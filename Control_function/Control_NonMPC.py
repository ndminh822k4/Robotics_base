import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import multiprocessing as mp
from queue import Full as QueueFull


# ============================================================
# 1. IMPORT FORWARD / INVERSE KINEMATICS / LQR
# ============================================================
FILE_PATH = os.path.abspath(__file__)
CONTROL_DIR = os.path.dirname(FILE_PATH)
KINEMATIC_DIR = os.path.dirname(CONTROL_DIR)
MUJUCO_ROOT_DIR = os.path.dirname(KINEMATIC_DIR)
sys.path.append(KINEMATIC_DIR)

from Forward import forward_kinematics
from Reverse_quat import DLS_quaternion
from Control_function.NonMPC import NMPCController

# ============================================================
# 2. LOAD MUJOCO MODEL
# ============================================================
xml_path = os.path.join(MUJUCO_ROOT_DIR, "mujoco_menagerie", "ufactory_lite6", "lite6.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ============================================================
# 3. END-EFFECTOR SITE
# ============================================================
site_name = "attachment_site"
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
if site_id == -1:
    raise ValueError(f"Site '{site_name}' not found in the model.")

# ============================================================
# 4. ĐẶT ĐIỂM ĐẦU VÀ ĐIỂM ĐÍCH
# ============================================================
P_start = np.array([0.087, 0.0, 0.1536])
quat_start = np.array([0.0, 1, 0.0, 0.0])  # Quaternion tương ứng với RPY (pi, 0, 0)

P_target = np.array([0.2, -0.3, 0.4])
quat_target = np.array([0.7071, 0.0, 0.7071, 0.0])  # Quaternion tương ứng với RPY (0, pi/2, 0)


# ============================================================
# 5. INVERSE KINEMATICS
# ============================================================
print("\n========================================")
print("INVERSE KINEMATICS")
print("========================================")

q_start, _, _ = DLS_quaternion(P_start, quat_start, q_init=np.zeros(6))
q_final, _, _ = DLS_quaternion(P_target, quat_target, q_init=q_start)


# ============================================================
# 6. DRAW FUNCTIONS
# ============================================================
def draw_line(scene, p1, p2, radius=0.01, rgba=(1.0, 0.0, 0.0, 1.0)):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    diff = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-9:
        return
    direction = diff / length
    center = (p1 + p2) / 2.0
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([radius, radius, length / 2.0]), center, np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, direction)
    c = np.dot(z, direction)
    if np.linalg.norm(v) > 1e-9:
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        R = np.eye(3) + vx + (vx @ vx) / (1.0 + c)
        geom.mat[:] = R
    elif c < 0:
        geom.mat[:] = np.diag([1.0, -1.0, -1.0])
    scene.ngeom += 1


def draw_sphere(scene, center, radius=0.012, rgba=(0.0, 1.0, 0.0, 1.0)):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(center, dtype=float), np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def draw_path(scene, points, radius=0.002, rgba=(0.0, 0.4, 1.0, 1.0)):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        draw_line(scene, points[i], points[i + 1], radius=radius, rgba=rgba)


# ============================================================
# 6b. LIVE PLOT CHẠY TRÊN MỘT PROCESS RIÊNG (song song với mô phỏng)
# ============================================================
# Chạy ở process khác (không phải thread) để KHÔNG chia sẻ GIL với vòng lặp
# mj_step/viewer -> vẽ đồ thị nặng bao nhiêu cũng không làm chậm mô phỏng
# chính. Giao tiếp 2 chiều qua multiprocessing.Queue: process mô phỏng chỉ
# "ném" các con số nhỏ (không phải mảng lớn/model MuJoCo) vào queue, process
# vẽ tự đọc và tự update đồ thị theo tốc độ của riêng nó.
def live_plot_worker(data_queue, torque_limit, max_points=3000, refresh_dt=0.05):
    import matplotlib
    matplotlib.use("TkAgg")  # backend GUI riêng cho process con
    import matplotlib.pyplot as _plt
    from collections import deque

    _plt.ion()
    fig, (ax1, ax2) = _plt.subplots(2, 1, figsize=(7, 6))
    fig.suptitle("Live monitor (process riêng) - MPC")
    fig.canvas.manager.set_window_title("Live Monitor - Control")

    t_buf = deque(maxlen=max_points)
    qerr_buf = deque(maxlen=max_points)
    dq_buf = deque(maxlen=max_points)
    torque_bufs = [deque(maxlen=max_points) for _ in range(6)]

    line_qerr, = ax1.plot([], [], color="tab:blue", label="|q_error|")
    line_dq, = ax1.plot([], [], color="tab:purple", label="|dq|")
    ax1.axhline(0.01, color="red", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("rad / rad·s⁻¹")
    ax1.set_title("Sai số vị trí & vận tốc (tổng hợp)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    torque_lines = []
    for i in range(6):
        ln, = ax2.plot([], [], label=f"tau{i+1}")
        torque_lines.append(ln)
    for lim in torque_limit:
        ax2.axhline(lim, color="gray", linestyle=":", linewidth=0.5)
        ax2.axhline(-lim, color="gray", linestyle=":", linewidth=0.5)
    ax2.set_title("Mô-men điều khiển từng khớp")
    ax2.set_xlabel("t (s)")
    ax2.legend(fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.3)

    _plt.tight_layout()
    _plt.show(block=False)

    while True:
        item = None
        stop_signal = False
        try:
            while True:
                latest = data_queue.get_nowait()
                if latest == "STOP":
                    stop_signal = True
                    break
                item = latest
        except Exception:
            pass

        if stop_signal:
            break

        if item is None:
            _plt.pause(refresh_dt)
            if not _plt.fignum_exists(fig.number):
                break
            continue

        t_buf.append(item["t"])
        qerr_buf.append(item["q_err_norm"])
        dq_buf.append(item["dq_norm"])
        for i in range(6):
            torque_bufs[i].append(item["torque"][i])

        line_qerr.set_data(t_buf, qerr_buf)
        line_dq.set_data(t_buf, dq_buf)
        for i in range(6):
            torque_lines[i].set_data(t_buf, torque_bufs[i])

        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()
        _plt.pause(0.001)

        if not _plt.fignum_exists(fig.number):
            break

    _plt.close(fig)

# ============================================================
# 7. INITIALIZE MUJOCO
# ============================================================
data.qpos[:6] = q_start
data.qvel[:6] = 0.0
mujoco.mj_forward(model, data)
mujoco_start = data.site_xpos[site_id].copy()

print("\n========================================")
print("INITIAL STATE")
print("========================================")
print("\nMuJoCo EE:")
print(mujoco_start)
print("\nExpected start:")
print(P_start)
print("\nPosition error:")
print(mujoco_start - P_start)

# ============================================================
# 8. LQR PARAMETERS (thay cho Kp/Kd/Ki của PID)
# ============================================================
# Q: phạt sai số vị trí (6 khớp đầu) + sai số vận tốc (6 khớp sau)
# R: phạt gia tốc điều khiển
# Q càng lớn -> bám sát hơn (dùng nhiều mô-men hơn)
# R càng lớn -> tiết kiệm lực hơn (bám kém hơn / đáp ứng êm hơn)
M = np.zeros((model.nu, model.nu))
mujoco.mj_fullM(model, data, M)

M_start = M[:6, :6]
Q_pos = np.diag([1000, 4000, 3000, 500, 600, 550])
Q_vel = np.diag(np.full(6, 5.0))

Q = np.block([
    [Q_pos, np.zeros((6, 6))],
    [np.zeros((6, 6)), Q_vel]
])

# NMPCController cần torque_limit THẬT (ràng buộc NLP), mà torque_limit
# thật chỉ được tính xong ở mục 11 (sau khi đọc forcerange/ctrlrange của
# actuator) -> NMPC được khởi tạo NGAY SAU mục 11 (xem dưới), không phải
# ở đây như bản MPCController tuyến tính cũ.
NMPC_R = np.array([0.05, 0.05, 0.05, 0.005, 0.01, 0.05])
NMPC_N = 5
NMPC_DT = 0.01


# ============================================================
# 9. SIMULATION PARAMETERS
# ============================================================
dt_sim = model.opt.timestep
print("\nMuJoCo timestep:")
print(dt_sim)

# ============================================================
# 9b. GRAVITY & CONTROLLER TOGGLE (bật/tắt bằng phím tắt)
# ============================================================
gravity_original = model.opt.gravity.copy()

gravity_state = {"enabled": False}
controller_state = {"enabled": False}
model.opt.gravity[:] = 0.0

GRAVITY_TOGGLE_KEY = " "     # Space: bật/tắt trọng lực
CONTROLLER_TOGGLE_KEY = "c"  # C: bật/tắt bộ điều khiển MPC


def set_gravity(enabled: bool):
    if enabled:
        model.opt.gravity[:] = gravity_original
        print("\n[INFO] Trọng lực: BẬT — robot sẽ rơi nếu bộ điều khiển không bù kịp.")
    else:
        model.opt.gravity[:] = 0.0
        print("\n[INFO] Trọng lực: TẮT.")
    gravity_state["enabled"] = enabled


def set_controller(enabled: bool):
    if enabled:
        print("\n[INFO] Bộ điều khiển MPC: BẬT — bắt đầu tính mô-men để bám điểm đích.")
    else:
        print("\n[INFO] Bộ điều khiển MPC: TẮT — mô-men = 0, robot rơi/đứng tự do theo vật lý.")
    controller_state["enabled"] = enabled


def key_callback(keycode):
    """Callback phím cho mujoco.viewer.launch_passive.
    Space: bật/tắt trọng lực.
    C    : bật/tắt bộ điều khiển MPC.
    """
    try:
        key_char = chr(keycode)
    except (ValueError, OverflowError):
        return

    if key_char == GRAVITY_TOGGLE_KEY:
        set_gravity(not gravity_state["enabled"])
    elif key_char.lower() == CONTROLLER_TOGGLE_KEY:
        set_controller(not controller_state["enabled"])

# ============================================================
# 10. ACTUATOR LIMIT
# ============================================================
forcerange = model.actuator_forcerange[:6].copy()
gear = model.actuator_gear[:6, 0].copy()

print("\n========================================")
print("ACTUATOR")
print("========================================")
print("\nForce range:")
print(forcerange)
print("\nGear:")
print(gear)

# ============================================================
# 11. TORQUE LIMIT
# ============================================================
# QUAN TRỌNG: nếu actuator KHÔNG khai forcelimited="true" trong XML,
# MuJoCo mặc định forcerange = [0, 0] -> đây KHÔNG phải là "giới hạn lực = 0",
# mà chỉ là placeholder chưa dùng tới. Nhiều model (kể cả lite6) giới hạn lực
# qua ctrlrange (áp cho actuator loại "motor" có gear=1, ctrl chính là torque)
# thay vì forcerange -> kiểm tra CẢ HAI trước khi quyết định giới hạn thật.
forcelimited = model.actuator_forcelimited[:6].astype(bool)
ctrllimited = model.actuator_ctrllimited[:6].astype(bool)
ctrlrange = model.actuator_ctrlrange[:6].copy()

# An toàn hơn nhiều so với 2000 Nm trước đây (phi thực tế với cánh tay nhỏ
# như lite6) -> nếu KHÔNG tìm được giới hạn thật nào, dùng tạm mức bảo thủ
# này để tránh mô-men khổng lồ làm mô phỏng bung/mất ổn định.
DEFAULT_TORQUE_BUDGET = 50.0  # Nm — CHỈNH LẠI theo thông số thật của lite6 nếu có

torque_limit = np.where(
    forcelimited,
    np.minimum(np.abs(forcerange[:, 0]), np.abs(forcerange[:, 1])),
    np.where(
        ctrllimited,
        np.minimum(np.abs(ctrlrange[:, 0]), np.abs(ctrlrange[:, 1])),
        DEFAULT_TORQUE_BUDGET
    )
)

print("\n========================================")
print("KIỂM TRA GIỚI HẠN ACTUATOR")
print("========================================")
print("\nforcelimited:", forcelimited)
print("ctrllimited :", ctrllimited)
print("ctrlrange   :\n", ctrlrange)
if not np.any(forcelimited) and not np.any(ctrllimited):
    print(f"[CẢNH BÁO] Không có giới hạn thật nào (forcerange lẫn ctrlrange đều "
          f"không bật limited). Đang dùng tạm ±{DEFAULT_TORQUE_BUDGET} Nm mọi khớp "
          f"— bạn NÊN thay bằng thông số mô-men thật của từng khớp lite6 nếu có "
          f"datasheet, để mô phỏng sát thực tế hơn.")

print("\nTorque limit (dùng thực tế):")
print(torque_limit)

# ============================================================
# 11b. KHỞI TẠO NMPC (phi tuyến, multiple shooting, dùng M(q)/bias thật
#      của MuJoCo tại từng bước dự báo, torque_limit là ràng buộc NLP
#      thật thay vì clip hậu-nghiệm)
# ============================================================
# ctrl = clip(torque/gear, -torque_limit, torque_limit) -> torque (đầu ra
# của controller, TRƯỚC khi chia gear) tương đương bị giới hạn trong
# [-torque_limit*gear, torque_limit*gear]. Đưa đúng giới hạn này vào NLP
# để ràng buộc |u|<=u_max là ràng buộc thật, không phải clip sau cùng.
nmpc_u_max = torque_limit * gear

MPC = NMPCController(
    model=model,
    Q=Q,
    R=NMPC_R,
    N=NMPC_N,
    dt=NMPC_DT,
    torque_limit=nmpc_u_max,
)

# ============================================================
# 12. OPEN VIEWER
# ============================================================
print("\n========================================")
print("STARTING MUJOCO")
print("========================================")

print("\n[HƯỚNG DẪN]")
print("  - Nhấn SPACE : bật/tắt TRỌNG LỰC (mặc định đang TẮT).")
print("  - Nhấn C     : bật/tắt BỘ ĐIỀU KHIỂN MPC (mặc định đang TẮT).")
print("  Quy trình test gợi ý: để mọi thứ TẮT -> robot đứng yên -> bật trọng")
print("  lực (SPACE) -> robot RƠI tự do -> bật bộ điều khiển (C) -> MPC sẽ tự")
print("  tính mô-men bù để kéo robot từ vị trí đang rơi về bám tới điểm đích.")

# ------------------------------------------------------------
# Khởi động process vẽ live (cửa sổ THỨ 2, tách biệt khỏi viewer MuJoCo)
# ------------------------------------------------------------
LIVE_PLOT_EVERY = 5  # cứ mỗi 5 bước mô phỏng mới đẩy 1 điểm dữ liệu vào queue
mp_ctx = mp.get_context("fork")  # fork: không tái chạy toàn bộ script trong process con
plot_queue = mp_ctx.Queue(maxsize=200)
live_plot_proc = mp_ctx.Process(
    target=live_plot_worker, args=(plot_queue, torque_limit), daemon=True
)
live_plot_proc.start()
print(f"[INFO] Đã mở process vẽ live (PID={live_plot_proc.pid}) — cửa sổ 'Live Monitor'.")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:

    # VẼ ĐIỂM START / TARGET
    draw_sphere(viewer.user_scn, P_start, radius=0.012, rgba=(0.0, 1.0, 0.0, 1.0))
    draw_sphere(viewer.user_scn, P_target, radius=0.012, rgba=(1.0, 0.0, 0.0, 1.0))
    base_ngeom = viewer.user_scn.ngeom
    ee_path = []

    model.opt.timestep = dt_sim  # đảm bảo timestep mô phỏng đúng với dt_sim

    # RESET ROBOT
    data.qpos[:6] = q_start
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)

    nv = model.nv  # số bậc tự do của model (dùng để lấy đúng shape ma trận khối lượng)

    reminder_shown = False  # nhắc 1 lần sau ~5s đứng yên (trọng lực + điều khiển đang tắt)

    # ------------------------------------------------------------
    # LOG DATA CHO ĐỒ THỊ ĐÁP ỨNG THỜI GIAN
    # ------------------------------------------------------------
    log_t = []
    log_q_error_norm = []
    log_q_error = []       # sai số từng khớp (6 cột)
    log_dq_norm = []
    log_dq = []             # vận tốc từng khớp (6 cột)
    log_torque = []         # mô-men điều khiển từng khớp (6 cột)
    log_ee_pos = []         # vị trí end-effector (x, y, z)
    log_gravity_on = []     # 0/1 -> để tô vùng trên đồ thị biết lúc nào trọng lực bật
    log_controller_on = []  # 0/1 -> để tô vùng trên đồ thị biết lúc nào controller bật

    step_counter = 0  # đếm bước mô phỏng để throttle tần suất đẩy dữ liệu live-plot

    # MAIN CONTROL LOOP
    while viewer.is_running():

        if (not reminder_shown) and (not gravity_state["enabled"]) and (not controller_state["enabled"]) and data.time >= 5.0:
            print("\n[NHẮC] Đã 5s. Nhấn SPACE để bật trọng lực (robot sẽ rơi), "
                  "sau đó nhấn C để bật bộ điều khiển MPC và xem robot tự bù về đích.")
            reminder_shown = True

        q_current = data.qpos[:6].copy()
        dq_current = data.qvel[:6].copy()

        position_error = q_final - q_current

        # Cập nhật động học/động lực học tại đúng trạng thái hiện tại trước khi
        # lấy M(q) và bias(q, qdot) (bias đã tự gồm cả trọng lực + Coriolis,
        # KHÔNG cần cộng thêm gravity compensation riêng nữa như PID trước đây)
        mujoco.mj_forward(model, data)

        M_full = np.zeros((nv, nv))
        mujoco.mj_fullM(model, data, M_full)
        M = M_full[:6, :6]
        bias = data.qfrc_bias[:6].copy()

        if controller_state["enabled"]:
            torque = MPC.compute(
                q_des=q_final,
                q_current=q_current,
                dq_des=np.zeros(6),
                dq_current=dq_current
            )

            # Chia theo tỉ số truyền actuator (gear=1 thì không đổi gì) rồi mới
            # clip theo giới hạn phần cứng thật (đã tính đúng ở bước 11, có xét
            # forcelimited/ctrllimited, KHÔNG dùng trực tiếp forcerange thô)
            torque = np.clip(torque / gear, -torque_limit, torque_limit)
            data.ctrl[:6] = torque
        else:
            # Bộ điều khiển đang TẮT -> không sinh mô-men, robot rơi/đứng tự do
            # theo vật lý (tuỳ trọng lực đang bật hay tắt).
            data.ctrl[:6] = 0.0

        mujoco.mj_step(model, data)

        ee_pos = data.site_xpos[site_id].copy()
        ee_path.append(ee_pos)

        # ---------------- LOG DATA CHO ĐỒ THỊ ----------------
        log_t.append(data.time)
        log_q_error_norm.append(np.linalg.norm(position_error))
        log_q_error.append(position_error.copy())
        log_dq_norm.append(np.linalg.norm(dq_current))
        log_dq.append(dq_current.copy())
        log_torque.append(data.ctrl[:6].copy())
        log_ee_pos.append(ee_pos.copy())
        log_gravity_on.append(1 if gravity_state["enabled"] else 0)
        log_controller_on.append(1 if controller_state["enabled"] else 0)

        # ---------------- ĐẨY DỮ LIỆU SANG PROCESS VẼ LIVE ----------------
        step_counter += 1
        if step_counter % LIVE_PLOT_EVERY == 0:
            try:
                plot_queue.put_nowait({
                    "t": float(data.time),
                    "q_err_norm": float(np.linalg.norm(position_error)),
                    "dq_norm": float(np.linalg.norm(dq_current)),
                    "torque": data.ctrl[:6].copy(),
                })
            except QueueFull:
                pass  # queue đầy -> bỏ qua điểm này, ưu tiên tốc độ mô phỏng chính

        viewer.user_scn.ngeom = base_ngeom
        draw_path(viewer.user_scn, ee_path, radius=0.0015, rgba=(0.0, 0.4, 1.0, 1.0))

        if int(data.time * 100) % 50 == 0:
            print(f"\rt = {data.time:6.2f} | |q_err| = {np.linalg.norm(position_error):8.5f} | |dq| = {np.linalg.norm(dq_current):8.5f} | EE = {ee_pos}", end="")
            print(f"\n q_error từng khớp là: ", q_final-q_current)

        q_error_norm = np.linalg.norm(position_error)
        dq_norm = np.linalg.norm(dq_current)


        if controller_state["enabled"] and q_error_norm < 0.01 and dq_norm < 0.05:
            print("\n\n========================================")
            print("ĐÃ TỚI ĐIỂM ĐÍCH")
            print("========================================")
            print("\nq_current:")
            print(q_current)
            print("\nq_final:")
            print(q_final)
            print("\nq error:")
            print(q_final - q_current)
            print("\nEE:")
            print(ee_pos)
            break

        viewer.sync()
        time.sleep(0.001)  # sleep 10ms để giảm tải CPU, không cần thiết nhưng giúp máy mát hơn

# ------------------------------------------------------------
# ĐÓNG PROCESS VẼ LIVE MỘT CÁCH SẠCH SẼ
# ------------------------------------------------------------
try:
    plot_queue.put_nowait("STOP")
except QueueFull:
    pass
live_plot_proc.join(timeout=3)
if live_plot_proc.is_alive():
    print("[CẢNH BÁO] Process vẽ live không tự thoát kịp -> buộc terminate().")
    live_plot_proc.terminate()
    live_plot_proc.join()

# ============================================================
# 13. TÍNH CHỈ SỐ ĐÁP ỨNG: RISE TIME / SETTLING TIME / OVERSHOOT
# ============================================================
def compute_step_response_metrics(t, log_q_error, log_controller_on, q_final,
                                   settle_band_ratio=0.02, min_settle_band=1e-3):
    """Tính rise time, settling time, overshoot cho từng khớp, coi thời điểm
    bộ điều khiển được BẬT (cạnh lên đầu tiên của controller_on) là mốc t0
    của một 'bước nhảy' (step response) từ q0 -> q_final.

    - rise time     : thời gian đi từ 10% -> 90% quãng đường tới đích.
    - settling time : thời gian (tính từ t0) để đáp ứng đi vào và ở lại
                       trong dải ±settle_band_ratio*|delta| quanh giá trị đích.
    - overshoot (%) : phần trăm vượt quá đích so với biên độ bước nhảy |delta|.

    Trả về (t0, list các dict theo từng khớp) hoặc (None, None) nếu controller
    chưa từng được bật trong lúc log dữ liệu.
    """
    t = np.asarray(t)
    controller_on = np.asarray(log_controller_on)
    q_error = np.asarray(log_q_error)  # (N, 6), q_error = q_final - q_current

    on_idx = np.where(controller_on > 0)[0]
    if len(on_idx) == 0:
        return None, None

    t0_idx = on_idx[0]
    t0 = t[t0_idx]
    q0 = q_final - q_error[t0_idx]  # trạng thái khớp tại thời điểm bật controller

    tt = t[t0_idx:]
    resp_all = q_final[None, :] - q_error[t0_idx:, :]  # quỹ đạo q(t) từ t0 trở đi, shape (M, 6)

    metrics = []
    for j in range(6):
        delta = q_final[j] - q0[j]
        resp = resp_all[:, j]

        if abs(delta) < 1e-9:
            metrics.append({"rise_time": 0.0, "settling_time": 0.0, "overshoot_pct": 0.0,
                             "delta": delta})
            continue

        norm_resp = (resp - q0[j]) / delta  # ~0 -> ~1 nếu tiến thẳng tới đích

        # --- Rise time: 10% -> 90% ---
        idx_10 = np.where(norm_resp >= 0.10)[0]
        idx_90 = np.where(norm_resp >= 0.90)[0]
        if len(idx_10) > 0 and len(idx_90) > 0 and idx_90[0] >= idx_10[0]:
            rise_time = tt[idx_90[0]] - tt[idx_10[0]]
        else:
            rise_time = np.nan  # chưa từng đạt 90% (chưa hội tụ trong log)

        # --- Settling time: thời điểm CUỐI CÙNG còn ra ngoài dải sai số cho phép ---
        band = max(settle_band_ratio * abs(delta), min_settle_band)
        out_of_band = np.abs(resp - q_final[j]) > band
        if np.any(out_of_band):
            last_out_idx = np.where(out_of_band)[0][-1]
            if last_out_idx == len(tt) - 1:
                settling_time = np.nan  # chưa bao giờ settle trong khoảng log
            else:
                settling_time = tt[last_out_idx] - tt[0]
        else:
            settling_time = 0.0  # đã nằm trong dải ngay từ đầu

        # --- Overshoot (%) so với biên độ bước nhảy |delta| ---
        if delta > 0:
            peak = np.max(resp)
            overshoot_pct = max(0.0, (peak - q_final[j]) / delta * 100.0)
        else:
            peak = np.min(resp)
            overshoot_pct = max(0.0, (q_final[j] - peak) / (-delta) * 100.0)

        metrics.append({"rise_time": rise_time, "settling_time": settling_time,
                         "overshoot_pct": overshoot_pct, "delta": delta})

    return t0, metrics


def print_metrics_table(t0, metrics):
    if metrics is None:
        print("\n[CẢNH BÁO] Bộ điều khiển chưa từng được BẬT (phím C) trong lúc mô "
              "phỏng chạy -> không có dữ liệu để tính rise time/settling time/overshoot.")
        return

    print("\n========================================")
    print(f"CHỈ SỐ ĐÁP ỨNG (mốc t0 = lúc bật controller = {t0:.3f}s)")
    print("========================================")
    header = f"{'Khớp':<6}{'Rise time (s)':<16}{'Settling time (s)':<20}{'Overshoot (%)':<16}"
    print(header)
    print("-" * len(header))
    for j, m in enumerate(metrics):
        rise_str = f"{m['rise_time']:.3f}" if not np.isnan(m["rise_time"]) else "N/A"
        settle_str = f"{m['settling_time']:.3f}" if not np.isnan(m["settling_time"]) else "N/A"
        print(f"q{j+1:<5}{rise_str:<16}{settle_str:<20}{m['overshoot_pct']:<16.2f}")
    print("-" * len(header))
    print("(N/A = chưa hội tụ đủ trong khoảng thời gian log được)")


def plot_time_response(log_t, log_q_error_norm, log_q_error, log_dq_norm,
                        log_dq, log_torque, log_ee_pos,
                        log_gravity_on, log_controller_on,
                        torque_limit, metrics_t0=None, metrics=None, save_path=None):
    if len(log_t) == 0:
        print("\n[CẢNH BÁO] Không có dữ liệu để vẽ đồ thị (chưa chạy bước mô phỏng nào).")
        return

    t = np.asarray(log_t)
    q_error_norm = np.asarray(log_q_error_norm)
    q_error = np.asarray(log_q_error)          # (N, 6)
    dq_norm = np.asarray(log_dq_norm)
    dq = np.asarray(log_dq)                    # (N, 6)
    torque = np.asarray(log_torque)            # (N, 6)
    ee_pos = np.asarray(log_ee_pos)            # (N, 3)
    gravity_on = np.asarray(log_gravity_on)
    controller_on = np.asarray(log_controller_on)

    def shade_events(ax):
        ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1],
                         where=gravity_on > 0, color="orange", alpha=0.08,
                         label="Trọng lực BẬT", step="pre")
        ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1],
                         where=controller_on > 0, color="green", alpha=0.08,
                         label="Controller BẬT", step="pre")

    fig, axes = plt.subplots(4, 2, figsize=(14, 13),
                              gridspec_kw={"height_ratios": [1, 1, 1, 0.9]})
    fig.suptitle("Đáp ứng thời gian của bộ điều khiển - MPC", fontsize=14)

    # --- (1) Sai số vị trí khớp (norm) ---
    ax = axes[0, 0]
    ax.plot(t, q_error_norm, color="tab:blue", label="|q_error|")
    ax.axhline(0.01, color="red", linestyle="--", linewidth=1, label="Ngưỡng hội tụ (0.01)")
    ax.set_ylabel("|q_error| (rad)")
    ax.set_title("Sai số vị trí khớp (tổng hợp)")
    shade_events(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (2) Sai số từng khớp + đánh dấu rise/settling time ---
    ax = axes[0, 1]
    for i in range(q_error.shape[1]):
        ax.plot(t, q_error[:, i], label=f"q{i+1}")
    if metrics_t0 is not None and metrics is not None:
        ax.axvline(metrics_t0, color="black", linestyle="-", linewidth=1,
                    label="t0 (bật controller)")
        for j, m in enumerate(metrics):
            if not np.isnan(m["settling_time"]):
                ax.axvline(metrics_t0 + m["settling_time"], color=f"C{j}",
                            linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("Sai số từng khớp (nét đứt chấm = settling time mỗi khớp)")
    ax.set_ylabel("q_error (rad)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # --- (3) Vận tốc khớp (norm) ---
    ax = axes[1, 0]
    ax.plot(t, dq_norm, color="tab:purple", label="|dq|")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="Ngưỡng hội tụ (0.05)")
    ax.set_ylabel("|dq| (rad/s)")
    ax.set_title("Vận tốc khớp (tổng hợp)")
    shade_events(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (4) Mô-men điều khiển từng khớp ---
    ax = axes[1, 1]
    for i in range(torque.shape[1]):
        ax.plot(t, torque[:, i], label=f"tau{i+1}")
    for i in range(len(torque_limit)):
        ax.axhline(torque_limit[i], color="gray", linestyle=":", linewidth=0.6)
        ax.axhline(-torque_limit[i], color="gray", linestyle=":", linewidth=0.6)
    ax.set_title("Mô-men điều khiển từng khớp (nét đứt xám = giới hạn)")
    ax.set_ylabel("Torque (Nm)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # --- (5) Vị trí end-effector theo trục X/Y/Z ---
    ax = axes[2, 0]
    ax.plot(t, ee_pos[:, 0], label="EE_x")
    ax.plot(t, ee_pos[:, 1], label="EE_y")
    ax.plot(t, ee_pos[:, 2], label="EE_z")
    ax.axhline(P_target[0], color="C0", linestyle="--", linewidth=0.8)
    ax.axhline(P_target[1], color="C1", linestyle="--", linewidth=0.8)
    ax.axhline(P_target[2], color="C2", linestyle="--", linewidth=0.8)
    ax.set_title("Vị trí end-effector (nét đứt = target)")
    ax.set_ylabel("Position (m)")
    ax.set_xlabel("Thời gian (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (6) Trạng thái gravity / controller theo thời gian ---
    ax = axes[2, 1]
    ax.step(t, gravity_on, where="post", label="Gravity ON/OFF", color="orange")
    ax.step(t, controller_on + 1.1, where="post", label="Controller ON/OFF", color="green")
    ax.set_yticks([0, 1, 1.1, 2.1])
    ax.set_yticklabels(["OFF", "ON", "OFF", "ON"])
    ax.set_title("Trạng thái trọng lực & bộ điều khiển")
    ax.set_xlabel("Thời gian (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (7) Bảng chỉ số Rise time / Settling time / Overshoot ---
    axes[3, 0].remove()
    axes[3, 1].remove()
    ax_table = fig.add_subplot(4, 1, 4)
    ax_table.axis("off")
    if metrics is not None:
        col_labels = ["Khớp", "Rise time (s)", "Settling time (s)", "Overshoot (%)"]
        rows = []
        for j, m in enumerate(metrics):
            rise_str = f"{m['rise_time']:.3f}" if not np.isnan(m["rise_time"]) else "N/A"
            settle_str = f"{m['settling_time']:.3f}" if not np.isnan(m["settling_time"]) else "N/A"
            rows.append([f"q{j+1}", rise_str, settle_str, f"{m['overshoot_pct']:.2f}"])
        table = ax_table.table(cellText=rows, colLabels=col_labels,
                                cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        ax_table.set_title(f"Chỉ số đáp ứng (t0 = lúc bật controller = {metrics_t0:.3f}s)",
                            fontsize=10, pad=12)
    else:
        ax_table.text(0.5, 0.5, "Chưa từng bật controller (phím C) -> không có chỉ số.",
                       ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n[INFO] Đã lưu đồ thị đáp ứng thời gian tại: {save_path}")

    plt.show()


metrics_t0, metrics = compute_step_response_metrics(
    log_t, log_q_error, log_controller_on, q_final
)
print_metrics_table(metrics_t0, metrics)

# plot_save_path = os.path.join(CONTROL_DIR, "mpc_time_response.png")
plot_time_response(
    log_t, log_q_error_norm, log_q_error, log_dq_norm, log_dq,
    log_torque, log_ee_pos, log_gravity_on, log_controller_on,
    torque_limit, metrics_t0=metrics_t0, metrics=metrics, save_path=None
)

# ============================================================
# 14. DONE
# ============================================================
print("\n\n========================================")
print("SIMULATION FINISHED")
print("========================================")