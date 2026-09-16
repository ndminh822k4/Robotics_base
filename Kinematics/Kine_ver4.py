import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from spatialmath import SE3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Forward import forward_kinematics
from Reverse import Newton_Raphson_inverse_kinematics

LITE6_XML = os.path.join(BASE_DIR, "mujoco_menagerie", "ufactory_lite6", "lite6.xml")

# ============================================================
# 1. GHÉP 2 ROBOT VÀO CHUNG 1 SCENE BẰNG mujoco.MjSpec
# ============================================================
spec_main = mujoco.MjSpec.from_file(LITE6_XML)   # robot chính (dùng IK tự viết)
spec_ref  = mujoco.MjSpec.from_file(LITE6_XML)   # robot tham chiếu (dùng IK của MuJoCo)

# Dịch robot tham chiếu sang bên cạnh (+Y 0.6m) để không đè lên robot chính
frame = spec_main.worldbody.add_frame(pos=[0, 0.6, 0])
frame.attach_body(spec_ref.worldbody, prefix="ref_")

model = spec_main.compile()
data = mujoco.MjData(model)
model.opt.gravity[:] = 0.0

site_id_main = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
site_id_ref = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ref_attachment_site")

if site_id_main == -1 or site_id_ref == -1:
    raise ValueError("Không tìm thấy site — kiểm tra lại tên site sau khi attach (prefix 'ref_').")

# ============================================================
# 1b. TÔ MÀU RIÊNG CHO 2 ROBOT ĐỂ DỄ PHÂN BIỆT
#     Robot chính (main) -> xanh dương, robot tham chiếu (ref) -> cam
# ============================================================
COLOR_MAIN = np.array([0.2, 0.4, 1.0, 1.0])   # xanh dương
COLOR_REF  = np.array([1.0, 0.55, 0.0, 1.0])  # cam

for geom_id in range(model.ngeom):
    body_id = model.geom_bodyid[geom_id]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    if body_name.startswith("ref_"):
        model.geom_rgba[geom_id] = COLOR_REF
    else:
        model.geom_rgba[geom_id] = COLOR_MAIN

# Giả định mỗi robot có 6 khớp, robot ref nối ngay sau robot main trong qpos
qpos_main = slice(0, 6)
qpos_ref = slice(6, 12)
# Lite6 toàn khớp revolute (1 dof/khớp) -> chỉ số trong qvel/Jacobian trùng với qpos
dof_main = qpos_main
dof_ref = qpos_ref

# Offset vị trí của robot ref trong world frame (do attach_body ở +Y 0.6m)
OFFSET_REF = np.array([0.0, 0.6, 0.0])

# ============================================================
# 2. NHẬP ĐIỂM ĐÍCH VÀ TẠO ĐƯỜNG THẲNG NỘI SUY
# ============================================================
P_start = np.array([0.1, 0.2, 0.3])
T_start = SE3.Trans(0.1, 0.2, 0.3) * SE3.RPY(0, np.pi / 2, 0, unit='rad')

x_target = input("Nhập tọa độ điểm cuối (x, y, z) cách nhau bằng dấu phẩy: ")
x_target = np.array([float(x) for x in x_target.split(",")])
T_target = SE3.Trans(x_target) * SE3.RPY(0, np.pi / 2, 0, unit='rad')

print(f"Điểm đầu: {P_start}")
print(f"Điểm cuối: {T_target.t}")

def create_straight_line(P_a, P_b, N):
    s = np.linspace(0.0, 1.0, N)
    return P_a + s[:, None] * (P_b - P_a)

N_POINTS = 100
trajectory_forward = create_straight_line(P_start, T_target.t, N_POINTS)

# Chỉ đi 1 chiều từ điểm đầu -> điểm cuối rồi dừng (không lặp đi lặp lại)
cartersian_trajectory = trajectory_forward

# Quỹ đạo tương ứng cho robot ref, cộng thêm offset +0.6 theo Y
# vì site của robot ref nằm trong world frame đã bị dịch bởi attach_body
cartersian_trajectory_ref = cartersian_trajectory + OFFSET_REF

# Quaternion mục tiêu cho hướng bàn kẹp (giữ nguyên orientation cố định
# RPY(0, pi/2, 0) như code gốc), dùng cho IK của robot ref
target_quat = np.zeros(4)
mujoco.mju_mat2Quat(target_quat, T_target.R.flatten())

# ============================================================
# 3. GIẢI IK - ROBOT CHÍNH DÙNG HÀM TỰ VIẾT (Newton_Raphson)
# ============================================================
def solve_ik_for_trajectory(traj_points, q_seed, warm_start=True):
    """Giải IK cho từng điểm trên quỹ đạo.
    warm_start=True  -> mỗi điểm dùng nghiệm điểm trước làm q_init (mượt, robot chính)
    warm_start=False -> mỗi điểm giải lại từ đầu (q_init cố định, để so sánh)
    """
    qs = []
    q_current = q_seed.copy()
    for p in traj_points:
        T = SE3.Trans(p) * SE3.RPY(0, np.pi / 2, 0, unit='rad')
        q_init = q_current if warm_start else q_seed
        q_current, _, _ = Newton_Raphson_inverse_kinematics(
            T, q_init=q_init, max_iterations=50, tolerance=1e-4
        )
        qs.append(q_current.copy())
    return np.array(qs)

# ============================================================
# 3b. GIẢI IK - ROBOT THAM CHIẾU DÙNG JACOBIAN CỦA MUJOCO
#     (thay cho Newton_Raphson_inverse_kinematics tự viết)
# ============================================================
def mujoco_ik(model, data, site_id, qpos_slice, dof_slice,
              target_pos, target_quat=None,
              q_init=None, max_iterations=100, tolerance=1e-4,
              damping=1e-2, step_scale=0.01):
    """
    Giải IK cho 1 site bằng chính Jacobian của MuJoCo (mj_jacSite),
    dùng phương pháp damped least squares.

    - site_id: id của site cần điều khiển (vd site_id_ref)
    - qpos_slice: slice trong data.qpos ứng với robot đó (vd qpos_ref)
    - dof_slice: slice cột Jacobian ứng với robot đó
                 (với lite6 toàn khớp revolute -> dof_slice == qpos_slice)
    - target_pos: vị trí đích (3,) trong world frame
    - target_quat: quaternion đích (w,x,y,z) hoặc None nếu chỉ cần vị trí
    """
    if q_init is not None:
        data.qpos[qpos_slice] = q_init
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iterations):
        mujoco.mj_forward(model, data)

        site_pos = data.site_xpos[site_id].copy()
        pos_err = target_pos - site_pos

        if target_quat is not None:
            site_mat = data.site_xmat[site_id].reshape(3, 3)
            site_quat = np.zeros(4)
            mujoco.mju_mat2Quat(site_quat, site_mat.flatten())

            neg_site_quat = np.zeros(4)
            mujoco.mju_negQuat(neg_site_quat, site_quat)
            err_quat = np.zeros(4)
            mujoco.mju_mulQuat(err_quat, target_quat, neg_site_quat)
            quat_err = np.zeros(3)
            mujoco.mju_quat2Vel(quat_err, err_quat, 1.0)

            err = np.concatenate([pos_err, quat_err])
        else:
            err = pos_err

        if np.linalg.norm(err) < tolerance:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        if target_quat is not None:
            J = np.vstack([jacp[:, dof_slice], jacr[:, dof_slice]])
        else:
            J = jacp[:, dof_slice]

        # Damped least squares: dq = J^T (J J^T + lam^2 I)^-1 err
        JJt = J @ J.T
        lam2I = (damping ** 2) * np.eye(JJt.shape[0])
        dq = J.T @ np.linalg.solve(JJt + lam2I, err) * step_scale

        data.qpos[qpos_slice] = data.qpos[qpos_slice] + dq
        mujoco.mj_forward(model, data)

    return data.qpos[qpos_slice].copy()


def solve_ik_for_trajectory_mujoco(model, data, site_id, qpos_slice, dof_slice,
                                    traj_points, q_seed, target_quat=None,
                                    warm_start=False, max_iterations=100, tolerance=1e-4):
    """
    Bản thay thế solve_ik_for_trajectory, dùng mujoco_ik() (Jacobian MuJoCo)
    thay cho Newton_Raphson_inverse_kinematics tự viết.
    """
    qs = []
    q_current = np.array(q_seed, dtype=float).copy()
    for p in traj_points:
        q_init = q_current if warm_start else q_seed
        q_current = mujoco_ik(
            model, data, site_id, qpos_slice, dof_slice,
            target_pos=p, target_quat=target_quat,
            q_init=q_init, max_iterations=max_iterations, tolerance=tolerance
        )
        qs.append(q_current.copy())
    return np.array(qs)


print("\nĐang giải IK cho toàn bộ quỹ đạo (robot chính - warm start, hàm tự viết Newton_Raphson)...")
q_init0, _, _ = Newton_Raphson_inverse_kinematics(T_start, q_init=[0, 0, 0, 0, 0, 0],
                                                    max_iterations=100, tolerance=1e-4)
qs_main = solve_ik_for_trajectory(cartersian_trajectory, q_seed=q_init0, warm_start=True)

print("Đang giải IK cho toàn bộ quỹ đạo (robot tham chiếu - dùng IK Jacobian của MuJoCo)...")
qs_ref = solve_ik_for_trajectory_mujoco(
    model, data, site_id_ref, qpos_ref, dof_ref,
    cartersian_trajectory_ref, q_seed=qs_main[-1],
    target_quat=target_quat, warm_start=False,
    max_iterations=100, tolerance=1e-4
)
print("Xong.")

# Kiểm tra sai lệch góc khớp giữa 2 cách giải tại 1 vài điểm mẫu
sample_idx = [0, len(qs_main) // 2, -1]
for i in sample_idx:
    diff = np.linalg.norm(qs_main[i] - qs_ref[i])
    print(f"Điểm {i}: |q_main - q_ref| = {diff:.5f} rad")

# ============================================================
# 4. HÀM VẼ (giữ nguyên như code gốc)
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
        print("WARNING: user_scn không còn đủ geom.")
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, radius, length / 2.0]),
        center, np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32)
    )
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, direction)
    c = np.dot(z, direction)
    if np.linalg.norm(v) > 1e-9:
        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ])
        R = np.eye(3) + vx + (vx @ vx) / (1.0 + c)
        geom.mat[:] = R
    elif c < 0:
        geom.mat[:] = np.diag([1.0, -1.0, -1.0])
    scene.ngeom += 1

def draw_sphere(scene, center, radius=0.012, rgba=(0.0, 1.0, 0.0, 1.0)):
    if scene.ngeom >= scene.maxgeom:
        print("WARNING: user_scn không còn đủ geom.")
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0]),
        np.asarray(center, dtype=float),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32)
    )
    scene.ngeom += 1

# ============================================================
# 5. MÔ PHỎNG: 2 ROBOT CHẠY SONG SONG THEO 2 BỘ QS ĐÃ GIẢI
# ============================================================
data.qpos[qpos_main] = qs_main[0]
data.qpos[qpos_ref] = qs_ref[0]
mujoco.mj_forward(model, data)

print("Starting MuJoCo viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Vẽ đường thẳng target 1 lần duy nhất (tĩnh, không tốn geom mỗi frame)
    draw_line(viewer.user_scn, P_start, T_target.t, radius=0.003, rgba=(0.0, 0.4, 1.0, 1.0))
    draw_sphere(viewer.user_scn, P_start, radius=0.012, rgba=(0.0, 1.0, 0.0, 1.0))
    draw_sphere(viewer.user_scn, T_target.t, radius=0.012, rgba=(0.0, 1.0, 0.0, 1.0))

    idx = 0
    n_pts = len(cartersian_trajectory)

    # ------------------------------------------------------------
    # SIM_DT: thời gian chờ giữa 2 frame (giây). Tăng lên để chạy CHẬM hơn.
    # MAX_JOINT_STEP: chỉ áp dụng cho robot REF (cold-start MuJoCo IK hay
    # nhảy nghiệm giữa các điểm gần nhau) để hiển thị mượt hơn.
    # Robot MAIN dùng nguyên trạng qs_main (đã mượt sẵn nhờ warm-start
    # Newton-Raphson) -> KHÔNG rate-limit, hiển thị đúng kết quả IK Newton.
    # ------------------------------------------------------------
    SIM_DT = 0.05          # tăng từ 0.02 -> 0.05 (chậm hơn ~2.5 lần)
    MAX_JOINT_STEP = 0.05  # rad/frame, chỉ dùng để làm mượt robot ref

    reached_end_message_printed = False

    q_ref_current = qs_ref[0].copy()

    while viewer.is_running():
        # Robot main: hiển thị trực tiếp nghiệm Newton-Raphson, không làm mượt
        data.qpos[qpos_main] = qs_main[idx]

        # Robot ref: rate-limit để tránh giật khi IK cold-start nhảy nghiệm
        q_ref_target = qs_ref[idx]
        d_ref = np.clip(q_ref_target - q_ref_current, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        q_ref_current = q_ref_current + d_ref
        data.qpos[qpos_ref] = q_ref_current

        mujoco.mj_forward(model, data)

        # In sai lệch vị trí Cartesian thực tế giữa 2 robot (đã bù offset +0.6 theo Y)
        ee_main = data.site_xpos[site_id_main]
        ee_ref = data.site_xpos[site_id_ref] - OFFSET_REF
        # (bỏ comment dòng dưới nếu muốn xem log liên tục)
        # print(f"idx={idx} | lệch EE main-ref = {np.linalg.norm(ee_main - ee_ref):.5f}")

        viewer.sync()
        time.sleep(SIM_DT)

        # Đi từ điểm đầu -> điểm cuối rồi DỪNG hẳn (không lặp lại từ đầu).
        # Viewer vẫn mở để bạn xem/xoay góc nhìn tại tư thế cuối cùng.
        if idx < n_pts - 1:
            idx += 1
        elif not reached_end_message_printed:
            print("Đã đi hết quỹ đạo từ điểm đầu -> điểm cuối. Dừng tại đây (đóng cửa sổ viewer để thoát).")
            reached_end_message_printed = True

print("Done!")