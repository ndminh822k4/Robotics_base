import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from spatialmath import SE3


# ============================================================
# 1. IMPORT FORWARD + INVERSE KINEMATICS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Forward import forward_kinematics
from Reverse import Newton_Raphson_inverse_kinematics, DLS
from Interpolation import cubic_trajectory, quintic_trajectory
from Interpolation_tg import plan_joint_trajectory_cubic_spline


# 2. LOAD MUJOCO MODEL
model = mujoco.MjModel.from_xml_path(os.path.join(BASE_DIR, "mujoco_menagerie", "ufactory_lite6", "lite6.xml"))
data = mujoco.MjData(model)

#3. TẮT GRAVITY
model.opt.gravity[:] = 0.0

#4. SETUP EE

site_name = "attachment_site"
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

if site_id == -1:
    raise ValueError(f"Site '{site_name}' not found in the model.")

# 5. ĐẶT ĐIỂM ĐẦU-CUỐI 
P_start = np.array([0.087, 0.0, 0.1536])  # Điểm đầu
T_start = SE3.Trans(0.087, 0.0, 0.1536) * SE3.RPY(np.pi, 0, 0, unit='rad')
x_target = input("Nhập tọa độ điểm cuối (x, y, z) cách nhau bằng dấu phẩy: ")
x_target = np.array([float(x) for x in x_target.split(",")])

T_target = SE3.Trans(x_target) * SE3.RPY(0, np.pi / 2, 0, unit='rad')




print(f"Điểm đầu: {P_start}")
print(f"Điểm cuối: {T_target.t}")

# TẠO N ĐIỂM TRUNG GIAN (MẶC ĐỊNH 4)
P_trajectory = np.array([
        [0.087, 0.0, 0.1536],    # P0: Vị trí Home
        [0.4, -0.1, 0.4],   # P1: Điểm gắp vật
        [0.2, -0.2, 0.2],   # P2: Vị trí tránh cản
        x_target   # P3: Vị trí đặt vật
    ]) 

def solve_interpolation(P_trajectory, q_init, warm_start = True):
    q_trajectory = []

    q_current = q_init.copy()
    for i, p in enumerate(P_trajectory):
        if i == 0:
            T_seed = SE3.Trans(p) * SE3.RPY(np.pi, 0, 0, unit='rad')
        else:
            T_seed = SE3.Trans(p) * SE3.RPY(0, np.pi/2, 0, unit='rad')

        q_seed = q_current if warm_start==True else q_init
        q_current, _, _ = DLS(T_seed, q_init=q_seed, max_iterations=200)
        print(q_current)
        q_trajectory.append(q_current)
    return np.array(q_trajectory)



# 6. TÍNH TOÁN INVERSE KINEMATICS
q_tra = solve_interpolation(P_trajectory=P_trajectory, q_init = np.zeros(6))
q_start = q_tra[0]
q_final = q_tra[-1]
print("Quy dao dau vao: ", q_start)
print("Quy dao dau ra: ", q_final)
print("\nIK hoàn thành.")

#Tao ham noi suy bang cubic spline
t, q_trajectory, _, _ = plan_joint_trajectory_cubic_spline(q_tra, dt_waypoints=4)
print("Shape cua noi suy: ", q_trajectory.shape)


#7. KIỂM TRA LẠI BẰNG FORWARD KINEMATICS
T_final = forward_kinematics(q_final)
P_final = T_final[:3, 3]
print(f"\nKết quả Forward Kinematics từ giá trị khớp cuối cùng: {T_final[:3, 3]}")
position_error = np.linalg.norm(P_final - T_target.t)
print(f"Lỗi vị trí: {position_error:.6f} m")


#8. HÀM VẼ ĐƯƠNG THẲNG-CẦU
def draw_line(scene, p1, p2, radius=0.01, rgba=(1.0, 0.0, 0.0, 1.0)):
    """Vẽ một đoạn capsule từ p1 -> p2 trực tiếp trong MuJoCo viewer."""

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

    # Lấy geom có sẵn trong buffer (KHÔNG tạo geom mới, KHÔNG gán đè tuple)
    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, radius, length / 2.0]),
        center,
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32)
    )

    # Xoay capsule: trục Z mặc định -> direction
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
        # direction ngược hoàn toàn với +Z
        geom.mat[:] = np.diag([1.0, -1.0, -1.0])
    # c > 0: giữ nguyên mat = eye(3) đã init ở trên

    scene.ngeom += 1

def draw_sphere(scene, center, radius=0.012, rgba=(0.0, 1.0, 0.0, 1.0)):
    """Vẽ một quả cầu đánh dấu tại 1 điểm - để so sánh trực quan
    với đường thẳng target, kiểm tra có bị lệch frame hay không."""
 
    if scene.ngeom >= scene.maxgeom:
        print("WARNING: user_scn không còn đủ geom.")
        return
 
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0]),
        np.asarray(center, dtype=float),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32)
    )
    scene.ngeom += 1
 
 
def draw_path(scene, points, radius=0.002, rgba=(0.0, 0.4, 1.0, 1.0)):
    """
    Vẽ đường đi THỰC TẾ của end-effector, dựa trên danh sách điểm
    (thường là data.site_xpos đọc mỗi frame trong lúc mô phỏng).
 
    points: list/array các điểm (N, 3), sẽ nối các điểm liên tiếp
            bằng draw_line() để tạo thành 1 polyline.
 
    Lưu ý: hàm này CHỈ thêm geom cho các đoạn nối giữa points,
    không tự xoá geom cũ. Muốn vẽ lại (redraw) mỗi frame mà không
    tràn scene, hãy reset scene.ngeom về đúng mốc trước khi gọi
    hàm này (xem ví dụ dùng bên dưới, biến base_ngeom).
    """
 
    points = np.asarray(points, dtype=float)
 
    if len(points) < 2:
        return
 
    for i in range(len(points) - 1):
        draw_line(
            scene,
            points[i],
            points[i + 1],
            radius=radius,
            rgba=rgba
        )


#9. MÔ PHỎNG MUJOCO
# ============================================================
# 9. MÔ PHỎNG MUJOCO THEO q_trajectory
# ============================================================

# Đặt robot tại vị trí bắt đầu
data.qpos[:6] = q_start
data.qvel[:6] = 0.0

mujoco.mj_forward(model, data)

mujoco_start = data.site_xpos[site_id].copy()

print("\n========================================")
print("MÔ PHỎNG MUJOCO")
print("========================================")

print("q_start:")
print(q_start)

print("\nq_final:")
print(q_final)

print("\nMuJoCo EE tại điểm đầu:")
print(mujoco_start)

print("\nFK EE tại điểm đầu:")
print(P_start)

print(
    "Sai số EE đầu:",
    np.linalg.norm(mujoco_start - P_start)
)


# ============================================================
# 10. MỞ MUJOCO VIEWER
# ============================================================

print("\nStarting MuJoCo viewer...")

with mujoco.viewer.launch_passive(model, data) as viewer:

    # --------------------------------------------------------
    # Vẽ điểm bắt đầu
    # --------------------------------------------------------

    draw_sphere(
        viewer.user_scn,
        P_start,
        radius=0.012,
        rgba=(0.0, 1.0, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # Vẽ điểm đích
    # --------------------------------------------------------

    draw_sphere(
        viewer.user_scn,
        T_target.t,
        radius=0.012,
        rgba=(0.0, 1.0, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # Lưu số geom ban đầu
    # --------------------------------------------------------

    base_ngeom = viewer.user_scn.ngeom

    # ========================================================
    # CHẠY q_trajectory
    # ========================================================

    for i in range(len(q_trajectory)):

        # Nếu đóng viewer thì thoát
        if not viewer.is_running():
            break

        # ----------------------------------------------------
        # Lấy bộ góc tại thời điểm i
        # ----------------------------------------------------

        q = q_trajectory[i]

        # ----------------------------------------------------
        # Gán vị trí khớp
        # ----------------------------------------------------

        data.qpos[:6] = q

        # Vì đang kiểm tra FK/trajectory
        # nên vận tốc không cần thiết
        data.qvel[:6] = 0

        # ----------------------------------------------------
        # Tính lại toàn bộ kinematics của MuJoCo
        # ----------------------------------------------------

        mujoco.mj_forward(model, data)

        # ----------------------------------------------------
        # Lấy vị trí EE từ MuJoCo
        # ----------------------------------------------------

        mujoco_position = data.site_xpos[site_id].copy()
        time.sleep(0.005)

        # ----------------------------------------------------
        # In ra một số frame
        # ----------------------------------------------------

        # if i % 20 == 0 or i == len(q_trajectory) - 1:

        #     print(
        #         f"Step {i:3d}/{len(q_trajectory)-1}: "
        #         f"q = {np.rad2deg(q)}"
        #     )

        #     print(
        #         f"    EE = {mujoco_position}"
        #     )

        # ----------------------------------------------------
        # Cập nhật viewer
        # ----------------------------------------------------

        viewer.sync()

        # ----------------------------------------------------
        # Chờ theo timestep của trajectory
        # ----------------------------------------------------

        # if i < len(t) - 1:
        #     dt = t[i + 1] - t[i]
        #     time.sleep(dt)


    # ========================================================
    # KIỂM TRA ĐIỂM CUỐI
    # ========================================================

    mujoco_final = data.site_xpos[site_id].copy()

    print("\n========================================")
    print("KIỂM TRA ĐIỂM CUỐI")
    print("========================================")

    print("MuJoCo EE:")
    print(mujoco_final)

    print("\nFK EE:")
    print(P_final)

    error = np.linalg.norm(
        mujoco_final - P_final
    )

    print(
        f"\nSai số vị trí MuJoCo vs FK: "
        f"{error:.6f} m"
    )

    # --------------------------------------------------------
    # Đánh dấu vị trí cuối MuJoCo
    # --------------------------------------------------------

    draw_sphere(
        viewer.user_scn,
        mujoco_final,
        radius=0.012,
        rgba=(1.0, 0.0, 0.0, 1.0)
    )

    viewer.sync()

    # ========================================================
    # GIỮ VIEWER SAU KHI CHẠY XONG
    # ========================================================

    while viewer.is_running():

        viewer.sync()

        time.sleep(20)
print("Done!")
