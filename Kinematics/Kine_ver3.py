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


# 2. LOAD MUJOCO MODEL
model = mujoco.MjModel.from_xml_path(os.path.join(BASE_DIR, "mujoco_menagerie", "ufactory_lite6", "lite6.xml"))
data = mujoco.MjData(model)

# 3. TẮT GRAVITY
model.opt.gravity[:] = 0.0

# 4. SETUP EE

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

# Waypoint trung gian dùng cho nhánh SLERP (Cartesian)
P_trajectory = np.array([
    [0.087, 0.0, 0.1536],   # P0: Vị trí Home
    [0.4, -0.1, 0.4],        # P1: Điểm gắp vật
    [0.2, -0.2, 0.2],        # P2: Vị trí tránh cản
    x_target                 # P3: Vị trí đặt vật
])


# ============================================================
# HÀM SLERP + NỘI SUY CARTESIAN (bổ sung, còn thiếu trong bản gốc)
# ============================================================

def rotmat_to_quat(R):
    """Ma trận xoay 3x3 -> quaternion [w, x, y, z]."""
    R = np.array(R, dtype=float)
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    """Quaternion [w, x, y, z] -> ma trận xoay 3x3."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x*y - z*w),         2 * (x*z + y*w)],
        [2 * (x*y + z*w),           1 - 2 * (x**2 + z**2),   2 * (y*z - x*w)],
        [2 * (x*z - y*w),           2 * (y*z + x*w),         1 - 2 * (x**2 + y**2)]
    ])


def slerp(q0, q1, t, epsilon=1e-6):
    """SLERP giữa 2 quaternion [w, x, y, z]."""
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 1.0 - epsilon:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    return (s0 * q0) + (s1 * q1)


def generate_cartesian_trajectory(P_traj, R_traj, num_points_per_segment=20):
    """
    Nội suy vị trí (LERP) + hướng (SLERP) giữa các waypoint,
    trả về danh sách SE3 mượt cho toàn bộ quỹ đạo.
    """
    T_list = []

    for i in range(len(P_traj) - 1):
        p0, p1 = P_traj[i], P_traj[i + 1]
        q0 = rotmat_to_quat(R_traj[i])
        q1 = rotmat_to_quat(R_traj[i + 1])

        for k in range(num_points_per_segment):
            t = k / (num_points_per_segment - 1)

            p_interp = (1 - t) * p0 + t * p1
            q_interp = slerp(q0, q1, t)
            R_interp = quat_to_rotmat(q_interp)

            T_list.append(SE3.Rt(R_interp, p_interp))

    return T_list


def solve_cartesian_ik_trajectory(T_list, q_init, warm_start=True):
    """Giải IK (DLS) cho từng T trong T_list, warm-start từ nghiệm trước."""
    q_traj = []
    q_current = q_init.copy()

    for T in T_list:
        q_seed = q_current if warm_start else q_init
        q_current, _, _ = DLS(T, q_init=q_seed, max_iterations=200)
        q_traj.append(q_current)

    return np.array(q_traj)


# ============================================================
# 6. TÍNH TOÁN INVERSE KINEMATICS + CHỌN KIỂU QUỸ ĐẠO
# ============================================================

q_start, _, _ = DLS(T_start, q_init=np.zeros(6), max_iterations=200)
q_final, _, error = DLS(T_target=T_target, q_init=q_start, max_iterations=200)

if error > 1e-5:
    print(f"  [WARNING] Điểm cuối: IK chưa hội tụ tốt, error = {error:.3e}")

print("\nIK hoàn thành.")

# Tạo quỹ đạo theo lựa chọn của người dùng
inter = input("Nhập quỹ đạo mong muốn (1 = cubic, 2 = quintic, 3 = SLERP Cartesian): ")
inter = float(inter)

qd_tra = None  # mặc định, phòng trường hợp nhánh không tạo vận tốc

if inter == 1:
    t, q_trajectory, qd_tra, _ = cubic_trajectory(q_start=q_start, q_goal=q_final, T=5, N=200)

elif inter == 2:
    t, q_trajectory, qd_tra, _ = quintic_trajectory(q_start=q_start, q_goal=q_final, T=5, N=200)

else:
    # ---- Nhánh SLERP: nội suy Cartesian (vị trí LERP + hướng SLERP) ----
    R_trajectory = []
    for i, p in enumerate(P_trajectory):
        if i == 0:
            T_way = SE3.Trans(p) * SE3.RPY(np.pi, 0, 0, unit='rad')
        else:
            T_way = SE3.Trans(p) * SE3.RPY(0, np.pi / 2, 0, unit='rad')
        R_trajectory.append(T_way.R)  # fix: append cho MỌI waypoint, không chỉ else

    T_list = generate_cartesian_trajectory(P_trajectory, R_trajectory, num_points_per_segment=20)
    q_trajectory = solve_cartesian_ik_trajectory(T_list, q_init=np.zeros(6))

    # Nhánh này không có sẵn t/qd_tra như cubic/quintic -> tự tạo để phần mô phỏng dùng chung được
    t = np.linspace(0, 5, len(q_trajectory))
    qd_tra = np.zeros_like(q_trajectory)  # không tính vận tốc, gán 0 cho an toàn

    # Cập nhật lại q_start/q_final theo đúng quỹ đạo Cartesian vừa tạo
    q_start = q_trajectory[0]
    q_final = q_trajectory[-1]


# 7. KIỂM TRA LẠI BẰNG FORWARD KINEMATICS
T_final = forward_kinematics(q_final)
P_final = T_final[:3, 3]
print(f"\nKết quả Forward Kinematics từ giá trị khớp cuối cùng: {T_final[:3, 3]}")
position_error = np.linalg.norm(P_final - T_target.t)
print(f"Lỗi vị trí: {position_error:.6f} m")


# 8. HÀM VẼ ĐƯỜNG THẲNG-CẦU
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

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, radius, length / 2.0]),
        center,
        np.eye(3).flatten(),
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
    """Vẽ một quả cầu đánh dấu tại 1 điểm."""

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
    """Vẽ đường đi thực tế của end-effector dựa trên danh sách điểm."""

    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return

    for i in range(len(points) - 1):
        draw_line(scene, points[i], points[i + 1], radius=radius, rgba=rgba)


# ============================================================
# 9. MÔ PHỎNG MUJOCO THEO q_trajectory
# ============================================================

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

print("Sai số EE đầu:", np.linalg.norm(mujoco_start - P_start))


# ============================================================
# 10. MỞ MUJOCO VIEWER
# ============================================================

print("\nStarting MuJoCo viewer...")

with mujoco.viewer.launch_passive(model, data) as viewer:

    draw_sphere(viewer.user_scn, P_start, radius=0.012, rgba=(0.0, 1.0, 0.0, 1.0))
    draw_sphere(viewer.user_scn, T_target.t, radius=0.012, rgba=(0.0, 1.0, 0.0, 1.0))

    base_ngeom = viewer.user_scn.ngeom

    for i in range(len(q_trajectory)):

        if not viewer.is_running():
            break

        q = q_trajectory[i]

        data.qpos[:6] = q
        data.qvel[:6] = qd_tra[i] if qd_tra is not None else 0.0

        mujoco.mj_forward(model, data)

        mujoco_position = data.site_xpos[site_id].copy()

        viewer.sync()

        if i < len(t) - 1:
            dt = t[i + 1] - t[i]
            time.sleep(dt)

    mujoco_final = data.site_xpos[site_id].copy()

    print("\n========================================")
    print("KIỂM TRA ĐIỂM CUỐI")
    print("========================================")

    print("MuJoCo EE:")
    print(mujoco_final)

    print("\nFK EE:")
    print(P_final)


    draw_sphere(viewer.user_scn, mujoco_final, radius=0.012, rgba=(1.0, 0.0, 0.0, 1.0))

    viewer.sync()

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)

print("Done!")