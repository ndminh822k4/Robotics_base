import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer


# =========================================================
# 1. IMPORT FK CỦA M
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from Kinematic.Forward import forward_kinematics


# =========================================================
# 2. LOAD MODEL MUJOCO
# =========================================================

xml_path = os.path.join(
    BASE_DIR,
    "mujoco_menagerie",
    "ufactory_lite6",
    "lite6.xml"
)

if not os.path.exists(xml_path):
    raise FileNotFoundError(
        f"Không tìm thấy file XML:\n{xml_path}"
    )

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Lay id cua site "attachment_site" (dat san o goc link6 trong file XML)
# --> dung site nay an toan hon la doan index body bang tay (vd data.xpos[6]),
# vi index body se doi neu them/bot body trong cay kinematic.
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")


# =========================================================
# 3. TẮT GRAVITY
# =========================================================

model.opt.gravity[:] = 0


# =========================================================
# 4. NHẬP Q BAN ĐẦU
# =========================================================
q_start_input = input(
    "Nhập q_start (độ), cách nhau bằng dấu phẩy: "
)
# =========================================================
# 5. NHẬP Q ĐÍCH
# =========================================================

q_goal_input = input(
    "Nhập q_goal (độ), cách nhau bằng dấu phẩy: "
)

# =========================================================
# 5b. NHẬP SỐ LẦN ĐI QUA LẠI
# =========================================================

num_cycles_input = input(
    "Nhập số lần đi qua lại (start->goal->start tính là 1 lần), Enter = 3: "
)
num_cycles = int(num_cycles_input.strip()) if num_cycles_input.strip() else 3


# =========================================================
# 6. ĐỔI ĐỘ → RADIAN
# =========================================================
q_start_deg = np.array([
    float(x.strip())
    for x in q_start_input.split(",")
], dtype=float)

if len(q_start_deg) != 6:
    raise ValueError("Phải nhập đúng 6 giá trị!")

q_start = np.deg2rad(q_start_deg)

q_goal_deg = np.array([
    float(x.strip())
    for x in q_goal_input.split(",")
], dtype=float)

if len(q_goal_deg) != 6:
    raise ValueError("Phải nhập đúng 6 giá trị!")

q_goal = np.deg2rad(q_goal_deg)
# =========================================================
# 7. TÍNH FK TẠI ĐIỂM ĐẦU
# =========================================================

T_start = forward_kinematics(q_start)

print("\n================================")
print("FK TẠI ĐIỂM ĐẦU")
print("================================")
print(T_start)
print("\nVị trí EE ban đầu:")
print(T_start[:3, 3])

# =========================================================
# 8. TÍNH FK TẠI ĐIỂM ĐÍCH
# =========================================================

T_goal = forward_kinematics(q_goal)

print("\n================================")
print("FK TẠI ĐIỂM ĐÍCH")
print("================================")

print(T_goal)

print("\nVị trí EE đích:")
print(T_goal[:3, 3])


# =========================================================
# 9. TẠO QUỸ ĐẠO JOINT (mỗi chiều đi N điểm)
# =========================================================

N = 80

traj_forward = np.linspace(q_start, q_goal, N)
traj_backward = np.linspace(q_goal, q_start, N)


# =========================================================
# TÍNH EE START VÀ EE GOAL BẰNG FK-DH
# =========================================================

T_start = forward_kinematics(q_start)
p_start = T_start[:3, 3]

T_goal = forward_kinematics(q_goal)
p_goal = T_goal[:3, 3]

print("\nEE start =", np.round(p_start, 4))
print("EE goal  =", np.round(p_goal, 4))


# =========================================================
# HÀM PHỤ: DỰNG MA TRẬN XOAY CHO CAPSULE NỐI 2 ĐIỂM
# =========================================================

def capsule_rotation(p_from, p_to):
    """Tra ve (center, length, rotation 3x3) cho 1 capsule noi p_from -> p_to."""
    direction = p_to - p_from
    length = np.linalg.norm(direction)

    if length < 1e-9:
        return None

    center = (p_from + p_to) / 2
    z_axis = direction / length

    temp = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(temp, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return center, length, rotation


def draw_capsule(scn, p_from, p_to, radius, rgba):
    """Ve 1 doan capsule tu p_from -> p_to vao scene, neu con cho (ngeom < maxgeom)."""
    res = capsule_rotation(p_from, p_to)
    if res is None:
        return False

    if scn.ngeom >= scn.maxgeom:
        return False

    center, length, rotation = res

    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, length / 2, 0.0]),
        center,
        rotation.flatten(),
        np.array(rgba, dtype=np.float64)
    )
    scn.ngeom += 1
    return True


def point_to_segment_distance(p, a, b):
    """Khoang cach tu diem p toi doan thang a-b (dung de danh gia sai lech)."""
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 < 1e-12:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


# =========================================================
# MUJOCO VIEWER
# =========================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    # -----------------------------------------------------
    # Vẽ đường thẳng lý tưởng (đỏ) START -> GOAL
    # -----------------------------------------------------

    draw_capsule(
        viewer.user_scn,
        p_start,
        p_goal,
        radius=0.005,
        rgba=[1.0, 0.0, 0.0, 1.0]
    )

    # =====================================================
    # DI CHUYỂN QUA LẠI GIỮA Q_START <-> Q_GOAL
    # =====================================================

    deviations = []       # sai lech (m) cua tat ca cac diem trong toan bo qua trinh
    prev_point = None     # diem EE thuc te (MuJoCo) truoc do, de noi thanh trail

    for cycle in range(num_cycles):

        print(f"\n=== Vòng {cycle + 1}/{num_cycles}: START -> GOAL ===")
        legs = [("START -> GOAL", traj_forward), ("GOAL -> START", traj_backward)]

        for leg_name, traj in legs:
            print(f"--- {leg_name} ---")

            for q in traj:

                if not viewer.is_running():
                    break

                # -----------------------------------------
                # Đưa q hiện tại vào MuJoCo
                # -----------------------------------------
                data.qpos[:6] = q
                mujoco.mj_forward(model, data)

                # -----------------------------------------
                # FK bằng bảng DH của m
                # -----------------------------------------
                T_DH = forward_kinematics(q)
                p_DH = T_DH[:3, 3]

                # -----------------------------------------
                # Vị trí thực tế theo MuJoCo (ground truth)
                # -----------------------------------------
                p_mj = data.site_xpos[ee_site_id].copy()

                # -----------------------------------------
                # Vẽ trail đường đi thực tế (xanh dương)
                # -----------------------------------------
                if prev_point is not None:
                    draw_capsule(
                        viewer.user_scn,
                        prev_point,
                        p_mj,
                        radius=0.003,
                        rgba=[0.1, 0.4, 1.0, 1.0]
                    )
                prev_point = p_mj.copy()

                # -----------------------------------------
                # Sai lệch so với đường thẳng lý tưởng start->goal
                # -----------------------------------------
                dev = point_to_segment_distance(p_mj, p_start, p_goal)
                deviations.append(dev)

                # -----------------------------------------
                # In vị trí EE
                # -----------------------------------------
                print(
                    "q =", np.round(q, 3),
                    "| EE (DH) =", np.round(p_DH, 4),
                    "| EE (MuJoCo) =", np.round(p_mj, 4),
                    "| lệch khỏi đường thẳng =", round(dev * 1000, 3), "mm"
                )

                # -----------------------------------------
                # Hiển thị MuJoCo
                # -----------------------------------------
                viewer.sync()
                time.sleep(0.02)

            if not viewer.is_running():
                break

        if not viewer.is_running():
            break

    # =====================================================
    # TỔNG KẾT SAI LỆCH
    # =====================================================
    if deviations:
        deviations = np.array(deviations)
        print("\n================================")
        print("TỔNG KẾT SAI LỆCH SO VỚI ĐƯỜNG THẲNG LÝ TƯỞNG")
        print("================================")
        print(f"Sai lệch lớn nhất : {deviations.max() * 1000:.3f} mm")
        print(f"Sai lệch trung bình: {deviations.mean() * 1000:.3f} mm")

    # Giữ viewer mở để quan sát kết quả cuối
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)


print("\nRobot đã kết thúc chuyển động.")