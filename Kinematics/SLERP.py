import numpy as np
# ============================================================
# THÊM: SLERP CHO NỘI SUY HƯỚNG TRONG KHÔNG GIAN CARTESIAN
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
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
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


def generate_cartesian_trajectory(P_trajectory, R_trajectory, num_points_per_segment=20):
    """
    Nội suy vị trí (LERP) + hướng (SLERP) giữa các waypoint,
    trả về danh sách các SE3 (T_i) mượt cho từng đoạn.

    P_trajectory: (N, 3) - vị trí các waypoint
    R_trajectory: list N ma trận xoay 3x3 tương ứng mỗi waypoint
    """
    T_list = []

    for i in range(len(P_trajectory) - 1):
        p0, p1 = P_trajectory[i], P_trajectory[i + 1]
        q0 = rotmat_to_quat(R_trajectory[i])
        q1 = rotmat_to_quat(R_trajectory[i + 1])

        for k in range(num_points_per_segment):
            t = k / (num_points_per_segment - 1)

            p_interp = (1 - t) * p0 + t * p1          # LERP vị trí
            q_interp = slerp(q0, q1, t)                # SLERP hướng
            R_interp = quat_to_rotmat(q_interp)

            T = SE3.Rt(R_interp, p_interp)
            T_list.append(T)

    return T_list


def solve_cartesian_ik_trajectory(T_list, q_init, warm_start=True):
    """
    Giải IK (DLS) cho từng T trong T_list, dùng warm-start từ nghiệm trước
    để đảm bảo chuỗi q liên tục, không nhảy giữa các nghiệm IK khác nhau.
    """
    q_trajectory = []
    q_current = q_init.copy()

    for i, T in enumerate(T_list):
        q_seed = q_current if warm_start else q_init
        q_current, _, _ = DLS(T, q_init=q_seed, max_iterations=200)
        q_trajectory.append(q_current)

    return np.array(q_trajectory)