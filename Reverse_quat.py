import os
import sys
import numpy as np
import mujoco
import mujoco.viewer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Chi lay thong so DH va joint_limits, KHONG dung dh_transform (ma tran T)
from Forward import d1, a2, a3, d4, d5, d6, joint_limits

# ==========================================
# QUATERNION UTILS (thuan numpy)
# Quy uoc: q = [w, x, y, z]
# ==========================================

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_multiply(q1, q2):
    """q1 (x) q2 - thu tu quan trong (khong giao hoan)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_rotate_vector(q, v):
    """Quay vector v boi quaternion q (khong can dung ma tran R)."""
    w = q[0]
    qv = q[1:]
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (w * uv + uuv)


def qz(theta):
    """Quaternion quay quanh truc z mot goc theta."""
    return np.array([np.cos(theta/2), 0.0, 0.0, np.sin(theta/2)])


def qx(alpha):
    """Quaternion quay quanh truc x mot goc alpha."""
    return np.array([np.cos(alpha/2), np.sin(alpha/2), 0.0, 0.0])


def quat_error_vec(q_target, q_current):
    """
    Sai so huong dang vector 3D (tuyen tinh hoa) tu quaternion error.
    Tu dong chon duong quay ngan nhat (shortest path).
    """
    q_target = quat_normalize(q_target)
    q_current = quat_normalize(q_current)

    q_err = quat_multiply(q_target, quat_conjugate(q_current))

    if q_err[0] < 0:
        q_err = -q_err

    return 2.0 * q_err[1:]


# ==========================================
# FORWARD KINEMATICS THUAN QUATERNION
# (khong tao ma tran 4x4 nao ca)
# ==========================================

def dh_step_quat(theta, d, a, alpha):
    """
    Tra ve (q_local, t_local) tuong duong 1 khau DH:
        A = Rz(theta) * Trans(a, 0, d) * Rx(alpha)
    q_local: quaternion phan quay cuc bo
    t_local: vector tinh tien cuc bo [a*cos(theta), a*sin(theta), d]
    """
    q_local = quat_multiply(qz(theta), qx(alpha))
    t_local = np.array([a * np.cos(theta), a * np.sin(theta), d])
    return q_local, t_local


def forward_kinematics_quat_all(q, joint_limits=joint_limits):
    """
    Tra ve 2 danh sach song song (thay cho T_list):
        P_list = [p0, p1, ..., p6]   vi tri tung frame (base -> EE)
        Q_list = [q0, q1, ..., q6]   quaternion tung frame (base -> EE)

    p0 = [0,0,0], q0 = [1,0,0,0]  (frame base)
    p6, q6                          (frame end-effector)

    Dung cho Jacobian: frame i la frame ma khop (i+1) quay quanh
    truc z cua no (quy uoc DH chuan).
    """
    for i, (theta, limits) in enumerate(zip(q, joint_limits)):
        if not (limits[0] <= theta <= limits[1]):
            raise ValueError(
                f"Joint {i+1} angle {np.degrees(theta):.2f}° out of limits "
                f"{np.degrees(limits[0]):.2f}° to {np.degrees(limits[1]):.2f}°."
            )

    theta1, theta2, theta3, theta4, theta5, theta6 = q

    dh_params = [
        (theta1,            d1, 0,  -np.pi/2),
        (theta2 - np.pi/2,  0,  a2,  np.pi),
        (theta3 - np.pi/2,  0,  a3,  np.pi/2),
        (theta4,            d4, 0,   np.pi/2),
        (theta5,            d5, 0,  -np.pi/2),
        (theta6,            d6, 0,   0),
    ]

    P_list = [np.zeros(3)]
    Q_list = [np.array([1.0, 0.0, 0.0, 0.0])]

    p, q_ = P_list[0], Q_list[0]
    for (th, d, a, al) in dh_params:
        q_local, t_local = dh_step_quat(th, d, a, al)

        p = p + quat_rotate_vector(q_, t_local)
        q_ = quat_normalize(quat_multiply(q_, q_local))

        P_list.append(p)
        Q_list.append(q_)

    return P_list, Q_list


# ==========================================
# JACOBIAN HINH HOC THUAN QUATERNION
# (khong dung T[:3,2] hay T[:3,3] tu ma tran nua,
#  ma dung truc tiep p_i, q_i)
# ==========================================

def jacobian_6dof_quat(P_list, Q_list):
    """
    P_list, Q_list: tu forward_kinematics_quat_all(q)

    Tra ve J 6x6 (space Jacobian, bieu dien trong he BASE):
        J[:3, :] -> linear velocity Jacobian
        J[3:, :] -> angular velocity Jacobian
    """
    n = 6
    J = np.zeros((6, n))

    p_e = P_list[6]

    for i in range(n):
        # truc z cua frame i, bieu dien trong he base = quay [0,0,1] boi q_i
        z_i = quat_rotate_vector(Q_list[i], np.array([0.0, 0.0, 1.0]))
        p_i = P_list[i]

        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i

    return J


# ==========================================
# DLS IK - DAU VAO/DAU RA THUAN QUATERNION
# ==========================================

def DLS_quaternion(
    target_pos,
    target_quat,
    q_init,
    max_iterations=200,
    tolerance=1e-6,
    damping=1e-3,
    step_clip=0.3,
    pos_weight=1.0,
    ori_weight=1.0,
):
    """
    target_pos:  vector 3, vi tri dich
    target_quat: quaternion [w,x,y,z], huong dich (khong can chuan hoa truoc)
    q_init:      goc khop ban dau (radian)

    Tra ve: q, iteration, err_norm
    """
    q = np.array(q_init, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    target_quat = quat_normalize(np.asarray(target_quat, dtype=float))

    W = np.diag([pos_weight]*3 + [ori_weight]*3)

    err_norm = np.inf
    for iteration in range(max_iterations):
        P_list, Q_list = forward_kinematics_quat_all(q)

        current_pos = P_list[6]
        current_quat = Q_list[6]

        e_pos = target_pos - current_pos
        e_ori = quat_error_vec(target_quat, current_quat)
        error_vector = np.concatenate([e_pos, e_ori])

        err_norm = np.linalg.norm(error_vector)
        if err_norm < tolerance:
            return q, iteration, err_norm

        J = jacobian_6dof_quat(P_list, Q_list)

        Jw = W @ J
        ew = W @ error_vector

        JT = Jw.T
        delta_q = JT @ np.linalg.solve(
            Jw @ JT + (damping ** 2) * np.eye(6),
            ew
        )

        step_norm = np.linalg.norm(delta_q)
        if step_norm > step_clip:
            delta_q = delta_q * (step_clip / step_norm)

        q = q + delta_q

    raise ValueError(
        f"IK (quaternion) khong hoi tu sau {max_iterations} vong lap "
        f"(sai so con lai = {err_norm:.6f})."
    )


if __name__ == "__main__":
    # Vi du: dau vao THUAN quaternion, khong dung T o dau ca
    target_pos = np.array([0.2, -0.3, 0.4])

    theta = np.pi / 2
    target_quat = np.array([np.cos(theta/2), 0.0, np.sin(theta/2), 0.0])  # quay 90 deg quanh y

    q_init = np.zeros(6)

    q_sol, n_iter, err = DLS_quaternion(target_pos, target_quat, q_init)

    print("So vong lap:", n_iter)
    print("Sai so cuoi:", err)
    print("Goc khop giai duoc (deg):", np.degrees(q_sol))

    # Kiem tra lai bang chinh forward_kinematics_quat_all (khong dung T)
    P_check, Q_check = forward_kinematics_quat_all(q_sol)
    print("\nVi tri EE dat duoc:", P_check[6])
    print("Vi tri EE mong muon:", target_pos)
    print("\nQuaternion EE dat duoc:", Q_check[6])
    print("Quaternion EE mong muon:", target_quat)