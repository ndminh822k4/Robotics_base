import os
import sys
import numpy as np
from spatialmath import SE3
import time
import mujoco
import mujoco.viewer


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(BASE_DIR)

from Forward import forward_kinematics, forward_kinematics_all

# ==========================================
# JACOBIAN HINH HOC (SPACE JACOBIAN, 6x6)
# ==========================================

def jacobian_6dof(T_list):
    """
    T_list: [T0, T1, ..., T6] tu forward_kinematics_all(q)
        T_list[i][:3, 3]  -> vi tri origin cua frame i (truoc khop i+1)
        T_list[i][:3, :3] -> rotation cua frame i

    Tra ve:
        J: Jacobian 6x6 (space Jacobian, bieu dien trong he BASE)
           J[:3, :] -> linear velocity Jacobian
           J[3:, :] -> angular velocity Jacobian
    """
    n = 6
    J = np.zeros((6, n))

    p_e = T_list[6][:3, 3]  # vi tri end-effector

    for i in range(n):
        T_i = T_list[i]
        z_i = T_i[:3, 2]
        p_i = T_i[:3, 3]

        # Joint quay (revolute)
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i

    return J


# ==========================================
# NEWTON-RAPHSON IK (co damping cho on dinh)
# ==========================================

def Newton_Raphson_inverse_kinematics(
    T_target,
    q_init,
    max_iterations=200,
    tolerance=1e-6,
    damping=1e-3,
    step_clip=0.3
):
    """
    T_target: SE3 target
    q_init: initial joint angles (radians)

    Tra ve:
        q_solution: joint angles dat T_target (radian)
    """
    q = np.array(q_init, dtype=float)

    for iteration in range(max_iterations):

        T_list = forward_kinematics_all(q)   # [T0..T6], khong lam tron
        T_current_SE3 = SE3(T_list[6], check=False)

        # QUAN TRONG: sai so phai tinh trong he BASE (spatial/space twist)
        # de khop voi Jacobian khong gian o tren, KHONG dung
        # T_current.inv() * T_target (do la body twist, sai he quy chieu).
        error_SE3 = T_target * T_current_SE3.inv()
        error_vector = error_SE3.log(twist=True)

        err_norm = np.linalg.norm(error_vector)
        if err_norm < tolerance:
            return q, iteration, err_norm

        J = jacobian_6dof(T_list)

        # Damped least squares (Levenberg-Marquardt) thay vi pinv thuan tuy,
        # de tranh buoc nhay qua lon / phan ky gan diem ky di (singularity).
        JT = J.T
        delta_q = JT @ np.linalg.solve(
            J @ JT + (damping ** 2) * np.eye(6),
            error_vector
        )

        # Gioi han do lon moi buoc de tranh vong lap dau tien nhay qua xa
        step_norm = np.linalg.norm(delta_q)
        if step_norm > step_clip:
            delta_q = delta_q * (step_clip / step_norm)

        q = q + delta_q

    raise ValueError(
        f"IK khong hoi tu sau {max_iterations} vong lap "
        f"(sai so con lai = {err_norm:.6f})."
    )

# ==========================================
# JACOBIAN HINH HOC (SPACE JACOBIAN, 6x6)
# ==========================================

def jacobian_6dof(T_list):
    """
    T_list: [T0, T1, ..., T6] tu forward_kinematics_all(q)
        T_list[i][:3, 3]  -> vi tri origin cua frame i (truoc khop i+1)
        T_list[i][:3, :3] -> rotation cua frame i

    Tra ve:
        J: Jacobian 6x6 (space Jacobian, bieu dien trong he BASE)
           J[:3, :] -> linear velocity Jacobian
           J[3:, :] -> angular velocity Jacobian
    """
    n = 6
    J = np.zeros((6, n))

    p_e = T_list[6][:3, 3]  # vi tri end-effector

    for i in range(n):
        T_i = T_list[i]
        z_i = T_i[:3, 2]
        p_i = T_i[:3, 3]

        # Joint quay (revolute)
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i

    return J


# ==========================================
# NEWTON-RAPHSON IK (co damping cho on dinh)
# ==========================================

def DLS(
    T_target,
    q_init,
    max_iterations=200,
    tolerance=1e-6,
    damping=1e-3,
    step_clip=0.3
):
    """
    T_target: SE3 target
    q_init: initial joint angles (radians)

    Tra ve:
        q_solution: joint angles dat T_target (radian)
    """
    q = np.array(q_init, dtype=float)

    for iteration in range(max_iterations):
        T_list = forward_kinematics_all(q)   # [T0..T6], khong lam tron
        T_current_SE3 = SE3(T_list[6], check=False)

        # QUAN TRONG: sai so phai tinh trong he BASE (spatial/space twist)
        # de khop voi Jacobian khong gian o tren, KHONG dung
        # T_current.inv() * T_target (do la body twist, sai he quy chieu).
        error_SE3 = T_target * T_current_SE3.inv()
        error_vector = error_SE3.log(twist=True)

        err_norm = np.linalg.norm(error_vector)
        if err_norm < tolerance:
            return q, iteration, err_norm

        J = jacobian_6dof(T_list)

        # Damped least squares (Levenberg-Marquardt) thay vi pinv thuan tuy,
        # de tranh buoc nhay qua lon / phan ky gan diem ky di (singularity).
        JT = J.T
        delta_q = JT @ np.linalg.solve(
            J @ JT + (damping ** 2) * np.eye(6),
            error_vector
        )

        # Gioi han do lon moi buoc de tranh vong lap dau tien nhay qua xa
        step_norm = np.linalg.norm(delta_q)
        if step_norm > step_clip:
            delta_q = delta_q * (step_clip / step_norm)

        q = q + delta_q

    raise ValueError(
        f"IK khong hoi tu sau {max_iterations} vong lap "
        f"(sai so con lai = {err_norm:.6f})."
    )


# ==========================================
# GIAI IK BANG JACOBIAN TICH HOP CUA MUJOCO
# ==========================================

def mujoco_ik(
    model, data, site_name, target_pos, target_mat=None,
    q_init=None, max_iters=200, tol=1e-4, step_size=0.5
):
    """
    Giai Inverse Kinematics su dung Jacobian tich hop cua MuJoCo
    (mj_jacSite) + Damped Least Squares.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Khong tim thay site '{site_name}' trong model.")

    # 1. Dat cau hinh goc ban dau
    if q_init is not None:
        data.qpos[:len(q_init)] = q_init

    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))

    # Gioi han khop lay tu chinh model (KHONG the dua vao mj_normalizeQuat -
    # ham do chi chuan hoa quaternion cua free joint, khong lien quan gi
    # den joint limit ca).
    qmin = model.jnt_range[:, 0]
    qmax = model.jnt_range[:, 1]
    has_limit = model.jnt_limited.astype(bool)

    success = False

    for iteration in range(max_iters):
        # Cap nhat Forward Kinematics trong MuJoCo
        mujoco.mj_forward(model, data)

        # Lay vi tri va huong hien tai cua site
        current_pos = data.site_xpos[site_id]
        err_pos = target_pos - current_pos

        # Tinh toan sai so huong (neu co yeu cau huong target_mat 3x3)
        if target_mat is not None:
            current_mat = data.site_xmat[site_id].reshape(3, 3)
            err_rot = 0.5 * (
                np.cross(current_mat[:, 0], target_mat[:, 0]) +
                np.cross(current_mat[:, 1], target_mat[:, 1]) +
                np.cross(current_mat[:, 2], target_mat[:, 2])
            )
            error_vector = np.concatenate([err_pos, err_rot])

            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
            J = np.vstack([jac_pos, jac_rot])
        else:
            error_vector = err_pos
            mujoco.mj_jacSite(model, data, jac_pos, None, site_id)
            J = jac_pos

        # Kiem tra dieu kien hoi tu
        if np.linalg.norm(error_vector) < tol:
            success = True
            break

        # Giai bang Damped Least Squares (DLS)
        diag = 1e-4 * np.eye(J.shape[0])
        delta_q = J.T @ np.linalg.solve(J @ J.T + diag, error_vector)

        # Cap nhat goc khop
        data.qpos[:model.nv] += step_size * delta_q

        # Ap gioi han khop THUC SU (clip ve [qmin, qmax] cho tung khop co limit)
        data.qpos[has_limit] = np.clip(
            data.qpos[has_limit],
            qmin[has_limit],
            qmax[has_limit]
        )

    mujoco.mj_forward(model, data)
    return data.qpos.copy(), success



def move_robot_direct(
    model, data, q_start, q_target, site_name="attachment_site"
):
    """Gán thẳng mục tiêu và để bộ điều khiển PD tự kéo robot đến đích."""
    # 1. Đặt trạng thái ban đầu
    data.qpos[: len(q_start)] = q_start
    mujoco.mj_forward(model, data)

    model.opt.gravity[:] = 0.0  # Tat gravity de robot tu dong keo khop bang PD controller

    # 2. Gán thẳng góc đích cho bộ điều khiển
    data.ctrl[: len(q_target)] = q_target

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Bước mô phỏng để PD controller tự sinh lực kéo khớp
            mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(model.opt.timestep)



if __name__ == "__main__":
    # Diem muc tieu (x=0.3m, y=0.2m, z=0.4m)
    T_target = SE3.Trans(0.2, -0.3, 0.4) * SE3.RPY(0, np.pi / 2, 0, unit='rad')

    # Goc doan ban dau (6 khop)
    T_start = SE3.Trans(0.087, 0.0, 0.1536) * SE3.RPY(np.pi, 0, 0, unit='rad')
    q_start, n_iter, err = DLS(T_target, q_init = np.zeros(6) )
    q_start2, _, _ = Newton_Raphson_inverse_kinematics(T_target, q_init=np.zeros(6))
    print("DLS:", np.degrees(q_start))


    
