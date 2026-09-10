import numpy as np


def cubic_trajectory(q_start, q_goal, T=2.0, N=100):
    """
    Tạo quỹ đạo đa thức bậc 3 cho robot nhiều khớp.

    Parameters
    ----------
    q_start : array-like
        Góc khớp ban đầu, đơn vị rad.
    q_goal : array-like
        Góc khớp cuối, đơn vị rad.
    T : float
        Tổng thời gian chuyển động (s).
    N : int
        Số điểm trên quỹ đạo.

    Returns
    -------
    t : ndarray
        Vector thời gian, shape (N,).

    q_trajectory : ndarray
        Quỹ đạo vị trí khớp, shape (N, n_joints).

    qd_trajectory : ndarray
        Quỹ đạo vận tốc khớp, shape (N, n_joints).

    qdd_trajectory : ndarray
        Quỹ đạo gia tốc khớp, shape (N, n_joints).
    """

    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    # Vector thời gian
    t = np.linspace(0, T, N)

    # Sai khác vị trí
    dq = q_goal - q_start

    # Hệ số đa thức bậc 3
    a0 = q_start
    a1 = np.zeros_like(q_start)
    a2 = 3 * dq / T**2
    a3 = -2 * dq / T**3

    # Reshape để broadcasting
    t_col = t[:, None]

    # Vị trí
    q_trajectory = (
        a0
        + a1 * t_col
        + a2 * t_col**2
        + a3 * t_col**3
    )

    # Vận tốc
    qd_trajectory = (
        a1
        + 2 * a2 * t_col
        + 3 * a3 * t_col**2
    )

    # Gia tốc
    qdd_trajectory = (
        2 * a2
        + 6 * a3 * t_col
    )

    return t, q_trajectory, qd_trajectory, qdd_trajectory



def quintic_trajectory(q_start, q_goal, T=3.0, N=200):
    """
    Tạo quỹ đạo đa thức bậc 5 cho robot nhiều khớp.

    Điều kiện biên:
        q(0)   = q_start
        qd(0)  = 0
        qdd(0) = 0

        q(T)   = q_goal
        qd(T)  = 0
        qdd(T) = 0

    Parameters
    ----------
    q_start : array-like
        Góc khớp ban đầu, rad.

    q_goal : array-like
        Góc khớp cuối, rad.

    T : float
        Thời gian chuyển động, giây.

    N : int
        Số điểm trên quỹ đạo.

    Returns
    -------
    t : ndarray
        Thời gian, shape (N,)

    q : ndarray
        Vị trí khớp, shape (N, n_joints)

    qd : ndarray
        Vận tốc khớp, shape (N, n_joints)

    qdd : ndarray
        Gia tốc khớp, shape (N, n_joints)
    """

    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    # Vector thời gian
    t = np.linspace(0, T, N)

    # Độ thay đổi góc
    dq = q_goal - q_start

    # =========================================================
    # HỆ SỐ ĐA THỨC BẬC 5
    # =========================================================

    a0 = q_start
    a1 = np.zeros_like(q_start)
    a2 = np.zeros_like(q_start)

    a3 = 10 * dq / T**3
    a4 = -15 * dq / T**4
    a5 = 6 * dq / T**5

    # Đưa t về dạng (N, 1) để broadcasting với 6 joint
    t = t[:, None]

    # =========================================================
    # VỊ TRÍ
    # =========================================================

    q = (
        a0
        + a1 * t
        + a2 * t**2
        + a3 * t**3
        + a4 * t**4
        + a5 * t**5
    )

    # =========================================================
    # VẬN TỐC
    # dq/dt
    # =========================================================

    qd = (
        a1
        + 2 * a2 * t
        + 3 * a3 * t**2
        + 4 * a4 * t**3
        + 5 * a5 * t**4
    )

    # =========================================================
    # GIA TỐC
    # d²q/dt²
    # =========================================================

    qdd = (
        2 * a2
        + 6 * a3 * t
        + 12 * a4 * t**2
        + 20 * a5 * t**3
    )

    return t.flatten(), q, qd, qdd
