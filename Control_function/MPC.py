"""
MPC_theory.py — Model Predictive Control tuyến tính, dạng QP condensed,
xây dựng ĐÚNG theo lý thuyết MPC (không mượn cấu trúc LQR).

--------------------------------------------------------------------
1) MÔ HÌNH DỰ BÁO (linear, time-varying qua M(q) tại thời điểm hiện tại)
--------------------------------------------------------------------
Trạng thái TUYỆT ĐỐI (không phải sai số):
    x = [q; dq] ∈ R^12          (q, dq ∈ R^6)

Động học liên tục (tuyến tính hóa quanh M(q) hiện tại, giống robot
manipulator dynamics rút gọn phần quán tính, bỏ qua Coriolis/gravity vì
phần đó được cộng bù bên ngoài, giống Control_system.py):
    q̇  = dq
    d(dq)/dt = M(q)^-1 u
=>  ẋ = A x + B u,   A = [[0, I],[0, 0]],   B = [[0],[M^-1]]

Rời rạc Euler:  Ad = I + A dt,   Bd = B dt

--------------------------------------------------------------------
2) DỰ BÁO N BƯỚC, GỘP THÀNH DẠNG BATCH
--------------------------------------------------------------------
    X = Sx x0 + Su U,     X = [x1;...;xN],   U = [u0;...;u_{N-1}]

--------------------------------------------------------------------
3) HÀM CHI PHÍ MPC CHUẨN (stage cost + terminal cost, so với QUỸ ĐẠO
   THAM CHIẾU cho cả horizon, KHÔNG phải một điểm cân bằng cố định
   như LQR):

    J(U) = Σ_{k=1}^{N-1} (x_k - x_ref_k)^T Q (x_k - x_ref_k)
         +           (x_N - x_ref_N)^T Qf (x_N - x_ref_N)
         + Σ_{k=0}^{N-1} u_k^T R u_k

   Đây chính là hàm mục tiêu lý thuyết của MPC hữu hạn horizon
   (finite-horizon optimal control), KHÔNG phải hàm Lyapunov của LQR.
   Q, R, Qf ở đây là "trọng số cost của bài toán QP", không phải
   nghiệm của Riccati equation.

--------------------------------------------------------------------
4) NGHIỆM KHÔNG RÀNG BUỘC (đóng dạng, để dùng ngay; nếu cần ràng buộc
   |u| <= u_max thật sự thì phải giải QP có ràng buộc bằng
   quadprog/OSQP/qpsolvers — phần torque_limit hiện tại chỉ là clip
   hậu-nghiệm, KHÔNG phải nghiệm QP có ràng buộc đúng nghĩa).

    H = Su^T Qbar Su + Rbar
    g = Su^T Qbar (Sx x0 - Xref)
    U* = -H^-1 g
--------------------------------------------------------------------
"""

import numpy as np


# ============================================================
# MÔ HÌNH DỰ BÁO
# ============================================================

def discretize(M, dt):
    """Ad (12x12), Bd (12x6) từ M(q) hiện tại."""
    n_q = M.shape[0]
    Minv = np.linalg.inv(M)

    A = np.zeros((2 * n_q, 2 * n_q))
    A[:n_q, n_q:] = np.eye(n_q)

    B = np.zeros((2 * n_q, n_q))
    B[n_q:, :] = Minv

    Ad = np.eye(2 * n_q) + A * dt
    Bd = B * dt
    return Ad, Bd


def build_prediction_matrices(Ad, Bd, N):
    """X = Sx x0 + Su U."""
    n, m = Ad.shape[0], Bd.shape[1]
    Sx = np.zeros((n * N, n))
    Su = np.zeros((n * N, m * N))

    Ad_pow = [np.eye(n)]
    for _ in range(N):
        Ad_pow.append(Ad_pow[-1] @ Ad)

    for k in range(N):
        Sx[k * n:(k + 1) * n, :] = Ad_pow[k + 1]
        for j in range(k + 1):
            Su[k * n:(k + 1) * n, j * m:(j + 1) * m] = Ad_pow[k - j] @ Bd
    return Sx, Su


# ============================================================
# HÀM CHI PHÍ MPC (stage cost + terminal cost so với QUỸ ĐẠO THAM CHIẾU)
# ============================================================

def build_cost_matrices(Q, R, N, Qf=None):
    """Qbar = blkdiag(Q,...,Q,Qf), Rbar = blkdiag(R,...,R)."""
    n, m = Q.shape[0], R.shape[0]
    Qbar = np.kron(np.eye(N), Q)
    if Qf is not None:
        Qbar[(N - 1) * n:N * n, (N - 1) * n:N * n] = Qf
    Rbar = np.kron(np.eye(N), R)
    return Qbar, Rbar


def build_reference_trajectory(q_des_traj, dq_des_traj):
    """
    Xref cho cả horizon, ĐÂY LÀ ĐIỂM KHÁC BIỆT LỚN NHẤT với bản LQR-style:
    thay vì trừ sẵn sai số tại state, MPC giữ trạng thái tuyệt đối và so
    với một QUỸ ĐẠO THAM CHIẾU x_ref_k = [q_des_k; dq_des_k] theo từng
    bước dự báo k = 1..N.

    q_des_traj, dq_des_traj: (N,6) — quỹ đạo mong muốn tại các bước dự báo
        1..N. Nếu robot chỉ có 1 điểm đích cố định (point-to-point), bạn
        có thể lặp lại q_des N lần — vẫn đúng lý thuyết MPC, chỉ là quỹ
        đạo tham chiếu "hằng số theo thời gian".

    Trả về Xref (12N,).
    """
    q_des_traj = np.asarray(q_des_traj, dtype=float)
    dq_des_traj = np.asarray(dq_des_traj, dtype=float)
    N = q_des_traj.shape[0]
    Xref = np.zeros(12 * N)
    for k in range(N):
        Xref[k * 12:k * 12 + 6] = q_des_traj[k]
        Xref[k * 12 + 6:k * 12 + 12] = dq_des_traj[k]
    return Xref


# ============================================================
# GIẢI QP CONDENSED KHÔNG RÀNG BUỘC
# ============================================================

def solve_mpc_qp(Sx, Su, Qbar, Rbar, x0, Xref):
    """U* = -(Su^T Qbar Su + Rbar)^-1 Su^T Qbar (Sx x0 - Xref)."""
    H = Su.T @ Qbar @ Su + Rbar
    g = Su.T @ Qbar @ (Sx @ x0 - Xref)
    U = -np.linalg.solve(H, g)
    return H, U


# ============================================================
# API SỬ DỤNG: điều khiển robot bám quỹ đạo giữa 2 vị trí
# ============================================================

def MPCControl(q_current, dq_current, q_des_traj, dq_des_traj,
               M_now, Q, R, dt=0.01, Qf=None, torque_limit=None):
    """
    Một bước MPC receding-horizon, giải lại QP mỗi lần gọi.

    q_current, dq_current : trạng thái tuyệt đối hiện tại (6,)
    q_des_traj, dq_des_traj: quỹ đạo tham chiếu (N,6) cho N bước dự báo
                              tới (bước 1..N kể từ hiện tại)
    M_now  : M(q_current) (6x6)
    Q      : (6,6) hoặc (12,12) — trọng số cost stage cho x=[q;dq]
    R      : (6,6) — trọng số cost cho u=tau
    Qf     : (12,12) tuỳ chọn — trọng số cost terminal
    torque_limit: (6,) tuỳ chọn — clip hậu-nghiệm (không phải QP-constrained thật)

    Trả về torque (6,) = u0, và U (6N,) toàn bộ chuỗi dự báo.
    """
    N = q_des_traj.shape[0]
    Ad, Bd = discretize(M_now, dt)
    Sx, Su = build_prediction_matrices(Ad, Bd, N)

    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        R = np.diag(R)
    if Q.ndim == 1:
        Q = np.diag(Q)
    if Qf is not None:
        Qf = np.asarray(Qf, dtype=float)
        if Qf.ndim == 1:
            Qf = np.diag(Qf)

    Q_mat = Q if Q.shape[0] == 12 else np.block([
        [Q, np.zeros((6, 6))],
        [np.zeros((6, 6)), np.zeros((6, 6))]
    ])
    Qbar, Rbar = build_cost_matrices(Q_mat, R, N, Qf=Qf)

    x0 = np.concatenate([q_current, dq_current])
    Xref = build_reference_trajectory(q_des_traj, dq_des_traj)

    _, U = solve_mpc_qp(Sx, Su, Qbar, Rbar, x0, Xref)

    m = R.shape[0]
    torque = U[:m]
    if torque_limit is not None:
        torque = np.clip(torque, -np.abs(torque_limit), np.abs(torque_limit))
    return torque, U


# ============================================================
# API DẠNG CLASS — dùng trong vòng lặp điều khiển (receding horizon)
# ============================================================

class MPCController:
    """
    Wrapper tiện dùng trong control loop, cùng "hình dáng" gọi hàm như
    MPCController cũ (Control_function/MPC.py) để bạn thay thế 1-1 trong
    Control3.py, nhưng bên trong dùng ĐÚNG cost function lý thuyết MPC
    (so với quỹ đạo tham chiếu x_ref, không phải sai số lồng sẵn vào state).

    Vì M(q) đổi theo từng bước, controller này GIẢI LẠI Sx/Su/H mỗi lần
    compute() được gọi nếu bạn truyền M_now mới (giống LQR_RECOMPUTE_
    EVERY_STEP=True). Nếu bạn không truyền M_now, nó tái dùng M0 ban đầu
    (fixed-point, rẻ hơn nhưng kém chính xác khi q đổi nhiều).
    """

    def __init__(self, M0, Q, R, N=10, dt=0.01, Qf=None, torque_limit=None):
        self.N = N
        self.dt = dt
        self.torque_limit = torque_limit

        # Cho phép truyền Q, R, Qf dạng vector 1 chiều (đường chéo) hoặc
        # ma trận đầy đủ sẵn — tự chuẩn hóa về ma trận vuông để tránh lỗi
        # shape khi build_cost_matrices/np.kron.
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        if R.ndim == 1:
            R = np.diag(R)
        if Q.ndim == 1:
            Q = np.diag(Q)

        self.m = R.shape[0]

        self.Q = Q if Q.shape[0] == 12 else np.block([
            [Q, np.zeros((6, 6))],
            [np.zeros((6, 6)), np.zeros((6, 6))]
        ])
        self.R = R

        if Qf is not None:
            Qf = np.asarray(Qf, dtype=float)
            if Qf.ndim == 1:
                Qf = np.diag(Qf)
            if Qf.shape[0] != 12:
                Qf = np.block([
                    [Qf, np.zeros((6, 6))],
                    [np.zeros((6, 6)), np.zeros((6, 6))]
                ])
        self.Qf = Qf
        self.M0 = M0

        # Cache cho trường hợp fixed-point (không truyền M_now mỗi bước)
        Ad, Bd = discretize(M0, dt)
        self.Sx0, self.Su0 = build_prediction_matrices(Ad, Bd, N)
        self.Qbar0, self.Rbar0 = build_cost_matrices(self.Q, self.R, N, Qf=self.Qf)

    def compute(self, q_des, q_current, dq_des, dq_current, M_now=None):
        """
        q_des, dq_des: điểm đích cố định (6,) — quỹ đạo tham chiếu được
                       tạo bằng cách lặp lại điểm này N lần (point-to-point
                       MPC). Nếu bạn có quỹ đạo trung gian mượt hơn, dùng
                       hàm MPCControl() ở trên với q_des_traj/dq_des_traj
                       thay vì class này.
        M_now         : nếu truyền vào (khuyến nghị, vì M(q) đổi theo q),
                        sẽ giải lại Sx/Su/H mỗi bước — giống MPCControl().
                        Nếu bỏ trống, dùng lại ma trận đã cache từ M0.
        """
        N = self.N
        q_des_traj = np.tile(np.asarray(q_des, dtype=float), (N, 1))
        dq_des_traj = np.tile(np.asarray(dq_des, dtype=float), (N, 1))
        Xref = build_reference_trajectory(q_des_traj, dq_des_traj)

        x0 = np.concatenate([np.asarray(q_current, dtype=float),
                              np.asarray(dq_current, dtype=float)])

        if M_now is not None:
            Ad, Bd = discretize(M_now, self.dt)
            Sx, Su = build_prediction_matrices(Ad, Bd, N)
            Qbar, Rbar = build_cost_matrices(self.Q, self.R, N, Qf=self.Qf)
        else:
            Sx, Su, Qbar, Rbar = self.Sx0, self.Su0, self.Qbar0, self.Rbar0

        _, U = solve_mpc_qp(Sx, Su, Qbar, Rbar, x0, Xref)
        torque = U[:self.m]

        if self.torque_limit is not None:
            torque = np.clip(torque, -np.abs(self.torque_limit), np.abs(self.torque_limit))
        return torque


if __name__ == "__main__":
    # Demo: di chuyển robot 6 khớp từ q_current=0 tới q_des cố định.
    # Quỹ đạo tham chiếu = lặp lại điểm đích N lần (point-to-point MPC).
    N = 10
    dt = 0.01
    M_demo = np.diag([8.0, 6.0, 4.0, 2.0, 1.0, 0.5])

    q_target = np.deg2rad([30, 40, 50, 60, 70, 80])
    q_current = np.zeros(6)
    dq_current = np.zeros(6)

    q_des_traj = np.tile(q_target, (N, 1))
    dq_des_traj = np.zeros((N, 6))

    # Trọng số cost MPC: phạt lệch vị trí + vận tốc so với quỹ đạo tham chiếu
    Q = np.diag(np.concatenate([np.full(6, 500.0), np.full(6, 5.0)]))
    R = np.diag(np.full(6, 0.05))

    torque, U_full = MPCControl(q_current, dq_current, q_des_traj, dq_des_traj,
                                 M_demo, Q, R, dt=dt)
    print("u0 (torque áp dụng ngay):", torque)