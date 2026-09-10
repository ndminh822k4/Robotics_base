"""
NMPC_mujoco.py — Nonlinear MPC (multiple shooting) dùng TRỰC TIẾP động
học phi tuyến của MuJoCo làm model dự báo, thay cho bản MPCController
tuyến tính (Control_function/MPC.py) đang tuyến tính hoá quanh M(q) tại
1 thời điểm rồi giữ cố định suốt cả horizon.

--------------------------------------------------------------------
SỰ KHÁC BIỆT VỚI MPCController (tuyến tính, condensed QP)
--------------------------------------------------------------------
Bản cũ:
    ẋ = A x + B u,   A = [[0,I],[0,0]],  B = [[0],[M(q0)^-1]]
    -> tuyến tính hoá 1 lần tại q hiện tại, DÙNG NGUYÊN B ĐÓ cho cả N
       bước dự báo phía trước (dù q sẽ đổi rất nhiều nếu robot di
       chuyển xa) -> M(q) "cứng" suốt horizon.
    -> giải bằng công thức đóng U* = -H^-1 g vì X = Sx x0 + Su U tuyến
       tính, không có ràng buộc cứng thật (torque_limit chỉ clip
       hậu-nghiệm).

Bản này (NMPC thật):
    q̈ = M(q)^-1 ( u - qfrc_bias(q, q̇) )
    -> qfrc_bias của MuJoCo đã tự gồm CẢ Coriolis/centrifugal VÀ
       gravity một cách chính xác (không phải suy diễn/bù riêng), và
       M(q) được TÍNH LẠI ở MỌI bước dự báo k=0..N-1 theo q dự báo tại
       bước đó (không đóng băng ở q hiện tại) -> đúng bản chất "phi
       tuyến" của robot khi di chuyển xa/đổi hướng nhanh.
    -> Biến quyết định gồm CẢ trạng thái x0..xN VÀ điều khiển u0..u_{N-1}
       (multiple shooting), ràng buộc động học là ràng buộc ĐẲNG THỨC
       (defect constraint) x_{k+1} - F(x_k,u_k) = 0, giải bằng NLP
       solver (SLSQP) theo lý thuyết MPC phi tuyến tổng quát.
    -> torque_limit là ràng buộc BẤT ĐẲNG THỨC THẬT trong NLP (bounds),
       không phải clip hậu-nghiệm.

API compute() giữ NGUYÊN chữ ký như MPCController.compute() để bạn chỉ
cần đổi phần khởi tạo + import trong Control_MPC.py, không phải sửa gì
trong main control loop.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize


# ============================================================
# XÂY dynamics_func(q, dq, u) -> ddq TỪ CHÍNH MODEL MUJOCO
# ============================================================

def make_mujoco_dynamics_func(model, n_q=6):
    """
    Trả về dynamics_func(q, dq, u) -> ddq dùng đúng M(q) và qfrc_bias(q,dq)
    (Coriolis + centrifugal + gravity) của MuJoCo tại TRẠNG THÁI DỰ BÁO
    (q, dq) — không phải trạng thái mô phỏng thật `data` đang chạy.

    Dùng một MjData "scratch" riêng (KHÔNG đụng vào `data` chính của
    vòng lặp mô phỏng) để có thể gọi hàm này nhiều lần với các (q,dq)
    giả định khác nhau trong lúc NLP dò nghiệm, mà không làm hỏng trạng
    thái mô phỏng thật.
    """
    scratch = mujoco.MjData(model)
    nv = model.nv

    def dynamics_func(q, dq, u):
        scratch.qpos[:n_q] = q
        scratch.qvel[:n_q] = dq
        mujoco.mj_forward(model, scratch)

        M_full = np.zeros((nv, nv))
        mujoco.mj_fullM(model, scratch, M_full)
        M = M_full[:n_q, :n_q]
        bias = scratch.qfrc_bias[:n_q].copy()

        ddq = np.linalg.solve(M, u - bias)
        return ddq

    return dynamics_func


# ============================================================
# TÍCH PHÂN ĐỘNG HỌC (RK4) — F(x_k, u_k)
# ============================================================

def _f(x, u, dynamics_func, n_q):
    q, dq = x[:n_q], x[n_q:]
    ddq = dynamics_func(q, dq, u)
    return np.concatenate([dq, ddq])


def _rk4_step(x, u, dynamics_func, dt, n_q):
    k1 = _f(x, u, dynamics_func, n_q)
    k2 = _f(x + 0.5 * dt * k1, u, dynamics_func, n_q)
    k3 = _f(x + 0.5 * dt * k2, u, dynamics_func, n_q)
    k4 = _f(x + dt * k3, u, dynamics_func, n_q)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class _Layout:
    """Index cho z = [x0..xN, u0..u_{N-1}] phẳng (1D) mà scipy cần."""

    def __init__(self, n_x, n_u, N):
        self.n_x, self.n_u, self.N = n_x, n_u, N
        self.n_states_block = n_x * (N + 1)

    def x_k(self, z, k):
        return z[k * self.n_x:(k + 1) * self.n_x]

    def u_k(self, z, k):
        off = self.n_states_block
        return z[off + k * self.n_u: off + (k + 1) * self.n_u]

    def pack(self, X_list, U_list):
        return np.concatenate(X_list + U_list)


# ============================================================
# API DẠNG CLASS — cùng "hình dáng" MPCController cũ để cắm thay thế 1-1
# ============================================================

class NMPCController:
    """
    Drop-in thay thế cho Control_function.MPC.MPCController, nhưng dùng
    NLP multiple-shooting + động học MuJoCo thật thay vì QP condensed
    tuyến tính hoá.

    Tham số:
        model         : mujoco.MjModel (dùng để xây dynamics_func nội bộ)
        Q, R, Qf       : giống MPCController cũ (vector 1D hoặc ma trận
                          vuông đều được, tự chuẩn hoá về ma trận)
        N, dt          : horizon và bước thời gian dự báo
        torque_limit   : (6,) — RÀNG BUỘC THẬT trong NLP (bounds), khác
                          bản cũ chỉ clip hậu-nghiệm
        n_q            : số khớp (mặc định 6)
        maxiter        : số vòng lặp tối đa cho SLSQP mỗi lần compute()
    """

    def __init__(self, model, Q, R, N=10, dt=0.01, Qf=None,
                 torque_limit=None, n_q=6, maxiter=60):
        self.model = model
        self.n_q = n_q
        self.n_x = 2 * n_q
        self.N = N
        self.dt = dt
        self.maxiter = maxiter

        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        if R.ndim == 1:
            R = np.diag(R)
        if Q.ndim == 1:
            Q = np.diag(Q)
        self.n_u = R.shape[0]
        self.Q = Q if Q.shape[0] == self.n_x else np.block([
            [Q, np.zeros((n_q, n_q))],
            [np.zeros((n_q, n_q)), np.zeros((n_q, n_q))]
        ])
        self.R = R

        if Qf is not None:
            Qf = np.asarray(Qf, dtype=float)
            if Qf.ndim == 1:
                Qf = np.diag(Qf)
            if Qf.shape[0] != self.n_x:
                Qf = np.block([
                    [Qf, np.zeros((n_q, n_q))],
                    [np.zeros((n_q, n_q)), np.zeros((n_q, n_q))]
                ])
            self.Qf = Qf
        else:
            self.Qf = self.Q

        self.torque_limit = np.asarray(torque_limit, dtype=float) if torque_limit is not None else None

        self.dynamics_func = make_mujoco_dynamics_func(model, n_q=n_q)
        self._layout = _Layout(self.n_x, self.n_u, N)
        self._z_prev = None  # warm start giữa các lần compute() (receding horizon)

    # -------------------------------------------------------
    def _warm_start(self, x_current):
        layout = self._layout
        N, n_u = layout.N, layout.n_u
        if self._z_prev is None:
            X_list = [x_current.copy() for _ in range(N + 1)]
            U_list = [np.zeros(n_u) for _ in range(N)]
            return layout.pack(X_list, U_list)

        z = self._z_prev
        X_list = [layout.x_k(z, k) for k in range(1, N + 1)] + [layout.x_k(z, N)]
        U_list = [layout.u_k(z, k) for k in range(1, N)] + [layout.u_k(z, N - 1)]
        X_list[0] = x_current.copy()
        return layout.pack(X_list, U_list)

    # -------------------------------------------------------
    def compute(self, q_des, q_current, dq_des, dq_current, M_now=None):
        """
        Chữ ký GIỐNG HỆT MPCController.compute() cũ để không phải sửa
        main loop trong Control_MPC.py.

        q_des, dq_des     : điểm đích cố định (6,) — lặp lại N lần thành
                              quỹ đạo tham chiếu (point-to-point NMPC)
        M_now             : KHÔNG dùng nữa — NMPC tự tính lại M(q) tại
                              từng bước dự báo bên trong dynamics_func,
                              chính xác hơn việc truyền M_now 1 lần.
        """
        layout = self._layout
        N, n_q, n_u = self.N, self.n_q, self.n_u

        x_current = np.concatenate([
            np.asarray(q_current, dtype=float),
            np.asarray(dq_current, dtype=float)
        ])

        Xref = [x_current]
        x_ref_step = np.concatenate([
            np.asarray(q_des, dtype=float),
            np.asarray(dq_des, dtype=float)
        ])
        for _ in range(N):
            Xref.append(x_ref_step)

        z0 = self._warm_start(x_current)

        def cost(z):
            J = 0.0
            for k in range(1, N):
                e = layout.x_k(z, k) - Xref[k]
                J += e @ self.Q @ e
            eN = layout.x_k(z, N) - Xref[N]
            J += eN @ self.Qf @ eN
            for k in range(N):
                u = layout.u_k(z, k)
                J += u @ self.R @ u
            return J

        def eq_constraints(z):
            residual = np.zeros(self.n_x * (N + 1))
            residual[:self.n_x] = layout.x_k(z, 0) - x_current
            for k in range(N):
                xk = layout.x_k(z, k)
                uk = layout.u_k(z, k)
                x_next_pred = _rk4_step(xk, uk, self.dynamics_func, self.dt, n_q)
                x_next_var = layout.x_k(z, k + 1)
                residual[self.n_x * (k + 1):self.n_x * (k + 2)] = x_next_var - x_next_pred
            return residual

        bounds = [(None, None)] * layout.n_states_block
        if self.torque_limit is not None:
            u_bounds = [(-abs(self.torque_limit[i]), abs(self.torque_limit[i])) for i in range(n_u)]
        else:
            u_bounds = [(None, None)] * n_u
        bounds += u_bounds * N

        res = minimize(
            cost, z0, method="SLSQP",
            constraints=[{"type": "eq", "fun": eq_constraints}],
            bounds=bounds,
            options={"maxiter": self.maxiter, "ftol": 1e-6}
        )

        self._z_prev = res.x
        torque = layout.u_k(res.x, 0)
        return torque