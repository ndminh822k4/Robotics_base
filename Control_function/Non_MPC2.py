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


def _rollout(x0, U, dynamics_func, dt, N, n_q):
    """
    SINGLE SHOOTING: X không phải biến quyết định, mà được TÍNH RA bằng
    cách mô phỏng xuôi (rollout) động học phi tuyến từ x0 với chuỗi U
    đang thử — khác multiple shooting ở chỗ:

        x_{k+1} = F(x_k, u_k),   x_0 = x hiện tại

    KHÔNG có ràng buộc đẳng thức nào cả, vì X luôn tự động khớp động
    học theo đúng định nghĩa (thế bằng đệ quy, không cần ép qua NLP).
    Đổi lại: mất khả năng ràng buộc trực tiếp lên x (chỉ ràng buộc được
    u), và có thể kém ổn định số hơn multiple shooting khi N lớn/hệ rất
    nhạy (lỗi tích luỹ qua rollout). Bù lại: số biến quyết định giảm từ
    n_x*(N+1)+n_u*N xuống còn đúng n_u*N, và KHÔNG còn ràng buộc đẳng
    thức -> solver không phải ước lượng Jacobian của (N+1)*n_x residual
    mỗi vòng lặp nữa -> nhẹ đi rất nhiều, đây là điều bạn cần khi chi
    phí tính toán của multiple shooting quá cao cho vòng lặp thời gian
    thực.
    """
    X = [x0]
    x = x0
    for k in range(N):
        x = _rk4_step(x, U[k], dynamics_func, dt, n_q)
        X.append(x)
    return X  # list gồm N+1 phần tử: X[0]=x0 .. X[N]


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
        self._U_prev = None  # warm start giữa các lần compute() (receding horizon)

    # -------------------------------------------------------
    def _warm_start(self):
        """U_prev dịch đi 1 bước (u1..u_{N-1} rồi lặp lại u cuối) —
        vẫn đúng tinh thần receding horizon dù giờ chỉ có U là biến."""
        N, n_u = self.N, self.n_u
        if self._U_prev is None:
            return np.zeros(N * n_u)
        U_prev = self._U_prev.reshape(N, n_u)
        U0 = np.vstack([U_prev[1:], U_prev[-1:]])
        return U0.reshape(-1)

    # -------------------------------------------------------
    def compute(self, q_des, q_current, dq_des, dq_current, M_now=None):
        """
        Chữ ký GIỐNG HỆT MPCController.compute() cũ để không phải sửa
        main loop trong Control_MPC.py.

        SINGLE SHOOTING: biến quyết định CHỈ CÒN U = [u0..u_{N-1}] (đúng
        yêu cầu — vì multiple shooting (cả X lẫn U) quá nặng cho vòng
        lặp thời gian thực khi dynamics phi tuyến). X được suy ra bằng
        rollout (_rollout), không phải biến tự do -> không còn ràng
        buộc đẳng thức nào để solver phải ước lượng Jacobian.

        q_des, dq_des : điểm đích cố định (6,) — lặp lại N lần thành
                        quỹ đạo tham chiếu (point-to-point NMPC)
        M_now         : KHÔNG dùng — NMPC tự tính lại M(q) tại từng
                        bước dự báo (qua rollout) bên trong dynamics_func.
        """
        N, n_q, n_u = self.N, self.n_q, self.n_u

        x_current = np.concatenate([
            np.asarray(q_current, dtype=float),
            np.asarray(dq_current, dtype=float)
        ])

        x_ref_step = np.concatenate([
            np.asarray(q_des, dtype=float),
            np.asarray(dq_des, dtype=float)
        ])

        def cost(U_flat):
            U = U_flat.reshape(N, n_u)
            X = _rollout(x_current, U, self.dynamics_func, self.dt, N, n_q)
            J = 0.0
            for k in range(1, N):
                e = X[k] - x_ref_step
                J += e @ self.Q @ e
            eN = X[N] - x_ref_step
            J += eN @ self.Qf @ eN
            for k in range(N):
                J += U[k] @ self.R @ U[k]
            return J

        U0 = self._warm_start()

        if self.torque_limit is not None:
            lo = -np.abs(self.torque_limit)
            hi = np.abs(self.torque_limit)
            bounds = [(lo[i % n_u], hi[i % n_u]) for i in range(N * n_u)]
        else:
            bounds = None

        # KHÔNG còn ràng buộc đẳng thức -> dùng L-BFGS-B (chỉ cần bounds,
        # không phải ước lượng Jacobian của constraint) thay vì SLSQP ->
        # nhẹ hơn nhiều so với bản multiple shooting.
        res = minimize(
            cost, U0, method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter, "ftol": 1e-8}
        )

        self._U_prev = res.x
        torque = res.x[:n_u]
        return torque

