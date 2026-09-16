"""
hybrid_dynamics.py
=====================
Kết hợp:
  - MLP nominal (đã train offline trên Colab, dyn_model.npz) -> dự báo
    ddq "gốc", chính xác trên đúng model MuJoCo lúc sinh dữ liệu.
  - GP residual (train ONLINE, cửa sổ trượt vài trăm điểm gần nhất) ->
    học phần SAI LỆCH giữa MLP và thực tế đang đo được khi vận hành
    (tải trọng đổi, ma sát đổi, model thật khác XML...).

    ddq_predict(q,dq,u) = f_MLP(q,dq,u) + f_GP_residual(q,dq,u)

Đây chính là phần biến bộ điều khiển thành "adaptive": mỗi vòng lặp
điều khiển, nếu đo được ddq THẬT (xem ghi chú compute_ddq_measured bên
dưới), gọi hybrid.update(...) để GP học thêm điểm dữ liệu mới -> mô hình
dự báo tự chỉnh theo sai lệch thực tế, không cần train lại MLP.

Cách dùng trong Control_NonMPC.py
-----------------------------------
    from hybrid_dynamics import HybridDynamics

    hybrid = HybridDynamics(mlp_path="dyn_model.npz", window_size=200,
                             retrain_every=5)

    MPC.dynamics_func = hybrid.dynamics_func   # cắm vào NMPC như cũ

    # Trong vòng lặp mô phỏng/điều khiển chính, MỖI BƯỚC (không phải mỗi
    # lần gọi MPC.compute(), vì cần lấy mẫu (q,dq,u,ddq) ở tần số vật lý
    # để finite-difference ddq cho chuẩn):
    dq_prev = data.qvel[:6].copy()
    ...
    mujoco.mj_step(model, data)
    dq_now = data.qvel[:6].copy()
    ddq_measured = (dq_now - dq_prev) / model.opt.timestep   # xem ghi chú dưới

    hybrid.update(q_prev, dq_prev, torque_applied, ddq_measured)

Ghi chú về ddq_measured
--------------------------
Trên MÔ PHỎNG MuJoCo, bạn có thể lấy ddq "thật" chính xác bằng
`data.qacc[:6]` ngay sau `mj_step` — không cần finite-difference (đây là
cách gian lận hợp lệ để TEST logic hybrid trước khi lên robot thật).

Trên ROBOT THẬT, không có `qacc` trực tiếp -> phải ước lượng bằng sai
phân vận tốc đo từ encoder: ddq ≈ (dq[k] - dq[k-1]) / dt. Cách này có
nhiễu (đạo hàm số khuếch đại nhiễu đo), nên nếu robot thật có nhiễu đo
lớn, ĐÂY LÀ CHỖ CẦN KALMAN FILTER trước khi đưa vào hybrid.update() —
lọc dq mượt trước, rồi mới sai phân ra ddq, residual GP sẽ học đúng
hơn nhiều so với việc học luôn cả nhiễu đo.
"""

import numpy as np
from collections import deque

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
except ImportError as e:
    raise SystemExit("Cần cài: pip install scikit-learn") from e

from learned_dynamics import load_learned_dynamics


class HybridDynamics:
    def __init__(self, mlp_path: str, n_q: int = 6, n_u: int = 6,
                 window_size: int = 200, retrain_every: int = 5,
                 min_points_to_fit: int = 15):
        """
        window_size      : số điểm residual gần nhất giữ lại (sliding
                            window) -> GP chỉ "nhớ" sai lệch GẦN ĐÂY,
                            tự động quên dữ liệu cũ (đúng tinh thần
                            adaptive: thích nghi điều kiện hiện tại,
                            không bị kéo lùi bởi dữ liệu quá khứ xa).
        retrain_every     : chỉ refit GP mỗi N điểm mới (không refit mỗi
                            bước để tiết kiệm tính toán -> refit O(n^3)
                            với n=200 vẫn rất nhanh, nhưng cứ để dư ra).
        min_points_to_fit : cần ít nhất bấy nhiêu điểm mới bắt đầu train
                            GP -> trước đó chỉ dùng thuần MLP.
        """
        self.f_mlp = load_learned_dynamics(mlp_path)
        self.n_q = n_q
        self.n_u = n_u
        self.window_size = window_size
        self.retrain_every = retrain_every
        self.min_points_to_fit = min_points_to_fit

        self.X_buf = deque(maxlen=window_size)   # mỗi phần tử: (q,dq,u) nối (18,)
        self.Y_buf = deque(maxlen=window_size)   # residual ddq_true - ddq_mlp, (n_q,)

        self.gp_models = None            # list n_q model GP độc lập (1 output/khớp)
        self._n_new_since_fit = 0
        self._is_fitted = False

        # Chuẩn hoá input GP bằng chính in_mean/in_std đã lưu lúc train MLP
        # (tận dụng lại, không cần tính riêng) -> giúp kernel RBF hoạt động
        # đúng trên các chiều có đơn vị/độ lớn khác nhau (q vs dq vs u).
        d = np.load(mlp_path)
        self._in_mean = d["in_mean"]
        self._in_std = d["in_std"]

    # ------------------------------------------------------------
    def _make_kernel(self):
        # RBF với ARD (length_scale riêng từng chiều input) + white noise
        # để GP tự học độ nhiễu của residual, không overfit vào nhiễu đo.
        n_dim = self.n_q + self.n_q + self.n_u
        return C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(n_dim),
                                          length_scale_bounds=(1e-2, 1e2)) \
               + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))

    # ------------------------------------------------------------
    def update(self, q, dq, u, ddq_measured):
        """Thêm 1 điểm dữ liệu mới (đo được lúc vận hành) vào cửa sổ
        trượt, và refit GP nếu đã đủ điểm mới theo retrain_every."""
        q = np.asarray(q, dtype=float).reshape(self.n_q)
        dq = np.asarray(dq, dtype=float).reshape(self.n_q)
        u = np.asarray(u, dtype=float).reshape(self.n_u)
        ddq_measured = np.asarray(ddq_measured, dtype=float).reshape(self.n_q)

        ddq_mlp = self.f_mlp(q, dq, u)
        residual = ddq_measured - ddq_mlp   # cái GP cần học: phần MLP CHƯA giải thích được

        x_raw = np.concatenate([q, dq, u])
        x_n = (x_raw - self._in_mean) / self._in_std   # chuẩn hoá giống lúc train MLP

        self.X_buf.append(x_n)
        self.Y_buf.append(residual)
        self._n_new_since_fit += 1

        n_points = len(self.X_buf)
        if n_points >= self.min_points_to_fit and self._n_new_since_fit >= self.retrain_every:
            self._refit()
            self._n_new_since_fit = 0

    # ------------------------------------------------------------
    def _refit(self):
        X = np.array(self.X_buf)          # (n_points, 18)
        Y = np.array(self.Y_buf)          # (n_points, n_q)

        models = []
        for j in range(self.n_q):
            gp = GaussianProcessRegressor(kernel=self._make_kernel(),
                                           normalize_y=True,
                                           n_restarts_optimizer=1,
                                           alpha=1e-6)
            gp.fit(X, Y[:, j])
            models.append(gp)
        self.gp_models = models
        self._is_fitted = True

    # ------------------------------------------------------------
    def predict(self, q, dq, u, return_std=False):
        """ddq_predict = f_MLP(q,dq,u) + f_GP_residual(q,dq,u).
        Nếu chưa đủ dữ liệu để fit GP, trả về thuần MLP (residual=0)."""
        q = np.asarray(q, dtype=float).reshape(self.n_q)
        dq = np.asarray(dq, dtype=float).reshape(self.n_q)
        u = np.asarray(u, dtype=float).reshape(self.n_u)

        ddq_mlp = self.f_mlp(q, dq, u)

        if not self._is_fitted:
            if return_std:
                return ddq_mlp, np.zeros(self.n_q)
            return ddq_mlp

        x_raw = np.concatenate([q, dq, u])
        x_n = ((x_raw - self._in_mean) / self._in_std).reshape(1, -1)

        residual = np.zeros(self.n_q)
        std = np.zeros(self.n_q)
        for j in range(self.n_q):
            if return_std:
                mean_j, std_j = self.gp_models[j].predict(x_n, return_std=True)
                residual[j] = mean_j[0]
                std[j] = std_j[0]
            else:
                residual[j] = self.gp_models[j].predict(x_n)[0]

        ddq = ddq_mlp + residual
        if return_std:
            return ddq, std
        return ddq

    # ------------------------------------------------------------
    def dynamics_func(self, q, dq, u):
        """Chữ ký tương thích trực tiếp với NMPCController.dynamics_func
        (Non_MPC2.py): gán thẳng MPC.dynamics_func = hybrid.dynamics_func"""
        return self.predict(q, dq, u, return_std=False)


if __name__ == "__main__":
    # Demo nhanh không cần MuJoCo: kiểm tra hybrid chạy được, residual
    # học đúng 1 hàm sai lệch giả lập.
    import sys
    path = "Control_function/dyn_model (1).npz" 
    hybrid = HybridDynamics(path, retrain_every=5, min_points_to_fit=15)

    d = np.load(path)
    n_q = int(d["n_q"])
    rng = np.random.default_rng(0)

    print("Trước khi có dữ liệu residual (chỉ MLP thuần):")
    q0, dq0, u0 = np.zeros(n_q), np.zeros(n_q), np.zeros(n_q)
    print(hybrid.predict(q0, dq0, u0))

    # Giả lập: thực tế luôn lệch +0.5 rad/s^2 so với MLP (vd do tải trọng mới)
    for _ in range(60):
        q = rng.uniform(-0.5, 0.5, n_q)
        dq = rng.uniform(-1, 1, n_q)
        u = rng.uniform(-5, 5, n_q)
        ddq_mlp = hybrid.f_mlp(q, dq, u)
        ddq_measured = ddq_mlp + 0.5   # sai lệch giả lập cố định
        hybrid.update(q, dq, u, ddq_measured)

    print("\nSau khi GP học ~60 điểm residual (kỳ vọng gần +0.5 mỗi khớp):")
    ddq_pred, std = hybrid.predict(q0, dq0, u0, return_std=True)
    ddq_mlp_only = hybrid.f_mlp(q0, dq0, u0)
    print("residual học được:", ddq_pred - ddq_mlp_only)
    print("std (độ bất định):", std)