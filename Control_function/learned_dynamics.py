"""
learned_dynamics.py
=====================
Đọc file trọng số đã train trên Colab (dyn_model.npz) và dựng lại đúng
forward-pass đó bằng NUMPY THUẦN (không cần torch lúc chạy MPC).

Trả về đúng hàm có chữ ký GIỐNG HỆT make_mujoco_dynamics_func() trong
Non_MPC2.py:

    dynamics_func(q, dq, u) -> ddq

Vì NMPCController.compute() gọi self.dynamics_func bên trong _rollout/_f
(qua RK4), nên chỉ cần THAY 1 DÒNG là dùng được ngay, không phải sửa gì
khác trong Non_MPC2.py:

    from learned_dynamics import load_learned_dynamics

    mpc = NMPCController(model=model, Q=Q, R=NMPC_R, N=NMPC_N, dt=NMPC_DT,
                          torque_limit=nmpc_u_max, maxiter=50)

    # Thay động lực học MuJoCo (chậm, gọi mj_forward mỗi lần) bằng mạng đã học:
    mpc.dynamics_func = load_learned_dynamics("dyn_model.npz")

Mọi thứ khác (RK4, rollout, cost, L-BFGS-B, warm start, torque_limit) giữ
nguyên 100% vì chúng chỉ gọi self.dynamics_func(q, dq, u), không quan tâm
bên trong là MuJoCo hay mạng nơ-ron.
"""

import numpy as np


def load_learned_dynamics(npz_path: str):
    d = np.load(npz_path)

    n_layers = int(d["n_layers"])
    activation = str(d["activation"])
    n_q = int(d["n_q"])
    n_u = int(d["n_u"])

    in_mean = d["in_mean"]
    in_std = d["in_std"]
    out_mean = d["out_mean"]
    out_std = d["out_std"]

    Ws = [d[f"W{i}"] for i in range(n_layers)]   # mỗi W: (out, in) kiểu PyTorch nn.Linear
    bs = [d[f"b{i}"] for i in range(n_layers)]

    if activation == "tanh":
        act = np.tanh
    elif activation == "relu":
        act = lambda z: np.maximum(z, 0.0)
    else:
        raise ValueError(f"Activation không hỗ trợ: {activation}")

    def dynamics_func(q, dq, u):
        q = np.asarray(q, dtype=float).reshape(n_q)
        dq = np.asarray(dq, dtype=float).reshape(n_q)
        u = np.asarray(u, dtype=float).reshape(n_u)

        inp = np.concatenate([q, dq, u])
        h = (inp - in_mean) / in_std

        for i in range(n_layers):
            h = Ws[i] @ h + bs[i]
            if i < n_layers - 1:      # lớp cuối không qua activation
                h = act(h)

        ddq = h * out_std + out_mean   # (n_q,)
        return ddq

    return dynamics_func


if __name__ == "__main__":
    # Demo nhanh: kiểm tra hàm chạy được, không cần MuJoCo/torch
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dyn_model (1).npz"
    f = load_learned_dynamics(path)
    d = np.load(path)
    n_q = int(d["n_q"])
    q0 = np.zeros(n_q)
    dq0 = np.zeros(n_q)
    u0 = np.zeros(n_q)
    print("ddq demo tại (q=0, dq=0, u=0):")
    print(f(q0, dq0, u0))
