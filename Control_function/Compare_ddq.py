"""
compare_ddq.py
================
So sánh TRỰC TIẾP đầu ra q̈ (ddq) giữa:
  - dynamics_func THẬT của MuJoCo (make_mujoco_dynamics_func, dùng M(q)
    và qfrc_bias chính xác -> coi là "chuẩn"/ground truth)
  - dynamics_func XẤP XỈ bằng mạng MLP đã học (learned_dynamics.py)

trên CÙNG một tập điểm (q, dq, u) — KHÔNG mô phỏng/rollout gì cả, chỉ so
số ra số. Đây là cách nhanh nhất để biết mạng học lệch bao nhiêu so với
động lực học thật, trước khi đem vào NMPC.

Chạy:
    python compare_ddq.py --dyn_model dyn_model.npz
"""

import argparse
import os
import numpy as np
import mujoco
import sys

FILE_PATH = os.path.abspath(__file__)
CONTROL_DIR = os.path.dirname(FILE_PATH)
KINEMATIC_DIR = os.path.dirname(CONTROL_DIR)
MUJUCO_ROOT_DIR = os.path.dirname(KINEMATIC_DIR)
sys.path.append(KINEMATIC_DIR)
xml_path = os.path.join(MUJUCO_ROOT_DIR, "mujoco_menagerie", "ufactory_lite6", "lite6.xml")
from Non_MPC2 import make_mujoco_dynamics_func
from learned_dynamics import load_learned_dynamics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn_model", type=str, default="Control_function/dyn_model (1).npz")
    ap.add_argument("--n_test", type=int, default=2000,
                     help="Số điểm (q,dq,u) ngẫu nhiên để test")
    ap.add_argument("--q_margin", type=float, default=1.2,
                     help="Nên trùng với Q_MARGIN lúc train trên Colab, để "
                          "test đúng vùng mạng đã học (test ngoài vùng này "
                          "thì sai số cao là bình thường, không phải lỗi)")
    ap.add_argument("--dq_range", type=float, default=2.0,
                     help="Nên trùng với DQ_RANGE lúc train")
    ap.add_argument("--torque_scale", type=float, default=0.5,
                     help="Nên trùng với TORQUE_SCALE lúc train")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(xml_path)
    n_q = 6
    n_u = 6

    joint_range = model.jnt_range.copy()
    torque_range = model.actuator_forcerange.copy()

    if model.nkey > 0:
        q_center = model.key_qpos[0][:n_q].copy()
    else:
        q_center = np.zeros(n_q)

    q_lo = np.maximum(joint_range[:, 0], q_center - args.q_margin)
    q_hi = np.minimum(joint_range[:, 1], q_center + args.q_margin)
    u_lo = torque_range[:, 0] * args.torque_scale
    u_hi = torque_range[:, 1] * args.torque_scale

    rng = np.random.default_rng(args.seed)
    Q = rng.uniform(q_lo, q_hi, size=(args.n_test, n_q))
    DQ = rng.uniform(-args.dq_range, args.dq_range, size=(args.n_test, n_q))
    U = rng.uniform(u_lo, u_hi, size=(args.n_test, n_u))

    f_true = make_mujoco_dynamics_func(model, n_q=n_q)   # MuJoCo thật -> chuẩn
    f_learned = load_learned_dynamics(args.dyn_model)     # mạng MLP đã học

    ddq_true = np.zeros((args.n_test, n_q))
    ddq_learned = np.zeros((args.n_test, n_q))

    for i in range(args.n_test):
        ddq_true[i] = f_true(Q[i], DQ[i], U[i])
        ddq_learned[i] = f_learned(Q[i], DQ[i], U[i])

    err = ddq_learned - ddq_true
    abs_err = np.abs(err)
    std_true = ddq_true.std(axis=0)

    print("=" * 70)
    print(f"So sánh q̈ (ddq): MuJoCo thật (chuẩn) vs Mạng MLP  —  {args.n_test} điểm test")
    print("=" * 70)
    print(f"{'Khớp':<8}{'MAE':>10}{'RMSE':>10}{'Max err':>12}{'std(ddq thật)':>16}{'MAE/std':>10}")
    for j in range(n_q):
        mae = abs_err[:, j].mean()
        rmse = np.sqrt((err[:, j] ** 2).mean())
        max_e = abs_err[:, j].max()
        rel = mae / (std_true[j] + 1e-9) * 100
        print(f"joint{j+1:<3}{mae:>10.4f}{rmse:>10.4f}{max_e:>12.4f}{std_true[j]:>16.4f}{rel:>9.2f}%")

    print("-" * 70)
    mae_all = abs_err.mean()
    rmse_all = np.sqrt((err ** 2).mean())
    rel_all = mae_all / (std_true.mean() + 1e-9) * 100
    print(f"{'TỔNG':<8}{mae_all:>10.4f}{rmse_all:>10.4f}{'':>12}{std_true.mean():>16.4f}{rel_all:>9.2f}%")

    # R^2 kiểu hồi quy (1 - SS_res/SS_tot), càng gần 1 càng tốt
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((ddq_true - ddq_true.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\nR² tổng thể: {r2:.4f}  (1.0 = khớp hoàn hảo với MuJoCo thật)")


if __name__ == "__main__":
    main()