import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def plan_joint_trajectory_cubic_spline(
    joint_waypoints, 
    dt_waypoints=1.0, 
    control_dt=0.01, 
    v_start=None, 
    v_end=None
):
    """
    Quy hoạch quỹ đạo đa khớp robot bằng Cubic Spline.
    
    Parameters:
    -----------
    joint_waypoints : np.ndarray hoặc list dạng (N_points, N_joints)
        Ma trận góc các khớp tại các điểm mốc (đơn vị: rad hoặc deg).
    dt_waypoints : float hoặc list/np.ndarray dạng (N_points - 1,)
        - Nếu là float: Khoảng thời gian cố định giữa 2 điểm mốc liên tiếp (giây).
        - Nếu là list: Mảng chứa khoảng thời gian chi tiết giữa từng chặng.
    control_dt : float
        Chu kỳ lấy mẫu gửi xuống servo/driver (ví dụ: 0.01s = 100Hz).
    v_start : float hoặc list/np.ndarray (N_joints,), mặc định là 0 (đứng yên)
        Vận tốc ban đầu của từng khớp.
    v_end : float hoặc list/np.ndarray (N_joints,), mặc định là 0 (dừng hẳn)
        Vận tốc kết thúc của từng khớp.
        
    Returns:
    --------
    t_samples : np.ndarray (M,)
        Trục thời gian đã băm mẫu mịn.
    q_traj : np.ndarray (M, N_joints)
        Vị trí góc của các khớp theo thời gian.
    dq_traj : np.ndarray (M, N_joints)
        Vận tốc góc của các khớp theo thời gian.
    ddq_traj : np.ndarray (M, N_joints)
        Gia tốc góc của các khớp theo thời gian.
    """
    waypoints = np.array(joint_waypoints, dtype=np.float64)
    num_points, num_joints = waypoints.shape

    # 1. Xác định mốc thời gian t_knots cho từng điểm waypoint
    if np.isscalar(dt_waypoints):
        t_knots = np.arange(num_points) * float(dt_waypoints)
    else:
        t_knots = np.zeros(num_points)
        t_knots[1:] = np.cumsum(dt_waypoints)

    # 2. Thiết lập điều kiện biên kẹp vận tốc (Clamped Spline)
    if v_start is None:
        v_start = np.zeros(num_joints)
    else:
        v_start = np.full(num_joints, v_start) if np.isscalar(v_start) else np.array(v_start)

    if v_end is None:
        v_end = np.zeros(num_joints)
    else:
        v_end = np.full(num_joints, v_end) if np.isscalar(v_end) else np.array(v_end)

    # Đóng gói điều kiện biên: ((đạo hàm bậc 1 tại t_đầu), (đạo hàm bậc 1 tại t_cuối))
    bc_conditions = ((1, v_start), (1, v_end))

    # 3. Tạo hàm Cubic Spline đa kênh (nội suy đồng thời trên trục axis=0)
    cs = CubicSpline(t_knots, waypoints, axis=0, bc_type=bc_conditions)

    # 4. Băm mịn thời gian theo chu kỳ điều khiển control_dt
    total_time = t_knots[-1]
    t_samples = np.arange(0, total_time + control_dt, control_dt)

    # 5. Tính toán Vị trí, Vận tốc, Gia tốc
    q_traj = cs(t_samples)          # Vị trí (rad hoặc deg)
    dq_traj = cs(t_samples, 1)      # Vận tốc (rad/s hoặc deg/s)
    ddq_traj = cs(t_samples, 2)     # Gia tốc (rad/s^2 hoặc deg/s^2)

    return t_samples, q_traj, dq_traj, ddq_traj


# ==========================================
# VÍ DỤ SỬ DỤNG VỚI ROBOT 3 BẬC TỰ DO (3 KHỚP)
# ==========================================
if __name__ == "__main__":
    # Robot có 3 khớp [q1, q2, q3] đi qua 4 điểm mốc (đơn vị: độ)
    waypoints_deg = np.array([
        [0.0,   0.0,   0.0],    # P0: Vị trí Home
        [30.0, -45.0,  60.0],   # P1: Điểm gắp vật
        [60.0, -20.0,  30.0],   # P2: Vị trí tránh cản
        [90.0,   0.0,   0.0]    # P3: Vị trí đặt vật
    ])

    # Mỗi chặng di chuyển trong 2 giây (tổng hành trình 6 giây), tần số lấy mẫu 100Hz (dt = 0.01s)
    t, q, dq, ddq = plan_joint_trajectory_cubic_spline(
        joint_waypoints=waypoints_deg,
        dt_waypoints=2.0,
        control_dt=0.01,
        v_start=0.0,
        v_end=0.0
    )

    print(f"Tổng số điểm điều khiển sinh ra: {len(t)} điểm.")

    # Vẽ đồ thị kết quả Vị trí - Vận tốc - Gia tốc của 3 khớp
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    joint_names = ['Khớp 1 (q1)', 'Khớp 2 (q2)', 'Khớp 3 (q3)']

    for i in range(3):
        # Vị trí
        axes[0].plot(t, q[:, i], label=joint_names[i])
        # Vận tốc
        axes[1].plot(t, dq[:, i], label=joint_names[i])
        # Gia tốc
        axes[2].plot(t, ddq[:, i], label=joint_names[i])

    # Đánh dấu các điểm mốc ban đầu trên đồ thị vị trí
    t_knots = np.array([0, 2, 4, 6])
    for i in range(3):
        axes[0].plot(t_knots, waypoints_deg[:, i], 'o', color='black', alpha=0.6)

    axes[0].set_ylabel('Vị trí (°)')
    axes[0].grid(True)
    axes[0].legend(loc='upper right')
    axes[0].set_title('Quy hoạch quỹ đạo các khớp Robot bằng Cubic Spline')

    axes[1].set_ylabel('Vận tốc (°/s)')
    axes[1].grid(True)

    axes[2].set_ylabel('Gia tốc (°/s²)')
    axes[2].set_xlabel('Thời gian (giây)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
    