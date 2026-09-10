import numpy as np
from scipy.linalg import solve_continuous_are


def compute_LQR_gain(n_joints, Q, R):
    """
    Tính gain phản hồi trạng thái K cho bộ điều khiển LQR, dùng để bám quỹ đạo
    khớp quay của robot SAU KHI đã tuyến tính hóa hồi tiếp (feedback
    linearization) toàn bộ động lực học phi tuyến của robot (xem LQRControl).

    Sau khi tuyến tính hóa, hệ trở thành n_joints chuỗi tích phân kép ĐỘC LẬP
    theo sai số:
        e = q_desired - q_current
        e_ddot = -u          (suy ra trong LQRControl, với u là gia tốc hiệu
                                chỉnh do LQR tạo ra)

    Biến trạng thái z = [e ; e_dot]   (kích thước 2*n_joints)

        A = [[0, I], [0, 0]]      (2n x 2n)
        B = [[0], [-I]]           (2n x n)

    Q (2n x 2n), R (n x n): ma trận trọng số bài toán LQR.
        - Q phạt sai số vị trí/vận tốc: Q càng lớn -> bám càng "gấp", càng sát
        - R phạt gia tốc điều khiển: R càng lớn -> điều khiển càng "tiết kiệm"
          lực, đáp ứng chậm/êm hơn

    LƯU Ý: Bản này hỗ trợ K THAY ĐỔI THEO THỜI GIAN (time-varying), tức
    Q, R có thể khác nhau ở mỗi bước mô phỏng (ví dụ đổi trọng số theo pha
    quỹ đạo, thích nghi theo sai số...). Vì vậy hàm này giờ được gọi lại
    MỖI BƯỚC bên trong LQRControl thay vì gọi một lần duy nhất trước vòng
    lặp. Nếu Q, R không đổi theo thời gian, việc giải lại Riccati mỗi bước
    sẽ tốn thêm chi phí tính toán so với cách cũ (gọi 1 lần) -> chỉ dùng
    cách này khi thực sự cần K thay đổi theo thời gian.

    Returns
    -------
    K : ndarray (n_joints, 2*n_joints)
        Gain phản hồi trạng thái, dùng trong LQRControl: u = -K @ z
    """
    n = n_joints
    I = np.eye(n)
    Z = np.zeros((n, n))

    A = np.block([
        [Z, I],
        [Z, Z]
    ])
    B = np.block([
        [Z],
        [-I]
    ])

    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    return K


def LQRControl(q_desired, q_current, dq_desired, dq_current,
               M, bias, Q, R, qdd_desired=None, torque_limit=0):
    """
    Bộ điều khiển LQR + tuyến tính hóa hồi tiếp (feedback linearization /
    computed-torque control) cho robot tay máy nhiều khớp quay.

    KHÁC BIỆT SO VỚI BẢN CŨ: thay vì nhận gain K đã tính sẵn (tính 1 lần
    trước vòng lặp), hàm này nhận trực tiếp ma trận trọng số Q, R và tự
    gọi compute_LQR_gain() để giải lại phương trình Riccati -> tính K MỖI
    BƯỚC THỜI GIAN. Điều này cho phép Q, R (và do đó K) thay đổi theo thời
    gian, ví dụ theo pha quỹ đạo hoặc thích nghi theo trạng thái hệ thống.

    Nguyên lý
    ---------
    Phương trình động lực học robot:
        M(q) * qddot + bias(q, qdot) = tau
    với bias = C(q,qdot)*qdot + g(q) (lực Coriolis/ly tâm + TRỌNG LỰC gộp
    chung, chính là data.qfrc_bias lấy trực tiếp từ MuJoCo -> KHÔNG cần hàm
    bù trọng lực riêng nữa, LQR này tự bù cả trọng lực lẫn Coriolis).

    Đặt:
        tau = M(q) @ a + bias
    thì:
        qddot = a     (tuyến tính hóa hoàn toàn động lực học phi tuyến)

    Với a = qdd_desired + u (u là gia tốc hiệu chỉnh), sai số
    e = q_desired - q_current thỏa:
        e_ddot = qdd_desired - qddot = qdd_desired - (qdd_desired + u) = -u

    Bài toán thiết kế u cho hệ tuyến tính e_ddot = -u được giải tối ưu bằng
    LQR (xem compute_LQR_gain), cho ra:
        u = -K @ [e ; e_dot]

    Parameters
    ----------
    q_desired, q_current, dq_desired, dq_current : array-like, shape (n,)
    M : ndarray (n, n)
        Ma trận khối lượng tại q_current (lấy từ MuJoCo qua mj_fullM).
    bias : array-like (n,)
        Lực Coriolis + trọng lực tại (q_current, dq_current)
        (data.qfrc_bias trong MuJoCo).
    Q : ndarray (2n, 2n)
        Ma trận trọng số sai số vị trí/vận tốc. Có thể thay đổi mỗi bước.
    R : ndarray (n, n)
        Ma trận trọng số gia tốc điều khiển. Có thể thay đổi mỗi bước.
    qdd_desired : array-like (n,) hoặc None
        Gia tốc khớp mong muốn (feedforward), lấy từ quỹ đạo nếu có.
        Mặc định None -> coi như 0.
    torque_limit : float, array-like, hoặc None
        Giới hạn mô-men từng khớp (giống PIDControl).

    Returns
    -------
    torque : ndarray (n,)
        Mô-men điều khiển cho các khớp [Nm]
    K : ndarray (n, 2n)
        Gain LQR vừa tính được ở bước này (trả thêm ra để debug/log nếu cần).
    """
    q_desired = np.asarray(q_desired, dtype=float)
    q_current = np.asarray(q_current, dtype=float)
    dq_desired = np.asarray(dq_desired, dtype=float)
    dq_current = np.asarray(dq_current, dtype=float)
    M = np.asarray(M, dtype=float)
    bias = np.asarray(bias, dtype=float)

    n = q_current.shape[0]
    qdd_desired = np.zeros(n) if qdd_desired is None else np.asarray(qdd_desired, dtype=float)

    # 0. Tính lại gain K cho bước thời gian hiện tại
    K = compute_LQR_gain(n, Q, R)

    # 1. Sai số vị trí + vận tốc
    e = q_desired - q_current
    edot = dq_desired - dq_current
    z = np.concatenate([e, edot])

    # 2. Gia tốc hiệu chỉnh tối ưu theo LQR
    u = -K @ z

    # 3. Tổng gia tốc khớp mong muốn (feedforward + hiệu chỉnh)
    a = qdd_desired + u

    # 4. Ánh xạ sang mô-men bằng tuyến tính hóa hồi tiếp (đã tự gồm bù
    #    trọng lực + Coriolis thông qua "bias")
    torque = M @ a + bias

    # 5. Giới hạn mô-men theo phần cứng thật
    if torque_limit is not None:
        torque = np.clip(
            torque,
            -np.asarray(torque_limit),
            np.asarray(torque_limit)
        )

    return torque, K


if __name__ == "__main__":
    # Ví dụ nhanh: mô phỏng vài bước, K được tính lại mỗi bước
    n = 6
    Q = np.diag([100]*n + [10]*n)   # phạt sai số vị trí mạnh hơn vận tốc
    R = np.eye(n) * 0.1

    q_desired = np.zeros(n)
    q_current = np.ones(n) * 0.1
    dq_desired = np.zeros(n)
    dq_current = np.zeros(n)
    M = np.eye(n)
    bias = np.zeros(n)

    for step in range(3):
        torque, K = LQRControl(q_desired, q_current, dq_desired, dq_current,
                                M, bias, Q, R)
        print(f"Step {step}: torque =", torque)
        # (trong vòng lặp mô phỏng thật, q_current/dq_current sẽ được cập
        #  nhật từ động lực học sau mỗi bước)