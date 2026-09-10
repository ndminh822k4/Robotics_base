import numpy as np

def PIDControl(q_desired, q_current, dq_desired, dq_current, integral_error, dt, kp, ki, kd, torque_limit = 0):
    """
    PID position control cho các khớp quay robot.
    Parameters
    ----------
    q_desired : array-like
        Góc khớp mong muốn [rad], shape (n,)
    q_current : array-like
        Góc khớp hiện tại [rad], shape (n,)
    dq_desired : array-like
        Vận tốc khớp mong muốn [rad/s], shape (n,)
    dq_current : array-like
        Vận tốc khớp hiện tại [rad/s], shape (n,)
    integral_error : array-like
        Tích phân sai số từ các bước trước, shape (n,)
    dt : float
        Chu kỳ điều khiển [s]
    Kp : array-like hoặc scalar
        Hệ số P
    Ki : array-like hoặc scalar
        Hệ số I
    Kd : array-like hoặc scalar
        Hệ số D
    torque_limit : float hoặc array-like hoặc None
        Giới hạn moment của từng khớp.

    Returns
    -------
    torque : ndarray
        Moment điều khiển cho các khớp [Nm]
    integral_error : ndarray
        Integral error đã được cập nhật
    """

    q_desired = np.asarray(q_desired, dtype=float)
    q_current = np.asarray(q_current, dtype=float)

    dq_desired = np.asarray(dq_desired, dtype=float)
    dq_current = np.asarray(dq_current, dtype=float)

    integral_error = np.asarray(integral_error, dtype=float)

    #1. Position error
    pos_err = q_desired-q_current

    #2. Integral error
    integral_error += pos_err * dt

    #3. Velocity error
    v_err = dq_desired - dq_current

    #4. PID
    torque = kp *pos_err + ki * integral_error + kd * v_err

    #5. Torque satuation
    if torque_limit is not None:
        torque = np.clip(
            torque,
            -np.asarray(torque_limit),
            np.asarray(torque_limit)
        )

    return torque, integral_error

if __name__ == "__main__":

    # Góc mong muốn
    q_desired = np.deg2rad([
        30,
        40,
        50,
        60,
        70,
        80
    ])

    # Góc hiện tại
    q_current = np.deg2rad([
        0,
        0,
        0,
        0,
        0,
        0
    ])

    # Vận tốc mong muốn
    dq_desired = np.zeros(6)

    # Vận tốc hiện tại
    dq_current = np.zeros(6)

    # Integral ban đầu
    integral_error = np.zeros(6)

    # Thời gian control
    dt = 0.002

    # PID gains
    Kp = np.array([
        100,
        100,
        100,
        50,
        30,
        20
    ])

    Ki = np.array([
        0,
        0,
        0,
        0,
        0,
        0
    ])

    Kd = np.array([
        10,
        10,
        10,
        5,
        3,
        2
    ])

    # Tính torque
    torque, integral_error = PIDControl(
        q_desired,
        q_current,
        dq_desired,
        dq_current,
        integral_error,
        dt,
        Kp,
        Ki,
        Kd,
        torque_limit=50
    )

    print("Torque:", torque)