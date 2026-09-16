
#Khai bao (da hieu chinh theo kich thuoc thuc te trong file XML MuJoCo)
d1 = 0.2435   # Độ cao chân đế
a2 = 0.2002   # Độ dài cánh tay 1
a3 = 0.0870   # Độ dài cánh tay 2
d4 = 0.22761  # Độ dài khâu 4
d5 = 0.0000   # Khoảng cách khớp 5
d6 = 0.0625   # Độ dài đầu gắp

#Ham chuyen doi Denavit-Hartenberg
from pyexpat import model
from time import time

import numpy as np
import mujoco.viewer
import mujoco
import os

joint_limits = [
    [-np.radians(360), np.radians(360)],  # Khớp 1
    [-5 * np.pi / 6, 5 * np.pi / 6],      # Khớp 2 (-150° đến 150°)
    [-np.radians(3.5), np.radians(300)],  # Khớp 3
    [-np.radians(360), np.radians(360)],  # Khớp 4
    [-np.radians(124), np.radians(124)],  # Khớp 5
    [-np.radians(360), np.radians(360)],  # Khớp 6
]

def dh_transform(theta, d, a, alpha):

    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),
         np.sin(theta)*np.sin(alpha), a*np.cos(theta)],

        [np.sin(theta), np.cos(theta)*np.cos(alpha),
         -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],

        [0, np.sin(alpha), np.cos(alpha), d],

        [0, 0, 0, 1]
    ])

#Thiet lap thong so cho khop

def forward_kinematics(q, joint_limits=joint_limits):
    
    for i, (theta, limits) in enumerate(zip(q, joint_limits)):
        if not (limits[0] <= theta <= limits[1]):
            raise ValueError(f"Joint {i+1} angle {np.degrees(theta):.2f}° is out of limits {np.degrees(limits[0]):.2f}° to {np.degrees(limits[1]):.2f}°.")

    #Thay the cac gia tri vao ma tran chuyen doi
    theta1_val, theta2_val, theta3_val, theta4_val, theta5_val, theta6_val = q
    # Luu y: khop 2 va khop 3 can cong them offset -pi/2 de khop voi
    # he quy chieu (frame) cua cac body trong file XML MuJoCo (mj_forward).
    A1 = dh_transform(theta1_val, d1, 0, -np.pi/2)
    A2 = dh_transform(theta2_val - np.pi/2, 0, a2, np.pi)
    A3 = dh_transform(theta3_val - np.pi/2, 0, a3, np.pi/2)
    A4 = dh_transform(theta4_val, d4, 0, np.pi/2)
    A5 = dh_transform(theta5_val, d5, 0, -np.pi/2)
    A6 = dh_transform(theta6_val, d6, 0, 0)
    VF = A1 @ A2 @ A3 @ A4 @ A5 @ A6

 # Làm tròn các giá trị trong ma trận VF đến 4 chữ số thập phân
    VF = np.round(VF,4)
    return VF

def forward_kinematics_all(q, joint_limits=joint_limits):
    """
    Tra ve danh sach 7 ma tran bien doi tu base den tung frame trung gian:
    [T0, T1, T2, T3, T4, T5, T6]
 
    T0 = eye(4)                  (frame base, truoc khop 1)
    T1 = A1                      (frame sau khop 1, truoc khop 2)
    ...
    T6 = A1 @ A2 @ ... @ A6      (frame end-effector)
 
    Dung cho tinh Jacobian: T_list[i] la frame ma khop (i+1) quay quanh
    truc z cua no (quy uoc DH chuan). KHONG lam tron so (khac voi
    forward_kinematics) de tranh loi khi dung lam dao ham so.
    """

    for i, (theta, limits) in enumerate(zip(q, joint_limits)):
            if not (limits[0] <= theta <= limits[1]):
                raise ValueError(f"Joint {i+1} angle {np.degrees(theta):.2f}° is out of limits {np.degrees(limits[0]):.2f}° to {np.degrees(limits[1]):.2f}°.")
    theta1_val, theta2_val, theta3_val, theta4_val, theta5_val, theta6_val = q
 
    A1 = dh_transform(theta1_val, d1, 0, -np.pi/2)
    A2 = dh_transform(theta2_val - np.pi/2, 0, a2, np.pi)
    A3 = dh_transform(theta3_val - np.pi/2, 0, a3, np.pi/2)
    A4 = dh_transform(theta4_val, d4, 0, np.pi/2)
    A5 = dh_transform(theta5_val, d5, 0, -np.pi/2)
    A6 = dh_transform(theta6_val, d6, 0, 0)
 
    T0 = np.eye(4)
    T1 = T0 @ A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5
    T6 = T5 @ A6
 
    return [T0, T1, T2, T3, T4, T5, T6]


if __name__ == "__main__":
    #Print
    VF_simplified = forward_kinematics(q = (0, 0, 0, 0, 0, 0))

    print("Forward Kinematics Transformation Matrix (VF):")
    print(VF_simplified)


    print("Tọa độ EE:", VF_simplified[:3, 3])

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(BASE_DIR, "mujoco_menagerie", "ufactory_lite6", "lite6.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[:6] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # data.qpos[: len(q_init)] = q_init
    # mujoco.mj_forward(model, data)

    # #In kq cua MUJOCO
    # site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    # pos_mujoco = data.site_xpos[site_id]
    # rot_mujoco = data.site_xmat[site_id].reshape(3, 3)
    # T_mujoco = np.eye(4)
    # T_mujoco[:3, :3] = rot_mujoco
    # T_mujoco[:3, 3] = pos_mujoco
    # T_mujoco = np.round(T_mujoco, 4)  # Làm tròn các giá trị trong ma trận T_mujoco đến 4 chữ số thập phân
    # print("\nKết quả Forward Kinematics từ MuJoCo:")
    # print("Tọa độ EE:", pos_mujoco)
    # print("Ma trận quay EE:\n", rot_mujoco)
    # print("Ma trận biến đổi T:\n", T_mujoco)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()