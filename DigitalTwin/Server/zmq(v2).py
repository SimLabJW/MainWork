#!/usr/bin/env python3
import zmq
import signal
import sys
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from Mapping.lidar_to_grid_map.lidar_to_grid_map import OccGridMap

# ---- ZMQ 서버 설정 ----
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:8788")

def signal_handler(sig, frame):
    """Ctrl+C 종료 시 ZMQ 소켓과 Matplotlib 창을 안전하게 닫는 핸들러"""
    print("\n서버를 종료합니다.")
    plt.close('all')
    socket.close()
    context.term()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
print("ZMQ REP 서버가 시작되었습니다 (Ctrl+C로 종료).")

# --- Occupancy Grid Map 초기화 ---
MAP_SIZE_M = 20.0
MAP_RESOLUTION = 0.1
occupancy_grid_map = OccGridMap(MAP_SIZE_M, MAP_RESOLUTION)

# --- 로봇 상태 및 데이터 초기화 ---
robot_pose = np.array([0.0, 0.0, 0.0])  # [x(m), y(m), theta(rad)]
path_history = []
previous_scan_points = None

# --- 시각화 설정 ---
plt.ion()
fig, ax = plt.subplots()

map_image = ax.imshow(
    np.zeros((occupancy_grid_map.xw, occupancy_grid_map.yw)),
    cmap="gray",
    origin="lower",
    extent=[-MAP_SIZE_M / 2, MAP_SIZE_M / 2, -MAP_SIZE_M / 2, MAP_SIZE_M / 2]
)

robot_plot, = ax.plot([], [], "ro", label="Robot")
path_plot, = ax.plot([], [], "b-", linewidth=1, label="Path")

ax.set_title("Unity LiDAR SLAM Visualization")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.legend()
ax.grid(True)

# --- 메인 루프: Unity로부터 데이터 수신 및 SLAM 처리 ---
while True:
    try:
        message = socket.recv_string()
        socket.send_string("ack")

        lines = message.strip().splitlines()
        current_scan_points = []
        odometry_delta = None

        for line in lines:
            parts = line.split(",")
            if parts[0] == "POSE":
                try:
                    odometry_delta = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                except (ValueError, IndexError):
                    print("오도메트리 데이터 파싱 오류")
                    continue
            else:
                try:
                    angle_rad = float(parts[0])
                    distance_m = float(parts[1]) / 1000.0
                    intensity = float(parts[2])
                    if intensity > 0:
                        x = distance_m * np.cos(angle_rad)
                        y = distance_m * np.sin(angle_rad)
                        current_scan_points.append([x, y])
                except (ValueError, IndexError):
                    print(f"LiDAR 데이터 파싱 오류: {line}")
                    continue

        if not current_scan_points:
            print("스캔 데이터가 없습니다. 다음 프레임을 기다립니다.")
            continue

        current_scan_points = np.array(current_scan_points)

        # 첫 번째 스캔 데이터는 비교할 이전 데이터가 없으므로 건너뜁니다.
        if previous_scan_points is None:
            previous_scan_points = current_scan_points
            continue

        # --- SLAM: 위치 추정 및 맵 업데이트 ---
        
        # 1. 위치 추정 (Localization) - NDT 대신 Open3D의 ICP 사용
        # Open3D의 NDT 정합을 사용하려면 포인트 클라우드를 구성해야 합니다.
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(np.hstack((current_scan_points, np.zeros((current_scan_points.shape[0], 1)))))
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(np.hstack((previous_scan_points, np.zeros((previous_scan_points.shape[0], 1)))))
        
        # Open3D의 ICP를 사용하여 두 포인트 클라우드 간의 변환 행렬을 추정합니다.
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        
        # Open3D 변환 행렬에서 회전 및 평행이동 값을 추출합니다.
        transformation_matrix = reg_p2p.transformation
        T = transformation_matrix[:2, 2]
        R = transformation_matrix[:2, :2]
        
        dtheta = np.arctan2(R[1, 0], R[0, 0])
        dx, dy = T[0], T[1]
        
        robot_pose[0] += dx
        robot_pose[1] += dy
        robot_pose[2] += dtheta

        # 2. 맵 생성 (Mapping)
        cos_theta, sin_theta = np.cos(robot_pose[2]), np.sin(robot_pose[2])
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        global_scan_points = (rotation_matrix @ current_scan_points.T).T + robot_pose[:2]
        
        occupancy_grid_map.update(robot_pose, global_scan_points)

        path_history.append(robot_pose[:2].tolist())
        
        # --- 시각화 업데이트 ---
        map_image.set_data(occupancy_grid_map.grid.T)
        robot_plot.set_data([robot_pose[0]], [robot_pose[1]])
        
        if path_history:
            path_array = np.array(path_history)
            path_plot.set_data(path_array[:, 0], path_array[:, 1])

        plt.draw()
        plt.pause(0.01)

        previous_scan_points = current_scan_points

    except zmq.error.ContextTerminated:
        break
    except Exception as e:
        print(f"오류 발생: {e}")