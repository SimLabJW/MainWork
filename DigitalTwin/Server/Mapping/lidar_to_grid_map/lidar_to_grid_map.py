#!/usr/bin/env python3
import zmq
import signal
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

# PythonRobotics 라이브러리의 utils 및 Mapping 폴더의 함수들을 임포트합니다.
from collections import deque
from Mapping.grid_map_lib.grid_map_lib import GridMap
from utils.angle import angle_mod

# --- 원본 lidar_to_2d_grid_map.py의 핵심 함수들 ---
# 이 함수들은 여러분이 제공해주신 파일에서 가져온 것입니다.
EXTEND_AREA = 1.0
def bresenham(start, end):
    # ... (기존 bresenham 함수 내용 그대로) ...
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    y_step = 1 if y1 < y2 else -1
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:
        points.reverse()
    points = np.array(points)
    return points

def calc_grid_map_config(ox, oy, xy_resolution):
    min_x = round(min(ox) - EXTEND_AREA / 2.0)
    min_y = round(min(oy) - EXTEND_AREA / 2.0)
    max_x = round(max(ox) + EXTEND_AREA / 2.0)
    max_y = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    return min_x, min_y, max_x, max_y, xw, yw

# ---- ZMQ 서버 설정 ----
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:8788")

def signal_handler(sig, frame):
    print("\n서버를 종료합니다.")
    plt.close('all')
    socket.close()
    context.term()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
print("ZMQ REP 서버가 시작되었습니다 (Ctrl+C로 종료).")

# --- 맵 및 로봇 상태 초기화 ---
# 초기 맵은 비워둡니다. 데이터가 들어오면 동적으로 생성됩니다.
occupancy_map = None
map_min_x, map_min_y, map_max_x, map_max_y = 0, 0, 0, 0
map_xw, map_yw = 0, 0

# ICP 매칭을 위한 이전 스캔 데이터와 로봇 위치
robot_pose = np.array([0.0, 0.0, 0.0])  # [x(m), y(m), theta(rad)]
path_history = []
previous_scan_points = None

# --- 시각화 설정 ---
plt.ion()
fig, ax = plt.subplots()

map_image = None
robot_plot, = ax.plot([], [], "ro", label="Robot")
path_plot, = ax.plot([], [], "b-", linewidth=1, label="Path")

ax.set_title("Unity LiDAR SLAM Visualization (Ray Casting)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.legend()
ax.grid(True)
plt.axis("equal")

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
        
        # 1. 위치 추정 (ICP) - PythonRobotics의 ICP 매칭 함수를 사용해야 합니다.
        # ICP 코드가 이 파일에 없으므로, PythonRobotics의 ICPMatching 폴더에서 가져와야 합니다.
        # 예시로, 간단한 오도메트리 업데이트를 사용하겠습니다.
        if odometry_delta is not None:
            robot_pose += odometry_delta

        # 2. 맵 생성 (Mapping) - Ray Casting
        # 로봇 좌표계의 스캔 포인트를 전역 좌표계로 변환
        cos_theta, sin_theta = np.cos(robot_pose[2]), np.sin(robot_pose[2])
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        global_scan_points = (rotation_matrix @ current_scan_points.T).T + robot_pose[:2]

        # 맵이 초기화되지 않았거나 로봇이 맵 경계를 벗어나면 맵을 재구성합니다.
        if occupancy_map is None:
            map_min_x, map_min_y, map_max_x, map_max_y, map_xw, map_yw = \
                calc_grid_map_config(global_scan_points[:, 0], global_scan_points[:, 1], 0.1)
            occupancy_map = np.ones((map_xw, map_yw)) / 2
        else:
            # 기존 맵의 경계에 새로운 스캔 포인트가 포함되는지 확인
            all_points = np.vstack((global_scan_points, np.array([[map_min_x, map_min_y], [map_max_x, map_max_y]])))
            new_min_x, new_min_y, new_max_x, new_max_y, new_xw, new_yw = \
                calc_grid_map_config(all_points[:, 0], all_points[:, 1], 0.1)

            # 맵의 크기가 변하면 새로운 맵을 생성하고 기존 맵의 내용을 복사합니다.
            if new_xw > map_xw or new_yw > map_yw:
                new_map = np.ones((new_xw, new_yw)) / 2
                offset_x = int(round((map_min_x - new_min_x) / 0.1))
                offset_y = int(round((map_min_y - new_min_y) / 0.1))
                new_map[offset_x:offset_x+map_xw, offset_y:offset_y+map_yw] = occupancy_map
                
                occupancy_map = new_map
                map_min_x, map_min_y, map_max_x, map_max_y = new_min_x, new_min_y, new_max_x, new_max_y
                map_xw, map_yw = new_xw, new_yw
                
                ax.set_xlim(map_min_x, map_max_x)
                ax.set_ylim(map_min_y, map_max_y)

        # Ray Casting을 사용하여 맵 업데이트
        center_x_grid = int(round((robot_pose[0] - map_min_x) / 0.1))
        center_y_grid = int(round((robot_pose[1] - map_min_y) / 0.1))

        for (x, y) in global_scan_points:
            ix = int(round((x - map_min_x) / 0.1))
            iy = int(round((y - map_min_y) / 0.1))
            
            laser_beams = bresenham((center_x_grid, center_y_grid), (ix, iy))
            
            for laser_beam in laser_beams:
                if 0 <= laser_beam[0] < map_xw and 0 <= laser_beam[1] < map_yw:
                    occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0  # free area
            
            if 0 <= ix < map_xw and 0 <= iy < map_yw:
                occupancy_map[ix][iy] = 1.0  # occupied area
        
        path_history.append(robot_pose[:2].tolist())
        
        # --- 시각화 업데이트 ---
        if map_image is None:
            map_image = ax.imshow(
                occupancy_map.T,
                cmap="gray",
                origin="lower",
                extent=[map_min_x, map_max_x, map_min_y, map_max_y]
            )
        else:
            map_image.set_data(occupancy_map.T)
        
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