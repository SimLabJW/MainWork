import zmq
import signal
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:8788")

def signal_handler(sig, frame):
    print("\n서버를 종료합니다.")
    socket.close()
    context.term()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
print("REP 서버 시작됨 (Ctrl+C로 종료)")

# --- occupancy grid 설정 ---
GRID_RES = 0.1  # 10cm
MAP_SIZE = 400  # 40m x 40m
MAP_CENTER = MAP_SIZE // 2
occupancy = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)

# --- matplotlib 초기화 ---
plt.ion()
fig, ax = plt.subplots()
map_img = ax.imshow(occupancy, cmap="gray", origin="lower",
                    extent=[-MAP_SIZE*GRID_RES/2, MAP_SIZE*GRID_RES/2,
                            -MAP_SIZE*GRID_RES/2, MAP_SIZE*GRID_RES/2])
robot_marker, = ax.plot([], [], 'ro', markersize=6, label="Robot")
path_line, = ax.plot([], [], 'k-', linewidth=1, label="Path")

ax.set_title("Unity LiDAR SLAM-like (Occupancy Grid)")
ax.legend()

pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
path = []

def to_grid(x, y):
    i = int(round(x / GRID_RES)) + MAP_CENTER
    j = int(round(y / GRID_RES)) + MAP_CENTER
    if 0 <= i < MAP_SIZE and 0 <= j < MAP_SIZE:
        return i, j
    return None

while True:
    try:
        message = socket.recv_string()
        socket.send_string("ack")

        lines = message.strip().splitlines()
        local_angles, local_distances = [], []
        delta_pose = None

        for line in lines:
            parts = line.split(",")
            if parts[0] == "POSE":
                dx, dy, dtheta = float(parts[1]), float(parts[2]), float(parts[3])
                delta_pose = (dx, dy, dtheta)
            else:
                try:
                    angle = float(parts[0])
                    dist = float(parts[1]) / 1000.0  # mm -> m
                    inten = float(parts[2])
                    if inten == 0: 
                        continue
                    local_angles.append(angle)
                    local_distances.append(dist)
                except:
                    continue

        # pose 업데이트
        if delta_pose:
            dx, dy, dtheta = delta_pose
            c, s = cos(pose[2]), sin(pose[2])
            global_dx = dx * c - dy * s
            global_dy = dx * s + dy * c
            pose[0] += global_dx
            pose[1] += global_dy
            pose[2] += dtheta
            path.append((pose[0], pose[1]))

        # 스캔 → occupancy grid 업데이트
        for angle, dist in zip(local_angles, local_distances):
            lx = dist * cos(angle)
            ly = dist * sin(angle)
            gx = pose[0] + lx * cos(pose[2]) - ly * sin(pose[2])
            gy = pose[1] + lx * sin(pose[2]) + ly * cos(pose[2])
            grid_idx = to_grid(gx, gy)
            if grid_idx:
                occupancy[grid_idx] = 255  # occupied

        # 시각화
        map_img.set_data(occupancy)
        robot_marker.set_data([pose[0]], [pose[1]])
        if path:
            px, py = zip(*path)
            path_line.set_data(px, py)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    except zmq.error.ContextTerminated:
        break
