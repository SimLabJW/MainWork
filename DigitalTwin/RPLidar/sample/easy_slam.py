#!/usr/bin/env python3
import numpy as np
from math import cos, sin, radians, atan2
from rplidar import RPLidar
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
import time

PORT_NAME = 'COM3'
DMAX = 10000
GRID_RESOLUTION = 0.05  # 50mm
TTL = 3.0
STATIC_THRESHOLD = 5
MAP_SIZE = 400
MAP_CENTER = MAP_SIZE // 2
MERGE_RADIUS = 0.1  # 점 병합 기준 거리 (10cm)


def round_coord(x, y):
    return (round(x / GRID_RESOLUTION) * GRID_RESOLUTION,
            round(y / GRID_RESOLUTION) * GRID_RESOLUTION)

def to_grid_index(x, y):
    i = int(round(x / GRID_RESOLUTION)) + MAP_CENTER
    j = int(round(y / GRID_RESOLUTION)) + MAP_CENTER
    if 0 <= i < MAP_SIZE and 0 <= j < MAP_SIZE:
        return i, j
    else:
        return None

def filter_scan(scan):
    pts = []
    for quality, angle_deg, distance in scan:
        if 0 < distance < DMAX:
            theta = radians(angle_deg)
            x = distance * cos(theta)
            y = distance * sin(theta)
            pts.append([x, y])
    return np.array(pts)

def icp(a_pts, b_pts, max_iter=20, tolerance=1e-4):
    R = np.eye(2)
    t = np.zeros((2,))
    prev_error = float('inf')

    for _ in range(max_iter):
        a_trans = (a_pts @ R.T) + t
        tree = cKDTree(b_pts)
        dists, idxs = tree.query(a_trans)
        b_match = b_pts[idxs]

        centroid_a = np.mean(a_trans, axis=0)
        centroid_b = np.mean(b_match, axis=0)
        aa = a_trans - centroid_a
        bb = b_match - centroid_b

        H = aa.T @ bb
        U, _, Vt = np.linalg.svd(H)
        R_i = Vt.T @ U.T
        if np.linalg.det(R_i) < 0:
            Vt[1,:] *= -1
            R_i = Vt.T @ U.T
        t_i = centroid_b - (centroid_a @ R_i.T)

        R = R_i @ R
        t = (t @ R_i.T) + t_i

        mean_error = np.mean(dists)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return R, t

def find_nearby_key(global_map, pt, radius=MERGE_RADIUS):
    pt_array = np.array(pt)
    for key in global_map.keys():
        if np.linalg.norm(np.array(key) - pt_array) <= radius:
            return key
    return None

def run_slam():
    lidar = RPLidar(PORT_NAME)
    lidar.clean_input()

    plt.ion()
    fig, ax = plt.subplots()
    sc_static = ax.scatter([], [], s=2, c='blue', label='Static')
    sc_dynamic = ax.scatter([], [], s=2, c='red', label='Dynamic')
    sc_current = ax.scatter([], [], s=10, c='orange', marker='x', label='Live Dynamic')
    robot_marker = Circle((0, 0), 100, color='green', label='Robot')
    direction_line, = ax.plot([], [], color='green')
    path_line, = ax.plot([], [], color='black', lw=1, label='Path')

    ax.add_patch(robot_marker)
    ax.set_aspect('equal')
    ax.set_xlim(-DMAX, DMAX)
    ax.set_ylim(-DMAX, DMAX)
    ax.set_title("SLAM Visualization (Static vs Dynamic)")
    ax.legend()

    global_map = {}
    occupancy_grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    path_history = []

    pose = np.array([0.0, 0.0, 0.0])
    prev_scan_pts = None

    iterator = lidar.iter_scans()
    try:
        for scan in iterator:
            now = time.time()
            scan_pts = filter_scan(scan)
            if len(scan_pts) < 10:
                continue

            if prev_scan_pts is None:
                prev_scan_pts = scan_pts
                continue

            try:
                R, t = icp(scan_pts, prev_scan_pts)
            except Exception as e:
                print("ICP 실패:", e)
                prev_scan_pts = scan_pts
                continue

            delta_theta = atan2(R[1,0], R[0,0])
            delta_x, delta_y = t
            c, s = cos(pose[2]), sin(pose[2])
            global_dx = delta_x * c - delta_y * s
            global_dy = delta_x * s + delta_y * c
            pose[0] += global_dx
            pose[1] += global_dy
            pose[2] += delta_theta

            pts_rot = scan_pts @ np.array([
                [cos(pose[2]), -sin(pose[2])],
                [sin(pose[2]),  cos(pose[2])]
            ])
            pts_trans = pts_rot + pose[:2]

            static_pts = []
            dynamic_pts = []
            live_dynamic_pts = []

            for pt in pts_trans:
                key = round_coord(pt[0], pt[1])
                nearby_key = find_nearby_key(global_map, key)

                if nearby_key:
                    entry = global_map[nearby_key]
                    entry['last_seen'] = now
                    entry['seen_count'] += 1
                    if entry['seen_count'] >= STATIC_THRESHOLD:
                        entry['is_static'] = True
                    if entry['is_static']:
                        static_pts.append(nearby_key)
                    else:
                        dynamic_pts.append(nearby_key)
                        live_dynamic_pts.append(pt)
                    grid_idx = to_grid_index(*nearby_key)
                else:
                    global_map[key] = {
                        'last_seen': now,
                        'seen_count': 1,
                        'is_static': False
                    }
                    dynamic_pts.append(key)
                    live_dynamic_pts.append(pt)
                    grid_idx = to_grid_index(*key)

                if grid_idx:
                    occupancy_grid[grid_idx] = 1

            keys_to_delete = [
                k for k, v in global_map.items()
                if not v['is_static'] and now - v['last_seen'] > TTL
            ]
            for k in keys_to_delete:
                del global_map[k]
                grid_idx = to_grid_index(*k)
                if grid_idx:
                    occupancy_grid[grid_idx] = 0

            path_history.append((pose[0], pose[1]))
            path_array = np.array(path_history)

            sc_static.set_offsets(np.array(static_pts) if static_pts else [])
            sc_dynamic.set_offsets(np.array(dynamic_pts) if dynamic_pts else [])
            sc_current.set_offsets(np.array(live_dynamic_pts) if live_dynamic_pts else [])

            robot_marker.center = (pose[0], pose[1])
            heading_length = 300
            dx = heading_length * cos(pose[2])
            dy = heading_length * sin(pose[2])
            direction_line.set_data([pose[0], pose[0] + dx], [pose[1], pose[1] + dy])
            path_line.set_data(path_array[:, 0], path_array[:, 1])

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

            prev_scan_pts = scan_pts

    except KeyboardInterrupt:
        print("\n중단됨")
        np.save("occupancy_grid.npy", occupancy_grid)
        print("📦 맵 저장됨: occupancy_grid.npy")

    finally:
        lidar.stop()
        lidar.disconnect()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    run_slam()