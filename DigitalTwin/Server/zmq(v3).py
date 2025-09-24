#!/usr/bin/env python3
import zmq
import signal
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from collections import deque

# Import the angle_mod function from your existing utils module
from utils.angle import angle_mod

# --- Core functions from PythonRobotics/Mapping ---
EXTEND_AREA = 1.0

def bresenham(start, end):
    """Bresenham's line algorithm."""
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
    """Calculates the map size based on obstacle points."""
    min_x = round(min(ox) - EXTEND_AREA / 2.0)
    min_y = round(min(oy) - EXTEND_AREA / 2.0)
    max_x = round(max(ox) + EXTEND_AREA / 2.0)
    max_y = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    return min_x, min_y, max_x, max_y, xw, yw

# ---- ZMQ Server Setup ----
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:8788")

def signal_handler(sig, frame):
    print("\nShutting down server.")
    plt.close('all')
    socket.close()
    context.term()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
print("ZMQ REP server started (Press Ctrl+C to exit).")

# --- Map and Robot State Initialization ---
occupancy_map = None
map_min_x, map_min_y, map_max_x, map_max_y = 0, 0, 0, 0
map_xw, map_yw = 0, 0
xy_resolution = 0.1 # 10cm resolution

robot_pose = np.array([0.0, 0.0, 0.0])  # [x(m), y(m), theta(rad)]
path_history = []
previous_scan_points = None

# --- Visualization Setup ---
plt.ion()
fig, ax = plt.subplots()

map_image = None
robot_plot, = ax.plot([], [], "ro", label="Robot")
path_plot, = ax.plot([], [], "b-", linewidth=1, label="Path")

ax.set_title("Unity LiDAR SLAM Visualization (ICP + Ray Casting)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.legend()
ax.grid(True)
plt.axis("equal")

# --- Main Loop: Receive Data and Perform SLAM ---
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
                    print("Odometry data parsing error.")
                    continue
            else:
                try:
                    # Unity sends: angle(rad), distance(mm), intensity(int)
                    angle_rad = float(parts[0])
                    distance_m = float(parts[1]) / 1000.0
                    
                    # Use only valid distances
                    if distance_m > 0.15 and distance_m < 20.0:
                        x = distance_m * np.cos(angle_rad)
                        y = distance_m * np.sin(angle_rad)
                        current_scan_points.append([x, y])
                except (ValueError, IndexError):
                    print(f"LiDAR data parsing error: {line}")
                    continue

        if not current_scan_points:
            print("No scan data. Waiting for next frame.")
            continue
        
        current_scan_points = np.array(current_scan_points)

        # The first scan has no previous data for comparison, so just save it and continue.
        if previous_scan_points is None:
            previous_scan_points = current_scan_points
            continue

        # --- SLAM: ICP-based Localization and Mapping ---
        
        # 1. Localization - ICP Matching
        # Convert 2D points to 3D point cloud (z-axis is 0)
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(np.hstack((current_scan_points, np.zeros((current_scan_points.shape[0], 1)))))
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(np.hstack((previous_scan_points, np.zeros((previous_scan_points.shape[0], 1)))))
        
        # Set initial ICP guess based on odometry
        initial_guess = np.eye(4)
        if odometry_delta is not None:
             theta = odometry_delta[2]
             T_odom = np.array([
                 [np.cos(theta), -np.sin(theta), 0, odometry_delta[0]],
                 [np.sin(theta), np.cos(theta), 0, odometry_delta[1]],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]
             ])
             initial_guess = T_odom
        
        # Align with ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source=source_pcd,
            target=target_pcd,
            max_correspondence_distance=0.5,
            init=initial_guess,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        
        # Extract the transformation matrix from ICP result
        transformation_matrix = reg_p2p.transformation
        
        # Extract translation and rotation
        T = transformation_matrix[:2, 3] # x, y translation
        R = transformation_matrix[:2, :2] # rotation matrix
        
        dtheta = np.arctan2(R[1, 0], R[0, 0])
        dx, dy = T[0], T[1]

        # Accumulate the refined pose using ICP results
        robot_pose[0] += dx
        robot_pose[1] += dy
        robot_pose[2] = angle_mod(robot_pose[2] + dtheta) # Normalize the angle using your function

        # 2. Mapping - Ray Casting
        # Transform LiDAR points to global coordinates based on the new pose
        cos_theta, sin_theta = np.cos(robot_pose[2]), np.sin(robot_pose[2])
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        global_scan_points = (rotation_matrix @ current_scan_points.T).T + robot_pose[:2]

        # Re-initialize the map if boundaries change
        if occupancy_map is None:
            map_min_x, map_min_y, map_max_x, map_max_y, map_xw, map_yw = \
                calc_grid_map_config(global_scan_points[:, 0], global_scan_points[:, 1], xy_resolution)
            occupancy_map = np.ones((map_xw, map_yw)) / 2 # Initialize as unknown space
        else:
            all_points = np.vstack((global_scan_points, np.array([[map_min_x, map_min_y], [map_max_x, map_max_y]])))
            new_min_x, new_min_y, new_max_x, new_max_y, new_xw, new_yw = \
                calc_grid_map_config(all_points[:, 0], all_points[:, 1], xy_resolution)

            # Expand the map if needed
            if new_xw > map_xw or new_yw > map_yw or new_min_x < map_min_x or new_min_y < map_min_y:
                new_map = np.ones((new_xw, new_yw)) / 2
                offset_x = int(round((map_min_x - new_min_x) / xy_resolution))
                offset_y = int(round((map_min_y - new_min_y) / xy_resolution))
                new_map[offset_x:offset_x+map_xw, offset_y:offset_y+map_yw] = occupancy_map
                
                occupancy_map = new_map
                map_min_x, map_min_y, map_max_x, map_max_y = new_min_x, new_min_y, new_max_x, new_max_y
                map_xw, map_yw = new_xw, new_yw
                
                ax.set_xlim(map_min_x, map_max_x)
                ax.set_ylim(map_min_y, map_max_y)

        # Update map using Ray Casting
        center_x_grid = int(round((robot_pose[0] - map_min_x) / xy_resolution))
        center_y_grid = int(round((robot_pose[1] - map_min_y) / xy_resolution))

        for (x, y) in global_scan_points:
            ix = int(round((x - map_min_x) / xy_resolution))
            iy = int(round((y - map_min_y) / xy_resolution))
            
            # Update the path traversed by the laser as 'free space'
            laser_beams = bresenham((center_x_grid, center_y_grid), (ix, iy))
            
            for laser_beam in laser_beams:
                if 0 <= laser_beam[0] < map_xw and 0 <= laser_beam[1] < map_yw:
                    occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0  # free area
            
            # Update the end point of the laser beam as 'occupied space'
            if 0 <= ix < map_xw and 0 <= iy < map_yw:
                occupancy_map[ix][iy] = 1.0  # occupied area
        
        path_history.append(robot_pose[:2].tolist())
        
        # --- Visualization Update ---
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
        
        # Save the current scan for the next frame
        previous_scan_points = current_scan_points

    except zmq.error.ContextTerminated:
        break
    except Exception as e:
        print(f"Error occurred: {e}")