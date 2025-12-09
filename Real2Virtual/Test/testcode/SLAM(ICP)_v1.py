import zmq
import signal
import sys
import threading
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

# ==============================
# ICP Matching Class
# ==============================
class ICP_Matching:
    def __init__(self):
        self.show_animation = False  # True면 매 step마다 점군 표시
        self.EPS = 0.0001
        self.MAX_ITER = 50

    def icp_matching(self, previous_points, current_points):
        """
        Iterative Closest Point matching
        - input
        previous_points: (2, N) array (누적 맵)
        current_points: (2, N) array (현재 스캔)
        - output
        R: 2x2 Rotation matrix
        T: 2D Translation vector
        """
        H = None
        dError = np.inf
        preError = np.inf
        count = 0

        while dError >= self.EPS:
            count += 1

            indexes, error = self.nearest_neighbor_association(previous_points, current_points)
            Rt, Tt = self.svd_motion_estimation(previous_points[:, indexes], current_points)
            
            # update current points
            current_points = (Rt @ current_points) + Tt[:, np.newaxis]

            dError = preError - error
            if dError < 0:
                break

            preError = error
            H = self.update_homogeneous_matrix(H, Rt, Tt)

            if dError <= self.EPS or count >= self.MAX_ITER:
                break

        if H is None:
            return np.eye(2), np.zeros(2)

        R = np.array(H[0:-1, 0:-1])
        T = np.array(H[0:-1, -1])
        return R, T

    def update_homogeneous_matrix(self, Hin, R, T):
        H = np.eye(3)
        H[0:2, 0:2] = R
        H[0:2, 2] = T
        if Hin is None:
            return H
        else:
            return Hin @ H

    def nearest_neighbor_association(self, previous_points, current_points):
        # 🔴 잘못된 부분: 같은 크기 가정
        # delta_points = previous_points - current_points
        # d = np.linalg.norm(delta_points, axis=0)
        # error = sum(d)

        # ✅ 수정: 최근접점 매칭으로 error 계산
        d = np.linalg.norm(np.repeat(current_points, previous_points.shape[1], axis=1)
                        - np.tile(previous_points, (1, current_points.shape[1])), axis=0)
        indexes = np.argmin(d.reshape(current_points.shape[1], previous_points.shape[1]), axis=1)

        # error = 현재 스캔 점과 매칭된 previous 점 사이의 거리 합
        matched_prev = previous_points[:, indexes]
        delta_points = matched_prev - current_points
        error = np.sum(np.linalg.norm(delta_points, axis=0))

        return indexes, error


    def svd_motion_estimation(self, previous_points, current_points):
        pm = np.mean(previous_points, axis=1)
        cm = np.mean(current_points, axis=1)

        p_shift = previous_points - pm[:, np.newaxis]
        c_shift = current_points - cm[:, np.newaxis]

        W = c_shift @ p_shift.T
        u, _, vh = np.linalg.svd(W)

        R = (u @ vh).T
        t = pm - (R @ cm)
        return R, t


# ==============================
# Realtime SLAM Class
# ==============================
class RealtimeSLAM:
    def __init__(self):
        self.icp = ICP_Matching()
        self.global_map_points = None  # 누적 맵 (2, N)
        self.path_points = deque()
        self.current_pose = (0.0, 0.0, 0.0)  # x, y, yaw(rad)
        self.current_scan = []

        # ZMQ 세팅
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:8788")

        self.running = True

        # 시각화 세팅
        self.fig = None
        self.ax = None
        self.scan_plot = None
        self.path_line = None
        self.pose_marker = None

        print("RealtimeSLAM initialized, waiting for messages...")

    def parse_and_update(self, message):
        """
        Unity → Python 메시지
        - LiDAR: angle(rad), distance(mm), intensity
        - POSE: POSE,x,y,theta
        """
        for line in message.strip().split("\n"):
            parts = line.strip().split(",")
            if not parts or len(parts) < 3:
                continue

            if parts[0] == "POSE":
                try:
                    x, y, theta = map(float, parts[1:4])
                    self.current_pose = (x, y, theta)
                except Exception as e:
                    print("❌ Error parsing POSE:", e, "parts:", parts)
                    return

                # === 현재 스캔을 전역 좌표로 변환 ===
                theta_rad = self.current_pose[2]
                rot = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                [np.sin(theta_rad),  np.cos(theta_rad)]])
                offset = np.array([self.current_pose[0], self.current_pose[1]])
                current_points = np.array([rot.dot(np.array([px, py])) + offset
                                           for px, py in self.current_scan]).T  # (2, N)

                # === ICP (Scan-to-Map) ===
                if self.global_map_points is not None and self.global_map_points.shape[1] > 10:
                    R, T = self.icp.icp_matching(self.global_map_points, current_points)

                    dx, dy = T[0], T[1]
                    dtheta = np.arctan2(R[1, 0], R[0, 0])
                    self.current_pose = (
                        self.current_pose[0] + dx,
                        self.current_pose[1] + dy,
                        self.current_pose[2] + dtheta
                    )

                    # pose 보정 후 다시 스캔 변환
                    rot_corr = np.array([[np.cos(self.current_pose[2]), -np.sin(self.current_pose[2])],
                                         [np.sin(self.current_pose[2]),  np.cos(self.current_pose[2])]])
                    offset_corr = np.array([self.current_pose[0], self.current_pose[1]])
                    current_points = np.array([rot_corr.dot(np.array([px, py])) + offset_corr
                                               for px, py in self.current_scan]).T

                # === 맵에 누적 ===
                if self.global_map_points is None:
                    self.global_map_points = current_points
                else:
                    self.global_map_points = np.hstack((self.global_map_points, current_points))

                # 경로 기록
                self.path_points.append((self.current_pose[0], self.current_pose[1]))
                self.current_scan = []

            else:
                try:
                    angle, distance, intensity = map(float, parts)
                    if 0 < distance < 40000:
                        r = distance / 1000.0  # m 단위
                        px = r * np.cos(angle)
                        py = r * np.sin(angle)
                        self.current_scan.append((px, py))
                except Exception as e:
                    print("❌ Error parsing LiDAR:", e, "parts:", parts)

    def zmq_loop(self):
        while self.running:
            try:
                msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                self.socket.send_string("ack")
                self.parse_and_update(msg)
            except zmq.error.Again:
                time.sleep(0.01)
            except Exception as e:
                print("❌ ZMQ Error:", e)
                time.sleep(0.1)

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True)
        self.scan_plot = self.ax.scatter([], [], s=5, c='blue')
        self.path_line, = self.ax.plot([], [], 'r-')
        self.pose_marker, = self.ax.plot([], [], 'ro')

    def viz_loop(self):
        if self.fig is None:
            self.setup_plot()

        while self.running:
            try:
                scan_arr = self.global_map_points.T if self.global_map_points is not None else np.empty((0, 2))
                path_arr = np.array(self.path_points) if len(self.path_points) > 0 else np.empty((0, 2))

                if scan_arr.shape[0] > 0:
                    self.scan_plot.set_offsets(scan_arr)

                if path_arr.shape[0] > 0:
                    self.path_line.set_data(path_arr[:, 0], path_arr[:, 1])
                    self.pose_marker.set_data([path_arr[-1, 0]], [path_arr[-1, 1]])

                # === 축 범위 자동 설정 + 최소 크기 보장 ===
                if scan_arr.shape[0] > 0 or path_arr.shape[0] > 0:
                    if scan_arr.shape[0] > 0 and path_arr.shape[0] > 0:
                        all_x = np.concatenate([scan_arr[:, 0], path_arr[:, 0]])
                        all_y = np.concatenate([scan_arr[:, 1], path_arr[:, 1]])
                    elif scan_arr.shape[0] > 0:
                        all_x, all_y = scan_arr[:, 0], scan_arr[:, 1]
                    else:
                        all_x, all_y = path_arr[:, 0], path_arr[:, 1]

                    margin = 1.0  # 1m 여유
                    x_min, x_max = all_x.min() - margin, all_x.max() + margin
                    y_min, y_max = all_y.min() - margin, all_y.max() + margin

                    # ✅ 최소 크기 보장 (예: 10m × 10m)
                    min_size = 10.0
                    if (x_max - x_min) < min_size:
                        cx = (x_max + x_min) / 2
                        x_min, x_max = cx - min_size / 2, cx + min_size / 2
                    if (y_max - y_min) < min_size:
                        cy = (y_max + y_min) / 2
                        y_min, y_max = cy - min_size / 2, cy + min_size / 2

                    self.ax.set_xlim(x_min, x_max)
                    self.ax.set_ylim(y_min, y_max)

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                time.sleep(0.05)

            except Exception as e:
                print("❌ Viz Error:", e)
                time.sleep(0.1)


    def run(self):
        def handler(sig, frame):
            print("\n🛑 Shutting down...")
            self.running = False
            self.socket.close()
            self.context.term()
            plt.close('all')
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)

        t_zmq = threading.Thread(target=self.zmq_loop, daemon=True)
        t_viz = threading.Thread(target=self.viz_loop, daemon=True)

        t_zmq.start()
        t_viz.start()

        while self.running:
            time.sleep(0.5)


if __name__ == "__main__":
    slam = RealtimeSLAM()
    slam.run()
