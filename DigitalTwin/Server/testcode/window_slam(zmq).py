import zmq
import signal
import sys
import threading
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

class RealtimeSLAM:
    def __init__(self):
        # self.scan_points = deque(maxlen=20000)
        # self.path_points = deque(maxlen=5000)
        self.scan_points = deque()
        self.path_points = deque()
        self.current_pose = (0.0, 0.0, 0.0)
        self.current_scan = []

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:8788")

        self.running = True

        self.fig = None
        self.ax = None
        self.scan_plot = None
        self.path_line = None
        self.pose_marker = None

        print("RealtimeSLAM initialized, waiting for messages...")

    def parse_and_update(self, message):
        for line in message.strip().split("\n"):
            parts = line.strip().split(",")
            if not parts or len(parts) < 3:
                continue
            if parts[0] == "POSE":
                
                theta_rad = self.current_pose[2]
                rot = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                [np.sin(theta_rad),  np.cos(theta_rad)]])
                offset = np.array([self.current_pose[1], self.current_pose[0]])
                for px, py in self.current_scan:
                    global_pt = rot.dot(np.array([px, py])) + offset
                    self.scan_points.append(global_pt)
                self.current_scan = []

                try:
                    x, y, theta = map(float, parts[1:4])
                    self.current_pose = (x, y, theta)
                    self.path_points.append((y, x))
                
                except Exception as e:
                    print("❌ Error parsing POSE:", e, "parts:", parts)
            else:
                try:
                    angle, distance, intensity = map(float, parts)
                    if 0 < distance < 40000:
                        r = distance / 1000.0
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
        self.scan_plot = self.ax.scatter([], [], s=10, c='blue')  # 점 크기 확대
        self.path_line, = self.ax.plot([], [], 'r-')
        self.pose_marker, = self.ax.plot([], [], 'ro')


    def viz_loop(self):
        if self.fig is None:
            self.setup_plot()

        while self.running:
            try:
                scan_arr = np.array(self.scan_points) if len(self.scan_points) > 0 else np.empty((0, 2))
                path_arr = np.array(self.path_points) if len(self.path_points) > 0 else np.empty((0, 2))

                if scan_arr.shape[0] > 0:
                    self.scan_plot.set_offsets(scan_arr)

                if path_arr.shape[0] > 0:
                    self.path_line.set_data(path_arr[:, 0], path_arr[:, 1])
                    self.pose_marker.set_data([path_arr[-1, 0]], [path_arr[-1, 1]])

                # ✅ 축 범위 자동 설정
                if scan_arr.shape[0] > 0 or path_arr.shape[0] > 0:
                    all_x = np.concatenate([scan_arr[:, 0], path_arr[:, 0]]) if scan_arr.size > 0 else path_arr[:, 0]
                    all_y = np.concatenate([scan_arr[:, 1], path_arr[:, 1]]) if scan_arr.size > 0 else path_arr[:, 1]
                    margin = 2.0
                    self.ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
                    self.ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                time.sleep(0.1)
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
