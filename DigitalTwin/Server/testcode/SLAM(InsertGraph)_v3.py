import zmq
import signal
import sys
import threading
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

# Graph-SLAM backend
from newslam.graph import Graph
from newslam.pose_se2 import PoseSE2


class RealtimeSLAM:
    def __init__(self):
        # === 설정 ===
        self.MAX_POINTS = 300_000         # 점군 상한 조금 여유
        self.VIS_EVERY = 5                # 축 갱신 주기
        self.SLEEP_VIZ = 0.02

        # === 데이터 저장 ===
        self.scan_x = deque()
        self.scan_y = deque()
        self.path_x = deque()
        self.path_y = deque()
        self.current_pose = (0.0, 0.0, 0.0)
        self.current_scan = []            # Pose 오기 전까지 임시 저장 (로봇 좌표계)

        # === Graph-SLAM 관련 ===
        self.graph = Graph(edges=[], vertices=[])
        self.node_id = 0
        self.prev_pose = (0.0, 0.0, 0.0)
        self.last_opt_time = time.time()
        self.OPT_INTERVAL = 5.0  # 5초마다 최적화

        # === ZMQ ===
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:8788")

        # === 시각화 ===
        self.fig = None
        self.ax = None
        self.scan_dots = None
        self.path_line = None
        self.pose_marker = None
        self._frame = 0

        self.running = True
        print("RealtimeSLAM initialized, waiting for messages...")

    # ========== Graph-SLAM 기능 ==========
    def add_pose_node(self, pose_tuple):
        x, y, theta = pose_tuple
        pose = PoseSE2([x, y], theta)

        self.graph.add_vertex(self.node_id, pose)

        if self.node_id > 0:
            dx = x - self.prev_pose[0]
            dy = y - self.prev_pose[1]
            dtheta = theta - self.prev_pose[2]
            measurement = PoseSE2([dx, dy], dtheta)
            self.graph.add_edge([self.node_id - 1, self.node_id],
                                measurement=measurement,
                                information=np.identity(3))

        self.prev_pose = (x, y, theta)
        self.node_id += 1

    def try_loop_closure(self, pose_tuple):
        x, y, theta = pose_tuple
        for past_v in self.graph._vertices:
            dx = x - past_v.pose.position[0]
            dy = y - past_v.pose.position[1]
            if np.hypot(dx, dy) < 1.0 and past_v.id != self.node_id - 1:
                measurement = PoseSE2([dx, dy], theta - past_v.pose.orientation)
                self.graph.add_edge([past_v.id, self.node_id - 1],
                                    measurement=measurement,
                                    information=np.identity(3))
                print(f"🔗 Loop closure between {past_v.id} and {self.node_id-1}")
                break

    # ========== 맵 누적 ==========
    def add_scan_to_map(self, pose, scan):
        """로봇 좌표계 스캔 점들을 전역 좌표로 변환 후 누적"""
        x, y, theta = pose
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for px, py in scan:
            gx = x + cos_t * px - sin_t * py
            gy = y + sin_t * px + cos_t * py
            self._append_scan_point(gx, gy)

    def _append_scan_point(self, gx, gy):
        self.scan_x.append(gx)
        self.scan_y.append(gy)
        # 오래된 점 안정 제거
        while len(self.scan_x) > self.MAX_POINTS:
            self.scan_x.popleft()
            self.scan_y.popleft()

    # ========== 메시지 처리 ==========
    def parse_and_update(self, message):
        for line in message.strip().split("\n"):
            parts = line.strip().split(",")
            if not parts or len(parts) < 3:
                continue

            if parts[0] == "POSE":
                try:
                    x, y, theta = map(float, parts[1:4])
                    self.current_pose = (x, y, theta)

                    # === Pose 갱신 후, 누적된 LiDAR 점들을 전역 변환 ===
                    if self.current_scan:
                        self.add_scan_to_map((x, y, theta), self.current_scan)
                        print(
                            f"POSE {x:.2f}, {y:.2f}, {theta:.2f} | "
                            f"Added {len(self.current_scan)} scan points | total={len(self.scan_x)}"
                        )
                        self.current_scan = []

                    # Graph-SLAM 노드 추가/루프클로저
                    self.add_pose_node((x, y, theta))
                    self.try_loop_closure((x, y, theta))

                    # 최적화 실행
                    if time.time() - self.last_opt_time > self.OPT_INTERVAL:
                        print("🔧 Optimizing graph...")
                        self.graph.optimize()
                        self.last_opt_time = time.time()

                    # 경로 업데이트
                    poses = [v.pose for v in self.graph._vertices]
                    self.path_x = deque([p.position[0] for p in poses])
                    self.path_y = deque([p.position[1] for p in poses])

                except Exception as e:
                    print("❌ Error parsing POSE:", e, "parts:", parts)

            else:  # LiDAR 데이터
                try:
                    angle, distance, intensity = map(float, parts)
                    if 0 < distance < 40000:  # mm 가정
                        # ⬇️ 들어오는 angle이 deg이면 자동 변환
                        if abs(angle) > (2*np.pi + 1e-3):
                            angle = np.deg2rad(angle)
                        r = distance / 1000.0  # m
                        px = r * np.cos(angle)
                        py = r * np.sin(angle)
                        self.current_scan.append((px, py))
                except Exception:
                    continue

    # ========== ZMQ & 시각화 ==========
    def zmq_loop(self):
        while self.running:
            try:
                msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                self.socket.send_string("ack")
                self.parse_and_update(msg)
            except zmq.error.Again:
                time.sleep(0.005)
            except Exception as e:
                print("❌ ZMQ Error:", e)
                time.sleep(0.05)

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 9))
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True, alpha=0.3)

        # ⬇️ 점/선 가시성 개선
        self.scan_dots, = self.ax.plot([], [], '.', color='tab:blue',
                                       ms=4, linestyle='None', alpha=0.9, zorder=2, label='Map')
        self.path_line, = self.ax.plot([], [], color='tab:red',
                                       lw=1.5, alpha=0.9, zorder=3, label='Path')
        self.pose_marker, = self.ax.plot([], [], 'o', ms=5,
                                         color='tab:red', alpha=0.9, zorder=4, label='Robot')
        self.ax.legend(loc='upper right')

    def viz_loop(self):
        if self.fig is None:
            self.setup_plot()

        while self.running:
            try:
                self._frame += 1

                # 데이터 반영
                if self.scan_x:
                    self.scan_dots.set_data(self.scan_x, self.scan_y)
                if self.path_x:
                    self.path_line.set_data(self.path_x, self.path_y)
                    self.pose_marker.set_data([self.path_x[-1]], [self.path_y[-1]])

                # ⬇️ 축을 "스캔 + 경로" 전체로 갱신 (이게 핵심!)
                if self._frame % self.VIS_EVERY == 0:
                    xs, ys = [], []
                    if self.scan_x:
                        xs += list(self.scan_x)
                        ys += list(self.scan_y)
                    if self.path_x:
                        xs += list(self.path_x)
                        ys += list(self.path_y)
                    if xs:
                        xs = np.asarray(xs); ys = np.asarray(ys)
                        xmin, xmax = np.nanmin(xs), np.nanmax(xs)
                        ymin, ymax = np.nanmin(ys), np.nanmax(ys)
                        pad = max(1.5, 0.05 * max(xmax-xmin, ymax-ymin))
                        self.ax.set_xlim(xmin - pad, xmax + pad)
                        self.ax.set_ylim(ymin - pad, ymax + pad)

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                time.sleep(self.SLEEP_VIZ)
            except Exception as e:
                print("❌ Viz Error:", e)
                time.sleep(0.05)

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
            time.sleep(0.2)


if __name__ == "__main__":
    slam = RealtimeSLAM()
    slam.run()
