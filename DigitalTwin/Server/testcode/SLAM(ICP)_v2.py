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
        # === 설정 (가볍게 만드는 핵심) ===
        self.MAX_POINTS = 120_000     # 누적 포인트 상한 (슬라이딩 윈도우)
        self.VIS_EVERY  = 10          # n 프레임마다 축/리밋 갱신
        self.SLEEP_VIZ  = 0.02        # 그리기 주기

        # === 원래 구조 유지 ===
        self.scan_x = deque()
        self.scan_y = deque()
        self.path_x = deque()
        self.path_y = deque()
        self.current_pose = (0.0, 0.0, 0.0)
        self.current_scan = []

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:8788")

        self.running = True

        self.fig = None
        self.ax = None
        self.scan_dots = None
        self.path_line = None
        self.pose_marker = None

        self._frame = 0  # 시각화 프레임 카운터

        print("RealtimeSLAM initialized, waiting for messages...")

    def _append_scan_point(self, gx, gy):
        self.scan_x.append(gx); self.scan_y.append(gy)
 

    def parse_and_update(self, message):
        for line in message.strip().split("\n"):
            parts = line.strip().split(",")
            if not parts or len(parts) < 3:
                continue

            if parts[0] == "POSE":
                # === 네 원래 로직: 이전 pose로 현재 스캔을 전역(플롯) 변환 후 누적 ===
                theta_rad = self.current_pose[2]
                c, s = np.cos(theta_rad), np.sin(theta_rad)
                rot = np.array([[c, -s],[s, c]])
                offset_yx = np.array([self.current_pose[1], self.current_pose[0]])  # (y,x)

                # current_scan -> global(y,x)
                for px, py in self.current_scan:
                    gy, gx = rot.dot(np.array([px, py])) + offset_yx
                    self._append_scan_point(gx, gy)  # 플롯 좌표계(x=gx, y=gy)

                self.current_scan = []

                # === 새로운 오도메트리 포즈 저장 & 경로 기록(원래처럼 y,x 저장) ===
                try:
                    x, y, theta = map(float, parts[1:4])
                    self.current_pose = (x, y, theta)
                    self.path_x.append(x)  # 플롯 x <- y
                    self.path_y.append(y)  # 플롯 y <- x
                except Exception as e:
                    print("❌ Error parsing POSE:", e, "parts:", parts)

            else:
                # LiDAR 한 빔 (angle[rad], distance[mm], intensity)
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
                time.sleep(0.005)
            except Exception as e:
                print("❌ ZMQ Error:", e)
                time.sleep(0.05)

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True, alpha=0.3)

        # scatter 대신 점 플롯(Line2D) — 업데이트가 훨씬 가벼움
        self.scan_dots, = self.ax.plot([], [], '.', color='blue', ms=2, linestyle='None', label='Global map')

        self.path_line,  = self.ax.plot([], [], color='red', lw=1.5, label='Path')
        self.pose_marker, = self.ax.plot([], [], 'ro', ms=5, label='Robot')
        self.ax.legend(loc='upper right')

    def viz_loop(self):
        if self.fig is None:
            self.setup_plot()

        while self.running:
            try:
                self._frame += 1

                # 포인트/경로 데이터 꺼내서 세팅 (형변환 비용 최소화)
                if len(self.scan_x) > 0:
                    self.scan_dots.set_data(self.scan_x, self.scan_y)

                if len(self.path_x) > 0:
                    self.path_line.set_data(self.path_x, self.path_y)
                    self.pose_marker.set_data([self.path_x[-1]], [self.path_y[-1]])

                # 축/리밋은 매 N프레임마다만 갱신
                if self._frame % self.VIS_EVERY == 0:
                    xs = []
                    ys = []
                    if len(self.scan_x) > 0:
                        xs.extend((self.scan_x[0], self.scan_x[-1]))  # 실제 min/max 계산을 줄이려면 간단 추정
                        ys.extend((self.scan_y[0], self.scan_y[-1]))
                        # 더 정확히 하려면 아래 두 줄 사용(조금 느림)
                        xs = list(self.scan_x); ys = list(self.scan_y)
                    if len(self.path_x) > 0:
                        xs += list(self.path_x); ys += list(self.path_y)

                    if xs:
                        xs = np.asarray(xs); ys = np.asarray(ys)
                        m = 2.0
                        self.ax.set_xlim(xs.min()-m, xs.max()+m)
                        self.ax.set_ylim(ys.min()-m, ys.max()+m)

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
