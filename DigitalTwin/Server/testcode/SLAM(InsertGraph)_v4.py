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
        # === 시각화 / 성능 설정 ===
        self.MAX_POINTS = 300_000       # 점군 상한(슬라이딩 윈도우)
        self.VIS_EVERY = 5              # 축 갱신 주기
        self.SLEEP_VIZ = 0.02
        self.BEAM_DEG_THRESHOLD = 2.0   # (시각화용) 빔 간 각도가 너무 촘촘할 때 디시메이션

        # === OGM(Occupancy Grid) 설정 ===
        self.OGM_RES = 0.10             # m/cell (해상도)
        self.OGM_INIT_SIZE = (600, 600) # 초기 셀 크기 (HxW) -> 60m x 60m
        self.OGM_L_FREE = -1.0          # 자유 공간 log-odds
        self.OGM_L_OCC  = +2.0          # 점유 log-odds
        self.OGM_CLAMP  = (-5.0, 5.0)   # log-odds 클램핑
        self.OGM_SUBSAMPLE = 1          # 빔 디시메이션(1이면 모든 빔 사용)

        # === 데이터 저장 ===
        self.scan_x = deque()
        self.scan_y = deque()
        self.path_x = deque()
        self.path_y = deque()
        self.current_pose = (0.0, 0.0, 0.0)
        self.current_scan = []          # Pose 오기 전까지 임시 저장 (로봇 좌표계)

        # === Graph-SLAM 관련 ===
        self.graph = Graph(edges=[], vertices=[])
        self.node_id = 0
        self.prev_pose = (0.0, 0.0, 0.0)
        self.last_opt_time = time.time()
        self.OPT_INTERVAL = 5.0         # 5초마다 최적화

        # === OGM 버퍼 준비 ===
        H, W = self.OGM_INIT_SIZE
        self.grid_logodds = np.zeros((H, W), dtype=np.float32)
        # 그리드의 (0,0) 셀에 해당하는 월드 좌표 (m)
        # 초기에는 로봇이 중앙에 오도록 원점을 설정
        self.grid_origin_world = (-W/2 * self.OGM_RES, -H/2 * self.OGM_RES)  # (x0, y0) [m]

        # === ZMQ ===
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        this_addr = "tcp://*:8788"
        self.socket.bind(this_addr)

        # === 시각화 ===
        self.fig = None
        self.ax = None
        self.scan_dots = None
        self.path_line = None
        self.pose_marker = None
        self.ogm_img = None
        self._frame = 0

        self.running = True
        print("RealtimeSLAM initialized, waiting for messages...")

    # ================== 좌표계 & OGM 유틸 ==================
    def world_to_map(self, x, y):
        """월드(m) -> 맵 셀 인덱스(iy, ix) [행,열]"""
        x0, y0 = self.grid_origin_world
        ix = int(np.floor((x - x0) / self.OGM_RES))
        iy = int(np.floor((y - y0) / self.OGM_RES))
        return iy, ix

    def map_to_world(self, iy, ix):
        """맵 셀 인덱스(iy, ix) -> 월드(m)"""
        x0, y0 = self.grid_origin_world
        x = x0 + (ix + 0.5) * self.OGM_RES
        y = y0 + (iy + 0.5) * self.OGM_RES
        return x, y

    def _ensure_in_grid(self, iy, ix):
        """인덱스가 그리드 바운드 밖이면 np.pad로 동적 확장."""
        H, W = self.grid_logodds.shape
        pad_top = pad_bottom = pad_left = pad_right = 0

        if iy < 0: pad_top = -iy
        if ix < 0: pad_left = -ix
        if iy >= H: pad_bottom = iy - H + 1
        if ix >= W: pad_right  = ix - W + 1

        if (pad_top or pad_bottom or pad_left or pad_right):
            self.grid_logodds = np.pad(
                self.grid_logodds,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant', constant_values=0.0
            )
            # 원점(world)에 대한 그리드 셀 (0,0)이 위/왼쪽으로 늘었으므로 origin 이동
            dx = -pad_left * self.OGM_RES
            dy = -pad_top  * self.OGM_RES
            x0, y0 = self.grid_origin_world
            self.grid_origin_world = (x0 + dx, y0 + dy)

    @staticmethod
    def _bresenham(iy0, ix0, iy1, ix1):
        """Bresenham ray-tracing: (iy0,ix0) -> (iy1,ix1)까지 경로 셀 나열(끝 점 포함 X)."""
        cells = []
        dy = abs(iy1 - iy0)
        dx = abs(ix1 - ix0)
        sy = 1 if iy0 < iy1 else -1
        sx = 1 if ix0 < ix1 else -1
        err = dx - dy
        y, x = iy0, ix0
        while not (y == iy1 and x == ix1):
            cells.append((y, x))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    def ogm_update_scan(self, pose, scan):
        """현재 pose에서 받은 스캔(로봇 좌표계)을 OGM에 반영."""
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)

        # 센서 원점(월드 좌표 -> 맵 인덱스)
        iy0, ix0 = self.world_to_map(x, y)
        self._ensure_in_grid(iy0, ix0)

        for i, (px, py) in enumerate(scan):
            if (self.OGM_SUBSAMPLE > 1) and (i % self.OGM_SUBSAMPLE != 0):
                continue

            # 로봇좌표 -> 월드
            gx = x + c*px - s*py
            gy = y + s*px + c*py

            # 월드 -> 맵 인덱스
            iy1, ix1 = self.world_to_map(gx, gy)
            self._ensure_in_grid(iy1, ix1)

            # 레이 트레이싱 (자유 공간)
            free_cells = self._bresenham(iy0, ix0, iy1, ix1)
            if free_cells:
                ys, xs = zip(*free_cells)
                self.grid_logodds[ys, xs] += self.OGM_L_FREE

            # 엔드포인트(점유)
            self.grid_logodds[iy1, ix1] += self.OGM_L_OCC

        # 클램핑
        lo_min, lo_max = self.OGM_CLAMP
        np.clip(self.grid_logodds, lo_min, lo_max, out=self.grid_logodds)

    # ================== Graph-SLAM 기능 ==================
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

    # ========== 점군 누적(그림용, 선택) ==========
    def add_scan_to_map_points(self, pose, scan):
        x, y, theta = pose
        c, s = np.cos(theta), np.sin(theta)
        for px, py in scan:
            gx = x + c*px - s*py
            gy = y + s*px + c*py
            self._append_scan_point(gx, gy)

    def _append_scan_point(self, gx, gy):
        self.scan_x.append(gx)
        self.scan_y.append(gy)
        while len(self.scan_x) > self.MAX_POINTS:
            self.scan_x.popleft()
            self.scan_y.popleft()

    # ================== 메시지 처리 ==================
    def parse_and_update(self, message):
        for line in message.strip().split("\n"):
            parts = line.strip().split(",")
            if not parts or len(parts) < 3:
                continue

            if parts[0] == "POSE":
                try:
                    x, y, theta = map(float, parts[1:4])
                    self.current_pose = (x, y, theta)

                    # 누적된 LiDAR 점들을 OGM + (선택)점군 지도에 반영
                    if self.current_scan:
                        self.ogm_update_scan((x, y, theta), self.current_scan)
                        self.add_scan_to_map_points((x, y, theta), self.current_scan)
                        print(
                            f"POSE {x:.2f}, {y:.2f}, {theta:.2f} | "
                            f"Added {len(self.current_scan)} scan points | total_pts={len(self.scan_x)}"
                        )
                        self.current_scan = []

                    # Graph-SLAM 업데이트
                    self.add_pose_node((x, y, theta))
                    self.try_loop_closure((x, y, theta))

                    if time.time() - self.last_opt_time > self.OPT_INTERVAL:
                        print("🔧 Optimizing graph...")
                        self.graph.optimize()
                        self.last_opt_time = time.time()

                    poses = [v.pose for v in self.graph._vertices]
                    self.path_x = deque([p.position[0] for p in poses])
                    self.path_y = deque([p.position[1] for p in poses])

                except Exception as e:
                    print("❌ Error parsing POSE:", e, "parts:", parts)

            else:  # LiDAR 데이터 (angle, distance(mm), intensity)
                try:
                    angle, distance, intensity = map(float, parts)
                    if 0 < distance < 40000:
                        if abs(angle) > (2*np.pi + 1e-3):
                            angle = np.deg2rad(angle)
                        r = distance / 1000.0
                        px = r * np.cos(angle)
                        py = r * np.sin(angle)
                        self.current_scan.append((px, py))
                except Exception:
                    continue

    # ================== ZMQ & 시각화 ==================
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
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True, alpha=0.3)

        # --- OGM layer (imshow) ---
        # p_occ = sigmoid(logodds); intensity = 1 - p_occ (occupied=black, free=white)
        p_occ = 1.0 / (1.0 + np.exp(-self.grid_logodds))
        intensity = 1.0 - p_occ
        x0, y0 = self.grid_origin_world
        H, W = self.grid_logodds.shape
        extent = [x0, x0 + W*self.OGM_RES, y0, y0 + H*self.OGM_RES]
        self.ogm_img = self.ax.imshow(
            intensity, origin='lower', extent=extent,
            cmap='gray', vmin=0.0, vmax=1.0, alpha=0.8, zorder=1
        )

        # --- overlays ---
        # self.scan_dots, = self.ax.plot([], [], '.', color='tab:blue',
        #                                ms=3, linestyle='None', alpha=0.7, zorder=2, label='Map')
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

                # --- OGM 갱신(배경 이미지) ---
                x0, y0 = self.grid_origin_world
                H, W = self.grid_logodds.shape
                extent = [x0, x0 + W*self.OGM_RES, y0, y0 + H*self.OGM_RES]
                self.ogm_img.set_extent(extent)
                # p_occ = sigmoid(logodds); intensity = 1 - p_occ
                p_occ = 1.0 / (1.0 + np.exp(-self.grid_logodds))
                intensity = 1.0 - p_occ
                self.ogm_img.set_data(intensity)

                # --- 오버레이 ---
                # if self.scan_x:
                #     self.scan_dots.set_data(self.scan_x, self.scan_y)
                if self.path_x:
                    self.path_line.set_data(self.path_x, self.path_y)
                    self.pose_marker.set_data([self.path_x[-1]], [self.path_y[-1]])

                # 축 자동 맞춤 (스캔+경로+OGM 범위)
                if self._frame % self.VIS_EVERY == 0:
                    xs, ys = [], []
                    if self.scan_x:
                        xs += list(self.scan_x); ys += list(self.scan_y)
                    if self.path_x:
                        xs += list(self.path_x); ys += list(self.path_y)
                    xs += [extent[0], extent[1]]
                    ys += [extent[2], extent[3]]

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
