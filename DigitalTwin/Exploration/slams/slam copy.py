# import zmq
# import signal
# import threading
# import matplotlib.pyplot as plt
# import numpy as np
# from collections import deque
# import time
# import json

# # ===== 프론티어-RL 선택 토글(온/오프) =====
# FRONTIER_RL_ENABLED = False
# # ===== 프론티어-RL 선택 토글(온/오프) ===== ON : MRTSP / OFF : Experiments Frontier
# FRONTIER_WHAT_ENABLED = False

# # Graph-SLAM backend
# from slams.newslam.graph import Graph
# from slams.newslam.pose_se2 import PoseSE2

# # RL(보조 점수만 제공)
# from rl.RL_Agent_v2 import RLAgent

# class RealtimeSLAM:
#     def __init__(self, socket):
#         # === Visualization / perf ===
#         self.MAX_POINTS = 300_000
#         self.VIS_EVERY = 5
#         self.SLEEP_VIZ = 0.02

#         # === LiDAR params ===
#         self.LIDAR_MAX_RANGE_MM = 40000
#         self.NO_HIT_MARGIN_MM = 5

#         # === OGM params ===
#         self.OGM_RES = 0.10
#         self.OGM_INIT_SIZE = (600, 600)
#         self.OGM_L_FREE = -1.0
#         self.OGM_L_OCC = +2.0
#         self.OGM_CLAMP = (-5.0, 5.0)
#         self.OGM_SUBSAMPLE = 1

#         # === Buffers ===
#         self.scan_x = deque()
#         self.scan_y = deque()
#         self.path_x = deque()
#         self.path_y = deque()
#         self.current_pose = (0.0, 0.0, 0.0)
#         self.current_scan = []  # (px, py, hit)

#         # === Graph-SLAM ===
#         self.graph = Graph(edges=[], vertices=[])
#         self.node_id = 0
#         self.prev_pose = (0.0, 0.0, 0.0)
#         self.last_opt_time = time.time()
#         self.OPT_INTERVAL = 5.0

#         # === OGM & TRACE buffer ===
#         H, W = self.OGM_INIT_SIZE
#         self.grid_logodds = np.zeros((H, W), dtype=np.float32)
#         self.trace = np.zeros((H, W), dtype=np.float32)
#         self.trace_decay = 0.01                     
#         self.grid_origin_world = (-W / 2 * self.OGM_RES, -H / 2 * self.OGM_RES)

#         # === ZMQ ===
#         self.socket = socket

#         # === Viz handles ===
#         self.fig = None
#         self.ax = None
#         self.ogm_img = None
#         self.path_line = None
#         self.pose_marker = None
#         self.goal_marker = None  # 선택된 프론티어
#         self._frame = 0

#         self.running = True
#         print("RealtimeSLAM initialized, waiting for messages.")

#         self.use_what_frontier = FRONTIER_WHAT_ENABLED
#         if (self.use_what_frontier):
#             self._set_MRTSP_frontier()
#         else:
#             self._set_experiment_frontier()
        
#         # === State ===
#         self.last_frontiers = []          # list[Frontier] (선택된 1개만 담음)
#         self.last_goal_center_xy = None   # (x, y)
#         self.last_path_xy = []            # [(x,y), ...]

#         # === RL 인스턴스 ===
#         self.rl = RLAgent()

#         # === Frontier Selection Mode (강화학습 온오프프) ===
#         self.use_rl_frontier = FRONTIER_RL_ENABLED
        

#     # ================== Coord utils ==================
#     def world_to_map(self, x, y):
#         x0, y0 = self.grid_origin_world
#         ix = int(np.floor((x - x0) / self.OGM_RES))
#         iy = int(np.floor((y - y0) / self.OGM_RES))
#         return iy, ix

#     def _ensure_in_grid(self, iy, ix):
#         H, W = self.grid_logodds.shape
#         pt = pb = pl = pr = 0
#         if iy < 0: pt = -iy
#         if ix < 0: pl = -ix
#         if iy >= H: pb = iy - H + 1
#         if ix >= W: pr = ix - W + 1
#         if pt or pb or pl or pr:
#             self.grid_logodds = np.pad(
#                 self.grid_logodds, ((pt, pb), (pl, pr)),
#                 mode="constant", constant_values=0.0
#             )
#             self.trace = np.pad(  # trace도 동일 패딩
#                 self.trace, ((pt, pb), (pl, pr)),
#                 mode="constant", constant_values=0.0
#             )
#             dx = -pl * self.OGM_RES
#             dy = -pt * self.OGM_RES
#             x0, y0 = self.grid_origin_world
#             self.grid_origin_world = (x0 + dx, y0 + dy)

#     @staticmethod
#     def _bresenham(iy0, ix0, iy1, ix1):
#         cells = []
#         dy = abs(iy1 - iy0)
#         dx = abs(ix1 - ix0)
#         sy = 1 if iy0 < iy1 else -1
#         sx = 1 if ix0 < ix1 else -1
#         err = dx - dy
#         y, x = iy0, ix0
#         while not (y == iy1 and x == ix1):
#             cells.append((y, x))
#             e2 = 2 * err
#             if e2 > -dy:
#                 err -= dy
#                 x += sx
#             if e2 < dx:
#                 err += dx
#                 y += sy
#         return cells

#     def ogm_update_scan(self, pose, scan):
#         x, y, th = pose
#         c, s = np.cos(th), np.sin(th)

#         iy0, ix0 = self.world_to_map(x, y)
#         self._ensure_in_grid(iy0, ix0)

#         # trace decay (매 스캔)
#         self.trace *= (1.0 - self.trace_decay)

#         for i, item in enumerate(scan):
#             if (self.OGM_SUBSAMPLE > 1) and (i % self.OGM_SUBSAMPLE != 0):
#                 continue
#             if len(item) == 3:
#                 px, py, hit = item
#             else:
#                 px, py = item
#                 hit = True

#             gx = x + c * px - s * py
#             gy = y + s * px + c * py

#             iy1, ix1 = self.world_to_map(gx, gy)
#             self._ensure_in_grid(iy1, ix1)

#             free_cells = self._bresenham(iy0, ix0, iy1, ix1)
#             if free_cells:
#                 ys, xs = zip(*free_cells)
#                 self.grid_logodds[ys, xs] += self.OGM_L_FREE
#                 self.trace[ys, xs] = 1.0  # 지나간 free는 trace 리프레시
#             if hit:
#                 self.grid_logodds[iy1, ix1] += self.OGM_L_OCC

#         np.clip(self.grid_logodds, *self.OGM_CLAMP, out=self.grid_logodds)

#         # Frontier/Planner refresh
#         self.detector.origin_xy = self.grid_origin_world
#         self.planner.update_map(self.grid_logodds, self.grid_origin_world)

#     # ================== Graph-SLAM ==================
#     def add_pose_node(self, pose_tuple):
#         x, y, theta = pose_tuple
#         pose = PoseSE2([x, y], theta)
#         self.graph.add_vertex(self.node_id, pose)
#         if self.node_id > 0:
#             dx = x - self.prev_pose[0]
#             dy = y - self.prev_pose[1]
#             dtheta = theta - self.prev_pose[2]
#             meas = PoseSE2([dx, dy], dtheta)
#             self.graph.add_edge(
#                 [self.node_id - 1, self.node_id],
#                 measurement=meas,
#                 information=np.identity(3),
#             )
#         self.prev_pose = (x, y, theta)
#         self.node_id += 1

#     def try_loop_closure(self, pose_tuple):
#         x, y, theta = pose_tuple
#         for past_v in self.graph._vertices:
#             dx = x - past_v.pose.position[0]
#             dy = y - past_v.pose.position[1]
#             if np.hypot(dx, dy) < 1.0 and past_v.id != self.node_id - 1:
#                 meas = PoseSE2([dx, dy], theta - past_v.pose.orientation)
#                 self.graph.add_edge(
#                     [past_v.id, self.node_id - 1],
#                     measurement=meas,
#                     information=np.identity(3),
#                 )
#                 break

#     # ================== Frontier 선택 (휴리스틱 + RL 보조) ==================
#     def _select_frontier(self, candidates, robot_xy):
#         if not candidates:
#             return None

#         if not self.use_rl_frontier:
#             return max(candidates, key=lambda f: f.score)

#         try:
#             out = self.rl.score_and_select(
#                 logodds=self.grid_logodds,
#                 origin_xy=self.grid_origin_world,
#                 res_m=self.OGM_RES,
#                 robot_xy=robot_xy,
#                 frontiers=candidates,
#             )
#             preds = out.get("rl_preds", [])
#             w_rl = self.RL_WEIGHT

#             # 같은 candidates 순서에서 보너스만 더해 최종 스코어 계산
#             blended_scores = []
#             for i, f in enumerate(candidates):
#                 bonus = float(preds[i]) if i < len(preds) else 0.0
#                 blended_scores.append(f.score + w_rl * bonus)
#             best_idx = int(np.argmax(blended_scores))
#             return candidates[best_idx]

#         except Exception as e:
#             print("⚠️ RL frontier hook failed, fallback to heuristic:", e)
#             return max(candidates, key=lambda f: f.score)

#     # ================== Message handling ==================
#     def parse_and_update(self, message):
#         """
#         Message format:
#           - 'POSE,x,y,theta' lines (radians, meters)
#           - 'angle,dist_mm,intensity' lines for LiDAR
#           - 'RENEWAL' : 지속적으로 통신 의미(로봇이 frontier에 도착하기전까지 이 통신으로 보냄)
#           - 'PROCESS' : 로봇이 frontier에 도착하거나 처음에만 보냄
#         """
#         lidar_parts, pose_parts, command_parts = self._classify_message(message)

#         for lp in lidar_parts:
#             self._set_LidarUpdate(lp)

#         self._set_poseUpdate(pose_parts, command_parts)

#         if command_parts == "PROCESS":
#             done = self.planner.notify_frontier_presence(self.last_goal_center_xy is not None)
#             if done:
#                 print("✅ 종료 (프론티어 없음 또는 맵 완성 기준 충족)")
#                 return {
#                     "status": "done",
#                     "frontier_rl": self.use_rl_frontier,
#                     "goal_xy": None,
#                     "path": []
#                 }

#             payload_plan = {
#                 "status": "continue",
#                 "frontier_rl": self.use_rl_frontier,
#                 "goal_xy": self.last_goal_center_xy,
#                 "path": self.last_path_xy,
#             }
#             return payload_plan

#         elif command_parts == "RENEWAL":

#             payload_plan = {
#                 "status": "renewal",
#                 "frontier_rl": "None",
#                 "goal_xy": "None",
#                 "path": [],
#             }
#             return payload_plan

#     # ================== ZMQ loop (request → plan → reply) ==================
#     def zmq_loop(self):
#         while self.running:
#             try:
#                 msg = self.socket.recv_string(flags=zmq.NOBLOCK)
#                 result_msg = self.parse_and_update(msg)

#                 payload = {
#                     "ok": True,
#                     "result": result_msg
#                 }
#                 self.socket.send_string(json.dumps(payload))

#             except zmq.error.Again:
#                 time.sleep(0.005)
#             except Exception as e:
#                 print("❌ ZMQ Error:", e)
#                 break
#                 time.sleep(0.05)

#     # ================== Visualization ==================
#     def setup_plot(self):
#         plt.ion()
#         self.fig, self.ax = plt.subplots(figsize=(7, 5))
#         self.ax.set_aspect("equal", "box")
#         self.ax.grid(True, alpha=0.3)

#         p_occ = 1.0 / (1.0 + np.exp(-self.grid_logodds))
#         intensity = 1.0 - p_occ
#         x0, y0 = self.grid_origin_world
#         H, W = self.grid_logodds.shape
#         extent = [x0, x0 + W * self.OGM_RES, y0, y0 + H * self.OGM_RES]
#         self.ogm_img = self.ax.imshow(
#             intensity,
#             origin="lower",
#             extent=extent,
#             cmap="gray",
#             vmin=0.0,
#             vmax=1.0,
#             alpha=0.8,
#             zorder=1,
#         )

#         (self.path_line,) = self.ax.plot([], [], lw=1.5, alpha=0.9, zorder=3, label="Path")
#         (self.pose_marker,) = self.ax.plot([], [], "o", ms=5, alpha=0.9, zorder=4, label="Robot")

#         (self.goal_marker,) = self.ax.plot([], [], "o", ms=8, zorder=6, label="Selected Frontier", color="#32CD32")
#         self.ax.legend(loc="upper right")

#     def viz_loop(self):
#         if self.fig is None:
#             self.setup_plot()

#         while self.running:
#             try:
#                 self._frame += 1

#                 x0, y0 = self.grid_origin_world
#                 H, W = self.grid_logodds.shape
#                 extent = [x0, x0 + W * self.OGM_RES, y0, y0 + H * self.OGM_RES]
#                 self.ogm_img.set_extent(extent)
#                 p_occ = 1.0 / (1.0 + np.exp(-self.grid_logodds))
#                 self.ogm_img.set_data(1.0 - p_occ)

#                 if self.path_x:
#                     self.path_line.set_data(self.path_x, self.path_y)
#                     self.pose_marker.set_data([self.path_x[-1]], [self.path_y[-1]])

#                 if self.last_goal_center_xy is not None:
#                     self.goal_marker.set_data(
#                         [self.last_goal_center_xy[0]],
#                         [self.last_goal_center_xy[1]],
#                     )
#                 else:
#                     self.goal_marker.set_data([], [])

#                 if self._frame % self.VIS_EVERY == 0:
#                     xs, ys = [], []
#                     if self.path_x:
#                         xs += list(self.path_x)
#                         ys += list(self.path_y)
#                     xs += [extent[0], extent[1]]
#                     ys += [extent[2], extent[3]]
#                     xs = np.asarray(xs)
#                     ys = np.asarray(ys)
#                     xmin, xmax = np.nanmin(xs), np.nanmax(xs)
#                     ymin, ymax = np.nanmin(ys), np.nanmax(ys)
#                     pad = max(1.5, 0.05 * max(xmax - xmin, ymax - ymin))
#                     self.ax.set_xlim(xmin - pad, xmax + pad)
#                     self.ax.set_ylim(ymin - pad, ymax + pad)

#                 self.fig.canvas.draw()
#                 self.fig.canvas.flush_events()
#                 time.sleep(self.SLEEP_VIZ)
#             except Exception as e:
#                 print("❌ Viz Error:", e)
#                 time.sleep(0.05)

#     def run(self):
#         def handler(sig, frame):
#             print("\n🛑 Shutting down slam.")
#             self.running = False

#         signal.signal(signal.SIGINT, handler)

#         t_zmq = threading.Thread(target=self.zmq_loop, daemon=True)
#         t_zmq.start()

#         self.setup_plot()
#         print("Starting visualization loop in main thread.")
#         self.viz_loop()

#         # Teardown
#         if t_zmq.is_alive():
#             self.stop = True
#             t_zmq.join(timeout=1.0)
#         try:
#             plt.close("all")
#         except Exception:
#             pass

#     # ================== create frontier ================
#     def _set_poseUpdate(self, parts, command_parts):
#         x, y, theta = map(float, parts[1:4])
#         self.current_pose = (x, y, theta)

#         # 1) 누적된 스캔을 OGM에 반영
#         if self.current_scan:
#             self.ogm_update_scan((x, y, theta), self.current_scan)
#             self.current_scan = []

#         # 2) 그래프/루프클로저/최적화 갱신
#         self.add_pose_node((x, y, theta))
#         self.try_loop_closure((x, y, theta))

#         if time.time() - self.last_opt_time > self.OPT_INTERVAL and len(self.graph._vertices) >= 2:
#             self.graph.optimize()
#             self.last_opt_time = time.time()

#         poses = [v.pose for v in self.graph._vertices]
#         self.path_x = deque([p.position[0] for p in poses])
#         self.path_y = deque([p.position[1] for p in poses])

#         if (self.use_what_frontier and command_parts =="PROCESS"):
#             self._make_MRTSP_frontier(x, y)
#         elif(self.use_what_frontier == False and command_parts =="PROCESS"):
#             self._make_experiment_frontier(x, y)

#     def _set_LidarUpdate(self, parts):
#         angle, distance, intensity = map(float, parts)
#         if abs(angle) > (2 * np.pi + 1e-3):
#             angle = np.deg2rad(angle)
#         no_hit = (distance <= 0) or (
#             distance >= self.LIDAR_MAX_RANGE_MM - self.NO_HIT_MARGIN_MM
#         ) or (intensity <= 0.5)

#         r = (self.LIDAR_MAX_RANGE_MM if no_hit else distance) / 1000.0
#         px = r * np.cos(angle)
#         py = r * np.sin(angle)
#         self.current_scan.append((px, py, not no_hit))

#     # ======================================================================================
#     # MRTSP Set
#     def _set_MRTSP_frontier(self):
#         from frontier.mrtsp_selector import FrontierMRTSPSelector
#         self.mrtsp_frontier = FrontierMRTSPSelector(
#             ogm_res_m=self.OGM_RES,
#             grid_origin_world_xy=self.grid_origin_world,
#             occ_thresh=0.65,
#             free_thresh=0.35,
#             min_cluster_size=12,
#             dilate_free=1,
#             # === 논문 방식 ===
#             use_map_optimization=True,     # 양방향 필터 + 확장 (지도 최적화) 사용
#             bilateral_sigma_s_px=2,
#             bilateral_sigma_r_val=30.0,
#             expansion_iters=1,
#             # 비용식 파라미터 (식 (5)–(8))
#             sensor_range_m=30.0,           # r_s : 실효 센서 사거리(예: 30m). 필요시 35~40으로 조정
#             Wd=1.0,                        # 거리 항 가중치
#             Ws=1.0,                        # 정보이득(프론티어 크기) 가중치
#             Vmax=0.8,                      # 최대 선속도(식 (8)에서 t_lb)
#             Wmax=1.2,                      # 최대 각속도(식 (8)에서 t_lb)
#             # 나머지 보조 옵션(접근성/안전여유 등)은 기본값 유지
#             require_reachable=True,
#             min_clearance_m=0.35,
#             ignore_border_unknown_margin_m=0.8,
#             min_free_before_unknown_m=0.6,
#         )
    
#     def _make_MRTSP_frontier(self, robot_x, robot_y, robot_theta):
#         all_frontiers = self.frontier.extract(
#             self.grid_logodds,
#             robot_xy=(robot_x, robot_y),
#             robot_yaw=robot_theta,            
#             exploration_trace=self.trace,
#             top_k=None
#         )
        
#     # Experiment Set
#     def _set_experiment_frontier(self):
#         from frontier.frontier import FrontierDetector
#         from frontier.experiment_selector import FrontierExSelector, ScoredFrontier
#         from frontier.global_planner import GlobalPlanner

#          # 1) 탐지기 (frontier.py)
#         self.detector = FrontierDetector(
#             ogm_res_m=self.OGM_RES,
#             grid_origin_world_xy=self.grid_origin_world,
#             occ_thresh=0.65,
#             free_thresh=0.35,
#             min_cluster_size=12,
#             dilate_free=1,
#             min_clearance_m=0.35,
#             require_reachable=True,
#             ignore_border_unknown_margin_m=0.8,
#         )

#         # 2) 선택기/점수화기 (experiment_selector.py)
#         self.selector = FrontierExSelector(
#             ogm_res_m=self.OGM_RES,
#             grid_origin_world_xy=self.grid_origin_world,
#             info_radius_m=1.0,
#             visible_rays=64,
#             ray_step_px=1,
#             min_free_before_unknown_m=0.6,
#             merge_min_sep_m=1.5,
#             w_info=0.7, w_size=0.1, w_dist=0.05,
#             w_open=1.0, w_trace=0.7,
#         )

#         # 3) 전역 플래너
#         self.planner = GlobalPlanner(
#             ogm_res_m=self.OGM_RES,
#             occ_thresh=0.65,
#             free_thresh=0.35,
#             coverage_done_thresh=0.95,
#             unknown_left_thresh=0.02,
#             no_frontier_patience=10,
#         )

#     def _make_experiment_frontier(self, robot_x, robot_y):
#         # 1) 탐지
#         det_out = self.detector.detect(
#             self.grid_logodds,
#             robot_xy=(robot_x, robot_y)
#         )
#         masks = det_out["masks"]
#         candidates = det_out["candidates"]

#         # 2) 점수화/정렬/병합
#         all_frontiers = self.selector.score_and_select(
#             candidates=candidates,
#             masks=masks,
#             robot_xy=(robot_x, robot_y),
#             exploration_trace=self.trace,
#             do_merge=True,
#             top_k=None
#         )

#         chosen = self._select_frontier(all_frontiers, (robot_x, robot_y))
#         self.last_frontiers = [chosen] if chosen else []
#         self.last_goal_center_xy = chosen.center_xy if chosen else None

#         self._make_experiment_A_start_path(robot_x, robot_y)

#     def _make_experiment_A_start_path(self, x, y):
#         path_xy = []
#         if self.last_goal_center_xy is not None:
#             path_xy = self.planner.plan_path(
#                 start_xy=(x, y),
#                 goal_xy=tuple(self.last_goal_center_xy),
#                 safety_inflate_m=0.75,
#                 allow_diagonal=True,
#             )
#             print(f"🧭 A* start={x:.2f},{y:.2f}  goal={self.last_goal_center_xy}  path_len={len(path_xy)}")
#             if not path_xy:
#                 print("⚠️ A* returned empty path. Goal may be inside inflated obstacles / unreachable.")
#             path_xy = [[float(px), float(py)] for (px, py) in path_xy]

#         self.last_path_xy = path_xy

#     #============================= message classify ==============================
#     def _classify_message(self, message):
#         lidar_parts = []  
#         pose_parts  = None 
#         command     = None

#         for line in message.strip().split("\n"):
#             ln = line.strip()
#             if not ln:
#                 continue

#             if ln == "PROCESS" or ln == "RENEWAL":
#                 command = ln
#                 continue

#             parts = ln.split(",")

#             if parts[0] == "POSE":
#                 pose_parts = parts
#                 continue

#             if len(parts) >= 3:
#                 lidar_parts.append(parts)
#                 continue

#         return lidar_parts, pose_parts, command

import zmq
import signal
import threading
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
import json

# ===== 프론티어-RL 선택 토글(온/오프) =====
FRONTIER_RL_ENABLED = False
# ===== 프론티어-RL 선택 토글(온/오프) ===== ON : MRTSP / OFF : Experiments Frontier
FRONTIER_WHAT_ENABLED = False

# Graph-SLAM backend
from slams.newslam.graph import Graph
from slams.newslam.pose_se2 import PoseSE2

# RL(보조 점수만 제공)
from rl.RL_Agent_v2 import RLAgent


class RealtimeSLAM:
    def __init__(self, socket):
        # === Visualization / perf ===
        self.MAX_POINTS = 300_000
        self.VIS_EVERY = 5
        self.SLEEP_VIZ = 0.02

        # === LiDAR params ===
        self.LIDAR_MAX_RANGE_MM = 40000
        self.NO_HIT_MARGIN_MM = 5

        # === OGM params ===
        self.OGM_RES = 0.10
        self.OGM_INIT_SIZE = (600, 600)
        self.OGM_L_FREE = -1.0
        self.OGM_L_OCC = +2.0
        self.OGM_CLAMP = (-5.0, 5.0)
        self.OGM_SUBSAMPLE = 1

        # === Buffers ===
        self.scan_x = deque()
        self.scan_y = deque()
        self.path_x = deque()
        self.path_y = deque()
        self.current_pose = (0.0, 0.0, 0.0)
        self.current_scan = []  # (px, py, hit)

        # === Graph-SLAM ===
        self.graph = Graph(edges=[], vertices=[])
        self.node_id = 0
        self.prev_pose = (0.0, 0.0, 0.0)
        self.last_opt_time = time.time()
        self.OPT_INTERVAL = 5.0

        # === OGM & TRACE buffer ===
        H, W = self.OGM_INIT_SIZE
        self.grid_logodds = np.zeros((H, W), dtype=np.float32)
        self.trace = np.zeros((H, W), dtype=np.float32)
        self.trace_decay = 0.01
        self.grid_origin_world = (-W / 2 * self.OGM_RES, -H / 2 * self.OGM_RES)

        # === ZMQ ===
        self.socket = socket

        # === Viz handles ===
        self.fig = None
        self.ax = None
        self.ogm_img = None
        self.path_line = None
        self.pose_marker = None
        self.goal_marker = None  # 선택된 프론티어
        self._frame = 0

        # === Frontier overlays (탐지/선택 시각화) ===
        self.det_frontier_img = None   # 탐지(노랑) imshow 레이어
        self.sel_frontier_img = None   # 점수/병합(빨강) imshow 레이어
        self.det_overlay = None        # np.uint8 (H,W,4) RGBA
        self.sel_overlay = None        # np.uint8 (H,W,4) RGBA

        self.running = True
        print("RealtimeSLAM initialized, waiting for messages.")

        self.use_what_frontier = FRONTIER_WHAT_ENABLED
        if (self.use_what_frontier):
            self._set_MRTSP_frontier()
        else:
            self._set_experiment_frontier()

        # === State ===
        self.last_frontiers = []          # list[Frontier] (선택된 1개만 담음)
        self.last_goal_center_xy = None   # (x, y)
        self.last_path_xy = []            # [(x,y), ...]

        # === RL 인스턴스 ===
        self.rl = RLAgent()

        # === Frontier Selection Mode (강화학습 온오프프) ===
        self.use_rl_frontier = FRONTIER_RL_ENABLED

    # ================== Coord utils ==================
    def world_to_map(self, x, y):
        x0, y0 = self.grid_origin_world
        ix = int(np.floor((x - x0) / self.OGM_RES))
        iy = int(np.floor((y - y0) / self.OGM_RES))
        return iy, ix

    def _ensure_in_grid(self, iy, ix):
        H, W = self.grid_logodds.shape
        pt = pb = pl = pr = 0
        if iy < 0: pt = -iy
        if ix < 0: pl = -ix
        if iy >= H: pb = iy - H + 1
        if ix >= W: pr = ix - W + 1
        if pt or pb or pl or pr:
            self.grid_logodds = np.pad(
                self.grid_logodds, ((pt, pb), (pl, pr)),
                mode="constant", constant_values=0.0
            )
            self.trace = np.pad(  # trace도 동일 패딩
                self.trace, ((pt, pb), (pl, pr)),
                mode="constant", constant_values=0.0
            )
            dx = -pl * self.OGM_RES
            dy = -pt * self.OGM_RES
            x0, y0 = self.grid_origin_world
            self.grid_origin_world = (x0 + dx, y0 + dy)

    @staticmethod
    def _bresenham(iy0, ix0, iy1, ix1):
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
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)

        iy0, ix0 = self.world_to_map(x, y)
        self._ensure_in_grid(iy0, ix0)

        # trace decay (매 스캔)
        self.trace *= (1.0 - self.trace_decay)

        for i, item in enumerate(scan):
            if (self.OGM_SUBSAMPLE > 1) and (i % self.OGM_SUBSAMPLE != 0):
                continue
            if len(item) == 3:
                px, py, hit = item
            else:
                px, py = item
                hit = True

            gx = x + c * px - s * py
            gy = y + s * px + c * py

            iy1, ix1 = self.world_to_map(gx, gy)
            self._ensure_in_grid(iy1, ix1)

            free_cells = self._bresenham(iy0, ix0, iy1, ix1)
            if free_cells:
                ys, xs = zip(*free_cells)
                self.grid_logodds[ys, xs] += self.OGM_L_FREE
                self.trace[ys, xs] = 1.0  # 지나간 free는 trace 리프레시
            if hit:
                self.grid_logodds[iy1, ix1] += self.OGM_L_OCC

        np.clip(self.grid_logodds, *self.OGM_CLAMP, out=self.grid_logodds)

        # Frontier/Planner refresh
        # (실험 구조에서 detector 사용)
        if hasattr(self, "detector"):
            self.detector.origin_xy = self.grid_origin_world
        self.planner.update_map(self.grid_logodds, self.grid_origin_world)

    # ================== Graph-SLAM ==================
    def add_pose_node(self, pose_tuple):
        x, y, theta = pose_tuple
        pose = PoseSE2([x, y], theta)
        self.graph.add_vertex(self.node_id, pose)
        if self.node_id > 0:
            dx = x - self.prev_pose[0]
            dy = y - self.prev_pose[1]
            dtheta = theta - self.prev_pose[2]
            meas = PoseSE2([dx, dy], dtheta)
            self.graph.add_edge(
                [self.node_id - 1, self.node_id],
                measurement=meas,
                information=np.identity(3),
            )
        self.prev_pose = (x, y, theta)
        self.node_id += 1

    def try_loop_closure(self, pose_tuple):
        x, y, theta = pose_tuple
        for past_v in self.graph._vertices:
            dx = x - past_v.pose.position[0]
            dy = y - past_v.pose.position[1]
            if np.hypot(dx, dy) < 1.0 and past_v.id != self.node_id - 1:
                meas = PoseSE2([dx, dy], theta - past_v.pose.orientation)
                self.graph.add_edge(
                    [past_v.id, self.node_id - 1],
                    measurement=meas,
                    information=np.identity(3),
                )
                break

    # ================== Frontier 선택 (휴리스틱 + RL 보조) ==================
    def _select_frontier(self, candidates, robot_xy):
        if not candidates:
            return None

        if not self.use_rl_frontier:
            return max(candidates, key=lambda f: f.score)

        try:
            out = self.rl.score_and_select(
                logodds=self.grid_logodds,
                origin_xy=self.grid_origin_world,
                res_m=self.OGM_RES,
                robot_xy=robot_xy,
                frontiers=candidates,
            )
            preds = out.get("rl_preds", [])
            w_rl = self.RL_WEIGHT

            # 같은 candidates 순서에서 보너스만 더해 최종 스코어 계산
            blended_scores = []
            for i, f in enumerate(candidates):
                bonus = float(preds[i]) if i < len(preds) else 0.0
                blended_scores.append(f.score + w_rl * bonus)
            best_idx = int(np.argmax(blended_scores))
            return candidates[best_idx]

        except Exception as e:
            print("⚠️ RL frontier hook failed, fallback to heuristic:", e)
            return max(candidates, key=lambda f: f.score)

    # ================== Message handling ==================
    def parse_and_update(self, message):
        """
        Message format:
          - 'POSE,x,y,theta' lines (radians, meters)
          - 'angle,dist_mm,intensity' lines for LiDAR
          - 'RENEWAL' : 지속적으로 통신 의미(로봇이 frontier에 도착하기전까지 이 통신으로 보냄)
          - 'PROCESS' : 로봇이 frontier에 도착하거나 처음에만 보냄
        """
        lidar_parts, pose_parts, command_parts = self._classify_message(message)

        for lp in lidar_parts:
            self._set_LidarUpdate(lp)

        self._set_poseUpdate(pose_parts, command_parts)

        if command_parts == "PROCESS":
            done = self.planner.notify_frontier_presence(self.last_goal_center_xy is not None)
            if done:
                print("✅ 종료 (프론티어 없음 또는 맵 완성 기준 충족)")
                return {
                    "status": "done",
                    "frontier_rl": self.use_rl_frontier,
                    "goal_xy": None,
                    "path": []
                }

            payload_plan = {
                "status": "continue",
                "frontier_rl": self.use_rl_frontier,
                "goal_xy": self.last_goal_center_xy,
                "path": self.last_path_xy,
            }
            return payload_plan

        elif command_parts == "RENEWAL":

            payload_plan = {
                "status": "renewal",
                "frontier_rl": "None",
                "goal_xy": "None",
                "path": [],
            }
            return payload_plan

    # ================== ZMQ loop (request → plan → reply) ==================
    def zmq_loop(self):
        while self.running:
            try:
                msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                result_msg = self.parse_and_update(msg)

                payload = {
                    "ok": True,
                    "result": result_msg
                }
                self.socket.send_string(json.dumps(payload))

            except zmq.error.Again:
                time.sleep(0.005)
            except Exception as e:
                print("❌ ZMQ Error:", e)
                break
                time.sleep(0.05)

    # ================== Visualization ==================
    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_aspect("equal", "box")
        self.ax.grid(True, alpha=0.3)

        p_occ = 1.0 / (1.0 + np.exp(-self.grid_logodds))
        intensity = 1.0 - p_occ
        x0, y0 = self.grid_origin_world
        H, W = self.grid_logodds.shape
        extent = [x0, x0 + W * self.OGM_RES, y0, y0 + H * self.OGM_RES]
        self.ogm_img = self.ax.imshow(
            intensity,
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            alpha=0.8,
            zorder=1,
        )

        # === 빈 오버레이 두 장 생성 (탐지=노랑, 선택=빨강) ===
        empty_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        self.det_frontier_img = self.ax.imshow(
            empty_rgba, origin="lower", extent=extent, zorder=5
        )
        self.sel_frontier_img = self.ax.imshow(
            empty_rgba, origin="lower", extent=extent, zorder=6
        )

        (self.path_line,) = self.ax.plot([], [], lw=1.5, alpha=0.9, zorder=3, label="Path")
        (self.pose_marker,) = self.ax.plot([], [], "o", ms=5, alpha=0.9, zorder=4, label="Robot")
        (self.goal_marker,) = self.ax.plot([], [], "o", ms=8, zorder=7, label="Selected Frontier", color="#32CD32")
        self.ax.legend(loc="upper right")

    def viz_loop(self):
        if self.fig is None:
            self.setup_plot()

        while self.running:
            try:
                self._frame += 1

                x0, y0 = self.grid_origin_world
                H, W = self.grid_logodds.shape
                extent = [x0, x0 + W * self.OGM_RES, y0, y0 + H * self.OGM_RES]
                self.ogm_img.set_extent(extent)
                p_occ = 1.0 / (1.0 + np.exp(-self.grid_logodds))
                self.ogm_img.set_data(1.0 - p_occ)

                # === 오버레이 갱신 ===
                if self.sel_overlay is not None:
                    self.sel_frontier_img.set_extent(extent)
                    self.sel_frontier_img.set_data(self.sel_overlay)

                if self.path_x:
                    self.path_line.set_data(self.path_x, self.path_y)
                    self.pose_marker.set_data([self.path_x[-1]], [self.path_y[-1]])

                if self.last_goal_center_xy is not None:
                    self.goal_marker.set_data(
                        [self.last_goal_center_xy[0]],
                        [self.last_goal_center_xy[1]],
                    )
                else:
                    self.goal_marker.set_data([], [])

                if self._frame % self.VIS_EVERY == 0:
                    xs, ys = [], []
                    if self.path_x:
                        xs += list(self.path_x)
                        ys += list(self.path_y)
                    xs += [extent[0], extent[1]]
                    ys += [extent[2], extent[3]]
                    xs = np.asarray(xs)
                    ys = np.asarray(ys)
                    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
                    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
                    pad = max(1.5, 0.05 * max(xmax - xmin, ymax - ymin))
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
            print("\n🛑 Shutting down slam.")
            self.running = False

        signal.signal(signal.SIGINT, handler)

        t_zmq = threading.Thread(target=self.zmq_loop, daemon=True)
        t_zmq.start()

        self.setup_plot()
        print("Starting visualization loop in main thread.")
        self.viz_loop()

        # Teardown
        if t_zmq.is_alive():
            self.stop = True
            t_zmq.join(timeout=1.0)
        try:
            plt.close("all")
        except Exception:
            pass

    # ================== create frontier ================
    def _set_poseUpdate(self, parts, command_parts):
        x, y, theta = map(float, parts[1:4])
        self.current_pose = (x, y, theta)

        # 1) 누적된 스캔을 OGM에 반영
        if self.current_scan:
            self.ogm_update_scan((x, y, theta), self.current_scan)
            self.current_scan = []

        # 2) 그래프/루프클로저/최적화 갱신
        self.add_pose_node((x, y, theta))
        self.try_loop_closure((x, y, theta))

        if time.time() - self.last_opt_time > self.OPT_INTERVAL and len(self.graph._vertices) >= 2:
            self.graph.optimize()
            self.last_opt_time = time.time()

        poses = [v.pose for v in self.graph._vertices]
        self.path_x = deque([p.position[0] for p in poses])
        self.path_y = deque([p.position[1] for p in poses])

        if (self.use_what_frontier and command_parts == "PROCESS"):
            self._make_MRTSP_frontier(x, y)
        elif (self.use_what_frontier == False and command_parts == "PROCESS"):
            self._make_experiment_frontier(x, y)

    def _set_LidarUpdate(self, parts):
        angle, distance, intensity = map(float, parts)
        if abs(angle) > (2 * np.pi + 1e-3):
            angle = np.deg2rad(angle)
        no_hit = (distance <= 0) or (
            distance >= self.LIDAR_MAX_RANGE_MM - self.NO_HIT_MARGIN_MM
        ) or (intensity <= 0.5)

        r = (self.LIDAR_MAX_RANGE_MM if no_hit else distance) / 1000.0
        px = r * np.cos(angle)
        py = r * np.sin(angle)
        self.current_scan.append((px, py, not no_hit))

    # ======================================================================================
    # MRTSP Set
    def _set_MRTSP_frontier(self):
        from frontier.mrtsp_selector import FrontierMRTSPSelector
        self.mrtsp_frontier = FrontierMRTSPSelector(
            ogm_res_m=self.OGM_RES,
            grid_origin_world_xy=self.grid_origin_world,
            occ_thresh=0.65,
            free_thresh=0.35,
            min_cluster_size=12,
            dilate_free=1,
            # === 논문 방식 ===
            use_map_optimization=True,     # 양방향 필터 + 확장 (지도 최적화) 사용
            bilateral_sigma_s_px=2,
            bilateral_sigma_r_val=30.0,
            expansion_iters=1,
            # 비용식 파라미터 (식 (5)–(8))
            sensor_range_m=30.0,           # r_s : 실효 센서 사거리(예: 30m). 필요시 35~40으로 조정
            Wd=1.0,                        # 거리 항 가중치
            Ws=1.0,                        # 정보이득(프론티어 크기) 가중치
            Vmax=0.8,                      # 최대 선속도(식 (8)에서 t_lb)
            Wmax=1.2,                      # 최대 각속도(식 (8)에서 t_lb)
            # 나머지 보조 옵션(접근성/안전여유 등)은 기본값 유지
            require_reachable=True,
            min_clearance_m=0.35,
            ignore_border_unknown_margin_m=0.8,
            min_free_before_unknown_m=0.6,
        )

    def _make_MRTSP_frontier(self, robot_x, robot_y, robot_theta):
        # NOTE: MRTSP 경로는 현재 사용하지 않는 구성이라 그대로 둡니다.
        all_frontiers = self.frontier.extract(
            self.grid_logodds,
            robot_xy=(robot_x, robot_y),
            robot_yaw=robot_theta,
            exploration_trace=self.trace,
            top_k=None
        )

    # Experiment Set
    def _set_experiment_frontier(self):
        from frontier.frontier import FrontierDetector
        from frontier.experiment_selector import FrontierExSelector, ScoredFrontier
        from frontier.global_planner import GlobalPlanner

        # 1) 탐지기 (frontier.py)
        self.detector = FrontierDetector(
            ogm_res_m=self.OGM_RES,
            grid_origin_world_xy=self.grid_origin_world,
            occ_thresh=0.65,
            free_thresh=0.35,
            min_cluster_size=12,
            dilate_free=1,
            min_clearance_m=0.35,
            require_reachable=True,
            ignore_border_unknown_margin_m=0.8,
        )

        # 2) 선택기/점수화기 (experiment_selector.py)
        self.selector = FrontierExSelector(
            ogm_res_m=self.OGM_RES,
            grid_origin_world_xy=self.grid_origin_world,
            info_radius_m=1.0,
            visible_rays=64,
            ray_step_px=1,
            min_free_before_unknown_m=0.6,
            merge_min_sep_m=1.5,
            w_info=0.7, w_size=0.1, w_dist=0.05,
            w_open=1.0, w_trace=0.7,
        )

        # 3) 전역 플래너
        self.planner = GlobalPlanner(
            ogm_res_m=self.OGM_RES,
            occ_thresh=0.65,
            free_thresh=0.35,
            coverage_done_thresh=0.95,
            unknown_left_thresh=0.02,
            no_frontier_patience=10,
        )

    def _make_experiment_frontier(self, robot_x, robot_y):
        # 1) 탐지
        det_out = self.detector.detect(
            self.grid_logodds,
            robot_xy=(robot_x, robot_y)
        )
        masks = det_out["masks"]
        candidates = det_out["candidates"]

        # === 탐지 오버레이(노랑) 생성 ===
        H, W = masks["free"].shape

        # 2) 점수화/정렬/병합
        all_frontiers = self.selector.score_and_select(
            candidates=candidates,
            masks=masks,
            robot_xy=(robot_x, robot_y),
            exploration_trace=self.trace,
            do_merge=True,
            top_k=None
        )

        # === 점수/병합 오버레이(빨강) 생성 ===
        sel_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        for f in all_frontiers:
            # 선택기 결과가 Frontier/ScoredFrontier 등 무엇이든
            # 최종적으로 (iy, ix) 리스트만 확보하면 됨
            coords = getattr(f, "pixel_inds", None)
            if coords is None and hasattr(f, "frontier"):
                coords = getattr(f.frontier, "pixel_inds", None)
            if not coords:
                continue
            ys, xs = zip(*coords)
            sel_rgba[ys, xs, 0] = 255  # R
            sel_rgba[ys, xs, 1] = 0    # G
            sel_rgba[ys, xs, 2] = 0    # B
            sel_rgba[ys, xs, 3] = 220  # A (더 진하게)
        self.sel_overlay = sel_rgba


        # === 기존 선택 로직 유지 ===
        chosen = self._select_frontier(all_frontiers, (robot_x, robot_y))
        self.last_frontiers = [chosen] if chosen else []
        self.last_goal_center_xy = chosen.center_xy if chosen else None

        self._make_experiment_A_start_path(robot_x, robot_y)

    def _make_experiment_A_start_path(self, x, y):
        path_xy = []
        if self.last_goal_center_xy is not None:
            path_xy = self.planner.plan_path(
                start_xy=(x, y),
                goal_xy=tuple(self.last_goal_center_xy),
                safety_inflate_m=0.75,
                allow_diagonal=True,
            )
            print(f"🧭 A* start={x:.2f},{y:.2f}  goal={self.last_goal_center_xy}  path_len={len(path_xy)}")
            if not path_xy:
                print("⚠️ A* returned empty path. Goal may be inside inflated obstacles / unreachable.")
            path_xy = [[float(px), float(py)] for (px, py) in path_xy]

        self.last_path_xy = path_xy

    #============================= message classify ==============================
    def _classify_message(self, message):
        lidar_parts = []
        pose_parts = None
        command = None

        for line in message.strip().split("\n"):
            ln = line.strip()
            if not ln:
                continue

            if ln == "PROCESS" or ln == "RENEWAL":
                command = ln
                continue

            parts = ln.split(",")

            if parts[0] == "POSE":
                pose_parts = parts
                continue

            if len(parts) >= 3:
                lidar_parts.append(parts)
                continue

        return lidar_parts, pose_parts, command
