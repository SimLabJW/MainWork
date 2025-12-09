import zmq
import signal
import threading
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
# import json

import cv2
import os, io, gzip, base64, json
from pathlib import Path
from typing import Union
from datetime import datetime

# ===== 프론티어-RL 선택 토글(온/오프) =====
FRONTIER_RL_ENABLED = True
# ===== 프론티어-RL 선택 토글(온/오프) ===== ON : MRTSP / OFF : Experiments Frontier
FRONTIER_WHAT_ENABLED = False

# Graph-SLAM backend
from slams.newslam.graph import Graph
from slams.newslam.pose_se2 import PoseSE2

# RL(보조 점수만 제공)
from rl.Frontier_Agent import FrontierRLAgent
from rl.rl_data_logger import RLDataConfig, RLDataModule

from frontier.global_planner import GlobalPlanner

RETURN_ORIGIN_POSITION = False

class RealtimeSLAM:
    def __init__(self, socket):
        # === Visualization / perf ===
        self.MAX_POINTS = 300_000
        self.VIS_EVERY = 5
        self.SLEEP_VIZ = 0.02

        # === LiDAR params ===
        # self.LIDAR_MAX_RANGE_MM = 40000 # unity
        self.LIDAR_MAX_RANGE_MM = 10000 # real
        self.NO_HIT_MARGIN_MM = 12

        # === OGM params ===
        # self.OGM_RES = 0.1 # unity
        # self.OGM_INIT_SIZE = (600, 600)# unity
        self.OGM_RES = 0.025 # real
        self.OGM_INIT_SIZE = (200, 200)# real
        self.OGM_L_FREE = -1.2 #-1.0
        self.OGM_L_OCC = +1.8  #2.0
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
        self.goal_frontier_pts = None
        self.frontier_pts = None # 프론티어(전체 리스트) 점 표시
        self._frame = 0

        self.goal_frontier_xs = []
        self.goal_frontier_ys = []

        self.frontier_xs = []
        self.frontier_ys = []

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
        model_path = "./rl/models/best_model.zip"
        
        self.rl = FrontierRLAgent(model_path=model_path)
        
        if self.rl.is_ready():
            print("✅ RL Agent initialized successfully")
            print(f"   Model loaded from: {model_path}")
        else:
            print("⚠️ RL Agent not ready - will use heuristic only")
            print(f"   Model not found at: {model_path}")
            print("   Please run train_frontier_v3.py first to create the model")

        self._rl_cfg = RLDataConfig(top_k=8, feat_dim=12, log_path="logs/frontier_dqn.jsonl")
        self._rl_mod = RLDataModule(self._rl_cfg)

        # 이전 PROCESS 스텝의 전이 보상 계산을 위한 pending 버퍼
        self._rl_prev = {
            "unknown_ratio": None,   # 이전 PROCESS에서의 미지영역 비율
            "obs": None,             # 이전 PROCESS 관측 [K,D]
            "action": None,          # 이전 PROCESS 행동(Top-K 내 인덱스)
            "path_len_m": None,      # 이전 선택 경로 길이(m)
            "success": None,         # 이전 선택 시 경로 유효여부(True/False)
            "Exploration_Time":None,
        }

        # === Frontier Selection Mode (강화학습 온오프프) ===
        self.use_rl_frontier = FRONTIER_RL_ENABLED

        self.planner = GlobalPlanner(
            ogm_res_m=self.OGM_RES,
            occ_thresh=0.65,
            free_thresh=0.35,
            coverage_done_thresh=0.90,
            unknown_left_thresh=0.02,
            no_frontier_patience=10,
        )

        # == slam returning ==
        self.return_to_origin = RETURN_ORIGIN_POSITION

        # == exploration score ==
        self.frontier_number = [] # 각 프론티어 탐색 할시의 프론티어 개수
        self.exploration_number = 0 # 전체 탐색 횟수 = 프론티어 탐색 전체 횟수

        self.frontier_exploration_time = [] # 각 프론티어 탐색 시간
        self.path_length = [] # 각 프론티어와 로봇 사이의 거리
        self.exploration_success = 0.0 # 탐색 성공률(free와 unkown 이 붙어있는 비율)

        self.frontier_s = 0.0
        self.frontier_e = 0.0

        self.exploration_s = 0.0
        self.exploration_e = 0.0



        self.runtime_check = 0

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
        print(f"in ogm th : {th} ")
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

            if free_cells and (not hit):
                trimmed = []
                for (yy, xx) in free_cells:
                    if 0 <= yy < self.grid_logodds.shape[0] and 0 <= xx < self.grid_logodds.shape[1]:
                        # 점유(양의 로그우도)를 만나면 그 앞까지만 free
                        if self.grid_logodds[yy, xx] > 0.0:
                            break
                        trimmed.append((yy, xx))
                    else:
                        break
                free_cells = trimmed

            # --- free 적용: 히트 직전 1칸은 남겨두기(가장자리 보호) ---
            if free_cells:
                # 끝 1칸 제외
                upto = max(0, len(free_cells) - 1)
                core = free_cells[:upto]
                if core:
                    ys, xs = zip(*core)
                    self.grid_logodds[ys, xs] += self.OGM_L_FREE
                    self.trace[ys, xs] = 1.0

            # --- hit 적용: 3x3 스탬핑으로 두께 확보 ---
            if hit:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = iy1 + dy
                        xx = ix1 + dx
                        if 0 <= yy < self.grid_logodds.shape[0] and 0 <= xx < self.grid_logodds.shape[1]:
                            self.grid_logodds[yy, xx] += self.OGM_L_OCC

        np.clip(self.grid_logodds, *self.OGM_CLAMP, out=self.grid_logodds)

        # Frontier/Planner refresh
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

        # RL 모델이 준비되지 않았으면 휴리스틱으로 폴백
        if not self.rl.is_ready():
            print("⚠️ RL model not ready, using heuristic")
            return max(candidates, key=lambda f: f.score)
        
        try:
            # === (1) RL 점수 계산 ===
            rl_scores = self.rl.evaluate_frontiers(
                logodds=self.grid_logodds,
                origin_xy=self.grid_origin_world,
                res_m=self.OGM_RES,
                planner=self.planner,
                robot_xy=tuple(robot_xy),
                robot_yaw=float(self.current_pose[2]),
                frontiers=candidates,
            )

            # === (2) 정규화 (0~1 스케일) ===
            heuristic_scores = np.array([f.score for f in candidates], dtype=float)
            rl_scores = np.array(rl_scores, dtype=float)

            def normalize(arr):
                arr_min, arr_max = np.min(arr), np.max(arr)
                if arr_max - arr_min < 1e-6:  # 값이 거의 같을 경우 0으로
                    return np.zeros_like(arr)
                return (arr - arr_min) / (arr_max - arr_min)

            heuristic_norm = normalize(heuristic_scores)
            rl_norm = normalize(rl_scores)

            # === (3) 휴리스틱 + RL 점수 결합 (1:1 비율) ===
            W_HEURISTIC = 0.3
            W_RL = 0.7

            combined_scores = W_HEURISTIC * heuristic_norm + W_RL * rl_norm

            # === (4) 최고 점수 선택 ===
            best_idx = int(np.argmax(combined_scores))
            best_frontier = candidates[best_idx]

            # === 디버깅 출력 ===
            print(f"🤖 Combined Frontier Selection (Weighted):")
            print(f"   Total candidates: {len(candidates)}")
            print(f"   Heuristic weight = {W_HEURISTIC}, RL weight = {W_RL}")
            print(f"   Heuristic max: {max(f.score for f in candidates):.2f}")
            print(f"   RL max: {max(rl_scores):.2f}")
            print(f"   Selected index: {best_idx}")
            print(f"   Combined score: {combined_scores[best_idx]:.2f}")
            print(f"   Frontier center: ({best_frontier.center_xy[0]:.2f}, {best_frontier.center_xy[1]:.2f})")

            return best_frontier

        except Exception as e:
            print("⚠️ RL frontier hook failed, fallback to heuristic:", e)
            return max(candidates, key=lambda f: f.score)

    # ================== Message handling ==================
    def parse_and_update(self, message):
        lidar_parts, pose_parts, command_parts = self._classify_message(message)

        print("status : "+ command_parts)

        for lp in lidar_parts:
            self._set_LidarUpdate(lp)

        self._set_poseUpdate(pose_parts, command_parts)

        if command_parts == "RENEWAL":
            payload_plan = {"status": "renewal", "frontier_rl": "None", "goal_xy": "None", "path": [], "slam_pose": {
                        "x": float(self.current_pose[0]),
                        "y": float(self.current_pose[1]),
                        "theta": float(self.current_pose[2]),
                    }}
            return payload_plan

        elif command_parts == "PROCESS":

            frontier_exists = (self.last_goal_center_xy is not None)   # change
            path_exists     = bool(self.last_path_xy)
    
            if self.runtime_check == 2:
                done = self.planner.notify_frontier_presence(frontier_exists, path_exists)
                if done:
                    
                    if self.return_to_origin == False:
                        # 이 자리에 self.exploration_success 계산이 필요함
                        try:
                            self.planner.update_dynamic_metrics((self.current_pose[0], self.current_pose[1]))
                            self.exploration_success = float(self.planner.exploration_success_ratio())
                        except Exception as e:
                            print("⚠️ exploration_success update failed:", e)
                            self.exploration_success = 0.0

                        print("✅ 종료 - return origin position")
                        origin_xy = (0.0, 0.0)
                        # self.last_goal_center_xy = origin_xy
                        
                        path_xy = self.planner.plan_path_return(
                            start_xy=(self.current_pose[0], self.current_pose[1]),
                            goal_xy=origin_xy,
                        )
                        # self.last_path_xy = path_xy

                        if not path_xy:
                            print("⚠️ A* returned empty path. Goal may be inside inflated obstacles / unreachable.")

                        self.return_to_origin = True

                        payload_plan = {
                            "status": "continue",
                            "frontier_rl": self.use_rl_frontier,
                            "goal_xy": origin_xy,
                            "path": path_xy,
                            "slam_pose": {
                                "x": float(self.current_pose[0]),
                                "y": float(self.current_pose[1]),
                                "theta": float(self.current_pose[2]),
                            }
                        }

                        # self._making_dqn_Data()

                        return payload_plan
                    else:
                        print("✅ 종료 - data download")
                        #요기서 전체적인 데이터 저장
                        self._save_metrics_json()
                        self._gridmap_binaryData_zip()
                        return {"status": "done", "frontier_rl": self.use_rl_frontier, "goal_xy": None, "path": []}


                payload_plan = {
                    "status": "continue",
                    "frontier_rl": self.use_rl_frontier,
                    "goal_xy": self.last_goal_center_xy,
                    "path": self.last_path_xy,
                    "slam_pose": {
                        "x": float(self.current_pose[0]),
                        "y": float(self.current_pose[1]),
                        "theta": float(self.current_pose[2]),
                    }
                }
                self.exploration_number += 1

                # self._making_dqn_Data()

                return payload_plan
            else:
                self.runtime_check += 1 
                payload_plan = {
                    "status": "continue",
                    "frontier_rl": self.use_rl_frontier,
                    "goal_xy": self.last_goal_center_xy,
                    "path": self.last_path_xy,
                    "slam_pose": {
                        "x": float(self.current_pose[0]),
                        "y": float(self.current_pose[1]),
                        "theta": float(self.current_pose[2]),
                    }
                }
                return payload_plan

    # ================== ZMQ loop (request → plan → reply) ==================
    def zmq_loop(self):
        self.exploration_s = time.time()
        while self.running:
            try:
                msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                result_msg = self.parse_and_update(msg)
                payload = {"ok": True, "result": result_msg}
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
            intensity, origin="lower", extent=extent, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.8, zorder=1
        )

        (self.path_line,) = self.ax.plot([], [], lw=1.5, alpha=0.9, zorder=3, label="Path")
        (self.pose_marker,) = self.ax.plot([], [], "o", ms=5, alpha=0.9, zorder=4, label="Robot")

        (self.goal_marker,) = self.ax.plot([], [], "o", ms=8, zorder=7, label="Selected Frontier", color="#32CD32")
        (self.gola_frontier_pts,) = self.ax.plot([], [], ".", ms=4, alpha=0.85, zorder=2, label="Selected Goal Frontier", color="#90EE90")
        # (self.frontier_pts,) = self.ax.plot([], [], ".", ms=3, color="red", alpha=0.9, zorder=5, label="Frontiers")
        
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

                # if self.frontier_pts is not None:
                #     self.frontier_pts.set_data(self.frontier_xs, self.frontier_ys)

                if self.gola_frontier_pts is not None:
                    self.gola_frontier_pts.set_data(self.goal_frontier_xs, self.goal_frontier_ys)

                if self.path_x:
                    self.path_line.set_data(self.path_x, self.path_y)
                    self.pose_marker.set_data([self.path_x[-1]], [self.path_y[-1]])

                if self.last_goal_center_xy is not None:
                    self.goal_marker.set_data([self.last_goal_center_xy[0]], [self.last_goal_center_xy[1]])
                else:
                    self.goal_marker.set_data([], [])

                if self._frame % self.VIS_EVERY == 0:
                    xs, ys = [], []
                    if self.path_x:
                        xs += list(self.path_x); ys += list(self.path_y)
                    xs += [extent[0], extent[1]]
                    ys += [extent[2], extent[3]]
                    xs = np.asarray(xs); ys = np.asarray(ys)
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

        if self.current_scan:
            self.ogm_update_scan((x, y, theta), self.current_scan)
            self.current_scan = []

        self.add_pose_node((x, y, theta))
        self.try_loop_closure((x, y, theta))

        if time.time() - self.last_opt_time > self.OPT_INTERVAL and len(self.graph._vertices) >= 2:
            self.graph.optimize()
            self.last_opt_time = time.time()

        poses = [v.pose for v in self.graph._vertices]
        self.path_x = deque([p.position[0] for p in poses])
        self.path_y = deque([p.position[1] for p in poses])


        if (self.use_what_frontier and command_parts == "PROCESS"):
            self._make_MRTSP_frontier(x, y, theta)
        elif (self.use_what_frontier == False and command_parts == "PROCESS"):
            self._make_experiment_frontier(x, y)

    def _set_LidarUpdate(self, parts):
        angle, distance, intensity = map(float, parts)
        if abs(angle) > (2 * np.pi + 1e-3):
            angle = np.deg2rad(angle)
        no_hit = (distance <= 0) or (distance >= self.LIDAR_MAX_RANGE_MM - self.NO_HIT_MARGIN_MM) or (intensity <= 0.5)
        r = (self.LIDAR_MAX_RANGE_MM if no_hit else distance) / 1000.0
        px = r * np.cos(angle); py = r * np.sin(angle)
        self.current_scan.append((px, py, not no_hit))

    # ======================================================================================
    # MRTSP Set
    def _set_MRTSP_frontier(self):
        from frontier.frontier_wfd import FrontierDetector
        from frontier.mrtsp_selector import FrontierMRTSPSelector
        # 검출기 (WFD)
        self.detector = FrontierDetector(
            ogm_res_m=self.OGM_RES,
            grid_origin_world_xy=self.grid_origin_world,
            occ_thresh=0.65,
            free_thresh=0.35,
            min_cluster_size=12,
            dilate_free=2,
            min_clearance_m=0.35,
            require_reachable=True,
            ignore_border_unknown_margin_m=0.2,
        )

        # 선택기 (MRTSP 전용)
        self.selector_mrtsp = FrontierMRTSPSelector(
            ogm_res_m=self.OGM_RES,
            grid_origin_world_xy=self.grid_origin_world,
            sensor_range_m=10.0,
            Wd=1.0, Ws=1.0,
            Vmax=0.8, Wmax=1.2,
        )

    def _make_MRTSP_frontier(self, robot_x, robot_y, robot_theta):
        from collections import deque

        # ---------- 헬퍼 ----------
        def _binary_dilate_bool(mask_bool: np.ndarray, iters: int) -> np.ndarray:
            out = mask_bool.astype(bool).copy()
            H, W = out.shape
            for _ in range(max(0, int(iters))):
                grown = out.copy()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        yy0 = max(0, -dy); yy1 = min(H, H - dy)
                        xx0 = max(0, -dx); xx1 = min(W, W - dx)
                        grown[yy0:yy1, xx0:xx1] |= out[yy0+dy:yy1+dy, xx0+dx:xx1+dx]
                out = grown
            return out

        def _flood_fill_reachable_free(free_bool: np.ndarray, start_iy_ix) -> np.ndarray:
            H, W = free_bool.shape
            sy, sx = start_iy_ix
            reachable = np.zeros((H, W), dtype=bool)
            if not (0 <= sy < H and 0 <= sx < W): return reachable
            if not free_bool[sy, sx]: return reachable
            q = deque([(sy, sx)])
            reachable[sy, sx] = True
            while q:
                y, x = q.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0: continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and (not reachable[ny, nx]) and free_bool[ny, nx]:
                            reachable[ny, nx] = True
                            q.append((ny, nx))
            return reachable

        # ---------- 1) 프런티어 검출 ----------
        det = self.detector.detect(self.grid_logodds, robot_xy=(robot_x, robot_y))
        cands = det["candidates"]
        if not cands:
            self.last_frontiers = []
            self.last_goal_center_xy = None
            self.last_path_xy = []
            return

        self.frontier_s = time.time()

        # ---------- 2) 맵 이진화 + 안전 팽창 ----------
        p_occ = 1.0 / (1.0 + np.exp(-self.grid_logodds))
        free = (p_occ <= getattr(self.planner, "free_thresh", 0.35))
        occ  = (p_occ >= getattr(self.planner, "occ_thresh", 0.65))

        SAFETY_INFLATE_M = 1.6
        inflate_px = max(1, int(round(SAFETY_INFLATE_M / float(self.OGM_RES))))
        occ_inflated = _binary_dilate_bool(occ, inflate_px)
        free_safe = free & (~occ_inflated)

        # ---------- 3) Reachable Free 마스크 ----------
        iy0, ix0 = self.world_to_map(robot_x, robot_y)
        self._ensure_in_grid(iy0, ix0)
        reachable = _flood_fill_reachable_free(free_safe, (iy0, ix0))

        # ---------- 4) 사전 필터링: 도달 가능한 후보만 ----------
        filtered = []
        H, W = free.shape
        for f in cands:
            pix = getattr(f, "pixel_inds", None)
            used = False
            if pix:
                for (iy, ix) in pix:
                    if 0 <= iy < H and 0 <= ix < W and reachable[iy, ix]:
                        used = True
                        break
            else:
                cy, cx = getattr(f, "center_ij", (-1, -1))
                if 0 <= cy < H and 0 <= cx < W and reachable[cy, cx]:
                    used = True
            if used:
                filtered.append(f)
        cands_for_select = filtered if filtered else cands  # 폴백

        # ---------- 5) MRTSP 선택(순서 결정만) ----------
        res = self.selector_mrtsp.select(
            candidates=cands_for_select,
            robot_xy=(robot_x, robot_y),
            robot_yaw=robot_theta,
            return_sequence=True,
            return_matrix=False,
        )
        

        seq = getattr(res, "sequence_idx", None)
        if seq:
            self.frontier_number.append(len(seq))
            chosen = cands_for_select[seq[0]]
        elif getattr(res, "chosen", None) is not None:
            self.frontier_number.append(1)
            chosen = res.chosen
        else:
            self.frontier_number.append(0)
            chosen = None

        if chosen is None:
            self.last_frontiers = []
            self.last_goal_center_xy = None
            self.last_path_xy = []
            return

        # ---------- 6) 시각화용 픽셀 ----------
        xs, ys = [], []
        x0, y0 = self.grid_origin_world
        pix = getattr(chosen, "pixel_inds", None)
        if pix:
            for (iy, ix) in pix:
                xs.append(x0 + (ix + 0.5) * self.OGM_RES)
                ys.append(y0 + (iy + 0.5) * self.OGM_RES)
        self.goal_frontier_xs = xs
        self.goal_frontier_ys = ys

        # ---------- 7) 목표점 스냅 (표준 핵심) ----------
        # 선택된 클러스터의 픽셀들 중 reachable∩free_safe에서 로봇과 가장 가까운 셀로 스냅
        goal_xy = None
        if pix:
            best_d2 = float("inf")
            for (iy, ix) in pix:
                if 0 <= iy < H and 0 <= ix < W and reachable[iy, ix] and free_safe[iy, ix]:
                    gx = x0 + (ix + 0.5) * self.OGM_RES
                    gy = y0 + (iy + 0.5) * self.OGM_RES
                    d2 = (gx - robot_x)*(gx - robot_x) + (gy - robot_y)*(gy - robot_y)
                    if d2 < best_d2:
                        best_d2 = d2
                        goal_xy = (gx, gy)

        # 스냅 실패 시: 대표점(center_xy)을 마지막 폴백으로 사용
        if goal_xy is None:
            goal_xy = tuple(chosen.center_xy)

        self.last_frontiers = [chosen]
        self.last_goal_center_xy = goal_xy  # ⬅️ 스냅된 free&reachable 좌표를 사용

        self.frontier_e = time.time()
        self.frontier_exploration_time.append(self.frontier_e - self.frontier_s)

        # ---------- 8) 실제 경로계획 ----------
        self._make_A_start_path(robot_x, robot_y)

   

    # Experiment Set
    def _set_experiment_frontier(self):
        from frontier.frontier_wfd import FrontierDetector
        from frontier.experiment_selector import FrontierExSelector, ScoredFrontier
        
        self.detector = FrontierDetector(
            ogm_res_m=self.OGM_RES,
            grid_origin_world_xy=self.grid_origin_world,
            occ_thresh=0.65,
            free_thresh=0.35,
            # min_cluster_size=12,
            min_cluster_size=3,
            dilate_free=2,
            # min_clearance_m=0.35,
            min_clearance_m=0.2,
            require_reachable=True,
            # ignore_border_unknown_margin_m=0.2,
            ignore_border_unknown_margin_m=0.1,
        )

        self.selector = FrontierExSelector(
            ogm_res_m=self.OGM_RES,
            grid_origin_world_xy=self.grid_origin_world,
            # info_radius_m=1.0,
            info_radius_m=2.5,
            visible_rays=64,
            ray_step_px=1,
            # min_free_before_unknown_m=0.6,
            # merge_min_sep_m=1.5,
            min_free_before_unknown_m=0.3,
            merge_min_sep_m=0.8,

            w_info=0.7, w_size=0.1, w_dist=0.05,
            w_open=1.0, w_trace=0.7,
        )

    def _make_experiment_frontier(self, robot_x, robot_y):
        # 1) 탐지
        det_out = self.detector.detect(self.grid_logodds, robot_xy=(robot_x, robot_y))
        masks = det_out["masks"]
        candidates = det_out["candidates"]

        self.frontier_s = time.time()

        # 2) 점수화/정렬/병합
        all_frontiers = self.selector.score_and_select(
            candidates=candidates,
            masks=masks,
            robot_xy=(robot_x, robot_y),
            exploration_trace=self.trace,
            do_merge=True,
            top_k=None
        )
        
        self.frontier_number.append(len(all_frontiers))

        self._last_all_frontiers = all_frontiers

        # # 점수화 후의 클러스터 중심 시각화
        # centers_x, centers_y = [], []
        # for sf in all_frontiers:
        #     cx, cy = sf.center_xy
        #     centers_x.append(cx); centers_y.append(cy)
        # self.frontier_xs = centers_x
        # self.frontier_ys = centers_y

        # === 기존 선택 로직 유지 ===
        chosen = self._select_frontier(all_frontiers, (robot_x, robot_y))
        
        self.last_frontiers = [chosen] if chosen else []

        sel_xs, sel_ys = [], []
        if chosen:
            pix = getattr(chosen, "candidate", None)
            if pix is not None:
                pix = getattr(chosen.candidate, "pixel_inds", None)
            if pix is None:
                pix = getattr(chosen, "pixel_inds", None)
            if pix:
                x0, y0 = self.grid_origin_world
                for (iy, ix) in pix:
                    sel_xs.append(x0 + (ix + 0.5) * self.OGM_RES)
                    sel_ys.append(y0 + (iy + 0.5) * self.OGM_RES)

        self.goal_frontier_xs = sel_xs  
        self.goal_frontier_ys = sel_ys

        self.frontier_e = time.time()

        self.frontier_exploration_time.append(self.frontier_e - self.frontier_s)

        self.last_goal_center_xy = chosen.center_xy if chosen else None

        self._make_A_start_path(robot_x, robot_y)

    def _make_A_start_path(self, x, y):
        path_xy = []
        if self.last_goal_center_xy is not None:
            path_xy = self.planner.plan_path(
                start_xy=(x, y),
                goal_xy=tuple(self.last_goal_center_xy),
                safety_inflate_m=0.3,
                allow_diagonal=True,
            )
            print(f"🧭 A* start={x:.2f},{y:.2f}  goal={self.last_goal_center_xy}  path_len={len(path_xy)}")
            self.path_length.append(len(path_xy))
            if not path_xy:
                print("⚠️ A* returned empty path. Goal may be inside inflated obstacles / unreachable.")
                self.last_path_xy = []
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

    # ========================== DQN Data Save ===============================
    def _making_dqn_Data(self):
        """
        PROCESS 시점에 호출.
        - 직전 PROCESS에서 pending된 (obs, action)에 대해 보상을 계산/finish 기록
        - 현재 PROCESS의 (obs, action)을 start 기록
        """
        # 후보/선택이 없으면 스킵
        if not hasattr(self, "_last_all_frontiers"):
            return
        candidates = getattr(self, "_last_all_frontiers", [])
        if not candidates or not self.last_frontiers:
            return
        chosen = self.last_frontiers[0]
        # chosen이 candidates 어디에 있는지 찾기
        try:
            chosen_pos = next(i for i, c in enumerate(candidates) if c is chosen)
        except StopIteration:
            # 객체 동일성으로 못 찾을 경우 좌표로 fallback
            def _key(f): return (float(f.center_xy[0]), float(f.center_xy[1]))
            ck = _key(chosen)
            idxs = [i for i, c in enumerate(candidates) if _key(c) == ck]
            if not idxs:
                return
            chosen_pos = idxs[0]

        # === 현재 PROCESS 관측/마스크 생성 (Top-K) ===
        def _plan_fn(start_xy, goal_xy, safety_inflate_m=1.6, allow_diagonal=True):
            return self.planner.plan_path(
                start_xy=start_xy,
                goal_xy=goal_xy,
                safety_inflate_m=safety_inflate_m,
                allow_diagonal=allow_diagonal,
            )

        obs_2d, mask_1d, paths = self._rl_mod.make_obs_and_mask(
            grid_logodds=self.grid_logodds,
            planner=self.planner,
            grid_origin_world=self.grid_origin_world,
            ogm_res=self.OGM_RES,
            robot_xy=(self.current_pose[0], self.current_pose[1]),
            robot_yaw=float(self.current_pose[2]),
            candidates=candidates,          # 전체 후보 그대로 전달 (내부에서 Top-K 슬라이스)
            plan_path_fn=_plan_fn,
        )

        top_k = self._rl_cfg.top_k
        # 선택한 후보가 Top-K 범위 밖이면 이번 스텝은 로그 스킵(학습 일관성)
        if chosen_pos >= top_k:
            # 그래도 다음 보상 계산을 위해 이전 unknown_ratio 갱신은 해두자
            # self._rl_prev["unknown_ratio"] = float(getattr(self.planner, "unknown_ratio", 0.0))
            val = getattr(self.planner, "unknown_ratio", 0.0)
            if callable(val): val = val()
            self._rl_prev["unknown_ratio"] = float(val)
            return

        action_idx = int(chosen_pos)           # Top-K 내에서의 인덱스(동일 순서 가정)
        path_len_m = float(len(self.last_path_xy)) * float(self.OGM_RES)
        success = bool(len(self.last_path_xy) > 0)

        # === 1) 직전 PROCESS 스텝 마무리 (보상/next_obs) ===
        prev_unknown = self._rl_prev.get("unknown_ratio", None)
        prev_obs     = self._rl_prev.get("obs", None)
        prev_action  = self._rl_prev.get("action", None)
        prev_path_m  = self._rl_prev.get("path_len_m", None)
        prev_success = self._rl_prev.get("success", None)

        # curr_unknown = float(getattr(self.planner, "unknown_ratio", 0.0))
        val = getattr(self.planner, "unknown_ratio", 0.0)
        if callable(val): val = val()
        curr_unknown = float(val)


        if (prev_obs is not None) and (prev_action is not None) and (prev_unknown is not None) and (prev_path_m is not None) and (prev_success is not None):
            # 보상 계산: (직전 unknown → 현재 unknown) 변화 사용
            reward = self._rl_mod.compute_reward(
                before_unknown_ratio=float(prev_unknown),
                after_unknown_ratio=float(curr_unknown),
                path_len_m=float(prev_path_m),
                success=bool(prev_success),
                replan=False,
            )
            # 다음 관측은 "현재 PROCESS의 obs"
            self._rl_mod.finish_step(reward=float(reward), next_obs_2d=obs_2d, done=False)

        # === 2) 현재 PROCESS 스텝 시작 (관측/행동 기록) ===
        self._rl_mod.start_step(obs_2d=obs_2d, action=int(action_idx))

        # === 3) 다음 보상 계산을 위해 현재 상태를 pending에 저장 ===
        self._rl_prev["unknown_ratio"] = curr_unknown
        self._rl_prev["obs"] = obs_2d
        self._rl_prev["action"] = int(action_idx)
        self._rl_prev["path_len_m"] = float(path_len_m)
        self._rl_prev["success"] = bool(success)

    # ========================== SLAM Map Optimization ===================

    def _ensure_dir(self, path: Union[str, Path]):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _gridmap_binaryData_zip(self):
        try:
            export_dir = self._ensure_dir("./exports")
            ts = self._timestamp()

            rx, ry, rth = map(float, self.current_pose)
            iy, ix = self.world_to_map(rx, ry)  

            H, W = self.grid_logodds.shape
            res = float(self.OGM_RES)
            x0, y0 = map(float, self.grid_origin_world)
            
            # 1) log-odds -> p_occ
            p_occ = 1.0 / (1.0 + np.exp(-self.grid_logodds))

            occ_thresh  = getattr(self.planner, "occ_thresh", 0.65)
            free_thresh = getattr(self.planner, "free_thresh", 0.35)

            # 2) ROS 규격 데이터 값(-1: unknown, 0: free, 100: occupied)
            data = np.full((H, W), -1, dtype=np.int8)
            data[p_occ <= free_thresh] = 0
            data[p_occ >= occ_thresh]  = 100

            # 3) row-major bytes
            raw_bytes = data.tobytes(order="C")

            # 4) gzip -> base64
            gz_buf = io.BytesIO()
            with gzip.GzipFile(fileobj=gz_buf, mode="wb") as f:
                f.write(raw_bytes)
            gz_bytes = gz_buf.getvalue()
            b64 = base64.b64encode(gz_bytes).decode("ascii")

            payload = {
                "robot" : {
                    "x": rx,
                    "y": ry,
                    "theta": rth,
                    "map_index": {               
                        "iy": int(iy),
                        "ix": int(ix),
                    },
                },
                "width": int(W),
                "height": int(H),
                "resolution": res,
                "origin": {"x": x0, "y": y0, "z": 0.0, "yaw": 0.0},
                "data_gzip_b64": b64,
            }

            json_path = Path(export_dir) / f"gridmap_{ts}.json"
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)


            print(f"📦 Saved grid binary JSON and gzip:\n - {json_path}")

        except Exception as e:
            print(f"❌ _gridmap_binaryData_zip error: {e}")
    
    def _save_metrics_json(self):

        try:
            export_dir = self._ensure_dir("./exports")
            ts = self._timestamp()

            # 셀 수 -> 미터로 변환
            path_len_m_list = [float(n) * float(self.OGM_RES) for n in self.path_length]

            # boundary_unknown_ratio도 같이 남기고 싶으면 planner에 최신 업데이트가 되어있다는 가정 하에 시도
            boundary_unknown_ratio = None
            try:
                if hasattr(self.planner, "boundary_unknown_ratio") and callable(self.planner.boundary_unknown_ratio):
                    boundary_unknown_ratio = float(self.planner.boundary_unknown_ratio())
            except Exception:
                boundary_unknown_ratio = None

            payload = {
                "frontier_number": list(map(int, self.frontier_number)),
                "exploration_number": int(self.exploration_number),
                "frontier_exploration_time_s": list(map(float, self.frontier_exploration_time)),
                "path_length_cells": list(map(int, self.path_length)),
                "path_length_m": path_len_m_list,
                "exploration_success": float(self.exploration_success),
                "boundary_unknown_ratio": boundary_unknown_ratio,
                "exploration_all_time" : float(time.time() - self.exploration_s),
                "timestamp": ts
            }

            out_path = Path(export_dir) / f"metrics_{ts}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            print(f"📄 Saved metrics JSON -> {out_path}")

        except Exception as e:
            print(f"❌ _save_metrics_json error: {e}")
