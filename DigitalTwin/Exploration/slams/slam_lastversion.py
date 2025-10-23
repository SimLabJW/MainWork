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
FRONTIER_RL_ENABLED = False
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
        self.LIDAR_MAX_RANGE_MM = 40000
        self.NO_HIT_MARGIN_MM = 5

        # === OGM params ===
        self.OGM_RES = 0.1
        self.OGM_INIT_SIZE = (600, 600)
        self.OGM_L_FREE = -1.5 #-1.0
        self.OGM_L_OCC = +1.5  #2.0
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
        self.rl = FrontierRLAgent()

        self._rl_cfg = RLDataConfig(top_k=8, feat_dim=12, log_path="logs/frontier_dqn.jsonl")
        self._rl_mod = RLDataModule(self._rl_cfg)

        # 이전 PROCESS 스텝의 전이 보상 계산을 위한 pending 버퍼
        self._rl_prev = {
            "unknown_ratio": None,   # 이전 PROCESS에서의 미지영역 비율
            "obs": None,             # 이전 PROCESS 관측 [K,D]
            "action": None,          # 이전 PROCESS 행동(Top-K 내 인덱스)
            "path_len_m": None,      # 이전 선택 경로 길이(m)
            "success": None,         # 이전 선택 시 경로 유효여부(True/False)
        }

        # === Frontier Selection Mode (강화학습 온오프프) ===
        self.use_rl_frontier = FRONTIER_RL_ENABLED

        self.planner = GlobalPlanner(
            ogm_res_m=self.OGM_RES,
            occ_thresh=0.65,
            free_thresh=0.35,
            coverage_done_thresh=0.95,
            unknown_left_thresh=0.02,
            no_frontier_patience=10,
        )

        # == slam returning ==
        self.return_to_origin = RETURN_ORIGIN_POSITION

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
        try:
            bonuses = self.rl.predict_bonus_for_frontiers(
            logodds=self.grid_logodds,
            origin_xy=self.grid_origin_world,
            res_m=self.OGM_RES,
            planner=self.planner,
            robot_xy=tuple(robot_xy),
            robot_yaw=float(self.current_pose[2]),
            frontiers=candidates,   
            )  

            best_idx = int(np.argmax([c.score + float(bonuses[i]) for i, c in enumerate(candidates)]))
            return candidates[best_idx]
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
            payload_plan = {"status": "renewal", "frontier_rl": "None", "goal_xy": "None", "path": []}
            return payload_plan

        elif command_parts == "PROCESS":

            frontier_exists = (self.last_goal_center_xy is not None)   # change
            path_exists     = bool(self.last_path_xy)
    
            done = self.planner.notify_frontier_presence(frontier_exists, path_exists)
            if done:
                
                if self.return_to_origin == False:
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
                    }

                    self._making_dqn_Data()

                    return payload_plan
                else:
                    print("✅ 종료 - data download")
                    self._gridmap_binaryData_zip()
                    return {"status": "done", "frontier_rl": self.use_rl_frontier, "goal_xy": None, "path": []}


            payload_plan = {
                "status": "continue",
                "frontier_rl": self.use_rl_frontier,
                "goal_xy": self.last_goal_center_xy,
                "path": self.last_path_xy,
            }

            self._making_dqn_Data()

            return payload_plan

    # ================== ZMQ loop (request → plan → reply) ==================
    def zmq_loop(self):
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
            dilate_free=1,
            min_clearance_m=0.35,
            require_reachable=True,
            ignore_border_unknown_margin_m=0.8,
        )

        # 선택기 (MRTSP 전용)
        self.selector_mrtsp = FrontierMRTSPSelector(
            ogm_res_m=self.OGM_RES,
            grid_origin_world_xy=self.grid_origin_world,
            sensor_range_m=30.0,
            Wd=1.0, Ws=1.0,
            Vmax=0.8, Wmax=1.2,
        )

    def _make_MRTSP_frontier(self, robot_x, robot_y, robot_theta):

        # 1) 탐지
        det = self.detector.detect(self.grid_logodds, robot_xy=(robot_x, robot_y))
        cands = det["candidates"]

        # 2)
        res = self.selector_mrtsp.select(
            candidates=cands,
            robot_xy=(robot_x, robot_y),
            robot_yaw=robot_theta,
            return_sequence=True,
            return_matrix=False,
        )

        chosen = res.chosen
        if chosen is None:
            # 프런티어 없음 → 종료 조건으로 넘기거나, fallback 처리
            self.last_frontiers = []
            self.last_goal_center_xy = None
            self.last_path_xy = []
            return

        xs, ys = [], []
        x0, y0 = self.grid_origin_world
        for (iy, ix) in chosen.pixel_inds:
            xs.append(x0 + (ix + 0.5) * self.OGM_RES)
            ys.append(y0 + (iy + 0.5) * self.OGM_RES)
        self.goal_frontier_xs = xs
        self.goal_frontier_ys = ys

        self.last_frontiers = [chosen]
        self.last_goal_center_xy = tuple(chosen.center_xy)

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
            min_cluster_size=12,
            dilate_free=1,
            min_clearance_m=0.35,
            require_reachable=True,
            ignore_border_unknown_margin_m=0.8,
        )

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

    def _make_experiment_frontier(self, robot_x, robot_y):
        # 1) 탐지
        det_out = self.detector.detect(self.grid_logodds, robot_xy=(robot_x, robot_y))
        masks = det_out["masks"]
        candidates = det_out["candidates"]

        # 2) 점수화/정렬/병합
        all_frontiers = self.selector.score_and_select(
            candidates=candidates,
            masks=masks,
            robot_xy=(robot_x, robot_y),
            exploration_trace=self.trace,
            do_merge=True,
            top_k=None
        )

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

        self.last_goal_center_xy = chosen.center_xy if chosen else None

        self._make_A_start_path(robot_x, robot_y)

    def _make_A_start_path(self, x, y):
        path_xy = []
        if self.last_goal_center_xy is not None:
            path_xy = self.planner.plan_path(
                start_xy=(x, y),
                goal_xy=tuple(self.last_goal_center_xy),
                safety_inflate_m=1.8,
                allow_diagonal=True,
            )
            print(f"🧭 A* start={x:.2f},{y:.2f}  goal={self.last_goal_center_xy}  path_len={len(path_xy)}")
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
        """
        마지막 grid map을 ROS nav_msgs/OccupancyGrid 관례에 맞는 데이터로 직렬화 후
        gzip + base64로 압축하여 JSON 파일로 저장한다.

        JSON 스키마:
        {
            "width": W,
            "height": H,
            "resolution": <meters per cell>,
            "origin": {"x": <world_x_of_cell00>, "y": <world_y_of_cell00>, "z": 0.0, "yaw": 0.0},
            "data_gzip_b64": "<base64.gz>"
        }
        별도로 .bin.gz 원시 파일도 저장한다.
        """
        try:
            export_dir = self._ensure_dir("./exports")
            ts = self._timestamp()

            H, W = self.grid_logodds.shape
            res = float(self.OGM_RES)
            x0, y0 = map(float, self.grid_origin_world)  # 맵 좌하단 월드 좌표

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
                "width": int(W),
                "height": int(H),
                "resolution": res,
                "origin": {"x": x0, "y": y0, "z": 0.0, "yaw": 0.0},
                "data_gzip_b64": b64,
            }

            json_path = Path(export_dir) / f"gridmap_{ts}.json"
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)

            # 원시 바이너리도 별도로 저장(선택)
            bin_gz_path = Path(export_dir) / f"gridmap_{ts}.bin.gz"
            with open(bin_gz_path, "wb") as fp:
                fp.write(gz_bytes)

            print(f"📦 Saved grid binary JSON and gzip:\n - {json_path}\n - {bin_gz_path}")

        except Exception as e:
            print(f"❌ _gridmap_binaryData_zip error: {e}")
    
 