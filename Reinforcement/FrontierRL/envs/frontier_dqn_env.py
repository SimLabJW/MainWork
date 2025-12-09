# -*- coding: utf-8 -*-
"""
FrontierDQNEnv v3 - Pure Reinforcement Learning

핵심:
- What-if simulation 완전 제거
- DQN이 시행착오를 통해 스스로 학습
- 실제 이동 + 실제 리워드만 사용
- 관측 가능한 특징만 사용 (센서로 측정 가능)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import threading
import time

# ======= LiDAR / OGM 파라미터 =======
OGM_L_FREE = -1.2
OGM_L_OCC  = +1.8
OGM_CLAMP  = (-5.0, 5.0)
FREE_T     = 0.35
OCC_T      = 0.65

LIDAR_MAX_RANGE_M = 40.0
RAY_COUNT  = 360

from .frontier.frontier_wfd import FrontierDetector
from .frontier.experiment_selector import FrontierExSelector, ScoredFrontier
from .frontier.global_planner import GlobalPlanner


def p_occ_from_logodds(L: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-L))


def _xy_to_ij(x: float, y: float, origin_xy: Tuple[float, float], res: float) -> Tuple[int, int]:
    x0, y0 = origin_xy
    ix = int(math.floor((x - x0) / res))
    iy = int(math.floor((y - y0) / res))
    return iy, ix


def _ij_to_xy(iy: int, ix: int, origin_xy: Tuple[float, float], res: float) -> Tuple[float, float]:
    x0, y0 = origin_xy
    x = x0 + (ix + 0.5) * res
    y = y0 + (iy + 0.5) * res
    return (x, y)


def _in_bounds(iy: int, ix: int, H: int, W: int) -> bool:
    return (0 <= iy < H) and (0 <= ix < W)


def _bresenham(iy0: int, ix0: int, iy1: int, ix1: int):
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


class FrontierDQNEnv(gym.Env):
    """순수 강화학습 기반 Frontier 선택 환경"""
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        maps: Dict[str, Dict],
        *,
        lidar_max_range_m: float = LIDAR_MAX_RANGE_M,
        ogm_res: float = 0.1,
        occ_thresh: float = OCC_T,
        free_thresh: float = FREE_T,
        max_steps: int = 200,
        top_k_frontiers: int = 5,
        reward_info_gain: float = 10.0,
        reward_distance_penalty: float = -0.1,
        reward_quality_bonus: float = 5.0,
        reward_invalid: float = -5.0,
        episodes_per_map: int = 50,
        coverage_done_thresh: float = 0.54,
        seed: Optional[int] = None,
        enable_visualization: bool = True,
        view_mode: str = "fit",
        follow_window_m: float = 5.0,
        follow_margin_m: float = 0.0,
        robot_speed_mps: float = 6.0,
        step_dt: float = 0.3,
        lidar_scan_interval_steps: int = 3,
    ):
        super().__init__()

        self.maps = maps
        self.map_names = sorted(list(maps.keys()))
        assert len(self.map_names) > 0

        self.episodes_per_map = int(episodes_per_map)
        self._current_map_idx = 0
        self._episode_count_on_current_map = 0
        self._total_episode_count = 0

        self.lidar_max_range_m = float(lidar_max_range_m)
        self.ogm_res = float(ogm_res)
        self.occ_thresh = float(occ_thresh)
        self.free_thresh = float(free_thresh)
        self.max_steps = int(max_steps)
        self.top_k = int(top_k_frontiers)
        self.coverage_done_thresh = float(coverage_done_thresh)

        self.reward_info_gain = float(reward_info_gain)
        self.reward_distance_penalty = float(reward_distance_penalty)
        self.reward_quality_bonus = float(reward_quality_bonus)
        self.reward_invalid = float(reward_invalid)

        self.rng = np.random.default_rng(seed)

        self.current_map_name: Optional[str] = None
        self.current_logodds: Optional[np.ndarray] = None
        self.origin_xy: Optional[Tuple[float, float]] = None
        self.converted_coords: List[Tuple[int, int]] = []
        self._initial_spawn_ij: Optional[Tuple[int, int]] = None
        self.robot_xy: Tuple[float, float] = (0.0, 0.0)
        self.robot_yaw: float = 0.0
        self.step_count: int = 0

        self.detector: Optional[FrontierDetector] = None
        self.selector: Optional[FrontierExSelector] = None
        self.planner: Optional[GlobalPlanner] = None

        self._frontiers: List[Optional[ScoredFrontier]] = []

        self._last_frontiers: List = []
        self._last_path_xy: List[Tuple[float, float]] = []
        self._last_chosen_frontier_xy: Optional[Tuple[float, float]] = None
        self._last_reward: float = 0.0
        self._last_info_gain: float = 0.0
        self._last_action: Optional[int] = None

        # ⭐ Observation: 관측 가능한 특징만 (센서로 측정 가능)
        feature_dim = 15
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.top_k * feature_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.top_k)

        self.enable_visualization = enable_visualization
        self.view_mode = view_mode
        self.follow_window_m = follow_window_m
        self.follow_margin_m = follow_margin_m

        self.robot_speed_mps = float(robot_speed_mps)
        self.step_dt = float(step_dt)
        self.lidar_scan_interval_steps = int(lidar_scan_interval_steps)

        self._viz_fig = None
        self._viz_ax = None
        self._viz_thread = None
        self._viz_running = False

        self._map_lock = threading.RLock()
        self.lidar_full_sweep_atomic = True
        self.scans_per_render = 1
        self.viz_sleep_scale = 0.0

        if self.enable_visualization:
            try:
                if matplotlib.get_backend().lower() == "agg":
                    matplotlib.use("TkAgg", force=False)
            except Exception:
                pass
            self._start_visualization()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._episode_count_on_current_map >= self.episodes_per_map:
            self._current_map_idx = (self._current_map_idx + 1) % len(self.map_names)
            self._episode_count_on_current_map = 0
            self._initial_spawn_ij = None

        self.current_map_name = self.map_names[self._current_map_idx]
        self._episode_count_on_current_map += 1
        self._total_episode_count += 1

        md = self.maps[self.current_map_name]
        self.current_logodds = md["logodds"].copy()
        self.origin_xy = tuple(md["origin_xy"])
        self.converted_coords = md.get("converted_coords", [])
        H, W = self.current_logodds.shape

        # 로봇 스폰
        if len(self.converted_coords) > 0:
            if self._episode_count_on_current_map == 1:
                cy, cx = self.converted_coords[self.rng.integers(len(self.converted_coords))]
                self._initial_spawn_ij = (cy, cx)
            else:
                base_cy, base_cx = self._initial_spawn_ij
                nearby = [(y, x) for y, x in self.converted_coords 
                         if abs(y - base_cy) < 50 and abs(x - base_cx) < 50]
                cy, cx = nearby[self.rng.integers(len(nearby))] if nearby else self._initial_spawn_ij
        else:
            cy, cx = H // 2, W // 2

        self.robot_xy = _ij_to_xy(cy, cx, self.origin_xy, self.ogm_res)
        self.robot_yaw = float(self.rng.uniform(0, 2 * math.pi))
        self.step_count = 0

        # 컴포넌트 초기화
        self.detector = FrontierDetector(
            ogm_res_m=self.ogm_res,
            grid_origin_world_xy=self.origin_xy,
            occ_thresh=self.occ_thresh,
            free_thresh=self.free_thresh,
            min_cluster_size=10,
        )
        self.selector = FrontierExSelector(
            ogm_res_m=self.ogm_res,
            grid_origin_world_xy=self.origin_xy,
            w_info=0.7, w_size=0.1, w_dist=0.05, w_open=1.0, w_trace=0.5,
        )
        self.planner = GlobalPlanner(
            ogm_res_m=self.ogm_res,
            occ_thresh=self.occ_thresh,
            free_thresh=self.free_thresh,
        )
        with self._map_lock:
            self.planner.update_map(self.current_logodds, self.origin_xy)

        # 초기 스캔
        self._lidar_scan_and_update()

        # Frontier 탐지
        self._detect_frontiers()

        obs = self._get_observation()
        info = {
            "map_name": self.current_map_name,
            "episode_on_map": self._episode_count_on_current_map,
            "total_episode": self._total_episode_count,
        }
        return obs, info

    def step(self, action: int):
        """
        ⭐ 순수 강화학습:
        1. DQN이 frontier 선택
        2. 실제 이동 + 스캔 (A* + LiDAR 유지)
        3. 실제 리워드 계산
        4. DQN이 경험으로 학습
        """
        self._last_action = int(action)

        if self._no_candidates():
            obs = self._get_observation()
            return obs, 0.0, True, False, {"reason": "no_frontier", "step": self.step_count}

        if action < 0 or action >= self.top_k or self._frontiers[action] is None:
            # 잘못된 액션
            ig = self._lidar_scan_and_update()
            reward = self.reward_invalid + self.reward_info_gain * ig
            self._last_reward = reward
            self._last_info_gain = ig
            self.step_count += 1
            self._detect_frontiers()
            obs = self._get_observation()
            return obs, reward, False, self.step_count >= self.max_steps, {"step": self.step_count}

        # ⭐ DQN이 선택한 frontier로 이동
        chosen = self._frontiers[action]
        self._last_chosen_frontier_xy = chosen.center_xy
        goal_xy = chosen.center_xy

        # A* 경로 계획 (유지)
        path = self.planner.plan_path(
            start_xy=self.robot_xy,
            goal_xy=goal_xy,
            safety_inflate_m=0.6,
            allow_diagonal=True,
        )
        self._last_path_xy = path

        if len(path) <= 1:
            # 경로 없음
            reward = self.reward_invalid
            self._last_reward = reward
            self._last_info_gain = 0.0
            self.step_count += 1
            self._detect_frontiers()
            obs = self._get_observation()
            return obs, reward, False, self.step_count >= self.max_steps, {"step": self.step_count}

        # ⭐ 실제 이동 + LiDAR 스캔 (완전 유지)
        info_gain_total, distance_traveled = self._follow_path_smooth(path)

        # ⭐ 도착 후 품질 평가 (센서로 측정)
        quality_score = self._evaluate_arrived_quality(goal_xy)

        # ⭐ 리워드 계산 (실제 결과만 사용)
        reward = (
            self.reward_info_gain * info_gain_total
            + self.reward_distance_penalty * distance_traveled
            + self.reward_quality_bonus * quality_score
        )
        
        self._last_reward = reward
        self._last_info_gain = info_gain_total
        self.step_count += 1

        # 다음 step을 위한 frontier 재탐지
        self._detect_frontiers()

        coverage = self._get_coverage()
        terminated = coverage >= self.coverage_done_thresh
        truncated = self.step_count >= self.max_steps

        obs = self._get_observation()
        info = {
            "info_gain": info_gain_total,
            "distance": distance_traveled,
            "quality": quality_score,
            "step": self.step_count,
            "coverage": coverage,
        }
        return obs, reward, terminated, truncated, info

    def _detect_frontiers(self):
        """Frontier 탐지 및 선택"""
        det_out = self.detector.detect(self.current_logodds, robot_xy=self.robot_xy)
        candidates = det_out.get("candidates", [])
        masks = det_out.get("masks", {})

        if len(candidates) == 0:
            self._frontiers = [None] * self.top_k
            self._last_frontiers = []
            return

        scored = self.selector.score_and_select(
            candidates=candidates,
            masks=masks,
            robot_xy=self.robot_xy,
            exploration_trace=None,
            do_merge=True,
            top_k=self.top_k,
        )
        self._frontiers = scored + [None] * (self.top_k - len(scored))
        self._last_frontiers = scored

    def _no_candidates(self) -> bool:
        return (len(self._frontiers) == 0) or all(f is None for f in self._frontiers)

    def _get_coverage(self) -> float:
        p = p_occ_from_logodds(self.current_logodds)
        free = (p <= self.free_thresh)
        return float(free.sum()) / float(p.size)

    def _get_observation(self) -> np.ndarray:
        """
        ⭐ 관측 가능한 특징만 사용 (센서로 측정 가능한 것)
        - Frontier 위치, 크기 (시각적으로 보임)
        - 현재 맵에서 보이는 주변 환경
        - 거리 및 방향
        """
        feature_dim = 15
        obs = np.zeros((self.top_k, feature_dim), dtype=np.float32)

        for i in range(self.top_k):
            f = self._frontiers[i] if i < len(self._frontiers) else None
            if f is None:
                continue

            fx, fy = f.center_xy
            dx = fx - self.robot_xy[0]
            dy = fy - self.robot_xy[1]
            dist = float(math.hypot(dx, dy))

            # 기본 특징 (관측 가능)
            obs[i, 0] = fx / 50.0
            obs[i, 1] = fy / 50.0
            obs[i, 2] = float(f.n) / 500.0  # frontier 크기
            obs[i, 3] = float(f.score) / 100.0
            obs[i, 4] = dx / 50.0
            obs[i, 5] = dy / 50.0
            obs[i, 6] = dist / 50.0

            # ⭐ 현재 맵에서 관측 가능한 주변 환경
            wall_proximity = self._get_wall_proximity(fx, fy)
            unknown_density = self._get_unknown_density(fx, fy)
            openness = self._get_openness(fx, fy)

            obs[i, 7] = wall_proximity
            obs[i, 8] = unknown_density
            obs[i, 9] = openness

            # 로봇 상태
            obs[i, 10] = self.robot_xy[0] / 50.0
            obs[i, 11] = self.robot_xy[1] / 50.0
            obs[i, 12] = float(math.cos(self.robot_yaw))
            obs[i, 13] = float(math.sin(self.robot_yaw))
            obs[i, 14] = float(self.step_count) / float(self.max_steps)

        return obs.flatten()

    def _get_wall_proximity(self, x: float, y: float) -> float:
        """벽 근접도 (현재 맵 기준, 0~1)"""
        H, W = self.current_logodds.shape
        iy, ix = _xy_to_ij(x, y, self.origin_xy, self.ogm_res)
        
        if not _in_bounds(iy, ix, H, W):
            return 1.0

        p = p_occ_from_logodds(self.current_logodds)
        radius = 5
        wall_count = 0
        total = 0

        for dy in range(-radius, radius + 1):
            for dx_val in range(-radius, radius + 1):
                ny, nx = iy + dy, ix + dx_val
                if _in_bounds(ny, nx, H, W):
                    if p[ny, nx] >= self.occ_thresh:
                        wall_count += 1
                    total += 1

        return float(wall_count / total) if total > 0 else 0.0

    def _get_unknown_density(self, x: float, y: float) -> float:
        """미지 영역 밀도 (현재 맵 기준, 0~1)"""
        H, W = self.current_logodds.shape
        iy, ix = _xy_to_ij(x, y, self.origin_xy, self.ogm_res)
        
        if not _in_bounds(iy, ix, H, W):
            return 0.0

        p = p_occ_from_logodds(self.current_logodds)
        radius = 8
        unknown_count = 0
        total = 0

        for dy in range(-radius, radius + 1):
            for dx_val in range(-radius, radius + 1):
                ny, nx = iy + dy, ix + dx_val
                if _in_bounds(ny, nx, H, W):
                    if self.free_thresh < p[ny, nx] < self.occ_thresh:
                        unknown_count += 1
                    total += 1

        return float(unknown_count / total) if total > 0 else 0.0

    def _get_openness(self, x: float, y: float) -> float:
        """개방성 (현재 맵 기준, 0~1)"""
        H, W = self.current_logodds.shape
        iy, ix = _xy_to_ij(x, y, self.origin_xy, self.ogm_res)
        
        if not _in_bounds(iy, ix, H, W):
            return 0.0

        p = p_occ_from_logodds(self.current_logodds)
        radius = 8
        free_count = 0
        total = 0

        for dy in range(-radius, radius + 1):
            for dx_val in range(-radius, radius + 1):
                ny, nx = iy + dy, ix + dx_val
                if _in_bounds(ny, nx, H, W):
                    if p[ny, nx] <= self.free_thresh:
                        free_count += 1
                    total += 1

        return float(free_count / total) if total > 0 else 0.0

    def _evaluate_arrived_quality(self, goal_xy: Tuple[float, float]) -> float:
        """
        도착 후 품질 평가 (센서로 측정)
        - Unknown 밀도 높음 → 좋음
        - 벽 근접 높음 → 나쁨
        - 개방성 높음 → 좋음
        """
        wall_prox = self._get_wall_proximity(*goal_xy)
        unknown_dens = self._get_unknown_density(*goal_xy)
        openness = self._get_openness(*goal_xy)

        # 점수 계산
        score = (
            unknown_dens * 3.0      # 미지 영역 많으면 좋음
            - wall_prox * 2.0       # 벽 가까우면 나쁨
            + openness * 1.5        # 열린 공간 좋음
        )
        return float(np.clip(score, -5.0, 5.0))

    # ============= Smooth Motion (완전 유지) =============
    def _follow_path_smooth(self, path: List[Tuple[float, float]]) -> Tuple[float, float]:
        """경로를 일정 속도로 따라가며 주기적으로 LiDAR 스캔"""
        if len(path) <= 1:
            ig = self._lidar_scan_and_update()
            return float(ig), 0.0

        ig_total = 0.0
        traveled = 0.0
        substep_count = 0

        cur_x, cur_y = self.robot_xy
        cur_idx = 0
        next_x, next_y = path[1]

        while True:
            dx = next_x - cur_x
            dy = next_y - cur_y
            seg_len = float(math.hypot(dx, dy))

            if seg_len < 1e-6:
                cur_x, cur_y = next_x, next_y
                cur_idx += 1
                if cur_idx >= len(path) - 1:
                    self.robot_xy = (cur_x, cur_y)
                    ig_total += self._lidar_scan_and_update()
                    break
                next_x, next_y = path[cur_idx + 1]
                continue

            step_len = min(self.robot_speed_mps * self.step_dt, seg_len)
            ux, uy = dx / seg_len, dy / seg_len
            cur_x += ux * step_len
            cur_y += uy * step_len
            traveled += step_len

            self.robot_yaw = math.atan2(uy, ux)
            self.robot_xy = (cur_x, cur_y)

            if (substep_count % self.lidar_scan_interval_steps) == 0:
                ig_total += self._lidar_scan_and_update()

            substep_count += 1

            if self.enable_visualization and self.viz_sleep_scale > 0.0:
                time.sleep(self.step_dt * self.viz_sleep_scale)

        return float(ig_total), float(traveled)

    # ============= LiDAR (완전 유지) =============
    def _lidar_scan_and_update(self) -> float:
        """풀 360° 스윕"""
        pose = (self.robot_xy[0], self.robot_xy[1], self.robot_yaw)
        H, W = self.current_logodds.shape
        free_set = set()

        for _ in range(self.scans_per_render):
            for i in range(RAY_COUNT):
                ang = (i / RAY_COUNT) * 2.0 * math.pi
                x, y, th = pose
                th_global = th + ang

                gx = x + self.lidar_max_range_m * math.cos(th_global)
                gy = y + self.lidar_max_range_m * math.sin(th_global)

                iy0, ix0 = _xy_to_ij(x, y, self.origin_xy, self.ogm_res)
                iy1, ix1 = _xy_to_ij(gx, gy, self.origin_xy, self.ogm_res)
                if not _in_bounds(iy0, ix0, H, W):
                    continue

                ray_cells = _bresenham(iy0, ix0, iy1, ix1)

                hit_index = None
                for k, (yy, xx) in enumerate(ray_cells):
                    if not _in_bounds(yy, xx, H, W):
                        hit_index = k
                        break
                    pcell = 1.0 / (1.0 + math.exp(-float(self.current_logodds[yy, xx])))
                    if pcell >= self.occ_thresh:
                        hit_index = k
                        break

                upto = len(ray_cells) if hit_index is None else max(0, hit_index)
                if upto > 0:
                    for (yy, xx) in ray_cells[:upto]:
                        free_set.add((yy, xx))

        with self._map_lock:
            before = self.current_logodds.copy()
            if free_set:
                ys, xs = zip(*free_set)
                self.current_logodds[ys, xs] += OGM_L_FREE
                np.clip(self.current_logodds, *OGM_CLAMP, out=self.current_logodds)

            p_before = 1.0 / (1.0 + np.exp(-before))
            p_after = 1.0 / (1.0 + np.exp(-self.current_logodds))
            became_free = (
                (p_before > self.free_thresh)
                & (p_before < self.occ_thresh)
                & (p_after <= self.free_thresh)
            )
            ig_total = float(became_free.sum())

            self.planner.update_map(self.current_logodds, self.origin_xy)

        return ig_total

    # ============= Visualization (유지) =============
    def _start_visualization(self):
        self._viz_running = True
        self._viz_thread = threading.Thread(
            target=self._visualization_loop, daemon=True
        )
        self._viz_thread.start()

    def _visualization_loop(self):
        plt.ion()
        self._viz_fig, self._viz_ax = plt.subplots(figsize=(12, 9))
        self._viz_ax.set_aspect('equal', 'box')
        self._viz_ax.set_title(
            'Frontier DQN (Pure RL)', fontsize=14, fontweight='bold'
        )

        while self._viz_running:
            try:
                self._render_frame()
                plt.pause(0.05)
            except Exception as e:
                time.sleep(0.1)

    def _render_frame(self):
        if self.current_logodds is None:
            return

        with self._map_lock:
            snapshot = self.current_logodds.copy()
            origin_xy = self.origin_xy
            robot_xy = self.robot_xy
            robot_yaw = self.robot_yaw
            last_frontiers = list(self._last_frontiers) if self._last_frontiers else []
            last_chosen = self._last_chosen_frontier_xy
            last_path = list(self._last_path_xy) if self._last_path_xy else []

        self._viz_ax.clear()

        H, W = snapshot.shape
        x0, y0 = origin_xy
        x_max = x0 + W * self.ogm_res
        y_max = y0 + H * self.ogm_res

        p = p_occ_from_logodds(snapshot)
        self._viz_ax.imshow(
            1.0 - p,
            cmap='gray',
            origin='lower',
            extent=[x0, x_max, y0, y_max],
            alpha=0.95,
            vmin=0,
            vmax=1,
        )

        # 로봇
        self._viz_ax.add_patch(
            Circle(robot_xy, radius=2.5, color='blue', alpha=0.95, zorder=10)
        )
        dx = 4.0 * math.cos(robot_yaw)
        dy = 4.0 * math.sin(robot_yaw)
        self._viz_ax.add_patch(
            FancyArrow(
                robot_xy[0],
                robot_xy[1],
                dx,
                dy,
                width=1.0,
                head_width=2.5,
                head_length=2.0,
                color='blue',
                alpha=0.95,
                zorder=11,
                linewidth=2,
            )
        )

        # Frontiers
        if last_frontiers:
            fx = [f.center_xy[0] for f in last_frontiers]
            fy = [f.center_xy[1] for f in last_frontiers]
            self._viz_ax.scatter(
                fx,
                fy,
                s=300,
                marker='*',
                color='yellow',
                alpha=0.7,
                zorder=5,
                edgecolors='black',
                linewidths=2,
            )

        if last_chosen:
            self._viz_ax.scatter(
                [last_chosen[0]],
                [last_chosen[1]],
                s=600,
                marker='*',
                color='red',
                edgecolors='black',
                linewidths=3,
                alpha=1.0,
                zorder=6,
            )

        # 경로
        if last_path and len(last_path) > 1:
            px = [p_[0] for p_ in last_path]
            py = [p_[1] for p_ in last_path]
            self._viz_ax.plot(px, py, 'r-', linewidth=2.5, alpha=0.9, zorder=4)

        # 정보
        unknown_cells = ((p > self.free_thresh) & (p < self.occ_thresh)).sum()
        free_cells = (p <= self.free_thresh).sum()
        info_text = (
            f"Map: {self.current_map_name}\n"
            f"Episode: {self._episode_count_on_current_map}/{self._total_episode_count}\n"
            f"Step: {self.step_count} | Action: {self._last_action}\n"
            f"Reward: {self._last_reward:.2f} | InfoGain: {self._last_info_gain:.0f}\n"
            f"Coverage: {self._get_coverage():.1%} | Frontiers: {len([f for f in self._frontiers if f is not None])}\n"
            f"Unknown: {unknown_cells:,} | Free: {free_cells:,}"
        )
        self._viz_ax.text(
            0.02,
            0.98,
            info_text,
            transform=self._viz_ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            family='monospace',
            bbox=dict(
                boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8
            ),
        )

        # 뷰 모드
        if self.view_mode == "fit":
            x1, x2, y1, y2 = x0, x_max, y0, y_max
            if self.follow_margin_m > 0:
                self._viz_ax.set_xlim(x1 - self.follow_margin_m,
                                      x2 + self.follow_margin_m)
                self._viz_ax.set_ylim(y1 - self.follow_margin_m,
                                      y2 + self.follow_margin_m)
            else:
                self._viz_ax.set_xlim(x1, x2)
                self._viz_ax.set_ylim(y1, y2)
        else:  # follow
            self._viz_ax.set_xlim(
                robot_xy[0] - self.follow_window_m,
                robot_xy[0] + self.follow_window_m,
            )
            self._viz_ax.set_ylim(
                robot_xy[1] - self.follow_window_m,
                robot_xy[1] + self.follow_window_m,
            )

        self._viz_ax.set_aspect('equal', 'box')
        self._viz_ax.grid(True, alpha=0.2, linestyle=':')
        self._viz_fig.canvas.draw()

    def close(self):
        self._viz_running = False
        if self._viz_thread:
            self._viz_thread.join(timeout=1.0)
        try:
            plt.close(self._viz_fig)
        except Exception:
            pass
        plt.close('all')