# frontier_rl_agent.py
import os
import math
import numpy as np
from typing import List, Tuple, Any, Optional
from stable_baselines3 import DQN


class FrontierRLAgent:
    """
    학습된 DQN 모델을 사용하여 프런티어 선택
    (train_frontier_v3.py + frontier_dqn_env_v3.py 환경과 동일한 입력 구조)
    """

    def __init__(self, model_path: str = "./rl/models/best_model.zip", device: str = "cpu"):
        self.model: Optional[DQN] = None
        self.top_k = 5          # frontier_dqn_env_v3.py의 top_k_frontiers
        self.feature_dim = 15   # frontier_dqn_env_v3.py의 obs_dim

        if os.path.exists(model_path):
            try:
                self.model = DQN.load(model_path, device=device)
                print(f"✅ RL model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load RL model: {e}")
        else:
            print(f"⚠️ Model not found at {model_path}, using heuristic only")

    def is_ready(self) -> bool:
        return self.model is not None

    # ---------------------------------------------------------------------
    # RL 기반 프런티어 선택
    # ---------------------------------------------------------------------
    def select_frontier(
        self,
        *,
        logodds: np.ndarray,
        origin_xy: Tuple[float, float],
        res_m: float,
        planner: Any,
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        frontiers: List[Any],
    ) -> int:
        """RL 모델이 프런티어 후보 중 하나를 선택"""
        if not self.is_ready() or not frontiers:
            return 0

        obs = self._build_observation(
            logodds=logodds,
            origin_xy=origin_xy,
            res_m=res_m,
            planner=planner,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            frontiers=frontiers[: self.top_k],
        )

        try:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
            if 0 <= action < len(frontiers):
                return action
            else:
                print(f"⚠️ Invalid action {action}, using 0")
                return 0
        except Exception as e:
            print(f"⚠️ RL prediction failed: {e}")
            return 0

    def evaluate_frontiers(self, logodds, origin_xy, res_m, planner, robot_xy, robot_yaw, frontiers):
        """
        각 프런티어에 대해 RL 모델이 예측한 점수를 반환합니다.
        항상 관측 차원(75,)을 유지하기 위해 padding 적용.
        """
        if not self.is_ready() or not frontiers:
            return [0.0] * len(frontiers)

        try:
            q_values = []
            for f in frontiers[: self.top_k]:
                obs = self._build_observation(
                    logodds=logodds,
                    origin_xy=origin_xy,
                    res_m=res_m,
                    planner=planner,
                    robot_xy=robot_xy,
                    robot_yaw=robot_yaw,
                    frontiers=[f],
                )

                # --- 패딩 (길이 75 고정) ---
                if obs.shape[0] < self.top_k * self.feature_dim:
                    pad_len = (self.top_k * self.feature_dim) - obs.shape[0]
                    obs = np.concatenate([obs, np.zeros(pad_len, dtype=np.float32)])

                obs = obs.reshape(1, -1)  # (1, 75) 형태로 맞추기

                # --- DQN 예측 ---
                action, _ = self.model.predict(obs, deterministic=True)
                q_values.append(float(action))

            # 프런티어 수보다 top_k가 클 경우 0으로 패딩
            while len(q_values) < len(frontiers):
                q_values.append(0.0)

            return q_values

        except Exception as e:
            print(f"⚠️ evaluate_frontiers() failed: {e}")
            return [0.0] * len(frontiers)




    # ---------------------------------------------------------------------
    # Observation 구성 (학습 환경과 동일하게)
    # ---------------------------------------------------------------------
    def _build_observation(
        self,
        *,
        logodds: np.ndarray,
        origin_xy: Tuple[float, float],
        res_m: float,
        planner: Any,
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        frontiers: List[Any],
    ) -> np.ndarray:
        """
        학습 시와 동일한 관측 상태 벡터 생성
        [frontier_feat_0, frontier_feat_1, ..., yaw_sin, yaw_cos]
        """
        obs = np.zeros((self.top_k, self.feature_dim), dtype=np.float32)
        num_frontiers = min(len(frontiers), self.top_k)

        for i in range(num_frontiers):
            obs[i, :] = self._encode_frontier_feature(
                logodds=logodds,
                origin_xy=origin_xy,
                res_m=res_m,
                planner=planner,
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                frontier=frontiers[i],
            )

        return obs.flatten()

    # ---------------------------------------------------------------------
    # frontier 피처 구성 (train 시 _get_obs() 구조 복제)
    # ---------------------------------------------------------------------
    def _encode_frontier_feature(
        self,
        *,
        logodds: np.ndarray,
        origin_xy: Tuple[float, float],
        res_m: float,
        planner: Any,
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        frontier: Any,
    ) -> np.ndarray:
        """학습 환경과 동일한 15차원 피처"""
        fx, fy = frontier.center_xy
        dx, dy = fx - robot_xy[0], fy - robot_xy[1]
        dist = math.hypot(dx, dy)

        feat = np.zeros(self.feature_dim, dtype=np.float32)

        # ① 프런티어 중심 좌표 (정규화)
        feat[0] = fx / 50.0
        feat[1] = fy / 50.0

        # ② 프런티어 크기 (픽셀 수)
        feat[2] = float(getattr(frontier, "n", 0)) / 500.0

        # ③ 휴리스틱 점수 (정규화)
        feat[3] = float(getattr(frontier, "score", 0)) / 1000.0

        # ④ 로봇과의 상대 거리 및 방향 (정규화)
        feat[4] = dx / 50.0
        feat[5] = dy / 50.0
        feat[6] = dist / 50.0

        # ⑤ 환경 통계 (벽 근접, unknown 비율, free 비율)
        feat[7] = self._get_wall_proximity(fx, fy, logodds, origin_xy, res_m)
        feat[8] = self._get_unknown_density(fx, fy, logodds, origin_xy, res_m)
        feat[9] = self._get_free_ratio(fx, fy, logodds, origin_xy, res_m)

        # ⑥ 로봇 상태 (정규화 좌표 + yaw sin/cos)
        feat[10] = robot_xy[0] / 50.0
        feat[11] = robot_xy[1] / 50.0
        feat[12] = math.sin(robot_yaw)
        feat[13] = math.cos(robot_yaw)

        # ⑦ frontier 크기 대비 거리 비 (효율성)
        feat[14] = (getattr(frontier, "n", 1)) / max(dist, 1.0)

        return feat

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _world_to_grid(self, x: float, y: float, origin_xy: Tuple[float, float], res_m: float):
        x0, y0 = origin_xy
        ix = int((x - x0) / res_m)
        iy = int((y - y0) / res_m)
        return iy, ix

    def _in_bounds(self, iy: int, ix: int, H: int, W: int):
        return 0 <= iy < H and 0 <= ix < W

    def _p_occ_from_logodds(self, logodds: np.ndarray):
        return 1.0 / (1.0 + np.exp(-logodds))

    def _get_wall_proximity(self, x, y, logodds, origin_xy, res_m):
        H, W = logodds.shape
        iy, ix = self._world_to_grid(x, y, origin_xy, res_m)
        if not self._in_bounds(iy, ix, H, W):
            return 0.0
        p = self._p_occ_from_logodds(logodds)
        occ_thresh = 0.65
        radius = 5
        occ = 0
        total = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = iy + dy, ix + dx
                if self._in_bounds(ny, nx, H, W):
                    if p[ny, nx] >= occ_thresh:
                        occ += 1
                    total += 1
        return occ / total if total > 0 else 0.0

    def _get_unknown_density(self, x, y, logodds, origin_xy, res_m):
        H, W = logodds.shape
        iy, ix = self._world_to_grid(x, y, origin_xy, res_m)
        if not self._in_bounds(iy, ix, H, W):
            return 0.0
        p = self._p_occ_from_logodds(logodds)
        free_t, occ_t = 0.35, 0.65
        radius = 8
        unknown = 0
        total = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = iy + dy, ix + dx
                if self._in_bounds(ny, nx, H, W):
                    if free_t < p[ny, nx] < occ_t:
                        unknown += 1
                    total += 1
        return unknown / total if total > 0 else 0.0

    def _get_free_ratio(self, x, y, logodds, origin_xy, res_m):
        H, W = logodds.shape
        iy, ix = self._world_to_grid(x, y, origin_xy, res_m)
        if not self._in_bounds(iy, ix, H, W):
            return 0.0
        p = self._p_occ_from_logodds(logodds)
        free_t = 0.35
        radius = 8
        free = 0
        total = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = iy + dy, ix + dx
                if self._in_bounds(ny, nx, H, W):
                    if p[ny, nx] <= free_t:
                        free += 1
                    total += 1
        return free / total if total > 0 else 0.0
