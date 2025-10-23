# frontier_rl_agent.py
import os
import numpy as np
from typing import List, Tuple, Any, Optional
from stable_baselines3 import DQN

# 고정 크기 (학습/운영 동일해야 함)
K = 8   # Top-K 후보 수 (액션 수)
D = 12  # 후보 1개 피처 차원

class FrontierRLAgent:
    """
    운영 시 사용:
      - predict_bonus_for_frontiers(): 프런티어 리스트만 넘기면 내부에서
        obs[K,D] + mask[K]를 구성하고, 학습된 DQN으로 보너스 벡터를 반환.
      - predict_bonus()/pick_action(): 낮은 단계 API (obs,mask 직접 전달용)
    """
    def __init__(self, model_path: str = "models/frontier_dqn.zip", device: str = "cpu"):
        self.model: Optional[DQN] = None
        if os.path.exists(model_path):
            self.model = DQN.load(model_path, device=device)

    # ---------- 상태 ----------
    def is_ready(self) -> bool:
        return self.model is not None

    # ---------- 상위 레벨 API: 프런티어 리스트만 입력 ----------
    def predict_bonus_for_frontiers(
        self,
        *,
        logodds: np.ndarray,             # (H,W) float32 log-odds grid
        origin_xy: Tuple[float, float],  # (x0, y0) world origin of grid
        res_m: float,                    # grid resolution (m/cell)
        planner: Any,                    # planner.plan_path(start_xy, goal_xy, ...)
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        frontiers: List[Any],            # ScoredFrontier 리스트(점수화/병합 완료)
        top_k: int = K,
        safety_inflate_m: float = 1.6,
        allow_diagonal: bool = True,
        p_mean_sample_cap: int = 30,
        band: int = 1,
        vmax: float = 0.8,
    ) -> np.ndarray:
        """
        반환: bonuses np.ndarray [len(frontiers)]
              - 상위 K에 대해서만 모델 보너스(불가/패딩은 -1e3 포함) 계산
              - K 밖 후보는 0 보너스로 확장
        """
        N = len(frontiers)
        if N == 0:
            return np.zeros((0,), dtype=np.float32)

        K_eff = min(int(top_k), N)
        cands_k = frontiers[:K_eff]

        # obs[K,D], mask[K] 구성
        obs = np.zeros((top_k, D), dtype=np.float32)
        mask = np.ones((top_k,), dtype=np.uint8)  # 1=불가/패딩, 0=유효

        for i, f in enumerate(cands_k):
            # 경로 유무로 마스크 설정
            path_xy = planner.plan_path(
                start_xy=tuple(robot_xy),
                goal_xy=tuple(f.center_xy),
                safety_inflate_m=safety_inflate_m,
                allow_diagonal=allow_diagonal,
            )
            if path_xy:
                mask[i] = 0

            obs[i] = self._build_feature_12(
                logodds=logodds,
                origin_xy=origin_xy,
                res_m=res_m,
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                cand=f,
                path_xy=path_xy or [],
                p_mean_sample_cap=p_mean_sample_cap,
                band=band,
                vmax=vmax,
            )

        if not self.is_ready():
            # 모델이 없으면 보너스 0으로 처리
            bonuses = np.zeros((N,), dtype=np.float32)
            return bonuses

        # 상위 K에 대한 보너스(선택된 액션=1.0, 불가/패딩=-1e3)
        bonus_topk = self.predict_bonus(obs, mask)  # [top_k]

        # 전체 길이 N으로 확장 (K 밖은 0)
        bonuses = np.zeros((N,), dtype=np.float32)
        bonuses[:K_eff] = bonus_topk[:K_eff]
        return bonuses

    # ---------- 저수준 API ----------
    def pick_action(self, obs_2d: np.ndarray, mask_1d: np.ndarray) -> int:
        """
        obs_2d: [K, D] float32
        mask_1d: [K] uint8 (1=불가/패딩, 0=유효)
        """
        valid = np.where(mask_1d == 0)[0]
        if len(valid) == 0:
            return 0
        if self.model is None:
            return int(valid[0])

        flat = obs_2d.reshape(K * D)
        a = int(self.model.predict(flat, deterministic=True)[0])
        # 마스킹 위반 시 fallback
        if mask_1d[a] == 0:
            return a
        return int(valid[0])

    def predict_bonus(self, obs_2d: np.ndarray, mask_1d: np.ndarray) -> np.ndarray:
        """
        간단 보너스: 선택된 액션에만 1.0, 나머지 0(불가/패딩은 -1e3).
        (Q-벡터 기반 연속 보너스를 쓰려면 커스텀 정책으로 Q를 노출하거나 밴딧을 고려)
        """
        K_local = obs_2d.shape[0]
        bonus = np.zeros((K_local,), dtype=np.float32)
        a = self.pick_action(obs_2d, mask_1d)
        if 0 <= a < K_local and mask_1d[a] == 0:
            bonus[a] = 1.0
        # 불가/패딩은 큰 음수로 내려서 재랭킹에서 탈락시키기
        if mask_1d.ndim == 1:
            bonus[mask_1d.astype(bool)] = -1e3
        else:
            bonus[mask_1d.reshape(-1).astype(bool)] = -1e3
        return bonus

    # ---------- 내부: D=12 피처 생성 ----------
    def _build_feature_12(
        self,
        *,
        logodds: np.ndarray,                 # (H,W)
        origin_xy: Tuple[float, float],      # (x0,y0)
        res_m: float,
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        cand: Any,                           # ScoredFrontier (center_xy, n, info_gain, candidate.open_score 등)
        path_xy: List[Tuple[float, float]],  # [(x,y), ...]
        p_mean_sample_cap: int = 30,
        band: int = 1,
        vmax: float = 0.8,
    ) -> np.ndarray:
        """
        D=12 스키마(학습/운영 동일):
        [0] L, [1] T=L/vmax, [2] cos(dθ), [3] sin(dθ), [4] dx, [5] dy,
        [6] info_gain, [7] n, [8] open_score, [9] p_mean, [10] trace_mean(0), [11] coverage(0)
        """
        # 1) 경로 길이/시간
        L = float(len(path_xy)) * float(res_m)
        T = (L / vmax) if L > 0 else 0.0

        # 2) 기하/자세
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        cx, cy = float(cand.center_xy[0]), float(cand.center_xy[1])
        dx = cx - rx
        dy = cy - ry
        heading = float(np.arctan2(dy, dx))
        dth = float(np.arctan2(np.sin(heading - robot_yaw), np.cos(heading - robot_yaw)))
        cos_d, sin_d = float(np.cos(dth)), float(np.sin(dth))

        # 3) 후보 속성
        info_gain = float(getattr(cand, "info_gain", 0.0))
        size      = float(getattr(cand, "n", 0.0))
        # ScoredFrontier.candidate.open_score 또는 cand.open_score 중 존재하는 값 사용
        open_s    = float(getattr(getattr(cand, "candidate", cand), "open_score", 0.0))

        # 4) 경로 밴드 평균 점유확률 p_mean
        p_list = []
        if path_xy:
            step = max(1, len(path_xy)//max(1, p_mean_sample_cap))
            H, W = logodds.shape[:2]
            x0, y0 = origin_xy
            for (px, py) in path_xy[::step]:
                iy = int(np.floor((py - y0) / res_m))
                ix = int(np.floor((px - x0) / res_m))
                for by in range(-band, band + 1):
                    for bx in range(-band, band + 1):
                        yy, xx = iy + by, ix + bx
                        if 0 <= yy < H and 0 <= xx < W:
                            lo = float(logodds[yy, xx])
                            p_list.append(1.0 / (1.0 + np.exp(-lo)))
        p_mean = float(np.mean(p_list)) if p_list else 0.5

        # trace_mean/coverage는 운영에서 주기 어렵다면 0 고정 (학습도 동일 스키마로)
        trace_mean = 0.0
        coverage   = 0.0

        feat = np.array(
            [L, T, cos_d, sin_d, dx, dy, info_gain, size, open_s, p_mean, trace_mean, coverage],
            dtype=np.float32
        )
        if feat.shape[0] != D:
            raise ValueError(f"Feature dim mismatch: {feat.shape[0]} != {D}")
        return feat
