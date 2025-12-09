# rl_data_logger.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import numpy as np


# =========================
# 설정/파라미터 컨테이너
# =========================
@dataclass
class RLDataConfig:
    top_k: int = 8          # 후보 최대 개수(K)
    feat_dim: int = 12      # 피처 차원(D)
    vmax: float = 0.8       # 예상 속도(m/s)
    safety_inflate_m: float = 1.6
    allow_diagonal: bool = True
    band: int = 1           # 경로 밴드 폭(안전성 추정)
    p_mean_sample_cap: int = 30  # 경로 샘플링 최대 개수(성능 제한)
    log_path: str = "logs/frontier_dqn_log.jsonl"


# =========================
# 유틸
# =========================
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# =========================
# 피처 빌더
# =========================
class FeatureBuilder:
    """
    프런티어 후보 1개에 대한 피처(D차원) 생성기.
    """
    def __init__(self, cfg: RLDataConfig):
        self.cfg = cfg

    def build_features(
        self,
        grid_logodds: np.ndarray,
        planner: Any,
        grid_origin_world: Tuple[float, float],
        ogm_res: float,
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        cand: Any,
        path_xy: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        반환: shape=(D,) float32
        """
        # 1) 기하/동역학
        L = float(len(path_xy)) * float(ogm_res)
        T = (L / self.cfg.vmax) if L > 0 else 0.0

        dx = float(cand.center_xy[0] - robot_xy[0])
        dy = float(cand.center_xy[1] - robot_xy[1])
        heading = float(np.arctan2(dy, dx))
        dth = float(np.arctan2(np.sin(heading - robot_yaw), np.cos(heading - robot_yaw)))
        cos_d, sin_d = float(np.cos(dth)), float(np.sin(dth))

        # 2) 후보 속성
        info_gain = float(getattr(cand, "info_gain", 0.0))
        size      = float(getattr(cand, "n", 0.0))
        open_s    = float(getattr(cand, "open_score", 0.0))

        # 3) 경로 밴드 평균 점유확률(안전성 지표)
        p_list = []
        step = max(1, len(path_xy)//self.cfg.p_mean_sample_cap or 1)
        for (px, py) in path_xy[::step] if path_xy else []:
            iy = int(np.floor((py - grid_origin_world[1]) / ogm_res))
            ix = int(np.floor((px - grid_origin_world[0]) / ogm_res))
            for by in range(-self.cfg.band, self.cfg.band + 1):
                for bx in range(-self.cfg.band, self.cfg.band + 1):
                    yy, xx = iy + by, ix + bx
                    if 0 <= yy < grid_logodds.shape[0] and 0 <= xx < grid_logodds.shape[1]:
                        p_list.append(_sigmoid(float(grid_logodds[yy, xx])))
        p_mean = float(np.mean(p_list)) if p_list else 0.5

        # 4) trace 평균(선택: planner.trace_ref가 있을 때)
        trace_mean = 0.0
        if hasattr(planner, "trace_ref"):
            tr = planner.trace_ref
            vals = []
            pix = getattr(cand, "pixel_inds", getattr(getattr(cand, "candidate", None), "pixel_inds", [])) or []
            for (iy, ix) in pix:
                if 0 <= iy < tr.shape[0] and 0 <= ix < tr.shape[1]:
                    vals.append(float(tr[iy, ix]))
            trace_mean = float(np.mean(vals)) if vals else 0.0

        # coverage = float(getattr(planner, "coverage_ratio", 0.0))
        cov = getattr(planner, "coverage_ratio", 0.0)
        if callable(cov): cov = cov()
        coverage = float(cov)

        feat = np.array(
            [L, T, cos_d, sin_d, dx, dy, info_gain, size, open_s, p_mean, trace_mean, coverage],
            dtype=np.float32,
        )
        assert feat.shape[0] == self.cfg.feat_dim, f"Feature dim mismatch: got {feat.shape[0]}, expect {self.cfg.feat_dim}"
        return feat


# =========================
# 보상 계산기
# =========================
class RewardCalculator:
    """
    도착/갱신 후 보상 계산(미지영역 감소 효율 중심).
    """
    def __init__(self, cfg: RLDataConfig):
        self.cfg = cfg

    def compute(
        self,
        before_unknown_ratio: float,
        after_unknown_ratio: float,
        path_len_m: float,
        success: bool,
        replan: bool,
    ) -> float:
        delta = max(0.0, float(before_unknown_ratio) - float(after_unknown_ratio))
        eff   = (delta / max(0.2, float(path_len_m))) if path_len_m > 0 else delta
        r = 5.0 * eff
        if not success:
            r -= 2.0
        if replan:
            r -= 1.0
        r -= 0.03 * float(path_len_m)
        return float(r)


# =========================
# 관측/마스크 빌더
# =========================
class ObsMaskBuilder:
    """
    후보 리스트 → [K,D] obs + [K] mask(1=불가/패딩)
    """
    def __init__(self, cfg: RLDataConfig, feature_builder: FeatureBuilder):
        self.cfg = cfg
        self.fb = feature_builder

    def build(
        self,
        grid_logodds: np.ndarray,
        planner: Any,
        grid_origin_world: Tuple[float, float],
        ogm_res: float,
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        candidates: List[Any],
        plan_path_fn: Any,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[float, float]]]]:
        feats, masks, paths = [], [], []
        valid = candidates[: self.cfg.top_k]
        for c in valid:
            path = plan_path_fn(
                start_xy=robot_xy,
                goal_xy=tuple(c.center_xy),
                safety_inflate_m=self.cfg.safety_inflate_m,
                allow_diagonal=self.cfg.allow_diagonal,
            )
            paths.append(path)
            masks.append(0 if path else 1)
            feats.append(
                self.fb.build_features(
                    grid_logodds, planner, grid_origin_world, ogm_res,
                    robot_xy, robot_yaw, c, path or [],
                )
            )

        # 패딩
        while len(feats) < self.cfg.top_k:
            feats.append(np.zeros((self.cfg.feat_dim,), np.float32))
            masks.append(1)
            paths.append([])

        obs_2d = np.stack(feats, axis=0).astype(np.float32)      # [K,D]
        mask = np.asarray(masks, dtype=np.uint8)                  # [K]
        return obs_2d, mask, paths


# =========================
# JSONL 로거
# =========================
class FrontierDQNLogger:
    """
    (obs, action) → (reward, next_obs, done) 트랜지션을 JSONL로 저장.
    """
    def __init__(self, cfg: RLDataConfig):
        self.cfg = cfg
        os.makedirs(os.path.dirname(self.cfg.log_path) or ".", exist_ok=True)
        self._pending: Optional[Tuple[List[float], int]] = None

    @staticmethod
    def pack_obs(obs_2d: np.ndarray, expected_len: int) -> List[float]:
        flat = np.asarray(obs_2d, dtype=np.float32).reshape(-1)
        assert flat.size == expected_len, f"obs size mismatch: {flat.size} != {expected_len}"
        return flat.tolist()

    def start_step(self, obs_2d: np.ndarray, action: int) -> None:
        """
        선택 직후 호출: 이번 스텝의 (관측, 행동) 임시 저장.
        """
        self._pending = (self.pack_obs(obs_2d, self.cfg.top_k * self.cfg.feat_dim), int(action))

    def finish_step(
        self,
        reward: float,
        next_obs_2d: Optional[np.ndarray] = None,
        done: bool = False,
    ) -> None:
        """
        도착/갱신 후 호출: 보상/다음 관측 기록.
        """
        if self._pending is None:
            return
        rec = {
            "obs": self._pending[0],
            "action": self._pending[1],
            "reward": float(reward),
            "next_obs": (
                self.pack_obs(next_obs_2d, self.cfg.top_k * self.cfg.feat_dim)
                if next_obs_2d is not None else None
            ),
            "done": bool(done),
        }
        with open(self.cfg.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._pending = None


# =========================
# 통합 파사드 (선택사항)
# =========================
class RLDataModule:
    """
    한 번에 쓰라고 만든 통합 파사드:
      - 관측/마스크 생성
      - 로깅 시작/종료
      - 보상 계산
    """
    def __init__(self, cfg: RLDataConfig):
        self.cfg = cfg
        self.feature_builder = FeatureBuilder(cfg)
        self.obs_builder = ObsMaskBuilder(cfg, self.feature_builder)
        self.reward_calc = RewardCalculator(cfg)
        self.logger = FrontierDQNLogger(cfg)

    # 관측/마스크
    def make_obs_and_mask(
        self,
        grid_logodds: np.ndarray,
        planner: Any,
        grid_origin_world: Tuple[float, float],
        ogm_res: float,
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        candidates: List[Any],
        plan_path_fn: Any,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[float, float]]]]:
        return self.obs_builder.build(
            grid_logodds, planner, grid_origin_world, ogm_res,
            robot_xy, robot_yaw, candidates, plan_path_fn
        )

    # 보상
    def compute_reward(
        self,
        before_unknown_ratio: float,
        after_unknown_ratio: float,
        path_len_m: float,
        success: bool,
        replan: bool,
    ) -> float:
        return self.reward_calc.compute(
            before_unknown_ratio, after_unknown_ratio, path_len_m, success, replan
        )

    # 로깅
    def start_step(self, obs_2d: np.ndarray, action: int) -> None:
        self.logger.start_step(obs_2d, action)

    def finish_step(
        self, reward: float, next_obs_2d: Optional[np.ndarray] = None, done: bool = False
    ) -> None:
        self.logger.finish_step(reward, next_obs_2d, done)
