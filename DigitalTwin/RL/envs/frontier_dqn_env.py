# envs/frontier_dqn_env.py
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Any, Optional

# 로그 설계 상 고정
K = 8
D = 12
OBS_DIM = K * D  # 96

class FrontierReplayEnv(gym.Env):
    """
    JSONL transition 로그를 재생하는 오프라인 학습용 Env.
    - 관측: [K*D] 연속값
    - 행동: Discrete(K)
    - done: 로그 끝 또는 레코드의 done=True
    옵션:
      - normalize: z-score 정규화(로그 전체 평균/표준편차)
      - shuffle_each_reset: reset마다 샘플 순서를 섞어 순환
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        log_path: str,
        normalize: bool = True,
        shuffle_each_reset: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._rng = np.random.RandomState(seed) if seed is not None else np.random
        self.normalize = bool(normalize)
        self.shuffle_each_reset = bool(shuffle_each_reset)

        # === 데이터 로드 ===
        self.log: List[Dict[str, Any]] = self._load(log_path)
        self.N = len(self.log)

        # === 관측/액션 공간 ===
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(K)

        # === 정규화 통계(옵션) ===
        if self.normalize:
            obs_stack = []
            for r in self.log:
                o = np.asarray(r["obs"], dtype=np.float32).reshape(-1)
                if o.size != OBS_DIM:
                    o = self._fix_dim(o)
                obs_stack.append(o)
                if r.get("next_obs") is not None:
                    no = np.asarray(r["next_obs"], dtype=np.float32).reshape(-1)
                    if no.size != OBS_DIM:
                        no = self._fix_dim(no)
                    obs_stack.append(no)
            obs_stack = np.stack(obs_stack, axis=0)
            self._mu = obs_stack.mean(axis=0)
            self._std = obs_stack.std(axis=0)
            # 분산 0 방지
            self._std[self._std < 1e-6] = 1.0
        else:
            self._mu = np.zeros((OBS_DIM,), dtype=np.float32)
            self._std = np.ones((OBS_DIM,), dtype=np.float32)

        # === 인덱스(셔플용) ===
        self._order = np.arange(self.N, dtype=np.int64)
        self.ptr = 0

    # ---------- 필수 Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.shuffle_each_reset:
            self._rng.shuffle(self._order)
        self.ptr = 0
        first = self.log[self._order[self.ptr]]
        obs = self._get_obs(first.get("obs"))
        return obs, {}

    def step(self, action: int):
        idx = self._order[self.ptr]
        rec = self.log[idx]

        # 보상/종료 읽기
        reward = float(rec.get("reward", 0.0))
        done_flag = bool(rec.get("done", False))

        # next_obs: 없으면 자기 자신 obs로 대체
        next_obs_raw = rec.get("next_obs", rec.get("obs"))
        next_obs = self._get_obs(next_obs_raw)

        # 포인터 이동
        self.ptr += 1
        if self.ptr >= self.N:
            done_flag = True  # 에피소드 종료

        # Gymnasium step 반환
        info = {"a_logged": int(rec.get("action", -1)), "idx": int(idx)}
        # terminated, truncated 분리 — 여기서는 오프라인이므로 모두 terminated로 처리
        return next_obs, reward, done_flag, False, info

    # ---------- 내부 유틸 ----------
    def _load(self, path: str) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                # 최소 필드 검사
                if "obs" not in j or "action" not in j:
                    continue
                # obs/next_obs 차원 보정(패딩/자르기)
                j["obs"] = self._fix_dim(np.asarray(j["obs"], dtype=np.float32))
                if j.get("next_obs") is not None:
                    j["next_obs"] = self._fix_dim(np.asarray(j["next_obs"], dtype=np.float32))
                data.append(j)
        if not data:
            raise RuntimeError(f"No transitions found in JSONL: {path}")
        return data

    def _fix_dim(self, arr: np.ndarray) -> np.ndarray:
        """입력 길이를 OBS_DIM으로 강제(길면 자르고, 짧으면 0 패딩)."""
        flat = arr.reshape(-1)
        if flat.size == OBS_DIM:
            return flat.astype(np.float32)
        out = np.zeros((OBS_DIM,), dtype=np.float32)
        n = min(OBS_DIM, flat.size)
        out[:n] = flat[:n]
        return out

    def _get_obs(self, raw: Any) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size != OBS_DIM:
            arr = self._fix_dim(arr)
        # z-score 정규화
        if self.normalize:
            arr = (arr - self._mu) / self._std
        return arr.astype(np.float32)
