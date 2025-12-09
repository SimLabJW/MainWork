"""
train_frontier_v3.py
강화학습 기반 Frontier 선택 학습 - 단일 환경 최적화 버전

변경 사항:
1. 병렬 환경 제거 (SubprocVecEnv → DummyVecEnv)
2. 단일 환경에서만 학습 수행
3. 학습 스텝 10,000으로 축소
4. 평가 주기 단축 (1,000)
5. reset 관련 문제 해결
"""
import os
import json
import gzip
import base64
import io
import numpy as np
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.frontier_dqn_env_v3 import FrontierDQNEnv

from stable_baselines3.common.logger import configure

def load_exploration_maps(json_path: str = "exploration_env.json"):
    """
    exploration_env.json에서 맵 로드
    Free(0) → Unknown(-1) 변환 및 좌표 추적
    """
    import math

    def _infer_hw(n_bytes: int) -> tuple[int, int]:
        best = None
        root = int(math.sqrt(n_bytes))
        for h in range(1, root + 1):
            if n_bytes % h == 0:
                w = n_bytes // h
                score = abs(w - h)
                if (best is None) or (score < best[0]):
                    best = (score, (h, w))
        return best[1] if best else (n_bytes, 1)

    def _decode_one(name: str, md: dict):
        gz_b64 = md["data_gzip_b64"]
        gz_bytes = base64.b64decode(gz_b64)
        with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes), mode="rb") as gz:
            raw = gz.read()
        n = len(raw)

        H = md.get("height")
        W = md.get("width")
        if (H is None) or (W is None) or (int(H) * int(W) != n):
            H, W = _infer_hw(n)
            print(f"ℹ️ [{name}] shape inferred → (H,W)=({H},{W}) from {n} bytes")

        arr_i8 = np.frombuffer(raw, dtype=np.int8).reshape((int(H), int(W))).copy()

        # Free(0) → Unknown(-1) 변환
        converted_coords = []
        for i in range(int(H)):
            for j in range(int(W)):
                if arr_i8[i, j] == 0:
                    arr_i8[i, j] = -1
                    converted_coords.append((i, j))

        print(f"✅ [{name}] {len(converted_coords)} Free cells converted to Unknown")

        # log-odds 변환
        logodds = np.zeros((int(H), int(W)), dtype=np.float32)
        logodds[arr_i8 == 0] = -2.0
        logodds[arr_i8 == 100] = +2.0
        logodds[arr_i8 == -1] = 0.0

        origin = md.get("origin", {"x": 0.0, "y": 0.0})
        res = float(md.get("resolution", 0.1))

        return {
            "logodds": logodds,
            "origin_xy": (float(origin.get("x", 0.0)), float(origin.get("y", 0.0))),
            "res": res,
            "converted_coords": converted_coords,
        }

    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    maps = {}
    if isinstance(root, dict) and "data_gzip_b64" not in root:
        for k, v in root.items():
            if isinstance(v, dict) and "data_gzip_b64" in v:
                maps[k] = _decode_one(k, v)
        if maps:
            return maps

    raise ValueError("지원하지 않는 exploration_env.json 구조입니다.")


def make_env(maps, seed=0, enable_visualization = False):
    """단일 환경 생성 함수"""
    any_map = next(iter(maps.values()))
    env = FrontierDQNEnv(
        maps=maps,
        lidar_max_range_m=40.0,
        ogm_res=any_map["res"],
        occ_thresh=0.65,
        free_thresh=0.35,
        # === 속도 최적화 설정 ===
        max_steps=70,
        top_k_frontiers=5,
        episodes_per_map=50,
        # === 리워드 설정 ===
        reward_info_gain=10.0,
        reward_distance_penalty=-0.1,
        reward_invalid=-5.0,
        # === 로봇 속도 설정 ===
        robot_speed_mps=6.0,
        step_dt=0.7,
        lidar_scan_interval_steps=5,
        # === 시각화 비활성화 ===
        enable_visualization=enable_visualization,
        seed=seed,
    )
    env = Monitor(env)
    return env


def main():
    # ========== 설정 ==========
    TOTAL_TIMESTEPS = 10_000   # ✅ 단일 학습 1만 스텝

    # ========== 1. 맵 로드 ==========
    print("📂 Loading exploration maps...")
    maps = load_exploration_maps("exploration_env.json")
    print(f"✅ Loaded {len(maps)} maps: {list(maps.keys())}")

    # ========== 2. 단일 환경 생성 ==========
    print("\n🏗️ Creating single environment...")
    env = DummyVecEnv([lambda: make_env(maps, seed=0, enable_visualization=True)])
    eval_env = DummyVecEnv([lambda: make_env(maps, seed=1, enable_visualization=False)])
    print("✅ Environment ready (single mode)")

    # ========== 3. DQN 모델 생성 ==========
    print("\n🧠 Creating DQN model...")

    log_dir = "./logs/tensorboard_v2/"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=128,
        tau=0.01,
        gamma=0.95,
        train_freq=4,
        gradient_steps=2,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
    )
    print("✅ DQN model created")
    model.set_logger(new_logger)

    # ========== 4. 콜백 ==========
    checkpoint_dir = "./logs/checkpoints_v2/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000,
        save_path=checkpoint_dir,
        name_prefix="frontier_dqn_single",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model_v2/",
        log_path="./logs/eval_v2/",
        eval_freq=1_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    # ========== 5. 학습 ==========
    print(f"\n🚀 Starting single-environment training for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        log_interval=100,
        progress_bar=True,
    )

    # ========== 6. 최종 저장 ==========
    final_model_path = "./logs/final_model_v2/frontier_dqn_single_final"
    os.makedirs("./logs/final_model_v2/", exist_ok=True)
    model.save(final_model_path)
    print(f"\n✅ Training complete! Model saved to {final_model_path}.zip")

    # ========== 7. 평가 ==========
    print("\n📊 Evaluating final model...")
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=5, deterministic=True
    )
    print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    eval_env.close()
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
