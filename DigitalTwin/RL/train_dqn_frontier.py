# train_dqn_frontier.py
import os
import torch
from stable_baselines3 import DQN
from envs.frontier_dqn_env import FrontierReplayEnv

def main():
    # === 경로 ===
    log_path = "logs/frontier_dqn.jsonl"   # SLAM이 생성한 JSONL
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # === 디바이스 자동 선택 (cuda 가능하면 cuda) ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device={device}")

    # === 오프라인 재생 Env ===
    # - normalize=True: 데이터 기준 z-score 정규화
    # - shuffle_each_reset=True: 매 epoch(=reset)마다 샘플 순서 섞기
    env = FrontierReplayEnv(
        log_path=log_path,
        normalize=True,
        shuffle_each_reset=True,
        seed=42,
    )

    # === DQN 설정 ===
    # - policy_kwargs: MLP 크기 늘림(조정 가능)
    # - optimize_memory_usage: 메모리 사용 최적화
    # - exploration_*: 오프라인 재생에서도 약간의 탐험(무해)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.05,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        optimize_memory_usage=False,          # ← 끔
        verbose=1,
        tensorboard_log="./tb_frontier/",
        device=device,
    )


    # === 학습 ===
    model.learn(total_timesteps=200_000, log_interval=10)

    # === 저장 ===
    out_path = os.path.join(save_dir, "frontier_dqn.zip")
    model.save(out_path)
    print("✅ Saved:", out_path)

if __name__ == "__main__":
    main()
