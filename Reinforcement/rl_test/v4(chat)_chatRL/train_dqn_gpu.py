import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import CallbackList
from callbacks.train_live_view import TrainingLivePlotCallback

from envs.explore_env import ExploreLidarEnv

def make_env(map_path: str):
    def _thunk():
        return Monitor(ExploreLidarEnv(
            map_path=map_path,
            meters_per_pixel=0.10,
            beams=360,
            lidar_max_range=30.0,
            max_steps=1500,
            forward_step=0.4,
            turn_step_rad= np.deg2rad(15),
            robot_radius_m=0.25,
            time_penalty=0.008,
            stop_penalty=0.01,
            coverage_gain=0.001,
            collision_penalty=1.0,
            seed=42
        ))
    return _thunk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, required=True, help="PNG occupancy map path")
    args = parser.parse_args()

    env = DummyVecEnv([make_env(args.map)])

    model = DQN(
        "MlpPolicy",
        env,
        device="cuda",
        verbose=1,
        learning_rate=1e-4,
        buffer_size=400_000,      # (↑) 리플레이 더 크게
        learning_starts=5_000,    # (↑) 조금 늦게 학습 시작
        batch_size=256,           # (↑) 한 번에 더 큰 배치 → GPU 효율↑
        train_freq=16,            # (↑) 환경 16스텝 모아서
        gradient_steps=4,         # (↑) 매 업데이트에 4회 역전파
        gamma=0.99,
        target_update_interval=10_000,  # (↑) 타깃 업데이트 주기 늘림
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
        tensorboard_log="./logs/dqn_explore/"
    )

    callbacks = CallbackList([
        TrainingLivePlotCallback(update_every=1),  # 처음부터, 매 스텝 업데이트
    ])

    model.learn(total_timesteps=300_000, callback=callbacks, log_interval=1)
    # model.learn(total_timesteps=300_000)
    model.save("dqn_lidar_explore")
    print("Saved model to dqn_lidar_explore.zip")
