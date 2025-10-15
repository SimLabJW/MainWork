import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from envs.explore_env import ExploreLidarEnv

# === 콘솔 하트비트 콜백 ===
class HeartbeatCallback(BaseCallback):
    def __init__(self, print_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self.ep_count = 0
        self.last_cov = 0.0
        self.last_ep_rew = None
        self.last_ep_len = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            cov = infos[-1].get("coverage", None)
            if cov is not None:
                self.last_cov = float(cov)
            for info in infos:
                if "episode" in info:
                    self.ep_count += 1
                    self.last_ep_rew = float(info["episode"]["r"])
                    self.last_ep_len = int(info["episode"]["l"])
        if self.num_timesteps % self.print_every == 0:
            msg = f"[HB] step={self.num_timesteps:,}  episodes={self.ep_count:,}  coverage={self.last_cov:.0f}"
            if self.last_ep_rew is not None:
                msg += f"  last_ep_rew={self.last_ep_rew:.3f}  last_ep_len={self.last_ep_len}"
            print(msg)
        return True

def make_env(map_path: str):
    def _thunk():
        return Monitor(ExploreLidarEnv(
            map_path=map_path,
            meters_per_pixel=0.10,
            beams=360,
            lidar_max_range=40.0,
            max_steps=2000,
            forward_step=0.20,
            turn_step_rad=0.261799,  # 15도
            robot_radius_m=0.25,
            time_penalty=0.01,
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
        verbose=1,
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=5_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
        tensorboard_log="./logs/dqn_explore/"
    )

    callbacks = CallbackList([
        HeartbeatCallback(print_every=5000),
    ])

    model.learn(total_timesteps=300_000, callback=callbacks)
    model.save("dqn_lidar_explore")
    print("Saved model to dqn_lidar_explore.zip")
