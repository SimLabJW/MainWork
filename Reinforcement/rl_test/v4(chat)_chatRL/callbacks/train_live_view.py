# callbacks/train_live_view.py
import numpy as np

# 백엔드 먼저 강제(윈도우에서 창 안뜨는 문제 방지)
import matplotlib
matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from envs.lidar_utils import cast_lidar
from envs.explore_env import ExploreLidarEnv

def _unwrap_to_env(maybe_wrapper):
    """
    DummyVecEnv -> Monitor -> ExploreLidarEnv 까지 벗겨서 실제 env를 얻는다.
    (DummyVecEnv 하나만 쓰는 전제)
    """
    env = maybe_wrapper
    # DummyVecEnv: .envs[0]
    if hasattr(env, "envs"):
        env = env.envs[0]
    # Monitor: .env
    if hasattr(env, "env"):
        env = env.env
    # 혹시 한 번 더 래핑되어 있으면 반복
    if hasattr(env, "env"):
        inner = env.env
        if isinstance(inner, ExploreLidarEnv):
            env = inner
    return env

class TrainingLivePlotCallback(BaseCallback):
    """
    학습에 사용 중인 '훈련 env' 상태를 매 스텝 읽어와 실시간 플롯.
    - 추가 rollout 없음 (학습 느려짐 최소화)
    - DummyVecEnv(n=1) 전제
    """
    def __init__(self, update_every: int = 1, title: str = "Live Training (green path, black walls)", verbose: int = 0):
        super().__init__(verbose)
        self.update_every = update_every
        self.title = title

        self.env_ref = None
        self.fig = None
        self.ax = None
        self.robot_plot = None
        self.path_plot = None
        self.scan_plot = None

        self.xs = []
        self.ys = []

    def _setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        H, W = self.env_ref.occ.shape
        self.ax.imshow(self.env_ref.occ, cmap="gray_r", origin="upper")
        (self.robot_plot,) = self.ax.plot([], [], "go", ms=4)
        (self.path_plot,) = self.ax.plot([], [], "g-", lw=1)
        self.scan_plot = self.ax.scatter([], [], s=1)
        self.ax.set_title(self.title)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        plt.tight_layout()
        plt.show(block=False)

    def _on_step(self) -> bool:
        # 최초 접근 시 훈련 env 포인터 확보
        if self.env_ref is None:
            self.env_ref = _unwrap_to_env(self.training_env)
            if not isinstance(self.env_ref, ExploreLidarEnv):
                print("[LivePlot] Could not unwrap ExploreLidarEnv; live view disabled.")
                return True

        # 첫 스텝에 바로 창 띄우기
        if self.fig is None:
            self._setup_plot()

        # 업데이트 간격 제어
        if (self.num_timesteps % self.update_every) != 0:
            return True

        env = self.env_ref
        # 현재 로봇 위치 기록(픽셀좌표로 변환)
        self.xs.append(env.x / env.mpp)
        self.ys.append(env.y / env.mpp)

        # 현재 포즈 기준 LiDAR 히트 포인트 계산
        dists = cast_lidar(env.occ, env.x, env.y, env.th, env.beams, env.lidar_max_range, env.mpp)
        angles = env.th + np.linspace(-np.pi, np.pi, env.beams, endpoint=False)
        hit_x = (env.x + dists * np.cos(angles)) / env.mpp
        hit_y = (env.y + dists * np.sin(angles)) / env.mpp

        # 플롯 업데이트
        self.robot_plot.set_data([self.xs[-1]], [self.ys[-1]])
        self.path_plot.set_data(self.xs, self.ys)
        self.scan_plot.remove()
        self.scan_plot = self.ax.scatter(hit_x, hit_y, s=1)

        plt.pause(0.001)
        if self.fig and self.fig.canvas:
            self.fig.canvas.flush_events()

        return True
