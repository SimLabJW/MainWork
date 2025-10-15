import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from envs.explore_env import ExploreLidarEnv
from envs.lidar_utils import cast_lidar

def main(map_path: str, model_path: str):
    env = ExploreLidarEnv(map_path=map_path, seed=0)
    model = DQN.load(model_path)

    obs, _ = env.reset()
    done = False
    trunc = False

    xs, ys = [], []

    plt.ion()
    fig, ax = plt.subplots()
    H, W = env.occ.shape
    # 맵 표시: 벽=검정, 자유=흰색
    ax.imshow(env.occ, cmap="gray_r", origin="upper")
    robot_plot, = ax.plot([], [], "go", ms=4)     # 로봇 위치
    path_plot,  = ax.plot([], [], "g-", lw=1)     # 로봇 경로
    scan_plot   = ax.scatter([], [], s=1)         # 스캔 포인트
    ax.set_title("Evaluation (green path, black walls)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # 이미지 좌표계 반전
    plt.tight_layout()

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)

        # 로봇/경로 업데이트
        xs.append(env.x / env.mpp)
        ys.append(env.y / env.mpp)

        # LiDAR 스캔 포인트 표시(충돌 셀에 dot)
        dists = cast_lidar(env.occ, env.x, env.y, env.th,
                           env.beams, env.lidar_max_range, env.mpp)
        angles = env.th + np.linspace(-np.pi, np.pi, env.beams, endpoint=False)
        hit_x = (env.x + dists * np.cos(angles)) / env.mpp
        hit_y = (env.y + dists * np.sin(angles)) / env.mpp

        robot_plot.set_data([xs[-1]], [ys[-1]])
        path_plot.set_data(xs, ys)
        scan_plot.remove()
        scan_plot = ax.scatter(hit_x, hit_y, s=1)

        plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--model", type=str, default="dqn_lidar_explore.zip")
    args = parser.parse_args()
    main(args.map, args.model)
