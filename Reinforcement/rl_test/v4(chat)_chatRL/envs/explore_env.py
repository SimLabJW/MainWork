import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


from .lidar_utils import (
    load_occupancy_png, inflate_obstacles, random_free_pose,
    cast_lidar, mark_visibility, clamp_pose_to_map
)

class ExploreLidarEnv(gym.Env):
    """
    2D LiDAR(360°, 40m) 부착 로봇이 PNG 맵에서 미지영역을 탐색.
    - 관측: LiDAR 거리 360개 (0~1 정규화)
    - 액션: 0=전진, 1=좌회전, 2=우회전, 3=정지
    - 보상: 가시영역 증가 +, 시간/정지 패널티 -, 충돌 큰 패널티
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_path: str,
        meters_per_pixel: float = 0.10,
        beams: int = 360,
        lidar_max_range: float = 40.0,
        max_steps: int = 2000,
        forward_step: float = 0.20,     # 한 스텝 전진 거리[m]
        turn_step_rad: float = np.deg2rad(15),  # 회전 스텝[rad]
        robot_radius_m: float = 0.25,   # 충돌 여유 반경
        time_penalty: float = 0.01,
        stop_penalty: float = 0.01,
        coverage_gain: float = 0.001,
        collision_penalty: float = 1.0,
        seed: Optional[int] = None,

    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.mpp = meters_per_pixel
        self.beams = beams
        self.lidar_max_range = lidar_max_range
        self.max_steps = max_steps
        self.forward_step = forward_step
        self.turn_step = turn_step_rad
        self.robot_radius_px = max(1, int(round(robot_radius_m / self.mpp)))
        self.time_penalty = time_penalty
        self.stop_penalty = stop_penalty
        self.coverage_gain = coverage_gain
        self.collision_penalty = collision_penalty

        # 맵 로드 및 충돌 여유 팽창
        base_occ = load_occupancy_png(map_path)
        self.occ = inflate_obstacles(base_occ, self.robot_radius_px)
        self.H, self.W = self.occ.shape

        # 관측/행동 공간
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.beams,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # F, L, R, Stop

        # 상태
        self.x = self.y = self.th = 0.0
        self.step_count = 0
        self.visited = np.zeros_like(self.occ, dtype=np.uint8)

    def seed(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _sample_free_pose(self):
        # 자유셀에서 시작. 확률적으로 막힌 곳을 피하기 위해 여러 번 시도
        for _ in range(500):
            x, y, th = random_free_pose(self.occ, self.mpp, self.rng)
            if not self._is_collision(x, y):
                return x, y, th
        # 실패 시 강제 배치
        return random_free_pose(self.occ, self.mpp, self.rng)

    def _is_collision(self, x: float, y: float) -> bool:
        i = int(x / self.mpp)
        j = int(y / self.mpp)
        if not (0 <= i < self.W and 0 <= j < self.H):
            return True
        return self.occ[j, i] == 1

    def _observe(self):
        dists = cast_lidar(self.occ, self.x, self.y, self.th,
                           self.beams, self.lidar_max_range, self.mpp)
        # 0~1 정규화
        obs = (dists / self.lidar_max_range).astype(np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.step_count = 0
        self.visited[:] = 0
        self.x, self.y, self.th = self._sample_free_pose()
        self.x, self.y = clamp_pose_to_map(self.x, self.y, self.occ, self.mpp)

        # 리셋 시에도 현재 시야 방문 처리(초기 커버리지)
        mark_visibility(self.visited, self.occ, self.x, self.y, self.th,
                        self.beams, self.lidar_max_range, self.mpp)
        obs = self._observe()
        return obs, {}

    def step(self, action: int):
        self.step_count += 1

        # === 동역학 ===
        if action == 0:      # forward
            nx = self.x + self.forward_step * np.cos(self.th)
            ny = self.y + self.forward_step * np.sin(self.th)
            nth = self.th
        elif action == 1:    # turn left
            nx, ny, nth = self.x, self.y, self.th + self.turn_step
        elif action == 2:    # turn right
            nx, ny, nth = self.x, self.y, self.th - self.turn_step
        elif action == 3:    # stop
            nx, ny, nth = self.x, self.y, self.th
        else:
            raise ValueError("invalid action")

        # 맵 경계/충돌 체크
        nx, ny = clamp_pose_to_map(nx, ny, self.occ, self.mpp)
        collision = self._is_collision(nx, ny)

        # 상태 갱신 (충돌이어도 일단 포즈 갱신 후 보상에 반영)
        self.x, self.y, self.th = nx, ny, nth

        # 관측/가시영역 갱신
        newly = mark_visibility(self.visited, self.occ, self.x, self.y, self.th,
                                self.beams, self.lidar_max_range, self.mpp)
        obs = self._observe()

        # === 보상 ===
        reward = 0.0
        reward += self.coverage_gain * float(newly)   # 새로 본 셀 보상
        reward -= self.time_penalty                   # 시간 패널티
        if action == 3:
            reward -= self.stop_penalty               # 정지 억제
        if collision:
            reward -= self.collision_penalty          # 충돌 큰 패널티

        # === 종료/트렁케이션 ===
        terminated = bool(collision)                  # 충돌 시 에피소드 종료
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "coverage": float(self.visited.sum()),
            "step": self.step_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        # 필요 시 evaluate.py에서 구현된 시각화 사용
        pass
