import numpy as np
from PIL import Image

def load_occupancy_png(path: str, thresh: int = 128, invert: bool = False):
    """
    흑/백(또는 그레이) PNG를 occupancy grid로 변환.
    - 흑(0)   = 벽(1)
    - 백(255) = 자유(0)
    """
    img = Image.open(path).convert("L")
    arr = np.array(img)
    if invert:
        arr = 255 - arr
    occ = (arr < thresh).astype(np.uint8)  # 1=벽, 0=자유
    return occ

def inflate_obstacles(occ: np.ndarray, pixels: int) -> np.ndarray:
    """로봇 반경만큼 벽 팽창(충돌 여유). 사각 커널로 간단 팽창."""
    if pixels <= 0:
        return occ.copy()
    h, w = occ.shape
    infl = occ.copy()
    pad = pixels
    padded = np.pad(occ, ((pad, pad), (pad, pad)), mode="edge")
    for y in range(h):
        y0 = y
        ys = y0
        for x in range(w):
            xs = x
            window = padded[ys:ys+2*pad+1, xs:xs+2*pad+1]
            infl[y, x] = 1 if window.max() == 1 else infl[y, x]
    return infl

def world_to_map(x: float, y: float, meters_per_pixel: float):
    """월드(m) → 맵(픽셀). (0,0)은 맵 좌상단이 아닌 '맵 중앙' 기준으로 설정."""
    return int(x / meters_per_pixel), int(y / meters_per_pixel)

def map_to_world(i: int, j: int, meters_per_pixel: float):
    """맵(픽셀) → 월드(m)."""
    return i * meters_per_pixel, j * meters_per_pixel

def clamp_pose_to_map(x, y, occ, mpp):
    """맵 경계 밖으로 나가지 않도록 좌표 클램프."""
    H, W = occ.shape
    x = np.clip(x, 0.0, (W - 1) * mpp)
    y = np.clip(y, 0.0, (H - 1) * mpp)
    return x, y

def random_free_pose(occ: np.ndarray, meters_per_pixel: float, rng: np.random.Generator):
    """자유 셀에서 임의 시작 위치 뽑기(연속좌표, 임의 yaw)."""
    H, W = occ.shape
    while True:
        i = rng.integers(0, W)
        j = rng.integers(0, H)
        if occ[j, i] == 0:
            x, y = map_to_world(i + 0.5, j + 0.5, meters_per_pixel)
            th = rng.uniform(-np.pi, np.pi)
            return x, y, th

def dda_raycast(occ: np.ndarray, x0: float, y0: float, th: float,
                max_range: float, meters_per_pixel: float):
    """
    DDA(Grid Traversal) 레이캐스트: (x0,y0)에서 각 th 방향으로 max_range까지 진행.
    - 충돌 시: 충돌 지점까지 거리 반환
    - 충돌 없으면: max_range 반환
    """
    H, W = occ.shape
    # 시작 셀
    sx = x0 / meters_per_pixel
    sy = y0 / meters_per_pixel
    i = int(sx)
    j = int(sy)

    # 레이 방향
    dx = np.cos(th)
    dy = np.sin(th)

    # 셀 경계까지의 거리 계산 준비
    step_i = 1 if dx > 0 else -1
    step_j = 1 if dy > 0 else -1

    # 다음 수직/수평 경계까지 거리
    def frac(a): return a - np.floor(a)
    if dx > 0:
        tMaxX = (1.0 - frac(sx)) / (dx if dx != 0 else 1e-9)
    else:
        tMaxX = (frac(sx)) / (-dx if dx != 0 else 1e-9)
    if dy > 0:
        tMaxY = (1.0 - frac(sy)) / (dy if dy != 0 else 1e-9)
    else:
        tMaxY = (frac(sy)) / (-dy if dy != 0 else 1e-9)

    tDeltaX = abs(1.0 / (dx if dx != 0 else 1e-9))
    tDeltaY = abs(1.0 / (dy if dy != 0 else 1e-9))

    # 최대 스텝 수(안전)
    max_steps = int((max_range / meters_per_pixel) * 2) + 5

    # 시작점 충돌 체크
    if 0 <= i < W and 0 <= j < H and occ[j, i] == 1:
        return 0.0

    t = 0.0
    for _ in range(max_steps):
        if t * meters_per_pixel > max_range:
            break
        # 더 가까운 경계로 이동
        if tMaxX < tMaxY:
            i += step_i
            t = tMaxX
            tMaxX += tDeltaX
        else:
            j += step_j
            t = tMaxY
            tMaxY += tDeltaY

        if not (0 <= i < W and 0 <= j < H):
            # 맵 밖 → 현재까지의 거리
            return min(max_range, t * meters_per_pixel)
        if occ[j, i] == 1:
            # 벽 충돌
            return min(max_range, t * meters_per_pixel)

    return max_range

def cast_lidar(occ: np.ndarray, x: float, y: float, th: float,
               beams: int, max_range: float, meters_per_pixel: float):
    """360° LiDAR 스캔(정균등 분해). 거리 배열 반환[m]."""
    angles = th + np.linspace(-np.pi, np.pi, beams, endpoint=False)
    dists = np.empty(beams, dtype=np.float32)
    for k, a in enumerate(angles):
        d = dda_raycast(occ, x, y, a, max_range, meters_per_pixel)
        dists[k] = d
    return dists

def mark_visibility(visited: np.ndarray, occ: np.ndarray,
                    x: float, y: float, th: float,
                    beams: int, max_range: float, mpp: float) -> int:
    """
    현재 포즈에서 LiDAR 가시 경로(자유 셀)와 히트(벽 셀)를 방문 처리.
    새로 방문된 셀 수를 반환(보상 계산용).
    """
    H, W = occ.shape
    newly = 0
    angles = th + np.linspace(-np.pi, np.pi, beams, endpoint=False)
    sx = int(x / mpp)
    sy = int(y / mpp)

    for a in angles:
        # DDA로 충돌/범위 한계까지 전진하면서 경로 셀 마킹
        dx = np.cos(a)
        dy = np.sin(a)
        step_i = 1 if dx > 0 else -1
        step_j = 1 if dy > 0 else -1

        # 시작 셀
        i = sx
        j = sy
        if 0 <= i < W and 0 <= j < H and visited[j, i] == 0:
            visited[j, i] = 1
            newly += 1

        # DDA 파라미터
        def frac(z): return z - np.floor(z)
        fx, fy = x / mpp, y / mpp
        if dx > 0:
            tMaxX = (1.0 - frac(fx)) / (dx if dx != 0 else 1e-9)
        else:
            tMaxX = (frac(fx)) / (-dx if dx != 0 else 1e-9)
        if dy > 0:
            tMaxY = (1.0 - frac(fy)) / (dy if dy != 0 else 1e-9)
        else:
            tMaxY = (frac(fy)) / (-dy if dy != 0 else 1e-9)

        tDeltaX = abs(1.0 / (dx if dx != 0 else 1e-9))
        tDeltaY = abs(1.0 / (dy if dy != 0 else 1e-9))

        max_cells = int(max_range / mpp) + 2
        for _ in range(max_cells):
            if tMaxX < tMaxY:
                i += step_i
                tMaxX += tDeltaX
            else:
                j += step_j
                tMaxY += tDeltaY

            if not (0 <= i < W and 0 <= j < H):
                break

            # 벽 만나면 그 셀만 마킹하고 중단
            if occ[j, i] == 1:
                if visited[j, i] == 0:
                    visited[j, i] = 1
                    newly += 1
                break
            else:
                if visited[j, i] == 0:
                    visited[j, i] = 1
                    newly += 1
    return newly
