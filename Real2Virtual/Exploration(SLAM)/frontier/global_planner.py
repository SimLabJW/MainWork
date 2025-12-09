# frontier/global_planner.py (초슬림 + A* 경로계획 + 방향성 브리지)
import heapq
import numpy as np
from typing import Tuple, Optional, List


class GlobalPlanner:
    def __init__(
        self,
        ogm_res_m: float,
        occ_thresh: float = 0.65,
        free_thresh: float = 0.35,
        coverage_done_thresh: float = 0.95,
        unknown_left_thresh: float = 0.02,
        no_frontier_patience: int = 10,
        allow_unknown: bool = True,
        unknown_cost_factor: float = 1.25,
    ):
        self.res = ogm_res_m
        self.occ_thresh = occ_thresh
        self.free_thresh = free_thresh
        self.coverage_done_thresh = coverage_done_thresh
        self.unknown_left_thresh = unknown_left_thresh
        self.no_frontier_patience = no_frontier_patience

        self.allow_unknown = bool(allow_unknown)
        self.unknown_cost_factor = float(unknown_cost_factor)

        self._p: Optional[np.ndarray] = None
        self._origin_xy: Tuple[float, float] = (0.0, 0.0)
        self._coverage: float = 0.0
        self._unknown_ratio: float = 1.0
        self.no_frontier_count: int = 0
        self.is_done: bool = False

        # 추가 지표
        self._boundary_unknown_ratio: float = 1.0
        self._exploration_success_ratio: float = 0.0

    # ---------- 기본 이진 연산 유틸 ----------
    def _dilate8_(self, mask: np.ndarray, r: int = 1) -> np.ndarray:
        out = mask.copy()
        H, W = mask.shape
        for _ in range(r):
            tmp = out.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy0 = max(0, -dy)
                    yy1 = min(H, H - dy)
                    xx0 = max(0, -dx)
                    xx1 = min(W, W - dx)
                    tmp[yy0:yy1, xx0:xx1] |= out[yy0 + dy : yy1 + dy, xx0 + dx : xx1 + dx]
            out = tmp
        return out

    def _erode8_(self, mask: np.ndarray, r: int = 1) -> np.ndarray:
        out = mask.copy()
        H, W = mask.shape
        for _ in range(r):
            tmp = out.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy0 = max(0, -dy)
                    yy1 = min(H, H - dy)
                    xx0 = max(0, -dx)
                    xx1 = min(W, W - dx)
                    tmp[yy0:yy1, xx0:xx1] &= out[yy0 + dy : yy1 + dy, xx0 + dx : xx1 + dx]
            out = tmp
        return out

    def _seal_walls(self, occ: np.ndarray, r: int = 2) -> np.ndarray:
        # closing(팽창 후 침식)으로 미세틈 봉인
        return self._erode8_(self._dilate8_(occ, r=r), r=r)

    # ---------- 방향성 브리지(문틈 직선으로 메우기) ----------
    def _close_gaps_directional(self, occ: np.ndarray, max_gap_px: int) -> np.ndarray:
        """
        선형 구조요소(0,45,90,135도)로 '닫힘(closing)'을 적용해
        벽 사이의 가는 틈을 방향성 있게 메운다.
        - occ: True=벽(점유)
        - max_gap_px: 메울 최대 틈 폭(픽셀)
        """
        H, W = occ.shape
        out = occ.copy()

        def _line_dilate(mask: np.ndarray, r: int, dir8: tuple) -> np.ndarray:
            dy, dx = dir8
            res = mask.copy()
            for k in range(1, r + 1):
                # 정방향
                yy0 = max(0, -dy * k)
                yy1 = min(H, H - dy * k)
                xx0 = max(0, -dx * k)
                xx1 = min(W, W - dx * k)
                res[yy0:yy1, xx0:xx1] |= mask[yy0 + dy * k : yy1 + dy * k, xx0 + dx * k : xx1 + dx * k]
                # 역방향
                yy0 = max(0, dy * k)
                yy1 = min(H, H + dy * k)
                xx0 = max(0, dx * k)
                xx1 = min(W, W + dx * k)
                res[yy0:yy1, xx0:xx1] |= mask[yy0 - dy * k : yy1 - dy * k, xx0 - dx * k : xx1 - dx * k]
            return res

        def _line_close(mask: np.ndarray, r: int, dir8: tuple) -> np.ndarray:
            dil = _line_dilate(mask, r, dir8)
            er = dil.copy()
            dy, dx = dir8
            for k in range(1, r + 1):
                # 정방향 침식
                yy0 = max(0, -dy * k)
                yy1 = min(H, H - dy * k)
                xx0 = max(0, -dx * k)
                xx1 = min(W, W - dx * k)
                er[yy0:yy1, xx0:xx1] &= dil[yy0 + dy * k : yy1 + dy * k, xx0 + dx * k : xx1 + dx * k]
                # 역방향 침식
                yy0 = max(0, dy * k)
                yy1 = min(H, H + dy * k)
                xx0 = max(0, dx * k)
                xx1 = min(W, W + dx * k)
                er[yy0:yy1, xx0:xx1] &= dil[yy0 - dy * k : yy1 - dy * k, xx0 - dx * k : xx1 - dx * k]
            return er

        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for d in dirs:
            out = _line_close(out, max_gap_px, d)
        return out

    # ---------- 도달 가능(open) 영역 확장 ----------
    def _reachable_open_from(self, occ: np.ndarray, seed_iy: int, seed_ix: int) -> np.ndarray:
        H, W = occ.shape
        open_ = ~occ  # 벽만 막고 free/unknown 모두 통과
        if not (0 <= seed_iy < H and 0 <= seed_ix < W) or (not open_[seed_iy, seed_ix]):
            return np.zeros_like(open_, dtype=bool)

        vis = np.zeros_like(open_, dtype=bool)
        stack = [(seed_iy, seed_ix)]
        vis[seed_iy, seed_ix] = True
        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while stack:
            y, x = stack.pop()
            for dy, dx in neigh:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and (not vis[yy, xx]) and open_[yy, xx]:
                    vis[yy, xx] = True
                    stack.append((yy, xx))
        return vis

    # ---------- 동적 지표(실내 기준, 바깥 무시) 갱신 ----------
    def update_dynamic_metrics(self, robot_xy):
        """
        원하는 방식:
        1) 현재 OCC(벽)를 '방향성 브리지 + closing'으로 길게 연장해 가짜 구조 확보
        2) 로봇 위치를 시드로, '벽이 아닌 곳(~occ_sealed)'을 flood-fill → 실내(interior)
        3) 실내에서만 free/unknown 비율 계산:
           exploration_success = free_inside / (free_inside + unknown_inside)
           boundary_unknown_ratio = unknown_inside / (free_inside + unknown_inside)
        바깥(outside) 연결 여부는 고려하지 않음.
        """
        if self._p is None or robot_xy is None:
            return

        free = self._p <= self.free_thresh
        occ = self._p >= self.occ_thresh
        unk = ~(free | occ)

        # (A) 벽을 '방향성'으로 연장해 틈 메우기 (기본 0.6 m 권장)
        gap_px = max(1, int(round(0.60 / self.res)))
        occ_bridged = self._close_gaps_directional(occ, gap_px)

        # (B) 남은 미세 틈은 8-이웃 closing으로 봉인(강도 2~3 권장)
        occ_sealed = self._seal_walls(occ_bridged, r=2)

        # (C) 로봇 seed로 '벽이 아닌 곳(~occ_sealed)'을 확장 → 실내
        ry, rx = self._xy_to_ij(*robot_xy)
        interior = self._reachable_open_from(occ_sealed, ry, rx)

        # (D) 실내에서 free/unknown만 유효로 보고 비율 계산
        interior_fu = interior & (free | unk)
        denom = float(interior_fu.sum())

        if denom <= 0.0:
            # 실내가 없거나 전부 OCC로 잡힌 특수 케이스
            # 정책: 탐사 성공 0.0, 경계 unknown 1.0 (필요시 NaN 처리로 바꿀 수 있음)
            self._boundary_unknown_ratio = 1.0
            self._exploration_success_ratio = 0.0
            return

        free_inside = float((interior & free).sum())
        unknown_inside = float((interior & unk).sum())

        self._boundary_unknown_ratio = unknown_inside / denom
        self._exploration_success_ratio = free_inside / denom

    # ---------- 게터 ----------
    def boundary_unknown_ratio(self) -> float:
        return float(getattr(self, "_boundary_unknown_ratio", 1.0))

    def exploration_success_ratio(self) -> float:
        return float(getattr(self, "_exploration_success_ratio", 0.0))

    # ---------- 맵 통계 ----------
    def update_map(self, logodds: np.ndarray, origin_xy: Tuple[float, float]) -> None:
        self._p = 1.0 / (1.0 + np.exp(-logodds))
        self._origin_xy = origin_xy

        free = self._p <= self.free_thresh
        occ = self._p >= self.occ_thresh
        unk = ~(occ | free)

        total = float(self._p.size) + 1e-9
        self._coverage = float(free.sum()) / total
        self._unknown_ratio = float(unk.sum()) / total

        if (self._coverage >= self.coverage_done_thresh) or (self._unknown_ratio <= self.unknown_left_thresh):
            self.is_done = True

    def coverage(self) -> float:
        return float(self._coverage)

    def unknown_ratio(self) -> float:
        return float(self._unknown_ratio)

    def notify_frontier_presence(self, frontier_exists: bool, path_exists: bool) -> bool:
        if self.is_done:
            return True
        if not frontier_exists:
            self.is_done = True
            return True
        if not path_exists:
            self.is_done = True
            return True
        return False

    # ---------- 좌표 변환 ----------
    def _xy_to_ij(self, x: float, y: float) -> Tuple[int, int]:
        x0, y0 = self._origin_xy
        ix = int(np.floor((x - x0) / self.res))
        iy = int(np.floor((y - y0) / self.res))
        return iy, ix

    def _ij_to_xy(self, iy: int, ix: int) -> Tuple[float, float]:
        x0, y0 = self._origin_xy
        x = x0 + (ix + 0.5) * self.res
        y = y0 + (iy + 0.5) * self.res
        return (x, y)

    # ---------- A* 경로계획 ----------
    def plan_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        safety_inflate_m: float = 0.8,
        allow_diagonal: bool = True,
        max_nodes: int = 50000,
    ) -> List[Tuple[float, float]]:
        if self._p is None:
            return []

        H, W = self._p.shape
        occ = self._p >= self.occ_thresh
        free = self._p <= self.free_thresh
        unk = ~(occ | free)

        # 1) 차단 마스크: 장애물만 확실히 차단
        blocked = occ.copy()

        # 2) 안전 마진 팽창(장애물만 대상으로)
        inflate_px = max(0, int(round(safety_inflate_m / self.res)))
        if inflate_px > 0:
            infl = blocked.copy()
            for dy in range(-inflate_px, inflate_px + 1):
                for dx in range(-inflate_px, inflate_px + 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy0 = max(0, -dy)
                    yy1 = min(H, H - dy)
                    xx0 = max(0, -dx)
                    xx1 = min(W, W - dx)
                    infl[yy0:yy1, xx0:xx1] |= blocked[yy0 + dy : yy1 + dy, xx0 + dx : xx1 + dx]
            blocked = infl

        # 3) 시작/목표 인덱스
        sy, sx = self._xy_to_ij(*start_xy)
        gy, gx = self._xy_to_ij(*goal_xy)
        if not (0 <= sy < H and 0 <= sx < W and 0 <= gy < H and 0 <= gx < W):
            return []

        # 4) 시작/목표가 막혔으면 주변 unblocked로 스냅
        def _nearest_unblocked(y, x, max_r=10):
            if 0 <= y < H and 0 <= x < W and not blocked[y, x]:
                return (y, x)
            for r in range(1, max_r + 1):
                y0 = max(0, y - r)
                y1 = min(H, y + r + 1)
                x0 = max(0, x - r)
                x1 = min(W, W + r + 1)
                for xx in range(x0, x1):
                    if not blocked[y0, xx]:
                        return (y0, xx)
                    if not blocked[y1 - 1, xx]:
                        return (y1 - 1, xx)
                for yy in range(y0, y1):
                    if not blocked[yy, x0]:
                        return (yy, x0)
                    if not blocked[yy, x1 - 1]:
                        return (yy, x1 - 1)
            return None

        start_ij = _nearest_unblocked(sy, sx, max_r=8)
        goal_ij = _nearest_unblocked(gy, gx, max_r=8)
        if (start_ij is None) or (goal_ij is None):
            return []

        sy, sx = start_ij
        gy, gx = goal_ij

        # 5) 이웃/이동거리
        if allow_diagonal:
            neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            stepc = [1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
        else:
            neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            stepc = [1, 1, 1, 1]

        # 6) 휴리스틱(옥타일)
        def h(y, x):
            dy = abs(y - gy)
            dx = abs(x - gx)
            if allow_diagonal:
                return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)
            return dx + dy

        # 7) 가중치: free=1.0, unknown=unknown_cost_factor, blocked=∞
        free_cost = 1.0
        unk_cost = self.unknown_cost_factor if self.allow_unknown else np.inf

        def cell_cost(y, x):
            if blocked[y, x]:
                return np.inf
            if free[y, x]:
                return free_cost
            return unk_cost  # unknown

        g = np.full((H, W), np.inf, dtype=np.float32)
        parent_yx = np.full((H, W, 2), -1, dtype=np.int32)
        visited = np.zeros((H, W), dtype=bool)

        g[sy, sx] = 0.0
        openq = [(h(sy, sx), 0.0, sy, sx)]
        nodes = 0

        while openq:
            f, gc, y, x = heapq.heappop(openq)
            if visited[y, x]:
                continue
            visited[y, x] = True
            nodes += 1
            if nodes > max_nodes:
                break
            if y == gy and x == gx:
                break

            for (base_d, (dy, dx)) in zip(stepc, neigh):
                yy = y + dy
                xx = x + dx
                if not (0 <= yy < H and 0 <= xx < W):
                    continue
                cc = cell_cost(yy, xx)
                if not np.isfinite(cc):
                    continue
                ng = g[y, x] + base_d * cc
                if ng < g[yy, xx]:
                    g[yy, xx] = ng
                    parent_yx[yy, xx] = (y, x)
                    heapq.heappush(openq, (ng + h(yy, xx), ng, yy, xx))

        if not visited[gy, gx]:
            return []

        # 역추적
        path_xy: List[Tuple[float, float]] = []
        cy, cx = gy, gx
        while not (cy == sy and cx == sx):
            wx, wy = self._ij_to_xy(cy, cx)
            path_xy.append((wx, wy))
            py, px = parent_yx[cy, cx]
            cy, cx = int(py), int(px)
        wx, wy = self._ij_to_xy(sy, sx)
        path_xy.append((wx, wy))
        path_xy.reverse()

        if len(path_xy) > 2:
            step = max(1, len(path_xy) // 50)
            path_xy = path_xy[::step] + [path_xy[-1]]
        return path_xy

    # ---------- 복귀 전용 A* (Unknown 차단) ----------
    def plan_path_return(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        allow_diagonal: bool = True,
    ) -> List[Tuple[float, float]]:
        if self._p is None:
            return []

        H, W = self._p.shape
        occ = self._p >= self.occ_thresh
        free = self._p <= self.free_thresh
        unk = ~(occ | free)

        # 1) 차단 마스크: Occupied
        blocked = occ.copy()

        # 2) 안전 마진 팽창 (로봇 크기 고려, 예: 1.6m)
        inflate_px = max(0, int(round(1.6 / self.res)))
        if inflate_px > 0:
            infl = blocked.copy()
            for dy in range(-inflate_px, inflate_px + 1):
                for dx in range(-inflate_px, inflate_px + 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy0 = max(0, -dy)
                    yy1 = min(H, H - dy)
                    xx0 = max(0, -dx)
                    xx1 = min(W, W - dx)
                    infl[yy0:yy1, xx0:xx1] |= blocked[yy0 + dy : yy1 + dy, xx0 + dx : xx1 + dx]
            blocked = infl

        # 3) 시작/목표 인덱스
        sy, sx = self._xy_to_ij(*start_xy)
        gy, gx = self._xy_to_ij(*goal_xy)
        if not (0 <= sy < H and 0 <= sx < W and 0 <= gy < H and 0 <= gx < W):
            return []

        # 4) 시작/목표 스냅 (blocked 또는 unknown 금지)
        def _nearest_unblocked(y, x, max_r=50):
            if 0 <= y < H and 0 <= x < W and (not blocked[y, x]) and (not unk[y, x]):
                return (y, x)
            for r in range(1, max_r + 1):
                y0 = max(0, y - r)
                y1 = min(H, H + r + 1)
                x0 = max(0, x - r)
                x1 = min(W, W + r + 1)
                for xx in range(x0, x1):
                    if not blocked[y0, xx] and not unk[y0, xx]:
                        return (y0, xx)
                    if not blocked[y1 - 1, xx] and not unk[y1 - 1, xx]:
                        return (y1 - 1, xx)
                for yy in range(y0, y1):
                    if not blocked[yy, x0] and not unk[yy, x0]:
                        return (yy, x0)
                    if not blocked[yy, x1 - 1] and not unk[yy, x1 - 1]:
                        return (yy, x1 - 1)
            return None

        start_ij = _nearest_unblocked(sy, sx, max_r=50)
        goal_ij = _nearest_unblocked(gy, gx, max_r=50)
        if (start_ij is None) or (goal_ij is None):
            return []

        sy, sx = start_ij
        gy, gx = goal_ij

        # 5) 이웃/이동거리
        if allow_diagonal:
            neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            stepc = [1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
        else:
            neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            stepc = [1, 1, 1, 1]

        # 6) 휴리스틱
        def h(y, x):
            dy = abs(y - gy)
            dx = abs(x - gx)
            if allow_diagonal:
                return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)
            return dx + dy

        # 7) 가중치: free=1, unknown=INF(차단), blocked=INF
        free_cost = 1.0
        unk_cost = np.inf

        def cell_cost(y, x):
            if blocked[y, x]:
                return np.inf
            if unk[y, x]:
                return unk_cost
            if free[y, x]:
                return free_cost
            return np.inf  # 중간 확률대(0.35~0.65)는 차단

        g = np.full((H, W), np.inf, dtype=np.float32)
        parent_yx = np.full((H, W, 2), -1, dtype=np.int32)
        visited = np.zeros((H, W), dtype=bool)

        g[sy, sx] = 0.0
        openq = [(h(sy, sx), 0.0, sy, sx)]

        while openq:
            f, gc, y, x = heapq.heappop(openq)
            if visited[y, x]:
                continue
            visited[y, x] = True
            if y == gy and x == gx:
                break

            for (base_d, (dy, dx)) in zip(stepc, neigh):
                yy = y + dy
                xx = x + dx
                if not (0 <= yy < H and 0 <= xx < W):
                    continue
                cc = cell_cost(yy, xx)
                if not np.isfinite(cc):
                    continue
                ng = g[y, x] + base_d * cc
                if ng < g[yy, xx]:
                    g[yy, xx] = ng
                    parent_yx[yy, xx] = (y, x)
                    heapq.heappush(openq, (ng + h(yy, xx), ng, yy, xx))

        if not visited[gy, gx]:
            return []

        # 역추적
        path_xy: List[Tuple[float, float]] = []
        cy, cx = gy, gx
        while not (cy == sy and cx == sx):
            wx, wy = self._ij_to_xy(cy, cx)
            path_xy.append((wx, wy))
            py, px = parent_yx[cy, cx]
            cy, cx = int(py), int(px)
        wx, wy = self._ij_to_xy(sy, sx)
        path_xy.append((wx, wy))
        path_xy.reverse()

        if len(path_xy) > 2:
            step = max(1, len(path_xy) // 50)
            path_xy = path_xy[::step] + [path_xy[-1]]
        return path_xy
