# frontier/global_planner.py (초슬림 + A* 경로계획 추가)
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

    # ==== 맵 통계 ====
    def update_map(self, logodds: np.ndarray, origin_xy: Tuple[float, float]) -> None:
        self._p = 1.0 / (1.0 + np.exp(-logodds))
        self._origin_xy = origin_xy

        free = (self._p <= self.free_thresh)
        occ  = (self._p >= self.occ_thresh)
        unk  = ~(free | occ)

        total = float(self._p.size) + 1e-9
        self._coverage = float(free.sum()) / total
        self._unknown_ratio = float(unk.sum()) / total

        if (self._coverage >= self.coverage_done_thresh) or (self._unknown_ratio <= self.unknown_left_thresh):
            self.is_done = True

    def coverage(self) -> float:
        return float(self._coverage)

    def unknown_ratio(self) -> float:
        return float(self._unknown_ratio)

    def notify_frontier_presence(self, frontier_exists: bool, path_exists : bool) -> bool:
        if self.is_done:
            return True
        if not frontier_exists:
            self.is_done = True
            return True
        if not path_exists:
            self.is_done = True
            return True
        return False

    # ==== 좌표 변환 ====
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

    # ==== A* 경로계획 ====
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
        occ  = (self._p >= self.occ_thresh)
        free = (self._p <= self.free_thresh)
        unk  = ~(occ | free)

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
                    yy0 = max(0, -dy); yy1 = min(H, H - dy)
                    xx0 = max(0, -dx); xx1 = min(W, W - dx)
                    infl[yy0:yy1, xx0:xx1] |= blocked[yy0 + dy:yy1 + dy, xx0 + dx:xx1 + dx]
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
                y0 = max(0, y - r); y1 = min(H, y + r + 1)
                x0 = max(0, x - r); x1 = min(W, x + r + 1)
                for xx in range(x0, x1):
                    if not blocked[y0, xx]: return (y0, xx)
                    if not blocked[y1 - 1, xx]: return (y1 - 1, xx)
                for yy in range(y0, y1):
                    if not blocked[yy, x0]: return (yy, x0)
                    if not blocked[yy, x1 - 1]: return (yy, x1 - 1)
            return None

        start_ij = _nearest_unblocked(sy, sx, max_r=8)
        goal_ij  = _nearest_unblocked(gy, gx, max_r=8)
     

        if (start_ij is None) or (goal_ij is None):
            return []

        sy, sx = start_ij
        gy, gx = goal_ij

        

        # 5) 이웃/기본 이동거리
        if allow_diagonal:
            neigh = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            stepc = [1,1,1,1, np.sqrt(2),np.sqrt(2),np.sqrt(2),np.sqrt(2)]
        else:
            neigh = [(-1,0),(1,0),(0,-1),(0,1)]
            stepc = [1,1,1,1]

        # 6) 휴리스틱(옥타일)
        def h(y, x):
            dy = abs(y - gy); dx = abs(x - gx)
            if allow_diagonal:
                return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)
            return dx + dy

        # 7) 가중치 함수: free=1.0, unknown=unknown_cost_factor, blocked=∞
        free_cost = 1.0
        unk_cost  = self.unknown_cost_factor if self.allow_unknown else np.inf

        def cell_cost(y, x):
            if blocked[y, x]:
                return np.inf
            if free[y, x]:
                return free_cost
            # unknown
            return unk_cost

        import heapq
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
                yy = y + dy; xx = x + dx
                if not (0 <= yy < H and 0 <= xx < W):
                    continue
                cc = cell_cost(yy, xx)
                if not np.isfinite(cc):
                    continue
                ng = g[y, x] + base_d * cc  # ← 타겟 셀의 지형 가중치 반영
                if ng < g[yy, xx]:
                    g[yy, xx] = ng
                    parent_yx[yy, xx] = (y, x)
                    heapq.heappush(openq, (ng + h(yy, xx), ng, yy, xx))

        if not visited[gy, gx]:
            return []

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



    def plan_path_return( # ⬅️ 새로운 A* 로직을 직접 포함합니다.
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        allow_diagonal: bool = True
    ) -> List[Tuple[float, float]]:
        """
        원점 복귀 전용: self.plan_path를 호출하지 않고,
        장애물에 대한 팽창(Inflation)을 적용하고,
        Unknown 영역은 무조건 차단하며, 최대 노드 수 제한을 해제합니다.
        """
        if self._p is None:
            return []

        H, W = self._p.shape
        occ  = (self._p >= self.occ_thresh)
        free = (self._p <= self.free_thresh)
        unk  = ~(occ | free)

        # 1) 차단 마스크: Occupied 픽셀 복사
        blocked = occ.copy()

        # 2) 안전 마진 팽창(장애물만 대상으로)
        # 로봇의 크기에 기반하여 팽창 반경(픽셀)을 계산합니다. (예: 1.6m / 해상도)
        inflate_px = max(0, int(round(1.6 / self.res))) 
        if inflate_px > 0:
            infl = blocked.copy()
            # Dilation (팽창) 로직
            for dy in range(-inflate_px, inflate_px + 1):
                for dx in range(-inflate_px, inflate_px + 1):
                    if dy == 0 and dx == 0: 
                        continue
                    yy0 = max(0, -dy); yy1 = min(H, H - dy)
                    xx0 = max(0, -dx); xx1 = min(W, W - dx)
                    infl[yy0:yy1, xx0:xx1] |= blocked[yy0 + dy:yy1 + dy, xx0 + dx:xx1 + dx]
            blocked = infl # ⬅️ blocked 마스크에 팽창된 장애물 영역이 포함됨

        # 3) 시작/목표 인덱스
        sy, sx = self._xy_to_ij(*start_xy)
        gy, gx = self._xy_to_ij(*goal_xy)
        if not (0 <= sy < H and 0 <= sx < W and 0 <= gy < H and 0 <= gx < W):
            print(f"🚨 [A* FAIL] 1. Start/Goal 맵 외부: Start=({sy},{sx}), Goal=({gy},{gx})")
            return []

        # 4) 시작/목표가 막혔으면 주변 unblocked로 스냅
        # Goal 픽셀 오염에 대비하여 스냅 반경을 50px (5.0m)로 최대 확장
        def _nearest_unblocked(y, x, max_r=50): 
            # 팽창된 blocked 픽셀 AND Unknown 픽셀도 모두 차단
            if 0 <= y < H and 0 <= x < W and not blocked[y, x] and not unk[y, x]:
                return (y, x)
            
            for r in range(1, max_r + 1):
                y0 = max(0, y - r); y1 = min(H, y + r + 1)
                x0 = max(0, x - r); x1 = min(W, x + r + 1)
                for xx in range(x0, x1):
                    if not blocked[y0, xx] and not unk[y0, xx]: return (y0, xx)
                    if not blocked[y1 - 1, xx] and not unk[y1 - 1, xx]: return (y1 - 1, xx)
                for yy in range(y0, y1):
                    if not blocked[yy, x0] and not unk[yy, x0]: return (yy, x0)
                    if not blocked[yy, x1 - 1] and not unk[yy, x1 - 1]: return (yy, x1 - 1)
            return None

        start_ij = _nearest_unblocked(sy, sx, max_r=50) 
        goal_ij  = _nearest_unblocked(gy, gx, max_r=50) 

        # ... (스냅 실패 시 에러 출력 및 반환 로직) ...
        if (start_ij is None):
                print(f"🚨 [A* FAIL] 2. Start 지점 스냅 실패: ({sy},{sx}) 주변 50px 내 안전 구역 없음. safety_inflate_m=0.00m")
        if (goal_ij is None):
                print(f"🚨 [A* FAIL] 3. Goal 지점 스냅 실패: ({gy},{gx}) 주변 50px 내 안전 구역 없음. safety_inflate_m=0.00m")
        
        if (start_ij is None) or (goal_ij is None):
            return []
            
        if start_ij != (sy, sx) or goal_ij != (gy, gx):
            print(f"ℹ️ [A* SNAP] Start: ({sy},{sx}) -> ({start_ij[0]},{start_ij[1]}), Goal: ({gy},{gx}) -> ({goal_ij[0]},{goal_ij[1]})")

        sy, sx = start_ij
        gy, gx = goal_ij

        
        # 5) 이웃/기본 이동거리 
        if allow_diagonal:
            neigh = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            stepc = [1,1,1,1, np.sqrt(2),np.sqrt(2),np.sqrt(2),np.sqrt(2)]
        else:
            neigh = [(-1,0),(1,0),(0,-1),(0,1)]
            stepc = [1,1,1,1]

        # 6) 휴리스틱(옥타일)
        def h(y, x):
            dy = abs(y - gy); dx = abs(x - gx)
            if allow_diagonal:
                return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)
            return dx + dy

        # 7) 가중치 함수: free=1.0, unknown=INF (복귀 시), blocked=INF 
        free_cost = 1.0
        unk_cost = np.inf # ⬅️ Unknown 영역을 무조건 차단 (np.inf)으로 처리

        def cell_cost(y, x):
            # 팽창된 장애물(blocked) 영역 차단
            if blocked[y, x]:
                return np.inf
            # Unknown 영역 차단 (복귀 모드)
            if unk[y, x]: 
                return unk_cost 
            # Free 영역 통과
            if free[y, x]:
                return free_cost
            
            return np.inf # 나머지 예외 픽셀 (0.35~0.65)도 차단
        
        # 8) A* 탐색 루프 (plan_path의 로직 그대로)
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
            # ⚠️ 최대 노드 수 제한 (max_nodes) 검사 코드를 완전히 제거합니다.
            
            if y == gy and x == gx:
                break

            for (base_d, (dy, dx)) in zip(stepc, neigh):
                yy = y + dy; xx = x + dx
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
            print(f"🚨 [A* FAIL] 4. 경로 탐색 실패: Goal 픽셀 ({gy},{gx})에 도달하지 못함.")
            print("⚠️ A* returned empty path. The Goal is likely unreachable (blocked by Occupied/Unknown cells or disconnected map).")
            return []

        # 9) 경로 역추적 (plan_path의 로직 그대로)
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