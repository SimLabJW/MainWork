import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict

class FrontierCandidate:
    __slots__ = ("pixel_inds", "bbox_ij", "center_ij", "center_xy", "n")
    def __init__(self, pixel_inds, bbox_ij, center_ij, center_xy, n):
        self.pixel_inds = pixel_inds
        self.bbox_ij = bbox_ij
        self.center_ij = center_ij
        self.center_xy = center_xy
        self.n = n

class FrontierDetector:
    """
    WFD(Wavefront Frontier Detection):
      - 로봇 근처 free 셀에서 시작하는 Map BFS로 free 영역을 훑는다.
      - unknown 이면서 인접 8-이웃 중 하나가 free이면 frontier.
      - frontier 성분은 Frontier BFS(8-연결)로 묶어 클러스터 생성.
    """
    def __init__(
        self,
        ogm_res_m: float,
        grid_origin_world_xy: Tuple[float, float],
        *,
        occ_thresh: float = 0.65,
        free_thresh: float = 0.35,
        min_cluster_size: int = 10,
        # 아래 파라미터들은 API 호환용 (WFD에서는 미사용)
        dilate_free: int = 0,
        min_clearance_m: float = 0.0,
        require_reachable: bool = True,
        ignore_border_unknown_margin_m: float = 0.0,
    ):
        self.res = float(ogm_res_m)
        self.origin_xy = grid_origin_world_xy
        self.occ_thresh = float(occ_thresh)
        self.free_thresh = float(free_thresh)
        self.min_cluster = int(min_cluster_size)
        self.require_reachable = bool(require_reachable)

    # ---------- 외부 API ----------
    def detect(
        self,
        logodds: np.ndarray,
        *,
        robot_xy: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, object]:

        masks = self._basic_masks(logodds)  # free, occ, unk
        H, W = masks["free"].shape

        frontier_mask = np.zeros((H, W), dtype=bool)
        candidates: List[FrontierCandidate] = []

        # 로봇 위치가 없으면 마스크 기반 fallback
        if robot_xy is None:
            frontier_mask = self._mask_frontier_fallback(masks)
            clusters = self._clusters_from_mask(frontier_mask)
            candidates = self._candidates_from_clusters(clusters)
            return {"masks": masks, "frontier_mask": frontier_mask, "candidates": candidates}

        # 1) 시드: 로봇에 가장 가까운 free
        seed = self._closest_free(masks["free"], robot_xy)
        if seed is None:
            return {"masks": masks, "frontier_mask": frontier_mask, "candidates": []}

        # 2) WFD
        clusters = self._wfd_frontiers(masks["free"], masks["unk"], seed)

        # 3) 마스크/후보
        for comp in clusters:
            for (iy, ix) in comp:
                frontier_mask[iy, ix] = True
        candidates = self._candidates_from_clusters(clusters)

        return {
            "masks": masks,
            "frontier_mask": frontier_mask,
            "candidates": candidates,
        }

    # ---------- 기본 마스크(확률 → 이산) ----------
    def _basic_masks(self, logodds: np.ndarray) -> Dict[str, np.ndarray]:
        p = 1.0 / (1.0 + np.exp(-logodds))
        free = (p <= self.free_thresh)
        occ  = (p >= self.occ_thresh)
        unk  = ~(free | occ)
        return {"p": p, "free": free, "occ": occ, "unk": unk}

    # ---------- Fallback (로봇 좌표 없을 때만 사용) ----------
    def _mask_frontier_fallback(self, masks: Dict[str, np.ndarray]) -> np.ndarray:
        free, unk = masks["free"], masks["unk"]
        H, W = free.shape
        out = np.zeros_like(free, dtype=bool)
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        ys, xs = np.where(free)
        for y, x in zip(ys, xs):
            for dy, dx in neigh:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and unk[yy, xx]:
                    out[y, x] = True
                    break
        return out

    # ---------- 로봇에서 가장 가까운 free ----------
    def _closest_free(self, free_mask: np.ndarray, robot_xy: Tuple[float, float]) -> Optional[Tuple[int,int]]:
        H, W = free_mask.shape
        rx, ry = robot_xy
        x0, y0 = self.origin_xy
        ixc = int(np.floor((rx - x0) / self.res))
        iyc = int(np.floor((ry - y0) / self.res))

        if 0 <= iyc < H and 0 <= ixc < W and free_mask[iyc, ixc]:
            return (iyc, ixc)

        q = deque()
        seen = np.zeros_like(free_mask, dtype=bool)
        def push(y, x):
            if 0 <= y < H and 0 <= x < W and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
        push(iyc, ixc)
        while q:
            y, x = q.popleft()
            if 0 <= y < H and 0 <= x < W and free_mask[y, x]:
                return (y, x)
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                push(y+dy, x+dx)
        return None

    # ---------- WFD 본체 ----------
    def _wfd_frontiers(self, free: np.ndarray, unk: np.ndarray, seed_ij: Tuple[int,int]):
        H, W = free.shape
        map_open  = np.zeros((H, W), dtype=bool)
        map_closed = np.zeros((H, W), dtype=bool)
        fr_open  = np.zeros((H, W), dtype=bool)
        fr_closed = np.zeros((H, W), dtype=bool)

        # ★ 변경: "free 셀이고, 8-이웃 중에 unknown이 하나라도 있으면 frontier"
        def is_frontier_free(y, x) -> bool:
            if not free[y, x]:
                return False
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < H and 0 <= xx < W and unk[yy, xx]:
                        return True
            return False

        map_q = deque()
        sy, sx = seed_ij
        map_open[sy, sx] = True
        map_q.append((sy, sx))

        clusters = []
        while map_q:
            y, x = map_q.popleft()
            if map_closed[y, x]:
                continue

            # ★ 변경: free-경계면 frontier BFS 시작
            if is_frontier_free(y, x):
                comp = []
                fr_q = deque()
                fr_open[y, x] = True
                fr_q.append((y, x))

                while fr_q:
                    fy, fx = fr_q.popleft()
                    if fr_closed[fy, fx] or map_closed[fy, fx]:
                        continue
                    if is_frontier_free(fy, fx):
                        comp.append((fy, fx))
                        # 8-연결로 같은 경계 따라가기
                        for dy in (-1,0,1):
                            for dx in (-1,0,1):
                                if dy == 0 and dx == 0:
                                    continue
                                yy, xx = fy+dy, fx+dx
                                if 0 <= yy < H and 0 <= xx < W:
                                    if not (fr_open[yy, xx] or fr_closed[yy, xx] or map_closed[yy, xx]):
                                        fr_open[yy, xx] = True
                                        fr_q.append((yy, xx))
                    fr_closed[fy, fx] = True

                for (fy, fx) in comp:
                    map_closed[fy, fx] = True

                if len(comp) >= self.min_cluster:
                    clusters.append(comp)

            # free 영역만 4-연결로 확장
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W:
                    if (not map_open[yy, xx]) and (not map_closed[yy, xx]) and free[yy, xx]:
                        map_open[yy, xx] = True
                        map_q.append((yy, xx))

            map_closed[y, x] = True

        return clusters


    # ---------- 클러스터 → 후보 ----------
    def _candidates_from_clusters(self, clusters: List[List[Tuple[int,int]]]) -> List[FrontierCandidate]:
        out: List[FrontierCandidate] = []
        for pix in clusters:
            ys, xs = zip(*pix)
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            cy = int(round(np.mean(ys)))
            cx = int(round(np.mean(xs)))
            cxw, cyw = self._ij_to_xy(cy, cx)
            out.append(FrontierCandidate(
                pixel_inds=pix,
                bbox_ij=(y0, x0, y1, x1),
                center_ij=(cy, cx),
                center_xy=(cxw, cyw),
                n=len(pix),
            ))
        return out

    # ---------- 좌표 변환 ----------
    def _ij_to_xy(self, iy: int, ix: int) -> Tuple[float, float]:
        x0, y0 = self.origin_xy
        x = x0 + (ix + 0.5) * self.res
        y = y0 + (iy + 0.5) * self.res
        return (x, y)
