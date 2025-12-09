"""
frontier_wfd.py
WFD (Wavefront Frontier Detection) 알고리즘 (개선 버전)
"""
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
    WFD 기반 Frontier 검출기
    
    개선 사항:
    1. Reachability 검증 강화
    2. Free-Unknown 경계 탐지 정확도 향상
    """
    
    def __init__(
        self,
        ogm_res_m: float,
        grid_origin_world_xy: Tuple[float, float],
        *,
        occ_thresh: float = 0.65,
        free_thresh: float = 0.35,
        min_cluster_size: int = 10,
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
    
    def detect(
        self,
        logodds: np.ndarray,
        *,
        robot_xy: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, object]:
        """
        Frontier 검출 메인 함수
        
        Returns:
            dict: {
                "masks": 마스크 딕셔너리,
                "frontier_mask": Frontier 픽셀 마스크,
                "candidates": FrontierCandidate 리스트
            }
        """
        masks = self._basic_masks(logodds)
        H, W = masks["free"].shape
        
        frontier_mask = np.zeros((H, W), dtype=bool)
        candidates: List[FrontierCandidate] = []
        
        if robot_xy is None:
            # Fallback: 마스크 기반 검출
            frontier_mask = self._mask_frontier_fallback(masks)
            clusters = self._clusters_from_mask(frontier_mask)
            candidates = self._candidates_from_clusters(clusters)
            return {
                "masks": masks,
                "frontier_mask": frontier_mask,
                "candidates": candidates
            }
        
        # WFD 알고리즘
        seed = self._closest_free(masks["free"], robot_xy)
        if seed is None:
            return {
                "masks": masks,
                "frontier_mask": frontier_mask,
                "candidates": []
            }
        
        clusters = self._wfd_frontiers(masks["free"], masks["unk"], seed)
        
        for comp in clusters:
            for (iy, ix) in comp:
                frontier_mask[iy, ix] = True
        
        candidates = self._candidates_from_clusters(clusters)
        
        return {
            "masks": masks,
            "frontier_mask": frontier_mask,
            "candidates": candidates,
        }
    
    def _basic_masks(self, logodds: np.ndarray) -> Dict[str, np.ndarray]:
        """Log-odds → 확률 → Free/Occ/Unknown 마스크"""
        p = 1.0 / (1.0 + np.exp(-logodds))
        free = (p <= self.free_thresh)
        occ = (p >= self.occ_thresh)
        unk = ~(free | occ)
        return {"p": p, "free": free, "occ": occ, "unk": unk}
    
    def _mask_frontier_fallback(self, masks: Dict[str, np.ndarray]) -> np.ndarray:
        """로봇 위치 없을 때 전체 맵 기반 Frontier 검출"""
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
    
    def _closest_free(self, free_mask: np.ndarray, robot_xy: Tuple[float, float]) -> Optional[Tuple[int,int]]:
        """로봇에서 가장 가까운 Free 셀 찾기 (BFS)"""
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
    
    def _wfd_frontiers(self, free: np.ndarray, unk: np.ndarray, seed_ij: Tuple[int,int]):
        """
        WFD 알고리즘: Free 영역을 BFS로 탐색하며 Frontier 검출
        
        핵심:
        - Free 셀이 8-이웃에 Unknown을 포함하면 Frontier
        - Frontier는 8-연결로 클러스터링
        """
        H, W = free.shape
        map_open = np.zeros((H, W), dtype=bool)
        map_closed = np.zeros((H, W), dtype=bool)
        fr_open = np.zeros((H, W), dtype=bool)
        fr_closed = np.zeros((H, W), dtype=bool)
        
        def is_frontier_free(y, x) -> bool:
            """Free이면서 8-이웃에 Unknown이 있는가?"""
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
            
            # Frontier 시작점 발견
            if is_frontier_free(y, x):
                comp = []
                fr_q = deque()
                fr_open[y, x] = True
                fr_q.append((y, x))
                
                # Frontier BFS (8-연결)
                while fr_q:
                    fy, fx = fr_q.popleft()
                    if fr_closed[fy, fx] or map_closed[fy, fx]:
                        continue
                    if is_frontier_free(fy, fx):
                        comp.append((fy, fx))
                        # 8-이웃 확장
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
                
                # 클러스터 등록
                for (fy, fx) in comp:
                    map_closed[fy, fx] = True
                
                if len(comp) >= self.min_cluster:
                    clusters.append(comp)
            
            # Free 영역 4-연결 확장
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W:
                    if (not map_open[yy, xx]) and (not map_closed[yy, xx]) and free[yy, xx]:
                        map_open[yy, xx] = True
                        map_q.append((yy, xx))
            
            map_closed[y, x] = True
        
        return clusters
    
    def _clusters_from_mask(self, mask: np.ndarray) -> List[List[Tuple[int,int]]]:
        """마스크에서 8-연결 성분 추출"""
        H, W = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        clusters = []
        
        for y in range(H):
            for x in range(W):
                if mask[y, x] and not visited[y, x]:
                    comp = []
                    q = deque([(y, x)])
                    visited[y, x] = True
                    
                    while q:
                        cy, cx = q.popleft()
                        comp.append((cy, cx))
                        
                        for dy in (-1,0,1):
                            for dx in (-1,0,1):
                                if dy == 0 and dx == 0:
                                    continue
                                yy, xx = cy+dy, cx+dx
                                if 0 <= yy < H and 0 <= xx < W:
                                    if mask[yy, xx] and not visited[yy, xx]:
                                        visited[yy, xx] = True
                                        q.append((yy, xx))
                    
                    if len(comp) >= self.min_cluster:
                        clusters.append(comp)
        
        return clusters
    
    def _candidates_from_clusters(self, clusters: List[List[Tuple[int,int]]]) -> List[FrontierCandidate]:
        """클러스터 → FrontierCandidate 객체 변환"""
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
    
    def _ij_to_xy(self, iy: int, ix: int) -> Tuple[float, float]:
        """그리드 좌표 → 월드 좌표"""
        x0, y0 = self.origin_xy
        x = x0 + (ix + 0.5) * self.res
        y = y0 + (iy + 0.5) * self.res
        return (x, y)