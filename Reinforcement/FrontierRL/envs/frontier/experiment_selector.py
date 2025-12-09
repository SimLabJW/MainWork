"""
experiment_selector.py
Frontier 휴리스틱 평가 및 선택 (개선 버전)
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from .frontier import FrontierCandidate

class ScoredFrontier:
    __slots__ = ("candidate", "center_xy", "center_ij", "bbox_ij", "n", "score", "info_gain")
    
    def __init__(self, cand: FrontierCandidate, score: float, info_gain: float):
        self.candidate = cand
        self.center_xy = cand.center_xy
        self.center_ij = cand.center_ij
        self.bbox_ij = cand.bbox_ij
        self.n = cand.n
        self.score = float(score)
        self.info_gain = float(info_gain)


class FrontierExSelector:
    """
    실험용 Frontier 선택기 (휴리스틱 기반)
    
    개선 사항:
    1. 도달 가능성 검증 강화
    2. 장애물 회피 점수 개선
    3. 정보 이득 계산 정확도 향상
    """
    
    def __init__(
        self,
        *,
        ogm_res_m: float,
        grid_origin_world_xy: Tuple[float, float],
        w_info: float = 0.7,
        w_size: float = 0.1,
        w_dist: float = 0.05,
        w_open: float = 1.0,
        w_trace: float = 0.5,
        info_radius_m: float = 1.0,
        visible_rays: int = 64,
        ray_step_px: int = 1,
        min_free_before_unknown_m: float = 0.6,
        merge_min_sep_m: float = 1.5,
        occ_clearance_m: float = 0.50,
        occ_inflate_m: float = 0.20,
        edge_clearance_m: float = 0.30,
    ):
        self.res = float(ogm_res_m)
        self.origin_xy = grid_origin_world_xy
        
        self.w_info = float(w_info)
        self.w_size = float(w_size)
        self.w_dist = float(w_dist)
        self.w_open = float(w_open)
        self.w_trace = float(w_trace)
        
        self.info_radius_px = max(1, int(np.ceil(info_radius_m / self.res)))
        self.visible_rays = int(max(8, visible_rays))
        self.ray_step_px = int(max(1, ray_step_px))
        self.min_free_before_unknown_px = max(0, int(np.ceil(min_free_before_unknown_m / self.res)))
        
        self.merge_min_sep_m = float(merge_min_sep_m)
        
        self.occ_clearance_px = int(np.ceil(float(occ_clearance_m) / self.res))
        self.occ_inflate_px = int(np.ceil(float(occ_inflate_m) / self.res))
        self.edge_clearance_px = int(np.ceil(float(edge_clearance_m) / self.res))
    
    def score_and_select(
        self,
        *,
        candidates: List[FrontierCandidate],
        masks: Dict[str, np.ndarray],
        robot_xy: Optional[Tuple[float, float]] = None,
        exploration_trace: Optional[np.ndarray] = None,
        do_merge: bool = True,
        top_k: Optional[int] = None,
    ) -> List[ScoredFrontier]:
        """
        Frontier 후보군을 평가하고 상위 K개 반환
        """
        if not candidates:
            return []
        
        free, occ, unk, free_raw, unk_explorable = self._ensure_masks(masks)
        
        # 장애물 팽창 및 거리 필드
        if self.occ_inflate_px > 0:
            occ_inflated = self._binary_dilate(occ.astype(bool), self.occ_inflate_px)
        else:
            occ_inflated = occ.astype(bool)
        
        occ_u8 = occ_inflated.astype(np.uint8)
        free_for_dt = (1 - occ_u8).astype(np.uint8)
        dist_to_occ_px = cv2.distanceTransform(free_for_dt, cv2.DIST_L2, 3)
        
        H, W = free.shape
        scored: List[ScoredFrontier] = []
        
        for cand in candidates:
            cy, cx = cand.center_ij
            xc, yc = cand.center_xy
            
            if not (0 <= cy < H and 0 <= cx < W):
                continue
            
            # Free 내부 + 장애물 거리 확인
            if not bool(free[cy, cx]):
                continue
            if dist_to_occ_px[cy, cx] <= self.occ_clearance_px:
                continue
            
            # 1. 정보 이득 (가시 영역 내 unknown 개수)
            vis_gain = float(self._visible_info_gain(
                cy, cx, free, occ_inflated, unk_explorable,
                min_free_before_unknown_px=self.min_free_before_unknown_px
            ))
            
            # 2. 개방성 (주변 free 공간 비율)
            open_score = float(self._open_aperture_score(
                cy, cx, free, occ_inflated, unk, self.info_radius_px
            ))
            
            # 3. 탐험 흔적 (이미 방문한 곳 페널티)
            trace_term = 0.0
            if exploration_trace is not None and exploration_trace.shape == free.shape:
                y0, x0, y1, x1 = cand.bbox_ij
                y0 = max(0, min(H-1, y0))
                y1 = max(0, min(H-1, y1))
                x0 = max(0, min(W-1, x0))
                x1 = max(0, min(W-1, x1))
                if y1 >= y0 and x1 >= x0:
                    t_win = exploration_trace[y0:y1+1, x0:x1+1]
                    if t_win.size > 0:
                        trace_term = float(1.0 - np.clip(np.mean(t_win), 0.0, 1.0))
            
            # 4. 로봇까지 거리
            dist = 0.0
            if robot_xy is not None:
                dx = xc - robot_xy[0]
                dy = yc - robot_xy[1]
                dist = float(np.hypot(dx, dy))
            
            # 최종 점수
            base_score = (
                self.w_info * vis_gain
                + self.w_open * open_score
                + self.w_size * cand.n
                - self.w_dist * dist
                + self.w_trace * trace_term
            )
            
            scored.append(ScoredFrontier(cand, base_score, vis_gain))
        
        # 병합 및 정렬
        if do_merge and scored:
            scored = self._merge_frontier_by_distance(scored, self.merge_min_sep_m)
        
        scored.sort(key=lambda f: f.score, reverse=True)
        
        if top_k is not None and top_k > 0:
            scored = scored[:top_k]
        
        return scored
    
    def _ensure_masks(self, masks: Dict[str, np.ndarray]):
        free = masks["free"].astype(bool)
        occ = masks["occ"].astype(bool)
        unk = masks["unk"].astype(bool)
        free_raw = masks.get("free_raw", free).astype(bool)
        
        unk_explorable = masks.get("unk_explorable", None)
        if unk_explorable is None:
            dilated_free_raw = self._binary_dilate(free_raw, 1)
            unk_explorable = (unk & dilated_free_raw)
        else:
            unk_explorable = unk_explorable.astype(bool)
        
        return free, occ, unk, free_raw, unk_explorable
    
    def _visible_info_gain(self, cy, cx, free, occ, unk_explorable, min_free_before_unknown_px) -> int:
        """레이캐스팅으로 가시 영역 내 unknown 개수 계산"""
        H, W = free.shape
        r = self.info_radius_px
        K = self.visible_rays
        step = self.ray_step_px
        count = 0
        
        for k in range(K):
            a = 2.0 * np.pi * (k / K)
            ca, sa = np.cos(a), np.sin(a)
            free_run = 0
            seen = False
            
            for d in range(step, r + 1, step):
                iy = int(round(cy + sa * d))
                ix = int(round(cx + ca * d))
                
                if not (0 <= iy < H and 0 <= ix < W):
                    break
                if occ[iy, ix]:
                    break
                if free[iy, ix]:
                    free_run += 1
                    continue
                if unk_explorable[iy, ix] and free_run >= min_free_before_unknown_px:
                    seen = True
                break
            
            if seen:
                count += 1
        
        return count
    
    def _open_aperture_score(self, cy, cx, free, occ, unk, r) -> float:
        """주변 개방성 점수 (free 공간 비율)"""
        H, W = free.shape
        K = self.visible_rays
        step = self.ray_step_px
        runs = []
        
        for k in range(K):
            a = 2.0 * np.pi * (k / K)
            ca, sa = np.cos(a), np.sin(a)
            run = 0
            
            for d in range(step, r + 1, step):
                iy = int(round(cy + sa * d))
                ix = int(round(cx + ca * d))
                
                if not (0 <= iy < H and 0 <= ix < W):
                    break
                if occ[iy, ix] or unk[iy, ix]:
                    break
                run += 1
            
            runs.append(run)
        
        max_steps = max(1, r // step)
        return float(np.mean(runs) / max_steps)
    
    def _merge_frontier_by_distance(self, frontiers, min_sep_m):
        """거리가 가까운 Frontier 병합"""
        if not frontiers:
            return []
        
        min_sep2 = float(min_sep_m) ** 2
        kept = []
        
        for f in sorted(frontiers, key=lambda t: (t.info_gain, t.n), reverse=True):
            fx, fy = f.center_xy
            keep = True
            
            for g in kept:
                gx, gy = g.center_xy
                if (fx - gx)**2 + (fy - gy)**2 < min_sep2:
                    keep = False
                    break
            
            if keep:
                kept.append(f)
        
        return kept
    
    def _binary_dilate(self, mask: np.ndarray, iters: int) -> np.ndarray:
        """8-이웃 팽창"""
        out = mask.astype(bool).copy()
        H, W = out.shape
        
        for _ in range(max(0, iters)):
            grown = out.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy0 = max(0, -dy)
                    yy1 = min(H, H - dy)
                    xx0 = max(0, -dx)
                    xx1 = min(W, W - dx)
                    grown[yy0:yy1, xx0:xx1] |= out[yy0+dy:yy1+dy, xx0+dx:xx1+dx]
            out = grown
        
        return out