
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict


class FrontierCandidate:
    """점수/정보이득 등은 없음. 탐지 결과만 담는 단순 컨테이너"""
    __slots__ = ("pixel_inds", "bbox_ij", "center_ij", "center_xy", "n")
    def __init__(self, pixel_inds, bbox_ij, center_ij, center_xy, n):
        self.pixel_inds = pixel_inds
        self.bbox_ij = bbox_ij
        self.center_ij = center_ij
        self.center_xy = center_xy
        self.n = n


class FrontierDetector:
    """
    역할: OGM → free/occ/unk 마스크 → frontier 마스크 → (옵션) reachable 필터 → 클러스터링 → 후보리스트
    출력: {masks, frontier_mask, candidates}
    """
    def __init__(
        self,
        ogm_res_m: float,
        grid_origin_world_xy: Tuple[float, float],
        *,
        occ_thresh: float = 0.65,
        free_thresh: float = 0.35,
        min_cluster_size: int = 10,
        dilate_free: int = 1,
        min_clearance_m: float = 0.30,
        require_reachable: bool = True,
        ignore_border_unknown_margin_m: float = 0.8,
    ):
        self.res = float(ogm_res_m)
        self.origin_xy = grid_origin_world_xy

        self.occ_thresh = float(occ_thresh)
        self.free_thresh = float(free_thresh)
        self.min_cluster = int(min_cluster_size)
        self.dilate_free = int(max(0, dilate_free))

        self.min_clearance_px = max(0, int(round(min_clearance_m / self.res)))
        self.require_reachable = bool(require_reachable)

        self.border_margin_px = max(0, int(round(ignore_border_unknown_margin_m / self.res)))

    # ---------- 외부 API ----------
    def detect(
        self,
        logodds: np.ndarray,
        *,
        robot_xy: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, object]:
        """
        Returns:
          {
            "masks": {
              "p": (H,W) float,
              "free": bool,
              "occ":  bool,
              "unk":  bool,
              "unk_explorable": bool,
              "occ_inflate": bool,
            },
            "frontier_mask": bool (reachable 적용 이후),
            "candidates": List[FrontierCandidate]
          }
        """
        masks = self._build_masks(logodds)
        frontier_mask = self._compute_frontier_mask(masks)

        if self.require_reachable and (robot_xy is not None):
            reach = self._reachable_mask(masks["free"], robot_xy)
            frontier_mask &= reach

        clusters = self._clusters_from_mask(frontier_mask)
        candidates: List[FrontierCandidate] = []
        for pix in clusters:
            if len(pix) < self.min_cluster:
                continue
            ys, xs = zip(*pix)
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            cy, cx = pix[len(pix) // 2]
            cxw, cyw = self._ij_to_xy(cy, cx)
            candidates.append(
                FrontierCandidate(
                    pixel_inds=pix,
                    bbox_ij=(y0, x0, y1, x1),
                    center_ij=(cy, cx),
                    center_xy=(cxw, cyw),
                    n=len(pix),
                )
            )

        return {
            "masks": masks,
            "frontier_mask": frontier_mask,
            "candidates": candidates,
        }

    # ---------- 좌표/유틸 ----------
    def _ij_to_xy(self, iy: int, ix: int) -> Tuple[float, float]:
        x0, y0 = self.origin_xy
        x = x0 + (ix + 0.5) * self.res
        y = y0 + (iy + 0.5) * self.res
        return (x, y)

    def _xy_to_ij(self, x: float, y: float) -> Tuple[int, int]:
        x0, y0 = self.origin_xy
        ix = int(np.floor((x - x0) / self.res))
        iy = int(np.floor((y - y0) / self.res))
        return (iy, ix)

    def _binary_dilate(self, mask: np.ndarray, iters: int) -> np.ndarray:
        out = mask.copy()
        H, W = out.shape
        for _ in range(max(0, iters)):
            grown = out.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy0 = max(0, -dy); yy1 = min(H, H - dy)
                    xx0 = max(0, -dx); xx1 = min(W, W - dx)
                    grown[yy0:yy1, xx0:xx1] |= out[yy0+dy:yy1+dy, xx0+dx:xx1+dx]
            out = grown
        return out

    # ---------- Stage 1: 마스크 ----------
    def _build_masks(self, logodds: np.ndarray) -> Dict[str, np.ndarray]:
        """
        log-odds → p → free_raw / free / occ / unk, unk_explorable, occ_inflate
        """
        p = 1.0 / (1.0 + np.exp(-logodds))

        # 1) 팽창 전 free (raw)과 점유
        free_raw = (p <= self.free_thresh)
        occ      = (p >= self.occ_thresh)

        # 2) unknown 은 'raw_free' 기준으로 정의  (※ dilate된 free 사용 금지)
        unk = ~(free_raw | occ)

        # 3) 주행 여유/경로 계산용으로만 쓸 dilate된 free
        free = self._binary_dilate(free_raw, self.dilate_free) if self.dilate_free > 0 else free_raw

        # 4) 탐색 가능한 unknown = unknown ∧ (raw_free 인접)
        unk_explorable = unk & self._binary_dilate(free_raw, 1)

        # 5) 지도 외곽 margin unknown 제거
        if self.border_margin_px > 0:
            bm = self.border_margin_px
            border = np.zeros_like(unk_explorable, dtype=bool)
            border[:bm, :] = True; border[-bm:, :] = True
            border[:, :bm] = True; border[:, -bm:] = True
            unk_explorable &= ~border

        # 6) 안전거리 확보용 점유 팽창
        occ_inflate = self._binary_dilate(occ, self.min_clearance_px) if self.min_clearance_px > 0 else occ

        return {
            "p": p,
            "free_raw": free_raw,    # ★ 추가
            "free": free,            # (주행/도달가능 등엔 이걸 써도 됨)
            "occ": occ,
            "unk": unk,
            "unk_explorable": unk_explorable,
            "occ_inflate": occ_inflate,
        }


    # ---------- Stage 2: frontier 마스크 ----------
    def _compute_frontier_mask(self, masks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        frontier := free_raw ∧ (8-이웃 중 unk_explorable 존재)
        + 안전거리: occ_inflate 근접 제거
        + free–occ 접촉부 제거(명시)
        """
        free_raw        = masks["free_raw"]       # ★ raw 사용
        unk_explorable  = masks["unk_explorable"]
        occ_inflate     = masks["occ_inflate"]
        occ             = masks["occ"]

        H, W = free_raw.shape
        frontier_mask = np.zeros_like(free_raw, dtype=bool)
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        ys, xs = np.where(free_raw)
        for y, x in zip(ys, xs):
            for dy, dx in neigh:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and unk_explorable[yy, xx]:
                    frontier_mask[y, x] = True
                    break

        # 안전거리: 팽창 점유와 맞닿은 프론티어 제거
        if self.min_clearance_px > 0:
            frontier_mask &= ~occ_inflate

        # ★ free–occ 경계 명시적으로 제거(바로 옆이 occ면 제외)
        frontier_mask &= ~self._binary_dilate(occ, 1)

        return frontier_mask

    # ---------- Stage 3: reachable ----------
    def _reachable_mask(self, free_mask: np.ndarray, robot_xy: Tuple[float, float]) -> np.ndarray:
        H, W = free_mask.shape
        rx, ry = robot_xy
        x0, y0 = self.origin_xy
        ixc = int(np.floor((rx - x0) / self.res))
        iyc = int(np.floor((ry - y0) / self.res))

        if not (0 <= iyc < H and 0 <= ixc < W) or not free_mask[iyc, ixc]:
            seeds = [(iyc+dy, ixc+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                     if 0 <= iyc+dy < H and 0 <= ixc+dx < W and free_mask[iyc+dy, ixc+dx]]
            if not seeds:
                return np.zeros_like(free_mask, dtype=bool)
            sy, sx = seeds[0]
        else:
            sy, sx = iyc, ixc

        q = deque([(sy, sx)])
        seen = np.zeros_like(free_mask, dtype=bool)
        seen[sy, sx] = True
        while q:
            y, x = q.popleft()
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and free_mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    q.append((yy, xx))
        return seen

    # ---------- Stage 4: 클러스터 ----------
    def _clusters_from_mask(self, mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        H, W = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        comps: List[List[Tuple[int,int]]] = []
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        ys, xs = np.where(mask)
        for sy, sx in zip(ys, xs):
            if visited[sy, sx]:
                continue
            q = deque([(sy, sx)])
            visited[sy, sx] = True
            cur = []
            while q:
                y, x = q.popleft()
                cur.append((y, x))
                for dy, dx in neigh:
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        q.append((yy, xx))
            if cur:
                comps.append(cur)
        return comps
