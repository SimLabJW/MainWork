# frontier/mrtsp_selector.py
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
from frontier.frontier import FrontierCandidate


class MRTSPResult:
    __slots__ = ("chosen", "sequence_idx", "cost_matrix")
    def __init__(self, chosen, sequence_idx, cost_matrix):
        self.chosen = chosen
        self.sequence_idx = list(sequence_idx) if sequence_idx is not None else []
        self.cost_matrix = cost_matrix


class FrontierMRTSPSelector:
    """
    논문식 프런티어 선택 전용:
      - 입력: FrontierCandidate 리스트(이미 검출됨), 로봇 (x,y[,yaw])
      - 비용함수:
          d_ij = max(dm+du, dn+dv) - r_s
          P_j  = S_j (= frontier size)
          0->j에는 시간 하한 t_lb 포함 (min( ||p0-cj||/Vmax, Δψ/Wmax ))
        최종 M 행렬로 '0에서 시작, 1..N 각 1회 방문, 0 복귀 없음' 경로 최소화
      - 출력: 첫 방문 프런티어(=다음 목표), 전체 방문 시퀀스, 비용행렬(옵션)

    ※ OGM 확률맵/마스크 재계산 없음. candidates의 pixel_inds/center_xy 만 사용.
    """

    def __init__(
        self,
        *,
        ogm_res_m: float,
        grid_origin_world_xy: Tuple[float, float],
        sensor_range_m: float = 6.0,   # r_s
        Wd: float = 1.0,
        Ws: float = 1.0,
        Vmax: float = 0.8,
        Wmax: float = 1.2,
    ):
        self.res = float(ogm_res_m)
        self.origin_xy = tuple(grid_origin_world_xy)
        self.rs = float(sensor_range_m)
        self.Wd = float(Wd)
        self.Ws = float(Ws)
        self.Vmax = float(max(1e-6, Vmax))
        self.Wmax = float(max(1e-6, Wmax))

    # ----------------------- Public API -----------------------
    def select(
        self,
        *,
        candidates: List[FrontierCandidate],
        robot_xy: Tuple[float, float],
        robot_yaw: Optional[float] = None,
        return_sequence: bool = False,
        return_matrix: bool = False,
    ) -> MRTSPResult:
        """
        반환:
          - chosen: 첫 방문 프런티어(없으면 None)
          - sequence_idx: 전체 방문 순서(프런티어 인덱스, 0-based)
          - cost_matrix: (N+1)x(N+1) 비용행렬(요청 시)
        """
        N = len(candidates)
        if N == 0:
            return MRTSPResult(None, [], None)

        keys = self._compute_keypoints(candidates, robot_xy)
        M = self._build_cost_matrix(keys, candidates, robot_xy, robot_yaw)
        seq_nodes = self._solve_mrtsp(M)  # 예: [0, 3, 2, 5, 1, 4]
        seq_fidx = [j - 1 for j in seq_nodes if j != 0]  # 0-based

        chosen = candidates[seq_fidx[0]] if seq_fidx else None
        return MRTSPResult(
            chosen=chosen,
            sequence_idx=seq_fidx if return_sequence else (seq_fidx[:1] if chosen else []),
            cost_matrix=(M if return_matrix else None),
        )

    # ----------------------- internals -----------------------
    def _compute_keypoints(self, frontiers: List[FrontierCandidate], robot_xy: Tuple[float, float]):
        """
        각 프런티어에 대해:
          - center_xy: 후보에 이미 있음
          - centroid_xy: pixel_inds를 (i,j)->(x,y) 변환 평균
          - start_xy(Oa): 로봇에 가장 가까운 픽셀의 (x,y)
          - size: cand.n
        """
        out = []
        rx, ry = robot_xy
        for f in frontiers:
            cx, cy = f.center_xy

            # centroid
            if f.pixel_inds:
                xs, ys = [], []
                for (iy, ix) in f.pixel_inds:
                    x, y = self._ij_to_xy(iy, ix)
                    xs.append(x); ys.append(y)
                gx = float(np.mean(xs)) if xs else cx
                gy = float(np.mean(ys)) if ys else cy
            else:
                gx, gy = cx, cy

            # start(Oa): 로봇과 가장 가까운 픽셀
            best = None
            best_d2 = float("inf")
            for (iy, ix) in f.pixel_inds:
                x, y = self._ij_to_xy(iy, ix)
                d2 = (x - rx) * (x - rx) + (y - ry) * (y - ry)
                if d2 < best_d2:
                    best_d2 = d2
                    best = (x, y)
            sx, sy = best if best is not None else (cx, cy)

            out.append({
                "center": (cx, cy),
                "centroid": (gx, gy),
                "start": (sx, sy),
                "size": int(getattr(f, "n", len(getattr(f, "pixel_inds", [])))),
            })
        return out

    def _build_cost_matrix(
        self,
        keys: List[dict],
        frontiers: List[FrontierCandidate],
        robot_xy: Tuple[float, float],
        robot_yaw: Optional[float],
    ) -> np.ndarray:
        """
        M 크기: (N+1)x(N+1)
          - 0행/열: 로봇
          - 0->j: (Wd/Ws)*d(0,j)/P_j + t_lb
          - i->j: (Wd/Ws)*d(i,j)/P_j
          - i->0: 0   (복귀 없음)
        """
        N = len(frontiers)
        M = np.zeros((N + 1, N + 1), dtype=np.float64)

        centers   = [k["center"]   for k in keys]
        centroids = [k["centroid"] for k in keys]
        starts    = [k["start"]    for k in keys]
        sizes     = [k["size"]     for k in keys]

        # 0 -> j
        for j in range(1, N + 1):
            cj, gj, sj = centers[j - 1], centroids[j - 1], starts[j - 1]
            dm = self._dist(robot_xy, cj)
            dn = self._dist(robot_xy, gj)
            # ✅ 논문식: u,v는 프런티어 내부에서 start까지
            du = self._dist(cj, sj)
            dv = self._dist(gj, sj)

            # ✅ 센서 범위 보정(음수 방지)
            d = max(0.0, max(dm + du, dn + dv) - self.rs)

            P = max(1, sizes[j - 1])

            # t_lb: 방향 고려 (선형 vs 회전 하한 중 더 작은 값)
            if robot_yaw is not None:
                ang = math.atan2(cj[1] - robot_xy[1], cj[0] - robot_xy[0])
                dpsi = abs((ang - robot_yaw + math.pi) % (2 * math.pi) - math.pi)
                t_lin = self._dist(robot_xy, cj) / self.Vmax
                t_ang = dpsi / self.Wmax
                t_lb = min(t_lin, t_ang)
            else:
                t_lb = 0.0

            M[0, j] = (self.Wd * d) / (self.Ws * P) + t_lb

        # i -> j
        for i in range(1, N + 1):
            ci, gi = centers[i - 1], centroids[i - 1]
            for j in range(1, N + 1):
                if i == j:
                    M[i, j] = np.inf
                    continue
                cj, gj, sj = centers[j - 1], centroids[j - 1], starts[j - 1]
                dm = self._dist(ci, cj)
                dn = self._dist(gi, gj)
                du = self._dist(ci, sj)
                dv = self._dist(gi, sj)
                # ✅ 절삭
                d = max(0.0, max(dm + du, dn + dv) - self.rs)
                P = max(1, sizes[j - 1])
                M[i, j] = (self.Wd * d) / (self.Ws * P)

        # i -> 0 : 복귀 없음(0)
        for i in range(1, N + 1):
            M[i, 0] = 0.0

        return M

    # ----------------------- MRTSP solver -----------------------
    def _solve_mrtsp(self, M: np.ndarray) -> List[int]:
        """0에서 시작, 1..N 각 1회 방문, 0 복귀 없음."""
        N = M.shape[0] - 1
        if N <= 1:
            return [0] + ([1] if N == 1 else [])
        if N <= 14:
            return self._held_karp_path(M)
        return self._nn_2opt_path(M)

    def _held_karp_path(self, M: np.ndarray) -> List[int]:
        N = M.shape[0] - 1
        ALL = 1 << N
        DP: Dict[Tuple[int, int], Tuple[float, int]] = {(1 << (j - 1), j): (M[0, j], 0) for j in range(1, N + 1)}
        for mask in range(1, ALL):
            for last in range(1, N + 1):
                if not (mask & (1 << (last - 1))):
                    continue
                cur = DP.get((mask, last))
                if cur is None:
                    continue
                cur_cost, _ = cur
                for nxt in range(1, N + 1):
                    if mask & (1 << (nxt - 1)):
                        continue
                    nmask = mask | (1 << (nxt - 1))
                    cost = cur_cost + M[last, nxt]
                    old = DP.get((nmask, nxt))
                    if (old is None) or (cost < old[0]):
                        DP[(nmask, nxt)] = (cost, last)
        full = ALL - 1
        best_cost = float("inf"); best_last = None
        for last in range(1, N + 1):
            val = DP.get((full, last))
            if val and val[0] < best_cost:
                best_cost = val[0]; best_last = last
        # 경로 복원
        seq = [best_last]
        mask = full; last = best_last
        while mask:
            _, prev = DP[(mask, last)]
            if prev == 0:
                seq.append(0); break
            seq.append(prev)
            mask ^= (1 << (last - 1))
            last = prev
        seq.reverse()
        return seq

    def _nn_2opt_path(self, M: np.ndarray) -> List[int]:
        N = M.shape[0] - 1
        unv = set(range(1, N + 1))
        path = [0]
        cur = 0
        # Greedy NN
        while unv:
            nxt = min(unv, key=lambda j: M[cur, j])
            path.append(nxt)
            unv.remove(nxt)
            cur = nxt
        # 2-opt
        improved = True
        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for k in range(i + 1, len(path) - 1):
                    a, b = path[i - 1], path[i]
                    c, d = path[k], path[k + 1]
                    if M[a, c] + M[b, d] < M[a, b] + M[c, d]:
                        path[i:k + 1] = reversed(path[i:k + 1])
                        improved = True
        return path

    # ----------------------- utils -----------------------
    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _ij_to_xy(self, iy: int, ix: int) -> Tuple[float, float]:
        x0, y0 = self.origin_xy
        x = x0 + (ix + 0.5) * self.res
        y = y0 + (iy + 0.5) * self.res
        return (x, y)
