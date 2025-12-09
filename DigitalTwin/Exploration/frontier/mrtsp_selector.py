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
    MRTSP 기반 프런티어 선택기 (origin 이동/그리드 확장에 안전한 버전)
    - ij→xy 변환을 하지 않는다.
    - FrontierCandidate.center_xy(월드좌표)만 사용한다.
    - centroid/start도 center로 근사(du=dv=0). 경로계획은 slam.py에서 수행.
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
        min_size: int = 10,            # 너무 작은 후보 제거(옵션)
    ):
        # 인터페이스 호환용으로 보관만; 내부 계산엔 사용하지 않음
        self.res = float(ogm_res_m)
        self.origin_xy = tuple(grid_origin_world_xy)

        self.rs = float(sensor_range_m)
        self.Wd = float(Wd)
        self.Ws = float(Ws)
        self.Vmax = float(max(1e-6, Vmax))
        self.Wmax = float(max(1e-6, Wmax))
        self.min_size = int(max(1, min_size))

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
        # 0) 경량 프리필터
        cands = [c for c in candidates
                 if int(getattr(c, "n", len(getattr(c, "pixel_inds", [])))) >= self.min_size]

        N = len(cands)
        if N == 0:
            return MRTSPResult(None, [], (None if not return_matrix else np.zeros((1, 1))))

        # 1) 키포인트(월드좌표만) : center=centroid=start
        centers = [tuple(map(float, f.center_xy)) for f in cands]
        sizes   = [int(getattr(f, "n", len(getattr(f, "pixel_inds", [])))) for f in cands]

        # 2) 비용행렬 (N+1)x(N+1)
        M = np.zeros((N + 1, N + 1), dtype=np.float64)

        # 0 -> j : (Wd/Ws)*(max(dm, dn) - rs)_+ / P_j + t_lb
        # 여기서는 centroid=start=center 이므로 dm==dn, du==dv==0
        for j in range(1, N + 1):
            cj = centers[j - 1]
            dm = self._dist(robot_xy, cj)
            d = max(0.0, dm - self.rs)  # _+ 절삭
            P = max(1, sizes[j - 1])

            if robot_yaw is not None:
                ang = math.atan2(cj[1] - robot_xy[1], cj[0] - robot_xy[0])
                dpsi = abs((ang - robot_yaw + math.pi) % (2 * math.pi) - math.pi)
                t_lin = dm / self.Vmax
                t_ang = dpsi / self.Wmax
                t_lb = min(t_lin, t_ang)
            else:
                t_lb = 0.0

            M[0, j] = (self.Wd * d) / (self.Ws * P) + t_lb

        # i -> j : (Wd/Ws)*(max(dm, dn) - rs)_+ / P_j  (여기서도 dm==dn)
        for i in range(1, N + 1):
            ci = centers[i - 1]
            for j in range(1, N + 1):
                if i == j:
                    M[i, j] = np.inf
                    continue
                cj = centers[j - 1]
                dm = self._dist(ci, cj)
                d = max(0.0, dm - self.rs)
                P = max(1, sizes[j - 1])
                M[i, j] = (self.Wd * d) / (self.Ws * P)

        # i -> 0 : 복귀 없음
        for i in range(1, N + 1):
            M[i, 0] = 0.0

        # 3) MRTSP 경로(0에서 시작, 1..N 방문, 0 복귀 없음)
        seq_nodes = self._solve_mrtsp(M)              # 예) [0, 3, 2, 5, 1]
        seq_fidx  = [j - 1 for j in seq_nodes if j != 0]  # 0-based

        chosen = cands[seq_fidx[0]] if seq_fidx else None
        return MRTSPResult(
            chosen=chosen,
            sequence_idx=seq_fidx if return_sequence else (seq_fidx[:1] if chosen else []),
            cost_matrix=(M if return_matrix else None),
        )

    # ----------------------- MRTSP solver -----------------------
    def _solve_mrtsp(self, M: np.ndarray) -> List[int]:
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
