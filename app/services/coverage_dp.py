import heapq
import itertools
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence
import numpy as np
import pandas as pd

# Tolerance used (a) to validate that demand lies in [0, 1] and (b) to ignore
# float rounding noise when deciding whether a marginal gain is worth taking.
_EPS = 1e-9

# utilities
def _check_demand_range(d: np.ndarray) -> None:
    """Demand is a load in [0, 1]: 1 = slot saturates the row, 0 = nobody needs it."""
    if d.size and (d.min() < -_EPS or d.max() > 1.0 + _EPS):
        raise ValueError(f"demand must lie in [0, 1]")

def extend_demand(demand: np.ndarray, L: int) -> np.ndarray:
    """
    Append L-1 empty (zero-demand) slots on the right of the last axis.
    """
    if L < 1:
        raise ValueError(f"need L >= 1, got L={L}")
    d = np.asarray(demand, dtype=np.float64)
    width = [(0, 0)] * (d.ndim - 1) + [(0, L - 1)]
    return np.pad(d, width)

# output of solve_coverage and solve_coverage_batch
@dataclass
class Solution:
    """
    Per-slot payoff of a covered slot is  d - alpha * (1 - d):
    d is served, 1 - d is capacity paid for but not needed.
    """
    value: float                 # optimal objective = inside - alpha * outside
    starts: np.ndarray           # sorted start indices of the chosen rectangles
    coverage: np.ndarray         # bool[T], covered slots
    n_rects: int = field(init=False)
    inside: float = field(init=False, default=0.0)    # served demand:      sum(d * cov)
    outside: float = field(init=False, default=0.0)   # idle capacity paid: sum((1 - d) * cov)
    missed: float = field(init=False, default=0.0)    # demand uncovered:   sum(d * ~cov)

    def __post_init__(self) -> None:
        self.n_rects = int(len(self.starts))

    def _fill_stats(self, demand: np.ndarray) -> "Solution":
        d = np.asarray(demand, dtype=np.float64)
        cov = self.coverage
        self.inside = float(d[cov].sum())
        self.outside = float((1.0 - d)[cov].sum())
        self.missed = float(d[~cov].sum())
        return self

# ---------------------------------------------------------------------------
# rectangle values via prefix sums
def rectangle_values(
    demand: Sequence[float] | np.ndarray,
    L: int,
    alpha: float,
    allow_overhang: bool = True,
) -> np.ndarray:
    """
    Compute the value of placing a rectangle of length L at every possible start slot s.

    allow_overhang=True (default): the demand is first extended with L-1 empty slots on the right
    """
    d = np.asarray(demand, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("demand must be 1-D; use solve_coverage_batch for many rows")
    if L < 1:
        raise ValueError(f"need L >= 1, got L={L}")
    _check_demand_range(d)
    if allow_overhang:
        d = extend_demand(d, L)
    T = d.size
    if L > T:
        raise ValueError(f"need L <= T when allow_overhang=False, got L={L}, T={T}")
    score = d - alpha * (1.0 - d)              # per-slot payoff when covered
    csum = np.concatenate(([0.0], np.cumsum(score)))
    return csum[L:] - csum[:-L]

# main solver
def solve_coverage(
    demand: Sequence[float] | np.ndarray,
    L: int,
    alpha: float,
    max_rects: Optional[int] = None,
    allow_overhang: bool = True,
) -> Solution:
    """
    Exact DP to place at most max_rects rectangles of length L on a 1-D demand vector, maximizing the objective.

    F[t, k] = best objective achievable using only rectangles entirely inside
              slots [0, t), with at most k of them.
    """
    d = np.asarray(demand, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("demand must be 1-D; use solve_coverage_batch for many rows")
    if allow_overhang:
        d = extend_demand(d, L)                # from here on T is the extended length
    T = d.size
    # w[s] = payoff of placing one rectangle at start s (see rectangle_values).
    w = rectangle_values(d, L, alpha, allow_overhang=False)
    # Cap on rectangles: at most T // L can physically fit; max_rects tightens that.
    K = (T // L) if max_rects is None else int(min(max_rects, T // L))
    if K <= 0:
        cov = np.zeros(T, dtype=bool)
        return Solution(0.0, np.empty(0, dtype=int), cov)._fill_stats(d)

    wl = w.tolist()

    # F[t][k] = best objective using slots [0, t) with at most k rectangles. 
    # take[t][k] records whether the optimum at (t, k) ends a rectangle right at t, 
    # so the backtrack below can recover the shift starts.
    F: list[list[float]] = [[0.0] * (K + 1) for _ in range(T + 1)]
    take: list[list[bool]] = [[False] * (K + 1) for _ in range(T + 1)]

    for t in range(1, T + 1):
        prev, cur = F[t - 1], F[t]
        cur[:] = prev  # default: carry forward "no rectangle ends at t" for every k
        if t >= L:
            # Placing a rectangle on [t-L, t) uses up one of the k rectangles,
            # so it competes against F[t-L][k-1] + w[t-L] for each k.
            src, tk, wv = F[t - L], take[t], wl[t - L]
            for k in range(1, K + 1):
                cand = src[k - 1] + wv
                if cand > cur[k]:
                    cur[k] = cand
                    tk[k] = True

    # backtrack
    # Start at the final state (T, K) and undo whichever choice take[t][k]
    # recorded, shrinking t (and k, when a rectangle is placed) until t == 0.
    starts: list[int] = []
    t, k = T, K
    while t > 0:
        if k > 0 and take[t][k]:
            starts.append(t - L)
            t -= L
            k -= 1
        else:
            t -= 1
    starts.reverse()

    coverage = np.zeros(T, dtype=bool)
    for s in starts:
        coverage[s:s + L] = True

    return Solution(float(F[T][K]), np.asarray(starts, dtype=int), coverage)._fill_stats(d)

# ---------------------------------------------------------------------------
# batched version: many demand vectors at once
def solve_coverage_batch(
    demand: np.ndarray,
    L: int,
    alpha: float,
    max_rects: Optional[int] = None,
    return_curve: bool = False,
    allow_overhang: bool = True,
) -> np.ndarray:
    """
    For each row compute the best achievable value for every number of shifts k = 0..K

    allow_overhang=True (default): demand is extended with L-1 empty slots on the right first.

    return_curve=False (default): each row solved independently with its own
        cap of `max_rects` rectangles -> (N,) array of optimal values.
    return_curve=True: returns the full (N, K+1) value curve instead, where
        column k is each row's optimal value using at most k rectangles
        (0 <= k <= K). This curve is what allocate_budget needs to split a
        single, shared rectangle budget across rows instead of giving every
        row its own independent cap.
    """
    D = np.asarray(demand, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("demand must be (N, T)")
    if L < 1:
        raise ValueError(f"need L >= 1, got L={L}")
    _check_demand_range(D)
    if allow_overhang:
        D = extend_demand(D, L)
    N, T = D.shape
    if L > T:
        raise ValueError(f"need L <= T when allow_overhang=False, got L={L}, T={T}")
    score = D - alpha * (1.0 - D)
    csum = np.concatenate([np.zeros((N, 1)), np.cumsum(score, axis=1)], axis=1)
    W = csum[:, L:] - csum[:, :-L]                       # (N, T-L+1)

    K = (T // L) if max_rects is None else int(min(max_rects, T // L))
    if K <= 0:
        return np.zeros((N, 1)) if return_curve else np.zeros(N)

    F = np.zeros((T + 1, N, K + 1), dtype=np.float64)
    for t in range(1, T + 1):
        F[t] = F[t - 1]
        if t >= L:
            cand = F[t - L, :, :-1] + W[:, t - L, None]
            np.maximum(F[t, :, 1:], cand, out=F[t, :, 1:])
    return F[T].copy() if return_curve else F[T, :, K].copy()

@dataclass
class Allocation:
    k: np.ndarray               # int[N], rectangles assigned to each row
    value: float                 # total objective, summed across rows
    per_row_value: np.ndarray    # float[N], each row's value at its assigned k

def allocate_budget(curves: np.ndarray, K_total: int, exact: bool = False) -> Allocation:
    """
    Split the shared budget K_total among the rows by maximising the sum of their curves

    curves is the (N, K+1) array from solve_coverage_batch(..., return_curve=True)

    exact=False (default): greedy over marginal gains curves[i,k]-curves[i,k-1], always awarding 
    the next rectangle to whichever row's next unit is currently most valuable. 
    This is optimal if every row's curve is concave (marginal gains non-increasing in k)
    exact=True: exact DP over (rows, budget spent), O(N * K_total * K).
    """
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError("curves must be (N, K+1), as returned by solve_coverage_batch(..., return_curve=True)")
    N, Kp1 = curves.shape
    K_max = Kp1 - 1
    K_total = int(max(0, min(K_total, N * K_max)))

    k = _allocate_exact(curves, K_total) if exact else _allocate_greedy(curves, K_total)
    per_row_value = curves[np.arange(N), k]
    return Allocation(k=k, value=float(per_row_value.sum()), per_row_value=per_row_value)

def _allocate_greedy(curves: np.ndarray, K_total: int) -> np.ndarray:
    N, Kp1 = curves.shape
    K_max = Kp1 - 1
    k = np.zeros(N, dtype=int)

    # Max-heap (via negated gain) of each row's next available increment.
    heap: list[tuple[float, int]] = []
    for i in range(N):
        if K_max >= 1:
            heapq.heappush(heap, (-(curves[i, 1] - curves[i, 0]), i))

    taken = 0
    while heap and taken < K_total:
        neg_gain, i = heapq.heappop(heap)
        if -neg_gain <= _EPS:
            break  # best remaining increment isn't worth taking (or is float noise)
        k[i] += 1
        taken += 1
        if k[i] < K_max:
            next_gain = curves[i, k[i] + 1] - curves[i, k[i]]
            heapq.heappush(heap, (-next_gain, i))

    return k

def _allocate_exact(curves: np.ndarray, K_total: int) -> np.ndarray:
    N, Kp1 = curves.shape
    K_max = Kp1 - 1
    NEG = float("-inf")

    dp = [0.0] + [NEG] * K_total          # dp[b]: best value so far, budget b spent
    choice: list[list[int]] = []          # choice[i][b]: k picked for row i at budget b

    for i in range(N):
        row = curves[i]
        row_choice = [0] * (K_total + 1)
        new_dp = [NEG] * (K_total + 1)
        for b in range(K_total + 1):
            best_val, best_k = NEG, 0
            for kk in range(min(K_max, b) + 1):
                prev = dp[b - kk]
                if prev == NEG:
                    continue
                cand = prev + row[kk]
                if cand > best_val:
                    best_val, best_k = cand, kk
            new_dp[b] = best_val
            row_choice[b] = best_k
        dp = new_dp
        choice.append(row_choice)

    ks = [0] * N
    b = K_total
    for i in range(N - 1, -1, -1):
        ks[i] = choice[i][b]
        b -= ks[i]

    return np.asarray(ks, dtype=int)

def solve_allocated(
    demand: np.ndarray,
    L: int,
    alpha: float,
    alloc: Allocation,
    allow_overhang: bool = True,
    ) -> list[Solution]:
    """
    Re-solve each row with its own number of shifts k to recover the actual start times.
    """
    D = np.asarray(demand, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("demand must be (N, T)")
    if len(alloc.k) != D.shape[0]:
        raise ValueError(f"alloc has {len(alloc.k)} rows, demand has {D.shape[0]}")
    return [solve_coverage(row, L, alpha, max_rects=int(k), allow_overhang=allow_overhang)
            for row, k in zip(D, alloc.k)]

@dataclass
class CoverageReport:
    blocks: pd.DataFrame
    rows: pd.DataFrame
    total: pd.Series

def coverage_report(D, sols, L, time_values, row_labels=None, alpha=None) -> CoverageReport:
    """
    Prepare a report of the coverage DP solution, including per-block and per-row statistics.

    D    : (n_rows, T) demand grid in [0, 1]
    sols : list of Solution for the first len(sols) rows (the rows that have a server); the remaining
           rows of D have no server: they count as demand with 0 blocks, 0 coverage
    L    : shift length in bins
    time_values : (T,) datetime-like, the start time of each bin
    row_labels : optional names for the index of rows (e.g. server labels, "no server" beyond)
    alpha : optional, the alpha used to compute the solutions (for reporting only)
    """
    D = np.asarray(D, dtype=np.float64)
    if len(sols) > D.shape[0]:
        raise ValueError(f"{len(sols)} solutions for {D.shape[0]} rows")
    with_obj = alpha is not None
    row_demand = D.sum(axis=1)
    total_demand = float(D.sum())
    safe = lambda num, den: num / den if den > 0 else np.nan
 
    tv = pd.to_datetime(pd.Series(time_values).reset_index(drop=True))
    step = tv.iloc[1] - tv.iloc[0]
 
    recs = []
    for c, sol in enumerate(sols):
        for b, s0 in enumerate(sol.starts):
            s0 = int(s0)
            covered = float(D[c, s0:s0 + L].sum())
            rec = dict(row=c, block=b, start_slot=s0, end_slot=s0 + L)
            rec.update(start_time=tv.iloc[s0], end_time=tv.iloc[s0] + L * step)
            rec.update(
                covered_demand=covered,
                idle_capacity=L - covered,
                utilization=covered / L,
            )
            if with_obj:                                           
                rec["objective"] = covered - alpha * (L - covered)
            recs.append(rec)
    blocks = pd.DataFrame(recs)
    if blocks.empty:
        cols = ["row", "block", "start_slot", "end_slot", "covered_demand", "idle_capacity", "utilization"]
        blocks = pd.DataFrame(columns=cols + (["objective"] if with_obj else []))

    n_rows = D.shape[0]
    covered_by_row = blocks.groupby("row")["covered_demand"].sum().reindex(range(n_rows), fill_value=0.0)
    n_blocks = blocks.groupby("row").size().reindex(range(n_rows), fill_value=0)
    capacity = n_blocks.to_numpy() * L
    rows = pd.DataFrame({
        "n_blocks": n_blocks.to_numpy(),
        "demand": row_demand,
        "covered_demand": covered_by_row.to_numpy(),
        "missed_demand": np.clip(row_demand - covered_by_row.to_numpy(), 0.0, None),
        "idle_capacity": capacity - covered_by_row.to_numpy(),
        "coverage": [safe(cv, dm) for cv, dm in zip(covered_by_row, row_demand)],
        "utilization": [safe(cv, cp) for cv, cp in zip(covered_by_row, capacity)],
    }).rename_axis("row")
    if with_obj:                                                       
        rows["objective"] = rows["covered_demand"] - alpha * rows["idle_capacity"]
    rows.insert(0, "server", list(row_labels) if row_labels is not None else [None] * n_rows)

    covered_tot = float(covered_by_row.sum())
    cap_tot = float(capacity.sum())
    total_dict = {                                                     
        "n_blocks": int(n_blocks.sum()),
        "demand": total_demand,
        "covered_demand": covered_tot,
        "missed_demand": max(total_demand - covered_tot, 0.0),
        "idle_capacity": cap_tot - covered_tot,
        "coverage": safe(covered_tot, total_demand),
    }
    if with_obj:                          
        total_dict["objective"] = covered_tot - alpha * (cap_tot - covered_tot)    
    return CoverageReport(blocks=blocks, rows=rows, total=pd.Series(total_dict))







#-------------------------non implementato
# --------------------------------------------------------------------------- #
# multi-length generalization: rectangles of several allowed sizes at once
# --------------------------------------------------------------------------- #
@dataclass
class MultiSolution:
    value: float                  # optimal objective
    starts: np.ndarray            # sorted start indices of the chosen rectangles
    lengths: np.ndarray           # length of each chosen rectangle, same order as starts
    coverage: np.ndarray          # bool[T], covered slots
    n_rects: int = field(init=False)
    rects_by_length: dict = field(init=False, default_factory=dict)
    inside: int = field(init=False, default=0)   # 1s covered
    outside: int = field(init=False, default=0)  # 0s covered
    missed: int = field(init=False, default=0)   # 1s left uncovered

    def __post_init__(self) -> None:
        self.n_rects = int(len(self.starts))
        self.rects_by_length = {}
        for l in self.lengths.tolist():
            self.rects_by_length[l] = self.rects_by_length.get(l, 0) + 1

    def _fill_stats(self, demand: np.ndarray) -> "MultiSolution":
        d = demand.astype(bool)
        self.inside = int(np.count_nonzero(self.coverage & d))
        self.outside = int(np.count_nonzero(self.coverage & ~d))
        self.missed = int(np.count_nonzero(~self.coverage & d))
        return self


def _length_caps(T: int, lengths: list[int], max_rects: Optional[dict]) -> list[int]:
    max_rects = max_rects or {}
    caps = []
    for l in lengths:
        cap = max_rects.get(l)
        caps.append(max(0, (T // l) if cap is None else int(min(cap, T // l))))
    return caps


def _check_state_space(cells: int, what: str, limit: int) -> None:
    if cells > limit:
        raise ValueError(
            f"{what} state space is ~{cells:,} cells, over the {limit:,} guard "
            f"limit -- this grows multiplicatively with the number of distinct "
            f"lengths; tighten max_rects/totals or reduce the number of lengths"
        )


def _validate_lengths(T: int, lengths: Sequence[int]) -> list[int]:
    lengths = sorted(set(int(l) for l in lengths))
    if not lengths:
        raise ValueError("lengths must be non-empty")
    for l in lengths:
        if not (1 <= l <= T):
            raise ValueError(f"need 1 <= L <= T for every length, got L={l}, T={T}")
    return lengths


def solve_coverage_multi(
    demand: Sequence[float] | np.ndarray,
    lengths: Sequence[int],
    alpha: float = 1.0,
    max_rects: Optional[dict[int, Optional[int]]] = None,
) -> MultiSolution:
    """
    Generalization of solve_coverage to several candidate rectangle lengths
    at once (e.g. several shift durations), each with its own independent
    row-level cardinality cap. Rectangles of any length still may not
    overlap each other.

    lengths: distinct rectangle lengths to choose from.
    max_rects: optional {length: cap} dict, one entry per length -- a length
        missing from the dict, or mapped to None, defaults to its natural
        T // length cap (i.e. unconstrained beyond physical fit).

    F[t, k_1, .., k_m] = best objective using slots [0, t) with at most k_i
    rectangles of lengths[i] remaining unused, for the m distinct lengths:

        F[t, ...] = max( F[t-1, ...],
                          F[t-L_i, ..., k_i-1, ...] + w_i[t-L_i]  for each i
                          with t >= L_i and k_i >= 1 )

    Ties are broken towards *not* placing a rectangle (matching solve_coverage).

    Complexity: O(T * prod(K_i + 1) * m). Unlike the single-length DP, this
    grows multiplicatively with the number of distinct lengths -- keep caps
    modest.
    """
    d = np.asarray(demand, dtype=np.float64)
    T = d.size
    lengths = _validate_lengths(T, lengths)
    caps = _length_caps(T, lengths, max_rects)
    m = len(lengths)
    shape = tuple(k + 1 for k in caps)
    _check_state_space(T * int(np.prod(shape)), "solve_coverage_multi", limit=20_000_000)

    if all(k == 0 for k in caps):
        cov = np.zeros(T, dtype=bool)
        return MultiSolution(0.0, np.empty(0, dtype=int), np.empty(0, dtype=int), cov)._fill_stats(d)

    W = [rectangle_values(d, l, alpha).tolist() for l in lengths]

    F = np.zeros((T + 1,) + shape, dtype=np.float64)
    # take[t][state]: which length index ends a rectangle exactly at t for
    # that state, or -1 if the optimum there carries F[t-1] forward instead.
    take = np.full((T + 1,) + shape, -1, dtype=np.int8)

    for t in range(1, T + 1):
        F[t] = F[t - 1]
        for i, l in enumerate(lengths):
            if t < l or caps[i] == 0:
                continue
            wv = W[i][t - l]
            # slice axis i only: src drops its last index, dst drops its first,
            # so dst's k_i lines up with src's k_i - 1 (one rectangle consumed).
            src = tuple(slice(None, -1) if j == i else slice(None) for j in range(m))
            dst = tuple(slice(1, None) if j == i else slice(None) for j in range(m))
            cand = F[t - l][src] + wv
            cur = F[t][dst]
            better = cand > cur
            F[t][dst] = np.where(better, cand, cur)
            take[t][dst] = np.where(better, i, take[t][dst])

    # ---- backtrack -------------------------------------------------------- #
    starts: list[int] = []
    starts_len: list[int] = []
    state = list(caps)
    t = T
    while t > 0:
        i = int(take[(t,) + tuple(state)])
        if i == -1:
            t -= 1
        else:
            l = lengths[i]
            starts.append(t - l)
            starts_len.append(l)
            t -= l
            state[i] -= 1
    starts.reverse()
    starts_len.reverse()

    coverage = np.zeros(T, dtype=bool)
    for s, l in zip(starts, starts_len):
        coverage[s:s + l] = True

    value = float(F[(T,) + tuple(caps)])
    return MultiSolution(
        value, np.asarray(starts, dtype=int), np.asarray(starts_len, dtype=int), coverage
    )._fill_stats(d)


def solve_coverage_batch_multi(
    demand: np.ndarray,
    lengths: Sequence[int],
    alpha: float = 1.0,
    max_rects: Optional[dict[int, Optional[int]]] = None,
) -> np.ndarray:
    """
    Vectorised value curve for solve_coverage_multi, over N rows at once.

    Returns an (N, K_1+1, .., K_m+1) array, lengths sorted ascending: entry
    [i, k_1, .., k_m] is row i's best value using at most k_j rectangles of
    lengths[j] simultaneously, for every j -- the same joint DP as
    solve_coverage_multi, values only (no backtrack, so no starts/coverage).

    Feeds allocate_budget_multi when a *total* budget per length must also be
    shared across rows, on top of each row's own per-length cap set here.
    """
    D = np.asarray(demand, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("demand must be (N, T)")
    N, T = D.shape
    lengths = _validate_lengths(T, lengths)
    caps = _length_caps(T, lengths, max_rects)
    m = len(lengths)
    shape = tuple(k + 1 for k in caps)
    _check_state_space(T * N * int(np.prod(shape)), "solve_coverage_batch_multi", limit=50_000_000)

    if all(k == 0 for k in caps):
        return np.zeros((N,) + shape)

    score = D - alpha * (1.0 - D)
    csum = np.concatenate([np.zeros((N, 1)), np.cumsum(score, axis=1)], axis=1)
    W = [csum[:, l:] - csum[:, :-l] for l in lengths]     # each (N, T-l+1)

    F = np.zeros((T + 1, N) + shape, dtype=np.float64)
    for t in range(1, T + 1):
        F[t] = F[t - 1]
        for i, l in enumerate(lengths):
            if t < l or caps[i] == 0:
                continue
            wv = W[i][:, t - l].reshape((N,) + (1,) * m)
            src = (slice(None),) + tuple(slice(None, -1) if j == i else slice(None) for j in range(m))
            dst = (slice(None),) + tuple(slice(1, None) if j == i else slice(None) for j in range(m))
            cand = F[t - l][src] + wv
            np.maximum(F[t][dst], cand, out=F[t][dst])
    return F[T].copy()


# --------------------------------------------------------------------------- #
# pooled-budget allocation, generalized: a separate shared total per length
# --------------------------------------------------------------------------- #
@dataclass
class MultiAllocation:
    k: np.ndarray                 # int[N, m], rectangles of lengths[j] assigned to row i
    lengths: list                 # length values, matching k's columns and curves' axes order
    value: float                  # total objective, summed across rows
    per_row_value: np.ndarray     # float[N]


def allocate_budget_multi(
    curves: np.ndarray,
    lengths: Sequence[int],
    totals: dict[int, int],
    exact: bool = False,
) -> MultiAllocation:
    """
    Split a separate, shared *total* budget per length across N rows, on top
    of the per-row per-length caps already baked into `curves` (via
    solve_coverage_batch_multi(..., max_rects=...) -- that's the "row-level
    maximum"; `totals` here is the "total constraint", both per length).

    curves: (N, K_1+1, .., K_m+1) from solve_coverage_batch_multi -- a row's
        choice for one length still competes for the same time slots as its
        other lengths, so that coupling already lives inside `curves`.
    lengths: length values in the same order as curves' trailing axes (pass
        the same sorted list solve_coverage_batch_multi was called with).
    totals: {length: pooled total across all rows}. A length missing from
        totals is left uncapped at the batch level (every row may use up to
        its own row-level cap for that length, independently).

    exact=False (default): heap-based greedy, one rectangle at a time,
        awarding whichever (row, length) increment currently has the best
        marginal gain. Unlike the single-length allocate_budget, optimality
        isn't guaranteed even when curves look concave along each axis --
        lengths can interact within a row (taking one rectangle of length A
        can change the marginal value of length B), which breaks the clean
        exchange argument. Still always feasible; good default for exploring.
    exact=True: exact DP over rows and the combined per-length remaining
        budget. The state space multiplies over every length that has a
        total in `totals`, so keep this to a couple of lengths and modest
        totals -- it's pure Python.
    """
    curves = np.asarray(curves, dtype=np.float64)
    lengths = list(lengths)
    m = len(lengths)
    if curves.ndim != m + 1:
        raise ValueError(
            f"curves must have {m + 1} axes -- (N, K_1+1, .., K_{m}+1) -- to "
            f"match `lengths` ({m} of them); got shape {curves.shape}"
        )
    N = curves.shape[0]
    caps = [s - 1 for s in curves.shape[1:]]  # per-row max index along each length axis
    totals_arr = [
        max(0, int(min(totals.get(l, N * caps[j]), N * caps[j])))
        for j, l in enumerate(lengths)
    ]

    if exact:
        ops = N * int(np.prod([t + 1 for t in totals_arr])) * int(np.prod([c + 1 for c in caps]))
        _check_state_space(ops, "allocate_budget_multi(exact=True)", limit=2_000_000)
        k = _allocate_multi_exact(curves, caps, totals_arr)
    else:
        k = _allocate_multi_greedy(curves, caps, totals_arr)

    idx = (np.arange(N),) + tuple(k[:, j] for j in range(m))
    per_row_value = curves[idx]
    return MultiAllocation(k=k, lengths=lengths, value=float(per_row_value.sum()), per_row_value=per_row_value)


def solve_allocated_multi(
    demand: np.ndarray,
    alpha: float,
    alloc: MultiAllocation,
) -> list[MultiSolution]:
    """
    Turn a MultiAllocation into the actual per-row solutions (starts/lengths/
    coverage) -- same idea as solve_allocated, generalized to a per-length
    cap per row. Re-solves each row with alloc.k[i] as its per-length cap.
    """
    D = np.asarray(demand, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("demand must be (N, T)")
    if alloc.k.shape[0] != D.shape[0]:
        raise ValueError(f"alloc has {alloc.k.shape[0]} rows, demand has {D.shape[0]}")
    sols = []
    for row, k_row in zip(D, alloc.k):
        caps = {l: int(k_row[j]) for j, l in enumerate(alloc.lengths)}
        sols.append(solve_coverage_multi(row, alloc.lengths, alpha, max_rects=caps))
    return sols


def _allocate_multi_greedy(curves: np.ndarray, caps: list[int], totals: list[int]) -> np.ndarray:
    N = curves.shape[0]
    m = len(caps)
    k = np.zeros((N, m), dtype=int)
    remaining = list(totals)
    version = [0] * N  # bumped whenever row i's state changes, to invalidate stale heap entries

    heap: list[tuple[float, int, int, int]] = []

    def push_all(i: int) -> None:
        version[i] += 1
        v = version[i]
        state = tuple(k[i])
        for j in range(m):
            if state[j] < caps[j] and remaining[j] > 0:
                nxt = list(state)
                nxt[j] += 1
                gain = float(curves[i][tuple(nxt)] - curves[i][state])
                heapq.heappush(heap, (-gain, i, j, v))

    for i in range(N):
        push_all(i)

    while heap:
        neg_gain, i, j, v = heapq.heappop(heap)
        if v != version[i]:
            continue  # row's state moved (another axis advanced) since this was pushed
        if -neg_gain <= 0:
            break  # best remaining increment anywhere isn't worth taking
        if remaining[j] <= 0:
            continue  # this length's pooled budget is exhausted; try the next candidate
        k[i, j] += 1
        remaining[j] -= 1
        push_all(i)  # re-derive fresh marginal gains for every axis of row i

    return k


def _allocate_multi_exact(curves: np.ndarray, caps: list[int], totals: list[int]) -> np.ndarray:
    N = curves.shape[0]
    m = len(caps)
    budget_shape = tuple(t + 1 for t in totals)
    row_choices = list(itertools.product(*(range(c + 1) for c in caps)))
    NEG = float("-inf")

    dp = np.full(budget_shape, NEG)          # dp[b]: best value so far, budget vector b spent
    dp[(0,) * m] = 0.0
    choice_per_row: list[dict[tuple, tuple]] = []  # choice[i][b]: local (k_1,..,k_m) picked for row i

    for i in range(N):
        new_dp = np.full(budget_shape, NEG)
        row_choice: dict[tuple, tuple] = {}
        row_curve = curves[i]
        it = np.nditer(dp, flags=["multi_index"])
        for val in it:
            prev = float(val)
            if prev == NEG:
                continue
            b = it.multi_index
            for combo in row_choices:
                nb = tuple(b[j] + combo[j] for j in range(m))
                if any(nb[j] > totals[j] for j in range(m)):
                    continue
                cand = prev + float(row_curve[combo])
                if cand > new_dp[nb]:
                    new_dp[nb] = cand
                    row_choice[nb] = combo
        dp = new_dp
        choice_per_row.append(row_choice)

    best_state = np.unravel_index(int(np.argmax(dp)), dp.shape)

    k = np.zeros((N, m), dtype=int)
    state = best_state
    for i in range(N - 1, -1, -1):
        combo = choice_per_row[i][state]
        k[i] = combo
        state = tuple(state[j] - combo[j] for j in range(m))
    return k


# --------------------------------------------------------------------------- #
# reference implementation for testing (exponential — small T only)
# --------------------------------------------------------------------------- #
def brute_force(demand: Sequence[float], L: int, alpha: float,
                max_rects: Optional[int] = None) -> float:
    from itertools import combinations
    d = np.asarray(demand, dtype=np.float64)
    T = d.size
    w = rectangle_values(d, L, alpha)
    positions = range(T - L + 1)
    K = (T // L) if max_rects is None else min(max_rects, T // L)
    best = 0.0
    for k in range(1, K + 1):
        for combo in combinations(positions, k):
            if any(b - a < L for a, b in zip(combo, combo[1:])):
                continue
            best = max(best, float(w[list(combo)].sum()))
    return best


def brute_force_multi(demand: Sequence[float], lengths: Sequence[int], alpha: float,
                       max_rects: Optional[dict] = None) -> float:
    """Reference implementation for testing solve_coverage_multi -- exponential, small T only."""
    d = np.asarray(demand, dtype=np.float64)
    T = d.size
    lengths = _validate_lengths(T, lengths)
    cap_map = dict(zip(lengths, _length_caps(T, lengths, max_rects)))

    candidates = []  # (start, length, value)
    for l in lengths:
        w = rectangle_values(d, l, alpha)
        for s in range(T - l + 1):
            candidates.append((s, l, float(w[s])))

    best = 0.0
    n = len(candidates)
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            items = sorted(candidates[idx] for idx in combo)
            if any(b[0] < a[0] + a[1] for a, b in zip(items, items[1:])):
                continue
            counts: dict = {}
            for _, l, _ in items:
                counts[l] = counts.get(l, 0) + 1
            if any(counts.get(l, 0) > cap_map[l] for l in counts):
                continue
            best = max(best, sum(v for _, _, v in items))
    return best


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
def _fmt(t: int) -> str:
    return f"{(t * 15) // 60:02d}:{(t * 15) % 60:02d}"


def pretty(demand: Iterable[int], sol: Solution, L: int, alpha: float) -> str:
    d = np.asarray(list(demand), dtype=int)
    lines = [
        f"L = {L} slots ({L * 15} min), alpha = {alpha}",
        "demand : " + "".join("#" if x else "." for x in d),
        "cover  : " + "".join("=" if x else " " for x in sol.coverage),
        f"objective = {sol.value:.3f} | rectangles = {sol.n_rects} | "
        f"1s covered = {sol.inside} | 0s covered = {sol.outside} | 1s missed = {sol.missed}",
        "shifts : " + ", ".join(f"{_fmt(s)}-{_fmt(s + L)}" for s in sol.starts),
    ]
    return "\n".join(lines)
