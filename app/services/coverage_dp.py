import heapq
import itertools
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import logging

TIE_EPS = 1e-9

# utilities
def _check_demand_range(d: np.ndarray) -> None:
    """Demand is a load in [0, 1]: 1 = slot saturates the row, 0 = nobody needs it."""
    if d.size and (d.min() < -TIE_EPS or d.max() > 1.0 + TIE_EPS):
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

# ---------------------------------------------------------------------------
# single L algorithm
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
    wl = w.tolist()

    # Cap on rectangles: at most T // L can physically fit; max_rects tightens that.
    K = (T // L) if max_rects is None else int(min(max_rects, T // L))
    if K <= 0:
        cov = np.zeros(T, dtype=bool)
        return Solution(0.0, np.empty(0, dtype=int), cov)._fill_stats(d)

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

def solve_coverage_batch(
    demand: np.ndarray,
    L: int,
    alpha: float,
    max_rects: Optional[int] = None,
    allow_overhang: bool = True,
) -> np.ndarray:
    """
    For each row compute the best achievable value for every number of shifts k = 0..K

    allow_overhang=True (default): demand is extended with L-1 empty slots on the right first.

    returns the full (N, K+1) value curve, where column k is each row's optimal value using at most k rectangles (0 <= k <= K). 
    This curve is what allocate_budget needs to split a single, shared rectangle budget across 
    rows instead of giving every row its own independent cap.
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
        return np.zeros((N, 1))

    F = np.zeros((T + 1, N, K + 1), dtype=np.float64)
    for t in range(1, T + 1):
        F[t] = F[t - 1]
        if t >= L:
            cand = F[t - L, :, :-1] + W[:, t - L, None]
            np.maximum(F[t, :, 1:], cand, out=F[t, :, 1:])
    return F[T].copy()

@dataclass
class Allocation:
    k: np.ndarray               # int[N], rectangles assigned to each row
    value: float                 # total objective, summed across rows
    per_row_value: np.ndarray    # float[N], each row's value at its assigned k

def allocate_budget(curves: np.ndarray, K_total: int, exact: bool = False) -> Allocation:
    """
    Split the shared budget K_total among the rows by maximising the sum of their curves

    curves is the (N, K+1) array from solve_coverage_batch

    exact=False (default): greedy over marginal gains curves[i,k]-curves[i,k-1], always awarding 
    the next rectangle to whichever row's next unit is currently most valuable. 
    This is optimal if every row's curve is concave (marginal gains non-increasing in k)
    exact=True: exact DP over (rows, budget spent), O(N * K_total * K).
    """
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError("curves must be (N, K+1)")
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
        if -neg_gain <= TIE_EPS:
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

# ---------------------------------------------------------------------------
# multi lengths algorithm

Caps = Optional[Union[int, dict]]

@dataclass
class MultiSolution:
    """Same as Solution, plus the length of every rectangle."""
    value: float                  # optimal objective = inside - alpha * outside
    starts: np.ndarray            # sorted start indices of the chosen rectangles
    lengths: np.ndarray           # length of each chosen rectangle, same order as starts
    coverage: np.ndarray          # bool[T_ext], covered slots
    n_rects: int = field(init=False)
    rects_by_length: dict = field(init=False, default_factory=dict)
    inside: float = field(init=False, default=0.0)    # served demand:      sum(d * cov)
    outside: float = field(init=False, default=0.0)   # idle capacity paid: sum((1 - d) * cov)
    missed: float = field(init=False, default=0.0)    # demand uncovered:   sum(d * ~cov)

    def __post_init__(self) -> None:
        self.n_rects = int(len(self.starts))
        self.rects_by_length = {}
        for l in self.lengths.tolist():
            self.rects_by_length[l] = self.rects_by_length.get(l, 0) + 1

    def _fill_stats(self, demand: np.ndarray) -> "MultiSolution":
        d = np.asarray(demand, dtype=np.float64)
        cov = self.coverage
        self.inside = float(d[cov].sum())
        self.outside = float((1.0 - d)[cov].sum())
        self.missed = float(d[~cov].sum())
        return self

#preprocessing
def _normalize_lengths(lengths: Sequence[int]) -> list:
    ls = sorted({int(l) for l in lengths})
    if not ls:
        raise ValueError("lengths must be non-empty")
    if ls[0] < 1:
        raise ValueError(f"every length must be >= 1, got {ls}")
    return ls

def _row_caps(T_ext: int, lengths: list) -> list:
    """Per-row cap of every length: how many fit in the padded horizon."""
    caps = []
    for l in lengths:
        caps.append(T_ext // l)
    return caps

def _prepare(D: np.ndarray, lengths: Sequence[int], allow_overhang: bool):
    """D (N, T) -> D_ext (N, T_ext), sorted lengths, T_ext, number of valid starts per length, row caps."""
    lengths = _normalize_lengths(lengths)
    T = D.shape[1]
    if not allow_overhang and lengths[-1] > T:
        raise ValueError(f"need every length <= T when allow_overhang=False, got {lengths[-1]} > {T}")
    pad = lengths[-1] - 1 if allow_overhang else 0
    D_ext = np.pad(D, ((0, 0), (0, pad)))
    T_ext = T + pad
    # overhang: a rectangle may start at any real bin (0..T-1); otherwise it must fit in the day
    n_starts = [T if allow_overhang else T - l + 1 for l in lengths]
    caps = _row_caps(T_ext, lengths)
    return D_ext, lengths, T_ext, n_starts, caps

def _window_values(D_ext: np.ndarray, lengths: list, alpha: float, n_starts: list) -> list:
    """W[i][r, s] = value of a rectangle of lengths[i] starting at s on row r:  sum_t ( d - alpha (1 - d) )."""
    score = D_ext - alpha * (1.0 - D_ext)
    csum = np.concatenate([np.zeros((D_ext.shape[0], 1)), np.cumsum(score, axis=1)], axis=1)
    return [csum[:, l:l + n] - csum[:, :n] for l, n in zip(lengths, n_starts)]

def _axis_slices(m: int, i: int, lead: int = 0):
    """Slices that align k_i (destination) with k_i - 1 (source) along axis i of the state."""
    pre = (slice(None),) * lead
    src = pre + tuple(slice(None, -1) if j == i else slice(None) for j in range(m))
    dst = pre + tuple(slice(1, None) if j == i else slice(None) for j in range(m))
    return src, dst

def _check_state_space(cells: int, what: str, limit: int) -> None:
    if cells > limit:
        raise ValueError(
            f"{what}: state space is ~{cells:,} cells, over the {limit:,} guard. It grows multiplicatively "
            f"with the number of lengths: use fewer lengths"
        )

# one row DP
def solve_coverage_multi(
    demand: Sequence[float] | np.ndarray,
    lengths: Sequence[int],
    alpha: float,
    max_rects: Optional[dict] = None,  # {length: cap}: used by solve_allocated_multi
    allow_overhang: bool = True,
) -> MultiSolution:
    """
    Exact DP: place rectangles of the given lengths on one demand vector, disjoint from each other, maximizing the objective.

    F[t, k_1..k_m] = best objective using slots [0, t) with at most k_i rectangles of lengths[i]:

        F[t, k] = max( F[t-1, k],                                      # nothing ends at t
                       F[t-l_i, k - e_i] + w_i[t-l_i]  for each i )    # a rectangle of length l_i ends at t

    Ties go to not placing a rectangle, so a rectangle of value <= 0 is never used.
    """
    d = np.asarray(demand, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("demand must be 1-D; use solve_coverage_batch_multi for many rows")
    _check_demand_range(d)
    D, lengths, T, n_starts, caps = _prepare(d[None, :], lengths, allow_overhang)

    if max_rects is not None:
            unknown = set(max_rects) - set(lengths)
            if unknown:
                raise ValueError(f"max_rects has lengths {sorted(unknown)} that are not in lengths={lengths}")
            caps = [max(0, min(c, int(max_rects.get(l, c)))) for l, c in zip(lengths, caps)]

    m = len(lengths)
    shape = tuple(c + 1 for c in caps)
    _check_state_space(T * int(np.prod(shape)), "solve_coverage_multi", limit=20_000_000)

    if all(c == 0 for c in caps):
        cov = np.zeros(T, dtype=bool)
        return MultiSolution(0.0, np.empty(0, dtype=int), np.empty(0, dtype=int), cov)._fill_stats(D[0])

    W = _window_values(D, lengths, alpha, n_starts)

    F = np.zeros((T + 1,) + shape, dtype=np.float64)
    # take[t][state]: index of the length that ends a rectangle at t in that state, -1 = carry F[t-1]
    take = np.full((T + 1,) + shape, -1, dtype=np.int8)
    moves = [(i, l, *_axis_slices(m, i)) for i, l in enumerate(lengths) if caps[i] > 0]

    for t in range(1, T + 1):
        F[t] = F[t - 1]
        for i, l, src, dst in moves:
            if t < l or t - l >= n_starts[i]:
                continue
            cand = F[t - l][src] + W[i][0, t - l]
            cur = F[t][dst]                       # views: writing into them updates F[t] / take[t]
            better = cand > cur
            if better.any():
                cur[better] = cand[better]
                tk = take[t][dst]
                tk[better] = i

    # backtrack from the final state (T, caps)
    starts, starts_len = [], []
    state = list(caps)
    t = T
    while t > 0:
        i = int(take[(t,) + tuple(state)])
        if i < 0:
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
    )._fill_stats(D[0])

# compute values for many rows at once
def solve_coverage_batch_multi(
    demand: np.ndarray,
    lengths: Sequence[int],
    alpha: float,
    allow_overhang: bool = True,
) -> np.ndarray:
    """
    Compute the best coverage values for multiple rows simultaneously.
    Returns curves of shape (N, K_1+1, .., K_m+1), lengths sorted ascending: entry [r, k_1, .., k_m] is the
    best value of row r using at most k_i rectangles of lengths[i], for every i at once.
    This is the input of allocate_budget_multi
    """
    D = np.asarray(demand, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("demand must be (N, T)")
    _check_demand_range(D)
    D, lengths, T, n_starts, caps = _prepare(D, lengths, allow_overhang)
    N, m = D.shape[0], len(lengths)
    shape = tuple(c + 1 for c in caps)
    P = lengths[-1] + 1                                   # ring buffer: only F[t-l], l <= max length, is needed
    _check_state_space(P * N * int(np.prod(shape)), "solve_coverage_batch_multi", limit=50_000_000)

    if all(c == 0 for c in caps):
        return np.zeros((N,) + shape)

    W = _window_values(D, lengths, alpha, n_starts)
    moves = [(i, l, *_axis_slices(m, i, lead=1)) for i, l in enumerate(lengths) if caps[i] > 0]

    buf = np.zeros((P, N) + shape, dtype=np.float64)      # buf[t % P] holds F[t]
    for t in range(1, T + 1):
        cur = buf[t % P]
        cur[...] = buf[(t - 1) % P]
        for i, l, src, dst in moves:
            if t < l or t - l >= n_starts[i]:
                continue
            wv = W[i][:, t - l].reshape((N,) + (1,) * m)
            cand = buf[(t - l) % P][src] + wv
            view = cur[dst]
            np.maximum(view, cand, out=view)
    return buf[T % P].copy()

# pooled-budget allocation (greedy)
@dataclass
class MultiAllocation:
    k: np.ndarray                 # int[N, m]: shifts of lengths[j] given to row i
    lengths: list                 # length values = columns of k = trailing axes of the curves
    value: float                  # total objective, summed across rows
    per_row_value: np.ndarray     # float[N]

    @property
    def n_total(self) -> int:
        return int(self.k.sum())

    @property
    def by_length(self) -> dict:
        return {l: int(self.k[:, j].sum()) for j, l in enumerate(self.lengths)}

def allocate_budget_multi(
    curves: np.ndarray,
    lengths: Sequence[int],
    K_total: Optional[int] = None,
    length_totals: Optional[dict] = None,
) -> MultiAllocation:
    """
    Split the shared budget among the rows, greedily.

    curves        : (N, K_1+1, .., K_m+1) from solve_coverage_batch_multi (row-level caps are already inside)
    lengths       : the same sorted list used to build the curves
    K_total       : max number of shifts, of any length, across all rows (None = no limit)
    length_totals : {length: max number of shifts of that length across all rows}; missing = no limit

    One shift at a time goes to the (row, length) whose next increment has the highest marginal gain,
    stopping when the budget is over, a length is exhausted, or the best gain is not worth taking.
    Always feasible, not guaranteed optimal: V_r is in general not concave when lengths differ, and the greedy
    never swaps (e.g. two short shifts -> one long one). An increment that only pays together with another
    one (gain 0 now, positive after the next) is never found.
    """
    curves = np.asarray(curves, dtype=np.float64)
    lengths = _normalize_lengths(lengths)
    m = len(lengths)
    if curves.ndim != m + 1:
        raise ValueError(f"curves must have {m + 1} axes (N, K_1+1, .., K_{m}+1) for {m} lengths; got {curves.shape}")

    N = curves.shape[0]
    caps = [s - 1 for s in curves.shape[1:]]
    length_totals = length_totals or {}
    unknown = set(length_totals) - set(lengths)
    if unknown:
        raise ValueError(f"length_totals has lengths {sorted(unknown)} that are not in lengths={lengths}")

    remaining = [max(0, int(min(length_totals.get(l, N * caps[j]), N * caps[j]))) for j, l in enumerate(lengths)]
    budget_left = sum(remaining) if K_total is None else max(0, min(int(K_total), sum(remaining)))

    k = np.zeros((N, m), dtype=int)
    version = [0] * N                     # bumped when row i changes: invalidates its old heap entries
    heap: list = []

    def push_all(i: int) -> None:
        version[i] += 1
        state = tuple(int(x) for x in k[i])
        base = curves[(i,) + state]
        for j in range(m):
            if state[j] < caps[j] and remaining[j] > 0:
                nxt = state[:j] + (state[j] + 1,) + state[j + 1:]
                heapq.heappush(heap, (-float(curves[(i,) + nxt] - base), i, j, version[i]))

    for i in range(N):
        push_all(i)

    while heap and budget_left > 0:
        neg_gain, i, j, v = heapq.heappop(heap)
        if v != version[i]:
            continue                      # stale: row i moved since this entry was pushed
        if -neg_gain <= TIE_EPS:
            break                         # the best remaining increment is not worth taking
        if remaining[j] <= 0:
            continue                      # this length is exhausted
        k[i, j] += 1
        remaining[j] -= 1
        budget_left -= 1
        push_all(i)

    per_row_value = np.array([curves[(i,) + tuple(int(x) for x in k[i])] for i in range(N)])
    return MultiAllocation(k=k, lengths=lengths, value=float(per_row_value.sum()), per_row_value=per_row_value)

# MILP allocation
def allocate_milp(
    curves: np.ndarray,
    lengths: Sequence[int],
    n_days: int = 1,
    n_layers: Optional[int] = None,
    K_total: Optional[int] = None,
    length_per_day: Optional[dict] = None,
    K_day: Optional[Union[int, dict, Sequence[int]]] = None,
    time_limit: Optional[float] = None,
) -> MultiAllocation:
    """
    max  sum_{n,r} sum_{k in K*[n,r]} V[n,r,k] z[n,r,k]
    s.t. sum_{k in K*[n,r]} z[n,r,k] = 1                    for every day n, layer r
         sum_{n,r,k} |k|   z[n,r,k] <= K_total              (optional)
         sum_{r,k}   |k|   z[n,r,k] <= K_day[n]             for every day n (optional)
         sum_{r,k}   k_j   z[n,r,k] <= length_per_day[l_j]  for every day n and length j (optional)
    K*[n,r] = efficient states of the row: V(k) > V(k - e_j) for every j with k_j >= 1.
 
    n_days : number of days (default 1: the rows of curves are then the layers of a single day)
    n_layers : layers per day; default = curves.shape[0] // n_days
    curves : (n_days * n_layers, K_1+1, .., K_m+1), rows ordered day by day: row = n * n_layers + r
    K_day  : int (same for all days), dict {n: cap} or sequence indexed by n; a day missing from a dict = no limit
    length_per_day : {length: max number of shifts of that length on each day}, the same cap for every day
    Days n = 0..n_days-1 and layers r = 0..n_layers-1. Returns the same MultiAllocation as allocate_budget_multi
    (k has one row per (n, r), in the same order as curves).
    """
    curves = np.asarray(curves, dtype=np.float64)
    lengths = _normalize_lengths(lengths)
    m_len = len(lengths)
    if n_layers is None:
        if curves.shape[0] % n_days:
            raise ValueError(f"curves has {curves.shape[0]} rows, not a multiple of n_days={n_days}")
        n_layers = curves.shape[0] // n_days
    if curves.ndim != m_len + 1:
        raise ValueError(f"curves must have {m_len + 1} axes; got {curves.shape}")
    if curves.shape[0] != n_days * n_layers:
        raise ValueError(f"curves has {curves.shape[0]} rows, expected n_days * n_layers = {n_days * n_layers}")
    length_per_day = length_per_day or {}
    unknown = set(length_per_day) - set(lengths)
    if unknown:
        raise ValueError(f"length_per_day has lengths {sorted(unknown)} that are not in lengths={lengths}")
 
    # efficient states of every row: every window of the state must improve the value
    eff = np.ones(curves.shape, dtype=bool)
    for j in range(m_len):
        ax = 1 + j
        hi = [slice(None)] * curves.ndim; hi[ax] = slice(1, None)
        lo = [slice(None)] * curves.ndim; lo[ax] = slice(0, -1)
        gain = np.full(curves.shape, np.inf)                      # k_j = 0: nothing to remove
        gain[tuple(hi)] = curves[tuple(hi)] - curves[tuple(lo)]
        eff &= gain > TIE_EPS
    states = {}                                                   # (n, r) -> efficient states (tuples)
    for n in range(n_days):
        for r in range(n_layers):
            row = n * n_layers + r
            states[n, r] = [tuple(int(x) for x in s) for s in np.argwhere(eff[row])]
    V = {(n, r, kap): float(curves[(n * n_layers + r,) + kap]) for (n, r), ks in states.items() for kap in ks}
 
    m = pyo.ConcreteModel()
    m.N = pyo.RangeSet(0, n_days - 1)                  # days
    m.R = pyo.RangeSet(0, n_layers - 1)                # layers
    m.I = pyo.Set(initialize=[(n, r) + kap for (n, r), ks in states.items() for kap in ks], dimen=m_len + 2)
    m.z = pyo.Var(m.I, domain=pyo.Binary)              # z[n, r, k_1, .., k_m]
 
    m.obj = pyo.Objective(
        expr=sum(V[n, r, kap] * m.z[(n, r) + kap] for (n, r), ks in states.items() for kap in ks),
        sense=pyo.maximize)
 
    m.one = pyo.Constraint(m.N, m.R, rule=lambda m, n, r: sum(m.z[(n, r) + kap] for kap in states[n, r]) == 1)
 
    if K_total is not None:
        m.total = pyo.Constraint(
            expr=sum(sum(kap) * m.z[(n, r) + kap] for (n, r), ks in states.items() for kap in ks)
            <= max(0, int(K_total)))
 
    if length_per_day:
        m.JD = pyo.Set(initialize=[j for j, l in enumerate(lengths) if l in length_per_day])
        m.per_day_length = pyo.Constraint(
            m.N, m.JD,
            rule=lambda m, n, j: sum(kap[j] * m.z[(n, r) + kap] for r in m.R for kap in states[n, r])
            <= max(0, int(length_per_day[lengths[j]])))
 
    if K_day is not None:
        cap_of = (lambda n: K_day.get(n)) if isinstance(K_day, dict) else \
                 (lambda n: K_day) if np.isscalar(K_day) else (lambda n: K_day[n])
        m.D = pyo.Set(initialize=[n for n in range(n_days) if cap_of(n) is not None])
        m.per_day = pyo.Constraint(
            m.D, rule=lambda m, n: sum(sum(kap) * m.z[(n, r) + kap] for r in m.R for kap in states[n, r])
            <= max(0, int(cap_of(n))))
 
    opt = pyo.SolverFactory("appsi_highs")
    opt.config.stream_solver = False
    opt.config.mip_gap = 0.0
    if time_limit:
        opt.config.time_limit = float(time_limit)
    opt.highs_options["output_flag"] = False
    quiet = logging.getLogger("highs_quiet")
    quiet.setLevel(logging.CRITICAL)
    opt.config.solver_output_logger = quiet
    res = opt.solve(m)
 
    if hasattr(res, "termination_condition") and \
            res.termination_condition not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit):
        raise RuntimeError(f"MILP failed: {res.termination_condition}")
 
    # chosen state of every row (already minimal: only efficient states exist)
    k = np.zeros((n_days * n_layers, m_len), dtype=int)
    for n in range(n_days):
        for r in range(n_layers):
            kap = max(states[n, r], key=lambda s: pyo.value(m.z[(n, r) + s]))
            k[n * n_layers + r] = kap
 
    per_row_value = np.array([curves[(i,) + tuple(int(x) for x in k[i])] for i in range(len(k))])
    return MultiAllocation(k=k, lengths=lengths, value=float(per_row_value.sum()), per_row_value=per_row_value)

def solve_allocated_multi(
    demand: np.ndarray,
    alpha: float,
    alloc: MultiAllocation,
    allow_overhang: bool = True,
) -> list:
    """
    Re-solve every row with its own number of shifts per length to recover starts and lengths.
    Use the same allow_overhang as the curves. The shifts really placed may be fewer than alloc.k:
    count them with sol.n_rects / sol.rects_by_length.
    """
    D = np.asarray(demand, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("demand must be (N, T)")
    if alloc.k.shape[0] != D.shape[0]:
        raise ValueError(f"alloc has {alloc.k.shape[0]} rows, demand has {D.shape[0]}")
    return [
        solve_coverage_multi(row, alloc.lengths, alpha,
                             max_rects={l: int(k_row[j]) for j, l in enumerate(alloc.lengths)},
                             allow_overhang=allow_overhang)
        for row, k_row in zip(D, alloc.k)
    ]

# report
def coverage_report_multi(D, sols, time_values, row_labels=None, alpha=None) -> CoverageReport:
    """
    Same report as coverage_report, with a length per block.

    D    : (n_rows, T) demand grid in [0, 1] (not padded: the overhang bins have zero demand)
    sols : list of MultiSolution for the first len(sols) rows (the rows with a server); the remaining
           rows have no server: demand with 0 blocks, 0 coverage
    time_values : (T,) datetime-like, start time of each bin
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
        for b, (s0, l) in enumerate(zip(sol.starts, sol.lengths)):
            s0, l = int(s0), int(l)
            covered = float(D[c, s0:s0 + l].sum())
            rec = dict(row=c, block=b, length=l, start_slot=s0, end_slot=s0 + l,
                       start_time=tv.iloc[s0], end_time=tv.iloc[s0] + l * step,
                       covered_demand=covered, idle_capacity=l - covered, utilization=covered / l)
            if with_obj:
                rec["objective"] = covered - alpha * (l - covered)
            recs.append(rec)
    blocks = pd.DataFrame(recs)
    if blocks.empty:
        cols = ["row", "block", "length", "start_slot", "end_slot", "covered_demand", "idle_capacity", "utilization"]
        blocks = pd.DataFrame(columns=cols + (["objective"] if with_obj else []))

    n_rows = D.shape[0]
    by_row = lambda col, fill: blocks.groupby("row")[col].sum().reindex(range(n_rows), fill_value=fill).to_numpy()
    covered_by_row = by_row("covered_demand", 0.0).astype(float)
    capacity = by_row("length", 0).astype(float)
    n_blocks = blocks.groupby("row").size().reindex(range(n_rows), fill_value=0).to_numpy()

    rows = pd.DataFrame({
        "n_blocks": n_blocks,
        "demand": row_demand,
        "covered_demand": covered_by_row,
        "missed_demand": np.clip(row_demand - covered_by_row, 0.0, None),
        "idle_capacity": capacity - covered_by_row,
        "coverage": [safe(cv, dm) for cv, dm in zip(covered_by_row, row_demand)],
        "utilization": [safe(cv, cp) for cv, cp in zip(covered_by_row, capacity)],
    }).rename_axis("row")
    if with_obj:
        rows["objective"] = rows["covered_demand"] - alpha * rows["idle_capacity"]
    rows.insert(0, "server", list(row_labels) if row_labels is not None else [None] * n_rows)

    covered_tot, cap_tot = float(covered_by_row.sum()), float(capacity.sum())
    total = {
        "n_blocks": int(n_blocks.sum()),
        "demand": total_demand,
        "covered_demand": covered_tot,
        "missed_demand": max(total_demand - covered_tot, 0.0),
        "idle_capacity": cap_tot - covered_tot,
        "coverage": safe(covered_tot, total_demand),
    }
    if with_obj:
        total["objective"] = covered_tot - alpha * (cap_tot - covered_tot)
    for l, n in blocks.groupby("length").size().items():          # how many shifts of each length
        total[f"n_blocks_{int(l)}"] = int(n)
    return CoverageReport(blocks=blocks, rows=rows, total=pd.Series(total))


