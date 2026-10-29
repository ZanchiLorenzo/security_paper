"""
Coverage DP and budget allocation for the lane opening windows.
 
Pipeline (several window lengths: the one SecurityPlanner.run_multi uses). 
D is the demand grid (rows, T), row r = r-th lane, values in [0, 1]:
 
    solve_coverage_batch_multi   value curve V_r(k) of every row (DP, all rows at once)
    allocate_milp                how many windows of each length every row gets (MILP over the curves)
    solve_allocated_multi        where the windows of every row are (DP + backtracking, rows in order)
    coverage_report_multi        blocks / rows / total tables
 
A window of length l starting in slot s is worth  sum_t (d_t - theta)  over its slots: the demand it covers minus theta for
every slot in which the lane is open (theta = minimum utilization that justifies opening it). 
Everything uses lengths in bins.
"""
import heapq
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union
 
import numpy as np
import pandas as pd
import pyomo.environ as pyo
 
TIE_EPS = 1e-9

# data classes
@dataclass
class MultiSolution:
    """Windows chosen on one row: starts (sorted), length of each, and the slots they cover."""
    value: float                  # optimal objective of the row: covered - theta * capacity
    starts: np.ndarray            # sorted start slots of the windows
    lengths: np.ndarray           # length of each window, same order as starts
    coverage: np.ndarray          # bool[T]: slots in which the row is open
    inside: float = field(init=False, default=0.0)    # served demand:      sum(d * cov)
    outside: float = field(init=False, default=0.0)   # idle capacity paid: sum((1 - d) * cov)
    missed: float = field(init=False, default=0.0)    # demand uncovered:   sum(d * ~cov)
 
    @property
    def n_rects(self) -> int:
        return int(len(self.starts))
 
    @property
    def rects_by_length(self) -> dict:
        return {int(l): int((self.lengths == l).sum()) for l in np.unique(self.lengths)}
 
    def _fill_stats(self, demand: np.ndarray) -> "MultiSolution":
        d = np.asarray(demand, dtype=np.float64)
        self.inside = float(d[self.coverage].sum())
        self.outside = float((1.0 - d)[self.coverage].sum())
        self.missed = float(d[~self.coverage].sum())
        return self
 
@dataclass
class MultiAllocation:
    k: np.ndarray                 # int[N, m]: windows of lengths[j] given to row i
    lengths: list                 # length values = columns of k = trailing axes of the curves
    value: float                  # total objective, summed across rows
    per_row_value: np.ndarray     # float[N]
 
    @property
    def n_total(self) -> int:
        return int(self.k.sum())
 
    @property
    def by_length(self) -> dict:
        return {l: int(self.k[:, j].sum()) for j, l in enumerate(self.lengths)}
 
@dataclass
class CoverageReport:
    blocks: pd.DataFrame          # one line per window
    rows: pd.DataFrame            # one line per row
    total: pd.Series              # whole day(s)
 
# helpers
def _check_demand_range(d: np.ndarray) -> None:
    """Demand is a load in [0, 1]: 1 = the slot saturates the row, 0 = nobody needs it."""
    if d.size and (d.min() < -TIE_EPS or d.max() > 1.0 + TIE_EPS):
        raise ValueError("demand must lie in [0, 1]")
 
def _normalize_lengths(lengths: Sequence[int]) -> list:
    ls = sorted({int(l) for l in lengths})
    if not ls:
        raise ValueError("lengths must be non-empty")
    if ls[0] < 1:
        raise ValueError(f"every length must be >= 1, got {ls}")
    return ls
 
def _theta_rows(theta, n_rows: int) -> np.ndarray:
    """theta (None = 0, a number, or one value per row) -> float array (n_rows,), checked to lie in [0, 1]."""
    th = np.asarray(0.0 if theta is None else theta, dtype=np.float64)
    if th.ndim > 1 or (th.ndim == 1 and th.size != n_rows) or not np.isfinite(th).all() or (th < 0).any() or (th > 1).any():
        raise ValueError("theta must be a number in [0, 1] or one such value per row")
    return np.broadcast_to(th, (n_rows,))
 
def _prepare(D: np.ndarray, lengths: Sequence[int]):
    """
    Prepare the data for the coverage DP.
    Returns sorted lengths, T, valid starts per length, cap of every length.
    """
    lengths = _normalize_lengths(lengths)
    T = D.shape[1]
    if lengths[-1] > T:
        raise ValueError(f"need every length <= T, got {lengths[-1]} > {T}")

    n_starts = [T - l + 1 for l in lengths]
    caps = [T // l for l in lengths] # how many windows of each length fit in a row
    return D, lengths, T, n_starts, caps
 
def _window_values(D: np.ndarray, lengths: list, theta_rows: np.ndarray, n_starts: list) -> list:
    """W[i][r, s] = value of a window of lengths[i] starting at s on row r:  sum_t (d - theta_r)  (prefix sums)."""
    score = D - theta_rows[:, None]
    csum = np.concatenate([np.zeros((D.shape[0], 1)), np.cumsum(score, axis=1)], axis=1)
    return [csum[:, l:l + n] - csum[:, :n] for l, n in zip(lengths, n_starts)]
 
def _axis_slices(m: int, i: int, lead: int = 0):
    """Slices that align k_i (destination) with k_i - 1 (source) along axis i of the state."""
    pre = (slice(None),) * lead
    src = pre + tuple(slice(None, -1) if j == i else slice(None) for j in range(m))
    dst = pre + tuple(slice(1, None) if j == i else slice(None) for j in range(m))
    return src, dst
 
def _check_state_space(cells: int, what: str, limit: int) -> None:
    if cells > limit:
        raise ValueError(f"{what}: state space is ~{cells:,} cells, over the {limit:,} guard. It grows multiplicatively "
                         f"with the number of lengths: use fewer lengths")
 
def _solve_with_highs(model, time_limit: Optional[float]) -> None:
    """Solve a Pyomo MILP with HiGHS; stops at time_limit seconds if given."""
    opt = pyo.SolverFactory("appsi_highs")
    opt.config.stream_solver = False
    opt.config.mip_gap = 0.0
    opt.highs_options["output_flag"] = False
    if time_limit:
        opt.config.time_limit = float(time_limit)
        opt.highs_options["time_limit"] = float(time_limit)
    quiet = logging.getLogger("highs_quiet")
    quiet.setLevel(logging.CRITICAL)
    opt.config.solver_output_logger = quiet
    res = opt.solve(model)
    tc = getattr(res, "termination_condition", None)
    if tc is not None and tc not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit):
        raise RuntimeError(f"MILP failed: {tc}")
 
# DP on one row, and on many rows at once
def solve_coverage_multi(
    demand: Sequence[float] | np.ndarray,
    lengths: Sequence[int],
    theta: float,
    max_rects: Optional[dict] = None
) -> MultiSolution:
    """
    Exact DP on one row: place disjoint windows of the given lengths, maximizing the sum of their values.
 
    F[t, k_1..k_m] = best objective using slots [0, t) with at most k_i windows of lengths[i]:
 
        F[t, k] = max( F[t-1, k],                                      # no window ends at t
                       F[t-l_i, k - e_i] + w_i[t-l_i]  for each i )    # a window of length l_i ends at t
 
    Ties go to not placing a window, so a window of value <= 0 is never used.
    max_rects : {length: cap} on the windows of each length (default: as many as fit); used by solve_allocated_multi.
    """
    d = np.asarray(demand, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("demand must be 1-D; use solve_coverage_batch_multi for many rows")
    _check_demand_range(d)
    D, lengths, T, n_starts, caps = _prepare(d[None, :], lengths)
 
    if max_rects is not None:
        unknown = set(max_rects) - set(lengths)
        if unknown:
            raise ValueError(f"max_rects has lengths {sorted(unknown)} that are not in lengths={lengths}")
        caps = [max(0, min(c, int(max_rects.get(l, c)))) for l, c in zip(lengths, caps)]
 
    m = len(lengths)
    shape = tuple(c + 1 for c in caps)
    _check_state_space(T * int(np.prod(shape)), "solve_coverage_multi", limit=20_000_000)
 
    if all(c == 0 for c in caps):
        empty = np.empty(0, dtype=int)
        return MultiSolution(0.0, empty, empty, np.zeros(T, dtype=bool))._fill_stats(D[0])
 
    W = _window_values(D, lengths, _theta_rows(theta, 1), n_starts)
    F = np.zeros((T + 1,) + shape, dtype=np.float64)
    take = np.full((T + 1,) + shape, -1, dtype=np.int8)   # index of the length ending a window at t, -1 = carry F[t-1]
    moves = [(i, l, *_axis_slices(m, i)) for i, l in enumerate(lengths) if caps[i] > 0]
 
    for t in range(1, T + 1):
        F[t] = F[t - 1]
        for i, l, src, dst in moves:
            if t < l or t - l >= n_starts[i]:
                continue
            cand = F[t - l][src] + W[i][0, t - l]
            cur = F[t][dst]                                   # views: writing into them updates F[t] / take[t]
            better = cand > cur
            if better.any():
                cur[better] = cand[better]
                take[t][dst][better] = i
 
    # backtrack from the final state (T, caps)
    starts, lens = [], []
    state, t = list(caps), T
    while t > 0:
        i = int(take[(t,) + tuple(state)])
        if i < 0:
            t -= 1
        else:
            starts.append(t - lengths[i])
            lens.append(lengths[i])
            t -= lengths[i]
            state[i] -= 1
    starts.reverse()
    lens.reverse()
 
    coverage = np.zeros(T, dtype=bool)
    for s, l in zip(starts, lens):
        coverage[s:s + l] = True
    value = float(F[(T,) + tuple(caps)])
    return MultiSolution(value, np.asarray(starts, dtype=int), np.asarray(lens, dtype=int), coverage)._fill_stats(D[0])
 
def solve_coverage_batch_multi(
    demand: np.ndarray,
    lengths: Sequence[int],
    theta: Union[float, Sequence[float]]
) -> np.ndarray:
    """
    Value curves of many rows at once. demand (N, T); theta: a number or one value per row.
    Returns (N, K_1+1, .., K_m+1), lengths sorted ascending: entry [r, k_1, .., k_m] is the best value of row r using
    at most k_i windows of lengths[i], for every i at once. This is the input of allocate_milp.
    """
    D = np.asarray(demand, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("demand must be (N, T)")
    _check_demand_range(D)
    theta_rows = _theta_rows(theta, D.shape[0])
    D, lengths, T, n_starts, caps = _prepare(D, lengths)
    N, m = D.shape[0], len(lengths)
    shape = tuple(c + 1 for c in caps)
    P = lengths[-1] + 1                                       # ring buffer: only F[t-l], l <= max length, is needed
    _check_state_space(P * N * int(np.prod(shape)), "solve_coverage_batch_multi", limit=50_000_000)
 
    if all(c == 0 for c in caps):
        return np.zeros((N,) + shape)
 
    W = _window_values(D, lengths, theta_rows, n_starts)
    moves = [(i, l, *_axis_slices(m, i, lead=1)) for i, l in enumerate(lengths) if caps[i] > 0]
 
    buf = np.zeros((P, N) + shape, dtype=np.float64)          # buf[t % P] holds F[t]
    for t in range(1, T + 1):
        cur = buf[t % P]
        cur[...] = buf[(t - 1) % P]
        for i, l, src, dst in moves:
            if t < l or t - l >= n_starts[i]:
                continue
            wv = W[i][:, t - l].reshape((N,) + (1,) * m)
            view = cur[dst]
            np.maximum(view, buf[(t - l) % P][src] + wv, out=view)
    return buf[T % P].copy()

# budget allocation (MILP over the value curves)
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
    Decide how many windows of each length every row gets, maximizing the total value of the rows:
 
        max  sum_{n,r} sum_{k in K*[n,r]} V[n,r,k] z[n,r,k]
        s.t. sum_{k in K*[n,r]} z[n,r,k] = 1                    for every day n, layer r   (one state per row)
             sum_{n,r,k} |k|   z[n,r,k] <= K_total              (optional)
             sum_{r,k}   |k|   z[n,r,k] <= K_day[n]             for every day n (optional)
             sum_{r,k}   k_j   z[n,r,k] <= length_per_day[l_j]  for every day n and length j (optional)
 
    K*[n,r] = efficient states of the row: V(k) > V(k - e_j) for every j with k_j >= 1 (budgets are upper bounds, so a
    state that does not improve on a smaller one is never needed).
 
    curves         : (n_days * n_layers, K_1+1, .., K_m+1) from solve_coverage_batch_multi, rows ordered day by day:
                     row = n * n_layers + r
    n_days         : number of days (default 1: the rows of curves are the layers of a single day)
    n_layers       : layers per day (default curves.shape[0] // n_days)
    K_day          : int (same for all days), dict {n: cap} or sequence indexed by n; a day missing from a dict = no limit
    length_per_day : {length: max windows of that length on each day}, the same cap for every day
    Returns MultiAllocation with one row of k per (n, r), in the order of curves.
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
 
    # efficient states: removing any window of the state must lower the value
    efficient = np.ones(curves.shape, dtype=bool)
    for j in range(m_len):
        hi = [slice(None)] * curves.ndim; hi[1 + j] = slice(1, None)
        lo = [slice(None)] * curves.ndim; lo[1 + j] = slice(0, -1)
        gain = np.full(curves.shape, np.inf)                  # k_j = 0: nothing to remove
        gain[tuple(hi)] = curves[tuple(hi)] - curves[tuple(lo)]
        efficient &= gain > TIE_EPS
    cells = [(n, r) for n in range(n_days) for r in range(n_layers)]
    states = {(n, r): [tuple(int(x) for x in s) for s in np.argwhere(efficient[n * n_layers + r])] for n, r in cells}
    V = lambda n, r, kap: float(curves[(n * n_layers + r,) + kap])
 
    if K_day is None:
        day_cap = [None] * n_days
    elif isinstance(K_day, dict):
        day_cap = [K_day.get(n) for n in range(n_days)]
    elif np.isscalar(K_day):
        day_cap = [K_day] * n_days
    else:
        day_cap = list(K_day)
 
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=[(n, r) + kap for (n, r) in cells for kap in states[n, r]], dimen=m_len + 2)
    m.z = pyo.Var(m.I, domain=pyo.Binary)                     # z[n, r, k_1, .., k_m]: row (n, r) takes state k
    m.obj = pyo.Objective(expr=sum(V(n, r, kap) * m.z[(n, r) + kap] for (n, r) in cells for kap in states[n, r]),
                          sense=pyo.maximize)
 
    m.one = pyo.ConstraintList()
    for n, r in cells:
        m.one.add(sum(m.z[(n, r) + kap] for kap in states[n, r]) == 1)
    if K_total is not None:
        m.total = pyo.Constraint(expr=sum(sum(kap) * m.z[(n, r) + kap] for (n, r) in cells for kap in states[n, r])
                                 <= max(0, int(K_total)))
    m.per_day = pyo.ConstraintList()
    for n in range(n_days):
        rows_n = [(r, kap) for r in range(n_layers) for kap in states[n, r]]
        if day_cap[n] is not None:
            m.per_day.add(sum(sum(kap) * m.z[(n, r) + kap] for r, kap in rows_n) <= max(0, int(day_cap[n])))
        for j, l in enumerate(lengths):
            if l in length_per_day:
                m.per_day.add(sum(kap[j] * m.z[(n, r) + kap] for r, kap in rows_n) <= max(0, int(length_per_day[l])))
 
    _solve_with_highs(m, time_limit)
 
    k = np.zeros((n_days * n_layers, m_len), dtype=int)
    for n, r in cells:
        k[n * n_layers + r] = max(states[n, r], key=lambda s: pyo.value(m.z[(n, r) + s]))
    per_row_value = np.array([curves[(i,) + tuple(k[i])] for i in range(len(k))])
    return MultiAllocation(k=k, lengths=lengths, value=float(per_row_value.sum()), per_row_value=per_row_value)
 
# positions of the windows: re-solve every row with its own k
def _inherited_row(rho: np.ndarray, n_open: np.ndarray) -> np.ndarray:
    """Demand a row sees when n_open lower rows are open in each slot: clip(rho - n_open, 0, 1)."""
    return np.clip(rho - n_open, 0.0, 1.0)
 
def effective_grid(D: np.ndarray, coverages: Sequence) -> np.ndarray:
    """
    Demand each row sees when it inherits what the rows below left uncovered.
    D (n_rows, T): layers, rows sum to rho_t = arrivals / pax per lane-bin. coverages[r] = bool mask of the slots in which
    row r is open (rows beyond len(coverages) have no server). Row r sees clip(rho_t - n_t, 0, 1), n_t = number of rows
    < r open in slot t: if all the lower rows are open it is the usual layer r; if some are closed, their uncovered
    demand falls on row r. Then  sum_r covered demand of row r = sum_t min(a_t, capacity_t): the true coverage.
    """
    D = np.asarray(D, dtype=np.float64)
    T = D.shape[1]
    rho = D.sum(axis=0)
    n_open = np.zeros(T)
    out = np.empty_like(D)
    for r in range(D.shape[0]):
        out[r] = _inherited_row(rho, n_open)
        if r < len(coverages):
            n_open += np.asarray(coverages[r], dtype=bool)[:T]
    return out
 
def solve_allocated_multi(
    demand: np.ndarray,
    theta: Union[float, Sequence[float]],
    alloc: MultiAllocation,
    inherit: bool = True,
) -> list:
    """
    Recover the windows of every row, each with its own number of windows per length (alloc.k). 
    Fewer windows than alloc.k may be placed (a window with value <= 0 is not opened):
    count them with sol.n_rects / sol.rects_by_length.
 
    inherit=True: rows are solved in order, row r on the demand rows 0..r-1 left uncovered (see effective_grid), so its
    windows can fill the slots where a lower row is closed. alloc.k comes from the plain layers. For the true coverage, row
    r is never worse than on its plain layer, which is still a feasible choice.
    """
    D = np.asarray(demand, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("demand must be (N, T)")
    if alloc.k.shape[0] != D.shape[0]:
        raise ValueError(f"alloc has {alloc.k.shape[0]} rows, demand has {D.shape[0]}")
    theta_rows = _theta_rows(theta, D.shape[0])
 
    rho = D.sum(axis=0)
    n_open = np.zeros(D.shape[1])
    sols = []
    for r in range(D.shape[0]):
        row = _inherited_row(rho, n_open) if inherit else D[r]
        sol = solve_coverage_multi(row, alloc.lengths, float(theta_rows[r]),
                                   max_rects={l: int(alloc.k[r, j]) for j, l in enumerate(alloc.lengths)},
                                )
        sols.append(sol)
        n_open += sol.coverage[:D.shape[1]]
    return sols
 
# report
_BLOCK_COLS = ["row", "block", "length", "start_slot", "start_time", "end_time", "covered_demand", "utilization"]
 
def _block_records(D, sols, time_values, fixed_length=None) -> pd.DataFrame:
    """One line per window: where it is, how long, how much demand it covers, how busy it is."""
    tv = pd.to_datetime(pd.Series(time_values).reset_index(drop=True))
    step = tv.iloc[1] - tv.iloc[0]
    recs = []
    for c, sol in enumerate(sols):
        lengths = [fixed_length] * len(sol.starts) if fixed_length is not None else sol.lengths
        for b, (s0, l) in enumerate(zip(sol.starts, lengths)):
            s0, l = int(s0), int(l)
            covered = float(D[c, s0:s0 + l].sum())
            recs.append(dict(row=c, block=b, length=l, start_slot=s0, start_time=tv.iloc[s0],
                             end_time=tv.iloc[s0] + l * step, covered_demand=covered, utilization=covered / l))
    return pd.DataFrame(recs, columns=_BLOCK_COLS)
 
def _report_from_blocks(D, blocks, theta_rows, row_labels) -> CoverageReport:
    """Per-row and total tables from the block table. D (n_rows, T) gives the demand of the rows."""
    n_rows = D.shape[0]
    row_demand = D.sum(axis=1)
    safe = lambda num, den: num / den if den > 0 else np.nan
    by_row = lambda col: blocks.groupby("row")[col].sum().reindex(range(n_rows), fill_value=0).to_numpy().astype(float)
    covered, capacity = by_row("covered_demand"), by_row("length")
    n_blocks = blocks.groupby("row").size().reindex(range(n_rows), fill_value=0).to_numpy()
    
    rows = pd.DataFrame({
        "server": list(row_labels) if row_labels is not None else [None] * n_rows,
        "n_blocks": n_blocks,
        "demand": row_demand,
        "covered_demand": covered,
        "coverage": [safe(cv, dm) for cv, dm in zip(covered, row_demand)],
    }).rename_axis("row")
 
    total = {
        "n_blocks": int(n_blocks.sum()),
        "demand": float(row_demand.sum()),
        "covered_demand": float(covered.sum()),
        "coverage": safe(float(covered.sum()), float(row_demand.sum())),
        "objective": float((covered - theta_rows * capacity).sum()),    # sum of (covered - theta * length)
    }
    for l, n in blocks.groupby("length").size().items():                 # how many windows of each length
        total[f"n_blocks_{int(l)}"] = int(n)
    return CoverageReport(blocks=blocks, rows=rows, total=pd.Series(total))
 
def coverage_report_multi(D, sols, time_values, theta, row_labels=None, D_cov=None) -> CoverageReport:
    """
    Report of the DP solution.
 
    D    : (n_rows, T) demand grid in [0, 1]
    sols : list of MultiSolution for the first len(sols) rows (the rows with a server); the other rows of D have no
           server: demand with 0 blocks
    time_values : (T,) datetime-like, start time of each bin
    theta : the theta used to compute the solutions: a number or one value per row of D (None = 0)
    D_cov : demand grid the rows were solved on when they inherit uncovered demand (effective_grid); the covered demand
            of the blocks is read from it. Default D (rows independent). Demand and coverage of rows / total stay relative to D.
 
    blocks : row, block, length, start_slot, start_time, end_time, covered_demand, utilization
    rows   : server, n_blocks, demand, covered_demand, coverage
    total  : n_blocks, n_blocks_<length>, demand, covered_demand, coverage, objective = sum(covered - theta * length)
    """
    D = np.asarray(D, dtype=np.float64)
    if len(sols) > D.shape[0]:
        raise ValueError(f"{len(sols)} solutions for {D.shape[0]} rows")
    theta_rows = _theta_rows(theta, D.shape[0])
    blocks = _block_records(D if D_cov is None else np.asarray(D_cov, dtype=np.float64), sols, time_values)
    return _report_from_blocks(D, blocks, theta_rows, row_labels)







# ---------------------------------------------------------------------------
# single L algorithm

def extend_demand(demand: np.ndarray, L: int) -> np.ndarray:
    """
    Append L-1 empty (zero-demand) slots on the right of the last axis.
    """
    if L < 1:
        raise ValueError(f"need L >= 1, got L={L}")
    d = np.asarray(demand, dtype=np.float64)
    width = [(0, 0)] * (d.ndim - 1) + [(0, L - 1)]
    return np.pad(d, width)

@dataclass
class Solution:
    value: float                 # optimal objective = inside - theta * (inside + outside)
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
    theta: float,
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
    score = d - theta
    csum = np.concatenate(([0.0], np.cumsum(score)))
    return csum[L:] - csum[:-L]

def solve_coverage(
    demand: Sequence[float] | np.ndarray,
    L: int,
    theta: float,
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
    w = rectangle_values(d, L, theta, allow_overhang=False)
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
    theta: float,
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
    score = D - theta
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
    theta: float,
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
    return [solve_coverage(row, L, theta, max_rects=int(k), allow_overhang=allow_overhang)
            for row, k in zip(D, alloc.k)]
