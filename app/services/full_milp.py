"""
Full MILP on the fluid queue: decides, for every lane, how many windows of each length and where they start,
minimising the fluid queue directly. It replaces the whole chain  solve_coverage_batch_multi -> allocate_milp ->
solve_allocated_multi  (demand grid, DP, allocation), and is meant as the queue-optimal benchmark of the
coverage-optimal schedule.

Same conventions as coverage_dp.py:
    * lengths are in bins (cells), sorted;  no overhang: a window ends within the day;
    * budgets: K_total, length_per_day {length: cap per day}, K_day (int / dict / sequence) as in allocate_milp;
    * the output is the list of MultiSolution of solve_allocated_multi (one per row, day-major: row = n * R + r), so
      coverage_report_multi, SecurityPlanner._schedule_and_curves, schedule_to_blocks ... work unchanged.

Model (per day n, slots t = 0..T-1, dt = bin length in minutes, a_t = arrivals, mu_r = pax/bin of lane r)
    x[r,l,s] in {0,1}      a window of length l opens on lane r in slot s   (0 <= s <= T-l)
    sum_{l, s covering t} x[r,l,s] <= 1                                     windows of a lane do not overlap
    c_t   = sum_r mu_r * sum_{l, s covering t} x[r,l,s]                      capacity of slot t (pax)
    Q_t  >= Q_{t-1} + a_t - c_t,  Q_t >= 0,  Q_{-1} = 0                      fluid queue at the end of slot t
Objective, LEXICOGRAPHIC and without any weight (two solves, same constraints):
    stage 1   min  U = sum_n Q_{n,T-1}                                       fewest unserved passengers
    stage 2   min  dt * sum_t (Q_{t-1} + Q_t)/2   s.t.  U <= U* + unserved_tol   then the smallest total waiting time
Since a_t and c_t are uniform inside a slot, Q_t = max(0, Q_{t-1} + a_t - c_t) is exact at slot boundaries (it is the same
queue as local_search._fluid_from_cap); the area under Q is the trapezoid of the two end values (exact when the queue
does not empty inside the slot, a slight over-estimate otherwise). Q_{T-1} is the mass that never departs: the "unserved".
The stage-2 objective / N is the mean wait per passenger, unserved ones waiting until the end of the horizon (censored).
When U* = 0 it is exactly the mean wait of local_search.fluid_metrics; when U* > 0 the two differ slightly (that one
averages over the served mass only).
"""
import dataclasses
import logging
import math
import time
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyomo.environ as pyo                                            
from app.services.coverage_dp import MultiSolution, MultiAllocation, coverage_report_multi, _normalize_lengths

# helpers
def fluid_cost(arrivals, capacity, bin_minutes):
    """Fluid queue on a given capacity profile (pax/slot): (wait_area in pax*min, unserved pax at the end)."""
    a = np.asarray(arrivals, dtype=np.float64)
    c = np.asarray(capacity, dtype=np.float64)
    q_prev, area = 0.0, 0.0
    for at, ct in zip(a, c):
        q = max(0.0, q_prev + at - ct)
        area += bin_minutes * 0.5 * (q_prev + q)
        q_prev = q
    return area, q_prev

def allocation_from_solutions(sols, lengths, values=None) -> MultiAllocation:
    """MultiAllocation (k per row and length) of a list of MultiSolution, like the one allocate_milp returns."""
    lengths = _normalize_lengths(lengths)
    k = np.zeros((len(sols), len(lengths)), dtype=int)
    for i, sol in enumerate(sols):
        for j, l in enumerate(lengths):
            k[i, j] = int(np.sum(sol.lengths == l))
    per_row = np.array([s.value for s in sols], dtype=float) if values is None else np.asarray(values, dtype=float)
    return MultiAllocation(k=k, lengths=lengths, value=float(per_row.sum()), per_row_value=per_row)

def pack_to_lowest_lanes(wins_by_lane: dict, mu) -> dict:
    """
    Reassign the windows of one day to the lowest-index lanes, without changing the capacity profile.
    wins_by_lane {r: [(start, length), ...]}; mu rate of each lane. Only lanes with the same rate are interchangeable, so
    the windows are packed within each group of equal-rate lanes: sorted by start, each window goes to the lowest lane that
    is free at its start (first fit: uses the smallest possible number of lanes of the group, = the maximum overlap).
    """
    R = len(mu)
    groups = []
    for r in range(R):
        for g in groups:
            if np.isclose(mu[r], mu[g[0]]):
                g.append(r)
                break
        else:
            groups.append([r])
    out = {r: [] for r in range(R)}
    for g in groups:
        wins = sorted((s, l) for r in g for (s, l) in wins_by_lane.get(r, []))
        wins.sort(key=lambda w: (w[0], -w[1]))
        end = {r: -1 for r in g}
        for s, l in wins:
            r = next(r for r in g if end[r] <= s)
            out[r].append((s, l))
            end[r] = s + l
    return out


# MILP
def solve_fluid_milp(
    arrivals: np.ndarray,
    rates: Union[dict, Sequence[float]],
    labels: Sequence[str],
    lengths: Sequence[int],
    bin_minutes: float,
    K_total: Optional[int] = None,
    length_per_day: Optional[dict] = None,
    K_day: Optional[Union[int, dict, Sequence[int]]] = None,
    unserved_tol: float = 1e-3,
    demand: Optional[np.ndarray] = None,
    window_cost: float = 1e-6,
    symmetry_breaking: bool = True,
    pack_lanes: bool = True,
    prefix_lanes: bool = False,
    time_limit: Optional[float] = None,
    mip_gap: float = 0.0,
    return_info: bool = False,
    verbose: bool = False,
):
    """
    arrivals       : (T,) pax per bin of one day, or (N, T) for N days (queues do not carry over between days)
    rates          : pax per bin of each lane: dict {label: rate} (planner.rate_by_label) or a sequence aligned with labels
    labels         : lane labels in priority order; the lanes are labels[:] (pass labels[:n_usable] to match run_multi)
    lengths        : allowed window lengths in bins (e.g. lengths_cells)
    bin_minutes    : bin length (minutes)
    K_total        : max number of windows over all lanes and days (None = no limit)
    length_per_day : {length in bins: max windows of that length on each day}   (as in allocate_milp)
    K_day          : max windows per day: int, dict {n: cap} or sequence (as in allocate_milp)
    unserved_tol   : pax. Stage 2 may leave up to U* + unserved_tol unserved (U* = minimum found by stage 1); the
                     default makes the order strictly lexicographic. No penalty weight: nothing to tune.
    demand         : optional demand grid (R, T) or (N, R, T), R >= number of lanes: only to fill the stats of the solutions
                     (sol.inside / outside / missed) and sol.value (= covered load)
    window_cost    : tiny cost per window so that useless windows (no effect on the queue) are not opened
    symmetry_breaking : lanes with the same rate are interchangeable: order them by opened slots (valid, speeds up the MILP)
    pack_lanes     : after solving, reassign the windows to the lowest-index lanes (first fit within each group of equal-rate
                     lanes). The capacity profile, hence queue and unserved, does not change; only which lane hosts which window.
    prefix_lanes   : constraint inside the MILP: lane r can be open in a slot only if lane r-1 is (open lanes always form a
                     prefix of the priority order; then layer coverage = true coverage). It RESTRICTS the schedules (windows
                     have fixed lengths and cannot be split between lanes), so the queue can get worse: off by default.
    time_limit     : seconds in total (None = run to optimality): up to 25% for stage 1, the rest for stage 2;
                     mip_gap : relative gap of stage 2 (0 = exact)

    Returns sols (list of MultiSolution, row = n * R + r, R = len(labels)); with return_info=True also a dict with
    unserved (stage 1 minimum, as found) and its status / lower bound, wait_area, mean wait per pax (wait_mean),
    lower bound and relative gap of stage 2, status, n_windows.
    """
    A = np.asarray(arrivals, dtype=np.float64)
    if A.ndim == 1:
        A = A[None, :]
    if A.ndim != 2 or (A < 0).any():
        raise ValueError("arrivals must be (T,) or (N, T), non-negative")
    N, T = A.shape
    labels = [str(l) for l in labels]
    R = len(labels)
    lengths = _normalize_lengths(lengths)
    if lengths[-1] > T:
        raise ValueError(f"longest length ({lengths[-1]} bins) exceeds the day ({T} bins)")
    mu = (np.array([float(rates[l]) for l in labels]) if isinstance(rates, dict)
          else np.asarray(rates, dtype=np.float64))
    if mu.shape != (R,) or (mu <= 0).any():
        raise ValueError("rates must be positive, one per lane")
    length_per_day = length_per_day or {}
    unknown = set(length_per_day) - set(lengths)
    if unknown:
        raise ValueError(f"length_per_day has lengths {sorted(unknown)} that are not in lengths={lengths}")
    dt = float(bin_minutes)

    # starts of the windows of length l that cover slot t
    cover = {l: [list(range(max(0, t - l + 1), min(t, T - l) + 1)) for t in range(T)] for l in lengths}

    m = pyo.ConcreteModel()
    idx = [(n, r, l, s) for n in range(N) for r in range(R) for l in lengths for s in range(T - l + 1)]
    m.I = pyo.Set(initialize=idx, dimen=4)
    m.x = pyo.Var(m.I, domain=pyo.Binary)
    m.Q = pyo.Var(pyo.RangeSet(0, N - 1), pyo.RangeSet(0, T - 1), domain=pyo.NonNegativeReals)

    def open_expr(n, r, t):                       # 1 if lane r of day n is open in slot t (linear in x)
        return sum(m.x[n, r, l, s] for l in lengths for s in cover[l][t])

    # windows of a lane do not overlap
    m.no_overlap = pyo.ConstraintList()
    for n in range(N):
        for r in range(R):
            for t in range(T):
                m.no_overlap.add(open_expr(n, r, t) <= 1)

    # fluid queue
    m.queue = pyo.ConstraintList()
    for n in range(N):
        for t in range(T):
            prev = m.Q[n, t - 1] if t > 0 else 0.0
            cap = sum(float(mu[r]) * open_expr(n, r, t) for r in range(R))
            m.queue.add(m.Q[n, t] >= prev + float(A[n, t]) - cap)

    # open lanes form a prefix of the priority order
    if prefix_lanes:
        m.prefix = pyo.ConstraintList()
        for n in range(N):
            for r in range(1, R):
                for t in range(T):
                    m.prefix.add(open_expr(n, r, t) <= open_expr(n, r - 1, t))

    # budgets (same semantics as allocate_milp)
    def count(n=None, l=None):
        return sum(m.x[nn, r, ll, s] for (nn, r, ll, s) in idx
                   if (n is None or nn == n) and (l is None or ll == l))

    if K_total is not None:
        m.total = pyo.Constraint(expr=count() <= max(0, int(K_total)))
    if length_per_day:
        m.per_day_length = pyo.ConstraintList()
        for n in range(N):
            for l, cap in length_per_day.items():
                m.per_day_length.add(count(n=n, l=l) <= max(0, int(cap)))
    if K_day is not None:
        cap_of = (lambda n: K_day.get(n)) if isinstance(K_day, dict) else \
                 (lambda n: K_day) if np.isscalar(K_day) else (lambda n: K_day[n])
        m.per_day = pyo.ConstraintList()
        for n in range(N):
            if cap_of(n) is not None:
                m.per_day.add(count(n=n) <= max(0, int(cap_of(n))))

    # lanes with the same rate are interchangeable: order them by number of open slots
    if symmetry_breaking:
        m.sym = pyo.ConstraintList()
        for n in range(N):
            for r in range(R - 1):
                if np.isclose(mu[r], mu[r + 1]):
                    m.sym.add(sum(l * m.x[n, r, l, s] for l in lengths for s in range(T - l + 1))
                              >= sum(l * m.x[n, r + 1, l, s] for l in lengths for s in range(T - l + 1)))

    wait_area = sum(dt * 0.5 * ((m.Q[n, t - 1] if t > 0 else 0.0) + m.Q[n, t]) for n in range(N) for t in range(T))
    unserved = sum(m.Q[n, T - 1] for n in range(N))

    def solve(limit, gap):
        opt = pyo.SolverFactory("appsi_highs")
        opt.config.stream_solver = bool(verbose)
        opt.config.mip_gap = float(gap)
        opt.highs_options["output_flag"] = bool(verbose)
        if limit:
            opt.config.time_limit = float(limit)
            opt.highs_options["time_limit"] = float(limit)
        quiet = logging.getLogger("highs_quiet")
        quiet.setLevel(logging.CRITICAL)
        opt.config.solver_output_logger = quiet
        res = opt.solve(m)
        tc = getattr(res, "termination_condition", None)
        if tc is not None and tc not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit):
            raise RuntimeError(f"MILP failed: {tc}")
        try:
            bound = float(res.problem.lower_bound)              # best lower bound proved (minimisation)
        except Exception:
            bound = float("nan")
        return tc, bound

    t_start = time.time()
    # stage 1: fewest unserved passengers
    m.obj1 = pyo.Objective(expr=unserved, sense=pyo.minimize)
    tc1, bound1 = solve(0.25 * time_limit if time_limit else None, 0.0)
    u_star = float(pyo.value(unserved))
    # stage 2: smallest total waiting time among the schedules with (almost) that few unserved
    m.obj1.deactivate()
    m.keep_unserved = pyo.Constraint(expr=unserved <= u_star + float(unserved_tol))
    m.obj2 = pyo.Objective(expr=wait_area + float(window_cost) * sum(m.x[i] for i in idx), sense=pyo.minimize)
    left = max(1.0, float(time_limit) - (time.time() - t_start)) if time_limit else None
    tc, bound = solve(left, mip_gap)

    # ---- solutions
    Dm = None
    if demand is not None:
        Dm = np.asarray(demand, dtype=np.float64)
        Dm = Dm[None] if Dm.ndim == 2 else Dm
        if Dm.shape[0] != N or Dm.shape[1] < R or Dm.shape[2] != T:
            raise ValueError(f"demand must be (R>={R}, {T}) or ({N}, R>={R}, {T}); got {np.shape(demand)}")
    sols, cap_profile = [], np.zeros((N, T))
    chosen = [(nn, rr, l, s) for (nn, rr, l, s) in idx if pyo.value(m.x[nn, rr, l, s]) > 0.5]
    for n in range(N):
        by_lane = {r: sorted((s, l) for (nn, rr, l, s) in chosen if nn == n and rr == r) for r in range(R)}
        if pack_lanes:
            by_lane = pack_to_lowest_lanes(by_lane, mu)
        for r in range(R):
            wins = sorted(by_lane[r])
            starts = np.array([s for s, _ in wins], dtype=int)
            lens = np.array([l for _, l in wins], dtype=int)
            cov = np.zeros(T, dtype=bool)
            for s, l in wins:
                cov[s:s + l] = True
            cap_profile[n] += mu[r] * cov
            sol = MultiSolution(0.0, starts, lens, cov)
            if Dm is not None:
                sol._fill_stats(Dm[n, r])
                sol.value = sol.inside
            sols.append(sol)

    if not return_info:
        return sols
    queue = [fluid_cost(A[n], cap_profile[n], dt) for n in range(N)]
    area = float(sum(q[0] for q in queue))
    obj_val = float(pyo.value(m.obj2))
    n_pax = float(A.sum())
    info = dict(
        unserved=float(sum(q[1] for q in queue)), unserved_status=str(tc1), unserved_bound=bound1,
        wait_area=area, wait_mean=area / n_pax if n_pax > 0 else float("nan"),
        n_windows=int(sum(s.n_rects for s in sols)), status=str(tc),
        bound=bound, gap=(obj_val - bound) / abs(obj_val) if np.isfinite(bound) and obj_val else float("nan"),
    )
    return sols, info


# planner level (same shape as SecurityPlanner.run_multi)
def run_fluid_milp(planner, lengths, K_total, length_totals=None, unserved_tol=1e-3, n_lanes=None,
                   time_limit=None, mip_gap=0.0, symmetry_breaking=True, pack_lanes=True, prefix_lanes=False):
    """
    Queue-optimal counterpart of planner.run_multi(lengths, theta, K_total, length_totals): same arguments (lengths in
    hours; K_total is an upper bound, fewer windows may be used), objective lexicographic (unserved, then wait), same PlannerResults (schedule, df_curves, coverage, report in planner.report). theta is not used: the result
    has theta=0 and theta_rows=0, so its `objective` is the covered demand. Returns (PlannerResults, info).
    n_lanes : number of lanes (default: the same as run_multi, min(demand layers, servers)).
    """
    try:
        from app.services.security_class import PlannerResults
    except ImportError:
        from security_class import PlannerResults
    lengths_h = sorted({float(l) for l in lengths})
    bin_hours = planner.bin_minutes / 60.0
    cells_of = {h: max(1, math.floor(h / bin_hours + 0.5)) for h in lengths_h}
    lengths_cells = [cells_of[h] for h in lengths_h]
    totals_cells = None
    if length_totals is not None:
        totals_cells = {cells_of[float(h)]: int(v) for h, v in length_totals.items()}

    n_rows = planner.demand_grid.shape[0]
    n_usable = min(n_rows, len(planner.server_labels)) if n_lanes is None else int(n_lanes)
    labels = list(planner.server_labels[:n_usable])

    sols, info = solve_fluid_milp(
        planner.arrivals.to_numpy(), planner.rate_by_label, labels, lengths_cells, planner.bin_minutes,
        K_total=K_total, length_per_day=totals_cells, unserved_tol=unserved_tol,
        demand=planner.demand_grid[:n_usable], symmetry_breaking=symmetry_breaking,
        pack_lanes=pack_lanes, prefix_lanes=prefix_lanes,
        time_limit=time_limit, mip_gap=mip_gap, return_info=True)

    zeros = np.zeros(n_rows)
    row_labels = labels + ["no server"] * (n_rows - n_usable)
    planner.report = coverage_report_multi(D=planner.demand_grid, sols=sols, time_values=planner.time_values,
                                           row_labels=row_labels, theta=zeros)
    schedule, df_curves = planner._schedule_and_curves(sols, labels)
    planner.results = PlannerResults(
        L=lengths_h[-1], L_cells=lengths_cells[-1], theta=0.0, K_total=int(K_total), bin_minutes=planner.bin_minutes,
        demand_grid=planner.demand_grid, server_labels=tuple(labels), server_schedule=schedule, df_curves=df_curves,
        coverage=planner.report.total.coverage,
        lengths=tuple(lengths_h), lengths_cells=tuple(lengths_cells), length_totals=length_totals,
        theta_rows=tuple(0.0 for _ in labels),
        coverage_slot=float(np.minimum(df_curves["demand"], df_curves["processing"]).sum() / df_curves["demand"].sum()))
    return planner.results, info