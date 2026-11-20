"""
Loop DP + MILP + backtracking on the inherited demand.

Pipeline now: the value curves (DP) and the allocation (MILP) use the plain layers d_{r,t}; only the recovery (backtracking)
uses the inherited demand d^eff_{r,t} = clip(rho_t - o_{r,t}, 0, 1), o_{r,t} = number of lower rows open in slot t.
Loop: the inherited demand of iteration k is built with the coverages of iteration k-1 and fed to the FIRST steps too:

    iteration 0 :  curves(d)        -> MILP -> recovery on d^eff(coverages of the same solution, row by row)   (= current pipeline)
    iteration k :  curves(d^eff_k)  -> MILP -> recovery,   d^eff_k = effective_grid(d, coverages_{k-1})

The recovery always inherits row by row inside the iteration, so the final solution of every iteration is a valid schedule
and its objective is the true one: sum_r (covered demand of row r on the final inherited grid - theta_r * capacity_r).

Fixed point: the schedule (hence d^eff) repeats. If it does not converge, the schedule repeats after a cycle: stop says which.
Everything is in bins (lengths and length_totals in cells), like coverage_dp.
"""
from re import match

import numpy as np
import pandas as pd

try:
    from app.services.coverage_dp import (
        MultiAllocation, solve_coverage_batch_multi, allocate_milp, solve_allocated_multi, effective_grid,
    )
except ImportError:
    from coverage_dp import (
        MultiAllocation, solve_coverage_batch_multi, allocate_milp, solve_allocated_multi, effective_grid,
    )


def _plan(grid, D_plan, lengths, theta_rows, K_total, length_totals, fixed, bounds):
    """curves on D_plan -> MILP -> recovery (inheriting row by row) on the plain grid. Returns (sols, alloc)."""
    R = grid.shape[0]
    fixed = fixed or {}
    free = [r for r in range(R) if r not in fixed]
    if not free:
        return [fixed[r] for r in range(R)], None
    if free != list(range(R - len(free), R)):
        raise ValueError("fixed rows must be the first rows (lowest layers)")
    n_fixed = R - len(free)
    curves = solve_coverage_batch_multi(D_plan[n_fixed:], lengths, theta_rows[n_fixed:], bounds=bounds)
    alloc_free = allocate_milp(curves, lengths, K_total=K_total, length_per_day=length_totals)
    alloc = MultiAllocation(
        k=np.vstack([np.zeros((n_fixed, len(lengths)), dtype=int), alloc_free.k]),
        lengths=alloc_free.lengths, value=alloc_free.value,
        per_row_value=np.concatenate([np.zeros(n_fixed), alloc_free.per_row_value]))
    sols = solve_allocated_multi(grid, theta_rows, alloc, inherit=True, fixed=fixed, bounds=bounds)
    return sols, alloc


def _blocks(sols):
    return [sorted((int(s), int(l)) for s, l in zip(sol.starts, sol.lengths)) for sol in sols]


def inherited_loop(grid, lengths, theta, K_total, length_totals=None, n_iter: int = 10, fixed=None, bounds=None,
                   arrivals=None, rates=None, verbose: bool = True):
    """
    grid    : (R, T) demand layers.   lengths : window lengths in bins.   theta : number or one value per row.
    K_total, length_totals : as in allocate_milp (length_totals = {length in bins: max windows}).
    fixed   : {row: MultiSolution} rows decided by the user (the first rows), bounds : (start, end) slots for the free rows.
    arrivals, rates : (T,) pax per slot and pax per slot of each row's server (len R); if given, the trace has q_fluid,
                      the mean wait of the fluid queue (drain) of each iteration (needs fluid_approx.py).
    Returns (sols, trace, history): sols of the LAST iteration; trace one row per iteration:
      iter, objective (true: covered - theta * capacity on the final inherited grid), planned (MILP value on the curves
      of that iteration), coverage (sum_t min(rho_t, lanes open) / sum rho), n_blocks, capacity (bin of open lanes),
      moved (blocks whose start changed vs the previous iteration), d_eff_change (sum |d^eff_k - d^eff_{k-1}|, lane-bins,
      d^eff of the final solution of each iteration), q_fluid. trace.attrs["stop"] says why the loop ended.
    history : list of dict(iter, sols, blocks, D_eff).
    """
    grid = np.asarray(grid, dtype=np.float64)
    R, T = grid.shape
    theta_rows = np.broadcast_to(np.asarray(theta, dtype=np.float64), (R,)).copy()
    rho = grid.sum(axis=0)
    if arrivals is not None:
        try:
            from app.services.fluid_approx import fluid_wait
        except ImportError:
            from fluid_approx import fluid_wait

    rows, history, seen = [], [], {}
    D_plan, prev_blocks, prev_Deff = grid, None, None
    stop = "n_iter"
    for k in range(n_iter + 1):
        sols, alloc = _plan(grid, D_plan, lengths, theta_rows, K_total, length_totals, fixed, bounds)
        covs = [s.coverage[:T] for s in sols]
        D_eff = effective_grid(grid, covs)
        blocks = _blocks(sols)
        covered = sum(float(D_eff[r][covs[r]].sum()) for r in range(R))
        capacity = float(sum(c.sum() for c in covs))
        obj = covered - float((theta_rows * np.array([c.sum() for c in covs])).sum())
        rec = dict(iter=k, objective=obj, planned=float(alloc.value) if alloc is not None else np.nan,
                   coverage=covered / float(rho.sum()), n_blocks=int(sum(len(b) for b in blocks)), capacity=capacity,
                   moved=0 if prev_blocks is None else sum(
                       abs(len(a) - len(b)) + sum(1 for x, y in zip(a, b) if x != y) for a, b in zip(prev_blocks, blocks)),
                   d_eff_change=0.0 if prev_Deff is None else float(np.abs(D_eff - prev_Deff).sum()))
        if arrivals is not None:
            cap = sum(rates[r] * covs[r].astype(float) for r in range(R))
            rec["q_fluid"] = fluid_wait(np.asarray(arrivals, dtype=np.float64), np.asarray(cap), 15.0)
        rows.append(rec)
        history.append(dict(iter=k, sols=sols, blocks=blocks, D_eff=D_eff))
        if verbose:
            q = f"  q_fluid={rec['q_fluid']:.2f}" if "q_fluid" in rec else ""
            print(f"iter {k}: objective={obj:.3f} planned={rec['planned']:.3f} coverage={rec['coverage']:.4f} "
                  f"blocks={rec['n_blocks']} moved={rec['moved']} d_eff_change={rec['d_eff_change']:.3f}{q}")
        sig = tuple(tuple(b) for b in blocks)
        if sig in seen:
            stop = "fixed point" if seen[sig] == k - 1 else f"cycle (iteration {seen[sig]} again)"
            break
        seen[sig] = k
        prev_blocks, prev_Deff = blocks, D_eff
        D_plan = D_eff
    trace = pd.DataFrame(rows)
    trace.attrs["stop"] = stop
    if verbose:
        print(f"stop: {stop}")
    return sols, trace, history


# ------------------------------------------------------------------ is the loop worth it? many cases
def _first_server(grid, fixed_first):
    """fixed_first: None | "all" (row 0 open on the whole horizon) | (start_slot, length) -> {0: MultiSolution}."""
    if fixed_first is None:
        return None
    try:
        from app.services.coverage_dp import fixed_window_solution
    except ImportError:
        from coverage_dp import fixed_window_solution
    start, length = (0, grid.shape[1]) if isinstance(fixed_first, str) and fixed_first == "all" else fixed_first
    return {0: fixed_window_solution(grid[0], 0.0, start, length)}   # theta only enters the (unused) value of that row


def loop_vs_theta_sweep(grid, lengths, thetas, K_total, length_totals, arrivals, rates, theta_grid=None,
                        n_iter: int = 10, tol_q: float = 0.05, q_max: float = 60.0, fixed_first=None):
    """
    One day (one grid). For every theta in `thetas`: plain pipeline (A), loop at its fixed point (B), and the plain
    pipeline on a fine grid of thetas (the "sweep", same K_total / length_totals). The loop is useful only if it gives
    something the theta sweep does not, so B is compared with the best plain schedule that uses no more capacity than B.

    fixed_first : None, "all" (row 0 open on the whole horizon) or (start_slot, length): row 0 is a server decided by the
                  user, the same in A, loop and sweep. It counts in cap_A / cap_B and in the fluid queue; K_total and
                  length_totals refer to the other rows only.

    Returns a DataFrame, one row per theta:
      changed        : B has a different schedule than A
      same_as_theta  : theta' of the sweep whose plain schedule is exactly B (NaN if none): the loop = another theta
      cap_A, cap_B   : open lane-bins;  q_A, q_B : fluid mean wait (min);  cov_A, cov_B : true coverage
      obj_A, obj_B   : covered - theta * capacity (true, at this theta)
      q_front        : smallest q among plain schedules (A and the sweep) with capacity <= cap_B
      advantage      : q_front - q_B  (> tol_q: the loop beats every plain schedule of the same or lower capacity);
                       loop_beats_sweep / loop_worse_than_sweep are only set for the cases the loop changed
      overloaded     : q_A > q_max minutes (queue that never clears: waits of hundreds of minutes, not informative);
                       loop_verdict leaves these cases out of the comparison
    """
    try:
        from app.services.fluid_approx import fluid_wait
    except ImportError:
        from fluid_approx import fluid_wait
    grid = np.asarray(grid, dtype=np.float64)
    R, T = grid.shape
    rho = grid.sum(axis=0)
    fixed = _first_server(grid, fixed_first)       # same fixed first server in A, in the loop and in the sweep
    a = np.asarray(arrivals, dtype=np.float64)
    rates = (list(rates) + [float(np.mean(rates))] * R)[:R]
    if theta_grid is None:
        theta_grid = np.round(np.arange(0.0, 1.0001, 0.025), 4)

    def summary(sols, th):
        covs = [s.coverage[:T] for s in sols]
        De = effective_grid(grid, covs)
        covered = sum(float(De[r][covs[r]].sum()) for r in range(R))
        cap = float(sum(c.sum() for c in covs))
        pax = sum(rates[r] * covs[r].astype(float) for r in range(R))
        return dict(sig=tuple(tuple(b) for b in _blocks(sols)), cap=cap, cov=covered / float(rho.sum()),
                    obj=covered - th * cap, q=fluid_wait(a, np.asarray(pax), 15.0))

    def plain(th):
        sols, _ = _plan(grid, grid, lengths, np.full(R, float(th)), K_total, length_totals, fixed, None)
        return summary(sols, th)

    sweep = [dict(theta=float(t), **plain(t)) for t in theta_grid]
    rows = []
    for th in thetas:
        A = plain(th)
        sols, tr, hist = inherited_loop(grid, lengths, th, K_total, length_totals, n_iter=n_iter, fixed=fixed, verbose=False)
        B = summary(sols, th)
        same = [s["theta"] for s in sweep if s["sig"] == B["sig"]]
        match = [s["cap"] for s in sweep if s["q"] <= B["q"] + tol_q] + ([A["cap"]] if A["q"] <= B["q"] + tol_q else [])
        cap_match = min(match) if match else np.inf
        cand = [s["q"] for s in sweep if s["cap"] <= B["cap"] + 1e-9] + ([A["q"]] if A["cap"] <= B["cap"] + 1e-9 else [])
        q_front = min(cand) if cand else np.inf
        rows.append(dict(theta=float(th), changed=A["sig"] != B["sig"], same_as_theta=min(same, key=lambda x: abs(x - th)) if same else np.nan,
                         iters=len(tr) - 1, stop=tr.attrs["stop"], cap_A=A["cap"], cap_B=B["cap"], q_A=A["q"], q_B=B["q"],
                         cov_A=A["cov"], cov_B=B["cov"], obj_A=A["obj"], obj_B=B["obj"], q_front=q_front,
                         advantage=q_front - B["q"], overloaded=bool(A["q"] > q_max),
                         loop_beats_sweep=bool(A["sig"] != B["sig"] and q_front - B["q"] > tol_q),
                         loop_worse_than_sweep=bool(A["sig"] != B["sig"] and q_front - B["q"] < -tol_q),
                         cap_match=cap_match,  cap_gap=cap_match - B["cap"]))
    return pd.DataFrame(rows)


def loop_verdict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary of loop_vs_theta_sweep rows (all days stacked), by theta. Cases with an overloaded queue are counted apart and
    left out of the comparison with the sweep.
    """
    ok = df[~df["overloaded"]]
    ch = ok[ok["changed"]]

    def by_theta(g):
        return pd.Series(dict(
            cases=len(g), fixed_point=(g["stop"] == "fixed point").mean(), changed=g["changed"].mean(),
            adds_capacity=(g["changed"] & (g["cap_B"] > g["cap_A"])).sum(),
            same_as_some_theta=(g["changed"] & g["same_as_theta"].notna()).sum(),
            loop_beats_sweep=g["loop_beats_sweep"].sum(), loop_worse_than_sweep=g["loop_worse_than_sweep"].sum(),
            obj_worse=(g["obj_B"] < g["obj_A"] - 1e-9).sum()))

    out = ok.groupby("theta").apply(by_theta, include_groups=False)
    beats = ch[ch["loop_beats_sweep"]]
    print(f"{len(df)} cases ({int(df['overloaded'].sum())} overloaded, left out); of the other {len(ok)}, the loop changes the schedule in "
          f"{len(ch)} ({len(ch) / max(len(ok), 1):.0%}). Of those {len(ch)}: plain pipeline at another theta gives exactly the same "
          f"schedule in {int(ch['same_as_theta'].notna().sum())}; the loop beats every plain schedule of equal or lower capacity in "
          f"{len(beats)}" + (f" (median advantage {beats['advantage'].median():.2f} min)" if len(beats) else "") +
          f"; it is worse than the best plain one in {int(ch['loop_worse_than_sweep'].sum())}.")
    return out