"""
Neighbourhood search on a schedule, evaluated with the fluid queue.

blocks = {label: [(start_slot, length_slots), ...]}

fluid_metrics(arrivals, rates, blocks, bin_minutes)  -> (mean wait [min], unserved pax)
neighbors(blocks, T)                                 -> every feasible single-block +-1 move
local_search(arrivals, rates, D, labels, alpha, blocks0, bin_minutes, ...) -> (best_blocks, trace DataFrame)

Search criterion (default, "constraint"): minimise the fluid mean wait among the schedules with
    unserved <= unserved(start) + unserved_tol.
Coverage / objective are not optimised, only tracked in the trace.
"""
import numpy as np
import pandas as pd

def capacity_per_slot(blocks, rates, T):
    """pax/bin the open lanes can process in each slot (rates: {label: pax per bin})."""
    cap = np.zeros(T)
    for label, bl in blocks.items():
        for s, l in bl:
            cap[s:s + l] += rates[label]
    return cap

def fluid_metrics(arrivals, rates, blocks, bin_minutes, refine=15):
    """
    Deterministic FIFO fluid queue. Arrivals and capacity are uniform inside each slot, each slot is split in
    refine sub-steps. Lindley recursion in closed form:  Q_n = S_n - min_{j<=n} S_j,  S = cumsum(arrivals - capacity).
    The wait of the mass arriving in a sub-step is (time at which cumulative departures reach its cumulative
    arrivals) - (arrival time). Mass that never departs by the end of the horizon is "unserved".
    Returns (mean wait in minutes over the served mass, unserved pax).
    """
    a_slot = np.asarray(arrivals, dtype=np.float64)
    T = len(a_slot)
    cap = capacity_per_slot(blocks, rates, T)
    a = np.repeat(a_slot / refine, refine)
    c = np.repeat(cap / refine, refine)
    dt = bin_minutes / refine
    S = np.concatenate([[0.0], np.cumsum(a - c)])
    Q = S - np.minimum.accumulate(S)                                  # queue at the sub-step boundaries
    A = np.concatenate([[0.0], np.cumsum(a)])
    Dep = A - Q                                                       # cumulative departures (nondecreasing)
    total = A[-1]
    served = Dep[-1]
    unserved = float(max(total - served, 0.0))
    if served <= 1e-12:
        return float("inf"), unserved
    tau = np.arange(len(A)) * dt
    x = (A[:-1] + A[1:]) / 2                                          # cumulative level at the middle of each mass
    t_arr = (np.arange(len(a)) + 0.5) * dt
    ok = (x <= served) & (a > 0)
    xs, ta, ws = x[ok], t_arr[ok], a[ok]
    j = np.clip(np.searchsorted(Dep, xs, side="left"), 1, len(Dep) - 1)
    d0, d1 = Dep[j - 1], Dep[j]
    frac = np.where(d1 > d0, (xs - d0) / np.where(d1 > d0, d1 - d0, 1.0), 0.0)
    t_dep = tau[j - 1] + frac * dt
    wait = np.maximum(t_dep - ta, 0.0)
    return float((ws * wait).sum() / ws.sum()), unserved

# neighbourhood search
def feasible_shifts(bl, i, T, step=1):
    """Shifts in (-step, +step) that keep block i of the (sorted) lane `bl` between its neighbours and inside [0, T]."""
    s, l = bl[i]
    lo = bl[i - 1][0] + bl[i - 1][1] if i > 0 else 0
    hi = (bl[i + 1][0] if i < len(bl) - 1 else T) - l
    return [d for d in (-step, step) if lo <= s + d <= hi]

def apply_shift(blocks, label, i, d):
    new = {k: list(v) for k, v in blocks.items()}
    s, l = new[label][i]
    new[label][i] = (s + d, l)
    return new

def neighbors(blocks, T, step=1):
    """Yield (move, new_blocks) with move = (label, block index, shift)."""
    for label, bl in blocks.items():
        for i in range(len(bl)):
            for d in feasible_shifts(bl, i, T, step):
                yield (label, i, d), apply_shift(blocks, label, i, d)

# evaluation of a schedule
def evaluate(blocks, arrivals, rates, D, labels, alpha, bin_minutes):
    row_of = {l: c for c, l in enumerate(labels)}
    covered = idle = 0.0
    for label, bl in blocks.items():
        for s, l in bl:
            seg = D[row_of[label], s:s + l]
            covered += float(seg.sum()); idle += float((1.0 - seg).sum())
    wait, unserved = fluid_metrics(arrivals, rates, blocks, bin_minutes)
    return dict(q_fluid=wait, unserved_fluid=unserved, covered_demand=covered, idle_capacity=idle,
                coverage=covered / float(D.sum()), objective=covered - alpha * idle)

# local search
def local_search(arrivals, rates, D, labels, alpha, blocks0, bin_minutes,
                 unserved_tol=0.0, max_iter=200, min_gain=1e-6, step=1, verbose=True):
    """
    Steepest descent: at each iteration evaluate all feasible +-step moves of one block, take the one with the
    lowest fluid mean wait among those with unserved <= unserved(blocks0) + unserved_tol, stop when no move
    improves the wait by more than min_gain (local optimum) or after max_iter iterations.
    Lengths, number of blocks and lane assignment never change (same budgets as the starting solution).
    Returns (best_blocks, trace): trace has one row per iteration (0 = start) with the move taken and the metrics.
    """
    T = len(arrivals)
    cur = {k: list(v) for k, v in blocks0.items()}
    m = evaluate(cur, arrivals, rates, D, labels, alpha, bin_minutes)
    u_cap = m["unserved_fluid"] + unserved_tol
    rows = [dict(iter=0, move=None, n_neighbors=0, **m)]
    for it in range(1, max_iter + 1):
        best = None
        n_nb = 0
        for move, nb in neighbors(cur, T, step):
            n_nb += 1
            w, u = fluid_metrics(arrivals, rates, nb, bin_minutes)
            if u <= u_cap + 1e-9 and w < m["q_fluid"] - min_gain and (best is None or w < best[0]):
                best = (w, move, nb)
        if best is None:
            break
        _, move, cur = best
        m = evaluate(cur, arrivals, rates, D, labels, alpha, bin_minutes)
        rows.append(dict(iter=it, move=move, n_neighbors=n_nb, **m))
        if verbose and it % 10 == 0:
            print(f"iter {it}: q_fluid={m['q_fluid']:.3f} unserved={m['unserved_fluid']:.2f} cov={m['coverage']:.3f}")
    trace = pd.DataFrame(rows)
    if verbose:
        a, b = trace.iloc[0], trace.iloc[-1]
        print(f"{len(trace) - 1} moves | q_fluid {a.q_fluid:.3f} -> {b.q_fluid:.3f} | unserved "
              f"{a.unserved_fluid:.2f} -> {b.unserved_fluid:.2f} | coverage {a.coverage:.3f} -> {b.coverage:.3f}")
    return cur, trace