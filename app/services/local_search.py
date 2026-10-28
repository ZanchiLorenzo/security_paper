"""
Local search on a schedule, evaluated with the fluid queue.
 
blocks = {label: [(start_slot, length_slots), ...]}  sorted, 0-based, inside [0, T]
 
    fluid_metrics(arrivals, rates, blocks, bin_minutes)  -> (mean wait [min], unserved pax)
    fluid_diagnostics(arrivals, rates, blocks, bin_minutes) -> DataFrame (arrivals, capacity, queue per slot)
    local_search(...)                                      -> (best_blocks, trace DataFrame)
 
Criterion: `penalty=None` (default) -> LEXICOGRAPHIC: 1) fewer unserved passengers, 2) lower mean wait;
`penalty=P` -> single metric J = mean wait per passenger, an unserved one counting as P minutes.
    - A move is "an unserved improvement" if it lowers the unserved by more than `unserved_tol` pax.
    - Otherwise it can only be taken if it lowers the wait AND keeps unserved <= best unserved so far + unserved_tol
      (the ceiling follows the best value, so the search cannot spend a gain it already made).
    Coverage, coverage_slot and objective are not optimised, only tracked in the trace.
 
Moves: relocate one block anywhere in the day (|shift| <= max_shift if given), always in its own lane (no jumps
between lanes), without overlapping the other blocks of the lane (it may jump over them).
Number of blocks, their lengths, their lanes and the total budget never change.
Optional iterated local search: random kicks followed by a new descent, keep only if lexicographically better.
"""
import numpy as np
import pandas as pd
 
# fluid queue
def capacity_per_slot(blocks, rates, T):
    """pax/bin the open lanes can process in each slot (rates: {label: pax per bin})."""
    cap = np.zeros(T)
    for label, bl in blocks.items():
        for s, l in bl:
            cap[s:s + l] += rates[label]
    return cap
 
def _fluid_from_cap(a_slot, cap, bin_minutes, refine=15):
    """Deterministic FIFO fluid queue from arrivals per slot and capacity per slot (see fluid_metrics)."""
    a = np.repeat(a_slot / refine, refine)
    c = np.repeat(cap / refine, refine)
    dt = bin_minutes / refine
    S = np.concatenate([[0.0], np.cumsum(a - c)])
    Q = S - np.minimum.accumulate(S)                                  # queue at the sub-step boundaries
    A = np.concatenate([[0.0], np.cumsum(a)])
    Dep = A - Q                                                       # cumulative departures (nondecreasing)
    served = Dep[-1]
    unserved = float(max(A[-1] - served, 0.0))
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
 
def fluid_metrics(arrivals, rates, blocks, bin_minutes, refine=15):
    """
    Deterministic FIFO fluid queue. Arrivals and capacity are uniform inside each slot, each slot is split in
    `refine` sub-steps. Lindley recursion in closed form:  Q_n = S_n - min_{j<=n} S_j,  S = cumsum(arrivals - capacity).
    The wait of the mass arriving in a sub-step is (time at which cumulative departures reach its cumulative
    arrivals) - (arrival time). Mass that never departs by the end of the horizon is "unserved".
    Returns (mean wait in minutes over the served mass, unserved pax).
    """
    a_slot = np.asarray(arrivals, dtype=np.float64)
    return _fluid_from_cap(a_slot, capacity_per_slot(blocks, rates, len(a_slot)), bin_minutes, refine)
 
def fluid_diagnostics(arrivals, rates, blocks, bin_minutes, refine=15):
    """Per slot: arrivals, capacity, queue at the end of the slot. Shows where the unserved come from."""
    a_slot = np.asarray(arrivals, dtype=np.float64)
    T = len(a_slot)
    cap = capacity_per_slot(blocks, rates, T)
    a = np.repeat(a_slot / refine, refine); c = np.repeat(cap / refine, refine)
    S = np.concatenate([[0.0], np.cumsum(a - c)])
    Q = (S - np.minimum.accumulate(S))[refine::refine]
    return pd.DataFrame(dict(arrivals=a_slot, capacity=cap, queue_end=Q))

# evaluation of a schedule
def evaluate(blocks, arrivals, rates, D, labels, theta, bin_minutes):
    """theta: number, or one value per row of D (same order as labels), e.g. res.theta_rows."""
    row_of = {l: c for c, l in enumerate(labels)}
    theta_rows = np.broadcast_to(np.asarray(theta, dtype=np.float64), (np.shape(D)[0],))
    covered = idle = objective = 0.0
    for label, bl in blocks.items():
        for s, l in bl:
            seg = D[row_of[label], s:s + l]
            c = float(seg.sum())
            covered += c; idle += float((1.0 - seg).sum())
            objective += c - theta_rows[row_of[label]] * l
    a = np.asarray(arrivals, dtype=np.float64)
    cap = capacity_per_slot(blocks, rates, len(a))
    wait, unserved = _fluid_from_cap(a, cap, bin_minutes)
    return dict(q_fluid=wait, unserved_fluid=unserved, covered_demand=covered, idle_capacity=idle,
                coverage=covered / float(D.sum()),                          # layer coverage (what the DP maximizes)
                coverage_slot=float(np.minimum(a, cap).sum() / a.sum()),    # true coverage, pooled capacity
                objective=objective)
 
# neighbourhood
def _fits(lane_blocks, s, l, T):
    """The window [s, s+l) lies in [0, T] and does not overlap the (start, length) blocks of the lane."""
    if s < 0 or s + l > T:
        return False
    return all(s + l <= s2 or s >= s2 + l2 for s2, l2 in lane_blocks)
 
def _candidates(blocks, T, max_shift):
    """
    Yield move = (label, block index, new_start) for every relocation of one block inside its own lane:
    any start in [0, T - length] (within max_shift slots of the current one if given) that does not overlap
    the other blocks of the lane.
    """
    for label, bl in blocks.items():
        for i, (s, l) in enumerate(bl):
            others = [b for k, b in enumerate(bl) if k != i]
            lo = 0 if max_shift is None else max(0, s - max_shift)
            hi = T - l if max_shift is None else min(T - l, s + max_shift)
            for s1 in range(lo, hi + 1):
                if s1 != s and _fits(others, s1, l, T):
                    yield (label, i, s1)
 
def _apply(blocks, move, l):
    label, i, s1 = move
    new = {k: list(v) for k, v in blocks.items()}
    del new[label][i]
    new[label].append((s1, l))
    new[label].sort()
    return new
 
# local search
def local_search(arrivals, rates, D, labels, theta, blocks0, bin_minutes,
                 penalty=None, j_tol=1e-6,
                 unserved_tol=1e-3, wait_tol=1e-6, max_iter=500, max_shift=None,
                 n_kicks=0, kick_size=3, seed=0, refine=15, verbose=True):
    """
    Descent on the fluid queue with two possible criteria.
 
    penalty = P (minutes)  -> SINGLE METRIC:
        J = ((N - U) * wait + P * U) / N     N = total arrivals, U = unserved, wait = mean wait of the served.
        Mean wait per passenger, an unserved passenger counting as P minutes. Steepest descent on J: take the
        best relocation if it lowers J by more than j_tol, stop otherwise. Kicks are kept if J improves.
    penalty = None         -> LEXICOGRAPHIC (default): unserved first, then wait:
        unserved_tol : pax. A move that lowers the unserved by more than this is an "unserved move" (taken even if the
                       wait gets worse); the others must keep unserved <= best unserved so far + unserved_tol and lower
                       the wait by more than wait_tol. Tiny default (0.001 pax) = strictly lexicographic; raise it
                       (e.g. 0.5) to ignore sub-passenger gains in unserved.
 
    theta     : only tracked (objective = covered - theta * capacity in the trace): a number or one value per row of D
                (pass res.theta_rows to keep the first-row theta). It does not drive the search.
    max_shift : None = a block can jump anywhere in the day (in its own lane); int = at most that many slots away.
    n_kicks   : iterated local search: after the descent, `n_kicks` times apply `kick_size` random relocations
                to the best schedule, descend again, keep the result only if it is better under the criterion.
    Returns (best_blocks, trace). trace: one row per accepted move (iter 0 = start) with metrics (and J if penalty is
    given), `kick` = index of the kick that produced the row (0 = first descent), `move` = (label, block, shift): the block, as numbered
    in the lane before the move, moved by `shift` slots (negative = earlier).
    """
    a_slot = np.asarray(arrivals, dtype=np.float64)
    T = len(a_slot)
    N = float(a_slot.sum())
    D = np.asarray(D, dtype=np.float64)
    f = lambda cap: _fluid_from_cap(a_slot, cap, bin_minutes, refine)
    Jf = (lambda w, u: ((N - u) * w + penalty * u) / N) if penalty is not None else None
    rng = np.random.default_rng(seed)
 
    def row(blocks, it, kick, move):
        m = evaluate(blocks, a_slot, rates, D, labels, theta, bin_minutes)
        if Jf is not None:
            m["J"] = Jf(m["q_fluid"], m["unserved_fluid"])
        return dict(iter=it, kick=kick, move=move, **m)
 
    def better(w, u, w0, u0):                                           # is (w, u) better than (w0, u0)?
        if Jf is not None:
            return Jf(w, u) < Jf(w0, u0) - j_tol
        return u < u0 - unserved_tol or (u <= u0 + unserved_tol and w < w0 - wait_tol)
 
    def descend(blocks, rows, kick):
        cap = capacity_per_slot(blocks, rates, T)
        w, u = f(cap)
        u_best = u
        for _ in range(max_iter):
            ceiling = u_best + unserved_tol
            best_u = best_w = None             # lexicographic: best "unserved move" / best "wait move"; J: best_w only
            J0 = Jf(w, u) if Jf is not None else None
            for move in _candidates(blocks, T, max_shift):
                label, i, s1 = move
                s0, l = blocks[label][i]
                c2 = cap.copy()
                c2[s0:s0 + l] -= rates[label]; c2[s1:s1 + l] += rates[label]
                w2, u2 = f(c2)
                if Jf is not None:
                    J2 = Jf(w2, u2)
                    if J2 < J0 - j_tol and (best_w is None or J2 < best_w[1]):
                        best_w = (u2, J2, move, c2, l, w2)
                    continue
                if u2 < u - unserved_tol:
                    if best_u is None or (u2, w2) < best_u[:2]:
                        best_u = (u2, w2, move, c2, l, w2)
                elif u2 <= ceiling and w2 < w - wait_tol and (best_w is None or w2 < best_w[1]):
                    best_w = (u2, w2, move, c2, l, w2)
            pick = best_u or best_w
            if pick is None:
                break
            u, _, move, cap, l, w = pick
            shift = move[2] - blocks[move[0]][move[1]][0]                 # new start - old start (slots)
            blocks = _apply(blocks, move, l)
            u_best = min(u_best, u)
            r = row(blocks, len(rows), kick, (move[0], move[1], shift))
            rows.append(r)
            if verbose and len(rows) % 10 == 0:
                extra = f" J={r['J']:.3f}" if Jf is not None else ""
                print(f"iter {len(rows)-1}: unserved={r['unserved_fluid']:.2f} q_fluid={r['q_fluid']:.3f}{extra} "
                      f"cov={r['coverage']:.3f}")
        return blocks, (u, w)
 
    cur = {k: list(v) for k, v in blocks0.items()}
    rows = [row(cur, 0, 0, None)]
    best, (bu, bw) = descend(cur, rows, 0)
 
    for k in range(1, n_kicks + 1):
        trial = {kk: list(v) for kk, v in best.items()}
        for _ in range(kick_size):                                      # random relocations (no improvement test)
            cands = list(_candidates(trial, T, max_shift))
            if not cands:
                break
            move = cands[rng.integers(len(cands))]
            trial = _apply(trial, move, trial[move[0]][move[1]][1])
        new_rows = []
        t_blocks, (tu, tw) = descend(trial, new_rows, k)
        if better(tw, tu, bw, bu):
            for r in new_rows:
                r["iter"] = len(rows); rows.append(r)
            best, bu, bw = t_blocks, tu, tw
            if verbose:
                extra = f" J={Jf(tw, tu):.3f}" if Jf is not None else ""
                print(f"kick {k}: improved -> unserved={tu:.2f} q_fluid={tw:.3f}{extra}")
 
    trace = pd.DataFrame(rows)
    if verbose:
        a, b = trace.iloc[0], trace.iloc[-1]
        extra = f" | J {a.J:.3f} -> {b.J:.3f}" if Jf is not None else ""
        print(f"{len(trace) - 1} moves | unserved {a.unserved_fluid:.2f} -> {b.unserved_fluid:.2f} | "
              f"q_fluid {a.q_fluid:.3f} -> {b.q_fluid:.3f}{extra} | coverage {a.coverage:.3f} -> {b.coverage:.3f}")
    return best, trace