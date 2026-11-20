"""
Simulated annealing on a schedule, evaluated with the fluid queue (fluid_approx).

blocks = {label: [(start_slot, length_slots), ...]}  sorted, 0-based, inside [0, T]
    evaluate(...)                                           -> dict of metrics of a schedule
    simulated_annealing(...)                                -> (best_blocks, trace DataFrame)
    plot_annealing_trace(trace)

Objective of the annealing: minimise the mean queue time (q_fluid). Coverage and objective
(covered - theta * capacity) are not optimised, only tracked in the trace.

Moves: relocate one block inside its own lane (no jumps between lanes), without overlapping the other blocks of the
lane. `fixed` lanes never move. Number of blocks, lengths, lanes and the total budget never change.
"""
import numpy as np
import pandas as pd
from app.services.coverage_dp import effective_grid
from app.services.fluid_approx import capacity_per_slot, fluid_wait


# evaluation of a schedule
def evaluate(blocks, arrivals, rates, D, labels, theta, bin_minutes, inherit=True, refine=15):
    """
    Metrics of a schedule. D: demand grid of the lanes in `labels` (same order), theta: number or one value per row of D
    (e.g. res.theta_rows). inherit=True: the blocks of row r are valued on the demand the lower rows leave uncovered
    (coverage_dp.effective_grid), as in SecurityPlanner.run: then covered_demand is the true coverage of the layers and
    objective = covered - theta * capacity. q_fluid: mean wait of the fluid queue (fluid_approx.fluid_wait).
    """
    D = np.asarray(D, dtype=np.float64)
    row_of = {l: c for c, l in enumerate(labels)}
    theta_rows = np.broadcast_to(np.asarray(theta, dtype=np.float64), (D.shape[0],))
    cov = np.zeros(D.shape, dtype=bool)
    for label, bl in blocks.items():
        for s, l in bl:
            cov[row_of[label], s:s + l] = True
    De = effective_grid(D, list(cov)) if inherit else D
    covered = idle = objective = 0.0
    for label, bl in blocks.items():
        for s, l in bl:
            seg = De[row_of[label], s:s + l]
            c = float(seg.sum())
            covered += c; idle += float((1.0 - seg).sum())
            objective += c - theta_rows[row_of[label]] * l
    a = np.asarray(arrivals, dtype=np.float64)
    cap = capacity_per_slot(blocks, rates, len(a))
    return dict(q_fluid=fluid_wait(a, cap, bin_minutes, refine), covered_demand=covered, idle_capacity=idle,
                coverage=float(np.minimum(a, cap).sum() / a.sum()),    # true coverage, pooled capacity of real rates
                objective=objective)


# neighbourhood
def _fits(lane_blocks, s, l, T, bounds=None):
    """The window [s, s+l) lies in [0, T] (in [bounds[0], bounds[1]] if given) and does not overlap the (start, length) blocks of the lane."""
    lo, hi = (0, T) if bounds is None else (max(0, int(bounds[0])), min(T, int(bounds[1])))
    if s < lo or s + l > hi:
        return False
    return all(s + l <= s2 or s >= s2 + l2 for s2, l2 in lane_blocks)


def _apply(blocks, move, l):
    label, i, s1 = move
    new = {k: list(v) for k, v in blocks.items()}
    del new[label][i]
    new[label].append((s1, l))
    new[label].sort()
    return new


# simulated annealing
def simulated_annealing(arrivals, rates, D, labels, theta, blocks0, bin_minutes,
                        fixed=None, n_iter=5000, T0=None, T_end=None, p0=0.5, cooling_ratio=1e-3,
                        max_step=4, p_jump=0.1, full_every=0, bounds=None,
                        seed=0, refine=15, verbose=True):
    """
    Simulated annealing on the fluid queue. Energy E = q_fluid = mean queue time of all passengers (minutes), to be
    minimised.

    Neighbour: one random block (any lane not in `fixed`) is moved inside its own lane, by +-1..max_step slots
    (prob. 1 - p_jump) or to a uniformly random feasible start of the lane (prob. p_jump, to escape local minima).
    Accept if dE <= 0, else with probability exp(-dE / T). Geometric cooling T_k = T0 * (T_end / T0) ** (k / n_iter).

    T0       : initial temperature in minutes. None = calibrated on the start: a worsening move of typical size
               (median positive dE over 200 random moves) is accepted with probability p0 (default 0.5).
    T_end    : final temperature; None = T0 * cooling_ratio.
    fixed    : labels of the lanes that must not move (first_server lane).
    bounds   : (start_slot, end_slot): the blocks of the movable lanes must lie inside [start_slot, end_slot) (operating
               hours of the first server: no lane opens when the base lane is closed). The start schedule must respect it.
    full_every : if > 0, every that many iterations the current schedule is evaluated with `evaluate` (coverage,
               objective) and stored in the trace (NaN in the other rows). 0 = only q_fluid (fast).
    Returns (best_blocks, trace). trace = one row per iteration (iter 0 = start):
      iter, T, move=(label, block, shift) or None, E_cand, E_cur (after the accept step), E_best, accepted,
      q_fluid of the CURRENT schedule, n_accepted (cumulative), + coverage/objective if full_every.
    Plot with plot_annealing_trace(trace).
    """
    a_slot = np.asarray(arrivals, dtype=np.float64)
    T = len(a_slot)
    D = np.asarray(D, dtype=np.float64)
    fixed = set(fixed or ())
    rng = np.random.default_rng(seed)
    energy_of_cap = lambda cap: fluid_wait(a_slot, cap, bin_minutes, refine)

    cur = {k: list(v) for k, v in blocks0.items()}
    movable = [(lab, i) for lab, bl in cur.items() if lab not in fixed for i in range(len(bl))]
    if not movable:
        raise ValueError("no movable blocks (all lanes fixed or empty)")
    b0, b1 = (0, T) if bounds is None else (int(bounds[0]), int(bounds[1]))
    for lab, i in movable:
        s_, l_ = cur[lab][i]
        if s_ < b0 or s_ + l_ > b1:
            raise ValueError(f"start schedule: block ({s_}, {l_}) of lane {lab!r} is outside bounds {bounds}")
    cap = capacity_per_slot(cur, rates, T)
    E = energy_of_cap(cap)

    def propose(blocks):
        """-> (label, i, s1) or None. Block indices refer to the lane sorted by start (as _apply keeps it)."""
        for _ in range(30):
            lab = movable[rng.integers(len(movable))][0]
            bl = blocks[lab]
            i = int(rng.integers(len(bl)))
            s, l = bl[i]
            others = [b for k, b in enumerate(bl) if k != i]
            if rng.random() < p_jump:
                s1 = int(rng.integers(b0, b1 - l + 1)) if b1 - l >= b0 else s
            else:
                s1 = s + int(rng.choice([-1, 1])) * int(rng.integers(1, max_step + 1))
            if s1 != s and _fits(others, s1, l, T, (b0, b1)):
                return lab, i, s1
        return None

    def try_move(blocks, cap, move):
        """Energy and capacity after the move."""
        lab, i, s1 = move
        s0, l = blocks[lab][i]
        c2 = cap.copy()
        c2[s0:s0 + l] -= rates[lab]; c2[s1:s1 + l] += rates[lab]
        return energy_of_cap(c2), c2, l

    if T0 is None:
        d = []
        for _ in range(200):
            mv = propose(cur)
            if mv is None:
                continue
            E2, _, _ = try_move(cur, cap, mv)
            if E2 > E + 1e-9:
                d.append(E2 - E)
        T0 = float(np.median(d)) / -np.log(p0) if d else 1.0
    T_end = T0 * cooling_ratio if T_end is None else T_end
    if verbose:
        print(f"annealing: {n_iter} iters, T0={T0:.4g} -> T_end={T_end:.4g} min, E0={E:.4f}")

    best, E_best = {k: list(v) for k, v in cur.items()}, E
    nacc = 0
    rows = [dict(iter=0, T=T0, move=None, E_cand=E, E_cur=E, E_best=E_best, accepted=True, q_fluid=E, n_accepted=0)]
    for k in range(1, n_iter + 1):
        Tk = T0 * (T_end / T0) ** (k / n_iter)
        mv = propose(cur)
        rec = dict(iter=k, T=Tk, move=None, E_cand=np.nan, accepted=False)
        if mv is not None:
            E2, c2, l = try_move(cur, cap, mv)
            rec["E_cand"] = E2
            if E2 <= E or rng.random() < np.exp(-(E2 - E) / Tk):
                lab, i, s1 = mv
                rec["move"] = (lab, i, s1 - cur[lab][i][0])
                cur = _apply(cur, mv, l)
                cap, E = c2, E2
                rec["accepted"] = True
                nacc += 1
                if E < E_best - 1e-12:
                    best, E_best = {kk: list(v) for kk, v in cur.items()}, E
        rec.update(E_cur=E, E_best=E_best, q_fluid=E, n_accepted=nacc)
        if full_every and k % full_every == 0:
            m = evaluate(cur, a_slot, rates, D, labels, theta, bin_minutes, refine=refine)
            rec.update(coverage=m["coverage"], objective=m["objective"])
        rows.append(rec)
        if verbose and k % max(1, n_iter // 10) == 0:
            print(f"iter {k}: T={Tk:.4g} E_cur={E:.4f} E_best={E_best:.4f} acc={nacc / k:.1%}")
    return best, pd.DataFrame(rows)


def plot_annealing_trace(trace, window=None, title=None):
    """Energy of the current / best schedule per iteration, temperature (log, right axis), rolling acceptance rate."""
    import matplotlib.pyplot as plt
    window = window or max(10, len(trace) // 50)
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5), gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(trace["iter"], trace["E_cur"], color="tab:blue", lw=0.6, alpha=0.7, label="current")
    a1.plot(trace["iter"], trace["E_best"], color="tab:red", lw=1.5, label="best")
    a1.set_ylabel("energy = mean wait (min)")
    a1b = a1.twinx()
    a1b.plot(trace["iter"], trace["T"], color="gray", lw=1, ls="--", label="temperature")
    a1b.set_yscale("log"); a1b.set_ylabel("T (min)", color="gray")
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper right", handlelength=1.2, borderpad=0.4, labelspacing=0.3)
    acc = trace["accepted"].astype(float).rolling(window, min_periods=1).mean()
    a2.plot(trace["iter"], acc, color="tab:green", lw=1)
    a2.set_ylim(0, 1); a2.set_ylabel(f"accept. rate\n(rolling {window})"); a2.set_xlabel("iteration")
    a1.set_title(title or f"E {trace['E_cur'].iloc[0]:.2f} -> best {trace['E_best'].iloc[-1]:.2f}", fontsize=10)
    fig.tight_layout(); plt.show()