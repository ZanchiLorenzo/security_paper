import dataclasses
import json
import numpy as np
import pandas as pd
try:
    from app.services.coverage_dp import effective_grid
except ImportError:
    from coverage_dp import effective_grid

def schedule_to_blocks(planner, res) -> dict:
    """res.server_schedule {label: [(start_dt, end_dt), ...]} -> {label: [(start_slot, length_slots), ...]}."""
    t0, step = pd.Timestamp(planner.time_values[0]), planner.bin_duration
    out = {}
    for label, windows in res.server_schedule.items():
        out[label] = sorted(
            (int(round((pd.Timestamp(s) - t0) / step)), int(round((pd.Timestamp(e) - pd.Timestamp(s)) / step)))
            for s, e in windows
        )
    return out

def blocks_to_schedule(planner, blocks: dict) -> dict:
    """{label: [(start_slot, length_slots), ...]} -> {label: [(start_dt, end_dt), ...]}, same format as res.server_schedule."""
    t0, step = pd.Timestamp(planner.time_values[0]), planner.bin_duration
    return {
        label: [((t0 + int(s) * step).to_pydatetime(), (t0 + (int(s) + int(l)) * step).to_pydatetime())
                for s, l in bl]
        for label, bl in blocks.items() if len(bl)
    }

def blocks_to_json(blocks: dict) -> str:
    return json.dumps({k: [[int(s), int(l)] for s, l in v] for k, v in blocks.items()})

def parse_blocks(x) -> dict:
    """Inverse of blocks_to_json (also accepts an already parsed dict). Safe to store in xlsx/csv."""
    d = x if isinstance(x, dict) else json.loads(x)
    return {k: [(int(s), int(l)) for s, l in v] for k, v in d.items()}

# coverage of a schedule
def _cover_mask(labels, blocks: dict, shape) -> np.ndarray:
    """bool (n_rows, T): slots in which each row (labels order) is open."""
    cov = np.zeros(shape, dtype=bool)
    for r, label in enumerate(labels):
        for s, l in blocks.get(label, []):
            cov[r, int(s):int(s) + int(l)] = True
    return cov

def effective_grid_from_blocks(planner, res, blocks: dict, inherit: bool = True) -> np.ndarray:
    """Demand each row sees with the schedule `blocks` (coverage_dp.effective_grid); the plain layers if inherit=False."""
    D = planner.demand_grid
    if not inherit:
        return D
    return effective_grid(D, list(_cover_mask(list(res.server_labels), blocks, D.shape)))

def coverage_from_blocks(planner, res, blocks: dict, inherit: bool = True) -> dict:
    """
    Coverage of a schedule represented as blocks. Layer r of the demand grid is operated by res.server_labels[r].
    inherit=True: the blocks of row r are valued on the effective grid (demand the lower rows leave uncovered), so
    coverage = sum of block coverages / total load is the true coverage of the layers; False: plain layers.
    objective = sum over rows of (covered_r - theta_r * capacity_r), theta_r = res.theta_rows[r].
    """
    D = planner.demand_grid
    labels = list(res.server_labels)
    theta_rows = np.asarray(res.theta_rows, dtype=float)
    De = effective_grid_from_blocks(planner, res, blocks, inherit)
    covered = idle = objective = 0.0
    for r, label in enumerate(labels):
        for s, l in blocks.get(label, []):
            c = float(De[r, s:s + l].sum())
            covered += c
            idle += float(l) - c
            objective += c - theta_rows[r] * l
    return dict(covered_demand=covered, idle_capacity=idle, coverage=covered / float(D.sum()), objective=objective)

def _slotwise(demand, processing) -> float:
    """True coverage: sum_t min(demand_t, processing_t) / sum_t demand_t."""
    d = np.asarray(demand, float); c = np.asarray(processing, float)
    return float(np.minimum(d, c).sum() / d.sum()) if d.sum() > 0 else float("nan")


def coverage_slotwise(arrivals, rate_by_label, blocks: dict) -> float:
    """
    True coverage of a schedule: sum_t min(arrivals_t, capacity_t) / sum_t arrivals_t,
    with capacity_t = sum of the rates (pax per bin) of the lanes open at t.
    arrivals: array (T,) of pax per bin; blocks: {label: [(start_slot, length_slots), ...]}.
    """
    a = np.asarray(arrivals, dtype=np.float64)
    cap = np.zeros(len(a))
    for label, bl in blocks.items():
        for s, l in bl:
            cap[int(s):int(s) + int(l)] += rate_by_label[label]
    return _slotwise(a, cap)


def res_with_blocks(planner, res, blocks: dict, inherit: bool = True):
    """
    Copy of the planner result in which the schedule is replaced by `blocks`, so that any plot that
    takes a result shows the perturbed schedule.
    Updates server_schedule, coverage and df_curves (processing, servers) the way SecurityPlanner.run does (also demand_grid_eff).
    """
    labels = list(res.server_labels)
    n_bins = len(res.df_curves)
    active = np.zeros((len(labels), n_bins), dtype=bool)
    for r, label in enumerate(labels):
        for s, l in blocks.get(label, []):
            active[r, int(s):min(int(s) + int(l), n_bins)] = True
    rates = np.array([planner.rate_by_label[label] for label in labels])
    df = res.df_curves.copy()
    df["processing"] = rates @ active
    df["servers"] = pd.Series(
        [tuple(labels[c] for c in np.flatnonzero(active[:, t])) for t in range(n_bins)], dtype=object)
    return dataclasses.replace(
        res,
        server_schedule=blocks_to_schedule(planner, blocks),
        df_curves=df,
        demand_grid_eff=effective_grid_from_blocks(planner, res, blocks, inherit),
        coverage=coverage_from_blocks(planner, res, blocks, inherit)["coverage"],
        coverage_slot=_slotwise(df["demand"], df["processing"]),
    )

# random movement of some block
def perturb_blocks(blocks: dict, T: int, n_moves: int, rng: np.random.Generator, step: int = 1):
    """
    Move n_moves randomly chosen blocks (any lane), each by exactly `step` slots (default 1), to the left or
    to the right with equal probability. Lengths and lane assignment never change.

    The result is always feasible: a block stays between its neighbours in the same lane (adjacent blocks
    are allowed; neighbours already moved count with their new position) and ends by slot T
    (no overhang, as in the allocator). A block that cannot move in any direction is skipped
    and another one is drawn.

    Returns (new_blocks, moves): moves = [(label, block index, shift), ...] actually applied;
    len(moves) < n_moves only if too few blocks could move.
    """
    new = {k: list(v) for k, v in blocks.items()}
    candidates = [(label, i) for label, bl in new.items() for i in range(len(bl))]
    rng.shuffle(candidates)
    moves = []
    for label, i in candidates:
        if len(moves) == n_moves:
            break
        bl = new[label]
        s, l = bl[i]
        lo = bl[i - 1][0] + bl[i - 1][1] if i > 0 else 0
        hi = (bl[i + 1][0] if i < len(bl) - 1 else T) - l          # the block must end by T (no overhang)
        shifts = [d for d in (-step, step) if lo <= s + d <= hi]
        if not shifts:
            continue
        d = int(rng.choice(shifts))
        bl[i] = (s + d, l)
        moves.append((label, i, d))
    return new, moves

# what changed between two schedules of df_pert
_METRICS = ["coverage", "objective", "qtime_mean", "n_unserved"]

def _pick_row(df_pert: pd.DataFrame, x) -> pd.Series:
    """x = value of the 'rep' column, or directly a row (pd.Series) of df_pert."""
    return x if isinstance(x, pd.Series) else df_pert.loc[df_pert["rep"] == x].iloc[0]

def diff_schedules(df_pert: pd.DataFrame, rep_a, rep_b, planner=None, verbose: bool = True) -> pd.DataFrame:
    """
    Compare the schedules of two rows of df_pert (a = reference, e.g. the baseline rep=0; b = the other one).

    Returns one row per block that differs:
      kind = 'moved'   same lane, same length, different start (shift = start_b - start_a, in slots)
             'removed' block only in a
             'added'   block only in b
    Blocks are matched by their position in the lane, which is exact when blocks are only shifted (same
    number of blocks and same lengths per lane); otherwise the lane is compared as a set of (start, length).
    With `planner`, start/end times of both schedules are added.
    Prints the change of coverage / objective / queue KPIs between the two rows when verbose.
    """
    ra, rb = _pick_row(df_pert, rep_a), _pick_row(df_pert, rep_b)
    A, B = parse_blocks(ra["blocks"]), parse_blocks(rb["blocks"])
    labels = list(A) + [k for k in B if k not in A]

    recs = []
    for label in labels:
        a, b = A.get(label, []), B.get(label, [])
        if len(a) == len(b) and all(la == lb for (_, la), (_, lb) in zip(a, b)):
            for i, ((sa, l), (sb, _)) in enumerate(zip(a, b)):
                if sa != sb:
                    recs.append(dict(lane=label, block=i, kind="moved", length=l,
                                     start_a=sa, start_b=sb, shift=sb - sa))
        else:
            for s, l in sorted(set(a) - set(b)):
                recs.append(dict(lane=label, block=a.index((s, l)), kind="removed", length=l,
                                 start_a=s, start_b=None, shift=None))
            for s, l in sorted(set(b) - set(a)):
                recs.append(dict(lane=label, block=b.index((s, l)), kind="added", length=l,
                                 start_a=None, start_b=s, shift=None))
    diff = pd.DataFrame(recs, columns=["lane", "block", "kind", "length", "start_a", "start_b", "shift"])

    if planner is not None and len(diff):
        t0, step = pd.Timestamp(planner.time_values[0]), planner.bin_duration
        hhmm = lambda s: None if pd.isna(s) else (t0 + int(s) * step).strftime("%H:%M")
        for side in ("a", "b"):
            diff[f"time_{side}"] = [
                None if pd.isna(s) else f"{hhmm(s)}-{hhmm(s + l)}"
                for s, l in zip(diff[f"start_{side}"], diff["length"])]

    if verbose:
        print(f"rep {ra['rep']} (a)  vs  rep {rb['rep']} (b): "
              f"{(diff.kind == 'moved').sum()} moved, {(diff.kind == 'removed').sum()} removed, "
              f"{(diff.kind == 'added').sum()} added")
        kp = pd.DataFrame(
            {m: dict(a=ra[m], b=rb[m], delta=rb[m] - ra[m]) for m in _METRICS if m in ra.index}).T
        print(kp.round(4).to_string())
    return diff