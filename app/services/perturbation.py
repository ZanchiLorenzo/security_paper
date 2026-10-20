import dataclasses
import json
import numpy as np
import pandas as pd

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
def coverage_from_blocks(planner, res, blocks: dict) -> dict:
    """
    Compute the coverage of a schedule represented as blocks.
    Layer r of the demand grid is operated by res.server_labels[r].
    coverage = covered demand / total demand (dropped layers included in the denominator);
    idle capacity counts the slots of a block that overhang the horizon (zero demand).
    """
    D = planner.demand_grid
    pad = max((l for bl in blocks.values() for _, l in bl), default=1) - 1
    Dp = np.pad(D, ((0, 0), (0, pad)))
    covered = idle = 0.0
    for r, label in enumerate(res.server_labels):
        for s, l in blocks.get(label, []):
            seg = Dp[r, s:s + l]
            covered += float(seg.sum())
            idle += float((1.0 - seg).sum())
    return dict(
        covered_demand=covered,
        idle_capacity=idle,
        coverage=covered / float(D.sum()),
        objective=covered - res.alpha * idle,
    )

def res_with_blocks(planner, res, blocks: dict):
    """
    Copy of the planner result in which the schedule is replaced by `blocks`, so that any plot that
    takes a result shows the perturbed schedule.
    Updates server_schedule, coverage and df_curves (processing, servers) the way SecurityPlanner.run_multi does.
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
        coverage=coverage_from_blocks(planner, res, blocks)["coverage"],
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