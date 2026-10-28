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
    objective = sum over rows of (covered_r - theta_r * capacity_r), with theta_r = res.theta_rows[r]
    (res.theta on every row if the result has no per-row values).
    """
    D = planner.demand_grid
    labels = list(res.server_labels)
    theta_rows = getattr(res, "theta_rows", None)
    theta_rows = np.full(len(labels), float(res.theta)) if theta_rows is None else np.asarray(theta_rows, dtype=float)
    pad = max((l for bl in blocks.values() for _, l in bl), default=1) - 1
    Dp = np.pad(D, ((0, 0), (0, pad)))
    covered = idle = objective = 0.0
    for r, label in enumerate(labels):
        for s, l in blocks.get(label, []):
            seg = Dp[r, s:s + l]
            c = float(seg.sum())
            covered += c
            idle += float(l) - c
            objective += c - theta_rows[r] * l
    return dict(
        covered_demand=covered,
        idle_capacity=idle,
        coverage=covered / float(D.sum()),
        objective=objective,
    )

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
        coverage_slot=_slotwise(df["demand"], df["processing"]),
    )

# random movement of some block
def perturb_blocks(blocks: dict, T: int, n_moves: int, rng: np.random.Generator, step: int = 1):
    """
    Move n_moves randomly chosen blocks (any lane), by 1..step slots, to the left or
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
        shifts = [d for m in range(1, step + 1) for d in (-m, m) if lo <= s + d <= hi]
        if not shifts:
            continue
        d = int(rng.choice(shifts))
        bl[i] = (s + d, l)
        moves.append((label, i, d))
    return new, moves


_EDIT_KEYS = {"lane", "block", "shift", "length", "remove"}

def edit_blocks(blocks, labels, edits, T, add=None):
    """
    Modifica a mano una schedule {label: [(start_slot, length_slots), ...]}.

    edits : lista di dict, uno per blocco da cambiare
              lane   : etichetta ('3') oppure posizione 0-based in `labels`
              block  : indice 0-based del blocco nella lane (ordinato per start; riferito alla baseline)
              shift  : slot di spostamento (+ = piu' tardi), default 0
              length : nuova lunghezza in slot (opzionale), lo start resta quello dopo lo shift
              remove : True per eliminare il blocco
    add   : lista di dict(lane=..., start=..., length=...) con nuovi blocchi (slot, 0-based)
    T     : numero di slot del giorno; ogni blocco deve stare in [0, T] e non sovrapporsi nella lane.
    Ritorna (new_blocks, log). Solleva ValueError se una modifica non e' valida.
    """
    labels = list(labels)

    def lane_of(x):
        if isinstance(x, str):
            if x not in labels and x not in blocks:
                raise ValueError(f"unknown lane {x!r}; lanes are {labels}")
            return x
        i = int(x)
        if not 0 <= i < len(labels):
            raise ValueError(f"lane index {i} out of range 0..{len(labels) - 1} ({labels})")
        return labels[i]

    cur = {k: [[int(s), int(l)] for s, l in v] for k, v in blocks.items()}
    log, touched = [], set()
    for e in edits or []:
        bad = set(e) - _EDIT_KEYS
        if bad:
            raise ValueError(f"unknown keys {sorted(bad)} in edit {e}; allowed: {sorted(_EDIT_KEYS)}")
        lab, i = lane_of(e["lane"]), int(e["block"])
        n = len(cur.get(lab, []))
        if not 0 <= i < n:
            raise ValueError(f"lane {lab!r} has {n} blocks (0..{n - 1}); block={i}")
        if (lab, i) in touched:
            raise ValueError(f"block {i} of lane {lab!r} is edited twice")
        touched.add((lab, i))
        if e.get("remove"):
            log.append(f"{lab} block {i}: removed {tuple(cur[lab][i])}")
            cur[lab][i] = None
            continue
        s, l = cur[lab][i]
        shift, new_len = int(e.get("shift", 0)), e.get("length")
        s2, l2 = s + shift, (l if new_len is None else int(new_len))
        cur[lab][i] = [s2, l2]
        log.append(f"{lab} block {i}: ({s},{l}) -> ({s2},{l2})")
    for a in add or []:
        lab = lane_of(a["lane"])
        cur.setdefault(lab, []).append([int(a["start"]), int(a["length"])])
        log.append(f"{lab}: added ({int(a['start'])},{int(a['length'])})")

    out = {}
    for lab, bl in cur.items():
        bl = sorted(tuple(b) for b in bl if b is not None)
        for s, l in bl:
            if l < 1 or s < 0 or s + l > T:
                raise ValueError(f"lane {lab!r}: block (start={s}, length={l}) is outside the day [0, {T}]")
        for (s1, l1), (s2, l2) in zip(bl, bl[1:]):
            if s2 < s1 + l1:
                raise ValueError(f"lane {lab!r}: blocks (start={s1}, length={l1}) and (start={s2}, length={l2}) overlap")
        if bl:
            out[lab] = bl
    return out, log