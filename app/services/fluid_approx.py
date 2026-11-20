"""
Deterministic fluid approximation of the security queue: FIFO, one queue served by the pooled open lanes.

It mirrors the simulator (QueueSimulationSinglequeue, drain_queue=True): when the last window ends, the lanes open in its
last slot stay open until the queue is empty, so nobody is left unserved and the mean wait counts every passenger.

    capacity_per_slot(blocks, rates, T)            pax per slot the open lanes can process
    fluid_wait(arrivals, capacity, bin_minutes)    mean wait of all passengers [min]
    fluid_diagnostics(arrivals, capacity, ...)     DataFrame: arrivals, capacity, queue at the end of each slot
    fluid_from_result(res)                         both, straight from a PlannerResults (res.df_curves)

blocks = {label: [(start_slot, length_slots), ...]},  rates = {label: pax per slot}.
"""
import numpy as np
import pandas as pd

def capacity_per_slot(blocks, rates, T):
    """pax per slot the open lanes can process (rates: {label: pax per slot})."""
    cap = np.zeros(T)
    for label, bl in blocks.items():
        for s, l in bl:
            cap[s:s + l] += rates[label]
    return cap

def _substeps(arrivals, capacity, refine):
    """
    Arrivals and capacity (uniform inside each slot) on `refine` sub-steps per slot. After the last open slot the lanes
    that were open in it keep working, for as many extra sub-steps as the queue needs to empty (drain).
    """
    a_slot = np.asarray(arrivals, dtype=np.float64)
    cap = np.asarray(capacity, dtype=np.float64)
    open_slots = np.flatnonzero(cap > 1e-12)
    rate_end = float(cap[open_slots[-1]]) if open_slots.size else 0.0
    if open_slots.size:
        cap = cap.copy()
        cap[open_slots[-1] + 1:] = rate_end
    a = np.repeat(a_slot / refine, refine)
    c = np.repeat(cap / refine, refine)
    if rate_end > 0:
        S = np.concatenate([[0.0], np.cumsum(a - c)])
        q_end = float(S[-1] - S.min())                                   # queue left at the end of the horizon
        n_extra = int(np.ceil(q_end / (rate_end / refine))) + 1
        a = np.concatenate([a, np.zeros(n_extra)])
        c = np.concatenate([c, np.full(n_extra, rate_end / refine)])
    return a, c, cap

def fluid_wait(arrivals, capacity, bin_minutes, refine=15):
    """
    Mean wait [min] of all passengers in the fluid FIFO queue. arrivals / capacity: pax per slot (e.g. the columns
    "demand" and "processing" of df_curves). Each slot is split in `refine` sub-steps.
    Lindley recursion in closed form: Q_n = S_n - min_{j<=n} S_j, S = cumsum(arrivals - capacity). The wait of the mass
    arriving in a sub-step is (time at which the cumulative departures reach its cumulative arrivals) - (arrival time).
    nan if nobody arrives, inf if there are arrivals and no lane is ever open.
    """
    a, c, _ = _substeps(arrivals, capacity, refine)
    if a.sum() <= 1e-12:
        return float("nan")
    dt = bin_minutes / refine
    S = np.concatenate([[0.0], np.cumsum(a - c)])
    Q = S - np.minimum.accumulate(S)                                     # queue at the sub-step boundaries
    A = np.concatenate([[0.0], np.cumsum(a)])
    Dep = A - Q                                                          # cumulative departures (nondecreasing)
    if Dep[-1] <= 1e-12:
        return float("inf")
    tau = np.arange(len(A)) * dt
    ok = a > 0
    x = ((A[:-1] + A[1:]) / 2)[ok]                                       # cumulative level at the middle of each mass
    t_arr = ((np.arange(len(a)) + 0.5) * dt)[ok]
    j = np.clip(np.searchsorted(Dep, x, side="left"), 1, len(Dep) - 1)
    d0, d1 = Dep[j - 1], Dep[j]
    frac = np.where(d1 > d0, (x - d0) / np.where(d1 > d0, d1 - d0, 1.0), 0.0)
    wait = np.maximum(tau[j - 1] + frac * dt - t_arr, 0.0)
    return float((a[ok] * wait).sum() / a[ok].sum())

def fluid_diagnostics(arrivals, capacity, bin_minutes=None, refine=15):
    """
    Per slot: arrivals, capacity (with the drain after the last open slot) and queue [pax] at the end of the slot.
    The queue left after the last slot is emptied after the horizon and is not in the table. bin_minutes is not needed
    (the queue in pax does not depend on it); it is accepted to mirror fluid_wait.
    """
    a_slot = np.asarray(arrivals, dtype=np.float64)
    a, c, cap = _substeps(a_slot, capacity, refine)
    S = np.concatenate([[0.0], np.cumsum(a - c)])
    Q = S - np.minimum.accumulate(S)
    T = len(a_slot)
    return pd.DataFrame(dict(arrivals=a_slot, capacity=cap, queue_end=Q[refine:T * refine + 1:refine]))

def fluid_from_result(res, refine=15):
    """(mean wait [min], diagnostics DataFrame indexed by time) of a PlannerResults, from res.df_curves."""
    df = res.df_curves
    a, cap = df["demand"].to_numpy(dtype=float), df["processing"].to_numpy(dtype=float)
    diag = fluid_diagnostics(a, cap, res.bin_minutes, refine)
    diag.index = pd.DatetimeIndex(df["time"])
    return fluid_wait(a, cap, res.bin_minutes, refine), diag