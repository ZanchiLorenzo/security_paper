"""
Many schedules, one process pool.

SecuritySimulator.run() opens a new pool, regenerates the (identical) arrival scenarios and aggregates the
timelines / passenger tables / queue CDF for every schedule. When thousands of schedules are compared on the same
scenarios (perturbation experiments) that overhead dominates.

simulate_kpis() generates the scenarios once, ships them to the workers once, runs every (schedule, scenario) pair in a
single pool and returns only the KPIs, with the same definition as SecuritySimulator.df_kpis (mean over scenarios of
the numeric KPIs, scenario_id and seed excluded). It reuses SecuritySimulator (validation), SimulationTask and
single_scenario_worker from security_class unchanged.

Put this file next to security_class.py; it must live in a module (not in the notebook) so that the workers can import it.
"""
import concurrent.futures

import numpy as np
import pandas as pd

from app.services import security_class as sc

_SCENARIOS = None


def _init_worker(scenarios):
    global _SCENARIOS
    _SCENARIOS = scenarios

def _run_one(job):
    sched_idx, scenario_id, intervals, server_registry, timebin, start_dt, keep_timeline, keep_waits = job
    task = sc.SimulationTask(
        scenario_id=scenario_id,
        arrival_scenario=_SCENARIOS[scenario_id],
        intervals=intervals,
        server_registry=server_registry,
        timebin=timebin,
        start_dt=start_dt,
    )
    out = sc.single_scenario_worker(task)
    tl = None
    if keep_timeline:                      # only what the plot needs, to keep the inter-process traffic small
        tl = out["df_timeline"][["time_bin", "queue_length", "processing_rate", "theoretical_processing_rate"]]
    waits = None
    if keep_waits:                         # one float32 per passenger (minutes), NaN = never served
        waits = out["df_pax"]["queue_time"].to_numpy(dtype=np.float32)
    return sched_idx, out["kpis"].to_dict(), tl, waits

# queue percentiles over the scenarios, one value per time bin
_Q = {"p5": 5, "p25": 25, "median": 50, "p75": 75, "p95": 95}

def summarize_timeline(timelines, ndigits: int = 2) -> dict:
    """
    Timelines of the scenarios of ONE schedule -> small JSON-able dict (lists, one value per time bin):
    t0, step_min, queue_length_{p5,p25,median,p75,p95}, processing_rate_median, theoretical_processing_rate.
    Rebuild the DataFrame for plotting with timeline_to_df (same column names as SecuritySimulator.df_timeline).
    """
    all_tl = pd.concat(timelines, ignore_index=True)
    g = all_tl.groupby("time_bin")
    out = {}
    times = sorted(all_tl["time_bin"].unique())
    q = g["queue_length"]
    for name, p in _Q.items():
        out[f"queue_length_{name}"] = q.quantile(p / 100).reindex(times).round(ndigits).tolist()
    out["processing_rate_median"] = g["processing_rate"].median().reindex(times).round(ndigits).tolist()
    out["theoretical_processing_rate"] = g["theoretical_processing_rate"].first().reindex(times).round(ndigits).tolist()
    step = float((times[1] - times[0]).total_seconds() / 60) if len(times) > 1 else 0.0
    return dict(t0=pd.Timestamp(times[0]).isoformat(), step_min=step, **out)

def timeline_to_df(d) -> pd.DataFrame:
    """dict (or JSON string) from summarize_timeline -> DataFrame with time_bin and the aggregated columns."""
    import json
    d = json.loads(d) if isinstance(d, str) else d
    cols = {k: v for k, v in d.items() if isinstance(v, list)}
    n = len(next(iter(cols.values())))
    df = pd.DataFrame(cols)
    df.insert(0, "time_bin", pd.Timestamp(d["t0"]) + pd.to_timedelta(np.arange(n) * d["step_min"], unit="min"))
    return df

def simulate_kpis(
    flight_demand,
    intervals_list,
    server_registry,
    n_scenarios: int,
    n_workers: int,
    timebin: int,
    base_seed: int,
    chunksize: int | None = None,
    return_timeline: bool = False,
    return_waits: bool = False,
):
    """
    intervals_list : one list of Interval per schedule (e.g. utils.server_schedule_to_intervals(...))
    Returns a DataFrame with one row per schedule, in the same order, with the KPI columns of df_kpis
    (qtime_mean, n_unserved, qtime_above_30_pct, ...).
    return_timeline=True: returns (df_kpis, timelines), timelines[i] = summarize_timeline(...) of schedule i
    (queue percentiles and processing rate per time bin, over the scenarios), to be saved next to the row.
    return_waits=True: also returns waits, waits[i][s] = float32 array of the queue times (minutes) of all the
    passengers of scenario s for schedule i (length = number of pax of that scenario, NaN = not served).
    Return value: df_kpis, then timelines (if return_timeline), then waits (if return_waits).
    """
    scenarios = sc.ArrivalGenerator().generate_scenarios(
        flight_demand=flight_demand, n_scenarios=n_scenarios, base_seed=base_seed)
    arrival_dts = [p.arrival_dt for s in scenarios.values() for p in s.passengers]
    first_arrival = min(arrival_dts) if arrival_dts else None

    jobs = []
    for i, intervals in enumerate(intervals_list):
        # the constructor validates the intervals exactly as in run()
        sim = sc.SecuritySimulator(flight_demand=flight_demand, intervals=intervals, server_registry=server_registry)
        interval_start = min(iv.start for iv in sim.intervals)
        start = interval_start if first_arrival is None else min(first_arrival, interval_start)
        start = pd.Timestamp(start).floor(f"{timebin}min").to_pydatetime()      # same origin as run()
        for scenario_id in scenarios:
            jobs.append((i, scenario_id, sim.intervals, server_registry, timebin, start,
                         return_timeline, return_waits))

    if chunksize is None:
        chunksize = max(1, len(jobs) // (n_workers * 8))
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(scenarios,)) as ex:
        results = list(ex.map(_run_one, jobs, chunksize=chunksize))

    per_schedule = [[] for _ in intervals_list]
    tls = [[] for _ in intervals_list]
    waits = [[] for _ in intervals_list]
    for i, kpis, tl, w in results:
        per_schedule[i].append(kpis)
        if tl is not None:
            tls[i].append(tl)
        if w is not None:
            waits[i].append(w)

    rows = []
    for kpis in per_schedule:
        df = pd.DataFrame(kpis)
        numeric = [c for c in df.select_dtypes(include="number").columns if c not in {"scenario_id", "seed"}]
        rows.append(df[numeric].mean())
    df_kpis = pd.DataFrame(rows).reset_index(drop=True)
    out = [df_kpis]
    if return_timeline:
        out.append([summarize_timeline(t) for t in tls])
    if return_waits:
        out.append(waits)
    return out[0] if len(out) == 1 else tuple(out)