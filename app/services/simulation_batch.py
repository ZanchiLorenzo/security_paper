import concurrent.futures

import pandas as pd

from app.services import security_class as sc

_SCENARIOS = None


def _init_worker(scenarios):
    global _SCENARIOS
    _SCENARIOS = scenarios


def _run_one(job):
    sched_idx, scenario_id, intervals, server_registry, timebin, start_dt = job
    task = sc.SimulationTask(
        scenario_id=scenario_id,
        arrival_scenario=_SCENARIOS[scenario_id],
        intervals=intervals,
        server_registry=server_registry,
        timebin=timebin,
        start_dt=start_dt,
    )
    out = sc.single_scenario_worker(task)
    return sched_idx, out["kpis"].to_dict()


def simulate_kpis(
    flight_demand,
    intervals_list,
    server_registry,
    n_scenarios: int,
    n_workers: int,
    timebin: int,
    base_seed: int,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """
    intervals_list : one list of Interval per schedule (e.g. utils.server_schedule_to_intervals(...))
    Returns a DataFrame with one row per schedule, in the same order, with the KPI columns of df_kpis
    (qtime_mean, n_unserved, qtime_above_30_pct, ...).
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
            jobs.append((i, scenario_id, sim.intervals, server_registry, timebin, start))

    if chunksize is None:
        chunksize = max(1, len(jobs) // (n_workers * 8))
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(scenarios,)) as ex:
        results = list(ex.map(_run_one, jobs, chunksize=chunksize))

    per_schedule = [[] for _ in intervals_list]
    for i, kpis in results:
        per_schedule[i].append(kpis)

    rows = []
    for kpis in per_schedule:
        df = pd.DataFrame(kpis)
        numeric = [c for c in df.select_dtypes(include="number").columns if c not in {"scenario_id", "seed"}]
        rows.append(df[numeric].mean())
    return pd.DataFrame(rows).reset_index(drop=True)
