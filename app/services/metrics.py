import pandas as pd
import numpy as np
import datetime as dt
from typing import Tuple
from app.services.entities import Flight, Passenger, Interval
from app.services.utils import get_processing_rate_by_label
import dataclasses
from collections import defaultdict

@dataclasses.dataclass
class SimulationKPIs:
    qtime_p5: float
    qtime_p25: float
    qtime_p50: float
    qtime_p75: float
    qtime_p95: float
    qtime_mean: float
    n_arrived: int
    n_served: int
    n_unserved: int
    penalized_system_pct: float
    qtime_above_15_pct: float
    qtime_above_30_pct: float
    qtime_above_60_pct: float
    queue_length_p5: float
    queue_length_p25: float
    queue_length_p50: float
    queue_length_p75: float
    queue_length_p95: float
    avg_utilization_pct: float
    full_utilization_pct: float
    scenario_id: int = 0
    seed: int = 0
    id: object = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

def _empty_flow_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "minute",
            "arrival_count",
            "arrival_rate",
            "complete_count",
            "processing_rate",
        ]
    )

def _compute_flow_metrics(pax_data: list, bin_minutes: float) -> pd.DataFrame:
    if not pax_data:
        return _empty_flow_df()

    arrival_min = np.array([p.arrival_min for p in pax_data], dtype=float)

    end_min = np.array(
        [p.end_min if p.end_min is not None else np.nan for p in pax_data],
        dtype=float,
    )

    arrival_bin = np.floor(arrival_min / bin_minutes) * bin_minutes

    df_arrivals = (
        pd.DataFrame({"minute": arrival_bin})
        .groupby("minute")
        .size()
        .reset_index(name="arrival_count")
    )

    valid_end = ~np.isnan(end_min)

    if valid_end.any():
        complete_bin = np.floor(end_min[valid_end] / bin_minutes) * bin_minutes

        df_completed = (
            pd.DataFrame({"minute": complete_bin})
            .groupby("minute")
            .size()
            .reset_index(name="complete_count")
        )
    else:
        df_completed = pd.DataFrame(columns=["minute", "complete_count"])

    out = df_arrivals.merge(df_completed, on="minute", how="outer")

    out[["arrival_count", "complete_count"]] = out[
        ["arrival_count", "complete_count"]
    ].fillna(0)

    out["arrival_rate"] = out["arrival_count"]
    out["processing_rate"] = out["complete_count"]

    return out

def _interval_to_min(interval, start_dt: dt.datetime) -> tuple[float, float]:
    start_min = (interval.start - start_dt).total_seconds() / 60
    end_min = (interval.end - start_dt).total_seconds() / 60
    return start_min, end_min

def _compute_capacity(df_bins: pd.DataFrame, intervals: list, start_dt: dt.datetime,) -> pd.DataFrame:
    out = df_bins[["minute"]].copy()
    out["capacity"] = 0.0

    for iv in intervals:
        start_min, end_min = _interval_to_min(iv, start_dt)

        mask = (out["minute"] >= start_min) & (out["minute"] < end_min)
        out.loc[mask, "capacity"] = iv.n_servers

    return out

def _compute_service_rate(df_bins: pd.DataFrame, intervals: list, start_dt: dt.datetime, rate_by_label: dict, duration: float) -> pd.DataFrame:
    """
    Per-bin aggregate service rate (pax/minute), built from the specific server labels actually scheduled in that bin
    """
    out = df_bins[["minute"]].copy()
    out["service_rate"] = 0.0

    rate_per_minute = {
        label: rate / duration
        for label, rate in rate_by_label.items()
    }

    for iv in intervals:
        start_min, end_min = _interval_to_min(iv, start_dt)

        mask = (out["minute"] >= start_min) & (out["minute"] < end_min)

        labels = [str(label) for label in iv.server_labels]

        missing = [label for label in labels if label not in rate_per_minute]
        if missing:
            raise ValueError(
                f"Missing processing rate for server labels: {sorted(set(missing))}"
            )

        rate = sum(rate_per_minute[label] for label in labels)
        out.loc[mask, "service_rate"] = rate

    return out

def _time_weighted_cumulative(events: list, state_col: str, query_times: np.ndarray) -> np.ndarray:
    """
    Cumulative integral of a piecewise constant function defined by events.
    """
    df_events = (
        pd.DataFrame(events)[["sim_minute", state_col]]
        .sort_values("sim_minute", kind="stable")
        .groupby("sim_minute", as_index=False)
        .last()
    )

    t = df_events["sim_minute"].to_numpy(dtype=float)
    v = df_events[state_col].to_numpy(dtype=float)

    segment_areas = v[:-1] * np.diff(t)
    cum_area_at_breakpoint = np.concatenate([[0.0], np.cumsum(segment_areas)])

    idx = np.searchsorted(t, query_times, side="right") - 1
    safe_idx = np.clip(idx, 0, len(t) - 1)

    cum_integral = cum_area_at_breakpoint[safe_idx] + v[safe_idx] * (query_times - t[safe_idx])

    return np.where(idx < 0, 0.0, cum_integral)

def _compute_state(events: list, df_bins: pd.DataFrame, state_col: str, output_col: str) -> pd.DataFrame:
    """
    Compute the weighted average of a piecewise constant state variable over time bins.
    """
    out = df_bins[["minute", "minute_end"]].copy()

    if not events:
        out[output_col] = 0.0
        return out[["minute", output_col]]

    starts = out["minute"].to_numpy(dtype=float)
    ends = out["minute_end"].to_numpy(dtype=float)

    cum_start = _time_weighted_cumulative(events, state_col, starts)
    cum_end = _time_weighted_cumulative(events, state_col, ends)

    bin_duration = ends - starts

    out[output_col] = np.where(
        bin_duration > 0,
        (cum_end - cum_start) / bin_duration,
        0.0,
    )

    return out[["minute", output_col]]

def get_df_timeline(result, timebin, server_registry: dict) -> pd.DataFrame:
    pax_data = result.pax_data
    intervals = result.intervals
    start_dt = result.start_dt

    bin_minutes = float(timebin)

    candidate_end_minutes = []

    candidate_end_minutes.extend(x["sim_minute"] for x in result.queue_var)
    candidate_end_minutes.extend(x["sim_minute"] for x in result.server_var)
    candidate_end_minutes.extend(x["sim_minute"] for x in result.complete_pax_var)

    candidate_end_minutes.extend(
        p.arrival_min
        for p in pax_data
        if p.arrival_min is not None
    )

    candidate_end_minutes.extend(
        p.end_min
        for p in pax_data
        if p.end_min is not None
    )

    candidate_end_minutes.extend(
        _interval_to_min(iv, start_dt)[1]
        for iv in intervals
    )

    if candidate_end_minutes:
        end_min = np.ceil(max(candidate_end_minutes) / bin_minutes) * bin_minutes
        end_min += bin_minutes
    else:
        end_min = bin_minutes

    minutes = np.arange(0, end_min, bin_minutes)

    df = pd.DataFrame({"minute": minutes})
    df["minute_end"] = df["minute"] + bin_minutes

    df_flow = _compute_flow_metrics(pax_data, bin_minutes)
    df_capacity = _compute_capacity(df, intervals, start_dt)

    rate_by_label, duration = get_processing_rate_by_label(server_registry)
    df_rate = _compute_service_rate(df, intervals, start_dt, rate_by_label, duration=duration)

    df_queue = _compute_state(
        result.queue_var,
        df,
        state_col="queue_length",
        output_col="queue_length",
    )

    df_servers = _compute_state(
        result.server_var,
        df,
        state_col="occupied_servers",
        output_col="occupied_servers",
    )

    df = (
        df
        .merge(df_flow, on="minute", how="left")
        .merge(df_capacity, on="minute", how="left")
        .merge(df_rate, on="minute", how="left")
        .merge(df_queue, on="minute", how="left")
        .merge(df_servers, on="minute", how="left")
    )

    fill_cols = [
        "arrival_count",
        "arrival_rate",
        "complete_count",
        "processing_rate",
        "queue_length",
        "occupied_servers",
        "capacity",
    ]

    df[fill_cols] = df[fill_cols].fillna(0)

    df["theoretical_processing_rate"] = bin_minutes * df["service_rate"]

    df["expected_qtime_min"] = np.where(
        df["service_rate"] > 0,
        df["queue_length"] / df["service_rate"],
        np.nan,
    )

    df["utilization_avg_pct"] = np.where(
        df["capacity"] > 0,
        (df["occupied_servers"] / df["capacity"]).clip(upper=1) * 100,
        0,
    )

    df["time_bin"] = [
        start_dt + dt.timedelta(minutes=float(m))
        for m in df["minute"]
    ]

    return df[
        [
            "time_bin",
            "minute",
            "arrival_count",
            "arrival_rate",
            "complete_count",
            "processing_rate",
            "queue_length",
            "occupied_servers",
            "capacity",
            "service_rate",
            "theoretical_processing_rate",
            "expected_qtime_min",
            "utilization_avg_pct",
        ]
    ]

def compute_full_utilization_pct(
    result,
    observation_end_min: float,
    ) -> float:

    events = sorted(
        result.state_events,
        key=lambda x: x["sim_minute"],
    )

    if not events:
        return 0.0

    total_time = 0.0
    full_time = 0.0

    extended_events = events + [{
        "sim_minute": observation_end_min,
        "capacity": events[-1]["capacity"],
        "occupied_servers": (
            events[-1]["occupied_servers"]
        ),
    }]

    for current, nxt in zip(
        extended_events,
        extended_events[1:],
    ):
        duration = (
            nxt["sim_minute"]
            - current["sim_minute"]
        )

        if duration <= 0:
            continue

        capacity = current["capacity"]
        occupied = current["occupied_servers"]

        if capacity > 0:
            total_time += duration

            if occupied >= capacity:
                full_time += duration

    return (
        100 * full_time / total_time
        if total_time > 0
        else 0.0
    )

def _empty_kpis() -> SimulationKPIs:
    return SimulationKPIs(
        qtime_p5=0.0,
        qtime_p25=0.0,
        qtime_p50=0.0,
        qtime_p75=0.0,
        qtime_p95=0.0,
        qtime_mean=0.0,
        n_arrived=0,
        n_served=0,
        n_unserved=0,
        penalized_system_pct=0.0,
        qtime_above_15_pct=0.0,
        qtime_above_30_pct=0.0,
        qtime_above_60_pct=0.0,
        queue_length_p5=0.0,
        queue_length_p25=0.0,
        queue_length_p50=0.0,
        queue_length_p75=0.0,
        queue_length_p95=0.0,
        avg_utilization_pct=0.0,
        full_utilization_pct=0.0,
    )

def get_kpis(result, df_timeline: pd.DataFrame) -> SimulationKPIs:
    pax_data = result.pax_data

    if not pax_data:
        return _empty_kpis()

    processed_pax = [
        p
        for p in pax_data
        if p.start_min is not None
        and p.end_min is not None
    ]

    unserved_pax = [
        p
        for p in pax_data
        if p.start_min is None
        or p.end_min is None
    ]

    n_arrived = len(pax_data)
    n_served = len(processed_pax)
    n_unserved = len(unserved_pax)

    # pax waiting more than the threshold (minutes); a pax that never entered service has an unbounded wait, so it counts as waiting too long
    waits = np.array([p.start_min - p.arrival_min for p in pax_data if p.start_min is not None], dtype=float)
    n_never_started = n_arrived - waits.size
    qtime_above_15_pct = 100 * (int(np.sum(waits > 15)) + n_never_started) / n_arrived
    qtime_above_30_pct = 100 * (int(np.sum(waits > 30)) + n_never_started) / n_arrived
    qtime_above_60_pct = 100 * (int(np.sum(waits > 60)) + n_never_started) / n_arrived


    if not processed_pax:
        kpis = _empty_kpis()

        return dataclasses.replace(
            kpis,
            n_arrived=n_arrived,
            n_served=0,
            n_unserved=n_unserved,
            penalized_system_pct=100.0 if n_arrived > 0 else 0.0,
        )

    arrivals = np.array([p.arrival_min for p in processed_pax], dtype=float)
    starts = np.array([p.start_min for p in processed_pax], dtype=float)
    ends = np.array([p.end_min for p in processed_pax], dtype=float)

    queue_times = starts - arrivals
    qtime_p5, qtime_p25, qtime_p50, qtime_p75, qtime_p95 = np.percentile(
        queue_times,
        [5, 25, 50, 75, 95],
    )
    queue_lengths = df_timeline["queue_length"].to_numpy(dtype=float)

    (
        queue_length_p5,
        queue_length_p25,
        queue_length_p50,
        queue_length_p75,
        queue_length_p95,
    ) = np.percentile(queue_lengths, [5, 25, 50, 75, 95])

    df_util = df_timeline[df_timeline["capacity"] > 0]

    if df_util.empty:
        avg_utilization_pct = 0.0
        full_utilization_pct = 0.0
    else:
        avg_utilization_pct = df_util["utilization_avg_pct"].mean()
        observation_end_min = max(
            (
                interval.end - result.start_dt
            ).total_seconds() / 60
            for interval in result.intervals
        )
        full_utilization_pct = compute_full_utilization_pct(result, observation_end_min)

    last_interval_end_min = max(
        _interval_to_min(iv, result.start_dt)[1]
        for iv in result.intervals
    )

    n_penalized_served = int(np.sum(ends > last_interval_end_min))

    n_penalized = (n_unserved + n_penalized_served)

    penalized_system_pct = (
        100 * n_penalized / n_arrived
        if n_arrived > 0
        else 0.0
    )

    return SimulationKPIs(
        qtime_p5=qtime_p5,
        qtime_p25=qtime_p25,
        qtime_p50=qtime_p50,
        qtime_p75=qtime_p75,
        qtime_p95=qtime_p95,
        qtime_mean=queue_times.mean(),
        n_arrived=n_arrived,
        n_served=n_served,
        n_unserved=n_unserved,
        penalized_system_pct=penalized_system_pct,
        qtime_above_15_pct=qtime_above_15_pct,
        qtime_above_30_pct=qtime_above_30_pct,
        qtime_above_60_pct=qtime_above_60_pct,
        queue_length_p5=queue_length_p5,
        queue_length_p25=queue_length_p25,
        queue_length_p50=queue_length_p50,
        queue_length_p75=queue_length_p75,
        queue_length_p95=queue_length_p95,
        avg_utilization_pct=avg_utilization_pct,
        full_utilization_pct=full_utilization_pct,
    )

def get_df_pax(pax_data: list, simulation_start_dt: dt.datetime) -> pd.DataFrame:

    if not pax_data:
        return pd.DataFrame()

    df = pd.DataFrame({
        "id": [p.id for p in pax_data],
        "segment": [p.segment for p in pax_data],
        "flight_number": [p.flight_number for p in pax_data],
        "flight_departure_dt": [p.flight_departure_dt for p in pax_data],
        "arrival_dt": [p.arrival_dt for p in pax_data],
        "processing_time": [p.processing_time for p in pax_data],
        "server_id": [p.server_id for p in pax_data],
        "arrival_min": [p.arrival_min for p in pax_data],
        "start_min": [p.start_min for p in pax_data],
        "end_min": [p.end_min for p in pax_data],
    })

    for col in ["start_min", "end_min"]:
        df[col.replace("_min", "_dt")] = df[col].apply(
            lambda x: simulation_start_dt + dt.timedelta(minutes=x)
            if pd.notna(x) else pd.NaT
        )

    df["queue_time"] = df["start_min"] - df["arrival_min"]
    df["time_to_departure"] = (
        df["flight_departure_dt"] - df["end_dt"]
    ).dt.total_seconds() / 60

    df["served"] = df["end_dt"].notna()

    return df.drop(columns=["arrival_min", "start_min", "end_min"])

def build_empty_simulation_result(seed, intervals, timebin) -> dict:
    # timebin arriva come intero in minuti (single_scenario_worker passa
    # task.timebin invariato): serve l'unità "min" perché pandas accetti la
    # stringa come frequenza (bare "5" non è una freq valida).
    freq = f"{timebin}min"

    start = pd.Timestamp(min(iv.start for iv in intervals)).floor(freq)
    end = pd.Timestamp(max(iv.end for iv in intervals)).ceil(freq)

    bins = pd.date_range(start=start, end=end, freq=freq)

    df_timeline = pd.DataFrame({"time_bin": bins[:-1]})
    bin_minutes = float(timebin)
    df_timeline["minute"] = np.arange(len(df_timeline)) * bin_minutes

    cols_zero = [
        "arrival_count",
        "arrival_rate",
        "complete_count",
        "processing_rate",
        "queue_length",
        "occupied_servers",
        "capacity",
        "theoretical_processing_rate",
        "utilization_avg_pct",
    ]

    for col in cols_zero:
        df_timeline[col] = 0.0

    for iv in intervals:
        mask = (
            (df_timeline["time_bin"] >= iv.start) &
            (df_timeline["time_bin"] < iv.end)
        )
        df_timeline.loc[mask, "capacity"] = iv.n_servers

    df_timeline["seed"] = seed

    kpis = _empty_kpis()
    kpis = dataclasses.replace(
        kpis,
        seed=seed,
    )

    return {
        "df_timeline": df_timeline,
        "kpis": kpis,
        "df_pax": pd.DataFrame(),
        "df_queue_cdf": pd.DataFrame(),
    }

def get_queue_cdf(df_pax: pd.DataFrame, bin_width: float = 1.0) -> pd.DataFrame:
    """
    Compute the empirical cumulative distribution (CDF) of queue times,
    binned at bin_width.
    """
    empty = pd.DataFrame(columns=[
        "queue_time_bin_start", "queue_time_bin_end", "cum_count", "cum_pct",
    ])

    if df_pax.empty or "queue_time" not in df_pax.columns:
        return empty

    queue_times = df_pax["queue_time"].dropna().to_numpy(dtype=float)

    if queue_times.size == 0:
        return empty

    max_time = queue_times.max()
    bin_edges = np.arange(0, max_time + bin_width, bin_width)

    if bin_edges[-1] <= max_time:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)

    counts, edges = np.histogram(queue_times, bins=bin_edges)
    cum_count = np.cumsum(counts)

    return pd.DataFrame({
        "queue_time_bin_start": edges[:-1],
        "queue_time_bin_end": edges[1:],
        "cum_count": cum_count,
        "cum_pct": cum_count / cum_count[-1] * 100,
    })