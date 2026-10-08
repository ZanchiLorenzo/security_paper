import pandas as pd
import datetime as dt
import ast
from typing import List, Union
from app.services.entities import Flight, Interval, ArrivalSampler
import json
import re
import numpy as np
import dataclasses
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

def parse_schedule_time(time_ser):
    s = time_ser.copy()

    # Se è già datetime, estrai direttamente ora e minuto
    if pd.api.types.is_datetime64_any_dtype(s):
        hour = s.dt.hour
        minute = s.dt.minute
        normalized = (
            hour.astype(str).str.zfill(2)
            + minute.astype(str).str.zfill(2)
        )

        return normalized, hour.astype(int), minute.astype(int)

    # altrimenti normale parsing delle stringhe
    s = s.astype(str).str.strip()

    # eventualmente rimuovi ":" se accetti 06:00
    s = s.str.replace(":", "", regex=False).str.zfill(4)

    hour = pd.to_numeric(s.str[:2], errors="coerce")
    minute = pd.to_numeric(s.str[2:4], errors="coerce")

    invalid = (
        hour.isna()
        | minute.isna()
        | ~hour.between(0, 23)
        | ~minute.between(0, 59)
    )

    if invalid.any():
        raise ValueError(
            f"Invalid scheduled times found: "
            f"{time_ser.loc[invalid].unique().tolist()}"
        )

    return s, hour.astype(int), minute.astype(int)

def assign_profile(row: pd.Series, profile_specs: dict) -> str:
    # as for now it assigns the first occurrence of a matching criteria, if any, otherwise "standard"
    for profile, spec in profile_specs.items():
        field = spec.get("criteria_type")
        values = spec.get("criteria_values", [])

        if field == "schtime":
            hour = row.get("schh")
            minute = row.get("schm")

            if pd.isna(hour) or pd.isna(minute):
                continue

            value = dt.time(hour=int(hour), minute=int(minute))

            for timerange in values:
                start_str, end_str = (timerange.split("-"))

                start = dt.datetime.strptime(start_str, "%H:%M").time()

                end = dt.datetime.strptime(end_str, "%H:%M", ).time()

                if start <= end:
                    matches = start <= value <= end
                else:
                    matches = (
                        value >= start
                        or value <= end
                    )

                if matches:
                    return profile

        else:
            value = row.get(field)

            if pd.isna(value):
                continue

            if value in values:
                return profile

    return "standard"

def build_allocation_summary(grid: np.ndarray, alloc_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "y": np.arange(grid.shape[0]),
        "target": grid.sum(axis=1),
    })

    if alloc_df.empty:
        covered = pd.DataFrame({
            "y": pd.Series(dtype=int),
            "covered": pd.Series(dtype=int),
        })
    else:
        covered = (
            alloc_df
            .groupby("y", as_index=False)
            .agg(covered=("inside_cells", "sum"))
        )

    summary = summary.merge(
        covered,
        on="y",
        how="left",
    )

    summary["covered"] = (
        summary["covered"]
        .fillna(0)
        .astype(int)
    )

    summary["coverage_pct"] = np.where(
        summary["target"] > 0,
        100 * summary["covered"] / summary["target"],
        0.0,
    )

    return summary

def operational_service_date(moment: dt.datetime, cutoff_hour: int = 3) -> dt.date:
    """
    Flights before `cutoff_hour` belong to the previous operational day, not
    their calendar date.
    """
    if moment.hour < cutoff_hour:
        return (moment - dt.timedelta(days=1)).date()

    return moment.date()

# parsing registry

def label_sort_key(label):
    label = str(label).strip().upper()

    m = re.fullmatch(r"(\d+)([A-Z]*)", label)

    if m:
        number = int(m.group(1))
        suffix = m.group(2)
        return (0, number, suffix)

    return (1, label)
    
def sort_labels_by_priority(server_registry: dict[str, dict]) -> list[str]:
    """
    Sort server labels by priority and return the ordered list of labels.
    """
    ordered_servers = sorted(
        server_registry.values(),
        key=lambda server: (
            int(server.get("priority", 10**9)),
            label_sort_key(server["label"]),
        ),
    )

    return [
        str(server["label"])
        for server in ordered_servers
    ]

def get_processing_rate_by_label(server_registry: dict[str, dict], duration: int = 15) -> dict[str, float]:
    """
    Compute the processing rate (pax per 15-minute bin) for each server, keyed by its label.
    """
    if not server_registry:
        raise ValueError("Processing parameters are empty.")

    rate_by_label = {}

    for _, params in server_registry.items():
        dist = params["dist"]
        label = str(params["label"])

        if dist == "gamma":
            mean_processing_time = params.get("shift", 0.0) + params["shape"] * params["scale"]
        elif dist == "gaussian":
            mean_processing_time = params.get("shift", 0.0) + params["mean"]
        elif dist == "exponential":
            mean_processing_time = params.get("shift", 0.0) + params["scale"]
        else:
            raise ValueError(f"Unsupported processing distribution for server {label}: {dist}")

        if mean_processing_time <= 0:
            raise ValueError(f"Mean processing time for server {label} must be positive.")

        rate_by_label[label] = duration / mean_processing_time

    return rate_by_label, duration

# parsing server schedule and intervals

def server_schedule_to_json(server_schedule: dict[str, list[tuple[dt.datetime, dt.datetime]]]) -> dict[str, list[dict]]:
    """
    Serialize a server_schedule to JSON-compatible dicts.
    """

    results = {
        label: [
            {"start": start.isoformat(), "end": end.isoformat()}
            for start, end in periods
        ]
        for label, periods in server_schedule.items()
    }
    return results

def server_schedule_from_json(data: dict[str, list[dict]]) -> dict[str, list[tuple[dt.datetime, dt.datetime]]]:
    """
    Deserialize a server_schedule from its JSON-compatible dict form.
    """
    return {
        label: [
            (dt.datetime.fromisoformat(period["start"]), dt.datetime.fromisoformat(period["end"]))
            for period in periods
        ]
        for label, periods in data.items()
    }

def validate_intervals(intervals: list[Interval], require_contiguous: bool = True) -> list[Interval]:
    
    if not intervals:
        raise ValueError("Intervals cannot be empty.")

    ordered = sorted(intervals, key=lambda interval: interval.start)

    for previous, current in zip(
        ordered,
        ordered[1:],
    ):
        if current.start < previous.end:
            raise ValueError(
                "Overlapping intervals detected: "
                f"{previous.start.isoformat()}-"
                f"{previous.end.isoformat()} and "
                f"{current.start.isoformat()}-"
                f"{current.end.isoformat()}."
            )

        if (require_contiguous and current.start > previous.end):
            raise ValueError(
                "Gap between intervals detected: "
                f"{previous.end.isoformat()} to "
                f"{current.start.isoformat()}."
            )

    return ordered

def server_schedule_to_intervals(server_schedule: dict[str, list[tuple]]) -> list[Interval]:
    """
    Convert a server schedule dictionary to a list of Interval objects.
    Each key in the server_schedule represents a server label, and its value is a list of tuples representing the start and end times of the server's availability.
    """
    if not server_schedule or not any(server_schedule.values()):
        raise ValueError("Server schedule cannot be empty.")

    events_by_time: dict = {}

    for label, periods in server_schedule.items():
        for start, end in periods:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            events_by_time.setdefault(start, []).append((1, label))
            events_by_time.setdefault(end, []).append((-1, label))

    breakpoints = sorted(events_by_time.keys())

    active: set = set()
    raw_intervals = []

    for seg_start, seg_end in zip(breakpoints[:-1], breakpoints[1:]):
        for sign, label in events_by_time.get(seg_start, []):
            active.add(label) if sign == 1 else active.discard(label)

        active_labels = tuple(sorted(active, key=label_sort_key))
        raw_intervals.append(
            Interval(start=seg_start.to_pydatetime(), end=seg_end.to_pydatetime(), server_labels=active_labels)
        )

    intervals = []
    for interval in raw_intervals:
        if (
            intervals
            and intervals[-1].server_labels == interval.server_labels
            and intervals[-1].end == interval.start
        ):
            intervals[-1] = dataclasses.replace(intervals[-1], end=interval.end)
        else:
            intervals.append(interval)

    return validate_intervals(intervals, require_contiguous=True)

def server_schedule_to_raw_intervals(server_schedule: dict[str, list[tuple]]) -> list[Interval]:
    """
    Like server_schedule_to_intervals, but without aggregating across labels:
    one Interval per (label, period) pair, exactly as it appears in
    server_schedule - no breakpoint splitting, no cross-label merging.

    The result is NOT a partition of the day: periods from different labels
    can (and normally do) overlap, and each Interval carries a single-label
    server_labels tuple. validate_intervals would reject that, so this is not
    suitable input for SecuritySimulator (which needs a contiguous,
    non-overlapping timeline) - use server_schedule_to_intervals for that.
    """
    return [
        Interval(start=start, end=end, server_labels=(label,))
        for label, periods in (server_schedule or {}).items()
        for start, end in periods
    ]

def intervals_to_server_schedule(intervals: list[Interval]) -> dict[str, list[tuple]]:
    """
    Inverse of server_schedule_to_intervals: collapses a merged timeline of
    Interval (each carrying the set of labels active in it) back into a
    per-label schedule, merging the contiguous/overlapping periods a label
    picks up across consecutive intervals into as few periods as possible.
    """
    periods_by_label: dict[str, list[tuple]] = {}

    for interval in intervals or []:
        for label in interval.server_labels:
            periods_by_label.setdefault(label, []).append((interval.start, interval.end))

    server_schedule: dict[str, list[tuple]] = {}

    for label, periods in periods_by_label.items():
        periods = sorted(periods, key=lambda period: period[0])

        merged = []
        for start, end in periods:
            if merged and start <= merged[-1][1]:
                if end > merged[-1][1]:
                    merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        server_schedule[label] = merged

    return server_schedule

# parsin flights

def flights_to_json(flights: list[Flight]) -> list[dict]:
    """
    Serialize Flight objects to JSON-compatible dicts.
    """
    return [
        {
            "flight_key": flight.flight_key,
            "flight_number": flight.flight_number,
            "departure_dt": flight.departure_dt.isoformat(),
            "pax_segments": flight.pax_segments,
            "flight_type": flight.flight_type,
            "carrier": flight.carrier,
            "destination": flight.destination,
            "aircraft": flight.aircraft,
            "arrival_sampler": (
                {
                    "x": flight.arrival_sampler.x.tolist(),
                    "cdf": flight.arrival_sampler.cdf.tolist(),
                }
                if flight.arrival_sampler is not None
                else None
            ),
        }
        for flight in flights
    ]

def flights_from_json(flights: list[dict]) -> list[Flight]:
    """
    Deserialize Flight objects from JSON-compatible dicts.
    """
    return [
        Flight(
            flight_key=flight["flight_key"],
            flight_number=flight["flight_number"],
            departure_dt=pd.Timestamp(flight["departure_dt"]).to_pydatetime(),
            pax_segments={
                str(segment): int(value)
                for segment, value
                in flight["pax_segments"].items()
            },
            flight_type=flight.get("flight_type", "standard"),
            carrier=flight.get("carrier"),
            destination=flight.get("destination"),
            aircraft=flight.get("aircraft"),
            arrival_sampler=(
                ArrivalSampler(
                    x=np.asarray(flight["arrival_sampler"]["x"], dtype=float),
                    cdf=np.asarray(flight["arrival_sampler"]["cdf"], dtype=float),
                )
                if flight.get("arrival_sampler") is not None
                else None
            ),
        )
        for flight in flights
    ]

# Results post-processing

def df_to_json_records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()

    # datetime/date -> stringa ISO
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # NaN, NaT, inf -> None. astype(object) first: on a numeric column,
    # DataFrame.where can't hold None without it (the column keeps its float
    # dtype and None silently reverts back to NaN), and psycopg's JSONB codec
    # then writes the literal token `NaN`, which Postgres rejects as invalid
    # JSON.
    df = df.replace([pd.NaT, float("inf"), float("-inf")], None)
    df = df.astype(object).where(pd.notnull(df), None)

    return df.to_dict(orient="records")