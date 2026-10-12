import pandas as pd
import datetime as dt
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
import numpy as np
from typing import List, Tuple, Any
from app.services.entities import Passenger, Flight, ArrivalSampler
import joblib
import math
import os
import dataclasses
import logging
logger = logging.getLogger(__name__)

_ARRIVAL_STREAM = 0

# time grid shared by every per-bin arrivals series
def _bin_index(start, end, freq: str) -> pd.DatetimeIndex:
    """Bins of length freq covering [start, end]: first bin starts at floor(start), last bin contains end."""
    first = pd.Timestamp(start).floor(freq)
    stop = pd.Timestamp(end).floor(freq) + pd.Timedelta(freq)
    return pd.date_range(first, stop, freq=freq, inclusive="left")

def arrival_window(flight_demand: list[Flight]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Earliest and latest instant a passenger of these flights can arrive, from the support of each
    sampler (x = minutes before departure): [departure - x_max, departure - x_min].
    """
    _check_samplers(flight_demand)
    first = min(pd.Timestamp(f.departure_dt) - pd.Timedelta(minutes=float(f.arrival_sampler.x[-1])) for f in flight_demand)
    last = max(pd.Timestamp(f.departure_dt) - pd.Timedelta(minutes=float(f.arrival_sampler.x[0])) for f in flight_demand)
    return first, last

def arrivals_index(flight_demand: list[Flight], freq: str = "15min", start=None, end=None) -> pd.DatetimeIndex:
    """
    Index of the arrivals series. By default it covers the whole arrival window of the flights;
    pass start/end (e.g. one operational day, end excluded) to restrict it.
    """
    first, last = arrival_window(flight_demand)
    if end is None:
        return _bin_index(first if start is None else start, last, freq)
    # explicit end is an exclusive upper bound: [start, end)
    return pd.date_range(pd.Timestamp(first if start is None else start).floor(freq), pd.Timestamp(end).ceil(freq), freq=freq, inclusive="left")

def _check_samplers(flight_demand: list[Flight]) -> None:
    if not flight_demand:
        raise ValueError("flight_demand is empty.")
    missing = [f.flight_key for f in flight_demand if f.arrival_sampler is None]
    if missing:
        raise ValueError(f"Flights missing arrival_sampler: {missing[:5]}")

# sampling of passengers
def sample_passengers(flight_demand: list[Flight], rng: np.random.Generator) -> list[Passenger]:
    """One random draw of every passenger of every flight, sorted by arrival time."""
    _check_samplers(flight_demand)
    passengers: list[Passenger] = []
    for flight in flight_demand:
        for segment, n_pax in flight.pax_segments.items():
            if n_pax <= 0:
                continue
            minutes_before = flight.arrival_sampler.sample(int(n_pax), rng)
            for i, advance_min in enumerate(minutes_before):
                passengers.append(
                    Passenger(
                        id=f"{flight.flight_key}_{segment}_{i}",
                        flight_key=flight.flight_key,
                        segment=segment,
                        flight_number=flight.flight_number,
                        flight_departure_dt=flight.departure_dt,
                        arrival_dt=flight.departure_dt - dt.timedelta(minutes=float(advance_min)),
                    )
                )
    passengers.sort(key=lambda p: p.arrival_dt)
    return passengers

@dataclasses.dataclass(frozen=True)
class ArrivalScenario:
    scenario_id: int
    seed: int
    passengers: list[Passenger]

class ArrivalGenerator:
    """Draws passenger arrival times from the arrival_sampler attached to each Flight."""

    def generate(self, flight_demand: list[Flight], seed: int) -> list[Passenger]:
        rng = np.random.default_rng([int(seed), _ARRIVAL_STREAM])
        return sample_passengers(flight_demand, rng)

    def generate_scenarios(self, flight_demand: list[Flight], n_scenarios: int, base_seed: int) -> dict[int, ArrivalScenario]:
        scenarios = {}
        for i in range(n_scenarios):
            seed = base_seed + i
            scenarios[i + 1] = ArrivalScenario(
                scenario_id=i + 1,
                seed=seed,
                passengers=self.generate(flight_demand, seed),
            )
        return scenarios

# expected arrivals per bin (for coverage planning)
def expected_arrivals(flight_demand: list[Flight], freq: str = "15min", start=None, end=None) -> pd.Series:
    """
    Expected pax per bin, no sampling noise.
    E[pax in bin] = sum_flights n_pax * (G(right) - G(left)),  G(t) = P(arrival <= t) = 1 - CDF(minutes before departure of t).
    Over the full arrival window the total is exactly the number of passengers; with start/end the mass that falls outside is lost (and logged).
    """
    index = arrivals_index(flight_demand, freq, start, end)
    edges = index.append(pd.DatetimeIndex([index[-1] + (index[1] - index[0])]))
    out = np.zeros(len(index))
    total = 0.0
    for f in flight_demand:
        n = sum(v for v in f.pax_segments.values() if v > 0)
        if n == 0:
            continue
        total += n
        mins_before = np.asarray((pd.Timestamp(f.departure_dt) - edges) / pd.Timedelta(minutes=1), dtype=float)
        G = 1.0 - np.interp(mins_before, f.arrival_sampler.x, f.arrival_sampler.cdf)
        out += n * np.diff(G)
    lost = total - out.sum()
    if lost > 1e-6 * max(total, 1.0):
        logger.warning("expected_arrivals: %.1f of %.0f passengers fall outside [%s, %s)", lost, total, edges[0], edges[-1])
    return pd.Series(out, index=index, name="arrivals")

# legacy functions for fitting KDEs
def fit_kde(empirical_arrivals: pd.Series, kde_specs: dict[str, Any]) -> Tuple[gaussian_kde, float, float]:

    required_keys = {
        "min_minutes",
        "max_minutes",
    }

    missing_keys = required_keys - kde_specs.keys()
    if missing_keys:
        raise KeyError(f"Missing KDE specifications: {sorted(missing_keys)}")

    x_min = float(kde_specs["min_minutes"])
    x_max = float(kde_specs["max_minutes"])
    shift = float(kde_specs.get("shift", 0))
    bw_method = kde_specs.get("bw_method", "scott")

    if x_min >= x_max:
        raise ValueError(f"min_minutes must be lower than max_minutes")

    arrivals = pd.to_numeric(
        pd.Series(empirical_arrivals),
        errors="coerce",
    ).dropna()

    arrivals = arrivals + shift
    arrivals = arrivals[arrivals.between(x_min, x_max, inclusive="both")]

    if arrivals.empty or len(arrivals) < 2 or arrivals.nunique() < 2:
        raise ValueError("No/too few arrival observations remain after filtering")

    kde = gaussian_kde(arrivals.to_numpy(dtype=float), bw_method=bw_method,)

    return kde, x_min, x_max

def kde_to_sampler(kde: gaussian_kde, x_min: float, x_max: float) -> ArrivalSampler:

    if x_min >= x_max:
        raise ValueError("x_min must be lower than x_max")

    x_min_int = math.floor(x_min)
    x_max_int = math.ceil(x_max)

    if x_max_int - x_min_int < 1:
        raise ValueError("x_min/x_max span must cover at least 1 minute.")

    x = np.arange(x_min_int, x_max_int + 1, 1.0, dtype=float)

    pdf = np.asarray(kde.evaluate(x), dtype=float)
    pdf = np.clip(pdf, 0.0, None)

    cdf = cumulative_trapezoid(pdf, x, initial=0.0)

    total_mass = cdf[-1]

    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("The KDE has no finite positive probability mass inside the requested interval.")

    cdf /= total_mass
    cdf = np.maximum.accumulate(cdf)
    cdf[0] = 0.0
    cdf[-1] = 1.0

    return ArrivalSampler(x=x, cdf=cdf)