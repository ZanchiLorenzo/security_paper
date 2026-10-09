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

@dataclasses.dataclass(frozen=True)
class ArrivalScenario:
    scenario_id: int
    seed: int
    passengers: list[Passenger]

class ArrivalGenerator:
    """
    Generates passenger arrival times for a given flight demand scenario. 
    The generator uses the `arrival_sampler` attached to each `Flight` to sample 
    arrival times for passengers. 
    """
    def generate(self, flight_demand: list[Flight], seed: int) -> list[Passenger]:
        rng = np.random.default_rng(seed)
        passengers = []

        for flight in flight_demand:
            sampler = flight.arrival_sampler

            if sampler is None:
                raise ValueError(f"Flight {flight.flight_key} has no arrival_sampler attached.")

            for segment, n_pax in flight.pax_segments.items():
                if n_pax <= 0:
                    continue

                minutes_before_departure = sampler.sample(n_pax, rng)

                for i, advance_min in enumerate(minutes_before_departure):
                    arrival_dt = flight.departure_dt - dt.timedelta(minutes=float(advance_min))

                    passengers.append(
                        Passenger(
                            id=f"{flight.flight_key}_{segment}_{i}",
                            flight_key=flight.flight_key,
                            segment=segment,
                            flight_number=flight.flight_number,
                            flight_departure_dt=flight.departure_dt,
                            arrival_dt=arrival_dt,
                        )
                    )

        passengers.sort(key=lambda passenger: passenger.arrival_dt)
        return passengers

    def generate_scenarios(self, flight_demand: list[Flight], n_scenarios: int, base_seed: int) -> dict[int, ArrivalScenario]:
        arrival_scenarios = {}

        for i in range(n_scenarios):
            seed = base_seed + i
            passengers = self.generate(flight_demand=flight_demand, seed=seed)

            arrival_scenarios[i + 1] = ArrivalScenario(
                scenario_id=i + 1,
                seed=seed,
                passengers=passengers,
            )

        return arrival_scenarios

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