import pandas as pd
import datetime as dt
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
import numpy as np
from typing import List, Tuple, Any
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.services.entities import Passenger, Flight, ArrivalSampler
import joblib
import math
import os
import dataclasses
import logging
logger = logging.getLogger(__name__)


def sampler_from_json(payload: dict) -> ArrivalSampler:
    return ArrivalSampler(
        x=np.asarray(payload["x"], dtype=float),
        cdf=np.asarray(payload["cdf"], dtype=float),
    )


def build_preprocess_raw_response(flights_by_day: dict) -> dict:
    """security_preprocess.raw_response - stessa forma di utils.flights_to_json
    (arrival_sampler ripetuto su ogni volo), qui raggruppata per giorno.
    Ripetuto per volo anche se oggi build_flight_demand assegna lo stesso
    sampler a ogni volo del run, perché il codice che legge
    (samplers_by_flight_key/samplers_for_flight_keys) deve poter valere anche
    quando smetterà di essere vero, senza un'altra migrazione dei dati."""

    flights_json: dict[str, list[dict]] = {}

    for day, flights in flights_by_day.items():
        day_entries = []

        for flight in flights:
            entry = {
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

            day_entries.append(entry)

        flights_json[day.isoformat() if hasattr(day, "isoformat") else str(day)] = day_entries

    return {"flights": flights_json}


def _flight_sampler_json(flight: dict, legacy_samplers: dict) -> dict | None:
    """`flight`'s arrival sampler payload, embedded (current format) or, for
    rows persisted before build_preprocess_raw_response stopped deduplicating
    it, looked up via the legacy "sampler_key" -> raw_response["samplers"]
    indirection."""

    sampler = flight.get("arrival_sampler")
    if sampler is not None:
        return sampler

    sampler_key = flight.get("sampler_key")
    if sampler_key is not None:
        return legacy_samplers.get(sampler_key)

    return None


def samplers_by_flight_key(raw_response: dict) -> dict[str, ArrivalSampler]:
    """The arrival_sampler recorded for every flight in one
    security_preprocess.raw_response (app.services.samplers.build_preprocess_raw_response),
    keyed by flight_key."""

    raw_response = raw_response or {}
    legacy_samplers = raw_response.get("samplers") or {}

    result: dict[str, ArrivalSampler] = {}
    for day_flights in raw_response.get("flights", {}).values():
        for flight in day_flights:
            flight_key = flight.get("flight_key")
            sampler = _flight_sampler_json(flight, legacy_samplers)
            if flight_key is None or sampler is None:
                continue

            result[flight_key] = sampler_from_json(sampler)

    return result


def first_sampler_from_raw_response(raw_response: dict) -> ArrivalSampler | None:
    """The first arrival_sampler found in a raw_response blob, whichever
    flight/day it comes from - used by load_latest_arrival_sampler (every
    flight in a run shares the same sampler today, so "the first one" is
    "the run's sampler"). `None` if the row has no sampler recorded at all."""

    raw_response = raw_response or {}
    legacy_samplers = raw_response.get("samplers") or {}

    for day_flights in raw_response.get("flights", {}).values():
        for flight in day_flights:
            sampler = _flight_sampler_json(flight, legacy_samplers)
            if sampler is not None:
                return sampler_from_json(sampler)

    return None


def samplers_for_flight_keys(raw_response: dict, flight_keys: set[str]) -> dict[str, ArrivalSampler]:
    """Come samplers_by_flight_key, ma costruisce l'ArrivalSampler (le
    conversioni numpy di x/cdf) SOLO per i flight_key richiesti, invece che
    per ogni volo di ogni giorno coperto dal preprocess run - usata dal
    preview, che tipicamente ha bisogno del sampler di 1-2 voli soltanto, non
    dell'intera giornata/import."""

    raw_response = raw_response or {}
    if not flight_keys:
        return {}
    legacy_samplers = raw_response.get("samplers") or {}

    result: dict[str, ArrivalSampler] = {}
    remaining = set(flight_keys)
    for day_flights in raw_response.get("flights", {}).values():
        if not remaining:
            break
        for flight in day_flights:
            flight_key = flight.get("flight_key")
            if flight_key not in remaining:
                continue

            sampler = _flight_sampler_json(flight, legacy_samplers)
            if sampler is not None:
                result[flight_key] = sampler_from_json(sampler)
            remaining.discard(flight_key)

            if not remaining:
                break

    return result


def _bootstrap_arrival_sampler_from_joblib(db: Session, apt_id: int) -> ArrivalSampler | None:
    from app.db.models import Apt

    apt = db.get(Apt, apt_id)
    if apt is None:
        return None

    sampler_path = os.path.join("data", apt.iata, "arrival_sampler.joblib")
    if not os.path.isfile(sampler_path):
        return None

    legacy_sampler = joblib.load(sampler_path)
    return ArrivalSampler(
        x=np.asarray(legacy_sampler.x, dtype=float),
        cdf=np.asarray(legacy_sampler.cdf, dtype=float),
    )


def load_latest_arrival_sampler(db: Session, apt_id: int) -> ArrivalSampler:
    """Carry-forward input sampler for a new preprocess run: one of the
    samplers embedded in the most recent successful security_preprocess row
    for this airport, read back instead of a file on disk or a static
    per-airport value - which is what allows it to change over time (each run
    is a new row) and, once build_flight_demand assigns samplers per flight
    instead of uniformly, to vary flight by flight without a schema change.

    All flights of a run share one sampler today, so the first flight
    carrying one works; picks whichever comes first. For a brand-new airport
    with no security_preprocess row at all, falls back to
    data/{iata}/arrival_sampler.joblib (_bootstrap_arrival_sampler_from_joblib)."""

    from app.db.models import SecurityPreprocess

    latest = db.scalar(
        select(SecurityPreprocess)
        .where(SecurityPreprocess.apt_id == apt_id, SecurityPreprocess.status == "done")
        # updated_at, non created_at: ogni giorno ha la propria riga
        # (apt_id, service_date), sovrascritta ad ogni run - un giorno
        # preprocessato per la prima volta settimane fa ma rilanciato ogni
        # notte da allora è comunque "il run più recente", anche se il suo
        # created_at è vecchio (vedi il model docstring).
        .order_by(SecurityPreprocess.updated_at.desc())
    )

    if latest is None:
        bootstrapped = _bootstrap_arrival_sampler_from_joblib(db, apt_id)
        if bootstrapped is not None:
            return bootstrapped
        raise ValueError(
            f"No preprocess run recorded yet for apt_id={apt_id}: nothing to carry the arrival sampler forward from."
        )

    sampler = first_sampler_from_raw_response(latest.raw_response)
    if sampler is None:
        raise ValueError(f"Latest preprocess run (id={latest.id}) for apt_id={apt_id} has no arrival_sampler recorded.")
    return sampler

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