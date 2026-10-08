import dataclasses
import datetime as dt
import numpy as np
from enum import Enum

@dataclasses.dataclass(frozen=True)
class ArrivalSampler:
    x: np.ndarray
    cdf: np.ndarray

    def __post_init__(self):
        if self.x.shape != self.cdf.shape or self.x.ndim != 1:
            raise ValueError("x and cdf must be 1-D arrays of the same shape.")
        if self.x.shape[0] < 2:
            raise ValueError("ArrivalSampler needs at least 2 points.")
        if np.any(np.diff(self.x) < 0):
            raise ValueError("x must be non-decreasing.")
        if np.any(np.diff(self.cdf) < 0):
            raise ValueError("cdf must be non-decreasing.")
        if self.cdf[0] < 0 or self.cdf[-1] > 1 or not np.isclose(self.cdf[-1], 1.0, atol=1e-6):
            raise ValueError("cdf must range from 0 to 1.")

    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if size < 0:
            raise ValueError("size must be non-negative")
        rng = rng or np.random.default_rng()
        u = rng.random(size)
        return np.interp(u, self.cdf, self.x)

@dataclasses.dataclass(frozen=True)
class Flight:
    flight_key: str
    flight_number: str
    departure_dt: dt.datetime

    pax_segments: dict[str, int]

    flight_type: str = "standard"

    carrier: str | None = None
    destination: str | None = None
    aircraft: str | None = None

    arrival_sampler: ArrivalSampler | None = dataclasses.field(default=None, compare=False, repr=False)

    def __post_init__(self):
        # departure_dt è naive quando costruito da valori appena assegnati
        # in Python nella stessa sessione/transazione che li scrive (es.
        # scripts/nightly_task.py, che riusa gli oggetti ORM appena creati
        # con expire_on_commit=False - mai un vero round-trip dal db) ma
        # tz-aware quando ScheduleFlight.departure_at viene letto FRESCO dal
        # db in una request/sessione diversa da quella di scrittura (es.
        # /run e /update in app/api/airports.py, che interrogano sempre
        # ScheduleFlight da zero - colonna DateTime(timezone=True), quindi
        # tz-aware una volta letta). Stesso identico problema di
        # Interval.__post_init__ più sotto in questo file, e stessa
        # soluzione: normalizzare qui una volta per tutte, invece che in
        # ogni caller.
        if self.departure_dt.tzinfo is not None:
            object.__setattr__(self, "departure_dt", self.departure_dt.replace(tzinfo=None))

@dataclasses.dataclass(frozen=True)
class Passenger:
    id: str
    flight_key: str
    segment: str

    flight_number: str
    flight_departure_dt: dt.datetime
    arrival_dt: dt.datetime

    processing_time: float | None = None
    flight_departure_min: float | None = None
    arrival_min: float | None = None
    start_min: float | None = None
    end_min: float | None = None
    server_id: str | None = None

@dataclasses.dataclass(frozen=True)
class Interval:
    start: dt.datetime
    end: dt.datetime
    server_labels: tuple[str, ...]

    def __post_init__(self):
        # start/end sono naive ovunque nel dominio (Flight.departure_dt e gli
        # arrival_dt dei passeggeri derivano da orari locali senza tz, vedi
        # preprocessing.py) TRANNE quando un Interval viene ricostruito da
        # SecurityPlanInterval.start_at/end_at (colonne DateTime(timezone=True),
        # quindi tz-aware una volta lette dal db - es. il piano di un giorno
        # locked, riletto da nightly_task.py/airports.py). Confrontare/fare
        # min() tra un Interval così e un arrival_dt naive solleva
        # TypeError - stesso problema già incontrato e risolto con
        # .tz_localize(None) in persistence.py (get_processing_rate_by_label);
        # qui si normalizza una volta per tutte alla creazione, invece di
        # ricordarsi di farlo in ogni punto che costruisce un Interval da db.
        if self.start.tzinfo is not None:
            object.__setattr__(self, "start", self.start.replace(tzinfo=None))
        if self.end.tzinfo is not None:
            object.__setattr__(self, "end", self.end.replace(tzinfo=None))

        if self.end <= self.start:
            raise ValueError("Interval end must be after start.")

        labels = tuple(str(label) for label in self.server_labels)

        if len(set(labels)) != len(labels):
            raise ValueError("server_labels must be unique.")

        object.__setattr__(self, "server_labels", labels)

    @property
    def n_servers(self) -> int:
        return len(self.server_labels)