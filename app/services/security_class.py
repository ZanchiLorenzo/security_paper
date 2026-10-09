from app.services.entities import  Interval, Flight, Passenger
from app.services.optimizer import LanesOptimizer
from app.services.simulation import QueueSimulationSinglequeue
from app.services.samplers import ArrivalScenario, ArrivalGenerator
from app.services.utils import validate_intervals
from app.services.coverage_dp import CoverageReport, coverage_report, solve_coverage_batch, allocate_budget, solve_allocated
import app.services.metrics as metrics
import concurrent.futures
import datetime as dt
import re
import math
import pandas as pd
import numpy as np
import dataclasses
import logging
logger = logging.getLogger(__name__)

# server registry helpers
def _label_of(key: str, params: dict) -> str:
    return str(params.get("label", key))

def _mean_processing_time(params: dict, label: str) -> float:
    """Mean processing time of one passenger, in minutes."""
    dist = params["dist"]
    shift = params.get("shift", 0.0)
    if dist == "gamma":
        t = shift + params["shape"] * params["scale"]
    elif dist == "gaussian":
        t = shift + params["mean"]
    elif dist == "exponential":
        t = shift + params["scale"]
    else:
        raise ValueError(f"Unsupported processing distribution for server {label}: {dist}")
    if t <= 0:
        raise ValueError(f"Mean processing time for server {label} must be positive.")
    return float(t)

def mean_processing_rate(server_registry: dict[str, dict]) -> float:
    """Mean over the servers of their processing rate, in pax per hour."""
    if not server_registry:
        raise ValueError("server_registry is empty.")
    rates = [60.0 / _mean_processing_time(p, _label_of(k, p)) for k, p in server_registry.items()]
    return float(np.mean(rates))

def get_processing_rate_by_label(server_registry: dict[str, dict], bin_minutes: float = 15.0) -> dict[str, float]:
    """Processing rate of each server, in pax per bin of bin_minutes, keyed by label."""
    if not server_registry:
        raise ValueError("server_registry is empty.")
    return {
        _label_of(k, p): bin_minutes / _mean_processing_time(p, _label_of(k, p))
        for k, p in server_registry.items()
    }

def label_sort_key(label):
    label = str(label).strip().upper()
    m = re.fullmatch(r"(\d+)([A-Z]*)", label)
    if m:
        return (0, int(m.group(1)), m.group(2))
    return (1, label)

def sort_labels_by_priority(server_registry: dict[str, dict]) -> list[str]:
    """Server labels ordered by priority (then by label); the first label is opened first."""
    items = sorted(
        server_registry.items(),
        key=lambda kv: (int(kv[1].get("priority", 10**9)), label_sort_key(_label_of(*kv))),
    )
    labels = [_label_of(k, p) for k, p in items]
    if len(set(labels)) != len(labels):
        raise ValueError("server labels in the registry must be unique.")
    return labels

# sampling arrivals

def _bin_index(start, end, freq: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).floor(freq)
    end = pd.Timestamp(end).ceil(freq)
    return pd.date_range(start, end, freq=freq, inclusive="left")
 
def passengers_to_arrivals(passengers, freq: str = "15min") -> pd.Series:
    """
    passengers : list of Passenger (needs .arrival_dt)
    freq       : frequency of the output bins
    """
    if not passengers:
        raise ValueError("no passengers")
    t = pd.DatetimeIndex([p.arrival_dt for p in passengers])
    start = t.min()
    end = t.max() + pd.Timedelta(freq)
    idx = _bin_index(start, end, freq)
    pos = idx.searchsorted(t, side="right") - 1          # bin containing each arrival
    ok = (pos >= 0) & (t < idx[-1] + pd.Timedelta(freq))
    if not ok.all():
        import logging
        logging.getLogger(__name__).warning("%d passengers outside [%s, %s) dropped", (~ok).sum(), idx[0], idx[-1] + pd.Timedelta(freq))
    counts = np.bincount(pos[ok], minlength=len(idx)).astype(float)
    return pd.Series(counts, index=idx, name="arrivals")
 
def sampled_arrivals(flight_demand, seed: int, freq: str = "15min") -> pd.Series:
    """
    flight_demand : list of Flight (needs .arrival_sampler)
    seed          : random seed for the sampling
    freq          : frequency of the output bins
    """
    from app.services.samplers import ArrivalGenerator
    pax = ArrivalGenerator().generate(flight_demand, seed)
    return passengers_to_arrivals(pax, freq)
 
def expected_arrivals(flight_demand, freq: str = "15min") -> pd.Series:
    """
    E[pax in bin] = sum_flights n_pax * (G(right) - G(left)), with G(t) = P(arrival <= t)
    = 1 - CDF(minutes before departure of t). The total is exactly the number of passengers, but fractional per bin.
    """
    if not flight_demand:
        raise ValueError("no flights")
    for f in flight_demand:
        if f.arrival_sampler is None:
            raise ValueError(f"Flight {f.flight_key} has no arrival_sampler attached.")
        
    first = min(f.departure_dt - dt.timedelta(minutes=float(f.arrival_sampler.x[-1])) for f in flight_demand)
    last = max(f.departure_dt - dt.timedelta(minutes=float(f.arrival_sampler.x[0])) for f in flight_demand)
    idx = _bin_index(first, (last + pd.Timedelta(freq)), freq)
    edges = idx.append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(freq)]))
    out = np.zeros(len(idx))
    for f in flight_demand:
        n = sum(v for v in f.pax_segments.values() if v > 0)
        if n == 0:
            continue
        mins_before = (pd.Timestamp(f.departure_dt) - edges) / pd.Timedelta(minutes=1)
        G = 1.0 - np.interp(np.asarray(mins_before, dtype=float), f.arrival_sampler.x, f.arrival_sampler.cdf)
        out += n * np.diff(G)
    return pd.Series(out, index=idx, name="arrivals")
 
# arrivals -> demand grid
def arrivals_to_demand_grid(arrivals, mean_processing_rate: float, bin_minutes: float = 15.0):
    """
    arrivals             : (T,) pax arriving in each bin
    mean_processing_rate : pax per HOUR processed by one counter
    returns D
        D              : (n_counters, T) float in [0, 1]
    """
    a = np.asarray(arrivals, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("arrivals must be 1-D (T,)")
    if mean_processing_rate <= 0 or bin_minutes <= 0:
        raise ValueError("mean_processing_rate and bin_minutes must be > 0")
    pax_per_counter_bin = mean_processing_rate * (bin_minutes / 60.0)
    required_exact = a / pax_per_counter_bin
    n_counters = int(np.ceil(required_exact.max())) if a.size else 0
    if n_counters == 0:
        raise ValueError("no arrivals: the demand grid would be empty")
    D = np.clip(required_exact[None, :] - np.arange(n_counters)[:, None], 0.0, 1.0)
    return D

# results
@dataclasses.dataclass(frozen=True, eq=False)
class LanePlannerResults:
    # ---- parameters of the run ----
    L: float                                   # requested shift length, hours
    L_cells: int                               # shift length actually used, in bins
    alpha: float                               # penalty per unit of idle capacity
    K_total: int                               # shared budget of shifts
    bin_minutes: float                         # bin length, minutes
    # ---- inputs / outputs ----
    demand_grid: np.ndarray                    # (n_counters, T) demand in [0, 1], one row per counter
    server_labels: tuple[str]                  # ordered by priority (first opened first)
    server_schedule: dict[str, list[tuple[dt.datetime, dt.datetime]]]   # label -> [(start, end), ...]
    df_curves: pd.DataFrame                    # one row per bin (see SecurityLanePlanner.run)

class SecurityLanePlanner:
    """
    arrivals        : pd.Series of passengers per bin, indexed by a regular pd.DatetimeIndex
    server_registry : {key: {"label", "priority", "dist", ...}}; processing times in minutes
    """
    def __init__(self, arrivals: pd.Series, server_registry: dict[str, dict]):
        if not isinstance(arrivals, pd.Series) or not isinstance(arrivals.index, pd.DatetimeIndex):
            raise TypeError("arrivals must be a pd.Series indexed by a pd.DatetimeIndex")
        if (arrivals < 0).any():
            raise ValueError("arrivals must be >= 0")
        if not server_registry:
            raise ValueError("server_registry cannot be empty.")

        self.arrivals = arrivals.astype(float)
        self.time_values: pd.DatetimeIndex = arrivals.index
        self.bin_duration = self._regular_step(self.time_values)
        self.bin_minutes = self.bin_duration.total_seconds() / 60.0

        self.server_registry = server_registry
        self.server_labels = sort_labels_by_priority(server_registry)
        self.rate_by_label = get_processing_rate_by_label(server_registry, self.bin_minutes)

        self.demand_grid = arrivals_to_demand_grid(self.arrivals.to_numpy(), mean_processing_rate(server_registry), self.bin_minutes)

        self.results: LanePlannerResults | None = None
        self.report: CoverageReport | None = None

    def run(self, L: float, alpha: float, K_total: int, exact: bool = False) -> LanePlannerResults:
        """
        L       : shift length in hours (converted to a whole number of bins)
        alpha   : penalty per unit of idle capacity (objective = covered demand - alpha * idle)
        K_total : total number of shifts shared among the servers
        exact   : exact DP for the budget split instead of the greedy one

        results.df_curves has one row per bin with columns
            time        bin start
            demand      passengers arriving in the bin
            processing  passengers the open servers can process in the bin (sum of their own rates)
            servers     tuple of the labels open in the bin
        """
        if L <= 0:
            raise ValueError("L must be > 0 (hours).")
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if int(K_total) != K_total or K_total < 0:
            raise ValueError("K_total must be an integer >= 0.")
        K_total = int(K_total)

        bin_hours = self.bin_minutes / 60.0
        L_cells = max(1, math.floor(L / bin_hours + 0.5))

        # only as many counter rows as there are servers to staff them
        n_rows = self.demand_grid.shape[0]
        n_usable = min(n_rows, len(self.server_labels))
        if n_usable < n_rows:
            logger.warning(
                "[SecurityLanePlanner] demand needs up to %d servers but the registry has %d: "
                "the demand of the upper %d row(s) (%.2f counter-bins) cannot be served.",
                n_rows, n_usable, n_rows - n_usable, float(self.demand_grid[n_usable:].sum()),
            )
        grid = self.demand_grid[:n_usable]
        labels = self.server_labels[:n_usable]

        # ---- optimisation ----
        curve = solve_coverage_batch(grid, L_cells, alpha, return_curve=True)
        alloc = allocate_budget(curve, K_total, exact=exact)
        sols = solve_allocated(grid, L_cells, alpha, alloc)

        row_labels = list(self.server_labels[:n_usable]) + ["no server"] * (n_rows - n_usable)
        self.report = coverage_report(self.demand_grid, sols, L_cells, self.time_values, row_labels)

        # ---- server schedule: row c is staffed by the c-th server in priority order ----
        schedule: dict[str, list[tuple[dt.datetime, dt.datetime]]] = {}
        for label, sol in zip(labels, sols):
            if sol.n_rects == 0:
                continue
            schedule[label] = [
                (
                    self.time_values[int(s0)].to_pydatetime(),
                    (self.time_values[int(s0)] + L_cells * self.bin_duration).to_pydatetime(),
                )
                for s0 in sorted(sol.starts)
            ]

        # ---- curves, from the same coverage the schedule comes from ----
        extra = L_cells - 1
        time_ext = self.time_values
        if extra:
            time_ext = time_ext.append(pd.date_range(self.time_values[-1] + self.bin_duration, periods=extra, freq=self.bin_duration))
        active = np.vstack([s.coverage for s in sols])                    # (n_usable, T + extra)
        rates = np.array([self.rate_by_label[label] for label in labels])
        df_curves = pd.DataFrame({
            "time": time_ext,
            "demand": np.concatenate([self.arrivals.to_numpy(), np.zeros(extra)]),
            "processing": rates @ active,
            "servers": pd.Series(
                [tuple(labels[c] for c in np.flatnonzero(active[:, t])) for t in range(active.shape[1])],
                dtype=object,
            ),
        })

        self.results = LanePlannerResults(
            L=L,
            L_cells=L_cells,
            alpha=alpha,
            K_total=K_total,
            bin_minutes=self.bin_minutes,
            demand_grid=self.demand_grid,
            server_labels=tuple(labels),
            server_schedule=schedule,
            df_curves=df_curves,
        )
        return self.results

    # ------------------------------------------------------------------ #
    @staticmethod
    def _regular_step(index: pd.DatetimeIndex) -> pd.Timedelta:
        if len(index) < 2:
            raise ValueError("arrivals needs at least 2 bins to infer the bin length.")
        steps = np.diff(index.values)
        if not (steps == steps[0]).all():
            raise ValueError("the time index of arrivals must be regular (constant bin length).")
        return pd.Timedelta(steps[0])



@dataclasses.dataclass
class SecuritySimulatorResults:
    df_timeline: pd.DataFrame
    df_kpis: pd.DataFrame
    # df_pax: pd.DataFrame
    # df_server: pd.DataFrame
    df_queue_cdf: pd.DataFrame

@dataclasses.dataclass(frozen=True)
class SimulationScenario:
    scenario_id: int                        # unique identifier for the scenario
    arrival_scenario: ArrivalScenario       # arrivals for the scenario
    intervals: list[Interval]               # intervals for the scenario
    server_registry: dict                   # server registry containing server processing parameters
    timebin: int                            # used for timeline aggregation
    buffer_threshold: float                 # threshold for buffer time in the simulation  
    start_dt: dt.datetime                   # simulation start datetime, used for converting passenger arrival times to relative minutes 

class SecuritySimulator:
    def __init__(
            self, 
            flight_demand: list[Flight],
            intervals: list[Interval], 
            server_registry: dict[str, dict]
        ):

        if not isinstance(intervals, list) or not intervals:
            raise ValueError("Intervals must be a non-empty list.")

        if not isinstance(flight_demand, list) or not flight_demand:
            raise ValueError("Flight demand must be a non-empty list.")

        missing_sampler = [f.flight_key for f in flight_demand if f.arrival_sampler is None]
        if missing_sampler:
            raise ValueError(f"Flights missing arrival_sampler: {missing_sampler[:5]}")

        if not isinstance(server_registry, dict) or not server_registry:
            raise ValueError("Server registry must be a non-empty dictionary.")
        
        self.flight_demand = flight_demand
        self.intervals = validate_intervals(intervals, require_contiguous=True)
        self.server_registry = server_registry

        operational_days = [pd.Timestamp(flight.departure_dt).normalize() for flight in self.flight_demand]
        self.day = pd.Series(operational_days).mode().iloc[0]

    def run(self, n_scenarios: int, n_workers: int, timebin: int, buffer_threshold: int, base_seed: int):

        # generate arrivals for each scenario
        self.arrival_scenarios = ArrivalGenerator().generate_scenarios(
            flight_demand=self.flight_demand,
            n_scenarios=n_scenarios,
            base_seed=base_seed
        )

        self.n_workers = n_workers
        self.timebin = timebin
        self.buffer_threshold = buffer_threshold
        self.base_seed = base_seed

        all_arrival_dts = [
            pax.arrival_dt
            for scenario in self.arrival_scenarios.values()
            for pax in scenario.passengers
        ]

        interval_start = min(
            interval.start
            for interval in self.intervals
        )

        if all_arrival_dts:
            simulation_start_dt = min(min(all_arrival_dts), interval_start)
        else:
            simulation_start_dt = interval_start

        simulation_start_dt = (
            pd.Timestamp(simulation_start_dt)
            .floor(f"{self.timebin}min")
            .to_pydatetime()
        )

        # prepare simulation tasks for each scenario
        tasks_list = []

        for scenario_id, arrival_scenario in self.arrival_scenarios.items():

            tasks_list.append(
                SimulationScenario(
                    scenario_id=scenario_id,
                    arrival_scenario=arrival_scenario,
                    intervals=self.intervals,
                    server_registry=self.server_registry,
                    timebin=self.timebin,
                    buffer_threshold=self.buffer_threshold,
                    start_dt=simulation_start_dt
                )
            )

        self.tasks = tasks_list

        # run the simulations for all scenarios
        self._run_simulations()

        return SecuritySimulatorResults(
            df_timeline=self.df_timeline,
            df_kpis=self.df_kpis,
            # df_pax=self.df_pax,
            # df_server=self.df_server,
            df_queue_cdf=self.df_queue_cdf
        )
    
    def _run_simulations(self):

        if not self.tasks:
            empty = self.add_run_metadata(pd.DataFrame())

            self.df_timeline = empty.copy()
            self.df_kpis = empty.copy()
            self.df_pax = empty.copy()
            # self.df_server = empty.copy()
            self.df_queue_cdf = metrics.get_queue_cdf(pd.DataFrame())
            return

        # launch parallel processing for the simulation tasks
        logger.info(f"[SecuritySimulator] Launching pool: {len(self.tasks)} total tasks, {self.n_workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            all_results = list(executor.map(single_scenario_worker, self.tasks))
        
        if len(self.tasks) != len(all_results):
            raise RuntimeError("Number of simulation results does not match number of tasks.")

        # Aggregate and format timeline
        self.df_timeline = (
            self.add_run_metadata(
                self._aggregate_timelines(all_results)
            )
            .sort_values("time_bin")
            .reset_index(drop=True)
        )

        # Aggregate and format KPIs
        df_kpis_scen = pd.DataFrame([r["kpis"].to_dict() for r in all_results])
        self.df_kpis = (
            self.add_run_metadata(
                self._aggregate_kpis(df_kpis_scen)
            )
            .reset_index(drop=True)
        )

        # Aggregate and format passenger data
        pax_list = [r["df_pax"] for r in all_results if not r["df_pax"].empty]

        if pax_list:
            df_pax = (
                pd.concat(pax_list, ignore_index=True)
                .sort_values(["scenario_id", "arrival_dt"])
                .reset_index(drop=True)
            )

            self.df_pax = self.add_run_metadata(df_pax)
        else:
            self.df_pax = self.add_run_metadata(pd.DataFrame())

        # Aggregate and format server data
        # servers_list = [r["df_server"] for r in all_results if not r["df_server"].empty]
        # self.df_server = (
        #     self.add_run_metadata(
        #         self._aggregate_servers(servers_list)
        #     ).reset_index(drop=True)
        # )

        self.df_queue_cdf = metrics.get_queue_cdf(self.df_pax)

    def add_run_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            date=pd.Timestamp(self.day),
            base_seed=self.base_seed,
            n_scenarios=len(self.tasks),
        )

        metadata_cols = [
            "date",
            "base_seed",
            "n_scenarios",
        ]

        return df[
            metadata_cols
            + [
                col
                for col in df.columns
                if col not in metadata_cols
            ]
        ]

    def _aggregate_timelines(self, results: list[dict]) -> pd.DataFrame:
        """
        Aggregate timelines from multiple scenarios by calculating median and percentiles for each time bin.
        """
        all_timelines = pd.concat([res['df_timeline'] for res in results], ignore_index=True)

        def percentile(q):
            return lambda x: np.percentile(x, q)
        
        p5 = percentile(5)
        p25 = percentile(25)
        p75 = percentile(75)
        p95 = percentile(95)

        stats = [
            ("mean", "mean"),
            ("median", "median"),
            ("p5", p5),
            ("p25", p25),
            ("p75", p75),
            ("p95", p95),
        ]
            
        agg_dict = {}

        for column in [
            "arrival_rate",
            "processing_rate",
            "queue_length",
            "occupied_servers",
            "expected_qtime_min"
        ]:
            agg_dict[column] = [
                (f"{column}_{suffix}", func)
                for suffix, func in stats
            ]

        agg_dict["utilization_avg_pct"] = [
            ("utilization_median_pct", "median"),
            ("utilization_p5_pct", p5),
            ("utilization_p25_pct", p25),
            ("utilization_p75_pct", p75),
            ("utilization_p95_pct", p95),
        ]

        agg_dict["capacity"] = [("capacity", "first")]

        agg_dict["service_rate"] = [("service_rate", "first")]

        agg_dict["theoretical_processing_rate"] = [("theoretical_processing_rate", "first")]

        named_agg = {
            new_name: pd.NamedAgg(
                column=column,
                aggfunc=func,
            )
            for column, funcs in agg_dict.items()
            for new_name, func in funcs
        }

        return (
            all_timelines
            .groupby("time_bin", as_index=False)
            .agg(**named_agg)
            .sort_values("time_bin")
            .reset_index(drop=True)
        )

    def _aggregate_kpis(self, df_kpis_scenarios: pd.DataFrame) -> pd.DataFrame:

        numeric_cols = [
            column
            for column in df_kpis_scenarios.select_dtypes(
                include="number"
            ).columns
            if column not in {
                "scenario_id",
                "seed",
            }
        ]

        return (
            df_kpis_scenarios[numeric_cols]
            .mean()
            .to_frame()
            .T
        )

    def _aggregate_servers(self, servers_list: list[pd.DataFrame]) -> pd.DataFrame:

        if not servers_list:
            return pd.DataFrame()

        all_servers = pd.concat(servers_list, ignore_index=True)

        def percentile(q):
            return lambda x: np.percentile(x, q)

        stats = [
            ("mean", "mean"),
            ("median", "median"),
            ("p5", percentile(5)),
            ("p95", percentile(95)),
        ]

        stochastic_cols = [
            "served_pax",
            "busy_minutes",
            "mean_processing_time",
            "idle_minutes",
            "utilization_pct",
        ]

        aggregations = {
            f"{column}_{suffix}": pd.NamedAgg(
                column=column,
                aggfunc=func,
            )
            for column in stochastic_cols
            for suffix, func in stats
        }

        aggregations["available_minutes"] = pd.NamedAgg(
            column="available_minutes",
            aggfunc="first",
        )

        return (
            all_servers
            .groupby("server_id", as_index=False)
            .agg(**aggregations)
            .sort_values("server_id")
            .reset_index(drop=True)
        )

def single_scenario_worker(task: SimulationScenario) -> dict:

    arrivals = task.arrival_scenario.passengers
    intervals = task.intervals
    server_registry = task.server_registry
    timebin = task.timebin
    buffer_threshold = task.buffer_threshold
    seed = task.arrival_scenario.seed
    simulation_start_dt = task.start_dt

    if not arrivals:
        return metrics.build_empty_simulation_result(seed=seed, intervals=intervals, timebin=timebin)
    
    # convert pax datetimes to minutes relative to simulation start, for easier processing in the simulation
    all_pax = [
        dataclasses.replace(
            p,
            arrival_min=(p.arrival_dt - simulation_start_dt).total_seconds() / 60,
            flight_departure_min=(
                p.flight_departure_dt - simulation_start_dt
            ).total_seconds() / 60,
        )
        for p in arrivals
    ]

    validate_simulation_inputs(
        passengers=all_pax,
        intervals=intervals,
        server_registry=server_registry,
    )

    # create and run the simulation for the scenario
    sim = QueueSimulationSinglequeue(
        all_pax=all_pax, 
        intervals=intervals, 
        server_registry=server_registry, 
        start_dt=simulation_start_dt, 
        seed=seed)
    
    result = sim.run()

    df_timeline = metrics.get_df_timeline(result, timebin=timebin, server_registry=server_registry)
    df_timeline["scenario_id"] = task.scenario_id

    kpis = metrics.get_kpis(result, df_timeline=df_timeline, buffer_threshold=buffer_threshold)

    df_pax = metrics.get_df_pax(result.pax_data, simulation_start_dt=simulation_start_dt)
    df_pax["simulation_start_dt"] = simulation_start_dt
    df_pax["scenario_id"] = task.scenario_id

    first_cols = ["scenario_id", "simulation_start_dt",]
    df_pax = df_pax[first_cols + [c for c in df_pax.columns if c not in first_cols]]

    # df_server = metrics.get_df_server(df_pax=df_pax, intervals=intervals)
    # df_server["scenario_id"] = task.scenario_id

    return {
        'df_timeline': df_timeline,
        'kpis': kpis,
        'df_pax': df_pax,
        # 'df_server': df_server,
    }

def validate_simulation_inputs(passengers: list[Passenger], intervals: list[Interval],server_registry: dict) -> None:

    if not intervals:
        raise ValueError("Intervals cannot be empty.")

    if not server_registry:
        raise ValueError("Server registry cannot be empty.")
    
    # Passenger times
    missing_arrival_min = [
        pax.id
        for pax in passengers
        if pax.arrival_min is None
    ]

    if missing_arrival_min:
        raise ValueError(f"Passengers without arrival_min: {missing_arrival_min[:5]}")

    # Intervals
    invalid_intervals = [
        interval
        for interval in intervals
        if interval.end <= interval.start
    ]

    if invalid_intervals:
        raise ValueError("Found intervals with end <= start.")

    # Registry labels

    registry_labels = [
        str(server["label"])
        for server in server_registry.values()
    ]

    if len(registry_labels) != len(set(registry_labels)):
        raise ValueError("Duplicate server labels found in server_registry.")

    registry_labels = set(registry_labels)

    interval_labels = {
        str(label)
        for interval in intervals
        for label in interval.server_labels
    }

    missing_labels = interval_labels - registry_labels

    if missing_labels:
        raise ValueError(f"Missing registry entries for servers: {sorted(missing_labels)}")