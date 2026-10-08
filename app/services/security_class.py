from app.services.entities import  Interval, Flight, Passenger
from app.services.optimizer import LanesOptimizer
from app.services.simulation import QueueSimulationSinglequeue
from app.services.samplers import ArrivalScenario, ArrivalGenerator
from app.services.utils import get_processing_rate_by_label, sort_labels_by_priority, validate_intervals
import app.services.metrics as metrics
import concurrent.futures
import datetime as dt
import pandas as pd
import numpy as np
import dataclasses
import logging
logger = logging.getLogger(__name__)

def _aggregate_arrivals(arrival_scenarios_dict, mean_processing_rate, freq="15min"):

    base_scenario = next(
        (scenario for scenario in arrival_scenarios_dict.values()
        if scenario.scenario_id == 1), 
        None,
        )

    if base_scenario is None:
        raise ValueError("Base arrival scenario 1 was not found.")

    if not base_scenario.passengers:
        raise ValueError("Base arrival scenario contains no passengers.")

    arrival_times = [p.arrival_dt for p in base_scenario.passengers]

    first_arrival = pd.Timestamp(min(arrival_times))

    start = min(
        first_arrival.floor(freq),
        first_arrival.normalize(),
    )

    end = pd.Timestamp(max(arrival_times)).ceil(freq) + pd.Timedelta(freq)

    time_axis = pd.date_range(
        start=start,
        end=end,
        freq=freq,
        inclusive="left",
    )

    out = (
        pd.DataFrame({"arrival_dt": arrival_times})
        .set_index("arrival_dt")
        .resample(freq)
        .size()
        .reindex(time_axis, fill_value=0)
        .rename("arrivals")
        .rename_axis("time")
        .reset_index()
    )

    bin_hours = pd.Timedelta(freq).total_seconds() / 3600
    pax_per_counter_bin = mean_processing_rate * bin_hours # mean_processing_rate in pax per hour

    out["required_counters_exact"] = out["arrivals"] / pax_per_counter_bin
    out["required_counters"] = np.ceil(out["required_counters_exact"]).astype(int)

    demand = out["required_counters"].to_numpy()

    grid = (np.arange(demand.max())[:, None] < demand[None, :]).astype(np.uint8)

    return grid, out

@dataclasses.dataclass
class LanePlannerResults:
    L: float
    scale: float
    alpha: float
    grid: np.ndarray
    df_alloc: pd.DataFrame
    df_requirements: pd.DataFrame
    server_schedule: dict[str, list[tuple]]
    df_arrivals: pd.DataFrame

class SecurityLanePlanner:
    def __init__(
        self,
        flight_demand: list[Flight],
        server_registry: dict[str, dict],
        ):
        
        if not isinstance(flight_demand, list) or not flight_demand:
            raise ValueError("Flight demand must be a non-empty list.")
        
        missing_sampler = [f.flight_key for f in flight_demand if f.arrival_sampler is None]
        if missing_sampler:
            raise ValueError(f"Flights missing arrival_sampler: {missing_sampler[:5]}")
        
        if not isinstance(server_registry, dict) or not server_registry:
            raise ValueError("Server registry must be a non-empty dictionary.")
        
        self.flight_demand = flight_demand
        self.server_registry = server_registry

    def run(self, L: float, scale: float, alpha: float, base_seed: int):

        self.arrival_scenarios = ArrivalGenerator().generate_scenarios(
                            flight_demand=self.flight_demand,
                            n_scenarios=1,
                            base_seed=base_seed
                        )

        # generate arrivals for each scenario
        self.df_arrivals = self._build_arrival_profiles()

        mean_processing_rate = self._get_mean_processing_rate()

        # optimize
        logger.info(f"[SecurityLanePlanner] Launching  optimization: L={L}, scale={scale}, alpha={alpha}")

        self._base_grid, self._base_arrival_curve = self._aggregate_arrivals(arrival_scenarios_dict, mean_processing_rate)

        self.lanes_optimizer = LanesOptimizer(
            arrival_scenarios_dict=self.arrival_scenarios,
            mean_processing_rate=mean_processing_rate
        )

        self.df_alloc, self.df_requirements = self.lanes_optimizer.optimize(L=L, scale=scale, alpha=alpha)

        # get server labels by priority for the intervals
        server_labels = sort_labels_by_priority(self.server_registry)
        rate_by_label, _ = get_processing_rate_by_label(self.server_registry)

        self.server_schedule = self._alloc_to_server_schedule(server_labels=server_labels)

        self.df_requirements["server_labels"] = self._server_labels_per_bin(
            server_schedule=self.server_schedule,
            server_labels=server_labels,
            time_values=self.df_requirements["time"],
        )

        self.df_requirements["processing_rate"] = (
            self.df_requirements["server_labels"].apply(lambda labels: sum(rate_by_label[label] for label in labels))
        )

        self.results = LanePlannerResults(
            L=L,
            scale=scale,
            alpha=alpha,
            grid=self.lanes_optimizer.grid,
            df_alloc=self.df_alloc,
            df_requirements=self.df_requirements,
            server_schedule=self.server_schedule,
            df_arrivals=self.df_arrivals
        )



    

    def _alloc_to_server_schedule(self, server_labels: list[str]) -> dict[str, list[tuple]]:
        """
        Convert the allocation DataFrame to a server schedule dictionary.
        Each server label will have a list of tuples representing the start and end times of its allocated intervals.
        The number of server_labels available is treated as an upper bound, not a hard requirement. 
        """
        df_alloc = self.df_alloc

        if df_alloc.empty:
            return {}

        n_labels = len(server_labels)
        max_y = int(df_alloc["y"].max())

        if max_y >= n_labels:
            dropped = int((df_alloc["y"] >= n_labels).sum())
            logger.warning(
                f"[SecurityLanePlanner] Optimizer allocated rows up to y={max_y}, "
                f"but only {n_labels} server_labels are available in the registry. "
                f"Dropping {dropped} allocated interval(s) beyond the available servers."
            )
            df_alloc = df_alloc[df_alloc["y"] < n_labels]

        if df_alloc.empty:
            return {}

        server_schedule: dict[str, list[tuple]] = {}

        for y, group in df_alloc.groupby("y"):
            label = server_labels[int(y)]
            server_schedule[label] = sorted(
                (pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime())
                for start, end in zip(group["start_time"], group["end_time"])
            )

        return server_schedule
    
    def _server_labels_per_bin(
        self,
        server_schedule: dict[str, list[tuple]],
        server_labels: list[str],
        time_values: pd.Series,
    ) -> list[tuple]:
        """
        Per ogni bin di df_requirements, quali server_labels sono attivi in quel
        momento, letti direttamente da server_schedule - non serve piu'
        ricostruire i list[Interval] condivisi solo per questo calcolo interno.
        """
        time_values = pd.to_datetime(time_values)
        n = len(time_values)

        active_matrix = np.zeros((len(server_labels), n), dtype=bool)

        for row, label in enumerate(server_labels):
            periods = server_schedule.get(label)
            if not periods:
                continue

            starts = [pd.Timestamp(s) for s, _ in periods]
            ends = [pd.Timestamp(e) for _, e in periods]

            interval_index = pd.IntervalIndex.from_arrays(starts, ends, closed="left")
            active_matrix[row] = interval_index.get_indexer(time_values) >= 0

        return [
            tuple(server_labels[row] for row in range(len(server_labels)) if active_matrix[row, col])
            for col in range(n)
        ]

    def _build_arrival_profiles(self, timebin_minutes: int = 15) -> pd.DataFrame:

        arrival_scenarios = self.arrival_scenarios
        flight_demand = self.flight_demand

        freq = f"{timebin_minutes}min"

        flight_metadata = pd.DataFrame(
            [
                {
                    "flight_key": flight.flight_key,
                    "flight_number": flight.flight_number,
                    "flight_departure_dt": pd.Timestamp(
                        flight.departure_dt
                    ),
                }
                for flight in flight_demand
            ]
        )

        if flight_metadata["flight_key"].duplicated().any():
            duplicate_keys = (
                flight_metadata.loc[
                    flight_metadata["flight_key"].duplicated(keep=False),
                    "flight_key",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(f"Duplicate flight_key values found: {duplicate_keys[:5]}")

        rows = []

        for scenario in arrival_scenarios.values():
            for passenger in scenario.passengers:
                rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "flight_key": passenger.flight_key,
                        "arrival_dt": pd.Timestamp(passenger.arrival_dt),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "aggregation_level",
                    "flight_key",
                    "flight_number",
                    "flight_departure_dt",
                    "time",
                    "median",
                    "p5",
                    "p25",
                    "p75",
                    "p95",
                ]
            )

        df = pd.DataFrame(rows)
        df["time"] = df["arrival_dt"].dt.floor(freq)

        scenario_ids = sorted(scenario.scenario_id for scenario in arrival_scenarios.values())

        flight_keys = flight_metadata["flight_key"].tolist()

        start = df["time"].min() - pd.Timedelta(minutes=timebin_minutes)
        end = df["time"].max() + pd.Timedelta(minutes= 2 * timebin_minutes)

        time_axis = pd.date_range(
            start=start,
            end=end,
            freq=freq,
            inclusive="left",
        )

        all_counts = (
            df.groupby(["scenario_id", "time"])
            .size()
            .rename("arrivals")
        )

        all_index = pd.MultiIndex.from_product(
            [scenario_ids, time_axis],
            names=["scenario_id", "time"],
        )

        all_counts = (
            all_counts
            .reindex(all_index, fill_value=0)
            .reset_index()
        )

        flight_counts = (
            df.groupby(["scenario_id", "flight_key", "time"])
            .size()
            .rename("arrivals")
        )

        flight_index = pd.MultiIndex.from_product(
            [scenario_ids, flight_keys, time_axis],
            names=["scenario_id", "flight_key", "time"],
        )

        flight_counts = (
            flight_counts
            .reindex(flight_index, fill_value=0)
            .reset_index()
        )

        def aggregate_stats(counts: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
            return (
                counts
                .groupby(group_columns)["arrivals"]
                .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                .unstack()
                .rename(columns={0.05: "p5", 0.25: "p25", 0.5: "median", 0.75: "p75", 0.95: "p95"})
                .reset_index()
                .sort_values(group_columns)
                .reset_index(drop=True)
            )

        df_all = aggregate_stats(
            counts=all_counts,
            group_columns=["time"],
        )

        df_all["aggregation_level"] = "all"

        df_by_flight = aggregate_stats(
            counts=flight_counts,
            group_columns=["flight_key", "time"],
        )

        df_by_flight["aggregation_level"] = "flight"

        df_arrivals = pd.concat(
            [
                df_all,
                df_by_flight,
            ],
            ignore_index=True,
            sort=False,
        )

        df_arrivals = df_arrivals.merge(
            flight_metadata,
            on="flight_key",
            how="left",
            validate="many_to_one",
        )

        df_arrivals = (
            df_arrivals[
                [
                    "aggregation_level",
                    "flight_key",
                    "flight_number",
                    "flight_departure_dt",
                    "time",
                    "median",
                    "p5",
                    "p25",
                    "p75",
                    "p95",
                ]
            ]
            .sort_values(
                ["aggregation_level", "flight_departure_dt", "time"],
                na_position="first",
            )
            .reset_index(drop=True)
        )

        return df_arrivals

    def _get_mean_processing_rate(self) -> float:
        """
        Compute the mean processing rate across all servers.
        """

        rates = []

        for server_label, params in self.server_registry.items():

            dist = params["dist"]

            if dist == "gamma":
                mean_processing_time = (
                    params.get("shift", 0.0)
                    + params["shape"] * params["scale"]
                )

            elif dist == "gaussian":
                mean_processing_time = (
                    params.get("shift", 0.0)
                    + params["mean"]
                )

            elif dist == "exponential":
                mean_processing_time = (
                    params.get("shift", 0.0)
                    + params["scale"]
                )

            else:
                raise ValueError(f"Unsupported processing distribution for server {server_label}: {dist}")

            if mean_processing_time <= 0:
                raise ValueError(f"Mean processing time for server {server_label} must be positive.")

            rates.append(60 / mean_processing_time)

        return float(np.mean(rates))

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