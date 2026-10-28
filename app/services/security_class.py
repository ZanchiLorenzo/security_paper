from app.services.entities import  Interval, Flight, Passenger
from app.services.simulation import QueueSimulationSinglequeue
from app.services.samplers import ArrivalScenario, ArrivalGenerator, expected_arrivals
from app.services.utils import sort_labels_by_priority, get_processing_rate_by_label, validate_intervals
from app.services.coverage_dp import CoverageReport, coverage_report, solve_coverage_batch, allocate_budget, solve_allocated, coverage_report_multi, solve_coverage_batch_multi, allocate_budget_multi, solve_allocated_multi, allocate_milp, MultiAllocation
import app.services.metrics as metrics
import concurrent.futures
import datetime as dt
import math
from typing import Optional, Sequence, Union
import pandas as pd
import numpy as np
import dataclasses
import logging
logger = logging.getLogger(__name__)

# arrivals -> demand grid
def arrivals_to_demand_grid_het(arrivals, rates):
    """
    Demand grid with the real rate of every lane (instead of the mean rate), layers in priority order.
 
    arrivals : (T,) or (N, T) pax per bin
    rates    : pax per bin of the lanes in priority order, mu_1..mu_M (planner.rate_by_label[label] for label in server_labels)
    returns  : D of shape (R, T) or (N, R, T), D[..., r, t] = clip((a_t - C_{r-1}) / mu_r, 0, 1), C_r = mu_1 + .. + mu_r
 
    Layer r is the share of lane r that is needed in the slot once the first r-1 lanes are full: when the lanes that are
    open are the first k, the covered demand is sum_r mu_r d_r = min(a_t, C_k). Layers beyond the M lanes (needed only if
    the lanes cannot absorb the peak) use the mean rate; they are dropped later, as before. With equal rates this is the
    previous grid: D = clip(a / mu - (r - 1), 0, 1).
    """
    a = np.asarray(arrivals, dtype=np.float64)
    c = np.asarray(rates, dtype=np.float64)
    if c.ndim != 1 or c.size == 0 or (c <= 0).any():
        raise ValueError("rates must be a non-empty 1-D array of positive pax per bin")
    a_max = float(a.max()) if a.size else 0.0
    if a_max <= 0:
        raise ValueError("no arrivals: the demand grid would be empty")
    ext = list(c)
    while sum(ext) < a_max:                                   # virtual layers with the mean rate beyond the M lanes
        ext.append(float(c.mean()))
    ext = np.asarray(ext)
    cum = np.cumsum(ext)
    R = int(np.searchsorted(cum, a_max - 1e-9 * float(c.mean()), side="left")) + 1     # smallest R with C_R >= max a
    mu_r, lo = ext[:R], np.concatenate([[0.0], cum[:R - 1]])
    if a.ndim == 1:
        return np.clip((a[None, :] - lo[:, None]) / mu_r[:, None], 0.0, 1.0)
    return np.clip((a[:, None, :] - lo[None, :, None]) / mu_r[None, :, None], 0.0, 1.0)

def slotwise_coverage(demand, processing) -> float:
    """
    "True" coverage of a schedule: sum_t min(arrivals_t, capacity_t) / sum_t arrivals_t, with the open lanes' capacity pooled
    (any lane serves any passenger). demand / processing: pax per bin, e.g. the columns of df_curves.
    The layer coverage (PlannerResults.coverage) assigns each layer to its own lane and is a lower bound of this one.
    """
    d = np.asarray(demand, dtype=np.float64)
    p = np.asarray(processing, dtype=np.float64)
    total = float(d.sum())
    return float(np.minimum(d, p).sum() / total) if total > 0 else float("nan")

def _slim_report(rep: CoverageReport, coverage_slot: float) -> CoverageReport:
    """The report already comes slim from coverage_dp; add the true (slot-wise) coverage to total."""
    total = rep.total.copy()
    total["coverage_slot"] = coverage_slot
    return CoverageReport(blocks=rep.blocks, rows=rep.rows, total=total)

# -- Planner on single day --
@dataclasses.dataclass(frozen=True, eq=False)
class PlannerResults:
    # ---- parameters of the run ----
    L: float                                   # requested shift length, hours
    L_cells: int                               # shift length actually used, in bins
    theta: float                               # minimum utilization to open a window (0 <= theta < 1)
    K_total: int                               # shared budget of shifts
    bin_minutes: float                         # bin length, minutes
    # ---- inputs / outputs ----
    demand_grid: np.ndarray                    # (n_counters, T) demand in [0, 1], one row per counter
    server_labels: tuple[str, ...]                  # ordered by priority (first opened first)
    server_schedule: dict[str, list[tuple[dt.datetime, dt.datetime]]]   # label -> [(start, end), ...]
    df_curves: pd.DataFrame                    # one row per bin (see SecurityPlanner.run)
    coverage: float                            # layer coverage: sum over blocks of the covered load of their layer / total load (what the DP maximises)
 
    # multi-length runs (empty for run())
    lengths: tuple = ()                        # allowed shift lengths, hours
    lengths_cells: tuple = ()                  # same, in bins
    length_totals: Optional[dict] = None       # {hours: max shifts across servers}
    theta_rows: Optional[tuple] = None         # theta actually used on each row (server_labels order); None = theta everywhere
    coverage_slot: Optional[float] = None      # true coverage: sum_t min(arrivals_t, capacity_t) / sum_t arrivals_t (see slotwise_coverage)

class SecurityPlanner:
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

        step = self._regular_step(arrivals.index)
        horizon_end = arrivals.index[0].normalize() + pd.Timedelta(days=1)
        if horizon_end is not None and pd.Timestamp(horizon_end) > arrivals.index[-1] + step:
            extra = pd.date_range(arrivals.index[-1] + step, pd.Timestamp(horizon_end), freq=step, inclusive="left")
            arrivals = pd.concat([arrivals, pd.Series(0.0, index=extra)])
        
        self.arrivals = arrivals.astype(float)
        self.time_values: pd.DatetimeIndex = arrivals.index
        self.bin_duration = step
    
        self.bin_minutes = self.bin_duration.total_seconds() / 60.0

        self.server_registry = server_registry
        self.server_labels = sort_labels_by_priority(server_registry)
        self.rate_by_label, _ = get_processing_rate_by_label(server_registry, self.bin_minutes)

        rates_in_order = [self.rate_by_label[l] for l in self.server_labels]
        self.demand_grid = arrivals_to_demand_grid_het(self.arrivals.to_numpy(), rates_in_order)

        self.results: PlannerResults | None = None
        self.report: CoverageReport | None = None

    def run(self, L: float, theta: float, K_total: int, exact: bool = False) -> PlannerResults:
        """
        L       : shift length in hours
        theta   : minimum utilization to open a window, in [0, 1): a window is worth opening iff its mean load >= theta
                  (objective = covered demand - theta * capacity)
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
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1].")
        if int(K_total) != K_total or K_total < 0:
            raise ValueError("K_total must be an integer >= 0.")
        K_total = int(K_total)
 
        bin_hours = self.bin_minutes / 60.0
        L_cells = max(1, math.floor(L / bin_hours + 0.5))
 
        # only as many counter rows as there are servers to staff them
        n_rows = self.demand_grid.shape[0]
        n_usable = min(n_rows, len(self.server_labels))
        if n_usable < n_rows:
            logger.warning(f"[SecurityPlanner] demand needs up to {n_rows} servers but the registry has {n_usable}")
        grid = self.demand_grid[:n_usable]
        labels = self.server_labels[:n_usable]
 
        # ---- optimization ----
        logger.info(f"[SecurityPlanner] Launching allocation DP: L={L:.2f}h, theta={theta:.2f}")
        # 1) For each row compute the best achievable value for every number of shifts k = 0..K
        curve = solve_coverage_batch(grid, L_cells, theta)
        # 2) Split the shared budget K_total among the rows by maximising the sum of their curves
        # (greedy: optimal only if the curves are concave; exact=True uses a DP over rows x budget).
        alloc = allocate_budget(curve, K_total, exact=exact)
        # 3) Re-solve each row with its own number of shifts k to recover the actual start times.
        sols = solve_allocated(grid, L_cells, theta, alloc)
 
        # get coverage report
        row_labels = list(self.server_labels[:n_usable]) + ["no server"] * (n_rows - n_usable)
        report = coverage_report(D=self.demand_grid, sols=sols, L=L_cells, time_values=self.time_values, theta=theta, row_labels=row_labels)
        
        # server schedule: row c is staffed by the c-th server in priority order
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
 
        # curves, for plotting: demand, processing, servers open
        extra = L_cells - 1
        time_ext = self.time_values
        if extra:
            time_ext = time_ext.append(pd.date_range(self.time_values[-1] + self.bin_duration, periods=extra, freq=self.bin_duration))
        active = np.vstack([s.coverage for s in sols])
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

        cov_slot = slotwise_coverage(df_curves["demand"], df_curves["processing"])
        self.report = _slim_report(report, cov_slot)

        # output results
        self.results = PlannerResults(
            L=L,
            L_cells=L_cells,
            theta=theta,
            K_total=K_total,
            bin_minutes=self.bin_minutes,
            demand_grid=self.demand_grid,
            server_labels=tuple(labels),
            server_schedule=schedule,
            df_curves=df_curves,
            coverage=self.report.total.coverage,
            theta_rows=tuple(float(theta) for _ in labels),
            coverage_slot=cov_slot,
        )
        return self.results
 
    def run_multi(self, lengths: Sequence[float], theta: float, K_total: int,
                  length_totals: Optional[dict] = None, theta_first: Optional[float] = None) -> PlannerResults:
        """
        lengths       : allowed shift lengths in hours, e.g. [2, 3]
        theta         : minimum utilization to open a window, in [0, 1)
        K_total       : total number of shifts (of any length) shared among the servers
        length_totals : {hours: max shifts of that length across all servers}
        theta_first   : theta of the first row (first server in priority order) if different from `theta`;
                        theta_first=0 keeps that server on for as long as the budget allows
        """
        lengths_h = sorted({float(l) for l in lengths})
        if not lengths_h or lengths_h[0] <= 0:
            raise ValueError("lengths must be a non-empty list of positive durations (hours).")
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1].")
        if theta_first is not None and not 0 <= theta_first <= 1:
            raise ValueError("theta_first must be in [0, 1].")
        if int(K_total) != K_total or K_total < 0:
            raise ValueError("K_total must be an integer >= 0.")
        K_total = int(K_total)
 
        bin_hours = self.bin_minutes / 60.0
        cells_of = {h: max(1, math.floor(h / bin_hours + 0.5)) for h in lengths_h}
        lengths_cells = [cells_of[h] for h in lengths_h]
        if len(set(lengths_cells)) < len(lengths_cells):
            raise ValueError(f"lengths {lengths_h} h collapse onto the same number of bins ({lengths_cells}) "
                             f"with {self.bin_minutes:g}-minute bins.")
 
        def to_cells(d: Optional[dict], name: str) -> Optional[dict]:
            if d is None:
                return None
            keys = {float(h) for h in d}
            if not keys <= set(lengths_h):
                raise ValueError(f"{name} has lengths {sorted(keys - set(lengths_h))} not in lengths={lengths_h}")
            return {cells_of[float(h)]: int(v) for h, v in d.items()}
 
        totals_cells = to_cells(length_totals, "length_totals")
 
        # only as many counter rows as there are servers to staff them
        n_rows = self.demand_grid.shape[0]
        n_usable = min(n_rows, len(self.server_labels))
        if n_usable < n_rows:
            logger.warning(f"[SecurityPlanner] demand needs up to {n_rows} servers but the registry has {n_usable}")
        grid = self.demand_grid[:n_usable]
        labels = self.server_labels[:n_usable]
 
        theta_rows = np.full(n_usable, float(theta))
        if theta_first is not None:
            theta_rows[0] = float(theta_first)
 
        # ---- optimization ----
        logger.info(f"[SecurityPlanner] Launching multi-length allocation: lengths={lengths_h} h, theta={theta:.2f}")
        curves = solve_coverage_batch_multi(grid, lengths_cells, theta_rows, allow_overhang=False)
        alloc = allocate_milp(curves, lengths_cells, K_total=K_total, length_per_day=totals_cells)
        sols = solve_allocated_multi(grid, theta_rows, alloc, allow_overhang=False)
 
        row_labels = list(labels) + ["no server"] * (n_rows - n_usable)
        theta_report = np.append(theta_rows, np.full(n_rows - n_usable, float(theta)))
 
        report = coverage_report_multi(D=self.demand_grid, sols=sols, time_values=self.time_values, row_labels=row_labels, theta=theta_report)
        schedule, df_curves = self._schedule_and_curves(sols, labels)
        cov_slot = slotwise_coverage(df_curves["demand"], df_curves["processing"])
        self.report = _slim_report(report, cov_slot)

        self.results = PlannerResults(
            L=lengths_h[-1], L_cells=lengths_cells[-1],
            theta=theta, K_total=K_total, bin_minutes=self.bin_minutes,
            demand_grid=self.demand_grid, server_labels=tuple(labels),
            server_schedule=schedule, df_curves=df_curves,
            coverage=self.report.total.coverage,
            lengths=tuple(lengths_h), lengths_cells=tuple(lengths_cells), length_totals=length_totals,
            theta_rows=tuple(float(t) for t in theta_rows),
            coverage_slot=cov_slot,
        )
        return self.results
 
    def _schedule_and_curves(self, sols, labels):
        """Schedule {label: [(start, end), ...]} and df_curves, from solutions with a length per block."""
        schedule = {}
        for label, sol in zip(labels, sols):
            if sol.n_rects == 0:
                continue
            order = np.argsort(sol.starts)
            schedule[label] = [
                (self.time_values[int(s)].to_pydatetime(),
                 (self.time_values[int(s)] + int(l) * self.bin_duration).to_pydatetime())
                for s, l in zip(sol.starts[order], sol.lengths[order])
            ]
 
        active = np.vstack([s.coverage for s in sols])
        rates = np.array([self.rate_by_label[label] for label in labels])
        df_curves = pd.DataFrame({
            "time": self.time_values,
            "demand": self.arrivals.to_numpy(),
            "processing": rates @ active,
            "servers": pd.Series([tuple(labels[c] for c in np.flatnonzero(active[:, t]))
                                  for t in range(active.shape[1])], dtype=object),
        })
        return schedule, df_curves
    @staticmethod
    def _regular_step(index: pd.DatetimeIndex) -> pd.Timedelta:
        if len(index) < 2:
            raise ValueError("arrivals needs at least 2 bins to infer the bin length.")
        steps = np.diff(index.values)
        if not (steps == steps[0]).all():
            raise ValueError("the time index of arrivals must be regular (constant bin length).")
        return pd.Timedelta(steps[0])

# -- Planner on multiple days --
@dataclasses.dataclass(frozen=True, eq=False)
class MultiDayResults:
    # ---- parameters of the run ----
    lengths: tuple                      # allowed shift lengths, hours
    lengths_cells: tuple                # shift lengths in bins
    theta: float
    K_total: Optional[int]
    length_per_day: Optional[dict]       # {length in hours: cap}
    K_day: Optional[dict]               # {day index: cap}
    bin_minutes: float
    # ---- inputs / outputs ----
    days: tuple                         # pd.Timestamp (midnight) of every planned day
    demand_grid: np.ndarray             # (N, R, T) demand in [0, 1]: day, layer, slot
    server_labels: tuple                # layer r is operated by server_labels[r] (only the first len(labels) layers)
    server_schedule: dict               # label -> [(start, end), ...] over all days: input of the simulator
    df_curves: pd.DataFrame             # all days one after the other: time, demand, processing, servers, day
    coverage: float                     # layer coverage (what the DP maximises): covered load / total load, all days
    coverage_slot: float                # true coverage: sum min(arrivals, capacity) / sum arrivals, all days
    allocation: MultiAllocation         # rows (day, layer) in day-major order: row = n * n_layers + r
    day_results: tuple                  # one single-day PlannerResults per day: use it with plot_allocation
 
class MultiDayPlanner:
    """
    Plans several days at once. Days are kept separate:
      1) the demand grid of every day (layers x T) goes to solve_coverage_batch_multi on its own: value curves;
      2) one MILP allocates the budgets across all the (day, layer) rows of the horizon, with limits on the number of shifts of each day;
      3) every day is rebuilt from the decision: solutions, report, schedule.
    A shift never crosses midnight.
 
    arrivals        : pd.Series of passengers per bin on a regular DatetimeIndex, e.g. expected_arrivals(flight_demand)
                      over all the days. Bins are assigned to the day of their start (calendar date).
    server_registry : same as SecurityPlanner
    days            : dates to plan (default: every calendar day between the first and the last bin). Arrivals falling
                      on other dates are dropped (with a warning).
    """
 
    def __init__(self, arrivals: pd.Series, server_registry: dict, days: Optional[Sequence] = None):
        if not isinstance(arrivals, pd.Series) or not isinstance(arrivals.index, pd.DatetimeIndex):
            raise TypeError("arrivals must be a pd.Series indexed by a pd.DatetimeIndex")
        if (arrivals < 0).any():
            raise ValueError("arrivals must be >= 0")
        if not server_registry:
            raise ValueError("server_registry cannot be empty.")
 
        self.bin_duration = _regular_step(arrivals.index)
        self.bin_minutes = self.bin_duration.total_seconds() / 60.0
        per_day = 24 * 60 / self.bin_minutes
        if abs(per_day - round(per_day)) > 1e-9:
            raise ValueError(f"a day must contain a whole number of bins; bin = {self.bin_minutes:g} min")
        self.T = int(round(per_day))                       # bins per day
 
        if days is None:
            days = pd.date_range(arrivals.index.min().normalize(), arrivals.index.max().normalize(), freq="D")
        self.days = tuple(pd.Timestamp(d).normalize() for d in days)
        if not self.days:
            raise ValueError("no days to plan")
        if len(set(self.days)) != len(self.days) or list(self.days) != sorted(self.days):
            raise ValueError("days must be distinct and increasing")
 
        # (N, T) arrivals, one row per day, on the calendar grid of each day
        index = pd.DatetimeIndex(
            np.concatenate([pd.date_range(d, periods=self.T, freq=self.bin_duration).values for d in self.days]))
        a = arrivals.astype(float)
        a = a[~a.index.duplicated()]
        aligned = a.reindex(index, fill_value=0.0)
        dropped = float(a.sum() - aligned.sum())
        if dropped > 1e-6:
            logger.warning(f"[MultiDayPlanner] {dropped:.1f} expected passengers fall outside the planned days: dropped")
        N = len(self.days)
        self.arrivals = aligned
        self.A = aligned.to_numpy().reshape(N, self.T)
        self.time_values_by_day = [index[n * self.T:(n + 1) * self.T] for n in range(N)]
 
        self.server_registry = server_registry
        self.server_labels = sort_labels_by_priority(server_registry)
        self.rate_by_label, _ = get_processing_rate_by_label(server_registry, self.bin_minutes)
 
        # demand grid (N, R, T)
        # R is the same for all days, so that the layers are comparable and the rows of the MILP line up.
        rates_in_order = [self.rate_by_label[l] for l in self.server_labels]
        self.demand_grid = arrivals_to_demand_grid_het(self.A, rates_in_order)     # (N, R, T)
        self.R = self.demand_grid.shape[1]
        
        self.results: Optional[MultiDayResults] = None
        self.reports: Optional[list] = None               # CoverageReport of each day
        self.report: Optional[CoverageReport] = None      # all days together

    def run(
        self,
        lengths: Sequence[float],
        theta: float,
        K_total: Optional[int] = None,
        length_per_day: Optional[dict] = None,
        K_day: Optional[Union[int, dict, Sequence[int]]] = None,
    ) -> MultiDayResults:
        """
        lengths       : allowed shift lengths, hours
        theta         : minimum utilization to open a window, in [0, 1) (objective = covered demand - theta * capacity)
        K_total       : max number of shifts over all days and servers (None = no limit)
        length_per_day : {length in hours: max number of shifts of that length on each day}
        K_day         : max number of shifts per day: an int (all days), a sequence with one value per day,
                        or a dict {day index or date: cap}; a day missing from the dict = no limit
        """
        N, T, R = len(self.days), self.T, self.R
        lengths_h = sorted({float(l) for l in lengths})
        if not lengths_h or lengths_h[0] <= 0:
            raise ValueError("lengths must be a non-empty list of positive durations (hours).")
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1].")
        if K_total is not None and (int(K_total) != K_total or K_total < 0):
            raise ValueError("K_total must be an integer >= 0.")
 
        bin_hours = self.bin_minutes / 60.0
        cells_of = {h: max(1, math.floor(h / bin_hours + 0.5)) for h in lengths_h}
        lengths_cells = [cells_of[h] for h in lengths_h]
        if len(set(lengths_cells)) < len(lengths_cells):
            raise ValueError(f"lengths {lengths_h} h collapse onto the same number of bins ({lengths_cells}).")
        if lengths_cells[-1] > T:
            raise ValueError(f"the longest shift ({lengths_cells[-1]} bins) is longer than a day ({T} bins).")
 
        per_day_cells = None
        if length_per_day is not None:
            keys = {float(h) for h in length_per_day}
            if not keys <= set(lengths_h):
                raise ValueError(f"length_per_day has lengths {sorted(keys - set(lengths_h))} not in lengths={lengths_h}")
            per_day_cells = {cells_of[float(h)]: int(v) for h, v in length_per_day.items()}
        day_caps = self._day_caps(K_day)
 
        n_use = min(R, len(self.server_labels))
        if n_use < R:
            logger.warning(f"[MultiDayPlanner] demand needs up to {R} servers but the registry has {n_use}")
        labels = list(self.server_labels[:n_use])
        grid = self.demand_grid[:, :n_use, :]                       # (N, n_use, T)
 
        # 1) value curves, day by day
        logger.info(f"[MultiDayPlanner] curves: {N} days x {n_use} layers, lengths={lengths_h} h, theta={theta:.2f}")
        day_curves = [
            solve_coverage_batch_multi(grid[n], lengths_cells, theta, allow_overhang=False)
            for n in range(N)
        ]
        shape = tuple(np.max([c.shape[1:] for c in day_curves], axis=0))
        curves = np.concatenate([self._pad_curves(c, shape) for c in day_curves], axis=0)   # (N * n_use, ...)
 
        # 2) one allocation of the budgets across all (day, layer) rows of the horizon
        alloc = allocate_milp(
            curves, lengths_cells, n_days=N, n_layers=n_use,
            K_total=K_total, length_per_day=per_day_cells, K_day=day_caps)
 
        # 3) day by day: re-solve the rows with their decision, report, schedule, curves
        reports, day_results, schedule, curve_parts = [], [], {}, []
        for n in range(N):
            sl = slice(n * n_use, (n + 1) * n_use)
            sub = MultiAllocation(k=alloc.k[sl], lengths=alloc.lengths,
                                  value=float(alloc.per_row_value[sl].sum()), per_row_value=alloc.per_row_value[sl])
            sols = solve_allocated_multi(grid[n], theta, sub, allow_overhang=False)
 
            tv = self.time_values_by_day[n]
            row_labels = labels + ["no server"] * (R - n_use)
            rep = coverage_report_multi(D=self.demand_grid[n], sols=sols, time_values=tv, row_labels=row_labels, theta=theta)
 
            sched_n, df_n = self._schedule_and_curves(sols, labels, tv, self.A[n])
            rep = _slim_report(rep, slotwise_coverage(df_n["demand"], df_n["processing"]))
            reports.append(rep)
            curve_parts.append(df_n.assign(day=n))
            for label, windows in sched_n.items():
                schedule.setdefault(label, []).extend(windows)
 
            day_results.append(self._day_result(
                n, lengths_h, lengths_cells, theta, K_total, length_per_day, labels, sched_n, df_n, rep))
 
        schedule = {label: sorted(w) for label, w in schedule.items()}
        self.reports = reports
        df_all = pd.concat(curve_parts, ignore_index=True)
        self.report = self._merge_reports(reports, df_all)
        self.results = MultiDayResults(
            lengths=tuple(lengths_h), lengths_cells=tuple(lengths_cells), theta=theta,
            K_total=None if K_total is None else int(K_total), 
            length_per_day=length_per_day, 
            K_day=day_caps,
            bin_minutes=self.bin_minutes, days=self.days, demand_grid=self.demand_grid,
            server_labels=tuple(labels), server_schedule=schedule,
            df_curves=df_all,
            coverage=float(self.report.total.coverage),
            coverage_slot=float(self.report.total.coverage_slot),
            allocation=alloc, 
            day_results=tuple(day_results))
        return self.results

    def shifts_per_day(self) -> pd.DataFrame:
        """Number of shifts of each length opened on each day (from the schedule actually built)."""
        if self.results is None:
            raise RuntimeError("run() first")
        out = []
        for n, rep in enumerate(self.reports):
            b = rep.blocks
            rec = {"day": self.days[n].date(), "n_blocks": int(len(b))}
            if len(b):
                rec.update({f"n_blocks_{int(l)}": int(c) for l, c in b.groupby("length").size().items()})
            out.append(rec)
        df = pd.DataFrame(out).fillna(0).set_index("day")
        return df.astype(int)
 
    def report_by_day(self) -> pd.DataFrame:
        """One row per day: demand, coverage, idle capacity, objective and shifts (from each day's report)."""
        if self.results is None:
            raise RuntimeError("run() first")
        rows = []
        for n, rep in enumerate(self.reports):
            rec = rep.total.to_dict()
            rec["day"] = self.days[n].date()
            rows.append(rec)
        df = pd.DataFrame(rows).fillna(0).set_index("day")
        counts = [c for c in df.columns if c.startswith("n_blocks")]
        df[counts] = df[counts].astype(int)
        return df
 
    def _day_caps(self, K_day) -> Optional[dict]:
        """K_day -> {day index: cap}."""
        N = len(self.days)
        if K_day is None:
            return None
        if isinstance(K_day, dict):
            out = {}
            for k, v in K_day.items():
                if isinstance(k, (int, np.integer)) and not isinstance(k, bool):
                    n = int(k)
                else:
                    ts = pd.Timestamp(k).normalize()
                    if ts not in self.days:
                        raise ValueError(f"K_day has the date {ts.date()} that is not a planned day")
                    n = self.days.index(ts)
                if not 0 <= n < N:
                    raise ValueError(f"K_day has day index {n} outside 0..{N - 1}")
                out[n] = int(v)
            return out
        if np.isscalar(K_day):
            return {n: int(K_day) for n in range(N)}
        K_day = list(K_day)
        if len(K_day) != N:
            raise ValueError(f"K_day has {len(K_day)} values for {N} days")
        return {n: int(v) for n, v in enumerate(K_day)}
 
    @staticmethod
    def _pad_curves(curves: np.ndarray, shape: tuple) -> np.ndarray:
        """Edge-pad the state axes of (rows, K_1+1, .., K_m+1) up to shape (K'_j >= K_j)."""
        pad = [(0, 0)] + [(0, int(t) - c) for c, t in zip(curves.shape[1:], shape)]
        return np.pad(curves, pad, mode="edge")
 
    def _schedule_and_curves(self, sols, labels, tv, arrivals_day):
        """Schedule {label: [(start, end), ...]} and df_curves of one day, from solutions with per-block lengths."""
        schedule = {}
        for label, sol in zip(labels, sols):
            if sol.n_rects == 0:
                continue
            order = np.argsort(sol.starts)
            schedule[label] = [
                (tv[int(s)].to_pydatetime(), (tv[int(s)] + int(l) * self.bin_duration).to_pydatetime())
                for s, l in zip(sol.starts[order], sol.lengths[order])
            ]
        active = np.vstack([s.coverage for s in sols])
        rates = np.array([self.rate_by_label[label] for label in labels])
        df = pd.DataFrame({
            "time": tv,
            "demand": arrivals_day,
            "processing": rates @ active,
            "servers": pd.Series(
                [tuple(labels[c] for c in np.flatnonzero(active[:, t])) for t in range(active.shape[1])], dtype=object),
        })
        return schedule, df
 
    def _day_result(self, n, lengths_h, lengths_cells, theta, K_total, length_per_day, labels, sched, df, rep):
        """Single-day PlannerResults, so that plotutils.plot_allocation(day_result) works as for SecurityPlanner."""
        values = dict(
            L=lengths_h[-1], L_cells=lengths_cells[-1], theta=theta, K_total=K_total, bin_minutes=self.bin_minutes,
            demand_grid=self.demand_grid[n], server_labels=tuple(labels), server_schedule=sched,
            df_curves=df, coverage=float(rep.total.coverage),
            lengths=tuple(lengths_h), lengths_cells=tuple(lengths_cells),
            theta_rows=tuple(float(theta) for _ in labels), coverage_slot=float(rep.total.coverage_slot),
        )
        fields = {f.name for f in dataclasses.fields(PlannerResults)}
        return PlannerResults(**{k: v for k, v in values.items() if k in fields})
 
    def _merge_reports(self, reports: list, df_curves: pd.DataFrame) -> CoverageReport:
        """Reports of all days in one: blocks and rows get a 'day' column, total is recomputed."""
        blocks, rows = [], []
        for n, rep in enumerate(reports):
            b = rep.blocks.copy()
            b.insert(0, "day", n)
            blocks.append(b)
            r = rep.rows.reset_index().rename(columns={"row": "layer"})
            r.insert(0, "day", n)
            rows.append(r)
        blocks = pd.concat(blocks, ignore_index=True)
        rows = pd.concat(rows, ignore_index=True)
 
        keys = []
        for rep in reports:                      # n_blocks_<length> only exists for the lengths actually used
            keys += [c for c in rep.total.index if c not in ("coverage", "coverage_slot") and c not in keys]
        total = {c: float(sum(rep.total.get(c, 0.0) for rep in reports)) for c in keys}
        total["n_blocks"] = int(total["n_blocks"])
        total["coverage"] = total["covered_demand"] / total["demand"] if total["demand"] > 0 else np.nan
        total["coverage_slot"] = slotwise_coverage(df_curves["demand"], df_curves["processing"])
        return CoverageReport(blocks=blocks, rows=rows, total=pd.Series(total))
 
def _regular_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Same regular-step check as SecurityPlanner._regular_step."""
    if len(index) < 2:
        raise ValueError("arrivals needs at least 2 bins to infer the bin length.")
    steps = np.diff(index.values)
    if not (steps == steps[0]).all():
        raise ValueError("the time index of arrivals must be regular (constant bin length).")
    return pd.Timedelta(steps[0])

# -- Simulator --
@dataclasses.dataclass
class SimulatorResults:
    df_timeline: pd.DataFrame
    df_kpis: pd.DataFrame
    df_pax: pd.DataFrame
    df_queue_cdf: pd.DataFrame

@dataclasses.dataclass(frozen=True)
class SimulationTask:
    """
    Represents a single simulation scenario to be run in parallel.
    """
    scenario_id: int                        # unique identifier for the scenario
    arrival_scenario: ArrivalScenario       # arrivals for the task
    intervals: list[Interval]               # intervals for the task
    server_registry: dict                   # server registry containing server processing parameters
    timebin: int                            # used for timeline aggregation
    start_dt: dt.datetime                   # simulation start datetime, used to convert passenger arrival times to relative minutes 

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

    def run(self, n_scenarios: int, n_workers: int, timebin: int, base_seed: int):

        # generate arrivals for each scenario
        self.arrival_scenarios = ArrivalGenerator().generate_scenarios(
            flight_demand=self.flight_demand,
            n_scenarios=n_scenarios,
            base_seed=base_seed
        )

        self.n_workers = n_workers
        self.timebin = timebin
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
                SimulationTask(
                    scenario_id=scenario_id,
                    arrival_scenario=arrival_scenario,
                    intervals=self.intervals,
                    server_registry=self.server_registry,
                    timebin=self.timebin,
                    start_dt=simulation_start_dt
                )
            )

        self.tasks = tasks_list

        # run the simulations for all scenarios
        self._run_simulations()

        return SimulatorResults(
            df_timeline=self.df_timeline,
            df_kpis=self.df_kpis,
            df_pax=self.df_pax,
            df_queue_cdf=self.df_queue_cdf
        )
    
    def _run_simulations(self):

        if not self.tasks:
            empty = self.add_run_metadata(pd.DataFrame())

            self.df_timeline = empty.copy()
            self.df_kpis = empty.copy()
            self.df_pax = empty.copy()
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

def single_scenario_worker(task: SimulationTask) -> dict:

    arrivals = task.arrival_scenario.passengers
    intervals = task.intervals
    server_registry = task.server_registry
    timebin = task.timebin
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

    kpis = metrics.get_kpis(result, df_timeline=df_timeline)

    df_pax = metrics.get_df_pax(result.pax_data, simulation_start_dt=simulation_start_dt)
    df_pax["simulation_start_dt"] = simulation_start_dt
    df_pax["scenario_id"] = task.scenario_id

    first_cols = ["scenario_id", "simulation_start_dt",]
    df_pax = df_pax[first_cols + [c for c in df_pax.columns if c not in first_cols]]

    return {
        'df_timeline': df_timeline,
        'kpis': kpis,
        'df_pax': df_pax,
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