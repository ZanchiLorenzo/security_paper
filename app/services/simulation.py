import datetime as dt
import simpy
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from bisect import bisect_right
import numpy as np

class SimulationState:
    def __init__(self, start_dt):
        self.start_dt = start_dt

        self.target_capacity = 0
        self.active_capacity = 0
        self.target_server_labels: set[str] = set()
        self.active_server_labels: set[str] = set()

        self.queue_length = 0
        self.occupied_servers = 0

        self.pax_data = []
        self.queue_var = []
        self.server_var = []
        self.complete_pax_var = []
        self.capacity = 0
        self.state_events = []

    def record_queue(self, sim_minute, variation):
        self.queue_length += variation
        self.queue_var.append({
            "sim_minute": sim_minute,
            "queue_length": self.queue_length,
            "variation": variation
        })
        self.record_state(sim_minute)

    def record_server(self, sim_minute, variation):
        self.occupied_servers += variation
        self.server_var.append({
            "sim_minute": sim_minute,
            "occupied_servers": self.occupied_servers,
            "variation": variation
        })
        self.record_state(sim_minute)

    def record_completed(self, pax_id, sim_minute):
        self.complete_pax_var.append({
            "pax_id": pax_id,
            "sim_minute": sim_minute
        })
        
    def record_capacity(self, sim_minute, capacity):
        self.capacity = capacity
        self.record_state(sim_minute)

    def record_state(self, sim_minute):
        """Record the current state of the system at a given simulation minute.
        if nothing changes in queue length, server occupancy, or capacity, no new state event is recorded to save memory.
        """
        if self.state_events:
            last = self.state_events[-1]
            if (last["queue_length"] == self.queue_length and
                last["occupied_servers"] == self.occupied_servers and
                last["capacity"] == self.capacity):
                return

        self.state_events.append({
            "sim_minute": sim_minute,
            "queue_length": self.queue_length,
            "occupied_servers": self.occupied_servers,
            "capacity": self.capacity,
        })

@dataclass
class SimulationResult:
    start_dt: dt.datetime
    intervals: list
    pax_data: list
    queue_var: list
    server_var: list
    complete_pax_var: list
    state_events: list

class QueueSimulationSinglequeue:
    """
    Single FIFO queue served by a time-varying set of labelled servers.
 
    Assumptions:
    - intervals are already validated and non-overlapping;
    - every interval contains server_labels;
    - every server label exists in server_registry;
    - passengers already contain arrival_min and flight_departure_min.
 
    drain_queue=True: when the last interval ends, its servers are not closed: they stay open until the queue is empty
    (and serve the passengers arriving after the closure, if any), so every passenger is eventually served and nobody
    is left "unserved". The queue time of the late passengers is part of the results. If the last interval has no
    servers there is nobody to drain the queue and the flag has no effect.
    drain_queue=False: closure is final, the passengers still waiting are the unserved ones.
    """

    def __init__(self, all_pax, intervals, server_registry, start_dt, seed, drain_queue=True):
        self.all_pax = sorted(all_pax, key=lambda x: x.arrival_min)
        self.intervals = sorted(intervals, key=lambda interval: interval.start)
        self.server_registry = server_registry
        self.start_dt = start_dt
        self.seed = seed
 
        self.rng = np.random.default_rng(seed)
 
        self.server_registry_by_label = {
            str(server["label"]): server
            for server in server_registry.values()
        }
 
        self.last_interval_end_min = (
            self.intervals[-1].end - self.start_dt
        ).total_seconds() / 60
 
        # Used to determine which servers are scheduled at any simulation time.
        self._interval_starts = [
            interval.start
            for interval in self.intervals
        ]
 
        self._interval_label_sets = [
            frozenset(
                str(label)
                for label in interval.server_labels
            )
            for interval in self.intervals
        ]
 
        self.drain_queue = bool(drain_queue) and len(self._interval_label_sets[-1]) > 0
        self._capacity_schedule = self._build_capacity_schedule()
    
    
    def run(self):
        state = SimulationState(start_dt=self.start_dt)

        env = simpy.Environment()
        available_servers = simpy.FilterStore(env)

        env.process(
            self.capacity_manager(
                env=env,
                servers=available_servers,
                state=state,
            )
        )

        for passenger in self.all_pax:
            env.process(
                self.pax_process(
                    env=env,
                    pax=passenger,
                    state=state,
                    servers=available_servers,
                )
            )

        # Runs until all arrivals, services and final timeouts are resolved.
        env.run()

        return SimulationResult(
            start_dt=self.start_dt,
            intervals=self.intervals,
            pax_data=state.pax_data,
            queue_var=state.queue_var,
            server_var=state.server_var,
            complete_pax_var=state.complete_pax_var,
            state_events=state.state_events,
        )
    
    def _build_capacity_schedule(self) -> list[tuple[dt.datetime, frozenset[str]]]:
        """
        Build the sequence of capacity changes.
 
        This also handles gaps between intervals: at the end of an interval,
        capacity becomes zero unless another interval starts at the same time.
        With drain_queue the last interval never ends: its servers stay open to empty the queue.
        """
        changes = {}
        last = len(self.intervals) - 1
 
        for i, (interval, labels) in enumerate(zip(
            self.intervals,
            self._interval_label_sets,
        )):
            changes[interval.start] = labels
            if not (self.drain_queue and i == last):
                changes[interval.end] = frozenset()
 
        schedule = []
 
        for change_dt, target_labels in sorted(changes.items()):
            # Skip consecutive changes that do not alter the target set.
            if schedule and schedule[-1][1] == target_labels:
                continue
 
            schedule.append(
                (change_dt, target_labels)
            )
 
        return schedule

    def capacity_manager(self, env, servers, state):
        """
        Manage the capacity changes over time.
        This process runs in the background and updates the set of active servers
        according to the capacity schedule.
        """

        for change_dt, target_labels in self._capacity_schedule:
            change_min = (
                change_dt - self.start_dt
            ).total_seconds() / 60

            if env.now < change_min:
                yield env.timeout(change_min - env.now)

            yield from self._apply_target_servers(
                env=env,
                servers=servers,
                state=state,
                target_labels=target_labels,
            )

    def _apply_target_servers(self, env, servers, state, target_labels):
        """
        Apply the target server set at the current simulation time.

        Idle servers are removed immediately.
        Busy servers finish their current passenger and are retired afterward.
        """
        target_labels = set(target_labels)

        state.target_server_labels = target_labels
        state.target_capacity = len(target_labels)

        # First remove servers no longer scheduled, when currently idle.
        servers_to_remove = (
            state.active_server_labels - target_labels
        )

        for server_label in sorted(servers_to_remove):
            if server_label not in servers.items:
                # The server is busy. It will be retired when service ends.
                continue

            yield servers.get(
                lambda item, expected=server_label: item == expected
            )

            state.active_server_labels.discard(server_label)

        # Then activate newly scheduled servers.
        servers_to_add = (
            target_labels - state.active_server_labels
        )

        for server_label in sorted(servers_to_add):
            yield servers.put(server_label)
            state.active_server_labels.add(server_label)

        state.active_capacity = len(
            state.active_server_labels
        )

        # Capacity records the planned/target capacity, not draining servers.
        state.record_capacity(
            sim_minute=env.now,
            capacity=state.target_capacity,
        )

    def pax_process(self, env, pax, state, servers):
        """
        Process a single passenger through the queue and service.
        """
 
        if env.now < pax.arrival_min:
            yield env.timeout(pax.arrival_min - env.now)
 
        state.record_queue(sim_minute=env.now, variation=1)
 
        # Passenger arriving at or after final closure (with drain_queue the servers are still there).
        if not self.drain_queue and env.now >= self.last_interval_end_min:
            state.pax_data.append(pax)
            return
 
        server_label = yield from self._acquire_server(
            env=env,
            servers=servers,
            state=state,
        )
 
        # No server obtained before final closure.
        # The passenger remains represented in the residual queue.
        if server_label is None:
            state.pax_data.append(pax)
            return
 
        processing_time = self._sample_processing_time(server_label=server_label)
 
        state.record_queue( sim_minute=env.now, variation=-1)
        state.record_server(sim_minute=env.now, variation=1,)
 
        service_start = env.now
 
        yield env.timeout(processing_time)
 
        service_end = env.now
 
        state.record_server(sim_minute=service_end, variation=-1)
        state.record_completed(pax_id=pax.id, sim_minute=service_end,)
 
        completed_pax = replace(
            pax,
            start_min=service_start,
            end_min=service_end,
            server_id=server_label,
            processing_time=processing_time,
        )
 
        state.pax_data.append(completed_pax)
 
        # The server may have been closed or changed during the service.
        if self._server_is_scheduled(server_label=server_label, sim_minute=service_end):
            state.active_server_labels.add(server_label)
            state.active_capacity = len(state.active_server_labels)
 
            yield servers.put(server_label)
 
        else:
            self._retire_server(server_label=server_label, state=state)
 
    def _acquire_server(self, env, servers, state):
        """
        Wait for an available server until final closure.
 
        A loop is used to handle the rare boundary case where a server token
        becomes available exactly when that server is no longer scheduled.
 
        With drain_queue there is no closure: wait for a server for as long as it takes.
        """
        if self.drain_queue:
            while True:
                server_label = str((yield servers.get()))
                if self._server_is_scheduled(server_label=server_label, sim_minute=env.now):
                    return server_label
                self._retire_server(server_label=server_label, state=state)
 
        while env.now < self.last_interval_end_min:
            remaining_time = (self.last_interval_end_min - env.now)
 
            server_request = servers.get()
            closure_timeout = env.timeout(remaining_time)
 
            outcome = yield server_request | closure_timeout
 
            # Final closure has priority over starting a new service.
            if env.now >= self.last_interval_end_min:
                if server_request in outcome:
                    acquired_label = str(outcome[server_request])
 
                    self._retire_server(server_label=acquired_label, state=state)
                else:
                    server_request.cancel()
 
                return None
 
            if server_request not in outcome:
                server_request.cancel()
                return None
 
            server_label = str(outcome[server_request])
 
            # Protect against simultaneous capacity-change/service events.
            if self._server_is_scheduled(server_label=server_label, sim_minute=env.now):
                return server_label
 
            # Stale token belonging to a server just closed.
            self._retire_server(server_label=server_label, state=state)
 
        return None
 
    def _server_is_scheduled(self, server_label: str, sim_minute: float) -> bool:
        """
        Return whether a server is scheduled at a given simulation minute.
 
        Intervals are interpreted as [start, end).
        """
        current_dt = (
            self.start_dt
            + dt.timedelta(minutes=float(sim_minute))
        )
 
        interval_index = (
            bisect_right(
                self._interval_starts,
                current_dt,
            )
            - 1
        )
 
        if interval_index < 0:
            return False
 
        interval = self.intervals[interval_index]
 
        if current_dt >= interval.end and not (self.drain_queue and interval_index == len(self.intervals) - 1):
            return False
 
        return (
            str(server_label)
            in self._interval_label_sets[interval_index]
        )

    def _retire_server(self, server_label: str, state) -> None:
        state.active_server_labels.discard(
            str(server_label)
        )

        state.active_capacity = len(
            state.active_server_labels
        )

    def _sample_processing_time(self, server_label: str,) -> float:
        params = self.server_registry_by_label.get(str(server_label))

        if params is None:
            raise ValueError(f"Server {server_label} not found in registry.")

        distribution = str(params["dist"]).lower()
        shift = float(params.get("shift", 0.0))

        for _ in range(1_000):
            if distribution == "gamma":
                sampled = self.rng.gamma(
                    shape=float(params["shape"]),
                    scale=float(params["scale"]),
                )

            elif distribution == "gaussian":
                sampled = self.rng.normal(
                    loc=float(params["mean"]),
                    scale=float(params["sigma"]),
                )

            elif distribution == "exponential":
                sampled = self.rng.exponential(
                    scale=float(params["scale"]),
                )

            else:
                raise ValueError(
                    "Unsupported distribution for server "
                    f"{server_label}: {distribution}"
                )

            processing_time = float(sampled + shift)

            if processing_time > 0:
                return processing_time

        raise RuntimeError(f"Unable to sample a positive processing time for server {server_label}.")