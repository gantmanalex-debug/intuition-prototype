"""Bounded discrete-time queue model. Family settings are evaluator-only."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
import random
from typing import Protocol
from uuid import uuid4

from intuition_prototype.records import Action, ActionKind, Intervention, Metrics


SCENARIOS = ("worker_capacity", "db_contention", "retry_amplification")
WINDOW_TICKS = 30
MAX_TICKS = 240
MAX_QUEUE = 4096


@dataclass(frozen=True)
class SimulatorConfig:
    """Evaluator-only initial conditions; the causal update rules are unchanged."""

    workers: int = 6
    db_capacity: float = 40.0
    lock_penalty: float = 0.01
    retry_timeout: int = 1000
    retries_enabled: bool = True
    arrival_min: int = 3
    arrival_max: int = 5

    def __post_init__(self) -> None:
        if type(self.workers) is not int or not 2 <= self.workers <= 16:
            raise ValueError("Initial workers must be an integer in [2, 16].")
        if (
            not math.isfinite(self.db_capacity) or not 1 <= self.db_capacity <= 64
            or not math.isfinite(self.lock_penalty) or not 0 <= self.lock_penalty <= 0.25
        ):
            raise ValueError("DB capacity must be in [1, 64] and lock penalty in [0, 0.25].")
        if type(self.retry_timeout) is not int or not 1 <= self.retry_timeout <= 1000:
            raise ValueError("Retry timeout must be an integer in [1, 1000].")
        if type(self.retries_enabled) is not bool:
            raise ValueError("Retry enablement must be a boolean.")
        if (
            type(self.arrival_min) is not int or type(self.arrival_max) is not int
            or not 0 <= self.arrival_min <= self.arrival_max <= 8
        ):
            raise ValueError("Arrival bounds must be integers with 0 <= min <= max <= 8.")


@dataclass(frozen=True)
class Snapshot:
    token: str


class InvestigationEnvironment(Protocol):
    def snapshot(self) -> Snapshot: ...
    def restore(self, snapshot: Snapshot) -> None: ...
    def perform(self, action: Action) -> Metrics | None: ...


@dataclass
class _Request:
    created: int
    next_retry: int
    retries: int = 0
    completed: bool = False


@dataclass
class _Attempt:
    request_id: int
    remaining: float = 2.0


@dataclass
class _State:
    tick: int
    workers: int
    db_capacity: float
    lock_penalty: float
    retry_timeout: int
    retries_enabled: bool
    random: random.Random
    requests: list[_Request]
    queue: deque[_Attempt]
    active: list[_Attempt]


class QueueSimulator:
    """Public operations expose metrics, actions, and opaque same-instance tokens.

    This Python API is an information boundary for the scripted runner, not a
    security sandbox against code inspecting private attributes.
    """

    def __init__(self, scenario: str, seed: int = 7):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario!r}. Choose from {SCENARIOS}.")
        config = {
            "worker_capacity": SimulatorConfig(),
            "db_contention": SimulatorConfig(db_capacity=3.0, lock_penalty=0.12),
            "retry_amplification": SimulatorConfig(
                db_capacity=9.0, lock_penalty=0.12, retry_timeout=2,
            ),
        }[scenario]
        self._configure(config, seed)

    @classmethod
    def from_config(cls, config: SimulatorConfig, seed: int = 7) -> "QueueSimulator":
        simulator = cls.__new__(cls)
        simulator._configure(config, seed)
        return simulator

    def _configure(self, config: SimulatorConfig, seed: int) -> None:
        if not isinstance(config, SimulatorConfig):
            raise ValueError("Expected a validated SimulatorConfig.")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer between 0 and 4294967295.")
        self._config = config
        self._seed = seed
        self._snapshots: dict[str, _State] = {}
        self.reset()

    def reset(self) -> None:
        config = self._config
        self._state = _State(
            0, config.workers, config.db_capacity, config.lock_penalty,
            config.retry_timeout, config.retries_enabled,
            random.Random(self._seed), [], deque(), [],
        )
        self._snapshots.clear()
        self._advance()

    def snapshot(self) -> Snapshot:
        if len(self._snapshots) >= 8:
            raise ValueError("Snapshot limit reached (8); reset to clear snapshots.")
        token = uuid4().hex
        self._snapshots[token] = deepcopy(self._state)
        return Snapshot(token)

    def restore(self, snapshot: Snapshot) -> None:
        if not isinstance(snapshot, Snapshot) or snapshot.token not in self._snapshots:
            raise ValueError("Unknown or expired snapshot for this simulator.")
        self._state = deepcopy(self._snapshots[snapshot.token])

    def perform(self, action: Action) -> Metrics | None:
        if not isinstance(action, Action):
            raise ValueError("Expected a validated Action.")
        if action.kind == ActionKind.REFRAME:
            return None
        if action.kind in (ActionKind.ANSWER, ActionKind.STOP):
            raise ValueError("Answer and stop are local policy actions, not simulator operations.")
        if self._state.tick + WINDOW_TICKS > MAX_TICKS:
            raise ValueError(f"Simulation tick limit ({MAX_TICKS}) reached; reset or restore.")
        if action.intervention == Intervention.DOUBLE_WORKERS:
            if self._state.workers != self._config.workers:
                raise ValueError("Workers already increased; restore before another trial.")
            self._state.workers *= 2
        elif action.intervention == Intervention.DISABLE_RETRIES:
            self._state.retries_enabled = False
        elif action.intervention == Intervention.DOUBLE_DB_CAPACITY:
            if self._state.db_capacity != self._config.db_capacity:
                raise ValueError("DB capacity already increased; restore before another trial.")
            self._state.db_capacity *= 2
        return self._advance()

    def capabilities(self) -> tuple[Intervention, ...]:
        return tuple(Intervention)

    def _advance(self) -> Metrics:
        state = self._state
        start = len(state.queue) + len(state.active)
        arrivals = completions = retries = duplicates = dropped = 0
        latencies = []
        for _ in range(WINDOW_TICKS):
            state.tick += 1
            incoming = state.random.randint(self._config.arrival_min, self._config.arrival_max)
            arrivals += incoming
            for _ in range(incoming):
                request_id = len(state.requests)
                state.requests.append(_Request(state.tick, state.tick + state.retry_timeout))
                if len(state.queue) + len(state.active) < MAX_QUEUE:
                    state.queue.append(_Attempt(request_id))
                else:
                    dropped += 1
            for request_id, request in enumerate(state.requests):
                if (
                    state.retries_enabled and not request.completed
                    and request.retries < 2 and state.tick >= request.next_retry
                ):
                    request.retries += 1
                    request.next_retry = state.tick + state.retry_timeout
                    retries += 1
                    if len(state.queue) + len(state.active) < MAX_QUEUE:
                        state.queue.append(_Attempt(request_id))
                    else:
                        dropped += 1
            while state.queue and len(state.active) < state.workers:
                state.active.append(state.queue.popleft())
            concurrency = len(state.active)
            if not concurrency:
                continue
            # More concurrent DB work consumes capacity in lock coordination.
            capacity = state.db_capacity / (1 + state.lock_penalty * (concurrency - 1))
            progress = min(1.0, capacity / concurrency)
            remaining = []
            for attempt in state.active:
                attempt.remaining -= progress
                if attempt.remaining > 1e-9:
                    remaining.append(attempt)
                    continue
                request = state.requests[attempt.request_id]
                if request.completed:
                    duplicates += 1
                else:
                    request.completed = True
                    completions += 1
                    latencies.append(state.tick - request.created + 1)
            state.active = remaining
        return Metrics(
            WINDOW_TICKS, arrivals, completions, retries, duplicates,
            start, len(state.queue) + len(state.active), dropped,
            round(sum(latencies) / len(latencies), 3) if latencies else None,
        )
