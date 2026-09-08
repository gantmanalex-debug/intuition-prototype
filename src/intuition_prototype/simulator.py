"""Bounded discrete-time queue model. Family settings are evaluator-only."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Protocol
from uuid import uuid4

from intuition_prototype.records import Action, ActionKind, Intervention, Metrics


SCENARIOS = ("worker_capacity", "db_contention", "retry_amplification")
WINDOW_TICKS = 30
MAX_TICKS = 240
MAX_QUEUE = 4096


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
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer between 0 and 4294967295.")
        self._scenario = scenario
        self._seed = seed
        self._snapshots: dict[str, _State] = {}
        self.reset()

    def reset(self) -> None:
        db_capacity, lock_penalty, timeout = {
            "worker_capacity": (40.0, 0.01, 1000),
            "db_contention": (3.0, 0.12, 1000),
            "retry_amplification": (9.0, 0.12, 2),
        }[self._scenario]
        self._base_db_capacity = db_capacity
        self._state = _State(
            0, 6, db_capacity, lock_penalty, timeout, True,
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
        if self._state.tick + WINDOW_TICKS > MAX_TICKS:
            raise ValueError(f"Simulation tick limit ({MAX_TICKS}) reached; reset or restore.")
        if action.intervention == Intervention.DOUBLE_WORKERS:
            if self._state.workers != 6:
                raise ValueError("Workers already increased; restore before another trial.")
            self._state.workers = 12
        elif action.intervention == Intervention.DISABLE_RETRIES:
            self._state.retries_enabled = False
        elif action.intervention == Intervention.DOUBLE_DB_CAPACITY:
            if self._state.db_capacity != self._base_db_capacity:
                raise ValueError("DB capacity already increased; restore before another trial.")
            self._state.db_capacity *= 2
        return self._advance()

    def _advance(self) -> Metrics:
        state = self._state
        start = len(state.queue) + len(state.active)
        arrivals = completions = retries = duplicates = dropped = 0
        latencies = []
        for _ in range(WINDOW_TICKS):
            state.tick += 1
            incoming = state.random.randint(3, 5)
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
