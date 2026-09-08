"""Claims reference evidence IDs; predictions are recorded before experiments."""

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    ASK = "ask"
    TEST = "test"
    REFRAME = "reframe"


class Intervention(str, Enum):
    DOUBLE_WORKERS = "double_workers"
    DISABLE_RETRIES = "disable_retries"
    DOUBLE_DB_CAPACITY = "double_db_capacity"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    intervention: Intervention | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ActionKind):
            raise ValueError("Action kind must be an ActionKind.")
        if self.kind == ActionKind.TEST:
            if not isinstance(self.intervention, Intervention):
                raise ValueError("Test actions require an allowed Intervention.")
        elif self.intervention is not None:
            raise ValueError("Only test actions accept an intervention.")

    @property
    def cost(self) -> int:
        return 2 if self.kind == ActionKind.TEST else 1


@dataclass(frozen=True)
class Metrics:
    ticks: int
    arrivals: int
    completions: int
    retries: int
    duplicate_completions: int
    queue_start: int
    queue_end: int
    dropped_attempts: int
    mean_latency: float | None

    @property
    def queue_growth(self) -> int:
        return self.queue_end - self.queue_start


@dataclass(frozen=True)
class Observation:
    id: str
    source: str
    metrics: Metrics


@dataclass(frozen=True)
class Hypothesis:
    id: str
    claim: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Prediction:
    id: str
    hypothesis_id: str
    claim: str
    minimum_completions: int
    maximum_queue_growth: int
    maximum_retries: int | None = None

    def matches(self, metrics: Metrics) -> bool:
        return (
            metrics.completions >= self.minimum_completions
            and metrics.queue_growth <= self.maximum_queue_growth
            and (self.maximum_retries is None or metrics.retries <= self.maximum_retries)
        )


@dataclass(frozen=True)
class Question:
    id: str
    text: str


@dataclass(frozen=True)
class InterventionRecord:
    id: str
    action: Action
    question_id: str
    prediction_id: str | None
    cost: int


@dataclass(frozen=True)
class Result:
    id: str
    intervention_id: str
    observation_id: str
    prediction_id: str
    matched: bool
    interpretation: str


Record = Observation | Hypothesis | Prediction | Question | InterventionRecord | Result


@dataclass(frozen=True)
class Episode:
    records: tuple[Record, ...]
    budget: int
    cost: int
    stop_reason: str
    interpretation: str
