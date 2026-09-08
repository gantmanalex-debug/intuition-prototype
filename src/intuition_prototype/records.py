"""Claims reference evidence IDs; predictions are recorded before experiments."""

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    ASK = "ask"
    TEST = "test"
    REFRAME = "reframe"
    ANSWER = "answer"
    STOP = "stop"


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
        if self.kind in (ActionKind.ANSWER, ActionKind.STOP):
            return 0
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


@dataclass(frozen=True)
class CandidateScore:
    key: str
    action: Action
    question: str
    assumption: str
    discrimination: float
    anomaly_relevance: float
    evidence_gap: float
    novelty: float
    redundancy: float
    cost: int
    score: float
    eligible: bool
    rejection_reasons: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if (
            self.action.kind != ActionKind.TEST
            or self.action.intervention is None
            or self.key != self.action.intervention.value
        ):
            raise ValueError("Candidate must identify a supported test action.")
        if self.cost != self.action.cost:
            raise ValueError("Candidate cost does not match its action.")
        components = (
            self.discrimination, self.anomaly_relevance,
            self.evidence_gap, self.novelty, self.redundancy,
        )
        if any(not 0 <= value <= 1 for value in components):
            raise ValueError("Heuristic score components must be in [0, 1].")
        if self.score != candidate_utility(*components, self.cost):
            raise ValueError("Candidate score contradicts its components.")
        if self.eligible != (not self.rejection_reasons):
            raise ValueError("Candidate eligibility contradicts its rejection reasons.")


def candidate_utility(
    discrimination: float, relevance: float, gap: float,
    novelty: float, redundancy: float, cost: int,
) -> float:
    return round((2 * discrimination + 2 * relevance + gap + novelty - 2 * redundancy) / cost, 6)


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    hypothesis_id: str
    evidence_ids: tuple[str, ...]
    remaining_budget: int
    candidates: tuple[CandidateScore, ...]
    chosen_key: str | None
    reason: str


@dataclass(frozen=True)
class FuseRecord:
    id: str
    evidence_ids: tuple[str, ...]
    triggers: tuple[str, ...]
    no_progress_count: int
    alternate_keys: tuple[str, ...]
    disposition: str
    reason: str


@dataclass(frozen=True)
class AnswerRecord:
    id: str
    action_id: str
    evidence_ids: tuple[str, ...]
    claim: str
    stop_reason: str


Record = (
    Observation | Hypothesis | Prediction | Question | InterventionRecord | Result
    | DecisionRecord | FuseRecord | AnswerRecord
)


@dataclass(frozen=True)
class Episode:
    records: tuple[Record, ...]
    budget: int
    cost: int
    stop_reason: str
    interpretation: str
    policy: str = "scripted"
    max_steps: int = 12
