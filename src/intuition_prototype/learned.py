"""Fitted ridge utility estimates over telemetry, selecting only predefined actions."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from intuition_prototype.benchmark_protocol import digest, generate_cases, source_digest
from intuition_prototype.navigator import NavigatorEnvironment, investigation_text, run_navigation
from intuition_prototype.records import Action, ActionKind, Episode, Intervention, LearnedCandidateScore, Metrics


MODEL_VERSION = "ridge-selector-v1"
FEATURE_NAMES = (
    "arrival_rate", "completion_fraction", "queue_growth_fraction", "initial_queue_fraction",
    "retry_fraction", "duplicate_fraction", "latency_windows", "latency_missing",
    "drop_fraction", "completion_deficit",
)


def features(metrics: Metrics) -> tuple[float, ...]:
    if not isinstance(metrics, Metrics) or type(metrics.ticks) is not int or metrics.ticks <= 0:
        raise ValueError("Feature extraction requires a valid telemetry window.")
    counts = (
        metrics.arrivals, metrics.completions, metrics.retries, metrics.duplicate_completions,
        metrics.queue_start, metrics.queue_end, metrics.dropped_attempts,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Telemetry counts must be nonnegative integers.")
    if metrics.mean_latency is not None and (
        not math.isfinite(metrics.mean_latency) or metrics.mean_latency < 0
    ):
        raise ValueError("Latency must be finite and nonnegative, or explicitly missing.")
    arrivals = max(1, metrics.arrivals)
    completed = min(4.0, metrics.completions / arrivals)
    return (
        metrics.arrivals / metrics.ticks,
        completed,
        max(-4.0, min(8.0, metrics.queue_growth / arrivals)),
        min(20.0, metrics.queue_start / arrivals),
        min(4.0, metrics.retries / arrivals),
        metrics.duplicate_completions / max(1, metrics.completions + metrics.duplicate_completions),
        0.0 if metrics.mean_latency is None else min(8.0, metrics.mean_latency / metrics.ticks),
        float(metrics.mean_latency is None),
        min(4.0, metrics.dropped_attempts / arrivals),
        max(0.0, 1.0 - completed),
    )


@dataclass(frozen=True)
class TrainingExample:
    config_id: str
    seed: int
    split: str
    observation: Metrics
    utilities: Mapping[str, float]


def _solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, target, strict=True)]
    size = len(target)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Ridge system is numerically singular; fitting aborted.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
                ]
    return [row[-1] for row in augmented]


def fit_model(examples: Sequence[TrainingExample], alpha: float, provenance: dict) -> "LearnedModel":
    if not examples or len(examples) > 4096 or any(example.split != "train" for example in examples):
        raise ValueError("Fitting requires 1..4096 development-train examples, never validation/test rows.")
    historical_ids = {
        case["config_id"] for split in generate_cases().values() for case in split
    }
    if any(example.config_id in historical_ids for example in examples):
        raise ValueError("This experiment excludes all historical Stage 3 configurations from fitting.")
    if len({(example.config_id, example.seed) for example in examples}) != len(examples):
        raise ValueError("Duplicate configuration/seed training rows would reweight the fit.")
    examples = sorted(examples, key=lambda example: (example.config_id, example.seed))
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive.")
    if any(set(example.utilities) != {item.value for item in Intervention} for example in examples):
        raise ValueError("Training targets must cover exactly the permitted interventions.")
    if any(not math.isfinite(value) for example in examples for value in example.utilities.values()):
        raise ValueError("Training targets must be finite.")
    vectors = [features(example.observation) for example in examples]
    centers = [mean(vector[index] for vector in vectors) for index in range(len(FEATURE_NAMES))]
    scales = [
        math.sqrt(mean((vector[index] - centers[index]) ** 2 for vector in vectors))
        for index in range(len(FEATURE_NAMES))
    ]
    scales = [value if value > 1e-12 else 1.0 for value in scales]
    normalized = [
        [(value - center) / scale for value, center, scale in zip(vector, centers, scales, strict=True)]
        for vector in vectors
    ]
    dimension = len(FEATURE_NAMES)
    matrix = [
        [
            math.fsum(vector[i] * vector[j] for vector in normalized)
            + (alpha * len(examples) if i == j else 0.0)
            for j in range(dimension)
        ] for i in range(dimension)
    ]
    models = {}
    for action in Intervention:
        targets = [example.utilities[action.value] for example in examples]
        intercept = mean(targets)
        rhs = [
            math.fsum(vector[i] * (value - intercept) for vector, value in zip(normalized, targets, strict=True))
            for i in range(dimension)
        ]
        models[action.value] = {"intercept": intercept, "coefficients": _solve(matrix, rhs)}
    body = {
        "version": MODEL_VERSION, "feature_names": list(FEATURE_NAMES),
        "centers": centers, "scales": scales, "models": models, "alpha": alpha, "threshold": 0.0,
        "provenance": {
            **provenance, "fit_split": "train",
            "fit_config_ids": sorted({example.config_id for example in examples}),
            "validation_config_ids": [],
            "training_source_sha256": source_digest(Path(__file__)),
        },
    }
    return LearnedModel({**body, "model_sha256": digest(body)})


class LearnedModel:
    def __init__(self, document: dict):
        body = {key: value for key, value in document.items() if key != "model_sha256"}
        if document.get("model_sha256") != digest(body):
            raise ValueError("Learned model fingerprint is missing or corrupt.")
        if set(body) != {
            "version", "feature_names", "centers", "scales", "models", "alpha", "threshold", "provenance",
        } or body["version"] != MODEL_VERSION or body["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Incompatible learned model or feature schema.")
        dimension = len(FEATURE_NAMES)
        if (
            not isinstance(body["centers"], list) or not isinstance(body["scales"], list)
            or not isinstance(body["models"], dict)
            or len(body["centers"]) != dimension or len(body["scales"]) != dimension
            or set(body["models"]) != {item.value for item in Intervention}
        ):
            raise ValueError("Invalid learned model dimensions/actions.")
        numeric = [*body["centers"], *body["scales"], body["alpha"], body["threshold"]]
        for parameters in body["models"].values():
            if (
                not isinstance(parameters, dict) or set(parameters) != {"intercept", "coefficients"}
                or not isinstance(parameters["coefficients"], list)
                or len(parameters["coefficients"]) != dimension
            ):
                raise ValueError("Invalid learned regression parameters.")
            numeric.extend([parameters["intercept"], *parameters["coefficients"]])
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in numeric):
            raise ValueError("Learned model numbers must be finite.")
        if min(body["scales"]) <= 0 or body["alpha"] <= 0 or body["threshold"] < 0:
            raise ValueError("Invalid normalization, regularization or abstention threshold.")
        provenance = body["provenance"]
        if not isinstance(provenance, dict) or not {
            "fit_split", "training_source_sha256", "fit_config_ids", "validation_config_ids",
        } <= set(provenance):
            raise ValueError("Learned model training provenance is missing.")
        if any(
            not isinstance(provenance[key], list)
            or any(not isinstance(value, str) for value in provenance[key])
            or len(set(provenance[key])) != len(provenance[key])
            for key in ("fit_config_ids", "validation_config_ids")
        ):
            raise ValueError("Invalid model development configuration IDs.")
        if (
            provenance["fit_split"] != "train"
            or provenance["training_source_sha256"] != source_digest(Path(__file__))
            or not provenance["fit_config_ids"]
            or set(provenance["fit_config_ids"]) & set(provenance["validation_config_ids"])
        ):
            raise ValueError("Model training provenance is incompatible or leaks validation into fitting.")
        self._document = json.loads(json.dumps(document))

    @property
    def fingerprint(self) -> str:
        return self._document["model_sha256"]

    def document(self) -> dict:
        return json.loads(json.dumps(self._document))

    def with_selection(self, threshold: float, validation_ids: Sequence[str]) -> "LearnedModel":
        body = {key: value for key, value in self.document().items() if key != "model_sha256"}
        body["threshold"] = threshold
        body["provenance"]["validation_config_ids"] = sorted(set(validation_ids))
        return LearnedModel({**body, "model_sha256": digest(body)})

    def score_candidates(
        self, baseline: Metrics, capabilities: Sequence[Intervention],
        hypothesis_states: Mapping[Intervention, str], remaining_budget: int, remaining_steps: int = 12,
    ) -> tuple[LearnedCandidateScore, ...]:
        if any(not isinstance(item, Intervention) for item in capabilities):
            raise ValueError("Capabilities must contain supported interventions.")
        if any(value not in ("untested", "weakened", "supported") for value in hypothesis_states.values()):
            raise ValueError("Unknown hypothesis state.")
        model = self._document
        normalized = tuple(
            (value - center) / scale
            for value, center, scale in zip(features(baseline), model["centers"], model["scales"], strict=True)
        )
        candidates = []
        for intervention in sorted(Intervention, key=lambda item: item.value):
            parameters = model["models"][intervention.value]
            prediction = parameters["intercept"] + math.fsum(
                weight * value for weight, value in zip(parameters["coefficients"], normalized, strict=True)
            )
            reasons = []
            if intervention not in capabilities:
                reasons.append("unsupported capability")
            state = hypothesis_states.get(intervention, "untested")
            if state != "untested":
                reasons.append(f"already tested ({state}); same-state repetition is redundant")
            if prediction <= model["threshold"]:
                reasons.append("estimated utility does not exceed the development-selected abstention threshold")
            if remaining_budget < 3:
                reasons.append("unaffordable: test plus one evidence update requires 3 units")
            if remaining_steps < 3:
                reasons.append("step limit: test, reframe, and terminal action require 3 steps")
            question, assumption = investigation_text(intervention)
            action = Action(ActionKind.TEST, intervention)
            candidates.append(LearnedCandidateScore(
                intervention.value, action, question, assumption, action.cost,
                round(prediction / action.cost, 6), prediction, model["threshold"],
                not reasons, tuple(reasons),
                "Fitted ridge estimate of independent utility, not a probability. "
                "Only telemetry is used for prediction; tested-state and bounds govern eligibility.",
                self.fingerprint, FEATURE_NAMES, normalized, tuple(parameters["coefficients"]),
                parameters["intercept"],
            ))
        return tuple(candidates)


def load_model(path: Path) -> LearnedModel:
    if not path.is_file():
        raise ValueError(f"Learned model is missing at {path}; run Stage 4 training first.")
    if path.stat().st_size > 1_000_000:
        raise ValueError("Learned model exceeds the supported size limit.")
    with path.open(encoding="utf-8") as source:
        document = json.load(source)
    if not isinstance(document, dict):
        raise ValueError("Learned model must be a JSON object.")
    return LearnedModel(document)


def run_learned(
    environment: NavigatorEnvironment, model: LearnedModel, budget: int = 10, max_steps: int = 12,
) -> Episode:
    return run_navigation(
        environment, budget, max_steps, selector=model.score_candidates, policy="learned",
    )
