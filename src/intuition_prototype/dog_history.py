"""Bounded history-aware question task and transparent offline ranker.

This is an illustrative synthetic story, not a model of real dogs and not
medical or veterinary advice.  The policy sees only an observation, its own
question/answer history, and legal question identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import re


HYPOTHESES = (
    "wants_outside",
    "returned_and_resting",
    "waiting_for_person",
    "possible_discomfort",
)
QUESTIONS = {
    "outside_orientation": "Is the dog repeatedly orienting from the door toward the outdoor side?",
    "recent_return": "Was the dog observed coming in recently and then remaining by the entry?",
    "person_departure": "Did the behavior begin just after a familiar person left through this door?",
    "outside_cue": "Does an ordinary outside cue, such as showing the leash, increase door-directed behavior?",
    "comfortable_settle": "When nobody approaches the door, does the dog settle in a comfortable resting posture?",
    "discomfort_indicator": (
        "Are there persistent observable discomfort indicators independent of anyone approaching the door? "
        "(This synthetic answer is not a diagnosis.)"
    ),
}
QUESTION_ORDER = tuple(QUESTIONS)
OPENING = (
    "A dog remains near a closed entry door, alternately standing and lying down. "
    "The opening observation alone does not distinguish the four illustrative explanations."
)
MAX_QUESTIONS = 4
RESOLUTION_POSTERIOR = 0.78
RESOLUTION_MARGIN = 0.28

# Prototypes are generator/evaluator semantics, not values exposed by DogQuestionAPI.
_PROTOTYPES = {
    "wants_outside": (1, 0, 0, 1, 0, 0),
    "returned_and_resting": (0, 1, 0, 0, 1, 0),
    "waiting_for_person": (0, 0, 1, 0, 0, 0),
    "possible_discomfort": (0, 0, 0, 0, 0, 1),
}
_INTERACTIONS = {
    "wants_outside": (("outside_orientation", "outside_cue", 4),),
    "returned_and_resting": (("recent_return", "comfortable_settle", 4),),
    "waiting_for_person": (("person_departure", "comfortable_settle", 3),),
    "possible_discomfort": (("discomfort_indicator", "comfortable_settle", 4),),
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def all_patterns() -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.product((0, 1), repeat=len(QUESTION_ORDER)))


def _unnormalized(hypothesis: str, pattern: tuple[int, ...]) -> int:
    prototype = _PROTOTYPES[hypothesis]
    # A mismatch remains possible.  Correlation terms create genuine evidence
    # interactions: the joint observation is not the product of marginals.
    weight = math.prod(3 if value == expected else 1 for value, expected in zip(pattern, prototype))
    answers = dict(zip(QUESTION_ORDER, pattern))
    for left, right, multiplier in _INTERACTIONS[hypothesis]:
        li, ri = QUESTION_ORDER.index(left), QUESTION_ORDER.index(right)
        if answers[left] == prototype[li] and answers[right] == prototype[ri]:
            weight *= multiplier
    return weight


_NORMALIZERS = {h: sum(_unnormalized(h, p) for p in all_patterns()) for h in HYPOTHESES}


def pattern_probability(hypothesis: str, pattern: tuple[int, ...]) -> float:
    if hypothesis not in HYPOTHESES or pattern not in all_patterns():
        raise ValueError("Unknown hypothesis or invalid answer pattern.")
    return _unnormalized(hypothesis, pattern) / _NORMALIZERS[hypothesis]


def _history_map(history: tuple[tuple[str, int], ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for question, answer in history:
        if question not in QUESTIONS or answer not in (0, 1) or question in result:
            raise ValueError("History must contain distinct known questions with binary answers.")
        result[question] = answer
    return result


@lru_cache(maxsize=None)
def likelihood(hypothesis: str, history: tuple[tuple[str, int], ...]) -> float:
    known = _history_map(history)
    total = 0.0
    for pattern in all_patterns():
        row = dict(zip(QUESTION_ORDER, pattern))
        if all(row[q] == a for q, a in known.items()):
            total += pattern_probability(hypothesis, pattern)
    return total


def belief(history: tuple[tuple[str, int], ...]) -> dict[str, float]:
    """Exact deterministic Bayes update under the declared finite model."""
    values = {h: likelihood(h, history) / len(HYPOTHESES) for h in HYPOTHESES}
    normalizer = sum(values.values())
    return {h: values[h] / normalizer for h in HYPOTHESES}


def resolution(history: tuple[tuple[str, int], ...]) -> dict:
    posterior = belief(history)
    ordered = sorted(posterior.items(), key=lambda item: (-item[1], item[0]))
    supported = (
        ordered[0][1] >= RESOLUTION_POSTERIOR
        and ordered[0][1] - ordered[1][1] >= RESOLUTION_MARGIN
    )
    return {
        "supported": supported,
        "answer": ordered[0][0] if supported else None,
        "posterior": posterior,
        "top_probability": ordered[0][1],
        "margin": ordered[0][1] - ordered[1][1],
        "rule": f"top>={RESOLUTION_POSTERIOR} and margin>={RESOLUTION_MARGIN}",
    }


def evidence_trace(history: tuple[tuple[str, int], ...]) -> list[dict]:
    trace, prefix = [], ()
    previous = belief(prefix)
    for index, (question, answer) in enumerate(history, 1):
        prefix += ((question, answer),)
        current = belief(prefix)
        trace.append({
            "step": index,
            "question_id": question,
            "question": QUESTIONS[question],
            "answer": "yes" if answer else "no",
            "evidence": f"Observed answer to {question}; no hidden label or future answer was used.",
            "prior": previous,
            "posterior": current,
            "interpretation": {
                h: ("increased" if current[h] > previous[h] else "decreased" if current[h] < previous[h] else "unchanged")
                for h in HYPOTHESES
            },
        })
        previous = current
    return trace


@dataclass(frozen=True)
class HiddenEpisode:
    episode_id: str
    hypothesis: str
    answers: tuple[int, ...]
    split: str

    def evaluator_record(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "hypothesis": self.hypothesis,
            "answers": list(self.answers),
            "split": self.split,
        }


class DogQuestionAPI:
    """Lawful policy view. Hidden label, answer vector and future answers stay private."""

    __slots__ = ("__episode", "__history")

    def __init__(self, episode: HiddenEpisode):
        self.__episode = episode
        self.__history: tuple[tuple[str, int], ...] = ()

    def observe(self) -> dict:
        return {
            "opening": OPENING,
            "history": [
                {"question_id": q, "question": QUESTIONS[q], "answer": "yes" if a else "no"}
                for q, a in self.__history
            ],
            "belief": belief(self.__history),
            "resolution": resolution(self.__history),
            "legal_questions": self.legal_questions(),
            "remaining_budget": MAX_QUESTIONS - len(self.__history),
        }

    def legal_questions(self) -> tuple[str, ...]:
        asked = {q for q, _ in self.__history}
        if len(asked) >= MAX_QUESTIONS:
            return ()
        return tuple(q for q in QUESTION_ORDER if q not in asked)

    def ask(self, question: str) -> dict:
        if question not in self.legal_questions():
            raise ValueError("Question is unknown, repeated, or exceeds the question budget.")
        answer = self.__episode.answers[QUESTION_ORDER.index(question)]
        self.__history += ((question, answer),)
        return self.observe()["history"][-1]

    @property
    def history(self) -> tuple[tuple[str, int], ...]:
        return self.__history


def expected_information_gain(history: tuple[tuple[str, int], ...], question: str) -> float:
    if question not in QUESTIONS or question in {q for q, _ in history}:
        raise ValueError("Information gain requires a legal unasked question.")
    prior = belief(history)
    prior_entropy = -sum(p * math.log2(p) for p in prior.values() if p)
    weighted = 0.0
    for answer in (0, 1):
        extended = history + ((question, answer),)
        predictive = sum(
            prior[h] * likelihood(h, extended) / likelihood(h, history) for h in HYPOTHESES
        )
        posterior = belief(extended)
        weighted += predictive * -sum(p * math.log2(p) for p in posterior.values() if p)
    return prior_entropy - weighted


def feature_names(history: tuple[tuple[str, int], ...], *, memoryless: bool = False) -> tuple[str, ...]:
    _history_map(history)
    if not history:
        return ("bias", "depth=0")
    rows = history[-1:] if memoryless else history
    features = ["bias", f"depth={len(history)}"]
    features.extend(f"qa:{q}={a}" for q, a in rows)
    if not memoryless:
        ordered = sorted(rows)
        features.extend(
            f"pair:{q1}={a1}&{q2}={a2}"
            for (q1, a1), (q2, a2) in itertools.combinations(ordered, 2)
        )
    return tuple(features)


@dataclass(frozen=True)
class PreferenceModel:
    """Multiclass averaged-count preference model fitted only to trajectory labels."""

    version: str
    memoryless: bool
    weights: dict[str, dict[str, float]]
    feature_counts: dict[str, int]
    provenance: dict
    model_sha256: str

    def scores(self, history: tuple[tuple[str, int], ...], legal: tuple[str, ...]) -> dict[str, float] | None:
        features = feature_names(history, memoryless=self.memoryless)
        # Explicit abstention rather than a silent fallback for out-of-support features.
        if any(self.feature_counts.get(feature, 0) == 0 for feature in features):
            return None
        return {
            q: sum(self.weights.get(q, {}).get(feature, 0.0) for feature in features) / len(features)
            for q in legal
        }

    def choose(self, history: tuple[tuple[str, int], ...], legal: tuple[str, ...]) -> tuple[str | None, dict]:
        scores = self.scores(history, legal)
        if not legal:
            return None, {"status": "no_legal_question", "scores": {}}
        if scores is None:
            return None, {"status": "unsupported_history_abstention", "scores": {}}
        chosen = min(legal, key=lambda q: (-scores[q], QUESTION_ORDER.index(q)))
        return chosen, {"status": "learned_preference", "scores": scores}

    def document(self) -> dict:
        body = {
            "version": self.version,
            "memoryless": self.memoryless,
            "weights": self.weights,
            "feature_counts": self.feature_counts,
            "provenance": self.provenance,
        }
        return {**body, "model_sha256": digest(body)}


def fit_preference(examples: list[dict], provenance: dict, *, memoryless: bool = False) -> PreferenceModel:
    """Fit positive-rate preferences; labels may contain every tied optimal question."""
    if not examples or any(row.get("split") != "train" for row in examples):
        raise ValueError("Preference fitting requires nonempty training-only examples.")
    counts: dict[str, int] = {}
    totals: dict[str, dict[str, int]] = {q: {} for q in QUESTION_ORDER}
    positives: dict[str, dict[str, int]] = {q: {} for q in QUESTION_ORDER}
    for row in examples:
        history = tuple((q, int(a)) for q, a in row["history"])
        optimal = set(row["optimal_questions"])
        legal = tuple(row["legal_questions"])
        if not optimal or not optimal <= set(legal):
            raise ValueError("Training labels must be nonempty subsets of legal questions.")
        for feature in feature_names(history, memoryless=memoryless):
            counts[feature] = counts.get(feature, 0) + 1
            for question in legal:
                totals[question][feature] = totals[question].get(feature, 0) + 1
                positives[question][feature] = positives[question].get(feature, 0) + int(question in optimal)
    weights = {
        q: {
            feature: (positives[q].get(feature, 0) + 1) / (total + 2)
            for feature, total in sorted(totals[q].items())
        }
        for q in QUESTION_ORDER
    }
    body = {
        "version": "dog-history-preference-v1",
        "memoryless": memoryless,
        "weights": weights,
        "feature_counts": dict(sorted(counts.items())),
        "provenance": provenance,
    }
    document = {**body, "model_sha256": digest(body)}
    _validate_model_document(document)
    return PreferenceModel(**body, model_sha256=document["model_sha256"])


def save_model(model: PreferenceModel, path: Path) -> None:
    path = Path(path)
    document = model.document()
    _validate_model_document(document)
    if document["model_sha256"] != model.model_sha256:
        raise ValueError("In-memory model fingerprint mismatch.")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")


_MODEL_KEYS = {
    "version", "memoryless", "weights", "feature_counts", "provenance", "model_sha256",
}
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_EPISODE_ID = re.compile(r"[0-9a-f]{20}")


def _valid_feature(feature: object, memoryless: bool) -> bool:
    if feature == "bias":
        return True
    if not isinstance(feature, str):
        return False
    if feature.startswith("depth="):
        value = feature.removeprefix("depth=")
        return value in {str(depth) for depth in range(MAX_QUESTIONS)}
    if feature.startswith("qa:"):
        item = feature.removeprefix("qa:")
        return any(item == f"{question}={answer}" for question in QUESTION_ORDER for answer in (0, 1))
    if feature.startswith("pair:") and not memoryless:
        pair = feature.removeprefix("pair:").split("&")
        if len(pair) != 2:
            return False
        parsed = []
        for item in pair:
            matches = [
                (question, answer) for question in QUESTION_ORDER for answer in (0, 1)
                if item == f"{question}={answer}"
            ]
            if len(matches) != 1:
                return False
            parsed.append(matches[0])
        return parsed[0][0] < parsed[1][0]
    return False


def _valid_json_value(value: object) -> bool:
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_valid_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _valid_json_value(item) for key, item in value.items())
    return False


def _validate_model_document(document: object) -> None:
    """Reject self-consistently hashed documents outside the declared model schema."""
    if type(document) is not dict or set(document) != _MODEL_KEYS:
        raise ValueError("Invalid dog-history model schema: exact top-level keys are required.")
    if document["version"] != "dog-history-preference-v1":
        raise ValueError("Incompatible dog-history model version.")
    if type(document["memoryless"]) is not bool:
        raise ValueError("Invalid dog-history model schema: memoryless must be boolean.")
    fingerprint = document["model_sha256"]
    if type(fingerprint) is not str or _HEX_64.fullmatch(fingerprint) is None:
        raise ValueError("Invalid dog-history model fingerprint.")

    counts = document["feature_counts"]
    if type(counts) is not dict or "bias" not in counts:
        raise ValueError("Invalid dog-history feature-count schema.")
    expected_depths = {f"depth={depth}" for depth in range(MAX_QUESTIONS)}
    if not expected_depths <= set(counts):
        raise ValueError("Invalid dog-history feature-count schema: all decision depths are required.")
    for feature, count in counts.items():
        if not _valid_feature(feature, document["memoryless"]):
            raise ValueError(f"Invalid dog-history feature name: {feature!r}.")
        if type(count) is not int or count <= 0:
            raise ValueError("Invalid dog-history feature count: counts must be positive integers.")

    weights = document["weights"]
    if type(weights) is not dict or tuple(weights) != QUESTION_ORDER:
        # JSON written with sort_keys has lexical order, so compare sets rather than serialized order.
        if type(weights) is not dict or set(weights) != set(QUESTION_ORDER):
            raise ValueError("Invalid dog-history action schema.")
    for action in QUESTION_ORDER:
        action_weights = weights[action]
        if type(action_weights) is not dict:
            raise ValueError(f"Invalid dog-history weights for action {action}.")
        required = {"bias", *(f"depth={depth}" for depth in range(MAX_QUESTIONS))}
        if not required <= set(action_weights):
            raise ValueError(f"Invalid dog-history weight schema for action {action}.")
        for feature, weight in action_weights.items():
            if feature not in counts or not _valid_feature(feature, document["memoryless"]):
                raise ValueError(f"Invalid dog-history weight feature: {feature!r}.")
            if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("Invalid dog-history weight: values must be finite numbers in [0, 1].")

    provenance = document["provenance"]
    if type(provenance) is not dict or provenance.get("fit_split") != "train":
        raise ValueError("Invalid dog-history provenance: fit_split must be train.")
    official_keys = {
        "fit_split", "fit_episode_ids", "validation_use",
        "training_example_count", "preparation_sha256",
    }
    allowed_schemas = ({"fit_split"}, {"fit_split", "fixture"}, official_keys)
    if set(provenance) not in allowed_schemas:
        raise ValueError("Invalid dog-history provenance schema.")
    if "fixture" in provenance and provenance["fixture"] is not True:
        raise ValueError("Invalid dog-history fixture provenance.")
    if not _valid_json_value(provenance):
        raise ValueError("Invalid dog-history provenance: values must be finite JSON data.")
    if "training_example_count" in provenance:
        count = provenance["training_example_count"]
        if type(count) is not int or count <= 0 or count != counts["bias"]:
            raise ValueError("Invalid dog-history provenance training-example count.")
    if "fit_episode_ids" in provenance:
        ids = provenance["fit_episode_ids"]
        if (
            type(ids) is not list or not ids or ids != sorted(set(ids))
            or any(type(item) is not str or _EPISODE_ID.fullmatch(item) is None for item in ids)
        ):
            raise ValueError("Invalid dog-history provenance episode identities.")
    if "preparation_sha256" in provenance and (
        type(provenance["preparation_sha256"]) is not str
        or _HEX_64.fullmatch(provenance["preparation_sha256"]) is None
    ):
        raise ValueError("Invalid dog-history provenance preparation identity.")
    if "validation_use" in provenance and (
        type(provenance["validation_use"]) is not str or not provenance["validation_use"]
    ):
        raise ValueError("Invalid dog-history provenance validation declaration.")

    body = {key: value for key, value in document.items() if key != "model_sha256"}
    if digest(body) != fingerprint:
        raise ValueError("Invalid dog-history model fingerprint.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in dog-history model: {key}.")
        result[key] = value
    return result


def load_model(path: Path) -> PreferenceModel:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing dog-history model: {path}.")
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(
                stream, object_pairs_hook=_unique_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Non-finite JSON constant: {value}.")
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Corrupt dog-history model: {path}.") from exc
    _validate_model_document(document)
    fingerprint = document["model_sha256"]
    body = {key: value for key, value in document.items() if key != "model_sha256"}
    model = PreferenceModel(**body, model_sha256=fingerprint)
    if model.document() != document:
        raise ValueError("Invalid dog-history model: document does not round-trip canonically.")
    return model
