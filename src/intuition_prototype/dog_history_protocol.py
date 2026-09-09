"""Preregistered construction for the history-aware dog-at-door toy benchmark."""

from __future__ import annotations

import random

from intuition_prototype import dog_history as task
from intuition_prototype.benchmark_protocol import digest


VERSION = "dog-history-v1"
GENERATOR_SEED = 2026090901
COUNTS = {"train": 28, "validation": 12, "test": 20}
POLICIES = ("fixed_order", "memoryless_learned", "learned_history", "information_gain")
PROTOCOL = {
    "version": VERSION,
    "task": "Synthetic history-conditioned next-question selection after each question/answer.",
    "safety_scope": (
        "Illustrative finite hypotheses only; not a real-dog behavioral model, medical diagnosis, "
        "or treatment recommendation."
    ),
    "opening": task.OPENING,
    "hypotheses": list(task.HYPOTHESES),
    "questions": task.QUESTIONS,
    "question_semantics": (
        "Each answer is a fixed binary episode attribute. The public joint likelihood assigns positive "
        "mass to all 64 answer vectors and includes declared pair interactions. The policy API reveals "
        "only answers to questions actually asked."
    ),
    "prior": "uniform over four hypotheses",
    "belief_update": "Exact marginal Bayes update over the public finite joint likelihood; never learned.",
    "resolution_rule": {
        "posterior": task.RESOLUTION_POSTERIOR,
        "margin": task.RESOLUTION_MARGIN,
        "answer": "Emit the posterior MAP only if both thresholds hold; otherwise abstain or ask.",
    },
    "goal": (
        "Reach an evidence-supported answer within budget. Learned ranking targets questions lying on "
        "a shortest eventual supported-resolution path for each development trajectory."
    ),
    "budget": task.MAX_QUESTIONS,
    "repeats": "Questions are distinct; repeats are illegal and cannot spend budget.",
    "reference_search": (
        "Evaluator exhaustively explores every legal distinct-question continuation against a fixed "
        "episode answer vector. Minimum depth certifies evidence ambiguity, not guessing impossibility. "
        "All next questions that attain the minimum are ties."
    ),
    "generator": {
        "seed": GENERATOR_SEED,
        "partition_counts": COUNTS,
        "unit": "one complete six-answer pattern assigned one conditional latent hypothesis",
        "pattern_partition": (
            "All 64 binary patterns are deterministically shuffled once; disjoint slices form train, "
            "validation and untouched test, leaving four unused. Exact full answer-pattern overlap is zero."
        ),
        "structural_overlap": (
            "Question semantics, hypotheses, feature schema, opening, and some partial histories overlap. "
            "This is finite-task compositional pattern holdout, not new-domain or mechanism generalization."
        ),
    },
    "fit": {
        "split": "train only",
        "learner": (
            "Add-one-smoothed per-action positive rates over bias/depth, question-answer and pair features; "
            "history model uses all prior answers, memoryless ablation uses latest answer only."
        ),
        "validation": (
            "Reports development behavior only. There are no hyperparameters or validation-selected choices, "
            "and no refit."
        ),
        "unsupported": "Any active feature absent from fitting causes explicit learned-policy abstention.",
    },
    "policies": {
        "fixed_order": list(task.QUESTION_ORDER),
        "memoryless_learned": "same fit and stop rule, but ranking cannot use earlier answers",
        "learned_history": "development-only fitted ranking plus nonlearned deterministic stop rule",
        "information_gain": "labeled nonlearned public-model expected entropy reduction",
    },
    "endpoints": {
        "primary": [
            "supported_resolution_rate",
            "correct_supported_resolution_rate",
            "mean_question_cost",
            "unsupported_confidence_rate",
            "abstention_rate",
            "legality_rate",
        ],
        "secondary": (
            "Next-question agreement with any episode-specific optimal tie. Agreement is not the primary "
            "outcome and the reference uses future answers unavailable to policies."
        ),
    },
    "test_use": (
        "Prepare, fit and inspect development, then freeze model/preprocessing/generator/evaluator/live sources. "
        "Execute the untouched test exactly once; no outcome filtering, tuning or rerun."
    ),
    "claims": (
        "No learned advantage is promised. Results describe this small known finite model only and do not "
        "validate intuition, real-world animal inference, or general question generation."
    ),
}


def generate_episodes(seed: int = GENERATOR_SEED) -> dict[str, list[dict]]:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Generator seed must be an integer in [0, 2**63).")
    patterns = list(task.all_patterns())
    random.Random(seed).shuffle(patterns)
    result, offset = {}, 0
    for split, count in COUNTS.items():
        entries = []
        for pattern in patterns[offset:offset + count]:
            posterior = task.belief(tuple(zip(task.QUESTION_ORDER, pattern)))
            # Conditional latent-label draw with an independent stable stream.
            rng = random.Random(int(digest({"seed": seed, "pattern": pattern})[:16], 16))
            needle, running, hypothesis = rng.random(), 0.0, task.HYPOTHESES[-1]
            for candidate in task.HYPOTHESES:
                running += posterior[candidate]
                if needle <= running:
                    hypothesis = candidate
                    break
            entries.append({
                "episode_id": digest({"split": split, "pattern": pattern, "hypothesis": hypothesis})[:20],
                "split": split,
                "hypothesis": hypothesis,
                "answers": list(pattern),
            })
        result[split] = entries
        offset += count
    return result


def validate_partitions(partitions: dict[str, list[dict]]) -> None:
    if {split: len(rows) for split, rows in partitions.items()} != COUNTS:
        raise ValueError("Partition count mismatch.")
    seen_patterns, seen_ids = set(), set()
    for split in COUNTS:
        for row in partitions[split]:
            pattern = tuple(row["answers"])
            if row["split"] != split or pattern not in task.all_patterns():
                raise ValueError("Invalid generated episode.")
            if pattern in seen_patterns or row["episode_id"] in seen_ids:
                raise ValueError("Answer patterns and episode IDs must be split-disjoint.")
            if row["hypothesis"] not in task.HYPOTHESES:
                raise ValueError("Invalid hidden hypothesis.")
            seen_patterns.add(pattern)
            seen_ids.add(row["episode_id"])
