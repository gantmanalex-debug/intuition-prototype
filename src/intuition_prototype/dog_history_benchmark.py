"""Lifecycle, evaluation, reporting and CLI for the dog-history benchmark."""

from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from intuition_prototype import dog_history as task
from intuition_prototype import dog_history_protocol as protocol
from intuition_prototype.benchmark_protocol import file_digest, read_json, source_digest, write_json
from intuition_prototype.stage4_protocol import environment, source_fingerprints, source_paths


CLI_INTERPRETATION_NOTE = (
    'Interpretation: "supported" means only that the declared working-model posterior and margin '
    "thresholds were met; it is not a calibrated-correctness guarantee. "
    '"unsupported_confidence" counts answers emitted without that support, so zero does not mean '
    "zero wrong supported answers. reference_optimal_ties is evaluator-only hindsight unavailable "
    "to the policy."
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_relative(name: object) -> bool:
    if type(name) is not str or not name or "\\" in name:
        return False
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts and path.as_posix() == name


def _sealed(directory: Path, name: str) -> dict:
    digest_path = directory / f"{name}.sha256"
    _require((directory / name).is_file() and digest_path.is_file(), f"Missing sealed manifest: {name}.")
    _require(
        file_digest(directory / name) == digest_path.read_text(encoding="ascii").strip(),
        f"Manifest changed: {name}.",
    )
    return read_json(directory / name)


def _check_files(directory: Path, files: dict, *, label: str = "Artifact") -> None:
    _require(type(files) is dict, f"{label} inventory is invalid.")
    for name, expected in files.items():
        _require(_safe_relative(name), f"Unsafe {label.lower()} path: {name!r}.")
        path = directory / name
        _require(type(expected) is str and path.is_file(), f"Missing {label.lower()}: {name}.")
        _require(file_digest(path) == expected, f"{label} changed: {name}.")


def _source_archive_names(directory: Path) -> set[str]:
    archive = directory / "sources"
    _require(archive.is_dir() and not archive.is_symlink(), "Missing source archive.")
    result = set()
    for path in archive.rglob("*"):
        _require(not path.is_symlink(), "Source archive symlinks are not supported.")
        if path.is_file():
            result.add(path.relative_to(archive).as_posix())
    return result


def _episode(row: dict) -> task.HiddenEpisode:
    return task.HiddenEpisode(
        episode_id=row["episode_id"], hypothesis=row["hypothesis"],
        answers=tuple(row["answers"]), split=row["split"],
    )


def _answer(episode: task.HiddenEpisode, question: str) -> int:
    return episode.answers[task.QUESTION_ORDER.index(question)]


def _extend(episode: task.HiddenEpisode, history: tuple[tuple[str, int], ...], question: str):
    return history + ((question, _answer(episode, question)),)


@lru_cache(maxsize=None)
def shortest_distance(
    episode: task.HiddenEpisode, history: tuple[tuple[str, int], ...] = (),
) -> int | None:
    """Exhaustive episode-specific evidence depth; no policy receives this oracle."""
    if task.resolution(history)["supported"]:
        return 0
    if len(history) >= task.MAX_QUESTIONS:
        return None
    asked = {q for q, _ in history}
    distances = []
    for question in task.QUESTION_ORDER:
        if question not in asked:
            remaining = shortest_distance(episode, _extend(episode, history, question))
            if remaining is not None:
                distances.append(1 + remaining)
    return min(distances, default=None)


def optimal_questions(
    episode: task.HiddenEpisode, history: tuple[tuple[str, int], ...],
) -> tuple[str, ...]:
    target = shortest_distance(episode, history)
    if target in (None, 0):
        return ()
    asked = {q for q, _ in history}
    return tuple(
        question for question in task.QUESTION_ORDER
        if question not in asked
        and shortest_distance(episode, _extend(episode, history, question)) == target - 1
    )


def reference(episode: task.HiddenEpisode) -> dict:
    minimum = shortest_distance(episode)
    terminal_paths = []
    for depth in range(1, task.MAX_QUESTIONS + 1):
        for path in itertools.permutations(task.QUESTION_ORDER, depth):
            history = tuple((q, _answer(episode, q)) for q in path)
            if task.resolution(history)["supported"]:
                terminal_paths.append(list(path))
    shortest = [path for path in terminal_paths if len(path) == minimum] if minimum else []
    return {
        "episode_id": episode.episode_id,
        "minimum_questions": minimum,
        "requires_at_least_three_distinct_questions": minimum is not None and minimum >= 3,
        "shortest_paths": shortest,
        "opening_optimal_questions": list(optimal_questions(episode, ())),
        "search": {
            "all_distinct_prefixes_checked": sum(
                len(tuple(itertools.permutations(task.QUESTION_ORDER, depth)))
                for depth in range(task.MAX_QUESTIONS + 1)
            ),
            "budget": task.MAX_QUESTIONS,
            "resolution_rule": task.resolution(())["rule"],
        },
    }


def training_examples(rows: list[dict]) -> list[dict]:
    examples = []
    for row in rows:
        episode = _episode(row)
        for depth in range(task.MAX_QUESTIONS):
            for path in itertools.permutations(task.QUESTION_ORDER, depth):
                history = tuple((q, _answer(episode, q)) for q in path)
                if task.resolution(history)["supported"]:
                    continue
                optimal = optimal_questions(episode, history)
                if not optimal:
                    continue
                examples.append({
                    "episode_id": episode.episode_id,
                    "split": row["split"],
                    "history": [list(item) for item in history],
                    "legal_questions": [q for q in task.QUESTION_ORDER if q not in path],
                    "optimal_questions": list(optimal),
                })
    return examples


def _choose(policy: str, history, legal, model, memoryless_model):
    if policy == "fixed_order":
        question = next((q for q in task.QUESTION_ORDER if q in legal), None)
        return question, {"status": "fixed_nonadaptive", "scores": {}}
    if policy == "information_gain":
        scores = {q: task.expected_information_gain(history, q) for q in legal}
        question = min(legal, key=lambda q: (-scores[q], task.QUESTION_ORDER.index(q))) if legal else None
        return question, {"status": "nonlearned_information_gain", "scores": scores}
    selected = model if policy == "learned_history" else memoryless_model
    return selected.choose(history, legal)


def run_policy(episode: task.HiddenEpisode, policy: str, model, memoryless_model) -> dict:
    _require(policy in protocol.POLICIES, "Unknown policy.")
    api = task.DogQuestionAPI(episode)
    decisions, agreement_hits, agreement_total = [], 0, 0
    legal_actions, unsupported_confidence = True, 0
    stop_reason = "budget_exhausted"
    while True:
        state = api.observe()
        result = state["resolution"]
        if result["supported"]:
            stop_reason = "evidence_supported_answer"
            break
        legal = api.legal_questions()
        if not legal:
            break
        history = api.history
        optimal = optimal_questions(episode, history)
        question, detail = _choose(policy, history, legal, model, memoryless_model)
        decision = {
            "history_before": [list(item) for item in history],
            "legal_questions": list(legal),
            "chosen_question": question,
            "selection": detail,
            "reference_optimal_ties": list(optimal),
            "agrees_with_optimal_tie": question in optimal if question else False,
        }
        if optimal:
            agreement_total += 1
            agreement_hits += int(question in optimal)
        if question is None:
            decisions.append(decision)
            stop_reason = detail["status"]
            break
        if question not in legal:
            legal_actions = False
            stop_reason = "illegal_question"
            decisions.append(decision)
            break
        api.ask(question)
        decisions.append(decision)
    final = task.resolution(api.history)
    answer = final["answer"] if final["supported"] else None
    if answer is not None and not final["supported"]:
        unsupported_confidence += 1
    return {
        "episode_id": episode.episode_id,
        "split": episode.split,
        "policy": policy,
        "opening": task.OPENING,
        "history": [list(item) for item in api.history],
        "evidence_trace": task.evidence_trace(api.history),
        "decisions": decisions,
        "final": final,
        "answer": answer,
        "abstained": answer is None,
        "correct": answer == episode.hypothesis if answer else None,
        "question_cost": len(api.history),
        "budget_adherent": len(api.history) <= task.MAX_QUESTIONS,
        "no_repeats": len({q for q, _ in api.history}) == len(api.history),
        "legal": legal_actions,
        "unsupported_confidence": unsupported_confidence,
        "next_question_agreement_hits": agreement_hits,
        "next_question_agreement_total": agreement_total,
        "stop_reason": stop_reason,
    }


def evaluate_rows(rows: list[dict], model, memoryless_model) -> list[dict]:
    output = []
    for row in rows:
        episode = _episode(row)
        ref = reference(episode)
        output.append({
            "episode_id": episode.episode_id,
            "split": episode.split,
            "reference": ref,
            "policies": {
                policy: run_policy(episode, policy, model, memoryless_model)
                for policy in protocol.POLICIES
            },
        })
    return output


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for policy in protocol.POLICIES:
        episodes = [row["policies"][policy] for row in rows]
        count = len(episodes)
        agreements = sum(row["next_question_agreement_hits"] for row in episodes)
        opportunities = sum(row["next_question_agreement_total"] for row in episodes)
        supported = sum(row["answer"] is not None for row in episodes)
        correct = sum(row["answer"] is not None and row["correct"] for row in episodes)
        summary[policy] = {
            "episodes": count,
            "supported_resolution_rate": supported / count,
            "correct_supported_resolution_rate": correct / count,
            "conditional_correctness": correct / supported if supported else None,
            "mean_question_cost": sum(row["question_cost"] for row in episodes) / count,
            "unsupported_confidence_rate": sum(row["unsupported_confidence"] > 0 for row in episodes) / count,
            "abstention_rate": sum(row["abstained"] for row in episodes) / count,
            "legality_rate": sum(
                row["legal"] and row["budget_adherent"] and row["no_repeats"] for row in episodes
            ) / count,
            "next_question_agreement_rate": agreements / opportunities if opportunities else None,
            "agreement_opportunities": opportunities,
        }
    depths = {}
    for row in rows:
        key = str(row["reference"]["minimum_questions"])
        depths[key] = depths.get(key, 0) + 1
    return {
        "episode_count": len(rows),
        "policies": summary,
        "reference_minimum_depth_distribution": dict(sorted(depths.items())),
        "requires_at_least_three_count": sum(
            row["reference"]["requires_at_least_three_distinct_questions"] for row in rows
        ),
    }


def _write_lines(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _read_lines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def prepare(
    directory: Path, seed: int = protocol.GENERATOR_SEED, *, development_only: bool = False,
) -> dict:
    directory = Path(directory)
    _require(type(seed) is int and type(development_only) is bool, "Invalid preparation declaration.")
    partitions = protocol.generate_episodes(seed)
    protocol.validate_partitions(partitions)
    if development_only:
        partitions = {**partitions, "test": []}
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", protocol.PROTOCOL)
    for split, rows in partitions.items():
        write_json(directory / f"{split}.json", {"episodes": rows})
    archive = directory / "sources"
    archive.mkdir()
    fingerprints = source_fingerprints()
    archived = {}
    for name, source in source_paths().items():
        _require(_safe_relative(name), f"Unsafe executable source path: {name!r}.")
        _require(source.is_file() and not source.is_symlink(), f"Invalid executable source: {name}.")
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        archived[name] = file_digest(target)
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True,
    ).strip()
    manifest = {
        "version": protocol.VERSION,
        "development_only": development_only,
        "classification": "development fixture" if development_only else "official declaration",
        "generator_seed": seed,
        "counts": {split: len(rows) for split, rows in partitions.items()},
        "exact_answer_pattern_overlap": 0,
        "structural_overlap": protocol.PROTOCOL["generator"]["structural_overlap"],
        "source_sha256": fingerprints,
        "archived_source_file_sha256": archived,
        "environment": environment(),
        "git_head": git_head,
        "includes_uncommitted_sources": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2], text=True,
        ).strip()),
        "files": {
            name: file_digest(directory / name)
            for name in ("protocol.json", "train.json", "validation.json", "test.json")
        },
        "outcomes_executed": False,
    }
    write_json(directory / "preparation.json", manifest)
    (directory / "preparation.json.sha256").write_text(
        file_digest(directory / "preparation.json") + "\n", encoding="ascii", newline="\n",
    )
    check_prepared(directory)
    return manifest


def check_prepared(directory: Path, *, execution: bool = True) -> dict:
    directory = Path(directory)
    manifest = _sealed(directory, "preparation.json")
    _require(type(manifest.get("development_only", False)) is bool, "Invalid development-only declaration.")
    for inventory in ("source_sha256", "archived_source_file_sha256"):
        _require(type(manifest.get(inventory)) is dict, f"Invalid {inventory} inventory.")
        _require(all(_safe_relative(name) for name in manifest[inventory]), "Unsafe source archive path.")
    _require(set(manifest["archived_source_file_sha256"]) == set(manifest["source_sha256"]),
             "Archived source inventory differs.")
    _require(_source_archive_names(directory) == set(manifest["source_sha256"]),
             "Exact source archive inventory changed.")
    for name, expected in manifest["archived_source_file_sha256"].items():
        path = directory / "sources" / name
        _require(path.is_file() and file_digest(path) == expected, f"Archived source changed: {name}")
        _require(source_digest(path) == manifest["source_sha256"][name],
                f"Archived executable source digest changed: {name}")
    _require(set(manifest["files"]) == {"protocol.json", "train.json", "validation.json", "test.json"},
             "Prepared artifact inventory differs.")
    _check_files(directory, manifest["files"], label="Prepared artifact")
    declaration = read_json(directory / "protocol.json")
    _require(declaration["version"] == manifest["version"], "Frozen protocol version mismatch.")
    if execution:
        _require(manifest["version"] == protocol.VERSION, "Unsupported preparation.")
        _require(manifest["source_sha256"] == source_fingerprints(),
                "Executable source drift since preparation.")
        _require(manifest["environment"] == environment(), "Environment drift since preparation.")
        _require(declaration == protocol.PROTOCOL, "Protocol drift.")
        generated = protocol.generate_episodes(manifest["generator_seed"])
        protocol.validate_partitions(generated)
        if manifest.get("development_only", False):
            generated = {**generated, "test": []}
        for split, rows in generated.items():
            _require(read_json(directory / f"{split}.json") == {"episodes": rows}, "Partition drift.")
        _require(manifest["counts"] == {split: len(rows) for split, rows in generated.items()},
                "Partition counts differ.")
    return manifest


def train(directory: Path) -> dict:
    directory = Path(directory)
    preparation = check_prepared(directory, execution=True)
    _require(not (directory / "model.json").exists(), "Training already completed.")
    train_rows = read_json(directory / "train.json")["episodes"]
    validation_rows = read_json(directory / "validation.json")["episodes"]
    examples = training_examples(train_rows)
    provenance = {
        "fit_split": "train",
        "fit_episode_ids": sorted(row["episode_id"] for row in train_rows),
        "validation_use": "report-only; no hyperparameter selection or refit",
        "training_example_count": len(examples),
        "preparation_sha256": task.digest(preparation),
    }
    model = task.fit_preference(examples, provenance)
    memoryless = task.fit_preference(examples, provenance, memoryless=True)
    task.save_model(model, directory / "model.json")
    task.save_model(memoryless, directory / "memoryless-model.json")
    _require(not task.load_model(directory / "model.json").memoryless,
             "History model readback has the wrong model type.")
    _require(task.load_model(directory / "memoryless-model.json").memoryless,
             "Memoryless model readback has the wrong model type.")
    development = evaluate_rows(train_rows + validation_rows, model, memoryless)
    _write_lines(directory / "development-results.jsonl", development)
    training = {
        "version": protocol.VERSION,
        "development_only": preparation.get("development_only", False),
        "model_sha256": model.model_sha256,
        "memoryless_model_sha256": memoryless.model_sha256,
        "fit_episode_count": len(train_rows),
        "validation_episode_count": len(validation_rows),
        "training_example_count": len(examples),
        "development_summary": summarize(development),
        "no_test_outcomes_executed": True,
    }
    write_json(directory / "training.json", training)
    frozen_files = {
        name: file_digest(directory / name) for name in (
            "protocol.json", "train.json", "validation.json", "test.json",
            "preparation.json", "model.json", "memoryless-model.json",
            "development-results.jsonl", "training.json",
        )
    }
    freeze = {
        "version": protocol.VERSION,
        "classification": "frozen before single official untouched test",
        "source_sha256": source_fingerprints(),
        "environment": environment(),
        "model_sha256": model.model_sha256,
        "memoryless_model_sha256": memoryless.model_sha256,
        "files": frozen_files,
        "test_attempts_allowed": 1,
        "test_outcomes_inspected": False,
    }
    write_json(directory / "official-freeze.json", freeze)
    (directory / "official-freeze.json.sha256").write_text(
        file_digest(directory / "official-freeze.json") + "\n", encoding="ascii", newline="\n",
    )
    check_frozen(directory)
    return training


def check_frozen(directory: Path, *, execution: bool = True) -> dict:
    directory = Path(directory)
    preparation = check_prepared(directory, execution=execution)
    freeze = _sealed(directory, "official-freeze.json")
    required = {
        "protocol.json", "train.json", "validation.json", "test.json", "preparation.json",
        "model.json", "memoryless-model.json", "development-results.jsonl", "training.json",
    }
    _require(set(freeze["files"]) == required, "Frozen artifact inventory differs.")
    _check_files(directory, freeze["files"], label="Frozen artifact")
    _require(freeze["source_sha256"] == preparation["source_sha256"],
             "Freeze/preparation source identity mismatch.")
    if execution:
        _require(freeze["source_sha256"] == source_fingerprints(), "Source drift after official freeze.")
        _require(freeze["environment"] == environment(), "Environment drift after official freeze.")
    model = task.load_model(directory / "model.json")
    memoryless = task.load_model(directory / "memoryless-model.json")
    _require(not model.memoryless and model.model_sha256 == freeze["model_sha256"],
             "History model identity mismatch.")
    _require(memoryless.memoryless and memoryless.model_sha256 == freeze["memoryless_model_sha256"],
             "Memoryless model identity mismatch.")
    training = read_json(directory / "training.json")
    _require(
        training["model_sha256"] == freeze["model_sha256"]
        and training["memoryless_model_sha256"] == freeze["memoryless_model_sha256"]
        and training["no_test_outcomes_executed"] is True,
        "Training/freeze identity mismatch.",
    )
    return freeze


def _validate_stored_results(directory: Path, rows: list[dict]) -> None:
    """Validate stored identities and trace shape without rerunning any policy or reference."""
    declared = read_json(directory / "test.json")["episodes"]
    declaration = read_json(directory / "protocol.json")
    policies = set(declaration["policies"])
    questions = set(declaration["questions"])
    _require(len(rows) == len(declared), "Raw result count differs from the declared test partition.")
    _require(
        [row.get("episode_id") for row in rows] == [row.get("episode_id") for row in declared],
        "Raw result episode order or identity differs.",
    )

    result_keys = {
        "episode_id", "split", "policy", "opening", "history", "evidence_trace", "decisions",
        "final", "answer", "abstained", "correct", "question_cost", "budget_adherent",
        "no_repeats", "legal", "unsupported_confidence", "next_question_agreement_hits",
        "next_question_agreement_total", "stop_reason",
    }
    for stored, episode in zip(rows, declared):
        _require(set(stored) == {"episode_id", "split", "reference", "policies"},
                 "Invalid raw result schema.")
        _require(stored["episode_id"] == episode["episode_id"] and stored["split"] == "test",
                 "Raw result/declaration identity mismatch.")
        reference = stored["reference"]
        _require(
            type(reference) is dict
            and set(reference) == {
                "episode_id", "minimum_questions", "requires_at_least_three_distinct_questions",
                "shortest_paths", "opening_optimal_questions", "search",
            }
            and reference["episode_id"] == episode["episode_id"],
            "Invalid stored reference schema or identity.",
        )
        minimum = reference["minimum_questions"]
        _require(
            minimum is None or type(minimum) is int and 1 <= minimum <= declaration["budget"],
            "Invalid stored reference depth.",
        )
        _require(
            type(reference["requires_at_least_three_distinct_questions"]) is bool
            and reference["requires_at_least_three_distinct_questions"]
            is (minimum is not None and minimum >= 3)
            and type(reference["shortest_paths"]) is list
            and type(reference["opening_optimal_questions"]) is list
            and set(reference["opening_optimal_questions"]) <= questions,
            "Invalid stored reference declaration.",
        )
        for path in reference["shortest_paths"]:
            _require(
                type(path) is list and len(path) == minimum == len(set(path))
                and set(path) <= questions,
                "Invalid stored shortest reference path.",
            )
        _require(
            type(reference["search"]) is dict
            and set(reference["search"]) == {
                "all_distinct_prefixes_checked", "budget", "resolution_rule",
            }
            and reference["search"]["budget"] == declaration["budget"]
            and type(reference["search"]["all_distinct_prefixes_checked"]) is int
            and reference["search"]["all_distinct_prefixes_checked"] > 0
            and type(reference["search"]["resolution_rule"]) is str,
            "Invalid stored reference-search declaration.",
        )
        _require(set(stored["policies"]) == policies, "Stored policy inventory differs.")
        for policy, result in stored["policies"].items():
            _require(type(result) is dict and set(result) == result_keys, "Invalid policy-result schema.")
            _require(
                result["episode_id"] == episode["episode_id"]
                and result["split"] == "test" and result["policy"] == policy
                and result["opening"] == declaration["opening"],
                "Policy-result identity mismatch.",
            )
            history = result["history"]
            _require(type(history) is list and len(history) <= declaration["budget"],
                     "Invalid stored question history.")
            seen = set()
            for item in history:
                _require(
                    type(item) is list and len(item) == 2 and item[0] in questions
                    and type(item[1]) is int and item[1] in (0, 1) and item[0] not in seen,
                    "Invalid stored question history.",
                )
                seen.add(item[0])
            _require(
                type(result["question_cost"]) is int and result["question_cost"] == len(history)
                and type(result["evidence_trace"]) is list
                and len(result["evidence_trace"]) == len(history)
                and type(result["decisions"]) is list,
                "Stored trace lengths differ.",
            )
            for index, evidence in enumerate(result["evidence_trace"]):
                question, answer = history[index]
                _require(
                    type(evidence) is dict
                    and set(evidence) == {
                        "step", "question_id", "question", "answer", "evidence",
                        "prior", "posterior", "interpretation",
                    }
                    and evidence["step"] == index + 1
                    and evidence["question_id"] == question
                    and evidence["question"] == declaration["questions"][question]
                    and evidence["answer"] == ("yes" if answer else "no")
                    and evidence["evidence"]
                    == f"Observed answer to {question}; no hidden label or future answer was used."
                    and set(evidence["prior"]) == set(declaration["hypotheses"])
                    and set(evidence["posterior"]) == set(declaration["hypotheses"])
                    and set(evidence["interpretation"]) == set(declaration["hypotheses"])
                    and all(
                        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                        for values in (evidence["prior"], evidence["posterior"])
                        for value in values.values()
                    )
                    and set(evidence["interpretation"].values())
                    <= {"increased", "decreased", "unchanged"},
                    "Invalid stored evidence trace.",
                )
            prefix = []
            for decision in result["decisions"]:
                _require(
                    type(decision) is dict
                    and set(decision) == {
                        "history_before", "legal_questions", "chosen_question", "selection",
                        "reference_optimal_ties", "agrees_with_optimal_tie",
                    }
                    and decision["history_before"] == prefix
                    and type(decision["legal_questions"]) is list
                    and len(decision["legal_questions"]) == len(set(decision["legal_questions"]))
                    and set(decision["legal_questions"]) <= questions - {item[0] for item in prefix}
                    and type(decision["reference_optimal_ties"]) is list
                    and set(decision["reference_optimal_ties"]) <= set(decision["legal_questions"])
                    and type(decision["agrees_with_optimal_tie"]) is bool,
                    "Invalid stored decision trace.",
                )
                selection = decision["selection"]
                _require(
                    type(selection) is dict and set(selection) == {"status", "scores"}
                    and type(selection["status"]) is str and type(selection["scores"]) is dict
                    and all(
                        key in decision["legal_questions"]
                        and type(value) in (int, float) and math.isfinite(value)
                        for key, value in selection["scores"].items()
                    ),
                    "Invalid stored selection detail.",
                )
                chosen = decision["chosen_question"]
                _require(
                    chosen is None or chosen in decision["legal_questions"],
                    "Stored decision selected an illegal question.",
                )
                _require(
                    decision["agrees_with_optimal_tie"]
                    is (chosen in decision["reference_optimal_ties"] if chosen else False),
                    "Stored decision/reference agreement mismatch.",
                )
                if chosen is not None:
                    _require(len(prefix) < len(history) and history[len(prefix)][0] == chosen,
                             "Stored decision/history progression mismatch.")
                    prefix.append(history[len(prefix)])
            _require(prefix == history, "Stored decisions do not account for the final history.")
            final = result["final"]
            _require(
                type(final) is dict
                and set(final) == {
                    "supported", "answer", "posterior", "top_probability", "margin", "rule",
                }
                and type(final["supported"]) is bool
                and final["answer"] == result["answer"]
                and (final["answer"] is None) is (not final["supported"])
                and type(final["posterior"]) is dict
                and set(final["posterior"]) == set(declaration["hypotheses"])
                and all(
                    type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                    for value in final["posterior"].values()
                )
                and abs(sum(final["posterior"].values()) - 1.0) <= 1e-12
                and type(final["top_probability"]) in (int, float)
                and math.isfinite(final["top_probability"]) and 0 <= final["top_probability"] <= 1
                and type(final["margin"]) in (int, float)
                and math.isfinite(final["margin"]) and 0 <= final["margin"] <= 1
                and type(final["rule"]) is str,
                "Invalid stored final resolution.",
            )
            _require(
                all(type(result[key]) is bool for key in
                    ("abstained", "budget_adherent", "no_repeats", "legal")),
                "Invalid stored policy flags.",
            )
            _require(result["abstained"] is (result["answer"] is None),
                     "Stored abstention/answer mismatch.")
            _require(result["answer"] is None or result["answer"] in declaration["hypotheses"],
                     "Invalid stored answer.")
            expected_correct = (
                result["answer"] == episode["hypothesis"] if result["answer"] is not None else None
            )
            _require(result["correct"] == expected_correct, "Stored answer/latent-label mismatch.")
            _require(
                type(result["unsupported_confidence"]) is int
                and result["unsupported_confidence"] >= 0
                and type(result["next_question_agreement_hits"]) is int
                and type(result["next_question_agreement_total"]) is int
                and 0 <= result["next_question_agreement_hits"]
                <= result["next_question_agreement_total"],
                "Invalid stored policy counters.",
            )


def _completion_files(directory: Path) -> set[str]:
    preparation = _sealed(directory, "preparation.json")
    sources = {"sources/" + name for name in preparation["source_sha256"]}
    legacy = {
        "protocol.json", "train.json", "validation.json", "test.json",
        "preparation.json", "preparation.json.sha256", "model.json", "memoryless-model.json",
        "development-results.jsonl", "training.json", "official-freeze.json",
        "official-freeze.json.sha256", "official-test-started.json", "test-results.jsonl",
        "evaluation.json", "evaluation.json.sha256", "report.md",
    }
    if (directory / "official-test-complete.json").exists():
        legacy |= {
            "official-test-started.json.sha256", "official-test-complete.json",
            "official-test-complete.json.sha256",
        }
    return legacy | sources


def _verify_completed_transport(directory: Path) -> tuple[dict, dict, list[dict]]:
    """Verify immutable completed bytes without depending on current executable source."""
    directory = Path(directory)
    preparation = check_prepared(directory, execution=False)
    freeze = _sealed(directory, "official-freeze.json")
    _require(
        set(freeze["files"]) == {
            "protocol.json", "train.json", "validation.json", "test.json", "preparation.json",
            "model.json", "memoryless-model.json", "development-results.jsonl", "training.json",
        },
        "Frozen artifact inventory differs.",
    )
    _check_files(directory, freeze["files"], label="Frozen artifact")
    _require(freeze["source_sha256"] == preparation["source_sha256"],
             "Freeze/preparation source identity mismatch.")
    evaluation = _sealed(directory, "evaluation.json")
    _require((directory / "official-test-started.json").is_file(), "Official start marker is missing.")
    marker = read_json(directory / "official-test-started.json")
    _require(
        marker == {
            "freeze_sha256": task.digest(freeze),
            "classification": "single official untouched test attempt",
        },
        "Official start marker identity mismatch.",
    )
    if (directory / "official-test-started.json.sha256").exists():
        _require(
            file_digest(directory / "official-test-started.json")
            == (directory / "official-test-started.json.sha256").read_text(encoding="ascii").strip(),
            "Official start marker changed.",
        )
    _require(
        evaluation["freeze_sha256"] == task.digest(freeze)
        and evaluation["test_result_sha256"] == file_digest(directory / "test-results.jsonl")
        and evaluation["official_test_attempt"] == 1
        and evaluation["no_tuning_or_rerun"] is True,
        "Evaluation/freeze/raw identity mismatch.",
    )
    training = read_json(directory / "training.json")
    _require(
        training["model_sha256"] == freeze["model_sha256"]
        and training["memoryless_model_sha256"] == freeze["memoryless_model_sha256"],
        "Training/model identity mismatch.",
    )
    rows = _read_lines(directory / "test-results.jsonl")
    _validate_stored_results(directory, rows)
    _require((directory / "report.md").is_file(), "Stored report is missing.")
    if (directory / "official-test-complete.json").exists():
        complete = _sealed(directory, "official-test-complete.json")
        _require(
            complete["freeze_sha256"] == task.digest(freeze)
            and complete["model_sha256"] == freeze["model_sha256"]
            and complete["memoryless_model_sha256"] == freeze["memoryless_model_sha256"]
            and complete["test_result_sha256"] == evaluation["test_result_sha256"],
            "Official completion identity mismatch.",
        )
        _check_files(directory, complete["files"], label="Completed artifact")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*") if path.is_file()
    }
    _require(actual == _completion_files(directory), "Exact completed artifact inventory changed.")
    return freeze, evaluation, rows


def evaluate(directory: Path) -> dict:
    directory = Path(directory)
    freeze = check_frozen(directory, execution=True)
    preparation = read_json(directory / "preparation.json")
    _require(not preparation.get("development_only", False),
             "Development-only declarations cannot execute the test partition.")
    marker = directory / "official-test-started.json"
    outputs = (
        marker, directory / "official-test-started.json.sha256",
        directory / "test-results.jsonl", directory / "evaluation.json",
        directory / "evaluation.json.sha256", directory / "report.md",
        directory / "official-test-complete.json", directory / "official-test-complete.json.sha256",
    )
    _require(not any(path.exists() for path in outputs),
             "Official test is single-use and has already started.")
    write_json(marker, {
        "freeze_sha256": task.digest(freeze),
        "classification": "single official untouched test attempt",
    })
    (directory / "official-test-started.json.sha256").write_text(
        file_digest(marker) + "\n", encoding="ascii", newline="\n",
    )
    rows = read_json(directory / "test.json")["episodes"]
    model = task.load_model(directory / "model.json")
    memoryless = task.load_model(directory / "memoryless-model.json")
    results = evaluate_rows(rows, model, memoryless)
    _write_lines(directory / "test-results.jsonl", results)
    result = {
        "version": protocol.VERSION,
        "freeze_sha256": task.digest(freeze),
        "test_result_sha256": file_digest(directory / "test-results.jsonl"),
        "summary": summarize(results),
        "official_test_attempt": 1,
        "no_tuning_or_rerun": True,
    }
    write_json(directory / "evaluation.json", result)
    (directory / "evaluation.json.sha256").write_text(
        file_digest(directory / "evaluation.json") + "\n", encoding="ascii", newline="\n",
    )
    _validate_stored_results(directory, results)
    report_text = render_report(result, read_json(directory / "training.json"))
    (directory / "report.md").write_text(report_text, encoding="utf-8", newline="\n")
    completed_names = (
        "official-test-started.json", "official-test-started.json.sha256",
        "test-results.jsonl", "evaluation.json", "evaluation.json.sha256", "report.md",
    )
    completed = {
        "version": protocol.VERSION,
        "freeze_sha256": task.digest(freeze),
        "model_sha256": freeze["model_sha256"],
        "memoryless_model_sha256": freeze["memoryless_model_sha256"],
        "test_result_sha256": result["test_result_sha256"],
        "files": {name: file_digest(directory / name) for name in completed_names},
        "classification": "completed single official untouched test attempt",
    }
    write_json(directory / "official-test-complete.json", completed)
    (directory / "official-test-complete.json.sha256").write_text(
        file_digest(directory / "official-test-complete.json") + "\n",
        encoding="ascii", newline="\n",
    )
    return result


def render_report(evaluation: dict, training: dict) -> str:
    lines = [
        "# Dog-at-door history-aware question benchmark",
        "",
        "Synthetic finite question-learning toy; not a real-dog model, diagnosis, or treatment advice.",
        "",
        f"- Model: `{training['model_sha256']}`",
        f"- Memoryless model: `{training['memoryless_model_sha256']}`",
        f"- Official freeze: `{evaluation['freeze_sha256']}`",
        f"- Untouched test episodes: {evaluation['summary']['episode_count']}",
        f"- Certified test episodes requiring at least three distinct questions: "
        f"{evaluation['summary']['requires_at_least_three_count']}",
        "",
        "| policy | supported | correct supported | mean cost | abstain | unsupported confidence | legal | optimal-tie agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in protocol.POLICIES:
        row = evaluation["summary"]["policies"][policy]
        lines.append(
            f"| {policy} | {row['supported_resolution_rate']:.3f} | "
            f"{row['correct_supported_resolution_rate']:.3f} | {row['mean_question_cost']:.3f} | "
            f"{row['abstention_rate']:.3f} | {row['unsupported_confidence_rate']:.3f} | "
            f"{row['legality_rate']:.3f} | "
            f"{row['next_question_agreement_rate']:.3f} |"
        )
    lines += [
        "",
        "Exact complete answer-pattern overlap across train/validation/test is zero. Partial histories, "
        "questions, hypotheses, feature schema, and the finite likelihood mechanism intentionally overlap.",
        "Agreement is secondary: its episode-specific reference can use future answers that policies cannot see.",
        "No superiority was assumed; this one frozen result was not used to tune or rerun the learner.",
    ]
    return "\n".join(lines) + "\n"


def _report_verified(directory: Path) -> str:
    """Read-only replay from stored references/raw records; never executes a policy."""
    directory = Path(directory)
    _, evaluation, rows = _verify_completed_transport(directory)
    recomputed = summarize(rows)
    _require(recomputed == evaluation["summary"], "Evaluation summary does not replay.")
    text = render_report(evaluation, read_json(directory / "training.json"))
    path = directory / "report.md"
    _require(path.read_text(encoding="utf-8") == text, "Existing report differs.")
    return text


def report(directory: Path) -> str:
    """Verify and render a completed experiment under its exact archived code."""
    directory = Path(directory).resolve()
    _verify_completed_transport(directory)
    archived_source = directory / "sources" / "src"
    _require(archived_source.is_dir(), "Archived Python source is missing.")
    script = r"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from intuition_prototype import dog_history_benchmark as benchmark

directory = Path(sys.argv[2])
preparation = benchmark.read_json(directory / "preparation.json")
benchmark.source_paths = lambda: {
    name: directory / "sources" / name for name in preparation["source_sha256"]
}
benchmark.source_fingerprints = lambda: preparation["source_sha256"]
benchmark.environment = lambda: preparation["environment"]
if hasattr(benchmark, "_report_verified"):
    text = benchmark._report_verified(directory)
else:
    freeze = benchmark.check_frozen(directory)
    evaluation = benchmark.read_json(directory / "evaluation.json")
    benchmark._require(
        benchmark.file_digest(directory / "evaluation.json")
        == (directory / "evaluation.json.sha256").read_text(encoding="ascii").strip(),
        "Evaluation manifest changed.",
    )
    benchmark._require(
        evaluation["freeze_sha256"] == benchmark.task.digest(freeze),
        "Evaluation/freeze mismatch.",
    )
    benchmark._require(
        evaluation["test_result_sha256"]
        == benchmark.file_digest(directory / "test-results.jsonl"),
        "Raw test results changed.",
    )
    rows = benchmark._read_lines(directory / "test-results.jsonl")
    benchmark._require(
        benchmark.summarize(rows) == evaluation["summary"],
        "Evaluation summary does not replay.",
    )
    text = benchmark.render_report(
        evaluation, benchmark.read_json(directory / "training.json")
    )
    benchmark._require(
        (directory / "report.md").read_text(encoding="utf-8") == text,
        "Existing report differs.",
    )
print(json.dumps(text, allow_nan=False))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(archived_source), str(directory)],
        capture_output=True, text=True, check=False,
    )
    _require(completed.returncode == 0, "Archived report verification failed:\n" + completed.stderr)
    return json.loads(completed.stdout)


def _decorate_demo(
    result: dict, model: task.PreferenceModel, memoryless: task.PreferenceModel, *,
    classification: str, execution_source: str,
) -> dict:
    selected = {
        "learned_history": model.model_sha256,
        "memoryless_learned": memoryless.model_sha256,
    }.get(result["policy"])
    return {
        **result,
        "classification": classification,
        "execution_source": execution_source,
        "model_artifacts": {
            "history_model_sha256": model.model_sha256,
            "memoryless_model_sha256": memoryless.model_sha256,
            "selected_ranker_sha256": selected,
            "fit_provenance": model.provenance,
            "shared_fit_provenance": model.provenance == memoryless.provenance,
        },
        "interpretation": {
            "support": (
                "Declared working-model threshold compliance only; not calibrated correctness."
            ),
            "unsupported_confidence": (
                "Counts answers emitted without declared support; zero does not mean zero wrong answers."
            ),
            "reference_optimal_ties": (
                "Evaluator-only hindsight using future answers; unavailable to the policy."
            ),
        },
    }


def _live_demo(directory: Path, pattern: str, policy_name: str) -> dict:
    """Development demo under current source; execution remains a strict live-source operation."""
    freeze = check_frozen(directory, execution=True)
    model = task.load_model(directory / "model.json")
    memoryless = task.load_model(directory / "memoryless-model.json")
    _require(model.model_sha256 == freeze["model_sha256"]
             and memoryless.model_sha256 == freeze["memoryless_model_sha256"],
             "Demo model identity mismatch.")
    answers = tuple(map(int, pattern))
    posterior = task.belief(tuple(zip(task.QUESTION_ORDER, answers)))
    hypothesis = max(task.HYPOTHESES, key=lambda h: posterior[h])
    episode = task.HiddenEpisode("interactive-demo", hypothesis, answers, "demo")
    result = run_policy(episode, policy_name, model, memoryless)
    return _decorate_demo(
        result, model, memoryless,
        classification="development demonstration; not an official test result",
        execution_source="current source strictly matched to this development freeze",
    )


def _historical_demo(directory: Path, pattern: str, policy_name: str) -> dict:
    """Run a demonstration under verified archived code; never touches test outcomes."""
    freeze, _, _ = _verify_completed_transport(directory)
    archived_source = directory / "sources" / "src"
    _require(archived_source.is_dir(), "Archived Python source is missing.")
    script = r"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from intuition_prototype import dog_history_benchmark as benchmark

directory = Path(sys.argv[2])
preparation = benchmark.read_json(directory / "preparation.json")
benchmark.source_paths = lambda: {
    name: directory / "sources" / name for name in preparation["source_sha256"]
}
benchmark.source_fingerprints = lambda: preparation["source_sha256"]
benchmark.environment = lambda: preparation["environment"]
run_demo = getattr(benchmark, "_live_demo", None)
if run_demo is None:
    run_demo = benchmark.demo
print(json.dumps(run_demo(directory, sys.argv[3], sys.argv[4]), allow_nan=False))
"""
    completed = subprocess.run(
        [
            sys.executable, "-I", "-B", "-c", script, str(archived_source),
            str(directory), pattern, policy_name,
        ],
        capture_output=True, text=True, check=False,
    )
    _require(completed.returncode == 0, "Archived demo execution failed:\n" + completed.stderr)
    result = json.loads(completed.stdout)
    model = task.load_model(directory / "model.json")
    memoryless = task.load_model(directory / "memoryless-model.json")
    _require(
        model.model_sha256 == freeze["model_sha256"]
        and memoryless.model_sha256 == freeze["memoryless_model_sha256"],
        "Historical demo model identity mismatch.",
    )
    return _decorate_demo(
        result, model, memoryless,
        classification=(
            "verified archived-code historical demonstration; not an official test result "
            "and not a new performance result"
        ),
        execution_source="frozen archived v1 source",
    )


def demo(directory: Path, pattern: str, policy_name: str) -> dict:
    _require(len(pattern) == len(task.QUESTION_ORDER) and set(pattern) <= {"0", "1"},
             "Demo pattern must contain six binary digits.")
    directory = Path(directory).resolve()
    _require(policy_name in protocol.POLICIES, "Unknown policy.")
    if (directory / "evaluation.json").is_file():
        return _historical_demo(directory, pattern, policy_name)
    return _live_demo(directory, pattern, policy_name)


def _observed_prefix(values: list[str]) -> tuple[tuple[str, int], ...]:
    _require(values, "At least one manually observed question/answer is required.")
    history = []
    answers = {"0": 0, "no": 0, "1": 1, "yes": 1}
    for value in values:
        question, separator, answer = value.partition("=")
        normalized = answer.lower()
        _require(
            separator and question in task.QUESTIONS and normalized in answers,
            "Observed answers must use QUESTION=0|1|no|yes with a known question.",
        )
        history.append((question, answers[normalized]))
    result = tuple(history)
    _require(len(result) <= task.MAX_QUESTIONS, "Observed prefix exceeds the question budget.")
    task.feature_names(result)  # Validates distinct lawful question identifiers and binary answers.
    return result


def continuation_demo(directory: Path, history: tuple[tuple[str, int], ...]) -> dict:
    """Continue a manually supplied lawful prefix using the exact archived fitted model.

    This is deliberately not an autonomous policy trajectory: in particular, its
    first supplied question must not be described as the model's chosen starter.
    """
    directory = Path(directory).resolve()
    _require(history, "A manually supplied observed prefix is required.")
    task.feature_names(history)
    _require(len(history) <= task.MAX_QUESTIONS, "Observed prefix exceeds the question budget.")
    freeze = check_frozen(directory, execution=False)
    archived_source = directory / "sources" / "src"
    _require(archived_source.is_dir(), "Archived Python source is missing.")
    script = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
from intuition_prototype import dog_history as task

history = tuple((question, int(answer)) for question, answer in json.loads(sys.argv[3]))
task.feature_names(history)
legal = tuple(question for question in task.QUESTION_ORDER if question not in {q for q, _ in history})
model = task.load_model(sys.argv[2])
question, detail = model.choose(history, legal)
print(json.dumps({
    "model_sha256": model.model_sha256,
    "model_provenance": model.provenance,
    "legal_questions": legal,
    "next_question": question,
    "selection": detail,
}, allow_nan=False))
"""
    completed = subprocess.run(
        [
            sys.executable, "-I", "-B", "-c", script, str(archived_source),
            str(directory / "model.json"), json.dumps(history),
        ],
        capture_output=True, text=True, check=False,
    )
    _require(completed.returncode == 0, "Archived continuation failed:\n" + completed.stderr)
    result = json.loads(completed.stdout)
    _require(result["model_sha256"] == freeze["model_sha256"],
             "Continuation model identity mismatch.")
    question = result["next_question"]
    return {
        "classification": "manual observed-prefix continuation; not a benchmark policy result",
        "starter_source": "manually supplied; not selected by the learned model",
        "history": [
            {
                "question_id": question_id,
                "question": task.QUESTIONS[question_id],
                "answer": "yes" if answer else "no",
            }
            for question_id, answer in history
        ],
        **result,
        "next_question_text": task.QUESTIONS[question] if question else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("directory", type=Path)
    prepare_parser.add_argument("--development-only", action="store_true")
    for name in ("train", "evaluate", "report"):
        command = sub.add_parser(name)
        command.add_argument("directory", type=Path)
    demo_parser = sub.add_parser("demo")
    demo_parser.add_argument("directory", type=Path)
    demo_parser.add_argument("--pattern", default="100100")
    demo_parser.add_argument("--policy", choices=protocol.POLICIES, default="learned_history")
    continuation_parser = sub.add_parser("continue")
    continuation_parser.add_argument("directory", type=Path)
    continuation_parser.add_argument(
        "--observed", action="append", required=True, metavar="QUESTION=ANSWER",
        help="Manually supplied lawful prefix, in order; ANSWER is 0/1/no/yes.",
    )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        output = prepare(args.directory, development_only=args.development_only)
    elif args.command == "train":
        output = train(args.directory)
    elif args.command == "evaluate":
        output = evaluate(args.directory)
    elif args.command == "report":
        print(CLI_INTERPRETATION_NOTE, file=sys.stderr)
        print(report(args.directory), end="")
        return
    elif args.command == "continue":
        output = continuation_demo(args.directory, _observed_prefix(args.observed))
    else:
        print(CLI_INTERPRETATION_NOTE, file=sys.stderr)
        output = demo(args.directory, args.pattern, args.policy)
    print(json.dumps(output, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
