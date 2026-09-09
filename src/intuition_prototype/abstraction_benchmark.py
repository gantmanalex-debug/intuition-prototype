"""Lifecycle and evaluation for context-first abstraction inquiry."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from intuition_prototype import abstraction_inquiry as task
from intuition_prototype import abstraction_protocol as protocol
from intuition_prototype.benchmark_protocol import (
    digest, file_digest, read_json, source_digest, write_json,
)
from intuition_prototype.stage4_protocol import environment


OFFICIAL_AUTHORIZATION = "PARENT-APPROVED-SINGLE-OFFICIAL-TEST"
FREEZE_AUTHORIZATION = "PARENT-APPROVED-DEVELOPMENT-FREEZE"
LABEL_RULE = (
    "maximize eventual scoped goal resolution; then immediate scoped projection reduction, "
    "broad-group reduction, fine-leaf reduction, and shorter remaining distance"
)
_TRAINING_EXAMPLE_CACHE: dict[str, list[dict]] = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _episode(row: dict) -> task.HiddenEpisode:
    return task.HiddenEpisode(
        episode_id=row["episode_id"],
        physical_world_id=row["physical_world_id"],
        domain=row["domain"],
        group=row["group"],
        leaf=row["leaf"],
        answers=tuple(row["answers"]),
        split=row["split"],
        case_type=row["case_type"],
        context_variant=row["context_variant"],
        context_explicit=row["context_explicit"],
        context_answer=row["context_answer"],
        goal=row["goal"],
        concern=row["concern"],
    )


def _supplied(episode: task.HiddenEpisode) -> str | None:
    return episode.context_answer if episode.context_explicit else None


def _answer(episode: task.HiddenEpisode, question: str) -> str:
    context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[episode.domain]
    physical = protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[episode.domain]
    return (
        episode.context_answer
        if question == context_question
        else episode.answers[physical.index(question)]
    )


def _extend(episode: task.HiddenEpisode, history, question: str):
    return tuple(history) + ((question, _answer(episode, question)),)


def _projection_size(history, supplied_context, domain) -> int:
    state = task.inquiry_state(history, supplied_context, domain)
    context, world = state["context"], state["world"]
    if context["status"] != "known":
        return len(protocol.CONTEXTS)
    if context["goal"] in ("explain_pattern", "recommend_representation"):
        return world["leaf_count"]
    if context["goal"] in ("characterize_change", "characterize_operations"):
        return world["group_count"]
    positive = {
        "assess_outside": {"outside_directed"},
        "assess_linked_fit": {"stable_identity", "splice_mutation"},
        "assess_indexing": {"indexed_collection"},
    }.get(context["goal"])
    if positive is not None:
        return len({
            group in positive for group in world["candidate_groups"]
        })
    return 0


@lru_cache(maxsize=None)
def shortest_distance(episode: task.HiddenEpisode, history=()) -> int | None:
    """Evaluator-only future-answer depth to a scoped answer."""
    supplied = _supplied(episode)
    state = task.inquiry_state(history, supplied, episode.domain)
    if state["goal_resolution"]["supported"]:
        return 0
    if (
        state["context"]["status"] in ("ambiguous", "inconsistent", "out_of_model")
        or state["world"]["status"] in ("inconsistent", "out_of_model")
    ):
        return None
    distances = []
    for question in task.legal_questions(history, supplied, episode.domain):
        remaining = shortest_distance(episode, _extend(episode, history, question))
        if remaining is not None:
            distances.append(1 + remaining)
    return min(distances, default=None)


def trajectory_scores(episode: task.HiddenEpisode, history) -> dict[str, float]:
    """Development-only composite labels using evaluator future answers."""
    supplied = _supplied(episode)
    before = task.inquiry_state(history, supplied, episode.domain)
    legal = task.legal_questions(history, supplied, episode.domain)
    if not legal:
        return {}
    scores = {}
    before_projection = _projection_size(history, supplied, episode.domain)
    for question in legal:
        extended = _extend(episode, history, question)
        after = task.inquiry_state(extended, supplied, episode.domain)
        distance = shortest_distance(episode, extended)
        reachable = distance is not None
        projection_reduction = before_projection - _projection_size(
            extended, supplied, episode.domain,
        )
        scores[question] = (
            100.0 * int(reachable)
            + 30.0 * int(after["goal_resolution"]["supported"])
            + 12.0 * projection_reduction
            + 3.0 * (before["world"]["group_count"] - after["world"]["group_count"])
            + 1.0 * (before["world"]["leaf_count"] - after["world"]["leaf_count"])
            - 2.0 * (1 + distance if distance is not None else protocol.MAX_TOTAL_QUESTIONS + 2)
        )
    return scores


def optimal_questions(episode: task.HiddenEpisode, history) -> tuple[str, ...]:
    scores = trajectory_scores(episode, tuple(history))
    if not scores:
        return ()
    best = max(scores.values())
    return tuple(question for question in task.QUESTION_ORDER if scores.get(question) == best)


def _reachable_histories(episode: task.HiddenEpisode):
    """Every legal observed prefix under the episode, including opening."""
    supplied = _supplied(episode)
    output = []

    def visit(history):
        output.append(history)
        for question in task.legal_questions(history, supplied, episode.domain):
            visit(_extend(episode, history, question))

    visit(())
    return output


def reference(episode: task.HiddenEpisode) -> dict:
    histories = _reachable_histories(episode)
    terminal = [
        [question for question, _ in history]
        for history in histories
        if task.goal_resolution(
            history, _supplied(episode), episode.domain,
        )["supported"]
    ]
    minimum = min(map(len, terminal), default=None)
    shortest = [path for path in terminal if len(path) == minimum] if minimum is not None else []
    return {
        "episode_id": episode.episode_id,
        "physical_world_id": episode.physical_world_id,
        "minimum_questions": minimum,
        "requires_at_least_three_distinct_questions": minimum is not None and minimum >= 3,
        "shortest_paths": shortest,
        "opening_optimal_questions": list(optimal_questions(episode, ())),
        "search": {
            "all_legal_prefixes_checked": len(histories),
            "physical_budget": protocol.MAX_PHYSICAL_QUESTIONS,
            "total_budget": (
                protocol.MAX_PHYSICAL_QUESTIONS
                if episode.context_explicit else protocol.MAX_TOTAL_QUESTIONS
            ),
            "shortcuts_rejected": all(
                not task.goal_resolution(
                    history, _supplied(episode), episode.domain,
                )["supported"]
                for history in histories if minimum is not None and len(history) < minimum
            ),
        },
        "claim_scope": "scoped resolution in this bounded API, not real-world diagnosis or guessing",
    }


def training_examples(rows: list[dict]) -> list[dict]:
    cache_key = digest(rows)
    if cache_key in _TRAINING_EXAMPLE_CACHE:
        return _TRAINING_EXAMPLE_CACHE[cache_key]
    examples = []
    for row in rows:
        _require(row["split"] == "train", "Training examples require train rows only.")
        episode = _episode(row)
        supplied = _supplied(episode)
        demonstrations = []

        def visit(history):
            demonstrations.append(history)
            for question in optimal_questions(episode, history):
                visit(_extend(episode, history, question))

        visit(())
        # Development-only history certificate: the latest answer and legal set
        # are held fixed across worlds while an earlier broad answer differs.
        # This supplies a real fitted comparison for the history model and its
        # latest-answer ablation; it is not substituted for policy behavior.
        prefix = ()
        if not episode.context_explicit:
            context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[episode.domain]
            if context_question in task.legal_questions(prefix, supplied, episode.domain):
                prefix = _extend(episode, prefix, context_question)
        certificate_questions = (
            ("dog_active_entry_pattern", "dog_upper_sequence_repeat")
            if episode.domain == "dog"
            else ("linked_identity_or_splice", "linked_upper_operation_repeat")
        )
        for question in certificate_questions:
            if question not in task.legal_questions(prefix, supplied, episode.domain):
                break
            prefix = _extend(episode, prefix, question)
        if len(prefix) >= (3 if not episode.context_explicit else 2):
            demonstrations.append(prefix)
        # Also label every state reached by the declared adaptive public tree.
        # This prevents an oracle-only training path from being presented as
        # actual learner behavior when aggregated fitted choices take another
        # tied branch.
        prefix = ()
        if not episode.context_explicit:
            context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[episode.domain]
            if context_question in task.legal_questions(prefix, supplied, episode.domain):
                prefix = _extend(episode, prefix, context_question)
                demonstrations.append(prefix)
        first = (
            "dog_active_entry_pattern"
            if episode.domain == "dog" else "linked_identity_or_splice"
        )
        if first in task.legal_questions(prefix, supplied, episode.domain):
            prefix = _extend(episode, prefix, first)
            demonstrations.append(prefix)
            first_answer = prefix[-1][1]
            second = (
                ("dog_outdoor_cue_response" if first_answer == "yes" else "dog_person_timing_link")
                if episode.domain == "dog"
                else ("linked_stable_handles" if first_answer == "yes" else "linked_fast_index")
            )
            route = (
                second,
                "dog_upper_sequence_repeat" if episode.domain == "dog" else "linked_upper_operation_repeat",
                "dog_lower_sequence_repeat" if episode.domain == "dog" else "linked_lower_operation_repeat",
            )
            for question in route:
                if question not in task.legal_questions(prefix, supplied, episode.domain):
                    break
                prefix = _extend(episode, prefix, question)
                demonstrations.append(prefix)
        demonstrations = list(dict.fromkeys(demonstrations))
        for history in demonstrations:
            legal = task.legal_questions(history, supplied, episode.domain)
            scores = trajectory_scores(episode, history)
            optimal = optimal_questions(episode, history)
            if not legal or not optimal:
                continue
            examples.append({
                "episode_id": episode.episode_id,
                "physical_world_id": episode.physical_world_id,
                "domain": episode.domain,
                "split": "train",
                "history": [list(item) for item in history],
                "supplied_context": supplied,
                "legal_questions": list(legal),
                "optimal_questions": list(optimal),
                "trajectory_scores": scores,
            })
    _TRAINING_EXAMPLE_CACHE[cache_key] = examples
    return examples


def _choose(policy, episode, history, legal, model, memoryless):
    supplied = _supplied(episode)
    if policy == "fixed_order":
        question = next((q for q in task.QUESTION_ORDER if q in legal), None)
        return question, {"status": "fixed_nonadaptive", "scores": {}}
    if policy == "information_gain":
        context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[episode.domain]
        if legal == (context_question,):
            return context_question, {
                "status": "nonlearned_required_context_clarification", "scores": {},
            }
        scores = {
            question: task.expected_information_gain(
                history, question, supplied, episode.domain,
            )["score"]
            for question in legal
        }
        question = min(legal, key=lambda q: (-scores[q], task.QUESTION_ORDER.index(q)))
        return question, {
            "status": "nonlearned_scoped_uniform_candidate_information",
            "scores": scores,
            "score_interpretation": "uniform candidate counts, not learned probability",
        }
    if policy == "reference":
        optimal = optimal_questions(episode, history)
        return (
            optimal[0] if optimal else None,
            {"status": "evaluator_only_future_answer_reference", "scores": {}},
        )
    selected = model if policy == "learned_history" else memoryless
    return selected.choose(history, legal, supplied, episode.domain)


def _goal_correct(episode: task.HiddenEpisode, resolution: dict) -> bool | None:
    if resolution["supported"] and resolution["goal"] == "honor_required_structure":
        return (
            episode.context_answer == "linked_required_structure"
            and resolution["value"] == "linked_list_required"
        )
    if not resolution["supported"] or episode.case_type in ("inconsistent", "out_of_model"):
        return None
    expected = {
        "explain_pattern": episode.leaf,
        "assess_outside": episode.group == "outside_directed",
        "characterize_change": episode.group,
        "recommend_representation": episode.leaf,
        "assess_linked_fit": episode.group in ("stable_identity", "splice_mutation"),
        "assess_indexing": episode.group == "indexed_collection",
        "characterize_operations": episode.group,
    }[episode.goal]
    return resolution["value"] == expected


def run_policy(episode, policy, model, memoryless) -> dict:
    _require(policy in protocol.POLICIES, "Unknown policy.")
    api = task.AbstractionQuestionAPI(episode)
    agent_trace, annotations = [], []
    legal_actions = True
    while True:
        observed = api.observe()
        state = observed["inquiry_state"]
        if state["goal_resolution"]["supported"]:
            stop_reason = "scoped_goal_resolved"
            break
        if state["context"]["status"] in ("ambiguous", "inconsistent", "out_of_model"):
            stop_reason = "context_abstention"
            break
        if state["world"]["status"] in ("inconsistent", "out_of_model"):
            stop_reason = f"{state['world']['status']}_world_abstention"
            break
        legal = api.legal_questions()
        if not legal:
            stop_reason = "budget_exhausted_abstention"
            break
        history = api.history
        optimal = optimal_questions(episode, history)
        question, selection = _choose(policy, episode, history, legal, model, memoryless)
        decision = {
            "step": len(history) + 1,
            "context_before": state["context"],
            "candidate_groups_before": list(state["world"]["candidate_groups"]),
            "candidate_leaves_before": list(state["world"]["candidate_leaves"]),
            "legal_questions": list(legal),
            "chosen_question": question,
            "chosen_question_text": task.QUESTIONS.get(question) if question else None,
            "selection": selection,
        }
        annotations.append({
            "step": len(history) + 1,
            "reference_optimal_ties": list(optimal),
            "chosen_agrees": question in optimal if question else False,
            "source": "evaluator-only future-answer annotation; unavailable to agent",
        })
        if question is None:
            agent_trace.append(decision)
            stop_reason = selection["status"]
            break
        if question not in legal:
            legal_actions = False
            agent_trace.append(decision)
            stop_reason = "illegal_question"
            break
        api.ask(question)
        after = api.observe()["inquiry_state"]
        decision.update({
            "observed_answer": api.history[-1][1],
            "context_after": after["context"],
            "candidate_groups_after": list(after["world"]["candidate_groups"]),
            "candidate_leaves_after": list(after["world"]["candidate_leaves"]),
        })
        agent_trace.append(decision)
    final = api.observe()["inquiry_state"]
    domain_groups = protocol.GROUPS_BY_DOMAIN[episode.domain]
    domain_leaves = protocol.LEAVES_BY_DOMAIN[episode.domain]
    physical = protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[episode.domain]
    context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[episode.domain]
    narrowing = [{
        "step": 0, "groups_remaining": len(domain_groups), "leaves_remaining": len(domain_leaves),
        "uniform_leaf_count_entropy_bits": math.log2(len(domain_leaves)),
    }]
    for item in task.evidence_trace(api.history, api.supplied_context, episode.domain):
        leaf_count = len(item["after_leaves"])
        narrowing.append({
            "step": item["step"],
            "kind": item["kind"],
            "groups_remaining": len(item["after_groups"]),
            "leaves_remaining": leaf_count,
            "uniform_leaf_count_entropy_bits": math.log2(leaf_count) if leaf_count else None,
        })
    resolution = final["goal_resolution"]
    return {
        "episode_id": episode.episode_id,
        "physical_world_id": episode.physical_world_id,
        "domain": episode.domain,
        "split": episode.split,
        "case_type": episode.case_type,
        "context_variant": episode.context_variant,
        "explicit_context": episode.context_answer if episode.context_explicit else None,
        "policy": policy,
        "agent_trace": agent_trace,
        "reference_annotations": annotations,
        "history": [list(item) for item in api.history],
        "evidence_trace": task.evidence_trace(api.history, api.supplied_context, episode.domain),
        "narrowing_by_step": narrowing,
        "final": final,
        "answer": resolution["answer"] if resolution["supported"] else None,
        "goal_resolved": resolution["supported"],
        "full_leaf_resolved": final["world"]["supported"],
        "abstained": not resolution["supported"],
        "correct": _goal_correct(episode, resolution),
        "question_cost": len(api.history),
        "context_question_cost": sum(q == context_question for q, _ in api.history),
        "physical_question_cost": sum(q in physical for q, _ in api.history),
        "budget_adherent": (
            len(api.history) <= (
                protocol.MAX_PHYSICAL_QUESTIONS
                if episode.context_explicit else protocol.MAX_TOTAL_QUESTIONS
            )
            and sum(q in physical for q, _ in api.history)
            <= protocol.MAX_PHYSICAL_QUESTIONS
        ),
        "no_repeats": len({q for q, _ in api.history}) == len(api.history),
        "legal": legal_actions,
        "unsupported_answer": int(resolution["answer"] is not None and not resolution["supported"]),
        "stop_reason": stop_reason,
    }


def evaluate_rows(rows, model, memoryless) -> list[dict]:
    return [{
        "episode_id": row["episode_id"],
        "physical_world_id": row["physical_world_id"],
        "split": row["split"],
        "reference": reference(_episode(row)),
        "policies": {
            policy: run_policy(_episode(row), policy, model, memoryless)
            for policy in protocol.POLICIES
        },
    } for row in rows]


def summarize(rows: list[dict]) -> dict:
    output = {}
    unit_count = len({row["physical_world_id"] for row in rows})
    for policy in protocol.POLICIES:
        results = [row["policies"][policy] for row in rows]
        resolved = [result for result in results if result["goal_resolved"]]
        scorable_resolved = [result for result in resolved if result["correct"] is not None]
        n = len(results)
        output[policy] = {
            "rows": n,
            "physical_units": unit_count,
            "goal_resolution_rate": len(resolved) / n if n else None,
            "correct_goal_resolution_rate": (
                sum(result["correct"] is True for result in scorable_resolved)
                / len(scorable_resolved)
                if scorable_resolved else None
            ),
            "full_leaf_resolution_rate": sum(result["full_leaf_resolved"] for result in results) / n,
            "abstention_rate": sum(result["abstained"] for result in results) / n,
            "unsupported_answer_rate": sum(result["unsupported_answer"] for result in results) / n,
            "mean_question_cost": sum(result["question_cost"] for result in results) / n,
            "mean_context_cost": sum(result["context_question_cost"] for result in results) / n,
            "mean_physical_cost": sum(result["physical_question_cost"] for result in results) / n,
            "legality_rate": sum(
                result["legal"] and result["no_repeats"] and result["budget_adherent"]
                for result in results
            ) / n,
            "mean_groups_remaining": {
                str(step): sum(
                    next(
                        (point["groups_remaining"] for point in result["narrowing_by_step"]
                         if point["step"] == step),
                        result["final"]["world"]["group_count"],
                    )
                    for result in results
                ) / n
                for step in range(protocol.MAX_TOTAL_QUESTIONS + 1)
            },
            "mean_leaves_remaining": {
                str(step): sum(
                    next(
                        (point["leaves_remaining"] for point in result["narrowing_by_step"]
                         if point["step"] == step),
                        result["final"]["world"]["leaf_count"],
                    )
                    for result in results
                ) / n
                for step in range(protocol.MAX_TOTAL_QUESTIONS + 1)
            },
            "next_question_optimal_tie_agreement": (
                sum(annotation["chosen_agrees"] for result in results
                    for annotation in result["reference_annotations"])
                / max(1, sum(len(result["reference_annotations"]) for result in results))
            ),
        }
    return {
        "statistical_unit": "physical_world_id with all context variants kept together",
        "policies": output,
        "interpretation": (
            "Context changes relevance and scoped stopping, never physical compatibility. "
            "Uniform candidate entropy and learned ranking counts are not calibrated probabilities."
        ),
    }


def _write_lines(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _read_lines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def _safe_name(name: str) -> bool:
    path = Path(name)
    return (
        type(name) is str and bool(name) and "\\" not in name and not path.is_absolute()
        and ".." not in path.parts and path.as_posix() == name
    )


def _seal(directory: Path, name: str, document: dict) -> dict:
    write_json(directory / name, document)
    (directory / f"{name}.sha256").write_text(
        file_digest(directory / name) + "\n", encoding="ascii", newline="\n",
    )
    return document


def _read_sealed(directory: Path, name: str) -> dict:
    path, checksum = directory / name, directory / f"{name}.sha256"
    _require(path.is_file() and checksum.is_file(), f"Missing sealed manifest: {name}.")
    _require(
        file_digest(path) == checksum.read_text(encoding="ascii").strip(),
        f"Manifest changed: {name}.",
    )
    return read_json(path)


def _check_files(directory: Path, files: dict[str, str]) -> None:
    _require(type(files) is dict, "Invalid file inventory.")
    for name, expected in files.items():
        _require(_safe_name(name), f"Unsafe executable source path: {name!r}.")
        path = directory / name
        _require(
            path.is_file() and file_digest(path) == expected,
            f"Artifact changed or missing: {name}.",
        )


def _archive_names(directory: Path) -> set[str]:
    root = directory / "sources"
    _require(root.is_dir() and not root.is_symlink(), "Missing source archive.")
    names = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "Source archive symlinks are forbidden.")
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
    return names


def source_fingerprints() -> dict[str, str]:
    return {name: source_digest(path) for name, path in protocol.source_paths().items()}


def prepare(directory: Path, seed: int = protocol.GENERATOR_SEED) -> dict:
    """Freeze protocol, crossed splits, and sources without any policy outcomes."""
    directory = Path(directory)
    partitions = protocol.generate_episodes(seed)
    sources = protocol.source_paths()
    for name, path in sources.items():
        _require(_safe_name(name), f"Unsafe executable source path: {name!r}.")
        _require(path.is_file() and not path.is_symlink(), f"Missing or unsafe source: {name}.")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", protocol.PROTOCOL)
    for split, rows in partitions.items():
        write_json(directory / f"{split}.json", {"episodes": rows})
    archive = directory / "sources"
    archive.mkdir()
    for name, source in sorted(sources.items()):
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    data_names = ("protocol.json", "train.json", "validation.json", "test.json")
    files = {name: file_digest(directory / name) for name in data_names}
    files.update({
        "sources/" + name: file_digest(directory / "sources" / name) for name in sources
    })
    frozen = _seal(directory, "preparation.json", {
        "version": protocol.VERSION,
        "seed": seed,
        "protocol_sha256": protocol.PROTOCOL_SHA256,
        "counts": protocol.COUNTS,
        "physical_units": protocol.PHYSICAL_UNITS,
        "source_sha256": source_fingerprints(),
        "environment": environment(),
        "files": files,
        "outcomes_executed": False,
        "official_test_executed": False,
        "classification": "preregistered crossed-context benchmark; test policy outcomes untouched",
    })
    check_prepared(directory)
    return frozen


def check_prepared(directory: Path, *, execution: bool = True) -> dict:
    directory = Path(directory)
    frozen = _read_sealed(directory, "preparation.json")
    _require(set(frozen) == {
        "version", "seed", "protocol_sha256", "counts", "physical_units",
        "source_sha256", "environment", "files", "outcomes_executed",
        "official_test_executed", "classification",
    }, "Invalid preparation schema.")
    _check_files(directory, frozen["files"])
    expected_archive = {
        name.removeprefix("sources/") for name in frozen["files"] if name.startswith("sources/")
    }
    _require(_archive_names(directory) == expected_archive, "Exact source archive inventory changed.")
    for name, expected in frozen["source_sha256"].items():
        _require(source_digest(directory / "sources" / name) == expected, "Archived source drift.")
    declaration = read_json(directory / "protocol.json")
    _require(digest(declaration) == frozen["protocol_sha256"], "Frozen protocol hash mismatch.")
    if execution:
        _require(frozen["version"] == protocol.VERSION, "Unsupported benchmark version.")
        _require(frozen["protocol_sha256"] == protocol.PROTOCOL_SHA256, "Protocol hash drift.")
        _require(frozen["source_sha256"] == source_fingerprints(), "Live executable source drift.")
        _require(frozen["environment"] == environment(), "Execution environment drift.")
        _require(declaration == protocol.PROTOCOL, "Frozen protocol differs.")
        partitions = protocol.generate_episodes(frozen["seed"])
        for split, rows in partitions.items():
            _require(
                read_json(directory / f"{split}.json") == {"episodes": rows},
                f"Frozen {split} partition differs.",
            )
        protocol.validate_partitions(partitions)
    return frozen


def _provenance(examples: list[dict]) -> dict:
    return {
        "fit_split": "train",
        "protocol_sha256": protocol.PROTOCOL_SHA256,
        "training_examples_sha256": digest(examples),
        "training_example_count": len(examples),
        "training_episode_ids": sorted({row["episode_id"] for row in examples}),
        "training_physical_world_ids": sorted({row["physical_world_id"] for row in examples}),
        "label_rule": LABEL_RULE,
    }


def render_report(split: str, summary: dict, metadata: dict) -> str:
    lines = [
        f"# {protocol.VERSION}: {split} report",
        "",
        "Context is explicit or clarified; it never changes physical compatible sets.",
        "This finite observed-pattern task is not diagnosis, treatment, or calibrated probability.",
        "No policy is required to win. Official test remains untouched unless this report says `test`.",
        "",
        f"- Protocol SHA-256: `{metadata['protocol_sha256']}`",
        f"- History model SHA-256: `{metadata['model_sha256']}`",
        f"- Memoryless model SHA-256: `{metadata['memoryless_model_sha256']}`",
        "",
        "| policy | scoped resolution | correct resolved | full leaf | abstain | total cost | context cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in protocol.POLICIES:
        row = summary["policies"][policy]
        correct = row["correct_goal_resolution_rate"]
        lines.append(
            f"| {policy} | {row['goal_resolution_rate']:.3f} | "
            f"{'n/a' if correct is None else f'{correct:.3f}'} | "
            f"{row['full_leaf_resolution_rate']:.3f} | {row['abstention_rate']:.3f} | "
            f"{row['mean_question_cost']:.3f} | {row['mean_context_cost']:.3f} |"
        )
    lines.extend([
        "",
        "All context variants of a physical world are one statistical unit and remain in one partition. "
        "Scoped answers may retain fine-world uncertainty, which is reported separately.",
        "",
    ])
    return "\n".join(lines)


def train(directory: Path) -> dict:
    """Fit train-only rankers and run validation; never execute official test."""
    directory = Path(directory)
    check_prepared(directory)
    _require(not (directory / "training.json").exists(), "Training already exists.")
    _require(not (directory / "official-test-started.json").exists(), "Official test already started.")
    rows = read_json(directory / "train.json")["episodes"]
    examples = training_examples(rows)
    provenance = _provenance(examples)
    model = task.fit_preference(examples, provenance)
    memoryless = task.fit_preference(examples, provenance, memoryless=True)
    _require(task.fit_preference(examples, provenance).model_sha256 == model.model_sha256,
             "History model reproduction failed.")
    _require(task.fit_preference(examples, provenance, memoryless=True).model_sha256
             == memoryless.model_sha256, "Memoryless model reproduction failed.")
    task.save_model(model, directory / "model.json")
    task.save_model(memoryless, directory / "memoryless-model.json")
    validation_rows = read_json(directory / "validation.json")["episodes"]
    outcomes = evaluate_rows(validation_rows, model, memoryless)
    _write_lines(directory / "validation-results.jsonl", outcomes)
    summary = summarize(outcomes)
    write_json(directory / "validation-summary.json", summary)
    metadata = {
        "protocol_sha256": protocol.PROTOCOL_SHA256,
        "model_sha256": model.model_sha256,
        "memoryless_model_sha256": memoryless.model_sha256,
    }
    (directory / "report.md").write_text(
        render_report("validation", summary, metadata), encoding="utf-8", newline="\n",
    )
    names = (
        "model.json", "memoryless-model.json", "validation-results.jsonl",
        "validation-summary.json", "report.md",
    )
    trained = _seal(directory, "training.json", {
        **metadata,
        "preparation_sha256": file_digest(directory / "preparation.json"),
        "training_examples_sha256": digest(examples),
        "training_example_count": len(examples),
        "fit_episode_ids": provenance["training_episode_ids"],
        "fit_physical_world_ids": provenance["training_physical_world_ids"],
        "files": {name: file_digest(directory / name) for name in names},
        "model_reproduction": {"history_exact": True, "memoryless_exact": True},
        "validation_only": True,
        "no_test_outcomes_executed": True,
    })
    check_frozen(directory)
    return trained


def check_frozen(directory: Path, *, execution: bool = True) -> dict:
    directory = Path(directory)
    check_prepared(directory, execution=execution)
    trained = _read_sealed(directory, "training.json")
    _require(set(trained) == {
        "protocol_sha256", "model_sha256", "memoryless_model_sha256",
        "preparation_sha256", "training_examples_sha256", "training_example_count",
        "fit_episode_ids", "fit_physical_world_ids", "files", "model_reproduction",
        "validation_only", "no_test_outcomes_executed",
    }, "Invalid training schema.")
    _check_files(directory, trained["files"])
    _require(trained["preparation_sha256"] == file_digest(directory / "preparation.json"),
             "Training refers to a different preparation.")
    for name, expected in (
        ("model.json", trained["model_sha256"]),
        ("memoryless-model.json", trained["memoryless_model_sha256"]),
    ):
        document = read_json(directory / name)
        body = {key: value for key, value in document.items() if key != "model_sha256"}
        _require(digest(body) == document.get("model_sha256") == expected,
                 "Frozen model content hash mismatch.")
    if not execution:
        return trained
    model = task.load_model(directory / "model.json")
    memoryless = task.load_model(directory / "memoryless-model.json")
    _require(not model.memoryless and memoryless.memoryless, "Model roles changed.")
    train_rows = read_json(directory / "train.json")["episodes"]
    examples = training_examples(train_rows)
    _require(digest(examples) == trained["training_examples_sha256"],
             "Frozen training examples no longer reproduce.")
    _require({row["episode_id"] for row in examples} == set(trained["fit_episode_ids"]),
             "Training episode membership mismatch.")
    _require({row["physical_world_id"] for row in examples}
             == set(trained["fit_physical_world_ids"]), "Training unit membership mismatch.")
    provenance = _provenance(examples)
    _require(task.fit_preference(examples, provenance).model_sha256 == model.model_sha256,
             "History model does not reproduce.")
    _require(task.fit_preference(examples, provenance, memoryless=True).model_sha256
             == memoryless.model_sha256, "Memoryless model does not reproduce.")
    validation_rows = read_json(directory / "validation.json")["episodes"]
    results = _read_lines(directory / "validation-results.jsonl")
    _require(digest(results) == digest(evaluate_rows(validation_rows, model, memoryless)),
             "Validation traces fail deterministic readback.")
    _require(digest(read_json(directory / "validation-summary.json")) == digest(summarize(results)),
             "Validation summary fails independent recomputation.")
    _require(not (directory / "test-results.jsonl").exists()
             or (directory / "official-test-started.json").exists(),
             "Test results exist without one-shot marker.")
    return trained


def freeze(directory: Path, authorization: str | None = None) -> dict:
    """Seal reviewed development training without executing official test."""
    directory = Path(directory)
    trained = check_frozen(directory)
    _require(
        authorization == FREEZE_AUTHORIZATION,
        "Development freeze blocked pending explicit parent review authorization.",
    )
    _require(
        not (directory / "freeze.json").exists()
        and not (directory / "official-test-started.json").exists(),
        "Development artifact is already frozen or official test has started.",
    )
    return _seal(directory, "freeze.json", {
        "authorization": authorization,
        "protocol_sha256": protocol.PROTOCOL_SHA256,
        "model_sha256": trained["model_sha256"],
        "memoryless_model_sha256": trained["memoryless_model_sha256"],
        "training_sha256": file_digest(directory / "training.json"),
        "reviewed_development_freeze": True,
        "official_test_executed": False,
    })


def check_reviewed_freeze(directory: Path, trained: dict | None = None) -> dict:
    directory = Path(directory)
    trained = trained or check_frozen(directory)
    frozen = _read_sealed(directory, "freeze.json")
    _require(set(frozen) == {
        "authorization", "protocol_sha256", "model_sha256", "memoryless_model_sha256",
        "training_sha256", "reviewed_development_freeze", "official_test_executed",
    }, "Invalid reviewed-freeze schema.")
    _require(
        frozen["authorization"] == FREEZE_AUTHORIZATION
        and frozen["protocol_sha256"] == protocol.PROTOCOL_SHA256
        and frozen["model_sha256"] == trained["model_sha256"]
        and frozen["memoryless_model_sha256"] == trained["memoryless_model_sha256"]
        and frozen["training_sha256"] == file_digest(directory / "training.json")
        and frozen["reviewed_development_freeze"] is True
        and frozen["official_test_executed"] is False,
        "Reviewed development freeze does not match training.",
    )
    return frozen


def evaluate(directory: Path, authorization: str | None = None) -> dict:
    """Execute frozen official test once, only with explicit parent authorization."""
    directory = Path(directory)
    trained = check_frozen(directory)
    check_reviewed_freeze(directory, trained)
    _require(authorization == OFFICIAL_AUTHORIZATION,
             "Official test blocked pending explicit parent authorization.")
    marker = directory / "official-test-started.json"
    _require(not marker.exists() and not (directory / "test-results.jsonl").exists(),
             "Official test is one-shot and already started.")
    _seal(directory, "official-test-started.json", {
        "authorization": authorization,
        "protocol_sha256": protocol.PROTOCOL_SHA256,
        "model_sha256": trained["model_sha256"],
        "one_shot": True,
    })
    model = task.load_model(directory / "model.json")
    memoryless = task.load_model(directory / "memoryless-model.json")
    rows = read_json(directory / "test.json")["episodes"]
    outcomes = evaluate_rows(rows, model, memoryless)
    _write_lines(directory / "test-results.jsonl", outcomes)
    summary = summarize(outcomes)
    write_json(directory / "test-summary.json", summary)
    (directory / "test-report.md").write_text(
        render_report("test", summary, {
            "protocol_sha256": protocol.PROTOCOL_SHA256,
            "model_sha256": model.model_sha256,
            "memoryless_model_sha256": memoryless.model_sha256,
        }),
        encoding="utf-8", newline="\n",
    )
    return _seal(directory, "evaluation.json", {
        "split": "test",
        "protocol_sha256": protocol.PROTOCOL_SHA256,
        "model_sha256": model.model_sha256,
        "memoryless_model_sha256": memoryless.model_sha256,
        "files": {
            name: file_digest(directory / name)
            for name in ("test-results.jsonl", "test-summary.json", "test-report.md")
        },
        "single_official_execution": True,
    })


def report(directory: Path) -> str:
    directory = Path(directory)
    check_frozen(directory, execution=False)
    if (directory / "evaluation.json").exists():
        evaluation = _read_sealed(directory, "evaluation.json")
        _check_files(directory, evaluation["files"])
        return (directory / "test-report.md").read_text(encoding="utf-8")
    return (directory / "report.md").read_text(encoding="utf-8")


def _demo_live(directory: Path, episode_id=None, policy="learned_history", domain=None) -> dict:
    trained = check_frozen(directory)
    rows = read_json(Path(directory) / "validation.json")["episodes"]
    if domain is not None:
        _require(domain in protocol.DOMAINS, "Unknown demonstration domain.")
        rows = [row for row in rows if row["domain"] == domain]
    if episode_id is None:
        row = rows[0]
    else:
        selected = [row for row in rows if row["episode_id"] == episode_id]
        _require(bool(selected), "Unknown validation episode ID for the selected domain.")
        row = selected[0]
    model = task.load_model(Path(directory) / "model.json")
    memoryless = task.load_model(Path(directory) / "memoryless-model.json")
    result = run_policy(_episode(row), policy, model, memoryless)
    return {
        "classification": "validation demonstration, not an official test result",
        "story": task.OPENINGS[row["domain"]],
        "policy": policy,
        "model_sha256": trained["model_sha256"],
        "model_fit_provenance": model.provenance,
        "result": result,
        "notes": [
            "Context is explicit/asked, never inferred.",
            "Context-only evidence leaves physical candidates unchanged.",
            "Scoped goal resolution and full-world uncertainty are reported separately.",
        ],
    }


def _archived_demo(directory: Path, episode_id, policy, domain) -> dict:
    directory = Path(directory).resolve()
    source_root, python_root = directory / "sources", directory / "sources" / "src"
    _require((python_root / "intuition_prototype" / "abstraction_benchmark.py").is_file(),
             "Frozen artifact has no archived demo implementation.")
    legacy_dependencies = {}
    expected_legacy = {
        "records": "3743a6437fefb4cdaf99bde5e78239b5c81345a325eebffb3c951b871d1c8e11",
        "simulator": "3edad28e96ef382ebe672a51d865c541296b3df4fc215a05c41b827374d8a65c",
    }
    for name, expected in expected_legacy.items():
        if not (python_root / "intuition_prototype" / f"{name}.py").is_file():
            dependency = Path(__file__).with_name(f"{name}.py")
            _require(source_digest(dependency) == expected,
                     f"Legacy demo dependency changed: {name}.")
            legacy_dependencies[name] = {"path": str(dependency), "sha256": expected}
    # V2 archived no package initializer; ordinary -m lookup can select the
    # installed live package and recursively redispatch instead of using the archive.
    bootstrap = """
import hashlib, importlib.util, json, sys, types
from pathlib import Path
root, arguments, legacy = Path(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3])
package = types.ModuleType("intuition_prototype")
package.__path__ = [str(root / "sources" / "src" / "intuition_prototype")]
sys.modules["intuition_prototype"] = package
for name in ("records", "simulator"):
    if name not in legacy:
        continue
    path = Path(legacy[name]["path"])
    if hashlib.sha256(path.read_bytes().replace(b"\\r\\n", b"\\n")).hexdigest() != legacy[name]["sha256"]:
        raise ValueError("Legacy dependency changed during archived execution.")
    spec = importlib.util.spec_from_file_location("intuition_prototype." + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from intuition_prototype import abstraction_benchmark as archived
if Path(archived.__file__).resolve().parent != Path(package.__path__[0]).resolve():
    raise ValueError("Archived demo imported live code.")
result = archived._demo_live(root, **arguments)
result["legacy_dependency_sha256"] = {name: entry["sha256"] for name, entry in legacy.items()}
print(json.dumps(result, sort_keys=True))
"""
    command = [
        sys.executable, "-I", "-B", "-c", bootstrap, str(directory),
        json.dumps({"episode_id": episode_id, "policy": policy, "domain": domain}),
        json.dumps(legacy_dependencies),
    ]
    completed = subprocess.run(
        command, cwd=source_root,
        text=True, capture_output=True, check=False,
    )
    _require(completed.returncode == 0,
             "Archived demo failed verification or execution: " + completed.stderr.strip())
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Archived demo returned invalid JSON.") from error
    _require(
        type(result) is dict
        and result.get("classification") == "validation demonstration, not an official test result",
        "Archived demo returned an incompatible record.",
    )
    return result


def demo(directory: Path, episode_id=None, policy="learned_history", domain=None) -> dict:
    directory = Path(directory)
    frozen = check_prepared(directory, execution=False)
    check_frozen(directory, execution=False)
    compatible = (
        frozen["version"] == protocol.VERSION
        and frozen["protocol_sha256"] == protocol.PROTOCOL_SHA256
        and frozen["source_sha256"] == source_fingerprints()
        and frozen["environment"] == environment()
    )
    return (
        _demo_live(directory, episode_id, policy, domain)
        if compatible else _archived_demo(directory, episode_id, policy, domain)
    )


def required_linked_list_demo(*, clarified: bool = False) -> dict:
    """Demonstrate a supplied requirement, without a hidden world or learned model."""
    context = "linked_required_structure"
    supplied = None if clarified else context
    history = ()
    initial = task.inquiry_state(history, supplied, "linked_list")
    if clarified:
        question = protocol.CONTEXT_QUESTION_BY_DOMAIN["linked_list"]
        _require(task.legal_questions(history, supplied, "linked_list") == (question,),
                 "Expected only the goal clarification.")
        history = ((question, context),)
    final = task.inquiry_state(history, supplied, "linked_list")
    return {
        "classification": "deterministic required-means acceptance demonstration; not a learned benchmark result",
        "protocol_version": protocol.VERSION,
        "initial": initial,
        "history": [{"question": task.QUESTIONS[q], "answer": a} for q, a in history],
        "evidence_trace": task.evidence_trace(history, supplied, "linked_list"),
        "final": final,
        "question_cost": len(history),
        "legal_next_questions": task.legal_questions(history, supplied, "linked_list"),
        "code_generated": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    required = commands.add_parser("required-linked-list")
    required.add_argument("--clarified", action="store_true")
    for command in ("prepare", "train", "freeze", "report", "demo", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("directory", type=Path)
        if command == "prepare":
            sub.add_argument("--seed", type=int, default=protocol.GENERATOR_SEED)
        if command == "demo":
            sub.add_argument("--episode-id")
            sub.add_argument("--domain", choices=protocol.DOMAINS)
            sub.add_argument("--policy", choices=protocol.POLICIES, default="learned_history")
        if command == "evaluate":
            sub.add_argument("--authorization")
        if command == "freeze":
            sub.add_argument("--authorization")
    args = parser.parse_args(argv)
    if args.command == "required-linked-list":
        print(json.dumps(required_linked_list_demo(clarified=args.clarified), sort_keys=True, indent=2))
        return
    if args.command == "prepare":
        value = prepare(args.directory, args.seed)
        print(json.dumps(value, sort_keys=True, indent=2))
    elif args.command == "train":
        print(json.dumps(train(args.directory), sort_keys=True, indent=2))
    elif args.command == "freeze":
        print(json.dumps(freeze(args.directory, args.authorization), sort_keys=True, indent=2))
    elif args.command == "report":
        print(report(args.directory), end="")
    elif args.command == "demo":
        print(json.dumps(
            demo(args.directory, args.episode_id, args.policy, args.domain),
            sort_keys=True, indent=2,
        ))
    else:
        print(json.dumps(evaluate(args.directory, args.authorization), sort_keys=True, indent=2))
    if args.command in ("demo", "report"):
        print(
            "Context is explicit/asked; reference ties are hindsight; counts are not calibrated.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
