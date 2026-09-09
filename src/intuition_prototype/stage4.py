"""Offline Stage 4 lifecycle. Importing this module does not fit or execute policies."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from intuition_prototype.benchmark import (
    AgentView, assess_policy, counterfactuals, score_effect, serialize_episode,
)
from intuition_prototype.benchmark_protocol import canonical_json, digest, file_digest, read_json, write_json
from intuition_prototype.inquiry import run_inquiry
from intuition_prototype.learned import LearnedModel, TrainingExample, fit_model, load_model, run_learned
from intuition_prototype.navigator import run_navigator
from intuition_prototype.records import Episode, Intervention, Metrics
from intuition_prototype.simulator import QueueSimulator, SimulatorConfig
from intuition_prototype.storage import _decode_record
from intuition_prototype.stage4_protocol import (
    ALPHAS, PROTOCOL, SEEDS, THRESHOLDS, VERSION, WORKERS, archive_files, check_files, check_prepared,
    environment, prepare, read_sealed, seal, source_fingerprints, source_paths,
)


POLICIES = ("scripted", "heuristic", "learned")
METRICS = ("utility", "regret", "cost", "cost_adjusted_utility")
TRAIN_FILES = ("development.json", "validation-grid.json", "model.json", "training.json",
               "training.json.sha256", "train-start.json",
               "training-runtime.json", "training-runtime.json.sha256")
PREP_FILES = ("preparation.json", "preparation.json.sha256", "protocol.json",
              "train.json", "validation.json", "test.json")


def _start(directory: Path, phase: str, provenance: dict) -> dict:
    marker = {**provenance, "phase": phase, "started_utc": datetime.now(timezone.utc).isoformat()}
    # Exclusive creation is the durable one-shot reservation, including failed attempts.
    write_json(directory / f"{phase}-start.json", marker)
    return marker


def _reference(case: dict, seed: int) -> dict:
    baseline, outcomes = counterfactuals(SimulatorConfig(**case["parameters"]), seed)
    return {
        "config_id": case["config_id"], "split": case["split"], "regime": case["regime"], "seed": seed,
        "baseline": asdict(baseline),
        "counterfactuals": {key: {"metrics": asdict(value), "effect": asdict(score_effect(baseline, value))}
                            for key, value in sorted(outcomes.items())},
        "oracle_compute": {"measurement_windows": 4, "advance_ticks_including_warmup": 150},
    }


def _policy(case: dict, seed: int, model: LearnedModel, reference: dict, name: str) -> dict:
    view = AgentView(QueueSimulator.from_config(SimulatorConfig(**case["parameters"]), seed))
    if name == "scripted":
        episode = run_inquiry(view, budget=10)
    elif name == "heuristic":
        episode = run_navigator(view, budget=10, max_steps=12)
    elif name == "learned":
        episode = run_learned(view, model, budget=10, max_steps=12)
    else:
        raise ValueError("Unknown policy.")
    if episode.budget != 10 or episode.max_steps != 12 or episode.policy != name:
        raise ValueError("Policies must share the declared bounds and correct identity.")
    baseline = Metrics(**reference["baseline"])
    outcomes = {key: Metrics(**value["metrics"]) for key, value in reference["counterfactuals"].items()}
    return {"episode": serialize_episode(episode), "assessment": assess_policy(episode, baseline, outcomes)}


def _evaluate_case(case: dict, seed: int, model: LearnedModel, reference: dict | None = None) -> dict:
    row = dict(reference if reference is not None else _reference(case, seed))
    row["model_sha256"] = model.fingerprint
    row["policies"] = {name: _policy(case, seed, model, row, name) for name in POLICIES}
    return row


def evaluate_case(case: dict, seed: int, model: LearnedModel) -> dict:
    """Explicit development preflight only. Official test execution is owned by evaluate()."""
    if case.get("split") not in ("train", "validation"):
        raise ValueError("evaluate_case accepts development cases only, never untouched test.")
    if (case["parameters"]["workers"] not in WORKERS[case["split"]]
            or case["config_id"] != digest(case["parameters"])[:20]):
        raise ValueError("Development case identity/support mismatch; relabeling test cases is prohibited.")
    from intuition_prototype.benchmark_protocol import generate_cases as historical_cases
    if case["config_id"] in {c["config_id"] for entries in historical_cases().values() for c in entries}:
        raise ValueError("Historical Stage 3 cases are not Stage 4 development data.")
    if seed not in SEEDS:
        raise ValueError("Undeclared repeat seed.")
    row = _evaluate_case(case, seed, model)
    row["classification"] = "development preflight"
    return row


def _selection_score(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["config_id"]].append(row["policies"]["learned"]["assessment"])
    return {key: mean(mean(a[key] for a in group) for group in groups.values())
            for key in ("cost_adjusted_utility", "utility", "cost")}


def _selection_key(candidate: dict) -> tuple:
    score = candidate["score"]
    return (score["cost_adjusted_utility"], score["utility"], -score["cost"],
            candidate["alpha"], candidate["threshold"])


def train(directory: Path) -> dict:
    directory = Path(directory)
    preparation = check_prepared(directory)
    if (directory / "freeze.json").exists() or (directory / "evaluate-start.json").exists():
        raise ValueError("Training after freeze/evaluation is prohibited.")
    _start(directory, "train", {"preparation_sha256": digest(preparation)})
    started = perf_counter()
    cases = {s: read_json(directory / f"{s}.json")["configurations"] for s in ("train", "validation")}
    oracle_start = perf_counter()
    references = [_reference(case, seed) for entries in cases.values() for case in entries for seed in SEEDS]
    oracle_seconds = perf_counter() - oracle_start
    by_identity = {(row["config_id"], row["seed"]): row for row in references}
    examples = [
        TrainingExample(row["config_id"], row["seed"], "train", Metrics(**row["baseline"]),
                        {key: value["effect"]["utility"] for key, value in row["counterfactuals"].items()})
        for row in references if row["split"] == "train"
    ]
    provenance = {
        "stage4_version": VERSION, "preparation_sha256": digest(preparation),
        "protocol_sha256": digest(read_json(directory / "protocol.json")),
        "training_data_sha256": digest([r for r in references if r["split"] == "train"]),
        "source_sha256": preparation["source_sha256"],
    }
    validation_ids = sorted(c["config_id"] for c in cases["validation"])
    grid, candidates = [], {}
    fit_seconds = validation_seconds = 0.0
    for alpha in ALPHAS:
        begin = perf_counter()
        fitted = fit_model(examples, alpha=alpha, provenance=provenance)
        fit_seconds += perf_counter() - begin
        for threshold in THRESHOLDS:
            model = fitted.with_selection(threshold, validation_ids)
            begin = perf_counter()
            rows = []
            for case in cases["validation"]:
                for seed in SEEDS:
                    reference = by_identity[(case["config_id"], seed)]
                    rows.append({
                        "config_id": case["config_id"], "seed": seed,
                        "policies": {"learned": _policy(case, seed, model, reference, "learned")},
                    })
            validation_seconds += perf_counter() - begin
            entry = dict(alpha=alpha, threshold=threshold, model_sha256=model.fingerprint,
                         score=_selection_score(rows), rows=rows)
            grid.append(entry)
            candidates[model.fingerprint] = model
    winner = max(grid, key=_selection_key)
    selected = candidates[winner["model_sha256"]]
    begin = perf_counter()
    development = [
        _evaluate_case(case, seed, selected, by_identity[(case["config_id"], seed)])
        for entries in cases.values() for case in entries for seed in SEEDS
    ]
    integration_seconds = perf_counter() - begin
    for row in development:
        row["classification"] = "development preflight"
    # Detect concurrent edits during a long training operation before publishing success.
    check_prepared(directory)
    validate_rows(development, cases, selected.fingerprint)
    write_json(directory / "development.json", {"classification": "development preflight", "rows": development})
    write_json(directory / "validation-grid.json", {
        "selection_split": "validation", "candidates": grid,
        "selected": {key: winner[key] for key in ("alpha", "threshold", "model_sha256", "score")},
        "no_refit": True,
    })
    write_json(directory / "model.json", selected.document())
    result = {
        "version": VERSION, "preparation_sha256": digest(preparation),
        "source_sha256": source_fingerprints(), "model_sha256": selected.fingerprint,
        "files": {name: file_digest(directory / name) for name in
                  ("development.json", "validation-grid.json", "model.json")},
        "compute": {
            "fit_examples": len(examples), "ridge_fits": len(ALPHAS), "validation_candidates": len(grid),
            "validation_policy_episodes": len(ALPHAS) * len(THRESHOLDS) * len(validation_ids) * len(SEEDS),
            "selected_development_policy_episodes": len(development) * len(POLICIES),
            "oracle_case_repeats": len(references), "oracle_measurement_windows": len(references) * 4,
            "oracle_advance_ticks_including_warmup": len(references) * 150,
            "oracle_included_in_agent_cost": False,
        },
    }
    seal(directory, "training.json", result)
    # Runtime observations bind to the reproducible manifest, never the other way around.
    # They are separately sealed now and included in the final exact-byte freeze archive.
    seal(directory, "training-runtime.json", {
        "training_sha256": digest(result),
        "files": {"train-start.json": file_digest(directory / "train-start.json")},
        "wall_seconds": dict(oracle=oracle_seconds, fit=fit_seconds, validation=validation_seconds,
                             integration=integration_seconds, total=perf_counter() - started),
        "timing_note": "Observed wall time is not deterministic and is excluded from model/training identity.",
    })
    # Verify the serialized results, not merely the in-memory objects.
    _check_training(directory)
    return result


def _check_training(directory: Path, *, execution: bool = True) -> dict:
    preparation = check_prepared(directory, execution=execution)
    training = read_sealed(directory, "training.json")
    check_files(directory, training["files"])
    runtime = read_sealed(directory, "training-runtime.json")
    check_files(directory, runtime["files"])
    if runtime["training_sha256"] != digest(training):
        raise ValueError("Runtime observations reference a different training manifest.")
    if read_json(directory / "train-start.json")["preparation_sha256"] != digest(preparation):
        raise ValueError("Training start references a different preparation.")
    if training["preparation_sha256"] != digest(preparation):
        raise ValueError("Training references a different preparation.")
    if training["source_sha256"] != preparation["source_sha256"]:
        raise ValueError("Training source differs from preparation.")
    document = read_json(directory / "model.json")
    body = {key: value for key, value in document.items() if key != "model_sha256"}
    if document["model_sha256"] != digest(body) or document["model_sha256"] != training["model_sha256"]:
        raise ValueError("Model fingerprint mismatch.")
    cases = {s: read_json(directory / f"{s}.json")["configurations"] for s in ("train", "validation")}
    provenance = document["provenance"]
    if (provenance["fit_split"] != "train"
            or provenance["fit_config_ids"] != sorted(c["config_id"] for c in cases["train"])
            or provenance["validation_config_ids"] != sorted(c["config_id"] for c in cases["validation"])
            or provenance["preparation_sha256"] != digest(preparation)
            or provenance["source_sha256"] != preparation["source_sha256"]
            or provenance["training_source_sha256"]
            != preparation["source_sha256"]["src/intuition_prototype/learned.py"]
            or provenance["protocol_sha256"] != digest(read_json(directory / "protocol.json"))):
        raise ValueError("Model partition/source provenance mismatch.")
    if execution:
        load_model(directory / "model.json")
        rows = read_json(directory / "development.json")["rows"]
        validate_rows(rows, cases, document["model_sha256"])
        references = [
            {key: value for key, value in row.items() if key not in ("policies", "model_sha256", "classification")}
            for row in rows if row["split"] == "train"
        ]
        if digest(references) != provenance["training_data_sha256"]:
            raise ValueError("Model training data fingerprint mismatch.")
        grid = read_json(directory / "validation-grid.json")
        if grid["selection_split"] != "validation" or grid["no_refit"] is not True:
            raise ValueError("Invalid validation selection declaration.")
        if [(g["alpha"], g["threshold"]) for g in grid["candidates"]] != [
            (a, t) for a in ALPHAS for t in THRESHOLDS
        ]:
            raise ValueError("Undeclared hyperparameter grid.")
        expected_ids = {(c["config_id"], seed) for c in cases["validation"] for seed in SEEDS}
        for candidate in grid["candidates"]:
            identities = [(r["config_id"], r["seed"]) for r in candidate["rows"]]
            if len(identities) != len(expected_ids) or set(identities) != expected_ids:
                raise ValueError("Validation grid leaked or omitted configuration repeats.")
            if candidate["score"] != _selection_score(candidate["rows"]):
                raise ValueError("Validation score mismatch.")
        winner = max(grid["candidates"], key=_selection_key)
        if (grid["selected"] != {key: winner[key] for key in ("alpha", "threshold", "model_sha256", "score")}
                or winner["model_sha256"] != document["model_sha256"]
                or document["alpha"] != winner["alpha"] or document["threshold"] != winner["threshold"]):
            raise ValueError("Model is not the declared validation winner.")
    return training


def freeze(directory: Path) -> dict:
    directory = Path(directory)
    training = _check_training(directory)
    if (directory / "evaluate-start.json").exists():
        raise ValueError("Cannot freeze after evaluation starts.")
    _start(directory, "freeze", {"training_sha256": digest(training)})
    sources = source_paths()
    files = {name: directory / name for name in (*PREP_FILES, *TRAIN_FILES, "freeze-start.json")}
    archives = archive_files(directory, {
        **{"sources/" + name: path for name, path in sources.items()},
        **{"artifacts/" + name: path for name, path in files.items()},
    })
    _check_training(directory)
    result = {
        "version": VERSION, "source_hash_mode": "crlf-to-lf-v1",
        "source_sha256": source_fingerprints(), "environment": environment(),
        "model_sha256": training["model_sha256"],
        "model_provenance": read_json(directory / "model.json")["provenance"],
        "training_sha256": digest(training),
        "files": {name: file_digest(path) for name, path in files.items()},
        "archives": archives,
    }
    seal(directory, "freeze.json", result)
    verify_freeze(directory)
    return {"freeze_sha256": digest(result), "model_sha256": training["model_sha256"]}


def verify_freeze(directory: Path, *, execution: bool = True) -> dict:
    frozen = read_sealed(directory, "freeze.json")
    check_files(directory, frozen["files"])
    check_files(directory, frozen["archives"])
    training = _check_training(directory, execution=execution)
    if frozen["training_sha256"] != digest(training) or frozen["model_sha256"] != training["model_sha256"]:
        raise ValueError("Freeze/training/model linkage mismatch.")
    if frozen["source_sha256"] != training["source_sha256"]:
        raise ValueError("Freeze/training source linkage mismatch.")
    if frozen["model_provenance"] != read_json(directory / "model.json")["provenance"]:
        raise ValueError("Frozen model provenance mismatch.")
    from intuition_prototype.benchmark_protocol import source_digest
    if frozen["source_hash_mode"] != "crlf-to-lf-v1":
        raise ValueError("Unknown source hash mode.")
    for name, expected in frozen["source_sha256"].items():
        if source_digest(directory / "archive" / "sources" / name) != expected:
            raise ValueError(f"Portable archived source changed: {name}")
    for name in frozen["files"]:
        if file_digest(directory / "archive" / "artifacts" / name) != frozen["files"][name]:
            raise ValueError(f"Archive/live artifact mismatch: {name}")
    archived_names = {
        path.relative_to(directory).as_posix() for path in (directory / "archive").rglob("*") if path.is_file()
    }
    if archived_names != set(frozen["archives"]):
        raise ValueError("Archive inventory changed.")
    if execution and (frozen["source_sha256"] != source_fingerprints() or frozen["environment"] != environment()):
        raise ValueError("Executable source or Python environment changed after freeze.")
    return frozen


def _decode_episode(document: dict) -> Episode:
    fields = dict(document)
    fields["records"] = tuple(_decode_record(r["kind"], canonical_json(r["data"])) for r in fields["records"])
    return Episode(**fields)


def validate_rows(rows: list[dict], cases: dict[str, list[dict]], model_sha256: str) -> None:
    """Roundtrip and re-assess persisted telemetry; never re-executes the simulator."""
    expected = {(c["config_id"], seed): c for entries in cases.values() for c in entries for seed in SEEDS}
    identities = [(r["config_id"], r["seed"]) for r in rows]
    if len(identities) != len(expected) or set(identities) != set(expected):
        raise ValueError("Incomplete, duplicate or undeclared case repeats.")
    for row in rows:
        case = expected[(row["config_id"], row["seed"])]
        if row["split"] != case["split"] or row["regime"] != case["regime"]:
            raise ValueError("Raw row partition/regime mismatch.")
        if row["model_sha256"] != model_sha256 or set(row["policies"]) != set(POLICIES):
            raise ValueError("Raw row model/policy mismatch.")
        baseline = Metrics(**row["baseline"])
        if baseline.ticks != 30:
            raise ValueError("Wrong measurement window.")
        outcomes = {}
        for key, value in row["counterfactuals"].items():
            outcomes[key] = Metrics(**value["metrics"])
            if asdict(score_effect(baseline, outcomes[key])) != value["effect"]:
                raise ValueError("Independent effect does not match raw telemetry.")
        for name, value in row["policies"].items():
            episode = _decode_episode(value["episode"])
            if episode.policy != name or episode.budget != 10 or episode.max_steps != 12:
                raise ValueError("Raw policy identity/bounds mismatch.")
            if digest(serialize_episode(episode)) != digest(value["episode"]):
                raise ValueError("Episode serialization did not roundtrip.")
            if assess_policy(episode, baseline, outcomes) != value["assessment"]:
                raise ValueError("Assessment does not match raw episode/effects.")
            for record in value["episode"]["records"]:
                if record["kind"] == "DecisionRecord" and name == "learned":
                    if any(c.get("model_sha256") != model_sha256 for c in record["data"]["candidates"]):
                        raise ValueError("Learned trace references another model.")


def _official_cases(directory: Path, preparation: dict) -> list[dict]:
    """Check the declaration only, without executing any test outcomes."""
    if preparation["development_only"]:
        raise ValueError("Development-only preparations cannot execute official test outcomes.")
    cases = read_json(directory / "test.json")["configurations"]
    if not cases or any(c["split"] != "test" for c in cases):
        raise ValueError("Official evaluation requires the declared untouched test split.")
    return cases


def evaluate(directory: Path) -> dict:
    """Official one-shot test. No resume, retry, overwrite or development-only promotion."""
    directory = Path(directory)
    frozen = verify_freeze(directory)
    preparation = check_prepared(directory)
    cases = _official_cases(directory, preparation)
    model = load_model(directory / "model.json")
    marker = _start(directory, "evaluate", {
        "freeze_sha256": digest(frozen), "model_sha256": model.fingerprint,
        "classification": "official one-shot test",
    })
    started = perf_counter()
    rows = []
    oracle_seconds = policy_seconds = 0.0
    # Write each completed pair immediately: a failed attempt retains its partial trace.
    with (directory / "test-raw.jsonl").open("x", encoding="utf-8", newline="\n") as output:
        for case in cases:
            for seed in SEEDS:
                begin = perf_counter()
                reference = _reference(case, seed)
                oracle_seconds += perf_counter() - begin
                begin = perf_counter()
                row = _evaluate_case(case, seed, model, reference)
                policy_seconds += perf_counter() - begin
                row.update(classification="official one-shot test", freeze_sha256=digest(frozen))
                output.write(canonical_json(row) + "\n")
                output.flush()
                rows.append(row)
    verify_freeze(directory)
    validate_rows(rows, {"test": cases}, model.fingerprint)
    reloaded = _read_raw(directory)
    if digest(reloaded) != digest(rows):
        raise ValueError("Raw JSONL roundtrip mismatch.")
    validate_rows(reloaded, {"test": cases}, model.fingerprint)
    development = read_json(directory / "development.json")["rows"]
    summary = summarize(reloaded, development_rows=development, classification="official one-shot test")
    summary["freeze_sha256"] = digest(frozen)
    write_json(directory / "report.json", summary)
    markdown = render_report(summary)
    with (directory / "report.md").open("x", encoding="utf-8", newline="\n") as output:
        output.write(markdown)
    if render_report(read_json(directory / "report.json")) != markdown:
        raise ValueError("Markdown report roundtrip mismatch.")
    seal(directory, "evaluation.json", {
        "version": VERSION, "freeze_sha256": digest(frozen), "start_sha256": digest(marker),
        "model_sha256": model.fingerprint,
        "files": {name: file_digest(directory / name) for name in
                  ("evaluate-start.json", "test-raw.jsonl", "report.json", "report.md")},
        "compute": {
            "test_case_repeats": len(rows), "policy_episodes": len(rows) * 3,
            "oracle_measurement_windows": len(rows) * 4, "oracle_advance_ticks_including_warmup": len(rows) * 150,
            "oracle_included_in_agent_cost": False,
            "wall_seconds": dict(oracle=oracle_seconds, policies=policy_seconds, total=perf_counter() - started),
        },
    })
    report(directory)  # Persisted report/raw integrity check, no policy execution.
    return summary


def _read_raw(directory: Path) -> list[dict]:
    with (directory / "test-raw.jsonl").open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def _signature(row: dict) -> str:
    return canonical_json({
        "zero_arrivals": row["baseline"]["arrivals"] == 0,
        "observed_retries": row["baseline"]["retries"] > 0,
        "effects": {key: [value["effect"]["beneficial"],
                          (value["effect"]["utility"] > 0) - (value["effect"]["utility"] < 0)]
                    for key, value in sorted(row["counterfactuals"].items())},
    })


def _exact_signatures(rows: list[dict]) -> dict[str, dict[str, str]]:
    groups = defaultdict(list)
    for row in rows:
        signature = digest({
            "baseline": row["baseline"],
            "outcomes": {key: value["metrics"] for key, value in row["counterfactuals"].items()},
        })
        groups[(row["split"], row["config_id"])].append((row["seed"], signature))
    by_split = defaultdict(dict)
    for (split, config_id), windows in groups.items():
        by_split[split][config_id] = digest(sorted(windows))
    return dict(by_split)


def _interval(units: list[dict]) -> dict:
    groups = defaultdict(list)
    for unit in units:
        groups[unit["regime"]].append(unit)
    distributions = {key: [] for key in METRICS}
    rng = random.Random(917)
    for _ in range(2000):
        sample = [rng.choice(group) for _, group in sorted(groups.items()) for _ in group]
        for key in METRICS:
            distributions[key].append(mean(unit[key] for unit in sample))

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        index = (len(ordered) - 1) * fraction
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {
        key: {
            "mean_delta": mean(u[key] for u in units),
            "configuration_delta_min": min(u[key] for u in units),
            "configuration_delta_max": max(u[key] for u in units),
            "paired_stratified_bootstrap_95": [percentile(distributions[key], .025),
                                             percentile(distributions[key], .975)],
            "positive": sum(u[key] > 1e-12 for u in units),
            "ties": sum(abs(u[key]) <= 1e-12 for u in units),
            "negative": sum(u[key] < -1e-12 for u in units),
        } for key in METRICS
    }


def summarize(rows: list[dict], *, development_rows: list[dict] | None = None,
              classification: str = "development preflight", protocol: dict | None = None) -> dict:
    protocol = PROTOCOL if protocol is None else protocol
    if not rows or classification not in ("development preflight", "official one-shot test"):
        raise ValueError("A nonempty, explicitly classified result is required.")
    if classification == "development preflight" and any(r["split"] == "test" for r in rows):
        raise ValueError("Untouched test rows cannot be presented as development preflight.")
    if classification == "official one-shot test" and any(r["split"] != "test" for r in rows):
        raise ValueError("Official report must contain test rows only.")
    identities = [(r["config_id"], r["seed"]) for r in rows]
    if len(identities) != len(set(identities)) or any(r["seed"] not in SEEDS for r in rows):
        raise ValueError("Duplicate/undeclared repeats.")
    groups = defaultdict(list)
    for row in rows:
        groups[row["config_id"]].append(row)
    if any(sorted(r["seed"] for r in group) != list(SEEDS) for group in groups.values()):
        raise ValueError("Every configuration requires both declared repeats.")
    if any(len({(r["split"], r["regime"]) for r in group}) != 1 for group in groups.values()):
        raise ValueError("Configuration crosses partitions or regimes.")
    report_data = {
        "version": VERSION, "classification": classification,
        "scope": protocol["scope"], "uncertainty": protocol["statistics"],
        "no_benefit_scope": protocol["no_benefit_scope"],
        "model_sha256": sorted({r["model_sha256"] for r in rows}),
        "configuration_count": len(groups), "seed_repeat_count": len(rows),
        "comparisons": {}, "policies": {},
        "oracle_compute": {
            "measurement_windows": sum(r["oracle_compute"]["measurement_windows"] for r in rows),
            "advance_ticks_including_warmup": sum(r["oracle_compute"]["advance_ticks_including_warmup"] for r in rows),
            "included_in_agent_cost": False,
        },
    }
    for other in ("scripted", "heuristic"):
        units = []
        for config_id, group in sorted(groups.items()):
            unit = dict(config_id=config_id, split=group[0]["split"], regime=group[0]["regime"])
            for key in METRICS:
                unit[key] = mean(r["policies"]["learned"]["assessment"][key]
                                 - r["policies"][other]["assessment"][key] for r in group)
            units.append(unit)
        name = "learned-minus-" + other
        report_data["comparisons"][name] = {
            "per_configuration": units, "overall": _interval(units),
            "by_regime": {regime: _interval([u for u in units if u["regime"] == regime])
                          for regime in sorted({u["regime"] for u in units})},
            "sample_configurations": {
                key: {
                    "positive": [u["config_id"] for u in units if u[key] > 1e-12][:5],
                    "ties": [u["config_id"] for u in units if abs(u[key]) <= 1e-12][:5],
                    "negative": [u["config_id"] for u in units if u[key] < -1e-12][:5],
                } for key in METRICS
            },
        }
    categories = (*protocol["failure_categories"], "no_benefit_false_positive")
    for name in POLICIES:
        assessments = [r["policies"][name]["assessment"] for r in rows]
        switches = recoveries = 0
        pointers = {key: [] for key in categories}
        for index, row in enumerate(rows):
            value = row["policies"][name]
            switch = sum(r["kind"] == "FuseRecord" and r["data"]["disposition"] == "switch_path"
                         for r in value["episode"]["records"])
            switches += switch
            recoveries += bool(switch and value["assessment"]["helpful"])
            for category in categories:
                if value["assessment"][category] and len(pointers[category]) < 5:
                    pointers[category].append({
                        "config_id": row["config_id"], "seed": row["seed"],
                        "trace": (f"test-raw.jsonl:{index + 1}#/policies/{name}/episode" if row["split"] == "test"
                                  else f"development.json#/rows/{index}/policies/{name}/episode"),
                    })
        report_data["policies"][name] = {
            "means": {key: mean(a[key] for a in assessments) for key in METRICS},
            "failures": {key: sum(a[key] for a in assessments) for key in categories},
            "failure_trace_examples": pointers,
            "supported": sum(a["supported"] for a in assessments),
            "abstentions": sum(a["abstained"] for a in assessments),
            "no_benefit_cases": sum(a["no_benefit_case"] for a in assessments),
            "stop_reasons": dict(sorted(Counter(a["stop_reason"] for a in assessments).items())),
            "bounds": {
                "budget": 10, "step_limit": 12, "max_cost": max(a["cost"] for a in assessments),
                "max_steps": max(a["steps"] for a in assessments),
                "violations": sum(not a["budget_adherent"] or a["steps"] > 12 for a in assessments),
            },
            "fuse_switches": switches, "externally_helpful_episodes_after_switch": recoveries,
            "failures_by_regime": {
                regime: {
                    category: sum(r["policies"][name]["assessment"][category]
                                  for r in rows if r["regime"] == regime)
                    for category in categories
                } for regime in sorted({r["regime"] for r in rows})
            },
        }
        values = report_data["policies"][name]
        fractions = {
            "supported_but_unhelpful_fraction": (
                values["failures"]["supported_but_unhelpful"], values["supported"],
            ),
            "no_benefit_false_positive_fraction": (
                values["failures"]["no_benefit_false_positive"], values["no_benefit_cases"],
            ),
            "abstention_available_benefit_fraction": (
                values["failures"]["abstained_with_available_benefit"], values["abstentions"],
            ),
            "benefit_case_abstention_fraction": (
                values["failures"]["abstained_with_available_benefit"], len(rows) - values["no_benefit_cases"],
            ),
        }
        values["descriptive_rates"] = {
            key: {"numerator": numerator, "denominator": denominator,
                  "fraction": numerator / denominator if denominator else None}
            for key, (numerator, denominator) in fractions.items()
        }
    all_rows = [*(development_rows or []), *rows]
    signatures = {}
    for split in sorted({r["split"] for r in all_rows}):
        subset = [r for r in all_rows if r["split"] == split]
        signatures[split] = {
            "episode_repeat_counts": dict(sorted(Counter(_signature(r) for r in subset).items())),
            "configuration_signature_counts": dict(sorted(Counter(
                canonical_json(sorted({_signature(r) for r in subset if r["config_id"] == cid}))
                for cid in sorted({r["config_id"] for r in subset})
            ).items())),
        }
    report_data["behavior_signatures"] = {
        "definition": protocol["signature"], "by_split": signatures,
        "warning": "Observed outcome patterns, not causal classes or independent novelty evidence.",
    }
    exact = _exact_signatures(all_rows)
    report_data["exact_telemetry_signatures"] = {
        "definition": protocol["exact_signature"],
        "by_split": {
            split: {
                "configurations": len(values), "distinct_paired_signatures": len(set(values.values())),
                "duplicate_configurations": len(values) - len(set(values.values())),
            } for split, values in sorted(exact.items())
        },
    }
    if classification == "official one-shot test":
        if not development_rows or any(r["split"] not in ("train", "validation") for r in development_rows):
            raise ValueError("Post-test novelty requires development-only reference outcomes.")
        development_signatures = {_signature(r) for r in development_rows}
        exact_development = {
            signature for split in ("train", "validation") for signature in exact[split].values()
        }
        report_data["exact_telemetry_signatures"]["test_configuration_overlap_with_development"] = sum(
            signature in exact_development for signature in exact["test"].values()
        )
        novel_ids = sorted(cid for cid, group in groups.items()
                           if all(_signature(r) not in development_signatures for r in group))
        report_data["behavior_signatures"].update({
            "test_repeat_overlap_with_development": sum(_signature(r) in development_signatures for r in rows),
            "test_repeat_novel_to_development": sum(_signature(r) not in development_signatures for r in rows),
            "all_repeats_novel_configuration_ids": novel_ids,
            "novel_subset": {
                name: {key: mean(u[key] for u in comp["per_configuration"] if u["config_id"] in novel_ids)
                       for key in METRICS}
                for name, comp in report_data["comparisons"].items()
            } if novel_ids else None,
            "novel_subset_warning": "Post-hoc descriptive subset; not a fresh or confirmatory test.",
        })
    return report_data


def render_report(summary: dict) -> str:
    lines = [
        "# Stage 4 offline selector benchmark", "", f"Classification: **{summary['classification']}**",
        "", summary["scope"], "",
        f"Configurations: {summary['configuration_count']}; seed repeats: {summary['seed_repeat_count']}.",
        "", "## Absolute policy means", "",
        "| Policy | Utility | Regret | Cost | Cost-adjusted utility | Supports | Abstentions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, policy in sorted(summary["policies"].items()):
        values = policy["means"]
        lines.append(
            f"| {name} | {values['utility']:.6f} | {values['regret']:.6f} | {values['cost']:.3f} "
            f"| {values['cost_adjusted_utility']:.6f} | {policy['supported']} | {policy['abstentions']} |"
        )
    lines += [
        "", "## Paired configuration results", "",
        "| Comparison | Metric | Mean delta | Paired 95% interval | Positive / tie / negative |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for name, comparison in sorted(summary["comparisons"].items()):
        for metric in METRICS:
            value = comparison["overall"][metric]
            low, high = value["paired_stratified_bootstrap_95"]
            lines.append(f"| {name} | {metric} | {value['mean_delta']:.6f} | [{low:.6f}, {high:.6f}] | "
                         f"{value['positive']} / {value['ties']} / {value['negative']} |")
    lines += ["", "Positive regret/cost deltas are worse; positive utility deltas are better.",
              "Intervals are 2,000 stratified paired configuration bootstrap resamples, not independent seeds.",
              "No generalization beyond this generator or causal identification is established.",
              "", "## Failures, bounds and fuse behavior", ""]
    for name, policy in sorted(summary["policies"].items()):
        lines += [f"### {name}", "", f"- Failures: `{canonical_json(policy['failures'])}`",
                  f"- Descriptive rates (null means no denominator): `{canonical_json(policy['descriptive_rates'])}`",
                  f"- Stop reasons: `{canonical_json(policy['stop_reasons'])}`",
                  f"- Bounds: `{canonical_json(policy['bounds'])}`",
                  f"- Fuse switches: {policy['fuse_switches']}; externally helpful episodes after switch: "
                  f"{policy['externally_helpful_episodes_after_switch']}.",
                  f"- Sample trace pointers: `{canonical_json(policy['failure_trace_examples'])}`", ""]
    lines += ["## Outcome signatures (descriptive only)", "",
              canonical_json(summary["behavior_signatures"]), "",
              "Exact paired telemetry duplicates: " + canonical_json(summary["exact_telemetry_signatures"]), "",
              "Oracle computation is separate from policy evidence cost. Full per-config/regime deltas, "
              "ties/losses and oracle counts are retained in report.json.",
              "Historical Stage 3 outcomes were not fitting data or fresh evidence.",
              summary["no_benefit_scope"], ""]
    return "\n".join(lines)


def report(directory: Path) -> dict:
    """Read-only historical reload: verifies artifacts without requiring current executable hashes."""
    directory = Path(directory)
    frozen = verify_freeze(directory, execution=False)
    evaluation = read_sealed(directory, "evaluation.json")
    check_files(directory, evaluation["files"])
    if (evaluation["freeze_sha256"] != digest(frozen)
            or evaluation["model_sha256"] != frozen["model_sha256"]
            or evaluation["start_sha256"] != digest(read_json(directory / "evaluate-start.json"))):
        raise ValueError("Evaluation provenance mismatch.")
    rows = _read_raw(directory)
    cases = read_json(directory / "test.json")["configurations"]
    expected = {(c["config_id"], seed) for c in cases for seed in SEEDS}
    identities = [(r["config_id"], r["seed"]) for r in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Incomplete official raw output.")
    if any(r["freeze_sha256"] != digest(frozen) or r["model_sha256"] != frozen["model_sha256"] for r in rows):
        raise ValueError("Raw output freeze/model mismatch.")
    summary = summarize(rows, development_rows=read_json(directory / "development.json")["rows"],
                        classification="official one-shot test", protocol=read_json(directory / "protocol.json"))
    summary["freeze_sha256"] = digest(frozen)
    if summary != read_json(directory / "report.json"):
        raise ValueError("Report does not roundtrip from raw output.")
    if render_report(summary) != (directory / "report.md").read_text(encoding="utf-8"):
        raise ValueError("Markdown does not roundtrip from report.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "freeze", "evaluate", "report"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--generator-seed", type=int, default=20260909)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--per-regime", type=int)
    args = parser.parse_args()
    try:
        if args.command != "prepare" and (args.development_only or args.per_regime is not None
                                          or args.generator_seed != 20260909):
            raise ValueError("Generation options are only valid for prepare.")
        if args.command == "prepare":
            result = prepare(args.directory, args.generator_seed, development_only=args.development_only,
                             per_regime=args.per_regime)
        else:
            result = {"train": train, "freeze": freeze, "evaluate": evaluate, "report": report}[args.command](
                args.directory)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"error": str(error), "command": args.command}, sort_keys=True))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
