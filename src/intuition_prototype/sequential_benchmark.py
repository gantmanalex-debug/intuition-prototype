"""One-shot, frozen cumulative-remediation benchmark (no training or import-time work).

Official preparation requires the published Stage 4 model's *content* digest.
Development-only declarations may use synthetic models and cannot execute test.
Reproduce the model using the original published Stage 4 commit/training procedure,
then copy its local model.json into this experiment at prepare; never commit it.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from intuition_prototype import sequential as task
from intuition_prototype import sequential_protocol as protocol
from intuition_prototype import stage4_protocol as integrity
from intuition_prototype.benchmark_protocol import digest, file_digest, read_json, source_digest, write_json
from intuition_prototype.learned import load_model
from intuition_prototype.records import Intervention, Metrics
from intuition_prototype.simulator import SimulatorConfig


COHORTS = ("deep", "shallow")
SPLITS = ("development", "test")
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 917
NOTES = [
    "Only separately versioned sequential adapters are evaluated; old policies are not cumulative planners.",
    "Original Stage 4 source/models/weights are untouched; this experiment does not erase Stage 4 failure.",
    "Fixed inherited local prediction, learned one-step utility target, and operational service success "
    "are different quantities. A matched local prediction is not service success.",
    "Depth interaction is NOT causal isolated depth: initial state changes and structural selection apply.",
    "Intervals are descriptive parent-unit percentile bootstrap intervals, not population guarantees; "
    "one parent yields a degenerate interval, not evidence of certainty.",
    "Reference trajectories/windows/time are separate from agent action cost; failed interventions cost 3.",
    "Integrity seals detect accidental drift, not adversarial rewriting or signed attestation.",
]


def _json(value):
    """Canonical JSON-shaped data (dataclasses may contain tuples or string enums)."""
    return json.loads(json.dumps(value, allow_nan=False))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _cases(development_only: bool = False, count: int | None = None) -> dict[str, list[dict]]:
    _require(type(development_only) is bool, "development_only must be boolean.")
    _require(count is None or (
        development_only and type(count) is int and 1 <= count <= protocol.COUNTS["development"]
    ), "Smaller count requires development-only and an integer in 1..80.")
    cases = protocol.generate_cases()
    if development_only:
        cases = {"development": cases["development"][:count], "test": []}
    return cases


def prepare(
    directory: Path, model: Path, *, development_only: bool = False, count: int | None = None,
) -> dict:
    """Declare both partitions, exact model, all executable code, and raw archives."""
    directory, model = Path(directory), Path(model)
    cases = _cases(development_only, count)
    loaded = load_model(model)
    _require(development_only or loaded.fingerprint == protocol.MODEL_SHA256,
             "Official model content digest mismatch; use the frozen published Stage 4 model.")
    sources = integrity.source_paths()
    fingerprints = integrity.source_fingerprints()
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model, directory / "model.json")
    write_json(directory / "protocol.json", protocol.PROTOCOL)
    for split, entries in cases.items():
        write_json(directory / f"{split}.json", {"configurations": entries})
    data_names = ("model.json", "protocol.json", "development.json", "test.json")
    files = {name: file_digest(directory / name) for name in data_names}
    files.update(integrity.archive_files(directory, {
        **sources, **{"data/" + name: directory / name for name in data_names},
    }))
    frozen = integrity.seal(directory, "preparation.json", {
        "version": protocol.VERSION, "development_only": development_only, "count": count,
        "classification": "development fixture" if development_only else "official frozen transfer",
        "parent_counts": {split: len(entries) for split, entries in cases.items()},
        "configuration_counts": {split: 2 * len(entries) for split, entries in cases.items()},
        "source_sha256": fingerprints, "environment": integrity.environment(),
        "model_sha256": loaded.fingerprint, "model_file_sha256": files["model.json"],
        "files": files,
        "reproduction": (
            "Use the original published Stage 4 commit and its original training procedure; copy its "
            "local model.json at prepare. No training/tuning here; do not commit the copied model."
        ),
    })
    check_prepared(directory)
    return frozen


def check_prepared(directory: Path, *, execution: bool = True) -> dict:
    directory = Path(directory)
    frozen = integrity.read_sealed(directory, "preparation.json")
    integrity.check_files(directory, frozen["files"])
    archived_names = {
        path.relative_to(directory).as_posix()
        for path in (directory / "archive").rglob("*") if path.is_file()
    }
    _require(
        archived_names == {name for name in frozen["files"] if name.startswith("archive/")},
        "Exact archive inventory changed.",
    )
    declaration = read_json(directory / "protocol.json")
    _require(frozen["version"] == declaration["version"], "Frozen protocol version mismatch.")
    for name, expected in frozen["source_sha256"].items():
        _require("archive/" + name in frozen["files"], "Missing executable archive entry.")
        _require(source_digest(directory / "archive" / name) == expected, "Archived source drift.")
    for name in ("model.json", "protocol.json", "development.json", "test.json"):
        _require(frozen["files"][name] == frozen["files"]["archive/data/" + name],
                 "Archived data differs from declaration.")
    document = read_json(directory / "model.json")
    body = {k: v for k, v in document.items() if k != "model_sha256"}
    _require(digest(body) == document["model_sha256"] == frozen["model_sha256"],
             "Frozen model content digest mismatch.")
    _require(frozen["model_file_sha256"] == file_digest(directory / "model.json"),
             "Frozen model file digest mismatch.")
    _require(frozen["development_only"] or frozen["model_sha256"] == declaration["model_sha256"],
             "Official model content digest mismatch.")
    if execution:
        _require(frozen["version"] == protocol.VERSION, "Unsupported sequential declaration.")
        _require(frozen["source_sha256"] == integrity.source_fingerprints(),
                 "Executable source drift since preparation.")
        _require(frozen["environment"] == integrity.environment(), "Execution environment drift.")
        _require(digest(declaration) == digest(protocol.PROTOCOL),
                 "Protocol declaration drift.")
        expected = _cases(frozen["development_only"], frozen["count"])
        for split, entries in expected.items():
            _require(read_json(directory / f"{split}.json") == {"configurations": entries},
                     "Partition differs from public generator.")
        _require(frozen["parent_counts"] == {s: len(v) for s, v in expected.items()},
                 "Parent count mismatch.")
        _require(frozen["configuration_counts"] == {s: 2 * len(v) for s, v in expected.items()},
                 "Configuration count mismatch.")
    return frozen


def _split(frozen, split):
    _require(split in SPLITS, "Unknown split.")
    _require(not (split == "test" and frozen["development_only"]),
             "Development-only declarations cannot execute test.")


def _write_lines(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _read_lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _identity(case, parent_id, cohort, seed, split, frozen):
    return dict(parent_id=parent_id, config_id=case["config_id"], cohort=cohort, seed=seed,
                split=split, model_sha256=frozen["model_sha256"], freeze_sha256=digest(frozen))


def _validate_reference(oracle):
    paths = [node["path"] for node in oracle["nodes"]]
    _require(paths == [list(path) for path in task.legal_paths()],
             "Reference must include every legal prefix, including repeated waits.")
    successes = []
    for node in oracle["nodes"]:
        metrics = Metrics(**node["metrics"])
        _require(node["success"] == task.service_restored(metrics)
                 and node["utility"] == task.service_utility(metrics), "Reference endpoint mismatch.")
        _require(type(node["action_cost"]) is int
                 and node["action_cost"] == 1 + sum(1 if key == task.WAIT else 3 for key in node["path"]),
                 "Reference action-cost accounting mismatch.")
        if node["success"]:
            successes.append(node["path"])
    minimum = min(map(len, successes), default=None)
    strict = minimum == 3 and all(task.WAIT not in path and len(set(path)) == 3 for path in successes)
    _require(oracle["successful_paths"] == successes and oracle["minimum_depth"] == minimum
             and oracle["requires_three_distinct_interventions"] == strict
             and oracle["best_utility"] == max(node["utility"] for node in oracle["nodes"]),
             "Reference structural summary mismatch.")
    _require(oracle["oracle_compute"] == {
        "trajectories": len(paths),
        "measurement_windows_including_warmup": sum(2 + len(path) for path in paths),
    }, "Reference compute accounting mismatch.")


def eligibility_rows(cases, references):
    """Structural-only membership, retaining both seeds and every excluded parent/control."""
    lookup = {(row["config_id"], row["seed"]): row["reference"] for row in references}
    rows = []
    for case in cases:
        base = [lookup[case["config_id"], seed] for seed in protocol.SEEDS]
        control = [lookup[case["control"]["config_id"], seed] for seed in protocol.SEEDS]
        deep, shallow = task.cohort(base, control)
        for config, cohort, included, oracles in (
            (case, "deep", deep, base), (case["control"], "shallow", shallow, control),
        ):
            rows.append({
                "parent_id": case["config_id"], "config_id": config["config_id"], "cohort": cohort,
                "included": included,
                "reason": ("included by both-seed structural rule" if included else
                           "parent fails both-seed strict depth-three rule" if not deep else
                           "control fails both-seed minimum-depth-one rule"),
                "seeds": [{
                    "seed": seed, "minimum_depth": oracle["minimum_depth"],
                    "requires_three_distinct_interventions": oracle["requires_three_distinct_interventions"],
                    "successful_paths": oracle["successful_paths"], "best_utility": oracle["best_utility"],
                } for seed, oracle in zip(protocol.SEEDS, oracles, strict=True)],
            })
    return rows


class _ReplayAPI:
    """Telemetry-only path playback; the policy never receives reference/configuration labels."""

    def __init__(self, oracle):
        self._metrics = {tuple(node["path"]): node["metrics"] for node in oracle["nodes"]}
        self._path = ()

    def observe(self):
        return Metrics(**self._metrics[self._path])

    def capabilities(self):
        if len(self._path) == task.MAX_DEPTH:
            return ()
        return tuple(sorted(item.value for item in Intervention if item.value not in self._path)) + (task.WAIT,)

    def move(self, key):
        _require(key in self.capabilities(), "Illegal replay move.")
        self._path += (key,)
        return self.observe()


def _assessment(episode, oracle):
    scores = task.assess_episode(episode, oracle)
    scores["helpful_recoveries"] = sum(
        decision["fuse"] == "switch_to_untried_intervention"
        and move["intermediate_utility_change"] > 0
        for decision, move in zip(episode["decisions"], episode["moves"])
    )
    scores["fuse_recovery_to_service_success"] = bool(scores["fuse_switches"] and scores["success"])
    return scores


def validate_records(cases, references, eligibility, rows, frozen, split, model):
    """Check exact coverage/order, all structural evidence, scores and full deterministic decisions."""
    expected = [
        _identity(config, parent["config_id"], cohort, seed, split, frozen)
        for parent in cases for config, cohort in ((parent, "deep"), (parent["control"], "shallow"))
        for seed in protocol.SEEDS
    ]
    _require(len(references) == len(expected), "Reference coverage mismatch.")
    for row, identity in zip(references, expected, strict=True):
        _require(set(row) == set(identity) | {"reference", "elapsed_seconds"}
                 and all(row[key] == value for key, value in identity.items()),
                 "Reference identity mismatch.")
        _require(type(row["elapsed_seconds"]) in (int, float)
                 and math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0,
                 "Invalid reference timing.")
        _validate_reference(row["reference"])
    _require(eligibility == eligibility_rows(cases, references), "Structural eligibility mismatch.")
    included = {row["config_id"] for row in eligibility if row["included"]}
    expected_rows = [(ref, policy) for ref in references if ref["config_id"] in included for policy in task.POLICIES]
    _require(len(rows) == len(expected_rows), "Policy coverage mismatch.")
    for row, (ref, policy) in zip(rows, expected_rows, strict=True):
        identity = {key: ref[key] for key in expected[0]}
        trace = f"{split}/{ref['parent_id']}/{ref['cohort']}/{ref['seed']}/{policy}"
        episode = _json(task.run_sequential(_ReplayAPI(ref["reference"]), policy, model, budget=10))
        scores = _assessment(episode, ref["reference"])
        _require(row == {**identity, "trace_id": trace, "policy": policy,
                         "episode": episode, "assessment": scores},
                 "Full policy trace or independent assessment mismatch.")


def paired_interval(values, *, lower_is_better=False):
    """One value per parent (not seed or action); learned minus baseline."""
    if not values:
        return dict(n_parents=0, mean=None, ci95=None, wins=0, ties=0, losses=0,
                    status="not estimable", resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED)
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAPS))
    oriented = [-value if lower_is_better else value for value in values]
    return dict(n_parents=n, mean=sum(values) / n,
                ci95=[samples[int(.025 * (BOOTSTRAPS - 1))], samples[int(.975 * (BOOTSTRAPS - 1))]],
                wins=sum(value > 0 for value in oriented), ties=sum(value == 0 for value in oriented),
                losses=sum(value < 0 for value in oriented),
                status="single-parent descriptive interval" if n == 1 else "descriptive",
                resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED)


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize(references, eligibility, rows, frozen, split):
    results = {"version": protocol.VERSION, "split": split, "classification": frozen["classification"],
               "freeze_sha256": digest(frozen), "model_sha256": frozen["model_sha256"],
               "notes": NOTES, "cohorts": {}, "paired": {}, "matched_depth_interaction": {}}
    means = {}
    for cohort in COHORTS:
        members = [item for item in eligibility if item["cohort"] == cohort]
        accepted = [item for item in members if item["included"]]
        group = [row for row in rows if row["cohort"] == cohort]
        result = {
            "candidate_parents": len(members), "included_parents": len(accepted),
            "excluded_parents": len(members) - len(accepted),
            "coverage": len(accepted) / len(members) if members else None,
            "seed_cases": 2 * len(accepted), "policies": {},
            "status": "descriptive" if accepted else "not estimable: no eligible configurations",
        }
        for policy in task.POLICIES:
            scores = [row["assessment"] for row in group if row["policy"] == policy]
            result["policies"][policy] = {
                "episodes": len(scores),
                **{("success_rate" if key == "success" else "mean_" + key): _mean([s[key] for s in scores])
                   for key in ("success", "utility", "cost", "intervention_depth", "regret",
                               "premature_stop_with_reachable_solution", "missed_available_solution",
                               "local_benefit_did_not_restore_service", "fuse_switches", "helpful_recoveries",
                               "fuse_recovery_to_service_success", "budget_violations")},
                "depth_distribution": dict(sorted(Counter(str(s["intervention_depth"]) for s in scores).items())),
                "success_distribution": dict(Counter(str(s["success"]).lower() for s in scores)),
                "stop_reasons": dict(Counter(s["stop_reason"] for s in scores)),
            }
            for member in accepted:
                selected = [row["assessment"] for row in group
                            if row["policy"] == policy and row["parent_id"] == member["parent_id"]]
                _require(len(selected) == 2, "Statistics require both seeds per parent/policy.")
                means[cohort, member["parent_id"], policy] = {
                    key: _mean([score[key] for score in selected]) for key in ("success", "utility", "cost")
                }
        results["cohorts"][cohort] = result
        results["paired"][cohort] = {}
        for baseline in task.POLICIES[:2]:
            results["paired"][cohort][baseline] = {
                metric: paired_interval([
                    means[cohort, member["parent_id"], task.POLICIES[2]][metric]
                    - means[cohort, member["parent_id"], baseline][metric] for member in accepted
                ], lower_is_better=metric == "cost") for metric in ("success", "utility", "cost")
            }
    matched = sorted(
        {item["parent_id"] for item in eligibility if item["included"] and item["cohort"] == "deep"}
        & {item["parent_id"] for item in eligibility if item["included"] and item["cohort"] == "shallow"}
    )
    for baseline in task.POLICIES[:2]:
        results["matched_depth_interaction"][baseline] = {
            metric: paired_interval([
                (means["deep", parent, task.POLICIES[2]][metric] - means["deep", parent, baseline][metric])
                - (means["shallow", parent, task.POLICIES[2]][metric] - means["shallow", parent, baseline][metric])
                for parent in matched
            ], lower_is_better=metric == "cost") for metric in ("success", "utility", "cost")
        }
    results["oracle_compute"] = {
        "configuration_seed_references": len(references),
        "trajectories": sum(r["reference"]["oracle_compute"]["trajectories"] for r in references),
        "measurement_windows_including_warmup": sum(
            r["reference"]["oracle_compute"]["measurement_windows_including_warmup"] for r in references),
        "elapsed_seconds": sum(r["elapsed_seconds"] for r in references),
    }
    results["agent_compute"] = {"episodes": len(rows), "action_cost": sum(r["episode"]["cost"] for r in rows)}
    results["reference_depth_distribution"] = {
        cohort: dict(Counter(str(r["reference"]["minimum_depth"]) for r in references if r["cohort"] == cohort))
        for cohort in COHORTS
    }
    failures = [row for row in rows if not row["assessment"]["success"]]
    results["failures"] = {
        "count": len(failures), "trace_ids": [r["trace_id"] for r in failures],
        "examples": [{"trace_id": r["trace_id"], "assessment": r["assessment"],
                      "path": [m["action"] for m in r["episode"]["moves"]]} for r in failures[:10]],
    }
    included_deep = {item["config_id"] for item in eligibility if item["included"] and item["cohort"] == "deep"}
    positive = [r for r in references if r["config_id"] in included_deep
                and r["reference"]["requires_three_distinct_interventions"]]
    results["positive_three_path_verification"] = [{
        "parent_id": r["parent_id"], "seed": r["seed"],
        "minimum_depth_including_wait": r["reference"]["minimum_depth"],
        "all_successes_three_distinct": True,
        "sample_endpoint": next(n for n in r["reference"]["nodes"] if n["success"]),
    } for r in positive[:4]]
    results["excluded_reference_summaries"] = [r for r in eligibility if not r["included"]]
    results["behavior_signatures"] = {
        "unique_full_reference_metrics": len({digest([n["metrics"] for n in r["reference"]["nodes"]])
                                             for r in references}),
        "unique_initial_evidence": len({digest(r["reference"]["nodes"][0]["metrics"]) for r in references}),
        "reference_count": len(references), "use": "descriptive only; never an inclusion rule",
    }
    return results


def markdown(result: dict) -> str:
    lines = [f"# Sequential remediation - {result['split']}", "", result["classification"], ""]
    lines.extend("- " + note for note in result["notes"])
    lines.extend(["", "| Cohort | Parents | Seed cases | Policy | Success | Utility | Cost |",
                  "|---|---:|---:|---|---:|---:|---:|"])
    for cohort, group in result["cohorts"].items():
        for policy, scores in group["policies"].items():
            values = [scores["success_rate"], scores["mean_utility"], scores["mean_cost"]]
            cells = ["not estimable" if value is None else f"{value:.6g}" for value in values]
            lines.append(f"| {cohort} | {group['included_parents']} | {group['seed_cases']} | "
                         f"{policy} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Full reproducible summary", "", "```json",
                  json.dumps(result, sort_keys=True, indent=2, allow_nan=False), "```", ""])
    return "\n".join(lines)


def run(directory: Path, split: str) -> dict:
    """Exhaust references and seal eligibility before calling any policy."""
    directory = Path(directory)
    frozen = check_prepared(directory)
    _split(frozen, split)
    model = load_model(directory / "model.json")
    start_name = f"{split}-start.json"
    integrity.seal(directory, start_name, {
        "split": split, "freeze_sha256": digest(frozen), "model_sha256": model.fingerprint,
        "started_unix": time.time(), "rule": "exclusive attempt; failures cannot resume or rerun",
    })
    try:
        cases = read_json(directory / f"{split}.json")["configurations"]
        references = []
        for parent in cases:
            for config, cohort in ((parent, "deep"), (parent["control"], "shallow")):
                for seed in protocol.SEEDS:
                    start = time.perf_counter()
                    oracle = task.reference(SimulatorConfig(**config["parameters"]), seed)
                    references.append({
                        **_identity(config, parent["config_id"], cohort, seed, split, frozen),
                        "reference": oracle, "elapsed_seconds": time.perf_counter() - start,
                    })
        for row in references:
            _validate_reference(row["reference"])
        eligibility = eligibility_rows(cases, references)
        reference_name, eligibility_name = f"{split}-references.jsonl", f"{split}-eligibility.jsonl"
        _write_lines(directory / reference_name, references)
        _write_lines(directory / eligibility_name, eligibility)
        reference_stage = integrity.seal(directory, f"{split}-reference-complete.json", {
            "split": split, "freeze_sha256": digest(frozen),
            "files": {name: file_digest(directory / name) for name in (reference_name, eligibility_name)},
            "selection": "structural only, persisted before any policy execution",
        })
        check_prepared(directory)
        included = {row["config_id"] for row in eligibility if row["included"]}
        configs = {c["config_id"]: c for parent in cases for c in (parent, parent["control"])}
        rows = []
        policy_start = time.perf_counter()
        for ref in references:
            if ref["config_id"] not in included:
                continue
            for policy in task.POLICIES:
                api = task.environment(SimulatorConfig(**configs[ref["config_id"]]["parameters"]), ref["seed"])
                episode = _json(task.run_sequential(api, policy, model, budget=10))
                rows.append({
                    **{k: v for k, v in ref.items() if k not in ("reference", "elapsed_seconds")},
                    "trace_id": f"{split}/{ref['parent_id']}/{ref['cohort']}/{ref['seed']}/{policy}",
                    "policy": policy, "episode": episode,
                    "assessment": _assessment(episode, ref["reference"]),
                })
        policy_elapsed = time.perf_counter() - policy_start
        raw_name = f"{split}-raw.jsonl"
        _write_lines(directory / raw_name, rows)
        references, eligibility, rows = (
            _read_lines(directory / name) for name in (reference_name, eligibility_name, raw_name)
        )
        validate_records(cases, references, eligibility, rows, frozen, split, model)
        result = summarize(references, eligibility, rows, frozen, split)
        report_name, md_name = f"{split}-report.json", f"{split}-report.md"
        write_json(directory / report_name, result)
        with (directory / md_name).open("x", encoding="utf-8", newline="\n") as out:
            out.write(markdown(result))
        check_prepared(directory)
        integrity.check_files(directory, reference_stage["files"])
        _require(read_json(directory / report_name) == result
                 and (directory / md_name).read_text(encoding="utf-8") == markdown(result),
                 "Report readback mismatch.")
        names = (reference_name, eligibility_name, raw_name, report_name, md_name, start_name,
                 start_name + ".sha256", f"{split}-reference-complete.json",
                 f"{split}-reference-complete.json.sha256")
        integrity.seal(directory, f"{split}-complete.json", {
            "split": split, "freeze_sha256": digest(frozen), "model_sha256": model.fingerprint,
            "files": {name: file_digest(directory / name) for name in names},
            "policy_elapsed_seconds": policy_elapsed, "validation": "full deterministic trace replay and scoring",
        })
        return result
    except (ValueError, RuntimeError, OSError) as exc:
        integrity.seal(directory, f"{split}-failure.json", {
            "split": split, "freeze_sha256": digest(frozen), "error_type": type(exc).__name__,
            "error": str(exc), "status": "invalid attempt; no automatic resume or rerun",
        })
        raise


def _report_verified(directory: Path, split: str) -> dict:
    """Executed using frozen archived code, without simulator calls or writes."""
    directory = Path(directory)
    frozen = check_prepared(directory, execution=False)
    _split(frozen, split)
    _require(not (directory / f"{split}-failure.json").exists(), "Attempt failed; no valid report.")
    complete = integrity.read_sealed(directory, f"{split}-complete.json")
    _require(complete["split"] == split and complete["freeze_sha256"] == digest(frozen)
             and complete["model_sha256"] == frozen["model_sha256"], "Completion identity mismatch.")
    integrity.check_files(directory, complete["files"])
    stage = integrity.read_sealed(directory, f"{split}-reference-complete.json")
    _require(stage["split"] == split and stage["freeze_sha256"] == digest(frozen),
             "Reference-stage identity mismatch.")
    integrity.check_files(directory, stage["files"])
    cases = read_json(directory / f"{split}.json")["configurations"]
    references, eligibility, rows = (
        _read_lines(directory / f"{split}-{suffix}.jsonl") for suffix in ("references", "eligibility", "raw")
    )
    validate_records(cases, references, eligibility, rows, frozen, split, load_model(directory / "model.json"))
    result = summarize(references, eligibility, rows, frozen, split)
    _require(read_json(directory / f"{split}-report.json") == result, "Report regeneration mismatch.")
    _require((directory / f"{split}-report.md").read_text(encoding="utf-8") == markdown(result),
             "Markdown regeneration mismatch.")
    return result


def report(directory: Path, split: str) -> dict:
    """Read-only historical verification under the exact archived scorer/model/runner.

    -I ignores ambient PYTHONPATH; -B prevents bytecode writes. Current source hashes
    and current platform are deliberately not execution gates for historical reads.
    """
    directory = Path(directory).resolve()
    frozen = check_prepared(directory, execution=False)
    _split(frozen, split)
    script = (
        "import sys,json; sys.path.insert(0,sys.argv[1]); "
        "from intuition_prototype.sequential_benchmark import _report_verified; "
        "print(json.dumps(_report_verified(sys.argv[2],sys.argv[3]),allow_nan=False))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(directory / "archive" / "src"),
         str(directory), split], capture_output=True, text=True, check=False,
    )
    _require(completed.returncode == 0, "Archived report verification failed:\n" + completed.stderr)
    return json.loads(completed.stdout)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--directory", required=True, type=Path)
    prep.add_argument("--model", required=True, type=Path)
    prep.add_argument("--development-only", action="store_true")
    prep.add_argument("--count", type=int, help="First N development parents; requires --development-only")
    for name in ("run", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--directory", required=True, type=Path)
        sub.add_argument("--split", required=True, choices=SPLITS)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.directory, args.model, development_only=args.development_only, count=args.count)
        print(json.dumps({"classification": result["classification"], "parent_counts": result["parent_counts"]}))
    elif args.command == "run":
        print(markdown(run(args.directory, args.split)))
    else:
        print(markdown(report(args.directory, args.split)))


if __name__ == "__main__":
    main()
