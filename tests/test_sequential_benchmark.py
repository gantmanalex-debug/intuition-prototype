"""Development fixtures/metadata only. Never execute untouched test configurations."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest

from intuition_prototype import learned
from intuition_prototype import sequential_benchmark as benchmark
from intuition_prototype import sequential_protocol as protocol
from intuition_prototype import stage4_protocol as integrity
from intuition_prototype.benchmark_protocol import digest, file_digest, read_json, source_digest, write_json
from intuition_prototype.records import Intervention
from intuition_prototype.simulator import QueueSimulator


ACTUAL_SOURCE_PATHS = integrity.source_paths


@pytest.fixture(scope="module", autouse=True)
def stable_inventory(tmp_path_factory):
    """Freeze a source snapshot, not concurrent parent edits; runtime behavior is unpatched."""
    root = tmp_path_factory.mktemp("sequential-source-snapshot")
    paths = {}
    for name, path in ACTUAL_SOURCE_PATHS().items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        paths[name] = target
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(integrity, "source_paths", lambda: paths)
        yield paths


@pytest.fixture(scope="module")
def synthetic_model(tmp_path_factory):
    """Handwritten constant selector; not fitted, tuned, or misrepresented as official."""
    body = {
        "version": learned.MODEL_VERSION, "feature_names": list(learned.FEATURE_NAMES),
        "centers": [0.0] * len(learned.FEATURE_NAMES), "scales": [1.0] * len(learned.FEATURE_NAMES),
        "models": {action.value: {"intercept": 0.25, "coefficients": [0.0] * len(learned.FEATURE_NAMES)}
                   for action in Intervention},
        "alpha": 1.0, "threshold": 1.0,
        "provenance": {
            "fit_split": "train", "fit_config_ids": ["synthetic-not-a-real-training-config"],
            "validation_config_ids": [], "training_source_sha256": source_digest(Path(learned.__file__)),
            "purpose": "handwritten development-only unit fixture; no actual training",
        },
    }
    path = tmp_path_factory.mktemp("sequential-model") / "fixture.json"
    write_json(path, {**body, "model_sha256": digest(body)})
    assert learned.load_model(path).fingerprint != protocol.MODEL_SHA256
    return path


def test_public_generator_exclusion_and_control_provenance():
    cases = protocol.generate_cases()
    assert cases == protocol.generate_cases()
    assert {key: len(value) for key, value in cases.items()} == protocol.COUNTS
    old = {c["config_id"] for generator in (protocol.stage3_cases, protocol.stage4_cases)
           for group in generator().values() for c in group}
    seen = set(old)
    workers = {}
    for split, parents in cases.items():
        workers[split] = set()
        for parent in parents:
            control = parent["control"]
            expected = {**parent["parameters"], "db_capacity": parent["parameters"]["db_capacity"] * 2,
                        "retries_enabled": False}
            assert control["parameters"] == expected
            for config in (parent, control):
                assert config["split"] == split
                assert config["config_id"] == digest(config["parameters"])[:20]
                assert config["config_id"] not in seen
                seen.add(config["config_id"])
                workers[split].add(config["parameters"]["workers"])
    assert workers["development"].isdisjoint(workers["test"])
    assert len(seen - old) == 480
    assert protocol.VERSION == "sequential-benchmark-v2"
    assert protocol.SEEDS == (307, 409)
    assert protocol.generate_cases()["development"][15]["config_id"] == "4a6b0d1760a86070d05d"
    assert "before any" in protocol.PROTOCOL["development_revision"]


def test_runtime_free_inventory(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "helper.py").write_text("value = 1\n")
    for folder in (".runtime", "tests", ".venv"):
        target = tmp_path / folder
        target.mkdir()
        (target / "private.py").write_text("raise RuntimeError('not executable inventory')\n")
    archived = tmp_path / "old-experiment"
    archived.mkdir()
    (archived / "preparation.json").write_text("{}")
    (archived / "archived.py").write_text("# data\n")
    monkeypatch.setattr(integrity, "project_root", lambda: tmp_path)
    assert set(ACTUAL_SOURCE_PATHS()) == {"pyproject.toml", "helper.py"}


def test_prepare_no_outcomes_and_exact_copies(tmp_path, synthetic_model, monkeypatch):
    monkeypatch.setattr(benchmark.task, "reference", lambda *a: pytest.fail("prepare ran reference"))
    monkeypatch.setattr(benchmark.task, "environment", lambda *a: pytest.fail("prepare ran policy"))
    directory = tmp_path / "prepared"
    before = synthetic_model.read_bytes()
    frozen = benchmark.prepare(directory, synthetic_model, development_only=True, count=2)
    assert benchmark.check_prepared(directory) == frozen
    assert frozen["classification"] == "development fixture"
    assert frozen["parent_counts"] == {"development": 2, "test": 0}
    assert frozen["configuration_counts"] == {"development": 4, "test": 0}
    assert read_json(directory / "test.json") == {"configurations": []}
    assert read_json(directory / "development.json")["configurations"] == protocol.generate_cases()["development"][:2]
    assert synthetic_model.read_bytes() == before == (directory / "model.json").read_bytes()
    assert (directory / "archive/data/model.json").read_bytes() == before
    original_modules = (
        "__init__.py", "app.py", "benchmark.py", "benchmark_protocol.py", "config.py", "inquiry.py",
        "learned.py", "llm.py", "navigator.py", "predictions.py", "records.py", "simulator.py",
        "stage1.py", "stage4.py", "stage4_protocol.py", "storage.py",
    )
    assert len(original_modules) + 1 == 17  # Includes original pyproject.toml.
    for name in (*original_modules, "sequential.py", "sequential_protocol.py", "sequential_benchmark.py"):
        assert "src/intuition_prototype/" + name in frozen["source_sha256"]
    assert "pyproject.toml" in frozen["source_sha256"]
    for name, source in integrity.source_paths().items():
        assert (directory / "archive" / name).read_bytes() == source.read_bytes()
    assert not list(directory.glob("*-start.json"))
    with pytest.raises(FileExistsError):
        benchmark.prepare(directory, synthetic_model, development_only=True, count=2)


def test_official_model_mismatch_is_explicit(tmp_path, synthetic_model, monkeypatch):
    monkeypatch.setattr(benchmark.task, "reference", lambda *a: pytest.fail("official outcomes forbidden"))
    directory = tmp_path / "not-official"
    with pytest.raises(ValueError, match="Official model content digest mismatch"):
        benchmark.prepare(directory, synthetic_model)
    assert not directory.exists()


def test_extra_archived_source_is_rejected(tmp_path, synthetic_model):
    directory = tmp_path / "prepared"
    benchmark.prepare(directory, synthetic_model, development_only=True, count=1)
    (directory / "archive" / "extra.py").write_text("# unexpected source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="archive inventory"):
        benchmark.check_prepared(directory, execution=False)


@pytest.mark.parametrize("kwargs", [
    {"count": 1}, {"development_only": True, "count": 0}, {"development_only": True, "count": 81},
    {"development_only": True, "count": True}, {"development_only": "yes"},
])
def test_invalid_declarations(tmp_path, synthetic_model, kwargs):
    with pytest.raises(ValueError):
        benchmark.prepare(tmp_path / "invalid", synthetic_model, **kwargs)


@pytest.mark.parametrize("filename", [
    "model.json", "archive/data/model.json", "development.json", "test.json", "protocol.json",
    "archive/src/intuition_prototype/sequential.py", "archive/src/intuition_prototype/learned.py",
])
def test_exact_artifact_mutation_detected(tmp_path, synthetic_model, filename):
    directory = tmp_path / "prepared"
    benchmark.prepare(directory, synthetic_model, development_only=True, count=1)
    path = directory / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Artifact changed"):
        benchmark.check_prepared(directory, execution=False)
    with pytest.raises(ValueError):
        benchmark.run(directory, "development")
    assert not (directory / "development-start.json").exists()


def test_live_source_mutation_blocks_execution_not_archive_reads(tmp_path, synthetic_model, stable_inventory):
    directory = tmp_path / "prepared"
    frozen = benchmark.prepare(directory, synthetic_model, development_only=True, count=1)
    path = stable_inventory["src/intuition_prototype/sequential_benchmark.py"]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# intentional unit-test drift\n")
        assert benchmark.check_prepared(directory, execution=False) == frozen
        with pytest.raises(ValueError, match="source drift"):
            benchmark.run(directory, "development")
        assert not (directory / "development-start.json").exists()
    finally:
        path.write_bytes(original)


@pytest.fixture(scope="module")
def development_run(tmp_path_factory, synthetic_model):
    """Only real lifecycle: first 16 declared development parents, including the known deep ID."""
    directory = tmp_path_factory.mktemp("sequential-development") / "experiment"
    frozen = benchmark.prepare(directory, synthetic_model, development_only=True, count=16)
    cases = read_json(directory / "development.json")["configurations"]
    allowed = {digest(config["parameters"]) for parent in cases for config in (parent, parent["control"])}
    factory = QueueSimulator.from_config
    runner = benchmark.task.run_sequential
    count = 0

    def guarded_factory(config, seed=7):
        assert digest(asdict(config)) in allowed, "Attempted non-development simulator call"
        assert seed in protocol.SEEDS
        return factory(config, seed)

    def checked_runner(*args, **kwargs):
        nonlocal count
        count += 1
        stage = integrity.read_sealed(directory, "development-reference-complete.json")
        integrity.check_files(directory, stage["files"])
        assert len(benchmark._read_lines(directory / "development-references.jsonl")) == 64
        assert len(benchmark._read_lines(directory / "development-eligibility.jsonl")) == 32
        return runner(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(QueueSimulator, "from_config", guarded_factory)
        patch.setattr(benchmark.task, "run_sequential", checked_runner)
        summary = benchmark.run(directory, "development")
    assert count > 0
    return directory, frozen, summary


def test_development_lifecycle_coverage_budget_and_exclusions(development_run):
    directory, frozen, summary = development_run
    refs = benchmark._read_lines(directory / "development-references.jsonl")
    eligibility = benchmark._read_lines(directory / "development-eligibility.jsonl")
    rows = benchmark._read_lines(directory / "development-raw.jsonl")
    assert len(refs) == 64
    assert all(len(row["reference"]["nodes"]) == 52 for row in refs)
    assert any([benchmark.task.WAIT] * 3 == node["path"] for node in refs[0]["reference"]["nodes"])
    for ref in refs:
        for node in ref["reference"]["nodes"]:
            assert node["action_cost"] == 1 + sum(
                1 if key == benchmark.task.WAIT else 3 for key in node["path"]
            )
    included = {row["config_id"] for row in eligibility if row["included"]}
    assert len(rows) == 6 * len(included)
    assert {r["config_id"] for r in rows} == included
    assert summary["oracle_compute"]["trajectories"] == 64 * 52
    assert summary["agent_compute"]["action_cost"] == sum(row["episode"]["cost"] for row in rows)
    assert summary["positive_three_path_verification"]
    assert summary["excluded_reference_summaries"]
    positive = next(row for row in eligibility if row["config_id"] == "4a6b0d1760a86070d05d")
    assert positive["included"] and all(seed["minimum_depth"] == 3 for seed in positive["seeds"])
    for row in rows:
        assert row["freeze_sha256"] == digest(frozen)
        assert row["model_sha256"] == frozen["model_sha256"]
        assert row["episode"]["cost"] == 1 + 3 * len(row["episode"]["moves"]) <= 10
        assert row["episode"]["budget"] == 10
        assert row["assessment"]["budget_violations"] == 0
    assert not (directory / "test-start.json").exists()
    assert not (directory / "test-references.jsonl").exists()
    with pytest.raises(FileExistsError):
        benchmark.run(directory, "development")


def test_early_stop_and_missed_solution_are_not_hidden(development_run):
    directory, _, summary = development_run
    rows = benchmark._read_lines(directory / "development-raw.jsonl")
    learned_rows = [row for row in rows if row["policy"] == benchmark.task.POLICIES[2]]
    assert learned_rows
    for row in learned_rows:
        assert row["episode"]["cost"] == 1
        assert row["episode"]["moves"] == []
        assert row["assessment"]["premature_stop_with_reachable_solution"]
        assert row["assessment"]["missed_available_solution"]
        assert row["episode"]["stop_reason"] == "no_eligible_intervention"
    assert set(r["trace_id"] for r in learned_rows) <= set(summary["failures"]["trace_ids"])


def test_reports_recompute_all_records_without_simulation(development_run, monkeypatch):
    directory, _, summary = development_run
    before = {p.relative_to(directory): file_digest(p) for p in directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(benchmark.task, "reference", lambda *a: pytest.fail("report executed reference"))
    monkeypatch.setattr(benchmark.task, "environment", lambda *a: pytest.fail("report executed simulator"))
    assert benchmark._report_verified(directory, "development") == summary
    assert benchmark.report(directory, "development") == summary
    after = {p.relative_to(directory): file_digest(p) for p in directory.rglob("*") if p.is_file()}
    assert before == after
    assert read_json(directory / "development-report.json") == summary
    assert (directory / "development-report.md").read_text() == benchmark.markdown(summary)


def test_historical_report_uses_archive_despite_live_drift(development_run, stable_inventory):
    directory, _, summary = development_run
    path = stable_inventory["src/intuition_prototype/sequential.py"]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# newer source, old archive still readable\n")
        assert benchmark.report(directory, "development") == summary
        with pytest.raises(ValueError, match="source drift"):
            benchmark.run(directory, "development")
    finally:
        path.write_bytes(original)


def test_historical_report_ignores_new_live_protocol_version(development_run, monkeypatch):
    directory, _, summary = development_run
    monkeypatch.setattr(protocol, "VERSION", "future-protocol-not-this-experiment")
    monkeypatch.setattr(protocol, "MODEL_SHA256", "future-model-not-this-experiment")
    assert benchmark.report(directory, "development") == summary
    with pytest.raises(ValueError, match="Unsupported"):
        benchmark.run(directory, "development")


@pytest.mark.parametrize("mutation", [
    "candidate", "assessment", "extra", "drop", "reference", "reference_cost", "eligibility",
])
def test_full_trace_validation_rejects_changes(development_run, mutation):
    directory, frozen, _ = development_run
    cases = read_json(directory / "development.json")["configurations"]
    refs = benchmark._read_lines(directory / "development-references.jsonl")
    eligibility = benchmark._read_lines(directory / "development-eligibility.jsonl")
    rows = benchmark._read_lines(directory / "development-raw.jsonl")
    if mutation == "candidate":
        rows[0]["episode"]["decisions"][0]["candidates"][0]["score"] += 1
    elif mutation == "assessment":
        rows[0]["assessment"]["cost"] += 1
    elif mutation == "extra":
        rows[0]["episode"]["unvalidated_field"] = "no"
    elif mutation == "drop":
        rows.pop()
    elif mutation == "reference":
        refs[0]["reference"]["nodes"].pop()
    elif mutation == "reference_cost":
        refs[0]["reference"]["nodes"][0]["action_cost"] += 1
    else:
        eligibility[0]["included"] = not eligibility[0]["included"]
    with pytest.raises(ValueError):
        benchmark.validate_records(cases, refs, eligibility, rows, frozen, "development",
                                   learned.load_model(directory / "model.json"))


def test_report_rejects_sealed_raw_mutation(development_run, tmp_path):
    source, _, _ = development_run
    directory = tmp_path / "copied"
    shutil.copytree(source, directory)
    path = directory / "development-raw.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Artifact changed"):
        benchmark._report_verified(directory, "development")


def test_dev_only_marker_guard_and_failure_is_one_shot(tmp_path, synthetic_model, monkeypatch):
    directory = tmp_path / "failed"
    benchmark.prepare(directory, synthetic_model, development_only=True, count=1)
    with pytest.raises(ValueError, match="cannot execute test"):
        benchmark.run(directory, "test")
    with pytest.raises(ValueError, match="cannot execute test"):
        benchmark.report(directory, "test")
    assert not (directory / "test-start.json").exists()

    def fail(*args):
        raise RuntimeError("intentional fixture failure")

    monkeypatch.setattr(benchmark.task, "reference", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        benchmark.run(directory, "development")
    assert (directory / "development-start.json").is_file()
    failure = integrity.read_sealed(directory, "development-failure.json")
    assert failure["error_type"] == "RuntimeError"
    assert not (directory / "development-complete.json").exists()
    with pytest.raises(FileExistsError):
        benchmark.run(directory, "development")
    with pytest.raises(ValueError, match="Attempt failed"):
        benchmark._report_verified(directory, "development")


def test_postrun_source_drift_invalidates_attempt(tmp_path, synthetic_model, stable_inventory, monkeypatch):
    directory = tmp_path / "drifted"
    benchmark.prepare(directory, synthetic_model, development_only=True, count=1)
    path = stable_inventory["src/intuition_prototype/sequential.py"]
    original_bytes = path.read_bytes()
    original_write = benchmark._write_lines

    def write_with_drift(target, rows):
        original_write(target, rows)
        if target.name == "development-raw.jsonl":
            path.write_bytes(original_bytes + b"\n# drift after policy phase\n")

    monkeypatch.setattr(benchmark, "_write_lines", write_with_drift)
    try:
        with pytest.raises(ValueError, match="source drift"):
            benchmark.run(directory, "development")
        assert (directory / "development-failure.json").exists()
        assert not (directory / "development-complete.json").exists()
    finally:
        path.write_bytes(original_bytes)


def test_statistics_parent_grouping_ties_and_nulls(development_run):
    directory, frozen, _ = development_run
    refs = benchmark._read_lines(directory / "development-references.jsonl")
    eligibility = benchmark._read_lines(directory / "development-eligibility.jsonl")
    rows = benchmark._read_lines(directory / "development-raw.jsonl")
    result = benchmark.summarize(refs, eligibility, rows, frozen, "development")
    for cohort in benchmark.COHORTS:
        parents = {row["parent_id"] for row in rows if row["cohort"] == cohort}
        paired = result["paired"][cohort][benchmark.task.POLICIES[0]]["cost"]
        assert paired["n_parents"] == len(parents)
        deltas = []
        for parent in sorted(parents):
            groups = {
                policy: [r["assessment"]["cost"] for r in rows if r["cohort"] == cohort
                         and r["parent_id"] == parent and r["policy"] == policy]
                for policy in (benchmark.task.POLICIES[0], benchmark.task.POLICIES[2])
            }
            assert all(len(values) == 2 for values in groups.values())
            deltas.append(sum(groups[benchmark.task.POLICIES[2]]) / 2
                          - sum(groups[benchmark.task.POLICIES[0]]) / 2)
        assert paired["mean"] == benchmark._mean(deltas)
    assert benchmark.paired_interval([])["mean"] is None
    assert benchmark.paired_interval([])["ci95"] is None
    interval = benchmark.paired_interval([-1, 0, 1])
    assert interval == benchmark.paired_interval([-1, 0, 1])
    assert interval["wins"] == interval["ties"] == interval["losses"] == 1
    assert interval["resamples"] == 2000 and interval["seed"] == 917
    assert benchmark.paired_interval([2])["ci95"] == [2, 2]
    assert benchmark.paired_interval([-2], lower_is_better=True)["wins"] == 1


def test_matched_interaction_resamples_parent_pairs(development_run):
    """Synthetic score records test arithmetic only, not simulator or official outcomes."""
    _, frozen, _ = development_run
    eligibility, rows = [], []
    for parent in ("a", "b"):
        for cohort in benchmark.COHORTS:
            eligibility.append(dict(parent_id=parent, config_id=parent + cohort, cohort=cohort, included=True))
            for seed in protocol.SEEDS:
                for index, policy in enumerate(benchmark.task.POLICIES):
                    utility = (1 if cohort == "deep" else .25) * index
                    score = dict(success=True, utility=utility, cost=4, intervention_depth=1, regret=0,
                                 premature_stop_with_reachable_solution=False, missed_available_solution=False,
                                 local_benefit_did_not_restore_service=0, fuse_switches=0, helpful_recoveries=0,
                                 fuse_recovery_to_service_success=False, budget_violations=0,
                                 stop_reason="service_restored")
                    rows.append(dict(parent_id=parent, cohort=cohort, seed=seed, policy=policy,
                                     assessment=score, episode={"cost": 4}))
    result = benchmark.summarize([], eligibility, rows, frozen, "development")
    for index, policy in enumerate(benchmark.task.POLICIES[:2]):
        interval = result["matched_depth_interaction"][policy]["utility"]
        assert interval["n_parents"] == 2
        assert interval["mean"] == (2 - index) * .75
        assert interval["ci95"] == [interval["mean"]] * 2
    assert any("NOT causal" in note for note in result["notes"])


def test_empty_cohorts_report_not_estimable(development_run):
    _, frozen, _ = development_run
    result = benchmark.summarize([], [], [], frozen, "development")
    for cohort in benchmark.COHORTS:
        assert result["cohorts"][cohort]["included_parents"] == 0
        assert result["cohorts"][cohort]["coverage"] is None
        for scores in result["cohorts"][cohort]["policies"].values():
            assert scores["success_rate"] is None and scores["mean_utility"] is None
    assert result["positive_three_path_verification"] == []
    assert "not estimable" in benchmark.markdown(result)


def test_cli_preparation_and_test_guard(tmp_path, synthetic_model, capsys):
    directory = tmp_path / "cli-fixture"
    benchmark.main(["prepare", "--directory", str(directory), "--model", str(synthetic_model),
                    "--development-only", "--count", "1"])
    assert json.loads(capsys.readouterr().out)["classification"] == "development fixture"
    with pytest.raises(ValueError, match="cannot execute test"):
        benchmark.main(["run", "--directory", str(directory), "--split", "test"])
