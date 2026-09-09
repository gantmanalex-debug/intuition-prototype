"""Stage 4 tests execute only declared development rows, NEVER untouched test outcomes."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from intuition_prototype import stage4, stage4_protocol as protocol
from intuition_prototype.benchmark_protocol import digest, generate_cases as historical_cases, read_json
from intuition_prototype.learned import TrainingExample, fit_model, load_model
from intuition_prototype.records import Metrics
from intuition_prototype.simulator import SimulatorConfig
from intuition_prototype.stage4_protocol import source_paths as actual_source_paths


@pytest.fixture(scope="module", autouse=True)
def stable_source_inventory(tmp_path_factory):
    """Concurrent integration edits must not turn a unit test into an official source freeze."""
    import shutil
    root = tmp_path_factory.mktemp("stage4-source-snapshot")
    paths = {}
    for name, source in protocol.source_paths().items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        paths[name] = target
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(protocol, "source_paths", lambda: paths)
        patch.setattr(stage4, "source_paths", lambda: paths)
        yield


def test_deterministic_partition_bounds_and_exclusion():
    cases = protocol.generate_cases()
    assert cases == protocol.generate_cases()
    assert cases != protocol.generate_cases(23)
    assert {s: len(v) for s, v in cases.items()} == {"train": 48, "validation": 16, "test": 48}
    old = {c["config_id"] for entries in historical_cases().values() for c in entries}
    seen = set(old)
    workers = {}
    for split, entries in cases.items():
        workers[split] = {c["parameters"]["workers"] for c in entries}
        for case in entries:
            config = SimulatorConfig(**case["parameters"])
            assert asdict(config) == case["parameters"]
            assert digest(case["parameters"])[:20] == case["config_id"]
            assert case["config_id"] not in seen
            seen.add(case["config_id"])
            assert case["split"] == split
            bounds = protocol.REGIMES[case["regime"]]
            assert config.workers in protocol.WORKERS[split]
            assert (config.arrival_min, config.arrival_max) in bounds["arrivals"]
            assert bounds["db"][0] <= config.db_capacity <= bounds["db"][1]
            assert bounds["locks"][0] <= config.lock_penalty <= bounds["locks"][1]
            assert config.retry_timeout in bounds["timeouts"]
            assert config.retries_enabled in bounds["retries"]
    assert workers["train"].isdisjoint(workers["validation"] | workers["test"])
    assert workers["validation"].isdisjoint(workers["test"])


def test_executable_inventory_never_depends_on_private_runtime_data(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("value = 1\n", encoding="utf-8")
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "local_notes.py").write_text("# ignored runtime content\n", encoding="utf-8")
    monkeypatch.setattr(protocol, "project_root", lambda: tmp_path)
    assert set(actual_source_paths()) == {"pyproject.toml", "helper.py"}


@pytest.mark.parametrize("kwargs", [
    {"per_regime": 1}, {"development_only": True, "per_regime": 0},
    {"development_only": True, "per_regime": True}, {"generator_seed": -1},
    {"generator_seed": True}, {"development_only": "yes"},
])
def test_invalid_declarations(kwargs):
    with pytest.raises(ValueError):
        protocol.generate_cases(**kwargs)


def test_duplicate_configuration_aborts(monkeypatch):
    monkeypatch.setattr(protocol, "digest", lambda value: "0" * 64)
    with pytest.raises(ValueError, match="Duplicate"):
        protocol.generate_cases()


def test_preparation_has_protocol_partitions_and_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr(stage4, "counterfactuals", lambda *args: pytest.fail("Preparation executed outcomes"))
    directory = tmp_path / "prepared"
    manifest = protocol.prepare(directory)
    assert protocol.check_prepared(directory) == manifest
    assert manifest["configuration_counts"] == {"train": 48, "validation": 16, "test": 48}
    declaration = read_json(directory / "protocol.json")
    assert declaration["alpha_grid"] == [.01, .1, 1, 10]
    assert declaration["threshold_grid"] == [0, .02, .05, .1]
    assert not (directory / "model.json").exists()
    assert not (directory / "test-raw.jsonl").exists()
    assert not (directory / "train-start.json").exists()
    with pytest.raises(FileExistsError):
        protocol.prepare(directory)
    assert "src/intuition_prototype/learned.py" in manifest["source_sha256"]
    assert "src/intuition_prototype/stage4.py" in manifest["source_sha256"]
    assert "src/intuition_prototype/benchmark.py" in manifest["source_sha256"]
    assert "src/intuition_prototype/app.py" in manifest["source_sha256"]
    assert "pyproject.toml" in manifest["source_sha256"]
    assert not any(name.startswith("tests/") for name in manifest["source_sha256"])
    monkeypatch.setattr(protocol, "source_fingerprints", lambda: {})
    with pytest.raises(ValueError, match="source drift"):
        stage4.train(directory)
    assert not (directory / "train-start.json").exists()


@pytest.fixture(scope="module")
def development_run(tmp_path_factory):
    """The only actual lifecycle fitting in this suite: 8 train + 8 validation configs."""
    directory = tmp_path_factory.mktemp("stage4-dev") / "run"
    protocol.prepare(directory, development_only=True, per_regime=1)
    # Guard the actual simulator-facing oracle and all policy construction against test configs.
    allowed = {
        digest(c["parameters"]) for split in ("train", "validation")
        for c in read_json(directory / f"{split}.json")["configurations"]
    }
    original = stage4.QueueSimulator.from_config
    with pytest.MonkeyPatch.context() as patch:
        def dev_factory(config, seed=7):
            assert digest(asdict(config)) in allowed, "Attempted execution outside development partitions"
            return original(config, seed)
        patch.setattr(stage4.QueueSimulator, "from_config", dev_factory)
        training = stage4.train(directory)
    return directory, training


def test_actual_dev_fit_selection_and_equal_evidence(development_run):
    directory, training = development_run
    model = load_model(directory / "model.json")
    document = model.document()
    assert model.fingerprint == document["model_sha256"] == training["model_sha256"]
    train_ids = sorted(c["config_id"] for c in read_json(directory / "train.json")["configurations"])
    validation_ids = sorted(c["config_id"] for c in read_json(directory / "validation.json")["configurations"])
    assert document["provenance"]["fit_config_ids"] == train_ids
    assert document["provenance"]["validation_config_ids"] == validation_ids
    assert document["provenance"]["fit_split"] == "train"
    assert document["provenance"]["preparation_sha256"] == digest(protocol.check_prepared(directory))
    grid = read_json(directory / "validation-grid.json")
    assert grid["no_refit"] is True
    assert len(grid["candidates"]) == 16
    assert grid["selected"]["model_sha256"] == model.fingerprint
    winner = max(grid["candidates"], key=stage4._selection_key)
    assert winner["model_sha256"] == model.fingerprint
    assert training["compute"]["fit_examples"] == 16
    assert training["compute"]["ridge_fits"] == 4
    assert training["compute"]["oracle_case_repeats"] == 32
    rows = read_json(directory / "development.json")["rows"]
    assert {r["split"] for r in rows} == {"train", "validation"}
    assert len(rows) == 32
    for row in rows:
        assert set(row["policies"]) == {"scripted", "heuristic", "learned"}
        assert row["classification"] == "development preflight"
        for policy in row["policies"].values():
            episode = policy["episode"]
            assert episode["budget"] == 10 and episode["max_steps"] == 12
            assert policy["assessment"]["budget_adherent"]
            assert policy["assessment"]["steps"] <= 12
            observations = [r for r in episode["records"] if r["kind"] == "Observation"]
            assert observations[0]["data"]["metrics"] == row["baseline"]
    # Refit selected alpha using train only reproduces final coefficients/normalization exactly.
    references = [row for row in rows if row["split"] == "train"]
    examples = [TrainingExample(
        r["config_id"], r["seed"], "train", Metrics(**r["baseline"]),
        {key: value["effect"]["utility"] for key, value in r["counterfactuals"].items()},
    ) for r in references]
    fitted = fit_model(examples, document["alpha"], document["provenance"]).with_selection(
        document["threshold"], validation_ids)
    assert fitted.fingerprint == model.fingerprint


def test_selection_ties_prefer_stronger_ridge_and_threshold():
    candidates = [
        dict(alpha=a, threshold=t, score=dict(cost_adjusted_utility=0, utility=0, cost=2))
        for a in protocol.ALPHAS for t in protocol.THRESHOLDS
    ]
    winner = max(candidates, key=stage4._selection_key)
    assert (winner["alpha"], winner["threshold"]) == (10, .1)
    better = dict(alpha=.01, threshold=0, score=dict(cost_adjusted_utility=.1, utility=.2, cost=1))
    assert max([*candidates, better], key=stage4._selection_key) is better


def test_actual_training_reproduction_is_path_and_timing_independent(development_run, tmp_path, monkeypatch):
    original, training = development_run
    reproduced = tmp_path / "different-output-path"
    preparation = protocol.prepare(reproduced, development_only=True, per_regime=1)
    assert preparation == protocol.check_prepared(original)
    allowed = {
        digest(c["parameters"]) for split in ("train", "validation")
        for c in read_json(reproduced / f"{split}.json")["configurations"]
    }
    factory = stage4.QueueSimulator.from_config
    def dev_factory(config, seed=7):
        assert digest(asdict(config)) in allowed, "Reproduction attempted non-development outcomes"
        return factory(config, seed)
    monkeypatch.setattr(stage4.QueueSimulator, "from_config", dev_factory)
    # Make timing obviously different; it must not affect any fitted-model identity.
    tick = iter(range(10000))
    monkeypatch.setattr(stage4, "perf_counter", lambda: float(next(tick)))
    repeated = stage4.train(reproduced)
    assert repeated == training
    assert digest(repeated) == digest(training)
    assert load_model(reproduced / "model.json").fingerprint == load_model(original / "model.json").fingerprint
    for name in ("preparation.json", "model.json", "training.json", "development.json", "validation-grid.json"):
        assert (reproduced / name).read_bytes() == (original / name).read_bytes()
    assert read_json(reproduced / "training-runtime.json") != read_json(original / "training-runtime.json")
    for name in ("model.json", "training.json"):
        text = (reproduced / name).read_text(encoding="utf-8")
        assert str(reproduced) not in text and str(original) not in text
        assert "started_utc" not in text and "wall_seconds" not in text
    # Report serialization itself is deterministic; no test outcomes are executed.
    rows = read_json(reproduced / "development.json")["rows"]
    summary = stage4.summarize(rows)
    assert summary == json.loads(json.dumps(summary))
    assert stage4.render_report(summary) == stage4.render_report(json.loads(json.dumps(summary)))


def test_fit_never_accepts_validation_or_old_stage3(development_run):
    directory, _ = development_run
    row = read_json(directory / "development.json")["rows"][0]
    utilities = {key: value["effect"]["utility"] for key, value in row["counterfactuals"].items()}
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="never validation/test"):
            fit_model([TrainingExample(row["config_id"], 101, split, Metrics(**row["baseline"]), utilities)],
                      .1, {})
    old = historical_cases()["heldout"][0]["config_id"]
    with pytest.raises(ValueError, match="historical"):
        fit_model([TrainingExample(old, 101, "train", Metrics(**row["baseline"]), utilities)], .1, {})


def test_public_preflight_refuses_test_without_execution(development_run, monkeypatch):
    directory, _ = development_run
    model = load_model(directory / "model.json")
    case = read_json(directory / "test.json")["configurations"][0]
    monkeypatch.setattr(stage4, "_reference", lambda *args: pytest.fail("Executed untouched test"))
    with pytest.raises(ValueError, match="development cases only"):
        stage4.evaluate_case(case, 101, model)
    with pytest.raises(ValueError, match="relabeling"):
        stage4.evaluate_case({**case, "split": "train"}, 101, model)


def test_dev_report_roundtrip_and_raw_validation(development_run):
    directory, _ = development_run
    rows = read_json(directory / "development.json")["rows"]
    cases = {s: read_json(directory / f"{s}.json")["configurations"] for s in ("train", "validation")}
    model = load_model(directory / "model.json")
    stage4.validate_rows(rows, cases, model.fingerprint)
    summary = stage4.summarize(rows)
    assert summary["classification"] == "development preflight"
    assert summary["configuration_count"] == 16
    assert set(summary["comparisons"]) == {"learned-minus-scripted", "learned-minus-heuristic"}
    assert set(summary["behavior_signatures"]["by_split"]) == {"train", "validation"}
    assert all(p["bounds"]["violations"] == 0 for p in summary["policies"].values())
    assert set(summary["exact_telemetry_signatures"]["by_split"]) == {"train", "validation"}
    assert all(
        item["fraction"] is None or 0 <= item["fraction"] <= 1
        for policy in summary["policies"].values() for item in policy["descriptive_rates"].values()
    )
    assert stage4._exact_signatures(rows) == stage4._exact_signatures(list(reversed(rows)))
    assert stage4.render_report(summary) == stage4.render_report(json.loads(json.dumps(summary)))
    assert "no_benefit_false_positive" in stage4.render_report(summary)
    for altered in (rows[:-1], [*rows, rows[0]]):
        with pytest.raises(ValueError, match="Incomplete"):
            stage4.validate_rows(altered, cases, model.fingerprint)
    corrupt = deepcopy(rows)
    corrupt[0]["policies"]["learned"]["assessment"]["utility"] += 1
    with pytest.raises(ValueError, match="Assessment"):
        stage4.validate_rows(corrupt, cases, model.fingerprint)
    corrupt = deepcopy(rows)
    corrupt[0]["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model/policy"):
        stage4.validate_rows(corrupt, cases, model.fingerprint)


def test_freeze_and_integrity_without_test_execution(development_run, tmp_path, monkeypatch):
    import shutil
    original, _ = development_run
    directory = tmp_path / "frozen-dev"
    shutil.copytree(original, directory)
    result = stage4.freeze(directory)
    frozen = stage4.verify_freeze(directory)
    assert result["freeze_sha256"] == digest(frozen)
    assert frozen["model_sha256"] == load_model(directory / "model.json").fingerprint
    monkeypatch.setattr(stage4, "_reference", lambda *args: pytest.fail("Executed untouched test"))
    with pytest.raises(ValueError, match="Development-only"):
        stage4._official_cases(directory, protocol.check_prepared(directory))
    assert not (directory / "evaluate-start.json").exists()
    with pytest.raises(FileExistsError):
        stage4.freeze(directory)
    with pytest.raises(ValueError, match="after freeze"):
        stage4.train(directory)
    live_copy = protocol.source_paths()["src/intuition_prototype/stage4.py"]
    before = live_copy.read_bytes()
    try:
        live_copy.write_bytes(before.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        assert stage4.verify_freeze(directory) == frozen
    finally:
        live_copy.write_bytes(before)
    # Every mutation is confined to the copied temporary artifacts, never real sources.
    for relative in (
        "model.json", "test.json", "development.json", "validation-grid.json",
        "archive/sources/src/intuition_prototype/learned.py",
        "archive/artifacts/model.json", "freeze.json", "training.json",
        "training-runtime.json", "train-start.json",
    ):
        path = directory / relative
        before = path.read_bytes()
        try:
            path.write_bytes(before + b"\n")
            with pytest.raises(ValueError, match="changed"):
                stage4.verify_freeze(directory)
        finally:
            path.write_bytes(before)
    extra = directory / "archive" / "unexpected.py"
    extra.write_text("# extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        stage4.verify_freeze(directory)
    extra.unlink()
    # Historical read-only verification does not instantiate a model against current learned.py.
    monkeypatch.setattr(stage4, "source_fingerprints", lambda: {})
    monkeypatch.setattr(protocol, "source_fingerprints", lambda: {})
    monkeypatch.setattr(stage4, "load_model", lambda *args: pytest.fail("Historical reload loaded live model"))
    assert stage4.verify_freeze(directory, execution=False) == frozen
    with pytest.raises(ValueError, match="source drift"):
        stage4.verify_freeze(directory)


def test_failed_development_training_reserves_attempt(tmp_path, monkeypatch):
    directory = tmp_path / "failed-dev"
    protocol.prepare(directory, development_only=True, per_regime=1)
    def fail(*args):
        raise RuntimeError("injected development failure")
    monkeypatch.setattr(stage4, "_reference", fail)
    with pytest.raises(RuntimeError, match="injected"):
        stage4.train(directory)
    assert (directory / "train-start.json").exists()
    with pytest.raises(FileExistsError):
        stage4.train(directory)
    with pytest.raises(FileNotFoundError):
        stage4.freeze(directory)


def test_exclusive_official_start_marker_without_running_any_case(tmp_path):
    # Marker atomicity can be tested in isolation, without calling official evaluate().
    marker = stage4._start(tmp_path, "evaluate", {"classification": "marker unit test, no outcomes"})
    with pytest.raises(FileExistsError):
        stage4._start(tmp_path, "evaluate", {})
    assert read_json(tmp_path / "evaluate-start.json") == marker


def test_cli_prepare_json(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "cli"
    monkeypatch.setattr("sys.argv", [
        "stage4", "prepare", "--directory", str(directory), "--development-only", "--per-regime", "1",
    ])
    stage4.main()
    output = json.loads(capsys.readouterr().out)
    assert output["development_only"] is True
    assert output["configuration_counts"] == {"train": 8, "validation": 8, "test": 8}
