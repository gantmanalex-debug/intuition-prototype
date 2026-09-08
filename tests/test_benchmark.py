from dataclasses import asdict, replace
import json

import pytest

from intuition_prototype import benchmark_protocol
from intuition_prototype.benchmark import (
    AgentView, assess_policy, counterfactuals, evaluate_case, load_results,
    paired_intervals, render_report, run_benchmark, score_effect, summarize,
)
from intuition_prototype.benchmark_protocol import (
    POLICY_NAMES, REGIMES, STAGE2_HASHES, check_frozen, digest, file_digest,
    generate_cases, prepare, read_json, source_digest, source_fingerprints,
)
from intuition_prototype.inquiry import run_inquiry
from intuition_prototype.records import Action, ActionKind, Intervention, Metrics
from intuition_prototype.simulator import QueueSimulator, SimulatorConfig, Snapshot


def test_configuration_split_is_deterministic_disjoint_and_not_seed_only():
    cases = generate_cases()
    assert cases == generate_cases()
    assert cases != generate_cases(generator_seed=1)
    assert {key: len(value) for key, value in cases.items()} == {"development": 16, "heldout": 48}
    dev = {case["config_id"] for case in cases["development"]}
    heldout = {case["config_id"] for case in cases["heldout"]}
    assert dev.isdisjoint(heldout)
    for split, entries in cases.items():
        for case in entries:
            config = SimulatorConfig(**case["parameters"])
            assert digest(asdict(config))[:20] == case["config_id"]
            workers, arrivals, databases, locks, timeouts, retries = REGIMES[case["regime"]][split]
            assert config.workers in workers
            assert (config.arrival_min, config.arrival_max) == arrivals
            assert databases[0] <= config.db_capacity <= databases[1]
            assert locks[0] <= config.lock_penalty <= locks[1]
            assert config.retry_timeout in timeouts
            assert config.retries_enabled in retries
    for ranges in REGIMES.values():
        assert set(ranges["development"][0]).isdisjoint(ranges["heldout"][0])


@pytest.mark.parametrize("changes", [
    {"workers": 1}, {"workers": 17}, {"workers": True}, {"db_capacity": 0},
    {"db_capacity": float("nan")}, {"lock_penalty": float("inf")}, {"lock_penalty": -0.1},
    {"retry_timeout": 0}, {"retry_timeout": 1.5}, {"retries_enabled": "yes"},
    {"arrival_min": 3, "arrival_max": 2}, {"arrival_max": 9}, {"arrival_min": -1},
])
def test_configuration_bounds(changes):
    with pytest.raises(ValueError):
        SimulatorConfig(**changes)


def test_parameterized_factory_preserves_curated_behavior_and_snapshot_controls():
    for scenario, config in [
        ("worker_capacity", SimulatorConfig()),
        ("db_contention", SimulatorConfig(db_capacity=3, lock_penalty=0.12)),
        ("retry_amplification", SimulatorConfig(db_capacity=9, lock_penalty=0.12, retry_timeout=2)),
    ]:
        assert run_inquiry(QueueSimulator(scenario)) == run_inquiry(QueueSimulator.from_config(config))
    configured = QueueSimulator.from_config(SimulatorConfig(workers=3))
    checkpoint = configured.snapshot()
    first = configured.perform(Action(ActionKind.TEST, Intervention.DOUBLE_WORKERS))
    with pytest.raises(ValueError, match="already increased"):
        configured.perform(Action(ActionKind.TEST, Intervention.DOUBLE_WORKERS))
    configured.restore(checkpoint)
    assert configured.perform(Action(ActionKind.TEST, Intervention.DOUBLE_WORKERS)) == first
    with pytest.raises(ValueError, match="SimulatorConfig"):
        QueueSimulator.from_config("unknown")


def test_external_score_uses_independent_absolute_and_relative_criteria():
    baseline = Metrics(30, 10, 1, 0, 0, 5, 14, 0, 10.0)
    weak = replace(baseline, completions=2, queue_end=13)
    effect = score_effect(baseline, weak)
    assert effect.raw_utility > 0
    assert not effect.beneficial
    assert effect.utility == 0
    helpful = replace(baseline, completions=4, queue_end=11)
    assert score_effect(baseline, helpful).beneficial
    harmful = replace(baseline, completions=0, queue_end=15)
    assert score_effect(baseline, harmful).utility < 0
    with pytest.raises(ValueError, match="same window"):
        score_effect(baseline, replace(weak, arrivals=11))


def test_supported_interpretation_is_not_external_correctness():
    baseline = Metrics(30, 10, 1, 0, 0, 5, 14, 0, 10.0)
    tiny_gain = replace(baseline, completions=2, queue_end=13)

    class WeakBenefit:
        def snapshot(self):
            return Snapshot("opaque")

        def restore(self, snapshot):
            assert snapshot.token == "opaque"

        def perform(self, action):
            if action.kind == ActionKind.REFRAME:
                return None
            return tiny_gain if action.intervention == Intervention.DOUBLE_WORKERS else baseline

    episode = run_inquiry(WeakBenefit())
    assert episode.stop_reason == "supported_interpretation"
    outcomes = {item.value: baseline for item in Intervention}
    outcomes["double_workers"] = tiny_gain
    score = assess_policy(episode, baseline, outcomes)
    assert score["evidence_backed"]
    assert score["supported_but_unhelpful"]
    assert score["no_benefit_false_positive"]
    assert not score["helpful"]
    assert score["utility"] == 0


def test_zero_benefit_and_mixed_effects_are_not_unique_cause_labels():
    baseline, outcomes = counterfactuals(
        SimulatorConfig(workers=16, db_capacity=64, lock_penalty=0, retries_enabled=False,
                        arrival_min=0, arrival_max=0),
        101,
    )
    assert all(not score_effect(baseline, outcome).beneficial for outcome in outcomes.values())
    assert all(score_effect(baseline, outcome).utility == 0 for outcome in outcomes.values())
    mixed = SimulatorConfig(db_capacity=9, lock_penalty=0.12, retry_timeout=2)
    baseline, outcomes = counterfactuals(mixed, 7)
    assert baseline.retries > 0
    assert sum(score_effect(baseline, outcome).beneficial for outcome in outcomes.values()) >= 2
    assert all(not hasattr(outcome, "cause") for outcome in outcomes.values())


def test_equal_observations_budget_and_opaque_agent_interface():
    case = generate_cases(development_per_regime=1, heldout_per_regime=1)["development"][0]
    paired = evaluate_case(case, 101)
    evidence = []
    for result in paired["policies"].values():
        episode = result["episode"]
        assert episode["budget"] == 10
        assert episode["max_steps"] == 12
        assert result["assessment"]["budget_adherent"]
        observation = next(r for r in episode["records"] if r["kind"] == "Observation")
        evidence.append(observation["data"]["metrics"])
    assert evidence[0] == evidence[1] == paired["baseline"]
    view = AgentView(QueueSimulator.from_config(SimulatorConfig()))
    assert not hasattr(view, "from_config")
    assert not hasattr(view, "parameters")
    assert not hasattr(view, "scenario")
    assert set(vars(view.snapshot())) == {"token"}


def test_preparation_freezes_original_policy_and_separate_manifests_without_results(tmp_path):
    directory = tmp_path / "prepared"
    manifest = prepare(directory)
    fingerprints = source_fingerprints()
    assert {name: fingerprints[name] for name in POLICY_NAMES} == STAGE2_HASHES
    assert manifest["configuration_counts"] == {"development": 16, "heldout": 48}
    assert not (directory / "raw.jsonl").exists()
    assert not (directory / "run-start.json").exists()
    assert check_frozen(directory) == manifest
    for name, expected in fingerprints.items():
        assert source_digest(directory / "sources" / name) == expected
        assert file_digest(directory / "sources" / name) == read_json(
            directory / "freeze.json"
        )["raw_source_sha256"][name]
    second = prepare(tmp_path / "second")
    assert second == manifest
    with pytest.raises(FileExistsError):
        prepare(directory)


def test_fingerprint_drift_prevents_execution_and_reruns_need_an_explicit_reason(tmp_path, monkeypatch):
    directory = tmp_path / "prepared"
    prepare(directory, development_per_regime=1, heldout_per_regime=1)
    modified = source_fingerprints()
    modified["benchmark.py"] = "changed"
    monkeypatch.setattr(benchmark_protocol, "source_fingerprints", lambda: modified)
    with pytest.raises(ValueError, match="fingerprint changed"):
        run_benchmark(directory, "development")
    assert not (directory / "run-start.json").exists()
    with pytest.raises(ValueError, match="rerun"):
        prepare(tmp_path / "invalid", rerun_of="prior-run")


def test_small_development_integration_raw_and_report_roundtrip(tmp_path):
    # No generated held-out policy outcomes are inspected by this integration test.
    directory = tmp_path / "development_only"
    manifest = prepare(directory, development_per_regime=1, heldout_per_regime=1)
    report = run_benchmark(directory, "development")
    rows, loaded = load_results(directory)
    assert loaded == report == summarize(rows, manifest)
    assert len(rows) == 16
    assert set(report["splits"]) == {"development"}
    split = report["splits"]["development"]
    assert split["configurations"] == 8
    assert split["paired_cases"] == 16
    assert split["agent_episodes"] == 32
    assert sum(split["configuration_utility_comparison"].values()) == 8
    assert report["oracle_compute"]["measurement_windows"] == 64
    assert all(row["manifest_sha256"] == digest(manifest) for row in rows)
    for scores in split["policies"].values():
        assert scores["budget_violations"] == 0
        assert sum(scores["stop_reasons"].values()) == 16
    assert "not external correctness" in render_report(report)
    assert render_report(loaded) == (directory / "report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_benchmark(directory, "development")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rows + rows[:1], manifest)
    with pytest.raises(ValueError, match="incomplete"):
        summarize(rows[:-1], manifest)
    (directory / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        load_results(directory)


def test_incomplete_attempt_is_not_a_valid_heldout_result(tmp_path):
    directory = tmp_path / "incomplete"
    prepare(directory, development_per_regime=1, heldout_per_regime=1)
    with pytest.raises(ValueError, match="incomplete/invalid"):
        load_results(directory)


def test_paired_intervals_resample_configuration_units_deterministically():
    units = [
        {"config_id": "one", "regime": "a", "utility": -0.2, "regret": 0.2,
         "cost": -3, "cost_adjusted_utility": -0.01},
        {"config_id": "two", "regime": "a", "utility": 0.2, "regret": -0.2,
         "cost": -1, "cost_adjusted_utility": 0.01},
    ]
    result = paired_intervals(units)
    assert result == paired_intervals(units)
    assert result["utility"]["mean_delta_heuristic_minus_scripted"] == 0
    assert result["cost"]["mean_delta_heuristic_minus_scripted"] == -2
    low, high = result["utility"]["paired_stratified_bootstrap_95"]
    assert low <= 0 <= high
    with pytest.raises(ValueError, match="empty"):
        paired_intervals([])


def test_source_identity_is_portable_but_does_not_ignore_code_changes(tmp_path):
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\nprint(x)\n")
    crlf.write_bytes(b"x = 1\r\nprint(x)\r\n")
    assert source_digest(lf) == source_digest(crlf)
    assert file_digest(lf) != file_digest(crlf)
    crlf.write_bytes(b"x = 2\r\nprint(x)\r\n")
    assert source_digest(lf) != source_digest(crlf)
