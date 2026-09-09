from dataclasses import asdict, replace
import itertools
import json
import subprocess
import sys

import pytest

from intuition_prototype.benchmark import counterfactuals, score_effect
from intuition_prototype.benchmark_protocol import digest, generate_cases
from intuition_prototype.config import Settings
from intuition_prototype.learned import (
    FEATURE_NAMES, LearnedModel, TrainingExample, features, fit_model, load_model, run_learned,
)
from intuition_prototype.navigator import choose_candidate
from intuition_prototype.records import (
    Action, ActionKind, DecisionRecord, FuseRecord, Hypothesis, Intervention, LearnedCandidateScore, Metrics,
)
from intuition_prototype.simulator import QueueSimulator, SimulatorConfig, Snapshot
from intuition_prototype.stage1 import render_trace, run_stage1
from intuition_prototype.storage import load_episode, save_episode, validate_episode


@pytest.fixture
def examples():
    configurations = [
        SimulatorConfig(workers=4, db_capacity=30, arrival_min=4, arrival_max=7, retries_enabled=False),
        SimulatorConfig(workers=10, db_capacity=3, lock_penalty=0.1, retries_enabled=False),
        SimulatorConfig(workers=4, db_capacity=8, lock_penalty=0.13, retry_timeout=2),
        SimulatorConfig(workers=12, db_capacity=40, arrival_min=0, arrival_max=1, retries_enabled=False),
    ]
    rows = []
    for config in configurations:
        for seed in (201, 301):
            baseline, outcomes = counterfactuals(config, seed)
            rows.append(TrainingExample(
                digest(asdict(config))[:20], seed, "train", baseline,
                {key: score_effect(baseline, outcome).utility for key, outcome in outcomes.items()},
            ))
    return rows


def test_fitting_is_real_deterministic_and_order_stable(examples):
    model = fit_model(examples, 0.01, {"purpose": "development-only test"})
    assert model.document() == fit_model(list(reversed(examples)), 0.01, {
        "purpose": "development-only test",
    }).document()
    assert any(abs(coefficient) > 1e-5 for parameters in model.document()["models"].values()
               for coefficient in parameters["coefficients"])
    changed = [replace(row, utilities={key: 0.0 for key in row.utilities}) for row in examples]
    zero = fit_model(changed, 0.01, {"purpose": "development-only test"})
    assert zero.fingerprint != model.fingerprint
    assert all(parameters["intercept"] == 0 for parameters in zero.document()["models"].values())
    evidence = [examples[index].observation for index in (0, 2, 4)]
    predictions = [
        tuple(item.predicted_utility for item in model.score_candidates(item, tuple(Intervention), {}, 10))
        for item in evidence
    ]
    assert len(set(predictions)) == 3


@pytest.mark.parametrize("split", ["validation", "test", "heldout", "development"])
def test_fit_rejects_nontraining_partitions(examples, split):
    with pytest.raises(ValueError, match="never validation/test"):
        fit_model([replace(examples[0], split=split)], 1, {})


def test_fit_rejects_historical_configs_and_reweighted_rows(examples):
    historical = generate_cases()["heldout"][0]["config_id"]
    with pytest.raises(ValueError, match="historical"):
        fit_model([replace(examples[0], config_id=historical)], 1, {})
    with pytest.raises(ValueError, match="Duplicate"):
        fit_model([examples[0], examples[0]], 1, {})
    with pytest.raises(ValueError, match="positive"):
        fit_model(examples, 0, {})


def test_schema_roundtrip_integrity_and_fail_closed_loading(examples, tmp_path):
    model = fit_model(examples, 0.1, {})
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model.document()), encoding="utf-8")
    assert load_model(path).document() == model.document()
    with pytest.raises(ValueError, match="missing"):
        load_model(tmp_path / "absent.json")
    corrupt = model.document()
    corrupt["threshold"] = 99
    with pytest.raises(ValueError, match="fingerprint"):
        LearnedModel(corrupt)
    for key, value in [("version", "future"), ("feature_names", []), ("scales", [0] * 10),
                       ("models", []), ("provenance", {})]:
        body = {key: value for key, value in model.document().items() if key != "model_sha256"}
        body[key] = value
        with pytest.raises(ValueError):
            LearnedModel({**body, "model_sha256": digest(body)})
    with pytest.raises(ValueError, match="leaks"):
        model.with_selection(0.02, [examples[0].config_id])
    body = {key: value for key, value in model.document().items() if key != "model_sha256"}
    body["provenance"]["training_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="incompatible"):
        LearnedModel({**body, "model_sha256": digest(body)})


def test_model_provenance_and_training_ids_do_not_drive_inference(examples):
    first = fit_model(examples, 0.1, {"irrelevant_evaluator_label": "capacity"})
    second = fit_model(examples, 0.1, {"irrelevant_evaluator_label": "retries"})
    assert first.fingerprint != second.fingerprint
    for capabilities in itertools.permutations(Intervention):
        left = first.score_candidates(examples[0].observation, capabilities, {}, 10)
        right = second.score_candidates(examples[0].observation, capabilities, {}, 10)
        assert [(item.key, item.score) for item in left] == [(item.key, item.score) for item in right]
        assert choose_candidate(left).key == choose_candidate(tuple(reversed(left))).key
    assert set(FEATURE_NAMES).isdisjoint({"scenario", "parameters", "seed", "regime", "config_id"})
    with pytest.raises(ValueError, match="telemetry"):
        features({"scenario": "capacity"})


class OpaqueEnvironment:
    __slots__ = ("recovery", "calls")

    def __init__(self, recovery=False):
        self.recovery, self.calls = recovery, []

    def capabilities(self):
        return tuple(Intervention)

    def snapshot(self):
        return Snapshot("opaque")

    def restore(self, snapshot):
        assert snapshot == Snapshot("opaque")

    def perform(self, action):
        self.calls.append(action)
        assert action.kind in (ActionKind.ASK, ActionKind.TEST)
        completion = 60 if self.recovery and action.intervention == Intervention.DOUBLE_WORKERS else 20
        return Metrics(30, 100, completion, 0, 0, 20, 120 - completion, 0, 10.0)


def constant_model():
    evidence = OpaqueEnvironment().perform(Action(ActionKind.ASK))
    return fit_model([TrainingExample("unit-fixture", 0, "train", evidence, {
        "double_db_capacity": 0.8, "double_workers": 0.6, "disable_retries": 0.0,
    })], 1, {"purpose": "synthetic unit fixture, not evaluation evidence"})


def test_fuse_recovers_after_contradiction_without_hidden_information():
    model = constant_model()
    episode = run_learned(OpaqueEnvironment(recovery=True), model)
    assert episode == run_learned(OpaqueEnvironment(recovery=True), model)
    choices = [record.chosen_key for record in episode.records
               if isinstance(record, DecisionRecord) and record.chosen_key]
    assert choices == ["double_db_capacity", "double_workers"]
    fuses = [record for record in episode.records if isinstance(record, FuseRecord)]
    assert fuses[0].triggers == ("contradictory_prediction", "no_progress")
    assert fuses[0].disposition == "switch_path"
    assert episode.stop_reason == "supported_interpretation"
    assert episode.cost == 7
    claims = [record.claim for record in episode.records if isinstance(record, Hypothesis)]
    assert any("Weakened assumption: The DB resource" in claim for claim in claims)
    assert any("Worker capacity" in claim for claim in claims)
    validate_episode(episode)


@pytest.mark.parametrize("budget,steps", [(0, 12), (1, 12), (3, 12), (4, 12), (7, 12), (20, 12), (10, 1), (10, 4)])
def test_learned_bounds_and_no_infinite_reframing(budget, steps):
    episode = run_learned(OpaqueEnvironment(), constant_model(), budget, steps)
    assert episode.cost <= budget
    validate_episode(episode)
    assert episode.stop_reason != "supported_interpretation"
    assert len([record for record in episode.records if isinstance(record, FuseRecord)]) <= 2


def test_abstention_legality_ties_and_contribution_validation():
    model = constant_model()
    evidence = Metrics(30, 100, 20, 0, 0, 20, 100, 0, 10.0)
    candidates = model.score_candidates(evidence, (Intervention.DOUBLE_WORKERS,), {}, 2)
    assert choose_candidate(candidates) is None
    assert all(not candidate.eligible for candidate in candidates)
    abstain = run_learned(OpaqueEnvironment(), model.with_selection(1, ["validation-fixture"]))
    assert abstain.cost == 1
    assert abstain.stop_reason == "inconclusive"
    selected = choose_candidate(model.score_candidates(evidence, tuple(Intervention), {}, 10))
    assert isinstance(selected, LearnedCandidateScore)
    with pytest.raises(ValueError, match="contributions"):
        replace(selected, predicted_utility=selected.predicted_utility + 1)
    other = replace(selected, key="double_workers",
                    action=replace(selected.action, intervention=Intervention.DOUBLE_WORKERS))
    assert choose_candidate((other, selected)).key == choose_candidate((selected, other)).key
    assert choose_candidate((other, selected)).key == "double_db_capacity"


def test_persistence_rejects_mixed_policy_and_model_provenance():
    episode = run_learned(OpaqueEnvironment(recovery=True), constant_model())
    with pytest.raises(ValueError, match="schema"):
        validate_episode(replace(episode, policy="heuristic"))
    records = list(episode.records)
    decisions = [index for index, record in enumerate(records) if isinstance(record, DecisionRecord)]
    later = records[decisions[-1]]
    records[decisions[-1]] = replace(later, candidates=tuple(
        replace(candidate, model_sha256="0" * 64) for candidate in later.candidates
    ))
    with pytest.raises(ValueError, match="mix model"):
        validate_episode(replace(episode, records=tuple(records)))


def test_learned_typed_storage_trace_cli_and_baseline_without_model(examples, tmp_path):
    model = fit_model(examples, 0.1, {})
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model.document()), encoding="utf-8")
    database = tmp_path / "episodes.sqlite3"
    episode = run_learned(QueueSimulator("retry_amplification"), model)
    identity = save_episode(database, "retry_amplification", 7, episode)
    saved = load_episode(database, identity)
    assert saved.episode == episode
    markdown = render_trace(saved)
    assert "Predicted utility" in markdown
    assert "standardized feature contributions" in markdown
    assert model.fingerprint in markdown
    for policy in ("scripted", "heuristic"):
        identity, _ = run_stage1(database, "retry_amplification", 7, 10, policy,
                                model_path=tmp_path / "missing.json")
        assert load_episode(database, identity).episode.policy == policy
    with pytest.raises(ValueError, match="missing"):
        run_stage1(database, "retry_amplification", 7, 10, "learned",
                   model_path=tmp_path / "missing.json")
    process = subprocess.run([
        sys.executable, "-m", "intuition_prototype.stage1", "--policy", "learned",
        "--model", str(model_path), "--database", str(database), "--json",
    ], capture_output=True, text=True, check=True)
    assert json.loads(process.stdout)["episode"]["policy"] == "learned"
    assert Settings.from_env({"INTUITION_MODEL_PATH": str(model_path)}).model_path == model_path
    with pytest.raises(ValueError, match="must not be empty"):
        Settings.from_env({"INTUITION_MODEL_PATH": " "})
