from dataclasses import asdict, replace
import itertools
import json
import sqlite3

import pytest

from intuition_prototype.inquiry import run_inquiry
from intuition_prototype.navigator import choose_candidate, run_navigator, score_candidates
from intuition_prototype.records import (
    Action, ActionKind, AnswerRecord, DecisionRecord, FuseRecord, Hypothesis,
    Intervention, InterventionRecord, Metrics, Observation, Result,
)
from intuition_prototype.simulator import QueueSimulator, SCENARIOS, Snapshot
from intuition_prototype.stage1 import run_stage1
from intuition_prototype.storage import initialize, load_episode, save_episode, validate_episode


def metrics(completions=75, retries=0, growth=25):
    return Metrics(30, 100, completions, retries, 0, 20, 20 + growth, 0, 10.0)


def selections(episode):
    return [
        record.chosen_key for record in episode.records
        if isinstance(record, DecisionRecord) and record.chosen_key is not None
    ]


@pytest.mark.parametrize("evidence,expected", [
    (metrics(), "double_workers"),
    (metrics(completions=20, growth=80), "double_db_capacity"),
    (metrics(completions=20, retries=200, growth=280), "disable_retries"),
])
def test_selection_depends_on_observed_evidence_not_constant_order(evidence, expected):
    candidates = score_candidates(evidence, tuple(Intervention), {}, 10)
    assert choose_candidate(candidates).key == expected
    for candidate in candidates:
        assert "heuristic" in candidate.rationale
        assert "probabilities" in candidate.rationale


def test_tie_and_input_order_stability():
    for capabilities in itertools.permutations(Intervention):
        scored = score_candidates(metrics(), capabilities, {}, 10)
        assert choose_candidate(scored).key == "double_workers"
        assert choose_candidate(tuple(reversed(scored))).key == "double_workers"
    worker = next(item for item in scored if item.key == "double_workers")
    database = next(item for item in scored if item.key == "double_db_capacity")
    database = replace(database, anomaly_relevance=1.0, score=worker.score)
    assert choose_candidate((worker, database)).key == "double_db_capacity"
    assert choose_candidate((database, worker)).key == "double_db_capacity"
    with pytest.raises(ValueError, match="unique"):
        choose_candidate((worker, worker))


def test_hypothesis_state_changes_gap_novelty_and_redundancy():
    candidates = score_candidates(
        metrics(), tuple(Intervention), {Intervention.DOUBLE_WORKERS: "weakened"}, 10,
    )
    worker = next(item for item in candidates if item.key == "double_workers")
    database = next(item for item in candidates if item.key == "double_db_capacity")
    assert not worker.eligible
    assert (worker.evidence_gap, worker.novelty, worker.redundancy) == (0, 0, 1)
    assert database.discrimination == 0.75
    assert choose_candidate(candidates).key == "double_db_capacity"


def test_capability_applicability_and_affordability_rejections_are_recorded():
    candidates = score_candidates(metrics(), (Intervention.DOUBLE_WORKERS,), {}, 2)
    assert choose_candidate(candidates) is None
    assert "unaffordable:" in " ".join(candidates[-1].rejection_reasons)
    retry = next(item for item in candidates if item.key == "disable_retries")
    assert "unsupported capability" in retry.rejection_reasons
    assert "no observed retry injection to suppress" in retry.rejection_reasons


@pytest.mark.parametrize("scenario,selected", [
    ("worker_capacity", "double_workers"),
    ("db_contention", "double_db_capacity"),
    ("retry_amplification", "disable_retries"),
])
def test_heuristic_family_replay_and_claim_scope(scenario, selected):
    simulator = QueueSimulator(scenario, 7)
    first = run_navigator(simulator)
    simulator.reset()
    assert first == run_navigator(simulator)
    assert first == run_navigator(QueueSimulator(scenario, 7))
    assert selections(first) == [selected]
    assert first.policy == "heuristic"
    assert first.cost == 4
    assert first.stop_reason == "supported_interpretation"
    assert "unique cause is not established" in first.interpretation
    if selected == "disable_retries":
        assert "Throughput recovery is not established" in first.interpretation
        evidence = [r.metrics for r in first.records if isinstance(r, Observation)]
        assert [(m.completions, m.retries, m.queue_growth) for m in evidence] == [
            (21, 244, 305), (21, 0, 61),
        ]
    validate_episode(first)


class LabelFreeEnvironment:
    """Test double with only the permitted API; it has no scenario or parameter access."""

    __slots__ = ("calls", "_baseline", "_recovery")

    def __init__(self, baseline=None, recovery=False):
        self.calls = []
        self._baseline = metrics(completions=20, growth=80) if baseline is None else baseline
        self._recovery = recovery

    def capabilities(self):
        self.calls.append("capabilities")
        return tuple(Intervention)

    def snapshot(self):
        self.calls.append("snapshot")
        return Snapshot("opaque-token")

    def restore(self, checkpoint):
        assert checkpoint == Snapshot("opaque-token")
        self.calls.append("restore")

    def perform(self, action):
        self.calls.append(action)
        assert action.kind in (ActionKind.ASK, ActionKind.TEST)
        if self._recovery and action.intervention == Intervention.DOUBLE_WORKERS:
            return metrics(completions=60, growth=40)
        return self._baseline


def test_contradiction_fuse_switches_path_with_meaningful_reframe():
    environment = LabelFreeEnvironment(recovery=True)
    episode = run_navigator(environment)
    assert selections(episode) == ["double_db_capacity", "double_workers"]
    fuses = [r for r in episode.records if isinstance(r, FuseRecord)]
    assert fuses[0].triggers == ("contradictory_prediction", "no_progress")
    assert fuses[0].alternate_keys == ("double_workers",)
    assert fuses[0].disposition == "switch_path"
    assert fuses[-1].disposition == "answer"
    assert episode.stop_reason == "supported_interpretation"
    hypotheses = [r for r in episode.records if isinstance(r, Hypothesis)]
    assert any("Weakened assumption: The DB resource" in r.claim for r in hypotheses)
    assert any("Working assumption to test (not a fact): Worker capacity" in r.claim for r in hypotheses)
    assert "Worker capacity" in episode.interpretation
    assert len({r.claim for r in hypotheses}) == len(hypotheses)
    assert environment.calls.count("restore") == 2
    validate_episode(episode)


def test_repeated_no_progress_exhausts_alternatives_not_an_endless_reframe_loop():
    episode = run_navigator(LabelFreeEnvironment())
    assert selections(episode) == ["double_db_capacity", "double_workers"]
    fuses = [r for r in episode.records if isinstance(r, FuseRecord)]
    assert fuses[-1].no_progress_count == 2
    assert "repeated_no_progress" in fuses[-1].triggers
    assert fuses[-1].disposition == "investigations_exhausted"
    assert episode.stop_reason == "investigations_exhausted"
    assert episode.interpretation.startswith("Unresolved")
    assert episode.cost == 7
    assert len([r for r in episode.records if isinstance(r, InterventionRecord)]) == 6
    assert not any(r.matched for r in episode.records if isinstance(r, Result))
    validate_episode(episode)


def test_repeated_no_progress_still_considers_a_third_affordable_alternative():
    episode = run_navigator(
        LabelFreeEnvironment(baseline=metrics(completions=20, retries=200, growth=280)),
    )
    assert len(selections(episode)) == 3
    fuses = [record for record in episode.records if isinstance(record, FuseRecord)]
    assert fuses[1].no_progress_count == 2
    assert fuses[1].disposition == "switch_path"
    assert fuses[1].alternate_keys
    assert fuses[2].no_progress_count == 3
    assert fuses[2].disposition == "investigations_exhausted"
    assert episode.cost == 10
    validate_episode(episode)


def test_runner_works_with_only_public_observation_action_and_opaque_snapshot_api():
    simulator = QueueSimulator("db_contention")

    class RestrictedAPI:
        __slots__ = ()

        def capabilities(self):
            return simulator.capabilities()

        def snapshot(self):
            return simulator.snapshot()

        def restore(self, checkpoint):
            simulator.restore(checkpoint)

        def perform(self, action):
            return simulator.perform(action)

    restricted = RestrictedAPI()
    assert not hasattr(restricted, "__dict__")
    assert not hasattr(restricted, "_state")
    assert not hasattr(restricted, "_scenario")
    assert run_navigator(restricted) == run_navigator(QueueSimulator("db_contention"))


def test_fuse_checks_alternatives_but_cannot_spend_beyond_budget():
    episode = run_navigator(LabelFreeEnvironment(), budget=4)
    assert selections(episode) == ["double_db_capacity"]
    fuse = next(r for r in episode.records if isinstance(r, FuseRecord))
    assert fuse.disposition == "budget_exhausted"
    assert not fuse.alternate_keys
    decision = [r for r in episode.records if isinstance(r, DecisionRecord)][-1]
    worker = next(c for c in decision.candidates if c.key == "double_workers")
    assert "unaffordable:" in " ".join(worker.rejection_reasons)
    assert episode.cost == 4
    assert episode.stop_reason == "budget_exhausted"


def test_healthy_observation_is_inconclusive_not_a_proven_cause():
    episode = run_navigator(LabelFreeEnvironment(baseline=metrics(completions=100, growth=0)))
    assert episode.cost == 1
    assert not selections(episode)
    assert episode.stop_reason == "inconclusive"
    assert episode.interpretation.startswith("Unresolved")


@pytest.mark.parametrize("budget", range(0, 11))
def test_budget_boundaries(budget):
    episode = run_navigator(LabelFreeEnvironment(), budget=budget)
    assert episode.cost <= budget
    assert sum(r.cost for r in episode.records if isinstance(r, InterventionRecord)) == episode.cost
    validate_episode(episode)
    if budget < 4:
        assert episode.stop_reason == "budget_exhausted"
        assert not selections(episode)


@pytest.mark.parametrize("limit", range(1, 13))
def test_step_boundaries_and_no_unfinished_investigation_admission(limit):
    episode = run_navigator(LabelFreeEnvironment(), max_steps=limit)
    actions = [r for r in episode.records if isinstance(r, InterventionRecord)]
    assert len(actions) <= limit
    validate_episode(episode)
    if limit < 4:
        assert not selections(episode)
        assert episode.stop_reason == "step_limit"
    if limit == 4:
        fuse = next(r for r in episode.records if isinstance(r, FuseRecord))
        assert fuse.disposition == "step_limit"


@pytest.mark.parametrize("kwargs", [
    {"budget": -1}, {"budget": 21}, {"budget": True},
    {"max_steps": 0}, {"max_steps": 13}, {"max_steps": 2.5},
])
def test_invalid_navigation_bounds(kwargs):
    with pytest.raises(ValueError):
        run_navigator(LabelFreeEnvironment(), **kwargs)


def test_invalid_capabilities_and_local_only_terminal_actions():
    with pytest.raises(ValueError, match="Capabilities"):
        score_candidates(metrics(), ("shell",), {}, 10)
    for kind in (ActionKind.ANSWER, ActionKind.STOP):
        action = Action(kind)
        assert action.cost == 0
        with pytest.raises(ValueError, match="local policy"):
            QueueSimulator("worker_capacity").perform(action)


def test_new_records_round_trip_and_corrupt_scores_fail(tmp_path):
    database = tmp_path / "heuristic.sqlite3"
    episode = run_navigator(LabelFreeEnvironment())
    episode_id = save_episode(database, "evaluator_metadata_only", 7, episode)
    assert load_episode(database, episode_id).episode == episode
    assert any(isinstance(r, DecisionRecord) for r in episode.records)
    assert any(isinstance(r, FuseRecord) for r in episode.records)
    assert any(isinstance(r, AnswerRecord) for r in episode.records)
    with sqlite3.connect(database) as connection:
        position, payload = connection.execute(
            "SELECT position, payload FROM inquiry_records WHERE kind = 'DecisionRecord' LIMIT 1"
        ).fetchone()
        fields = json.loads(payload)
        fields["candidates"][0]["score"] = 999
        connection.execute(
            "UPDATE inquiry_records SET payload = ? WHERE position = ?",
            (json.dumps(fields), position),
        )
    with pytest.raises(ValueError, match="score contradicts"):
        load_episode(database, episode_id)


def test_legacy_stage1_database_migrates_without_changing_records(tmp_path):
    database = tmp_path / "stage1.sqlite3"
    original = run_inquiry(QueueSimulator("retry_amplification"))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE inquiry_episodes (id TEXT PRIMARY KEY, scenario TEXT NOT NULL, "
            "seed INTEGER NOT NULL, budget INTEGER NOT NULL, cost INTEGER NOT NULL, "
            "stop_reason TEXT NOT NULL, interpretation TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE inquiry_records (episode_id TEXT NOT NULL, position INTEGER NOT NULL, "
            "record_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, "
            "PRIMARY KEY (episode_id, position), UNIQUE (episode_id, record_id))"
        )
        connection.execute(
            "INSERT INTO inquiry_episodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("legacy", "retry_amplification", 7, 10, 7, original.stop_reason, original.interpretation),
        )
        connection.executemany(
            "INSERT INTO inquiry_records VALUES (?, ?, ?, ?, ?)",
            [
                ("legacy", index, r.id, type(r).__name__, json.dumps(asdict(r)))
                for index, r in enumerate(original.records)
            ],
        )
    loaded = load_episode(database, "legacy")
    assert loaded.episode == original
    assert loaded.episode.policy == "scripted"
    initialize(database)
    assert load_episode(database, "legacy") == loaded
    new_id = save_episode(database, "worker_capacity", 7, run_navigator(QueueSimulator("worker_capacity")))
    assert load_episode(database, new_id).episode.policy == "heuristic"
    assert load_episode(database, "legacy") == loaded


@pytest.mark.parametrize("scenario,cost,completions", [
    ("worker_capacity", 4, [90, 144]),
    ("db_contention", 10, [24, 16, 24, 46]),
    ("retry_amplification", 7, [21, 18, 21]),
])
def test_scripted_baseline_remains_unchanged(scenario, cost, completions):
    episode = run_inquiry(QueueSimulator(scenario))
    assert episode.policy == "scripted"
    assert episode.cost == cost
    assert [r.metrics.completions for r in episode.records if isinstance(r, Observation)] == completions
    assert not any(isinstance(r, (DecisionRecord, FuseRecord, AnswerRecord)) for r in episode.records)


@pytest.mark.parametrize("policy", ("scripted", "heuristic"))
def test_policy_orchestration_and_readable_trace(tmp_path, policy):
    database = tmp_path / "ui.sqlite3"
    episode_id, trace = run_stage1(database, "retry_amplification", 7, 10, policy)
    assert load_episode(database, episode_id).episode.policy == policy
    assert policy.upper() in trace
    if policy == "heuristic":
        assert all(term in trace for term in ("Candidate", "Score", "Intelligence fuse", "AnswerRecord"))
    with pytest.raises(ValueError, match="Unknown policy"):
        run_stage1(database, "retry_amplification", 7, 10, "unknown")
