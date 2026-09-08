from dataclasses import replace
import sqlite3

import pytest

from intuition_prototype.inquiry import run_inquiry
from intuition_prototype.records import (
    Action, ActionKind, Hypothesis, Intervention, InterventionRecord, Metrics,
    Observation, Prediction, Question, Result,
)
from intuition_prototype.simulator import MAX_TICKS, QueueSimulator, SCENARIOS, Snapshot
from intuition_prototype.stage1 import reset_stage1, run_stage1
from intuition_prototype.storage import (
    initialize, load_episode, read_entry, save_episode, validate_episode, write_entry,
)


def observations(episode):
    return {record.source: record for record in episode.records if isinstance(record, Observation)}


@pytest.mark.parametrize("scenario,expected,cost", [
    ("worker_capacity", "Worker capacity is supported", 4),
    ("db_contention", "DB contention/capacity is supported", 10),
    ("retry_amplification", "Retry amplification contributes", 7),
])
def test_scripted_case_families(scenario, expected, cost):
    episode = run_inquiry(QueueSimulator(scenario))
    assert episode.stop_reason == "supported_interpretation"
    assert episode.interpretation.startswith(expected)
    assert episode.cost == cost
    validate_episode(episode)
    for record in observations(episode).values():
        metrics = record.metrics
        assert metrics.queue_start + metrics.arrivals + metrics.retries == (
            metrics.queue_end + metrics.completions + metrics.duplicate_completions
            + metrics.dropped_attempts
        )


def test_retry_acceptance_trace():
    episode = run_inquiry(QueueSimulator("retry_amplification", seed=7))
    measured = observations(episode)
    baseline = measured["baseline_same_state_window"].metrics
    workers = measured["double_workers"].metrics
    retry = measured["disable_retries"].metrics
    assert (baseline.completions, workers.completions, retry.completions) == (21, 18, 21)
    assert (baseline.queue_growth, workers.queue_growth, retry.queue_growth) == (305, 315, 61)
    assert (baseline.retries, workers.retries, retry.retries) == (244, 244, 0)
    assert baseline.arrivals == workers.arrivals == retry.arrivals
    assert baseline.queue_start == workers.queue_start == retry.queue_start
    results = [record for record in episode.records if isinstance(record, Result)]
    assert [result.matched for result in results] == [False, True]
    assert "Discrepancy" in results[0].interpretation
    assert "Throughput recovery is not established" in episode.interpretation
    hypotheses = [record for record in episode.records if isinstance(record, Hypothesis)]
    assert "insufficient worker" in hypotheses[0].claim
    assert "weakened" in hypotheses[1].claim
    assert measured["disable_retries"].id in hypotheses[-1].evidence_ids
    positions = {record.id: index for index, record in enumerate(episode.records)}
    for result in results:
        assert positions[result.prediction_id] < positions[result.intervention_id]
        assert positions[result.intervention_id] < positions[result.observation_id]
        assert positions[result.observation_id] < positions[result.id]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_deterministic_reset_snapshot_and_replay(scenario):
    simulator = QueueSimulator(scenario, 17)
    first = run_inquiry(simulator)
    simulator.reset()
    assert run_inquiry(simulator) == first
    assert run_inquiry(QueueSimulator(scenario, 17)) == first
    simulator.reset()
    checkpoint = simulator.snapshot()
    assert set(vars(checkpoint)) == {"token"}
    baseline = simulator.perform(Action(ActionKind.ASK))
    simulator.restore(checkpoint)
    assert simulator.perform(Action(ActionKind.ASK)) == baseline


@pytest.mark.parametrize("intervention", list(Intervention))
def test_allowed_interventions_and_fair_arrival_replay(intervention):
    simulator = QueueSimulator("retry_amplification")
    checkpoint = simulator.snapshot()
    baseline = simulator.perform(Action(ActionKind.ASK))
    simulator.restore(checkpoint)
    changed = simulator.perform(Action(ActionKind.TEST, intervention))
    assert changed.arrivals == baseline.arrivals
    assert changed.queue_start == baseline.queue_start
    simulator.restore(checkpoint)
    assert simulator.perform(Action(ActionKind.TEST, intervention)) == changed
    assert simulator.perform(Action(ActionKind.REFRAME)) is None


@pytest.mark.parametrize("factory", [
    lambda: Action("shell"),
    lambda: Action(ActionKind.TEST),
    lambda: Action(ActionKind.TEST, "delete_database"),
    lambda: Action(ActionKind.ASK, Intervention.DOUBLE_WORKERS),
    lambda: QueueSimulator("unknown"),
    lambda: QueueSimulator("worker_capacity", -1),
    lambda: QueueSimulator("worker_capacity", 1.5),
    lambda: QueueSimulator("worker_capacity", True),
])
def test_invalid_actions_and_inputs_are_explicit(factory):
    with pytest.raises(ValueError):
        factory()


def test_snapshot_ownership_expiry_and_limits():
    simulator = QueueSimulator("worker_capacity")
    checkpoint = simulator.snapshot()
    with pytest.raises(ValueError, match="Unknown or expired"):
        QueueSimulator("worker_capacity").restore(checkpoint)
    simulator.reset()
    with pytest.raises(ValueError, match="Unknown or expired"):
        simulator.restore(checkpoint)
    with pytest.raises(ValueError, match="Unknown or expired"):
        simulator.restore(Snapshot("unknown"))
    for _ in range(8):
        simulator.snapshot()
    with pytest.raises(ValueError, match="Snapshot limit"):
        simulator.snapshot()


def test_tick_limit_and_repeated_scaling_are_explicit():
    simulator = QueueSimulator("worker_capacity")
    for _ in range(MAX_TICKS // 30 - 1):
        simulator.perform(Action(ActionKind.ASK))
    with pytest.raises(ValueError, match="tick limit"):
        simulator.perform(Action(ActionKind.ASK))
    simulator.reset()
    for intervention in (Intervention.DOUBLE_WORKERS, Intervention.DOUBLE_DB_CAPACITY):
        simulator.reset()
        simulator.perform(Action(ActionKind.TEST, intervention))
        with pytest.raises(ValueError, match="already increased"):
            simulator.perform(Action(ActionKind.TEST, intervention))
    with pytest.raises(ValueError, match="validated Action"):
        simulator.perform("shell")


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 5, 6])
def test_budget_termination(budget):
    episode = run_inquiry(QueueSimulator("retry_amplification"), budget)
    assert episode.stop_reason == "budget_exhausted"
    assert episode.cost <= budget
    assert sum(
        record.cost for record in episode.records if isinstance(record, InterventionRecord)
    ) == episode.cost
    validate_episode(episode)


@pytest.mark.parametrize("budget", [-1, 21, 1.2, True])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match="Budget"):
        run_inquiry(QueueSimulator("worker_capacity"), budget)


class UnresponsiveEnvironment:
    """No family labels or parameters; all interventions return unchanged evidence."""

    def snapshot(self):
        return Snapshot("opaque")

    def restore(self, snapshot):
        assert snapshot.token == "opaque"

    def perform(self, action):
        if action.kind == ActionKind.REFRAME:
            return None
        return Metrics(30, 120, 20, 100, 10, 30, 220, 0, 12.0)


def test_exhaustion_and_no_revision_without_distinguishing_evidence():
    episode = run_inquiry(UnresponsiveEnvironment(), 20)
    assert episode.stop_reason == "investigations_exhausted"
    assert episode.cost == 10
    assert episode.interpretation.startswith("Unresolved")
    assert all(not record.matched for record in episode.records if isinstance(record, Result))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_persist_typed_episode_and_preserve_original_storage(tmp_path, scenario):
    database = tmp_path / "inquiry.sqlite3"
    initialize(database)
    entry = write_entry(database, "existing demo entry")
    episode = run_inquiry(QueueSimulator(scenario))
    episode_id = save_episode(database, scenario, 7, episode)
    loaded = load_episode(database, episode_id)
    assert loaded.episode == episode
    assert (loaded.scenario, loaded.seed) == (scenario, 7)
    assert read_entry(database, entry) == "existing demo entry"
    assert {type(record) for record in loaded.episode.records} == {
        Observation, Hypothesis, Prediction, Question, InterventionRecord, Result,
    }
    with pytest.raises(LookupError, match="not found"):
        load_episode(database, "' OR 1=1 --")


def test_reject_broken_claim_references_and_corrupt_record_kind(tmp_path):
    database = tmp_path / "corrupt.sqlite3"
    episode = run_inquiry(QueueSimulator("worker_capacity"))
    records = list(episode.records)
    index = next(i for i, record in enumerate(records) if isinstance(record, Hypothesis))
    records[index] = replace(records[index], evidence_ids=("missing",))
    with pytest.raises(ValueError, match="reference"):
        save_episode(database, "worker_capacity", 7, replace(episode, records=tuple(records)))
    episode_id = save_episode(database, "worker_capacity", 7, episode)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE inquiry_records SET kind = 'InventedClaim' WHERE position = 0")
    with pytest.raises(ValueError, match="Unknown persisted record"):
        load_episode(database, episode_id)


def test_reject_result_that_contradicts_prediction_and_evidence(tmp_path):
    episode = run_inquiry(QueueSimulator("retry_amplification"))
    records = tuple(
        replace(record, matched=not record.matched) if isinstance(record, Result) else record
        for record in episode.records
    )
    with pytest.raises(ValueError, match="contradicts"):
        save_episode(
            tmp_path / "invalid.sqlite3", "retry_amplification", 7,
            replace(episode, records=records),
        )


def test_ui_orchestration_persists_and_reset_replays_without_deleting_history(tmp_path):
    database = tmp_path / "ui.sqlite3"
    first_id, trace = run_stage1(database, "retry_amplification", 7, 10)
    for word in ("SCRIPTED", "Question", "Prediction", "Observation", "Discrepancy", "Cost:", "Stop:"):
        assert word in trace
    assert "does not validate the research hypothesis" in trace
    assert reset_stage1("retry_amplification", 7)[0] == ""
    second_id, _ = run_stage1(database, "retry_amplification", 7, 10)
    assert first_id != second_id
    assert load_episode(database, first_id).episode == load_episode(database, second_id).episode
