"""Explicitly scripted, bounded baseline; never reads evaluator family labels."""

from intuition_prototype.predictions import make_prediction
from intuition_prototype.records import (
    Action, ActionKind, Episode, Hypothesis, Intervention, InterventionRecord,
    Observation, Question, Record, Result,
)
from intuition_prototype.simulator import InvestigationEnvironment


def run_inquiry(environment: InvestigationEnvironment, budget: int = 10) -> Episode:
    if type(budget) is not int or not 0 <= budget <= 20:
        raise ValueError("Budget must be an integer between 0 and 20.")
    records: list[Record] = []
    spent = 0
    interpretation = "Unresolved: no evidence collected."

    def next_id() -> str:
        return f"r{len(records) + 1:03d}"

    def finish(reason: str) -> Episode:
        return Episode(tuple(records), budget, spent, reason, interpretation)

    def record_action(action: Action, question: str, prediction_id: str | None) -> str:
        nonlocal spent
        item = Question(next_id(), question)
        records.append(item)
        intervention = InterventionRecord(next_id(), action, item.id, prediction_id, action.cost)
        records.append(intervention)
        spent += action.cost
        return intervention.id

    if budget < 1:
        return finish("budget_exhausted")
    checkpoint = environment.snapshot()
    record_action(Action(ActionKind.ASK), "What is the baseline service behavior?", None)
    metrics = environment.perform(Action(ActionKind.ASK))
    if metrics is None:
        raise RuntimeError("Ask action returned no observations.")
    baseline = Observation(next_id(), "baseline_same_state_window", metrics)
    records.append(baseline)
    interpretation = "Provisional: insufficient worker capacity."
    hypothesis = Hypothesis(next_id(), interpretation, (baseline.id,))
    records.append(hypothesis)

    trials = (
        (
            Intervention.DOUBLE_WORKERS,
            "Does doubling workers improve useful completions and queue growth?",
            "Worker capacity is supported by the intervention; not proven uniquely.",
        ),
        (
            Intervention.DISABLE_RETRIES,
            "From the same state, does disabling new retries outperform baseline?",
            "Retry amplification contributes to queue growth: suppression reduces growth "
            "without reducing unique completions. Throughput recovery is not established.",
        ),
        (
            Intervention.DOUBLE_DB_CAPACITY,
            "From the same state, does doubling DB capacity improve useful service?",
            "DB contention/capacity is supported by the DB intervention.",
        ),
    )
    for intervention, question, supported in trials:
        action = Action(ActionKind.TEST, intervention)
        if spent + action.cost > budget:
            return finish("budget_exhausted")
        prediction = make_prediction(next_id(), hypothesis.id, baseline.metrics, intervention)
        records.append(prediction)
        action_id = record_action(action, question, prediction.id)
        environment.restore(checkpoint)
        outcome = environment.perform(action)
        if outcome is None:
            raise RuntimeError("Test action returned no observations.")
        observation = Observation(next_id(), intervention.value, outcome)
        records.append(observation)
        matched = prediction.matches(outcome)
        result_text = (
            "Prediction matched all preregistered criteria." if matched
            else "Discrepancy: prediction failed at least one preregistered criterion."
        )
        result = Result(
            next_id(), action_id, observation.id, prediction.id, matched, result_text,
        )
        records.append(result)
        if spent + 1 > budget:
            interpretation = "Unresolved: result recorded, but no budget for interpretation update."
            return finish("budget_exhausted")
        record_action(Action(ActionKind.REFRAME), "How should the interpretation change?", None)
        environment.perform(Action(ActionKind.REFRAME))
        if matched:
            interpretation = supported
        elif intervention == Intervention.DOUBLE_WORKERS:
            interpretation = (
                "Worker-only explanation weakened by discrepancy. "
                "Retry amplification and DB contention remain alternatives."
            )
        elif intervention == Intervention.DISABLE_RETRIES:
            interpretation = (
                "Retry suppression did not distinguish the case; investigate DB capacity next."
            )
        else:
            interpretation = "Unresolved: no available intervention met its prediction criteria."
        hypothesis = Hypothesis(
            next_id(), interpretation, (baseline.id, observation.id, result.id),
        )
        records.append(hypothesis)
        if matched:
            return finish("supported_interpretation")
    return finish("investigations_exhausted")
