"""Transparent hand-designed utility estimates and an evidence-path fuse, not learning."""

from typing import Mapping, Protocol, Sequence

from intuition_prototype.predictions import make_prediction
from intuition_prototype.records import (
    Action, ActionKind, AnswerRecord, CandidateScore, DecisionRecord, Episode, FuseRecord,
    Hypothesis, Intervention, InterventionRecord, Metrics, Observation, Question, Record, Result,
    candidate_utility,
)
from intuition_prototype.simulator import InvestigationEnvironment


class NavigatorEnvironment(InvestigationEnvironment, Protocol):
    def capabilities(self) -> tuple[Intervention, ...]: ...


_QUESTIONS = {
    Intervention.DOUBLE_WORKERS: (
        "Would doubling workers improve unique completions and queue growth?",
        "Worker capacity limits useful service.",
    ),
    Intervention.DISABLE_RETRIES: (
        "Would preventing new retries reduce queue growth without hurting unique completions?",
        "Retry injection contributes to queue pressure; old duplicates will remain.",
    ),
    Intervention.DOUBLE_DB_CAPACITY: (
        "Would doubling DB capacity improve unique completions and queue growth?",
        "The DB resource, rather than worker count alone, constrains useful service.",
    ),
}


def score_candidates(
    baseline: Metrics, capabilities: Sequence[Intervention],
    hypothesis_states: Mapping[Intervention, str], remaining_budget: int,
    remaining_steps: int = 12,
) -> tuple[CandidateScore, ...]:
    if any(not isinstance(item, Intervention) for item in capabilities):
        raise ValueError("Capabilities must contain only supported Intervention values.")
    if any(state not in ("untested", "weakened", "supported") for state in hypothesis_states.values()):
        raise ValueError("Unknown hypothesis state.")
    arrival_scale = max(1, baseline.arrivals)
    service_ratio = baseline.completions / arrival_scale
    retry_load = min(1.0, baseline.retries / arrival_scale)
    anomaly = baseline.queue_growth > 0 or baseline.completions < baseline.arrivals
    contradicted = "weakened" in hypothesis_states.values()
    candidates = []
    for intervention in sorted(Intervention, key=lambda item: item.value):
        state = hypothesis_states.get(intervention, "untested")
        untested = state == "untested"
        if intervention == Intervention.DISABLE_RETRIES:
            discrimination = 0.9 if baseline.retries else 0.1
            relevance = retry_load
        elif intervention == Intervention.DOUBLE_WORKERS:
            discrimination = 0.6
            relevance = 1.0 if service_ratio >= 0.5 and not baseline.retries else 0.3
        else:
            discrimination = 0.6
            relevance = 1.0 if service_ratio < 0.5 else 0.25
        if contradicted and untested:
            discrimination = min(1.0, discrimination + 0.15)
        reasons = []
        if intervention not in capabilities:
            reasons.append("unsupported capability")
        if not untested:
            reasons.append(f"already tested ({state}); same-state repetition is redundant")
        if not anomaly:
            reasons.append("no queue/service anomaly in the observed window")
        if intervention == Intervention.DISABLE_RETRIES and not baseline.retries:
            reasons.append("no observed retry injection to suppress")
        action = Action(ActionKind.TEST, intervention)
        if remaining_budget < action.cost + Action(ActionKind.REFRAME).cost:
            reasons.append("unaffordable: test plus one evidence update requires 3 units")
        if remaining_steps < 3:
            reasons.append("step limit: test, reframe, and terminal action require 3 steps")
        question, assumption = _QUESTIONS[intervention]
        components = (
            discrimination, relevance if anomaly else 0.0,
            float(untested), float(untested), float(not untested),
        )
        candidates.append(CandidateScore(
            intervention.value, action, question, assumption, *components,
            action.cost, candidate_utility(*components, action.cost), not reasons, tuple(reasons),
            f"Observed completion/arrival ratio={service_ratio:.3f}, retry load={retry_load:.3f}; "
            f"hypothesis={state}; alternative-gap bonus={0.15 if contradicted and untested else 0:.2f}. "
            "Scores are heuristic priorities, not probabilities or causal truth.",
        ))
    return tuple(candidates)


def choose_candidate(candidates: Sequence[CandidateScore]) -> CandidateScore | None:
    if len({candidate.key for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate keys must be unique.")
    eligible = [candidate for candidate in candidates if candidate.eligible]
    return min(eligible, key=lambda item: (-item.score, item.key)) if eligible else None


def _stop_reason(candidates: Sequence[CandidateScore]) -> str:
    # Ignore affordability once to distinguish a budget boundary from exhausted hypotheses.
    if any(
        candidate.rejection_reasons
        and all(reason.startswith(("unaffordable:", "step limit:")) for reason in candidate.rejection_reasons)
        and any(reason.startswith("step limit:") for reason in candidate.rejection_reasons)
        for candidate in candidates
    ):
        return "step_limit"
    if any(
        candidate.rejection_reasons
        and all(reason.startswith("unaffordable:") for reason in candidate.rejection_reasons)
        for candidate in candidates
    ):
        return "budget_exhausted"
    if all(any(
        reason.startswith(("already tested", "unsupported", "no observed retry"))
        for reason in candidate.rejection_reasons
    ) for candidate in candidates):
        return "investigations_exhausted"
    return "inconclusive"


def assess_fuse(
    record_id: str, baseline: Observation, observed: Observation, result: Result,
    candidates: Sequence[CandidateScore], no_progress_count: int,
) -> FuseRecord:
    unchanged = (
        observed.metrics.completions == baseline.metrics.completions
        and observed.metrics.queue_growth == baseline.metrics.queue_growth
        and observed.metrics.retries == baseline.metrics.retries
        and observed.metrics.duplicate_completions == baseline.metrics.duplicate_completions
    )
    count = no_progress_count + 1 if unchanged else 0
    triggers = []
    if not result.matched:
        triggers.append("contradictory_prediction")
    if count:
        triggers.append("repeated_no_progress" if count >= 2 else "no_progress")
    alternatives = tuple(
        item.key for item in sorted(candidates, key=lambda item: (-item.score, item.key))
        if item.eligible
    )
    if result.matched:
        disposition = "answer"
        reason = "Prediction met; report bounded evidence support, not proof of a unique cause."
    elif alternatives:
        disposition = "switch_path"
        reason = "Current assumption weakened; affordable untried alternatives remain. Reframe before switching."
    else:
        disposition = _stop_reason(candidates)
        reason = "Alternatives assessed; none is both applicable and affordable. Do not repeat or invent evidence."
    return FuseRecord(
        record_id, (baseline.id, observed.id, result.id), tuple(triggers), count,
        alternatives, disposition, reason,
    )


def run_navigator(
    environment: NavigatorEnvironment, budget: int = 10, max_steps: int = 12,
) -> Episode:
    if type(budget) is not int or not 0 <= budget <= 20:
        raise ValueError("Budget must be an integer between 0 and 20.")
    if type(max_steps) is not int or not 1 <= max_steps <= 12:
        raise ValueError("Step limit must be an integer between 1 and 12.")
    records: list[Record] = []
    spent = steps = no_progress = 0
    states: dict[Intervention, str] = {}
    baseline = None
    checkpoint = None
    hypothesis = None
    pending = None
    interpretation = "Unresolved: no observations collected."
    phase = ActionKind.ASK
    terminal_reason = "inconclusive"
    capabilities = environment.capabilities()

    def next_id() -> str:
        return f"r{len(records) + 1:03d}"

    def episode(reason: str) -> Episode:
        return Episode(tuple(records), budget, spent, reason, interpretation, "heuristic", max_steps)

    def act(action: Action, question: str, prediction_id: str | None = None) -> str:
        nonlocal spent, steps
        if spent + action.cost > budget:
            raise RuntimeError("Internal policy error: action exceeds budget.")
        item = Question(next_id(), question)
        records.append(item)
        record = InterventionRecord(next_id(), action, item.id, prediction_id, action.cost)
        records.append(record)
        spent += action.cost
        steps += 1
        return record.id

    while steps < max_steps:
        if phase == ActionKind.ASK:
            if budget < 1:
                phase, terminal_reason = ActionKind.STOP, "budget_exhausted"
                continue
            if max_steps < 2:
                phase, terminal_reason = ActionKind.STOP, "step_limit"
                continue
            checkpoint = environment.snapshot()
            act(Action(phase), "What can the simulator telemetry tell us about this window?")
            metrics = environment.perform(Action(phase))
            if metrics is None:
                raise RuntimeError("Ask action returned no observations.")
            baseline = Observation(next_id(), "baseline_same_state_window", metrics)
            records.append(baseline)
            interpretation = "Worker capacity, DB limits, and retry injection are untested alternatives."
            hypothesis = Hypothesis(next_id(), interpretation, (baseline.id,))
            records.append(hypothesis)
            phase = ActionKind.TEST
        elif phase == ActionKind.TEST:
            if baseline is None or hypothesis is None or checkpoint is None:
                raise RuntimeError("Cannot investigate without baseline evidence and a snapshot.")
            candidates = score_candidates(
                baseline.metrics, capabilities, states, budget - spent, max_steps - steps,
            )
            chosen = choose_candidate(candidates)
            records.append(DecisionRecord(
                next_id(), hypothesis.id, (baseline.id,), budget - spent, candidates,
                None if chosen is None else chosen.key,
                "Highest eligible utility/cost; ties broken lexically by action key."
                if chosen else "No applicable affordable candidate; record explicit stop reason.",
            ))
            if chosen is None:
                phase, terminal_reason = ActionKind.STOP, _stop_reason(candidates)
                continue
            intervention = chosen.action.intervention
            if intervention is None:
                raise RuntimeError("Selected test lacks an intervention.")
            hypothesis = Hypothesis(
                next_id(), f"Working assumption to test (not a fact): {chosen.assumption}",
                (baseline.id,),
            )
            records.append(hypothesis)
            prediction = make_prediction(next_id(), hypothesis.id, baseline.metrics, intervention)
            records.append(prediction)
            action_id = act(chosen.action, chosen.question, prediction.id)
            environment.restore(checkpoint)
            metrics = environment.perform(chosen.action)
            if metrics is None:
                raise RuntimeError("Test action returned no observations.")
            observation = Observation(next_id(), chosen.key, metrics)
            records.append(observation)
            matched = prediction.matches(metrics)
            result = Result(
                next_id(), action_id, observation.id, prediction.id, matched,
                "Prediction criteria matched; this is limited support, not causal proof."
                if matched else "Discrepancy: prediction contradicted by at least one measured criterion.",
            )
            records.append(result)
            states[intervention] = "supported" if matched else "weakened"
            alternatives = score_candidates(
                baseline.metrics, capabilities, states, budget - spent - 1, max_steps - steps - 1,
            )
            records.append(DecisionRecord(
                next_id(), hypothesis.id, (baseline.id, observation.id, result.id),
                budget - spent - 1, alternatives, None,
                "Fuse assessment of alternatives, reserving one unit for the pending reframe; not a selection.",
            ))
            fuse = assess_fuse(next_id(), baseline, observation, result, alternatives, no_progress)
            records.append(fuse)
            no_progress = fuse.no_progress_count
            pending = (chosen, observation, result, fuse)
            phase = ActionKind.REFRAME
        elif phase == ActionKind.REFRAME:
            if pending is None or baseline is None:
                raise RuntimeError("Reframe requires a new result, not repeated speculation.")
            chosen, observation, result, fuse = pending
            act(Action(phase), "Which working assumption changes in light of this new result?")
            if result.matched:
                interpretation = f"Evidence supports: {chosen.assumption} A unique cause is not established."
                if chosen.action.intervention == Intervention.DISABLE_RETRIES:
                    interpretation += " Throughput recovery is not established; only new retries were disabled."
            else:
                alternatives = ", ".join(fuse.alternate_keys) or "none currently eligible"
                interpretation = (
                    f"Weakened assumption: {chosen.assumption} Prediction discrepancy recorded. "
                    f"Replace this path with untried alternatives: {alternatives}."
                )
            hypothesis = Hypothesis(
                next_id(), interpretation, (baseline.id, observation.id, result.id),
            )
            records.append(hypothesis)
            pending = None
            if fuse.disposition == "answer":
                phase, terminal_reason = ActionKind.ANSWER, "supported_interpretation"
            elif fuse.disposition == "switch_path":
                phase = ActionKind.TEST
            else:
                phase, terminal_reason = ActionKind.STOP, fuse.disposition
        else:
            if phase == ActionKind.STOP:
                interpretation = f"Unresolved ({terminal_reason}). {interpretation}"
            action_id = act(Action(phase), "What can we conclude, and why stop this bounded inquiry?")
            evidence = () if hypothesis is None else (hypothesis.id,)
            records.append(AnswerRecord(next_id(), action_id, evidence, interpretation, terminal_reason))
            return episode(terminal_reason)
    interpretation = f"Unresolved (step_limit). {interpretation}"
    records.append(FuseRecord(
        next_id(), () if baseline is None else (baseline.id,), ("hard_step_bound",),
        no_progress, (), "step_limit", "Hard action-step limit reached; no further action is permitted.",
    ))
    return episode("step_limit")
