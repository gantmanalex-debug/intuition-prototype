"""UI/CLI orchestration owns evaluator metadata; the runner receives only an API."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from intuition_prototype.inquiry import run_inquiry
from intuition_prototype.navigator import run_navigator
from intuition_prototype.records import (
    Hypothesis, InterventionRecord, Observation, Prediction, Question, Result,
    AnswerRecord, DecisionRecord, FuseRecord,
)
from intuition_prototype.simulator import QueueSimulator, SCENARIOS
from intuition_prototype.storage import SavedEpisode, load_episode, save_episode


POLICIES = ("scripted", "heuristic")


def render_trace(saved: SavedEpisode) -> str:
    episode = saved.episode
    lines = [
        f"## {episode.policy.upper()} {'baseline' if episode.policy == 'scripted' else 'navigator'} "
        "trace (not an LLM or learned intuition)",
        f"Run `{saved.id}` | evaluator case `{saved.scenario}` | seed `{saved.seed}`",
        f"**Cost: {episode.cost}/{episode.budget} | Stop: {episode.stop_reason}**",
        f"Action steps: {sum(isinstance(record, InterventionRecord) for record in episode.records)}"
        f"/{episode.max_steps}",
        f"**Interpretation:** {episode.interpretation}",
        "",
        "All trial windows restore the same warm-state snapshot and arrival RNG.",
        "Claims below are interpretations, not hidden ground truth.",
    ]
    for record in episode.records:
        label = f"**{record.id} - {type(record).__name__}**"
        if isinstance(record, Observation):
            metrics = record.metrics
            text = (
                f"{record.source}: {metrics.completions} unique completions, "
                f"{metrics.arrivals} arrivals, {metrics.retries} retries, "
                f"{metrics.duplicate_completions} duplicate completions; "
                f"queue {metrics.queue_start} -> {metrics.queue_end} "
                f"(growth {metrics.queue_growth}); "
                f"completed-request mean latency {metrics.mean_latency} ticks; "
                f"dropped attempts {metrics.dropped_attempts}."
            )
        elif isinstance(record, Hypothesis):
            text = f"{record.claim} Evidence references: {', '.join(record.evidence_ids)}."
        elif isinstance(record, Prediction):
            text = (
                f"{record.claim} Thresholds: completions >= {record.minimum_completions}, "
                f"queue growth <= {record.maximum_queue_growth}"
                f", retries <= {record.maximum_retries if record.maximum_retries is not None else 'unconstrained'}."
                f" Claim reference: {record.hypothesis_id}."
            )
        elif isinstance(record, Question):
            text = record.text
        elif isinstance(record, InterventionRecord):
            action = record.action
            text = (
                f"`{action.kind.value}`"
                f"{' / ' + action.intervention.value if action.intervention else ''} "
                f"| cost {record.cost} | question {record.question_id} "
                f"| prediction {record.prediction_id or 'none'}."
            )
        elif isinstance(record, Result):
            text = (
                f"{record.interpretation} Action {record.intervention_id}; "
                f"prediction {record.prediction_id}; observation {record.observation_id}."
            )
        elif isinstance(record, DecisionRecord):
            rows = [
                f"Remaining budget {record.remaining_budget}; chosen: {record.chosen_key or 'none'}. "
                f"{record.reason} Hypothesis {record.hypothesis_id}; evidence {', '.join(record.evidence_ids)}.",
                "",
                "| Candidate | Score | Discrimination | Relevance | Gap | Novelty | Redundancy | Cost | Eligibility |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
            for candidate in record.candidates:
                rows.append(
                    f"| {candidate.key} | {candidate.score:.3f} | {candidate.discrimination:.2f} "
                    f"| {candidate.anomaly_relevance:.2f} | {candidate.evidence_gap:.0f} "
                    f"| {candidate.novelty:.0f} | {candidate.redundancy:.0f} | {candidate.cost} "
                    f"| {'eligible' if candidate.eligible else '; '.join(candidate.rejection_reasons)} |"
                )
            rows.extend(
                f"\n- **{candidate.key}**: {candidate.question} {candidate.rationale}"
                for candidate in record.candidates
            )
            text = "\n".join(rows)
        elif isinstance(record, FuseRecord):
            text = (
                f"**Intelligence fuse: {record.disposition}**. "
                f"Triggers: {', '.join(record.triggers) or 'none; prediction matched'}. "
                f"No-progress count {record.no_progress_count}; "
                f"eligible alternatives: {', '.join(record.alternate_keys) or 'none'}. "
                f"{record.reason} Evidence: {', '.join(record.evidence_ids) or 'none'}."
            )
        elif isinstance(record, AnswerRecord):
            text = (
                f"{record.claim} Stop: {record.stop_reason}; terminal action {record.action_id}; "
                f"evidence/claim references: {', '.join(record.evidence_ids) or 'none'}."
            )
        lines.extend(("", f"{label}: {text}"))
    lines.extend((
        "",
        "**Limitation:** a successful scripted trace does not validate the research hypothesis. "
        "Nor does a successful heuristic trace. Scores are hand-designed estimates, not learned "
        "intuition, calibrated probabilities, or causal truth.",
    ))
    return "\n\n".join(lines)


def run_stage1(
    database_path: Path, scenario: str, seed: int, budget: int,
    policy: str = "scripted", max_steps: int = 12,
) -> tuple[str, str]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy!r}. Choose from {POLICIES}.")
    if policy == "scripted" and max_steps != 12:
        raise ValueError("Custom step limits apply only to the heuristic policy.")
    simulator = QueueSimulator(scenario, seed)
    episode = (
        run_inquiry(simulator, budget) if policy == "scripted"
        else run_navigator(simulator, budget, max_steps)
    )
    episode_id = save_episode(database_path, scenario, seed, episode)
    return episode_id, render_trace(load_episode(database_path, episode_id))


def reset_stage1(scenario: str, seed: int) -> tuple[str, str]:
    QueueSimulator(scenario, seed)
    return "", (
        "Ready: fresh deterministic state will be constructed on the next Run. "
        "The trace viewer is cleared; previously persisted episodes are retained."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scripted or evidence-based heuristic inquiry.")
    parser.add_argument("--policy", choices=POLICIES, default="scripted")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--scenario", choices=SCENARIOS, default="retry_amplification")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--database", type=Path, default=Path(".runtime/intuition.sqlite3"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    episode_id, trace = run_stage1(
        args.database, args.scenario, args.seed, args.budget, args.policy, args.max_steps,
    )
    print(
        json.dumps(asdict(load_episode(args.database, episode_id)), indent=2)
        if args.json else trace
    )


if __name__ == "__main__":
    main()
