"""Paired, evaluator-only counterfactual benchmark of the frozen Stage 2 policies."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import mean

from intuition_prototype.benchmark_protocol import (
    PROTOCOL, REPEAT_SEEDS, canonical_json, check_frozen, check_manifest, digest, file_digest,
    prepare, read_json, write_json,
)
from intuition_prototype.inquiry import run_inquiry
from intuition_prototype.navigator import run_navigator
from intuition_prototype.records import (
    Action, ActionKind, Episode, Hypothesis, Intervention, InterventionRecord, Metrics,
    Observation, Result,
)
from intuition_prototype.simulator import QueueSimulator, SimulatorConfig, Snapshot
from intuition_prototype.storage import validate_episode


@dataclass(frozen=True)
class ExternalEffect:
    beneficial: bool
    utility: float
    raw_utility: float
    completion_gain: int
    queue_growth_reduction: int
    completion_threshold: int
    queue_threshold: int


def score_effect(baseline: Metrics, outcome: Metrics) -> ExternalEffect:
    """Independent reference: does not accept policy labels, predictions, or scores."""
    if (baseline.ticks, baseline.arrivals, baseline.queue_start) != (
        outcome.ticks, outcome.arrivals, outcome.queue_start,
    ):
        raise ValueError("Counterfactuals do not share the same window, arrivals, and initial queue.")
    spec = PROTOCOL["reference"]
    completions = outcome.completions - baseline.completions
    queue = baseline.queue_growth - outcome.queue_growth
    completion_threshold = max(
        spec["completion_absolute_gain"],
        math.ceil(spec["completion_relative_gain"] * max(1, baseline.completions)),
    )
    queue_threshold = max(
        spec["queue_absolute_reduction"],
        math.ceil(spec["queue_relative_reduction"] * max(1, abs(baseline.queue_growth), baseline.arrivals)),
    )
    beneficial = (
        completions >= 0 and queue >= 0
        and (completions >= completion_threshold or queue >= queue_threshold)
    )
    raw = (completions + spec["queue_weight"] * queue) / max(1, baseline.arrivals)
    return ExternalEffect(
        beneficial, round(raw if beneficial else min(0, raw), 12), round(raw, 12),
        completions, queue, completion_threshold, queue_threshold,
    )


class AgentView:
    """Same observation/action API for both policies; no evaluation metadata access."""

    __slots__ = ("_simulator",)

    def __init__(self, simulator: QueueSimulator):
        self._simulator = simulator

    def capabilities(self) -> tuple[Intervention, ...]:
        return self._simulator.capabilities()

    def snapshot(self) -> Snapshot:
        return self._simulator.snapshot()

    def restore(self, snapshot: Snapshot) -> None:
        self._simulator.restore(snapshot)

    def perform(self, action: Action) -> Metrics | None:
        return self._simulator.perform(action)


def counterfactuals(config: SimulatorConfig, seed: int) -> tuple[Metrics, dict[str, Metrics]]:
    simulator = QueueSimulator.from_config(config, seed)
    checkpoint = simulator.snapshot()
    baseline = simulator.perform(Action(ActionKind.ASK))
    if baseline is None:
        raise RuntimeError("Oracle baseline returned no metrics.")
    outcomes = {}
    for intervention in Intervention:
        simulator.restore(checkpoint)
        outcome = simulator.perform(Action(ActionKind.TEST, intervention))
        if outcome is None:
            raise RuntimeError("Oracle intervention returned no metrics.")
        outcomes[intervention.value] = outcome
    return baseline, outcomes


def assess_policy(
    episode: Episode, baseline: Metrics, outcomes: dict[str, Metrics],
) -> dict:
    validate_episode(episode)
    if set(outcomes) != {item.value for item in Intervention}:
        raise ValueError("Reference must include every permitted single intervention.")
    observed = [record for record in episode.records if isinstance(record, Observation)]
    if observed and observed[0].metrics != baseline:
        raise ValueError("Policy and oracle received different baseline evidence.")
    for observation in observed[1:]:
        if observation.source not in outcomes or observation.metrics != outcomes[observation.source]:
            raise ValueError("Policy experiment differs from the same-state oracle.")
    effects = {key: score_effect(baseline, outcome) for key, outcome in outcomes.items()}
    best = max(0.0, *(effect.utility for effect in effects.values()))
    tests = [
        record for record in episode.records
        if isinstance(record, InterventionRecord) and record.action.kind == ActionKind.TEST
    ]
    supported = episode.stop_reason == "supported_interpretation"
    recommendation = None
    evidence_backed = False
    if supported:
        if not tests or tests[-1].action.intervention is None:
            raise ValueError("Supported interpretation has no actual tested intervention.")
        recommendation = tests[-1].action.intervention.value
        results = [record for record in episode.records if isinstance(record, Result)]
        claims = [record for record in episode.records if isinstance(record, Hypothesis)]
        evidence_backed = bool(
            results and claims and results[-1].matched
            and results[-1].intervention_id == tests[-1].id
            and results[-1].id in claims[-1].evidence_ids
            and results[-1].observation_id in claims[-1].evidence_ids
        )
        if not evidence_backed:
            raise ValueError("Policy support is not linked to its last measured experiment.")
    selected = None if recommendation is None else effects[recommendation]
    utility = 0.0 if selected is None else selected.utility
    helpful = selected is not None and selected.beneficial
    best_tested = max(0.0, *(
        effects[record.action.intervention.value].utility
        for record in tests if record.action.intervention is not None
    )) if tests else 0.0
    return {
        "recommendation": recommendation, "supported": supported, "evidence_backed": evidence_backed,
        "helpful": helpful, "abstained": recommendation is None,
        "utility": utility, "regret": round(best - utility, 12), "best_available_utility": best,
        "cost": episode.cost, "cost_adjusted_utility": utility / (1 + episode.cost),
        "steps": len([r for r in episode.records if isinstance(r, InterventionRecord)]),
        "stop_reason": episode.stop_reason,
        "budget_adherent": episode.cost <= PROTOCOL["agent_budget"],
        "supported_but_unhelpful": supported and not helpful,
        "no_benefit_case": best == 0,
        "no_benefit_false_positive": best == 0 and supported,
        "abstained_with_available_benefit": recommendation is None and best > 0,
        "missed_despite_tested_benefit": recommendation is None and best_tested > 0,
        "suboptimal_recommendation": recommendation is not None and best - utility > 1e-9,
        "best_tested_utility": best_tested,
    }


def serialize_episode(episode: Episode) -> dict:
    data = asdict(episode)
    data["records"] = [
        {"kind": type(record).__name__, "data": asdict(record)} for record in episode.records
    ]
    return data


def evaluate_case(case: dict, seed: int) -> dict:
    config = SimulatorConfig(**case["parameters"])
    baseline, outcomes = counterfactuals(config, seed)
    policies = {}
    for name, runner in (("scripted", run_inquiry), ("heuristic", run_navigator)):
        view = AgentView(QueueSimulator.from_config(config, seed))
        episode = runner(view, budget=PROTOCOL["agent_budget"])
        if episode.budget != PROTOCOL["agent_budget"] or episode.max_steps != PROTOCOL["agent_step_limit"]:
            raise ValueError("Policies do not share the declared bounds.")
        policies[name] = {
            "assessment": assess_policy(episode, baseline, outcomes),
            "episode": serialize_episode(episode),
        }
    return {
        "config_id": case["config_id"], "regime": case["regime"], "split": case["split"], "seed": seed,
        "baseline": asdict(baseline),
        "counterfactuals": {
            key: {"metrics": asdict(outcome), "effect": asdict(score_effect(baseline, outcome))}
            for key, outcome in outcomes.items()
        },
        "policies": policies,
        "oracle_compute": {"measurement_windows": 4, "advance_ticks_including_warmup": 150},
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    low, high = math.floor(location), math.ceil(location)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def paired_intervals(units: list[dict]) -> dict:
    if not units:
        raise ValueError("Cannot bootstrap an empty configuration split.")
    groups: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        groups[unit["regime"]].append(unit)
    metrics = ("utility", "regret", "cost", "cost_adjusted_utility")
    distributions: dict[str, list[float]] = {metric: [] for metric in metrics}
    generator = random.Random(PROTOCOL["statistics"]["bootstrap_seed"])
    for _ in range(2000):
        sample = [
            generator.choice(group)
            for _, group in sorted(groups.items()) for _ in range(len(group))
        ]
        for metric in metrics:
            distributions[metric].append(mean(unit[metric] for unit in sample))
    return {
        metric: {
            "mean_delta_heuristic_minus_scripted": mean(unit[metric] for unit in units),
            "configuration_delta_min": min(unit[metric] for unit in units),
            "configuration_delta_max": max(unit[metric] for unit in units),
            "paired_stratified_bootstrap_95": [
                _percentile(distributions[metric], 0.025),
                _percentile(distributions[metric], 0.975),
            ],
        } for metric in metrics
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[dict], manifest: dict) -> dict:
    if not rows:
        raise ValueError("No benchmark observations to summarize.")
    seen = set()
    for row in rows:
        identity = (row["config_id"], row["seed"])
        if identity in seen or row["seed"] not in REPEAT_SEEDS:
            raise ValueError("Duplicate case repeat or undeclared repeat seed.")
        seen.add(identity)
        if row["policy_sha256"] != manifest["policy_sha256"]:
            raise ValueError("Raw results contain a different frozen policy.")
        if row["manifest_sha256"] != digest(manifest):
            raise ValueError("Raw results reference a different declared manifest.")
    report = {
        "version": manifest["version"], "manifest_sha256": digest(manifest),
        "policy_sha256": manifest["policy_sha256"],
        "attempt_classification": manifest["attempt_classification"],
        "rerun_of": manifest["rerun_of"], "invalidation_reason": manifest["invalidation_reason"],
        "scope": PROTOCOL["scope"], "llm_baseline": PROTOCOL["llm_baseline"],
        "independent_unit": PROTOCOL["statistics"]["unit"], "splits": {},
        "oracle_compute": {
            "measurement_windows": sum(row["oracle_compute"]["measurement_windows"] for row in rows),
            "advance_ticks_including_warmup": sum(
                row["oracle_compute"]["advance_ticks_including_warmup"] for row in rows
            ),
            "included_in_agent_cost": False,
        },
    }
    for split in sorted({row["split"] for row in rows}):
        subset = [row for row in rows if row["split"] == split]
        by_config: dict[str, list[dict]] = defaultdict(list)
        for row in subset:
            by_config[row["config_id"]].append(row)
        if len(by_config) != manifest["configuration_counts"][split]:
            raise ValueError("Incomplete configuration split; refusing a partial benchmark summary.")
        if any(len(repeats) != len(REPEAT_SEEDS) for repeats in by_config.values()):
            raise ValueError("Configuration repeats are incomplete.")
        units = [
            {
                "config_id": config_id, "regime": repeats[0]["regime"],
                **{
                    metric: mean(
                        row["policies"]["heuristic"]["assessment"][metric]
                        - row["policies"]["scripted"]["assessment"][metric]
                        for row in repeats
                    ) for metric in ("utility", "regret", "cost", "cost_adjusted_utility")
                },
            } for config_id, repeats in sorted(by_config.items())
        ]
        summary = {
            "configurations": len(units), "repeats_per_configuration": len(REPEAT_SEEDS),
            "paired_cases": len(subset), "agent_episodes": 2 * len(subset),
            "oracle_no_benefit_cases": sum(
                row["policies"]["scripted"]["assessment"]["no_benefit_case"] for row in subset
            ),
            "policies": {}, "paired": paired_intervals(units),
            "configuration_utility_comparison": {
                "heuristic_wins": sum(unit["utility"] > 1e-9 for unit in units),
                "scripted_wins": sum(unit["utility"] < -1e-9 for unit in units),
                "ties": sum(abs(unit["utility"]) <= 1e-9 for unit in units),
            },
            "by_regime": {},
            "failure_analysis": {},
        }
        for name in ("scripted", "heuristic"):
            scores = [row["policies"][name]["assessment"] for row in subset]
            supported = sum(score["supported"] for score in scores)
            unhelpful = sum(score["supported_but_unhelpful"] for score in scores)
            no_benefit = sum(score["no_benefit_case"] for score in scores)
            false_positives = sum(score["no_benefit_false_positive"] for score in scores)
            summary["policies"][name] = {
                **{f"mean_{metric}": mean(score[metric] for score in scores) for metric in (
                    "utility", "regret", "cost", "cost_adjusted_utility",
                )},
                "supported_count": supported, "helpful_recommendations": sum(s["helpful"] for s in scores),
                "abstentions": sum(s["abstained"] for s in scores),
                "supported_but_unhelpful_count": unhelpful,
                "supported_but_unhelpful_rate_among_supported": _fraction(unhelpful, supported),
                "no_benefit_false_positive_count": false_positives,
                "no_benefit_false_positive_rate": _fraction(false_positives, no_benefit),
                "budget_violations": sum(not s["budget_adherent"] for s in scores),
                "stop_reasons": dict(Counter(score["stop_reason"] for score in scores)),
            }
            summary["failure_analysis"][name] = {}
            for category in PROTOCOL["failure_categories"]:
                failures = [
                    row for row in subset if row["policies"][name]["assessment"][category]
                ]
                summary["failure_analysis"][name][category] = {
                    "count": len(failures),
                    "by_regime": dict(Counter(row["regime"] for row in failures)),
                    "samples": [
                        {
                            "config_id": row["config_id"], "seed": row["seed"], "regime": row["regime"],
                            "assessment": row["policies"][name]["assessment"],
                            "interpretation": row["policies"][name]["episode"]["interpretation"],
                            "trace_excerpt": [
                                record for record in row["policies"][name]["episode"]["records"]
                                if record["kind"] in ("InterventionRecord", "Result", "Hypothesis", "FuseRecord")
                            ],
                        } for row in failures[:2]
                    ],
                }
        for regime in sorted({unit["regime"] for unit in units}):
            matching = [unit for unit in units if unit["regime"] == regime]
            summary["by_regime"][regime] = {
                "configurations": len(matching),
                "mean_utility_delta": mean(unit["utility"] for unit in matching),
                "mean_cost_delta": mean(unit["cost"] for unit in matching),
                "heuristic_utility_losses": sum(unit["utility"] < -1e-9 for unit in matching),
            }
        report["splits"][split] = summary
    return report


def render_report(report: dict) -> str:
    lines = [
        "# Stage 3: frozen policy comparison",
        f"Attempt: **{report['attempt_classification']}**; protocol `{report['version']}`.",
        f"Manifest SHA256: `{report['manifest_sha256']}`.",
        f"Policy SHA256: `{report['policy_sha256']}`.",
        report["scope"],
        "**A supported_interpretation is not external correctness or proof of a unique cause.**",
        f"LLM baseline: {report['llm_baseline']}.",
        f"Statistical unit: {report['independent_unit']}. Intervals are paired configuration-level, "
        "stratified by generator regime; they do not establish general superiority.",
    ]
    if report["rerun_of"]:
        lines.append(f"Rerun of {report['rerun_of']}: {report['invalidation_reason']}")
    for split, summary in report["splits"].items():
        lines.extend((
            f"\n## {split}",
            f"{summary['configurations']} configurations, {summary['paired_cases']} paired seeded cases, "
            f"{summary['agent_episodes']} agent episodes; "
            f"{summary['oracle_no_benefit_cases']} cases without reference single-action benefit.",
            "| Policy | Utility | Regret | Cost | Utility/(1+cost) | Helpful / supported | Abstentions | Unhelpful support | No-benefit FP |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ))
        for name in ("scripted", "heuristic"):
            values = summary["policies"][name]
            lines.append(
                f"| {name} | {values['mean_utility']:.6f} | {values['mean_regret']:.6f} "
                f"| {values['mean_cost']:.3f} | {values['mean_cost_adjusted_utility']:.6f} "
                f"| {values['helpful_recommendations']} / {values['supported_count']} "
                f"| {values['abstentions']} | {values['supported_but_unhelpful_count']} "
                f"| {values['no_benefit_false_positive_count']} |"
            )
        for name in ("scripted", "heuristic"):
            values = summary["policies"][name]
            lines.append(
                f"\n{name} stop reasons: `{canonical_json(values['stop_reasons'])}`; "
                f"budget violations: {values['budget_violations']}.\n"
            )
        for metric in ("utility", "regret", "cost", "cost_adjusted_utility"):
            values = summary["paired"][metric]
            low, high = values["paired_stratified_bootstrap_95"]
            lines.append(
                f"- Paired {metric} delta (heuristic - scripted): "
                f"{values['mean_delta_heuristic_minus_scripted']:.6f}; descriptive 95% interval "
                f"[{low:.6f}, {high:.6f}]."
            )
        lines.append(f"Configuration-mean utility wins/losses/ties: `{summary['configuration_utility_comparison']}`.")
        lines.extend((
            "\n### Regime-level differences",
            "| Sampling regime (not cause label) | Configs | Utility delta | Cost delta | Heuristic utility losses |",
            "| --- | ---: | ---: | ---: | ---: |",
        ))
        for regime, values in summary["by_regime"].items():
            lines.append(
                f"| {regime} | {values['configurations']} | {values['mean_utility_delta']:.6f} "
                f"| {values['mean_cost_delta']:.3f} | {values['heuristic_utility_losses']} |"
            )
        lines.append("\n### Failure samples (full typed traces in raw.jsonl)")
        for policy in ("scripted", "heuristic"):
            categories = summary["failure_analysis"][policy]
            for category in PROTOCOL["failure_categories"]:
                values = categories[category]
                lines.append(
                    f"- {policy} / {category}: {values['count']}; "
                    f"regimes `{canonical_json(values['by_regime'])}`."
                )
                for sample in values["samples"]:
                    score = sample["assessment"]
                    lines.append(
                        f"  - Config `{sample['config_id']}`, seed {sample['seed']}: "
                        f"recommendation `{score['recommendation']}`, utility {score['utility']:.6f}, "
                        f"best {score['best_available_utility']:.6f}; {sample['interpretation']}"
                    )
    lines.extend((
        "\n## Scope and costs",
        f"Evaluator-only computation, excluded from agent cost: `{canonical_json(report['oracle_compute'])}`.",
        "The independent reference rewards unique completions and reduced queue growth using the "
        "predeclared absolute/relative criteria in protocol.json. Mixed bottlenecks are not assigned "
        "a sole true-cause label. No measured single-action benefit does not exclude multi-step cures.",
        "No policy weights, prediction thresholds, or selector rules were tuned on these results. "
        "The held-out parameter configurations still use the same known simulator formulas.",
    ))
    return "\n".join(lines) + "\n"


def run_benchmark(directory: Path, split: str = "all") -> dict:
    if split not in ("all", "development", "heldout"):
        raise ValueError("Unknown benchmark split.")
    manifest = check_frozen(directory)
    write_json(directory / "run-start.json", {
        "status": "started_not_yet_valid",
        "manifest_sha256": digest(manifest), "split": split,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "failure_rule": "Missing run-complete.json means invalid/incomplete, never silently resume or overwrite.",
    })
    selected = ("development", "heldout") if split == "all" else (split,)
    rows = []
    with (directory / "raw.jsonl").open("x", encoding="utf-8", newline="\n") as output:
        for selected_split in selected:
            cases = read_json(directory / f"{selected_split}.json")["configurations"]
            for case in cases:
                for seed in REPEAT_SEEDS:
                    row = evaluate_case(case, seed)
                    row["policy_sha256"] = manifest["policy_sha256"]
                    row["manifest_sha256"] = digest(manifest)
                    output.write(canonical_json(row) + "\n")
                    output.flush()
                    rows.append(row)
    check_frozen(directory)
    report = summarize(rows, manifest)
    write_json(directory / "report.json", report)
    with (directory / "report.md").open("x", encoding="utf-8", newline="\n") as output:
        output.write(render_report(report))
    write_json(directory / "run-complete.json", {
        "status": "complete",
        "manifest_sha256": digest(manifest),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "files": {name: file_digest(directory / name) for name in ("raw.jsonl", "report.json", "report.md")},
    })
    return report


def load_results(directory: Path) -> tuple[list[dict], dict]:
    if not (directory / "run-complete.json").is_file():
        raise ValueError("This attempt is incomplete/invalid; no valid held-out report exists.")
    receipt = read_json(directory / "run-complete.json")
    manifest = check_manifest(directory)
    if receipt["manifest_sha256"] != digest(manifest):
        raise ValueError("Completion receipt and declared manifest disagree.")
    for name, expected in receipt["files"].items():
        if file_digest(directory / name) != expected:
            raise ValueError(f"Completed result artifact changed: {name}")
    with (directory / "raw.jsonl").open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    return rows, read_json(directory / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Stage 3 paired parameter-holdout evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    declaration = commands.add_parser("prepare", help="Freeze manifests and sources WITHOUT running policies.")
    declaration.add_argument("--output", type=Path, required=True)
    declaration.add_argument("--generator-seed", type=int, default=20260908)
    declaration.add_argument("--rerun-of")
    declaration.add_argument("--invalidation-reason")
    run = commands.add_parser("run", help="Execute once; never overwrite or silently resume an attempt.")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--split", choices=("all", "development", "heldout"), default="all")
    inspect = commands.add_parser("report", help="Read and verify a completed report without new experiments.")
    inspect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.output, args.generator_seed,
            rerun_of=args.rerun_of, invalidation_reason=args.invalidation_reason,
        )
    elif args.command == "run":
        result = run_benchmark(args.output, args.split)
    else:
        _, result = load_results(args.output)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
