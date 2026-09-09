"""Cumulative service-remediation task, separate from same-state diagnosis."""

from dataclasses import asdict, dataclass
import itertools
from typing import Protocol

from intuition_prototype.learned import LearnedModel
from intuition_prototype.navigator import choose_candidate, investigation_text, score_candidates
from intuition_prototype.predictions import make_prediction
from intuition_prototype.records import Action, ActionKind, Hypothesis, Intervention, Metrics
from intuition_prototype.simulator import QueueSimulator, SimulatorConfig, Snapshot


VERSION = "sequential-remediation-v1"
WAIT = "observe_wait"
POLICIES = ("scripted-sequential-v1", "heuristic-sequential-v1", "learned-transfer-sequential-v1")
ORDER = (Intervention.DOUBLE_WORKERS, Intervention.DISABLE_RETRIES, Intervention.DOUBLE_DB_CAPACITY)
MAX_DEPTH = 3


def service_restored(metrics: Metrics) -> bool:
    """Finite-window operational SLO, never a diagnosis or a minimum-action gate."""
    return (
        metrics.arrivals > 0 and metrics.completions >= metrics.arrivals
        and metrics.queue_growth <= 0 and metrics.dropped_attempts == 0
    )


def service_utility(metrics: Metrics) -> float:
    arrivals = max(1, metrics.arrivals)
    return round(
        min(1.0, metrics.completions / arrivals)
        - max(0, metrics.queue_growth) / arrivals - metrics.dropped_attempts / arrivals, 12,
    )


class SequentialAPI(Protocol):
    def observe(self) -> Metrics: ...
    def capabilities(self) -> tuple[str, ...]: ...
    def move(self, key: str) -> Metrics: ...


class SequentialEnvironment:
    """Only telemetry and legal moves are exposed; effects persist between tests."""

    __slots__ = ("_simulator", "_current", "_used", "_moves", "_snapshots")

    def __init__(self, simulator: QueueSimulator):
        self._simulator = simulator
        initial = simulator.perform(Action(ActionKind.ASK))
        if initial is None:
            raise RuntimeError("Initial service measurement is missing.")
        self._current = initial
        self._used: set[str] = set()
        self._moves = 0
        self._snapshots: dict[str, tuple[Metrics, frozenset[str], int]] = {}

    def observe(self) -> Metrics:
        return self._current

    def capabilities(self) -> tuple[str, ...]:
        if self._moves >= MAX_DEPTH:
            return ()
        return tuple(sorted({action.value for action in Intervention} - self._used)) + (WAIT,)

    def move(self, key: str) -> Metrics:
        if key not in self.capabilities():
            raise ValueError(f"Illegal, repeated, or depth-exhausted sequential move: {key!r}.")
        action = Action(ActionKind.ASK) if key == WAIT else Action(ActionKind.TEST, Intervention(key))
        outcome = self._simulator.perform(action)
        if outcome is None:
            raise RuntimeError("Sequential move did not return evidence.")
        self._moves += 1
        if key != WAIT:
            self._used.add(key)
        self._current = outcome
        return outcome

    def snapshot(self) -> Snapshot:
        token = self._simulator.snapshot()
        self._snapshots[token.token] = (self._current, frozenset(self._used), self._moves)
        return token

    def restore(self, snapshot: Snapshot) -> None:
        if not isinstance(snapshot, Snapshot) or snapshot.token not in self._snapshots:
            raise ValueError("Unknown sequential snapshot.")
        self._simulator.restore(snapshot)
        current, used, moves = self._snapshots[snapshot.token]
        self._current, self._used, self._moves = current, set(used), moves


def environment(config: SimulatorConfig, seed: int) -> SequentialEnvironment:
    return SequentialEnvironment(QueueSimulator.from_config(config, seed))


def legal_paths(max_depth: int = MAX_DEPTH) -> tuple[tuple[str, ...], ...]:
    if type(max_depth) is not int or not 0 <= max_depth <= MAX_DEPTH:
        raise ValueError("Reference depth must be an integer in [0, 3].")
    actions = tuple(sorted(action.value for action in Intervention)) + (WAIT,)
    return tuple(
        path for depth in range(max_depth + 1) for path in itertools.product(actions, repeat=depth)
        if len([key for key in path if key != WAIT]) == len({key for key in path if key != WAIT})
    )


def path_key(path: tuple[str, ...] | list[str]) -> str:
    return ",".join(path)


def reference(config: SimulatorConfig, seed: int) -> dict:
    """Exhaust every legal prefix, including wait substitutions; no policy is run."""
    nodes = []
    for path in legal_paths():
        simulator = environment(config, seed)
        metrics = simulator.observe()
        for action in path:
            metrics = simulator.move(action)
        nodes.append({
            "path": list(path), "metrics": asdict(metrics), "success": service_restored(metrics),
            "utility": service_utility(metrics),
            "action_cost": 1 + sum(1 if key == WAIT else 3 for key in path),
        })
    successes = [node for node in nodes if node["success"]]
    depth = min((len(node["path"]) for node in successes), default=None)
    strict_three = depth == 3 and all(
        WAIT not in node["path"] and len(set(node["path"])) == 3 for node in successes
    )
    return {
        "minimum_depth": depth, "requires_three_distinct_interventions": strict_three,
        "nodes": nodes, "best_utility": max(node["utility"] for node in nodes),
        "successful_paths": [node["path"] for node in successes],
        "oracle_compute": {
            "trajectories": len(nodes),
            "measurement_windows_including_warmup": sum(2 + len(node["path"]) for node in nodes),
        },
    }


def run_sequential(
    api: SequentialAPI, policy: str, model: LearnedModel, budget: int = 10,
) -> dict:
    if policy not in POLICIES:
        raise ValueError("Unknown versioned sequential policy.")
    if type(budget) is not int or not 0 <= budget <= 20:
        raise ValueError("Budget must be an integer in [0, 20].")
    if budget < 1:
        return dict(version=VERSION, policy=policy, budget=budget, cost=0, initial=None,
                    moves=[], decisions=[], stop_reason="budget_exhausted", success=False,
                    model_sha256=model.fingerprint if policy.startswith("learned") else None)
    current = api.observe()
    initial = current
    spent = 1
    moves, decisions = [], []
    stop_reason = "inconclusive"
    pending_trigger = None
    for depth in range(MAX_DEPTH + 1):
        if service_restored(current):
            stop_reason = "service_restored"
            break
        if depth == MAX_DEPTH:
            stop_reason = "depth_exhausted"
            break
        keys = api.capabilities()
        if any(key not in {action.value for action in Intervention} | {WAIT} for key in keys):
            raise ValueError("Environment exposed an unsupported sequential move.")
        available = tuple(Intervention(key) for key in keys if key != WAIT)
        remaining = budget - spent
        if policy == "scripted-sequential-v1":
            candidates = [{
                "key": item.value, "eligible": item in available and remaining >= 3,
                "score": float(len(ORDER) - rank),
                "rejection_reasons": (
                    ["unavailable after prior intervention"] if item not in available else []
                ) + (["unaffordable: test plus evidence update requires 3 units"] if remaining < 3 else []),
                "rationale": "Preserved fixed worker/retry/DB order, but cumulative goal-based stopping.",
            } for rank, item in enumerate(ORDER)]
            eligible = [item for item in candidates if item["eligible"]]
            chosen = min(eligible, key=lambda item: (-item["score"], item["key"])) if eligible else None
            chosen_key = None if chosen is None else chosen["key"]
        else:
            selector = model.score_candidates if policy.startswith("learned") else score_candidates
            scored = selector(current, available, {}, remaining, remaining_steps=3 * (MAX_DEPTH - depth) + 1)
            chosen = choose_candidate(scored)
            chosen_key = None if chosen is None else chosen.key
            candidates = [asdict(candidate) for candidate in scored]
        decisions.append({
            "depth": depth, "evidence": asdict(current), "remaining_budget": remaining,
            "candidates": candidates, "chosen_key": chosen_key,
            "wait": {
                "legal": WAIT in keys, "selected": False,
                "reason": "Inherited selectors rank interventions only; no learned or heuristic wait estimate.",
            },
            "fuse": (
                "switch_to_untried_intervention" if pending_trigger and chosen_key else
                "stop_no_eligible_alternative" if pending_trigger else "not_triggered"
            ),
            "trigger": pending_trigger,
        })
        if chosen_key is None:
            stop_reason = "budget_exhausted" if remaining < 3 else "no_eligible_intervention"
            break
        question, assumption = investigation_text(Intervention(chosen_key))
        hypothesis = Hypothesis(
            f"h{depth + 1}", f"Working assumption, not a diagnosis: {assumption}", (f"o{depth}",),
        )
        prediction = make_prediction(f"p{depth + 1}", hypothesis.id, current, Intervention(chosen_key))
        outcome = api.move(chosen_key)
        matched = prediction.matches(outcome)
        progress = service_utility(outcome) - service_utility(current)
        trigger = []
        if not matched:
            trigger.append("local_prediction_discrepancy")
        if progress <= 0:
            trigger.append("no_service_utility_progress")
        if matched and not service_restored(outcome):
            trigger.append("local_prediction_met_but_service_goal_unmet")
        spent += 3
        moves.append({
            "depth": depth + 1, "action": chosen_key, "cost": 3, "cumulative_cost": spent,
            "question": question, "hypothesis": asdict(hypothesis),
            "before_observation_id": f"o{depth}", "after_observation_id": f"o{depth + 1}",
            "before": asdict(current), "prediction": asdict(prediction),
            "after": asdict(outcome), "prediction_matched": matched,
            "service_success": service_restored(outcome), "service_utility": service_utility(outcome),
            "intermediate_utility_change": round(progress, 12), "fuse_triggers": trigger,
            "interpretation": (
                f"{chosen_key}: local prediction {'matched' if matched else 'weakened'}; "
                f"operational service goal {'met' if service_restored(outcome) else 'not met'}. "
                "The intervention remains applied. Next decisions use this new telemetry; "
                "consecutive windows are not a same-state causal contrast."
            ),
        })
        pending_trigger = trigger or None
        current = outcome
    return {
        "version": VERSION, "policy": policy, "budget": budget, "cost": spent,
        "initial": asdict(initial), "moves": moves, "decisions": decisions, "stop_reason": stop_reason,
        "success": service_restored(current),
        "model_sha256": model.fingerprint if policy.startswith("learned") else None,
    }


def cohort(base_references: list[dict], control_references: list[dict]) -> tuple[bool, bool]:
    if len(base_references) != 2 or len(control_references) != 2:
        raise ValueError("Eligibility needs the two declared seed references for base and control.")
    deep = all(item["requires_three_distinct_interventions"] for item in base_references)
    shallow = deep and all(item["minimum_depth"] == 1 for item in control_references)
    return deep, shallow


def assess_episode(episode: dict, oracle: dict) -> dict:
    if episode["version"] != VERSION or episode["policy"] not in POLICIES:
        raise ValueError("Invalid sequential episode version/policy.")
    if episode["stop_reason"] not in (
        "service_restored", "depth_exhausted", "budget_exhausted", "no_eligible_intervention",
    ):
        raise ValueError("Invalid sequential terminal reason.")
    nodes = {path_key(node["path"]): node for node in oracle["nodes"]}
    initial = episode["initial"]
    path = []
    previous = initial
    if initial != nodes[""]["metrics"]:
        raise ValueError("Episode did not receive the reference initial evidence.")
    for index, move in enumerate(episode["moves"]):
        if service_restored(Metrics(**previous)):
            raise ValueError("Episode continues after the operational goal was already met.")
        path.append(move["action"])
        if path_key(path) not in nodes or move["action"] == WAIT or len(set(path)) != len(path):
            raise ValueError("Episode contains an illegal/repeated intervention.")
        expected = nodes[path_key(path)]
        if move["before"] != previous or move["after"] != expected["metrics"] or move["depth"] != index + 1:
            raise ValueError("Sequential evidence does not follow the reference state transition.")
        prediction = make_prediction(f"p{index + 1}", f"h{index + 1}", Metrics(**previous),
                                     Intervention(move["action"]))
        question, assumption = investigation_text(Intervention(move["action"]))
        after = Metrics(**move["after"])
        if (
            move["prediction"] != asdict(prediction)
            or move["question"] != question
            or move["hypothesis"]["id"] != prediction.hypothesis_id
            or move["hypothesis"]["claim"] != f"Working assumption, not a diagnosis: {assumption}"
            or tuple(move["hypothesis"]["evidence_ids"]) != (f"o{index}",)
            or move["before_observation_id"] != f"o{index}"
            or move["after_observation_id"] != f"o{index + 1}"
            or move["prediction_matched"] != prediction.matches(after)
            or move["service_success"] != service_restored(after)
            or move["service_utility"] != service_utility(after)
            or move["cost"] != 3 or move["cumulative_cost"] != 1 + 3 * (index + 1)
        ):
            raise ValueError("Sequential prediction, goal or cost record contradicts evidence.")
        previous = move["after"]
    final = nodes[path_key(path)]
    if (
        episode["cost"] != 1 + 3 * len(path) or not episode["cost"] <= episode["budget"] <= 20
        or episode["success"] != final["success"]
        or (episode["stop_reason"] == "service_restored") != final["success"]
    ):
        raise ValueError("Sequential terminal claim or budget contradicts measured outcome.")
    reachable = [
        node for node in oracle["nodes"] if node["path"][:len(path)] == path
        and node["success"] and node["action_cost"] <= episode["budget"]
    ]
    return {
        "success": final["success"], "utility": final["utility"], "cost": episode["cost"],
        "regret": round(oracle["best_utility"] - final["utility"], 12),
        "intervention_depth": len(path), "stop_reason": episode["stop_reason"],
        "missed_available_solution": bool(oracle["successful_paths"]) and not final["success"],
        "premature_stop_with_reachable_solution": not final["success"] and bool(reachable),
        "fuse_switches": sum(
            decision["fuse"] == "switch_to_untried_intervention" for decision in episode["decisions"]
        ),
        "local_benefit_did_not_restore_service": sum(
            move["prediction_matched"] and not move["service_success"] for move in episode["moves"]
        ),
        "budget_violations": int(episode["cost"] > episode["budget"]),
    }
