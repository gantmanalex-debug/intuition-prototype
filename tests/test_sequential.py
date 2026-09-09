from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import pytest

from intuition_prototype.benchmark_protocol import canonical_json, digest
from intuition_prototype.learned import TrainingExample, fit_model
from intuition_prototype.records import Action, ActionKind, Intervention, Metrics
from intuition_prototype.sequential import (
    MAX_DEPTH, ORDER, POLICIES, WAIT, assess_episode, cohort, environment, legal_paths,
    reference, run_sequential, service_restored, service_utility,
)
from intuition_prototype.sequential_protocol import generate_cases, SEEDS
from intuition_prototype.simulator import QueueSimulator, SimulatorConfig, Snapshot


@pytest.fixture(scope="module")
def deep_case():
    return next(
        case for case in generate_cases()["development"] if case["config_id"] == "4a6b0d1760a86070d05d"
    )


@pytest.fixture(scope="module")
def deep_reference(deep_case):
    return reference(SimulatorConfig(**deep_case["parameters"]), SEEDS[0])


def dev_model(metrics, values=None):
    values = values or {"double_workers": .2, "disable_retries": .8, "double_db_capacity": .4}
    return fit_model([TrainingExample("synthetic-sequential-unit-fixture", 307, "train", metrics, values)],
                     .1, {"purpose": "development unit control flow, not the frozen transfer experiment"})


def test_exhaustive_lower_bound_requires_three_distinct_cumulative_repairs(deep_reference):
    paths = legal_paths()
    assert len(paths) == 52
    assert len({tuple(path) for path in paths}) == 52
    assert {length: sum(len(path) == length for path in paths) for length in range(4)} == {
        0: 1, 1: 4, 2: 13, 3: 34,
    }
    assert {tuple(node["path"]) for node in deep_reference["nodes"]} == set(paths)
    assert deep_reference["minimum_depth"] == 3
    assert deep_reference["requires_three_distinct_interventions"]
    assert all(not node["success"] for node in deep_reference["nodes"] if len(node["path"]) < 3)
    assert all(not node["success"] for node in deep_reference["nodes"] if WAIT in node["path"])
    assert deep_reference["successful_paths"] == [
        ["disable_retries", "double_db_capacity", "double_workers"],
        ["disable_retries", "double_workers", "double_db_capacity"],
    ]
    assert all(node["action_cost"] <= 10 for node in deep_reference["nodes"])
    assert not any(
        node["success"] for node in deep_reference["nodes"]
        if len([key for key in node["path"] if key != WAIT]) < 3
    )


def test_positive_three_step_path_and_order_dependence(deep_case, deep_reference):
    simulator = environment(SimulatorConfig(**deep_case["parameters"]), SEEDS[0])
    evidence = [simulator.observe()]
    for index, action in enumerate(deep_reference["successful_paths"][0]):
        evidence.append(simulator.move(action))
        assert service_restored(evidence[-1]) == (index == 2)
    assert not simulator.capabilities()
    assert len({digest(asdict(item)) for item in evidence}) == 4
    assert evidence[1].queue_start == evidence[0].queue_end
    assert evidence[2].queue_start == evidence[1].queue_end
    assert evidence[3].queue_start == evidence[2].queue_end
    wrong_order = next(node for node in deep_reference["nodes"]
                       if node["path"] == [item.value for item in ORDER])
    assert not wrong_order["success"]
    assert wrong_order["metrics"] != asdict(evidence[-1])
    direct = QueueSimulator.from_config(SimulatorConfig(**deep_case["parameters"]), SEEDS[0])
    assert direct.perform(Action(ActionKind.ASK)) == evidence[0]
    for index, action in enumerate(deep_reference["successful_paths"][0], start=1):
        assert direct.perform(Action(ActionKind.TEST, Intervention(action))) == evidence[index]


def test_shallow_control_has_a_real_one_step_solution_not_three_step_gate(deep_case, deep_reference):
    reference_control = reference(SimulatorConfig(**deep_case["control"]["parameters"]), SEEDS[0])
    assert reference_control["minimum_depth"] == 1
    assert not reference_control["requires_three_distinct_interventions"]
    simulator = environment(SimulatorConfig(**deep_case["control"]["parameters"]), SEEDS[0])
    episode = run_sequential(simulator, POLICIES[0], dev_model(simulator.observe()))
    assert episode["success"]
    assert episode["cost"] == 4
    assert len(episode["moves"]) == 1
    assert episode["moves"][0]["action"] == "double_workers"
    assert assess_episode(episode, reference_control)["success"]
    assert cohort([deep_reference, deep_reference], [reference_control, reference_control]) == (True, True)
    assert cohort([reference_control, deep_reference], [reference_control, reference_control]) == (False, False)
    with pytest.raises(ValueError, match="two declared"):
        cohort([deep_reference], [reference_control])


def test_api_legality_snapshot_and_no_information_leakage(deep_case):
    config = SimulatorConfig(**deep_case["parameters"])
    simulator = environment(config, 307)
    baseline = simulator.observe()
    snapshot = simulator.snapshot()
    assert set(vars(snapshot)) == {"token"}
    assert all(not hasattr(simulator, field) for field in ("config", "scenario", "seed", "parameters", "oracle"))
    initial_capabilities = simulator.capabilities()
    changed = simulator.move("disable_retries")
    assert changed.retries == 0
    with pytest.raises(ValueError, match="Illegal"):
        simulator.move("disable_retries")
    simulator.restore(snapshot)
    assert simulator.observe() == baseline
    assert simulator.capabilities() == initial_capabilities
    assert simulator.move("disable_retries") == changed
    with pytest.raises(ValueError, match="Unknown"):
        simulator.restore(Snapshot("not-issued"))
    with pytest.raises(ValueError, match="Illegal"):
        simulator.move("invented_tool")
    simulator.move(WAIT)
    simulator.move(WAIT)
    with pytest.raises(ValueError, match="depth-exhausted"):
        simulator.move(WAIT)


@pytest.mark.parametrize("budget", [-1, True, 21, 1.5])
def test_invalid_budget_is_an_explicit_error(deep_case, budget):
    api = environment(SimulatorConfig(**deep_case["parameters"]), SEEDS[0])
    with pytest.raises(ValueError, match="Budget must"):
        run_sequential(api, POLICIES[0], dev_model(api.observe()), budget)


@pytest.mark.parametrize("depth", [-1, True, 4])
def test_reference_cannot_silently_shorten_or_expand_its_depth(depth):
    with pytest.raises(ValueError, match="Reference depth"):
        legal_paths(depth)


@pytest.mark.parametrize("policy", POLICIES)
def test_deterministic_policy_and_independent_assessment(deep_case, deep_reference, policy):
    config = SimulatorConfig(**deep_case["parameters"])
    simulator = environment(config, SEEDS[0])
    model = dev_model(simulator.observe())
    first = run_sequential(simulator, policy, model)
    second = run_sequential(environment(config, SEEDS[0]), policy, model)
    assert first == second
    assert canonical_json(json.loads(json.dumps(first))) == canonical_json(first)
    score = assess_episode(first, deep_reference)
    assert score["cost"] <= 10 and score["intervention_depth"] <= 3
    assert score["success"] == service_restored(simulator.observe())
    assert score["utility"] == service_utility(simulator.observe())
    assert first["decisions"][0]["evidence"] == deep_reference["nodes"][0]["metrics"]
    if first["success"]:
        assert len(first["moves"]) == 3
    corrupt = deepcopy(first)
    corrupt["success"] = not corrupt["success"]
    with pytest.raises(ValueError, match="terminal"):
        assess_episode(corrupt, deep_reference)
    if first["moves"]:
        corrupt = deepcopy(first)
        corrupt["moves"][0]["after"]["completions"] += 1
        with pytest.raises(ValueError, match="transition"):
            assess_episode(corrupt, deep_reference)


@pytest.mark.parametrize("budget", [0, 1, 3, 4, 7, 10, 20])
@pytest.mark.parametrize("policy", POLICIES)
def test_budget_counts_every_actual_test_and_failure(deep_case, budget, policy):
    simulator = environment(SimulatorConfig(**deep_case["parameters"]), SEEDS[0])
    result = run_sequential(simulator, policy, dev_model(simulator.observe()), budget)
    assert result["cost"] <= budget
    assert result["cost"] == (0 if budget == 0 else 1 + 3 * len(result["moves"]))
    assert len(result["moves"]) <= MAX_DEPTH
    if budget < 10:
        assert not result["success"]
    assert all(move["cost"] == 3 for move in result["moves"])
    assert len({move["action"] for move in result["moves"]}) == len(result["moves"])


def test_fuse_and_local_prediction_are_not_terminal_goal(deep_case):
    simulator = environment(SimulatorConfig(**deep_case["parameters"]), SEEDS[0])
    result = run_sequential(simulator, POLICIES[2], dev_model(simulator.observe()))
    assert result["success"] and len(result["moves"]) == 3
    assert any(not move["prediction_matched"] for move in result["moves"])
    assert any(decision["fuse"] == "switch_to_untried_intervention" for decision in result["decisions"][1:])
    assert result["decisions"][1]["evidence"] == result["moves"][0]["after"]
    assert result["moves"][1]["before"] == result["moves"][0]["after"]
    assert all("not a same-state causal contrast" in move["interpretation"] for move in result["moves"])
    low = dev_model(environment(SimulatorConfig(**deep_case["parameters"]), SEEDS[0]).observe(),
                    {item.value: -.1 for item in Intervention})
    abstention = run_sequential(environment(SimulatorConfig(**deep_case["parameters"]), SEEDS[0]), POLICIES[2], low)
    assert abstention["cost"] == 1
    assert abstention["stop_reason"] == "no_eligible_intervention"
    assert all(candidate["predicted_utility"] < 0
               for candidate in abstention["decisions"][0]["candidates"])


def test_local_benefit_does_not_end_the_versioned_remediation_adapter():
    class LocalBenefit:
        def __init__(self):
            self.current = Metrics(30, 100, 20, 100, 0, 20, 100, 0, 10)
            self.available = [item.value for item in Intervention]

        def observe(self):
            return self.current

        def capabilities(self):
            return tuple(self.available)

        def move(self, key):
            self.available.remove(key)
            completion, growth = {
                "disable_retries": (20, 40), "double_db_capacity": (50, 10), "double_workers": (100, 0),
            }[key]
            self.current = Metrics(30, 100, completion, 0, 0, 100, 100 + growth, 0, 10)
            return self.current

    api = LocalBenefit()
    result = run_sequential(api, POLICIES[2], dev_model(api.observe()))
    assert result["moves"][0]["prediction_matched"]
    assert not result["moves"][0]["service_success"]
    assert "local_prediction_met_but_service_goal_unmet" in result["moves"][0]["fuse_triggers"]
    assert result["success"] and len(result["moves"]) == 3


def test_agent_only_needs_observation_capabilities_and_move(deep_case):
    simulator = environment(SimulatorConfig(**deep_case["parameters"]), SEEDS[0])

    class SafeView:
        __slots__ = ()

        def observe(self):
            return simulator.observe()

        def capabilities(self):
            return simulator.capabilities()

        def move(self, key):
            return simulator.move(key)

    episode = run_sequential(SafeView(), POLICIES[2], dev_model(simulator.observe()))
    assert episode["success"]


def test_original_executable_sources_remain_byte_identical_to_published_stage4():
    root = Path(__file__).resolve().parents[1]
    historical = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "88a7de2236e839a06cced5a5b34c220c3ffada17",
         "src", "pyproject.toml"], cwd=root, text=True,
    ).splitlines()
    for name in historical:
        prior = subprocess.check_output(
            ["git", "show", f"88a7de2236e839a06cced5a5b34c220c3ffada17:{name}"], cwd=root,
        )
        assert (root / name).read_bytes().replace(b"\r\n", b"\n") == prior, name
