"""Declared structural eligibility, controls and frozen transfer experiment."""

from dataclasses import asdict
import random

from intuition_prototype.benchmark_protocol import digest, generate_cases as stage3_cases
from intuition_prototype.simulator import SimulatorConfig
from intuition_prototype.stage4_protocol import generate_cases as stage4_cases


VERSION = "sequential-benchmark-v2"
SEEDS = (307, 409)
MODEL_SHA256 = "ea8ceb42894b03a1500b594855c4d45a9da5cba21f12d1cd5ca7993ac1e375fe"
COUNTS = {"development": 80, "test": 160}
WORKERS = {"development": (4, 6, 8, 10), "test": (3, 5, 7, 9, 11, 12)}
PROTOCOL = {
    "version": VERSION, "generator_seed": 20260910,
    "hypothesis": (
        "A frozen one-step learned selector might compare differently on remediation requiring three "
        "distinct interventions. Worse performance is acceptable and does not invalidate prior results."
    ),
    "task": "Operational service remediation, NOT evidence-supported diagnosis or steps needed to guess a cause.",
    "mechanisms": "Original queue simulator unchanged; actions now persist and time advances after each move.",
    "goal": "Positive arrivals; unique completions >= arrivals; queue growth <= 0; dropped attempts == 0.",
    "utility": "min(1,completions/max(1,arrivals))-max(0,queue_growth)/max(1,arrivals)-drops/max(1,arrivals)",
    "cost": "Initial measurement 1; each intervention 2 plus evidence update 1; no rollback or free failed tests.",
    "budget": 10, "max_interventions": 3, "window_ticks": 30, "warmup_ticks": 30,
    "initial_measurement_ticks": 30, "seeds": SEEDS,
    "sampling": {
        "counts": COUNTS, "worker_supports": WORKERS,
        "arrival_bounds": ((2, 4), (3, 5), (4, 6), (3, 7), (5, 7)),
        "db_range": (5, 32), "lock_range": (0.0, 0.12),
        "timeouts": (1, 2, 3, 4, 6), "retries_enabled": True,
        "sampling_rule": "Uniform choices/ranges; DB rounded to4decimals and lock to6decimals.",
    },
    "history": "Exclude exact configuration IDs from all original Stage3 and Stage4 partitions.",
    "oracle": (
        "Exhaust all52legal prefixes depth0..3: three once-only interventions plus repeatable observe/wait. "
        "Report every endpoint; lower bound is in this bounded action/time model, not human reasoning."
    ),
    "eligibility": (
        "Deep: both seeds have minimum depth3 and ALL successful prefixes use three distinct interventions, "
        "never a wait substitution. Shallow: both seeds have minimum depth1. All others excluded with "
        "reference summaries retained. Filtering uses reference only, before policy execution, never policy wins."
    ),
    "controls": (
        "For every base configuration declare a paired control with DB capacity already doubled and retries "
        "disabled, otherwise identical parameters/seeds. Include it in the matched shallow cohort only if "
        "its parent qualifies deep and BOTH control seeds have minimum depth1. Same goal/dynamics/budget; "
        "initial remediation changes the telemetry/parameters, so this is not causal isolation of depth."
    ),
    "selection": "Evaluate every eligible deep parent and matched shallow control; no policy-outcome selection.",
    "development_revision": (
        "v1's declared80development base configurations yielded3certified deep and0shallow before any "
        "policy outcomes or test reference execution. v2 adds structurally derived paired shallow controls, "
        "preserving every base configuration. No test outcomes guided this change; v1 declaration retained."
    ),
    "adapters": {
        "names": ["scripted-sequential-v1", "heuristic-sequential-v1", "learned-transfer-sequential-v1"],
        "scripted": "Original worker/retry/DB order, but continue until service goal, not local benefit.",
        "heuristic": "Original score on latest telemetry with only unused capabilities; no hidden config.",
        "learned": (
            "Frozen Stage4 one-step utility coefficients/normalization/threshold; rescore current telemetry. "
            "Strict original utility threshold applies also after failed tests; no forced three-step gate."
        ),
        "wait": "Environment allows waiting; inherited selectors have no wait estimate and rank interventions only.",
        "prediction": "Existing local hypothesis criteria retained separately from learned utility and service goal.",
        "fuse": "After mismatch/stall or local-only benefit, assess unused candidates; switch only if eligible.",
    },
    "training": "None. Frozen transfer only; no sequential-target model fitting or tuning.",
    "model_sha256": MODEL_SHA256,
    "statistics": (
        "Parent configuration unit keeps both seeds, all action labels and its matched control together. "
        "Paired learned-minus-each-baseline means;2000parent-unit bootstrap resamples within each depth "
        "cohort, seed917. Also report matched depth difference of policy deltas, not causal evidence."
    ),
    "failure_rule": "One-shot markers; failed test requires explicit invalidation, no silent reruns.",
    "scope": "Same known mechanisms, changed operational task; not an explanation that erases Stage4 failure.",
}


def generate_cases() -> dict[str, list[dict]]:
    previous = {
        case["config_id"]
        for collection in (stage3_cases(), stage4_cases())
        for cases in collection.values() for case in cases
    }
    result = {}
    for index, (split, count) in enumerate(COUNTS.items()):
        rng = random.Random(PROTOCOL["generator_seed"] + index * 2**32)
        entries = []
        for _ in range(count):
            low, high = rng.choice(PROTOCOL["sampling"]["arrival_bounds"])
            config = SimulatorConfig(
                workers=rng.choice(WORKERS[split]),
                db_capacity=round(rng.uniform(*PROTOCOL["sampling"]["db_range"]), 4),
                lock_penalty=round(rng.uniform(*PROTOCOL["sampling"]["lock_range"]), 6),
                retry_timeout=rng.choice(PROTOCOL["sampling"]["timeouts"]),
                retries_enabled=True, arrival_min=low, arrival_max=high,
            )
            parameters = asdict(config)
            identity = digest(parameters)[:20]
            if identity in previous:
                raise ValueError("Duplicate or historical sequential configuration; declaration rejected.")
            previous.add(identity)
            entries.append(dict(config_id=identity, split=split, parameters=parameters))
        result[split] = entries
    for split, entries in result.items():
        for case in entries:
            parameters = {**case["parameters"], "db_capacity": case["parameters"]["db_capacity"] * 2,
                          "retries_enabled": False}
            identity = digest(parameters)[:20]
            if identity in previous:
                raise ValueError("Duplicate or historical control configuration; declaration rejected.")
            previous.add(identity)
            case["control"] = dict(config_id=identity, split=split, parameters=parameters)
    return result
