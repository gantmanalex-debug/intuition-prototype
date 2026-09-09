"""Predeclared Stage 3 construction/provenance. No policy execution happens here."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess

from intuition_prototype.simulator import SimulatorConfig


VERSION = "stage3-v3-baseline-refactor"
REPEAT_SEEDS = (101, 202)
POLICY_NAMES = ("inquiry.py", "navigator.py", "predictions.py", "records.py")
STAGE2_HASHES = {
    "inquiry.py": "c91d1517637bad61bc4cc288735d43c292146d5d077dc8445d315b1343009989",
    "navigator.py": "1d74c0220fba508f10b92c41a748fd74d7948e596228eb852224af8b01542c1c",
    "predictions.py": "f04637fe96ba28a00659031162029cb542a98ec67d9627496b04bdac885f39fc",
    "records.py": "3743a6437fefb4cdaf99bde5e78239b5c81345a325eebffb3c951b871d1c8e11",
}
SOURCE_NAMES = (
    "__init__.py", *POLICY_NAMES, "simulator.py", "storage.py",
    "benchmark_protocol.py", "benchmark.py",
)

# Each tuple declares workers, arrival bounds, DB range, lock range, timeouts, retry settings.
REGIMES = {
    "worker_pressure": {
        "development": ((5, 6), (3, 5), (24, 32), (0.01, 0.03), (1000,), (False,)),
        "heldout": ((3, 4), (5, 7), (36, 44), (0, 0.009), (1000,), (False,)),
    },
    "db_pressure": {
        "development": ((5, 6), (3, 5), (2.5, 4), (0.08, 0.12), (1000,), (False,)),
        "heldout": ((8, 9), (5, 7), (1, 2.4), (0.16, 0.22), (1000,), (False,)),
    },
    "retry_pressure": {
        "development": ((5, 6), (3, 5), (10, 14), (0.05, 0.1), (3, 4), (True,)),
        "heldout": ((3, 4), (5, 7), (7, 9), (0.12, 0.18), (1, 2), (True,)),
    },
    "mixed_pressure": {
        "development": ((5, 6), (3, 5), (4, 7), (0.1, 0.14), (3, 4), (True,)),
        "heldout": ((3, 4), (5, 7), (1, 3), (0.18, 0.24), (1, 2), (True,)),
    },
    "healthy_headroom": {
        "development": ((10, 12), (1, 2), (36, 44), (0.01, 0.015), (1000,), (False,)),
        "heldout": ((14, 16), (0, 1), (48, 60), (0, 0.005), (1000,), (False,)),
    },
    "low_signal": {
        "development": ((8, 10), (1, 2), (16, 20), (0.02, 0.04), (1000,), (False,)),
        "heldout": ((12, 14), (0, 1), (22, 26), (0.05, 0.07), (1000,), (False,)),
    },
    "retry_headroom": {
        "development": ((10, 12), (1, 2), (36, 44), (0.01, 0.02), (5, 6), (True,)),
        "heldout": ((14, 16), (0, 1), (48, 60), (0, 0.005), (8, 10), (True,)),
    },
    "near_balance": {
        "development": ((5, 6), (2, 3), (8, 12), (0.02, 0.05), (5, 6), (False, True)),
        "heldout": ((8, 9), (3, 4), (14, 18), (0.06, 0.09), (8, 10), (False, True)),
    },
}

PROTOCOL = {
    "version": VERSION,
    "hypothesis": (
        "Heuristic selection may save investigation cost but may miss independently useful effects. "
        "No improvement in utility or regret is required or assumed."
    ),
    "primary_endpoint": "paired heldout mean utility difference: heuristic minus scripted",
    "policies": ["scripted", "heuristic"],
    "llm_baseline": "omitted: no provider/API configured; no calls made",
    "agent_budget": 10,
    "agent_step_limit": 12,
    "warmup_ticks": 30,
    "measurement_ticks": 30,
    "repeat_seeds": REPEAT_SEEDS,
    "split_unit": "whole parameter configuration, never repeated seed",
    "construction": REGIMES,
    "regime_tuple_fields": ["workers", "arrivals", "db_range", "lock_range", "timeouts", "retries"],
    "curated_stage1_families": "sanity/development only; not members of heldout",
    "data_use": (
        "Development traces may support future fitting. Held-out outcomes are evaluation-only; "
        "once inspected, these configurations are regression data, not a fresh future held-out test."
    ),
    "scope": "parameter-range/configuration holdout within known formulas; NOT mechanism generalization",
    "reference": {
        "completion_absolute_gain": 3,
        "completion_relative_gain": 0.10,
        "queue_absolute_reduction": 10,
        "queue_relative_reduction": 0.15,
        "queue_weight": 0.25,
        "rule": (
            "No loss of unique completions or increased queue growth; AND either completion gain "
            ">=max(3,ceil(.10*max(1,baseline completions))) OR queue-growth reduction "
            ">=max(10,ceil(.15*max(1,abs(baseline growth),arrivals)))."
        ),
        "utility": (
            "(completion gain + .25*queue-growth reduction)/max(1,arrivals) if independently "
            "beneficial; otherwise min(0,that value). Abstention utility=0."
        ),
        "regret": "max(0,all permitted single-intervention utilities) minus recommendation utility",
        "recommendation": "last tested action only when policy ends supported_interpretation; otherwise abstain",
        "cost_adjusted_utility": "recommendation utility / (1 + agent cost); not monetary cost",
    },
    "statistics": {
        "unit": "configuration mean over its two correlated seed repeats",
        "bootstrap": "2000 paired configuration resamples within each regime; percentile 95% intervals",
        "bootstrap_seed": 917,
        "interpretation": "descriptive conditional on this generator, not population causal inference",
    },
    "failure_categories": [
        "supported_but_unhelpful", "abstained_with_available_benefit",
        "missed_despite_tested_benefit", "suboptimal_recommendation",
    ],
    "no_benefit_scope": "no qualifying effect of the permitted single interventions in this window, not no possible cure",
    "preflight_invalidation": (
        "Development-only integration exposed non-deterministic Markdown ordering after JSON reload "
        "and table spacing. Those preflight artifacts were invalidated and regenerated after a renderer "
        "fix. No held-out policy outcomes had been executed or inspected; policy bytes and scoring "
        "criteria were unchanged. The original v1 full run was the first held-out attempt."
    ),
    "portability_revision": (
        "v1 raw source hashes were checkout-line-ending dependent. v2 normalizes CRLF to LF "
        "for source identity and separately retains exact raw archive hashes. Policies, generator "
        "and scoring rules are unchanged. Revalidation of already inspected v1 cases is a rerun, "
        "not fresh held-out evidence."
    ),
    "baseline_refactor_revision": (
        "v3 shares the unchanged heuristic inquiry loop with Stage 4 and extends typed records. "
        "Its fixed source checkpoints differ from abdc78b; weights, predictions, simulator and "
        "external scoring are unchanged. The original v1/v2 artifacts remain readable and intact. "
        "New Stage 3 runs are historical regression, never fresh held-out evidence or Stage 4 training."
    ),
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_fingerprints() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: source_digest(root / name) for name in SOURCE_NAMES}


def generate_cases(
    generator_seed: int = 20260908, development_per_regime: int = 2, heldout_per_regime: int = 6,
) -> dict[str, list[dict]]:
    if type(generator_seed) is not int or not 0 <= generator_seed < 2**32:
        raise ValueError("Generator seed must be an integer in [0, 4294967295].")
    if any(type(count) is not int or not 1 <= count <= 8 for count in (
        development_per_regime, heldout_per_regime,
    )):
        raise ValueError("Each split must contain 1..8 configurations per regime.")
    random_source = random.Random(generator_seed)
    result: dict[str, list[dict]] = {"development": [], "heldout": []}
    ids = set()
    for split, count in (("development", development_per_regime), ("heldout", heldout_per_regime)):
        for regime, ranges in REGIMES.items():
            workers, arrivals, database, locks, timeouts, retries = ranges[split]
            for _ in range(count):
                config = SimulatorConfig(
                    workers=random_source.choice(workers),
                    db_capacity=round(random_source.uniform(*database), 4),
                    lock_penalty=round(random_source.uniform(*locks), 6),
                    retry_timeout=random_source.choice(timeouts),
                    retries_enabled=random_source.choice(retries),
                    arrival_min=arrivals[0], arrival_max=arrivals[1],
                )
                parameters = asdict(config)
                config_id = digest(parameters)[:20]
                if config_id in ids:
                    raise ValueError("Duplicate parameter configuration; preparation aborted.")
                ids.add(config_id)
                result[split].append({
                    "config_id": config_id, "split": split, "regime": regime, "parameters": parameters,
                })
    return result


def write_json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def prepare(
    directory: Path, generator_seed: int = 20260908,
    development_per_regime: int = 2, heldout_per_regime: int = 6,
    rerun_of: str | None = None, invalidation_reason: str | None = None,
) -> dict:
    if (rerun_of is None) != (invalidation_reason is None):
        raise ValueError("A rerun must name its prior run and explicit invalidation reason.")
    fingerprints = source_fingerprints()
    if {name: fingerprints[name] for name in POLICY_NAMES} != STAGE2_HASHES:
        raise ValueError("Frozen Stage 2 policy source changed. Do not tune or silently refreeze this protocol.")
    cases = generate_cases(generator_seed, development_per_regime, heldout_per_regime)
    directory.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).parent
    snapshots = directory / "sources"
    snapshots.mkdir()
    for name in SOURCE_NAMES:
        shutil.copyfile(source_root / name, snapshots / name)
    freeze = {
        "source_hash_mode": "crlf-to-lf-v1",
        "source_sha256": fingerprints,
        "raw_source_sha256": {name: file_digest(source_root / name) for name in SOURCE_NAMES},
        "policy_sha256": digest({name: fingerprints[name] for name in POLICY_NAMES}),
        "python": platform.python_version(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root.parents[1], text=True,
        ).strip(),
        "includes_uncommitted_sources": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_root.parents[1], text=True,
        ).strip()),
    }
    write_json(directory / "freeze.json", freeze)
    write_json(directory / "protocol.json", PROTOCOL)
    for split, entries in cases.items():
        write_json(directory / f"{split}.json", {"configurations": entries})
    manifest = {
        "version": VERSION, "generator_seed": generator_seed,
        "files": {name: file_digest(directory / name) for name in (
            "freeze.json", "protocol.json", "development.json", "heldout.json",
        )},
        "configuration_counts": {split: len(entries) for split, entries in cases.items()},
        "policy_sha256": freeze["policy_sha256"],
        "attempt_classification": "rerun" if rerun_of else "historical_regression",
        "rerun_of": rerun_of, "invalidation_reason": invalidation_reason,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def check_manifest(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    if manifest["version"] not in ("stage3-v1", "stage3-v2-portable", VERSION):
        raise ValueError("Unknown benchmark version.")
    for name, expected in manifest["files"].items():
        if file_digest(directory / name) != expected:
            raise ValueError(f"Declared benchmark artifact changed: {name}")
    freeze = read_json(directory / "freeze.json")
    legacy = manifest["version"] == "stage3-v1"
    if not legacy and freeze["source_hash_mode"] != "crlf-to-lf-v1":
        raise ValueError("Unknown source hash normalization.")
    for name, expected in freeze["source_sha256"].items():
        archived = directory / "sources" / name
        actual = file_digest(archived) if legacy else source_digest(archived)
        if actual != expected:
            raise ValueError(f"Archived source fingerprint changed: {name}")
        if not legacy and file_digest(archived) != freeze["raw_source_sha256"][name]:
            raise ValueError(f"Raw archived source bytes changed: {name}")
    return manifest


def check_frozen(directory: Path) -> dict:
    manifest = check_manifest(directory)
    if manifest["version"] != VERSION:
        raise ValueError("Legacy results are readable, but execution requires a declared regression replay.")
    freeze = read_json(directory / "freeze.json")
    if source_fingerprints() != freeze["source_sha256"]:
        raise ValueError("Source fingerprint changed after preparation; invalidate and explicitly declare a rerun.")
    if {name: freeze["source_sha256"][name] for name in POLICY_NAMES} != STAGE2_HASHES:
        raise ValueError("Frozen policy provenance is not the recorded Stage 2 baseline refactor.")
    if platform.python_version() != freeze["python"]:
        raise ValueError("Python version differs from prepared benchmark.")
    return manifest
