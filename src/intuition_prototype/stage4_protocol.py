"""Predeclared Stage 4 partitions and integrity controls; never executes outcomes."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import platform
import random
import shutil
import sys

from intuition_prototype.benchmark_protocol import (
    PROTOCOL as STAGE3_PROTOCOL, digest, file_digest, generate_cases as historical_cases,
    read_json, source_digest, write_json,
)
from intuition_prototype.simulator import SimulatorConfig


VERSION = "stage4-v1"
SEEDS = (101, 202)
ALPHAS = (0.01, 0.1, 1.0, 10.0)
THRESHOLDS = (0.0, 0.02, 0.05, 0.1)
COUNTS = {"train": 6, "validation": 2, "test": 6}
WORKERS = {
    "train": (4, 7, 10, 13),
    "validation": (5, 8, 11, 14),
    "test": (2, 3, 6, 9, 12, 15, 16),
}
# Arrival pairs are sampled as whole bounds, not fitted to observed behavior.
REGIMES = {
    "worker_pressure": dict(arrivals=((3, 6), (5, 8), (7, 8)), db=(28, 64),
                            locks=(0, 0.025), timeouts=(1000,), retries=(False,)),
    "db_pressure": dict(arrivals=((2, 5), (4, 8), (6, 8)), db=(1, 5),
                        locks=(0.08, 0.25), timeouts=(1000,), retries=(False,)),
    "retry_pressure": dict(arrivals=((1, 4), (3, 7), (5, 8)), db=(6, 24),
                           locks=(0.02, 0.2), timeouts=(1, 2, 3, 5), retries=(True,)),
    "mixed_pressure": dict(arrivals=((0, 8), (3, 6), (5, 8)), db=(1, 12),
                           locks=(0.06, 0.25), timeouts=(1, 2, 4, 8), retries=(False, True)),
    "healthy_headroom": dict(arrivals=((0, 0), (0, 1), (1, 1)), db=(32, 64),
                             locks=(0, 0.02), timeouts=(1000,), retries=(False,)),
    "low_signal": dict(arrivals=((0, 0), (0, 1), (0, 2), (1, 2)), db=(8, 32),
                       locks=(0, 0.08), timeouts=(1000,), retries=(False,)),
    "retry_headroom": dict(arrivals=((0, 0), (0, 1), (1, 2)), db=(28, 64),
                           locks=(0, 0.03), timeouts=(4, 8, 16), retries=(True,)),
    "near_balance": dict(arrivals=((1, 3), (2, 4), (3, 5), (0, 8)), db=(5, 24),
                         locks=(0.01, 0.12), timeouts=(3, 8, 1000), retries=(False, True)),
}
PROTOCOL = {
    "version": VERSION,
    "hypothesis": "A telemetry-only offline ridge action selector may help or hurt; no gain is assumed.",
    "scope": "Coarse worker-support/configuration holdout within known simulator mechanisms; not causal labels.",
    "history": "All 64 original Stage 3 IDs excluded from every split; old heldout is inspected history.",
    "split_unit": "Whole configuration; seed repeats never cross partitions.",
    "counts_per_regime": COUNTS,
    "worker_support": WORKERS,
    "sampling_bounds": REGIMES,
    "sampling": "Uniform discrete choices; uniform continuous DB/lock bounds rounded to 4/6 decimals.",
    "repeat_seeds": SEEDS,
    "agent_budget": 10,
    "agent_step_limit": 12,
    "warmup_ticks": 30,
    "measurement_ticks": 30,
    "policies": ["scripted", "heuristic", "learned"],
    "fit_split": "train",
    "selection_split": "validation",
    "test_use": "One official attempt after final source/model/data freeze; no outcome-based prefiltering.",
    "alpha_grid": ALPHAS,
    "threshold_grid": THRESHOLDS,
    "selection_order": [
        "maximum configuration-mean cost_adjusted_utility", "maximum configuration-mean utility",
        "minimum configuration-mean cost", "maximum alpha", "maximum threshold",
    ],
    "no_refit": "Selected model retains training-only normalization and coefficients; no validation refit.",
    "reference": deepcopy(STAGE3_PROTOCOL["reference"]),
    "statistics": {
        "unit": "Configuration mean over two correlated seed repeats",
        "bootstrap_resamples": 2000, "bootstrap_seed": 917,
        "method": "Paired configuration resampling within each regime, percentile 95% intervals",
        "comparisons": ["learned-minus-scripted", "learned-minus-heuristic"],
        "primary_endpoint": "test mean learned-minus-heuristic utility",
        "interpretation": "Descriptive conditional on this generator; no population or causal generalization.",
    },
    "failure_categories": list(STAGE3_PROTOCOL["failure_categories"]),
    "signature": "Sorted per-action beneficial flag and utility sign, with baseline zero-arrival/retry flags.",
    "signature_use": "Descriptive overlap/novelty only, after test; never selection or test prefiltering.",
    "exact_signature": (
        "SHA256 of all baseline and counterfactual telemetry, excluding labels/parameters; "
        "configuration signature retains the ordered pair of seed-window signatures."
    ),
    "abstention_rates": (
        "Report missed-benefit fraction among abstentions and abstention fraction among benefit cases; "
        "false-support fraction among supports and no-benefit false-positive fraction among no-benefit cases. "
        "Zero denominators are explicitly null, not zero. These are descriptive rates, not calibrated probabilities."
    ),
    "no_benefit_scope": STAGE3_PROTOCOL["no_benefit_scope"],
    "costs": "Agent evidence action units; oracle windows and wall-clock compute recorded separately.",
    "reproduction": (
        "Model and training manifest identity exclude timestamps, elapsed timing and absolute output paths. "
        "Nondeterministic runtime observations are separately sealed in training-runtime.json and the final archive."
    ),
    "integrity": "Portable CRLF->LF executable hashes plus exact raw archived bytes; not a signed attestation.",
    "llm_baseline": "Omitted; offline experiment, no external model calls.",
}


def generate_cases(generator_seed: int = 20260909, *, development_only: bool = False,
                   per_regime: int | None = None) -> dict[str, list[dict]]:
    if type(generator_seed) is not int or not 0 <= generator_seed < 2**32:
        raise ValueError("Generator seed must be an integer in [0, 2**32).")
    if type(development_only) is not bool:
        raise ValueError("development_only must be boolean.")
    if per_regime is not None and (
        not development_only or type(per_regime) is not int or not 1 <= per_regime <= 6
    ):
        raise ValueError("Smaller counts require development_only=True and 1..6 per regime.")
    historical = {c["config_id"] for entries in historical_cases().values() for c in entries}
    seen = set(historical)
    result = {}
    for index, split in enumerate(COUNTS):
        # Independent RNG streams keep a split stable when other split counts change.
        rng = random.Random(generator_seed + index * 2**32)
        entries = []
        for regime, bounds in REGIMES.items():
            for _ in range(per_regime or COUNTS[split]):
                low, high = rng.choice(bounds["arrivals"])
                config = SimulatorConfig(
                    workers=rng.choice(WORKERS[split]),
                    db_capacity=round(rng.uniform(*bounds["db"]), 4),
                    lock_penalty=round(rng.uniform(*bounds["locks"]), 6),
                    retry_timeout=rng.choice(bounds["timeouts"]),
                    retries_enabled=rng.choice(bounds["retries"]),
                    arrival_min=low, arrival_max=high,
                )
                parameters = asdict(config)
                config_id = digest(parameters)[:20]
                if config_id in seen:
                    raise ValueError("Duplicate or historical configuration; preparation aborted.")
                seen.add(config_id)
                entries.append(dict(config_id=config_id, split=split, regime=regime, parameters=parameters))
        result[split] = entries
    return result


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_paths() -> dict[str, Path]:
    """Project Python files, excluding runtime data, docs/tests and local build environments.

    Artifact directories are excluded only when they contain our preparation marker:
    their archived executable bytes are independently verified, never treated as live code.
    """
    root = project_root()
    ignored = {".git", ".runtime", ".venv", "venv", "__pycache__", ".pytest_cache", "tests", "docs",
               "build", "dist", "node_modules", ".mypy_cache", ".ruff_cache"}
    result = {"pyproject.toml": root / "pyproject.toml"}

    def walk(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            if path.is_symlink():
                if path.suffix == ".py":
                    raise ValueError("Executable source symlinks are not supported.")
                continue
            if path.is_dir():
                if path.name not in ignored and not (path / "preparation.json").is_file():
                    # Stage 3 archived historical source is data, not executable project code.
                    if not ((path / "manifest.json").is_file() and (path / "sources").is_dir()):
                        walk(path)
            elif path.suffix == ".py":
                result[path.relative_to(root).as_posix()] = path

    walk(root)
    return dict(sorted(result.items()))


def source_fingerprints() -> dict[str, str]:
    return {name: source_digest(path) for name, path in source_paths().items()}


def environment() -> dict:
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "platform": platform.platform(), "byteorder": sys.byteorder,
    }


def seal(directory: Path, name: str, document: dict) -> dict:
    """Exclusive write with an external digest to detect accidental manifest modification."""
    write_json(directory / name, document)
    with (directory / (name + ".sha256")).open("x", encoding="ascii", newline="\n") as out:
        out.write(file_digest(directory / name) + "\n")
    return document


def read_sealed(directory: Path, name: str) -> dict:
    expected = (directory / (name + ".sha256")).read_text(encoding="ascii").strip()
    if file_digest(directory / name) != expected:
        raise ValueError(f"Manifest digest changed: {name}")
    return read_json(directory / name)


def check_files(directory: Path, files: dict[str, str]) -> None:
    for name, expected in files.items():
        path = directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file():
            raise ValueError(f"Missing or invalid artifact: {name}")
        if file_digest(path) != expected:
            raise ValueError(f"Artifact changed: {name}")


def prepare(directory: Path, generator_seed: int = 20260909, *,
            development_only: bool = False, per_regime: int | None = None) -> dict:
    directory = Path(directory)
    cases = generate_cases(generator_seed, development_only=development_only, per_regime=per_regime)
    sources = source_fingerprints()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", PROTOCOL)
    for split, entries in cases.items():
        write_json(directory / f"{split}.json", {"configurations": entries})
    return seal(directory, "preparation.json", {
        "version": VERSION, "generator_seed": generator_seed, "development_only": development_only,
        "per_regime": per_regime, "configuration_counts": {s: len(v) for s, v in cases.items()},
        "source_sha256": sources, "environment": environment(),
        "files": {name: file_digest(directory / name) for name in
                  ("protocol.json", "train.json", "validation.json", "test.json")},
    })


def check_prepared(directory: Path, *, execution: bool = True) -> dict:
    document = read_sealed(directory, "preparation.json")
    if document["version"] != VERSION:
        raise ValueError("Unsupported Stage 4 protocol.")
    check_files(directory, document["files"])
    if execution:
        if document["source_sha256"] != source_fingerprints():
            raise ValueError("Executable source drift since preparation.")
        if document["environment"] != environment():
            raise ValueError("Python environment drift since preparation.")
        if digest(read_json(directory / "protocol.json")) != digest(PROTOCOL):
            raise ValueError("Protocol does not match executable declaration.")
        expected = generate_cases(document["generator_seed"], development_only=document["development_only"],
                                  per_regime=document["per_regime"])
        for split, cases in expected.items():
            if read_json(directory / f"{split}.json") != {"configurations": cases}:
                raise ValueError("Partition does not match public generator.")
        if document["configuration_counts"] != {s: len(v) for s, v in expected.items()}:
            raise ValueError("Declared configuration counts differ.")
    return document


def archive_files(directory: Path, files: dict[str, Path]) -> dict[str, str]:
    archive = directory / "archive"
    archive.mkdir(exist_ok=False)
    hashes = {}
    for name, source in sorted(files.items()):
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes["archive/" + name] = file_digest(target)
    return hashes
