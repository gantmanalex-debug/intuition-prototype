# Intuition: local environment and Stage 1 scripted baseline

The environment scaffold now includes **Stage 1: a bounded service simulator
and an explicitly SCRIPTED inquiry baseline**, not learned intuition.
It provides an importable Python package, standard-library SQLite storage,
pytest tests, and a loopback-only Gradio health/demo and investigation interface.
No GPU, paid provider, credentials, model downloads, or LLM API calls are needed.
Successful scripted traces do **not** validate the intuition research hypothesis.

## Selected platform

- Windows host with existing WSL2 Ubuntu 24.04.3 LTS.
- Ubuntu Python 3.12.3 and SQLite 3.45.1.
- Verified with Gradio 6.26.0, pytest 9.1.1, and pip 26.2.1.
- Project: `C:\Users\AlexGantman\Projects\intuition-prototype`.
- Isolated **Linux** virtual environment: `.venv` in the project.
  Do not use this environment with Windows Python.

No Windows features, distributions, global Python packages, or global settings
were changed. Ubuntu lacks pip/ensurepip, so pip was bootstrapped into `.venv`
without sudo. The existing Windows `uv` is not needed for this WSL environment.

## Open, run, and test

From PowerShell, enter Ubuntu at the project directory:

```powershell
wsl -d Ubuntu --cd 'C:\Users\AlexGantman\Projects\intuition-prototype'
```

Then run these commands **inside Ubuntu**, not PowerShell:

```bash
source .venv/bin/activate
python -m pytest -q
python -m intuition_prototype.app
```

Open **http://127.0.0.1:7860** in the Windows browser. The app always binds to
`127.0.0.1`, uses `share=False`, and disables Gradio analytics and monitoring.
Stop a foreground server with Ctrl+C. Stop an already running server before
launching a second instance; an occupied port is reported rather than silently
switching ports. `intuition-ui` is an equivalent entry point after activation.
Browser run history is disabled, and the theme uses local system fonts.

In VS Code, use the WSL extension and **WSL: Reopen Folder in WSL** if available,
then select `.venv/bin/python` as the Python interpreter. Installing editor
extensions is optional and was not part of setup.

## Recreate dependencies

An internet connection is needed for installation, not for an LLM service.
With Python 3.12+ and working ensurepip, run inside Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip --isolated install -e '.[dev]'
python -m pytest -q
```

On this Ubuntu installation, ensurepip is absent. For a **new** environment,
use this no-admin alternative instead:

```bash
python3 -m venv --without-pip .venv
mkdir -p .runtime
curl --fail --location https://bootstrap.pypa.io/pip/pip.pyz --output .runtime/pip.pyz
.venv/bin/python .runtime/pip.pyz --isolated install pip
rm .runtime/pip.pyz
.venv/bin/python -m pip --isolated install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Dependencies are declared in `pyproject.toml`: Gradio >=6.26,<7 and the `dev` extra
with pytest. SQLite comes with Python. This modest scaffold does not pin a
complete transitive lockfile; package versions may change on a fresh install.
Installation on this Windows-mounted directory can be slow because WSL crosses
the filesystem boundary. No administrator installation is required.

## Configuration and data

The app reads process environment variables. `.env.example` documents them;
it is **not automatically loaded**, and no real `.env` or credentials are needed.

| Variable | Default | Meaning |
| --- | --- | --- |
| `INTUITION_MODE` | `demo` | Only supported mode; other values fail explicitly. |
| `INTUITION_DB_PATH` | `.runtime/intuition.sqlite3` | SQLite path, relative to the current working directory unless absolute. |

The environment panel initializes the SQLite schema and displays runtime versions, database
location, and the result of a database query. The demo simply echoes input with
an explicit **not an LLM response** label; echo inputs are not persisted.
The storage module has a small parameterized write/read API verified by tests.
This original API and its data remain intact. Stage 1 adds separate episode and
typed-record tables; its simulator traces **are** persisted locally.

Local databases, runtime files, credentials, virtual environments, and caches
are excluded by `.gitignore`. This is a local development UI, not an authenticated
multi-user service. Do not expose it through a proxy, public tunnel, or LAN bind.

## Stage 1: running and evaluating

Use the existing environment/run/test commands above. In the UI select
`worker_capacity`, `db_contention`, or `retry_amplification`, set a seed and
budget, and click **Run scripted investigation**. The trace shows questions,
preregistered predictions, actions, observations, discrepancies, claim updates,
evidence references, action costs, and the stop reason.

Every Run constructs a fresh simulator. **Reset scenario / clear viewer** clears
the displayed run and validates the scenario/seed; it does not delete saved
episodes. Rerunning with the same seed/budget reproduces the same evidence and
decisions, but receives a new persistence ID.

Inside the activated WSL environment, evaluate without Gradio:

```bash
python -m intuition_prototype.stage1 --scenario retry_amplification --seed 7 --budget 10
python -m intuition_prototype.stage1 --scenario worker_capacity --seed 7 --json
python -m intuition_prototype.stage1 --scenario db_contention --seed 7 --json
python -m pytest -q tests/test_stage1.py tests/test_smoke.py
```

The CLI also persists the episode; `--database` selects another local SQLite
file. `storage.load_episode(path, id)` returns the typed saved episode, preserving
ordered observations, hypotheses, predictions, questions, actions, and results.
Claims reference earlier evidence/claims rather than replacing observations.
SQLite writes an entire episode transactionally; the original demo table is
unchanged. Predictions are stored before the corresponding intervention/result.

### Simulator mechanisms and bounds

This is a discrete-time, single-service FIFO queue, not a table of canned
intervention responses. All outcomes are calculated by advancing attempts:

- Each tick receives a seeded uniform integer of 3..5 new requests. Each attempt
  requires 2 units of DB work and occupies one worker until it completes.
- Default workers: 6. At most one work unit per worker per tick is processed.
  With `n` busy workers, total effective DB capacity is
  `db_capacity / (1 + lock_penalty * (n - 1))`; capacity is shared equally.
  Extra workers can therefore increase lock-coordination overhead.
- At a request's deadline, an unfinished request creates another attempt,
  up to two retries. Existing duplicates are **not cancelled** when a request
  completes. Their completions consume work but do not count as unique service.
- Retry suppression prevents **new** retries only; already queued duplicates
  remain. Short-window useful throughput need not recover.
- State includes active/queued attempts, request completion/retry state, current
  tick, resource settings, and RNG state. Opaque same-instance snapshot tokens
  restore all of these before each counterfactual trial. The warm-up is 30 ticks;
  baseline and each intervention measure the next 30 ticks with identical arrivals.
- Each timeline has at most 240 ticks, queue plus active work is capped at 4096
  attempts, requests have at most two retries, and each instance holds at most
  eight snapshots. A reset invalidates snapshots. Tick/snapshot/action validation
  errors are explicit; queue overflow counts dropped attempts.

Evaluator-only family parameters are:

| Family | DB capacity | Lock penalty | Retry deadline (ticks) |
| --- | ---: | ---: | ---: |
| Worker capacity | 40 | 0.01 | 1000 |
| DB contention | 3 | 0.12 | 1000 |
| Retry amplification | 9 | 0.12 | 2 |

The runner receives only the observation/action/snapshot protocol, not the family,
seed, parameter table, or ground-truth labels. Case selection and metadata belong
to the UI/CLI evaluation harness. Python private fields are an architectural
boundary, not a security sandbox against hostile introspection.

### SCRIPTED policy, predictions, and costs

The fixed policy first asks for baseline evidence (cost 1), provisionally proposes
insufficient workers, then tests doubling workers (cost 2). Success requires
at least 20% more unique completions **and** strictly lower queue growth.
A reframe/interpretation update costs 1. On discrepancy it next tests disabling
new retries, then (if inconclusive) doubling DB capacity, always from the same
snapshot. These are the only three permitted interventions.

The retry prediction is deliberately different: no fewer unique completions,
at least 30% lower queue growth (also strictly lower), and zero new retries.
It tests retry contribution to queue pressure, not immediate throughput recovery.
The DB prediction uses the same throughput/growth criteria as the worker test.
All thresholds are recorded before each trial. A match supports an interpretation;
it does not prove a unique cause. The policy is fixed, with outcome-based branches,
not a learned selector, LLM, or an open-ended planner.

Budgets are integer 0..20 (default 10). No action runs if it cannot be afforded.
A supported interpretation terminates the run; otherwise it stops on budget
exhaustion or after all three investigations. Results may be recorded without
an interpretation update if the remaining budget cannot afford a reframe.

For seed 7, the retry acceptance episode's 30-tick measurements are:

| Same-state trial | Unique completions | New retries | Queue growth |
| --- | ---: | ---: | ---: |
| Baseline | 21 | 244 | 305 |
| Double workers | 18 | 244 | 315 |
| Disable new retries | 21 | 0 | 61 |

The first prediction fails; the runner records the discrepancy and weakens
the worker-only explanation. The retry test reduces growth without reducing
completions, so it supports retry contribution to queue growth at total cost 7.
It explicitly does **not** claim throughput recovery.

### Limitations and deferred research

The lock formula, retry rule, family parameters, 30-tick horizon, and thresholds
are declared modeling assumptions, not measurements of a production service.
Families can involve overlapping bottlenecks. Outcomes depend on these choices;
this small suite is not a causal-identifiability or generalization benchmark.
Disabling retry injection mechanically reduces offered attempt load; that result
alone is not evidence of a learned investigative ability.

There is no network, real DB contention, backoff/jitter, cancellation, fairness
scheduler, workload drift, adaptive arrival process, or independent CPU stage.
Unused per-tick capacity is not redistributed after an attempt completes.
Mean latency covers completed requests only, so it is survivor-biased; unfinished
work is visible separately in queue growth. Replay is checked in the selected
Python environment; bit-for-bit cross-version replay is not promised.
SQLite retains run history until manually managed; there is no retention service.

Learned intuition, learned action selection, fuse training, real remote LLM
integration, unrestricted real-world tools, and research validation remain deferred.

## Architecture boundary

- `config.py`: explicit validated demo configuration.
- `llm.py`: replaceable `LLMAdapter` protocol and deterministic `DemoAdapter`.
- `records.py`: typed evidence, claims, predictions, unified ask/test/reframe actions,
  results, and episodes.
- `simulator.py`: bounded queue dynamics and opaque snapshot/reset API.
- `inquiry.py`: explicitly scripted, budgeted investigation policy.
- `storage.py`: original demo storage plus transactional typed episode persistence.
- `stage1.py`: evaluator orchestration, persisted trace rendering, and CLI.
- `app.py`: environment health, offline echo, and local Stage 1 Gradio interface.
- `tests/test_smoke.py`: imports/configuration, SQLite initialize/write/read,
  health reporting, and UI construction.
- `tests/test_stage1.py`: three families, acceptance trace, deterministic replay,
  action/budget bounds, evidence separation, persistence, and reset semantics.

The Stage 1 records and scripted ask/test/reframe loop are foundations for later
research, not an implementation of the proposed learned controller.

A future remote adapter must explicitly implement the protocol, extend mode
validation and the factory, and define credentials, timeouts, error handling,
and user-approved cost controls. The current scaffold deliberately selects no
provider and rejects `remote` rather than silently falling back to demo.
