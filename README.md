# Intuition: bounded simulation and transparent inquiry policies

## Abstract

This project investigates whether an intelligent system can improve its reasoning
by choosing what to question next, rather than merely producing longer answers.
Clarification, experimentation, and reframing are treated as related ways to
investigate uncertainty: evidence can justify continuing an inquiry, changing its
direction, or stopping. The long-term goal is a learned selection policy that
recognizes promising investigations, including opportunities revealed by
unexpected results.

The current prototype uses a reproducible service simulator, explicit hypotheses
and evidence, persistent inquiry traces, a scripted baseline, a transparent
heuristic navigator, and a small offline-fitted action selector with a bounded
intelligence fuse. The learned component selects among predefined investigations;
it does not discover new questions or tools. Frozen-policy comparative
evaluation measures investigation usefulness and cost independently of the
policies' own conclusions, within the simulator's declared mechanisms. This is
research infrastructure, not a demonstration of AGI, consciousness, or learned
intuition, and it currently makes no LLM calls.

## Current status

**Stage 4 adds a small offline LEARNED action selector:** fitted ridge utility
regressors select among the same three bounded interventions, alongside the
preserved SCRIPTED default and transparent HEURISTIC navigator/intelligence fuse.
None is learned intuition or an LLM. Stage 4 uses new development-only fitting
and validation data and a separately declared, frozen-model test split.
Its first frozen test found **lower utility than the heuristic despite lower
investigation cost**; the model was not retuned after that result.
It provides an importable Python package, standard-library SQLite storage,
pytest tests, and a loopback-only Gradio health/demo and investigation interface.
No GPU, paid provider, credentials, model downloads, or LLM API calls are needed.
Successful scripted, heuristic, or learned traces do **not** validate the intuition research
hypothesis. Stage 3 adds a frozen, configuration-level comparative evaluation
within the same simulator, not a test of unknown mechanisms.

## Latest findings and conclusions

The frozen Stage 4 test compared 48 configurations with two paired seeds each.
Mean **utility / investigation cost** was **0.183276 / 8.281250** for scripted,
**0.214807 / 4.500000** for heuristic, and **0.147435 / 2.593750** for learned.
Utility is a synthetic score of useful completions and queue relief; cost is
bounded inquiry-action units, **not money, runtime, or LLM tokens**.

The learned selector used **42% less cost but delivered 31% lower utility than
the heuristic**, and missed available benefit in **15/96 seeded cases**. Its
conservative value threshold sometimes rejected an affordable alternative after
the first intervention failed. **It is not a superior replacement for the
heuristic**: this experiment demonstrates working learning/evaluation
infrastructure and an underexploration failure, not validated learned intuition
or generalization beyond this simulator.

**Proposed, not implemented:** investigate underexploration and utility-estimate/
abstention calibration using development data only, then freeze any revised
policy before evaluating another untouched configuration split. Do not tune on
these inspected test results. See [the detailed Stage 4 results](#first-frozen-test-negative-utility-result-lower-cost)
and [limitations](#diversity-computation-artifacts-and-limitations) for uncertainty,
failure traces, correlated behavior, and the full comparison.

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
| `INTUITION_MODEL_PATH` | `.runtime/stage4-v1/model.json` | Local fitted selector model for the UI; learned selection fails explicitly if missing or incompatible. |

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
budget, leave **Inquiry policy** set to `scripted`, and click **Run investigation**.
The trace shows questions,
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

Stage 4 now implements limited learned action selection. Learned intuition, fuse training, real remote LLM
integration, unrestricted real-world tools, and research validation remain deferred.

## Stage 2: heuristic navigator and intelligence fuse

Select `heuristic` in **Inquiry policy** to see scored alternatives and fuse
decisions. `scripted` remains the default and retains its original investigation
order, predictions, costs, and simulator outcomes. Environment health, offline
echo, non-destructive reset, loopback binding, and privacy settings are unchanged.

The existing CLI module supports both policies:

```bash
python -m intuition_prototype.stage1 --policy scripted --scenario retry_amplification --seed 7 --json
python -m intuition_prototype.stage1 --policy heuristic --scenario retry_amplification --seed 7 --json
python -m intuition_prototype.stage1 --policy heuristic --scenario db_contention --budget 4 --max-steps 6
python -m pytest -q tests/test_navigator.py tests/test_stage1.py tests/test_smoke.py
```

The `stage1` CLI name is retained for compatibility. `--max-steps` defaults to 12;
custom limits (integer 1..12) apply only to `heuristic`. Invalid policies, actions,
capabilities, budget/step values, and persisted record data fail explicitly.
No new dependencies, provider configuration, training, or model downloads are used.

### Inspectable selection formula

The agent-facing API adds only the supported intervention capability list. The
navigator receives no case label, simulator parameters, seed, request internals,
or snapshot contents. It never simulates a candidate just to score it.
The evaluator alone selects the family and owns persistence metadata.

For each supported intervention, the policy produces a question and a testable
assumption. It also shows unsupported/inapplicable candidates as rejected audit
rows. Estimates use observed baseline telemetry and the current hypothesis states
(`untested`, `weakened`, `supported`):

```text
score = (2 * discrimination + 2 * relevance + evidence_gap
         + novelty - 2 * redundancy) / test_action_cost
```

Components are 0..1 heuristic weights, **not probabilities**. Scores are rounded
to six decimals, stored with their components, and validated on readback.
The implementation uses these declared, hand-designed choices:

| Component | Rule |
| --- | --- |
| Service ratio | unique completions / max(1, arrivals) in the baseline window |
| Retry load | min(1, retries / max(1, arrivals)) |
| Discrimination | worker and DB: 0.6; retry: 0.9 when retries exist, otherwise 0.1 |
| Alternative-gap adjustment | after any weakened hypothesis, add 0.15 to untested alternatives' discrimination, capped at 1 |
| Worker relevance | 1 if service ratio >=0.5 and no retries; otherwise 0.3 |
| DB relevance | 1 if service ratio <0.5; otherwise 0.25 |
| Retry relevance | observed retry load |
| Evidence gap / novelty | each 1 for an untested assumption, otherwise 0 |
| Redundancy | 1 for an already tested assumption, otherwise 0 |
| Cost divisor | simulator test cost 2; a further reframe cost 1 is reserved for affordability |

No service/queue anomaly means nonpositive queue growth **and** completions at
least equal arrivals; all relevance scores become zero and tests are rejected.
Retry suppression is rejected without observed new retries. Unsupported and
already-tested interventions are rejected. Candidates also need budget for the
test plus reframe (3 units) and slots for test, reframe, and terminal action
(3 steps). Rejection reasons remain visible with the estimated scores.
Eligible candidates are ordered by descending score, then lexical intervention
key; input enumeration order cannot change a tie.

The chosen question and prediction are recorded **before** execution. Both
policies use the exact same declared prediction criteria described in Stage 1.
Candidate nomination is not evidence: only a new measured result changes an
assumption to supported/weakened.

### Unified loop and fuse behavior

The heuristic state machine uses one bounded `ask/test/reframe/answer/stop` loop:

- **ask (cost 1):** read a simulator telemetry window, not a human question/API.
- **test (cost 2):** apply one listed intervention from the identical warm
  snapshot. Each intervention is tested at most once.
- **reframe (cost 1):** locally replace a working assumption with an evidence-linked
  supported/weakened interpretation. It requires a new result; it is not a
  cosmetic repeat or an extra simulator operation.
- **answer / stop (cost 0):** emit a local evidence-linked conclusion or unresolved
  stop record. These consume a step and are not simulator operations or LLM calls.

After every test, the **intelligence fuse** records triggers, alternatives,
remaining-path assessment, and disposition. Any failed prediction triggers
`contradictory_prediction`. Unchanged unique completions, queue growth, new retries,
and duplicate completions increment a consecutive no-progress counter; the first
is `no_progress`, the second is `repeated_no_progress`. These operational metrics
do not exhaust every notion of information gain: a failed prediction can still
usefully weaken a claim.

The fuse immediately assesses untried alternatives on a contradiction rather than
waiting for repeated failure. If another applicable action is affordable within
the step cap, it records `switch_path`, pays for a meaningful reframe, and the
loop selects again from the scored alternatives. It does not repeatedly spend on
the same state/intervention. If a prediction matches, the answer reports limited
support, never proof or calibrated confidence; model/policy agreement is not truth.

Stop states distinguish `supported_interpretation`, `budget_exhausted`,
`step_limit`, `investigations_exhausted`, and `inconclusive`. A healthy-looking
window is inconclusive, not evidence of a unique cause. Budget can remain unused
when it cannot fund a test plus evidence update. At most 12 actions and budget
20 are allowed; the three available interventions further bound useful work.

### Persistence compatibility

SQLite initialization transactionally adds `policy` and `max_steps` columns to
the original episode table. Existing Stage 1 rows are explicitly marked
`scripted` with the original compatible step ceiling of 12; their records and IDs
are not rewritten. No inference from missing or unknown record types is used.
Typed `DecisionRecord`, `CandidateScore`, `FuseRecord`, and `AnswerRecord` add
auditable decisions while preserving the original evidence and claims.
Validation checks referenced records, costs/steps, candidate scores and choices,
and the terminal answer. Legacy-schema readback and new-record roundtrips are tested.

### Measured in-scope comparison (not held-out evaluation)

Both policies were run at budget 10 on all three families with seeds **7, 17, 29**
(18 episodes total). All reported `supported_interpretation` within this simulator:

| Family | Scripted cost per seed | Heuristic cost per seed | Heuristic first action |
| --- | ---: | ---: | --- |
| Worker capacity | 4 | 4 | double_workers |
| DB contention | 10 | 4 | double_db_capacity |
| Retry amplification | 7 | 4 | disable_retries |

For those seeds respectively, the final trial's unique completions / queue growth
were identical across policies: workers **144/-23, 156/-29, 153/-25**;
DB **46/75, 42/85, 42/86**; retries **21/61, 18/67, 18/68**.
Retry scores for seed 7 are suppression **2.9**, DB **2.6**, workers **1.9**.
The policy does not claim recovered retry-case throughput: the old duplicate
backlog remains, and only new retry generation was disabled.

A separate label-free test double checks fuse recovery: unchanged evidence after
the first DB experiment triggers contradiction/no-progress and a reframe to the
untried worker explanation. If workers then improve service, the policy answers;
if they also return unchanged evidence, the fuse records repeated-no-progress and
exhaustion. This synthetic test checks control flow, not simulator or research validity.

The heuristics were designed with this simulator in view. These costs do **not**
establish general superiority, causal identification, learned intelligence, or
research validation. Stage 2 itself has no calibrated information-gain model,
learned selector, independent CPU/DB diagnosis, or formal held-out benchmark;
Stages 3 and 4 add evaluation and limited learning separately. The original
simulator limitations remain.

## Stage 3: frozen configuration-holdout comparison

The benchmark compares the **unchanged Stage 2 policies**. Its predeclared primary
endpoint is paired held-out utility difference (heuristic minus scripted); cost,
regret, abstention and false-support metrics are secondary. Improvement is not an
acceptance requirement. There is no LLM baseline because no provider/API is
configured. No policy weights, scoring priorities, prediction thresholds, or fuse
rules may be tuned after inspecting held-out outcomes.

### Preparation and protocol

Inside the activated WSL environment:

```bash
python -m intuition_prototype.benchmark prepare --output .runtime/stage3-replay --rerun-of published-stage3-v2 --invalidation-reason "Historical regression after shared-loop refactor; not fresh evidence"
python -m intuition_prototype.benchmark run --output .runtime/stage3-replay
python -m intuition_prototype.benchmark report --output .runtime/stage3-replay
python -m pytest -q tests/test_benchmark.py tests/test_navigator.py tests/test_stage1.py tests/test_smoke.py
```

`prepare` performs **no experiments**. It writes separate development/held-out
configuration manifests, the scoring/construction protocol, source hashes,
the Git base plus uncommitted-source provenance, and frozen source copies.
Source identity normalizes CRLF to LF for portability across Git checkouts; exact
raw hashes are also retained and verified for archived source files.
Only then may `run` execute. All commands emit JSON. `report` verifies the stored
artifacts without executing policies. Preparation requires a new output directory;
execution is one-shot and will not overwrite or silently resume an earlier attempt.

Default construction uses generator seed **20260908**, eight sampling regimes,
**16 development configurations** (2/regime) and **48 held-out configurations**
(6/regime). Each configuration is paired across policies with arrival seeds
**101 and 202** at budget **10** and step cap **12**. Thus the full benchmark has
128 paired seeded cases / 256 agent episodes, but only **64 configuration units**.
The curated Stage 1 scenarios remain sanity/development tests, not held-out cases.

Regimes cover worker pressure, DB pressure, retry pressure, mixed pressure,
healthy headroom, low signal, retry-enabled headroom, and near balance.
They are construction tags, **not sole true-cause labels**. Exact ranges are
declared in `benchmark_protocol.py` and frozen in `protocol.json`. Within each
regime, held-out initial-worker choices are disjoint from development choices;
load, DB-capacity, lock-penalty and timeout ranges also shift. This is a parameter
support/configuration holdout, **not new-mechanism generalization**.

`SimulatorConfig` adds evaluator-only initial conditions: workers 2..16, DB
capacity 1..64, lock penalty 0..0.25, retry timeout 1..1000, retry enablement, and
integer arrival bounds 0..8. Doubling interventions operate relative to those
initial resource values. The FIFO service, lock formula, work-per-attempt,
timeouts, retry cap and horizon are unchanged. The original three families retain
exactly their original observations and policy outcomes.

Both policies receive the same restricted observation/action/snapshot interface.
They never see configuration labels/parameters, manifests, external scores, or
counterfactual measurements. Every trial restores the identical warm-state
snapshot and arrival RNG; evaluator checks compare recorded policy observations
against the reference windows. Warm-up and measurement windows remain 30 ticks.

### Independent reference, not self-reported correctness

The evaluator measures the baseline and **all three permitted single
interventions** from the same state, separately from each policy. Let `C` be unique
completions, `G` queue growth and `A` baseline arrivals. An intervention qualifies
as independently beneficial only if it does not reduce `C` or increase `G`, and:

```text
completion gain >= max(3, ceil(0.10 * max(1, baseline C)))
OR
queue-growth reduction >= max(10, ceil(0.15 * max(1, abs(baseline G), A)))
```

These absolute/relative criteria are deliberately independent of policy
predictions. Raw utility is `(completion gain + 0.25 * queue-growth reduction) /
max(1, A)`. Qualifying interventions retain raw utility; otherwise utility is
`min(0, raw utility)` so tiny unsupported gains are not rewarded and harms remain
negative. Abstention has utility zero. The reference optimum is the maximum of
zero and all permitted single-intervention utilities; regret is that optimum
minus the policy recommendation's utility.

A recommendation is the last actual tested intervention **only if** the policy
ends with `supported_interpretation`; otherwise the policy abstains. This label
selects what claim to score, **not whether it is correct**. Its evidence links are
checked, and independent benefit/utility determines helpfulness. Reports include:

- recommended utility and regret vs the best permitted single intervention;
- agent cost and utility / (1 + agent cost), a declared index, not monetary cost;
- helpful vs supported counts, supported-but-unhelpful rate among supports;
- abstentions, missed available benefits and missed benefits despite testing them;
- false-positive rate on oracle no-benefit cases, stop reasons and budget adherence;
- paired configuration-mean ties/losses/wins, regime summaries and failure traces.

No single-intervention benefit in a finite window is **not** proof that a multi-step
cure is impossible. Retry suppression still leaves old duplicate work outstanding;
reduced queue injection does not establish recovered throughput or a unique cause.
Oracle work (4 measurement windows plus warm-up per paired seeded case) is reported
separately from agent action costs.

### Statistical unit and reproducibility

Each configuration's two seed results are averaged **before** paired comparisons.
The report uses 2,000 deterministic paired bootstrap resamples of configuration
units **within sampling regimes**, seed 917, with percentile 95% intervals.
These are descriptive intervals conditional on the declared generator, not
population/generalization claims. Conditional false-positive/support rates are
reported with explicit denominators; seed repeats are not independent samples.

Artifacts under the ignored output directory:

- `manifest.json`, `development.json`, `heldout.json`, `protocol.json`;
- `freeze.json` and `sources/`: exact uncommitted policy/engine/evaluator provenance;
- `raw.jsonl`: paired results, full typed traces and evaluator-only counterfactuals;
- `report.json`, `report.md`: summaries, intervals, failure groups and sample traces;
- `run-start.json`, `run-complete.json`: attempt lifecycle and output integrity hashes.

Missing completion means an invalid/incomplete attempt. Changed sources,
manifests or archived code fail verification. A corrected implementation must use
a **new directory** with `prepare --rerun-of <prior-directory>
--invalidation-reason "<explicit reason>"`; it is labeled a rerun, never a pristine
held-out result. Frozen Stage 2 policy hashes must still match.

Development-only preflight found a Markdown ordering/spacing defect on JSON
roundtrip. Those preflight artifacts were invalidated and the renderer corrected
**before any generated held-out policy outcomes were run**. The protocol records
this history; policies and reference scoring criteria were not changed.

The ordinary UI remains the small simulator/policy UI; no full benchmark runs at
startup or from a UI callback. Inspect the report artifacts separately. Development
traces can seed future learned-policy work, but inspected held-out configurations
must not be treated as fresh test data after subsequent tuning.

### First held-out run: measured findings

The original declared full run completed once with unchanged frozen policy bytes
and no policy tuning. Its local artifacts remain in `.runtime/stage3-v1/`.
A subsequent publication-portability replay is explicitly distinguished below.

- Manifest SHA256:
  `ef396b6531608c1af65aff9ccb6770eeba1c0926056df20ad51e6af0665966bd`
- Combined frozen policy SHA256:
  `82866677c92969108d7069b12e1303b09ac02886398397127a2fcbbbb47b3101`
- Preflight: **122 tests passed**, including all 98 pre-existing tests.
- Full run: **64 configurations / 128 paired seeded cases / 256 policy episodes**.
- Held-out analysis: **48 configurations / 96 paired seeded cases / 192 episodes**.
- Evaluator-only work: **512 measurement windows / 19,200 advance ticks including
  warm-ups**, not charged to the agents.

Held-out means:

| Policy | Independent utility | Regret | Agent cost | Utility/(1+cost) | Helpful supports | Abstentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scripted | 0.146397 | 0.037748 | 8.250 | 0.021720 | 48/48 | 48/96 |
| Heuristic | 0.184145 | 0.000000 | 5.875 | 0.033323 | 48/48 | 48/96 |

Paired configuration-level differences (heuristic minus scripted), with the
predeclared descriptive 95% intervals:

- Utility: **+0.037748**, interval **[+0.018787, +0.056623]**.
- Regret: **-0.037748**, interval **[-0.056623, -0.018787]**.
- Cost: **-2.375**, interval **[-2.500, -2.250]**.
- Cost-adjusted utility: **+0.011602**, interval **[+0.009112, +0.013958]**.

This was not a uniform win. **44/48 configurations tied on utility**, 4 favored
the heuristic, and none favored the scripted policy on utility. In contrast,
**6/48 configurations favored the scripted policy on cost and cost-adjusted
utility**: every held-out worker-pressure configuration took heuristic cost 7
versus scripted cost 4. Utility gains were concentrated in four retry-pressure
configurations, not spread across the whole benchmark.

Actual cost-loss trace: configuration `ec4a87945839fa220fef`, seed 101.
The heuristic interpreted low completion/arrival ratio as a reason to test DB
capacity first (score 2.6 versus workers 1.9). Baseline and DB trial both completed
45 requests with queue growth 133. The fuse recorded contradiction/no-progress,
reframed, and tried workers, which completed 90 with growth 88. It recovered the
same independently useful recommendation as the scripted policy, but spent 3
extra units. This is a real simulator trace, not a scripted test-double response.

Actual scripted weakness: configuration `4e7ae414f529bae539a5`, seed 101. Doubling
workers met its prediction and gave utility **0.043539**, so it stopped. The
permitted reference optimum was **0.502809**, giving regret **0.459270**. Across
held-out seeds, the scripted policy made 8 helpful-but-suboptimal recommendations;
the heuristic made none under this particular reference.

Both policies had zero budget violations, zero supported-but-unhelpful reports
out of 48 supports, and zero no-benefit false positives out of 48 no-benefit cases.
Both stopped with 48 supports and 48 exhausted investigations; neither missed a
reference benefit through abstention. These zeros do **not** establish robust
correctness: the suite did not expose those failure modes. A unit test explicitly
demonstrates that policy support can fail the independent scorer.

Important weaknesses of this benchmark:

- Half of held-out seeded cases had no qualifying single-action benefit, and
  only **20 distinct paired telemetry/counterfactual signatures** occurred among
  the 48 parameter configurations. Quantized dynamics reduce behavioral diversity.
- No-benefit cases still consumed heuristic cost 7 and scripted cost 10. The
  policies can spend effort investigating tiny finite-window fluctuations.
- Regimes and metric weights were hand-designed. Queue-relief weighting can favor
  retry suppression; it is a declared utility choice, not universal correctness.
- Construction and scoring were authored with knowledge of both policies. Cases
  were held out from execution/tuning, not sourced from an independent benchmark.
- Conditional bootstrap intervals describe this generator, not unknown
  mechanisms, production systems, learned intuition, or general superiority.

Data readiness: development contains 16 configurations, 32 paired seed cases,
64 policy traces and all corresponding single-action reference outcomes. It is
usable for a small future learned-policy prototype, not enough to substantiate
learning claims. Held-out configurations have now been inspected; retain their
evaluation provenance and construct a new untouched configuration split for any
later tuned/learned policy. Stage 4 deliberately excludes **all 64** Stage 3
configurations, including development, from its new training/validation/test sets.

### Publication portability correction and labeled replay

Pre-push verification found that v1's raw source fingerprints depended on the
local CRLF files, whereas the repository checks out LF files. This would have
blocked benchmark preparation from a fresh clone. Protocol `stage3-v2-portable`
normalizes only CRLF-to-LF for source identity, while separately preserving and
validating raw archive hashes. The original v1 reports remain readable and intact;
legacy execution requires a new declared attempt rather than silently changing it.

No policy source, weight, threshold, simulator formula, generator or external
scoring rule changed. A full replay under `.runtime/stage3-v2-portable/` records
`attempt_classification: rerun`, its v1 predecessor and the portability correction
as its invalidation reason. All **128 paired rows**, including complete policy
traces, oracle effects and scores, match the original exactly after excluding the
two provenance hashes. Split statistics and intervals also match exactly.
This replay is **not fresh held-out evidence**.

- Portable manifest SHA256:
  `1c16d7c31f7c89e64469ec3220a82639e09772fac8fce2f4a5ec1cd6c9239fda`
- LF-normalized policy SHA256:
  `1c0a54add89307e8db8de00b42cd706c87e97848b2d681276e0ef4df433bed7f`
- Publication validation: **123 tests passed**, including a line-ending portability
  regression test. The differing hash is a representation change, not policy tuning.

### Stage 4 baseline compatibility

`stage3-v3-baseline-refactor` records new source fingerprints for sharing the
unchanged navigation loop and extending typed records. It does not change the
heuristic weights, predictions, simulator, or external utility rule. Both v1 and
v2 report archives remain readable and untouched. An explicit historical regression
replayed all **128 paired cases / 256 complete serialized episodes** from v2:
both baseline policies' traces, assessments, and initial observations matched
exactly. This is regression evidence, never Stage 4 training or fresh testing.
New Stage 3 runs are labeled historical regression; reproducing the original
source checkpoint itself requires the published `abdc78b` checkout/archived sources.

## Stage 4: offline learned candidate selection

### What is actually learned

Three ridge regressions predict independently defined **single-intervention
utility**, one for each permitted intervention. They are actually fitted from
development simulator observations and intervention outcomes, not hand-labeled
heuristic rules. Fitting uses only the Python standard library; no downloaded
model, GPU, provider, credentials, or API is involved.

The actions/questions remain the predefined worker doubling, DB doubling, and
new-retry suppression investigations. This is **selection among known candidates**,
not question generation, mechanism discovery, general reasoning, or learned intuition.
Scripted remains the default. Heuristic and learned policies share the same
bounded ask/test/reframe/answer/stop loop, pre-action hypothesis predictions,
same-state experiments, evidence-based reframing, and intelligence fuse.

For each action the fit minimizes mean squared target error plus
`alpha * sum(coefficients**2)`, with an unpenalized intercept. Training-only means
and population standard deviations standardize ten telemetry features:

1. Arrivals per tick.
2. Unique completions / `max(1, arrivals)`, clipped to 0..4.
3. Queue growth / arrival denominator, clipped to -4..8.
4. Initial queue / arrival denominator, clipped to 0..20.
5. New retries / arrival denominator, clipped to 0..4.
6. Duplicate completions / `max(1, unique + duplicate completions)`.
7. Mean completed-request latency / window ticks, clipped to 0..8; zero if absent.
8. Explicit missing-latency indicator.
9. Dropped attempts / arrival denominator, clipped to 0..4.
10. `max(0, 1 - completion_fraction)`.

A constant feature has scale 1. No scenario ID, regime, seed, hidden parameter,
snapshot contents, or evaluator counterfactual is an inference feature.
Training targets use exactly the independent Stage 3 utility rule above, not a
policy's support label. Estimates are **not calibrated probabilities** and can be
wrong or negative.

Eligible candidates are ranked by `predicted_utility / test_cost`, rounded to six
decimals, with lexical action-key ties. The predicted utility must strictly exceed
the development-selected abstention threshold. Unsupported/already-tested actions
are rejected; admission reserves test + reframe cost and test/reframe/terminal
steps. The fuse considers all untried alternatives after contradiction/no-progress,
including explicit low-value rejection reasons. Evidence updates eligibility and
interpretations, **not fitted weights or other actions' numeric estimates**.
Low estimated value produces an inconclusive abstention, not a proof of no cause.

The trace records model SHA256, predictions, threshold, standardized features,
coefficients, intercept, individual contributions, chosen/rejected alternatives,
actual observations, and fuse decisions. SQLite roundtrip validates the learned
score schema while preserving Stage 1/2 episodes and original demo entries.

### Model use and reproducibility

The model is a versioned JSON artifact with normalization, coefficients, selected
hyperparameters, training/validation IDs, source provenance, and a content digest.
Missing, corrupt, incompatible, or changed-preprocessing models fail explicitly;
there is no silent fallback. A health-panel file-present flag only reports file
existence; actual model validation occurs when learned selection is requested.

Inside the activated WSL environment, prepare and train before selecting learned:

```bash
python -m intuition_prototype.stage4 prepare --directory .runtime/stage4-v1
python -m intuition_prototype.stage4 train --directory .runtime/stage4-v1
python -m intuition_prototype.stage1 --policy learned --model .runtime/stage4-v1/model.json --scenario retry_amplification --seed 7 --json
python -m intuition_prototype.stage4 freeze --directory .runtime/stage4-v1
python -m intuition_prototype.stage4 evaluate --directory .runtime/stage4-v1
python -m intuition_prototype.stage4 report --directory .runtime/stage4-v1
```

Use a **new directory** for reproduction; existing artifacts are not overwritten.
Re-executing the published split is a reproduction of inspected data, not another
untouched test. No training or benchmark runs during UI startup. Set **Inquiry
policy** to `learned` after training; the UI loads `INTUITION_MODEL_PATH` for each
run. CLI `--model` is explicit and independent of that environment variable.

Development-only model tests exercise actual fitting, deterministic replay,
feature/provenance boundaries, missing/corrupt models, bounds, alternate-path fuse
recovery, contribution validation, SQLite, and CLI. The test suite never executes
the new untouched test partition.

### Declared split, selection, and freeze

Protocol `stage4-v1`, generator seed **20260909**, declares all configurations
before fitting: **48 train, 16 validation, 48 test**, with seeds **101 and 202**
paired across all three policies. There are eight sampling regimes, respectively
6/2/6 configurations per regime. Worker supports are globally disjoint:
train `{4,7,10,13}`, validation `{5,8,11,14}`, test `{2,3,6,9,12,15,16}`.
Exact parameter fingerprints exclude all 64 original Stage 3 configurations.

The declared generator broadens load/capacity variation and includes zero/low
arrivals, healthy headroom, retries on/off, mixed bottlenecks and wide arrival
ranges including 0..8. Initial DB capacity spans declared regime-specific bounds
within 1..64, lock penalties within 0..0.25, and retry timeouts 1..1000.
Regime tags are sampling metadata, not causes. **No simulator formula or reference
utility was changed to help the learned policy.** The public generator reproduces
training data without any private runtime files.

Four ridge strengths `{0.01,0.1,1,10}` are fitted on the 96 training seed windows.
The 16 combinations with thresholds `{0,0.02,0.05,0.1}` are compared on validation
only. The declared order maximizes configuration-mean cost-adjusted utility,
then utility, then minimizes cost, then prefers stronger ridge and a higher
threshold. There is no refit on validation. Selected **alpha 0.1, threshold 0.1**:
validation mean utility 0.153526, cost 2.5, cost-adjusted utility 0.030705.
These are selection data, not test findings.

Both an actual second full development run and unit-level reproduction produced
byte-identical preparation, model, training, development-output, and selection-grid
files. Timestamps and timings are sealed separately, outside model identity.
Before the official test, the final model, preprocessing, every project executable
source, scorer, protocol, partitions, and training outputs were frozen.
Source identity normalizes CRLF to LF; exact raw archive bytes and inventory are
also verified. Runtime data/credentials are excluded from live-source inventory.
An exclusive start marker prevents silent overwrite/resume of failed attempts.

The primary endpoint is configuration-mean learned-minus-heuristic test utility.
Seeds are averaged within configuration; 2,000 deterministic, paired bootstrap
resamples within regime provide descriptive 95% intervals. The secondary
learned-minus-scripted comparison uses the same procedure. All policies receive
the same agent-safe interface, snapshots, 30-tick windows, budget 10 and step cap 12.
The reference remains the existing evaluator-only, best permitted **single-action**
counterfactual, not a policy's support label or a unique-cause diagnosis.

### First frozen test: negative utility result, lower cost

One official attempt completed on **48 configurations / 96 paired seed cases /
288 agent episodes**. No test outcome was inspected before model selection and
freeze; there was no test-triggered retuning, source change, or rerun.

| Policy | Mean utility | Mean regret | Mean cost | Cost-adjusted utility | Helpful supports | Abstentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scripted | 0.183276 | 0.031531 | 8.281250 | 0.027568 | 52 | 44 |
| Heuristic | 0.214807 | 0.000000 | 4.500000 | 0.038479 | 52 | 44 |
| Learned | 0.147435 | 0.067372 | 2.593750 | 0.029487 | 37 | 59 |

- Learned minus heuristic utility: **-0.067372**, interval
  **[-0.101448, -0.036232]**; 0 configuration wins, 40 ties, 8 losses.
- Learned minus heuristic cost: **-1.906250**, interval
  **[-2.437500, -1.375000]**. Cost-adjusted utility also fell by 0.008992,
  interval [-0.014734, -0.004243].
- Learned minus scripted utility: **-0.035841**, interval
  **[-0.079313, 0.005014]**; 5 wins, 35 ties, 8 losses.
  Cost fell by 5.687500; cost-adjusted utility rose by 0.001919 with interval
  [-0.007178, 0.009782]. This does not establish a reliable advantage.

The learned selector abstained despite available benefit in **15/96** seed cases:
2 worker-pressure, 4 mixed-pressure, and 9 near-balance cases. That is **15/59
(25.4%) of its abstentions**, or **15/52 (28.8%) of independently beneficial
cases**. Both baselines missed none through abstention. All three policies had
zero supported-but-unhelpful reports, zero no-benefit false positives among 44
no-benefit cases, and zero budget/step violations. These zeros are not proof of
robust correctness or calibrated confidence.

Concrete failure: configuration `66d2b35a87a368f8c7cc`, seed 101. Baseline:
30 completions, 198 arrivals, queue growth 168. The model estimated DB utility
**0.158689**, workers **0.094610**, and retry suppression **0.007376**. It tested DB;
the outcome was unchanged, so the fuse recorded contradiction/no-progress.
Workers were affordable but rejected below threshold 0.1. The result was an
inconclusive abstention at cost 4, although the evaluator measured worker utility
**0.189394**. Scripted found that benefit at cost 4; heuristic recovered it at
cost 7. A separately tagged mixed configuration with identical telemetry received
the same predictions, demonstrating that evaluator regime labels did not drive selection.

The learned policy had **zero test-set fuse switches/recoveries**: its value gate
often rejected the remaining alternative. The heuristic had 37 switches and 14
externally helpful episodes after a switch. Development-only synthetic tests
verify that the shared learned loop can genuinely reframe and recover when an
eligible alternative remains; that control-flow test is not evidence of observed
test-set recovery. We leave this weakness intact rather than adjusting the threshold
or inventing successful demonstrations after inspecting test results.

### Diversity, computation, artifacts, and limitations

Disjoint parameters are not disjoint behavior. Exact paired telemetry signatures
number **32/48 train**, **13/16 validation**, and **34/48 test** configurations.
Fourteen test configurations duplicate another test signature; **19 test
configurations share an exact paired signature with development**. Coarse
benefit/sign patterns number 8/5/7 across train/validation/test; **94/96 test
windows** have a coarse pattern seen in development. Only one configuration has
both repeats novel under that coarse definition. Novelty summaries are descriptive
after-test diagnostics, not independently selected confirmatory subsets.

Broader parameters therefore do not establish new-mechanism or robust behavioral
generalization. Configuration-level intervals remain conditional on this
hand-designed generator and contain correlated/duplicate behavior. The model has
only ten features, 48 fitting configurations and 16 selection configurations.
It cannot discover candidates, reason about unmodeled mechanisms, or update
cross-action utility estimates after a contradiction. Completed-request latency
is survivor-biased; retry suppression does not clear old backlog or establish
throughput recovery. No LLM baseline exists because none is configured.

Observed primary training computation: 4 ridge fits, 512 validation-policy
episodes, 384 selected-model development-policy episodes; evaluator-only oracle
cost 512 windows / 19,200 simulated ticks including warmup. Measured wall time was
7.35 seconds total, including 0.08 seconds fitting and 0.72 seconds oracle work.
The official test used 384 oracle windows / 14,400 ticks separately from agent
cost; observed wall times were 0.69 seconds oracle, 1.36 seconds policy execution,
and 8.13 seconds through result generation/validation. Timings are machine-specific,
not a general performance benchmark; final read-only reload checks and training
reproduction are additional computation.

Persistent, gitignored artifacts are under `.runtime/stage4-v1/`:
`preparation.json`, separate split files, `protocol.json`, `development.json`,
`validation-grid.json`, `model.json`, training/runtime manifests, `freeze.json`,
exact `archive/` copies, `test-raw.jsonl`, `report.json`, `report.md`, and the sealed
evaluation manifest. The complete primary artifact set is about 44.7 MiB.
The independent development reproduction is retained in
`.runtime/stage4-training-reproduction/`; it has **no test execution**.

- Preparation digest:
  `d9d0858cda7bbffc265928ea740aa7488097fda71ead44e21e3e0e46949706d9`
- Selected model digest:
  `ea8ceb42894b03a1500b594855c4d45a9da5cba21f12d1cd5ca7993ac1e375fe`
- Final freeze digest:
  `3261eb9641b7334dee54924e526a75a773ae7f99388585bf9e937134b5d075fa`
- Raw test file SHA256:
  `bf059aa9e082eb9d3afca0471fc2875f9ec4895639b893e8ef683b5117c9d79b`

Verification: **163 pytest tests passed**; actual train/evaluate/report CLI JSON,
persisted report/Markdown reload, source/archive/model integrity, all three live
Windows-browser policy callbacks, typed SQLite readback, reset, health, and offline
echo were verified. The UI remains localhost-only with analytics disabled and no
external theme stylesheets. The old inactive browser page needed replacement;
no application workaround was introduced. Startup can be slow on the Windows-mounted
virtual environment. The live learned budget-3 callback stopped at cost 1 with
`budget_exhausted`; budget 10 completed the retry demonstration at cost 4.

These inspected Stage 4 test cases are now evaluation history, not a new future
test set. Future model improvements must use development data and another declared
untouched evaluation split. The current result demonstrates working learning and
evaluation infrastructure, **not a successful intuition research hypothesis**.

## Architecture boundary

- `config.py`: explicit validated demo configuration.
- `llm.py`: replaceable `LLMAdapter` protocol and deterministic `DemoAdapter`.
- `records.py`: typed evidence, claims, predictions, unified ask/test/reframe actions,
  results, and episodes.
- `simulator.py`: bounded queue dynamics and opaque snapshot/reset API.
- `inquiry.py`: explicitly scripted, budgeted investigation policy.
- `predictions.py`: shared declared prediction criteria for all three policies.
- `navigator.py`: heuristic scoring and shared bounded selection/intelligence-fuse loop.
- `learned.py`: telemetry preprocessing, ridge fitting, strict model loading and learned scores.
- `stage4_protocol.py`: new isolated configuration partitions and frozen experiment provenance.
- `stage4.py`: development training/selection and three-policy test evaluation/reporting.
- `benchmark_protocol.py`: declared parameter splits, frozen source/config fingerprints.
- `benchmark.py`: independent counterfactual reference, paired evaluation and reports.
- `storage.py`: original demo storage plus transactional typed episode persistence.
- `stage1.py`: evaluator orchestration, persisted trace rendering, and CLI.
- `app.py`: environment health, offline echo, and local policy/trace interface.
- `tests/test_smoke.py`: imports/configuration, SQLite initialize/write/read,
  health reporting, and UI construction.
- `tests/test_stage1.py`: three families, acceptance trace, deterministic replay,
  action/budget bounds, evidence separation, persistence, and reset semantics.
- `tests/test_navigator.py`: evidence-sensitive selection, tie stability, fuse recovery,
  bounds, legacy-schema migration, score validation, and preserved baseline behavior.
- `tests/test_benchmark.py`: split/config bounds, external scoring, frozen provenance,
  development-only integration, integrity and report roundtrip.
- `tests/test_learned.py`, `tests/test_stage4.py`: fitted-model behavior, data isolation,
  model/experiment integrity, development-only lifecycle integration and persistence.

All three policies are bounded research infrastructure, not the proposed open-ended
learned investigative controller. Candidate discovery and fuse training remain deferred.

A future remote adapter must explicitly implement the protocol, extend mode
validation and the factory, and define credentials, timeouts, error handling,
and user-approved cost controls. The current scaffold deliberately selects no
provider and rejects `remote` rather than silently falling back to demo.
