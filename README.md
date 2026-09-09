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
A separate cumulative three-intervention benchmark now tests the depth hypothesis
with explicitly named sequential adapters; it leaves the original policies and
frozen Stage 4 model unchanged.
A further isolated **dog-at-door history benchmark** tests learned next-question
preferences after each question/answer. It is a finite synthetic pattern-learning
toy, not sequential repair, a real-dog model, diagnosis, treatment guidance, or
validated intuition.

The current extension targets **context-first inquiry**, rather than object-only
classification: clarify what the asker needs when it is unknown, narrow relevant
interpretation families using evidence, then ask finer discriminating questions.
The two illustrative interfaces are a dog-at-door question and a request to use a
linked list. Asker context and desired means are not evidence about the underlying
world or requirements. Both CLI demonstrations and the new frozen comparison are
verified. Historical results below remain separate evidence.

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

The [sequential extension](#sequential-benchmark-three-meaningful-interventions)
also does not support a learned advantage: on certified three-intervention test
cases, frozen learned transfer restored service in **0/10**, versus **8/10** for
the heuristic adapter and **0/10** for scripted. These are remediation outcomes
in a small, structurally selected sample, not general reasoning accuracy or durable
recovery. Retrospective inspection of the frozen reference also finds a single
fixed repair order meeting the window goal in **10/10**: this benchmark does not
establish that adaptive reasoning is needed.

The first frozen dog-history test also does not show a history-model advantage.
The history-aware learner reached supported answers in **12/20** cases, versus
**13/20** for its memoryless learned ablation, with the same **6/20** supported
and latent-label-correct outcomes. It did clearly outperform the deliberately
nonadaptive fixed order on supported resolution (12/20 versus 2/20), but this
small known finite task does not establish general question learning.

The newer [context-first comparison](#context-first-inquiry-asker-purpose-abstractions-then-details)
implements asker clarification and explicit abstraction narrowing in both domains.
History-aware and memoryless learned policies **tie at 180/228 scoped resolutions
and 2.509 questions on average**. The learner's actual history-dependent branches
work, but this experiment does **not** show a performance advantage from the extra
history. Twelve resolved exception cases have no scorable modeled truth; this is
not a claim of perfect accuracy or reliable out-of-model detection.

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

## Sequential benchmark: three meaningful interventions

The expectation that learned selection might fare better when a solution is more
than one investigation away is a **hypothesis**, not an explanation that removes
the Stage 4 failure. This separate benchmark tests operational remediation with
cumulative effects. It does **not** measure how many reasoning steps are required
to guess or justify a diagnosis.

### Task and rigorous depth criterion

The original simulator formulas and the Stage 3/4 policies, learned model, and
saved evidence are unchanged. A new wrapper lets the same worker/DB/retry
interventions **remain applied**. The first observation follows the usual 30-tick
warmup and a 30-tick measurement. Each subsequent intervention advances another
30 ticks and returns the next window's telemetry. Queue contents, unfinished work,
resource settings, and arrival RNG advance rather than resetting between tests.

External success is a public finite-window service objective: positive arrivals,
unique completions **at least arrivals**, non-growing queue, and no dropped
attempts. There is no action-count requirement in this predicate. It means the
service meets this window's load; it does not mean all old backlog is cleared,
the cause is identified, or permanent stability is proven.
The task deadline is three 30-tick action windows after the initial measurement.
Success is checked after every move, not only at the deadline.

**SLO and persistence precision:** the pre-run protocol and implementation defined
success using aggregate metrics in **one 30-tick window**, not a sustained
multi-window condition or an every-tick SLO. No confirmation windows, latency SLO,
or subsequent relapse test were specified or measured. The saved
`service_restored`/`success` labels and "restored service" wording below mean only
**this window's goal was met**. A stochastic completion/arrival fluctuation can
meet that goal; paired seeds do not establish longitudinal persistence. This
limitation is explicit here following feedback received after the run, without
retrospectively changing its frozen goal. A durable-recovery experiment would need
a separately preregistered persistence criterion and new evaluation evidence.

An evaluator-only exhaustive reference enumerates **all 52 legal prefixes**
of depth 0..3: each of the three interventions at most once, plus a repeatable
30-tick observe/wait move. A deep case is eligible only when **both seeds** have
minimum success depth exactly 3 and **every successful path uses three distinct
interventions**, never a wait substitution. Thus a one/two-repair solution, passive
recovery, or merely consuming three measurement windows cannot satisfy eligibility.
This is a lower bound within the declared action durations and horizon, not a
mathematical bound on cognition, arbitrary tools, or unbounded waiting.
In particular, failure of all waits within the remaining 90-tick horizon does
**not** prove that waiting longer or forever could never help. Repairs and waits
each consume one of the three available 30-tick moves; shorter-duration actions,
repeated repairs, extra moves, and other tools are outside the certificate.

Waiting is legal and included in the stronger reference, but the inherited
selectors do not estimate or select waiting. The three-intervention bound limits
all policies equally; it is not a minimum-step gate. A shallow control can and
does terminate after its first successful intervention. Failed or unhelpful
interventions remain applied and consume their full cost.

### Separately named adapters, not silently upgraded planners

The original default scenario UI remains the same-state inquiry interface.
Use the separate sequential CLI/report below rather than interpreting its UI
trace as a cumulative experiment.

- `scripted-sequential-v1` is the **non-adaptive all-repairs reference**: workers,
  retry suppression, then DB capacity in fixed order, with early goal/budget stops.
  Its next repair does not depend on intermediate telemetry.
- `heuristic-sequential-v1` uses the unchanged heuristic formula on the latest
  telemetry and remaining legal interventions.
- `learned-transfer-sequential-v1` uses the frozen Stage 4 coefficients,
  training-only normalization, and strict utility threshold 0.1 on the latest
  telemetry. It is **one-step-target transfer**, not a trained sequential planner.

All share one explicit sequential loop and the same initial evidence, service
goal, depth and budget. Initial measurement costs 1; each intervention costs 2
plus 1 evidence-update unit, so budget 10 permits three actual interventions.
Reference waits cost 1. These are synthetic action units, not money or CPU time.
The reference utility is `min(1, completions/arrivals) - max(0, growth)/arrivals -
drops/arrivals`, using denominator `max(1, arrivals)`. It measures final service
performance, **not the Stage 4 single-intervention utility**; scores across the
two benchmarks are not directly comparable.

Existing hypothesis prediction criteria remain separately recorded before each
move. A prediction match/local improvement does not imply the service objective
is met. Mismatch, no utility progress, or local-only benefit prompts assessment
of unused alternatives. The fuse switches only when an eligible action remains;
it can still stop prematurely, including through the unchanged learned threshold.
No policy receives reference paths, configuration labels, or hidden parameters.
Consecutive observation windows are **not same-state causal contrasts**.

### Declared construction and matched shallow controls

`sequential-benchmark-v2` declares 80 development and 160 untouched-test parent
configurations, each with seeds **307 and 409**, before test outcomes. Worker
supports are disjoint: development `{4,6,8,10}`, test `{3,5,7,9,11,12}`. Public
configuration fingerprints exclude every original Stage 3 and Stage 4 configuration.
Declared arrival ranges, DB/lock ranges and timeouts use existing mechanisms only.

An initial v1 declaration found three eligible deep cases and zero shallow cases
in development, before any policy or test-reference execution. The retained v2
declaration adds a paired control to **every** parent: DB capacity is already
doubled and retries are disabled, with remaining parameters and seeds unchanged.
All original parent configurations are preserved. A control enters the shallow
cohort only if its parent qualifies deep and both control seeds have minimum
success depth 1. Already-restored and unsolved controls are explicitly excluded.

All structural reference outcomes and exclusions are retained before any policy
run. Every eligible case is compared, not only favorable outcomes. Both seeds,
all counterfactual paths, and the paired control belong to one parent statistical
unit. The same budget, dynamics and objective apply to both cohorts, but initial
partial remediation changes telemetry and bottlenecks: **depth is not causally
isolated**. Sparse structural eligibility also limits representativeness.
There is no sequential-target fitting, model selection, or test-set tuning.

Inside the activated WSL environment:

```bash
python -m intuition_prototype.sequential_benchmark prepare --directory .runtime/sequential-v2 --model .runtime/stage4-v1/model.json
python -m intuition_prototype.sequential_benchmark run --directory .runtime/sequential-v2 --split development
python -m intuition_prototype.sequential_benchmark run --directory .runtime/sequential-v2 --split test
python -m intuition_prototype.sequential_benchmark report --directory .runtime/sequential-v2 --split test
python -m pytest -q tests/test_sequential.py tests/test_sequential_benchmark.py
```

Preparation requires a new directory and copies the exact frozen transfer model.
Recreate that model from the Stage 4 source checkpoint `88a7de2` and its documented
training commands if necessary; changing its training-source inventory produces
a different model provenance and is not the declared frozen-transfer experiment.
Executable sources, model, protocol, partitions and raw archive bytes are checked
before and after execution. Each split is one-shot; a failed or inspected test
must not be silently overwritten and presented as fresh evidence.

### Verified results: deeper tasks did not rescue frozen learned selection

The official frozen test examined **160 parent configurations plus 160 paired
controls**, each at two seeds. All **640 configuration-seed references** were
exhausted and their structural eligibility persisted **before any policy was run**.
Only **5/160 parents (3.125%)** satisfied the strict three-distinct-intervention
criterion at both seeds. Of those five, two had qualifying one-step controls;
three controls were already restored initially and were excluded from the
one-step cohort. The 155 other parents and all control exclusions remain in the
reference/eligibility artifacts. No case was selected by policy success.

The 155 excluded parents comprise 117 with neither seed solving within the
declared horizon, 21 already meeting the goal at both seeds, six with depth 3 at
both seeds but a successful wait-substitution path, and 11 other mixed-depth or
shorter-path pairs. All 155 corresponding controls are excluded regardless of
their own performance; a further three controls fail the both-seed depth-1 rule
because both seeds already meet the goal. These are reference-based exclusions,
not policy failures used to choose cases.

Shortest successful depths for every included deep parent and its paired control
are shown separately by seed. Depth 0 means the initial window already met the
goal; it does not mean permanent health.

| Parent configuration | Deep seed 307 | Deep seed 409 | Control seed 307 | Control seed 409 | Control inclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| `15169e1deb0dfcc4f073` | 3 | 3 | 0 | 0 | Excluded: initially meets goal |
| `3080e16887b90bb987b9` | 3 | 3 | 0 | 0 | Excluded: initially meets goal |
| `41116140a9181655a4f6` | 3 | 3 | 1 | 1 | Included |
| `4c3adbe20432c620cd8c` | 3 | 3 | 1 | 1 | Included |
| `b1969ef09ab74eede376` | 3 | 3 | 0 | 0 | Excluded: initially meets goal |

The stored `test-eligibility.jsonl` records minimum depth, strict-three status,
successful paths, and inclusion/exclusion for **every configuration at each seed**;
`test-references.jsonl` retains all endpoints. A null depth means no success within
this finite search, not impossibility at longer horizons. Inclusion requires both
seeds independently satisfying the rule; a favorable seed cannot qualify a parent.

Thus there are **five deep parent units / ten seed cases**, **two matched shallow
parent units / four seed cases**, and **42 policy episodes**, not hundreds of
independent performance samples. Separately, development had three deep parents
and two matched shallow controls: deep successes were scripted 0/6, heuristic
4/6, learned 0/6; all three achieved 4/4 on its shallow controls. No adapter,
threshold, model, scoring rule, or source changed after the final freeze or in
response to either split's policy results.

| Test cohort | Sequential adapter | Restored service | Mean final utility | Mean cost | Actual intervention depth |
| --- | --- | ---: | ---: | ---: | --- |
| Three-step | Scripted | 0/10 | 0.636723 | 10 | 3 in all cases |
| Three-step | Heuristic | 8/10 | 0.959225 | 10 | 3 in all cases |
| Three-step | Frozen learned transfer | 0/10 | 0.299082 | 7 | 2 in all cases |
| Matched one-step | Scripted | 4/4 | 1.000000 | 4 | 1 in all cases |
| Matched one-step | Heuristic | 4/4 | 1.000000 | 4 | 1 in all cases |
| Matched one-step | Frozen learned transfer | 2/4 | 0.926067 | 2.5 | 0 in two cases, 1 in two |

The learned adapter stopped for `no_eligible_intervention` after two tests in
all ten deep cases, despite having three cost units left. **Eight of those ten
stops still had a successful affordable continuation from their actual current
state**. The other two had already taken an order that could not finish within
the deadline. Scripted consumed all three interventions without restoring service;
heuristic exhausted the deadline in two cases. All had zero budget violations.

The shared goal-aware fuse switched to another untried action 20 times for
scripted, 20 for heuristic, and 10 for learned in the deep cohort. Service was
restored after a switch in 0, 8, and 0 episodes, respectively. These counts belong
to the **new shared adapters**, not the old Stage 4 fuse evaluation. Scripted had
19 local prediction matches without terminal service recovery; heuristic had 6
and learned 4. This directly demonstrates why local benefit is not the new goal.

### Concrete externally certified three-step path

For test parent `41116140a9181655a4f6`, seed 307:

| Cumulative path | Unique completions / arrivals | Queue growth | New retries | Goal met |
| --- | ---: | ---: | ---: | --- |
| Initial observation | 62 / 170 | 351 | 346 | No |
| Disable retries | 51 / 179 | 14 | 0 | No |
| Then double DB capacity | 56 / 177 | 12 | 0 | No |
| Then double workers | 186 / 181 | -149 | 0 | Yes |

Each row is a new window with persistent state. Every legal prefix of length
0, 1, or 2 fails, and no three-move path substituting a wait succeeds. The middle
resource change is an enabling intervention, not a cosmetic reframe: cumulative
capacity and retained duplicate backlog determine the later worker intervention's
effect. Paths, all endpoint telemetry, and the certificate are preserved in
`test-references.jsonl`.

The learned adapter followed the first two interventions, then estimated worker
utility **-0.223422**, below its frozen threshold **0.1**, and stopped at cost 7.
The independently measured third intervention would restore service at cost 10.
This exposes the mismatch between the **old one-step utility target** and a
cumulative resource configuration; it does not establish that a differently
trained sequential selector would necessarily succeed.

### Order sensitivity does not establish adaptive reasoning necessity

With exactly three permitted once-only repairs, "try all repairs" is an obvious
non-adaptive strategy. The existing scripted adapter is that reference in one
predeclared order, not a planner. Three necessary physical repairs do **not**
establish that three adaptive reasoning steps or information-dependent choices
are necessary.

The exhaustive reference already contains all six fixed orders, so order
sensitivity can be characterized without running another policy or simulator.
The following is **retrospective descriptive analysis of stored reference
outcomes**, not a new preregistered policy comparison or an independently selected
winning strategy. Let W = double workers, R = disable retries, D = double DB.
All orders cost 10 on the ten included deep seed cases:

| Non-adaptive repair order | Window goals met |
| --- | ---: |
| W, R, D (the evaluated scripted reference) | 0/10 |
| W, D, R | 0/10 |
| R, W, D | 10/10 |
| R, D, W | 8/10 |
| D, W, R | 0/10 |
| D, R, W | 0/10 |

Parent `3080e16887b90bb987b9` requires R,W,D at both seeds; the other four included
parents permit either retry-first order. The heuristic's two failures used R,D,W
on that parent. Thus order matters under the fixed deadline, but **one fixed order
works across all included cases**. The heuristic's 8/10 versus scripted's 0/10
cannot by itself demonstrate an advantage from planning, adaptation, or intuition.
No adapter was changed to the retrospectively successful order, and no reference
path was exposed to policy inference.

### What can and cannot be concluded

Across five parent units, learned-minus-heuristic success was **-0.8**, with a
descriptive paired-bootstrap 95% interval **[-1.0, -0.4]**. The utility difference
was **-0.660143**, interval [-0.758633, -0.549812]. Against scripted, success tied
at zero but learned utility was **0.337641 lower**; its three-unit cost saving
came with worse final remediation progress.

On the two matched parent pairs, the deep-minus-shallow change in the
learned-minus-heuristic success gap was **-0.5**, interval [-1, 0]. Against scripted
it was **+0.5**, interval [0, 1], because scripted also failed on deep tasks,
**not because learned transfer succeeded there**. Both seeds and both cohorts
stay together in parent-unit resampling. With only two matched parents, these
intervals are highly discrete and provide no robust causal depth estimate.

**Conclusion:** this implemented three-step test does not support the expectation
that the existing frozen learned selector becomes better simply because the
solution is farther away. It reveals underexploration/target-transfer limitations,
while the heuristic adapter succeeds on most of these selected cases. It does
not disprove the broader hypothesis for a future sequential-target learner.
The controls change initial remediation state as well as depth; structural
filtering is selective; the horizon is finite; all mechanisms remain familiar.
Among all 640 test references, there are only 372 distinct full-reference telemetry
signatures and 286 distinct initial observations, so unique parameter IDs also
overstate behavioral diversity. These are not independent new mechanisms.

Possible future work remains **proposed, not implemented**: use development-only
data to investigate sequential utility targets, underexploration, or calibrated
abstention, then freeze any changes before another untouched test. Do not tune on
this now-inspected split or reinterpret the original Stage 4 result as explained away.

### Reproducibility and verification

**239 pytest tests passed**, covering development-only exhaustive lower bounds,
positive three-step paths, rejection of shallow/wait solutions, meaningful state
transitions, pre-action evidence references, budget/capability/snapshot boundaries,
frozen model/provenance/archive checks, full stored-trace replay, matched statistics,
CLI/report readback, and all existing regressions. Every one of the original 17
Stage 4 executable/configuration source identities is unchanged; only three new
executable modules implement the separate task. Existing Stage 3/4 artifacts and
the source/model contents are preserved. Their old live-inventory execution check
correctly notices new modules; historical read-only artifact verification remains valid.

The official sequential preparation, development run, test run, and report CLI
were exercised. Report reload validates original archived code, model, all 52-node
reference trees, complete decision traces, independent scores, and raw/Markdown
roundtrip without advancing a simulator. There was one official test attempt,
no post-test source changes, and no training. The original local UI is unchanged;
this task is intentionally accessed through its own CLI/report.

Artifacts under `.runtime/sequential-v2/` include frozen source/model/protocol
archives, separate parent/control manifests, per-split exhaustive references and
eligibility, policy traces, reports, and sealed start/reference-complete/completion
markers. The set is about **21.7 MiB / 52 files**. Earlier development declarations
are retained in `.runtime/sequential-declaration-v1/` and
`.runtime/sequential-declaration-v2/`.

- Sequential freeze digest:
  `ac4dd4c8e0f2b29c68a91857c6200a41fcd9703754c5b243a70dc457b783a8f4`
- Frozen transfer model digest (unchanged from Stage 4):
  `ea8ceb42894b03a1500b594855c4d45a9da5cba21f12d1cd5ca7993ac1e375fe`

Test reference computation used **33,280 trajectories / 151,040 measurement
windows including warmup** (4,531,200 simulated ticks), taking about 66.1 seconds
on this machine. Policy execution took about 0.08 seconds and 312 total synthetic
action-cost units across 42 episodes; validation/reload and development computation
are additional. Oracle cost is explicitly excluded from policy costs. This is a
small local simulation benchmark, not a runtime-speed claim.

## Historical four-label question-learning baseline

This benchmark is conceptually separate from the service remediation work above.
After every actual **question/answer**, a policy may select a different next
question using the full prior history. Nothing is physically repaired. The bounded
story has four illustrative hypotheses: `wants_outside`, `returned_and_resting`,
`waiting_for_person`, and `possible_discomfort`. They are synthetic labels, not a
model of real animal behavior or a medical/veterinary diagnosis. The task offers
six fixed binary observation questions and a four-question budget.
This completed experiment predates the clarification that broad abstraction
families should be narrowed before finer interpretations. It has four leaf labels,
not an explicit group/leaf ontology, and **does not satisfy that refined goal**.
Its frozen results are retained rather than silently reinterpreted as evidence
for abstraction narrowing.

Every episode begins with exactly the same ambiguous observation. A policy-facing
API returns only that opening, already observed question/answer pairs, the explicit
posterior, remaining budget, and legal unasked questions. It does not return the
hidden sampled label, complete answer vector, future answers, generator parameters,
or reference paths. Repeats are illegal. The evaluator retains the full answer
vector solely to answer lawful questions and certify reference depth.

### Public evidence model and actual learned component

The finite public likelihood gives positive mass to all 64 six-answer patterns
under every hypothesis. Declared pair factors make the evidence naturally
interactive: for example, outdoor orientation plus response to an outside cue is
not treated as the product of two independent observations. After each answer,
exact marginal Bayesian updating over that table produces the displayed belief.
This update and the evidence-support rule are deterministic and **not learned**.
An answer is emitted only when top posterior is at least 0.78 and its margin over
second place is at least 0.28; otherwise the episode asks again or explicitly
abstains. Here **"supported" means crossing these working-model thresholds**, not
independent confirmation of the explanation. Every hypothesis assigns positive
probability to every full answer pattern, so even all six answers cannot logically
eliminate every competing hypothesis.

**Calibration limitation identified during review:** the benchmark partitions
uniformly shuffled full answer patterns, then draws a latent label conditional
on each full pattern. This is not sampling patterns according to the public
likelihood's own marginal distribution. Consequently the public model's posterior
given a *partial* history need not equal the benchmark generator's conditional
label probability. A displayed value of 0.78 is **not a validated 78% chance of
being right**, nor a probability about a real dog. Sampling and thresholds were
frozen before the run; they were not corrected or tuned after seeing its results.
Model-threshold compliance and actual sampled-label correctness are reported
separately. The experiment does not establish calibrated confidence.

The modest standard-library learner is real but narrow. Development-only
trajectory examples label every next question lying on an evaluator-certified
shortest path to eventual evidence-supported resolution. Add-one-smoothed
per-question positive rates are fitted over bias/depth, question-answer, and
answer-pair features. Ranking uses the complete observed history; model weights
never alter the Bayesian evidence update. A separately fitted memoryless ablation
uses only the latest answer. Features absent from fitting cause an explicitly
labeled unsupported-history abstention rather than a confident fallback.
`information_gain` is a labeled **nonlearned** public-model entropy policy, not
called intuition. `fixed_order` always asks the same questions.

Concrete reference branch checks demonstrate available discriminating questions:

- With the identical opening, asking `outside_orientation` first and receiving
  **yes** on pattern `101111` makes `outside_cue` the sole shortest-path next
  question; receiving **no** on pattern `000110` makes `comfortable_settle` the
  sole useful next question.
- With latest evidence fixed at `recent_return=no`, history
  `[outside_orientation=yes, recent_return=no]` requires `outside_cue`, while
  `[outside_cue=yes, recent_return=no]` requires `outside_orientation`. A
  latest-answer-only representation cannot distinguish these two states.

These are exhaustive-reference facts for fixed synthetic patterns, not fabricated
free-form reasoning or evidence that the learned ranker made those choices. In
fact, its autonomous first question is `outside_orientation`, and it next chooses
`comfortable_settle` after **either** first answer. That first learned decision
does not branch as the oracle example does.

The frozen learner **does** demonstrably use earlier answers at the next step:
in stored development trajectories, the same latest `comfortable_settle=no`,
same two asked question IDs, same remaining budget, and same legal next questions
lead to different choices:

| Earlier answer | Latest answer | Actual learned next question |
| --- | --- | --- |
| `outside_orientation=no` | `comfortable_settle=no` | `discomfort_indicator` |
| `outside_orientation=yes` | `comfortable_settle=no` | `outside_cue` |

Both observed branches occur in eleven development episodes. Thus history
conditioning is implemented, but it does not guarantee that every interaction
changes the next question or that every learned choice is optimal. The
memoryless **ranking** ablation drops earlier answers but still knows asked
question IDs/budget and uses the shared full-history evidence/stop rule; it is
not an agent with no memory anywhere.

Two concrete **training-split demonstrations**, not selected evaluation successes,
also have useful learned choices under the hindsight reference:

- Pattern `011100`: outdoor orientation **no**, comfortable settling **no**,
  discomfort indicator **no**, person departure **yes**; the model-threshold
  answer is `waiting_for_person`. Its third question is one of two optimal ties.
- Pattern `101101`: outdoor orientation **yes**, comfortable settling **no**,
  outside cue **yes**; the model-threshold answer is `wants_outside`.
  Its third question is the unique optimal continuation.

These demonstrate actual branching after the same latest answer, not a claim that
every displayed sequence is shortest from the opening or every answer is correct.

Each trace records the actual question text, answer, prior,
posterior, per-hypothesis increase/decrease interpretation, learned scores or
nonlearned policy label, and legal set. The separately marked
`reference_optimal_ties` annotations are evaluator-only hindsight, not information
available to the learner or its explanation for choosing a question. Scores are
fitted preference statistics, not calibrated probabilities of eventual success.

### Preregistration, split, depth, and first frozen result

Protocol `dog-history-v1`, generator seed **2026090901**, assigned disjoint complete
answer patterns to **28 training / 12 validation / 20 untouched test** episodes
before fitting. Exact complete-pattern overlap is zero. The opening, hypotheses,
questions, likelihood mechanism, feature schema, and some partial histories
intentionally overlap; this is finite-task compositional pattern holdout, not
new-domain or new-mechanism generalization. Validation is report-only: there are
no selected hyperparameters and no refit.

The fit uses **1,747 derived trajectory examples from 28 training patterns**,
not 1,747 independent episodes. Counting all nonempty unordered observed
question/answer subsets of sizes 1..3, there are 225 distinct training subsets,
178 validation subsets, and 230 test subsets; **223/230 test subsets already occur
in training**. This count covers possible evidence subsets, not only histories
visited by a policy. Full-pattern disjointness therefore coexists with substantial
structural overlap. The history learner pools all observed answers and pairs;
the static story does not require modeling temporal changes in answers.

An evaluator-only search checks all **517 legal distinct-question prefixes** of
depth 0..4 for each episode. Minimum depth is the first prefix satisfying the
declared evidence rule; it does not count steps or forbid early success and does
not claim guessing is impossible. On untouched test, minimum depth was 2 for 15
episodes and 4 for 4; one could not resolve within budget. Thus four cases have a
direct certificate requiring at least three distinct informative questions.
This is a certificate of **working-model threshold depth under the four-question
API**, not logical identification of the hidden label, a worst-case optimal
adaptive decision tree, or inability to guess correctly. The episode-specific
reference knows future answers; all equally short next questions are accepted
as ties in the secondary agreement metric. No test episode is filtered out:
short, deep, and unresolved cases all contribute to the main comparison.

One official test attempt ran only after model, preprocessing, generator,
evaluator, protocol, split files, environment, and exact/normalized live source
identities were frozen. No outcome-selected inclusion, retuning, or rerun occurred:

| Policy | Supported | Correct supported | Mean questions | Abstain | Unsupported confidence | Legal | Optimal-tie agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed order | 0.100 | 0.100 | 4.000 | 0.900 | 0.000 | 1.000 | 0.456 |
| Memoryless learned | 0.650 | 0.300 | 3.000 | 0.350 | 0.000 | 1.000 | 0.771 |
| History learned | 0.600 | 0.300 | 3.350 | 0.400 | 0.000 | 1.000 | 0.706 |
| Information gain | 0.400 | 0.250 | 3.400 | 0.600 | 0.000 | 1.000 | 0.667 |

The fixed order often spent all four questions without reaching support, so this
task contains branches where nonadaptivity wastes the budget. However, the
history-aware fit did **not** beat the memoryless ablation: it resolved one fewer
case, had the same correct-supported rate, cost 0.35 more questions, and had lower
secondary optimal-tie agreement. This honest negative comparison remains frozen.
Twenty test episodes are too few for broad conclusions, and reference agreement
uses future answers unavailable to policies, so it is secondary rather than the
optimization claim.

In counts, the history learner emitted 12 threshold-qualified answers, of which
**six were correct and six wrong**, and abstained eight times. Memoryless emitted
13 (six correct, seven wrong); information gain emitted eight (five correct,
three wrong); fixed order emitted two (both correct). **Zero unsupported
confidence is only compliance with the programmed threshold**, not zero wrong
confident answers, calibrated uncertainty, or independent truth verification.
Among emitted answers, history's measured correctness was 50%, not at least 78%.

Within the four certified depth-4 test cases, history and information gain each
answered one correctly and abstained on three; fixed and memoryless abstained on
all four. All spent four questions on each such case. This small subgroup is
descriptive, not evidence that the history learner is generally superior on
deeper tasks. Paired on all twenty episodes, history versus memoryless has
threshold-resolution wins/ties/losses **1/17/2**, and correctness wins/ties/losses
**1/18/1**. The first frozen comparison supports neither an overall history-model
advantage nor validation of learned intuition.

Run the isolated CLI inside WSL:

```bash
python -m intuition_prototype.dog_history_benchmark prepare .runtime/dog-history-v1
python -m intuition_prototype.dog_history_benchmark train .runtime/dog-history-v1
python -m intuition_prototype.dog_history_benchmark evaluate .runtime/dog-history-v1
python -m intuition_prototype.dog_history_benchmark report .runtime/dog-history-v1
python -m intuition_prototype.dog_history_benchmark demo .runtime/dog-history-v1 --pattern 011100 --policy learned_history
python -m intuition_prototype.dog_history_benchmark demo .runtime/dog-history-v1 --pattern 101101 --policy learned_history
python -m intuition_prototype.dog_history_benchmark continue .runtime/dog-history-v1 --observed comfortable_settle=yes
```

Use a new directory to reproduce: the official directory is single-use. Persisted
artifacts include split/protocol/preparation files, archived source, both strict
JSON models, development and raw test traces, official start/freeze/evaluation
seals, and replayed Markdown. Identifiers:

Both frozen model documents were independently reproduced exactly, using the
original archived code, the original 28 training patterns and original provenance.
That read-only development fit did not select parameters, write replacement
models, or execute the official test again.

`continue` accepts an explicitly **manually supplied observation history** and
shows the frozen model's next-question preference. For the starter
`comfortable_settle=no`, it chooses `outside_orientation`; for
`comfortable_settle=yes`, it chooses `recent_return`. This is a continuation demo,
not a claim that the autonomous model chose that starter or a new benchmark result.
Report/demo commands send interpretation warnings to stderr while preserving
machine-readable stdout.

- History model: `9240127f45bf68abcd2d3650a553faeaaae70712c0b2dbe4b79251d20dade21c`
- Memoryless model: `5cfb729b02cc09ab7bb279942d825e5b4d43fecd505d50b12302929eaa834231`
- Official freeze: `e2aa234e89314bf8a5a89a84ccc80939e702b398ff81872c98edd87aaf56b086`
- Raw test traces:
  `3edbecaa2fc32107431e7f15ec2672222d94af3ef173162d06bb6b9b7ea8b7ad`

The original pre-test suite had **251 passing tests**, including history branches,
same-latest/different-earlier cases, API leakage boundaries, evidence-versus-guess
abstention, exhaustive depth certificates, budget/repeat legality, deterministic
model roundtrip, split/provenance integrity, lifecycle freeze, and all historical
regressions. Existing UI semantics and historical Stage 3/4/sequential artifacts
and models are unchanged.

Post-test review added strict model schema/type/feature/weight/provenance checks,
exact archive inventory and stored-record validation, historical archived-code
report replay, and the labeled continuation demo. The suite then passed **274
tests** (35 dog-history tests plus the 239 earlier tests). These are explicitly
**post-test validation/interface changes**, not new learned-policy performance:
original model weights, frozen source archive, raw results, and original report
bytes remain unchanged. Live execution correctly rejects that old source freeze;
historical readback uses its verified archive. No official test was rerun.

Seven historical dog-report/demo checks require the local frozen
`.runtime/dog-history-v1` artifact. They explicitly skip when it is absent, as on
a fresh source-only checkout; they do not download, publish, or reconstruct an
official test run. Development-only lifecycle/model tests remain runnable without
that artifact. This publication-portability correction changes tests only, not
the saved experiment or its results.

Publication review also fixed completed-experiment demo dispatch: current archives
call their direct demonstration implementation rather than re-entering historical
dispatch recursively; older archives retain their original direct entry point.
Separate subprocess regression tests cover both forms. This is an interface fix,
not another evaluation or a change to frozen models, sources, raw results or reports.

## Context-first inquiry: asker purpose, abstractions, then details

This separate bounded task implements the refined question: **what does the asker
need to find out, and which evidence should narrow the interpretation space next?**
It has two illustrative interfaces over one compatibility/learning engine:

- **Dog at the door:** distinguish an explanation request, an outside-directed
  assessment, or characterization of an observed change. Explicit worry is stored
  as concern, not as evidence of illness or discomfort. This is not a real animal
  model, diagnosis, or treatment recommendation.
- **"Use a linked list":** distinguish the requested means from underlying
  operation constraints, such as indexed access, stable handles, or splicing.
  When the structure is explicitly required for implementation or education,
  honor that requirement instead of automatically reconsidering the representation.
  The output is a supported toy requirement profile or scoped fit assessment,
  **not production data-structure design advice or generated linked-list code**.

Each interface declares four broad families and sixteen finer profiles. These are
hand-constructed illustrative categories, not a universal abstraction ontology.
The domains share the same implementation and deliberately similar branching
structure; they are not two independent demonstrations of general intelligence.

### Context is not evidence about the world

Unknown intent makes a context clarification the only lawful initial action.
This is an **explicit API rule**, not a learned discovery that clarification is
needed. The answer supplies a goal; it does not remove any physical/requirement
candidate. Known explicit context skips clarification. Ambiguous intent causes
abstention rather than an invented purpose.

The same physical answer pattern is crossed with six context variants. All six
stay in one split and one statistical unit. Worried and curious explanation
contexts share the same physical evidence model; changing a goal changes relevance
and the scope of an answer, never the underlying hidden profile.

There are at most four world/requirement questions, plus one clarification when
needed. Each actual question costs one. Scoped goals may resolve while several
fine profiles remain; a practical outside-interest question need not demand the
same detail as "explain the pattern." Repeats are illegal, unknown observations
retain compatible candidates, and observed contradictions or unlisted answer
tokens cause explicit abstention. No minimum-question gate forces a long trace.

### What narrows, and what is learned

An explicit **nonlearned compatibility filter** retains every profile consistent
with the observed answers. Traces show before/after groups and leaves, evidence
links, count reductions, and log2 candidate-count entropy. Counts assume uniform
candidate weighting for description; they are **not calibrated probabilities**.
Zero surviving profiles means inconsistency, not perfect narrowing or certainty.
An out-of-model cause that mimics a modeled observation sequence cannot necessarily
be detected, especially if inquiry stops before the contradictory answer is asked.

The actual learned component is an add-one-smoothed question-preference table,
fitted to **1,014 development trajectory states**. Keys contain the supplied or
clarified goal, concern, asked-question set and compatible state derived from the
full history. The memoryless ranking ablation keeps the goal/concern and latest
answer; both policies share the full-history evidence and terminal-goal machinery.
Unseen learned states explicitly abstain.

Training labels use development-only future-answer search to reward eventual
scoped resolution, immediate relevant narrowing, group/leaf reductions and cost.
The numeric score is
`100*reachable + 30*resolved + 12*projection_reduction + 3*group_reduction +
leaf_reduction - 2*remaining_cost`, with the declared finite-budget penalty when
unreachable. All maximum-score ties are accepted. These are design weights, not
probabilities. Training histories include declared reference trajectories,
history-contrast examples, and a hand-authored public branching traversal:
**the learner learns preferences on deliberately supplied examples; it does not
discover the ontology, questions, or traversal curriculum**.

The comparators are fixed order, memoryless learned ranking, history-aware learned
ranking, and nonlearned public-model information gain. A fifth `reference` row
uses evaluator-only future answers and is **an oracle, not a fair deployable agent**.
Its choices and hindsight tie annotations are kept separate from agent evidence.

### Actual learned branches and context-first demonstrations

Development checks verify the fitted choices, not just oracle-optimal examples:

| Domain / fine-explanation goal | Learned first world question | If yes, next question | If no, next question |
| --- | --- | --- | --- |
| Dog | Active movement at the entry? | Outdoor-cue response? | Timing linked to a person? |
| Linked-list requirements | Stable handles or cheap splicing needed? | Stable external handles? | Frequent numeric indexing? |

Additional checks hold the latest answer, legal questions, and budget identical
while changing an earlier answer; the history learner changes its next question
in both domains, while the latest-answer ablation does not. Explicit goals also
change actual learned rankings without changing the initial compatible worlds.

Example validation demonstrations:

- Dog episode `7e7fb33077dc8eefd0f6`: clarify explanation goal; active-entry
  answer yes; outdoor cue no; first-half sequence yes; second-half sequence no.
  Result: `entry_pause_then_settle`, explicitly within the toy catalogue.
- Linked-list episode `e76a2c9402d67ea6f4ad`: clarify underlying requirements;
  handles/splicing yes; stable handles yes; first-half recurrence no; second-half
  recurrence no. Result: `stable_external_handles`, not blindly accepting the
  initially requested implementation.

In each example, clarification keeps **4 groups / 16 profiles** unchanged.
World observations then narrow to **2/8, 1/4, 1/2, 1/1**. These readable examples
demonstrate implemented behavior; they are not selected test cases or evidence
that the learned policy always follows an optimal path.

### Declaration, evaluation discipline, and commands

Each split has **38 world units**, nineteen per domain (sixteen modeled profiles
plus unknown-evidence, inconsistent, and out-of-model cases). Crossing six contexts
produces **228 rows per split**, not 228 independent worlds. Train, validation,
and test have distinct full domain/answer patterns but share all modeled profiles
and the finite compatibility mechanism; their variation is primarily nuisance
answers. This is not a held-out ontology, domain, or unknown-mechanism experiment.
No policy-outcome filtering selects the test cohort.

The original development artifact is retained at `.runtime/abstraction-experiment`.
Before official evaluation, review fixed an invalid demo-ID fallback and prevented
archived demo execution from writing bytecode into its immutable archive.
The final source snapshot is `.runtime/abstraction-context-final`; both learned
models and all validation result bytes reproduced **exactly** across this
pre-test packaging correction. No policy weights, generator, scoring criteria,
or validation outcomes were changed to improve performance.

Inside the activated WSL environment, inspect the final report and demonstrations:

```bash
python -m intuition_prototype.abstraction_benchmark report .runtime/abstraction-context-final
python -m intuition_prototype.abstraction_benchmark demo .runtime/abstraction-context-final --domain dog --episode-id 7e7fb33077dc8eefd0f6
python -m intuition_prototype.abstraction_benchmark demo .runtime/abstraction-context-final --domain linked_list --episode-id e76a2c9402d67ea6f4ad
```

Demo stdout is structured JSON with question text, actual answers, before/after
candidates, model scores/provenance, and the scoped result. It uses validation
episodes, never silently substitutes an unknown episode ID, and marks hindsight
reference annotations separately. The original Gradio resource UI is unchanged;
these are separate CLI interfaces.

To reproduce development in a **new** directory:

```bash
python -m intuition_prototype.abstraction_benchmark prepare .runtime/my-context-development
python -m intuition_prototype.abstraction_benchmark train .runtime/my-context-development
```

Official execution requires a reviewed `freeze` and an explicit one-use
`evaluate` authorization. Do not rerun an inspected test and describe it as fresh.
The archive preserves protocol/splits, exact executable sources, strict models,
validation traces, and sealed provenance; historical report reload checks the
stored artifact identities without silently using changed live rendering.

### Required means: post-evaluation acceptance correction

The original frozen v2 menu omitted the positive case where a linked list really
is required. The live **`abstraction-inquiry-v3-required-means`** menu now includes
`linked_required_structure`, covering an explicit implementation requirement or
an educational exercise about that structure.

This goal terminates at **scope acceptance**, without pretending implementation is
complete. With the requirement supplied explicitly, it asks zero questions; if
the requirement is learned from the goal clarification, it asks exactly one.
Both leave all **four requirement families and sixteen profiles** compatible:
the requested structure does not establish an operation profile or its suitability.
An ordinary, unexplained "use a linked list" request still permits clarification;
the positive case is never inferred just from the phrase.

Run either live acceptance demonstration without a model or runtime directory:

```bash
python -m intuition_prototype.abstraction_benchmark required-linked-list
python -m intuition_prototype.abstraction_benchmark required-linked-list --clarified
```

The JSON explicitly says `code_generated: false`, includes the retained uncertainty,
and states that no linked-list code was generated, implemented, or tested. This is
a **deterministic context/terminal-rule correction**, not newly learned behavior.
The live generator also includes explicit and clarified required-structure
variants for each linked-list world, so future training/evaluation can cover the
positive case rather than force every request through representation selection.
Future live partitions have six dog and eight linked-list contexts (266 rows per
split); the historical v2 results below still describe six contexts in each domain.

**No v3 model was fitted or test rerun.** The original v2 models, archive, raw
results and report remain unchanged, and historical report/demo CLI readback uses
that version's verified archive. The acceptance fix passed **17 targeted tests**,
including required-means CLI, zero/one-question depth, unchanged world candidates,
context separation, existing learned branches, grouped split regressions and
isolated archived imports.
The earlier 307-test result applies to the pre-correction v2 source.

Real historical-demo verification also exposed a v2 archive-loading defect:
without an archived package initializer, ordinary module lookup could select the
installed live package and recursively redispatch after source changes. The live
loader now explicitly imports the verified archive in an isolated, no-bytecode
process. V2's omitted `records`/`simulator` helper dependencies are checked against
their preserved source hashes and disclosed in the demo output; future archives
include them directly. The original v2 archive was not modified. Its historical
demo now completes with the entire artifact inventory byte-identical.

### Single frozen v2 test: actual outcomes and failures

After reviewing development traces and passing **307 tests**, the final source,
protocol, split and model identities were sealed and the official test executed
**once**. The final development preparation reproduced both original model
documents and validation results byte-for-byte. No weights, questions, scoring
rules, or source files were changed to tune these results. The later required-means
acceptance correction is separately versioned above and was not evaluated here.

The test contains **38 world units / 228 context-crossed rows / 1,140 policy
episodes**. The oracle row is included only as a separately labeled reference.

| Policy | Scoped resolutions | Scorable correct answers | Resolved but unscorable | Fine-profile resolutions | Mean questions | Abstentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed order | 90/228 | 85 | 5 | 0 | 2.6667 | 138 |
| Memoryless learned | 180/228 | 168 | 12 | 88 | 2.5088 | 48 |
| History-aware learned | 180/228 | 168 | 12 | 88 | 2.5088 | 48 |
| Nonlearned information gain | 177/228 | 165 | 12 | 85 | 2.5088 | 51 |
| Future-answer oracle, not deployable | 184/228 | 168 | 16 | 91 | 2.2895 | 44 |

**Denominators matter:** the generated report's `correct resolved` /
`correct_goal_resolution_rate` is correctness **conditional on a resolved answer
with scorable modeled truth**, not correctness across all 228 rows. Its value
1.0 therefore does not mean every episode was solved, nor that exception-case
answers were correct. The table above exposes the excluded counts. A fine-profile
resolution is also a compatibility result, not necessarily validated truth for
an exceptional world.

History-aware outcomes by interface:

| Interface | Rows | Scoped resolutions | Scorable correct | Mean questions |
| --- | ---: | ---: | ---: | ---: |
| Dog | 114 | 91 | 85 | 2.7456 |
| Linked-list requirements | 114 | 89 | 83 | 2.2719 |

History and memoryless tie on resolution at **all 38 world units**. Against fixed
order, history wins on 37 units and ties on one (mean paired difference +0.3947);
against information gain it wins on one and ties on 37 (+0.0132). These are
descriptive paired counts with all six contexts kept together, not independent
row-level significance tests or broad generalization claims.

All policies obeyed action legality, repeat and budget bounds. Each paid 76 total
clarifications across the 228 rows (mean 0.3333); explicit-context rows did not
repeat that question. History's remaining 496 cost units were actual observation
questions. Its 48 abstentions comprised 38 unresolved-intent cases, eight observed
out-of-model answers, and two physical-budget exhaustion cases.

**Observed boundary failure:** on the twelve inconsistent-world context variants,
history returned ten scoped answers and abstained on two unknown intents. On the
twelve out-of-model variants, it returned two scoped answers, detected the unusual
answer in eight, and abstained on two unknown intents. Those twelve emitted
exception answers are unscorable, not successes against ground truth. The policy
can stop on a compatible prefix before asking a contradiction-revealing question.
Zero `unsupported_answer` merely confirms its programmed compatibility gate;
it does not establish comprehensive model checking. Unknown-evidence cases had
eight scoped answers and four abstentions; all sixteen modeled profiles with
known intent were resolved by the history learner.

### Depth, overlap, and what the result establishes

Stored per-row reference certificates give shortest total inquiry depth:
**71 at depth 1, 28 at 2, 26 at 3, 42 at 4, 17 at 5, and 44 unresolved**.
Depth includes a clarification only when context is unknown. Thus 85 rows require
at least three total questions, but they are correlated context variants.
World-question depth is not merely clarification padding: among modeled explicit
fine-explanation rows, dog has 18 depth-4, 12 depth-3 and two depth-2 rows (two
concern variants per world); linked-list requirements has eight depth-4 and eight
depth-3 rows. The certificate applies only to these finite permitted questions,
compatibility rules, scoped goals and budget, not to the impossibility of guessing
or the necessary steps of human reasoning.

For example, explicit-goal test row `eda3ea8a7f6e72c4aec1` requires four dog
observations, and `114e4c2c386efdd8df26` requires four linked-list requirement
observations. Their shortest paths and per-row metadata are retained in the raw
test artifact. All short, deep, ambiguous and exceptional cases remain included;
none was selected because a learned policy succeeded.

Exact full domain/answer patterns do not overlap between splits, but **1,047 of
1,118 possible test partial answer subsets of sizes 1..3 also occur in training**.
All sixteen modeled profiles per domain occur in every split. Consequently this
test mostly checks new nuisance combinations within a familiar, hand-built
ontology. It does not establish learned abstraction discovery or transfer to an
unseen domain. Deliberately supplied training traversals further limit the claim.

**Conclusion:** the implementation now performs the requested context-first,
evidence-conditioned narrowing, rather than simply executing repairs or classifying
a dog label. Genuine learned branching is visible in both interfaces. However,
history did not improve the aggregate test outcomes over the memoryless ranker,
and unseen contradictions can escape detection. This is a working, bounded
pattern-learning experiment with explicit limitations, not validated intuition.

### Final provenance and verification

- Protocol: `65f7c64d138b1554cddf0a392c9cf147bf5c9039c3d27eafd3f60b3862141a6b`
- History model: `b6d66d907a4379d15ade07a9444417a2439ceba3edd88a72b1809349a0c66067`
- Memoryless model: `0a1378fbfb2a99d01e8375e11dbfe8e5d23a3657c574c5469c62e13cebc6127e`
- Final preparation file: `9febd4c85b9a22fa65836c8e389a320c54657de84caa74f3407d6f2d4efad5d6`
- Reviewed freeze file: `e1bbd18a901f3a4dbcf2e64082cba75a1194cb1bd8ee58143bdd6ef9eb44cbcf`
- Test raw traces: `28d65111b2c8a1a2f2ca97891815ec2426e5cda82acc750effba6ad4ba4d585e`

The full **307-test suite passed** before the official run, including both-domain
learned branches, context/truth isolation, source/model corruption checks,
shortest-depth checks, explicit-context skipping, unknown/contradictory evidence,
historical archived-demo byte preservation, and all prior regressions. Real CLI
demonstrations for both domains were verified before the final freeze. Freeze
verification plus official evaluation took about 269 seconds on this machine;
this includes development replay and oracle work and is not policy inference
time. No cloud calls, added dependencies, commit, or push were needed.

Final readback independently checked all **1,140 stored records** against their
allowed observed answers, compatible state, scoped goal, truth scoring where
defined, question costs and evidence traces; this did not rerun selection policies
or future-answer search. The real report CLI matched the sealed Markdown exactly.
The final artifact contains 28 files, about 25.4 MiB. Prior Stage 3/4, sequential,
and four-label dog artifacts remain intact and their historical reports reload.
The existing resource UI remained HTTP 200 without a restart.

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
- `sequential.py`: persistent intervention environment, exhaustive depth reference and named adapters.
- `sequential_protocol.py`, `sequential_benchmark.py`: declared structural filtering,
  matched shallow controls, frozen transfer evaluation and reports.
- `dog_history.py`: public finite evidence model, lawful question API, deterministic
  belief update, traces, and transparent fitted preference model.
- `dog_history_protocol.py`, `dog_history_benchmark.py`: disjoint pattern splits,
  exhaustive evidence-depth reference, four-policy lifecycle, freeze, reports, and CLI.
- `abstraction_inquiry.py`: context-separated compatibility state, scoped goals,
  history-conditioned fitted ranking, and two-domain observation API.
- `abstraction_protocol.py`, `abstraction_benchmark.py`: crossed context/world
  partitions, development training, bounded reference search, frozen comparison,
  and context-first CLI demonstrations.
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
- `tests/test_dog_history.py`: history-dependent branches, API/evidence boundaries,
  depth proof, fitted-model determinism, split isolation, and development lifecycle.
- `tests/test_abstraction_inquiry.py`: two-domain context isolation, actual learned
  branches, abstraction narrowing, bounded depth, strict provenance and CLI checks.

All three policies are bounded research infrastructure, not the proposed open-ended
learned investigative controller. Candidate discovery and fuse training remain deferred.

A future remote adapter must explicitly implement the protocol, extend mode
validation and the factory, and define credentials, timeouts, error handling,
and user-approved cost controls. The current scaffold deliberately selects no
provider and rejects `remote` rather than silently falling back to demo.
