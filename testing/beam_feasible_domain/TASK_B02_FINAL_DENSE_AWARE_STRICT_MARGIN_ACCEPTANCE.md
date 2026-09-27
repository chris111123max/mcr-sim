# Codex task: final B02 dense-aware strict-margin full-episode acceptance

## Goal

Run the final test-only engineering acceptance using the already-archived dense-aware strict solver.

This is the first full-episode run after the step662 CASE-B fix:

- strict requested margin = 0.100 mm
- independent dense gate = 0.100 mm
- committed penetration limit = 0.001 mm
- dense solver constraint includes the 10 um independent worst-clearance finite-difference gradient
- 2 x 5 ms physics substeps
- B02 / target_04 / seed 15204
- epoch74 deterministic PPO checkpoint
- max 2048 RL steps
- safety layer enabled from the first physics substep

Do not modify solver mathematics before this run.

## Runner

```text
testing/beam_feasible_domain/tests/b02_full_episode_dense_aware_strict_margin_0p100_final.py
```

This runner wraps the existing strict full-episode harness and writes to a fresh result namespace so historical runs are preserved.

It also refuses to run if the step662 dense-gradient fix is missing.

## Scope

All work remains test-only.

Do not modify production files.

Do not train.

Do not increase margin.

Do not lower the dense gate.

Do not change committed tolerance.

Do not change physics substeps.

Do not run a baseline OFF episode.

## Preflight

From:

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
```

Run:

```bash
git status --short
python -m py_compile \
  testing/beam_feasible_domain/tests/b02_full_episode_dense_aware_strict_margin_0p100_final.py \
  testing/beam_feasible_domain/strict_margin_solver/strict_b02_feasible_solve.py
```

Confirm the previous step662 diagnosis PASS exists if available:

```text
testing/beam_feasible_domain/_runtime/strict_margin_0p100/step662_diagnosis/results/step662_solver_result_fix1.json
```

Do not rerun step662.

## Execute exactly once

```bash
mkdir -p testing/beam_feasible_domain/_runtime/dense_aware_strict_margin_0p100_final/logs

python testing/beam_feasible_domain/tests/b02_full_episode_dense_aware_strict_margin_0p100_final.py \
  --max-rl-steps 2048 \
  --progress-every 25 \
  > testing/beam_feasible_domain/_runtime/dense_aware_strict_margin_0p100_final/logs/full_episode.log 2>&1
```

Do not launch a second copy in parallel.

The inherited runner is fail-fast.

## Hard PASS requirements

PASS requires the complete fixed episode / 2048-step horizon to satisfy all of:

- candidate requested margin = 0.100 mm
- every unsafe free state produces an independently dense-certified candidate >= 0.100 mm, allowing only the existing 1e-12 m roundoff tolerance
- solver failures = 0
- candidate certification failures = 0
- native failures = 0
- mapping propagation failures = 0
- velocity correction failures = 0
- committed violations = 0
- NaN/Inf = 0
- forbidden fallback count = 0
- max committed penetration <= 0.001 mm
- at least one unsafe free state genuinely challenges the solver
- episode ends normally or reaches 2048 RL steps

Route success itself is not required.

## If FAIL

Stop at the first real failure.

Do not automatically modify code.

Do not automatically try 0.11 / 0.12 / 0.15 mm.

Return the exact first failure, including if available:

- step/substep
- q_prev clearance
- q_free clearance
- solver status
- solver iterations/runtime
- candidate dense clearance
- candidate worst point
- native fire delta/status
- mapping change
- committed clearance
- candidate-to-committed clearance loss
- failure classification

## Results

Read:

```text
testing/beam_feasible_domain/_runtime/dense_aware_strict_margin_0p100_final/results/b02_full_episode_strict_margin_0p100.json
testing/beam_feasible_domain/_runtime/dense_aware_strict_margin_0p100_final/results/b02_full_episode_strict_margin_0p100_report.md
testing/beam_feasible_domain/_runtime/dense_aware_strict_margin_0p100_final/results/b02_full_episode_strict_margin_0p100_trace.jsonl
```

## Final response

Return:

```text
B02 FINAL DENSE-AWARE STRICT 0.100MM FULL-EPISODE ACCEPTANCE

Configuration:
seed = 15204
requested margin = 0.100 mm
dense candidate gate = 0.100 mm
committed penetration limit = 0.001 mm
physics = 2 x 5 ms
dense-aware step662 fix present = YES/NO
production files modified = NO
training launched = NO

Run:
RL steps executed = ...
physics substeps = ...
terminated/truncated = ...
terminal reason = ...

Solver:
unsafe free states = ...
solver calls = ...
strict candidate certifications = ...
candidate certification failures = ...
solver failures = ...
minimum candidate clearance = ...
runtime total / mean / max = ...

Native:
injections = ...
native failures = ...
mapping failures = ...
velocity failures = ...
fireCount final = ...

Committed:
worst committed clearance = ...
max committed penetration = ...
committed violations = ...

Other:
NaN/Inf = ...
forbidden fallback count = ...

FINAL:
PASS / FAIL / INCONCLUSIVE

Reason:
...

Q1. Did the dense-aware strict solver solve every unsafe free state?
Q2. Did every injected candidate independently satisfy >=0.100 mm?
Q3. Did every committed Beam state stay within the 0.001 mm penetration limit?
Q4. Did the fixed B02 episode complete / reach 2048?
Q5. If PASS, has the current B02 fixed worst-case engineering penetration acceptance passed?
Q6. Does this mathematically prove all future states/policies/vessels cannot penetrate?
```

Q6 must be NO.

After returning the result, stop. Do not train.
