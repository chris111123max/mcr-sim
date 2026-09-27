# Codex task: B02 step665 multi-dense active-set diagnosis

## Goal

Diagnose and test the first solver failure from the final dense-aware full-episode run:

- B02 / target_04 / seed 15204
- protected trajectory
- RL step 665 / physics substep 1
- requested margin = 0.100 mm
- q_prev clearance ~= +0.468945 mm
- q_free clearance ~= -0.032075 mm
- current strict solver: FAIL: FEASIBLE LINE SEARCH STAGNATION
- current strict solver internal sampled clearance near the failure: ~+0.102238 mm

The working hypothesis is:

> one scalar gradient of the current dense argmin is not enough when multiple
> near-worst dense locations compete or the argmin switches during line search.

This task compares the current single-argmin strict solver against a test-only
multi-dense active-set variant on the EXACT SAME q_prev/q_free target frame.

Do not run another full episode.

## New files

Baseline solver remains unchanged:

```text
testing/beam_feasible_domain/strict_margin_solver/strict_b02_feasible_solve.py
```

Test-only scientific variant:

```text
testing/beam_feasible_domain/strict_margin_solver/strict_b02_feasible_solve_multidense.py
```

Target runner:

```text
testing/beam_feasible_domain/tests/diagnose_step665_multidense_stagnation.py
```

## Scientific comparison

Prefix through step664 uses the current archived strict 0.100 mm solver so the
protected trajectory remains the same as the failed full-episode test.

At step665/substep1 only:

1. capture the real current q_prev and q_free;
2. run the CURRENT single-argmin strict solver on those states for trace only;
3. do not let that baseline call write SOFA or arm native;
4. run the multi-dense active-set variant on the SAME q_prev/q_free;
5. if and only if the variant returns a strict dense-certified >=0.100 mm
   candidate, arm the same native hook;
6. complete that one real SOFA physics substep;
7. check committed real-Beam clearance;
8. stop immediately before substep2.

## Multi-dense variant

The test-only variant keeps up to 8 separated local minima from the 10 um
independent dense profile as pointwise SLSQP inequalities.

It does NOT:

- change requested margin;
- relax certification;
- change objective;
- use CollisionDOFs as solver constraints;
- use native post-contact state;
- rollback/project;
- write committed Beam position.

The goal is to test active-set representation only.

## Hard scope

All code changes are already under testing/.

Do not modify:

```text
mcr_sim/
training/
```

Do not train.

Do not change:

- margin 0.100 mm;
- dense gate 0.100 mm;
- committed limit 0.001 mm;
- 2 x 5 ms physics;
- policy/checkpoint/seed.

Do not run a full 2048-step episode.

## Preflight

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python

git status --short

python -m py_compile \
  testing/beam_feasible_domain/strict_margin_solver/strict_b02_feasible_solve_multidense.py \
  testing/beam_feasible_domain/tests/diagnose_step665_multidense_stagnation.py
```

If py_compile exposes a simple test-only syntax/import issue, fix only under
testing/ and continue.

## Execute

```bash
mkdir -p \
  testing/beam_feasible_domain/_runtime/strict_margin_0p100/step665_multidense_diagnosis/logs \
  testing/beam_feasible_domain/_runtime/strict_margin_0p100/step665_multidense_diagnosis/results

python testing/beam_feasible_domain/tests/diagnose_step665_multidense_stagnation.py \
  --step665-validation \
  --max-rl-steps 665 \
  --progress-every 50 \
  > testing/beam_feasible_domain/_runtime/strict_margin_0p100/step665_multidense_diagnosis/logs/run.log 2>&1
```

Run only one copy.

## Replay gate

At target require approximately:

```text
q_prev clearance = +0.468945 mm
q_free clearance = -0.032075 mm
```

The runner uses a small tolerance for the rounded report values.

If target state is outside tolerance:

```text
STEP665_PROTECTED_STATE_MISMATCH
```

and stop.

Report the realized action-prefix SHA256, but do not invent an expected hash.

## Read results

Primary:

```text
testing/beam_feasible_domain/_runtime/strict_margin_0p100/step665_multidense_diagnosis/results/step665_multidense_result.json
```

Solver traces:

```text
testing/beam_feasible_domain/_runtime/strict_margin_0p100/step665_multidense_diagnosis/step665_single_argmin_baseline_trace.jsonl
testing/beam_feasible_domain/_runtime/strict_margin_0p100/step665_multidense_diagnosis/step665_multidense_trace.jsonl
```

Read both traces yourself.

For each outer iteration compare:

- current dense min;
- active dense constraint count;
- active dense indices;
- active dense clearances;
- line-search alpha;
- trial dense clearance;
- whether the active/worst indices switch;
- whether one near-worst location improves while another becomes limiting.

Do not paste the entire JSONL.

## Interpretation

### Supports multi-point competition hypothesis

Strong evidence is:

- baseline reproduces stagnation;
- baseline dense argmin/worst region switches or trials fail to improve the
  true dense minimum;
- multi-dense variant returns >=0.100 mm on exactly the same q_prev/q_free;
- native pipeline succeeds;
- committed penetration <=0.001 mm.

Then classify:

```text
ROOT CAUSE:
SINGLE_DENSE_ARGMIN_CONSTRAINT_INSUFFICIENT_UNDER_ACTIVE_SET_SWITCHING
```

### Rejects hypothesis

If the multi-dense variant also stagnates, report exactly how:

- active-set indices/count;
- best dense clearance reached;
- whether all directions degrade all near-worst points;
- SLSQP status;
- line-search behavior.

Do NOT automatically modify the solver again.

Do NOT run full episode.

## PASS for this diagnostic

PASS means only:

```text
target replay matched
baseline comparison collected
multi-dense candidate >= 0.100 mm
native fire delta = 1
native mapping/velocity = PASS
committed penetration <= 0.001 mm
no forbidden fallback
no NaN/Inf
```

It does NOT mean the full episode passes.

## Final response

Return:

```text
B02 STEP665 MULTI-DENSE ACTIVE-SET DIAGNOSIS

Replay:
step/substep = 665/1
action-prefix SHA256 = ...
q_prev clearance = ...
q_free clearance = ...
state matched = YES/NO

Baseline single-argmin:
status = ...
outer iterations = ...
line-search trials = ...
best dense clearance = ...
stagnation reproduced = YES/NO
worst/active switching evidence = ...

Multi-dense variant:
status = ...
outer iterations = ...
line-search trials = ...
max dense active constraints = ...
active indices / clearances = ...
candidate dense clearance = ...
solver runtime = ...

Native:
fire delta = ...
status = ...
mapping change = ...
velocity correction = ...
parent write error = ...

Committed:
clearance = ...
penetration = ...

Diagnosis:
multi-point competition hypothesis = SUPPORTED / REJECTED / INCONCLUSIVE
root cause = ...

Forbidden fallback = 0 / ...
NaN/Inf = 0 / ...

FINAL:
PASS / FAIL / INCONCLUSIVE

Q1. Did baseline reproduce the step665 stagnation?
Q2. Did the baseline trace show dense worst-point/near-worst competition?
Q3. Did multi-dense constraints produce a strict >=0.100 mm candidate?
Q4. Did that candidate commit safely through the native pipeline?
Q5. Should multi-dense active constraints be the next solver formulation to
    validate on a full episode?
```

After reporting, STOP.

Do not run full episode.
Do not train.
Do not modify production.
