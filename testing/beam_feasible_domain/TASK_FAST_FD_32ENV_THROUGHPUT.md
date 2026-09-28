# Codex task: 32-env Fast FD + Generic throughput stress benchmark

## Goal

Measure the REAL wall-clock throughput of the validated Fast local-point FD
Beam unilateral safety layer with the same multiprocessing model used by
training:

```text
32 independent SOFA21 processes
SubprocVecEnv(start_method="spawn")
batched epoch74 PPO inference
Fast local-point FD Beam unilateral safety in every worker
GenericConstraintSolver
```

This is NOT training. There are zero optimizer updates.

Run exactly:

```text
32 envs
1024 vector steps total
first 100 vector steps = warm-up only
last 924 vector steps = measured throughput
```

Total requested workload:

```text
32,768 env steps
65,536 physics substeps
```

Measured workload after warm-up:

```text
29,568 env steps
59,136 physics substeps
```

## Why fixed B02 copies

Every worker uses the same validated fixed:

```text
B02 / target_04
physics = 2 x 5 ms
requested margin = +0.100 mm
dense spacing = 0.010 mm
committed penetration limit = 0.001 mm
```

This deliberately creates a synchronized WORST-CASE contention benchmark.
When the deterministic trajectory enters the near-wall regime, many/all 32
workers can build unilateral rows in the same vector step.

Do not randomize this benchmark.

## New files

```text
testing/beam_feasible_domain/tests/fast_fd_32env_worker.py
testing/beam_feasible_domain/tests/benchmark_fast_fd_32env.py
```

Existing Fast FD code is reused:

```text
testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_fast_fd.py
testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_fast_full_episode.py
```

## Hard restrictions

Do NOT modify:

```text
mcr_sim/
training/
```

Do NOT modify the existing validated Fast FD algorithm.

Do NOT change:

- n_envs = 32;
- vector steps = 1024;
- warm-up = 100;
- physics = 2 x 5 ms;
- requested margin = +0.100 mm;
- dense spacing = 0.010 mm;
- penetration limit = 0.001 mm;
- GenericConstraintSolver;
- deterministic epoch74 policy.

Do NOT:

- train PPO;
- run optimizer updates;
- use LCP;
- use SLSQP q_candidate;
- use native candidate injection;
- use rollback/projection/action shielding;
- tune parameters after a failure;
- automatically rerun.

## 1. Work directory / git

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python

git status --short
git log -1 --oneline
```

Do not reset or clean unrelated files.

## 2. Record machine CPU limits

Run:

```bash
nproc
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
```

Do not change these variables for this benchmark. Just report them.

## 3. Python preflight

```bash
python -m py_compile \
  testing/beam_feasible_domain/tests/fast_fd_32env_worker.py \
  testing/beam_feasible_domain/tests/benchmark_fast_fd_32env.py
```

If a trivial syntax/import compatibility issue exists only in these two NEW
benchmark files, make the minimum fix there, show the diff, and continue.

Do not change the validated Fast FD scientific files unless compilation proves
an unavoidable import-only issue. Do not change physics or thresholds.

## 4. Plugin

Find the already validated component:

```bash
find testing/beam_feasible_domain/_runtime \
  -name 'libMCRBeamLinearizedUnilateral.so' -print
```

Reuse it.

If absent, rebuild the existing test-only plugin only. Do not alter its
algorithm.

## 5. Checkpoint

Use the same epoch74 checkpoint used by the full-episode acceptance.

The benchmark runner has the known checkpoint path as its default.

Before starting, verify that it exists.

## 6. Output protection

Runtime directory:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_32env_benchmark/
```

Confirm there is no old completed benchmark result from a previous attempt.

Do not delete unrelated runtime data.

## 7. Execute exactly one 32-env benchmark

Create directories:

```bash
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_32env_benchmark/logs
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_32env_benchmark/results
```

Run:

```bash
python testing/beam_feasible_domain/tests/benchmark_fast_fd_32env.py \
  --n-envs 32 \
  --vector-steps 1024 \
  --warmup-steps 100 \
  --device npu \
  --progress-every 25 \
  > testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_32env_benchmark/logs/run.log 2>&1
```

Run ONE copy only.

Do not use nohup unless the current shell/session requires it. If you do use
nohup, still run only one copy and record the PID.

## 8. Fail behavior

The parent benchmark checks all worker telemetry after every vector step.

If any worker reports:

- committed penetration > 0.001 mm;
- activeCount mismatch;
- Fast builder invariant failure;
- NaN/Inf;
- row-build/planning failure;
- wrong physics-substep count;
- missing benchmark telemetry;

the benchmark must stop after that vector step.

Do not tune or rerun.

## 9. Performance statistics

The first 100 vector steps are warm-up.

Only vector steps 101..1024 are used for primary throughput.

Report:

### End-to-end measured throughput

```text
measured wall time
vector steps/s
env steps/s
physics substeps/s
```

### Parent latencies

For policy inference:

```text
mean / p50 / p95 / p99 / max
```

For `SubprocVecEnv.step()` wall latency:

```text
mean / p50 / p95 / p99 / max
```

For full policy+env vector iteration:

```text
mean / p50 / p95 / p99 / max
```

### 32-env Fast FD contention

For individual row-build events across workers:

```text
count
mean / p50 / p95 / p99 / max
```

Also report:

```text
total rows
row-count distribution
support-mode counts
batched FD points
batched SDF calls
post-selection full-Beam profiles
```

Expected invariant:

```text
support mode = exact_selected_point_local_fd
one batched SDF call per row-build substep
zero post-selection full-Beam profiles
```

## 10. Concurrency

For each measured vector step, the parent counts how many of 32 workers
triggered at least one row-build.

Report distribution:

```text
0 envs -> N vector steps
1 env -> N
...
32 envs -> N
```

Also report the same distribution for workers with geometrically negative
`q_free < 0`.

Most important:

```text
max envs simultaneously building rows
```

Because all 32 copies follow the same fixed deterministic trajectory, a large
32-worker synchronized peak is expected and is useful as a stress condition.

## 11. Safety statistics

Safety is checked across ALL 1024 vector steps, including warm-up.

Report:

```text
total env steps
total physics substeps
q_free < 0 substeps
q_free < -0.001 mm substeps
worst free clearance
worst committed clearance
max committed penetration
committed violations
activeCount mismatches
fast-builder invariant failures
NaN/Inf
planning failures
first failure
```

The purpose is primarily performance, but do not accept performance numbers
from a physically invalid run.

## 12. Important interpretation

This benchmark includes:

- 32 spawned SOFA environments;
- real safety constraint work;
- batched PPO policy inference.

It does NOT include:

- PPO rollout-buffer processing;
- GAE;
- PPO epochs/minibatches;
- backpropagation;
- optimizer updates;
- validation callbacks.

Therefore it measures ENVIRONMENT/SAFETY rollout throughput, not full PPO
training epoch throughput.

Do not claim it is identical to end-to-end PPO training speed.

## 13. Results

Primary JSON:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_32env_benchmark/results/fast_fd_32env_benchmark.json
```

Report:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_32env_benchmark/results/fast_fd_32env_benchmark_report.md
```

Vector trace:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_32env_benchmark/results/fast_fd_32env_vector_trace.jsonl
```

Read the JSON/report yourself. Do not paste the full trace.

## Final response format

Return:

```text
32-ENV FAST FD + GENERIC THROUGHPUT BENCHMARK

Preflight:
git clean = YES/NO
py_compile = PASS/FAIL
plugin = PASS/FAIL
checkpoint = PASS/FAIL
nproc = ...
cgroup cpu.max = ...
thread env = ...

Configuration:
n_envs = 32
vector steps = 1024
warm-up = 100
measured steps = 924
policy device = npu
SubprocVecEnv = spawn
physics = 2 x 5 ms
production modified = NO
training = NO
optimizer updates = 0

Throughput:
measured wall = ... s
vector steps/s = ...
env steps/s = ...
physics substeps/s = ...

Policy inference latency:
mean / p50 / p95 / p99 / max = ...

VecEnv step latency:
mean / p50 / p95 / p99 / max = ...

Full vector iteration:
mean / p50 / p95 / p99 / max = ...

Fast FD row-build under 32-env contention:
events = ...
mean / p50 / p95 / p99 / max = ...
total rows = ...
row distribution = ...
support modes = ...
batched FD points = ...
batched SDF calls = ...
post-selection full-Beam profiles = ...

Concurrency:
envs-with-rows distribution = ...
max simultaneous row-build envs = ...
envs-with-negative-q_free distribution = ...

Safety:
env steps = ...
physics substeps = ...
q_free < 0 = ...
q_free < -0.001 mm = ...
worst free = ...
worst committed = ...
max committed penetration = ...
committed violations = ...
activeCount mismatches = ...
builder invariant failures = ...
NaN/Inf = ...
planning failures = ...
first failure = ...

FINAL = PASS / FAIL / INCONCLUSIVE

Q1. What is the measured 32-env env-steps/s?
Q2. What is the mean and p95 vector-step wall latency?
Q3. How much did individual Fast FD row-build latency inflate under 32-env contention compared with single-env 0.05623 s?
Q4. How many environments were simultaneously building rows at the worst vector step?
Q5. Did safety remain valid across the benchmark?
Q6. Is this enough to estimate rollout slowdown versus the old 32-env baseline?
Q7. Does this include PPO optimizer/update cost?
```

For Q7 answer NO.

If PASS, STOP. Do not begin training or production integration automatically.
