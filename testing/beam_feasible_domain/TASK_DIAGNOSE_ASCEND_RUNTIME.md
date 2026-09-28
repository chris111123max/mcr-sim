# Codex task: Diagnose Ascend NPU runtime before 32-env benchmark

## Goal

Do NOT modify the benchmark, Fast FD algorithm, production code, training code,
conda environment, or system libraries.

The current blocker is strictly runtime:

```text
resolve_device("npu", distributed=False, local_rank=0)
-> import torch_npu
-> torch_npu._C load fails
-> libascend_hal.so not found
```

Find why the current shell cannot load the same Ascend runtime previously used
for MCR PPO training.

Do not start the 32-env benchmark in this task.

## Work directory

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
```

## 1. Record shell / Python environment

Run:

```bash
echo "SHELL=$SHELL"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "ASCEND_OPP_PATH=$ASCEND_OPP_PATH"
echo "PYTHONPATH=$PYTHONPATH"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"

which python
python -V
python -c 'import sys; print(sys.executable); print("\n".join(sys.path))'
```

Also:

```bash
python - <<'PY'
import importlib.metadata as m
for name in ("torch", "torch-npu", "torch_npu"):
    try:
        print(name, m.version(name))
    except Exception as e:
        print(name, "NOT_FOUND", type(e).__name__, e)
PY
```

## 2. Locate torch_npu without importing its C extension

Run:

```bash
python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("torch_npu")
print("spec =", spec)
print("origin =", getattr(spec, "origin", None))
print("locations =", list(getattr(spec, "submodule_search_locations", []) or []))
PY
```

Then locate NPU shared objects in the active Python environment:

```bash
PYROOT="$(python - <<'PY'
import sys
print(sys.prefix)
PY
)"

find "$PYROOT" -path '*torch_npu*' -type f \( -name '*.so' -o -name '_C*' \) -print 2>/dev/null | head -100
```

For the primary torch_npu extension/shared library, run `ldd` and show only
missing dependencies plus Ascend-related dependencies:

```bash
find "$PYROOT" -path '*torch_npu*' -type f -name '*.so' -print 2>/dev/null | while read f; do
  echo "===== $f ====="
  ldd "$f" 2>/dev/null | grep -E 'not found|ascend|hccl|cann|acl|runtime|hal' || true
done
```

Do not modify anything.

## 3. Locate libascend_hal.so

Search likely system locations first:

```bash
for d in /usr/local/Ascend /opt/Ascend /usr/local /opt; do
  if [ -d "$d" ]; then
    find "$d" -name 'libascend_hal.so*' -print 2>/dev/null
  fi
done
```

If not found there, search the user's persistent data area:

```bash
find /data/home/3220251075 -name 'libascend_hal.so*' -print 2>/dev/null | head -100
```

Also:

```bash
ldconfig -p 2>/dev/null | grep -i ascend || true
ldconfig -p 2>/dev/null | grep -i 'libascend_hal' || true
```

## 4. Locate official Ascend environment scripts

Run:

```bash
find /usr/local/Ascend /opt/Ascend /data/home/3220251075 \
  -maxdepth 5 -type f \
  \( -name 'set_env.sh' -o -name 'setenv.bash' -o -name 'setenv.sh' \) \
  -print 2>/dev/null | head -100
```

Do NOT source any script yet.

For each candidate under an Ascend/CANN directory, print its path and relevant
exports only:

```bash
while read f; do
  echo "===== $f ====="
  grep -E 'ASCEND|LD_LIBRARY_PATH|PYTHONPATH|PATH=' "$f" | head -100
done < <(
  find /usr/local/Ascend /opt/Ascend /data/home/3220251075 \
    -maxdepth 5 -type f \
    \( -name 'set_env.sh' -o -name 'setenv.bash' -o -name 'setenv.sh' \) \
    -print 2>/dev/null
)
```

## 5. Check whether the known mcr_sofa environment differs

The historically used environment is expected around:

```text
/data/home/3220251075/mcr_env/miniforge3/envs/mcr_sofa
```

Do not activate/change it yet. Only inspect:

```bash
ls -ld /data/home/3220251075/mcr_env/miniforge3/envs/mcr_sofa 2>/dev/null || true

/data/home/3220251075/mcr_env/miniforge3/envs/mcr_sofa/bin/python -V 2>/dev/null || true

/data/home/3220251075/mcr_env/miniforge3/envs/mcr_sofa/bin/python - <<'PY' 2>/dev/null || true
import importlib.metadata as m
for name in ("torch", "torch-npu", "torch_npu"):
    try:
        print(name, m.version(name))
    except Exception as e:
        print(name, "NOT_FOUND", type(e).__name__, e)
PY
```

Also run the same `find_spec("torch_npu")` with that Python, but do NOT import
torch_npu directly yet.

## 6. Minimal controlled test only if an official CANN set_env script is found

Only if a clearly official Ascend/CANN environment script exists, run a
SUBSHELL so the parent shell remains unchanged:

```bash
bash -lc '
  set -e
  source "<OFFICIAL_ASCEND_SET_ENV_SCRIPT>"
  cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
  /data/home/3220251075/mcr_env/miniforge3/envs/mcr_sofa/bin/python - <<'"'"'PY'"'"'
from mcr_sim.distributed import resolve_device
d = resolve_device("npu", distributed=False, local_rank=0)
print("requested =", d.requested)
print("resolved =", d.resolved)
print("accelerator =", d.accelerator)
PY
'
```

Replace the placeholder only with the actual discovered official script path.

Do not manually construct LD_LIBRARY_PATH yet.
Do not install/reinstall torch or torch_npu.
Do not create symlinks.
Do not copy libascend_hal.so.
Do not edit shell rc files.

## Stop conditions

Stop after diagnosis. Do NOT run the 32-env benchmark.

## Final response format

Return:

```text
ASCEND NPU RUNTIME DIAGNOSIS

Current shell:
python = ...
CONDA_DEFAULT_ENV = ...
CONDA_PREFIX = ...
torch = ...
torch-npu = ...

torch_npu:
package location = ...
missing shared libraries = ...

libascend_hal.so:
found = YES/NO
paths = ...

Ascend/CANN env scripts:
found = YES/NO
candidate official script = ...

Known mcr_sofa:
exists = YES/NO
python = ...
torch = ...
torch-npu = ...
same as current python = YES/NO

Controlled official-env NPU test:
attempted = YES/NO
resolve_device result = PASS/FAIL
resolved device = ...
error = ...

ROOT CAUSE:
...

MINIMUM FIX:
...

SAFE NEXT COMMAND:
...

No production modification.
No package installation.
No symlink/copy.
No benchmark.
```

The minimum fix should prefer:
1. using the known mcr_sofa Python;
2. sourcing the official CANN/Ascend environment script in the benchmark shell;
3. no package/library mutation.

Do not make the fix automatically in this task.
