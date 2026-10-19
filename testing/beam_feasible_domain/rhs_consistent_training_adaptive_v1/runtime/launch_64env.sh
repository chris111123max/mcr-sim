#!/usr/bin/env bash
set -e
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
MODE=${1:?smoke or formal}
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=ppo_beam_rhs_adaptive_64env_${MODE}_${STAMP}
RUN=$PWD/../training_runs/$NAME
mkdir -p "$RUN/models" "$RUN/logs"
export MCR_CANDIDATE_RUN_DIR="$RUN"
if [ "$MODE" = smoke ]; then export MCR_ADAPTIVE_SMOKE=1; else unset MCR_ADAPTIVE_SMOKE; fi
nohup python testing/beam_feasible_domain/rhs_consistent_training_adaptive_v1/train_candidate.py \
 --device npu --epochs 100 --episodes-per-epoch 100 --n-envs 64 --n-steps 256 --batch-size 1024 --n-epochs 5 \
 --physics-substeps 2 --frame-skip 1 --time-step 0.005 --training-curriculum \
 --beam-safety-validation audit --beam-safety-audit-interval-substeps 16 --beam-safety-audit-near-clearance-mm 0.300 \
 --beam-safety-plugin mcr_sim/native/beam_unilateral/_build/lib/libMCRBeamLinearizedUnilateral.so \
 --exp-name "$NAME" > "$RUN/startup.out" 2>&1 < /dev/null &
PID=$!
python - "$RUN" "$PID" "$MODE" <<'META'
import sys,json
from pathlib import Path
p=Path(sys.argv[1]);(p/'launch.json').write_text(json.dumps({'pid':int(sys.argv[2]),'mode':sys.argv[3],'run':str(p)},indent=2))
print('LAUNCH',sys.argv[2],str(p),flush=True)
META
