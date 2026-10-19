#!/usr/bin/env bash
# Handoff only: NOT executed in this delivery.
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master
source ../../setup_mcr_sofa.sh
set -euo pipefail
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
python/stEVE_anatomical_vessels/.venv/bin/python -m mcr_sim.vessel_generator --seed 101
python -m mcr_sim.integration_smoke --seed 101 --steps 1 --action 0 0 0 \
  2>&1 | tee python/mcr_sim/reports/luna_seed101_step1.log
