#!/usr/bin/env bash

# 1. 继承 SOFA 必须的环境变量
export SOFA_ROOT=/home/chen/SOFAA/sofa_ws/sofa/build_plugins
export SOFAPYTHON3_ROOT=/home/chen/SOFAA/sofa_ws/sofa/build_plugins
export PYTHONPATH=/home/chen/SOFAA/sofa_ws/sofa/build_plugins/lib/python3/site-packages:/home/chen/SOFAA/sofa_ws/STLIB:/home/chen/SOFAA/sofa_ws/STLIB/python:/home/chen/SOFAA/sofa_ws/STLIB/python3/src:/home/chen/SOFAA/projects/mCR_simulator-master/python:$PYTHONPATH

# 2. 进入运行目录
cd /home/chen/SOFAA/projects/mCR_simulator-master/python

# 3. 运行模型，并把外部命令行参数继续传给 run_trained_mcr_ros.py
# 例如：
# ./run_inference_closed_loop.sh --force-model V1
# ./run_inference_closed_loop.sh --force-model 0207 --max-episodes 3
PYTHONUNBUFFERED=1 python3 run_trained_mcr_ros.py \
  --model /home/chen/SOFAA/projects/mCR_simulator-master/python/runs_tri/centerline_light_2mm_from_3mm_aortic_2mm_20260517_184506/models/sac_mcr_2mm_V1_y_noS_ckpt_3800000_steps.zip \
  "$@" \
  2>&1 | grep --line-buffered -v "Determinant is null"
