#!/usr/bin/env bash
set -euo pipefail

# 激活mcr_sofa虚拟环境并设置SOFA运行路径
source /data/home/3220251075/mcr_sim/setup_mcr_sofa.sh

# 当前平台中，SOFA核心和所有插件统一编译在build_plugins中
SOFA_BUILD=/data/home/3220251075/mcr_sim/mcr_env/sofa/build_plugins
SOFA_BUILD_PLUGINS=/data/home/3220251075/mcr_sim/mcr_env/sofa/build_plugins

STLIB_ROOT=/data/home/3220251075/mcr_sim/mcr_env/sofa/src/STLIB
SOFTROBOTS_LIB="$SOFA_BUILD_PLUGINS/external_directories/SoftRobots/lib"
BEAMADAPTER_LIB="$SOFA_BUILD_PLUGINS/external_directories/BeamAdapter/lib"
STLIB_LIB="$SOFA_BUILD_PLUGINS/external_directories/STLIB/lib"

PROJECT_PY=/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD_PLUGINS"

# Python模块路径
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$SOFA_BUILD_PLUGINS/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PROJECT_PY:${PYTHONPATH:-}"

# SOFA及插件动态库
export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# SOFA插件搜索路径
export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB:${SOFA_PLUGIN_PATH:-}"

# 防止多环境训练时每个进程再次创建大量CPU线程
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$PROJECT_PY"

# Unified training entrypoint; pass through all CLI args.
# This version does NOT save raw terminal logs.
# It only filters repetitive LocalMinDistance determinant warnings from console output.
python train_sac.py "$@" \
    2>&1 | awk '!/\[WARNING\] \[LocalMinDistance\(localmindistance\)\] Determinant is null/'
