#!/usr/bin/env bash

export SOFA_ROOT=/home/chen/SOFAA/sofa_ws/sofa/build_plugins
export SOFAPYTHON3_ROOT=/home/chen/SOFAA/sofa_ws/sofa/build_plugins
export PYTHONPATH=/home/chen/SOFAA/sofa_ws/sofa/build_plugins/lib/python3/site-packages:/home/chen/SOFAA/sofa_ws/STLIB:/home/chen/SOFAA/sofa_ws/STLIB/python:/home/chen/SOFAA/sofa_ws/STLIB/python3/src:/home/chen/SOFAA/projects/mCR_simulator-master/python:$PYTHONPATH

cd /home/chen/SOFAA/sofa_ws/sofa/build_plugins/bin

./runSofa-21.12.00 \
-l /home/chen/SOFAA/sofa_ws/sofa/build_plugins/lib/libSofaPython3.so \
-l /home/chen/SOFAA/sofa_ws/sofa/build_plugins/lib/libSoftRobots.so.1.0 \
-l /home/chen/SOFAA/sofa_ws/sofa/build_plugins/external_directories/BeamAdapter/lib/libBeamAdapter.so.21.12 \
/home/chen/SOFAA/projects/mCR_simulator-master/python/example_aortic_arch_ros.py 2>&1 | grep --line-buffered -v "Determinant is null"
