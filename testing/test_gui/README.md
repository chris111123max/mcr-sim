# MCR SOFA Web Viewer (non-ROS)

This diagnostic viewer renders the existing non-ROS SOFA scene through
pyglet/EGL and exposes only the latest PNG frame plus camera controls on a
loopback HTTP port. It does not change training, rewards, observations,
actions, collision, or the ROS simulator.

Manual catheter control defaults to training-equivalent mode. The browser
sends a three-dimensional action to the existing `MCREnv.step(action)` path,
so action clipping, rate limiting, insertion safety damping, magnetic control,
and SOFA collision handling are identical to SAC environment interaction.
Start the simulation first, then hold I/K for local-N magnetic rotation, J/L
for local-B rotation, W/S for insertion/retraction, or Space to neutralize.

The viewer defaults also match `train_sac.py`: SOFA time step 0.01 s, frame
skip 1, settle steps 8, target threshold 0.003 m, maximum episode length 4096,
radius observation scale 0.005 m, and actor history length 4. The forced model
selects the vessel being inspected; EGL rendering, camera placement, and vessel
alpha are display-only differences. Collision defaults are the same scene
defaults used for training: vessel TriangleCollisionModel only, and catheter
LineCollisionModel plus PointCollisionModel.

The viewer uses nominal vessel scale 1.00 by default. Use
`--vessel-scale-factor 0.90` to inspect the hardest shrink-only geometry with
the same collision and control path before training.

On the Ascend worker, stop any previous server using port 8765 and run:

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
bash testing/test_gui/run_mcr_web_viewer.sh --model B01 --port 8765
```

The command prints a randomly generated access token. In a second terminal:

```bash
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8765
```

If Cloudflare prints `https://example.trycloudflare.com`, open:

```text
https://example.trycloudflare.com/?token=TOKEN_PRINTED_BY_THE_VIEWER
```

The quick-tunnel hostname and token are temporary secrets. Do not share them.
Stop both processes with Ctrl+C after inspection.

Available artificial models are B01..B05 and C01..C05. Example:

```bash
bash testing/test_gui/run_mcr_web_viewer.sh \
  --model C05 --vessel-scale-factor 0.90 \
  --width 1920 --height 1080 --fps 3
```
