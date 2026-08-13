#!/usr/bin/env python3
"""Password-token protected browser viewer for the non-ROS MCR SOFA scene.

The SOFA scene is rendered off-screen through the existing pyglet/EGL path.
Only the latest PNG frame and small camera-control requests are served.  The
HTTP server intentionally binds to loopback; use an outbound tunnel such as
Cloudflare Tunnel when the compute platform does not expose user ports.
"""

import argparse
import binascii
import json
import math
import os
import queue
import secrets
import struct
import sys
import threading
import time
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np


TEST_GUI_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TEST_GUI_DIR.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.mcr_rl_env import ActionType, EnvType, MCREnv, ObservationType
from mcr_sim.rl_core.base import RenderFramework, RenderMode
from mcr_sim.training_config import (
    ACTOR_HISTORY_STEPS,
    ENTRY_TANGENT_POINTS,
    FRAME_SKIP,
    INITIAL_ORIENTATION_MAX_ANGLE_DEG,
    MAX_EPISODE_STEPS,
    RADIUS_OBSERVATION_SCALE_M,
    SETTLE_STEPS,
    SOFA_TIME_STEP_S,
    TARGET_THRESHOLD_M,
)
from mcr_sim.paths import VALID_MESH_DIR


ARTIFICIAL_MODELS = [f"B{i:02d}" for i in range(1, 6)] + [
    f"C{i:02d}" for i in range(1, 6)
] + [f"V{i:02d}" for i in range(1, 6)]


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as PNG using only the standard library."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {image.shape}")
    image = np.ascontiguousarray(image)
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in image)

    def chunk(name: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", binascii.crc32(name + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=1))
        + chunk(b"IEND", b"")
    )


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MCR SOFA Web Viewer</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #11151a; color: #e7edf3; overflow: hidden; }
    #top { height: 52px; display: flex; align-items: center; gap: 14px;
           padding: 8px 14px; background: #1c232b; border-bottom: 1px solid #34414e; }
    #title { font-weight: 700; white-space: nowrap; }
    #status { font-size: 13px; color: #a9bac9; overflow: hidden; text-overflow: ellipsis; }
    #stage { position: relative; height: calc(100vh - 52px); display: grid; place-items: center; }
    #frame { max-width: 100%; max-height: 100%; width: 100%; height: 100%;
             object-fit: contain; user-select: none; cursor: grab; background: #202f3d; }
    #frame.dragging { cursor: grabbing; }
    #controls { position: absolute; left: 14px; bottom: 14px; display: flex;
                flex-direction: column; gap: 7px; padding: 9px; border-radius: 9px; background: #111a22dd;
                border: 1px solid #415263; backdrop-filter: blur(5px); }
    .control-row { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; }
    .control-label { min-width: 64px; color: #a9bac9; font-size: 12px; }
    button { border: 1px solid #587086; background: #253646; color: #eef6fc;
             border-radius: 6px; padding: 7px 10px; cursor: pointer; }
    button:hover { background: #34516a; }
    button.active { background: #176b87; border-color: #69d8fb; }
    #hint { position: absolute; right: 14px; bottom: 14px; padding: 8px 10px;
            border-radius: 7px; background: #111a22cc; color: #b7c7d4; font-size: 12px; }
    #error { position: absolute; top: 16px; background: #741f2bcc; border: 1px solid #ef7181;
             padding: 8px 12px; border-radius: 6px; display: none; }
  </style>
</head>
<body>
  <div id="top"><div id="title">MCR SOFA Web Viewer</div><div id="status">连接中…</div></div>
  <div id="stage">
    <img id="frame" alt="MCR SOFA frame" draggable="false">
    <div id="error"></div>
    <div id="controls">
      <div class="control-row">
        <span class="control-label">仿真/相机</span>
        <button data-op="play">▶ 仿真</button>
        <button data-op="pause">⏸ 暂停</button>
        <button data-op="reset_episode">重置回合</button>
        <button data-op="reset_camera">重置视角</button>
        <button data-op="zoom_in">放大</button>
        <button data-op="zoom_out">缩小</button>
        <button data-op="left">左转</button>
        <button data-op="right">右转</button>
        <button data-op="up">上转</button>
        <button data-op="down">下转</button>
      </div>
      <div class="control-row">
        <span class="control-label">导管动作</span>
        <button data-action="rot_n_pos">N＋ (I)</button>
        <button data-action="rot_n_neg">N－ (K)</button>
        <button data-action="rot_b_pos">B＋ (J)</button>
        <button data-action="rot_b_neg">B－ (L)</button>
        <button data-action="insert">插入 (W)</button>
        <button data-action="retract">回撤 (S)</button>
        <button data-action="neutral">动作归零 (空格)</button>
      </div>
    </div>
    <div id="hint">先启动仿真；按住 I/K/J/L 调磁场，W/S 插入/回撤，空格归零</div>
  </div>
<script>
(() => {
  const token = new URLSearchParams(location.search).get('token') || '';
  const frame = document.getElementById('frame');
  const status = document.getElementById('status');
  const error = document.getElementById('error');
  const endpoint = (path) => `${path}?token=${encodeURIComponent(token)}`;
  let frameSeq = 0, dragging = false, button = 0, lastX = 0, lastY = 0, lastSend = 0;
  const heldKeys = new Set(), heldPointers = new Map();
  const actionVectors = {
    rot_n_pos: [1, 0, 0], rot_n_neg: [-1, 0, 0],
    rot_b_pos: [0, 1, 0], rot_b_neg: [0, -1, 0],
    insert: [0, 0, 1], retract: [0, 0, -1]
  };
  const keyActions = {
    KeyI: 'rot_n_pos', KeyK: 'rot_n_neg', KeyJ: 'rot_b_pos',
    KeyL: 'rot_b_neg', KeyW: 'insert', KeyS: 'retract'
  };

  function showError(message) { error.textContent = message; error.style.display = 'block'; }
  function clearError() { error.style.display = 'none'; }
  function refreshFrame() {
    const next = new Image();
    next.onload = () => { frame.src = next.src; clearError(); setTimeout(refreshFrame, 160); };
    next.onerror = () => { showError('画面连接暂时中断，正在重试…'); setTimeout(refreshFrame, 1000); };
    next.src = endpoint('/frame.png') + `&n=${frameSeq++}`;
  }
  async function command(op, extra = '') {
    try {
      const response = await fetch(endpoint('/api/control') + `&op=${op}${extra}`, {method:'POST'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    } catch (e) { showError(`控制失败：${e}`); }
  }
  function currentAction() {
    const value = [0, 0, 0];
    const names = [...heldKeys, ...heldPointers.values()];
    names.forEach(name => actionVectors[name].forEach((v, i) => value[i] += v));
    return value.map(v => Math.max(-1, Math.min(1, v)));
  }
  async function sendAction(action = currentAction(), keepalive = false) {
    const extra = `&n=${action[0]}&b=${action[1]}&insert=${action[2]}`;
    try {
      const response = await fetch(endpoint('/api/action') + extra,
        {method:'POST', keepalive});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    } catch (e) { if (!keepalive) showError(`导管控制失败：${e}`); }
  }
  function neutralize(keepalive = false) {
    heldKeys.clear(); heldPointers.clear();
    document.querySelectorAll('button[data-action]').forEach(b => b.classList.remove('active'));
    sendAction([0, 0, 0], keepalive);
  }
  async function refreshStatus() {
    try {
      const response = await fetch(endpoint('/api/status'), {cache:'no-store'});
      const data = await response.json();
      document.getElementById('title').textContent =
        `MCR SOFA Web Viewer x${data.sim_steps_per_frame}`;
      const episodeLabels = {
        paused: '手动暂停', running: '运行', success: '成功',
        out_of_vessel: '出血管', timeout: '超时',
        wrong_branch: '错误分支', non_finite: '数值异常', ended: '结束'
      };
      const episodeLabel = episodeLabels[data.episode_state] || data.episode_state;
      const distance = data.distance_to_goal_mm == null
        ? '--' : Number(data.distance_to_goal_mm).toFixed(2);
      const clearance = data.sdf_clearance_mm == null
        ? '--' : Number(data.sdf_clearance_mm).toFixed(2);
      const bodyClearance = data.sdf_body_clearance_mm == null
        ? '--' : Number(data.sdf_body_clearance_mm).toFixed(2);
      const inserted = data.sdf_inserted_length_mm == null
        ? '--' : Number(data.sdf_inserted_length_mm).toFixed(1);
      const tipWallSteps = Number(data.sdf_tip_near_wall_counter || 0);
      const routeGap = data.route_graph_gap_mm == null
        ? '--' : Number(data.route_graph_gap_mm).toFixed(2);
      const successQuality = data.episode_state === 'success'
        ? `　Tip安全成功 ${data.safe_success ? '是' : '否'}　Tip全程无穿壁 ${data.contact_free_success ? '是' : '否'}`
        : '';
      const requested = data.requested_action.map(v => Number(v).toFixed(1)).join(',');
      const applied = data.applied_action.map(v => Number(v).toFixed(2)).join(',');
      status.textContent = `模型 ${data.model}　${data.width}×${data.height}　` +
        `渲染 ${data.render_fps.toFixed(1)} FPS　回合 ${episodeLabel}　` +
        `终点距离 ${distance}mm　Tip净空 ${clearance}mm　Body最小净空 ${bodyClearance}mm　` +
        `Tip连续近壁 ${tipWallSteps}步　路径差 ${routeGap}mm　SDF检测段 ${inserted}mm${successQuality}　` +
        `训练等价控制　请求 [${requested}]　应用 [${applied}]　` +
        `有效插入 ${Number(data.effective_insert).toFixed(2)}　步数 ${data.steps}`;
    } catch (_) { status.textContent = '状态连接中断'; }
  }
  document.querySelectorAll('button[data-op]').forEach(b => b.onclick = () => command(b.dataset.op));
  document.querySelectorAll('button[data-action]').forEach(b => {
    b.addEventListener('contextmenu', e => e.preventDefault());
    b.addEventListener('pointerdown', e => {
      e.preventDefault();
      if (b.dataset.action === 'neutral') { neutralize(); return; }
      b.setPointerCapture(e.pointerId);
      heldPointers.set(e.pointerId, b.dataset.action); b.classList.add('active'); sendAction();
    });
    const release = e => {
      if (!heldPointers.has(e.pointerId)) return;
      heldPointers.delete(e.pointerId); b.classList.remove('active'); sendAction();
    };
    b.addEventListener('pointerup', release); b.addEventListener('pointercancel', release);
  });
  window.addEventListener('keydown', e => {
    if (e.code === 'Space') { e.preventDefault(); neutralize(); return; }
    const name = keyActions[e.code];
    if (!name || heldKeys.has(name)) return;
    e.preventDefault(); heldKeys.add(name); sendAction();
  });
  window.addEventListener('keyup', e => {
    const name = keyActions[e.code];
    if (!name) return;
    e.preventDefault(); heldKeys.delete(name); sendAction();
  });
  window.addEventListener('blur', () => neutralize(true));
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) neutralize(true);
  });
  frame.addEventListener('contextmenu', e => e.preventDefault());
  frame.addEventListener('pointerdown', e => {
    dragging = true; button = e.button; lastX = e.clientX; lastY = e.clientY;
    frame.classList.add('dragging'); frame.setPointerCapture(e.pointerId);
  });
  frame.addEventListener('pointerup', e => { dragging = false; frame.classList.remove('dragging'); });
  frame.addEventListener('pointermove', e => {
    if (!dragging) return;
    const now = performance.now(), dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (now - lastSend < 55) return;
    lastSend = now;
    command(button === 2 ? 'pan' : 'orbit', `&dx=${dx.toFixed(1)}&dy=${dy.toFixed(1)}`);
  });
  frame.addEventListener('wheel', e => {
    e.preventDefault(); command(e.deltaY < 0 ? 'zoom_in' : 'zoom_out');
  }, {passive:false});
  if (!token) showError('URL中缺少访问令牌 token');
  refreshFrame(); refreshStatus(); setInterval(refreshStatus, 1000);
})();
</script>
</body>
</html>"""


class ViewerState:
    def __init__(self, token: str, model: str, sim_steps_per_frame: int):
        self.token = token
        self.model = model
        self.sim_steps_per_frame = int(sim_steps_per_frame)
        self.frame = b""
        self.width = 0
        self.height = 0
        self.render_fps = 0.0
        self.playing = False
        self.episode_state = "paused"
        self.terminal_reason = "not_done"
        self.distance_to_goal_mm: Optional[float] = None
        self.sdf_clearance_mm: Optional[float] = None
        self.sdf_body_clearance_mm: Optional[float] = None
        self.sdf_inserted_length_mm: Optional[float] = None
        self.sdf_tip_near_wall_counter = 0
        self.route_graph_gap_mm: Optional[float] = None
        self.safe_success = False
        self.contact_free_success = False
        self.steps = 0
        self.requested_action = [0.0, 0.0, 0.0]
        self.applied_action = [0.0, 0.0, 0.0]
        self.effective_insert = 0.0
        self.error: Optional[str] = None
        self.lock = threading.Lock()
        self.commands: "queue.Queue[Tuple[str, float, float]]" = queue.Queue()

    def update_frame(self, frame: bytes, width: int, height: int, fps: float) -> None:
        with self.lock:
            self.frame = frame
            self.width = int(width)
            self.height = int(height)
            self.render_fps = float(fps)

    def set_playing(self, playing: bool) -> bool:
        with self.lock:
            terminal_states = {
                "success", "out_of_vessel", "wrong_branch", "timeout", "non_finite", "ended"
            }
            if playing and self.episode_state in terminal_states:
                return False
            self.playing = bool(playing)
            if self.playing:
                self.episode_state = "running"
            elif self.episode_state not in terminal_states:
                self.episode_state = "paused"
            if not self.playing:
                self.requested_action = [0.0, 0.0, 0.0]
            return True

    def is_playing(self) -> bool:
        with self.lock:
            return self.playing

    @staticmethod
    def _finite_optional(value, scale: float = 1.0) -> Optional[float]:
        try:
            value = float(value) * float(scale)
        except Exception:
            return None
        return value if math.isfinite(value) else None

    def record_step(self, info: dict, stopped: bool = False) -> None:
        with self.lock:
            self.steps += 1
            self.distance_to_goal_mm = self._finite_optional(
                info.get("current_dist_to_goal"), 1000.0
            )
            self.sdf_clearance_mm = self._finite_optional(
                info.get("sdf_tip_surface_clearance"), 1000.0
            )
            self.sdf_body_clearance_mm = self._finite_optional(
                info.get("sdf_body_min_surface_clearance"), 1000.0
            )
            self.sdf_inserted_length_mm = self._finite_optional(
                info.get("sdf_inserted_length"), 1000.0
            )
            self.sdf_tip_near_wall_counter = int(
                info.get("sdf_tip_near_wall_counter", 0)
            )
            self.route_graph_gap_mm = self._finite_optional(
                info.get("route_graph_distance_gap"), 1000.0
            )
            self.safe_success = bool(info.get("safe_success", False))
            self.contact_free_success = bool(
                info.get("contact_free_success", False)
            )
            if stopped:
                self.playing = False
                self.requested_action = [0.0, 0.0, 0.0]
                self.terminal_reason = str(info.get("terminal_reason", "other"))
                state_by_reason = {
                    "target": "success",
                    "out_of_vessel": "out_of_vessel",
                    "wrong_branch": "wrong_branch",
                    "timeout": "timeout",
                    "non_finite": "non_finite",
                }
                self.episode_state = state_by_reason.get(
                    self.terminal_reason, "ended"
                )
            else:
                self.episode_state = "running"

    def reset_episode(self) -> None:
        with self.lock:
            self.playing = False
            self.episode_state = "paused"
            self.terminal_reason = "not_done"
            self.distance_to_goal_mm = None
            self.sdf_clearance_mm = None
            self.sdf_body_clearance_mm = None
            self.sdf_inserted_length_mm = None
            self.sdf_tip_near_wall_counter = 0
            self.route_graph_gap_mm = None
            self.safe_success = False
            self.contact_free_success = False
            self.steps = 0
            self.requested_action = [0.0, 0.0, 0.0]
            self.applied_action = [0.0, 0.0, 0.0]
            self.effective_insert = 0.0

    def set_requested_action(self, action) -> None:
        with self.lock:
            self.requested_action = [float(value) for value in action]

    def get_requested_action(self) -> np.ndarray:
        with self.lock:
            return np.asarray(self.requested_action, dtype=np.float32)

    def update_applied_action(self, action, effective_insert: float) -> None:
        with self.lock:
            self.applied_action = [float(value) for value in action]
            self.effective_insert = float(effective_insert)

    def set_error(self, error: str) -> None:
        with self.lock:
            self.error = error

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return {
                "model": self.model,
                "sim_steps_per_frame": self.sim_steps_per_frame,
                "width": self.width,
                "height": self.height,
                "render_fps": self.render_fps,
                "playing": self.playing,
                "episode_state": self.episode_state,
                "terminal_reason": self.terminal_reason,
                "distance_to_goal_mm": self.distance_to_goal_mm,
                "sdf_clearance_mm": self.sdf_clearance_mm,
                "sdf_body_clearance_mm": self.sdf_body_clearance_mm,
                "sdf_inserted_length_mm": self.sdf_inserted_length_mm,
                "sdf_tip_near_wall_counter": self.sdf_tip_near_wall_counter,
                "route_graph_gap_mm": self.route_graph_gap_mm,
                "safe_success": self.safe_success,
                "contact_free_success": self.contact_free_success,
                "steps": self.steps,
                "control_mode": "training_equivalent",
                "requested_action": self.requested_action,
                "applied_action": self.applied_action,
                "effective_insert": self.effective_insert,
                "error": self.error,
            }


def make_handler(state: ViewerState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MCRWebViewer/1.0"

        def log_message(self, fmt: str, *args) -> None:
            # The token is part of the query string. Do not leak it through
            # BaseHTTPRequestHandler's default request-line access log.
            return

        def _query(self):
            return parse_qs(urlparse(self.path).query)

        def _authorized(self) -> bool:
            return secrets.compare_digest(self._query().get("token", [""])[0], state.token)

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if not self._authorized():
                self._send(HTTPStatus.FORBIDDEN, "text/plain; charset=utf-8", b"Forbidden\n")
                return
            if path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            elif path == "/frame.png":
                with state.lock:
                    frame = state.frame
                if not frame:
                    self._send(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain", b"Frame not ready\n")
                else:
                    self._send(HTTPStatus.OK, "image/png", frame)
            elif path == "/api/status":
                body = json.dumps(state.snapshot(), ensure_ascii=False).encode("utf-8")
                self._send(HTTPStatus.OK, "application/json; charset=utf-8", body)
            else:
                self._send(HTTPStatus.NOT_FOUND, "text/plain", b"Not found\n")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if not self._authorized():
                self._send(HTTPStatus.FORBIDDEN, "text/plain", b"Forbidden\n")
                return
            if path == "/api/action":
                query = self._query()
                try:
                    action = [
                        float(query.get("n", ["0"])[0]),
                        float(query.get("b", ["0"])[0]),
                        float(query.get("insert", ["0"])[0]),
                    ]
                except ValueError:
                    self._send(HTTPStatus.BAD_REQUEST, "text/plain", b"Bad action\n")
                    return
                if any(not math.isfinite(value) or abs(value) > 1.0 for value in action):
                    self._send(HTTPStatus.BAD_REQUEST, "text/plain", b"Action outside [-1, 1]\n")
                    return
                state.set_requested_action(action)
                self._send(HTTPStatus.OK, "application/json", b'{"ok":true}')
                return
            if path != "/api/control":
                self._send(HTTPStatus.NOT_FOUND, "text/plain", b"Not found\n")
                return
            query = self._query()
            op = query.get("op", [""])[0]
            try:
                dx = float(query.get("dx", ["0"])[0])
                dy = float(query.get("dy", ["0"])[0])
            except ValueError:
                self._send(HTTPStatus.BAD_REQUEST, "text/plain", b"Bad coordinates\n")
                return
            allowed = {
                "play", "pause", "reset_episode", "reset_camera", "zoom_in", "zoom_out",
                "left", "right", "up", "down", "orbit", "pan",
            }
            if op not in allowed:
                self._send(HTTPStatus.BAD_REQUEST, "text/plain", b"Bad operation\n")
                return
            state.commands.put((op, dx, dy))
            self._send(HTTPStatus.OK, "application/json", b'{"ok":true}')

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="B01", choices=ARTIFICIAL_MODELS)
    parser.add_argument(
        "--asset-root",
        default="",
        help=(
            "Optional vessel asset root. V01-V05 default to PROJECT_ROOT/mesh/valid; "
            "B/C models default to the existing train assets."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--sim-steps-per-frame",
        type=int,
        default=10,
        help=(
            "Training-equivalent environment steps before each rendered frame. "
            "This accelerates manual inspection without enlarging one action step."
        ),
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=float(os.environ.get("MCR_SOFA_DT", str(SOFA_TIME_STEP_S))),
        help="SOFA time step; defaults to the same 0.01 s used by train_sac.py.",
    )
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--settle-steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--target-threshold", type=float, default=TARGET_THRESHOLD_M)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument(
        "--vessel-scale-factor",
        type=float,
        default=1.0,
        help="Fixed shrink-only vessel scale for preflight testing (0.90 to 1.00).",
    )
    parser.add_argument("--vessel-alpha", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--access-token", default="")
    parser.add_argument("--start-playing", action="store_true")
    return parser.parse_args()


def apply_camera_command(env: MCREnv, state: ViewerState, command, initial_camera) -> None:
    op, dx, dy = command
    if op == "play":
        state.set_playing(True)
        return
    if op == "pause":
        state.set_playing(False)
        return
    if op == "reset_episode":
        env.reset()
        state.reset_episode()
        return
    if op == "reset_camera":
        env._set_camera_vector("position", initial_camera[0])
        env._set_camera_vector("lookAt", initial_camera[1])
        return
    if op in ("zoom_in", "zoom_out"):
        env._on_pyglet_mouse_scroll(0, 0, 0, 2 if op == "zoom_in" else -2)
        return
    if op in ("left", "right", "up", "down"):
        mapping = {
            "left": (-35.0, 0.0), "right": (35.0, 0.0),
            "up": (0.0, -35.0), "down": (0.0, 35.0),
        }
        dx, dy = mapping[op]
        op = "orbit"
    if op == "orbit":
        env._on_pyglet_mouse_drag(0, 0, dx, dy, env.pyglet.window.mouse.LEFT, 0)
    elif op == "pan":
        env._on_pyglet_mouse_drag(0, 0, dx, dy, env.pyglet.window.mouse.RIGHT, 0)


def main() -> int:
    args = parse_args()
    if args.width < 320 or args.height < 240:
        raise SystemExit("--width/--height must be at least 320x240")
    if not (0.2 <= args.fps <= 20.0):
        raise SystemExit("--fps must be between 0.2 and 20")
    if not (1 <= args.sim_steps_per_frame <= 100):
        raise SystemExit("--sim-steps-per-frame must be between 1 and 100")
    if args.time_step <= 0.0:
        raise SystemExit("--time-step must be positive")
    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be at least 1")
    if not (0.90 <= args.vessel_scale_factor <= 1.0):
        raise SystemExit("--vessel-scale-factor must be between 0.90 and 1.00")

    # The scene reads MCR_SOFA_DT when it creates the simulator. Keep that
    # value synchronized with MCREnv's time_step so the viewer and SAC
    # training execute the same physical step duration.
    os.environ["MCR_SOFA_DT"] = str(float(args.time_step))

    token = args.access_token.strip() or secrets.token_urlsafe(18)
    state = ViewerState(
        token=token,
        model=args.model,
        sim_steps_per_frame=int(args.sim_steps_per_frame),
    )
    state.set_playing(args.start_playing)

    asset_root = str(args.asset_root).strip()
    if not asset_root and str(args.model).upper().startswith("V"):
        asset_root = str(VALID_MESH_DIR)
    scene_kwargs = {
        "force_model": args.model,
        "radius_observation_scale": RADIUS_OBSERVATION_SCALE_M,
        "actor_history_steps": ACTOR_HISTORY_STEPS,
        "debug_rendering": True,
        "verbose_scene": True,
        "positioning_camera": True,
        "vessel_alpha": float(args.vessel_alpha),
        "randomize_start_target": False,
        "randomize_initial_orientation": False,
        "initial_orientation_max_angle_deg": INITIAL_ORIENTATION_MAX_ANGLE_DEG,
        "soft_randomize_single_vessel": True,
        "entry_tangent_points": ENTRY_TANGENT_POINTS,
        "vessel_scale_factor": float(args.vessel_scale_factor),
    }
    if asset_root:
        scene_kwargs["asset_root"] = asset_root

    env = MCREnv(
        image_shape=(int(args.height), int(args.width)),
        create_scene_kwargs=scene_kwargs,
        observation_type=ObservationType.STATE,
        action_type=ActionType.CONTINUOUS,
        time_step=float(args.time_step),
        frame_skip=int(args.frame_skip),
        settle_steps=int(args.settle_steps),
        render_mode=RenderMode.HEADLESS,
        render_framework=RenderFramework.PYGLET,
        env_type=EnvType.AORTIC,
        target_distance_threshold=float(args.target_threshold),
        max_episode_steps=int(args.max_episode_steps),
    )

    server = None
    server_thread = None
    try:
        print(f"[MCR WEB] Initializing SOFA model={args.model}...", flush=True)
        env.reset(seed=int(args.seed))
        initial_camera = (
            env._get_camera_vector("position").copy(),
            env._get_camera_vector("lookAt").copy(),
        )

        first_rgb = env._update_rgb_buffer()
        state.update_frame(
            encode_png(first_rgb), first_rgb.shape[1], first_rgb.shape[0], 0.0
        )

        server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(state))
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, name="mcr-web-http", daemon=True)
        server_thread.start()

        print("=" * 72, flush=True)
        print("MCR SOFA Web Viewer is ready", flush=True)
        print(
            "Training  : "
            f"dt={args.time_step} frame_skip={args.frame_skip} "
            f"settle_steps={args.settle_steps} "
            f"target_threshold={args.target_threshold} "
            f"max_episode_steps={args.max_episode_steps} "
            f"vessel_scale_factor={args.vessel_scale_factor} "
            f"sim_steps_per_frame={args.sim_steps_per_frame}",
            flush=True,
        )
        print(
            "Collision : vessel Triangle; catheter Line+Point "
            "(proximity values are reported in the SOFA log above)",
            flush=True,
        )
        print(f"Local URL : http://{args.host}:{args.port}/?token={token}", flush=True)
        print("Tunnel    : cloudflared tunnel --protocol http2 "
              f"--url http://{args.host}:{args.port}", flush=True)
        print(f"Open      : append /?token={token} to the trycloudflare URL", flush=True)
        print("Security  : do not share the URL or token; Ctrl+C stops the viewer", flush=True)
        print("=" * 72, flush=True)

        frame_period = 1.0 / float(args.fps)
        last_frame_time = time.monotonic()
        fps_ema = 0.0

        while True:
            loop_start = time.monotonic()
            while True:
                try:
                    command = state.commands.get_nowait()
                except queue.Empty:
                    break
                apply_camera_command(env, state, command, initial_camera)

            if state.is_playing():
                initial_requested_action = state.get_requested_action()
                active_manual_action = bool(
                    np.any(np.abs(initial_requested_action) > 1e-8)
                )
                steps_this_frame = (
                    int(args.sim_steps_per_frame) if active_manual_action else 1
                )
                for _ in range(steps_this_frame):
                    requested_action = state.get_requested_action()
                    _, _, terminated, truncated, info = env.step(requested_action)
                    state.update_applied_action(
                        getattr(env, "_last_smoothed_action", requested_action),
                        getattr(env, "current_effective_insert", requested_action[2]),
                    )
                    state.record_step(
                        info=info,
                        stopped=bool(terminated or truncated),
                    )
                    if terminated or truncated:
                        terminal_status = state.snapshot()
                        print(
                            "[MCR WEB EPISODE END]",
                            f"reason={info.get('terminal_reason', 'unknown')}",
                            f"steps={terminal_status['steps']}",
                            f"distance_mm={terminal_status['distance_to_goal_mm']}",
                            f"sdf_clearance_mm={terminal_status['sdf_clearance_mm']}",
                            f"sdf_body_clearance_mm={terminal_status['sdf_body_clearance_mm']}",
                            f"sdf_inserted_length_mm={terminal_status['sdf_inserted_length_mm']}",
                            f"safe_success={terminal_status['safe_success']}",
                            f"contact_free_success={terminal_status['contact_free_success']}",
                            f"route_graph_gap_mm={terminal_status['route_graph_gap_mm']}",
                            flush=True,
                        )
                        break
            else:
                env._update_rgb_buffer()

            rgb = env.render()
            png = encode_png(rgb)
            now = time.monotonic()
            instantaneous_fps = 1.0 / max(now - last_frame_time, 1e-6)
            fps_ema = instantaneous_fps if fps_ema <= 0.0 else 0.85 * fps_ema + 0.15 * instantaneous_fps
            last_frame_time = now
            state.update_frame(png, rgb.shape[1], rgb.shape[0], fps_ema)

            remaining = frame_period - (time.monotonic() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[MCR WEB] Stopping...", flush=True)
    except Exception as exc:
        state.set_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        # Initialization can fail before SofaEnv creates ``_sofa_root_node``.
        # Cleanup must never hide the original initialization traceback.
        if hasattr(env, "sofa_simulation") and hasattr(env, "_sofa_root_node"):
            try:
                env.close()
            except Exception as close_exc:
                print(
                    f"[MCR WEB] Cleanup warning: "
                    f"{type(close_exc).__name__}: {close_exc}",
                    file=sys.stderr,
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
