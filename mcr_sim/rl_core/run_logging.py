"""Persistent per-run console capture for headless distributed training."""

from __future__ import annotations

import atexit
import json
import sys
import threading
from pathlib import Path
from typing import TextIO


class _TeeStream:
    """Mirror a text stream to its original destination and a run log file."""

    def __init__(self, original: TextIO, log_stream: TextIO, lock: threading.Lock):
        self._original = original
        self._log_stream = log_stream
        self._lock = lock

    def write(self, text):
        with self._lock:
            original_result = self._original.write(text)
            self._log_stream.write(text)
        return original_result

    def flush(self):
        with self._lock:
            self._original.flush()
            self._log_stream.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


class RunLogCapture:
    """Capture one distributed rank's Python stdout/stderr under ``logs/``."""

    def __init__(self, log_dir: Path, rank: int):
        self.log_dir = Path(log_dir)
        self.rank = int(rank)
        self.path = self.log_dir / (
            "console.log" if self.rank == 0 else f"console_rank_{self.rank}.log"
        )
        self._original_stdout = None
        self._original_stderr = None
        self._stream = None

    def start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        lock = threading.Lock()
        sys.stdout = _TeeStream(self._original_stdout, self._stream, lock)
        sys.stderr = _TeeStream(self._original_stderr, self._stream, lock)
        # Keep capture active through an uncaught exception: Python writes the
        # traceback before running atexit handlers.
        atexit.register(self.close)
        print(f"[RUN_LOG] rank={self.rank} path={self.path}", flush=True)
        return self

    def close(self):
        if self._stream is None:
            return
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._stream.close()
            self._stream = None


def start_run_log_capture(log_dir: Path, rank: int) -> RunLogCapture:
    return RunLogCapture(log_dir=log_dir, rank=rank).start()


def write_run_config(path: Path, args, **extra) -> None:
    """Persist the effective CLI values used by one training run."""

    def _json_value(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _json_value(item) for key, item in value.items()}
        return str(value)

    payload = {key: _json_value(value) for key, value in vars(args).items()}
    payload.update({key: _json_value(value) for key, value in extra.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
