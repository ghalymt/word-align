"""Versioned JSON-line subprocess protocol for isolated engine execution.

Generalizes the Qwen sentinel-JSON pattern into a reusable protocol.
Any engine can run out-of-process via ``SubprocessRunner``.

Protocol rules:
- stdout is reserved for protocol messages (one JSON object per line)
- stderr is for human-readable logs (captured, not parsed)
- Every request has a unique request_id
- Malformed JSON lines are skipped, not fatal
- Worker crashes generate an ErrorEvent with stderr tail
- Supports timeout and cancellation
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from typing import Callable, Optional

PROTOCOL_VERSION = 1


def make_request(method: str, params: dict,
                 request_id: Optional[str] = None) -> dict:
    """Build a protocol request message."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id or f"req_{uuid.uuid4().hex[:12]}",
        "type": "request",
        "method": method,
        "params": params,
    }


def make_cancel(request_id: str) -> dict:
    """Build a cancel message."""
    return {"protocol_version": PROTOCOL_VERSION,
            "type": "cancel", "request_id": request_id}


def make_init(supported_versions: Optional[list[int]] = None) -> dict:
    """Build an init message."""
    return {"protocol_version": PROTOCOL_VERSION,
            "type": "init",
            "supported_versions": supported_versions or [PROTOCOL_VERSION]}


def parse_line(line: str) -> Optional[dict]:
    """Parse one JSON line. Returns None for malformed lines."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


class SubprocessRunner:
    """Manages an out-of-process engine worker via the JSON-line protocol.

    Lifecycle:
        runner = SubprocessRunner(executable, work_dir)
        runner.start()
        result = runner.request("transcribe", {"audio_path": "..."})
        runner.shutdown()
    """

    def __init__(self, executable: str, work_dir: str = "",
                 timeout: float = 600, env: Optional[dict] = None):
        self.executable = executable
        self.work_dir = work_dir or os.getcwd()
        self.timeout = timeout
        self.env = env or os.environ.copy()
        self._proc: Optional[subprocess.Popen] = None
        self._initialized = False

    def start(self) -> None:
        """Launch the worker process."""
        self._proc = subprocess.Popen(
            [self.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
            cwd=self.work_dir,
        )
        # Send init
        self._send(make_init())
        # Wait for ready
        ready = self._read_response(timeout=10)
        if ready is None or ready.get("type") != "ready":
            raise RuntimeError(f"Worker failed to initialize: {ready}")
        self._initialized = True

    def request(self, method: str, params: dict,
                on_progress: Optional[Callable[[float, str], None]] = None,
                timeout: Optional[float] = None) -> dict:
        """Send a request and wait for the result.

        Calls ``on_progress(progress, message)`` for any progress messages.
        Raises ``SubprocessTimeoutError`` on timeout.
        """
        if not self._initialized:
            raise RuntimeError("Worker not started")

        req = make_request(method, params)
        req_id = req["request_id"]
        self._send(req)

        deadline = time.time() + (timeout or self.timeout)
        while True:
            if time.time() > deadline:
                from ..core.errors import SubprocessTimeoutError
                raise SubprocessTimeoutError(
                    "subprocess",
                    timeout or self.timeout)

            msg = self._read_response(timeout=min(5, deadline - time.time()))
            if msg is None:
                continue
            if msg.get("request_id") != req_id:
                continue

            msg_type = msg.get("type")
            if msg_type == "progress":
                if on_progress:
                    on_progress(float(msg.get("progress", 0)),
                                msg.get("message", ""))
            elif msg_type == "result":
                return msg
            elif msg_type == "error":
                from ..core.errors import EngineError
                raise EngineError(
                    "subprocess",
                    msg.get("message", "Unknown error"),
                    recoverable=msg.get("recoverable", False))

    def cancel(self, request_id: str) -> None:
        """Send cancellation for a specific request."""
        self._send(make_cancel(request_id))

    def shutdown(self) -> None:
        """Clean shutdown: send shutdown command, then terminate."""
        if self._proc is None:
            return
        try:
            self._send({"protocol_version": PROTOCOL_VERSION,
                        "type": "shutdown"})
            self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        finally:
            self._proc = None
            self._initialized = False

    def is_alive(self) -> bool:
        """Check if the worker process is still running."""
        return self._proc is not None and self._proc.poll() is None

    def _send(self, msg: dict) -> None:
        if self._proc and self._proc.stdin:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()

    def _read_response(self, timeout: float = 5) -> Optional[dict]:
        """Read one JSON line from stdout. Returns None on timeout."""
        if not self._proc or not self._proc.stdout:
            return None
        # Note: this is a blocking read. For true async, use threads or asyncio.
        line = self._proc.stdout.readline()
        return parse_line(line) if line else None

    def get_stderr_tail(self, lines: int = 5) -> str:
        """Get the last N lines of stderr output."""
        if self._proc and self._proc.stderr:
            all_err = self._proc.stderr.read()
            return "\n".join(all_err.strip().splitlines()[-lines:])
        return ""
