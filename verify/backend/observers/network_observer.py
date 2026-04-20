"""
NetworkObserver — spawns mitmdump and captures every HTTP/S flow as a JSON
record. Used by BlackBoxAdapter to record what a closed-source app sends to
its backend.

Usage:
    with NetworkObserver(proxy_port=8082) as net:
        # drive the app ...
    # net.events is now populated
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_ADDON = Path(__file__).parent / "_mitm_addon.py"


class NetworkObserver:
    def __init__(self, proxy_port: int, log_path: Optional[Path] = None):
        self.proxy_port = proxy_port
        self._log_path = Path(log_path) if log_path else Path(
            tempfile.gettempdir()) / f"verify_mitm_{proxy_port}.jsonl"
        self._proc: Optional[subprocess.Popen] = None
        self.events: List[Dict[str, Any]] = []

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "NetworkObserver":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ── Start / stop ──────────────────────────────────────────────────────────

    def start(self) -> None:
        mitmdump = shutil.which("mitmdump")
        if not mitmdump:
            raise RuntimeError(
                "mitmdump not found on PATH. Install with: pip install mitmproxy"
            )

        # Fresh log per run.
        if self._log_path.exists():
            self._log_path.unlink()

        env = dict(os.environ)
        env["MITM_FLOW_LOG"] = str(self._log_path)

        self._proc = subprocess.Popen(
            [
                mitmdump,
                "-p", str(self.proxy_port),
                "-s", str(_ADDON),
                "--set", "flow_detail=0",
                "--set", "console_eventlog_verbosity=error",
                "-q",                # quiet
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait briefly for the proxy to bind the port.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self._port_open(self.proxy_port):
                return
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() or b"").decode(errors="replace") \
                    if self._proc.stderr else ""
                raise RuntimeError(f"mitmdump exited prematurely: {err}")
            time.sleep(0.25)
        raise RuntimeError(f"mitmdump did not open port {self.proxy_port} within 10s.")

    def stop(self) -> None:
        if self._proc is None:
            return

        if self._proc.poll() is None:
            # Graceful SIGINT first, then SIGTERM.
            try:
                self._proc.send_signal(signal.SIGINT)
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

        self._proc = None
        self._read_events()

    # ── Event parsing ─────────────────────────────────────────────────────────

    def _read_events(self) -> None:
        self.events = []
        if not self._log_path.exists():
            return
        with self._log_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # Sort by timestamp so downstream phase classification works reliably.
        self.events.sort(key=lambda e: e.get("ts", 0))

    @property
    def flows(self) -> List[Dict[str, Any]]:
        """Alias for events, matches the doc's naming in verify_report_blackbox.md."""
        return self.events

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _port_open(port: int) -> bool:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":  # quick sanity check: starts, stops, prints 0 events
    import sys
    obs = NetworkObserver(proxy_port=8082)
    obs.start()
    print(f"mitmdump up on :{obs.proxy_port}", file=sys.stderr)
    time.sleep(1)
    obs.stop()
    print(f"{len(obs.events)} flows captured.", file=sys.stderr)
