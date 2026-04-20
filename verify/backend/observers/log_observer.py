"""
LogObserver — captures the target app's logcat output during the driven
session. Filters to the app's UID so we don't drown in system-wide noise,
caps at 200 records with WARN+ priority preserved first.

Parallels the root-logger handler that _runtime_capture.py installs for
open-source adapters.

See verify_report_blackbox.md §6.3.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional


_LOGCAT_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"(?P<pid>\d+)\s+(?P<tid>\d+)\s+(?P<level>[VDIWEF])\s+(?P<tag>[^:]+?)\s*:\s*(?P<msg>.*)$"
)
_LEVEL_RANK = {"F": 5, "E": 4, "W": 3, "I": 2, "D": 1, "V": 0}
_MAX_RECORDS = 200


class LogObserver:
    """
    Context manager. On start() spawns `adb logcat` filtered to the target
    package's UID; on stop() reads collected lines and parses them into
    LOGGING events.
    """

    def __init__(self, serial: str, package: str):
        self.serial = serial
        self.package = package
        self.events: List[Dict[str, Any]] = []
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._lines: List[str] = []

    def __enter__(self) -> "LogObserver":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        adb = shutil.which("adb") or "adb"
        uid = self._get_uid()
        cmd = [adb, "-s", self.serial, "logcat", "-v", "threadtime", "-T", "1"]
        if uid is not None:
            cmd += ["--uid", str(uid)]

        # Clear existing buffer so we only capture lines from this run.
        subprocess.run(
            [adb, "-s", self.serial, "logcat", "-c"],
            capture_output=True, timeout=10,
        )

        self._lines = []
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._reader:
            self._reader.join(timeout=5)

        self.events = self._parse_lines(self._lines)
        self._proc = None
        self._reader = None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _pump(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            self._lines.append(line.rstrip("\n"))

    def _get_uid(self) -> Optional[int]:
        """Look up the app's UID so logcat --uid filters to just this app."""
        adb = shutil.which("adb") or "adb"
        res = subprocess.run(
            [adb, "-s", self.serial, "shell",
             "cmd", "package", "list", "packages", "-U", self.package],
            capture_output=True, text=True, timeout=10,
        )
        # Output: "package:com.foo.bar uid:10234"
        m = re.search(r"uid:(\d+)", res.stdout or "")
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _parse_lines(lines: List[str]) -> List[Dict[str, Any]]:
        """Parse logcat lines; prefer WARN+ when truncating."""
        records: List[Dict[str, Any]] = []
        now = time.time()
        for line in lines:
            m = _LOGCAT_RE.match(line)
            if not m:
                continue
            records.append({
                "ts": now,  # logcat timestamps lack year; use wall-clock for ordering
                "level": m.group("level"),
                "tag": m.group("tag").strip(),
                "pid": int(m.group("pid")),
                "msg": m.group("msg"),
            })
        # Severity-priority truncate: keep WARN+ first, then fill with the rest
        # in arrival order. Matches _runtime_capture.py's logging-filter spirit.
        high = [r for r in records if _LEVEL_RANK.get(r["level"], 0) >= 3]
        low = [r for r in records if _LEVEL_RANK.get(r["level"], 0) < 3]
        picked = high[:_MAX_RECORDS]
        remaining = _MAX_RECORDS - len(picked)
        if remaining > 0:
            picked += low[:remaining]
        return picked
