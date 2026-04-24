"""
FsObserver — captures filesystem side-effects by diffing a stat listing of
/data/data/<pkg> and /sdcard around the driven session.

Android has no fs_usage equivalent, so this is a sampling approach: snapshot
before the app runs, snapshot after, diff. Good enough to answer "did the app
write private data to a file it keeps around after inference?"

/data/data/<pkg> requires a rooted AVD; if not rooted we silently fall back to
/sdcard only (still catches media exports, download artifacts, shared storage).

See verify_report_blackbox.md §6.2.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple


# Filter rules — ignore cache/lock/tmp noise; keep data-looking extensions.
_NOISE_SUBSTRINGS = ("/cache/", "/code_cache/", "/tmp/")
_NOISE_SUFFIXES = (".lock", ".tmp", ".log", ".pid")
_INTERESTING_SUFFIXES = (
    ".json", ".db", ".sqlite", ".sqlite-journal", ".sqlite-shm", ".sqlite-wal",
    ".jpg", ".jpeg", ".png", ".webp", ".heic",
    ".pdf", ".txt", ".csv", ".xml", ".proto", ".bin",
)


class FsObserver:
    """
    Context manager. Captures file entries before/after the driven session;
    diff is emitted as STORAGE events in self.events on stop().
    """

    def __init__(self, serial: str, package: str):
        self.serial = serial
        self.package = package
        self.events: List[Dict[str, Any]] = []
        self._before: Dict[str, Tuple[int, int]] = {}  # path → (mtime, size)
        self._paths: List[str] = [f"/data/data/{package}", "/sdcard"]

    def __enter__(self) -> "FsObserver":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._before = self._snapshot()

    def stop(self) -> None:
        after = self._snapshot()
        ts = time.time()

        added: List[Dict[str, Any]] = []
        modified: List[Dict[str, Any]] = []
        for path, (mtime, size) in after.items():
            if self._is_noise(path):
                continue
            prev = self._before.get(path)
            if prev is None:
                added.append({"ts": ts, "kind": "added", "path": path, "size": size, "mtime": mtime})
            elif prev != (mtime, size):
                modified.append({
                    "ts": ts, "kind": "modified", "path": path,
                    "size": size, "mtime": mtime,
                    "prev_size": prev[1], "prev_mtime": prev[0],
                })

        # Prefer interesting extensions; cap total.
        ranked = sorted(added + modified, key=self._rank)
        self.events = ranked[:200]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _snapshot(self) -> Dict[str, Tuple[int, int]]:
        """Return {path: (mtime, size)} for tracked roots. Silent on failure."""
        out: Dict[str, Tuple[int, int]] = {}
        for root in self._paths:
            listing = self._list(root)
            for line in listing:
                parts = line.split("|", 2)
                if len(parts) != 3:
                    continue
                mtime_s, size_s, path = parts
                try:
                    out[path] = (int(mtime_s), int(size_s))
                except ValueError:
                    continue
        return out

    def _list(self, root: str) -> List[str]:
        """
        Stat every regular file under `root`. Uses `run-as <pkg>` for the
        app-private dir on user-build AVDs; falls back to plain `find` on root
        AVDs (and silently yields nothing if neither works).
        """
        # Plain find — works on /sdcard and on rooted AVDs for /data/data.
        plain = self._adb_shell(
            f'find "{root}" -type f -printf "%T@|%s|%p\\n" 2>/dev/null'
        )
        if plain.returncode == 0 and plain.stdout.strip():
            # %T@ is floating seconds; trim the fraction for a stable int mtime.
            return [self._normalize_ts_line(l) for l in plain.stdout.splitlines()]

        # Fallback for /data/data/<pkg> on non-root AVDs: run-as (debuggable apps).
        if root.startswith("/data/data/"):
            ra = self._adb_shell(
                f'run-as {self.package} find . -type f -printf "%T@|%s|%p\\n" 2>/dev/null'
            )
            if ra.returncode == 0:
                return [
                    self._normalize_ts_line(l.replace("./", f"{root}/", 1))
                    for l in ra.stdout.splitlines()
                ]
        return []

    @staticmethod
    def _normalize_ts_line(line: str) -> str:
        """Drop fractional seconds from %T@ so (mtime,size) tuples are stable."""
        head, _, rest = line.partition("|")
        int_ts = head.split(".", 1)[0]
        return f"{int_ts}|{rest}"

    def _adb_shell(self, cmd: str) -> subprocess.CompletedProcess:
        adb = shutil.which("adb") or "adb"
        return subprocess.run(
            [adb, "-s", self.serial, "shell", cmd],
            capture_output=True, text=True, timeout=60,
        )

    @staticmethod
    def _is_noise(path: str) -> bool:
        lower = path.lower()
        if any(sub in lower for sub in _NOISE_SUBSTRINGS):
            return True
        if any(lower.endswith(sfx) for sfx in _NOISE_SUFFIXES):
            return True
        return False

    @staticmethod
    def _rank(event: Dict[str, Any]) -> Tuple[int, str]:
        path = event.get("path", "")
        interesting = any(path.lower().endswith(s) for s in _INTERESTING_SUFFIXES)
        # Interesting first (0), then others (1); alphabetical within each band.
        return (0 if interesting else 1, path)
