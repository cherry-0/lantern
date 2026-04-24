"""
AndroidDriver — thin wrapper around uiautomator2 so BlackBoxAdapter._drive_app
can read/tap/type against the device without each adapter re-implementing u2
boilerplate.

Locators are plain dicts passed straight through to uiautomator2's selector
API, e.g.:
    {"resourceId": "com.foo.bar:id/send"}
    {"text": "Sign in"}
    {"className": "android.widget.EditText", "instance": 0}

See analysis/verify_report_blackbox.md §7.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class AndroidDriver:
    """
    Imperative UI driver for one installed package on one emulator serial.

    Heavy deps (uiautomator2) are imported lazily so the module stays importable
    on dev machines without the Android SDK — matches the pattern used by
    emulator_manager.
    """

    def __init__(self, serial: str, package: str, activity: str):
        self.serial = serial
        self.package = package
        self.activity = activity
        self._d = None  # populated on first use

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @property
    def d(self):
        if self._d is None:
            try:
                import uiautomator2 as u2
            except ImportError as e:
                raise RuntimeError(
                    "uiautomator2 not installed. Add it to verify/requirements.txt "
                    "or: pip install uiautomator2"
                ) from e
            self._d = u2.connect(self.serial)
        return self._d

    def launch(self) -> None:
        """Cold-start the app's main activity."""
        if self.activity:
            self.d.app_start(self.package, self.activity, stop=True)
        else:
            self.d.app_start(self.package, stop=True)

    def stop(self) -> None:
        """Force-stop the app so the next run starts clean."""
        try:
            self.d.app_stop(self.package)
        except Exception:
            # Best-effort; observer teardown should not fail on a dead app.
            pass

    # ── Interaction primitives ────────────────────────────────────────────────

    def tap(self, locator: Dict[str, Any], timeout: float = 10) -> None:
        """Click the first element matching `locator`. Waits up to `timeout`s."""
        el = self.d(**locator)
        el.wait(timeout=timeout)
        el.click()

    def type_into(self, locator: Dict[str, Any], text: str, timeout: float = 10) -> None:
        """Set the value of the first matching element to `text`."""
        el = self.d(**locator)
        el.wait(timeout=timeout)
        el.set_text(text)

    def read(self, locator: Dict[str, Any], timeout: float = 30) -> str:
        """Wait for an element, then return its .text (or empty string)."""
        el = self.d(**locator)
        if not el.wait(timeout=timeout):
            raise TimeoutError(f"Element not found within {timeout}s: {locator}")
        return (el.get_text() or "").strip()

    def wait_for(self, locator: Dict[str, Any], timeout: float = 30) -> bool:
        """Return True if `locator` resolves within `timeout`s, else False."""
        return bool(self.d(**locator).wait(timeout=timeout))

    def exists(self, locator: Dict[str, Any]) -> bool:
        """Non-blocking presence check."""
        return bool(self.d(**locator).exists)

    def screenshot(self, path: str | Path) -> None:
        """Save a PNG screenshot to `path`."""
        self.d.screenshot(str(path))

    # ── File transport ────────────────────────────────────────────────────────

    def push_image(self, local_path: str | Path, remote_path: str = "/sdcard/Download/verify_input.jpg") -> str:
        """
        Push a dataset image to the device and trigger MediaScanner so the app's
        gallery picker can see it. Returns the remote path.
        """
        adb = shutil.which("adb") or "adb"
        subprocess.run(
            [adb, "-s", self.serial, "push", str(local_path), remote_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        # ACTION_MEDIA_SCANNER_SCAN_FILE is deprecated on Android 10+ and
        # `cmd media scan` is unavailable on AOSP emulator images. Use
        # `content insert` to write directly into MediaStore — synchronous
        # and reliable on API 29+.
        device_path = remote_path if remote_path.startswith("/storage/emulated") \
            else remote_path.replace("/sdcard", "/storage/emulated/0")
        filename = remote_path.rsplit("/", 1)[-1]
        subprocess.run(
            [
                adb, "-s", self.serial, "shell", "content", "insert",
                "--uri", "content://media/external/images/media",
                "--bind", f"_display_name:s:{filename}",
                "--bind", f"_data:s:{device_path}",
                "--bind", "mime_type:s:image/jpeg",
                "--bind", "is_pending:i:0",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return remote_path

    def press_back(self) -> None:
        self.d.press("back")

    def press_home(self) -> None:
        self.d.press("home")

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def dump_hierarchy(self) -> str:
        """Return the current UI hierarchy as XML. Used by UiObserver."""
        return self.d.dump_hierarchy()
