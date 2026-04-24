"""
EmulatorManager — wraps the Android SDK CLIs (emulator, adb, avdmanager) so
BlackBoxAdapter can treat an Android Virtual Device as a first-class resource.

One AVD is shared across runs; state is reset between items via snapshot
restore rather than cold reboot. A proxy port is allocated once per boot and
the AVD is pointed at 10.0.2.2:<port> so host-side mitmproxy can MITM.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class BootResult:
    ok: bool
    serial: str
    message: str


class EmulatorManager:
    """
    Lifecycle manager for a single AVD. Not thread-safe; one manager per AVD
    per Verify process. Reuses a running emulator if one matches the AVD name.
    """

    DEFAULT_AVD = "verify_pixel7"
    BOOT_TIMEOUT_S = 180
    # AVD reaches the host via the magic IP 10.0.2.2 (see Android emulator docs).
    HOST_FROM_EMULATOR = "10.0.2.2"

    def __init__(self, avd_name: str = DEFAULT_AVD, proxy_port: Optional[int] = None):
        self.avd_name = avd_name
        self.proxy_port = proxy_port or self._pick_free_port()
        self.serial: Optional[str] = None
        self._emulator_proc: Optional[subprocess.Popen] = None

    # ── Probing (non-blocking) ────────────────────────────────────────────────

    @classmethod
    def probe(cls, avd_name: str = DEFAULT_AVD) -> Tuple[bool, str]:
        """Return (ok, message) without booting anything."""
        emu = cls._which("emulator")
        adb = cls._which("adb")
        if not emu:
            return False, "[BLACKBOX] 'emulator' not on PATH — install Android SDK + set ANDROID_HOME"
        if not adb:
            return False, "[BLACKBOX] 'adb' not on PATH — install Android platform-tools"

        try:
            out = subprocess.check_output([emu, "-list-avds"], text=True, timeout=10)
        except subprocess.SubprocessError as e:
            return False, f"[BLACKBOX] 'emulator -list-avds' failed: {e}"

        avds = [line.strip() for line in out.splitlines() if line.strip()]
        if avd_name not in avds:
            return False, (
                f"[BLACKBOX] AVD '{avd_name}' not found. Available: {avds or '(none)'}. "
                "Create with: avdmanager create avd -n {avd_name} -k 'system-images;android-34;"
                "google_apis;arm64-v8a' --device pixel_7"
            )
        return True, f"[BLACKBOX] AVD '{avd_name}' ready to boot."

    # ── Boot / shutdown ───────────────────────────────────────────────────────

    def ensure_booted(self, headless: bool = True) -> BootResult:
        """
        Boot the AVD if not already running. Reuses an existing running emulator
        (matched by avd name) if one is found. Returns BootResult(serial=...).

        Note: the boot command intentionally does NOT set `-http-proxy`. A
        boot-time proxy pointing at mitmdump would black-hole all traffic
        whenever mitmdump isn't running (provisioning, onboarding, snapshot
        save). Instead the proxy is toggled at runtime via set_runtime_proxy()
        during run_pipeline, and cleared after.
        """
        existing = self._find_running_serial()
        if existing:
            self.serial = existing
            return BootResult(True, existing, f"Attached to running emulator {existing}.")

        cmd: List[str] = [
            self._which("emulator") or "emulator",
            "-avd", self.avd_name,
            "-no-audio",
            "-no-snapshot-save",          # snapshots are managed explicitly via adb emu
        ]
        if headless:
            cmd += ["-no-window", "-no-boot-anim"]

        self._emulator_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        serial = self._wait_for_device(self.BOOT_TIMEOUT_S)
        if not serial:
            self.shutdown()
            return BootResult(False, "", f"Emulator did not come up within {self.BOOT_TIMEOUT_S}s.")

        self.serial = serial
        return BootResult(True, serial, f"Booted {self.avd_name} → {serial}.")

    def shutdown(self) -> None:
        if self.serial:
            self._adb("emu", "kill")
        if self._emulator_proc:
            try:
                self._emulator_proc.terminate()
                self._emulator_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._emulator_proc.kill()
        self.serial = None
        self._emulator_proc = None

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def save_snapshot(self, name: str) -> None:
        self._adb("emu", "avd", "snapshot", "save", name)

    def restore_snapshot(self, name: str) -> Tuple[bool, str]:
        """
        Restore a named snapshot. Returns (ok, message). If the snapshot does
        not exist, returns (False, reason) without aborting — caller can choose
        to proceed from the current AVD state.
        """
        res = self._adb_capture("emu", "avd", "snapshot", "load", name)
        if res.returncode != 0 or "failed" in (res.stdout + res.stderr).lower():
            return False, f"Snapshot '{name}' not restored: {res.stdout.strip() or res.stderr.strip()}"
        return True, f"Restored snapshot '{name}'."

    # ── APK install ───────────────────────────────────────────────────────────

    def install_apk(self, apk_path: Path | str) -> Tuple[bool, str]:
        """Install a single .apk file."""
        p = Path(apk_path)
        if not p.exists():
            return False, f"APK not found: {p}"
        res = self._adb_capture("install", "-r", "-t", str(p))
        if res.returncode != 0 or "Success" not in (res.stdout + res.stderr):
            return False, f"adb install failed: {res.stderr.strip() or res.stdout.strip()}"
        return True, f"Installed {p.name}."

    def _select_apkm_splits(
        self,
        apkm_path: Path | str,
        extract_dir: Optional[Path | str] = None,
        locales: Tuple[str, ...] = ("en",),
    ) -> Tuple[Optional[Path], List[Path], str]:
        """Return (extract_dir, selected_split_paths, message) for an APKM bundle."""
        p = Path(apkm_path)
        if not p.exists():
            return None, [], f"APKM not found: {p}"

        extract = Path(extract_dir) if extract_dir else p.parent / "_split"
        extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as z:
            z.extractall(extract)

        abi = self.get_prop("ro.product.cpu.abi") or "arm64-v8a"
        density_bucket = self._density_bucket(self.get_density())

        keep: List[Path] = []
        for f in extract.glob("*.apk"):
            n = f.name
            if n == "base.apk":
                keep.append(f)
            elif n.startswith("split_config."):
                tag = n[len("split_config."):-len(".apk")]
                if tag == abi.replace("-", "_") or tag == density_bucket:
                    keep.append(f)
                elif tag in locales:
                    keep.append(f)
            elif n.startswith("split_") and not n.startswith("split_config."):
                # Dynamic feature modules — always include.
                keep.append(f)

        if not keep or not any(k.name == "base.apk" for k in keep):
            return None, [], f"Could not find base.apk + matching splits in {extract}."

        return extract, keep, f"Selected splits from {p.name}."

    def install_split_apks(
        self,
        apks: List[Path | str],
        *,
        label: str = "split APK set",
    ) -> Tuple[bool, str]:
        """Install a pre-selected set of split APKs with `adb install-multiple`."""
        paths = [Path(p) for p in apks]
        if not paths:
            return False, f"No APKs provided for {label}."
        if not any(p.name == "base.apk" for p in paths):
            return False, f"{label} is missing base.apk."
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            return False, f"Missing APK(s) for {label}: {missing}"

        res = self._adb_capture("install-multiple", "-r", "-t", *[str(p) for p in paths])
        if res.returncode != 0 or "Success" not in (res.stdout + res.stderr):
            return False, (
                f"adb install-multiple failed: "
                f"{res.stderr.strip() or res.stdout.strip()}"
            )
        names = ", ".join(p.name for p in paths)
        return True, f"Installed {label} (splits: {names})."

    def install_apkm(
        self,
        apkm_path: Path | str,
        extract_dir: Optional[Path | str] = None,
        locales: Tuple[str, ...] = ("en",),
    ) -> Tuple[bool, str]:
        """
        Extract an APKMirror .apkm bundle and install only the splits matching
        this AVD's ABI and density, plus the requested locales and any dynamic
        feature splits.
        """
        _, keep, msg = self._select_apkm_splits(apkm_path, extract_dir=extract_dir, locales=locales)
        if not keep:
            return False, msg
        return self.install_split_apks(keep, label=Path(apkm_path).name)

    def uninstall(self, package: str) -> None:
        self._adb_capture("uninstall", package)

    # ── Runtime helpers ───────────────────────────────────────────────────────

    def get_prop(self, key: str) -> Optional[str]:
        res = self._adb_capture("shell", "getprop", key)
        out = (res.stdout or "").strip()
        return out or None

    def get_density(self) -> int:
        res = self._adb_capture("shell", "wm", "density")
        for tok in (res.stdout or "").split():
            if tok.isdigit():
                return int(tok)
        return 420  # xhdpi default for Pixel-7-class AVDs

    def grant_permissions(self, package: str, permissions: List[str]) -> None:
        for perm in permissions:
            self._adb_capture("shell", "pm", "grant", package, perm)

    def push_file(self, local: Path | str, remote: str) -> Tuple[bool, str]:
        res = self._adb_capture("push", str(local), remote)
        if res.returncode != 0:
            return False, f"adb push failed: {res.stderr.strip()}"
        return True, f"Pushed {local} → {remote}."

    # ── Proxy toggling (runtime) ──────────────────────────────────────────────

    # 10.0.2.2 is the Android emulator's alias for the host loopback, so an
    # app inside the AVD reaches a host-bound mitmdump at 10.0.2.2:<port>.
    HOST_LOOPBACK_ALIAS = "10.0.2.2"

    def set_runtime_proxy(self, port: Optional[int] = None) -> None:
        """Point the device's global HTTP proxy at host-side mitmdump."""
        p = port or self.proxy_port
        self._adb_capture(
            "shell", "settings", "put", "global",
            "http_proxy", f"{self.HOST_LOOPBACK_ALIAS}:{p}",
        )

    def clear_runtime_proxy(self) -> None:
        """Remove the device's global HTTP proxy (best-effort)."""
        # `put :0` is the documented way to disable; `delete` also works on most builds.
        self._adb_capture("shell", "settings", "put", "global", "http_proxy", ":0")
        self._adb_capture("shell", "settings", "delete", "global", "http_proxy")

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _which(binary: str) -> Optional[str]:
        found = shutil.which(binary)
        if found:
            return found
        # Fallback: ANDROID_HOME / ANDROID_SDK_ROOT layouts
        for env_var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
            root = os.environ.get(env_var)
            if not root:
                continue
            candidates = {
                "emulator": [Path(root) / "emulator" / "emulator"],
                "adb": [Path(root) / "platform-tools" / "adb"],
                "avdmanager": [Path(root) / "cmdline-tools" / "latest" / "bin" / "avdmanager"],
                "sdkmanager": [Path(root) / "cmdline-tools" / "latest" / "bin" / "sdkmanager"],
            }
            for cand in candidates.get(binary, []):
                if cand.exists():
                    return str(cand)
        return None

    @staticmethod
    def _pick_free_port(start: int = 8082) -> int:
        for port in range(start, start + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        raise RuntimeError("No free port found in 8082–8181 for mitmproxy.")

    @staticmethod
    def _density_bucket(density: int) -> str:
        # Matches split_config.<bucket> naming used by AABs.
        for bucket, lo in [
            ("xxxhdpi", 560),
            ("xxhdpi", 400),
            ("xhdpi", 280),
            ("hdpi", 200),
            ("mdpi", 140),
            ("ldpi", 0),
        ]:
            if density >= lo:
                return bucket
        return "xxhdpi"

    def _adb(self, *args: str) -> None:
        subprocess.check_call([self._which("adb") or "adb", "-s", self.serial or "", *args])

    def _adb_capture(self, *args: str) -> subprocess.CompletedProcess:
        adb = self._which("adb") or "adb"
        serial_args = ["-s", self.serial] if self.serial else []
        return subprocess.run(
            [adb, *serial_args, *args],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def _find_running_serial(self) -> Optional[str]:
        """Return the serial of a running emulator matching self.avd_name, if any."""
        adb = self._which("adb") or "adb"
        res = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=10)
        for line in res.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) == 2 and parts[1] == "device" and parts[0].startswith("emulator-"):
                serial = parts[0]
                # Match AVD name via emu command
                name_res = subprocess.run(
                    [adb, "-s", serial, "emu", "avd", "name"],
                    capture_output=True, text=True, timeout=5,
                )
                if name_res.returncode == 0 and self.avd_name in name_res.stdout:
                    return serial
        return None

    def _wait_for_device(self, timeout_s: int) -> Optional[str]:
        """Block until the emulator appears in adb devices and finishes booting."""
        adb = self._which("adb") or "adb"
        start = time.monotonic()

        # Step 1: serial appears
        serial: Optional[str] = None
        while time.monotonic() - start < timeout_s:
            res = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=10)
            for line in res.stdout.splitlines()[1:]:
                parts = line.split()
                if len(parts) == 2 and parts[1] == "device" and parts[0].startswith("emulator-"):
                    # Prefer the one whose AVD matches
                    cand = parts[0]
                    name_res = subprocess.run(
                        [adb, "-s", cand, "emu", "avd", "name"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if name_res.returncode == 0 and self.avd_name in name_res.stdout:
                        serial = cand
                        break
            if serial:
                break
            time.sleep(2)

        if not serial:
            return None

        # Step 2: boot completes
        while time.monotonic() - start < timeout_s:
            res = subprocess.run(
                [adb, "-s", serial, "shell", "getprop", "sys.boot_completed"],
                capture_output=True, text=True, timeout=10,
            )
            if res.stdout.strip() == "1":
                # Give package manager another moment
                time.sleep(3)
                return serial
            time.sleep(2)

        return None


if __name__ == "__main__":  # manual smoke test: python -m verify.backend.drivers.emulator_manager
    ok, msg = EmulatorManager.probe()
    print(msg, file=sys.stderr)
    sys.exit(0 if ok else 1)
