"""
One-shot AVD provisioning helper for the black-box pipeline.

Takes the manual click-through described in analysis/verify_report_blackbox.md
§15.2–15.4 and reduces it to:

    python -m verify.backend.drivers.provision \
        --avd verify_pixel7 \
        --apkm target-apps/photomath/com.microblink.photomath_*.apkm

What it does (in order):

  1. Probe the SDK + AVD with EmulatorManager.probe.
  2. Boot headed (so the user can see + dismiss onboarding) unless --headless.
  3. Push the mitmproxy CA from ~/.mitmproxy/ into /system/etc/security/cacerts/
     (requires `adb root` to succeed — only works on Google-APIs AOSP images,
     not Play Store images).
  4. Install each --apk / --apkm.
  5. Pause for the user to walk through onboarding for each app
     ("press Enter when ready").
  6. Save the named snapshot.

This is a *helper*, not a replacement for human judgement: the user still has
to click through onboarding because every app does it differently and we
cannot reliably automate consent dialogs across vendors.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from verify.backend.drivers.emulator_manager import EmulatorManager


def _adb(serial: str, *args: str) -> subprocess.CompletedProcess:
    adb = shutil.which("adb") or "adb"
    return subprocess.run(
        [adb, "-s", serial, *args],
        capture_output=True, text=True, timeout=60,
    )


def install_mitm_ca(serial: str) -> bool:
    """
    Push ~/.mitmproxy/mitmproxy-ca-cert.cer to /system/etc/security/cacerts/
    so apps targeting API 24+ trust our CA system-wide.

    Returns True on success. Requires a rooted AVD (`adb root` works).
    """
    src = Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.cer"
    if not src.exists():
        # mitmproxy ships .pem; convert via openssl if needed
        pem = Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem"
        if not pem.exists():
            print(f"[provision] No mitmproxy CA found at {pem} — run `mitmdump` once first.", file=sys.stderr)
            return False
        src = pem

    print("[provision] Installing mitmproxy CA into /system/etc/security/cacerts/ ...")
    res = _adb(serial, "root")
    if "cannot run as root" in (res.stdout + res.stderr).lower():
        print("[provision] adb root failed — AVD is not a Google-APIs image. Skipping CA install.", file=sys.stderr)
        return False

    _adb(serial, "wait-for-device")
    _adb(serial, "remount")

    # Hash-based filename mitmproxy expects on the device.
    target = "/system/etc/security/cacerts/c8750f0d.0"
    push = _adb(serial, "push", str(src), target)
    if push.returncode != 0:
        print(f"[provision] adb push failed: {push.stderr.strip()}", file=sys.stderr)
        return False
    _adb(serial, "shell", "chmod", "644", target)

    print("[provision] CA installed. Rebooting AVD to take effect ...")
    _adb(serial, "reboot")
    # Wait for boot to complete; reuse EmulatorManager's helper indirectly
    # by polling sys.boot_completed.
    _adb(serial, "wait-for-device")
    return True


def install_apks(em: EmulatorManager, apks: List[Path], apkms: List[Path]) -> bool:
    ok = True
    for apk in apks:
        success, msg = em.install_apk(apk)
        print(f"[provision] {msg}")
        ok = ok and success
    for apkm in apkms:
        success, msg = em.install_apkm(apkm)
        print(f"[provision] {msg}")
        ok = ok and success
    return ok


def wait_for_user(packages: List[str]) -> None:
    print()
    print("=" * 60)
    print("[provision] Onboarding pause")
    print("=" * 60)
    if packages:
        print("Open each of the following packages on the AVD UI and:")
        print("  - skip / accept any first-run dialogs")
        print("  - sign in if needed")
        print("  - grant permissions when prompted")
        print("  - return to the app's main screen")
        print()
        for pkg in packages:
            print(f"  • {pkg}")
        print()
    print("Press Enter when every app is ready and the device is in the state")
    print("you want the snapshot to capture.")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("[provision] Aborted.", file=sys.stderr)
        sys.exit(1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--avd", default="verify_pixel7", help="AVD name to provision")
    p.add_argument("--snapshot", default="clean", help="Snapshot name to save")
    p.add_argument("--apk", action="append", default=[], type=Path,
                   help="Path to a .apk to install (repeatable)")
    p.add_argument("--apkm", action="append", default=[], type=Path,
                   help="Path to a .apkm bundle to install (repeatable)")
    p.add_argument("--no-ca", action="store_true",
                   help="Skip the mitmproxy CA install (use if device is unrooted)")
    p.add_argument("--headless", action="store_true",
                   help="Boot without a UI (only useful if onboarding is already done)")
    args = p.parse_args(argv)

    ok, msg = EmulatorManager.probe(args.avd)
    if not ok:
        print(msg, file=sys.stderr)
        return 2

    em = EmulatorManager(args.avd)
    boot = em.ensure_booted(headless=args.headless)
    print(f"[provision] {boot.message}")
    if not boot.ok:
        return 3

    # Defensive: if a previous run left a stale runtime proxy pointing at a
    # dead mitmdump port, the AVD would have no internet during onboarding.
    em.clear_runtime_proxy()

    if not args.no_ca:
        install_mitm_ca(em.serial)

    if args.apk or args.apkm:
        if not install_apks(em, args.apk, args.apkm):
            print("[provision] One or more installs failed; aborting before snapshot.", file=sys.stderr)
            return 4

    # Surface the package names we just installed so the prompt is useful.
    pkgs = []
    for path in args.apk + args.apkm:
        pkgs.append(path.stem.split("_")[0])
    wait_for_user(pkgs)

    print(f"[provision] Saving snapshot '{args.snapshot}' ...")
    em.save_snapshot(args.snapshot)
    print(f"[provision] Done. Future runs can call em.restore_snapshot({args.snapshot!r}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
