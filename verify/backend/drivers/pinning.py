"""
Helpers for preparing MITM-/pinning-bypass-friendly install artifacts.

Phase 3 hardening starts with apk-mitm because it is the least invasive path
for apps that reject the system-installed mitmproxy CA. For split APK bundles
(`.apkm`), we patch `base.apk` and then reinstall it alongside the original
ABI/DPI/feature splits selected by EmulatorManager.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

from verify.backend.drivers.emulator_manager import EmulatorManager


def _apk_mitm_binary() -> str:
    binary = shutil.which("apk-mitm")
    if not binary:
        raise RuntimeError(
            "apk-mitm not found on PATH. Install it first, e.g. `npm install -g apk-mitm`."
        )
    return binary


def _build_tools_binary(name: str) -> str:
    sdk_root = Path.home() / "Library" / "Android" / "sdk" / "build-tools"
    candidates = sorted(sdk_root.glob(f"*/{name}"))
    if candidates:
        return str(candidates[-1])
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{name} not found in Android SDK build-tools or PATH.")


def patch_apk_with_apk_mitm(apk_path: Path | str, output_dir: Path | str) -> Tuple[Path, str]:
    """
    Run apk-mitm on a single APK and return the patched output path.

    apk-mitm writes `<stem>-patched.apk` next to the original by default; this
    helper moves/copies the result into `output_dir` so provisioning can rely on
    stable paths under the target-app directory.
    """
    apk = Path(apk_path)
    if not apk.exists():
        raise RuntimeError(f"APK not found: {apk}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = _apk_mitm_binary()
    before = {p.resolve() for p in apk.parent.glob("*-patched.apk")}
    res = subprocess.run(
        [binary, str(apk)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"apk-mitm failed: {res.stderr.strip() or res.stdout.strip()}")

    after = {p.resolve() for p in apk.parent.glob("*-patched.apk")}
    created = sorted(after - before)
    patched_src = created[-1] if created else apk.with_name(f"{apk.stem}-patched.apk")
    if not patched_src.exists():
        raise RuntimeError(
            f"apk-mitm reported success but no patched APK was found for {apk.name}."
        )

    patched_dst = out_dir / patched_src.name
    if patched_src.resolve() != patched_dst.resolve():
        shutil.copy2(patched_src, patched_dst)
    return patched_dst, res.stdout.strip() or res.stderr.strip()


def patch_apkm_base_with_apk_mitm(
    em: EmulatorManager,
    apkm_path: Path | str,
    *,
    extract_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Tuple[list[Path], str]:
    """
    Select the installable splits for an APKM, patch only `base.apk`, and
    return the split set ready for `install_split_apks()`.
    """
    extract, splits, msg = em._select_apkm_splits(apkm_path, extract_dir=extract_dir)
    if not splits or extract is None:
        raise RuntimeError(msg)

    base = next((p for p in splits if p.name == "base.apk"), None)
    if base is None:
        raise RuntimeError(f"No base.apk selected for {apkm_path}.")

    out_dir = Path(output_dir) if output_dir else extract / "_patched"
    patched_base, patch_msg = patch_apk_with_apk_mitm(base, out_dir)
    patched_splits = [patched_base if p.name == "base.apk" else p for p in splits]
    return patched_splits, patch_msg


def resign_apk_set(
    apks: Iterable[Path | str],
    *,
    output_dir: Path | str,
    keystore: Path | str | None = None,
    key_alias: str = "androiddebugkey",
    key_pass: str = "android",
    store_pass: str = "android",
) -> list[Path]:
    """
    Re-sign every APK in a split set with the same debug key so
    `adb install-multiple` accepts the bundle.
    """
    zipalign = _build_tools_binary("zipalign")
    apksigner = _build_tools_binary("apksigner")
    keystore_path = Path(keystore) if keystore else Path.home() / ".android" / "debug.keystore"
    if not keystore_path.exists():
        raise RuntimeError(f"Debug keystore not found: {keystore_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signed: list[Path] = []
    for apk in [Path(p) for p in apks]:
        if not apk.exists():
            raise RuntimeError(f"APK not found for signing: {apk}")
        aligned = out_dir / f"{apk.stem}-aligned.apk"
        signed_apk = out_dir / apk.name

        subprocess.run(
            [zipalign, "-f", "4", str(apk), str(aligned)],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        subprocess.run(
            [
                apksigner,
                "sign",
                "--ks", str(keystore_path),
                "--ks-key-alias", key_alias,
                "--ks-pass", f"pass:{store_pass}",
                "--key-pass", f"pass:{key_pass}",
                "--out", str(signed_apk),
                str(aligned),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        signed.append(signed_apk)
    return signed
