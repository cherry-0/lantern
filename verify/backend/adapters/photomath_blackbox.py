"""
Adapter for Photomath — first concrete closed-source (black-box) adapter.

Pipeline: push a math-problem image to /sdcard/DCIM → open Photomath → choose
"Import from gallery" → pick the image → read the OCR'd expression and the
step-by-step solution.

The locator dicts below are best-effort for Photomath 8.47.1 (see
target-apps/photomath/README.md). Real resource-ids are discovered at
snapshot-provisioning time with `uiautomator2 dump_hierarchy`; any that change
across Photomath updates are tracked per adapter rather than in the base class.
"""
from __future__ import annotations

from typing import Any, Dict

from verify.backend.adapters.blackbox_base import BlackBoxAdapter, BlackBoxConfig
from verify.backend.observers import DEFAULT_LLM_HOSTS


# Photomath is owned by Google and uses Google's inference backends. We add
# the math-intent service explicitly so phase_classifier has a high-signal
# fallback when no generic LLM host matches.
_PHOTOMATH_HOSTS = DEFAULT_LLM_HOSTS + (
    "*.photomath.net",
    "api.photomath.com",
)


class PhotomathAdapter(BlackBoxAdapter):
    name = "photomath"
    supported_modalities = ["image"]

    config = BlackBoxConfig(
        package_name="com.microblink.photomath",
        main_activity="",  # let uiautomator2 resolve the default launch activity
        apkm_filename="com.microblink.photomath_8.47.1-70001015_4arch_7dpi_f3942b2ec0cd28e15de4229bfd459de2_apkmirror.com.apkm",
        avd_name="verify_pixel7",
        snapshot_name="clean",
        llm_hosts=_PHOTOMATH_HOSTS,
        primary_backend_host="*.photomath.net",
        runtime_permissions=[
            "android.permission.READ_MEDIA_IMAGES",
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE",
        ],
        timeout_s=90,
    )

    def _drive_app(self, driver, input_item: Dict[str, Any]) -> str:
        import time

        src = input_item.get("path") or ""
        if not src:
            raise RuntimeError("photomath adapter requires input_item['path']")

        # push_image now uses `cmd media scan` (synchronous on API 29+) so no
        # sleep is needed before opening the picker.
        driver.push_image(src, remote_path="/sdcard/DCIM/verify_input.jpg")

        # Dismiss onboarding/camera-permission dialogs if any survived snapshot
        # restore. Best-effort — exists() is non-blocking.
        for locator in (
            {"text": "Continue"},
            {"text": "Allow"},
            {"text": "While using the app"},
            {"text": "Skip"},
        ):
            if driver.exists(locator):
                driver.tap(locator, timeout=2)

        # Open the gallery importer. In Photomath 8.47.1 this is an unlabeled
        # (NAF=true) ImageButton nested inside gallery_fragment_container, so
        # descriptionContains / text won't match — reach through the container.
        gallery = (
            driver.d(resourceId="com.microblink.photomath:id/gallery_fragment_container")
            .child(className="android.widget.ImageButton")
        )
        if gallery.wait(timeout=5):
            gallery.click()
        else:
            # Fallbacks for older/newer builds where the button is labeled or
            # lives outside that container.
            for locator in (
                {"descriptionContains": "gallery"},
                {"descriptionContains": "Gallery"},
                {"resourceIdMatches": r".*:id/gallery.*"},
            ):
                if driver.exists(locator):
                    driver.tap(locator)
                    break
            else:
                raise RuntimeError("Could not find Photomath's gallery import button.")

        # STALLED: Android 14 scoped photo-picker auto-dismisses within ~1s when
        # mitmproxy is active (proxy intercepts picker's Google Photos TLS request
        # → failure → auto-close). The thumbnail IS visible (confirmed via screencap)
        # but gone by the time the tap fires. Pending fix: clear proxy before gallery
        # open, or use VIEW intent to bypass picker, or save a post-picker snapshot.
        # See analysis/verify_report_blackbox.md §15.2 for options.
        import subprocess as _sp, shutil as _sh
        time.sleep(1)
        _adb = _sh.which("adb") or "adb"
        _sp.run([_adb, "-s", driver.serial, "shell", "input", "tap", "177", "825"],
                check=True, timeout=5)
        time.sleep(1)

        # After thumbnail selection Photomath shows a crop-adjustment screen.
        # Tap "Solve" to submit the image to the OCR+solver pipeline.
        if driver.wait_for({"resourceId": "com.microblink.photomath:id/button_solve"}, timeout=10):
            driver.tap({"resourceId": "com.microblink.photomath:id/button_solve"})

        # Wait for the solution card to appear.
        problem = ""
        solution = ""
        if driver.wait_for(
            {"resourceId": "com.microblink.photomath:id/solution_card_container"}, timeout=45
        ):
            try:
                problem = driver.read(
                    {"resourceId": "com.microblink.photomath:id/card_title"}, timeout=5
                )
            except Exception:
                pass
            try:
                solution = driver.read(
                    {"resourceId": "com.microblink.photomath:id/card_header"}, timeout=5
                )
            except Exception:
                pass

        if not (problem or solution):
            try:
                solution = driver.read({"className": "android.widget.TextView", "instance": 0}, timeout=5)
            except Exception:
                solution = ""

        parts = []
        if problem:
            parts.append(f"Problem: {problem}")
        if solution:
            parts.append(f"Solution type: {solution}")
        return "\n".join(parts) if parts else ""
