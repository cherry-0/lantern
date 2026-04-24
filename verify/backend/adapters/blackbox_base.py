"""
BlackBoxAdapter — base class for closed-source (binary-only) target apps.

Parallel to adapters that use CondaRunner to invoke open-source pipelines,
but drives a real Android APK inside an emulator and observes externalizations
from the host side (mitmproxy + adb). Subclasses provide UI-flow code and
per-app configuration only.

See analysis/verify_report_blackbox.md for architectural context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.drivers.emulator_manager import EmulatorManager
from verify.backend.observers import (
    NetworkObserver,
    FsObserver,
    LogObserver,
    UiObserver,
    classify_phases,
    DEFAULT_LLM_HOSTS,
)
from verify.backend.utils.config import use_app_servers


@dataclass
class BlackBoxConfig:
    """Per-adapter configuration. Subclasses override via class attributes."""
    package_name: str = ""
    main_activity: str = ""
    apk_filename: str = ""                 # single .apk, under target-apps/<name>/
    apkm_filename: str = ""                # .apkm bundle; one of apk/apkm required
    patched_apk_filename: str = ""         # optional locally patched .apk
    patched_apkm_base_filename: str = ""   # optional patched base.apk for split installs
    avd_name: str = "verify_pixel7"
    snapshot_name: str = "clean"
    llm_hosts: Tuple[str, ...] = DEFAULT_LLM_HOSTS
    primary_backend_host: Optional[str] = None
    runtime_permissions: List[str] = field(default_factory=list)
    pinning_bypass: str = "none"           # "none" | "apk_mitm" | "frida"
    timeout_s: int = 120


class BlackBoxAdapter(BaseAdapter):
    """
    Base class for black-box adapters. Concrete subclasses set `config` and
    override `_drive_app(driver, input_item)`.

    Subclasses inherit:
      - check_availability()  — validates AVD + SDK are present
      - run_pipeline()        — full boot/install/observe/drive/classify flow
    """

    config: BlackBoxConfig = BlackBoxConfig()

    # ── Required overrides ───────────────────────────────────────────────────

    def _drive_app(self, driver: "AndroidDriver", input_item: Dict[str, Any]) -> str:
        """
        Input the dataset item into the UI and return the app's response text.
        Called while observers are active; anything the app does network/disk/UI-wise
        during this call will be captured.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _drive_app()."
        )

    # ── BaseAdapter interface ────────────────────────────────────────────────

    def check_availability(self) -> Tuple[bool, str]:
        if not use_app_servers():
            return False, (
                "[BLACKBOX] Requires USE_APP_SERVERS=true and an Android emulator. "
                "Serverless fallback is not supported for closed-source adapters."
            )
        return EmulatorManager.probe(self.config.avd_name)

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") not in self.supported_modalities:
            return AdapterResult(
                success=False,
                error=f"Unsupported modality {input_item.get('modality')!r}.",
            )

        em = EmulatorManager(self.config.avd_name)
        boot = em.ensure_booted()
        if not boot.ok:
            return AdapterResult(success=False, error=boot.message)

        ok, msg = em.restore_snapshot(self.config.snapshot_name)
        # Not fatal — run continues from current state if snapshot missing.
        snapshot_note = msg

        if self.config.runtime_permissions:
            em.grant_permissions(self.config.package_name, self.config.runtime_permissions)

        response_text = ""
        network_events: List[Dict[str, Any]] = []
        storage_events: List[Dict[str, Any]] = []
        logging_events: List[Dict[str, Any]] = []
        ui_events: List[Dict[str, Any]] = []

        try:
            net = NetworkObserver(proxy_port=em.proxy_port)
            fs  = FsObserver(serial=em.serial, package=self.config.package_name)
            log = LogObserver(serial=em.serial, package=self.config.package_name)
            ui  = UiObserver(serial=em.serial)

            # Start mitmdump first, then point the device at it. The runtime
            # proxy is cleared in `finally` so leftover AVD state doesn't
            # black-hole traffic on the next run.
            with net, fs, log, ui:
                em.set_runtime_proxy(em.proxy_port)
                try:
                    from verify.backend.drivers.android_driver import AndroidDriver
                    driver = AndroidDriver(
                        serial=em.serial,
                        package=self.config.package_name,
                        activity=self.config.main_activity,
                    )
                    driver.launch()
                    response_text = self._drive_app(driver, input_item)
                    # Snapshot the final UI before tearing the app down.
                    ui.capture_post()
                    driver.stop()
                finally:
                    em.clear_runtime_proxy()

            network_events = net.flows
            storage_events = fs.events
            logging_events = log.events
            ui_events = ui.events
        except Exception as e:
            # Best-effort: make sure the device isn't left with a dead proxy.
            try:
                em.clear_runtime_proxy()
            except Exception:
                pass
            return AdapterResult(
                success=False,
                error=f"Black-box pipeline failed: {type(e).__name__}: {e}",
                metadata={"method": "blackbox_android", "snapshot": snapshot_note},
            )

        events = {
            "NETWORK": network_events,
            "STORAGE": storage_events,
            "LOGGING": logging_events,
            "UI": ui_events,
        }
        during, post = classify_phases(
            events,
            llm_hosts=self.config.llm_hosts,
            primary_backend_host=self.config.primary_backend_host,
        )

        return AdapterResult(
            success=True,
            output_text=response_text,
            externalizations=self._flatten_post(post),
            raw_output={
                "response_text": response_text,
                "during": during,
                "post": post,
            },
            metadata={
                "method": "blackbox_android",
                "avd": self.config.avd_name,
                "snapshot": snapshot_note,
                "n_network_events": len(network_events),
                "n_storage_events": len(storage_events),
                "n_logging_events": len(logging_events),
                "n_ui_events": len(ui_events),
            },
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_post(post: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """
        Convert POST-phase events into the flat {channel: str} format the
        evaluator consumes. Each channel is summarized line-by-line.
        """
        out: Dict[str, str] = {}

        net = post.get("NETWORK", [])
        if net:
            lines = []
            for e in net[:15]:
                lines.append(
                    f"[{e.get('method', '?')}] {e.get('url', '?')} → {e.get('status', '?')}"
                )
                body = (e.get("res_body") or "").strip()
                if body:
                    lines.append(f"  ↳ {body[:200]}")
            out["NETWORK"] = "\n".join(lines)

        storage = post.get("STORAGE", [])
        if storage:
            lines = [
                f"[{e.get('kind', '?')}] {e.get('path', '?')} ({e.get('size', '?')} bytes)"
                for e in storage[:20]
            ]
            out["STORAGE"] = "\n".join(lines)

        logging_ = post.get("LOGGING", [])
        if logging_:
            lines = [
                f"[{e.get('level', '?')}] {e.get('tag', '?')}: {e.get('msg', '')[:200]}"
                for e in logging_[:30]
            ]
            out["LOGGING"] = "\n".join(lines)

        ui = post.get("UI", [])
        if ui:
            lines = [
                f"[{e.get('kind', 'text')}] {e.get('value', '')[:200]}"
                for e in ui[:30]
            ]
            out["UI"] = "\n".join(lines)

        return out
