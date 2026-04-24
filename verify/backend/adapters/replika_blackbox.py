"""
Adapter stub for Replika — high-privacy AI companion chat app.

Modality: text (PrivacyLens-style items drop straight into the chat input).

Status: scaffold only. config is populated, but `_drive_app` raises
NotImplementedError until the locators are verified against a real
`uiautomator2.dump_hierarchy()` taken on the provisioned AVD. See
analysis/verify_report_blackbox.md §15.5 for the verification recipe.

Per the README in target-apps/replika/, the UI flow is:
  cold launch → splash → "Continue as <name>" (snapshot must contain a saved
  session) → main chat tab → text input → send → wait for response bubble →
  read last TextView in the message list.
"""
from __future__ import annotations

from typing import Any, Dict

from verify.backend.adapters.blackbox_base import BlackBoxAdapter, BlackBoxConfig
from verify.backend.observers import DEFAULT_LLM_HOSTS


# Replika fronts a custom backend that proxies inference internally; calls
# rarely hit a public LLM host directly. primary_backend_host is the fallback
# the phase classifier uses when no LLM-host pattern matches.
_REPLIKA_HOSTS = DEFAULT_LLM_HOSTS + (
    "*.replika.ai",
    "api.replika.ai",
)


class ReplikaAdapter(BlackBoxAdapter):
    name = "replika"
    supported_modalities = ["text"]

    config = BlackBoxConfig(
        package_name="ai.replika.app",
        main_activity="",  # let uiautomator2 resolve the launch activity
        apkm_filename="ai.replika.app_12.7.2-6340_2arch_7dpi_1feat_dbc5e8fca44b9a6183d124106f690f32_apkmirror.com.apkm",
        avd_name="verify_pixel7",
        snapshot_name="clean",
        llm_hosts=_REPLIKA_HOSTS,
        primary_backend_host="*.replika.ai",
        runtime_permissions=[],
        pinning_bypass="apk_mitm",
        timeout_s=120,
    )

    def _drive_app(self, driver, input_item: Dict[str, Any]) -> str:
        # TODO(blackbox-phase2): verify locators against a real dump_hierarchy.
        #
        # Sketch of the expected flow once locators are confirmed:
        #   text = input_item.get("data") or input_item.get("text") or ""
        #   driver.tap({"resourceIdMatches": r".*:id/chat_input.*"})
        #   driver.type_into({"resourceIdMatches": r".*:id/chat_input.*"}, text)
        #   driver.tap({"resourceIdMatches": r".*:id/send.*"})
        #   # Streaming UI: poll the last message bubble until it stops growing
        #   return driver.read({"resourceIdMatches": r".*:id/message_text.*"}, timeout=60)
        raise NotImplementedError(
            "ReplikaAdapter._drive_app: locators not yet verified. "
            "See analysis/verify_report_blackbox.md §15.5."
        )
