"""
Adapter for oxproxion — open-source Android LLM chat app.

Core pipeline (from source):
  ChatFragment → chatEditText (user types) → sendChatButton (tap) →
  Ktor HTTP POST to openrouter.ai/api/v1/chat/completions →
  response rendered in chatRecyclerView / messageTextView →
  on save: ChatRepository.insertSessionAndMessages() → Room (SQLite) DB.

NATIVE (USE_APP_SERVERS=true):
  BlackBoxAdapter path: boots the AVD, installs the APK built from
  target-apps/oxproxion/, drives the real Android app via uiautomator2,
  and observes all externalizations via:
    - mitmproxy  → NETWORK (the real Ktor HTTP call with private user text)
    - ADB FsObserver → STORAGE (Room DB writes to the app's private data dir)
    - ADB LogObserver → LOGGING (app logcat output)
  Resource IDs verified from fragment_chat.xml / item_message_ai.xml.

  AVD setup:
    - snapshot "clean" must have oxproxion installed and the OpenRouter API
      key already configured in Settings (the app stores it in SharedPreferences).
    - APK: build from source with `./gradlew assembleDebug` inside
      target-apps/oxproxion/, then place the resulting APK at
      target-apps/oxproxion/oxproxion.apk (or set apk_filename accordingly).

SERVERLESS (USE_APP_SERVERS=false):
  Direct OpenRouter call via BaseAdapter._call_openrouter(); no emulator needed.
  Externalizations from _build_serverless_externalizations() (3-tier priority).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import (
    AdapterResult,
    BaseAdapter,
    OPENROUTER_DEFAULT_MODEL,
)
from verify.backend.adapters.blackbox_base import BlackBoxAdapter, BlackBoxConfig
from verify.backend.observers import DEFAULT_LLM_HOSTS
from verify.backend.utils.config import get_openrouter_api_key, use_app_servers

# openrouter.ai is the primary LLM host for oxproxion (Ktor calls it directly).
# Adding it here ensures classify_phases() marks the inference POST request as
# DURING-phase, not as a post-inference externalization.
_OXPROXION_LLM_HOSTS = DEFAULT_LLM_HOSTS + (
    "openrouter.ai",
    "*.openrouter.ai",
)

_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_SYSTEM = (
    "You are a helpful AI assistant. "
    "Answer the user's questions clearly and concisely."
)
_APP_DIR = Path(__file__).resolve().parents[3] / "target-apps" / "oxproxion"


def _local_artifact_note() -> str:
    apk = _APP_DIR / "oxproxion.apk"
    if apk.exists():
        return f" Local APK found: {apk}."
    apkms = sorted(_APP_DIR.glob("*.apkm"))
    if apkms:
        return f" Local APKM found: {apkms[0]}."
    return (
        " No local APK/APKM artifact found under target-apps/oxproxion/."
        " Native runs therefore assume the app is already installed in the"
        " AVD snapshot, or you must build/copy oxproxion.apk first."
    )


class OxproxionAdapter(BlackBoxAdapter):
    """
    Wraps the oxproxion Android LLM chat pipeline.

    Inherits the full BlackBoxAdapter pipeline (EmulatorManager, mitmproxy,
    ADB observers, uiautomator2 driver) for native mode.  Adds a serverless
    fallback that calls OpenRouter directly when USE_APP_SERVERS=false.
    """

    name = "oxproxion"
    supported_modalities = ["text", "image"]

    config = BlackBoxConfig(
        package_name="io.github.stardomains3.oxproxion",
        main_activity="io.github.stardomains3.oxproxion.MainActivity",
        # Build with: cd target-apps/oxproxion && ./gradlew assembleDebug
        # Then copy app/build/outputs/apk/debug/app-debug.apk → oxproxion.apk
        apk_filename="oxproxion.apk",
        avd_name="verify_pixel7",
        snapshot_name="clean",
        llm_hosts=_OXPROXION_LLM_HOSTS,
        primary_backend_host="openrouter.ai",
        runtime_permissions=[],
        pinning_bypass="none",  # open-source app, no certificate pinning
        timeout_s=120,
    )

    # ── BaseAdapter interface overrides ──────────────────────────────────────

    def check_availability(self) -> Tuple[bool, str]:
        if not use_app_servers():
            api_key = get_openrouter_api_key()
            if api_key and not api_key.startswith("your_"):
                return True, "[SERVERLESS] Using OpenRouter chat fallback for oxproxion."
            return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."
        # Native: delegate to BlackBoxAdapter (checks AVD availability).
        ok, msg = super().check_availability()
        if not ok:
            return ok, msg
        return True, f"{msg}{_local_artifact_note()}"

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        modality = input_item.get("modality", "text")
        if modality not in self.supported_modalities:
            return AdapterResult(
                success=False,
                error=f"oxproxion does not support modality '{modality}'.",
            )
        if not use_app_servers():
            return self._run_serverless(input_item)
        if modality != "text":
            return AdapterResult(
                success=False,
                error=(
                    "oxproxion native automation currently implements the text chat flow only. "
                    "Image input is supported in serverless mode, but the black-box Android "
                    "driver does not yet automate attachmentButton/gallery upload."
                ),
            )
        # Native: full BlackBoxAdapter pipeline (boot → observe → drive → classify).
        return super().run_pipeline(input_item)

    # ── BlackBoxAdapter required override ────────────────────────────────────

    def _drive_app(self, driver: "AndroidDriver", input_item: Dict[str, Any]) -> str:
        """
        Drive the real oxproxion UI via uiautomator2.

        UI flow (resource IDs from fragment_chat.xml / item_message_ai.xml):
          1. Clear any existing chat (resetChatButton) so we start fresh.
          2. Tap chatEditText and type the user message.
          3. Tap sendChatButton to submit.
          4. Poll until the last messageTextView in chatRecyclerView stabilises
             (streaming responses update it in-place).
          5. Return the final response text.

        NOTE: Locators are derived from the layout XML but must be verified
        against a live dump_hierarchy on the provisioned AVD before use.
        Run:  adb shell uiautomator dump /sdcard/dump.xml && adb pull /sdcard/dump.xml
        to confirm resource IDs match the installed build.

        Known native failure mode:
          If the run fails with
            UiObjectNotFoundError: Selector [resourceId='...:id/chatEditText']
          the provisioned emulator is typically not on the expected chat
          screen, or the installed oxproxion build's runtime hierarchy does not
          match the resource IDs assumed from source/layout XML. In that case,
          re-verify the live hierarchy on the target AVD before trusting native
          IOC results for oxproxion.
        """
        import time

        pkg = self.config.package_name
        text = str(input_item.get("data", "")).strip()

        # 1. Clear previous conversation so we get a clean response.
        try:
            driver.tap({
                "resourceId": f"{pkg}:id/resetChatButton",
                "optional": True,
            })
            time.sleep(0.5)
        except Exception:
            pass  # not fatal — chat may already be empty

        # 2. Type the user message.
        driver.tap({"resourceId": f"{pkg}:id/chatEditText"})
        driver.type_into({"resourceId": f"{pkg}:id/chatEditText"}, text)

        # 3. Send.
        driver.tap({"resourceId": f"{pkg}:id/sendChatButton"})

        # 4. Wait for a non-empty AI response and poll until text stabilises
        #    (oxproxion supports streaming, so the bubble updates incrementally).
        ai_text_locator = {"resourceId": f"{pkg}:id/messageTextView"}
        prev = ""
        stable_count = 0
        deadline = time.time() + self.config.timeout_s
        while time.time() < deadline:
            try:
                current = driver.read_last(ai_text_locator)
            except Exception:
                current = ""
            if current and current == prev:
                stable_count += 1
                if stable_count >= 3:  # stable for ~1.5 s → streaming done
                    return current
            else:
                stable_count = 0
            prev = current
            time.sleep(0.5)

        # Timed out — return whatever we have.
        return prev

    # ── Serverless fallback ───────────────────────────────────────────────────

    def _run_serverless(self, input_item: Dict[str, Any]) -> AdapterResult:
        modality = input_item.get("modality", "text")
        text = str(input_item.get("data", "")).strip()
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        image_b64 = input_item.get("image_base64") if modality == "image" else None
        prompt = f"{_SYSTEM}\n\nUser: {text}"

        try:
            response = self._call_openrouter(
                prompt=prompt,
                image_b64=image_b64,
                max_tokens=1024,
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        # Post-inference externalizations mirror the real app:
        # NETWORK  — Ktor HTTP POST to OpenRouter (real call at tier-1)
        # STORAGE  — ChatRepository.insertSessionAndMessages() → Room/SQLite
        # UI       — response rendered in chatRecyclerView / messageTextView
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    f"[POST] {_OPENROUTER_ENDPOINT}"
                    f" (model={OPENROUTER_DEFAULT_MODEL}) → 200\n"
                    f"  ↳ Request body: messages=[system, user: {text}]\n"
                    f"  ↳ Response: {response}"
                ),
                "STORAGE": (
                    "[Room DB / SQLite] ChatSession inserted:"
                    " title=auto-generated, timestamp=now\n"
                    f"[Room DB / SQLite] ChatMessage(role=user): {text}\n"
                    f"[Room DB / SQLite] ChatMessage(role=assistant): {response}"
                ),
                "UI": (
                    "[DISPLAY_TEXT] Assistant response rendered in"
                    f" chatRecyclerView / messageTextView: {response}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"user_message": text, "ai_response": response},
            structured_output={"user_message": text, "ai_response": response},
            externalizations=externalizations,
            metadata={"method": "openrouter_serverless", "modality": modality},
        )
