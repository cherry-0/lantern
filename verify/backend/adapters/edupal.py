"""
Adapter for the edupal app.

Core pipeline (firebase_functions/functions/):
  Audio (m4a) → Whisper STT → transcript text
  → OpenAI GPT-3.5-turbo (character persona + conversation history) → bot response
  → ElevenLabs TTS → audio output (base64 mp3)

For Verify we target the text inference step — the GPT-3.5-turbo call in
generate_bot_response() — since that is where personal information in the
user's text is externalised to a third-party cloud LLM.

Externalizations per request:
  NETWORK  — POST to OpenAI API (api.openai.com):
             user transcript + last 10 conversation turns sent to gpt-3.5-turbo.
  STORAGE  — Firestore write: each message (user + bot) stored in
             messages collection with {message, side, session_id, character, timestamp}.
  NETWORK  — POST to ElevenLabs API (api.elevenlabs.io):
             bot response text sent for TTS synthesis.

The backend runs as Firebase Cloud Functions (Python) — there is no local
server to call, so there is no native HTTP mode.  Both modes replicate the
GPT-3.5-turbo generate_bot_response() call via OpenRouter.

Default character: Shiba Inu (as in the app's main demo screen).

Configuration (.env)
--------------------
OPENROUTER_API_KEY  — required for serverless (both) mode
"""

import sys
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_openrouter_api_key, use_app_servers

# Mirrors generate_bot_response() system prompt for the default character
_SYSTEM_PROMPT = (
    "you are a happy dog chatting with children, you want to be engaging and friendly. "
    "you want to keep the child interested and try to have a good conversation."
)


class EduPalAdapter(BaseAdapter):
    """
    Wraps the edupal AI character conversation pipeline.

    Replicates generate_bot_response() from speech_utils.py:
    user text + Shiba Inu system prompt → GPT-3.5-turbo → bot response.

    No native mode: the backend is Firebase Cloud Functions (cloud-only deployment).
    """

    name = "edupal"
    supported_modalities = ["text"]
    env_spec = None

    # ── Availability ──────────────────────────────────────────────────────────

    def check_availability(self) -> Tuple[bool, str]:
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            mode = "NATIVE (serverless equivalent)" if use_app_servers() else "SERVERLESS"
            return True, (
                f"[{mode}] Using OpenRouter to replicate GPT-3.5-turbo character response. "
                "edupal backend is Firebase Cloud Functions — no local server available."
            )
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"edupal only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        text = str(data).strip() if data else ""
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        return self._run_serverless(text)

    # ── SERVERLESS mode ───────────────────────────────────────────────────────

    def _run_serverless(self, text: str) -> AdapterResult:
        """Replicate generate_bot_response() via OpenRouter (openai/gpt-3.5-turbo)."""
        # Mirrors the prompt construction in speech_utils.py
        prompt = f"{_SYSTEM_PROMPT}\n\nUser: {text}"

        print(
            f"[edupal] Calling OpenRouter (gpt-3.5-turbo Shiba Inu)  text={text[:80]!r}",
            file=sys.stderr, flush=True,
        )

        try:
            response = self._call_openrouter(
                prompt=prompt,
                max_tokens=256,
                model="openai/gpt-3.5-turbo",
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        print(f"[edupal] Response: {response[:120]!r}", file=sys.stderr, flush=True)

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    f"[OpenAI API] POST https://api.openai.com/v1/chat/completions — "
                    f"model=gpt-3.5-turbo, system=Shiba Inu persona, "
                    f"user={text[:120]!r}"
                ),
                "STORAGE": (
                    f"[Firestore] messages collection write — "
                    f"{{message: {text[:80]!r}, side: 'user', character: 'Shiba Inu'}}; "
                    f"{{message: {response[:80]!r}, side: 'bot', character: 'Shiba Inu'}}"
                ),
                "NETWORK:TTS": (
                    f"[ElevenLabs API] POST https://api.elevenlabs.io/v1/text-to-speech/<voice_id> — "
                    f"model=eleven_multilingual_v2, text={response[:120]!r}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"text": text, "response": response},
            structured_output={"response": response},
            externalizations=externalizations,
            metadata={"method": "serverless_openrouter", "character": "Shiba Inu"},
        )
