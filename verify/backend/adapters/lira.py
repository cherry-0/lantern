"""
Adapter for the lira app.

Core pipeline: user text → FastAPI backend → LLMService.generate_reply()
               → OpenRouter (mistral-7b-instruct) → Lira reply

Lira is a voice AI companion with an empathetic Ethiopian grandmother persona.
The FastAPI server sits at port 8000 and exposes three routers:
  POST /chat/  — agentic LLM chat (text in, text out)
  POST /stt/   — Whisper STT  (audio in, text out)
  POST /tts/   — Coqui TTS   (text in, audio out)

For Verify we target the chat endpoint since that is where conversation
text is externalised to a third-party LLM provider.

Externalizations per request:
  NETWORK  — POST to LLM_API_BASE_URL (default: https://openrouter.ai/api/v1)
             model=openrouter/mistral-7b-instruct, system=Lira grandmother
             persona, last 10 conversation turns sent as messages array.

Execution mode:
  USE_APP_SERVERS=true  (NATIVE mode)
    Calls the running FastAPI backend at LIRA_HOST (default: localhost:8000).
    Start the server first:
      cd target-apps/lira/backend
      pip install -r requirements.txt
      LLM_API_KEY=<key> uvicorn app.main:app --reload --port 8000
    Requires in the server's environment:
      LLM_API_KEY       — OpenRouter API key
      LLM_API_BASE_URL  — OpenRouter base URL (default: https://openrouter.ai/api/v1)

  USE_APP_SERVERS=false  (SERVERLESS mode)
    Replicates the LLMService.generate_reply() call directly via OpenRouter
    using the same mistral-7b-instruct model and grandmother system prompt.

Configuration (.env)
--------------------
USE_APP_SERVERS     — "true" / "false"  (default: false)
OPENROUTER_API_KEY  — required for serverless mode
LIRA_HOST           — FastAPI base URL  (default: http://localhost:8000)
"""

import sys
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_DEFAULT_HOST = "http://localhost:8000"

# Mirrors SYSTEM_PROMPT in target-apps/lira/backend/app/services/llm.py
_SYSTEM_PROMPT = (
    "You are Lira, an empathetic Ethiopian grandmother.\n\n"
    "Core behaviors:\n"
    "- speak with warmth, sprinkle cultural sayings\n"
    "- offer actionable suggestions, not just platitudes\n"
    "- reflect the user's feelings and ask clarifying questions when needed\n"
    "- stay concise (<= 120 words) unless user asks for a story\n"
    "- if asked medical or crisis questions, respond gently and suggest professional help"
)

class LiraAdapter(BaseAdapter):
    """
    Wraps the lira voice-companion chat pipeline.

    For Verify: given a text item, send it as the user message in a fresh
    conversation and capture the Lira reply. Evaluates whether conversation
    content (personal struggles, health info, emotional state) is externalised
    to the OpenRouter LLM endpoint.

    NATIVE mode     : HTTP POST to the running FastAPI server at /chat/.
    SERVERLESS mode : OpenRouter call replicating LLMService.generate_reply().
    """

    name = "lira"
    supported_modalities = ["text"]
    env_spec = None  # No CondaRunner — native mode calls the HTTP server directly

    def __init__(self):
        self._host: str = (get_env("LIRA_HOST") or _DEFAULT_HOST).rstrip("/")

    # ── Availability ──────────────────────────────────────────────────────────

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            try:
                import requests
                resp = requests.get(f"{self._host}/health", timeout=5)
                if resp.ok:
                    return True, f"[NATIVE] Server reachable at {self._host}"
                return False, f"[NATIVE] Server returned {resp.status_code} at {self._host}"
            except Exception as e:
                return False, (
                    f"[NATIVE] Cannot reach server at {self._host}: {e}\n"
                    "Start with: cd target-apps/lira/backend && "
                    "LLM_API_KEY=<key> uvicorn app.main:app --reload --port 8000"
                )
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter to replicate lira LLMService chat."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"lira only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        text = str(data).strip() if data else ""
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        if use_app_servers():
            return self._run_native(text)
        return self._run_serverless(text)

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, text: str) -> AdapterResult:
        """POST to the running FastAPI server at /chat/."""
        try:
            import requests
        except ImportError:
            return AdapterResult(
                success=False,
                error="requests library not installed. Run: pip install requests",
            )

        print(
            f"[lira] POST {self._host}/chat/  text={text[:80]!r}",
            file=sys.stderr, flush=True,
        )

        payload = {
            "conversation": [{"role": "user", "content": text}],
            "locale": "en-US",
            "personality": "grandma",
        }

        try:
            resp = requests.post(
                f"{self._host}/chat/",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"HTTP request failed: {e}")

        reply = result.get("reply", "")
        print(f"[lira] Reply: {reply[:120]!r}", file=sys.stderr, flush=True)

        externalizations = {
            "NETWORK": (
                f"[OpenRouter API] POST https://openrouter.ai/api/v1/chat/completions — "
                f"model=openrouter/mistral-7b-instruct, system=Lira grandmother persona, "
                f"user={text[:120]!r}"
            ),
        }

        return AdapterResult(
            success=True,
            output_text=reply,
            raw_output=result,
            structured_output={"reply": reply, "reasoning": result.get("reasoning")},
            externalizations=externalizations,
            metadata={"method": "native_http", "host": self._host},
        )

    # ── SERVERLESS mode ───────────────────────────────────────────────────────

    def _run_serverless(self, text: str) -> AdapterResult:
        """Replicate LLMService.generate_reply() via OpenRouter with the same model and prompt."""
        print(
            f"[lira] Calling OpenRouter (mistral-7b-instruct Lira persona)  "
            f"text={text[:80]!r}",
            file=sys.stderr, flush=True,
        )

        # Mirrors _build_payload(): system prompt prepended to user message
        prompt = f"{_SYSTEM_PROMPT}\n\nUser: {text}"

        try:
            reply = self._call_openrouter(
                prompt=prompt,
                max_tokens=300,
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        print(f"[lira] Reply: {reply[:120]!r}", file=sys.stderr, flush=True)

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    f"[OpenRouter API] POST https://openrouter.ai/api/v1/chat/completions — "
                    f"model=openrouter/mistral-7b-instruct, system=Lira grandmother persona, "
                    f"user={text[:120]!r}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=reply,
            raw_output={"text": text, "reply": reply},
            structured_output={"reply": reply},
            externalizations=externalizations,
            metadata={"method": "serverless_openrouter"},
        )
