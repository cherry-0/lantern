"""
Adapter for the Waico (Wellbeing AI Companion) app.

Core pipeline: user wellbeing / mental health text → on-device Gemma 3n
(via MediaPipe LLM Inference) → counseling, fitness, or meditation response.

Waico is a Flutter/Dart Android app that runs all inference entirely on-device
using Gemma 3n + Mediapipe.  No data ever leaves the device.  Since native
Android automation is not wired here, both execution modes use the OpenRouter
serverless path replicating the Counselor agent's conversational workflow.
"""

import sys
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import AdapterResult, BaseAdapter, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_DEFAULT_MAX_TOKENS = 2048

_SYSTEM = (
    "You are Waico's on-device Counselor agent, a compassionate wellbeing AI companion. "
    "Your role is to provide empathetic emotional support and practical guidance "
    "grounded in evidence-based approaches (CBT, ACT, mindfulness). "
    "Listen actively, validate the user's feelings, and offer gentle, actionable suggestions. "
    "Always prioritise user safety: if there are signs of crisis, encourage professional help. "
    "Keep responses warm, concise, and non-judgmental."
)


class WaicoAdapter(BaseAdapter):
    """
    Replicates Waico's Counselor agent workflow via OpenRouter.

    Waico runs fully on-device (Gemma 3n + MediaPipe) and has no server
    endpoints; both USE_APP_SERVERS=true and false fall through to the
    serverless path.
    """

    name = "waico"
    supported_modalities = ["text"]
    env_spec = None

    def __init__(self):
        self._max_tokens = int(get_env("WAICO_MAX_TOKENS") or _DEFAULT_MAX_TOKENS)

    def check_availability(self) -> Tuple[bool, str]:
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            mode = "NATIVE (serverless equivalent)" if use_app_servers() else "SERVERLESS"
            return True, (
                f"[{mode}] Using OpenRouter to replicate Waico's on-device Gemma 3n "
                f"Counselor pipeline. Native Android/on-device inference is not wired."
            )
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"waico only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        text = str(data).strip() if data else ""
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        return self._run_serverless(text)

    def _run_serverless(self, text: str) -> AdapterResult:
        prompt = f"{_SYSTEM}\n\nUser: {text}"

        print(
            f"[waico] Calling OpenRouter ({OPENROUTER_DEFAULT_MODEL}) text={text[:80]!r}",
            file=sys.stderr,
            flush=True,
        )

        try:
            raw_response = self._call_openrouter(
                prompt=prompt,
                max_tokens=self._max_tokens,
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "UI": (
                    f"Waico Counselor screen streams response token by token: "
                    f"{raw_response!r}"
                ),
            }
        )
        # STORAGE always fires: ConversationProcessor saves to ObjectBox after every conversation
        externalizations["STORAGE"] = (
            f"[Waico / ObjectBox] ConversationRepository.save(): "
            f"summary=<AI-generated summary>, observations=<clinical notes>; "
            f"ConversationMemoryRepository.save(): episodic memories with Qwen3-Embedding-0.6B vectors; "
            f"UserRepository.updateUserInfo(): user profile updated from conversation content"
        )

        return AdapterResult(
            success=True,
            output_text=raw_response,
            raw_output={"input": text, "raw_response": raw_response},
            structured_output={"response": raw_response},
            externalizations=externalizations,
            metadata={
                "method": "serverless_openrouter",
                "workflow": "counselor_agent",
                "on_device_model": "gemma-3n",
            },
        )
