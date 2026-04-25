"""
Adapter for PocketPal AI — on-device SLM chat app (React Native + llama.rn).

Core pipeline:
  user message → LocalCompletionEngine (LlamaContext from llama.rn) →
  llama.cpp GGUF model inference (fully on-device, no network) →
  response rendered in ChatScreen →
  conversation saved to WatermelonDB (local SQLite-based database).

The app uses llama.rn, a React Native binding for llama.cpp. The closest Python
equivalent is llama-cpp-python, which wraps the same llama.cpp C++ library.
Running the same GGUF model file via llama-cpp-python IS the honest native path —
same format, same engine, same on-device computation.

NATIVE (USE_APP_SERVERS=true):
  CondaRunner runs pocketpal_runner.py inside a 'pocketpal-ai' conda env.
  Requires: POCKETPAL_GGUF_MODEL_PATH pointing to a local .gguf model file.
  Post-inference: writes a local JSON chat store (simulating WatermelonDB save)
  captured as a STORAGE externalization.

  Optional network externalizations in the real app (none triggered automatically):
    - Supabase PalsHub sync (SUPABASE_URL + SUPABASE_ANON_KEY, user-triggered)
    - Firebase Functions: feedback submission, benchmark leaderboard (user-triggered)
    - HuggingFace: model download (one-time setup, not per-inference)

SERVERLESS (USE_APP_SERVERS=false):
  Direct OpenRouter call; externalizations from _build_serverless_externalizations().

Configuration (.env):
  POCKETPAL_GGUF_MODEL_PATH  — absolute path to .gguf model file (required for native)
  POCKETPAL_MAX_TOKENS       — max tokens to generate (default: 512)
  POCKETPAL_CTX_SIZE         — context window size (default: 4096)
  POCKETPAL_SYSTEM_PROMPT    — system message override (default: generic assistant)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verify.backend.adapters.base import (
    AdapterResult,
    BaseAdapter,
    OPENROUTER_DEFAULT_MODEL,
)
from verify.backend.utils.config import (
    TARGET_APPS_DIR,
    get_env,
    get_openrouter_api_key,
    use_app_servers,
)
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

_RUNNER = Path(__file__).parent.parent / "runners" / "pocketpal_runner.py"

# llama-cpp-python wraps the same llama.cpp C++ library as llama.rn.
# --prefer-binary skips slow compilation when a pre-built wheel is available.
_ENV_SPEC = EnvSpec(
    name="pocketpal-ai",
    python="3.10",
    install_cmds=[["pip", "install", "llama-cpp-python", "--prefer-binary"]],
)

_DEFAULT_SYSTEM = (
    "You are a helpful AI assistant running entirely on-device. "
    "Respond clearly and concisely to the user's questions."
)

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_CTX_SIZE = 4096


def _get_model_path() -> Optional[str]:
    return get_env("POCKETPAL_GGUF_MODEL_PATH")


def _get_max_tokens() -> int:
    val = get_env("POCKETPAL_MAX_TOKENS")
    try:
        return int(val) if val else _DEFAULT_MAX_TOKENS
    except ValueError:
        return _DEFAULT_MAX_TOKENS


def _get_ctx_size() -> int:
    val = get_env("POCKETPAL_CTX_SIZE")
    try:
        return int(val) if val else _DEFAULT_CTX_SIZE
    except ValueError:
        return _DEFAULT_CTX_SIZE


class PocketPalAdapter(BaseAdapter):
    """
    Wraps the PocketPal AI on-device SLM chat pipeline.

    Native mode uses llama-cpp-python with the same GGUF model file the app
    loads via llama.rn, reproducing the on-device inference faithfully.
    Post-inference, the app persists the conversation to WatermelonDB
    (a SQLite-based local database) — captured as a STORAGE externalization.
    """

    name = "pocketpal-ai"
    supported_modalities = ["text"]
    env_spec = _ENV_SPEC

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            model_path = _get_model_path()
            if not model_path:
                return (
                    False,
                    "[NATIVE] POCKETPAL_GGUF_MODEL_PATH not set. "
                    "Point it to a local .gguf model file "
                    "(e.g. Qwen2.5-1.5B-Instruct-Q4_K_M.gguf).",
                )
            if not Path(model_path).exists():
                return (
                    False,
                    f"[NATIVE] GGUF model file not found: {model_path}",
                )
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter chat fallback for pocketpal-ai."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error="pocketpal-ai only supports 'text' modality.",
            )
        if use_app_servers():
            return self._run_native(input_item)
        return self._run_serverless(input_item)

    # ── NATIVE ────────────────────────────────────────────────────────────────

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        model_path = _get_model_path()
        if not model_path or not Path(model_path).exists():
            return AdapterResult(
                success=False,
                error=f"GGUF model not found: {model_path}. Set POCKETPAL_GGUF_MODEL_PATH.",
            )

        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        text = str(input_item.get("data", "")).strip()
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        system_prompt = get_env("POCKETPAL_SYSTEM_PROMPT") or _DEFAULT_SYSTEM

        ok, result, err = CondaRunner.run(
            _ENV_SPEC.name,
            _RUNNER,
            {
                "text_content": text,
                "model_path": model_path,
                "system_prompt": system_prompt,
                "max_tokens": _get_max_tokens(),
                "ctx_size": _get_ctx_size(),
            },
            timeout=300,  # first run may need time for model load
        )
        if not ok:
            return AdapterResult(success=False, error=err)
        if not result.get("success"):
            return AdapterResult(success=False, error=result.get("error"))

        response = result.get("ai_response", "")
        tokens_predicted = result.get("tokens_predicted", 0)
        system_prompt = get_env("POCKETPAL_SYSTEM_PROMPT") or _DEFAULT_SYSTEM
        max_tokens = _get_max_tokens()
        ctx_size = _get_ctx_size()
        return AdapterResult(
            success=True,
            output_text=response,
            raw_output=result,
            structured_output={
                "user_message": text,
                "ai_response": response,
                "tokens_predicted": tokens_predicted,
                "model_path": model_path,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "ctx_size": ctx_size,
            },
            externalizations=result.get("externalizations", {}),
            metadata={
                "method": "native_llama_cpp",
                "model_path": model_path,
                "model_name": Path(model_path).name,
                "tokens_predicted": tokens_predicted,
                "max_tokens": max_tokens,
                "ctx_size": ctx_size,
            },
        )

    # ── SERVERLESS ────────────────────────────────────────────────────────────

    def _run_serverless(self, input_item: Dict[str, Any]) -> AdapterResult:
        text = str(input_item.get("data", "")).strip()
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        system_prompt = get_env("POCKETPAL_SYSTEM_PROMPT") or _DEFAULT_SYSTEM
        prompt = f"{system_prompt}\n\nUser: {text}"

        try:
            response = self._call_openrouter(prompt=prompt, max_tokens=512)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        # Post-inference externalizations mirror the real app:
        # STORAGE — WatermelonDB (SQLite-based) persists ChatMessage + ChatSession rows
        # No NETWORK — inference is fully local; no automatic network calls post-inference
        # (Firebase feedback and Supabase PalsHub sync are user-triggered, not automatic)
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "STORAGE": (
                    "[WatermelonDB / SQLite] ChatSession created: id=auto, createdAt=now\n"
                    f"[WatermelonDB / SQLite] ChatMessage(role=user): {text}\n"
                    f"[WatermelonDB / SQLite] ChatMessage(role=assistant): {response}"
                ),
                "UI": (
                    f"[STREAMING_TEXT] {response}\n"
                    f"[DISPLAY_TEXT] On-device response streamed to ChatScreen: {response}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"user_message": text, "ai_response": response},
            structured_output={
                "user_message": text,
                "ai_response": response,
                "system_prompt": system_prompt,
            },
            externalizations=externalizations,
            metadata={
                "method": "openrouter_serverless",
                "model": OPENROUTER_DEFAULT_MODEL,
            },
        )
