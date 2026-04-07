"""
Adapter for the google-ai-edge-gallery app.

Core pipeline: user text/image → LiteRT LLM (on-device) → AI response.

The app downloads quantized LLM models from HuggingFace (Gemma, Qwen2.5, Phi, etc.)
and runs them via the Android LiteRT LLM SDK.  The on-device format (.task files) is
Android-only and has no Python SDK.

Execution mode is controlled by USE_APP_SERVERS in .env:

  USE_APP_SERVERS=true  (NATIVE mode)
    Loads one of the same base models from HuggingFace via the `transformers` library
    and runs local inference — the closest Python equivalent to the on-device LiteRT LLM.
    The model is configurable via GOOGLE_AI_EDGE_MODEL_ID (HuggingFace model ID).
    Requires:  pip install transformers accelerate torch

    Suggested models (same families as the app's allowlist):
      Qwen/Qwen2.5-1.5B-Instruct   (small, ~3 GB)
      google/gemma-3-1b-it           (requires HuggingFace login)
      microsoft/phi-2                (small, ~5 GB)

  USE_APP_SERVERS=false  (SERVERLESS mode)
    Calls OpenRouter with the same system prompt the app uses for LlmChat tasks,
    producing structurally identical output without any local model loading.

Configuration (.env)
--------------------
USE_APP_SERVERS             — "true" / "false"                (default: false)
GOOGLE_AI_EDGE_MODEL_ID     — HuggingFace model ID for native mode
                              (default: Qwen/Qwen2.5-1.5B-Instruct)
GOOGLE_AI_EDGE_MAX_TOKENS   — max new tokens for native mode  (default: 512)
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers, TARGET_APPS_DIR
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

_DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_DEFAULT_MAX_TOKENS = 512

_ENV_SPEC = EnvSpec(
    name="google-ai-edge-gallery",
    python="3.10",
    install_cmds=[["pip", "install", "transformers", "accelerate", "torch"]],
)
_RUNNER = Path(__file__).parent.parent / "runners" / "googleaiedge_runner.py"

# System prompt matching the app's LlmChat capability persona
_CHAT_SYSTEM = (
    "You are a helpful AI assistant running locally on a mobile device via the Google AI Edge "
    "Gallery app. You have multimodal capabilities: you can process both text and images. "
    "Respond helpfully and concisely to the user's input. If an image is provided, analyze it "
    "and incorporate your visual understanding into the response."
)


def _encode_image_b64(data_or_path) -> str:
    from PIL import Image as PILImage
    try:
        if isinstance(data_or_path, (str, Path)):
            img = PILImage.open(str(data_or_path)).convert("RGB")
        else:
            img = data_or_path.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to encode image: {e}")


class GoogleAIEdgeAdapter(BaseAdapter):
    """
    Wraps the google-ai-edge-gallery local multimodal chat pipeline.

    NATIVE mode     : local HuggingFace model via transformers (same model family as app).
    SERVERLESS mode : OpenRouter with same system prompt as the app's LlmChat task.
    """

    name = "google-ai-edge-gallery"
    supported_modalities = ["image", "text"]

    def __init__(self):
        self._model_id: str = get_env("GOOGLE_AI_EDGE_MODEL_ID") or _DEFAULT_MODEL_ID
        self._max_tokens: int = int(get_env("GOOGLE_AI_EDGE_MAX_TOKENS") or _DEFAULT_MAX_TOKENS)

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter to replicate LiteRT LLM output."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        modality = input_item.get("modality", "text")
        if modality not in self.supported_modalities:
            return AdapterResult(
                success=False,
                error=f"google-ai-edge-gallery does not support modality '{modality}'.",
            )
        if use_app_servers():
            return self._run_native(input_item, modality)
        return self._run_serverless(input_item, modality)

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, input_item: Dict[str, Any], modality: str) -> AdapterResult:
        """Run HuggingFace transformers in the 'google-ai-edge-gallery' conda env."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        image_description = ""
        if modality == "image":
            data = input_item.get("data")
            path = input_item.get("path", "")
            try:
                image_b64 = input_item.get("image_base64") or _encode_image_b64(
                    data if data is not None else path
                )
                # Describe image using OpenRouter so the text-only local model can process it
                image_description = self._call_openrouter(
                    "Describe this image in detail for an AI assistant to process.",
                    image_b64=image_b64,
                    max_tokens=256,
                )
            except Exception as e:
                return AdapterResult(success=False, error=f"Image encoding failed: {e}")

        ok, result, err = CondaRunner.run(
            _ENV_SPEC.name,
            _RUNNER,
            {
                "modality": modality,
                "text_content": input_item.get("data", "") if modality == "text" else "",
                "image_description": image_description,
                "model_id": self._model_id,
                "max_tokens": self._max_tokens,
            },
            timeout=300,  # model download + first inference can be slow
        )
        if not ok:
            return AdapterResult(success=False, error=err)

        response = result.get("ai_response", "")
        externalizations = result.get("externalizations", {})

        structured = {
            "input_modality": modality,
            "ai_response": response,
            "model_id": result.get("model_id", self._model_id),
        }
        return AdapterResult(
            success=result.get("success", False),
            output_text=response,
            raw_output=result,
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "native_transformers", "model_id": self._model_id},
        )

    # ── SERVERLESS mode ───────────────────────────────────────────────────────

    def _run_serverless(self, input_item: Dict[str, Any], modality: str) -> AdapterResult:
        """
        Call OpenRouter with the same system prompt as the app's LlmChat task.
        Structurally identical output to native mode.
        """
        if modality == "image":
            data = input_item.get("data")
            path = input_item.get("path", "")
            image_b64 = input_item.get("image_base64")
            if not image_b64:
                try:
                    image_b64 = _encode_image_b64(data if data is not None else path)
                except Exception as e:
                    return AdapterResult(success=False, error=str(e))

            prompt = (
                f"{_CHAT_SYSTEM}\n\n"
                "The user has shared an image. Analyze it and provide a helpful, "
                "informative response describing what you observe and any insights you can offer."
            )
            try:
                response = self._call_openrouter(prompt, image_b64=image_b64, max_tokens=512)
            except RuntimeError as e:
                return AdapterResult(success=False, error=str(e))

            user_message = "[image input]"

        else:
            data = input_item.get("data", "")
            user_message = str(data).strip() if data else ""
            if not user_message:
                return AdapterResult(success=False, error="Empty text input.")

            prompt = f"{_CHAT_SYSTEM}\n\nUser: {user_message}"
            try:
                response = self._call_openrouter(prompt=prompt, max_tokens=512)
            except RuntimeError as e:
                return AdapterResult(success=False, error=str(e))

        structured = {
            "input_modality": modality,
            "user_message": user_message[:500],
            "ai_response": response,
        }

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "UI": f"Rendering LlmChatScreen with: {response[:100]}...",
                "ANALYTICS": "[Firebase Fallback] Log: CAPABILITY_CHAT_UI_GENERATION",
            }
        )

        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"user_message": user_message, "response": response},
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "serverless", "workflow": "local_multimodal_chat"},
        )
