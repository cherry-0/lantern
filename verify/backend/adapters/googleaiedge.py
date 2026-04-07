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
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_DEFAULT_MAX_TOKENS = 512

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
        self._pipeline = None   # cached transformers pipeline
        self._transformers_ok: Optional[bool] = None

    # ── Availability ──────────────────────────────────────────────────────────

    def _check_transformers(self) -> Tuple[bool, str]:
        if self._transformers_ok is not None:
            return self._transformers_ok, ""
        try:
            import transformers  # noqa: F401
            self._transformers_ok = True
            return True, ""
        except ImportError:
            self._transformers_ok = False
            return False, "transformers not installed (pip install transformers accelerate torch)"

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            ok, msg = self._check_transformers()
            if not ok:
                return False, f"[NATIVE] {msg}"
            return (
                True,
                f"[NATIVE] transformers available. Will load {self._model_id} on first run "
                "(model is downloaded from HuggingFace if not cached locally).",
            )
        else:
            api_key = get_openrouter_api_key()
            if api_key and not api_key.startswith("your_"):
                return True, f"[SERVERLESS] Using OpenRouter to replicate LiteRT LLM output."
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

    def _get_pipeline(self):
        """Lazily load the transformers pipeline (downloads model on first call)."""
        if self._pipeline is not None:
            return self._pipeline

        from transformers import pipeline as hf_pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = hf_pipeline(
            "text-generation",
            model=self._model_id,
            device=device,
            torch_dtype=torch.float16 if device >= 0 else torch.float32,
            trust_remote_code=True,
        )
        return self._pipeline

    def _run_native(self, input_item: Dict[str, Any], modality: str) -> AdapterResult:
        """
        Run the same HuggingFace model family as the app's LiteRT allowlist via transformers.
        Image inputs are described in the prompt since most 1-2B text models are text-only.
        """
        ok, msg = self._check_transformers()
        if not ok:
            return AdapterResult(success=False, error=f"[NATIVE] {msg}")

        if modality == "image":
            # Describe the image as a text prompt (most small transformers models are text-only)
            data = input_item.get("data")
            path = input_item.get("path", "")
            try:
                image_b64 = input_item.get("image_base64") or _encode_image_b64(
                    data if data is not None else path
                )
                # Use OpenRouter vision briefly just to get an image description,
                # then feed that into the local model — mirrors how the app handles images
                image_description = self._call_openrouter(
                    "Describe this image in detail for an AI assistant to process.",
                    image_b64=image_b64,
                    max_tokens=256,
                )
            except Exception as e:
                return AdapterResult(success=False, error=f"Image encoding failed: {e}")
            user_message = (
                f"The user shared an image. Visual content: {image_description}\n\n"
                "Please analyze this and provide a helpful response."
            )
        else:
            data = input_item.get("data", "")
            user_message = str(data).strip() if data else ""
            if not user_message:
                return AdapterResult(success=False, error="Empty text input.")

        messages = [
            {"role": "system", "content": _CHAT_SYSTEM},
            {"role": "user",   "content": user_message},
        ]

        try:
            pipe = self._get_pipeline()
            outputs = pipe(
                messages,
                max_new_tokens=self._max_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=64,
                top_p=0.95,
            )
            response = outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            return AdapterResult(success=False, error=f"Local model inference failed: {e}")

        structured = {
            "input_modality": modality,
            "user_message": user_message[:500],
            "ai_response": response,
            "model_id": self._model_id,
        }

        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"user_message": user_message, "response": response},
            structured_output=structured,
            metadata={
                "method": "native_transformers",
                "model_id": self._model_id,
                "workflow": "local_multimodal_chat",
            },
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
            if image_b64 is None:
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

        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"user_message": user_message, "response": response},
            structured_output=structured,
            metadata={"method": "serverless", "workflow": "local_multimodal_chat"},
        )
