"""
Adapter for the snapdo app.

Core pipeline: image + task description → VLM verification → PASSED/FAILED + explanation.

Primary strategy: import VLMService from the snapdo backend.
Fallback: replicate the VLM call using OpenRouter directly (without Django context).
"""

import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key

SNAPDO_SERVER = TARGET_APPS_DIR / "snapdo" / "server"

# Default task description used when none is provided in input
DEFAULT_TASK = "Identify and describe all visible content in this image, including objects, text, people, and location cues."


def _encode_image_b64(data_or_path) -> str:
    """Encode a PIL Image or path to base64 JPEG."""
    import io

    try:
        from PIL import Image as PILImage

        if isinstance(data_or_path, (str, Path)):
            img = PILImage.open(str(data_or_path)).convert("RGB")
        else:
            img = data_or_path.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to encode image: {e}")


class SnapdoAdapter(BaseAdapter):
    """
    Wraps the snapdo image verification pipeline.

    For Verify: given an image, run the VLM to produce a rich description/analysis.
    The output text is then evaluated for privacy attribute inferability.
    """

    name = "snapdo"
    supported_modalities = ["image"]

    def __init__(self):
        self._native_available: Optional[bool] = None
        self._native_error: str = ""

    def _check_native(self) -> Tuple[bool, str]:
        if self._native_available is not None:
            return self._native_available, self._native_error

        snapdo_path = str(SNAPDO_SERVER)
        if snapdo_path not in sys.path:
            sys.path.insert(0, snapdo_path)

        try:
            import django
            import os

            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
            try:
                django.setup()
            except RuntimeError:
                pass

            from snapdo.services.vlm_service import VLMService  # noqa: F401

            self._native_available = True
            self._native_error = ""
        except Exception as e:
            self._native_available = False
            self._native_error = str(e)

        return self._native_available, self._native_error

    def check_availability(self) -> Tuple[bool, str]:
        native_ok, native_err = self._check_native()
        if native_ok:
            return True, "Native snapdo VLMService available."

        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return (
                True,
                f"Native pipeline unavailable ({native_err}); using OpenRouter vision fallback.",
            )

        return (
            False,
            f"snapdo native pipeline unavailable ({native_err}) and no valid OpenRouter API key.",
        )

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "image":
            return AdapterResult(
                success=False,
                error=f"snapdo only supports 'image' modality, got '{input_item.get('modality')}'.",
            )

        try:
            native_ok, _ = self._check_native()
            if native_ok:
                return self._run_native(input_item)
            else:
                return self._run_openrouter_fallback(input_item)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Use snapdo's VLMService directly."""
        from snapdo.services.vlm_service import VLMService

        data = input_item.get("data")
        path = input_item.get("path", "")
        image_b64 = input_item.get("image_base64") or _encode_image_b64(
            data if data is not None else path
        )

        task = input_item.get("task_description", DEFAULT_TASK)

        service = VLMService()
        raw = service.verify_evidence(image_b64, task)

        output_text = (
            f"Verdict: {raw.get('verdict', 'UNKNOWN')}\n"
            f"Confidence: {raw.get('confidence', 'N/A')}\n"
            f"Explanation: {raw.get('explanation', '')}"
        )

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output=raw,
            structured_output=raw,
            metadata={"method": "native_vlmservice"},
        )

    def _run_openrouter_fallback(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Replicate VLMService.verify_evidence using OpenRouter directly."""
        import requests

        api_key = get_openrouter_api_key()
        if not api_key or api_key.startswith("your_"):
            return AdapterResult(
                success=False,
                error="No valid OpenRouter API key for snapdo fallback.",
            )

        data = input_item.get("data")
        path = input_item.get("path", "")
        image_b64 = input_item.get("image_base64") or _encode_image_b64(
            data if data is not None else path
        )

        task = input_item.get("task_description", DEFAULT_TASK)

        prompt = (
            f"Please analyze the given image and tell me how it relates to the following task:\n"
            f"Task: {task}\n\n"
            "Respond in this exact format:\n"
            "Verdict: PASSED or FAILED\n"
            "Confidence: <0.0 to 1.0>\n"
            "Explanation: <your explanation>"
        )

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Verify",
                "X-Title": "Verify",
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"]

        # Parse the structured response
        verdict, confidence, explanation = "UNKNOWN", 0.5, response_text
        for line in response_text.splitlines():
            if line.lower().startswith("verdict:"):
                verdict = line.split(":", 1)[1].strip().upper()
            elif line.lower().startswith("confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.lower().startswith("explanation:"):
                explanation = line.split(":", 1)[1].strip()

        structured = {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
        }

        return AdapterResult(
            success=True,
            output_text=response_text,
            raw_output={"raw_response": response_text},
            structured_output=structured,
            metadata={"method": "openrouter_fallback"},
        )
