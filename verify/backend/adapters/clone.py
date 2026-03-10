"""
Adapter for the clone app.

Core pipeline: video → sampled frames → description/summary via LLM.

The clone app's primary inference runs in a TypeScript Electron process (ONNX CLIP/DRAGON
encoders). Direct Python import is not feasible. This adapter implements a standalone
OpenRouter-based fallback that replicates the spirit of the pipeline:
frames → vision model → descriptive summary.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_openrouter_api_key


def _encode_pil_b64(pil_image) -> str:
    """Encode a PIL Image to base64 JPEG string."""
    import io

    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class CloneAdapter(BaseAdapter):
    """
    Wraps the clone video → description pipeline.

    Since the original app uses a TypeScript Electron process with ONNX models,
    this adapter uses OpenRouter vision model as a standalone fallback.

    For Verify purposes: given sampled video frames, produce a descriptive summary
    that the privacy evaluator can then assess for attribute inferability.
    """

    name = "clone"
    supported_modalities = ["image", "video"]

    def check_availability(self) -> Tuple[bool, str]:  # type: ignore[override]
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return (
                True,
                "Using OpenRouter vision fallback (clone's native TypeScript/Electron pipeline "
                "is not directly importable from Python).",
            )
        return (
            False,
            "clone adapter requires OpenRouter API key. "
            "Native pipeline (TypeScript Electron + ONNX) is not directly callable from Python.",
        )

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        modality = input_item.get("modality", "image")
        if modality not in self.supported_modalities:
            return AdapterResult(
                success=False,
                error=f"clone adapter does not support modality '{modality}'.",
            )

        api_key = get_openrouter_api_key()
        if not api_key or api_key.startswith("your_"):
            return AdapterResult(
                success=False,
                error="No valid OpenRouter API key for clone adapter.",
            )

        try:
            if modality == "video":
                frames = input_item.get("frames", [])
                if not frames:
                    return AdapterResult(success=False, error="No video frames provided.")
                return self._run_on_frames(frames, api_key)
            else:
                # Image modality: treat single image as a "frame"
                data = input_item.get("data")
                if data is None:
                    from PIL import Image as PILImage

                    data = PILImage.open(input_item["path"]).convert("RGB")
                return self._run_on_frames([data], api_key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def _run_on_frames(self, frames: List[Any], api_key: str) -> AdapterResult:
        """Send sampled frames to OpenRouter and get a descriptive summary."""
        import requests

        # Encode all frames
        encoded_frames = [_encode_pil_b64(f) for f in frames]

        content = [
            {
                "type": "text",
                "text": (
                    "You are an AI personal assistant analyzing screen activity recordings "
                    "(like the clone app). You receive sampled frames from a video recording. "
                    "Describe:\n"
                    "1. What activity/scene is shown.\n"
                    "2. Any visible text, applications, or identifiable elements.\n"
                    "3. A concise summary suitable for a personal knowledge base.\n\n"
                    "Format:\n"
                    "Activity: <description>\n"
                    "Details: <visible elements>\n"
                    "Summary: <1-2 sentence summary>"
                ),
            }
        ]

        for i, b64 in enumerate(encoded_frames):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
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
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"]

        # Parse structured fields
        activity, details, summary = "", "", ""
        for line in response_text.splitlines():
            if line.lower().startswith("activity:"):
                activity = line.split(":", 1)[1].strip()
            elif line.lower().startswith("details:"):
                details = line.split(":", 1)[1].strip()
            elif line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()

        output_text = response_text

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"raw_response": response_text},
            structured_output={
                "activity": activity,
                "details": details,
                "summary": summary,
                "num_frames": len(frames),
            },
            metadata={"method": "openrouter_vision_fallback", "frame_count": len(frames)},
        )
