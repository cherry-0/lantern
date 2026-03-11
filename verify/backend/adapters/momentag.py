"""
Adapter for the momentag app.

Core pipeline: image → CLIP embedding + BLIP captions → tag keywords.

Primary strategy: import gpu_tasks from the momentag app directly.
Fallback: use OpenRouter vision model to generate image description and tags.
"""

import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key

MOMENTAG_BACKEND = TARGET_APPS_DIR / "momentag" / "backend"


def _encode_image_b64(image_or_path) -> str:
    """Encode a PIL Image or file path to base64 JPEG."""
    try:
        from PIL import Image as PILImage
        import io

        if isinstance(image_or_path, str) or isinstance(image_or_path, Path):
            img = PILImage.open(str(image_or_path)).convert("RGB")
        else:
            img = image_or_path.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to encode image: {e}")


def _run_openrouter_vision(image_b64: str, prompt: str, api_key: str) -> str:
    """Call OpenRouter with a vision model and return the text response."""
    import requests

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
    return resp.json()["choices"][0]["message"]["content"]


class MomentagAdapter(BaseAdapter):
    """
    Wraps the momentag image → tags/captions pipeline.

    Tries to use momentag's native CLIP/BLIP models if available.
    Falls back to OpenRouter vision model if not.
    """

    name = "momentag"
    supported_modalities = ["image"]

    def __init__(self):
        self._native_available: Optional[bool] = None
        self._native_error: str = ""

    def _check_native(self) -> Tuple[bool, str]:
        """Try to import momentag's GPU pipeline."""
        if self._native_available is not None:
            return self._native_available, self._native_error

        momentag_path = str(MOMENTAG_BACKEND)
        if momentag_path not in sys.path:
            sys.path.insert(0, momentag_path)

        try:
            import django
            import os

            os.environ.setdefault(
                "DJANGO_SETTINGS_MODULE", "config.settings.local"
            )
            # Purge any cached 'config' package from a previous adapter to
            # avoid sys.modules collision (momentag and xend both use a
            # top-level package named 'config').
            for _mod in list(sys.modules):
                if _mod == "config" or _mod.startswith("config."):
                    del sys.modules[_mod]
            try:
                django.setup()
            except RuntimeError:
                pass  # Already set up

            from gallery.gpu_tasks import get_image_captions, get_image_model  # noqa: F401

            self._native_available = True
            self._native_error = ""
        except Exception as e:
            self._native_available = False
            self._native_error = str(e)

        return self._native_available, self._native_error

    def check_availability(self) -> Tuple[bool, str]:
        """Available if either native OR OpenRouter fallback works."""
        native_ok, native_err = self._check_native()
        if native_ok:
            return True, "Native momentag CLIP/BLIP pipeline available."

        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return (
                True,
                f"Native pipeline unavailable ({native_err}); using OpenRouter vision fallback.",
            )

        return (
            False,
            f"momentag native pipeline unavailable ({native_err}) and no valid OpenRouter API key found.",
        )

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "image":
            return AdapterResult(
                success=False,
                error=f"momentag only supports 'image' modality, got '{input_item.get('modality')}'.",
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
        """Run the actual CLIP/BLIP pipeline from momentag's gpu_tasks."""
        from gallery.gpu_tasks import (
            get_image_captions,
            get_image_model,
            get_image_embedding,
            phrase_to_words,
        )
        from PIL import Image as PILImage

        data = input_item.get("data")
        path = input_item.get("path", "")

        if data is None:
            data = PILImage.open(path).convert("RGB")

        # Get image captions (BLIP)
        captions_data = get_image_captions(data)
        # Each caption is (caption_text, keyword_list)
        captions = []
        tags = []
        for item in captions_data:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                captions.append(item[0])
                if len(item) >= 2 and item[1]:
                    tags.extend(item[1])
            elif isinstance(item, str):
                captions.append(item)

        # Deduplicate tags
        tags = list(dict.fromkeys(tags))

        output_text = "Captions: " + " | ".join(captions)
        if tags:
            output_text += "\nTags: " + ", ".join(tags)

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"captions": captions, "tags": tags},
            structured_output={"captions": captions, "tags": tags},
            metadata={"method": "native_clip_blip"},
        )

    def _run_openrouter_fallback(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Use OpenRouter vision model to generate tags/captions as momentag would."""
        api_key = get_openrouter_api_key()
        if not api_key or api_key.startswith("your_"):
            return AdapterResult(
                success=False,
                error="No valid OpenRouter API key available for momentag fallback.",
            )

        data = input_item.get("data")
        path = input_item.get("path", "")
        image_b64 = input_item.get("image_base64")

        if image_b64 is None:
            image_b64 = _encode_image_b64(data if data is not None else path)

        prompt = (
            "You are an AI photo tagging assistant (like the momentag app). "
            "Analyze this image and provide:\n"
            "1. A brief descriptive caption (1-2 sentences).\n"
            "2. A list of 5-10 relevant tags (single words or short phrases).\n\n"
            "Respond in this exact format:\n"
            "Caption: <your caption>\n"
            "Tags: <tag1>, <tag2>, <tag3>, ..."
        )

        response = _run_openrouter_vision(image_b64, prompt, api_key)

        # Parse response
        caption = ""
        tags: List[str] = []
        for line in response.splitlines():
            if line.lower().startswith("caption:"):
                caption = line.split(":", 1)[1].strip()
            elif line.lower().startswith("tags:"):
                raw_tags = line.split(":", 1)[1].strip()
                tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

        output_text = f"Caption: {caption}\nTags: {', '.join(tags)}"

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"raw_response": response},
            structured_output={"captions": [caption], "tags": tags},
            metadata={"method": "openrouter_vision_fallback"},
        )
