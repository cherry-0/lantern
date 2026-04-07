"""
Adapter for the momentag app.

Core pipeline: image → CLIP embedding + BLIP captions → tag keywords.

Primary strategy: import gpu_tasks from the momentag app directly.
Fallback: use OpenRouter vision model to generate image description and tags.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

MOMENTAG_BACKEND = TARGET_APPS_DIR / "momentag" / "backend"

_ENV_SPEC = EnvSpec(
    name="momentag",
    python="3.13",
    install_cmds=[["pip", "install", "-e", str(MOMENTAG_BACKEND)]],
    cwd=MOMENTAG_BACKEND,
)
_RUNNER = Path(__file__).parent.parent / "runners" / "momentag_runner.py"


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




class MomentagAdapter(BaseAdapter):
    """
    Wraps the momentag image → tags/captions pipeline.

    Tries to use momentag's native CLIP/BLIP models if available.
    Falls back to OpenRouter vision model if not.
    """

    name = "momentag"
    supported_modalities = ["image"]

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter vision fallback for momentag."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "image":
            return AdapterResult(
                success=False,
                error=f"momentag only supports 'image' modality, got '{input_item.get('modality')}'.",
            )
        if use_app_servers():
            return self._run_native(input_item)
        return self._run_openrouter_fallback(input_item)

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Run momentag's CLIP/BLIP pipeline inside the 'momentag' conda env via subprocess."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        image_b64 = input_item.get("image_base64") or _encode_image_b64(
            input_item.get("data") or input_item.get("path", "")
        )
        ok, result, err = CondaRunner.run(
            _ENV_SPEC.name, _RUNNER,
            {"image_base64": image_b64},
            timeout=180,
        )
        if not ok:
            return AdapterResult(success=False, error=err)

        captions = result.get("captions", [])
        tags = result.get("tags", [])
        externalizations = result.get("externalizations", {})

        output_text = "Captions: " + " | ".join(captions)
        if tags:
            output_text += "\nTags: " + ", ".join(tags)
        return AdapterResult(
            success=result.get("success", False),
            output_text=output_text,
            raw_output=result,
            structured_output={"captions": captions, "tags": tags},
            externalizations=externalizations,
            metadata={"method": "native_clip_blip"},
        )

    def _run_openrouter_fallback(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Use OpenRouter vision model to generate tags/captions as momentag would."""
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

        try:
            response = self._call_openrouter(prompt, image_b64=image_b64, max_tokens=512)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

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

        # Simulated fallback externalizations
        externalizations = {
            "NETWORK": "[OpenRouter Fallback] Direct vision request for captions/tags.",
            "UI": f"Rendering search result thumbnails for: {', '.join(tags[:3])}..."
        }

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"raw_response": response},
            structured_output={"captions": [caption], "tags": tags},
            externalizations=externalizations,
            metadata={"method": "openrouter_vision_fallback"},
        )
