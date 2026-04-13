"""
Adapter for the momentag app.

Core pipeline: image → CLIP embedding + BLIP captions → tag keywords.

Primary strategy: import gpu_tasks from the momentag app directly.
Fallback: use OpenRouter vision model to generate image description and tags.
"""

import atexit
import base64
import os
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
    install_cmds=[
        ["bash", "-c", "uv export --frozen --no-hashes --no-dev | pip install -r /dev/stdin"],
        ["pip", "install", "fastapi", "uvicorn", "pydantic", "requests"]
    ],
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
    env_spec = _ENV_SPEC

    def __init__(self):
        self._server_process = None
        self._server_port = None
        atexit.register(self._cleanup_server)

    def _cleanup_server(self):
        if self._server_process is not None:
            import sys
            print(f"[momentag] Shutting down local API server (port {self._server_port})...", file=sys.stderr, flush=True)
            self._server_process.terminate()
            self._server_process.wait()
            self._server_process = None

    def _start_server(self):
        if self._server_process is not None and self._server_process.poll() is None:
            return

        import socket
        import subprocess
        import time
        import requests
        import sys

        s = socket.socket()
        s.bind(("", 0))
        self._server_port = s.getsockname()[1]
        s.close()

        conda = CondaRunner.find_conda()
        server_script = Path(__file__).parent.parent / "runners" / "momentag_server.py"

        print(f"[momentag] Starting local API server on port {self._server_port}...", file=sys.stderr, flush=True)
        self._server_process = subprocess.Popen(
            [conda, "run", "-n", _ENV_SPEC.name, "python", str(server_script), "--port", str(self._server_port)],
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )

        start_time = time.time()
        while time.time() - start_time < 60:  # Allow 60s since momentag loads large models
            try:
                resp = requests.get(f"http://127.0.0.1:{self._server_port}/health")
                if resp.status_code == 200:
                    print("[momentag] Server is ready.", file=sys.stderr, flush=True)
                    return
            except Exception:
                time.sleep(1)
        
        raise RuntimeError("momentag_server failed to start within 60 seconds.")

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
        """Run momentag's CLIP/BLIP pipeline inside the 'momentag' conda env via HTTP server."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        import requests
        import sys

        try:
            self._start_server()
        except Exception as e:
            return AdapterResult(success=False, error=f"Failed to start server: {e}")

        image_b64 = input_item.get("image_base64") or _encode_image_b64(
            input_item.get("data") or input_item.get("path", "")
        )

        try:
            payload = {"image_base64": image_b64}
            print("[momentag] Sending inference request to local server...", file=sys.stderr, flush=True)
            resp = requests.post(f"http://127.0.0.1:{self._server_port}/infer", json=payload, timeout=180)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"HTTP request to server failed: {e}")

        if not result.get("success"):
            return AdapterResult(success=False, error=result.get("error"))

        captions = result.get("captions", [])
        tags = result.get("tags", [])
        externalizations = result.get("externalizations", {})

        output_text = "Captions: " + " | ".join(captions)
        if tags:
            output_text += "\nTags: " + ", ".join(tags)
        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output=result,
            structured_output={"captions": captions, "tags": tags},
            externalizations=externalizations,
            metadata={"method": "native_clip_blip_server"},
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

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": "[OpenRouter Fallback] Direct vision request for captions/tags.",
                "UI": f"Rendering search result thumbnails for: {', '.join(tags[:3])}...",
            }
        )

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"raw_response": response},
            structured_output={"captions": [caption], "tags": tags},
            externalizations=externalizations,
            metadata={"method": "openrouter_vision_fallback"},
        )
