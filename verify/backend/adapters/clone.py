"""
Adapter for the clone app.

Core pipeline: video/image → sampled frames → LLM description → stored in clone chat session.

Clone's AI inference runs in the Electron frontend (TypeScript + ONNX/OpenAI). The Django
REST backend is a storage layer (auth, chat sessions, VectorDB proxy).

Execution mode is controlled by USE_APP_SERVERS in .env:

  USE_APP_SERVERS=true  (NATIVE mode)
    Runs the clone Django server inside the 'clone' conda env via CondaRunner.
    Bootstraps Django with a SQLite shim, creates a chat session via ORM, calls
    OpenRouter vision for the frame description, and persists the message.
    No running clone server required.

  USE_APP_SERVERS=false  (SERVERLESS mode)
    Calls OpenRouter vision directly with the same prompt structure.
    No clone server or conda env involved.

Both modes produce structurally identical outputs.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from verify.backend.adapters.base import BaseAdapter, AdapterResult, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import (
    TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers,
)
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

_CLONE_SERVER = TARGET_APPS_DIR / "clone" / "server"

_ENV_SPEC = EnvSpec(
    name="clone",
    python="3.12",
    install_cmds=[[
        "pip", "install",
        "django==5.2",
        "djangorestframework",
        "django-cors-headers",
        "djangorestframework-simplejwt",
        "drf-yasg",
        "python-dotenv",
        "requests",
        "Pillow",
        "django-storages",
        "boto3",
    ]],
    cwd=_CLONE_SERVER,
)
_RUNNER = Path(__file__).parent.parent / "runners" / "clone_runner.py"

# Prompt used in both modes so outputs are structurally identical
_FRAME_PROMPT = (
    "You are an AI personal assistant analyzing screen activity recordings "
    "(like the clone app). You receive sampled frames from a video or screenshot. "
    "Describe:\n"
    "1. What activity/scene is shown.\n"
    "2. Any visible text, applications, or identifiable elements.\n"
    "3. A concise summary suitable for a personal knowledge base.\n\n"
    "Format:\n"
    "Activity: <description>\n"
    "Details: <visible elements>\n"
    "Summary: <1-2 sentence summary>"
)


def _encode_pil_b64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_description(text: str) -> Tuple[str, str, str]:
    activity, details, summary = "", "", ""
    for line in text.splitlines():
        lower = line.lower()
        if lower.startswith("activity:"):
            activity = line.split(":", 1)[1].strip()
        elif lower.startswith("details:"):
            details = line.split(":", 1)[1].strip()
        elif lower.startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
    return activity, details, summary


class CloneAdapter(BaseAdapter):
    """
    Wraps the clone screen-activity pipeline.

    NATIVE mode     : CondaRunner bootstraps clone Django, creates chat session via ORM,
                      calls OpenRouter vision, persists the message.
    SERVERLESS mode : direct OpenRouter vision call, same prompt, no clone server required.
    """

    name = "clone"
    supported_modalities = ["image", "video"]

    # ── Availability ──────────────────────────────────────────────────────────

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter vision (no clone server required)."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        modality = input_item.get("modality", "image")
        if modality not in self.supported_modalities:
            return AdapterResult(
                success=False,
                error=f"clone adapter does not support modality '{modality}'.",
            )

        frames = self._collect_frames(input_item)
        if not frames:
            return AdapterResult(success=False, error="No frames available in input_item.")

        if use_app_servers():
            return self._run_native(frames, input_item)
        return self._run_serverless(frames)

    def _collect_frames(self, input_item: Dict[str, Any]) -> List[Any]:
        if input_item.get("modality") == "video":
            return input_item.get("frames", [])
        data = input_item.get("data")
        if data is None:
            from PIL import Image as PILImage
            data = PILImage.open(input_item["path"]).convert("RGB")
        return [data]

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, frames: List[Any], input_item: Dict[str, Any]) -> AdapterResult:
        """CondaRunner: Django ORM session + OpenRouter vision inside 'clone' conda env."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        frames_b64 = [_encode_pil_b64(f) for f in frames]
        filename = input_item.get("filename", "verify-input")
        path = input_item.get("path", "")

        ok, result, err = CondaRunner.run(
            _ENV_SPEC.name,
            _RUNNER,
            {
                "frames_base64": frames_b64,
                "openrouter_api_key": get_openrouter_api_key() or "",
                "model": OPENROUTER_DEFAULT_MODEL,
                "filename": filename,
                "path": path,
            },
            timeout=90,
        )
        if not ok:
            return AdapterResult(success=False, error=err)

        description = result.get("description", "")
        activity = result.get("activity", "")
        details = result.get("details", "")
        summary = result.get("summary", "")
        session_id: Optional[int] = result.get("session_id")
        externalizations = result.get("externalizations", {})

        return AdapterResult(
            success=result.get("success", False),
            output_text=description,
            raw_output=result,
            structured_output={
                "activity": activity,
                "details": details,
                "summary": summary,
                "num_frames": len(frames),
            },
            externalizations=externalizations,
            metadata={
                "method": "native_django_orm",
                "session_id": session_id,
                "frame_count": len(frames),
            },
        )

    # ── SERVERLESS mode ───────────────────────────────────────────────────────

    def _run_serverless(self, frames: List[Any]) -> AdapterResult:
        """Direct OpenRouter vision call, same prompt as native mode."""
        try:
            description = self._describe_frames(frames)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

        activity, details, summary = _parse_description(description)

        externalizations = {
            "NETWORK": "[OpenRouter Fallback] Direct vision request with sampled frames.",
            "UI": f"Displaying generated summary: {summary}",
        }

        return AdapterResult(
            success=True,
            output_text=description,
            raw_output={"raw_response": description},
            structured_output={
                "activity": activity,
                "details": details,
                "summary": summary,
                "num_frames": len(frames),
            },
            externalizations=externalizations,
            metadata={"method": "serverless", "frame_count": len(frames)},
        )

    # ── Vision helper (serverless only) ──────────────────────────────────────

    def _describe_frames(self, frames: List[Any]) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": _FRAME_PROMPT}]
        for frame in frames:
            b64 = _encode_pil_b64(frame)
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )

        api_key = get_openrouter_api_key()
        if not api_key or api_key.startswith("your_"):
            raise RuntimeError("No valid OPENROUTER_API_KEY configured.")

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Verify",
                "X-Title": "Verify",
            },
            json={
                "model": OPENROUTER_DEFAULT_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
