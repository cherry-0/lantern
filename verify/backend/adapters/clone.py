"""
Adapter for the clone app.

Core pipeline: video/image → sampled frames → LLM description → stored in clone chat session.

Clone's AI inference runs in the Electron frontend (TypeScript + ONNX/OpenAI). The Django
REST backend is a storage layer (auth, chat sessions, VectorDB proxy).

Execution mode is controlled by USE_APP_SERVERS in .env:

  USE_APP_SERVERS=true  (SERVER mode)
    Authenticates against the running clone Django server, creates a chat session, generates a
    frame description via OpenRouter (replicating what the Electron LLM would produce), and posts
    it as a user message.  Requires the server to be up — fails explicitly if not reachable.

  USE_APP_SERVERS=false  (SERVERLESS mode)
    Calls OpenRouter vision directly with the same prompt structure, no clone server involved.

Both modes use the same OpenRouter prompt so outputs are structurally comparable.

Configuration (.env)
--------------------
USE_APP_SERVERS    — "true" / "false"              (default: false)
CLONE_SERVER_URL   — base URL of clone Django server (default: http://localhost:8000)
CLONE_EMAIL        — Verify test-account email      (default: verify@lantern.local)
CLONE_PASSWORD     — Verify test-account password   (default: Verify_lantern_123!)
"""

import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from verify.backend.adapters.base import BaseAdapter, AdapterResult, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_DEFAULT_SERVER_URL = "http://localhost:8000"
_DEFAULT_EMAIL = "verify@lantern.local"
_DEFAULT_USERNAME = "verify_lantern"
_DEFAULT_PASSWORD = "Verify_lantern_123!"
_HTTP_TIMEOUT = 10

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

    SERVER mode     : auth → session → OpenRouter vision → post message to clone backend.
    SERVERLESS mode : direct OpenRouter vision call, same prompt, no clone server required.
    """

    name = "clone"
    supported_modalities = ["image", "video"]

    def __init__(self):
        self._server_url: str = (get_env("CLONE_SERVER_URL") or _DEFAULT_SERVER_URL).rstrip("/")
        self._email: str = get_env("CLONE_EMAIL") or _DEFAULT_EMAIL
        self._password: str = get_env("CLONE_PASSWORD") or _DEFAULT_PASSWORD
        self._access_token: Optional[str] = None
        self._server_available: Optional[bool] = None

    # ── Availability ──────────────────────────────────────────────────────────

    def _ping_server(self) -> bool:
        if self._server_available is not None:
            return self._server_available
        try:
            resp = requests.get(
                f"{self._server_url}/swagger/",
                timeout=_HTTP_TIMEOUT,
                allow_redirects=True,
            )
            self._server_available = resp.status_code < 500
        except requests.RequestException:
            self._server_available = False
        return self._server_available

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            if self._ping_server():
                return (
                    True,
                    f"[SERVER] Clone server reachable at {self._server_url}. "
                    "Pipeline: auth → session → OpenRouter vision → message.",
                )
            return (
                False,
                f"[SERVER] Clone server not reachable at {self._server_url}. "
                "Start the server or set USE_APP_SERVERS=false to use serverless mode.",
            )
        else:
            api_key = get_openrouter_api_key()
            if api_key and not api_key.startswith("your_"):
                return True, "[SERVERLESS] Using OpenRouter vision (no clone server required)."
            return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Auth helpers (server mode only) ──────────────────────────────────────

    def _ensure_auth(self) -> str:
        if self._access_token:
            return self._access_token
        token = self._try_login()
        if token:
            self._access_token = token
            return token
        self._try_signup()
        token = self._try_login()
        if token:
            self._access_token = token
            return token
        raise RuntimeError(
            f"Cannot authenticate with clone server at {self._server_url}. "
            "Check CLONE_EMAIL / CLONE_PASSWORD in .env, or create the account manually."
        )

    def _try_login(self) -> Optional[str]:
        try:
            resp = requests.post(
                f"{self._server_url}/api/auth/login/",
                json={"email": self._email, "password": self._password},
                timeout=_HTTP_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json().get("access")
        except requests.RequestException:
            pass
        return None

    def _try_signup(self) -> None:
        try:
            requests.post(
                f"{self._server_url}/api/auth/signup/",
                json={
                    "email": self._email,
                    "username": _DEFAULT_USERNAME,
                    "password": self._password,
                    "password_confirm": self._password,
                },
                timeout=_HTTP_TIMEOUT,
            )
        except requests.RequestException:
            pass

    def _auth_headers(self, token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _create_session(self, title: str, token: str) -> int:
        resp = requests.post(
            f"{self._server_url}/api/chat/sessions/create/",
            json={"title": title},
            headers=self._auth_headers(token),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def _post_message(self, session_id: int, content: str, token: str) -> None:
        requests.post(
            f"{self._server_url}/api/chat/sessions/{session_id}/messages/create/",
            json={
                "role": "user",
                "content": content,
                "timestamp": int(time.time() * 1000),
            },
            headers=self._auth_headers(token),
            timeout=_HTTP_TIMEOUT,
        )

    # ── Vision inference (shared by both modes) ───────────────────────────────

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
            return self._run_server(frames, filename=input_item.get("filename", "verify-input"))
        return self._run_serverless(frames)

    def _collect_frames(self, input_item: Dict[str, Any]) -> List[Any]:
        if input_item.get("modality") == "video":
            return input_item.get("frames", [])
        data = input_item.get("data")
        if data is None:
            from PIL import Image as PILImage
            data = PILImage.open(input_item["path"]).convert("RGB")
        return [data]

    # ── SERVER mode ───────────────────────────────────────────────────────────

    def _run_server(self, frames: List[Any], filename: str) -> AdapterResult:
        """
        SERVER mode: authenticate → create session → describe frames → post message.
        Fails explicitly on auth or server errors (no silent fallback to serverless).
        """
        if not self._ping_server():
            return AdapterResult(
                success=False,
                error=f"Clone server not reachable at {self._server_url}.",
            )

        try:
            token = self._ensure_auth()
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        try:
            description = self._describe_frames(frames)
        except Exception as e:
            return AdapterResult(success=False, error=f"OpenRouter vision failed: {e}")

        session_id: Optional[int] = None
        try:
            session_id = self._create_session(title=f"Verify: {filename[:80]}", token=token)
            self._post_message(session_id=session_id, content=description, token=token)
        except Exception:
            pass  # Session persistence failure doesn't invalidate the description output

        activity, details, summary = _parse_description(description)
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
            metadata={
                "method": "server",
                "server_url": self._server_url,
                "session_id": session_id,
                "frame_count": len(frames),
            },
        )

    # ── SERVERLESS mode ───────────────────────────────────────────────────────

    def _run_serverless(self, frames: List[Any]) -> AdapterResult:
        """
        SERVERLESS mode: direct OpenRouter vision call, same prompt as server mode.
        No clone server dependency.
        """
        try:
            description = self._describe_frames(frames)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

        activity, details, summary = _parse_description(description)
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
            metadata={"method": "serverless", "frame_count": len(frames)},
        )
