import base64
import json
import os
import sys
import time
import traceback
import argparse
import uvicorn
import requests
from fastapi import FastAPI
from pydantic import BaseModel

runners_dir = os.path.dirname(os.path.abspath(__file__))
clone_server = os.path.normpath(os.path.join(
    runners_dir, "..", "..", "..", "target-apps", "clone", "server"
))
sys.path.insert(0, clone_server)
sys.path.insert(0, runners_dir)

import _runtime_capture
_runtime_capture.install()

# Load clone .env
env_file = os.path.join(clone_server, ".env")
env_example = os.path.join(clone_server, ".env_example")
if not os.path.exists(env_file) and os.path.exists(env_example):
    import shutil
    shutil.copyfile(env_example, env_file)
if os.path.exists(env_file):
    for line in open(env_file).read().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

os.environ["DJANGO_SETTINGS_MODULE"] = "_clone_verify_settings"
import django
django.setup()
_runtime_capture.connect_django_signals()

from django.conf import settings as _django_settings
from django.core.management import call_command
db_path = _django_settings.DATABASES["default"]["NAME"]
if not os.path.exists(db_path):
    print("[clone-server] Creating SQLite database (first run) ...", file=sys.stderr, flush=True)
    call_command("migrate", "--run-syncdb", verbosity=0)
    print("[clone-server] Database ready.", file=sys.stderr, flush=True)

from user.models import User
from chat.models import ChatMessage, ChatSession

_email = "verify@lantern.local"
try:
    user = User.objects.get(email=_email)
except User.DoesNotExist:
    user = User.objects.create_user(
        email=_email, username="verify_lantern", password="Verify_lantern_123!"
    )

app = FastAPI()

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

def _parse_description(text: str):
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

class InferenceRequest(BaseModel):
    frames_base64: list[str] = []
    image_base64: str = ""
    openrouter_api_key: str
    model: str = "google/gemini-2.5-pro"
    filename: str = "verify-input"

class InferenceResponse(BaseModel):
    success: bool
    description: str
    activity: str
    details: str
    summary: str
    session_id: int | None = None
    externalizations: dict
    error: str | None = None

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    _runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": []}
    _runtime_capture.set_phase("DURING")

    try:
        frames_b64 = req.frames_base64
        if not frames_b64 and req.image_base64:
            frames_b64 = [req.image_base64]
        
        session = ChatSession.objects.create(
            user=user,
            title=f"Verify: {req.filename[:80]}",
            last_message_timestamp=int(time.time() * 1000),
        )
        session_id = session.id

        content = [{"type": "text", "text": _FRAME_PROMPT}]
        for b64 in frames_b64:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {req.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Verify",
                "X-Title": "Verify",
            },
            json={
                "model": req.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        description = resp.json()["choices"][0]["message"]["content"]

        ts = int(time.time() * 1000)
        ChatMessage.objects.create(session=session, role="user", content=description, timestamp=ts)
        session.last_message_timestamp = ts
        session.save(update_fields=["last_message_timestamp"])

        activity, details, summary = _parse_description(description)

        _runtime_capture.set_phase("POST")
        _runtime_capture.record_ui_event("DISPLAY_CHAT", summary)

        externalizations = _runtime_capture.finalize()

        return InferenceResponse(
            success=True,
            description=description,
            activity=activity,
            details=details,
            summary=summary,
            session_id=session_id,
            externalizations=externalizations
        )
    except Exception as e:
        err = traceback.format_exc()
        return InferenceResponse(
            success=False,
            description="",
            activity="",
            details="",
            summary="",
            externalizations={},
            error=err
        )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return {"status": "cleared"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"[clone-server] Starting API server on port {args.port}...", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
