import json
import os
import sys
import traceback
import argparse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

runners_dir = os.path.dirname(os.path.abspath(__file__))
xend_backend = os.path.normpath(os.path.join(
    runners_dir, "..", "..", "..", "target-apps", "xend", "backend"
))

sys.path.insert(0, xend_backend)
sys.path.insert(0, runners_dir)

import _runtime_capture
_runtime_capture.install()

# Load xend .env first
env_file = os.path.join(xend_backend, ".env")
env_example = os.path.join(xend_backend, ".env_example")
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

os.environ["DJANGO_SETTINGS_MODULE"] = "_xend_verify_settings"
import django
django.setup()
_runtime_capture.connect_django_signals()

from django.conf import settings as _django_settings
from django.core.management import call_command
db_path = _django_settings.DATABASES["default"]["NAME"]
if not os.path.exists(db_path):
    print("[xend-server] Creating SQLite database (first run) ...", file=sys.stderr, flush=True)
    call_command("migrate", "--run-syncdb", verbosity=0)
    print("[xend-server] Database ready.", file=sys.stderr, flush=True)

from apps.ai.services.chains import body_chain, subject_chain
from apps.mail.services import send_email_logic

app = FastAPI()

class InferenceRequest(BaseModel):
    text_content: str
    openrouter_api_key: str
    model: str = "google/gemini-2.5-pro"

class InferenceResponse(BaseModel):
    success: bool
    subject: str
    body: str
    externalizations: dict
    error: str | None = None

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    _runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": []}
    _runtime_capture.set_phase("DURING")

    base_url = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = req.openrouter_api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_MODEL"] = req.model

    try:
        inputs = {
            "body": req.text_content,
            "subject": "",
            "language": "en",
            "recipients": "",
            "group_name": "",
            "group_description": "",
            "prompt_text": "",
            "sender_role": "",
            "recipient_role": "",
            "plan_text": "",
            "analysis": None,
            "fewshots": None,
            "profile": "",
            "attachments": [],
            "locked_subject": "",
        }

        subject = (subject_chain.invoke(inputs) or "").strip()
        inputs["locked_subject"] = subject
        body = (body_chain.invoke(inputs) or "").strip()

        _runtime_capture.set_phase("POST")
        try:
            send_email_logic(
                access_token="dummy-verify-token",
                to=["recipient@example.com"],
                subject=subject,
                body=body,
                is_html=False
            )
        except Exception as e:
            pass

        _runtime_capture.record_ui_event("NOTIFICATION", f"Email sent: {subject[:50]}...")
        externalizations = _runtime_capture.finalize()

        return InferenceResponse(
            success=True,
            subject=subject,
            body=body,
            externalizations=externalizations
        )
    except Exception as e:
        err = traceback.format_exc()
        return InferenceResponse(success=False, subject="", body="", externalizations={}, error=err)

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
    print(f"[xend-server] Starting API server on port {args.port}...", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
