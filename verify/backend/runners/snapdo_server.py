import base64
import json
import os
import sys
import traceback
import argparse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

runners_dir = os.path.dirname(os.path.abspath(__file__))
snapdo_server = os.path.normpath(os.path.join(
    runners_dir, "..", "..", "..", "target-apps", "snapdo", "server"
))

sys.path.insert(0, snapdo_server)
sys.path.insert(0, runners_dir)

import _runtime_capture
_runtime_capture.install()

# Load snapdo's own .env
env_file = os.path.join(snapdo_server, "snapdo", ".env")
if os.path.exists(env_file):
    for line in open(env_file).read().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

os.environ["DJANGO_SETTINGS_MODULE"] = "server.settings"
import django
django.setup()
_runtime_capture.connect_django_signals()

from snapdo.services.vlm_service import VLMService
service = VLMService()

app = FastAPI()

class InferenceRequest(BaseModel):
    image_base64: str
    task_title: str = ""
    task_description: str = ""
    openrouter_api_key: str
    model: str = "google/gemini-2.5-pro"

class InferenceResponse(BaseModel):
    success: bool
    verdict: str
    confidence: float | None = None
    explanation: str
    task_title: str
    externalizations: dict
    error: str | None = None

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    _runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": []}
    _runtime_capture.set_phase("DURING")

    os.environ["VLM_API_KEY"] = req.openrouter_api_key
    os.environ["VLM_API_URL"] = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = req.openrouter_api_key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

    constraint = req.task_title
    if req.task_description:
        constraint += ". " + req.task_description
    if not constraint:
        constraint = (
            "Identify and describe all visible content in this image, "
            "including objects, text, people, and location cues."
        )

    try:
        raw = service.verify_evidence(req.image_base64, constraint, model=req.model)

        _runtime_capture.set_phase("POST")
        _runtime_capture.record_ui_event("DISPLAY_VERDICT", f"{raw.get('verdict')} - {raw.get('explanation', '')[:100]}")
        externalizations = _runtime_capture.finalize()

        return InferenceResponse(
            success=True,
            verdict=raw.get("verdict", "UNKNOWN"),
            confidence=raw.get("confidence"),
            explanation=raw.get("explanation", ""),
            task_title=req.task_title,
            externalizations=externalizations
        )
    except Exception as e:
        err = traceback.format_exc()
        return InferenceResponse(success=False, verdict="UNKNOWN", explanation="", task_title=req.task_title, externalizations={}, error=err)

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
    print(f"[snapdo-server] Starting API server on port {args.port}...", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
