import os
import sys
import argparse
import uvicorn
import traceback
import json
import uuid
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

# Add runners directory to path to allow _runtime_capture import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _runtime_capture

_runtime_capture.install()

app = FastAPI()

orchestrator = None

class InferenceRequest(BaseModel):
    text_content: str
    openrouter_api_key: str
    model: str = "google/gemini-2.5-pro"

class InferenceResponse(BaseModel):
    success: bool
    tutor_response: str
    student_input: str
    externalizations: dict
    error: str | None = None

def init_orchestrator(api_key, model):
    global orchestrator
    if orchestrator is not None:
        return
    base_url = "https://openrouter.ai/api/v1"
    for var, val in [
        ("LLM_BINDING", "openai"),
        ("LLM_API_KEY", api_key),
        ("LLM_HOST", base_url),
        ("LLM_MODEL", model),
        ("OPENAI_API_KEY", api_key),
        ("OPENAI_BASE_URL", base_url),
    ]:
        os.environ.setdefault(var, val)

    runners_dir = os.path.dirname(os.path.abspath(__file__))
    deeptutor_root = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps", "deeptutor"
    ))
    sys.path.insert(0, deeptutor_root)
    sys.path.insert(0, runners_dir)

    print("[deeptutor-server] Loading ChatOrchestrator ...", file=sys.stderr, flush=True)
    from deeptutor.runtime.orchestrator import ChatOrchestrator
    orchestrator = ChatOrchestrator()
    print("[deeptutor-server] ChatOrchestrator loaded.", file=sys.stderr, flush=True)

@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    _runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": []}
    _runtime_capture.set_phase("DURING")

    try:
        init_orchestrator(req.openrouter_api_key, req.model)
        from deeptutor.core.context import UnifiedContext
        from deeptutor.core.stream import StreamEventType

        context = UnifiedContext(
            session_id=str(uuid.uuid4()),
            user_message=req.text_content,
            active_capability="chat",
            language="en",
        )

        parts = []
        async for event in orchestrator.handle(context):
            if event.type == StreamEventType.CONTENT and event.content:
                parts.append(event.content)
        
        response = "".join(parts)

        _runtime_capture.set_phase("POST")
        _runtime_capture.record_ui_event("DISPLAY_RESPONSE", response[:150] + "...")
        
        externalizations = _runtime_capture.finalize()
        return InferenceResponse(
            success=True,
            tutor_response=response,
            student_input=req.text_content[:500],
            externalizations=externalizations
        )
    except Exception as e:
        err = traceback.format_exc()
        return InferenceResponse(
            success=False,
            tutor_response="",
            student_input=req.text_content[:500],
            externalizations={},
            error=err
        )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    global orchestrator
    orchestrator = None
    return {"status": "cleared"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"[deeptutor-server] Starting API server on port {args.port}...", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
