import asyncio
import json
import os
import sys
import traceback
import argparse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

runners_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, runners_dir)
import _runtime_capture
_runtime_capture.install()

app = FastAPI()

_VTUBER_SYSTEM = (
    "You are Shizuku, a cheerful and curious AI VTuber. You speak in a warm, "
    "expressive style and are genuinely interested in what the user shares with you. "
    "You respond naturally to conversation as a VTuber streamer would — engaging, "
    "friendly, and occasionally playful. Keep responses concise and conversational, "
    "as they will be spoken aloud via text-to-speech."
)

class InferenceRequest(BaseModel):
    text_content: str
    openrouter_api_key: str
    model: str = "google/gemini-2.5-pro"

class InferenceResponse(BaseModel):
    success: bool
    character_response: str
    user_message: str
    externalizations: dict
    error: str | None = None

@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    _runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": []}
    _runtime_capture.set_phase("DURING")

    try:
        user_message = req.text_content.strip()
        
        # Bootstrap llm-vtuber imports
        llmvtuber_src = os.path.normpath(os.path.join(
            runners_dir, "..", "..", "..", "target-apps", "llm-vtuber", "src"
        ))
        if llmvtuber_src not in sys.path:
            sys.path.insert(0, llmvtuber_src)

        from open_llm_vtuber.agent.stateless_llm.openai_compatible_llm import AsyncLLM

        llm = AsyncLLM(
            model=req.model,
            base_url="https://openrouter.ai/api/v1",
            llm_api_key=req.openrouter_api_key,
            temperature=1.0,
        )

        messages = [{"role": "user", "content": user_message}]

        parts = []
        async for chunk in llm.chat_completion(messages, system=_VTUBER_SYSTEM):
            if isinstance(chunk, str):
                parts.append(chunk)
        
        response = "".join(parts)

        # Post-Inference Externalization
        _runtime_capture.set_phase("POST")
        
        try:
            from open_llm_vtuber.tts.edge_tts import TTSEngine as EdgeTTS
            tts = EdgeTTS(voice="en-US-AvaMultilingualNeural")
            await tts.generate_audio("Hello! " + response[:20])
        except Exception:
            pass

        _runtime_capture.record_ui_event("DISPLAY_TEXT", response[:200])
        _runtime_capture.record_ui_event("ANIMATION", "Cheerful")

        externalizations = _runtime_capture.finalize()

        return InferenceResponse(
            success=True,
            character_response=response,
            user_message=user_message[:500],
            externalizations=externalizations
        )
    except Exception as e:
        err = traceback.format_exc()
        return InferenceResponse(
            success=False,
            character_response="",
            user_message=req.text_content[:500],
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
    print(f"[llmvtuber-server] Starting API server on port {args.port}...", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
