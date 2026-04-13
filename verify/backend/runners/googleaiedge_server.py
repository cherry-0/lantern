import base64
import io
import os
import sys
import argparse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Add runners directory to path to allow _runtime_capture import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _runtime_capture

# Initialize capture (it hooks httpx, urllib3, etc.)
_runtime_capture.install()

app = FastAPI()

# Global state for models
models = {}

DEFAULT_VLM_MODEL  = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_TEXT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_CHAT_SYSTEM = (
    "You are a helpful AI assistant running locally on a mobile device via the Google AI Edge "
    "Gallery app. You have multimodal capabilities: you can process both text and images. "
    "Respond helpfully and concisely to the user's input."
)

class InferenceRequest(BaseModel):
    modality: str
    text_content: str = ""
    image_base64: str = ""
    model_id: str = ""
    max_tokens: int = 512

class InferenceResponse(BaseModel):
    success: bool
    ai_response: str
    model_id: str
    externalizations: dict
    error: str | None = None

def _decode_image(image_b64: str):
    from PIL import Image as PILImage
    return PILImage.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

def load_model(model_id: str, task: str):
    import torch
    from transformers import pipeline as hf_pipeline
    if model_id not in models:
        device = 0 if torch.cuda.is_available() else -1
        print(f"[google-ai-edge-server] Loading {task} model {model_id} (device={'cuda' if device >= 0 else 'cpu'}) ...", file=sys.stderr, flush=True)
        pipe = hf_pipeline(
            task,
            model=model_id,
            device=device,
            torch_dtype=torch.float16 if device >= 0 else torch.float32,
            trust_remote_code=True,
        )
        models[model_id] = pipe
        print(f"[google-ai-edge-server] Model {model_id} loaded.", file=sys.stderr, flush=True)
    return models[model_id]

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    # Reset externalization events at the start of each request
    _runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": []}
    _runtime_capture.set_phase("DURING")

    model_id = req.model_id or (DEFAULT_VLM_MODEL if req.image_base64 else DEFAULT_TEXT_MODEL)
    
    try:
        if req.modality == "image":
            if not req.image_base64:
                raise ValueError("image_base64 is empty")
            pipe = load_model(model_id, "image-text-to-text")
            img = _decode_image(req.image_base64)
            user_text = req.text_content.strip() if req.text_content.strip() else "Please analyze this image."
            pipe.model.generation_config.max_new_tokens = req.max_tokens
            pipe.model.generation_config.max_length = None
            messages = [
                {"role": "system", "content": [{"type": "text", "text": _CHAT_SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_text}]},
            ]
            outputs = pipe(messages)
            generated = outputs[0]["generated_text"]
            if isinstance(generated, list):
                response = generated[-1].get("content", "")
            else:
                response = str(generated)
        else:
            if not req.text_content:
                raise ValueError("text_content is empty")
            pipe = load_model(model_id, "text-generation")
            pipe.model.generation_config.max_new_tokens = req.max_tokens
            pipe.model.generation_config.max_length = None
            pipe.model.generation_config.do_sample = True
            pipe.model.generation_config.temperature = 1.0
            pipe.model.generation_config.top_k = 64
            pipe.model.generation_config.top_p = 0.95
            messages = [
                {"role": "system", "content": _CHAT_SYSTEM},
                {"role": "user", "content": req.text_content},
            ]
            outputs = pipe(messages)
            response = outputs[0]["generated_text"][-1]["content"]

        _runtime_capture.set_phase("POST")
        _runtime_capture.record_ui_event("DISPLAY_RESPONSE", response[:150] + "...")
        
        externalizations = _runtime_capture.finalize()
        return InferenceResponse(success=True, ai_response=response, model_id=model_id, externalizations=externalizations)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return InferenceResponse(success=False, ai_response="", model_id=model_id, externalizations={}, error=err)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    models.clear()
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {"status": "cleared"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"[google-ai-edge-server] Starting API server on port {args.port}...", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")