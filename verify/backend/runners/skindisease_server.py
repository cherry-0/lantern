import base64
import io
import os
import sys
import argparse
import uvicorn
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Add runners directory to path to allow _runtime_capture import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _runtime_capture

# Initialize capture (it hooks httpx, urllib3, etc.)
_runtime_capture.install()

app = FastAPI()

_LABELS_CANCER   = ["Actinic Keratoses", "Basal Cell Carcinoma",
                     "Benign Keratosis like Lesions", "Dermatofibroma",
                     "Melanocytic Nevi", "Vascular Lesions"]
_LABELS_ALLERGY  = ["Acne and Rosacea", "Eczema and Atopic Dermatitis",
                     "Nail Fungus and other Nail Disease"]
_LABELS_MELANOMA = ["Melanoma", "Not Melanoma"]

# Global state for models
interpreters = {}

class InferenceRequest(BaseModel):
    image_base64: str

class InferenceResponse(BaseModel):
    success: bool
    cancer: dict | None = None
    allergy: dict | None = None
    melanoma: dict | None = None
    externalizations: dict
    error: str | None = None

def load_models():
    if interpreters:
        return
    
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

    runners_dir = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps",
        "skin-disease-detection", "app", "assets"
    ))

    print("[skindisease-server] Loading models into memory...", file=sys.stderr, flush=True)
    for name, fname in [
        ("cancer",   "quantized_pruned_model_cancer.tflite"),
        ("allergy",  "quantized_pruned_model_allergy.tflite"),
        ("melanoma", "quantized_pruned_model_melanoma.tflite"),
    ]:
        model_path = os.path.join(assets, fname)
        if os.path.exists(model_path):
            interp = tflite.Interpreter(model_path=str(model_path))
            interp.allocate_tensors()
            interpreters[name] = interp
            print(f"[skindisease-server] Loaded {name} model.", file=sys.stderr, flush=True)
        else:
            print(f"[skindisease-server] Model not found: {model_path}", file=sys.stderr, flush=True)

def run_model(name, pil_image, labels):
    if name not in interpreters:
        raise ValueError(f"Model {name} not loaded")
        
    interpreter = interpreters[name]
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    _, H, W, _ = inp["shape"]

    w, h = pil_image.size
    crop = min(w, h)
    left, top = (w - crop) // 2, (h - crop) // 2
    img = pil_image.crop((left, top, left + crop, top + crop))
    from PIL import Image as PILImage
    img = img.resize((W, H), PILImage.NEAREST)

    if inp["dtype"] == np.uint8:
        arr = np.array(img, dtype=np.uint8)
    else:
        arr = np.array(img, dtype=np.float32) / 255.0

    interpreter.set_tensor(inp["index"], arr[np.newaxis])
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"])[0]

    if out["dtype"] == np.uint8:
        scale, zero_point = out.get("quantization", (1.0, 0))
        raw = (raw.astype(np.float32) - zero_point) * scale

    probs = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "label": labels[top_idx] if top_idx < len(labels) else "Unknown",
        "confidence": float(probs[top_idx]),
    }

@app.on_event("startup")
def startup_event():
    load_models()

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    _runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": []}
    _runtime_capture.set_phase("DURING")

    try:
        from PIL import Image as PILImage
        img_bytes = base64.b64decode(req.image_base64)
        pil_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

        results = {}
        errors = []
        for name, labels in [
            ("cancer", _LABELS_CANCER),
            ("allergy", _LABELS_ALLERGY),
            ("melanoma", _LABELS_MELANOMA),
        ]:
            try:
                results[name] = run_model(name, pil_image, labels)
            except Exception as e:
                errors.append(f"{name}: {e}")

        _runtime_capture.set_phase("POST")
        _runtime_capture.record_ui_event("DISPLAY_RESULTS", str(results))
        
        externalizations = _runtime_capture.finalize()
        return InferenceResponse(
            success=bool(results),
            cancer=results.get("cancer"),
            allergy=results.get("allergy"),
            melanoma=results.get("melanoma"),
            externalizations=externalizations,
            error="; ".join(errors) if errors else None
        )
    except Exception as e:
        err = traceback.format_exc()
        return InferenceResponse(success=False, externalizations={}, error=err)

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
    print(f"[skindisease-server] Starting API server on port {args.port}...", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
