import base64
import io
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

# Set dummy environment variables to prevent momentag backend from crashing
# before _momentag_verify_settings sets them properly.
os.environ["SECRET_KEY"] = "dummy"
os.environ["DJANGO_SETTINGS_MODULE"] = "_momentag_verify_settings"

import django
django.setup()

from django.conf import settings as _django_settings
from django.core.management import call_command
db_path = _django_settings.DATABASES["default"]["NAME"]
if not os.path.exists(db_path):
    print("[momentag-server] Creating SQLite database (first run) ...", file=sys.stderr, flush=True)
    call_command("migrate", "--run-syncdb", verbosity=0)
    print("[momentag-server] Database ready.", file=sys.stderr, flush=True)

from gallery.gpu_tasks import get_image_captions

_runtime_capture.install()
_runtime_capture.connect_django_signals()

app = FastAPI()

class InferenceRequest(BaseModel):
    image_base64: str

class InferenceResponse(BaseModel):
    success: bool
    captions: list[str]
    tags: list[str]
    externalizations: dict
    error: str | None = None

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    _runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": []}
    _runtime_capture.set_phase("DURING")

    try:
        img_bytes = base64.b64decode(req.image_base64)
        from PIL import Image as PILImage
        pil_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

        captions_data = get_image_captions(pil_image)

        captions, tags = [], []
        for item in captions_data:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                captions.append(str(item[0]))
                if len(item) >= 2 and item[1]:
                    tags.extend(item[1])
            elif isinstance(item, str):
                captions.append(item)

        tags = list(dict.fromkeys(tags))  # deduplicate

        _runtime_capture.set_phase("POST")
        _runtime_capture.record_ui_event("DISPLAY_TAGS", ", ".join(tags))
        
        externalizations = _runtime_capture.finalize()

        return InferenceResponse(
            success=True,
            captions=captions,
            tags=tags,
            externalizations=externalizations
        )
    except Exception as e:
        err = traceback.format_exc()
        return InferenceResponse(
            success=False,
            captions=[],
            tags=[],
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
    print(f"[momentag-server] Starting API server on port {args.port}...", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
