import base64
import json
import os
import sys
import tempfile
import traceback
import argparse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

runners_dir = os.path.dirname(os.path.abspath(__file__))
budgetlens_root = os.path.normpath(os.path.join(
    runners_dir, "..", "..", "..", "target-apps", "budget-lens", "budgetlens"
))

sys.path.insert(0, budgetlens_root)
sys.path.insert(0, runners_dir)

import _runtime_capture
_runtime_capture.install()

os.environ["DJANGO_SETTINGS_MODULE"] = "_budgetlens_verify_settings"
import django
django.setup()
_runtime_capture.connect_django_signals()

from django.conf import settings as _django_settings
from django.core.management import call_command
db_path = _django_settings.DATABASES["default"]["NAME"]
if not os.path.exists(db_path):
    print("[budget-lens-server] Creating SQLite database (first run) ...", file=sys.stderr, flush=True)
    call_command("migrate", "--run-syncdb", verbosity=0)
    print("[budget-lens-server] Database ready.", file=sys.stderr, flush=True)

import core.views as _core_views
from core.views import process_receipt

app = FastAPI()

class InferenceRequest(BaseModel):
    image_base64: str
    openrouter_api_key: str

class InferenceResponse(BaseModel):
    success: bool
    category: str
    date: str
    amount: float | None = None
    currency: str
    externalizations: dict
    error: str | None = None

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    _runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": []}
    _runtime_capture.set_phase("DURING")

    base_url = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = req.openrouter_api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    # core.views creates its OpenAI client at import time, so patch it now
    import openai as _openai
    _core_views.client = _openai.OpenAI(api_key=req.openrouter_api_key, base_url=base_url)

    tmp = None
    try:
        img_bytes = base64.b64decode(req.image_base64)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(img_bytes)
        tmp.close()

        category, expense_date, amount, currency = process_receipt(tmp.name)

        _runtime_capture.set_phase("POST")
        _runtime_capture.record_ui_event("DISPLAY_RECEIPT", f"{category} - {amount} {currency}")
        externalizations = _runtime_capture.finalize()

        return InferenceResponse(
            success=True,
            category=category,
            date=str(expense_date),
            amount=float(amount) if amount is not None else None,
            currency=currency,
            externalizations=externalizations
        )
    except Exception as e:
        err = traceback.format_exc()
        return InferenceResponse(success=False, category="", date="", currency="", externalizations={}, error=err)
    finally:
        if tmp and os.path.exists(tmp.name):
            os.unlink(tmp.name)

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
    print(f"[budget-lens-server] Starting API server on port {args.port}...", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
