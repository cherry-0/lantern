"""
Adapter for the skin-disease-detection app.

Core pipeline: skin lesion photo → TFLite classifier → disease label + confidence.

The app runs three separate quantized TFLite models on-device in Flutter/Dart:
  - quantized_pruned_model_cancer.tflite   → 6-class skin cancer classifier
  - quantized_pruned_model_allergy.tflite  → 3-class allergy/nail classifier
  - quantized_pruned_model_melanoma.tflite → binary melanoma classifier

Execution mode is controlled by USE_APP_SERVERS in .env:

  USE_APP_SERVERS=true  (NATIVE mode)
    Loads the same .tflite files from the repo and runs inference via tflite_runtime.
    Preprocessing mirrors the Dart classifier: center-crop → resize → normalize [0, 1].
    Requires:  pip install tflite-runtime   (or tensorflow)

  USE_APP_SERVERS=false  (SERVERLESS mode)
    Uses OpenRouter vision to replicate the same three-model classification output
    structure (cancer category, allergy category, melanoma verdict) without loading
    any local model files.
"""

import atexit
import base64
import io
import json
import platform
import re
from pathlib import Path
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

# Model files in the repo
_ASSETS = TARGET_APPS_DIR / "skin-disease-detection" / "app" / "assets"
_MODEL_CANCER   = _ASSETS / "quantized_pruned_model_cancer.tflite"
_MODEL_ALLERGY  = _ASSETS / "quantized_pruned_model_allergy.tflite"
_MODEL_MELANOMA = _ASSETS / "quantized_pruned_model_melanoma.tflite"

# tensorflow-macos is only available on macOS; use standard tensorflow elsewhere
_TF_PACKAGE = "tensorflow-macos" if platform.system() == "Darwin" else "tensorflow"

_ENV_SPEC = EnvSpec(
    name="skin-disease-detection",
    python="3.10",
    install_cmds=[["pip", "install", _TF_PACKAGE, "pillow", "numpy", "fastapi", "uvicorn", "pydantic", "requests"]],
)
_RUNNER = Path(__file__).parent.parent / "runners" / "skindisease_runner.py"

_LABELS_CANCER   = ["Actinic Keratoses", "Basal Cell Carcinoma",
                     "Benign Keratosis like Lesions", "Dermatofibroma",
                     "Melanocytic Nevi", "Vascular Lesions"]
_LABELS_ALLERGY  = ["Acne and Rosacea", "Eczema and Atopic Dermatitis",
                     "Nail Fungus and other Nail Disease"]
_LABELS_MELANOMA = ["Melanoma", "Not Melanoma"]


def _encode_image_b64(data_or_path) -> str:
    from PIL import Image as PILImage
    try:
        if isinstance(data_or_path, (str, Path)):
            img = PILImage.open(str(data_or_path)).convert("RGB")
        else:
            img = data_or_path.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to encode image: {e}")



class SkinDiseaseAdapter(BaseAdapter):
    """
    Wraps the skin-disease-detection TFLite classification pipeline.

    NATIVE mode     : runs the three .tflite model files from the repo via tflite_runtime.
    SERVERLESS mode : OpenRouter vision replicating the same three-model output structure.
    """

    name = "skin-disease-detection"
    supported_modalities = ["image"]
    env_spec = _ENV_SPEC

    def __init__(self):
        self._server_process = None
        self._server_port = None
        atexit.register(self._cleanup_server)

    def _cleanup_server(self):
        if self._server_process is not None:
            import sys
            print(f"[skin-disease] Shutting down local API server (port {self._server_port})...", file=sys.stderr, flush=True)
            self._server_process.terminate()
            self._server_process.wait()
            self._server_process = None

    def _start_server(self):
        if self._server_process is not None and self._server_process.poll() is None:
            return

        import socket
        import subprocess
        import time
        import requests
        import sys

        s = socket.socket()
        s.bind(("", 0))
        self._server_port = s.getsockname()[1]
        s.close()

        conda = CondaRunner.find_conda()
        server_script = Path(__file__).parent.parent / "runners" / "skindisease_server.py"

        print(f"[skin-disease] Starting local API server on port {self._server_port}...", file=sys.stderr, flush=True)
        self._server_process = subprocess.Popen(
            [conda, "run", "-n", _ENV_SPEC.name, "python", str(server_script), "--port", str(self._server_port)],
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )

        start_time = time.time()
        while time.time() - start_time < 30:
            try:
                resp = requests.get(f"http://127.0.0.1:{self._server_port}/health")
                if resp.status_code == 200:
                    print("[skin-disease] Server is ready.", file=sys.stderr, flush=True)
                    return
            except Exception:
                time.sleep(1)
        
        raise RuntimeError("skindisease_server failed to start within 30 seconds.")

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter vision to replicate TFLite classification."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "image":
            return AdapterResult(
                success=False,
                error=f"skin-disease-detection only supports 'image' modality, "
                      f"got '{input_item.get('modality')}'.",
            )

        if use_app_servers():
            return self._run_native(input_item)
        return self._run_serverless(input_item)

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Run the three TFLite models inside the 'skin-disease-detection' conda env via HTTP server."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        import requests
        import sys

        try:
            self._start_server()
        except Exception as e:
            return AdapterResult(success=False, error=f"Failed to start server: {e}")

        image_b64 = input_item.get("image_base64") or _encode_image_b64(
            input_item.get("data") or input_item.get("path", "")
        )

        try:
            payload = {"image_base64": image_b64}
            print("[skin-disease] Sending inference request to local server...", file=sys.stderr, flush=True)
            resp = requests.post(f"http://127.0.0.1:{self._server_port}/infer", json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"HTTP request to server failed: {e}")

        if not result.get("success"):
            return AdapterResult(success=False, error=result.get("error"))

        structured = {k: result[k] for k in ("cancer", "allergy", "melanoma") if k in result and result[k]}
        externalizations = result.get("externalizations", {})

        return AdapterResult(
            success=result.get("success", False),
            output_text=self._format_output(structured),
            raw_output=result,
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "native_tflite_server"},
        )

    # ── SERVERLESS mode ───────────────────────────────────────────────────────

    def _run_serverless(self, input_item: Dict[str, Any]) -> AdapterResult:
        """
        Replicate the three-model classification structure via OpenRouter vision.
        Output fields mirror native mode: cancer, allergy, melanoma each with
        label + confidence, so results are directly comparable.
        """
        data = input_item.get("data")
        path = input_item.get("path", "")
        image_b64 = input_item.get("image_base64")
        if image_b64 is None:
            image_b64 = _encode_image_b64(data if data is not None else path)

        cancer_str  = ", ".join(_LABELS_CANCER)
        allergy_str = ", ".join(_LABELS_ALLERGY)

        prompt = (
            "You are replicating three separate on-device TFLite skin analysis classifiers "
            "from a medical AI app. Analyze the skin area in this image and return results "
            "for all three classifiers.\n\n"
            f"Classifier 1 — Cancer (choose one): {cancer_str}\n"
            f"Classifier 2 — Allergy/Nail (choose one): {allergy_str}\n"
            "Classifier 3 — Melanoma (choose one): Melanoma, Not Melanoma\n\n"
            "Respond ONLY with a JSON object:\n"
            "{\n"
            '  "cancer":   {"label": "<category>", "confidence": <0.0–1.0>},\n'
            '  "allergy":  {"label": "<category>", "confidence": <0.0–1.0>},\n'
            '  "melanoma": {"label": "<Melanoma|Not Melanoma>", "confidence": <0.0–1.0>}\n'
            "}"
        )

        try:
            raw_response = self._call_openrouter(prompt, image_b64=image_b64, max_tokens=256)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        structured = self._parse_json(raw_response)
        output_text = self._format_output(structured)

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": "[OpenRouter Fallback] Direct vision request for three TFLite classifiers.",
                "UI": f"Rendering patient results: {output_text[:100]}...",
            }
        )

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"raw_response": raw_response, "parsed": structured},
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "serverless", "workflow": "tflite_classification"},
        )

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_output(results: Dict[str, Any]) -> str:
        lines = []
        for classifier, data in results.items():
            label = data.get("label", "Unknown")
            conf  = data.get("confidence", "N/A")
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
            lines.append(f"{classifier.capitalize()}: {label} (confidence: {conf_str})")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}
