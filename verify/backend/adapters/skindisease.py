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

import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers

# Model files in the repo
_ASSETS = TARGET_APPS_DIR / "skin-disease-detection" / "app" / "assets"
_MODEL_CANCER   = _ASSETS / "quantized_pruned_model_cancer.tflite"
_MODEL_ALLERGY  = _ASSETS / "quantized_pruned_model_allergy.tflite"
_MODEL_MELANOMA = _ASSETS / "quantized_pruned_model_melanoma.tflite"

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


def _run_tflite_model(model_path: Path, pil_image, labels: List[str]) -> Dict[str, Any]:
    """
    Run a single TFLite classification model on a PIL image.

    Preprocessing mirrors the Dart classifier:
      1. Center-crop to square
      2. Resize to model input H×W via NEAREST_NEIGHBOUR
      3. Normalize: uint8 input → keep as-is; float32 input → divide by 255
    """
    import numpy as np
    from PIL import Image as PILImage
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite  # type: ignore

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    _, H, W, _ = inp["shape"]

    # Center-crop then resize (mirrors ResizeWithCropOrPadOp + ResizeOp in Dart)
    w, h = pil_image.size
    crop = min(w, h)
    left, top = (w - crop) // 2, (h - crop) // 2
    img = pil_image.crop((left, top, left + crop, top + crop))
    img = img.resize((W, H), PILImage.NEAREST)
    if inp["dtype"] == np.uint8:
        arr = np.array(img, dtype=np.uint8)
    else:
        arr = np.array(img, dtype=np.float32) / 255.0

    interpreter.set_tensor(inp["index"], arr[np.newaxis])
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"])[0]

    # Convert uint8 output back to float probabilities
    if out["dtype"] == np.uint8:
        scale, zero_point = out.get("quantization", (1.0, 0))
        raw = (raw.astype(np.float32) - zero_point) * scale

    probs = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))

    return {
        "label": labels[top_idx] if top_idx < len(labels) else "Unknown",
        "confidence": float(probs[top_idx]),
        "all_scores": {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))},
    }


class SkinDiseaseAdapter(BaseAdapter):
    """
    Wraps the skin-disease-detection TFLite classification pipeline.

    NATIVE mode     : runs the three .tflite model files from the repo via tflite_runtime.
    SERVERLESS mode : OpenRouter vision replicating the same three-model output structure.
    """

    name = "skin-disease-detection"
    supported_modalities = ["image"]

    def _check_tflite(self) -> Tuple[bool, str]:
        """Return (available, reason) for tflite_runtime."""
        try:
            import tflite_runtime.interpreter  # noqa: F401
            return True, "tflite_runtime"
        except ImportError:
            pass
        try:
            import tensorflow.lite  # noqa: F401
            return True, "tensorflow.lite"
        except ImportError:
            pass
        return False, "neither tflite_runtime nor tensorflow is installed"

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            tflite_ok, tflite_msg = self._check_tflite()
            if not tflite_ok:
                return False, f"[NATIVE] {tflite_msg}. Install with: pip install tflite-runtime"
            missing = [m.name for m in [_MODEL_CANCER, _MODEL_ALLERGY, _MODEL_MELANOMA]
                       if not m.exists()]
            if missing:
                return False, f"[NATIVE] Missing model files: {', '.join(missing)}"
            return True, f"[NATIVE] tflite_runtime ({tflite_msg}) + 3 model files found."
        else:
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
        """
        Run the three TFLite models from the app's assets directory.
        Mirrors the Dart classifier pipeline: preprocess → infer → top-label + confidence.
        """
        tflite_ok, tflite_msg = self._check_tflite()
        if not tflite_ok:
            return AdapterResult(success=False, error=f"[NATIVE] {tflite_msg}")

        from PIL import Image as PILImage
        data = input_item.get("data")
        path = input_item.get("path", "")
        if data is None:
            pil_image = PILImage.open(str(path)).convert("RGB")
        elif isinstance(data, (str, Path)):
            pil_image = PILImage.open(str(data)).convert("RGB")
        else:
            pil_image = data.convert("RGB")

        results: Dict[str, Any] = {}
        errors: List[str] = []

        for label, model_path, labels in [
            ("cancer",   _MODEL_CANCER,   _LABELS_CANCER),
            ("allergy",  _MODEL_ALLERGY,  _LABELS_ALLERGY),
            ("melanoma", _MODEL_MELANOMA, _LABELS_MELANOMA),
        ]:
            if not model_path.exists():
                errors.append(f"Model file not found: {model_path.name}")
                continue
            try:
                results[label] = _run_tflite_model(model_path, pil_image, labels)
            except Exception as e:
                errors.append(f"{label}: {e}")

        if not results:
            return AdapterResult(success=False, error="; ".join(errors))

        output_text = self._format_output(results)

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output=results,
            structured_output=results,
            metadata={
                "method": "native_tflite",
                "models_run": list(results.keys()),
                **({"errors": errors} if errors else {}),
            },
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

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"raw_response": raw_response, "parsed": structured},
            structured_output=structured,
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
