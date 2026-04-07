"""
Skin-disease runner — executed inside the 'skin-disease-detection' conda env.
Requires: pip install tensorflow-macos pillow numpy

Input JSON keys:
  image_base64  str   base64-encoded JPEG

Output JSON:
  success  bool
  cancer   {label, confidence}
  allergy  {label, confidence}
  melanoma {label, confidence}
  error    str | null
"""
import base64
import io
import json
import os
import sys
import traceback

_LABELS_CANCER   = ["Actinic Keratoses", "Basal Cell Carcinoma",
                     "Benign Keratosis like Lesions", "Dermatofibroma",
                     "Melanocytic Nevi", "Vascular Lesions"]
_LABELS_ALLERGY  = ["Acne and Rosacea", "Eczema and Atopic Dermatitis",
                     "Nail Fungus and other Nail Disease"]
_LABELS_MELANOMA = ["Melanoma", "Not Melanoma"]


def run_model(model_path, pil_image, labels):
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

    if out["dtype"] == np.uint8:
        scale, zero_point = out.get("quantization", (1.0, 0))
        raw = (raw.astype(np.float32) - zero_point) * scale

    probs = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "label": labels[top_idx] if top_idx < len(labels) else "Unknown",
        "confidence": float(probs[top_idx]),
    }


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    image_b64: str = data["image_base64"]
    img_bytes = base64.b64decode(image_b64)

    from PIL import Image as PILImage
    pil_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

    runners_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, runners_dir)
    from _runner_log import log_input
    log_input("skin-disease", "image", data.get("path", "<base64>"))

    assets = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps",
        "skin-disease-detection", "app", "assets"
    ))

    results = {}
    errors = []
    for name, fname, labels in [
        ("cancer",   "quantized_pruned_model_cancer.tflite",   _LABELS_CANCER),
        ("allergy",  "quantized_pruned_model_allergy.tflite",  _LABELS_ALLERGY),
        ("melanoma", "quantized_pruned_model_melanoma.tflite", _LABELS_MELANOMA),
    ]:
        model_path = os.path.join(assets, fname)
        if not os.path.exists(model_path):
            errors.append(f"Model not found: {fname}")
            continue
        print(f"[skin-disease] Running {name} classifier ({fname}) ...", file=sys.stderr, flush=True)
        try:
            results[name] = run_model(model_path, pil_image, labels)
            print(f"[skin-disease] {name}: {results[name]['label']} ({results[name]['confidence']:.2f})", file=sys.stderr, flush=True)
        except Exception as e:
            errors.append(f"{name}: {e}")

    # --- Capture Externalizations as identified in analysis/skin-disease-detection.md ---
    externalizations = {
        "NETWORK": (
            "[Email/WhatsApp] Intent: Sending lesion photo + classification results to oncologist. \n"
            "[Google Maps API] HTTP GET: Nearby search for 'dermatologist' near current GPS coordinates."
        ),
        "STORAGE": "[Firebase Realtime DB] push().set(): Saving user contact info and lesion inquiry history.",
        "LOGGING": f"DEBUG: classifier.dart: Inference completed in 124ms. Top prediction: {results.get('cancer', {}).get('label', 'N/A')}"
    }

    print(json.dumps({
        "success": bool(results),
        **results,
        "externalizations": externalizations,
        "error": "; ".join(errors) if errors else None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"success": False, "error": traceback.format_exc()}))
        sys.exit(1)
