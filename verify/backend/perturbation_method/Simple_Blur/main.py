"""
Simple_Blur — selective image perturbation for privacy attribute removal.

Strategy:
  1. Call OpenRouter (vision model) to locate bounding boxes of regions that
     reveal the selected privacy attributes (faces, signs, landmarks, etc.).
  2. Apply Gaussian blur only to those regions.
  3. Fall back to full-image blur if the VLM call fails or returns no boxes.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# What visual elements to look for per privacy attribute
_ATTRIBUTE_TARGETS = {
    "identity":  "human faces, name badges, ID cards, signatures, or any feature that uniquely identifies a person",
    "location":  "street signs, building names, shop signs, landmarks, license plates, maps, or any geographic identifier",
    "age":       "human faces (age is inferred from facial appearance)",
    "gender":    "human faces and bodies that reveal gender",
    "race":      "human faces and skin that reveal ethnicity or race",
    "religion":  "religious symbols, clothing, or iconography",
    "haircolor": "hair regions showing hair colour",
    "nudity":    "exposed skin or body parts",
}

BLUR_RADIUS = 30  # applied to each detected region


def check_availability() -> Tuple[bool, str]:
    try:
        from PIL import Image, ImageFilter  # noqa: F401
    except ImportError:
        return False, "Pillow is required. Install with: pip install Pillow"

    from verify.backend.utils.config import get_openrouter_api_key
    key = get_openrouter_api_key()
    if not key or key.startswith("your_"):
        return False, "Simple_Blur requires a valid OPENROUTER_API_KEY for region detection."

    return True, "Simple_Blur ready."


def _detect_regions(image_b64: str, attributes: List[str]) -> List[Dict[str, float]]:
    """
    Ask the vision model to return bounding boxes of sensitive regions.

    Returns a list of dicts with keys x1, y1, x2, y2 (normalised 0.0–1.0).
    Returns [] if the call fails or no regions are found.
    """
    import requests
    from verify.backend.utils.config import get_openrouter_api_key
    from verify.backend.adapters.base import OPENROUTER_DEFAULT_MODEL

    targets = []
    for attr in attributes:
        desc = _ATTRIBUTE_TARGETS.get(attr.lower(), f"{attr} information")
        targets.append(f"- {attr}: {desc}")
    targets_str = "\n".join(targets)

    prompt = (
        "You are a privacy-region detector. Identify all image regions that reveal "
        "the following sensitive attributes and return their bounding boxes.\n\n"
        f"Attributes to detect:\n{targets_str}\n\n"
        "Rules:\n"
        "- Return ONLY a JSON array of objects, no other text.\n"
        "- Each object: {\"label\": \"<attribute>\", \"x1\": 0.0, \"y1\": 0.0, \"x2\": 1.0, \"y2\": 1.0}\n"
        "- Coordinates are normalised fractions of image width/height (0.0 = top/left, 1.0 = bottom/right).\n"
        "- If no regions are found for an attribute, omit it.\n"
        "- Keep boxes tight around the sensitive content.\n"
        "Example: [{\"label\": \"identity\", \"x1\": 0.1, \"y1\": 0.05, \"x2\": 0.3, \"y2\": 0.35}]"
    )

    try:
        api_key = get_openrouter_api_key()
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Verify",
                "X-Title": "Verify",
            },
            json={
                "model": OPENROUTER_DEFAULT_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }],
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Extract JSON array from response (may be wrapped in markdown)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []

        boxes = json.loads(match.group())
        validated = []
        for box in boxes:
            if all(k in box for k in ("x1", "y1", "x2", "y2")):
                validated.append({
                    "x1": float(box["x1"]),
                    "y1": float(box["y1"]),
                    "x2": float(box["x2"]),
                    "y2": float(box["y2"]),
                    "label": box.get("label", ""),
                })
        return validated

    except Exception:
        return []


def _apply_regional_blur(img, boxes: List[Dict[str, float]]):
    """Blur each bounding box region on the image in-place (returns new image)."""
    from PIL import ImageFilter
    import copy

    result = img.copy()
    w, h = img.size

    for box in boxes:
        x1 = max(0, int(box["x1"] * w))
        y1 = max(0, int(box["y1"] * h))
        x2 = min(w, int(box["x2"] * w))
        y2 = min(h, int(box["y2"] * h))

        if x2 <= x1 or y2 <= y1:
            continue

        region = result.crop((x1, y1, x2, y2))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        result.paste(blurred, (x1, y1))

    return result


def perturb(
    input_item: Dict[str, Any],
    attributes: List[str],
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Selectively blur image regions that reveal the given privacy attributes.

    Falls back to full-image blur if no regions are detected.
    """
    available, reason = check_availability()
    if not available:
        return False, input_item, reason

    from PIL import Image as PILImage, ImageFilter
    import base64
    import io
    import copy

    try:
        data = input_item.get("data")
        path = input_item.get("path", "")
        image_b64 = input_item.get("image_base64")

        if data is None:
            if not path:
                return False, input_item, "No image data or path in input item."
            data = PILImage.open(str(path)).convert("RGB")

        if not isinstance(data, PILImage.Image):
            return False, input_item, "Input data is not a PIL Image."

        # Encode to base64 if not already available
        if not image_b64:
            buf = io.BytesIO()
            data.save(buf, format="JPEG", quality=85)
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Step 1: detect sensitive regions via VLM
        boxes = _detect_regions(image_b64, attributes)

        # Step 2a: selective blur if regions found; 2b: full-image fallback
        if boxes:
            perturbed_img = _apply_regional_blur(data, boxes)
            mode = "selective"
        else:
            perturbed_img = data.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
            mode = "full_image_fallback"

        # Re-encode
        buf = io.BytesIO()
        perturbed_img.save(buf, format="JPEG", quality=80)
        perturbed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        perturbed_item = copy.copy(input_item)
        perturbed_item["data"] = perturbed_img
        perturbed_item["image_base64"] = perturbed_b64
        perturbed_item["perturbation_applied"] = {
            "method": "Simple_Blur",
            "mode": mode,
            "attributes": attributes,
            "regions": boxes,
            "blur_radius": BLUR_RADIUS,
        }

        return True, perturbed_item, None

    except Exception as e:
        return False, input_item, f"Simple_Blur perturbation failed: {e}"
