"""
Imago_Obscura — image perturbation for privacy attribute removal.

Strategy: apply Gaussian blur (aggressive) to obscure location and identity cues.
Blur radius is scaled by the number of selected attributes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def check_availability() -> Tuple[bool, str]:
    """Check that Pillow is available (required for image operations)."""
    try:
        from PIL import Image, ImageFilter  # noqa: F401

        return True, "Imago_Obscura ready (PIL available)."
    except ImportError:
        return False, "Pillow is required for Imago_Obscura. Please install it: pip install Pillow"


def perturb(
    input_item: Dict[str, Any],
    attributes: List[str],
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Perturb an image to remove signals for the given privacy attributes.

    Strategy:
    - Apply Gaussian blur with a radius proportional to the number of attributes selected.
    - Aggressive perturbation is acceptable per Verify requirements.

    Args:
        input_item: item dict from the dataset loader (must have "data" PIL.Image
                    or "path" str, and optionally "image_base64").
        attributes: list of attribute names to remove (e.g. ["location", "identity"]).

    Returns:
        (success, perturbed_item_dict, error_message)
        perturbed_item_dict has same structure as input_item with "data", "image_base64" updated.
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

        if data is None:
            if not path:
                return False, input_item, "No image data or path in input item."
            data = PILImage.open(str(path)).convert("RGB")

        if not isinstance(data, PILImage.Image):
            return False, input_item, "Input data is not a PIL Image."

        # Blur radius: base 20, +10 per attribute (aggressive)
        blur_radius = 20 + 10 * (len(attributes) - 1) if attributes else 20

        perturbed_img = data.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Re-encode
        buf = io.BytesIO()
        perturbed_img.save(buf, format="JPEG", quality=80)
        perturbed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        perturbed_item = copy.copy(input_item)
        perturbed_item["data"] = perturbed_img
        perturbed_item["image_base64"] = perturbed_b64
        perturbed_item["perturbation_applied"] = {
            "method": "Imago_Obscura",
            "attributes": attributes,
            "blur_radius": blur_radius,
        }

        return True, perturbed_item, None

    except Exception as e:
        return False, input_item, f"Imago_Obscura perturbation failed: {e}"
