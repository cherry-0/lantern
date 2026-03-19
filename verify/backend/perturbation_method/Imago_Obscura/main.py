"""
Imago_Obscura — image perturbation for privacy attribute removal.

Strategy (in priority order):
  1. Florence-2 phrase grounding → detect sensitive regions → targeted blur/pixelate/fill
  2. Fallback: full-image Gaussian blur (used when torch/transformers are unavailable
     or when no regions are detected)

Set IMAGO_OBSCURA_MODE=blur|pixelate|fill and IMAGO_OBSCURA_BLUR_RADIUS=<int>
in the environment to tune behaviour without code changes.
"""

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the headless package importable regardless of working directory.
_SRC = Path(__file__).parent / "src" / "headless"
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))


def check_availability() -> Tuple[bool, str]:
    """Always runnable. Reports whether Florence-2 is available for semantic segmentation."""
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return False, "Pillow is required. Install it: pip install Pillow"

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True, "Imago_Obscura ready (Florence-2 semantic segmentation enabled)."
    except ImportError:
        return True, (
            "Imago_Obscura ready (fallback mode: full-image blur). "
            "Install torch + transformers for semantic region detection."
        )


def perturb(
    input_item: Dict[str, Any],
    attributes: List[str],
    mode: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Perturb an image to remove signals for the given privacy attributes.

    Args:
        input_item: item dict from the dataset loader (must have "data" PIL.Image
                    or "path" str, and optionally "image_base64").
        attributes: list of attribute names to remove (e.g. ["location", "identity"]).

    Returns:
        (success, perturbed_item_dict, error_message)
        perturbed_item_dict has same structure as input_item with "data", "image_base64" updated.
    """
    from PIL import Image as PILImage
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

        from headless.pipeline import run as _pipeline_run
        mode = mode or os.environ.get("IMAGO_OBSCURA_MODE", "blur")
        success, perturbed_img, error = _pipeline_run(data, attributes, mode=mode)

        if not success:
            return False, input_item, error

        buf = io.BytesIO()
        perturbed_img.save(buf, format="JPEG", quality=80)
        perturbed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        perturbed_item = copy.copy(input_item)
        perturbed_item["data"] = perturbed_img
        perturbed_item["image_base64"] = perturbed_b64
        perturbed_item["perturbation_applied"] = {
            "method": "Imago_Obscura",
            "attributes": attributes,
            "mode": mode,
        }

        return True, perturbed_item, None

    except Exception as e:
        return False, input_item, f"Imago_Obscura perturbation failed: {e}"
