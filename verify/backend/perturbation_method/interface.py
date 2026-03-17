"""
Perturbation interface — resolves and invokes the correct perturbation
module based on modality and config.

Reads the perturbation method mapping from config/perturbation_method.csv and
dynamically imports the matching module's `perturb()` function.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from verify.backend.utils.config import load_perturbation_method_map, BACKEND_DIR

# Path to perturbation_method modules
PERTURBATION_DIR = BACKEND_DIR / "perturbation_method"


def _load_method_module(method_name: str):
    """
    Dynamically load the main.py module for the given perturbation method name.
    Returns the module object, or None if not found.
    """
    module_path = PERTURBATION_DIR / method_name / "main.py"
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        f"perturbation_{method_name}", str(module_path)
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def list_methods_for_modality(modality: str) -> List[str]:
    """
    Return all perturbation method names that have a main.py and support the given modality.
    Scans the perturbation_method directory for available modules.
    Image methods: anything except PrivacyLens-Prompt.
    Text methods: anything except image-only methods (heuristic: name contains 'Blur' or 'Obscura').
    Falls back to the config default as the first entry.
    """
    IMAGE_ONLY_KEYWORDS = {"blur", "obscura"}
    TEXT_ONLY_KEYWORDS = {"privacylens", "prompt"}

    all_methods = [
        d.name
        for d in PERTURBATION_DIR.iterdir()
        if d.is_dir() and (d / "main.py").exists()
    ]

    def _modality_match(name: str) -> bool:
        lower = name.lower().replace("-", "").replace("_", "")
        if modality == "image":
            return not any(k in lower for k in TEXT_ONLY_KEYWORDS)
        if modality == "text":
            return not any(k in lower for k in IMAGE_ONLY_KEYWORDS)
        return True

    matched = [m for m in sorted(all_methods) if _modality_match(m)]

    # Put the config default first
    default = get_perturbation_method(modality)
    if default and default in matched:
        matched = [default] + [m for m in matched if m != default]

    return matched


def get_perturbation_method(modality: str) -> Optional[str]:
    """
    Return the default perturbation method name for the given modality, from config.
    Returns None if not configured.
    """
    mapping = load_perturbation_method_map()
    return mapping.get(modality)


def check_perturbation_availability(modality: str, method_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check whether a perturbation method is available.

    Args:
        modality: "image", "text", or "video".
        method_name: explicit method name override; falls back to config default if None.

    Returns:
        (available: bool, reason: str)
    """
    name = method_name or get_perturbation_method(modality)
    if not name:
        return False, f"No perturbation method configured for modality '{modality}'."

    module = _load_method_module(name)
    if module is None:
        return False, f"Perturbation module '{name}' not found or failed to load."

    if not hasattr(module, "perturb"):
        return False, f"Perturbation module '{name}' has no `perturb()` function."

    if hasattr(module, "check_availability"):
        return module.check_availability()

    return True, f"Perturbation method '{name}' is available."


def run_perturbation(
    input_item: Dict[str, Any],
    modality: str,
    attributes: List[str],
    method_name: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Apply privacy perturbation to an input item.

    Args:
        input_item: loaded dataset item dict.
        modality: "image", "text", or "video".
        attributes: list of privacy attributes to remove.
        method_name: explicit method override; falls back to config default if None.

    Returns:
        (success, perturbed_item_dict, error_message)
    """
    if not attributes:
        return False, input_item, "No attributes selected for perturbation."

    name = method_name or get_perturbation_method(modality)
    if not name:
        return (
            False,
            input_item,
            f"No perturbation method configured for modality '{modality}' in perturbation_method.csv.",
        )

    module = _load_method_module(name)
    if module is None:
        return False, input_item, f"Perturbation module '{name}/main.py' could not be loaded."

    if not hasattr(module, "perturb"):
        return False, input_item, f"Perturbation module '{name}' does not define a `perturb()` function."

    try:
        return module.perturb(input_item, attributes)
    except Exception as e:
        return False, input_item, f"Perturbation failed ({name}): {e}"


def list_available_methods() -> Dict[str, Dict[str, Any]]:
    """
    Return info about all perturbation methods for all configured modalities.

    Returns:
        {modality: {"method": str, "available": bool, "reason": str}}
    """
    mapping = load_perturbation_method_map()
    result = {}
    for modality, method_name in mapping.items():
        available, reason = check_perturbation_availability(modality)
        result[modality] = {
            "method": method_name,
            "available": available,
            "reason": reason,
        }
    return result
