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


def get_perturbation_method(modality: str) -> Optional[str]:
    """
    Return the perturbation method name for the given modality, from config.
    Returns None if not configured.
    """
    mapping = load_perturbation_method_map()
    return mapping.get(modality)


def check_perturbation_availability(modality: str) -> Tuple[bool, str]:
    """
    Check whether the perturbation method for the given modality is available.

    Returns:
        (available: bool, reason: str)
    """
    method_name = get_perturbation_method(modality)
    if not method_name:
        return False, f"No perturbation method configured for modality '{modality}'."

    module = _load_method_module(method_name)
    if module is None:
        return False, f"Perturbation module '{method_name}' not found or failed to load."

    if not hasattr(module, "perturb"):
        return False, f"Perturbation module '{method_name}' has no `perturb()` function."

    if hasattr(module, "check_availability"):
        return module.check_availability()

    return True, f"Perturbation method '{method_name}' is available."


def run_perturbation(
    input_item: Dict[str, Any],
    modality: str,
    attributes: List[str],
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Apply privacy perturbation to an input item.

    Resolves the perturbation method from config, loads the module, and calls perturb().

    Args:
        input_item: loaded dataset item dict.
        modality: "image", "text", or "video".
        attributes: list of privacy attributes to remove.

    Returns:
        (success, perturbed_item_dict, error_message)
    """
    if not attributes:
        return False, input_item, "No attributes selected for perturbation."

    method_name = get_perturbation_method(modality)
    if not method_name:
        return (
            False,
            input_item,
            f"No perturbation method configured for modality '{modality}' in perturbation_method.csv.",
        )

    module = _load_method_module(method_name)
    if module is None:
        return (
            False,
            input_item,
            f"Perturbation module '{method_name}/main.py' could not be loaded.",
        )

    if not hasattr(module, "perturb"):
        return (
            False,
            input_item,
            f"Perturbation module '{method_name}' does not define a `perturb()` function.",
        )

    try:
        return module.perturb(input_item, attributes)
    except Exception as e:
        return False, input_item, f"Perturbation failed ({method_name}): {e}"


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
