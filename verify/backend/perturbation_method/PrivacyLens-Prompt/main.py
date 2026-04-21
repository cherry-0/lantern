"""
PrivacyLens — text perturbation for privacy attribute removal.

Strategy: use OpenRouter LLM to rewrite text with selected privacy attributes
(e.g. location, identity) removed or anonymized.
"""

from typing import Any, Dict, List, Optional, Tuple

from verify.backend.utils.verbose_log import log_perturbation


def check_availability() -> Tuple[bool, str]:
    """Check that yaml is available and credentials are configured for the active backend."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        return False, "PyYAML is required. Please install it."

    from verify.backend.utils.config import is_infer_local, get_openrouter_api_key

    if is_infer_local():
        return True, "PrivacyLens ready (local Ollama backend)."

    api_key = get_openrouter_api_key()
    if not api_key or api_key.startswith("your_"):
        return False, "PrivacyLens requires a valid OPENROUTER_API_KEY (or set INFER_LOCAL=true)."

    return True, "PrivacyLens ready (OpenRouter)."


def perturb(
    input_item: Dict[str, Any],
    attributes: List[str],
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Perturb a text item to remove signals for the given privacy attributes.

    Uses OpenRouter to rewrite the text, removing or anonymizing the specified attributes.
    A joint perturbation prompt is constructed from all selected attributes.

    Args:
        input_item: item dict from the dataset loader (must have "text_content" str).
        attributes: list of attribute names to remove (e.g. ["location", "identity"]).

    Returns:
        (success, perturbed_item_dict, error_message)
        perturbed_item_dict has "text_content" and "data" updated.
    """
    available, reason = check_availability()
    if not available:
        log_perturbation(method_name="PrivacyLens", filename=input_item.get("filename", "<unknown>"), attributes=attributes, ok=False, error=reason)
        return False, input_item, reason

    import copy
    import yaml
    from pathlib import Path

    text_content = input_item.get("text_content", "")
    if not text_content:
        log_perturbation(method_name="PrivacyLens", filename=input_item.get("filename", "<unknown>"), attributes=attributes, ok=False, error="No text_content found in input item.")
        return False, input_item, "No text_content found in input item."

    # Load prompt and attributes from YAML
    prompt_yaml_path = Path(__file__).parent / "prompt" / "prompt_privacylens.yaml"
    try:
        with open(prompt_yaml_path, "r", encoding="utf-8") as f:
            prompt_data = yaml.safe_load(f)
    except Exception as e:
        log_perturbation(method_name="PrivacyLens", filename=input_item.get("filename", "<unknown>"), attributes=attributes, ok=False, error=f"Failed to load prompt from {prompt_yaml_path}: {e}")
        return False, input_item, f"Failed to load prompt from {prompt_yaml_path}: {e}"

    system_prompt_template = prompt_data.get("system_prompt", "")
    attr_descriptions = prompt_data.get("attributes", {})

    attr_bullets = []
    for attr in attributes:
        desc = attr_descriptions.get(attr, f"{attr} information")
        attr_bullets.append(f"- {attr.upper()}: {desc}")

    attr_list_str = "\n".join(attr_bullets)

    prompt = system_prompt_template.format(
        attr_list_str=attr_list_str,
        text_content=text_content
    )

    from verify.backend.utils.config import get_perturbation_text_model
    from verify.backend.utils.llm_client import call_llm

    try:
        perturbed_text = call_llm(
            [{"role": "user", "content": prompt}],
            model=get_perturbation_text_model(),
            max_tokens=2048,
            timeout=90,
        )
    except Exception as e:
        log_perturbation(method_name="PrivacyLens", filename=input_item.get("filename", "<unknown>"), attributes=attributes, ok=False, error=f"PrivacyLens call failed: {e}")
        return False, input_item, f"PrivacyLens call failed: {e}"

    perturbed_item = copy.copy(input_item)
    perturbed_item["text_content"] = perturbed_text
    perturbed_item["data"] = perturbed_text
    perturbed_item["perturbation_applied"] = {
        "method": "PrivacyLens",
        "attributes": attributes,
    }

    log_perturbation(method_name="PrivacyLens", filename=input_item.get("filename", "<unknown>"), attributes=attributes, ok=True)
    return True, perturbed_item, None
