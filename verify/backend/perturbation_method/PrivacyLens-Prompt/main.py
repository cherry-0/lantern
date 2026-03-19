"""
PrivacyLens — text perturbation for privacy attribute removal.

Strategy: use OpenRouter LLM to rewrite text with selected privacy attributes
(e.g. location, identity) removed or anonymized.
"""

from typing import Any, Dict, List, Optional, Tuple


def check_availability() -> Tuple[bool, str]:
    """Check that requests and yaml are available, and an OpenRouter API key is configured."""
    try:
        import requests  # noqa: F401
        import yaml      # noqa: F401
    except ImportError:
        return False, "requests and PyYAML libraries are required. Please install them."

    from verify.backend.utils.config import get_openrouter_api_key

    api_key = get_openrouter_api_key()
    if not api_key or api_key.startswith("your_"):
        return False, "PrivacyLens requires a valid OPENROUTER_API_KEY in the environment."

    return True, "PrivacyLens ready (OpenRouter API key found)."


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
        return False, input_item, reason

    import requests
    import copy
    import yaml
    from pathlib import Path
    from verify.backend.utils.config import get_openrouter_api_key

    text_content = input_item.get("text_content", "")
    if not text_content:
        return False, input_item, "No text_content found in input item."

    api_key = get_openrouter_api_key()

    # Load prompt and attributes from YAML
    prompt_yaml_path = Path(__file__).parent / "prompt" / "prompt_privacylens.yaml"
    try:
        with open(prompt_yaml_path, "r", encoding="utf-8") as f:
            prompt_data = yaml.safe_load(f)
    except Exception as e:
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

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Verify",
                "X-Title": "Verify",
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
            },
            timeout=90,
        )
        resp.raise_for_status()
        perturbed_text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return False, input_item, f"PrivacyLens API call failed: {e}"

    perturbed_item = copy.copy(input_item)
    perturbed_item["text_content"] = perturbed_text
    perturbed_item["data"] = perturbed_text
    perturbed_item["perturbation_applied"] = {
        "method": "PrivacyLens",
        "attributes": attributes,
    }

    return True, perturbed_item, None
