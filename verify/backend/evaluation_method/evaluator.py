"""
Privacy inferability evaluator.

Uses OpenRouter to assess whether selected privacy attributes are still
inferable from the target app's output (original and perturbed).

Evaluation is separate from the target app inference step:
  1. Target app runs on input → produces output.
  2. Evaluator analyzes the output → determines attribute inferability.
"""

import json
import re
import yaml
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from verify.backend.utils.config import get_openrouter_api_key

# Model to use for evaluation
EVAL_MODEL = "google/gemini-2.0-flash-001"

# Load EVAL_SYSTEM_PROMPT from prompts/prompt1.yaml
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_FILE  = PROMPTS_DIR / "prompt1.yaml"
PROMPT2_FILE = PROMPTS_DIR / "prompt2.yaml"
PROMPT3_FILE = PROMPTS_DIR / "prompt3.yaml"
PROMPT4_FILE = PROMPTS_DIR / "prompt4.yaml"

# ── Verdict constants (prompt4) ───────────────────────────────────────────────
VERDICT_CONFIRMED   = "confirmed leakage"
VERDICT_POSSIBLE    = "possible leakage"
VERDICT_NO_EVIDENCE = "no evidence"

_CHANNEL_GUIDANCE = (
    "\n\nNote: The input may contain data from multiple externalization channels labeled "
    "with [UI], [NETWORK], [STORAGE], or [LOGGING]. You must evaluate all channels "
    "for potential privacy attribute inferability. For example, a raw image sent via [NETWORK] "
    "may leak more PII than the summarized [UI] output."
)


def _load_system_prompt() -> str:
    """Load the system prompt from prompt1.yaml."""
    try:
        with open(PROMPT_FILE, "r") as f:
            data = yaml.safe_load(f)
            return data.get("EVAL_SYSTEM_PROMPT", "").strip() + _CHANNEL_GUIDANCE
    except Exception:
        return "You are a sophisticated privacy analyst. Evaluate privacy attributes inferability."


def _load_prompt2() -> tuple:
    """Load system prompt and MCQ choices from prompt2.yaml.

    Returns:
        (system_prompt: str, mcq_choices: dict)
        mcq_choices maps attribute name → {"choices": list|None, "instruction": str}
    """
    try:
        with open(PROMPT2_FILE, "r") as f:
            data = yaml.safe_load(f)
        system_prompt = data.get("EVAL_SYSTEM_PROMPT", "").strip() + _CHANNEL_GUIDANCE
        mcq_choices   = data.get("MCQ_CHOICES", {}) or {}
        return system_prompt, mcq_choices
    except Exception:
        return "You are a sophisticated privacy analyst. Evaluate privacy attributes inferability.", {}


def _load_prompt3() -> str:
    """Load the system prompt from prompt3.yaml."""
    try:
        with open(PROMPT3_FILE, "r") as f:
            data = yaml.safe_load(f)
            return data.get("EVAL_SYSTEM_PROMPT", "").strip() + _CHANNEL_GUIDANCE
    except Exception:
        return (
            "You are a sophisticated privacy analyst. Evaluate privacy attribute "
            "inferability for both aggregate and channel-specific outputs."
        )


def _load_prompt4() -> str:
    """Load the system prompt from prompt4.yaml."""
    try:
        with open(PROMPT4_FILE, "r") as f:
            data = yaml.safe_load(f)
            return data.get("EVAL_SYSTEM_PROMPT", "").strip() + _CHANNEL_GUIDANCE
    except Exception:
        return (
            "You are a sophisticated privacy analyst. Evaluate privacy attribute "
            "inferability and return a 3-way verdict per attribute."
        )


EVAL_SYSTEM_PROMPT = _load_system_prompt()
_PROMPT2_SYSTEM, _MCQ_CHOICES = _load_prompt2()
_PROMPT3_SYSTEM = _load_prompt3()
_PROMPT4_SYSTEM = _load_prompt4()

# Attribute-specific evaluation guidance
ATTRIBUTE_GUIDANCE = {
    "location": (
        "location information such as country, city, neighborhood, street address, GPS coordinates, "
        "place names, landmarks, building names, transit routes, or any geographic identifiers — "
        "including indirect cues like store names, airport codes, or regional dialect references."
    ),
    "identity": (
        "personally identifiable information such as real names, usernames, display names, "
        "email addresses, phone numbers, social media handles, account IDs, government ID numbers, "
        "or any combination of details that could uniquely identify or re-identify a specific person."
    ),
    "age": (
        "age-related information such as a stated age, birth year, date of birth, generational cohort "
        "(e.g. 'millennial', 'Gen Z'), or contextual cues that allow estimating someone's approximate age "
        "— including graduation years, retirement references, or descriptions of life stage."
    ),
    "gender": (
        "gender or sex information, including explicitly stated gender identity, pronouns used to refer "
        "to the person, gendered titles (Mr./Ms./Mx.), or implicit cues such as descriptions of gendered "
        "social roles or activities that reveal or strongly imply gender."
    ),
    "marital status": (
        "marital or relationship status such as being married, divorced, widowed, single, or in a "
        "domestic partnership — including direct statements, references to a spouse or partner, "
        "wedding events, divorce proceedings, or use of titles like 'Mrs.' or 'Mr.' implying marriage."
    ),
    # ── Image-modality attributes (HR-VISPR 18-class labels) ──────────────
    "face": (
        "the presence of a human face that is visible and potentially identifiable — including "
        "frontal, profile, or partial faces where a person's appearance could be recognised."
    ),
    "race": (
        "racial or ethnic appearance cues visible in the image, such as skin tone, facial features, "
        "or cultural markers that indicate or suggest a person's racial or ethnic background."
    ),
    "nudity": (
        "exposed skin, partial or full nudity, or sexually suggestive poses that reveal body parts "
        "beyond what is typical in professional or public contexts."
    ),
    "height": (
        "physical stature — whether a person appears notably tall or short relative to surroundings "
        "or other people, which could be used as a physical identifier."
    ),
    "weight": (
        "body size or weight — whether a person's build, body mass, or physique is discernible "
        "and could be used to characterise or identify them."
    ),
    "disability": (
        "visible signs of a physical disability or medical condition, such as prosthetics, "
        "wheelchairs, crutches, visible impairments, or assistive devices."
    ),
    "ethnic_clothing": (
        "clothing or accessories associated with a specific ethnic or cultural group, "
        "including traditional dress, religious garments, or culturally distinctive attire."
    ),
    "religion": (
        "religious affiliation indicators such as religious symbols, clothing (hijab, cross, kippah), "
        "places of worship in the background, or other visual markers of religious practice."
    ),
    "medical": (
        "medical context indicators such as hospital settings, medical equipment, bandages, "
        "IV lines, medical uniforms, or visible signs of illness or injury."
    ),
    "sports": (
        "sports participation or affiliation — uniforms, equipment, team logos, or activities "
        "that identify a person as a member of a specific sports team or athletic group."
    ),
    "formal": (
        "formal attire or professional dress such as suits, uniforms, or business clothing "
        "that may indicate occupational role, status, or organisational affiliation."
    ),
    "uniforms": (
        "occupational or institutional uniforms — military, police, medical, school, or corporate "
        "uniforms that reveal a person's role, rank, or institutional membership."
    ),
    "casual": (
        "casual everyday clothing that, while not inherently sensitive, may provide contextual "
        "clues about a person's lifestyle, socioeconomic status, or personal identity."
    ),
    "color": (
        "skin color or prominent color-based features that could be used to identify or "
        "characterise a person, particularly in combination with other attributes."
    ),
    "haircolor": (
        "hair color or style that is distinctive enough to serve as an identifying characteristic "
        "or reveal personal information about the individual."
    ),
    "troupe": (
        "group membership or association with a performing or social group (band, theatre troupe, "
        "choir, etc.) that could identify the person or reveal affiliations."
    ),
}

KNOWN_CHANNELS = ("UI", "NETWORK", "STORAGE", "LOGGING", "IPC")


def _extract_channels_from_text(output_text: str) -> Dict[str, str]:
    """Parse `[CHANNEL] content` blocks from joined ext_text."""
    if not output_text:
        return {}

    matches = list(re.finditer(r"^\[([A-Z]+)\]\s*", output_text, re.MULTILINE))
    if not matches:
        return {}

    channels: Dict[str, str] = {}
    for i, match in enumerate(matches):
        channel = match.group(1).strip().upper()
        if channel not in KNOWN_CHANNELS:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output_text)
        content = output_text[start:end].strip()
        channels[channel] = content
    return channels


def _normalize_binary_result(entry: Any) -> Dict[str, Any]:
    """Normalize a single inferability result entry into the flat prompt1 schema."""
    if not isinstance(entry, dict):
        return {
            "inferable": False,
            "score": 0,
            "reasoning": "Evaluator did not assess this attribute.",
        }
    inferable = bool(entry.get("inferable", False))
    reasoning = str(entry.get("reasoning", ""))
    return {
        "inferable": inferable,
        "score": 1 if inferable else 0,
        "reasoning": reasoning,
    }


def _normalize_verdict_result(entry: Any) -> Dict[str, Any]:
    """Normalize a single result entry into the prompt4 verdict schema.

    Returns a dict with verdict, score, reasoning, and inferable (for backward compat).
    """
    if not isinstance(entry, dict):
        return {
            "verdict":   VERDICT_NO_EVIDENCE,
            "score":     0,
            "reasoning": "Evaluator did not assess this attribute.",
            "inferable": False,
        }
    raw = str(entry.get("verdict", VERDICT_NO_EVIDENCE)).strip().lower()
    if "confirmed" in raw:
        verdict = VERDICT_CONFIRMED
    elif "possible" in raw:
        verdict = VERDICT_POSSIBLE
    else:
        verdict = VERDICT_NO_EVIDENCE
    score = 2 if verdict == VERDICT_CONFIRMED else (1 if verdict == VERDICT_POSSIBLE else 0)
    return {
        "verdict":   verdict,
        "score":     score,
        "reasoning": str(entry.get("reasoning", "")),
        "inferable": verdict != VERDICT_NO_EVIDENCE,
    }


def is_channelwise_eval_entry(entry: Any) -> bool:
    """Return True if entry matches the nested channel-wise schema."""
    if not isinstance(entry, dict):
        return False
    agg = entry.get("aggregate")
    return isinstance(agg, dict) and ("inferable" in agg or "verdict" in agg)


def is_verdict_eval_entry(entry: Any) -> bool:
    """Return True if entry matches the prompt4 verdict schema (aggregate with verdict)."""
    if not isinstance(entry, dict):
        return False
    agg = entry.get("aggregate")
    return isinstance(agg, dict) and "verdict" in agg


def get_aggregate_eval_entry(entry: Any) -> Dict[str, Any]:
    """Return the aggregate result for flat, prompt3, or prompt4 nested schemas."""
    if is_verdict_eval_entry(entry):
        return _normalize_verdict_result(entry.get("aggregate"))
    if is_channelwise_eval_entry(entry):
        return _normalize_binary_result(entry.get("aggregate"))
    return _normalize_binary_result(entry)


def get_channel_eval_entries(entry: Any) -> Dict[str, Dict[str, Any]]:
    """Return normalized channel results for prompt3 or prompt4 nested schemas."""
    if is_verdict_eval_entry(entry):
        channels = entry.get("channels", {})
        if not isinstance(channels, dict):
            return {}
        return {
            str(channel).upper(): _normalize_verdict_result(value)
            for channel, value in channels.items()
            if isinstance(value, dict)
        }
    if not is_channelwise_eval_entry(entry):
        return {}
    channels = entry.get("channels", {})
    if not isinstance(channels, dict):
        return {}
    return {
        str(channel).upper(): _normalize_binary_result(value)
        for channel, value in channels.items()
        if isinstance(value, dict)
    }


def entry_to_verdict(entry: Any) -> str:
    """Extract verdict string from any eval entry (binary flat/channelwise or verdict)."""
    agg = get_aggregate_eval_entry(entry)
    v = agg.get("verdict")
    if v in (VERDICT_CONFIRMED, VERDICT_POSSIBLE, VERDICT_NO_EVIDENCE):
        return v
    return VERDICT_CONFIRMED if agg.get("inferable") else VERDICT_NO_EVIDENCE


def verdict_to_icon(verdict: str) -> str:
    """Return a color indicator emoji for the given verdict string."""
    if verdict == VERDICT_CONFIRMED:
        return "🔴"
    if verdict == VERDICT_POSSIBLE:
        return "🟡"
    return "🟢"


def _build_eval_prompt(
    output_text: str,
    attributes: List[str],
) -> str:
    """Build the evaluation prompt for the given output and attributes."""
    attr_lines = []
    for attr in attributes:
        guidance = ATTRIBUTE_GUIDANCE.get(attr, f"{attr} information")
        attr_lines.append(f'- "{attr}": {guidance}')

    attributes_desc = "\n".join(attr_lines)

    schema_example = {
        attr: {"inferable": True, "score": 1, "reasoning": "example reasoning"}
        for attr in attributes
    }

    return f"""Analyze the following AI system output and determine whether each privacy attribute can be inferred from it.

Privacy attributes to assess:
{attributes_desc}

AI System Output:
---
{output_text}
---

Return a JSON object with this exact structure (no markdown, no extra text):
{json.dumps(schema_example, indent=2)}

Where:
- "inferable": boolean — true if the attribute CAN be inferred from the output
- "score": integer — 1 if inferable, 0 if not inferable (must match the inferable boolean)
- "reasoning": string — brief explanation of your assessment"""


def evaluate_inferability(
    output_text: str,
    attributes: List[str],
    api_key: Optional[str] = None,
    model: str = EVAL_MODEL,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Evaluate whether privacy attributes are inferable from the given output text.

    Args:
        output_text: the target app's output (e.g. tags, explanation, email body).
        attributes: list of attribute names to evaluate.
        api_key: optional override for OpenRouter API key.

    Returns:
        (success, results_dict, error_message)

        results_dict format:
            {
                "location": {"inferable": True, "score": 1, "reasoning": "..."},
                "identity": {"inferable": False, "score": 0, "reasoning": "..."},
            }
    """
    if not attributes:
        return True, {}, None

    if not output_text or not output_text.strip():
        return False, {}, "Empty output text provided for evaluation."

    key = api_key or get_openrouter_api_key()
    if not key or key.startswith("your_"):
        return False, {}, "No valid OpenRouter API key available for evaluation."

    import re
    import requests

    prompt = _build_eval_prompt(output_text, attributes)
    last_error: str = ""

    for attempt in range(5):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Verify",
                    "X-Title": "Verify",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4096,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip control characters that can appear in Gemini responses
            raw_content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_content)

            # Parse JSON response
            try:
                results = json.loads(raw_content)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if json_match:
                    results = json.loads(json_match.group())
                else:
                    last_error = f"Could not parse JSON from evaluator response: {raw_content[:200]}"
                    continue  # retry

            # Normalize results structure
            normalized: Dict[str, Any] = {}
            for attr in attributes:
                if attr in results:
                    entry = results[attr]
                    inferable = bool(entry.get("inferable", False))
                    normalized[attr] = {
                        "inferable": inferable,
                        "score": 1 if inferable else 0,
                        "reasoning": str(entry.get("reasoning", "")),
                    }
                else:
                    normalized[attr] = {
                        "inferable": False,
                        "score": 0,
                        "reasoning": "Evaluator did not assess this attribute.",
                    }

            return True, normalized, None

        except Exception as e:
            last_error = f"Evaluation API call failed: {e}"

    return False, {}, last_error


def _build_eval_prompt_v2(
    output_text: str,
    attributes: List[str],
    mcq_choices: Dict[str, Any],
) -> str:
    """Build an MCQ-style evaluation prompt using prompt2 choices."""
    attr_lines = []
    for attr in attributes:
        guidance  = ATTRIBUTE_GUIDANCE.get(attr, f"{attr} information")
        mcq_entry = mcq_choices.get(attr, {}) or {}
        choices   = mcq_entry.get("choices")
        instr     = mcq_entry.get("instruction", "")

        if choices:
            choice_str = " | ".join(f'"{c}"' for c in choices)
            attr_lines.append(
                f'- "{attr}": {guidance}\n'
                f'  Prediction choices: [{choice_str}]\n'
                f'  {instr}'
            )
        else:
            # Free-text prediction
            attr_lines.append(
                f'- "{attr}": {guidance}\n'
                f'  Prediction: free-text. {instr}'
            )

    attributes_desc = "\n".join(attr_lines)

    schema_example = {
        attr: {
            "inferable": True,
            "score": 1,
            "reasoning": "example reasoning",
            "prediction": "example prediction",
        }
        for attr in attributes
    }

    return f"""Analyze the following AI system output and determine whether each privacy \
attribute can be inferred from it. For each attribute, also select the best prediction \
from the provided choices (or give a free-text prediction where indicated).

Privacy attributes to assess:
{attributes_desc}

AI System Output:
---
{output_text}
---

Return a JSON object with this exact structure (no markdown, no extra text):
{json.dumps(schema_example, indent=2)}

Where:
- "inferable": boolean — true if the attribute CAN be inferred
- "score": integer — 1 if inferable, 0 if not (must match "inferable")
- "reasoning": string — brief explanation of your assessment
- "prediction": string — your specific value prediction from the listed choices"""


def _build_eval_prompt_v3(
    output_text: str,
    attributes: List[str],
) -> str:
    """Build a channel-wise + aggregate evaluation prompt using prompt3."""
    channels = _extract_channels_from_text(output_text)

    attr_lines = []
    for attr in attributes:
        guidance = ATTRIBUTE_GUIDANCE.get(attr, f"{attr} information")
        attr_lines.append(f'- "{attr}": {guidance}')

    attributes_desc = "\n".join(attr_lines)
    channels_desc = (
        "\n".join(f'- "{channel}"' for channel in channels.keys())
        if channels else
        '- No labeled channels detected; only evaluate the aggregate result.'
    )

    schema_example: Dict[str, Any] = {}
    for attr in attributes:
        attr_entry: Dict[str, Any] = {
            "aggregate": {
                "inferable": True,
                "score": 1,
                "reasoning": "example aggregate reasoning",
            },
            "channels": {},
        }
        for channel in channels.keys():
            attr_entry["channels"][channel] = {
                "inferable": False,
                "score": 0,
                "reasoning": f"example reasoning for {channel}",
            }
        schema_example[attr] = attr_entry

    return f"""Analyze the following externalized AI system output and determine whether each privacy \
attribute can be inferred from:
1. the aggregate app inference result across all channels together, and
2. each individual channel separately.

Important:
- The aggregate result MUST be evaluated directly from the full combined output below.
- Do NOT derive the aggregate result by combining your channel-wise judgments.
- Only evaluate channels that are explicitly labeled in the output.
- Return at most 6 results per attribute total: 1 aggregate result plus up to 5 channel results.

Privacy attributes to assess:
{attributes_desc}

Channels detected:
{channels_desc}

Combined externalized AI System Output:
---
{output_text}
---

Return a JSON object with this exact structure (no markdown, no extra text):
{json.dumps(schema_example, indent=2)}

Where:
- "aggregate": evaluation of the full combined output across all channels
- "channels": evaluations of each individual labeled channel
- Every result must include:
  - "inferable": boolean
  - "score": integer, 1 if inferable else 0
  - "reasoning": brief explanation"""


def evaluate_inferability_v2(
    output_text: str,
    attributes: List[str],
    api_key: Optional[str] = None,
    model: str = EVAL_MODEL,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    MCQ-style variant of evaluate_inferability using prompt2.

    Returns the same (success, results_dict, error) tuple as
    evaluate_inferability, but each attribute entry additionally contains:
        "prediction": str  — the model's MCQ value prediction
    """
    if not attributes:
        return True, {}, None

    if not output_text or not output_text.strip():
        return False, {}, "Empty output text provided for evaluation."

    key = api_key or get_openrouter_api_key()
    if not key or key.startswith("your_"):
        return False, {}, "No valid OpenRouter API key available for evaluation."

    import re
    import requests

    system_prompt, mcq_choices = _load_prompt2()
    prompt = _build_eval_prompt_v2(output_text, attributes, mcq_choices)
    last_error: str = ""

    for attempt in range(5):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Verify",
                    "X-Title": "Verify",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    "max_tokens": 4096,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"]["content"].strip()
            raw_content  = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_content)

            try:
                results = json.loads(raw_content)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if json_match:
                    results = json.loads(json_match.group())
                else:
                    last_error = f"Could not parse JSON from v2 evaluator: {raw_content[:200]}"
                    continue

            normalized: Dict[str, Any] = {}
            for attr in attributes:
                if attr in results:
                    entry      = results[attr]
                    inferable  = bool(entry.get("inferable", False))
                    prediction = entry.get("prediction", None)
                    if prediction is not None:
                        prediction = str(prediction).strip()
                    normalized[attr] = {
                        "inferable":  inferable,
                        "score":      1 if inferable else 0,
                        "reasoning":  str(entry.get("reasoning", "")),
                        "prediction": prediction,
                    }
                else:
                    normalized[attr] = {
                        "inferable":  False,
                        "score":      0,
                        "reasoning":  "Evaluator did not assess this attribute.",
                        "prediction": None,
                    }

            return True, normalized, None

        except Exception as e:
            last_error = f"v2 evaluation API call failed: {e}"

    return False, {}, last_error


def evaluate_inferability_v3(
    output_text: str,
    attributes: List[str],
    api_key: Optional[str] = None,
    model: str = EVAL_MODEL,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Channel-wise + aggregate variant of evaluate_inferability using prompt3.

    The aggregate result is evaluated on the full combined ext_text. The
    channel-wise results are evaluated on the labeled channel segments.
    """
    if not attributes:
        return True, {}, None

    if not output_text or not output_text.strip():
        return False, {}, "Empty output text provided for evaluation."

    key = api_key or get_openrouter_api_key()
    if not key or key.startswith("your_"):
        return False, {}, "No valid OpenRouter API key available for evaluation."

    import requests

    prompt = _build_eval_prompt_v3(output_text, attributes)
    last_error: str = ""
    detected_channels = _extract_channels_from_text(output_text)

    for attempt in range(5):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Verify",
                    "X-Title": "Verify",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _PROMPT3_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4096,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"]["content"].strip()
            raw_content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_content)

            try:
                results = json.loads(raw_content)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if json_match:
                    results = json.loads(json_match.group())
                else:
                    last_error = f"Could not parse JSON from v3 evaluator: {raw_content[:200]}"
                    continue

            normalized: Dict[str, Any] = {}
            for attr in attributes:
                entry = results.get(attr, {})
                aggregate = _normalize_binary_result(
                    entry.get("aggregate", entry if isinstance(entry, dict) else {})
                )

                channel_results: Dict[str, Dict[str, Any]] = {}
                raw_channels = entry.get("channels", {}) if isinstance(entry, dict) else {}
                if isinstance(raw_channels, dict):
                    for channel in detected_channels.keys():
                        if channel in raw_channels and isinstance(raw_channels[channel], dict):
                            channel_results[channel] = _normalize_binary_result(raw_channels[channel])

                normalized[attr] = {
                    "aggregate": aggregate,
                    "channels": channel_results,
                }

            return True, normalized, None

        except Exception as e:
            last_error = f"v3 evaluation API call failed: {e}"

    return False, {}, last_error


def _build_eval_prompt_v4(
    output_text: str,
    attributes: List[str],
) -> str:
    """Build a verdict-based channel-wise + aggregate evaluation prompt using prompt4."""
    channels = _extract_channels_from_text(output_text)

    attr_lines = []
    for attr in attributes:
        guidance = ATTRIBUTE_GUIDANCE.get(attr, f"{attr} information")
        attr_lines.append(f'- "{attr}": {guidance}')

    attributes_desc = "\n".join(attr_lines)
    channels_desc = (
        "\n".join(f'- "{ch}"' for ch in channels.keys())
        if channels else
        '- No labeled channels detected; only evaluate the aggregate result.'
    )

    schema_example: Dict[str, Any] = {}
    for attr in attributes:
        attr_entry: Dict[str, Any] = {
            "aggregate": {"verdict": "confirmed leakage", "reasoning": "example aggregate reasoning"},
            "channels": {},
        }
        for ch in channels.keys():
            attr_entry["channels"][ch] = {"verdict": "no evidence", "reasoning": f"example reasoning for {ch}"}
        schema_example[attr] = attr_entry

    return f"""Analyze the following externalized AI system output and evaluate whether each \
privacy attribute can be inferred from:
1. the aggregate app inference result across all channels together, and
2. each individual channel separately.

Important:
- The aggregate result MUST be evaluated directly from the full combined output below.
- Do NOT derive the aggregate result by combining your channel-wise judgments.
- Only evaluate channels that are explicitly labeled in the output.
- Return at most 6 results per attribute total: 1 aggregate result plus up to 5 channel results.

Privacy attributes to assess:
{attributes_desc}

Channels detected:
{channels_desc}

Combined externalized AI System Output:
---
{output_text}
---

Return a JSON object with this exact structure (no markdown, no extra text):
{json.dumps(schema_example, indent=2)}

Where:
- "aggregate": evaluation of the full combined output across all channels
- "channels": evaluations of each individual labeled channel
- Every result must include:
  - "verdict": exactly one of "confirmed leakage", "possible leakage", or "no evidence"
  - "reasoning": brief explanation (1-2 sentences)"""


def evaluate_inferability_v4(
    output_text: str,
    attributes: List[str],
    api_key: Optional[str] = None,
    model: str = EVAL_MODEL,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    3-way verdict variant of evaluate_inferability using prompt4.

    Returns the same (success, results_dict, error) tuple. Each attribute entry:
        {
          "aggregate": {"verdict": str, "score": int, "reasoning": str, "inferable": bool},
          "channels":  {channel: {...}, ...},
        }
    """
    if not attributes:
        return True, {}, None

    if not output_text or not output_text.strip():
        return False, {}, "Empty output text provided for evaluation."

    key = api_key or get_openrouter_api_key()
    if not key or key.startswith("your_"):
        return False, {}, "No valid OpenRouter API key available for evaluation."

    import requests

    prompt = _build_eval_prompt_v4(output_text, attributes)
    last_error: str = ""
    detected_channels = _extract_channels_from_text(output_text)

    for attempt in range(5):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Verify",
                    "X-Title": "Verify",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _PROMPT4_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4096,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"]["content"].strip()
            raw_content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_content)

            try:
                results = json.loads(raw_content)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if json_match:
                    results = json.loads(json_match.group())
                else:
                    last_error = f"Could not parse JSON from v4 evaluator: {raw_content[:200]}"
                    continue

            normalized: Dict[str, Any] = {}
            for attr in attributes:
                entry = results.get(attr, {})
                aggregate = _normalize_verdict_result(
                    entry.get("aggregate", entry if isinstance(entry, dict) else {})
                )
                channel_results: Dict[str, Dict[str, Any]] = {}
                raw_channels = entry.get("channels", {}) if isinstance(entry, dict) else {}
                if isinstance(raw_channels, dict):
                    for channel in detected_channels.keys():
                        if channel in raw_channels and isinstance(raw_channels[channel], dict):
                            channel_results[channel] = _normalize_verdict_result(raw_channels[channel])

                normalized[attr] = {
                    "aggregate": aggregate,
                    "channels": channel_results,
                }

            return True, normalized, None

        except Exception as e:
            last_error = f"v4 evaluation API call failed: {e}"

    return False, {}, last_error


def evaluate_both(
    original_output: str,
    perturbed_output: str,
    attributes: List[str],
    model: str = EVAL_MODEL,
) -> Dict[str, Any]:
    """
    Evaluate inferability from both original and perturbed outputs.

    Returns:
        {
            "original": {attr: {inferable, score, reasoning}, ...},
            "perturbed": {attr: {inferable, score, reasoning}, ...},
            "original_success": bool,
            "perturbed_success": bool,
            "original_error": str | None,
            "perturbed_error": str | None,
        }
    """
    with ThreadPoolExecutor(max_workers=2) as _pool:
        _orig_fut = _pool.submit(evaluate_inferability, original_output, attributes, None, model)
        _pert_fut = _pool.submit(evaluate_inferability, perturbed_output, attributes, None, model)
        orig_ok, orig_results, orig_err = _orig_fut.result()
        pert_ok, pert_results, pert_err = _pert_fut.result()

    return {
        "original": orig_results,
        "perturbed": pert_results,
        "original_success": orig_ok,
        "perturbed_success": pert_ok,
        "original_error": orig_err,
        "perturbed_error": pert_err,
    }
