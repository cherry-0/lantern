"""
Unified LLM client for the Verify pipeline.

When INFER_LOCAL=false (default): routes to OpenRouter. Requires OPENROUTER_API_KEY.
When INFER_LOCAL=true:           routes to local Ollama.
  - Auto-selects gemma4:26b when ≥8 GB VRAM/RAM is available.
  - Falls back to gemma4:e4b on constrained hardware.
  - Override with LOCAL_MODEL_NAME or per-pipeline env vars.
"""

import json
import re
from typing import Any, Dict, List, Optional

from verify.backend.utils.verbose_log import log_llm_call

_LOCAL_MODEL_LARGE = "gemma4:26b"
_LOCAL_MODEL_SMALL = "gemma4:e4b"
_LOCAL_VRAM_THRESHOLD_MB = 8192
_OLLAMA_BASE = "http://localhost:11434/v1"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _get_available_memory_mb() -> int:
    """
    Return available GPU/unified memory in MB for local model selection.

    - NVIDIA: queries free VRAM via nvidia-smi.
    - Apple Silicon: uses 55% of total unified RAM as a conservative estimate
      (leaves headroom for OS + other processes).
    - Returns 0 on any failure (caller falls back to small model).
    """
    import subprocess

    # NVIDIA GPU
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            values = [int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip()]
            if values:
                return max(values)
    except Exception:
        pass

    # Apple Silicon (unified memory)
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            total_bytes = int(r.stdout.strip())
            return int(total_bytes * 0.55 / (1024 * 1024))
    except Exception:
        pass

    return 0


def pick_local_model(explicit_model: Optional[str] = None) -> str:
    """
    Resolve which local model to use.

    Priority: explicit_model arg > LOCAL_MODEL_NAME env var > VRAM auto-select.
    OpenRouter model IDs (containing "/") are ignored — they don't work with Ollama.
    """
    if explicit_model and "/" not in explicit_model:
        return explicit_model
    from verify.backend.utils.config import get_env
    env_model = get_env("LOCAL_MODEL_NAME")
    if env_model:
        return env_model
    available_mb = _get_available_memory_mb()
    chosen = _LOCAL_MODEL_LARGE if available_mb >= _LOCAL_VRAM_THRESHOLD_MB else _LOCAL_MODEL_SMALL
    return chosen


def call_llm(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    image_b64: Optional[str] = None,
    response_format_json: bool = False,
    max_tokens: int = 2048,
    timeout: int = 90,
) -> str:
    """
    Make a single chat completion call and return the response text.

    Args:
        messages:             OpenAI-style messages list.
        model:                Model override. When None: uses env var or auto-select.
        image_b64:            Base64 JPEG to attach to the last user message (vision).
        response_format_json: Request JSON-object output (supported by both backends).
        max_tokens:           Token limit for the response.
        timeout:              HTTP request timeout in seconds.

    Returns:
        Raw response text (stripped).

    Raises:
        RuntimeError: on API error or missing credentials.
    """
    import requests
    from verify.backend.utils.config import is_infer_local, get_openrouter_api_key, get_env

    # Attach vision content to the last user message if an image is provided
    if image_b64:
        msgs = list(messages)
        last = msgs[-1]
        if last["role"] == "user":
            content = last["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            else:
                content = list(content)
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            )
            msgs[-1] = {"role": "user", "content": content}
        messages = msgs

    if is_infer_local():
        chosen_model = pick_local_model(model)
        base_url = (get_env("LOCAL_MODEL_URL") or _OLLAMA_BASE).rstrip("/")
        url = base_url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        body: Dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            body["response_format"] = {"type": "json_object"}
    else:
        api_key = get_openrouter_api_key()
        if not api_key or api_key.startswith("your_"):
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Set it in .env or set INFER_LOCAL=true to use a local model."
            )
        chosen_model = model or get_env("OPENROUTER_DEFAULT_MODEL") or "google/gemini-2.0-flash-001"
        url = _OPENROUTER_BASE + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Verify",
            "X-Title": "Verify",
        }
        body = {
            "model": chosen_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            body["response_format"] = {"type": "json_object"}

    _prompt_preview = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            _prompt_preview = c if isinstance(c, str) else str(c)
            break

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        log_llm_call(
            backend="local" if is_infer_local() else "openrouter",
            model=chosen_model,
            prompt_preview=_prompt_preview,
            response_preview="",
            ok=False,
            error=str(e),
        )
        raise RuntimeError(f"LLM call failed ({url}): {e}") from e

    _response_text = resp.json()["choices"][0]["message"]["content"].strip()
    log_llm_call(
        backend="local" if is_infer_local() else "openrouter",
        model=chosen_model,
        prompt_preview=_prompt_preview,
        response_preview=_response_text,
        ok=True,
    )
    return _response_text


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from LLM response text.

    Strips stray control characters (common in Gemma/Gemini responses) and falls
    back to regex extraction when the model wraps JSON in markdown fences.
    """
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No JSON object found in LLM response: {text[:300]}")
