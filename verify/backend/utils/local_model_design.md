# Local Model Integration — Design Proposal

Author: planner-3 (team `local-model-pipeline`)
Scope: Unify the two privacy-pipeline LLM entry points (perturbation + evaluation) behind a single backend-agnostic client so the same pipeline can run against OpenRouter, a local Ollama server, or a local `llama.cpp` server without code changes at the call sites.

---

## 1. Problem summary

Today the pipeline has **three** LLM integration points talking to OpenRouter:

| Caller | Purpose | Structured JSON needed? |
|---|---|---|
| `verify/backend/perturbation_method/PrivacyLens-Prompt/main.py` | Rewrites text with private attributes removed | No (plain text) |
| `verify/backend/evaluation_method/evaluator.py` | LLM-as-judge for inferability | Yes (`response_format={"type": "json_object"}`) |
| `verify/backend/adapters/base.py::BaseAdapter._call_openrouter` | Serverless fallback for target-app adapters (vision + text) | Optional (`extra_body`) |

All three hard-code `https://openrouter.ai/api/v1/chat/completions`, duplicate header/auth scaffolding, and treat the API key the same way. There is **no single seam** to swap the backend.

We want privacy-pipeline evaluations to be reproducible on a laptop/GPU box without an external API (budget, data-handling constraints, and offline CI).

---

## 2. Goals / Non-goals

**Goals**
1. One module (`verify/backend/utils/llm_client.py`) exposes a single `call_llm()` entry point used by perturbation + evaluation (and, later, by the adapter base).
2. Swap backend with one env var (`MODEL_BACKEND`) — no code edits required for a run.
3. Honor structured-JSON semantics uniformly across backends (not just OpenRouter).
4. Stay compatible with the adapter `_call_openrouter` capture mechanism so externalization reports still fire.
5. Zero new runtime dependencies — continue to use `requests`.

**Non-goals**
- Replacing the per-adapter env injection (`_inject_openrouter_env`). That layer targets *target-app processes* that read `OPENAI_*` env vars at import time; that is a different concern and stays untouched in this proposal.
- Vision support for local backends in phase 1. Text-only first; adapter-layer vision stays on OpenRouter until a local VLM path is in scope.
- Streaming. Both privacy callers use non-streaming responses.

---

## 3. Backend survey

### 3.1 Ollama (recommended default for local)
- OpenAI-compatible chat endpoint at `http://localhost:11434/v1/chat/completions` (same request/response shape as OpenAI, no API key).
- Native endpoint at `http://localhost:11434/api/chat` accepts `format: "json"` for structured output; **this is the well-tested path** for JSON. The OpenAI-compat `response_format={"type":"json_object"}` is accepted on recent Ollama (>= 0.1.30) and forwards to the same constrained-decoding path.
- Model list via `ollama pull <name>`; already installed models are named `llama3.1:8b`, `qwen2.5:7b`, `mistral-nemo:12b`, etc.
- Limitation: JSON mode is honored but **no JSON Schema enforcement** — the evaluator's existing regex/JSON-retry fallback is still required.

### 3.2 llama.cpp server
- `llama-server` ships an OpenAI-compatible `/v1/chat/completions` endpoint (default port 8080).
- Structured output is supported via the `grammar` (GBNF) or `json_schema` body field; the `response_format={"type":"json_object"}` hint is also accepted and internally compiled to a permissive JSON grammar.
- Typically one model per server — `LOCAL_MODEL_NAME` is informational; the model is fixed by how `llama-server` was launched.
- Pros: lowest-level control, best quantization options, no daemon/registry layer.
- Cons: user must start the server themselves; not drop-in like Ollama.

### 3.3 OpenRouter (existing)
- Remote, hosted. Unchanged behavior. Default when `MODEL_BACKEND` is unset.

### 3.4 Recommended local models (privacy text analysis)
All are permissive-license, tool-friendly, and run on a single 16–24 GB GPU or Apple Silicon:

| Model | Ollama tag | Strengths | Notes |
|---|---|---|---|
| **Llama 3.1 8B Instruct** | `llama3.1:8b` | Strong general reasoning, reliable JSON mode | **Recommended default** for both perturbation and evaluation |
| **Qwen 2.5 7B Instruct** | `qwen2.5:7b` | Best-in-class JSON/structured output at this size | Good evaluator choice; tends to be more literal than Llama for paraphrase |
| **Mistral Nemo 12B** | `mistral-nemo:12b` (or `mistral-nemo`) | Very strong instruction following, 128k context | Good perturbation choice when input text is long; needs ~16 GB VRAM |
| Llama 3.1 70B (optional) | `llama3.1:70b` | Closest to OpenRouter-tier quality | Only worth it if a big GPU is available; much slower |

Evaluator is the more sensitive of the two (LLM-as-judge drives the headline metric), so users running locally should prefer Qwen 2.5 7B or Llama 3.1 8B for evaluation and can use a smaller/faster model for perturbation if needed. To split backends per-stage see §6 (future extension).

---

## 4. `llm_client.py` — proposed interface

Single public function. Everything else is backend-internal.

```python
# verify/backend/utils/llm_client.py
"""
Backend-agnostic LLM client for the Verify privacy pipeline.

Supports three backends:
    openrouter — https://openrouter.ai/api/v1 (default)
    ollama     — http://localhost:11434/v1 (OpenAI-compat)
    llamacpp   — http://localhost:8080/v1  (OpenAI-compat)

Selected via the MODEL_BACKEND env var. All three speak the OpenAI chat
format, so a single requests.post call serves them. Differences are
confined to: base URL, auth header, and how structured JSON is requested.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from verify.backend.utils.config import (
    get_env,
    get_openrouter_api_key,
    get_model_backend,
    get_local_model_url,
    get_local_model_name,
)


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_LOCAL_MODEL      = "llama3.1:8b"

# Backend → (base_url, needs_api_key)
_BACKEND_DEFAULTS = {
    "openrouter": ("https://openrouter.ai/api/v1", True),
    "ollama":     ("http://localhost:11434/v1",    False),
    "llamacpp":   ("http://localhost:8080/v1",     False),
}


@dataclass
class LLMCall:
    """Record of a single call — compatible with BaseAdapter externalizations."""
    backend: str
    model: str
    base_url: str
    status: int
    prompt_preview: str
    response_preview: str
    has_image: bool = False


class LLMError(RuntimeError):
    pass


# ── Public API ────────────────────────────────────────────────────────────────

def call_llm(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    response_format_json: bool = False,
    max_tokens: int = 2048,
    timeout: int = 90,
    extra_body: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    record: Optional[List[LLMCall]] = None,
) -> str:
    """
    Send a chat-completion request to the configured backend and return the
    assistant text.

    Args:
        messages: OpenAI-style [{"role": ..., "content": ...}, ...].
        model:    Model name. When None, resolved from env per backend.
        response_format_json: If True, ask the backend to emit JSON.
                  - OpenRouter: response_format={"type": "json_object"}
                  - Ollama/llamacpp: same hint + backend-specific fallback
                    (Ollama honors the OpenAI hint; llamacpp compiles it
                    into a permissive JSON grammar).
        max_tokens: Upper bound on response length.
        timeout:  Seconds.
        extra_body: Passthrough merged into the request body (e.g. temperature).
        backend:  Override (otherwise resolve_backend() from env).
        record:   Optional list to append an LLMCall entry to (for
                  externalization capture, mirrors BaseAdapter._openrouter_calls).

    Returns:
        The assistant response string, control chars stripped.

    Raises:
        LLMError: if misconfigured or the request fails.
    """
    backend = (backend or get_model_backend() or "openrouter").lower()
    if backend not in _BACKEND_DEFAULTS:
        raise LLMError(f"Unknown MODEL_BACKEND={backend!r}; "
                       f"expected one of {list(_BACKEND_DEFAULTS)}")

    base_url, needs_key = _BACKEND_DEFAULTS[backend]
    if backend != "openrouter":
        base_url = get_local_model_url() or base_url

    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if backend == "openrouter":
        api_key = get_openrouter_api_key()
        if not api_key or api_key.startswith("your_"):
            raise LLMError("No valid OPENROUTER_API_KEY configured.")
        headers["Authorization"] = f"Bearer {api_key}"
        headers["HTTP-Referer"]  = "https://github.com/Verify"
        headers["X-Title"]       = "Verify"
    elif needs_key:
        # Reserved for future authenticated local backends (e.g. vLLM with a token)
        token = get_env("LOCAL_MODEL_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"

    # Resolve model per backend
    used_model = model or (
        DEFAULT_OPENROUTER_MODEL if backend == "openrouter"
        else (get_local_model_name() or DEFAULT_LOCAL_MODEL)
    )

    body: Dict[str, Any] = {
        "model": used_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        body["response_format"] = {"type": "json_object"}
        # llamacpp also accepts a top-level `json_schema` but the permissive
        # response_format hint is sufficient for our flat {attr: {...}} output.
    if extra_body:
        body.update(extra_body)

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        raise LLMError(f"{backend} request failed: {e}") from e
    except (KeyError, ValueError) as e:
        raise LLMError(f"{backend} returned unparseable body: {e}") from e

    # Strip control chars (already needed for Gemini on OpenRouter; cheap to keep)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()

    if record is not None:
        preview_prompt = _preview_from_messages(messages)
        record.append(LLMCall(
            backend=backend,
            model=used_model,
            base_url=base_url,
            status=resp.status_code,
            prompt_preview=preview_prompt[:200],
            response_preview=cleaned[:200],
            has_image=_messages_have_image(messages),
        ))
    return cleaned


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON parse used by the evaluator. Same recovery logic that is
    currently inline in evaluator.py: try strict first, then regex-extract the
    first {...} block.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


# ── Internal helpers ──────────────────────────────────────────────────────────

def _preview_from_messages(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


def _messages_have_image(messages: List[Dict[str, Any]]) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False
```

Key properties:
- **Single function** `call_llm(messages, model=..., response_format_json=..., max_tokens=..., timeout=..., extra_body=..., backend=..., record=...)`. Matches the signature the task spec asked for, with explicit-keyword extras (`extra_body`, `record`) needed for integration with existing call sites.
- **Backend resolution** is a one-liner (`get_model_backend()`), but callers may override via the `backend=` kwarg for tests and for a future per-stage split.
- **record** hook lets the adapter base class keep its externalization capture (`self._openrouter_calls`) working — it becomes `self._llm_calls` and appends `LLMCall` dataclass instances, keeping the existing `_build_serverless_externalizations` semantics.
- **No vision yet for local backends**: the function accepts image messages today (OpenRouter uses them), and `_messages_have_image` records the fact; local backends will just forward the message to the server and rely on the server rejecting if it is not a VLM. Adding real local VLM support is a later ticket.

---

## 5. Config additions — `verify/backend/utils/config.py`

Three new helpers, colocated with the existing `get_openrouter_api_key`:

```python
# Append to verify/backend/utils/config.py

def get_model_backend() -> str:
    """
    Return the configured LLM backend: "openrouter" | "ollama" | "llamacpp".
    Defaults to "openrouter" for backward compatibility.
    """
    val = (get_env("MODEL_BACKEND", "openrouter") or "openrouter").strip().lower()
    if val not in {"openrouter", "ollama", "llamacpp"}:
        # Unknown backend → fall back to openrouter rather than crashing at import.
        return "openrouter"
    return val


def get_local_model_url() -> Optional[str]:
    """
    Base URL for the local LLM server (OpenAI-compatible chat API).
    Default depends on backend:
        ollama    → http://localhost:11434/v1
        llamacpp  → http://localhost:8080/v1
    Override with LOCAL_MODEL_URL.
    """
    explicit = get_env("LOCAL_MODEL_URL")
    if explicit:
        return explicit.rstrip("/")
    backend = get_model_backend()
    return {
        "ollama":   "http://localhost:11434/v1",
        "llamacpp": "http://localhost:8080/v1",
    }.get(backend)


def get_local_model_name() -> Optional[str]:
    """
    Model name passed to the local backend. Informational for llama.cpp
    (which serves a single preloaded model) and meaningful for Ollama
    (which dispatches by name).
    """
    return get_env("LOCAL_MODEL_NAME")
```

**Env var spec (documentation to add to a project README or `.env` example):**

| Variable | Required when | Default | Example |
|---|---|---|---|
| `MODEL_BACKEND` | Always optional | `openrouter` | `ollama` |
| `OPENROUTER_API_KEY` | `MODEL_BACKEND=openrouter` | — | `sk-or-v1-...` |
| `LOCAL_MODEL_URL` | Optional override | See table above | `http://gpu-box.lan:11434/v1` |
| `LOCAL_MODEL_NAME` | Recommended for local backends | — | `llama3.1:8b` |
| `LOCAL_MODEL_API_KEY` | Optional, reserved for auth'd local (e.g. vLLM) | — | `sk-local-...` |

---

## 6. Migration — how callers change

### 6.1 Perturbation (`perturbation_method/PrivacyLens-Prompt/main.py`)

Current code (lines 85–102) is 18 lines of `requests.post` bolierplate + `choices[0].message.content` unwrap. After migration:

```python
from verify.backend.utils.llm_client import call_llm, LLMError

# ...inside perturb(), replace the try/except block:
try:
    perturbed_text = call_llm(
        messages=[{"role": "user", "content": prompt}],
        # Model defaults to DEFAULT_OPENROUTER_MODEL or DEFAULT_LOCAL_MODEL
        # per backend. Callers that want a specific OpenRouter model can pass
        # model="google/gemini-2.0-flash-001" explicitly.
        max_tokens=2048,
        timeout=90,
    )
except LLMError as e:
    return False, input_item, f"PrivacyLens API call failed: {e}"
```

`check_availability` relaxes: when `MODEL_BACKEND != "openrouter"`, an API key is not required, and we instead check that `LOCAL_MODEL_URL` resolves and `requests.get(f"{base}/models", timeout=3)` returns 2xx (soft check — skip if it fails with a warning rather than failing the whole run).

### 6.2 Evaluation (`evaluation_method/evaluator.py`)

The evaluator needs JSON structured output + retries. Refactor the inner loop of `evaluate_inferability` to use `call_llm(..., response_format_json=True)` and `parse_json_response`:

```python
from verify.backend.utils.llm_client import call_llm, parse_json_response, LLMError

# ...inside evaluate_inferability(), the for-attempt loop becomes:
for attempt in range(5):
    try:
        raw = call_llm(
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            model=model,  # caller still picks; defaults resolved in call_llm
            response_format_json=True,
            max_tokens=4096,
            timeout=60,
        )
        try:
            results = parse_json_response(raw)
        except json.JSONDecodeError:
            last_error = f"Could not parse JSON from evaluator response: {raw[:200]}"
            continue
        # existing normalization block unchanged
        ...
        return True, normalized, None
    except LLMError as e:
        last_error = str(e)
return False, {}, last_error
```

The `EVAL_MODEL` constant stays as the OpenRouter default; when `MODEL_BACKEND=ollama` the constant becomes a no-op (caller's `model=` override is ignored only if the resolved backend's default is used — i.e. if the evaluator explicitly passes the old OpenRouter name, we need to let it fall through). To keep this clean, drop the `model=model` argument when `get_model_backend() != "openrouter"`, or better: treat `EVAL_MODEL` as "the OpenRouter model name" and let `call_llm` resolve a local default when the backend is local. The cleanest rule is **in `call_llm`, ignore an explicit `model=` that clearly isn't valid for the resolved backend** — but that adds magic. Simpler rule for phase 1: **`model=None` when backend is local**, set explicitly via `LOCAL_MODEL_NAME`. Applied in the evaluator as:

```python
used_model = model if get_model_backend() == "openrouter" else None
raw = call_llm(messages=..., model=used_model, response_format_json=True, ...)
```

### 6.3 Adapter base (`adapters/base.py::_call_openrouter`) — optional phase 2

Not required by the task, but recommended to keep the three call sites consistent. `_call_openrouter` becomes a thin wrapper that builds the vision message and delegates to `call_llm(..., record=self._llm_calls)`. Rename `_openrouter_calls` → `_llm_calls` and keep a property alias so existing code paths still work. Because the 14 target-app adapters go through this one method, this change is O(1) in adapter count.

### 6.4 Step-by-step migration plan

1. **Add helpers** to `verify/backend/utils/config.py` (`get_model_backend`, `get_local_model_url`, `get_local_model_name`). Zero behavior change when env vars are unset.
2. **Create** `verify/backend/utils/llm_client.py` per §4. Pure addition, no callers yet.
3. **Unit tests** for `llm_client` — mock `requests.post` per backend, assert correct URL/headers/body shape and that `response_format_json=True` sets the right field. Covers the three backends without needing a local server.
4. **Migrate `evaluator.py`** first (it is the more sensitive caller). Add the `record=` plumbing only if we want externalization of evaluator calls — I recommend **not** doing that in phase 1 to keep evaluator output invisible to the inferability judge.
5. **Migrate `PrivacyLens-Prompt/main.py`**. Same pattern; no `record=`.
6. **Smoke test end-to-end** with `MODEL_BACKEND=openrouter` (regression) and `MODEL_BACKEND=ollama` + `ollama run llama3.1:8b` (new path). Verify reports come out shaped identically.
7. **(Phase 2)** Refactor `BaseAdapter._call_openrouter` to delegate. This also lets the adapter serverless fallback run against a local model — useful for cost-free bulk runs.
8. **Document** env vars in `verify/README.md` (or `CLAUDE.md`'s verify section) with the table from §5.

---

## 7. Structured JSON — known caveats

- **Ollama JSON mode (`response_format={"type":"json_object"}`)**: constrains output to *syntactically valid* JSON but not to a specific schema. The evaluator already has a shape-normalization pass (`normalized[attr] = {...}` with per-attr defaults), so schema drift is absorbed. No change needed.
- **llama.cpp**: accepts the OpenAI hint; internally compiles a permissive JSON grammar. For stricter enforcement one can pass `extra_body={"grammar": "..."}` with a GBNF grammar or `extra_body={"json_schema": {...}}` in future phases. Not required for the flat per-attribute schema the evaluator uses.
- **Small-model failure mode**: 7–8B models occasionally emit a truncated JSON if `max_tokens` is hit mid-object. The evaluator's existing "retry up to 5x, then regex-extract" path already covers this; local runs may want to bump `max_tokens` to 8192 (safe for all listed models' context windows).
- **Prompt header stripping**: Llama 3.1 sometimes prefixes responses with `Here is the JSON:`. The evaluator's `re.search(r"\{.*\}", raw, re.DOTALL)` fallback handles this already — one more reason to port that logic into `parse_json_response`.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Local model quality regresses inferability scores vs. OpenRouter | Document recommended models; allow per-stage backend split by reading `MODEL_BACKEND_EVAL` / `MODEL_BACKEND_PERTURB` in phase 2. Keep OpenRouter as the default for headline numbers. |
| Local server not running when pipeline starts | `call_llm` surfaces the connection error as `LLMError`; `check_availability` in perturbation can probe `/v1/models` and return a clear message. |
| Response-shape drift (e.g. Ollama returns different finish_reason) | We only read `choices[0].message.content`, which is identical across all three backends' OpenAI-compat paths. |
| Adapter externalization reports change format | Phase 1 does not touch adapters. Phase 2 rename `_openrouter_calls` → `_llm_calls` is a mechanical find/replace; `_build_serverless_externalizations` already iterates a list of dicts, so just update the label format. |

---

## 9. Summary of deliverables

- **New file**: `verify/backend/utils/llm_client.py` (see §4).
- **Edits**: `verify/backend/utils/config.py` — 3 helpers appended (see §5).
- **Edits**: `verify/backend/evaluation_method/evaluator.py` — swap `requests.post` loop for `call_llm` + `parse_json_response` (see §6.2).
- **Edits**: `verify/backend/perturbation_method/PrivacyLens-Prompt/main.py` — swap `requests.post` block for `call_llm` (see §6.1).
- **Optional phase 2**: `verify/backend/adapters/base.py::_call_openrouter` delegates to `call_llm` with `record=self._llm_calls`.
- **Env vars**: `MODEL_BACKEND`, `LOCAL_MODEL_URL`, `LOCAL_MODEL_NAME` (+ existing `OPENROUTER_API_KEY`).
- **Recommended local models**: `llama3.1:8b` (default), `qwen2.5:7b` (best JSON), `mistral-nemo:12b` (long context) — all runnable on a single 16–24 GB GPU.
