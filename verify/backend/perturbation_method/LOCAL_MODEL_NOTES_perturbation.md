# Perturbation Pipeline — Local Model Integration Notes

Scope: findings for planner task #1. Covers the three perturbation modules
(`PrivacyLens-Prompt`, `Simple_Blur`, `Imago_Obscura`), the dispatcher
(`__init__.py` + `interface.py`), and what a minimum local-model-agnostic
refactor would look like.

---

## 1. Registration & Dispatch

The perturbation package is discovered **filesystem-first**, not via a registry.

- `verify/backend/perturbation_method/__init__.py` is a stub:
  ```python
  # Perturbation method package
  ```
  No imports, no registry.

- The real dispatcher lives in `verify/backend/perturbation_method/interface.py`:
  - `PERTURBATION_DIR = BACKEND_DIR / "perturbation_method"` (line 18)
  - `_load_method_module(method_name)` (lines 21-41): uses
    `importlib.util.spec_from_file_location` to import
    `<PERTURBATION_DIR>/<method_name>/main.py` on demand.
  - `list_methods_for_modality(modality)` (lines 44-76): scans the directory
    and filters by keyword heuristic — names containing `blur`/`obscura` are
    image-only; names containing `privacylens`/`prompt` are text-only.
  - `get_perturbation_method(modality)` (lines 79-85) pulls the **default**
    method for a modality from `verify/config/perturbation_method.csv`
    (parsed by `load_perturbation_method_map()` in `utils/config.py`).
  - `check_perturbation_availability(...)` (lines 88-113) dynamically imports
    the module and calls its optional `check_availability()`.
  - `run_perturbation(input_item, modality, attributes, method_name=None, **kwargs)`
    (lines 116-171) is the single entry point orchestrators call. It passes
    `**kwargs` to the module's `perturb()` — so new per-method kwargs (e.g.
    `base_url`, `model`) flow through without changing the dispatcher.

Implication for local-model work: **no central registry needs editing.**
Each `main.py` is self-contained; the dispatcher already supports threading
extra kwargs to `perturb()`.

---

## 2. Hardcoded Model / API References

### 2.1 `PrivacyLens-Prompt/main.py` (text perturbation)
| Where | Value | Line |
|---|---|---|
| Endpoint URL | `https://openrouter.ai/api/v1/chat/completions` | 87 |
| Model ID | `google/gemini-2.0-flash-001` (hardcoded, not `OPENROUTER_DEFAULT_MODEL`) | 95 |
| Auth header | `Authorization: Bearer {api_key}` from `get_openrouter_api_key()` | 89 |
| Branding headers | `HTTP-Referer: https://github.com/Verify`, `X-Title: Verify` | 91-92 |
| Body shape | OpenAI chat completions (`messages=[{role:user, content:<str>}]`, `max_tokens=2048`) | 94-98 |
| Timeout | 90s | 99 |
| Response parse | `resp.json()["choices"][0]["message"]["content"]` | 102 |
| Availability gate | `OPENROUTER_API_KEY` present and not prefixed `your_` | 21-23 |

Prompt is loaded from YAML: `PrivacyLens-Prompt/prompt/prompt_privacylens.yaml`
(attribute descriptions + system-prompt template filled with
`attr_list_str` + `text_content`).

### 2.2 `Simple_Blur/main.py` (image perturbation)
| Where | Value | Line |
|---|---|---|
| Endpoint URL | `https://openrouter.ai/api/v1/chat/completions` | 77 |
| Model ID | `OPENROUTER_DEFAULT_MODEL` imported from `verify/backend/adapters/base.py` (currently `google/gemini-2.5-pro`) | 53, 85 |
| Auth header | `Authorization: Bearer {api_key}` from `get_openrouter_api_key()` | 75, 79 |
| Branding headers | `HTTP-Referer: https://github.com/Verify`, `X-Title: Verify` | 81-82 |
| Body shape | OpenAI chat completions with multimodal content (text + `image_url` data-URL of base64 JPEG), `max_tokens=512` | 84-94 |
| Timeout | 60s | 95 |
| Response parse | `resp.json()["choices"][0]["message"]["content"]`, then regex `\[.*\]` to extract JSON array | 98-105 |
| Availability gate | `OPENROUTER_API_KEY` present and not prefixed `your_` | 37-39 |

The prompt is a baked-in string (lines 61-72) — asks the VLM to return
normalized bounding boxes of sensitive regions. If detection fails or returns
nothing, the code falls back to a full-image `ImageFilter.GaussianBlur`
(PIL), which does **not** require any model. `BLUR_RADIUS = 30` is a module
constant (line 27).

### 2.3 `Imago_Obscura/main.py` (image perturbation)
No OpenRouter calls. Runs entirely locally via Florence-2 (`torch` +
`transformers`) phrase grounding, delegated to
`headless.pipeline.run(...)` under `Imago_Obscura/src/headless/`. Falls back
to full-image blur when `torch`/`transformers` are absent. Env toggles:
`IMAGO_OBSCURA_MODE` (`blur|pixelate|fill`) and `IMAGO_OBSCURA_BLUR_RADIUS`.
**Already model-agnostic from an API standpoint — only needs local compute.**

(Note: `Imago_Obscura/src/imago-obscura-study-prototype/image_manipulation/opanai_agent.py`
line 17 hardcodes `gpt-4o-2024-08-06`, but that file is part of the vendored
upstream prototype and is not reached by the headless `run()` path used in
`perturb()`.)

---

## 3. Config / Env Vars Currently Used

From `verify/backend/utils/config.py`:
- `OPENROUTER_API_KEY` (or fallback `OPENROUTER_KEY`) — read by
  `get_openrouter_api_key()`; both perturbation modules that call
  OpenRouter gate availability on this.
- `.env` is loaded from `LANTERN_ROOT / ".env"` (the repo root), not via
  python-dotenv — a minimal inline parser in `get_env()`.
- Other unrelated toggles also live here: `USE_APP_SERVERS`, `DEBUG`,
  `MALICIOUS_PROMPT_MODE`, `OPENAI_API_KEY`.

From `verify/backend/adapters/base.py`:
- Constant `OPENROUTER_DEFAULT_MODEL = "google/gemini-2.5-pro"` (line 13) —
  reused by `Simple_Blur` for its VLM call. `PrivacyLens-Prompt` does **not**
  use this constant and hardcodes a different model.
- `BaseAdapter._inject_openrouter_env()` (lines 130-162) sets
  `OPENAI_API_KEY`, `OPENAI_BASE_URL=https://openrouter.ai/api/v1`,
  `LLM_API_KEY`, `LLM_HOST`, `VLM_API_KEY`, `VLM_API_URL` into the process
  environment for target-app subprocesses. This is for **adapters**, not
  perturbation methods — but it's the same pattern we'd want to extend to
  point at a local server.

No perturbation-specific env vars exist today other than Imago_Obscura's two.

---

## 4. `perturb()` Function Signature

All three modules expose the same contract (with one extra kwarg on
Imago_Obscura):

```python
def perturb(
    input_item: Dict[str, Any],
    attributes: List[str],
    # Imago_Obscura additionally accepts:
    # mode: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
```

**Input** `input_item`:
- Text: `text_content: str` (plus other dataset fields).
- Image: `data: PIL.Image` and/or `path: str`; optional `image_base64`.
- Also carries `filename` used for logging.

**Output** `(success, perturbed_item, error_message)`:
- `perturbed_item` is a shallow copy of `input_item` with mutated fields:
  - Text: `text_content`, `data` both set to the rewritten string.
  - Image: `data` (PIL), `image_base64` (re-encoded JPEG).
- Always adds `perturbed_item["perturbation_applied"]` metadata dict
  (method name, attributes, and method-specific extras like
  `mode`/`regions`/`blur_radius`).

**Dispatcher (`run_perturbation`) threads `**kwargs` through to `perturb()`**,
so adding optional params like `base_url=...` or `model=...` would not
break any caller.

---

## 5. What a Local Model Adapter Would Look Like

Both OpenRouter-using perturbation methods hit the **OpenAI-compatible
chat completions endpoint** (`POST /v1/chat/completions`). Ollama exposes
the identical shape at `http://localhost:11434/v1/chat/completions`
(the OpenAI-compat layer) — so bodies and response parsing don't change.
vLLM, LM Studio, llama.cpp server, and Text Generation WebUI all expose
the same `/v1/chat/completions` contract.

### 5.1 Minimum viable interface

Add a single helper that both modules call instead of hand-rolling a
`requests.post`:

```python
# proposed: verify/backend/utils/llm_client.py
def chat_completion(
    prompt: str,
    image_b64: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,    # NEW
    api_key: Optional[str] = None,     # NEW, optional for local servers
    max_tokens: int = 1024,
    timeout: int = 60,
) -> str:
    """
    Route to OpenRouter or a local OpenAI-compatible server based on
    LLM_BACKEND env var (or base_url kwarg override).
    Returns the assistant text.
    """
```

Resolution order:
1. Explicit `base_url`/`model` kwargs (per-call override).
2. `LLM_BACKEND` env: one of `openrouter`, `ollama`, `openai_compat`.
3. `LLM_BASE_URL` env — free-form base URL (e.g. `http://localhost:11434/v1`).
4. `LLM_MODEL` env — model identifier (e.g. `llama3.2-vision:11b`,
   `qwen2.5:7b`).
5. Fallback to OpenRouter with `OPENROUTER_API_KEY` + `OPENROUTER_DEFAULT_MODEL`.

Auth: skip the `Authorization` header when no `api_key` is resolved (local
Ollama does not require one). OpenRouter-specific `HTTP-Referer` /
`X-Title` headers should only be set on the OpenRouter branch.

Vision-capability flag: both backends can handle `image_url` content parts
as long as the model is multimodal. For local models this means the caller
must choose a vision-capable model (e.g. `llama3.2-vision`, `llava`,
`qwen2.5-vl`). The helper need not know — it just forwards the content
array. We should, however, surface a clearer error if the server returns
a 400 "model does not support images" so users don't get silent full-image
blur fallback.

### 5.2 Minimum changes to each perturbation module

**`Simple_Blur/main.py`:**
- Replace the `requests.post(...)` block in `_detect_regions()` (lines 74-97)
  with a call to `chat_completion(prompt, image_b64=image_b64, ...)`.
- Drop the hardcoded `OPENROUTER_DEFAULT_MODEL` import in favor of
  reading from the helper (still allow override via `kwargs`).
- Broaden `check_availability()` — "no key" should only be fatal in the
  `openrouter` branch.

**`PrivacyLens-Prompt/main.py`:**
- Replace the `requests.post(...)` block (lines 86-104) with
  `chat_completion(prompt, ...)`.
- Replace the hardcoded `google/gemini-2.0-flash-001` with
  `model=os.environ.get("PERTURBATION_TEXT_MODEL")` → helper default.
- Same availability relaxation.

**`Imago_Obscura/main.py`:** no code change needed; it already runs local.

**`interface.py` / `__init__.py`:** no changes. `**kwargs` already flows
through.

### 5.3 Config knobs to add

New env vars (all optional, documented in `.env.example`):
- `LLM_BACKEND` — `openrouter` | `ollama` | `openai_compat` (default: `openrouter` for backward compat).
- `LLM_BASE_URL` — overrides backend default (e.g. `http://localhost:11434/v1`).
- `LLM_API_KEY` — optional; falls through to `OPENROUTER_API_KEY` when `LLM_BACKEND=openrouter`.
- `PERTURBATION_TEXT_MODEL` — overrides default for PrivacyLens.
- `PERTURBATION_VISION_MODEL` — overrides default for Simple_Blur region detection.
- `LLM_REQUEST_TIMEOUT` — optional uniform timeout.

Add corresponding `get_*` accessors to `verify/backend/utils/config.py`.

### 5.4 Open questions / gotchas

1. **Vision parity**: Ollama's OpenAI-compat endpoint accepts `image_url`
   data-URLs for vision models but swallow errors silently for text-only
   models. Worth surfacing a clearer message.
2. **Latency**: Simple_Blur has `timeout=60`, PrivacyLens `timeout=90`.
   Local models on CPU can exceed these; timeouts should be configurable.
3. **Response shape**: Both modules assume OpenAI-compat
   `choices[0].message.content` — all listed local backends match. Ollama's
   **native** `/api/chat` endpoint differs (`message.content` at top level),
   so we should specifically target Ollama's `/v1/chat/completions`.
4. **`OPENROUTER_DEFAULT_MODEL` leakage**: `Simple_Blur` imports from
   `verify/backend/adapters/base.py`. When we split routing, we should move
   the default constant(s) into `utils/llm_client.py` or equivalent to
   avoid a circular "adapters → perturbation" coupling.
5. **Branding headers**: `HTTP-Referer` / `X-Title` must not be forwarded
   to local servers — some will 400 on unrecognized headers.
6. **Existing `BaseAdapter._inject_openrouter_env()`** already standardizes
   env-var plumbing for target apps. We can extend it to also accept
   `LLM_BACKEND=ollama` and inject `OPENAI_BASE_URL=http://localhost:11434/v1`
   etc., so **target apps** under test will also route through local models
   — but that is adapter work (task #2 / #3), not perturbation work.

---

## 6. Summary

- Two of three perturbation modules call OpenRouter directly via
  hand-rolled `requests.post` to the `/v1/chat/completions` endpoint.
- The dispatcher (`interface.py`) already threads arbitrary `**kwargs` to
  `perturb()`, so no dispatcher changes are needed.
- Minimum change: introduce `verify/backend/utils/llm_client.chat_completion(...)`
  that reads `LLM_BACKEND` / `LLM_BASE_URL` / `LLM_MODEL` env vars (with
  OpenRouter as the default fallback), then swap the two inline
  `requests.post` blocks for calls to this helper.
- Add per-role model overrides (`PERTURBATION_TEXT_MODEL`,
  `PERTURBATION_VISION_MODEL`) so vision/text perturbation can target
  different local models.
- Move `OPENROUTER_DEFAULT_MODEL` out of `adapters/base.py` to avoid
  reverse coupling once the helper exists.
