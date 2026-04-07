# Verify App Adapter Analysis Report

This report documents the complete architecture of the `verify/` privacy evaluation framework: how adapters bridge target apps to the verification pipeline, how execution modes are controlled, how native app pipelines are isolated and invoked, and how results flow through evaluation.

---

## 1. Adapter Architecture

Each target app has a corresponding adapter inheriting from `verify.backend.adapters.base.BaseAdapter`. The base class defines a standardized interface:

- **`check_availability() → Tuple[bool, str]`** — non-blocking probe: returns `(available, reason)`.
- **`run_pipeline(input_item) → AdapterResult`** — main execution entry point.

### AdapterResult

| Field | Type | Description |
|---|---|---|
| `success` | bool | Whether pipeline succeeded |
| `output_text` | str | Human-readable output |
| `raw_output` | Dict | Raw output from app |
| `structured_output` | Dict | Structured fields (verdict, tags, category, etc.) |
| `externalizations` | Dict[str, str] | Captured channels (NETWORK, UI, STORAGE, LOGGING, etc.) |
| `error` | Optional[str] | Error message on failure |
| `metadata` | Dict | Method name, generated task, etc. |

The `combined_output` property concatenates `externalizations` with `[CHANNEL]` prefixes and falls back to `output_text` — this is what the evaluator receives.

All adapters inherit `_call_openrouter(prompt, image_b64, model, max_tokens, timeout)` from `BaseAdapter`. Default model: `google/gemini-2.5-pro`.

---

## 2. Execution Mode: `USE_APP_SERVERS`

A single environment variable in `.env` controls which execution path every adapter takes:

| Value | Mode | What runs |
|---|---|---|
| `true` | **NATIVE** | Actual app pipeline (local models, native code, or CondaRunner subprocess) |
| `false` (default) | **SERVERLESS** | OpenRouter-only fallback replicating the same output structure |

`use_app_servers() → bool` in `verify/backend/utils/config.py` reads this variable (accepts `true`, `1`, `yes`, case-insensitive).

The serverless path produces **structurally identical output** to the native path — same field names, same categories, same JSON shape — so results are directly comparable across modes.

### Debug mode

`is_debug() → bool` reads `DEBUG=true/false` from `.env`. When `DEBUG=true` and `USE_APP_SERVERS=false`, `_build_serverless_externalizations()` returns `{"NETWORK": "example output", "UI": "example output"}` instead of realistic-looking strings. This prevents confusion during development.

---

## 3. Native Pipeline Isolation: CondaRunner

Running native pipelines in-process causes `sys.modules` collisions (multiple apps define a `config` package; Django can only be initialized once per process). The solution is **process isolation via conda environments**.

### 3.1 `CondaRunner` (`verify/backend/utils/conda_runner.py`)

```
CondaRunner
  ├── find_conda()          — locate conda/mamba/micromamba on PATH (cached)
  ├── env_exists(name)      — check if conda env exists (shells to conda env list --json)
  ├── _sentinel(name)       — path to .verify_ready inside the env directory
  ├── is_ready(name)        — env_exists AND sentinel present
  ├── probe(spec)           — non-blocking; returns (True, msg) if conda found
  ├── ensure(spec)          — blocking first-time setup: create env + install + touch sentinel
  └── run(env, script, data, timeout)
         — write input to temp JSON
         → Popen: conda run -n {env} python {script} {input.json}
         → background thread relays stderr to terminal in real time
         → collect stdout, parse first { ... last } as JSON result
         → return (ok, result_dict, error_str)
```

**`EnvSpec` dataclass:** `name` (conda env name), `python` (version string), `install_cmds` (list of pip commands), `cwd` (optional working directory).

### 3.2 Sentinel File Caching

After `ensure()` completes it touches `.verify_ready` inside the env directory. Subsequent calls to `is_ready()` short-circuit immediately. Delete this file to force re-setup without removing the whole env.

### 3.3 Stderr Streaming

`CondaRunner.run()` uses `subprocess.Popen` with a background thread that relays runner `stderr` to the terminal line-by-line in real time. `stdout` is collected separately for JSON parsing. This ensures database creation and inference logs appear immediately without buffering.

### 3.4 JSON Extraction

Stdout is parsed with `find("{")` (first `{`) and `rfind("}")` (last `}`) to correctly handle multi-line JSON values containing `{` characters (e.g. email body text). All runner diagnostic logging goes to `stderr` so `stdout` is always clean JSON.

### 3.5 Per-App Conda Environments

| App | Env name | Python | Install command |
|---|---|---|---|
| clone | `clone` | 3.12 | `pip install django==5.2 djangorestframework django-cors-headers djangorestframework-simplejwt drf-yasg python-dotenv requests Pillow django-storages boto3` |
| snapdo | `snapdo` | 3.10 | `pip install -r requirements.txt` |
| momentag | `momentag` | 3.13 | `pip install -e .` (backend/) |
| xend | `xend` | 3.12 | `pip install poetry` → `poetry install --no-root` |
| budget-lens | `budget-lens` | 3.10 | `pip install -r requirements.txt` |
| deeptutor | `deeptutor` | 3.10 | `pip install -e .` |
| llm-vtuber | `llm-vtuber` | 3.11 | `pip install -e .` |
| skin-disease-detection | `skin-disease-detection` | 3.10 | `pip install tensorflow-macos pillow numpy` |
| google-ai-edge-gallery | `google-ai-edge-gallery` | 3.10 | `pip install transformers accelerate torch` |

Note: xend uses `poetry install --no-root` because `pyproject.toml` sets `package-mode = false`, which makes `pip install .` fail.  
Note: skin-disease uses `tensorflow-macos` (not `tflite-runtime`) on Apple Silicon — see TROUBLESHOOTING.md §10.

---

## 4. Runner Scripts

Each app has a standalone runner in `verify/backend/runners/`. Runners follow a strict contract:

1. Read a JSON input file from `sys.argv[1]`
2. Print diagnostic progress to `stderr`
3. Print a single JSON result to `stdout` as the last line
4. Exit 0 on success, 1 on failure (with `{"success": false, "error": "<traceback>"}` on stdout)

All runners call `log_input()` from `_runner_log.py` at startup to display the input item in the terminal.

All runners import `_runtime_capture` at the top (after sys.path setup) and call `_runtime_capture.install()` before any app code runs. Django-based runners additionally call `_runtime_capture.connect_django_signals()` after `django.setup()`.

### Runner input/output contract

| Runner | Input keys | Output keys |
|---|---|---|
| `clone_runner.py` | image_base64, openrouter_api_key, model | success, activity, details, summary, num_frames, externalizations |
| `snapdo_runner.py` | image_base64, task_title, task_description, openrouter_api_key, model | success, verdict, confidence, explanation, task_title, externalizations |
| `momentag_runner.py` | image_base64 | success, captions, tags, externalizations |
| `xend_runner.py` | text_content, openrouter_api_key, model | success, subject, body, externalizations |
| `budgetlens_runner.py` | image_base64, openrouter_api_key | success, category, date, amount, currency, externalizations |
| `deeptutor_runner.py` | text_content, openrouter_api_key, model | success, tutor_response, student_input, externalizations |
| `llmvtuber_runner.py` | text_content, openrouter_api_key, model | success, character_response, user_message, externalizations |
| `skindisease_runner.py` | image_base64 | success, cancer {label, confidence}, allergy {label, confidence}, melanoma {label, confidence}, externalizations |
| `googleaiedge_runner.py` | modality, text_content, image_description, model_id, max_tokens | success, ai_response, model_id, externalizations |

### Django runners and SQLite migration

Five apps (clone, xend, momentag, budget-lens, snapdo) require Django setup but use PostgreSQL/MySQL in production. Runners use `_<app>_verify_settings.py` shims that:

1. Inject dummy values for all required env vars (`if not os.environ.get(key)` — covers both missing and empty-string cases)
2. Override `DATABASES` to a local SQLite file
3. Auto-migrate on first run via `call_command("migrate", "--run-syncdb", verbosity=0)`

| Shim file | SQLite path | Key overrides |
|---|---|---|
| `_clone_verify_settings.py` | `runners/clone_verify.sqlite3` | SECRET_KEY, VECTORDB_CHAT_HOST, VECTORDB_SCREEN_HOST, SENDGRID_*, DB_* |
| `_xend_verify_settings.py` | `runners/xend_verify.sqlite3` | SECRET_KEY, CHANNEL_URL, CELERY_BROKER_URL, DATABASE_*, GOOGLE_CLIENT_*, ENCRYPTION_KEY, PII_MASKING_SECRET |
| `_momentag_verify_settings.py` | `runners/momentag_verify.sqlite3` | SECRET_KEY, QDRANT_CLUSTER_URL, QDRANT_API_KEY, DB_* (also appends "test" to sys.argv to trigger momentag's built-in SQLite path) |
| `_budgetlens_verify_settings.py` | `runners/budgetlens_verify.sqlite3` | SECRET_KEY, DB_* |

snapdo already uses SQLite with a hardcoded `SECRET_KEY` — no shim needed.

---

## 5. Externalization Capture System

### 5.1 `_runtime_capture.py` (runner-side, stdlib-only)

`verify/backend/runners/_runtime_capture.py` is a self-contained module (no verify package imports, works in any conda env). It captures actual runtime I/O from the app during execution.

**`install()`** patches three transport layers:
- `requests.Session.request` — captures HTTP requests/responses (URL, method, status)
- `httpx.Client.send` — captures sync httpx calls (used by LangChain, some Django apps)
- `httpx.AsyncClient.send` — captures async httpx calls

**`connect_django_signals()`** — call after `django.setup()` to hook `post_save` signal for model persistence events.

**Log filtering** — installs a `logging.Handler` at DEBUG level. Keeps WARNING+ always; keeps DEBUG/INFO only if the log record name or message contains inference-relevant keywords (`inference`, `llm`, `model`, `verdict`, `category`, `mail`, `tutor`, `response`, etc.). Skips noisy internals (`django.db.backends`, `urllib3`, `asyncio`, `transformers`, etc.).

**`finalize() → dict`** — returns captured channels, caps applied:
- `NETWORK`: up to 15 entries, formatted as `[METHOD] URL → STATUS`
- `STORAGE`: up to 10 Django model save events
- `LOGGING`: up to 8 deduplicated log lines

Empty channels are omitted from the returned dict.

### 5.2 `_openrouter_calls` tracking (adapter-side, serverless only)

`BaseAdapter._call_openrouter()` appends to `self._openrouter_calls` on each call:

```python
{
    "model": used_model,
    "has_image": bool(image_b64),
    "status": resp.status_code,
    "prompt_preview": prompt[:200],
    "response_preview": response_text[:200],
}
```

The orchestrator calls `adapter._reset_openrouter_calls()` before each `run_pipeline()` invocation (both original and perturbed) to ensure clean per-item state.

### 5.3 `_build_serverless_externalizations()` — 3-tier priority

```python
def _build_serverless_externalizations(self, realistic_fallback=None) -> dict:
```

Priority order (first match wins):

1. **Real captured calls** — if `self._openrouter_calls` is non-empty, format each call as:
   ```
   [POST] https://openrouter.ai/api/v1/chat/completions (model, vision/text) → 200
   ↳ Prompt: <first 200 chars>
   ↳ Response: <first 200 chars>
   ```
2. **DEBUG placeholder** — if `is_debug()` is True, return `{"NETWORK": "example output", "UI": "example output"}`
3. **Realistic fallback** — return the `realistic_fallback` dict passed by the adapter (descriptive strings that reflect what the real app would do)

### 5.4 What is shown in the UI

`View Results` page (`verify/frontend/pages/1_View_Results.py`) shows externalizations for both the original and perturbed pipeline results in a `🌐 Captured Externalizations` expander within the Inference Results section. Each channel is shown as a bold header + caption text.

---

## 6. Per-Adapter Design

### clone
**Native (`USE_APP_SERVERS=true`):** `CondaRunner.run(clone_runner.py)` — bootstraps Django with `_clone_verify_settings`, auto-migrates SQLite, creates/gets `verify@lantern.local` user via ORM, creates a ChatSession, describes frames via OpenRouter vision, persists ChatMessage. Captures externalizations via `_runtime_capture`.

**Serverless:** Direct OpenRouter vision with `_FRAME_PROMPT`. Externalizations via `_build_serverless_externalizations()`.

**Modality:** image, video

---

### snapdo
**Native:** `CondaRunner.run(snapdo_runner.py)` — bootstraps Django from `target-apps/snapdo/server`, calls `VLMService().verify_evidence(image_b64, constraint, model=model)` routed to OpenRouter via env-var injection.

**Serverless:** OpenRouter vision; pre-generates a realistic Todo task for the dataset item (cached per full file **path**, not filename, to avoid basename collisions) so the model has meaningful context for verification.

**Output:** verdict (PASSED/FAILED/UNKNOWN), confidence, explanation

**Modality:** image

---

### momentag
**Native:** `CondaRunner.run(momentag_runner.py)` — bootstraps Django, calls `gallery.gpu_tasks.get_image_captions(pil_image)` which runs CLIP (`clip-ViT-B-32`) for embeddings and BLIP (`blip-image-captioning-base`) for captions/keywords locally. Timeout: 180s.

**Serverless:** OpenRouter vision generates captions and tags in the same structure.

**Output:** captions (list[str]), tags (list[str])

**Modality:** image

---

### xend
**Native:** `CondaRunner.run(xend_runner.py)` — bootstraps Django with `_xend_verify_settings`, imports `subject_chain` and `body_chain` from `apps.ai.services.chains`, invokes both with a structured inputs dict (body, subject, language, recipients, etc.).

**Serverless:** OpenRouter with xend's email-drafting persona; produces `subject` + `body`.

**Output:** subject, body

**Modality:** text

---

### budget-lens
**Native:** `CondaRunner.run(budgetlens_runner.py)` — bootstraps Django with `_budgetlens_verify_settings`, writes image to a temp JPEG, calls `process_receipt(path)` from `core.views`. Captures externalizations: OpenAI request, exchange rate API call, Expense model save.

**Serverless:** OpenRouter vision extracts category, date, amount, currency plus `merchant_name` and `merchant_address` (relevant for SROIE ground-truth evaluation).

**Output:** category (one of 15 predefined), date (YYYY-MM-DD), amount, currency, optional merchant fields

**Modality:** image

---

### deeptutor
**Native:** `CondaRunner.run(deeptutor_runner.py)` — injects LLM env vars (`LLM_BINDING`, `LLM_HOST`, `LLM_API_KEY`, `LLM_MODEL`), imports `ChatOrchestrator` + `UnifiedContext` + `StreamEventType`, drives `orchestrator.handle(context)` async via `asyncio.run()`, collects `CONTENT` events. Captures externalizations: RAG embedding, LLM request, RAG retrieval, event bus, debug logging.

**Serverless:** OpenRouter with structured tutoring prompt (summary → key concepts → examples → caveats).

**Output:** tutor_response, student_input

**Modality:** text

---

### llm-vtuber
**Native:** `CondaRunner.run(llmvtuber_runner.py)` — imports `AsyncLLM` from `open_llm_vtuber.agent.stateless_llm.openai_compatible_llm`, calls `llm.chat_completion(messages, system=_VTUBER_SYSTEM)` async. Captures externalizations: STT (Whisper), LLM, TTS requests; Live2D UI rendering; chat_history.json storage.

**Serverless:** OpenRouter with the same Shizuku VTuber system prompt.

**Output:** character_response, user_message, character_name ("Shizuku")

**Modality:** text

---

### skin-disease-detection
**Native:** `CondaRunner.run(skindisease_runner.py)` — loads all three `.tflite` model files from `target-apps/skin-disease-detection/app/assets/` via `tflite_runtime` (falls back to `tensorflow.lite`). Preprocessing mirrors the Dart classifier: center-crop → resize NEAREST to model input dimensions → normalize [0, 1] (or uint8 for quantized inputs). Handles quantized output via scale + zero_point dequantization. Logs each classifier result (label + confidence) to stderr.

**Serverless:** OpenRouter vision prompt replicates the three-classifier output structure with confidence scores.

**Output:**
- `cancer`: 6-class (Actinic Keratoses, Basal Cell Carcinoma, Benign Keratosis like Lesions, Dermatofibroma, Melanocytic Nevi, Vascular Lesions)
- `allergy`: 3-class (Acne and Rosacea, Eczema and Atopic Dermatitis, Nail Fungus and other Nail Disease)
- `melanoma`: 2-class (Melanoma, Not Melanoma)

**Modality:** image

---

### google-ai-edge-gallery
**Native:** `CondaRunner.run(googleaiedge_runner.py)` — loads `transformers.pipeline("text-generation")` with `model_id` (default: `Qwen/Qwen2.5-1.5B-Instruct`), selects CUDA if available. The adapter first describes any image via OpenRouter (since small local models are typically text-only), then passes the description as `image_description` to the runner. Timeout: 300s (model download on first run).

**Serverless:** OpenRouter with the app's LlmChat system prompt; handles both text and image natively.

**Config:** `GOOGLE_AI_EDGE_MODEL_ID` (HuggingFace model ID), `GOOGLE_AI_EDGE_MAX_TOKENS` (default 512).

**Output:** ai_response, model_id, input_modality

**Modality:** image, text

---

## 7. How to Add a New App

### Step 1: Create the adapter

Create `verify/backend/adapters/<appname>.py`:

```python
"""
Adapter for <appname>.
Core pipeline: <describe the main data flow>
"""
from pathlib import Path
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

APPNAME_ROOT = TARGET_APPS_DIR / "<appname>"

_ENV_SPEC = EnvSpec(
    name="<appname>",
    python="3.10",                  # match app's Python requirement
    install_cmds=[["pip", "install", "-r", str(APPNAME_ROOT / "requirements.txt")]],
    # cwd=APPNAME_ROOT,             # set if pip install -e . needs a specific directory
)
_RUNNER = Path(__file__).parent.parent / "runners" / "<appname>_runner.py"

# System prompt for serverless fallback
_SYSTEM = "You are ..."


class AppNameAdapter(BaseAdapter):
    name = "<appname>"
    supported_modalities = ["text"]   # or ["image"], ["image", "text"]

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter fallback for <appname>."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") not in self.supported_modalities:
            return AdapterResult(success=False, error="Unsupported modality.")
        if use_app_servers():
            return self._run_native(input_item)
        return self._run_serverless(input_item)

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        ok, result, err = CondaRunner.run(
            _ENV_SPEC.name, _RUNNER,
            {"text_content": input_item.get("data", ""),
             "openrouter_api_key": get_openrouter_api_key() or "",
             "model": OPENROUTER_DEFAULT_MODEL},
            timeout=90,
        )
        if not ok:
            return AdapterResult(success=False, error=err)

        response = result.get("response", "")
        return AdapterResult(
            success=result.get("success", False),
            output_text=response,
            raw_output=result,
            structured_output={"response": response},
            externalizations=result.get("externalizations", {}),
            metadata={"method": "native"},
        )

    def _run_serverless(self, input_item: Dict[str, Any]) -> AdapterResult:
        text = str(input_item.get("data", "")).strip()
        prompt = f"{_SYSTEM}\n\nUser: {text}"
        try:
            response = self._call_openrouter(prompt=prompt, max_tokens=512)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": f"[OpenRouter Fallback] Sending request to API.",
                "UI": f"Displaying response: {response[:100]}...",
            }
        )
        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"response": response},
            structured_output={"response": response},
            externalizations=externalizations,
            metadata={"method": "openrouter_fallback"},
        )
```

### Step 2: Create the runner

Create `verify/backend/runners/<appname>_runner.py`:

```python
"""
Runner for <appname>. Executed inside the '<appname>' conda environment.
Requires: pip install ...
"""
import json, sys
from pathlib import Path

# ── sys.path setup ────────────────────────────────────────────────────────────
_RUNNERS_DIR = Path(__file__).parent
_VERIFY_ROOT = _RUNNERS_DIR.parent.parent
sys.path.insert(0, str(_RUNNERS_DIR))         # for _runtime_capture, _runner_log
sys.path.insert(0, str(_VERIFY_ROOT))         # for verify.backend.utils.config

import _runtime_capture
_runtime_capture.install()

from _runner_log import log_input

# ── Target app imports (after sys.path) ───────────────────────────────────────
_APP_ROOT = Path(__file__).parent.parent.parent / "target-apps" / "<appname>"
sys.path.insert(0, str(_APP_ROOT))

# from <appname_module> import <function>  # import the app's actual pipeline here

# ── Main ──────────────────────────────────────────────────────────────────────
def main(data: dict) -> dict:
    log_input("<appname>", data)
    text = data.get("text_content", "")
    api_key = data.get("openrouter_api_key", "")

    print(f"[<appname>] Running pipeline...", file=sys.stderr, flush=True)

    # --- call the app's actual pipeline here ---
    response = "..."

    externalizations = _runtime_capture.finalize()
    return {
        "success": True,
        "response": response,
        "externalizations": externalizations,
    }

if __name__ == "__main__":
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)
    try:
        result = main(data)
    except Exception:
        import traceback
        result = {"success": False, "error": traceback.format_exc()}
    print(json.dumps(result))
```

**For Django-based apps**, add before the target app imports:

```python
# ── Django settings shim ──────────────────────────────────────────────────────
import _<appname>_verify_settings   # sets env vars + DATABASES override
import django
django.setup()
_runtime_capture.connect_django_signals()
from django.core.management import call_command
call_command("migrate", "--run-syncdb", verbosity=0)
```

### Step 3: (Django only) Create the settings shim

Create `verify/backend/runners/_<appname>_verify_settings.py`:

```python
"""
Minimal Django settings shim for <appname> in the verify environment.
Injects dummy values for all required env vars, then overrides DATABASES to SQLite.
"""
import os
from pathlib import Path

_RUNNERS_DIR = Path(__file__).parent

# Inject required env vars (only if not already set — handles empty strings too)
for _k, _v in [
    ("SECRET_KEY", "verify-dummy-secret"),
    ("DB_NAME", "dummy"),
    ("DB_USER", "dummy"),
    ("DB_PASSWORD", "dummy"),
    ("DB_HOST", "localhost"),
    ("DB_PORT", "5432"),
    # Add any other required env vars here
].items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

# Import the app's actual settings
from <appname>.config.settings import *   # adjust import path as needed

# Override DB to local SQLite (no PostgreSQL/MySQL needed in verify)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(_RUNNERS_DIR / "<appname>_verify.sqlite3"),
    }
}
```

### Step 4: Register the adapter

In `verify/backend/adapters/__init__.py`, add to `ADAPTER_REGISTRY`:

```python
from verify.backend.adapters.<appname> import AppNameAdapter

ADAPTER_REGISTRY = {
    ...,
    "<appname>": AppNameAdapter,
}
```

### Step 5: Add to the dataset configuration

In `verify/frontend/app.py` (or wherever datasets are defined per app), add the new app to the app selector and configure which datasets it supports.

### Common Pitfalls

| Pitfall | Fix |
|---|---|
| `sys.modules` collision (`config` package already imported) | Runner isolation via CondaRunner subprocess handles this — do not import verify package inside runners |
| `os.environ.setdefault` doesn't override empty strings | Use `if not os.environ.get(k): os.environ[k] = v` in settings shims |
| Cache key collision (different images, same filename) | Use full `path` as cache key, not `os.path.basename(path)` |
| Silent exceptions returning default output for all items | Log exceptions to `stderr` explicitly; do not swallow in bare `except` |
| `tflite-runtime` not installable on Apple Silicon | Use `tensorflow-macos`; runner already has `try: import tflite_runtime` fallback |
| Django `migrate` fails because model fields reference unset env vars at import time | Inject all env vars **before** `from <settings> import *` in the shim |
| Runner stdout polluted with non-JSON | All diagnostics go to `stderr`; only the final `print(json.dumps(result))` goes to `stdout` |

---

## 8. Logging System

### Always-on logs (all environments)

| Event | Function | Output |
|---|---|---|
| Adapter availability | `log_availability()` | `✦ AVAILABILITY` block with OK/FAIL status |
| Conda env setup start | `log_setup_start()` | `⚙ SETUP` banner with python version |
| Install step | `log_setup_step()` | `INSTALL` row inside setup block |
| Setup complete/failed | `log_setup_done()` | `✔ Done` or `✘ FAILED` footer |
| Runner input | `log_input()` (in runner) | `▶ INPUT` block with TYPE + CONTENT/PATH |
| Runner progress | `print(..., file=sys.stderr)` | `[app] Loading ... / Running ... / complete.` |
| DB creation | `print(..., file=sys.stderr)` | `[app] Creating SQLite database (first run) ...` |

### VERBOSE-gated logs (`VERBOSE=true` required)

| Event | Function | Output |
|---|---|---|
| Inference result | `log_inference()` | `▶ INFERENCE` block with INPUT/MODULE/OUTPUT |

Set `VERBOSE=true` in `.env` and restart Streamlit to enable inference blocks.

---

## 9. OpenRouter Roles

OpenRouter is used in four distinct roles:

1. **Serverless inference** — primary LLM/VLM provider when `USE_APP_SERVERS=false`
2. **Image description bridge** — google-ai-edge native mode describes images via OpenRouter before passing text to the local HuggingFace model
3. **Transparent routing in runners** — runners inject OpenRouter credentials into app-native env vars (`OPENAI_API_KEY`, `LLM_API_KEY`, `VLM_API_KEY`) so the app's own LLM client calls OpenRouter without code changes
4. **Snapdo task generation** — serverless mode pre-generates a realistic task context per dataset item before calling verify_evidence

---

## 10. Dataset Support and Attribute Filtering

### Supported dataset formats

| Format | Detection | Privacy labels | Attribute field |
|---|---|---|---|
| HR-VISPR | Split subdirs (val2017/test2017/train2017) + PKL/JSON labels | 18-class labels (age, face, gender, race, …) | `privacy_labels` |
| SROIE2019 | `{split}/img/` + `{split}/entities/` | Entity field names (company, date, address, total) | `privacy_labels` + `sroie_entity_attrs` |
| PrivacyLens (HF) | `dataset_dict.json` or `*.arrow` | Mapped from seed.data_type via pre-built PKL | `data_type_attributes` |
| Flat files | Extension counting | None | — |

### Attribute filtering (orchestrator lines 315–321)

```python
item_labels = (
    item.get("privacy_labels")        # HR-VISPR / SROIE
    or item.get("data_type_attributes")  # PrivacyLens
    or []
)
if item_labels and not (set(item_labels) & set(self.attributes)):
    continue  # skip — no overlap with selected attributes
```

Items with no labels are always included. Items with labels are skipped unless at least one label overlaps the selected attributes. The expander title in the UI shows `🏷 {labels}` (privacy_labels) or `📄 {data_type} → {attributes}` (PrivacyLens).

---

## 11. Full Pipeline Summary

```
Dataset item
    │
    ├── Attribute filter (skip if no label overlap)
    │
    ├── Cache check (skip if already processed)
    │
    ├─▶ Original pipeline
    │       ├── adapter._reset_openrouter_calls()
    │       └── adapter.run_pipeline(item)
    │               └── [native]    CondaRunner.run() → runner subprocess
    │                               └── _runtime_capture patches HTTP/logging/Django ORM
    │                   [serverless] _call_openrouter() → tracked in _openrouter_calls
    │                               └── _build_serverless_externalizations() (3-tier priority)
    │
    ├─▶ Perturbation
    │       └── run_perturbation(item, modality, attributes, method)
    │
    ├─▶ Perturbed pipeline
    │       ├── adapter._reset_openrouter_calls()
    │       └── adapter.run_pipeline(perturbed_item)
    │
    ├─▶ Evaluation
    │       └── evaluate_both(orig_output, pert_output, attributes)
    │               └── OpenRouter judges inferability per attribute
    │                   → {score: 0.0–1.0, inferable: bool, reasoning: str}
    │
    └── Yield item_result → Streamlit UI (progressive display)
                               └── View Results: externalizations expander per pipeline run
```

### Summary table

| Adapter | Native strategy | Serverless strategy | Modality |
|---|---|---|---|
| clone | CondaRunner → Django ORM + OpenRouter vision | OpenRouter vision | image, video |
| snapdo | CondaRunner → VLMService | OpenRouter vision + task gen | image |
| momentag | CondaRunner → CLIP + BLIP | OpenRouter vision | image |
| xend | CondaRunner → LangChain chains | OpenRouter text | text |
| budget-lens | CondaRunner → process_receipt() | OpenRouter vision | image |
| deeptutor | CondaRunner → ChatOrchestrator | OpenRouter text | text |
| llm-vtuber | CondaRunner → AsyncLLM | OpenRouter text | text |
| skin-disease | CondaRunner → 3× TFLite | OpenRouter vision | image |
| google-ai-edge | CondaRunner → HuggingFace transformers | OpenRouter text+vision | image, text |
