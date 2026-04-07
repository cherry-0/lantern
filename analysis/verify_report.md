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
| `true` | **NATIVE** | Actual app pipeline (local models, native code, or live HTTP server) |
| `false` (default) | **SERVERLESS** | OpenRouter-only fallback replicating the same output structure |

`use_app_servers() → bool` in `verify/backend/utils/config.py` reads this variable (accepts `true`, `1`, `yes`, case-insensitive).

The serverless path produces **structurally identical output** to the native path — same field names, same categories, same JSON shape — so results are directly comparable across modes.

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
| snapdo | `snapdo` | 3.10 | `pip install -r requirements.txt` |
| momentag | `momentag` | 3.13 | `pip install -e .` (backend/) |
| xend | `xend` | 3.12 | `pip install poetry` → `poetry install --no-root` |
| budget-lens | `budget-lens` | 3.10 | `pip install -r requirements.txt` |
| deeptutor | `deeptutor` | 3.10 | `pip install -e .` |
| llm-vtuber | `llm-vtuber` | 3.11 | `pip install -e .` |
| skin-disease-detection | `skin-disease-detection` | 3.10 | `pip install tflite-runtime pillow numpy` |
| google-ai-edge-gallery | `google-ai-edge-gallery` | 3.10 | `pip install transformers accelerate torch` |

Note: `clone` uses HTTP requests to a live server — no conda isolation.  
Note: xend uses `poetry install --no-root` because `pyproject.toml` sets `package-mode = false`, which makes `pip install .` fail.

---

## 4. Runner Scripts

Each app has a standalone runner in `verify/backend/runners/`. Runners follow a strict contract:

1. Read a JSON input file from `sys.argv[1]`
2. Print diagnostic progress to `stderr`
3. Print a single JSON result to `stdout` as the last line
4. Exit 0 on success, 1 on failure (with `{"success": false, "error": "<traceback>"}` on stdout)

All runners call `log_input()` from `_runner_log.py` at startup to display the input item in the terminal.

### Runner input/output contract

| Runner | Input keys | Output keys |
|---|---|---|
| `snapdo_runner.py` | image_base64, task_title, task_description, openrouter_api_key, model | success, verdict, confidence, explanation, task_title |
| `momentag_runner.py` | image_base64 | success, captions, tags |
| `xend_runner.py` | text_content, openrouter_api_key, model | success, subject, body |
| `budgetlens_runner.py` | image_base64, openrouter_api_key | success, category, date, amount, currency, externalizations |
| `deeptutor_runner.py` | text_content, openrouter_api_key, model | success, tutor_response, student_input, externalizations |
| `llmvtuber_runner.py` | text_content, openrouter_api_key, model | success, character_response, user_message, externalizations |
| `skindisease_runner.py` | image_base64 | success, cancer {label, confidence}, allergy {label, confidence}, melanoma {label, confidence} |
| `googleaiedge_runner.py` | modality, text_content, image_description, model_id, max_tokens | success, ai_response, model_id, externalizations |

### Django runners and SQLite migration

Three apps (xend, momentag, budget-lens) require Django setup but use PostgreSQL/MySQL in production. Runners use `_<app>_verify_settings.py` shims that:

1. Inject dummy values for all required env vars (`if not os.environ.get(key)` — covers both missing and empty-string cases)
2. Override `DATABASES` to a local SQLite file
3. Auto-migrate on first run via `call_command("migrate", "--run-syncdb", verbosity=0)`

| Shim file | SQLite path | Key overrides |
|---|---|---|
| `_xend_verify_settings.py` | `runners/xend_verify.sqlite3` | SECRET_KEY, CHANNEL_URL, CELERY_BROKER_URL, DATABASE_*, GOOGLE_CLIENT_*, ENCRYPTION_KEY, PII_MASKING_SECRET |
| `_momentag_verify_settings.py` | `runners/momentag_verify.sqlite3` | SECRET_KEY, QDRANT_CLUSTER_URL, QDRANT_API_KEY, DB_* (also appends "test" to sys.argv to trigger momentag's built-in SQLite path) |
| `_budgetlens_verify_settings.py` | `runners/budgetlens_verify.sqlite3` | SECRET_KEY, DB_* |

snapdo already uses SQLite with a hardcoded `SECRET_KEY` — no shim needed.

---

## 5. Input Logging (`_runner_log.py`)

All runners import `log_input()` from `verify/backend/runners/_runner_log.py` — a self-contained stdlib-only helper (no verify package imports, works in any conda env).

Output goes to `stderr` and is relayed to the terminal in real time:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ INPUT   xend
──────────────────────────────────────────────────────────────────────
  TYPE      text
  CONTENT   Hello everyone, I hope this email finds you well. I am
            writing to share a brief update on my recent work...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

- **image:** shows file path or `<base64>` under `PATH`
- **text:** shows first 300 chars wrapped to 68-char width under `CONTENT`

---

## 6. Per-Adapter Design

### clone
**Native (`USE_APP_SERVERS=true`):** HTTP client pattern — auth (`POST /api/auth/login/`, auto-signup), session creation (`POST /api/chat/sessions/create/`), OpenRouter vision to describe frames, then post message (`POST /api/chat/sessions/{id}/messages/create/`). Fails explicitly if server is unreachable. Config: `CLONE_SERVER_URL`, `CLONE_EMAIL`, `CLONE_PASSWORD`.

**Serverless:** Direct OpenRouter vision with `_FRAME_PROMPT`. Structurally identical output (activity, details, summary, num_frames).

**Modality:** image, video

---

### snapdo
**Native:** `CondaRunner.run(snapdo_runner.py)` — bootstraps Django from `target-apps/snapdo/server`, calls `VLMService().verify_evidence(image_b64, constraint, model=model)` routed to OpenRouter via env-var injection.

**Serverless:** OpenRouter vision; pre-generates a realistic Todo task for the dataset item (cached per filename) so the model has meaningful context for verification.

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
**Native:** `CondaRunner.run(googleaiedge_runner.py)` — loads `transformers.pipeline("text-generation")` with `model_id` (default: `Qwen/Qwen2.5-1.5B-Instruct`), selects CUDA if available. The adapter first describes any image via OpenRouter (since small local models are typically text-only), then passes the description as `image_description` to the runner. Captures externalizations: WebView log, Android Intent, Firebase analytics. Timeout: 300s (model download on first run).

**Serverless:** OpenRouter with the app's LlmChat system prompt; handles both text and image natively.

**Config:** `GOOGLE_AI_EDGE_MODEL_ID` (HuggingFace model ID), `GOOGLE_AI_EDGE_MAX_TOKENS` (default 512).

**Output:** ai_response, model_id, input_modality

**Modality:** image, text

---

## 7. Logging System

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

## 8. OpenRouter Roles

OpenRouter is used in four distinct roles:

1. **Serverless inference** — primary LLM/VLM provider when `USE_APP_SERVERS=false`
2. **Image description bridge** — google-ai-edge native mode describes images via OpenRouter before passing text to the local HuggingFace model
3. **Transparent routing in runners** — runners inject OpenRouter credentials into app-native env vars (`OPENAI_API_KEY`, `LLM_API_KEY`, `VLM_API_KEY`) so the app's own LLM client calls OpenRouter without code changes
4. **Snapdo task generation** — serverless mode pre-generates a realistic task context per dataset item before calling verify_evidence

---

## 9. Dataset Support and Attribute Filtering

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

## 10. Full Pipeline Summary

```
Dataset item
    │
    ├── Attribute filter (skip if no label overlap)
    │
    ├── Cache check (skip if already processed)
    │
    ├─▶ Original pipeline
    │       └── adapter.run_pipeline(item)
    │               └── [native]    CondaRunner.run() → runner subprocess
    │                   [serverless] _call_openrouter()
    │
    ├─▶ Perturbation
    │       └── run_perturbation(item, modality, attributes, method)
    │
    ├─▶ Perturbed pipeline
    │       └── adapter.run_pipeline(perturbed_item)
    │
    ├─▶ Evaluation
    │       └── evaluate_both(orig_output, pert_output, attributes)
    │               └── OpenRouter judges inferability per attribute
    │                   → {score: 0.0–1.0, inferable: bool, reasoning: str}
    │
    └── Yield item_result → Streamlit UI (progressive display)
```

### Summary table

| Adapter | Native strategy | Serverless strategy | Modality |
|---|---|---|---|
| clone | HTTP REST (live Django server) | OpenRouter vision | image, video |
| snapdo | CondaRunner → VLMService | OpenRouter vision + task gen | image |
| momentag | CondaRunner → CLIP + BLIP | OpenRouter vision | image |
| xend | CondaRunner → LangChain chains | OpenRouter text | text |
| budget-lens | CondaRunner → process_receipt() | OpenRouter vision | image |
| deeptutor | CondaRunner → ChatOrchestrator | OpenRouter text | text |
| llm-vtuber | CondaRunner → AsyncLLM | OpenRouter text | text |
| skin-disease | CondaRunner → 3× TFLite | OpenRouter vision | image |
| google-ai-edge | CondaRunner → HuggingFace transformers | OpenRouter text+vision | image, text |
