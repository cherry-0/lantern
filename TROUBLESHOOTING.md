# Troubleshooting — Lantern Verify Framework

This document captures recurring errors, their root causes, and fixes encountered during development of the `verify/` privacy evaluation framework.

---

## 1. Runner path traversal off-by-one

**Symptom**
```
ModuleNotFoundError: No module named 'server'
ModuleNotFoundError: No module named 'config'
```
Django `setup()` fails immediately because the app's root directory is not on `sys.path`.

**Root cause**
Runner scripts compute their path relative to `__file__` using `..` segments. The runners live at:
```
verify/backend/runners/<runner>.py
```
Going up to the repo root requires **3** levels (`runners → backend → verify → lantern/`), but the original code used 4, overshooting to the parent of the repo.

**Fix**
```python
# Wrong — goes above the repo root
os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "target-apps", ...)

# Correct
os.path.join(os.path.dirname(__file__), "..", "..", "..", "target-apps", ...)
```

**Affected runners** (all fixed): `snapdo`, `momentag`, `xend`, `budgetlens`, `deeptutor`, `llmvtuber`, `skindisease`.

---

## 2. Runner errors swallowed by conda's generic message

**Symptom**
```
Original pipeline failed: Runner exited 1: ERROR conda.cli.main_run:execute(125): conda run ... failed. (See above for error)
```
The actual Python traceback is invisible — only conda's own error line is shown.

**Root cause**
`CondaRunner.run()` originally used `subprocess.run(capture_output=True)` and only surfaced `stderr` on non-zero exit. But conda wraps the runner's stderr with its own message, and the runner's JSON error output (on `stdout`) was discarded.

**Fix**
- On non-zero exit, parse `stdout` for a JSON error object first (runners print `{"success": false, "error": "<traceback>"}` then exit 1).
- Only fall back to `stderr` if no JSON is found.
- Switched to `Popen` with a background thread relaying `stderr` to the terminal in real time.

**Location**: `verify/backend/utils/conda_runner.py` → `CondaRunner.run()`

---

## 3. Django `SECRET_KEY` / database errors in native runners

**Symptom**
```
django.core.exceptions.ImproperlyConfigured: The SECRET_KEY setting must not be empty.
```
or
```
django.db.utils.OperationalError: could not connect to server: Connection refused
```

**Root cause**
Target apps that use PostgreSQL (xend, momentag, budget-lens) require a full set of environment variables and a live database server. Runners that call `django.setup()` without these configured will fail before any inference code runs.

Apps affected and their DB engine:
| App | Default DB | Required env vars |
|---|---|---|
| snapdo | SQLite (hardcoded) | `VLM_API_KEY` only |
| xend | PostgreSQL | `SECRET_KEY`, `DATABASE_*`, `CHANNEL_URL`, `CELERY_BROKER_URL`, `GOOGLE_CLIENT_*`, `ENCRYPTION_KEY`, `PII_MASKING_SECRET`, … |
| budget-lens | PostgreSQL | `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` |
| momentag | MySQL (dev) | `SECRET_KEY`, `QDRANT_*`, `DB_*` |

**Fix**
Create a `_<app>_verify_settings.py` shim in `verify/backend/runners/` for each app that needs it:
1. Inject dummy `os.environ.setdefault()` values for all required vars.
2. Import the app's own settings (`from config.settings.local import *`).
3. Override `DATABASES` to SQLite with a local file path.
4. In the runner, run `migrate --run-syncdb` once if the SQLite file does not yet exist.

```python
# Example: verify/backend/runners/_xend_verify_settings.py
os.environ.setdefault("SECRET_KEY", "verify-only-dummy-key")
os.environ.setdefault("PII_MASKING_SECRET", "00" * 32)  # bytes.fromhex() needs valid hex
# ... other required vars ...
from config.settings.local import *
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": str(_DB_PATH)}}
```

```python
# In runner: auto-migrate on first run
db_path = django_settings.DATABASES["default"]["NAME"]
if not os.path.exists(db_path):
    call_command("migrate", "--run-syncdb", verbosity=0)
```

SQLite files created: `verify/backend/runners/xend_verify.sqlite3`, `budgetlens_verify.sqlite3`, `momentag_verify.sqlite3`.

**Note on empty `.env` values**: The xend `.env` file may contain `SECRET_KEY=` (key present, value empty). `os.environ.setdefault()` does NOT override an existing key even if its value is an empty string — the key exists, so setdefault is a no-op. The shim must use `if not os.environ.get(key): os.environ[key] = value` to cover both missing and empty cases.

**Note on `PII_MASKING_SECRET`**: xend uses `bytes.fromhex(env("PII_MASKING_SECRET"))`. The dummy value must be a valid even-length hex string — `"00" * 32` (64 hex chars = 32 bytes) satisfies this.

**Note on momentag**: its settings switch to SQLite automatically when `"test"` is in `sys.argv`. The shim appends `"test"` to `sys.argv` before importing settings to trigger this path, then overrides with a file-based SQLite (not `:memory:`) so the schema persists.

---

## 4. Poetry non-package mode — `pip install .` fails

**Symptom**
```
RuntimeError: Building a package is not possible in non-package mode.
error: metadata-generation-failed
```

**Root cause**
xend's `pyproject.toml` sets `package-mode = false` (Poetry ≥ 1.2 feature). `pip install .` tries to build a wheel, which Poetry refuses in non-package mode.

**Fix**
Use `poetry install --no-root` instead, which installs dependencies only without building the project itself:
```python
# verify/backend/adapters/xend.py
_ENV_SPEC = EnvSpec(
    name="xend",
    python="3.12",
    install_cmds=[
        ["pip", "install", "poetry"],        # ensure poetry is in the conda env
        ["poetry", "install", "--no-root"],  # install deps only
    ],
    cwd=XEND_BACKEND,
)
```

After changing `install_cmds`, delete the stale sentinel to force re-setup:
```bash
find ~/miniconda3/envs/xend -name ".verify_ready" -delete
```

---

## 5. "No JSON found in runner output" despite valid JSON being present

**Symptom**
```
No JSON found in runner output: {"success": true, "subject": "...", "body": "Hello everyone,\n\nI hope...
```
The error message shows the start of a valid JSON object, but parsing still fails.

**Root cause**
`CondaRunner.run()` originally used `rfind("{")` to locate the JSON start — the intent was to skip any log lines that might precede the JSON. However, if the JSON value contains `{` characters (e.g. an email body or essay text), `rfind("{")` finds a `{` *inside* the body, not at the outer object boundary. The slice `stdout[rfind("{"):rfind("}")]` then extracts a fragment, not the full object.

**Fix**
Use `find("{")` (first occurrence) for the start and `rfind("}")` (last occurrence) for the end. This is safe because all runner logging now goes to `sys.stderr`, so `stdout` contains only the result JSON with no preceding text to skip.

```python
json_start = stdout.find("{")   # first '{' = start of outer object
json_end   = stdout.rfind("}") + 1  # last '}' = end of outer object
```

**Location**: `verify/backend/utils/conda_runner.py` → `CondaRunner.run()`

---

## 7. `USE_APP_SERVERS` / `VERBOSE` not taking effect

**Symptom**
Inference always runs in serverless mode, or no inference logs appear in the terminal.

**Root cause**
Both flags are read from environment variables, not from the Streamlit UI. If they are set in `.env` but Streamlit was launched before the file was edited, the process does not pick up the new values.

**Fix**
Set them in `.env` at the repo root:
```
USE_APP_SERVERS=true    # false = OpenRouter serverless (default)
VERBOSE=true            # false = only availability + setup logs shown (default)
```
Then restart the Streamlit process. Changes to `.env` require a restart — they are not hot-reloaded.

---

## 8. Conda env already exists but setup is incomplete

**Symptom**
Runner fails with import errors even though the conda env was previously created.

**Root cause**
The sentinel file (`.verify_ready`) was never written — usually because a previous `ensure()` call failed mid-install. `CondaRunner.is_ready()` returns `False` so `ensure()` should re-run, but if the env already exists it skips `conda create` and goes straight to the install commands. If those also fail, the sentinel is never written.

**Fix**
Delete the sentinel and optionally the env to force a clean re-setup:
```bash
# Force re-run of install steps only (env already exists)
find ~/miniconda3/envs/<app-name> -name ".verify_ready" -delete

# Full clean slate (slow — recreates the env from scratch)
conda env remove -n <app-name>
```

---

## 9. General debugging workflow for runner failures

When a native pipeline fails, follow this sequence:

1. **Read the full traceback** — the runner prints `{"success": false, "error": "<traceback>"}` to stdout; this now surfaces in the UI error message.

2. **Reproduce manually** to see stderr in real time:
   ```bash
   conda run -n <app-name> python verify/backend/runners/<app>_runner.py /tmp/test_input.json
   ```
   with a minimal `/tmp/test_input.json` (e.g. `{"image_base64": "", "openrouter_api_key": "sk-..."}`).

3. **Check the path** — verify the computed `target-apps/` path is correct:
   ```bash
   python -c "import os; print(os.path.normpath(os.path.join('verify/backend/runners', '..', '..', '..', 'target-apps')))"
   # Should print: target-apps  (relative to repo root)
   ```

4. **Check the sentinel** — if the env exists but setup seems incomplete:
   ```bash
   find ~/miniconda3/envs/<app-name> -name ".verify_ready"
   ```

5. **Check env vars** — for Django apps, confirm the settings shim is injecting the right values:
   ```bash
   conda run -n <app-name> python -c "
   import os, sys
   sys.path.insert(0, 'verify/backend/runners')
   sys.path.insert(0, 'target-apps/<app>/backend')
   os.environ['DJANGO_SETTINGS_MODULE'] = '_<app>_verify_settings'
   import django; django.setup()
   from django.conf import settings
   print(settings.DATABASES)
   print(bool(settings.SECRET_KEY))
   "
   ```

---

## 10. `tflite-runtime` not installable on Apple Silicon (arm64 macOS)

**Symptom**
```
Install step failed in env 'skin-disease-detection' (pip install tflite-runtime pillow numpy):
ERROR: Could not find a version that satisfies the requirement tflite-runtime
```

**Root cause**
Google discontinued the standalone `tflite-runtime` PyPI package and never published an arm64/macOS wheel. It is simply unavailable on Apple Silicon.

**Fix**
Use `tensorflow-macos` instead — it ships `tensorflow.lite` and works on arm64:
```bash
pip install tensorflow-macos pillow numpy
```

The runner already handles both import paths:
```python
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
```

**Code change**
`verify/backend/adapters/skindisease.py` — `_ENV_SPEC.install_cmds` changed from:
```python
[["pip", "install", "tflite-runtime", "pillow", "numpy"]]
```
to:
```python
[["pip", "install", "tensorflow-macos", "pillow", "numpy"]]
```

**After changing the install_cmds**, delete the stale sentinel so `ensure()` re-runs setup:
```bash
find ~/miniconda3/envs/skin-disease-detection -name ".verify_ready" -delete
```

---

## 11. Async phase drift in `_runtime_capture.py`

**Symptom**: Inference-phase network calls (e.g. the OpenRouter LLM request) appear in `externalizations`
as POST-phase events, inflating privacy leak scores.

**Root cause**: `_patched_async_send` recorded `_current_phase` **after** `await`. If `set_phase("POST")`
fires while the coroutine is suspended, the inference call is misclassified as a post-inference leak.

**Fix** (`_runtime_capture.py`):
```python
async def _patched_async_send(self, request, **kwargs):
    _phase_snap = _current_phase        # snapshot BEFORE await
    resp = await _orig_async_send(self, request, **kwargs)
    _record_network(..., phase=_phase_snap)
    return resp
```
Same fix applied to `_patched_aiohttp_request`. Added optional `phase` param to `_record_network()`.

---

## 12. `await` on synchronous TTS method (llmvtuber)

**Symptom**: `asyncio.run() cannot be called when another event loop is running` inside FastAPI handler.

**Root cause**: `EdgeTTS.generate_audio` is synchronous and internally calls `asyncio.run()`. Awaiting
it directly from an `async def` FastAPI handler re-enters the already-running event loop.

**Fix** (`llmvtuber_server.py`):
```python
await asyncio.to_thread(tts.generate_audio, response)
```

---

## 13. Missing `"IPC"` key in `_events` reset

**Symptom**: `KeyError: 'IPC'` in the server handler after `_runtime_capture` was updated to track IPC events.

**Root cause**: Server handlers that manually reset `_events` at the start of each request were written
before the `"IPC"` channel was added to `_runtime_capture.py`.

**Fix**: All server handlers must reset ALL channels:
```python
_runtime_capture._events = {"NETWORK": [], "STORAGE": [], "LOGGING": [], "UI": [], "IPC": []}
```
Check `llmvtuber_server.py`, `googleaiedge_server.py`, `clone_server.py`, `momentag_server.py` any time
a new channel is added to `_runtime_capture._events`.

---

## 14. HTTP timeout from concurrent local model inference (google-ai-edge)

**Symptom**: Workers time out after 300 s when running google-ai-edge with `--workers > 1`.

**Root cause**: `googleaiedge_server.py` had no serialization — all parallel workers sent requests to
the same FastAPI server simultaneously. Each request queued behind the slow HuggingFace model load/run,
causing compounding latency and timeout.

**Fix**:
- `googleaiedge_server.py`: `_inference_lock = threading.Lock()` wrapping the handler body.
- `googleaiedge.py` adapter: `self._inference_lock`, timeout raised 300 → 600 s.
- Use `max_items` or `--workers 1` for local-model apps in `batch_config.csv`.

---

## 15. Experiment Progress table not scanning timestamped output dirs

**Symptom**: Progress table shows ~0 items even though `verify/outputs/` has hundreds of results.

**Root cause**: `_scan_caches()` only matched `cache_*` dirs. Perturbation outputs land in timestamped
dirs named `{app}_{modality}_{attrs}_{timestamp}/` and were completely ignored.

**Fix** (`5_Experiment_Progress.py`): Replaced with `_scan_outputs()` which iterates **all**
subdirectories of `verify/outputs/` that contain `run_config.json`.

**Also fixed**: Items that appeared in multiple runs for the same (app, dataset, method) were
double-counted. Fixed with `defaultdict(set)` to deduplicate item filenames across runs.

---

## 16. Batch Runner log box not auto-scrolling to bottom

**Symptom**: Log output always shows the top after each Streamlit rerun.

**Root cause 1**: `st.components.v1.html(..., height=0)` — zero-height iframes may be suppressed
before JavaScript executes.

**Root cause 2**: Cross-frame JS (`window.parent.document.querySelectorAll('textarea')`) is
timing-fragile and targeted `<textarea>` elements that no longer exist.

**Fix**: Replaced with a self-contained HTML component (height=460) where log content and the
`el.scrollTop = el.scrollHeight` script are in the same frame.

---

## 17. Progress bars not showing M/N ratio

**Symptom**: Per-task progress bars show "Starting…" instead of `done / total`.

**Root cause**: When `max_items` is unset in both the CSV row AND the global flag, `total` stayed
`None` and `elif total and total > 0` never fired.

**Fix** (`4_Batch_Runner.py`): At batch launch, call `count_dataset_items(dataset, modality)` to
populate `total` from dataset metadata when no cap is configured.

**Also fixed**: `&nbsp;` literal in `st.progress(text=...)` — Streamlit progress text uses Markdown,
not HTML. Replaced with plain spaces.

---

## 18. Runner response truncation in externalizations

**Symptom**: `[UI]` externalization events show `response[:150] + "..."`, hiding full model output.

**Root cause**: Several runners hard-coded truncation in `record_ui_event` calls as a defensive measure.

**Fix**: Removed truncation from all `record_ui_event` calls in:
- `llmvtuber_server.py` + `llmvtuber_runner.py`
- `deeptutor_server.py`
- `googleaiedge_server.py`
- `xend_runner.py`

**Pending**: `xend_server.py:110` still has `f"Email sent: {subject[:50]}..."`. The `*_runner.py`
and `*_server.py` variants must always be kept in sync.

---

## 19. Clone returns "screen descriptions", not email drafts — not a bug

**Observation**: Clone's output looks like an image-captioning result ("Activity: …, Details: …, Summary: …").

**Explanation**: Intentional. The verify framework tests clone's **ingestion pipeline**:
screen recording → OpenRouter vision LLM description via `_FRAME_PROMPT` → stored as `ChatMessage` in Django DB.

The chat-retrieval step is not privacy-sensitive in isolation and is not tested. Clone's output fields are
`description / activity / details / summary` — there is no `captions` field. The `structured_output` only
contains `{activity, details, summary, num_frames}`.

---

## 20. Server vs runner inconsistency (general pattern)

When fixing a runner bug, always check the corresponding server variant:

| App | Runner | Server |
|-----|--------|--------|
| xend | `xend_runner.py` | `xend_server.py` |
| llm-vtuber | `llmvtuber_runner.py` | `llmvtuber_server.py` |
| momentag | `momentag_runner.py` | `momentag_server.py` |
| clone | — | `clone_server.py` |
| google-ai-edge | — | `googleaiedge_server.py` |
| deeptutor | — | `deeptutor_server.py` |

`*_runner.py` is used for CondaRunner subprocess mode (`USE_APP_SERVERS=false`-style scripted runs).
`*_server.py` is used when `USE_APP_SERVERS=true` (long-lived FastAPI server started per adapter).
Both must behave identically.
