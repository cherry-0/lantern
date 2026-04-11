# Lantern — Privacy Evaluation Framework (Verify)

Lantern is a research framework for evaluating how much private information AI-powered mobile and desktop applications infer, process, and externalize. The core tool is **Verify** — a Streamlit UI that runs each target app's AI pipeline against privacy-labeled datasets and measures attribute inferability at every stage (input → output → network/storage/UI channels).

---

## Repository Layout

```
lantern/
├── verify/                   # Evaluation framework (this is the main tool)
│   ├── frontend/             # Streamlit pages
│   │   └── pages/
│   │       ├── 0_Initialization.py       # Per-app env setup
│   │       ├── 1_Perturb_Input.py        # Run + perturb pipeline
│   │       ├── 1_View_Results.py         # Browse past run results
│   │       └── 2_Input_Output_Comparison.py  # Stage-wise attribute comparison
│   ├── backend/
│   │   ├── adapters/         # One adapter per target app
│   │   ├── runners/          # Subprocess runner scripts (run inside conda envs)
│   │   ├── datasets/         # Dataset loaders (HR-VISPR, SROIE, PrivacyLens, …)
│   │   ├── perturbation_method/
│   │   ├── evaluation_method/
│   │   └── utils/
│   ├── config/               # Attribute lists, dataset registry
│   └── requirements.txt      # Verify's own Python dependencies
├── target-apps/              # The apps under evaluation (git submodules)
│   ├── clone/
│   ├── momentag/
│   ├── snapdo/
│   ├── xend/
│   └── ...
├── analysis/                 # Research notes and architecture docs
│   └── verify_report.md      # Full adapter + pipeline reference
├── .env                      # API keys and feature flags (not committed)
├── install_all_reqs.sh       # Install target-app dependencies
└── TROUBLESHOOTING.md
```

---

## 1. Prerequisites

- **conda** (Miniconda or Anaconda) — used to create isolated envs per target app
- **Python 3.12** for the Verify host environment
- **git** with submodule support
- **OpenRouter API key** (required for serverless mode and evaluation)
- Optional: `uv`, `poetry` — needed by individual target apps (installed automatically where required)

---

## 2. Installation

### 2.1 Clone with submodules

```bash
git clone --recurse-submodules <repo-url> lantern
cd lantern
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2.2 Create the Verify host environment

```bash
conda create -n lantern python=3.12
conda activate lantern
pip install -r verify/requirements.txt
```

This installs Streamlit and the libraries used by Verify itself. It does **not** install the target apps' dependencies — those are managed separately (see §3).

### 2.3 Configure the `.env` file

Create `.env` in the repo root (next to `README.md`):

```env
# Required: used for serverless inference and all LLM-based evaluation
OPENROUTER_API_KEY=sk-or-...

# Execution mode
# true  → run the app's real pipeline inside a conda subprocess (native mode)
# false → call OpenRouter directly, mimicking the app's output structure (serverless mode)
USE_APP_SERVERS=false

# Optional flags
DEBUG=false    # true → externalizations show "example output" instead of realistic strings
VERBOSE=false  # true → print full inference I/O blocks to the terminal
```

---

## 3. Target-App Dependencies

Each target app runs in its own **conda environment**, completely isolated from the Verify host. This prevents `sys.modules` collisions (multiple apps define a `config` package; Django can only initialize once per process).

### Execution modes

| `USE_APP_SERVERS` | Mode | What runs |
|---|---|---|
| `false` (default) | **Serverless** | OpenRouter calls only — no local app code, no conda envs needed |
| `true` | **Native** | App's real AI pipeline inside a conda subprocess |

In **serverless mode** you only need an `OPENROUTER_API_KEY`. No target-app dependencies are required. This is the fastest way to get started.

In **native mode** each app needs its own conda environment. The Verify **Initialization page** (`⚙️` in the sidebar) manages this for you — it shows the status of every app's environment and provides an "Initialize" button that creates and populates the env on first use.

### Per-app environments (native mode)

| App | Conda env | Python | Install method |
|---|---|---|---|
| `clone` | `clone` | 3.12 | `pip install django djangorestframework …` |
| `momentag` | `momentag` | 3.13 | `uv export --frozen … \| pip install -r /dev/stdin` |
| `snapdo` | `snapdo` | 3.10 | `pip install -r requirements.txt` |
| `xend` | `xend` | 3.12 | `pip install poetry` → `poetry install --no-root` |
| `budget-lens` | `budget-lens` | 3.10 | `pip install -r requirements.txt` |
| `deeptutor` | `deeptutor` | 3.10 | `pip install -e .` |
| `llm-vtuber` | `llm-vtuber` | 3.11 | `pip install -e .` |
| `skin-disease-detection` | `skin-disease-detection` | 3.10 | `pip install tensorflow-macos pillow numpy` |
| `google-ai-edge-gallery` | `google-ai-edge-gallery` | 3.10 | `pip install transformers accelerate torch` |

You can also install target-app dependencies in bulk with the provided script (native mode only):

```bash
./install_all_reqs.sh
```

---

## 4. Running Verify

```bash
conda activate lantern
streamlit run verify/frontend/app.py
```

Navigate to `http://localhost:8501`. Use the sidebar to switch between pages:

| Page | Purpose |
|---|---|
| **Initialization** | Check and set up per-app conda environments |
| **Perturb Input** | Run the pipeline on original + perturbed inputs; compare inferability |
| **View Results** | Browse cached results from previous runs |
| **Input / Output Comparison** | Stage-wise analysis: Input → Raw Output → Externalized channels |

---

## 5. How to Add a New Target App

### Step 1: Add the app as a git submodule

```bash
git submodule add <app-repo-url> target-apps/<appname>
git submodule update --init target-apps/<appname>
```

### Step 2: Create the adapter

Create `verify/backend/adapters/<appname>.py` implementing `BaseAdapter`:

```python
from verify.backend.adapters.base import BaseAdapter, AdapterResult, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

_ENV_SPEC = EnvSpec(
    name="<appname>",
    python="3.10",
    install_cmds=[["pip", "install", "-r", str(TARGET_APPS_DIR / "<appname>" / "requirements.txt")]],
)
_RUNNER = Path(__file__).parent.parent / "runners" / "<appname>_runner.py"

class AppNameAdapter(BaseAdapter):
    name = "<appname>"
    supported_modalities = ["image"]   # or ["text"], ["image", "text"]
    env_spec = _ENV_SPEC

    def check_availability(self):
        if use_app_servers():
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter fallback."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item):
        if use_app_servers():
            return self._run_native(input_item)
        return self._run_serverless(input_item)

    def _run_native(self, input_item):
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)
        ok, result, err = CondaRunner.run(_ENV_SPEC.name, _RUNNER, {...}, timeout=90)
        if not ok:
            return AdapterResult(success=False, error=err)
        return AdapterResult(
            success=result.get("success", False),
            output_text=result.get("response", ""),
            externalizations=result.get("externalizations", {}),
        )

    def _run_serverless(self, input_item):
        response = self._call_openrouter(prompt="...", max_tokens=512)
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={"NETWORK": "OpenRouter call to infer output."}
        )
        return AdapterResult(success=True, output_text=response, externalizations=externalizations)
```

The `env_spec` class attribute is used by the Initialization page to manage the conda environment.

### Step 3: Create the runner script

Create `verify/backend/runners/<appname>_runner.py`. Runners execute inside the app's conda env and must:
- Read input from `sys.argv[1]` (a temp JSON file)
- Print all diagnostic output to `stderr`
- Print a single JSON result dict to `stdout` as the last line

```python
import json, sys
from pathlib import Path

_RUNNERS_DIR = Path(__file__).parent
sys.path.insert(0, str(_RUNNERS_DIR))
sys.path.insert(0, str(_RUNNERS_DIR.parent.parent))

import _runtime_capture
_runtime_capture.install()   # patches HTTP, logging, Django ORM

from _runner_log import log_input

_APP_ROOT = Path(__file__).parent.parent.parent / "target-apps" / "<appname>"
sys.path.insert(0, str(_APP_ROOT))
# from <appname> import <pipeline_function>

def main(data):
    log_input("<appname>", data)
    # ... call the app's actual pipeline ...
    result = "..."
    externalizations = _runtime_capture.finalize()
    return {"success": True, "response": result, "externalizations": externalizations}

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)
    try:
        print(json.dumps(main(data)))
    except Exception:
        import traceback
        print(json.dumps({"success": False, "error": traceback.format_exc()}))
```

**For Django-based apps**, add a settings shim (`verify/backend/runners/_<appname>_verify_settings.py`) that injects dummy env vars and overrides `DATABASES` to a local SQLite file. Import it in the runner before `django.setup()`. See `analysis/verify_report.md` §7 for the full template.

### Step 4: Register the adapter

In `verify/backend/adapters/__init__.py`:

```python
from verify.backend.adapters.<appname> import AppNameAdapter

ADAPTER_REGISTRY = {
    ...,
    "<appname>": AppNameAdapter,
}
```

That's all. The Initialization page, Perturb Input page, and Input/Output Comparison page all discover adapters from `ADAPTER_REGISTRY` automatically.

For the complete reference — including Django shim templates, externalization capture details, common pitfalls, and per-adapter design notes — see [`analysis/verify_report.md`](analysis/verify_report.md).

---

## 6. How the Pipeline Works

```
Dataset item
    │
    ├── (Perturb Input page)  perturb with selected method + attributes
    │
    ├── Run app pipeline
    │     ├── [native]     CondaRunner spawns: conda run -n <env> python runner.py
    │     │                _runtime_capture patches HTTP / logging / Django ORM
    │     └── [serverless] _call_openrouter() → OpenRouter API
    │
    ├── Evaluate inferability per privacy attribute
    │     └── OpenRouter judges each attribute: {score, inferable, reasoning}
    │
    └── Display result progressively in the UI (item-by-item)
          └── Results cached to verify/outputs/cache_<hash>/
```

### Externalization channels

The framework captures what the app does with inferred data beyond producing output text:

| Channel | What it captures |
|---|---|
| `NETWORK` | HTTP/HTTPS requests (URL, method, status, payload preview) |
| `UI` | WebSocket pushes, notifications, animations shown to the user |
| `STORAGE` | Django ORM `post_save` signals (database writes) |
| `LOGGING` | Inference-relevant WARNING+ log messages |

Channels are organized into two phases: **DURING** (while the AI model is running) and **POST** (after inference, when results are externalized).

---

## 7. Caching

Results are cached to `verify/outputs/cache_<hash>/` keyed by `{app, dataset, modality, attributes, method}`. Re-running with the same configuration skips already-processed items. Use the "Use cache" checkbox in the sidebar to disable caching for a fresh run.

To invalidate all caches, delete the cache directories under `verify/outputs/`:

```bash
rm -rf verify/outputs/cache_*/
```
