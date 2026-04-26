"""
Config utilities: load config files and environment variables for Verify.
"""

import os
import csv
import json
import threading
from pathlib import Path
from typing import Any, List, Dict, Optional

# Root of the verify/ directory
VERIFY_ROOT = Path(__file__).resolve().parent.parent.parent
LANTERN_ROOT = VERIFY_ROOT.parent

CONFIG_DIR = VERIFY_ROOT / "config"
BACKEND_DIR = VERIFY_ROOT / "backend"
DATASET_DIR = BACKEND_DIR / "datasets"
OUTPUTS_DIR = VERIFY_ROOT / "outputs"
TARGET_APPS_DIR = LANTERN_ROOT / "target-apps"
EVAL_PROMPT_CHOICES = ("prompt1", "prompt2", "prompt3", "prompt4")
DEFAULT_COLOR_PALETTE: Dict[str, Dict[str, str]] = {
    "verdict": {
        "confirmed leakage": "#f96d36",
        "possible leakage": "#f7b44f",
        "no evidence": "#98d198",
        "na": "#e6e6e6",
    },
    "stage": {
        "Input": "#5bc0de",
        "Raw Output": "#d9534f",
        "Externalized": "#f0ad4e",
    },
    "channel": {
        "AGGREGATE": "#f0ad4e",
        "Aggregate": "#f0ad4e",
        "UI": "#7f8c8d",
        "NETWORK": "#5b8def",
        "Network": "#5b8def",
        "STORAGE": "#27ae60",
        "Storage": "#27ae60",
        "LOGGING": "#f39c12",
        "Logging": "#f39c12",
        "Perturbed Aggregate": "#b9770e",
    },
    "heatmap": {
        "empty": "#ffffff",
        "missing": "#e6e6e6",
        "exposure_high": "#c62828",
    },
    "binary": {
        "positive": "#d9534f",
        "negative": "#5cb85c",
        "secondary": "#5bc0de",
    },
}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_dataset_list() -> List[str]:
    """Read dataset names from config/dataset_list.txt."""
    path = CONFIG_DIR / "dataset_list.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_attribute_list(modality: str = "text") -> List[str]:
    """
    Read privacy attributes for the given modality.
    image → config/attribute_list_image.txt  (HR-VISPR 18-class labels)
    text / other → config/attribute_list.txt
    """
    filename = "attribute_list_image.txt" if modality == "image" else "attribute_list.txt"
    path = CONFIG_DIR / filename
    if not path.exists():
        # fallback to default
        path = CONFIG_DIR / "attribute_list.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_perturbation_method_map() -> Dict[str, str]:
    """
    Parse config/perturbation_method.csv and return a mapping:
        {data_type: perturbation_method}
    data_type is treated as the modality (image, text, video).
    """
    path = CONFIG_DIR / "perturbation_method.csv"
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_type = row.get("data_type", "").strip()
            method = row.get(" perturbation_method", row.get("perturbation_method", "")).strip()
            if data_type and method:
                mapping[data_type] = method
    return mapping


def load_color_palette() -> Dict[str, Dict[str, str]]:
    """Load UI color palette from verify/config/color_palette.json."""
    path = CONFIG_DIR / "color_palette.json"
    if not path.exists():
        return DEFAULT_COLOR_PALETTE
    try:
        data = json.loads(path.read_text())
    except Exception:
        return DEFAULT_COLOR_PALETTE
    if not isinstance(data, dict):
        return DEFAULT_COLOR_PALETTE
    merged = _deep_merge_dict(DEFAULT_COLOR_PALETTE, data)
    return {
        section: {
            str(key): str(value)
            for key, value in values.items()
            if isinstance(values, dict)
        }
        for section, values in merged.items()
        if isinstance(values, dict)
    }


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read an environment variable. Attempts to load from the root .env if not set.
    Does NOT auto-install anything.
    """
    val = os.environ.get(key)
    if val:
        return val.strip()

    # Try loading from .env at LANTERN_ROOT
    env_path = LANTERN_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                if k.strip() == key:
                    v = v.strip()
                    # Strip surrounding quotes
                    if (v.startswith('"') and v.endswith('"')) or (
                        v.startswith("'") and v.endswith("'")
                    ):
                        v = v[1:-1]
                    return v
    return default


def get_openrouter_api_key() -> Optional[str]:
    """Return OpenRouter API key from environment."""
    return get_env("OPENROUTER_API_KEY") or get_env("OPENROUTER_KEY")


def get_default_eval_prompt() -> str:
    """
    Return the default IOC externalization evaluator prompt.

    Resolution order:
      1. VERIFY_EVAL_PROMPT
      2. EVAL_PROMPT
      3. prompt4
    """
    raw = get_env("VERIFY_EVAL_PROMPT") or get_env("EVAL_PROMPT") or "prompt4"
    value = raw.strip().lower()
    return value if value in EVAL_PROMPT_CHOICES else "prompt4"


def is_debug() -> bool:
    """
    Return True when DEBUG=true is set in .env or environment.

    Controls externalization fallback behaviour in serverless adapters:
      True  → placeholder "example output" when no real data is captured
      False → realistic-looking hardcoded strings (default)
    """
    val = get_env("DEBUG", "false") or "false"
    return val.strip().lower() in ("1", "true", "yes")


# ── Per-app mode overrides ────────────────────────────────────────────────────
# Maps app_name → "native" | "serverless".
# Set by the Streamlit Settings page; overrides USE_APP_SERVERS for that app only.
_APP_MODE_OVERRIDES: Dict[str, str] = {}

# Per-thread name of the adapter currently executing; used by use_app_servers()
# to look up per-app overrides without needing to change every adapter's call
# site.  threading.local() makes this safe when multiple items are processed
# concurrently in a ThreadPoolExecutor.
_app_context_local: threading.local = threading.local()


def set_app_mode_override(app_name: str, mode: str) -> None:
    """
    Set the execution mode for a specific app.
    mode: "native" | "serverless" | "auto" (removes override, falls back to .env)
    """
    if mode == "auto":
        _APP_MODE_OVERRIDES.pop(app_name, None)
    else:
        _APP_MODE_OVERRIDES[app_name] = mode


def get_app_mode_override(app_name: str) -> Optional[str]:
    """Return the current override for an app, or None if using the global default."""
    return _APP_MODE_OVERRIDES.get(app_name)


def set_current_app_context(app_name: str) -> None:
    """Tell use_app_servers() which app is currently executing (per-thread)."""
    _app_context_local.name = app_name


def use_app_servers() -> bool:
    """
    Return True when native (app server) mode is in effect for the current app.

    Resolution order:
      1. Per-app override set via Streamlit Settings page (_APP_MODE_OVERRIDES)
      2. Global USE_APP_SERVERS value from .env / environment

    Controls the adapter execution mode:
      True  → HTTP / native pipeline  (requires target app servers to be running)
      False → OpenRouter serverless fallback  (no target app dependency)
    """
    _current_app_context = getattr(_app_context_local, "name", "")
    if _current_app_context and _current_app_context in _APP_MODE_OVERRIDES:
        return _APP_MODE_OVERRIDES[_current_app_context] == "native"
    val = get_env("USE_APP_SERVERS", "false") or "false"
    return val.strip().lower() in ("1", "true", "yes")


def is_malicious_prompt_mode() -> bool:
    """
    Return True when MALICIOUS_PROMPT_MODE=true is set in .env or environment.

    Controls task/prompt generation in adapters that call a VLM to create
    an input prompt before running the app pipeline:
      True  → _generate_malicious_task(): privacy-maximizing prompt tailored
               to the specific image, designed to surface maximum private info
      False → _generate_task(): natural, realistic prompt a real user would send
    """
    val = get_env("MALICIOUS_PROMPT_MODE", "false") or "false"
    return val.strip().lower() in ("1", "true", "yes")


def get_openai_api_key() -> Optional[str]:
    """Return OpenAI API key from environment."""
    return get_env("OPENAI_API_KEY")


def list_target_apps() -> List[str]:
    """Return names of directories inside target-apps/."""
    if not TARGET_APPS_DIR.exists():
        return []
    return [
        d.name
        for d in TARGET_APPS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]


def get_dataset_path(dataset_name: str) -> Optional[Path]:
    """Return the path to a dataset directory, or None if not found."""
    p = DATASET_DIR / dataset_name
    return p if p.exists() else None


def ensure_outputs_dir() -> Path:
    """Ensure the outputs directory exists and return its path."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR
