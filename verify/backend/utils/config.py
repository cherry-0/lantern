"""
Config utilities: load config files and environment variables for Verify.
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Optional

# Root of the verify/ directory
VERIFY_ROOT = Path(__file__).resolve().parent.parent.parent
LANTERN_ROOT = VERIFY_ROOT.parent

CONFIG_DIR = VERIFY_ROOT / "config"
BACKEND_DIR = VERIFY_ROOT / "backend"
DATASET_DIR = BACKEND_DIR / "datasets"
OUTPUTS_DIR = VERIFY_ROOT / "outputs"
TARGET_APPS_DIR = LANTERN_ROOT / "target-apps"


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


def use_app_servers() -> bool:
    """
    Return True when USE_APP_SERVERS is set to a truthy value in .env or environment.

    Controls the global adapter execution mode:
      True  → HTTP / native pipeline  (requires target app servers to be running)
      False → OpenRouter serverless fallback  (no target app dependency)
    """
    val = get_env("USE_APP_SERVERS", "false") or "false"
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
