"""
Simple disk-based caching for Verify pipeline runs.

Cache key components: app_name, dataset_name, modality, attributes, method identifiers.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from verify.backend.utils.config import OUTPUTS_DIR


def normalize_eval_prompt(eval_prompt: Optional[str]) -> str:
    """Normalize IOC eval prompt values, defaulting legacy/missing values to prompt1."""
    value = str(eval_prompt or "").strip()
    return value or "prompt1"


def _make_cache_key(
    app_name: str,
    dataset_name: str,
    modality: str,
    attributes: List[str],
    perturbation_method: str = "",
    evaluation_method: str = "openrouter",
) -> str:
    """Generate a stable hash key for a run configuration."""
    parts = {
        "app": app_name,
        "dataset": dataset_name,
        "modality": modality,
        "attributes": sorted(attributes),
        "perturbation": perturbation_method,
        "evaluation": evaluation_method,
    }
    payload = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_cache_dir(
    app_name: str,
    dataset_name: str,
    modality: str,
    attributes: List[str],
    perturbation_method: str = "",
    evaluation_method: str = "openrouter",
) -> Path:
    """Return the cache directory for the given run config (creates it if needed)."""
    key = _make_cache_key(
        app_name,
        dataset_name,
        modality,
        attributes,
        perturbation_method,
        evaluation_method,
    )
    cache_dir = OUTPUTS_DIR / f"cache_{key}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_item_cache(
    cache_dir: Path,
    filename: str,
    *,
    expected_eval_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load cached result for a specific item file, or None if not cached."""
    cache_file = cache_dir / f"{filename}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if expected_eval_prompt is not None:
                actual = normalize_eval_prompt(data.get("eval_prompt"))
                expected = normalize_eval_prompt(expected_eval_prompt)
                if actual != expected:
                    return None
                data["eval_prompt"] = actual
            return data
        except Exception:
            return None
    return None


def save_item_cache(cache_dir: Path, filename: str, data: Dict[str, Any]) -> None:
    """Persist a per-item result to disk."""
    cache_file = cache_dir / f"{filename}.json"
    try:
        cache_file.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass  # Non-fatal; caching is best-effort


def load_run_config(
    cache_dir: Path,
    *,
    expected_eval_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load the run config JSON from the cache dir, or None."""
    config_file = cache_dir / "run_config.json"
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
            if expected_eval_prompt is not None:
                actual = normalize_eval_prompt(config.get("eval_prompt"))
                expected = normalize_eval_prompt(expected_eval_prompt)
                if actual != expected:
                    return None
                config["eval_prompt"] = actual
            return config
        except Exception:
            return None
    return None


def save_run_config(cache_dir: Path, config: Dict[str, Any]) -> None:
    """Save a run configuration to disk."""
    config_file = cache_dir / "run_config.json"
    try:
        config_file.write_text(json.dumps(config, indent=2, default=str))
    except Exception:
        pass
