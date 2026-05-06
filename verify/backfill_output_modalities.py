"""
Backfill input/output modality fields in verify/outputs/*/run_config.json.

Older output caches often stored only `modality` and sometimes `generation_task`.
That makes Streamlit pages render ambiguous labels like `image` even when the
actual workflow is `image->text`. This script fills:

  input_modality   from input_modality or modality
  output_modality  from output_modality, generation_task, batch_config.csv, or
                   item JSON shape
  generation_task  from output_modality for compatibility

It only edits run_config.json files under verify/outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

VERIFY_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = VERIFY_ROOT / "outputs"
BATCH_CONFIG = VERIFY_ROOT / "batch_config.csv"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _load_batch_lookup(path: Path) -> Dict[Tuple[str, str, str, str], str]:
    lookup: Dict[Tuple[str, str, str, str], str] = {}
    if not path.exists():
        return lookup
    with path.open(newline="") as f:
        reader = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in reader:
            app = _clean(row.get("app_name"))
            dataset = _clean(row.get("dataset_name"))
            input_modality = _clean(row.get("input_modality") or row.get("modality"))
            output_modality = _clean(row.get("output_modality") or row.get("generation_task"))
            method = _clean(row.get("perturbation_method"))
            if app and dataset and input_modality and output_modality:
                lookup[(app, dataset, input_modality, method)] = output_modality
    return lookup


def _looks_like_image_output(value: Any) -> bool:
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys & {"image_base64", "generated_image_base64", "image_path", "generated_image_path"}:
            return True
        return any(_looks_like_image_output(v) for v in value.values())
    if isinstance(value, list):
        return any(_looks_like_image_output(v) for v in value)
    return False


def _infer_from_items(run_dir: Path) -> Optional[str]:
    for item_path in sorted(run_dir.glob("*.json")):
        if item_path.name in {"run_config.json", "dir_summary.json"}:
            continue
        try:
            item = json.loads(item_path.read_text())
        except Exception:
            continue

        for key in ("output_modality", "generation_task"):
            value = _clean(item.get(key))
            if value:
                return value

        if _looks_like_image_output(item.get("output_text")):
            return "image"
        if _looks_like_image_output(item.get("original_output")) or _looks_like_image_output(item.get("perturbed_output")):
            return "image"
        if item.get("output_text") or item.get("ext_text"):
            return "text"
        if isinstance(item.get("original_output"), dict) and item["original_output"].get("output_text"):
            return "text"
    return None


def _infer_output_modality(
    run_dir: Path,
    cfg: Dict[str, Any],
    batch_lookup: Dict[Tuple[str, str, str, str], str],
) -> str:
    existing = _clean(cfg.get("output_modality") or cfg.get("generation_task"))
    if existing:
        return existing

    app = _clean(cfg.get("app_name"))
    dataset = _clean(cfg.get("dataset_name"))
    input_modality = _clean(cfg.get("input_modality") or cfg.get("modality"))
    method = _clean(cfg.get("perturbation_method"))
    from_batch = batch_lookup.get((app, dataset, input_modality, method))
    if from_batch:
        return from_batch

    from_items = _infer_from_items(run_dir)
    if from_items:
        return from_items

    # Conservative app-pipeline fallback: most Verify target app outputs are text.
    return "text"


def backfill(outputs_dir: Path, *, dry_run: bool = False) -> Tuple[int, int]:
    batch_lookup = _load_batch_lookup(BATCH_CONFIG)
    scanned = changed = 0
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "run_config.json"
        if not cfg_path.exists():
            continue
        scanned += 1
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue

        before = json.dumps(cfg, sort_keys=True, default=str)
        input_modality = _clean(cfg.get("input_modality") or cfg.get("modality"))
        output_modality = _infer_output_modality(run_dir, cfg, batch_lookup)

        if input_modality:
            cfg["input_modality"] = input_modality
            cfg["modality"] = cfg.get("modality") or input_modality
        if output_modality:
            cfg["output_modality"] = output_modality
            cfg["generation_task"] = cfg.get("generation_task") or output_modality

        after = json.dumps(cfg, sort_keys=True, default=str)
        if after != before:
            changed += 1
            print(f"{run_dir.name}: {input_modality}->{output_modality}")
            if not dry_run:
                cfg_path.write_text(json.dumps(cfg, indent=2, default=str))

    return scanned, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing.")
    args = parser.parse_args()

    scanned, changed = backfill(OUTPUTS_DIR, dry_run=args.dry_run)
    action = "would update" if args.dry_run else "updated"
    print(f"Scanned {scanned} run_config.json files; {action} {changed}.")


if __name__ == "__main__":
    main()
