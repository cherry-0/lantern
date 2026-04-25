"""
patch_ioc_eval_prompt.py — Backfill missing IOC eval_prompt fields on disk.

Legacy IOC caches and run directories may predate the explicit `eval_prompt`
field. This script patches those records in place by writing:

  eval_prompt = "prompt1"

to:
  - run_config.json
  - each per-item JSON file in the directory

Only IOC directories are touched:
  run_config.json["perturbation_method"] == "ioc_comparison"

Usage:
    python verify/patch_ioc_eval_prompt.py
    python verify/patch_ioc_eval_prompt.py --dry-run
    python verify/patch_ioc_eval_prompt.py --dir verify/outputs/cache_xxxxx
    python verify/patch_ioc_eval_prompt.py --app oxproxion --dataset PrivacyLens
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_VERIFY_ROOT = Path(__file__).resolve().parent
_OUTPUTS_DIR = _VERIFY_ROOT / "outputs"
_DIR_SUMMARY = "dir_summary.json"


def _iter_target_dirs(
    *,
    root: Path,
    single_dir: Optional[Path],
    apps: List[str],
    datasets: List[str],
) -> Iterable[Path]:
    dirs = [single_dir] if single_dir else sorted(root.iterdir()) if root.exists() else []
    app_filter = set(apps)
    dataset_filter = set(datasets)

    for d in dirs:
        if d is None or not d.is_dir():
            continue
        cfg_path = d / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if cfg.get("perturbation_method") != "ioc_comparison":
            continue
        app_name = str(cfg.get("app_name") or "").strip()
        dataset_name = str(cfg.get("dataset_name") or "").strip()
        if app_filter and app_name not in app_filter:
            continue
        if dataset_filter and dataset_name not in dataset_filter:
            continue
        yield d


def _patch_json_file(path: Path, *, dry_run: bool) -> str:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return "error"

    current = str(data.get("eval_prompt") or "").strip()
    if current:
        return "present"

    if dry_run:
        return "patched"

    data["eval_prompt"] = "prompt1"
    try:
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        return "error"
    return "patched"


def _process_dir(d: Path, *, dry_run: bool) -> Dict[str, int]:
    counts = {
        "dirs_seen": 1,
        "run_configs_patched": 0,
        "items_patched": 0,
        "already_present": 0,
        "errors": 0,
    }

    cfg_result = _patch_json_file(d / "run_config.json", dry_run=dry_run)
    if cfg_result == "patched":
        counts["run_configs_patched"] += 1
    elif cfg_result == "present":
        counts["already_present"] += 1
    else:
        counts["errors"] += 1

    for f in sorted(d.iterdir()):
        if f.suffix != ".json" or f.name in ("run_config.json", _DIR_SUMMARY):
            continue
        result = _patch_json_file(f, dry_run=dry_run)
        if result == "patched":
            counts["items_patched"] += 1
        elif result == "present":
            counts["already_present"] += 1
        else:
            counts["errors"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing IOC eval_prompt fields.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    parser.add_argument("--dir", type=Path, help="Patch a single output directory.")
    parser.add_argument("--app", nargs="*", default=[], help="Filter IOC directories by app name.")
    parser.add_argument("--dataset", nargs="*", default=[], help="Filter IOC directories by dataset name.")
    args = parser.parse_args()

    totals = {
        "dirs_seen": 0,
        "run_configs_patched": 0,
        "items_patched": 0,
        "already_present": 0,
        "errors": 0,
    }

    matched_dirs = list(
        _iter_target_dirs(
            root=_OUTPUTS_DIR,
            single_dir=args.dir,
            apps=args.app,
            datasets=args.dataset,
        )
    )

    if not matched_dirs:
        print("No matching IOC directories found.")
        return

    for d in matched_dirs:
        counts = _process_dir(d, dry_run=args.dry_run)
        for key, value in counts.items():
            totals[key] += value
        print(
            f"{d}: run_config +{counts['run_configs_patched']}, "
            f"items +{counts['items_patched']}, "
            f"already_present={counts['already_present']}, "
            f"errors={counts['errors']}"
        )

    mode = "DRY RUN" if args.dry_run else "PATCHED"
    print(
        f"\n[{mode}] dirs={totals['dirs_seen']} "
        f"run_configs_patched={totals['run_configs_patched']} "
        f"items_patched={totals['items_patched']} "
        f"already_present={totals['already_present']} "
        f"errors={totals['errors']}"
    )


if __name__ == "__main__":
    main()
