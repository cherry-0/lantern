#!/usr/bin/env python3
"""
Delete cached Verify item JSONs that re-eval would classify as "no_data".

In verify/reeval.py, an item is counted as no_data when:
  - item["status"] == "success", but
  - ext_text is empty, or output_eval has no attribute keys.

Removing those item JSONs lets the normal batch/cache path regenerate them on
the next run. The script is dry-run by default; pass --apply to delete files.

Examples:
  python scripts/delete_no_data_items.py
  python scripts/delete_no_data_items.py --apply
  python scripts/delete_no_data_items.py --apply --app tool-neuron --dataset HR-VISPR
  python scripts/delete_no_data_items.py --apply --dir verify/outputs/cache_abc123
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS = ROOT / "verify" / "outputs"
SKIP_NAMES = {"run_config.json", "dir_summary.json", "report.json", "results.json", "run_info.json"}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _resolve_dir(raw: str, outputs_dir: Path) -> Optional[Path]:
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    candidate = outputs_dir / raw
    if candidate.exists():
        return candidate.resolve()
    return None


def _iter_dirs(outputs_dir: Path, explicit_dirs: List[str]) -> Iterable[Path]:
    if explicit_dirs:
        for raw in explicit_dirs:
            resolved = _resolve_dir(raw, outputs_dir)
            if resolved and resolved.is_dir():
                yield resolved
            else:
                print(f"[WARN] Could not resolve directory: {raw}")
        return

    if not outputs_dir.exists():
        return
    for path in sorted(outputs_dir.iterdir()):
        if path.is_dir() and (path / "run_config.json").exists():
            yield path


def _matches_dir_filters(cfg: Dict[str, Any], app: Optional[str], dataset: Optional[str]) -> bool:
    if app and str(cfg.get("app_name", "")).strip() != app:
        return False
    if dataset and str(cfg.get("dataset_name", "")).strip() != dataset:
        return False
    return True


def _is_no_data_item(item: Dict[str, Any]) -> tuple[bool, str]:
    if item.get("status") != "success":
        return False, "not_success"

    ext_text = str(item.get("ext_text") or "")
    output_eval = item.get("output_eval") or {}
    has_attrs = isinstance(output_eval, dict) and bool(output_eval.keys())

    missing = []
    if not ext_text.strip():
        missing.append("ext_text")
    if not has_attrs:
        missing.append("output_eval_attrs")

    if missing:
        return True, "+".join(missing)
    return False, "has_data"


def _item_json_files(run_dir: Path) -> Iterable[Path]:
    for path in sorted(run_dir.iterdir()):
        if path.suffix == ".json" and path.name not in SKIP_NAMES:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        default=str(DEFAULT_OUTPUTS),
        help="Verify outputs directory to scan. Defaults to verify/outputs.",
    )
    parser.add_argument(
        "--dir",
        nargs="+",
        default=[],
        help="Specific output/cache directories to scan. Accepts paths or bare directory names.",
    )
    parser.add_argument("--app", help="Only clean directories with this app_name.")
    parser.add_argument("--dataset", help="Only clean directories with this dataset_name.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only prints what would be deleted.",
    )
    parser.add_argument(
        "--keep-summary",
        action="store_true",
        help="Do not delete dir_summary.json in directories where item files were deleted.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    total_seen = 0
    total_matches = 0
    total_deleted = 0
    touched_dirs: set[Path] = set()
    reason_counts: Dict[str, int] = {}

    mode = "DELETE" if args.apply else "DRY RUN"
    print(f"[{mode}] scanning {outputs_dir}")

    for run_dir in _iter_dirs(outputs_dir, args.dir):
        cfg = _read_json(run_dir / "run_config.json") or {}
        if not _matches_dir_filters(cfg, args.app, args.dataset):
            continue

        for item_path in _item_json_files(run_dir):
            total_seen += 1
            item = _read_json(item_path)
            if item is None:
                continue
            is_match, reason = _is_no_data_item(item)
            if not is_match:
                continue

            total_matches += 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            print(f"[MATCH] {item_path}  reason={reason}")

            if args.apply:
                item_path.unlink()
                total_deleted += 1
                touched_dirs.add(run_dir)

    summary_deleted = 0
    if args.apply and not args.keep_summary:
        for run_dir in sorted(touched_dirs):
            summary_path = run_dir / "dir_summary.json"
            if summary_path.exists():
                summary_path.unlink()
                summary_deleted += 1
                print(f"[SUMMARY] deleted stale {summary_path}")

    print()
    print(f"Item JSONs scanned : {total_seen}")
    print(f"No-data matches    : {total_matches}")
    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")
    if args.apply:
        print(f"Item JSONs deleted : {total_deleted}")
        print(f"Summaries deleted  : {summary_deleted}")
    else:
        print("No files deleted. Re-run with --apply to delete matches.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
