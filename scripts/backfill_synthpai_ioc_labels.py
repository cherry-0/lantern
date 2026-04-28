#!/usr/bin/env python3
"""
Backfill SynthPAI IOC result files with the current input-label logic.

This does not rerun apps and does not rerun external-channel evaluation. It only
rewrites stored input_labels in existing result JSON files by recomputing labels
from each saved input_item.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


LANTERN_ROOT = Path(__file__).resolve().parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

from verify.backend.datasets.label_mapper import get_input_labels  # noqa: E402


SKIP_JSON_NAMES = {"run_config.json", "dir_summary.json", "report.json", "summary.json"}


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"[skip] failed to read {path}: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    return data


def _is_synthpai_ioc_dir(run_dir: Path) -> bool:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return False
    cfg = _load_json(config_path)
    if not cfg:
        return False
    return (
        cfg.get("dataset_name") == "SynthPAI"
        and cfg.get("perturbation_method") == "ioc_comparison"
    )


def _result_files(outputs_root: Path) -> Iterable[Path]:
    for run_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        if not _is_synthpai_ioc_dir(run_dir):
            continue
        for path in sorted(run_dir.glob("*.json")):
            if path.name not in SKIP_JSON_NAMES:
                yield path


def _recompute_labels(result: Dict[str, Any]) -> Dict[str, int] | None:
    input_item = result.get("input_item") or {}
    if input_item.get("label_source") != "synthpai":
        return None

    old_labels = result.get("input_labels") or {}
    attrs = list(old_labels.keys())
    if not attrs:
        cfg_attrs = result.get("unified_attrs") or []
        attrs = [a for a in cfg_attrs if isinstance(a, str)]
    if not attrs:
        return None

    return get_input_labels(input_item, attrs)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def backfill(outputs_root: Path, apply: bool) -> Tuple[int, int, int]:
    scanned = changed = written = 0

    for path in _result_files(outputs_root):
        scanned += 1
        result = _load_json(path)
        if result is None:
            continue

        new_labels = _recompute_labels(result)
        if new_labels is None:
            continue

        old_labels = result.get("input_labels") or {}
        if new_labels == old_labels:
            continue

        changed += 1
        old_pos = sorted(k for k, v in old_labels.items() if v == 1)
        new_pos = sorted(k for k, v in new_labels.items() if v == 1)
        print(f"[change] {path}")
        print(f"  positives: {old_pos} -> {new_pos}")

        if apply:
            result["input_labels"] = new_labels
            _write_json(path, result)
            written += 1

    return scanned, changed, written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill SynthPAI IOC input_labels in verify/outputs result JSONs."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=LANTERN_ROOT / "verify" / "outputs",
        help="Directory containing cache_* output directories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changed input_labels. Without this flag, runs as a dry run.",
    )
    args = parser.parse_args()

    if not args.outputs_root.exists():
        print(f"outputs root not found: {args.outputs_root}", file=sys.stderr)
        return 2

    scanned, changed, written = backfill(args.outputs_root, args.apply)
    mode = "applied" if args.apply else "dry-run"
    print(
        f"\n{mode}: scanned={scanned}, changed={changed}, written={written}"
    )
    if not args.apply and changed:
        print("Run again with --apply to update the JSON files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
