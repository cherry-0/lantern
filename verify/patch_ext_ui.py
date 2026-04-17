"""
patch_ext_ui.py — Reconstruct truncated UI externalization events from output_text.

Several runners used to call record_ui_event() with a truncated response:
  deeptutor              response[:150] + "..."
  google-ai-edge-gallery response[:150] + "..."
  llm-vtuber             response[:200]  (no trailing "...")
  xend                   subject[:50] + "..."

The full response is always stored in output_text, so the correct UI event
can be derived from that field.  NETWORK / STORAGE / IPC channels were never
truncated and need no patching.

For each patched item:
  - externalizations["UI"] is replaced with the reconstructed full value
  - ext_text is rebuilt from the corrected externalizations
  - ext_eval_stale is set to True  (ext_eval was scored on the old truncated
    text; re-run with --eval to fix the scores as well)

After processing each output directory, a dir_summary.json is written with
aggregated counts.  The Experiment Progress page reads this file as a fast
path to show success / failed / stale counts without re-reading every item.

Usage:
    python verify/patch_ext_ui.py               # patch all dirs, write summaries
    python verify/patch_ext_ui.py --dry-run     # detect only, no file writes
    python verify/patch_ext_ui.py --eval        # also re-evaluate ext_eval via LLM
    python verify/patch_ext_ui.py --dir PATH    # process a single directory
    python verify/patch_ext_ui.py --summarize   # write summaries only, do not patch
    python verify/patch_ext_ui.py --app deeptutor xend          # filter by app name
    python verify/patch_ext_ui.py --dataset PrivacyLens         # filter by dataset name
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **_):  # type: ignore[misc]
        return it

_VERIFY_ROOT  = Path(__file__).resolve().parent
_LANTERN_ROOT = _VERIFY_ROOT.parent
_OUTPUTS_DIR  = _VERIFY_ROOT / "outputs"

DIR_SUMMARY = "dir_summary.json"


# ── Per-app UI reconstruction ─────────────────────────────────────────────────

def _reconstruct_ui(app_name: str, output_text: str) -> Optional[str]:
    """
    Return the canonical full UI externalization string for this app,
    derived from output_text.  Returns None if no reconstruction is needed
    (app was never truncated) or the output_text is empty.
    """
    if not output_text:
        return None

    if app_name == "llm-vtuber":
        return f"[DISPLAY_TEXT] {output_text}"

    if app_name == "deeptutor":
        return f"[DISPLAY_RESPONSE] {output_text}"

    if app_name == "google-ai-edge-gallery":
        return f"[DISPLAY_RESPONSE] {output_text}"

    if app_name == "xend":
        # output_text = "Subject: {subject}\n\nBody:\n{body}"
        subject = ""
        for line in output_text.splitlines():
            if line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
                break
        return f"[NOTIFICATION] Email sent: {subject}"

    # clone / momentag / snapdo / budget-lens / etc.: UI was never truncated
    return None


def _build_ext_text(externalizations: Dict[str, str]) -> str:
    return "\n".join(
        f"[{ch.upper()}] {c}" for ch, c in externalizations.items()
    ) if externalizations else ""


# ── Item-level patch ──────────────────────────────────────────────────────────

_STATUS_CLEAN   = "clean"    # UI is already correct
_STATUS_STALE   = "stale"    # would patch (dry-run mode)
_STATUS_PATCHED = "patched"  # patched successfully
_STATUS_FAILED  = "failed"   # item status != success; skipped
_STATUS_ERROR   = "error"    # JSON parse / write error


def _patch_item(
    item_path: Path,
    app_name: str,
    *,
    dry_run: bool,
    do_eval: bool,
    summarize_only: bool,
) -> str:
    """
    Read, patch if stale, write back.  Returns one of the _STATUS_* constants.
    """
    try:
        item = json.loads(item_path.read_text())
    except Exception:
        return _STATUS_ERROR

    if item.get("status") != "success":
        return _STATUS_FAILED

    output_text = item.get("output_text", "")
    ext = item.get("externalizations") or {}

    reconstructed_ui = _reconstruct_ui(app_name, output_text)
    if reconstructed_ui is None:
        # App doesn't need patching; mark as clean so summary is accurate
        # but also respect any pre-existing stale flag
        if item.get("ext_eval_stale"):
            return "stale_eval_only"   # eval stale but text is fine
        return _STATUS_CLEAN

    cached_ui = ext.get("UI", "")
    if cached_ui == reconstructed_ui:
        if item.get("ext_eval_stale"):
            return "stale_eval_only"
        return _STATUS_CLEAN

    # ── Item is stale ─────────────────────────────────────────────────────────
    if dry_run or summarize_only:
        return _STATUS_STALE

    # Build patched item
    new_ext = {**ext, "UI": reconstructed_ui}
    new_ext_text = _build_ext_text(new_ext)

    item["externalizations"] = new_ext
    item["ext_text"]         = new_ext_text
    item["ext_eval_stale"]   = True   # ext_eval not yet re-scored

    if do_eval:
        try:
            if str(_LANTERN_ROOT) not in sys.path:
                sys.path.insert(0, str(_LANTERN_ROOT))
            from verify.backend.evaluation_method.evaluator import evaluate_inferability
            attrs = list(item.get("output_eval", {}).keys())
            if new_ext_text.strip() and attrs:
                ok, ext_eval, err = evaluate_inferability(new_ext_text, attrs)
                item["ext_eval"]       = ext_eval
                item["ext_eval_ok"]    = ok
                item["ext_eval_error"] = err
                item["ext_eval_stale"] = False
        except Exception:
            pass  # leave stale flag set

    try:
        item_path.write_text(json.dumps(item, indent=2, default=str))
    except Exception:
        return _STATUS_ERROR

    return _STATUS_PATCHED


# ── Directory-level processing ────────────────────────────────────────────────

def _process_dir(
    d: Path,
    *,
    dry_run: bool,
    do_eval: bool,
    summarize_only: bool,
    verbose: bool,
    show_progress: bool = False,
) -> Optional[Dict]:
    """
    Process all items in an output directory.
    Returns a summary dict or None if the directory is not a valid output dir.
    """
    config_path = d / "run_config.json"
    if not config_path.exists():
        return None

    try:
        cfg = json.loads(config_path.read_text())
        app     = cfg.get("app_name", "").strip()
        dataset = cfg.get("dataset_name", "").strip()
        if not app or not dataset:
            return None
    except Exception:
        return None

    counts: Dict[str, int] = {
        "total": 0, "success": 0, "failed": 0,
        "stale": 0, "patched": 0, "error": 0,
    }

    item_files = [
        f for f in d.iterdir()
        if f.suffix == ".json" and f.name not in ("run_config.json", DIR_SUMMARY)
    ]

    tag = f"{app}/{dataset}"
    item_iter = _tqdm(
        item_files,
        desc=tag,
        unit="item",
        leave=False,
        disable=not show_progress,
    )
    for f in item_iter:
        counts["total"] += 1
        result = _patch_item(
            f, app, dry_run=dry_run, do_eval=do_eval, summarize_only=summarize_only
        )
        if result == _STATUS_PATCHED:
            counts["success"] += 1
            counts["patched"] += 1
        elif result == _STATUS_STALE:
            counts["success"] += 1   # was a success run, just stale
            counts["stale"]   += 1
        elif result == _STATUS_CLEAN:
            counts["success"] += 1
        elif result == "stale_eval_only":
            counts["success"] += 1
            counts["stale"]   += 1
        elif result == _STATUS_FAILED:
            counts["failed"] += 1
        else:
            counts["error"] += 1

    # Write dir_summary.json (unless dry-run)
    if not dry_run:
        summary_data = {
            "app_name":            app,
            "dataset_name":        dataset,
            "modality":            cfg.get("modality", ""),
            "perturbation_method": cfg.get("perturbation_method", ""),
            **counts,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            (d / DIR_SUMMARY).write_text(json.dumps(summary_data, indent=2))
        except Exception:
            pass

    if verbose:
        mode = cfg.get("perturbation_method") or "ioc"
        line = (
            f"  [{mode:>14s}]  {tag:<45s}  "
            f"total={counts['total']:3d}  success={counts['success']:3d}  "
            f"failed={counts['failed']:3d}  stale={counts['stale']:3d}  "
            f"patched={counts['patched']:3d}"
        )
        # Use tqdm.write so the line appears above the progress bar cleanly
        try:
            from tqdm import tqdm
            tqdm.write(line)
        except ImportError:
            print(line)

    return {
        "dir":     str(d),
        "app":     app,
        "dataset": dataset,
        "cfg":     cfg,
        **counts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detect stale items and report counts without writing any files",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Re-evaluate ext_eval via LLM after patching (slow — one LLM call per item)",
    )
    parser.add_argument(
        "--summarize", action="store_true",
        help="Write dir_summary.json files only; do not patch item JSON files",
    )
    parser.add_argument(
        "--dir", metavar="PATH",
        help="Process only this single output directory",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-directory stats",
    )
    parser.add_argument(
        "--app", metavar="APP", nargs="+",
        help="Filter to specific app name(s), e.g. --app deeptutor xend",
    )
    parser.add_argument(
        "--dataset", metavar="DATASET", nargs="+",
        help="Filter to specific dataset name(s), e.g. --dataset PrivacyLens",
    )
    args = parser.parse_args()

    if args.dir:
        dirs = [Path(args.dir)]
    else:
        if not _OUTPUTS_DIR.exists():
            print(f"[ERROR] Outputs directory not found: {_OUTPUTS_DIR}")
            sys.exit(1)
        dirs = sorted(d for d in _OUTPUTS_DIR.iterdir() if d.is_dir())

    # ── App / dataset filtering ───────────────────────────────────────────────
    app_filter:     Optional[Set[str]] = set(args.app)     if args.app     else None
    dataset_filter: Optional[Set[str]] = set(args.dataset) if args.dataset else None

    if app_filter or dataset_filter:
        filtered: List[Path] = []
        for d in dirs:
            cfg_path = d / "run_config.json"
            if not cfg_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text())
                if app_filter and cfg.get("app_name", "").strip() not in app_filter:
                    continue
                if dataset_filter and cfg.get("dataset_name", "").strip() not in dataset_filter:
                    continue
                filtered.append(d)
            except Exception:
                pass
        dirs = filtered

    mode_label = (
        "DRY RUN"        if args.dry_run    else
        "SUMMARIZE ONLY" if args.summarize  else
        "PATCH" + (" + EVAL" if args.eval else "")
    )
    filter_label = "  ".join(filter(None, [
        f"app={','.join(sorted(app_filter))}"         if app_filter     else "",
        f"dataset={','.join(sorted(dataset_filter))}" if dataset_filter else "",
    ]))
    print(f"\n[patch_ext_ui]  mode={mode_label}  dirs={len(dirs)}"
          + (f"  filter: {filter_label}" if filter_label else "") + "\n")

    show_progress = not args.verbose   # verbose uses tqdm.write; non-verbose shows bars

    results: List[Dict] = []
    dir_iter = _tqdm(dirs, desc="Dirs", unit="dir", disable=args.verbose)
    for d in dir_iter:
        r = _process_dir(
            d,
            dry_run=args.dry_run,
            do_eval=args.eval,
            summarize_only=args.summarize,
            verbose=args.verbose,
            show_progress=show_progress,
        )
        if r is not None:
            results.append(r)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    total_items   = sum(r["total"]   for r in results)
    total_patched = sum(r["patched"] for r in results)
    total_stale   = sum(r["stale"]   for r in results)
    total_failed  = sum(r["failed"]  for r in results)
    total_success = sum(r["success"] for r in results)

    print(f"\n{'─' * 60}")
    print(f"  Dirs processed    : {len(results)}")
    print(f"  Items total       : {total_items}")
    print(f"  Items success     : {total_success}")
    print(f"  Items failed      : {total_failed}")
    print(f"  Stale UI events   : {total_stale}")
    print(f"  Patched           : {total_patched}")
    if args.dry_run or args.summarize:
        print(f"  (no files modified — {'dry-run' if args.dry_run else 'summarize-only'} mode)")

    # ── Fully failed directories ──────────────────────────────────────────────
    fully_failed = [r for r in results if r["success"] == 0 and r["total"] > 0]
    if fully_failed:
        print(f"\n{'─' * 60}")
        print(f"  Fully-failed directories ({len(fully_failed)}):")
        for r in fully_failed:
            mode = r["cfg"].get("perturbation_method") or "ioc_comparison"
            print(f"    {r['dir']}")
            print(
                f"      app={r['app']}  dataset={r['dataset']}  "
                f"method={mode}  total={r['total']}  failed={r['failed']}"
            )
    else:
        print("\n  No fully-failed directories found.")

    print()


if __name__ == "__main__":
    main()
