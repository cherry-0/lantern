"""
reeval.py — Re-evaluate ext_eval scores with a specified model.

Run --init once to stamp each cached result with the model that produced
the current ext_eval scores (no API calls — provenance labeling only).
Then use --model MODEL to re-evaluate with any OpenRouter-compatible model.

Add --prompt2 to use the MCQ prompt (prompt2.yaml) which produces a
"prediction" field per attribute alongside inferable/score/reasoning.
Useful for SynthPAI validation: run with --prompt2 first, then open the
"Evaluation Validation" Streamlit page to compare predictions against
the SynthPAI ground-truth profile.

Usage:
    python verify/reeval.py --init                              # stamp defaults
    python verify/reeval.py --model google/gemini-2.5-pro       # re-eval all
    python verify/reeval.py --model MODEL --dir PATH [PATH …]   # specific dirs
    python verify/reeval.py --model MODEL --app deeptutor xend  # filter by app
    python verify/reeval.py --model MODEL --dataset PrivacyLens # filter by dataset
    python verify/reeval.py --model MODEL --prompt2             # MCQ prompt (adds prediction field)
    python verify/reeval.py --dry-run --model MODEL             # preview only
    python verify/reeval.py --dry-run --init                    # preview init

--init and --model are mutually exclusive.
--prompt2 only applies to --model mode (ignored with --init).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DIR_SUMMARY   = "dir_summary.json"

if str(_LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(_LANTERN_ROOT))

from verify.backend.evaluation_method.evaluator import (
    EVAL_MODEL,
    evaluate_inferability,
    evaluate_inferability_v2,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _read_cfg(d: Path) -> Optional[Dict]:
    p = d / "run_config.json"
    if not p.exists():
        return None
    try:
        cfg = json.loads(p.read_text())
        if not cfg.get("app_name", "").strip() or not cfg.get("dataset_name", "").strip():
            return None
        return cfg
    except Exception:
        return None


def _item_files(d: Path) -> List[Path]:
    return [
        f for f in d.iterdir()
        if f.suffix == ".json" and f.name not in ("run_config.json", DIR_SUMMARY)
    ]


def _patch_summary(d: Path, eval_model: str) -> None:
    """Update eval_model / last_reeval in dir_summary.json, preserving other fields."""
    p = d / DIR_SUMMARY
    s: Dict = {}
    if p.exists():
        try:
            s = json.loads(p.read_text())
        except Exception:
            pass
    s["eval_model"]  = eval_model
    s["last_reeval"] = datetime.now(timezone.utc).isoformat()
    try:
        p.write_text(json.dumps(s, indent=2))
    except Exception:
        pass


def _tqdm_write(line: str) -> None:
    try:
        from tqdm import tqdm
        tqdm.write(line)
    except ImportError:
        print(line)


# ── Init mode ─────────────────────────────────────────────────────────────────

def _init_one(f: Path, *, dry_run: bool) -> str:
    """
    Stamp eval_model = EVAL_MODEL on an item that has ext_eval but no eval_model.
    Returns: "labeled" | "already" | "no_eval" | "skipped" | "error"
    """
    try:
        item = json.loads(f.read_text())
    except Exception:
        return "error"
    if item.get("status") != "success":
        return "skipped"
    if not item.get("ext_eval"):
        return "no_eval"
    if item.get("eval_model"):
        return "already"
    if dry_run:
        return "labeled"   # would label — counted as labeled for dry-run reporting
    item["eval_model"] = EVAL_MODEL
    try:
        f.write_text(json.dumps(item, indent=2, default=str))
    except Exception:
        return "error"
    return "labeled"


def _process_dir_init(
    d: Path,
    *,
    dry_run: bool,
    verbose: bool,
    show_progress: bool,
) -> Optional[Dict]:
    cfg = _read_cfg(d)
    if cfg is None:
        return None
    app     = cfg["app_name"].strip()
    dataset = cfg["dataset_name"].strip()
    tag     = f"{app}/{dataset}"
    files   = _item_files(d)

    counts: Dict[str, int] = {
        "labeled": 0, "already": 0, "no_eval": 0, "skipped": 0, "error": 0,
    }
    it = _tqdm(files, desc=tag, unit="item", leave=False, disable=not show_progress)
    for f in it:
        r = _init_one(f, dry_run=dry_run)
        counts[r] = counts.get(r, 0) + 1

    if not dry_run and counts["labeled"] > 0:
        _patch_summary(d, EVAL_MODEL)

    if verbose:
        mode = cfg.get("perturbation_method") or "ioc"
        _tqdm_write(
            f"  [{mode:>14s}]  {tag:<45s}  "
            f"labeled={counts['labeled']:3d}  already={counts['already']:3d}  "
            f"no_eval={counts['no_eval']:3d}  skipped={counts['skipped']:3d}"
        )
    return {"dir": str(d), "app": app, "dataset": dataset, "cfg": cfg, **counts}


# ── Re-eval mode ──────────────────────────────────────────────────────────────

def _reeval_one(
    f: Path, model: str, *, dry_run: bool, prompt_v2: bool = False
) -> Tuple[str, Optional[str]]:
    """
    Re-evaluate a single item with *model*.

    prompt_v2=True uses evaluate_inferability_v2 (MCQ prompt), which adds a
    "prediction" field to each attribute entry in ext_eval.

    Returns: (status, error_or_None)
    status: "success" | "failed" | "skipped" | "no_data" | "error"
    """
    try:
        item = json.loads(f.read_text())
    except Exception:
        return "error", "JSON parse error"

    if item.get("status") != "success":
        return "skipped", None

    ext_text = item.get("ext_text", "")
    attrs    = list(item.get("output_eval", {}).keys())

    if not ext_text.strip() or not attrs:
        return "no_data", None

    if dry_run:
        return "success", None   # would evaluate

    _eval_fn = evaluate_inferability_v2 if prompt_v2 else evaluate_inferability
    ok, ext_eval, err = _eval_fn(ext_text, attrs, model=model)

    item["ext_eval"]        = ext_eval
    item["ext_eval_ok"]     = ok
    item["ext_eval_error"]  = err
    item["eval_model"]      = model
    item["eval_prompt"]     = "prompt2" if prompt_v2 else "prompt1"
    item["ext_eval_stale"]  = False

    try:
        f.write_text(json.dumps(item, indent=2, default=str))
    except Exception:
        return "error", "write error"

    return ("success" if ok else "failed"), err


def _process_dir_reeval(
    d: Path,
    model: str,
    *,
    workers: int,
    dry_run: bool,
    verbose: bool,
    show_progress: bool,
    prompt_v2: bool = False,
) -> Optional[Dict]:
    cfg = _read_cfg(d)
    if cfg is None:
        return None
    app     = cfg["app_name"].strip()
    dataset = cfg["dataset_name"].strip()
    tag     = f"{app}/{dataset}"
    files   = _item_files(d)

    counts: Dict[str, int] = {
        "success": 0, "failed": 0, "skipped": 0, "no_data": 0, "error": 0,
    }

    def _tally(status: str, _err: Optional[str]) -> None:
        counts[status] = counts.get(status, 0) + 1

    if dry_run or workers <= 1:
        it = _tqdm(files, desc=tag, unit="item", leave=False, disable=not show_progress)
        for f in it:
            _tally(*_reeval_one(f, model, dry_run=dry_run, prompt_v2=prompt_v2))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_reeval_one, f, model, dry_run=False, prompt_v2=prompt_v2): f
                for f in files
            }
            prog = _tqdm(
                as_completed(futs), total=len(files),
                desc=tag, unit="item", leave=False, disable=not show_progress,
            )
            for fut in prog:
                _tally(*fut.result())

    if not dry_run:
        _patch_summary(d, model)

    if verbose:
        mode   = cfg.get("perturbation_method") or "ioc"
        prompt = "prompt2" if prompt_v2 else "prompt1"
        _tqdm_write(
            f"  [{mode:>14s}]  {tag:<45s}  [{prompt}]  "
            f"success={counts['success']:3d}  failed={counts['failed']:3d}  "
            f"skipped={counts['skipped']:3d}  no_data={counts['no_data']:3d}"
        )
    return {
        "dir": str(d), "app": app, "dataset": dataset,
        "cfg": cfg, "model": model, "prompt": "prompt2" if prompt_v2 else "prompt1",
        **counts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--init", action="store_true",
        help=(
            f"Stamp existing ext_eval items with eval_model={EVAL_MODEL!r}. "
            "No API calls; items already labeled are skipped."
        ),
    )
    mode_group.add_argument(
        "--model", metavar="MODEL",
        help="Re-evaluate all matching items with this OpenRouter model ID",
    )

    parser.add_argument(
        "--dir", metavar="PATH", nargs="+",
        help="Process specific output directories (space-separated absolute paths)",
    )
    parser.add_argument(
        "--app", metavar="APP", nargs="+",
        help="Filter to specific app name(s), e.g. --app deeptutor xend",
    )
    parser.add_argument(
        "--dataset", metavar="DATASET", nargs="+",
        help="Filter to specific dataset name(s), e.g. --dataset PrivacyLens",
    )
    parser.add_argument(
        "--prompt2", action="store_true",
        help=(
            "Use the MCQ evaluation prompt (prompt2.yaml). "
            "Adds a 'prediction' field to each attribute in ext_eval. "
            "Only applies with --model; ignored with --init."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel API calls per directory for --model (default: 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview actions without writing any files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-directory stats",
    )
    args = parser.parse_args()

    # ── Collect directories ───────────────────────────────────────────────────
    if args.dir:
        dirs = [Path(p) for p in args.dir]
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
                if app_filter     and cfg.get("app_name",     "").strip() not in app_filter:
                    continue
                if dataset_filter and cfg.get("dataset_name", "").strip() not in dataset_filter:
                    continue
                filtered.append(d)
            except Exception:
                pass
        dirs = filtered

    # ── Header ────────────────────────────────────────────────────────────────
    prompt_v2  = bool(args.prompt2) and not args.init
    prompt_tag = "  prompt=prompt2" if prompt_v2 else ""
    mode_label = "INIT" if args.init else f"REEVAL  model={args.model}{prompt_tag}"
    if args.dry_run:
        mode_label = f"DRY RUN ({mode_label})"

    filter_parts: List[str] = []
    if app_filter:     filter_parts.append(f"app={','.join(sorted(app_filter))}")
    if dataset_filter: filter_parts.append(f"dataset={','.join(sorted(dataset_filter))}")
    filter_str = ("  filter: " + "  ".join(filter_parts)) if filter_parts else ""

    print(f"\n[reeval]  mode={mode_label}  dirs={len(dirs)}{filter_str}\n")

    # ── Process ───────────────────────────────────────────────────────────────
    show_progress = not args.verbose
    results: List[Dict] = []

    dir_iter = _tqdm(dirs, desc="Dirs", unit="dir", disable=args.verbose)
    for d in dir_iter:
        if args.init:
            r = _process_dir_init(
                d,
                dry_run=args.dry_run,
                verbose=args.verbose,
                show_progress=show_progress,
            )
        else:
            r = _process_dir_reeval(
                d, args.model,
                workers=args.workers,
                dry_run=args.dry_run,
                verbose=args.verbose,
                show_progress=show_progress,
                prompt_v2=prompt_v2,
            )
        if r is not None:
            results.append(r)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Dirs processed : {len(results)}")

    if args.init:
        print(f"  Labeled        : {sum(r.get('labeled',  0) for r in results)}")
        print(f"  Already had    : {sum(r.get('already',  0) for r in results)}")
        print(f"  No eval data   : {sum(r.get('no_eval',  0) for r in results)}")
        print(f"  Skipped        : {sum(r.get('skipped',  0) for r in results)}")
    else:
        print(f"  Re-eval success: {sum(r.get('success',  0) for r in results)}")
        print(f"  Re-eval failed : {sum(r.get('failed',   0) for r in results)}")
        print(f"  Skipped        : {sum(r.get('skipped',  0) for r in results)}")
        print(f"  No data        : {sum(r.get('no_data',  0) for r in results)}")

    if args.dry_run:
        print("  (dry-run — no files modified)")
    print()


if __name__ == "__main__":
    main()
