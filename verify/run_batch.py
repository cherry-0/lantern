"""
Batch evaluation runner for the Verify privacy framework.

Reads a CSV of (app, modality, dataset, perturbation_method) configs and
runs Input-Output Comparison (IOC) and/or perturbation analysis for each row,
parallelising across rows with a ThreadPoolExecutor.

Attributes are always loaded from the modality-specific config files:
  image → verify/config/attribute_list_image.txt
  text  → verify/config/attribute_list.txt

Results land in the standard verify/outputs/ structure so they are immediately
viewable in the Streamlit frontend (View Results / View IOC Results pages).

Usage:
    # Run all enabled rows, both IOC and perturbation analysis
    python run_batch.py

    # IOC only, 4 parallel workers, limit 20 items per run
    python run_batch.py --mode ioc --workers 4 --max-items 20

    # Perturbation only, use a custom config file
    python run_batch.py --mode perturb --config my_config.csv

    # Dry run: print the execution plan without running anything
    python run_batch.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── sys.path: make verify importable regardless of CWD ───────────────────────
_VERIFY_ROOT = Path(__file__).resolve().parent
_LANTERN_ROOT = _VERIFY_ROOT.parent
for _p in (str(_LANTERN_ROOT), str(_VERIFY_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Attribute loading from config files ──────────────────────────────────────

_CONFIG_DIR = _VERIFY_ROOT / "config"

# Cache so each file is only read once
_ATTR_CACHE: Dict[str, List[str]] = {}
_EVAL_PROMPT_CHOICES = ("prompt1", "prompt2", "prompt3")


def _load_attrs_for_modality(modality: str) -> List[str]:
    """
    Return the attribute list for the given modality by reading the
    appropriate config file:
      image  → config/attribute_list_image.txt
      text   → config/attribute_list.txt
      other  → config/attribute_list_unified.txt  (full union)
    """
    if modality in _ATTR_CACHE:
        return _ATTR_CACHE[modality]

    filename_map = {
        "image": "attribute_list_image.txt",
        "text":  "attribute_list.txt",
    }
    fname = filename_map.get(modality, "attribute_list_unified.txt")
    path = _CONFIG_DIR / fname

    if not path.exists():
        # Graceful fallback to unified list
        path = _CONFIG_DIR / "attribute_list_unified.txt"

    attrs = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ] if path.exists() else []

    _ATTR_CACHE[modality] = attrs
    return attrs


# ── Thread-safe logger ────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def _log(tag: str, msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{ts}] {level:5s} [{tag}] {msg}", flush=True)


# ── Terminal bar chart helpers ────────────────────────────────────────────────

_BAR_FILLED = "█"
_BAR_EMPTY  = "░"
_BAR_WIDTH  = 10       # visible characters per bar segment

_ANSI_BLUE   = "\033[94m"   # Input / Perturbed
_ANSI_RED    = "\033[91m"   # Raw Output / Original
_ANSI_YELLOW = "\033[93m"   # Externalized
_ANSI_RESET  = "\033[0m"
_USE_ANSI    = sys.stdout.isatty()


def _colorize(text: str, code: str) -> str:
    return f"{code}{text}{_ANSI_RESET}" if _USE_ANSI else text


def _bar(rate: float) -> str:
    """Return a fixed-width block-character bar for rate ∈ [0, 1]."""
    filled = round(max(0.0, min(1.0, rate)) * _BAR_WIDTH)
    return _BAR_FILLED * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)


_CHART_SEP = "─" * 74


def _print_ioc_chart(result: Dict[str, Any]) -> None:
    from verify.backend.evaluation_method.evaluator import get_aggregate_eval_entry

    """
    Print an attribute-wise terminal bar chart for an IOC result.
    Three bars per attribute — Input (annotation rate), Raw Output (inferability),
    Externalized (ext inferability) — matching the Streamlit aggregated chart.
    Called once per config row as soon as IOC finishes.
    """
    items = [r for r in result.get("items", []) if "output_eval" in r]
    if not items:
        return

    attrs = _load_attrs_for_modality(result.get("modality", ""))
    if not attrs:
        return

    n = len(items)
    tag_str = f"{result.get('app', '?')} / {result.get('dataset', '?')}"

    lines: List[str] = []
    lines.append(_CHART_SEP)
    lines.append(f"  IOC Results: {tag_str}  ({n} item{'s' if n != 1 else ''})")
    lines.append(_CHART_SEP)
    lines.append(
        f"  {'Attribute':<22}  "
        f"{'Input':<17}  "
        f"{'Raw Output':<17}  "
        f"{'Externalized':<17}"
    )
    lines.append(f"  {'─'*22}  {'─'*17}  {'─'*17}  {'─'*17}")

    for attr in attrs:
        in_cnt = sum(1 for r in items if r.get("input_labels", {}).get(attr))
        out_cnt = sum(
            1 for r in items
            if isinstance(r.get("output_eval", {}).get(attr), dict)
            and r["output_eval"][attr].get("inferable")
        )
        ext_cnt = sum(
            1 for r in items
            if bool(get_aggregate_eval_entry(r.get("ext_eval", {}).get(attr)).get("inferable"))
        )
        in_bar  = _colorize(_bar(in_cnt  / n), _ANSI_BLUE)
        out_bar = _colorize(_bar(out_cnt / n), _ANSI_RED)
        ext_bar = _colorize(_bar(ext_cnt / n), _ANSI_YELLOW)
        lines.append(
            f"  {attr:<22}  "
            f"{in_bar} {in_cnt}/{n:<5}  "
            f"{out_bar} {out_cnt}/{n:<5}  "
            f"{ext_bar} {ext_cnt}/{n:<5}"
        )

    lines.append("")
    with _print_lock:
        print("\n".join(lines), flush=True)


def _print_perturb_chart(result: Dict[str, Any]) -> None:
    """
    Print an attribute-wise terminal bar chart for a perturbation result.
    Two bars per attribute — Original (red) and Perturbed (blue) — plus delta,
    matching the Streamlit aggregated chart.
    Called once per config row as soon as perturbation finishes.
    """
    attrs = result.get("attributes", [])
    eval_results = result.get("eval_results", [])  # per-item evaluation dicts

    # ── Path A: item-level evaluations available ──────────────────────────────
    if eval_results:
        n = len(eval_results)
        tag_str = f"{result.get('app', '?')} / {result.get('dataset', '?')}"

        lines: List[str] = []
        lines.append(_CHART_SEP)
        lines.append(f"  Perturb Results: {tag_str}  ({n} item{'s' if n != 1 else ''})")
        lines.append(_CHART_SEP)
        lines.append(
            f"  {'Attribute':<22}  "
            f"{'Original':<17}  "
            f"{'Perturbed':<17}  "
            f"{'Δ':>7}"
        )
        lines.append(f"  {'─'*22}  {'─'*17}  {'─'*17}  {'─'*7}")

        for attr in attrs:
            orig_cnt = sum(
                1 for ev in eval_results
                if isinstance(ev, dict)
                and ev.get("original", {}).get(attr, {}).get("inferable")
            )
            pert_cnt = sum(
                1 for ev in eval_results
                if isinstance(ev, dict)
                and ev.get("perturbed", {}).get(attr, {}).get("inferable")
            )
            orig_rate = orig_cnt / n
            pert_rate = pert_cnt / n
            delta = pert_rate - orig_rate

            o_bar = _colorize(_bar(orig_rate), _ANSI_RED)
            p_bar = _colorize(_bar(pert_rate), _ANSI_BLUE)
            lines.append(
                f"  {attr:<22}  "
                f"{o_bar} {orig_cnt}/{n:<5}  "
                f"{p_bar} {pert_cnt}/{n:<5}  "
                f"{delta:>+7.2f}"
            )

        lines.append("")
        with _print_lock:
            print("\n".join(lines), flush=True)
        return

    # ── Path B: fall back to aggregated scores from the summary event ─────────
    summary = result.get("summary") or {}
    agg = summary.get("aggregated_scores", {})
    n = result.get("n_success", 0) + result.get("n_cached", 0)
    if not agg or not attrs or not n:
        return

    orig_agg = agg.get("original", {})
    pert_agg = agg.get("perturbed", {})
    tag_str = f"{result.get('app', '?')} / {result.get('dataset', '?')}"

    lines = []
    lines.append(_CHART_SEP)
    lines.append(f"  Perturb Results: {tag_str}  ({n} item{'s' if n != 1 else ''}, avg scores)")
    lines.append(_CHART_SEP)
    lines.append(
        f"  {'Attribute':<22}  "
        f"{'Original':<17}  "
        f"{'Perturbed':<17}  "
        f"{'Δ':>7}"
    )
    lines.append(f"  {'─'*22}  {'─'*17}  {'─'*17}  {'─'*7}")

    for attr in attrs:
        o_rate = orig_agg.get(attr) or 0.0
        p_rate = pert_agg.get(attr) or 0.0
        delta  = p_rate - o_rate
        o_bar  = _colorize(_bar(o_rate), _ANSI_RED)
        p_bar  = _colorize(_bar(p_rate), _ANSI_BLUE)
        lines.append(
            f"  {attr:<22}  "
            f"{o_bar} {o_rate:.2f}     "
            f"  {p_bar} {p_rate:.2f}     "
            f"  {delta:>+7.2f}"
        )

    lines.append("")
    with _print_lock:
        print("\n".join(lines), flush=True)


# ── Config loading ────────────────────────────────────────────────────────────

def _load_unified_attrs() -> List[str]:
    path = _VERIFY_ROOT / "config" / "attribute_list_unified.txt"
    if not path.exists():
        return _IMAGE_DEFAULT_ATTRS + [a for a in _TEXT_DEFAULT_ATTRS if a not in _IMAGE_DEFAULT_ATTRS]
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_csv(path: Path) -> List[Dict[str, str]]:
    """Parse the batch config CSV, skipping comment lines and blank rows."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#")),
        )
        for row in reader:
            # Skip rows where all values are blank
            cleaned = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if not any(v for v in cleaned.values()):
                continue
            cleaned.setdefault("generation_task", "")
            rows.append(cleaned)
    return rows




def _row_tag(row: Dict[str, str], mode: str) -> str:
    generation_task = row.get("generation_task", "") or "text"
    if row.get("app_name") == "tool-neuron" and row.get("modality") == "text":
        return f"{mode}/{row['app_name']}/{row['dataset_name']}/{generation_task}"
    return f"{mode}/{row['app_name']}/{row['dataset_name']}"


def _ioc_cache_eval_method(eval_prompt: str, generation_task: str = "text") -> str:
    method = "openrouter" if eval_prompt == "prompt1" else f"openrouter:{eval_prompt}"
    if generation_task != "text":
        method = f"{method}:task={generation_task}"
    return method


def _select_ioc_ext_eval_fn(eval_prompt: str):
    from verify.backend.evaluation_method.evaluator import (
        evaluate_inferability,
        evaluate_inferability_v2,
        evaluate_inferability_v3,
    )

    if eval_prompt == "prompt3":
        return evaluate_inferability_v3
    if eval_prompt == "prompt2":
        return evaluate_inferability_v2
    return evaluate_inferability


# ── IOC (Input-Output Comparison) pipeline ───────────────────────────────────

def _run_ioc(
    row: Dict[str, str],
    unified_attrs: List[str],
    max_items: Optional[int],
    use_cache: bool,
    item_workers: int = 1,
    eval_prompt: str = "prompt1",
) -> Dict[str, Any]:
    """
    Headless IOC pipeline matching run_comparison_pipeline() in
    2_Input_Output_Comparison.py exactly:
      - same cache key (ioc_comparison, no attrs, eval_prompt-specific)
      - same result schema (input_item, prompt_text, error field on all rows)
      - failed items saved to item_results with full schema
      - PIL stripped from input_item before writing to cache
    """
    app_name = row["app_name"]
    dataset_name = row["dataset_name"]
    modality = row["modality"]
    generation_task = row.get("generation_task", "") or "text"
    tag = _row_tag(row, "ioc")
    row_max = int(row["max_items"]) if row.get("max_items") else max_items

    from verify.backend.adapters import get_adapter
    from verify.backend.datasets.loader import iter_dataset
    from verify.backend.datasets.label_mapper import get_input_labels
    from verify.backend.evaluation_method.evaluator import evaluate_inferability
    from verify.backend.utils import cache as cache_module
    from verify.backend.utils.config import set_current_app_context

    def _strip_pil(item: Dict) -> Dict:
        """Return item without PIL Image objects (key 'data')."""
        return {k: v for k, v in item.items() if k != "data"}

    def _failed_result(filename: str, error: str, input_item: Dict,
                       input_labels: Dict) -> Dict[str, Any]:
        return {
            "filename": filename,
            "status": "failed",
            "error": error,
            "modality": modality,
            "input_item": _strip_pil(input_item),
            "input_labels": input_labels,
            "output_text": "",
            "externalizations": {},
            "ext_text": "",
            "output_eval": {},
            "ext_eval": {},
            "output_eval_ok": False,
            "ext_eval_ok": False,
            "output_eval_error": None,
            "ext_eval_error": None,
            "prompt_text": "",
            "eval_prompt": eval_prompt,
            "from_cache": False,
        }

    adapter = get_adapter(app_name)
    if adapter is None:
        _log(tag, f"No adapter registered for '{app_name}'", "ERROR")
        return {"mode": "ioc", "tag": tag, "error": f"No adapter for '{app_name}'",
                "n_success": 0, "n_failed": 0, "n_cached": 0, "items": [],
                "app": app_name, "dataset": dataset_name, "modality": modality,
                "eval_prompt": eval_prompt}

    cache_dir = (
        cache_module.get_cache_dir(
            app_name,
            dataset_name,
            modality,
            [],
            "ioc_comparison",
            _ioc_cache_eval_method(eval_prompt, generation_task),
        )
        if use_cache else None
    )

    if cache_dir is not None:
        run_config_path = cache_dir / "run_config.json"
        if not run_config_path.exists():
            cache_module.save_run_config(cache_dir, {
                "app_name": app_name,
                "dataset_name": dataset_name,
                "modality": modality,
                "generation_task": generation_task,
                "unified_attrs": unified_attrs,
                "perturbation_method": "ioc_comparison",
                "evaluation_method": _ioc_cache_eval_method(eval_prompt, generation_task),
                "eval_prompt": eval_prompt,
            })

    n_success = n_failed = n_cached = 0
    item_results: List[Dict] = []

    _log(tag, f"Starting IOC ({dataset_name}, {modality}, task={generation_task})")
    ext_eval_fn = _select_ioc_ext_eval_fn(eval_prompt)

    def _process_one(ok: bool, item: Dict, err: Optional[str]) -> Dict:
        """Process a single non-cached dataset item (runs in a worker thread)."""
        filename = item.get("filename", "unknown")
        if app_name == "tool-neuron":
            item = {**item, "generation_task": generation_task}

        if not ok:
            _log(tag, f"  load error {filename}: {err}", "WARN")
            return _failed_result(filename, err or "Dataset load error", item, {})

        # 1. Input labels
        input_labels = get_input_labels(item, unified_attrs)

        # 2. Run original pipeline
        try:
            set_current_app_context(app_name)
            adapter._reset_openrouter_calls()
            pipeline_result = adapter.run_pipeline(item)
        except Exception as e:
            _log(tag, f"  pipeline error {filename}: {e}", "ERROR")
            return _failed_result(filename, f"Pipeline error: {e}", item, input_labels)

        if not pipeline_result.success:
            _log(tag, f"  pipeline failed {filename}: {pipeline_result.error}", "WARN")
            return _failed_result(
                filename, pipeline_result.error or "Adapter returned failure.",
                item, input_labels,
            )

        # 3. Evaluate output and externalizations
        output_text    = pipeline_result.output_text or ""
        externalizations = pipeline_result.externalizations or {}
        ext_text = "\n".join(
            f"[{ch.upper()}] {c}" for ch, c in externalizations.items()
        ) if externalizations else ""

        out_ok, output_eval, out_err = evaluate_inferability(output_text, unified_attrs)
        if ext_text.strip():
            ext_ok, ext_eval, ext_err = ext_eval_fn(ext_text, unified_attrs)
        else:
            ext_ok, ext_eval, ext_err = True, {}, None

        ioc_item: Dict[str, Any] = {
            "filename": filename,
            "status": "success",
            "error": None,
            "modality": modality,
            "input_item": item,
            "input_labels": input_labels,
            "output_text": output_text,
            "externalizations": externalizations,
            "ext_text": ext_text,
            "output_eval": output_eval,
            "ext_eval": ext_eval,
            "output_eval_ok": out_ok,
            "ext_eval_ok": ext_ok,
            "output_eval_error": out_err,
            "ext_eval_error": ext_err,
            "prompt_text": (pipeline_result.metadata or {}).get("prompt_text", ""),
            "eval_prompt": eval_prompt,
            "from_cache": False,
        }

        # ── Save to cache (strip PIL objects) ─────────────────────────────────
        if cache_dir is not None:
            saveable = {**ioc_item, "input_item": _strip_pil(item)}
            cache_module.save_item_cache(cache_dir, filename, saveable)

        _log(tag, f"  ✓ {filename}  out_ok={out_ok}  ext_ok={ext_ok}")
        return ioc_item

    # Collect all dataset items first (avoids tqdm races when parallelising)
    all_items: List[Tuple[bool, Dict, Optional[str]]] = list(
        iter_dataset(dataset_name, modality, max_items=row_max)
    )

    # Separate cached from non-cached in the main thread
    pending: List[Tuple[bool, Dict, Optional[str]]] = []
    for ok, item, err in all_items:
        if app_name == "tool-neuron":
            item = {**item, "generation_task": generation_task}
        filename = item.get("filename", "unknown")
        if cache_dir is not None:
            cached = cache_module.load_item_cache(
                cache_dir,
                filename,
                expected_eval_prompt=eval_prompt,
            )
            if cached is not None:
                cached.setdefault("input_item", _strip_pil(item))
                cached["eval_prompt"] = cache_module.normalize_eval_prompt(cached.get("eval_prompt"))
                cached["from_cache"] = True
                n_cached += 1
                item_results.append(cached)
                continue
        pending.append((ok, item, err))

    _log(
        tag,
        f"Cache scan — cached={n_cached} pending={len(pending)} total={len(all_items)}",
    )

    # Process non-cached items — parallel when item_workers > 1
    if item_workers > 1 and pending:
        with ThreadPoolExecutor(max_workers=item_workers) as ipool:
            futs = [ipool.submit(_process_one, ok, item, err) for ok, item, err in pending]
            for fut in as_completed(futs):
                res = fut.result()
                item_results.append(res)
                if res.get("status") == "success":
                    n_success += 1
                else:
                    n_failed += 1
    else:
        for ok, item, err in pending:
            res = _process_one(ok, item, err)
            item_results.append(res)
            if res.get("status") == "success":
                n_success += 1
            else:
                n_failed += 1

    _log(tag, f"Done — success={n_success} cached={n_cached} failed={n_failed}")

    result = {
        "mode": "ioc",
        "tag": tag,
        "app": app_name,
        "dataset": dataset_name,
        "modality": modality,
        "generation_task": generation_task,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_cached": n_cached,
        "items": item_results,
        "error": None,
        "eval_prompt": eval_prompt,
    }
    _print_ioc_chart(result)
    return result


# ── Perturbation analysis pipeline ───────────────────────────────────────────

def _run_perturb(
    row: Dict[str, str],
    unified_attrs: List[str],  # unused here; attrs come from _load_attrs_for_modality
    max_items: Optional[int],
    use_cache: bool,
    item_workers: int = 1,
    eval_prompt: str = "prompt1",
) -> Dict[str, Any]:
    """
    Perturbation analysis: original pipeline → perturb → perturbed pipeline → evaluate.
    Delegates entirely to Orchestrator; results land in verify/outputs/ run directories.
    """
    app_name = row["app_name"]
    dataset_name = row["dataset_name"]
    modality = row["modality"]
    generation_task = row.get("generation_task", "") or "text"
    perturbation_method = row.get("perturbation_method") or None
    cache_method = perturbation_method
    if app_name == "tool-neuron" and generation_task != "text":
        suffix = f"task={generation_task}"
        cache_method = f"{perturbation_method}::{suffix}" if perturbation_method else suffix
    attributes = _load_attrs_for_modality(modality)
    tag = _row_tag(row, "perturb")
    row_max = int(row["max_items"]) if row.get("max_items") else max_items

    if not attributes:
        _log(tag, "No valid attributes — skipping", "WARN")
        return {"mode": "perturb", "tag": tag, "error": "No valid attributes",
                "n_success": 0, "n_failed": 0, "n_cached": 0, "summary": None}

    from verify.backend.orchestrator import Orchestrator

    _log(tag, f"Starting perturb ({dataset_name}, {modality}, task={generation_task}, attrs={attributes}, method={perturbation_method})")

    orch = Orchestrator(
        app_name=app_name,
        dataset_name=dataset_name,
        modality=modality,
        attributes=attributes,
        use_cache=use_cache,
        max_items=row_max,
        perturbation_method=cache_method,
        item_workers=item_workers,
        adapter_kwargs={"generation_task": generation_task} if app_name == "tool-neuron" else None,
    )

    summary: Optional[Dict] = None
    n_success = n_failed = n_cached = 0
    eval_results: List[Dict] = []   # per-item evaluation dicts for charting

    for event in orch.run():
        etype = event.get("type", "")
        if etype == "summary":
            summary = event
        elif etype == "item_result":
            fname = event.get("filename", "")
            status = event.get("status", "")
            cached = event.get("from_cache", False)
            if cached:
                n_cached += 1
            elif status == "success":
                n_success += 1
            else:
                n_failed += 1
            # Collect evaluation for charting (success items only)
            if status == "success":
                ev = event.get("evaluation")
                if isinstance(ev, dict):
                    eval_results.append(ev)
            _log(tag, f"  {'(cached) ' if cached else ''}{fname}  [{status}]")
        elif etype == "error":
            _log(tag, f"  error: {event.get('error', '')}", "ERROR")

    run_dir = summary.get("run_dir", "") if summary else ""
    _log(tag, f"Done — success={n_success} cached={n_cached} failed={n_failed}  → {run_dir}")

    result = {
        "mode": "perturb",
        "tag": tag,
        "app": app_name,
        "dataset": dataset_name,
        "modality": modality,
        "generation_task": generation_task,
        "attributes": attributes,
        "perturbation_method": perturbation_method,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_cached": n_cached,
        "summary": summary,
        "eval_results": eval_results,
        "error": None,
    }
    _print_perturb_chart(result)
    return result


# ── Summary printer ───────────────────────────────────────────────────────────

def _avg(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _print_summary(results: List[Dict[str, Any]]) -> None:
    from verify.backend.evaluation_method.evaluator import get_aggregate_eval_entry

    print("\n" + "=" * 72)
    print("BATCH SUMMARY")
    print("=" * 72)

    for r in results:
        if r.get("error"):
            print(f"\n  ✗ {r['tag']}  ERROR: {r['error']}")
            continue

        mode = r["mode"]
        tag = r["tag"]
        n_ok = r["n_success"]
        n_ca = r["n_cached"]
        n_fa = r["n_failed"]
        total = n_ok + n_ca + n_fa

        print(f"\n  {'IOC' if mode == 'ioc' else 'PERTURB'}  {tag}")
        print(f"    items: {total} total  ({n_ok} new  {n_ca} cached  {n_fa} failed)")

        if mode == "ioc":
            items = r.get("items", [])
            # Compute avg output_inferable and ext_inferable per attr
            out_scores: Dict[str, List[int]] = {}
            ext_scores: Dict[str, List[int]] = {}
            for item in items:
                for attr, ev in item.get("output_eval", {}).items():
                    out_scores.setdefault(attr, []).append(int(ev.get("score", 0)))
                for attr, ev in item.get("ext_eval", {}).items():
                    ext_scores.setdefault(attr, []).append(
                        int(get_aggregate_eval_entry(ev).get("score", 0))
                    )

            if out_scores:
                print("    eval prompt:")
                print(
                    f"      externalized = {r.get('eval_prompt', 'prompt1')}  "
                    f"raw_output = prompt1"
                )
                print("    avg output_inferable:")
                for attr in sorted(out_scores):
                    avg = _avg(out_scores[attr])
                    bar = "█" * int((avg or 0) * 10)
                    print(f"      {attr:<20s}  {avg:.2f}  {bar}" if avg is not None else f"      {attr:<20s}  —")
            if ext_scores:
                print("    avg ext_inferable:")
                for attr in sorted(ext_scores):
                    avg = _avg(ext_scores[attr])
                    bar = "█" * int((avg or 0) * 10)
                    print(f"      {attr:<20s}  {avg:.2f}  {bar}" if avg is not None else f"      {attr:<20s}  —")

        elif mode == "perturb":
            summary = r.get("summary") or {}
            agg = summary.get("aggregated_scores", {})
            attrs = r.get("attributes", [])
            if agg and attrs:
                print(f"    attributes: {', '.join(attrs)}")
                print(f"    {'attr':<20s}  {'orig':>6s}  {'pert':>6s}  {'Δ':>6s}")
                print(f"    {'-'*20}  {'------':>6s}  {'------':>6s}  {'------':>6s}")
                orig_agg = agg.get("original", {})
                pert_agg = agg.get("perturbed", {})
                for attr in attrs:
                    o = orig_agg.get(attr)
                    p = pert_agg.get(attr)
                    delta = (p - o) if (o is not None and p is not None) else None
                    o_str = f"{o:.3f}" if o is not None else "  —  "
                    p_str = f"{p:.3f}" if p is not None else "  —  "
                    d_str = f"{delta:+.3f}" if delta is not None else "  —  "
                    print(f"    {attr:<20s}  {o_str:>6s}  {p_str:>6s}  {d_str:>6s}")

    print("\n" + "=" * 72 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation runner for Verify.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=str(_VERIFY_ROOT / "batch_config.csv"),
        help="Path to batch config CSV (default: verify/batch_config.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["ioc", "perturb", "both"],
        default="both",
        help="Which analysis to run (default: both)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max parallel workers across configs (default: 4)",
    )
    parser.add_argument(
        "--item-workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel workers per config for item-level processing (default: 4)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help="Global item cap per run; row-level max_items overrides this",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching (re-run everything)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan and exit without running",
    )
    parser.add_argument(
        "--eval-prompt",
        choices=_EVAL_PROMPT_CHOICES,
        default="prompt1",
        help=(
            "IOC externalization evaluation prompt "
            "(prompt1=binary, prompt2=prediction, prompt3=channel-wise aggregate)"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    unified_attrs = _load_unified_attrs()
    all_rows = _load_csv(config_path)

    # Filter enabled rows (default: enabled if column absent or blank)
    rows = [
        r for r in all_rows
        if r.get("enabled", "true").lower() not in ("false", "0", "no")
    ]

    if not rows:
        print("No enabled rows found in config. Exiting.")
        return

    # Build task list: (fn, row)
    Task = Tuple  # (callable, row_dict)
    tasks: List[Tuple[Any, Dict]] = []
    for row in rows:
        if args.mode in ("ioc", "both"):
            tasks.append((_run_ioc, row))
        if args.mode in ("perturb", "both"):
            tasks.append((_run_perturb, row))

    # Dry-run: print plan and exit
    if args.dry_run:
        print(f"\nBatch plan  ({len(tasks)} tasks, {args.workers} workers):\n")
        for fn, row in tasks:
            mode_label = "IOC    " if fn is _run_ioc else "PERTURB"
            attrs = _load_attrs_for_modality(row.get("modality", ""))
            method = row.get("perturbation_method") or "(config default)"
            row_max = row.get("max_items") or args.max_items or "all"
            generation_task = row.get("generation_task", "") or "text"
            print(f"  {mode_label}  {row['app_name']:<30s}  {row['dataset_name']:<12s}"
                  f"  {row['modality']:<6s}  task={generation_task:<5s}  method={method}  items={row_max}")
            if fn is _run_ioc:
                print(f"            ext_eval_prompt: {args.eval_prompt}  (raw_output=prompt1)")
            if fn is _run_perturb:
                print(f"            attrs: {', '.join(attrs) or '(none — row will be skipped)'}")
        print()
        return

    use_cache = not args.no_cache

    # Pre-import dataset libraries in the main thread to avoid tqdm thread-init
    # races when multiple workers start loading HuggingFace datasets simultaneously.
    has_text = any(r.get("modality") == "text" for r in rows)
    if has_text:
        try:
            import tqdm  # noqa: F401
            import datasets as _ds  # noqa: F401
        except ImportError:
            pass

    print(f"\nRunning {len(tasks)} tasks across {len(rows)} config rows  "
          f"(workers={args.workers}, item_workers={args.item_workers}, "
          f"cache={'on' if use_cache else 'off'}, "
          f"max_items={args.max_items or 'all'}, "
          f"ioc_ext_eval={args.eval_prompt})\n")

    results: List[Dict[str, Any]] = []
    futures_map: Dict[Any, Tuple[Any, Dict]] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fn, row in tasks:
            fut = pool.submit(fn, row, unified_attrs, args.max_items, use_cache,
                              args.item_workers, args.eval_prompt)
            futures_map[fut] = (fn, row)

        for fut in as_completed(futures_map):
            fn, row = futures_map[fut]
            try:
                result = fut.result()
            except Exception:
                tb = traceback.format_exc()
                tag = _row_tag(row, "ioc" if fn is _run_ioc else "perturb")
                _log(tag, f"Unhandled exception:\n{tb}", "ERROR")
                result = {
                    "mode": "ioc" if fn is _run_ioc else "perturb",
                    "tag": tag,
                    "error": tb,
                    "n_success": 0, "n_failed": 0, "n_cached": 0,
                }
            results.append(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
