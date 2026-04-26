"""
Experiment Progress — app × dataset coverage matrix.

Shows how many items have been successfully cached (M) out of the full dataset
size (N) for every (app, dataset) combination listed in batch_config.csv.
Failed items are excluded from M.  Items whose ext_eval was scored on a
truncated UI event are annotated as "(S stale)".

Run  verify/patch_ext_ui.py  to reconstruct full UI events and write
per-directory summaries that power the stale / failed counts shown here.

IOC and Perturb results are shown in separate tabs.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT  = Path(__file__).resolve().parent.parent.parent

if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import pandas as pd
import streamlit as st
from verify.backend.utils.cache import normalize_eval_prompt


_BATCH_CONFIG = VERIFY_ROOT / "batch_config.csv"
_OUTPUTS_DIR  = VERIFY_ROOT / "outputs"
_DIR_SUMMARY  = "dir_summary.json"   # written by patch_ext_ui.py


def _delete_output_dirs(dirs: List[str]) -> Tuple[int, List[str]]:
    """Delete output directories for one progress group."""
    deleted = 0
    errors: List[str] = []
    for raw in dirs:
        path = Path(raw)
        try:
            if not path.exists() or not path.is_dir():
                continue
            if path.parent != _OUTPUTS_DIR:
                errors.append(f"Refused outside outputs: {path}")
                continue
            shutil.rmtree(path)
            deleted += 1
        except Exception as e:
            errors.append(f"{path.name}: {e}")
    return deleted, errors


def _group_label(c: Dict) -> str:
    mode_label = "IOC" if c["method"] == "ioc_comparison" else (c["method"] or "unknown")
    prompt_label = c.get("eval_prompt") or "-"
    in_mod = c.get("input_modality", "") or c.get("modality", "?")
    out_mod = c.get("output_modality", "") or c.get("modality", "?")
    mod_display = f"{in_mod}->{out_mod}" if in_mod != out_mod else in_mod
    return (
        f"{c['app_name']} / {c['dataset_name']} / {mod_display} "
        f"[{mode_label}] / {prompt_label}"
    )


def _render_delete_group(c: Dict, key_prefix: str) -> None:
    dirs = c.get("dirs", [])
    if not dirs:
        st.caption("No output directories recorded for this group.")
        return

    confirm_key = f"{key_prefix}_confirm"
    confirmed = st.checkbox("Confirm delete", key=confirm_key)
    if st.button("Delete output group", key=f"{key_prefix}_delete", type="secondary", disabled=not confirmed):
        deleted, errors = _delete_output_dirs(dirs)
        if errors:
            st.error("; ".join(errors))
        if deleted:
            st.success(f"Deleted {deleted} output director{'y' if deleted == 1 else 'ies'}.")
            st.cache_data.clear()
            st.rerun()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    def _clean(value: object) -> str:
        return str(value).strip() if value is not None else ""

    def _normalize_legacy_row(row: Dict[str, str]) -> Dict[str, str]:
        """
        Support older 6-column batch_config rows that predate generation_task:
          enabled,app_name,modality,dataset_name,perturbation_method,max_items

        When parsed against the current 7-column header, those rows land as:
          generation_task=<dataset>, dataset_name=<method>, perturbation_method=<max_items>, max_items=""
        """
        generation_task = row.get("generation_task", "")
        if generation_task in ("", "text", "image"):
            return row

        return {
            **row,
            "generation_task": "",
            "dataset_name": generation_task,
            "perturbation_method": row.get("dataset_name", ""),
            "max_items": row.get("perturbation_method", ""),
        }

    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#"))
        )
        for row in reader:
            cleaned = {_clean(k): _clean(v) for k, v in row.items() if k is not None}
            if not any(cleaned.values()):
                continue
            rows.append(_normalize_legacy_row(cleaned))
    return rows


@st.cache_data(ttl=15)
def _scan_outputs() -> List[Dict]:
    """
    Scan ALL verify/outputs/ directories (both cache_* and timestamped run dirs).

    Returns a list of dicts:
      app_name, dataset_name, input_modality, output_modality, method, eval_prompt,
      success_count, failed_count, stale_count,
      prompt_success_counts, has_summary, dirs (list of directory paths for this key)

    Counts are aggregated across all directories for the same
    (app_name, dataset_name, input_modality, output_modality, perturbation_method, eval_prompt) key by
    deduplicating item filenames. This avoids dropping coverage when a run was
    split across multiple output directories.
    """
    if not _OUTPUTS_DIR.exists():
        return []

    def _merge_item_state(
        existing: Optional[Tuple[str, bool]],
        incoming_status: str,
        incoming_stale: bool,
    ) -> Tuple[str, bool]:
        if existing is None:
            return incoming_status, incoming_stale

        status_rank = {"success": 3, "failed": 2, "error": 1}
        existing_status, existing_stale = existing
        if status_rank.get(incoming_status, 0) > status_rank.get(existing_status, 0):
            return incoming_status, incoming_stale
        if incoming_status == existing_status == "success":
            return "success", existing_stale or incoming_stale
        return existing_status, existing_stale

    # Per-key state. key = (app, dataset, input_modality, output_modality, method, eval_prompt)
    key_input_modality: Dict[tuple, str] = {}
    key_output_modality: Dict[tuple, str] = {}
    key_prompt: Dict[tuple, str] = {}
    key_dirs: Dict[tuple, List[str]] = defaultdict(list)
    key_items: Dict[tuple, Dict[str, Tuple[str, bool]]] = defaultdict(dict)
    key_summary_fallback: Dict[tuple, Dict[str, int]] = defaultdict(
        lambda: {"success": 0, "failed": 0, "stale": 0, "total": 0}
    )

    for d in _OUTPUTS_DIR.iterdir():
        if not d.is_dir():
            continue
        config_path = d / "run_config.json"
        if not config_path.exists():
            continue
        try:
            cfg      = json.loads(config_path.read_text())
            app      = cfg.get("app_name", "").strip()
            dataset  = cfg.get("dataset_name", "").strip()
            # Support both new (input/output_modality) and legacy (modality/generation_task) formats
            input_modality = cfg.get("input_modality", "").strip() or cfg.get("modality", "").strip()
            output_modality = cfg.get("output_modality", "").strip() or cfg.get("generation_task", "").strip() or input_modality
            method   = cfg.get("perturbation_method", "").strip()
            eval_prompt = str(cfg.get("eval_prompt") or "").strip()
            if method == "ioc_comparison":
                eval_prompt = normalize_eval_prompt(eval_prompt)
            if not app or not dataset:
                continue
            key = (app, dataset, input_modality, output_modality, method, eval_prompt)
            key_input_modality[key] = input_modality
            key_output_modality[key] = output_modality
            key_prompt[key] = eval_prompt
            key_dirs[key].append(str(d))

            summary_path = d / _DIR_SUMMARY
            summary = None
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except Exception:
                    summary = None

            saw_item_json = False
            for f in d.iterdir():
                if f.suffix != ".json" or f.name in ("run_config.json", _DIR_SUMMARY):
                    continue
                saw_item_json = True
                try:
                    data = json.loads(f.read_text())
                    if method == "ioc_comparison":
                        item_prompt = normalize_eval_prompt(data.get("eval_prompt"))
                        if item_prompt != eval_prompt:
                            continue
                    item_name = str(data.get("filename") or f.stem)
                    merged = _merge_item_state(
                        key_items[key].get(item_name),
                        str(data.get("status", "")),
                        bool(data.get("ext_eval_stale", False)),
                    )
                    key_items[key][item_name] = merged
                except Exception:
                    item_name = f.stem
                    key_items[key][item_name] = _merge_item_state(
                        key_items[key].get(item_name),
                        "error",
                        False,
                    )

            if not saw_item_json and summary is not None:
                fb = key_summary_fallback[key]
                fb["success"] += int(summary.get("success", 0) or 0)
                fb["failed"] += int(summary.get("failed", 0) or 0)
                fb["stale"] += int(summary.get("stale", 0) or 0)
                fb["total"] += int(summary.get("total", 0) or 0)

        except Exception:
            pass

    # Build result list
    all_keys = set(key_dirs.keys()) | set(key_items.keys()) | set(key_summary_fallback.keys())
    result: List[Dict] = []

    for key in all_keys:
        app, dataset, input_modality, output_modality, method, eval_prompt = key
        input_modality = key_input_modality.get(key, "")
        output_modality = key_output_modality.get(key, "")
        eval_prompt = key_prompt.get(key, eval_prompt)
        dirs = key_dirs.get(key, [])
        items = key_items.get(key, {})

        if items:
            success_count = sum(1 for st, _ in items.values() if st == "success")
            failed_count = sum(1 for st, _ in items.values() if st == "failed")
            stale_count = sum(1 for st, stl in items.values() if st == "success" and stl)
            has_summary = False
        else:
            fb = key_summary_fallback.get(key, {})
            success_count = int(fb.get("success", 0) or 0)
            failed_count = int(fb.get("failed", 0) or 0)
            stale_count = int(fb.get("stale", 0) or 0)
            has_summary = True

        result.append({
            "app_name": app,
            "dataset_name": dataset,
            "modality": input_modality,  # Legacy
            "input_modality": input_modality,
            "output_modality": output_modality,
            "method": method,
            "eval_prompt": eval_prompt,
            "success_count": success_count,
            "failed_count": failed_count,
            "stale_count": stale_count,
            "prompt_success_counts": {eval_prompt: success_count} if eval_prompt else {},
            "has_summary": has_summary,
            "dirs": dirs,
        })

    return result


@st.cache_data(ttl=60)
def _dataset_size(dataset_name: str, modality: str) -> int:
    try:
        from verify.backend.datasets.loader import count_dataset_items
        return count_dataset_items(dataset_name, modality)
    except Exception:
        return 0


# ── Table builder ─────────────────────────────────────────────────────────────

def _build_table(
    batch_rows: List[Dict[str, str]],
    caches: List[Dict],
    mode: str,          # "ioc" or "perturb"
) -> pd.DataFrame:
    """
    Return an (app, input_modality, output_modality) × dataset DataFrame with cell strings:
      "M / N"              — M successful items out of N total
      "M / N (S stale)"    — S items have outdated ext_eval scores
      "—"                  — combo not in batch config
    Failed items are excluded from M.
    
    Rows are indexed by (app, input_modality, output_modality) to separate different
    modality combinations (e.g., image->text vs text->text for the same app).
    """
    # (app, input_modality, output_modality, dataset) tuples from batch config
    combo_keys = {
        (r["app_name"], r.get("input_modality", "") or r.get("modality", ""), 
         r.get("output_modality", "") or r.get("modality", ""), r["dataset_name"])
        for r in batch_rows
    }
    
    # Unique row identifiers (app, input_modality, output_modality)
    row_ids = sorted({(r["app_name"], 
                       r.get("input_modality", "") or r.get("modality", ""),
                       r.get("output_modality", "") or r.get("modality", ""))
                     for r in batch_rows})
    
    # Unique datasets
    datasets = sorted({r["dataset_name"] for r in batch_rows})

    # (app, input_modality, output_modality, dataset) → (success, stale, prompt_counts)
    cache_lookup: Dict[Tuple[str, str, str, str], Tuple[int, int, Dict[str, int]]] = {}
    for c in caches:
        app = c["app_name"]
        dataset = c["dataset_name"]
        in_mod = c.get("input_modality", "") or c.get("modality", "")
        out_mod = c.get("output_modality", "") or c.get("modality", "")
        is_ioc = c["method"] == "ioc_comparison"
        if (mode == "ioc" and is_ioc) or (mode == "perturb" and not is_ioc):
            key = (app, in_mod, out_mod, dataset)
            success = c.get("success_count", 0)
            stale   = c.get("stale_count",   0)
            prompt_counts = dict(c.get("prompt_success_counts", {}) or {})
            existing = cache_lookup.get(key, (-1, 0, {}))
            if success > existing[0]:
                cache_lookup[key] = (success, stale, prompt_counts)
            elif success == existing[0]:
                merged_counts = dict(existing[2])
                for prompt, count in prompt_counts.items():
                    merged_counts[prompt] = merged_counts.get(prompt, 0) + count
                cache_lookup[key] = (success, max(stale, existing[1]), merged_counts)

    # Build DataFrame with MultiIndex for rows
    data: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row_id in row_ids:
        app, in_mod, out_mod = row_id
        row: Dict[str, str] = {"input_modality": in_mod, "output_modality": out_mod}
        for dataset in datasets:
            combo_key = (app, in_mod, out_mod, dataset)
            if combo_key not in combo_keys:
                row[dataset] = "—"
            else:
                # Use input modality for dataset size lookup
                n = _dataset_size(dataset, in_mod)
                m, stale, prompt_counts = cache_lookup.get(combo_key, (0, 0, {}))
                cell = f"{m} / {n}" if n > 0 else f"{m} / ?"
                if stale > 0:
                    cell += f" ({stale} stale)"
                version_parts = [
                    f"{prompt}: {count}"
                    for prompt, count in sorted(prompt_counts.items())
                    if prompt and count > 0
                ]
                if version_parts:
                    cell += " [" + ", ".join(version_parts) + "]"
                row[dataset] = cell
        data[row_id] = row

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.names = ["app", "input_modality", "output_modality"]
    return df


def _style_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return same-shape DataFrame of CSS strings for st.dataframe styling."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for idx in df.index:
        for dataset in df.columns:
            # Skip modality columns
            if dataset in ("input_modality", "output_modality"):
                continue
            val = str(df.loc[idx, dataset])
            if val == "—":
                styles.loc[idx, dataset] = (
                    "background-color:#f5f5f5; color:#c0c0c0"
                )
            elif "/" in val:
                # Strip annotations: "340 / 347 (12 stale) [prompt1: 340]" → "340 / 347"
                base = val.split("(")[0].split("[")[0].strip()
                parts = base.split("/")
                stale = "stale" in val
                try:
                    m     = int(parts[0].strip())
                    n_raw = parts[1].strip()
                    n     = int(n_raw) if n_raw != "?" else 0
                    if n > 0 and m >= n:
                        # Complete (possibly stale)
                        styles.loc[idx, dataset] = (
                            "background-color:#c3e6cb; color:#155724; font-weight:600"
                            if not stale else
                            "background-color:#d4edda; color:#155724; font-weight:600; "
                            "border-bottom:2px dashed #856404"
                        )
                    elif m > 0:
                        # Partial (possibly stale)
                        styles.loc[idx, dataset] = (
                            "background-color:#fff3cd; color:#856404"
                        )
                    else:
                        # Not started
                        styles.loc[idx, dataset] = (
                            "background-color:#f8d7da; color:#721c24"
                        )
                except (ValueError, IndexError):
                    pass
    return styles


def _parse_m_n(val: str) -> Tuple[int, int]:
    """Parse 'M / N' with optional annotations → (M, N).  Returns (-1, 0) on failure."""
    try:
        base  = val.split("(")[0].split("[")[0].strip()
        parts = base.split("/")
        m     = int(parts[0].strip())
        n_raw = parts[1].strip()
        n     = int(n_raw) if n_raw != "?" else 0
        return m, n
    except (ValueError, IndexError, AttributeError):
        return -1, 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📋 Experiment Progress")
    st.markdown(
        "Coverage of **successfully cached** results vs. full dataset size "
        "for every (app, dataset) pair in `batch_config.csv`.  "
        "Failed items are excluded from M.  "
        "Run `verify/patch_ext_ui.py` to detect and fix stale UI events."
    )

    batch_rows = _load_csv_rows(_BATCH_CONFIG)
    if not batch_rows:
        st.error("Could not load `batch_config.csv`. Expected at `verify/batch_config.csv`.")
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Options")
        if st.button("🔄 Refresh", width="stretch"):
            st.cache_data.clear()
            st.rerun()

        st.caption(
            "**Legend**\n\n"
            "🟩 M = N — complete\n\n"
            "🟨 0 < M < N — partial\n\n"
            "🟥 M = 0 — not started\n\n"
            "⚠️ (S stale) — ext_eval scored on truncated text;\n"
            "run `patch_ext_ui.py` to fix\n\n"
            "░ — not in batch config\n\n"
            "**M** = successful items only (failed excluded)"
        )

    caches = _scan_outputs()
    batch_combo_keys = {
        (r["app_name"], 
         r.get("input_modality", "") or r.get("modality", ""),
         r.get("output_modality", "") or r.get("modality", ""),
         r["dataset_name"])
        for r in batch_rows
    }
    unmapped_caches = [
        c for c in caches
        if (c["app_name"], 
            c.get("input_modality", ""), 
            c.get("output_modality", ""), 
            c["dataset_name"]) not in batch_combo_keys
    ]

    # ── Stats row ─────────────────────────────────────────────────────────────
    enabled_rows = [
        r for r in batch_rows
        if r.get("enabled", "true").lower() not in ("false", "0", "no")
    ]
    n_apps     = len({r["app_name"]    for r in enabled_rows})
    n_datasets = len({r["dataset_name"] for r in enabled_rows})
    n_combos   = len(enabled_rows)
    total_stale  = sum(c.get("stale_count",  0) for c in caches)
    total_failed = sum(c.get("failed_count", 0) for c in caches)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Apps",        n_apps)
    c2.metric("Datasets",    n_datasets)
    c3.metric("Combos",      n_combos)
    c4.metric("Run groups",  len(caches))
    c5.metric("Stale items", total_stale,
              delta=f"{total_failed} failed" if total_failed else None,
              delta_color="inverse")

    st.divider()

    # ── Tabs: IOC / Perturb ───────────────────────────────────────────────────
    tab_ioc, tab_perturb = st.tabs(["IOC", "Perturb"])

    for tab, mode in ((tab_ioc, "ioc"), (tab_perturb, "perturb")):
        with tab:
            df = _build_table(batch_rows, caches, mode)

            if df.empty:
                st.info("No data to show.")
                continue

            # Compute aggregate progress (success items / dataset total)
            total_m = total_n = 0
            for val in df.values.flatten():
                m, n = _parse_m_n(str(val))
                if m >= 0 and n > 0:
                    total_m += m
                    total_n += n

            if total_n > 0:
                st.caption(
                    f"**Overall: {total_m} / {total_n} items successfully cached "
                    f"({total_m / total_n:.0%})**"
                )

            styled = df.style.apply(_style_table, axis=None)
            st.dataframe(styled, width="stretch", height=40 * (len(df) + 1) + 36)

    # ── Delete output groups ──────────────────────────────────────────────────
    if caches:
        st.divider()
        with st.expander("Delete Output Groups", expanded=False):
            st.caption(
                "Deletes cached output directories from `verify/outputs/`. "
                "This does not edit `verify/batch_config.csv`; it only removes recorded results."
            )
            sorted_caches = sorted(
                caches,
                key=lambda x: (
                    x["app_name"],
                    x["dataset_name"],
                    x.get("input_modality", "") or x.get("modality", ""),
                    x.get("output_modality", "") or x.get("modality", ""),
                    x["method"],
                    x.get("eval_prompt", ""),
                ),
            )
            for idx, c in enumerate(sorted_caches):
                with st.container(border=True):
                    st.markdown(f"**{_group_label(c)}**")
                    st.caption(
                        f"{c.get('success_count', 0)} success, "
                        f"{c.get('failed_count', 0)} failed, "
                        f"{len(c.get('dirs', []))} director{'y' if len(c.get('dirs', [])) == 1 else 'ies'}"
                    )
                    for d in c.get("dirs", []):
                        st.code(d, language=None)
                    _render_delete_group(c, f"progress_del_{idx}")

    # ── Fully Failed Runs ─────────────────────────────────────────────────────
    fully_failed = [
        c for c in caches
        if c.get("success_count", 0) == 0 and c.get("failed_count", 0) > 0
    ]
    if fully_failed:
        st.divider()
        with st.expander(
            f"⛔ Fully Failed Runs ({len(fully_failed)}) — all items failed, 0 successes",
            expanded=False,
        ):
            st.caption(
                "These output directories have no successful items. "
                "Check runner logs or re-run with `--no-cache`."
            )
            for c in sorted(fully_failed, key=lambda x: (x["app_name"], x["dataset_name"])):
                st.markdown(
                    f"**{_group_label(c)}** &nbsp; "
                    f"— {c.get('failed_count', '?')} failed items",
                    unsafe_allow_html=True,
                )
                for d in c.get("dirs", []):
                    st.code(d, language=None)

    if unmapped_caches:
        st.divider()
        with st.expander(
            f"🧭 Unmapped Output Groups ({len(unmapped_caches)}) — on disk but not in batch_config.csv",
            expanded=False,
        ):
            st.caption(
                "These output groups were found under `verify/outputs/`, but their "
                "(app, dataset, modality) combination is not present in `verify/batch_config.csv`, "
                "so they are not represented in the matrix."
            )
            for c in sorted(unmapped_caches, key=lambda x: (x["app_name"], x["dataset_name"], 
                                                              x.get("input_modality", "") or x.get("modality", ""),
                                                              x.get("output_modality", "") or x.get("modality", ""), 
                                                              x["method"])):
                st.markdown(
                    f"**{_group_label(c)}** &nbsp; "
                    f"— {c.get('success_count', 0)} success, {c.get('failed_count', 0)} failed",
                    unsafe_allow_html=True,
                )
                for d in c.get("dirs", []):
                    st.code(d, language=None)

    # ── Auto-refresh while batch is running ───────────────────────────────────
    bs = st.session_state.get("batch_state", {})
    if bs.get("running"):
        time.sleep(10)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
