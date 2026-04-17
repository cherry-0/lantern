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

st.set_page_config(
    page_title="Experiment Progress — Verify",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

_BATCH_CONFIG = VERIFY_ROOT / "batch_config.csv"
_OUTPUTS_DIR  = VERIFY_ROOT / "outputs"
_DIR_SUMMARY  = "dir_summary.json"   # written by patch_ext_ui.py


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#"))
        )
        for row in reader:
            if not any(v.strip() for v in row.values()):
                continue
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


@st.cache_data(ttl=15)
def _scan_outputs() -> List[Dict]:
    """
    Scan ALL verify/outputs/ directories (both cache_* and timestamped run dirs).

    Fast path: reads dir_summary.json written by patch_ext_ui.py — O(1) per dir.
    Slow path: reads every item JSON to extract status / ext_eval_stale — used
               for dirs that have not been processed by patch_ext_ui.py yet.

    Returns a list of dicts:
      app_name, dataset_name, modality, method,
      success_count, failed_count, stale_count,
      has_summary, dirs (list of directory paths for this key)
    """
    if not _OUTPUTS_DIR.exists():
        return []

    # Per-key state.  key = (app, dataset, method)
    key_modality: Dict[tuple, str]        = {}
    key_dirs:     Dict[tuple, List[str]]  = defaultdict(list)
    # Best summary seen for this key (highest success_count wins)
    key_summary:  Dict[tuple, Dict]       = {}
    # Fallback: per-filename (status, stale) dict — deduplicated across dirs
    key_items:    Dict[tuple, Dict[str, Tuple[str, bool]]] = defaultdict(dict)
    # Keys that have at least one dir without a summary (need slow-path merge)
    needs_slow: set = set()

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
            modality = cfg.get("modality", "").strip()
            method   = cfg.get("perturbation_method", "").strip()
            if not app or not dataset:
                continue
            key = (app, dataset, method)
            key_modality[key] = modality
            key_dirs[key].append(str(d))

            # Fast path — dir_summary.json exists
            summary_path = d / _DIR_SUMMARY
            if summary_path.exists():
                try:
                    s = json.loads(summary_path.read_text())
                    existing = key_summary.get(key, {})
                    if s.get("success", 0) >= existing.get("success", 0):
                        key_summary[key] = s
                    continue   # skip item enumeration for this dir
                except Exception:
                    pass  # fall through to slow path

            # Slow path — read individual item files
            needs_slow.add(key)
            for f in d.iterdir():
                if f.suffix != ".json" or f.name in ("run_config.json", _DIR_SUMMARY):
                    continue
                try:
                    data = json.loads(f.read_text())
                    key_items[key][f.name] = (
                        data.get("status", ""),
                        bool(data.get("ext_eval_stale", False)),
                    )
                except Exception:
                    key_items[key][f.name] = ("error", False)

        except Exception:
            pass

    # Build result list
    all_keys = set(key_summary.keys()) | set(key_items.keys())
    result: List[Dict] = []

    for key in all_keys:
        app, dataset, method = key
        modality = key_modality.get(key, "")
        dirs     = key_dirs.get(key, [])

        if key in key_summary and key not in needs_slow:
            # All dirs for this key had summaries — use the best one
            s = key_summary[key]
            result.append({
                "app_name":      app,
                "dataset_name":  dataset,
                "modality":      modality,
                "method":        method,
                "success_count": s.get("success", 0),
                "failed_count":  s.get("failed",  0),
                "stale_count":   s.get("stale",   0),
                "has_summary":   True,
                "dirs":          dirs,
            })
        else:
            # At least one dir lacked a summary; use slow-path item data
            items = key_items.get(key, {})
            # If we also have a summary for some dirs, take whichever gives more successes
            slow_success = sum(1 for st, _ in items.values() if st == "success")
            slow_failed  = sum(1 for st, _ in items.values() if st == "failed")
            slow_stale   = sum(1 for st, stl in items.values() if st == "success" and stl)

            if key in key_summary and key_summary[key].get("success", 0) > slow_success:
                s = key_summary[key]
                result.append({
                    "app_name":      app,
                    "dataset_name":  dataset,
                    "modality":      modality,
                    "method":        method,
                    "success_count": s.get("success", 0),
                    "failed_count":  s.get("failed",  0),
                    "stale_count":   s.get("stale",   0),
                    "has_summary":   True,
                    "dirs":          dirs,
                })
            else:
                result.append({
                    "app_name":      app,
                    "dataset_name":  dataset,
                    "modality":      modality,
                    "method":        method,
                    "success_count": slow_success,
                    "failed_count":  slow_failed,
                    "stale_count":   slow_stale,
                    "has_summary":   False,
                    "dirs":          dirs,
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
    Return an app × dataset DataFrame with cell strings of the form:
      "M / N"              — M successful items out of N total
      "M / N (S stale)"    — S items have outdated ext_eval scores
      "—"                  — combo not in batch config
    Failed items are excluded from M.
    """
    apps     = sorted({r["app_name"]    for r in batch_rows})
    datasets = sorted({r["dataset_name"] for r in batch_rows})

    # (app, dataset) → modality from batch config
    combo_modality: Dict[Tuple[str, str], str] = {
        (r["app_name"], r["dataset_name"]): r["modality"]
        for r in batch_rows
    }

    # (app, dataset) → (success_count, stale_count) for the best run
    cache_lookup: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for c in caches:
        key    = (c["app_name"], c["dataset_name"])
        is_ioc = c["method"] == "ioc_comparison"
        if (mode == "ioc" and is_ioc) or (mode == "perturb" and not is_ioc):
            success = c.get("success_count", 0)
            stale   = c.get("stale_count",   0)
            existing = cache_lookup.get(key, (-1, 0))
            if success > existing[0]:
                cache_lookup[key] = (success, stale)

    data: Dict[str, Dict[str, str]] = {}
    for app in apps:
        row: Dict[str, str] = {}
        for dataset in datasets:
            combo = (app, dataset)
            if combo not in combo_modality:
                row[dataset] = "—"
            else:
                modality = combo_modality[combo]
                n = _dataset_size(dataset, modality)
                m, stale = cache_lookup.get(combo, (0, 0))
                cell = f"{m} / {n}" if n > 0 else f"{m} / ?"
                if stale > 0:
                    cell += f" ({stale} stale)"
                row[dataset] = cell
        data[app] = row

    return pd.DataFrame(data, index=datasets).T   # apps as rows, datasets as cols


def _style_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return same-shape DataFrame of CSS strings for st.dataframe styling."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for app in df.index:
        for dataset in df.columns:
            val = str(df.loc[app, dataset])
            if val == "—":
                styles.loc[app, dataset] = (
                    "background-color:#f5f5f5; color:#c0c0c0"
                )
            elif "/" in val:
                # Strip stale annotation: "340 / 347 (12 stale)" → "340 / 347"
                base  = val.split("(")[0].strip()
                parts = base.split("/")
                stale = "(stale)" in val
                try:
                    m     = int(parts[0].strip())
                    n_raw = parts[1].strip()
                    n     = int(n_raw) if n_raw != "?" else 0
                    if n > 0 and m >= n:
                        # Complete (possibly stale)
                        styles.loc[app, dataset] = (
                            "background-color:#c3e6cb; color:#155724; font-weight:600"
                            if not stale else
                            "background-color:#d4edda; color:#155724; font-weight:600; "
                            "border-bottom:2px dashed #856404"
                        )
                    elif m > 0:
                        # Partial (possibly stale)
                        styles.loc[app, dataset] = (
                            "background-color:#fff3cd; color:#856404"
                        )
                    else:
                        # Not started
                        styles.loc[app, dataset] = (
                            "background-color:#f8d7da; color:#721c24"
                        )
                except (ValueError, IndexError):
                    pass
    return styles


def _parse_m_n(val: str) -> Tuple[int, int]:
    """Parse 'M / N' or 'M / N (S stale)' → (M, N).  Returns (-1, 0) on failure."""
    try:
        base  = val.split("(")[0].strip()
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
        if st.button("🔄 Refresh", use_container_width=True):
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
            st.dataframe(styled, use_container_width=True, height=40 * (len(df) + 1) + 36)

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
                mode_label = "IOC" if c["method"] == "ioc_comparison" else (c["method"] or "IOC")
                st.markdown(
                    f"**{c['app_name']}** / `{c['dataset_name']}` &nbsp; "
                    f"`[{mode_label}]` &nbsp; "
                    f"— {c.get('failed_count', '?')} failed items",
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
