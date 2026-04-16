"""
Experiment Progress — app × dataset coverage matrix.

Shows how many items have been cached (M) out of the full dataset size (N)
for every (app, dataset) combination listed in batch_config.csv.
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
    Deduplicates item filenames across multiple runs for the same
    (app_name, dataset_name, method) so counts are not inflated.

    Returns list of {app_name, dataset_name, modality, method, cached_count}.
    """
    if not _OUTPUTS_DIR.exists():
        return []

    # (app, dataset, method) → set of item filenames (deduplicated across runs)
    item_sets: Dict[tuple, set] = defaultdict(set)
    # (app, dataset, method) → modality (last seen wins; should be consistent)
    key_modality: Dict[tuple, str] = {}

    for d in _OUTPUTS_DIR.iterdir():
        if not d.is_dir():
            continue
        config_path = d / "run_config.json"
        if not config_path.exists():
            continue
        try:
            cfg = json.loads(config_path.read_text())
            app     = cfg.get("app_name", "").strip()
            dataset = cfg.get("dataset_name", "").strip()
            modality = cfg.get("modality", "").strip()
            method  = cfg.get("perturbation_method", "").strip()
            if not app or not dataset:
                continue
            key = (app, dataset, method)
            key_modality[key] = modality
            for f in d.iterdir():
                if f.suffix == ".json" and f.name != "run_config.json":
                    item_sets[key].add(f.name)
        except Exception:
            pass

    return [
        {
            "app_name":     app,
            "dataset_name": dataset,
            "modality":     key_modality.get((app, dataset, method), ""),
            "method":       method,
            "cached_count": len(items),
        }
        for (app, dataset, method), items in item_sets.items()
    ]


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
    """Return an app × dataset DataFrame with "M / N" or "—" cell strings."""

    apps     = sorted({r["app_name"]    for r in batch_rows})
    datasets = sorted({r["dataset_name"] for r in batch_rows})

    # (app, dataset) → modality from batch config
    combo_modality: Dict[Tuple[str, str], str] = {
        (r["app_name"], r["dataset_name"]): r["modality"]
        for r in batch_rows
    }

    # (app, dataset) → best (most complete) cached count
    cache_lookup: Dict[Tuple[str, str], int] = {}
    for c in caches:
        key = (c["app_name"], c["dataset_name"])
        is_ioc = c["method"] == "ioc_comparison"
        if (mode == "ioc" and is_ioc) or (mode == "perturb" and not is_ioc):
            cache_lookup[key] = max(cache_lookup.get(key, 0), c["cached_count"])

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
                m = cache_lookup.get(combo, 0)
                row[dataset] = f"{m} / {n}" if n > 0 else f"{m} / ?"
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
                parts = val.split("/")
                try:
                    m = int(parts[0].strip())
                    n_raw = parts[1].strip()
                    n = int(n_raw) if n_raw != "?" else 0
                    if n > 0 and m >= n:
                        # Complete
                        styles.loc[app, dataset] = (
                            "background-color:#d4edda; color:#155724; font-weight:600"
                        )
                    elif m > 0:
                        # Partial
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📋 Experiment Progress")
    st.markdown(
        "Coverage of cached results vs. full dataset size "
        "for every (app, dataset) pair in `batch_config.csv`."
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
            "░ — not in batch config"
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Apps",     n_apps)
    c2.metric("Datasets", n_datasets)
    c3.metric("Combos",   n_combos)
    c4.metric("Run groups", len(caches))

    st.divider()

    # ── Tabs: IOC / Perturb ───────────────────────────────────────────────────
    tab_ioc, tab_perturb = st.tabs(["IOC", "Perturb"])

    for tab, mode in ((tab_ioc, "ioc"), (tab_perturb, "perturb")):
        with tab:
            df = _build_table(batch_rows, caches, mode)

            if df.empty:
                st.info("No data to show.")
                continue

            # Compute aggregate progress for the caption
            total_m = total_n = 0
            for val in df.values.flatten():
                if "/" in str(val) and val != "—":
                    parts = str(val).split("/")
                    try:
                        total_m += int(parts[0].strip())
                        n_raw = parts[1].strip()
                        if n_raw != "?":
                            total_n += int(n_raw)
                    except (ValueError, IndexError):
                        pass

            if total_n > 0:
                st.caption(
                    f"**Overall: {total_m} / {total_n} items cached "
                    f"({total_m / total_n:.0%})**"
                )

            styled = df.style.apply(_style_table, axis=None)
            st.dataframe(styled, use_container_width=True, height=40 * (len(df) + 1) + 36)

    # ── Auto-refresh while batch is running ───────────────────────────────────
    bs = st.session_state.get("batch_state", {})
    if bs.get("running"):
        time.sleep(10)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
