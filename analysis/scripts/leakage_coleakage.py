"""
Confirmed Co-leakage Correspondence Heatmap
21×21 attribute-by-attribute matrix: cell (A, B) = number of (item, config)
pairs where BOTH attribute A AND attribute B were confirmed leaked in ext_eval
aggregate. Diagonal = per-attribute confirmed count.

Usage: python analysis/scripts/leakage_coleakage.py
Outputs:
  analysis/attachments/leakage_coleakage_1x1.png  (7×7 in, 150 dpi)
  analysis/attachments/leakage_coleakage_2x1.png  (3.5×3.5 in, 200 dpi)
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
LANTERN_ROOT = SCRIPT_DIR.parent.parent
ATTACH_DIR   = LANTERN_ROOT / "analysis" / "attachments"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from _leakage_common import (
    load_data, ALL_ATTRS, ATTR_FAMILIES, ATTR_TO_FAMILY, FAMILY_COLORS,
    P_TEAL, P_YELLOW, P_ORANGE, P_GREEN, P_BLUE, P_MAUVE, P_SLATE, S_OUTPUT,
)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams["axes.grid"] = False

# Family colors for tick labels (Activity uses warm orange = S_OUTPUT from FIGURE.md)
FAM_TICK_COLORS = {
    "Identity & Identifiability": P_BLUE,
    "Demographic":                 P_ORANGE,
    "Health & Medical":            P_MAUVE,
    "Location & Spatial":          P_TEAL,
    "Religion & Cultural":         P_YELLOW,
    "Appearance & Body":           P_GREEN,
    "Attire, Role & Group":        P_SLATE,
    "Activity & Lifestyle":        S_OUTPUT,
}

# Canonical attribute order: family order, then alpha within family
FAMILY_ORDER = list(ATTR_FAMILIES.keys())
ORDERED_ATTRS: list[str] = []
for fam in FAMILY_ORDER:
    ORDERED_ATTRS.extend(sorted(ATTR_FAMILIES[fam]))
N_ATTRS = len(ORDERED_ATTRS)
ATTR_IDX = {a: i for i, a in enumerate(ORDERED_ATTRS)}


# ── Load data ────────────────────────────────────────────────────────────────
def load_corpus() -> pd.DataFrame:
    """Load with min_items=1, then keep configs with >=100 items OR app==tool-neuron."""
    df, _ = load_data(min_items=1)

    # Identify which (dir_name) configs to keep
    config_sizes = df.groupby("dir_name")["full_key"].nunique()
    keep_dirs = set()
    for dname, n_items in config_sizes.items():
        app_rows = df[df["dir_name"] == dname]["app"]
        app = app_rows.iloc[0] if len(app_rows) > 0 else ""
        if n_items >= 100 or app == "tool-neuron":
            keep_dirs.add(dname)

    filtered = df[df["dir_name"].isin(keep_dirs)].copy()
    return filtered


# ── Build co-leakage matrix ──────────────────────────────────────────────────
def build_co_matrix(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """
    For each unique (full_key) item, find the set of attrs with confirmed
    leakage in ext_eval aggregate. Increment co_matrix[A][B] for each pair
    (A, B) both confirmed in that item (including A==B for diagonal).

    Returns co_matrix (N_ATTRS x N_ATTRS) and total item count.
    """
    co = np.zeros((N_ATTRS, N_ATTRS), dtype=int)

    # Get per-item confirmed attrs
    conf_df = df[df["ext_conf"] == 1][["full_key", "attr"]].drop_duplicates()
    item_attrs: dict[str, set[str]] = defaultdict(set)
    for _, row in conf_df.iterrows():
        a = row["attr"]
        if a in ATTR_IDX:
            item_attrs[row["full_key"]].add(a)

    for fk, attrs in item_attrs.items():
        attr_list = [a for a in attrs if a in ATTR_IDX]
        for i, a in enumerate(attr_list):
            ia = ATTR_IDX[a]
            # diagonal: count this item confirmed for A
            co[ia][ia] += 1
            for b in attr_list[i + 1:]:
                ib = ATTR_IDX[b]
                co[ia][ib] += 1
                co[ib][ia] += 1

    total_items = df["full_key"].nunique()
    return co, total_items


# ── Family boundary positions ─────────────────────────────────────────────────
def family_boundaries() -> list[int]:
    """Return the x/y positions between family groups (0-indexed, as separators)."""
    boundaries = []
    pos = 0
    for fam in FAMILY_ORDER[:-1]:  # no boundary after last family
        pos += len(ATTR_FAMILIES[fam])
        boundaries.append(pos)
    return boundaries


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_heatmap(co: np.ndarray, total_items: int, figsize: tuple[float, float],
                 out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # Colormap: high=black, low=white ("Greys" reversed = black for high)
    cmap = plt.get_cmap("Greys")

    # Normalize for coloring: 0 -> white (0.0), max -> black (1.0)
    vmax = co.max()
    vmin = 0

    im = ax.imshow(co, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")

    # ── Cell annotations ──
    threshold = 0.6 * vmax  # use white text above this
    font_size = max(5, min(7, figsize[0] * 0.8))
    for i in range(N_ATTRS):
        for j in range(N_ATTRS):
            val = co[i, j]
            if val > 0:
                # text color: white if cell is dark, else black
                text_color = "white" if val >= threshold else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=font_size, color=text_color, fontweight="normal")

    # ── Grid lines between cells ──
    for x in range(N_ATTRS + 1):
        ax.axvline(x - 0.5, color="lightgray", linewidth=0.3, zorder=3)
        ax.axhline(x - 0.5, color="lightgray", linewidth=0.3, zorder=3)

    # ── Family separator lines ──
    boundaries = family_boundaries()
    for b in boundaries:
        ax.axvline(b - 0.5, color="black", linewidth=0.8, zorder=4)
        ax.axhline(b - 0.5, color="black", linewidth=0.8, zorder=4)

    # ── Axis ticks and labels ──
    ax.set_xticks(range(N_ATTRS))
    ax.set_yticks(range(N_ATTRS))
    tick_fontsize = max(5, min(7, figsize[0] * 0.75))
    ax.set_xticklabels(ORDERED_ATTRS, rotation=45, ha="right",
                       fontsize=tick_fontsize)
    ax.set_yticklabels(ORDERED_ATTRS, fontsize=tick_fontsize)

    # Color tick labels by family
    for tick, attr in zip(ax.get_xticklabels(), ORDERED_ATTRS):
        fam = ATTR_TO_FAMILY.get(attr, "")
        color = FAM_TICK_COLORS.get(fam, "black")
        tick.set_color(color)

    for tick, attr in zip(ax.get_yticklabels(), ORDERED_ATTRS):
        fam = ATTR_TO_FAMILY.get(attr, "")
        color = FAM_TICK_COLORS.get(fam, "black")
        tick.set_color(color)

    # ── No title per FIGURE.md: "No plot titles" ──
    # Title goes in LaTeX caption instead.

    # ── Spines ──
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_summary(co: np.ndarray, total_items: int) -> None:
    print(f"\n=== Co-leakage Summary (total corpus items: {total_items:,}) ===")

    # Per-attribute confirmed count (diagonal)
    print("\nPer-attribute confirmed count (diagonal):")
    diag_vals = [(ORDERED_ATTRS[i], co[i, i]) for i in range(N_ATTRS)]
    diag_vals.sort(key=lambda x: -x[1])
    for attr, cnt in diag_vals:
        print(f"  {attr:<20s}: {cnt:>5d}")

    # Top 10 co-leaking pairs (off-diagonal, upper triangle)
    pairs = []
    for i in range(N_ATTRS):
        for j in range(i + 1, N_ATTRS):
            if co[i, j] > 0:
                pairs.append((ORDERED_ATTRS[i], ORDERED_ATTRS[j], co[i, j]))
    pairs.sort(key=lambda x: -x[2])
    print("\nTop 10 co-leaking attribute pairs:")
    for a, b, cnt in pairs[:10]:
        print(f"  ({a}, {b}): {cnt}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading corpus...")
    df = load_corpus()
    print(f"  Corpus: {df['full_key'].nunique():,} items, "
          f"{df['dir_name'].nunique()} configs, "
          f"{df['app'].nunique()} apps")

    print("Building co-leakage matrix...")
    co, total_items = build_co_matrix(df)
    print(f"  Matrix max = {co.max()}, diagonal max = {co.diagonal().max()}")

    print_summary(co, total_items)

    print("\nPlotting...")
    # 1x1: 7x7 inches, 150 dpi
    plot_heatmap(co, total_items,
                 figsize=(7.0, 7.0),
                 out_path=ATTACH_DIR / "leakage_coleakage_1x1.png",
                 dpi=150)

    # 2x1 per task spec: 3.5x3.5 inches (column-width), 200 dpi
    plot_heatmap(co, total_items,
                 figsize=(3.5, 3.5),
                 out_path=ATTACH_DIR / "leakage_coleakage_2x1.png",
                 dpi=200)

    print("Done.")


if __name__ == "__main__":
    main()
