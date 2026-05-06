"""
Input-GT → Output Transition Heatmaps  (21×21 and collapsed 8×8)

For each item with ≥1 GT-positive input attribute, counts (input_attr, output_attr)
pairs across:
  - raw output  (output_conf == 1)
  - externalized output (ext_conf == 1)

Compared to the co-leakage script (output-output pairs), these matrices capture
*transformation*: row = what was present in the input, column = what appeared
in the output.

Outputs
-------
analysis/attachments/
  leakage_input_flow_raw_1x1.png   — 21×21 GT→raw-output  (7×7 in, 150 dpi)
  leakage_input_flow_raw_2x1.png   — 21×21 GT→raw-output  (3.5×3.5 in, 200 dpi)
  leakage_input_flow_ext_1x1.png   — 21×21 GT→externalized (7×7 in, 150 dpi)
  leakage_input_flow_ext_2x1.png   — 21×21 GT→externalized (3.5×3.5 in, 200 dpi)
  leakage_family_collapsed_coleakage_1x1.png  — 8×8 co-leakage family matrix
  leakage_family_collapsed_raw_1x1.png        — 8×8 GT→raw family matrix
  leakage_family_collapsed_ext_1x1.png        — 8×8 GT→ext family matrix
  leakage_family_collapsed_altair.html        — Altair 3-panel interactive HTML

paper/26_CCS_Lantern/figures/
  leakage_input_flow_raw_2x1.png
  leakage_input_flow_ext_2x1.png
  leakage_family_collapsed_coleakage_2x1.png
  leakage_family_collapsed_raw_2x1.png
  leakage_family_collapsed_ext_2x1.png

Usage: python analysis/scripts/leakage_input_flow.py
"""

from __future__ import annotations
import sys
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import altair as alt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
LANTERN_ROOT = SCRIPT_DIR.parent.parent
ATTACH_DIR   = LANTERN_ROOT / "analysis" / "attachments"
FIGURES_DIR  = LANTERN_ROOT / "paper" / "26_CCS_Lantern" / "figures"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from _leakage_common import (
    load_data, ALL_ATTRS, ATTR_FAMILIES, ATTR_TO_FAMILY, FAMILY_COLORS,
    P_TEAL, P_YELLOW, P_ORANGE, P_GREEN, P_BLUE, P_MAUVE, P_SLATE, S_OUTPUT,
)

# ── Shared attribute order (same as co-leakage script) ───────────────────────
FAMILY_ORDER = list(ATTR_FAMILIES.keys())
ORDERED_ATTRS: list[str] = []
for _fam in FAMILY_ORDER:
    ORDERED_ATTRS.extend(sorted(ATTR_FAMILIES[_fam]))
N_ATTRS = len(ORDERED_ATTRS)
ATTR_IDX = {a: i for i, a in enumerate(ORDERED_ATTRS)}

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

# For 8×8 family-level labels (short names for heatmap axes)
FAM_SHORT = {
    "Identity & Identifiability": "Identity",
    "Demographic":                 "Demographic",
    "Health & Medical":            "Health",
    "Location & Spatial":          "Location",
    "Religion & Cultural":         "Religion",
    "Appearance & Body":           "Appearance",
    "Attire, Role & Group":        "Attire",
    "Activity & Lifestyle":        "Activity",
}
ORDERED_FAMS = FAMILY_ORDER  # same order as attributes
FAM_SHORT_LIST = [FAM_SHORT[f] for f in ORDERED_FAMS]

# ── Load corpus ───────────────────────────────────────────────────────────────
def load_corpus() -> pd.DataFrame:
    df, _ = load_data(min_items=1)
    config_sizes = df.groupby("dir_name")["full_key"].nunique()
    keep_dirs = set()
    for dname, n_items in config_sizes.items():
        app_rows = df[df["dir_name"] == dname]["app"]
        app = app_rows.iloc[0] if len(app_rows) > 0 else ""
        if n_items >= 100 or app == "tool-neuron":
            keep_dirs.add(dname)
    return df[df["dir_name"].isin(keep_dirs)].copy()


# ── Build input→output transition matrices ───────────────────────────────────
def build_flow_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """
    mat_raw[i][j] = # items where GT attribute i is present AND attribute j is
                    confirmed in raw output (output_conf==1).
    mat_ext[i][j] = same but for externalized confirmed (ext_conf==1).

    Rows = input GT attributes, columns = output attributes.
    Diagonal (i==j): attribute present in input AND found in output (passthrough).
    Off-diagonal (i≠j): attribute i in input but attribute j appeared in output.
    """
    mat_raw = np.zeros((N_ATTRS, N_ATTRS), dtype=int)
    mat_ext = np.zeros((N_ATTRS, N_ATTRS), dtype=int)

    # Group by item (full_key)
    for fk, grp in df.groupby("full_key"):
        # GT-present attributes for this item
        gt_attrs = [
            row["attr"] for _, row in grp.iterrows()
            if row.get("input_label", 0) == 1 and row["attr"] in ATTR_IDX
        ]
        # Raw-output confirmed attributes for this item
        raw_attrs = [
            row["attr"] for _, row in grp.iterrows()
            if row.get("output_conf", 0) == 1 and row["attr"] in ATTR_IDX
        ]
        # Externalized confirmed attributes for this item
        ext_attrs = [
            row["attr"] for _, row in grp.iterrows()
            if row.get("ext_conf", 0) == 1 and row["attr"] in ATTR_IDX
        ]

        # Count all (gt_attr, out_attr) pairs
        for ga in gt_attrs:
            gi = ATTR_IDX[ga]
            for ra in raw_attrs:
                mat_raw[gi][ATTR_IDX[ra]] += 1
            for ea in ext_attrs:
                mat_ext[gi][ATTR_IDX[ea]] += 1

    total_items = df["full_key"].nunique()
    return mat_raw, mat_ext, total_items


def build_co_matrix(df: pd.DataFrame) -> np.ndarray:
    """Re-build the co-leakage matrix (output-output) for the collapsed chart."""
    co = np.zeros((N_ATTRS, N_ATTRS), dtype=int)
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
            co[ia][ia] += 1
            for b in attr_list[i + 1:]:
                ib = ATTR_IDX[b]
                co[ia][ib] += 1
                co[ib][ia] += 1
    return co


# ── Collapse 21×21 → 8×8 by family ───────────────────────────────────────────
def collapse_to_families(mat: np.ndarray) -> np.ndarray:
    """Sum attribute-level matrix into family-level matrix (8×8)."""
    n_fam = len(ORDERED_FAMS)
    fam_mat = np.zeros((n_fam, n_fam), dtype=int)
    for ri, rf in enumerate(ORDERED_FAMS):
        for ci, cf in enumerate(ORDERED_FAMS):
            row_attrs = [ATTR_IDX[a] for a in ATTR_FAMILIES[rf] if a in ATTR_IDX]
            col_attrs = [ATTR_IDX[a] for a in ATTR_FAMILIES[cf] if a in ATTR_IDX]
            fam_mat[ri][ci] = mat[np.ix_(row_attrs, col_attrs)].sum()
    return fam_mat


# ── Family boundary positions ─────────────────────────────────────────────────
def family_boundaries() -> list[int]:
    boundaries, pos = [], 0
    for fam in FAMILY_ORDER[:-1]:
        pos += len(ATTR_FAMILIES[fam])
        boundaries.append(pos)
    return boundaries


# ── Matplotlib 21×21 heatmap ──────────────────────────────────────────────────
def plot_21x21(mat: np.ndarray, figsize: tuple[float, float],
               out_path: Path, dpi: int, ylabel: str = "", xlabel: str = "") -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    cmap = plt.get_cmap("Greys")
    vmax = mat.max() if mat.max() > 0 else 1
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="equal",
                   interpolation="nearest")

    threshold = 0.6 * vmax
    font_size = max(5, min(7, figsize[0] * 0.8))
    for i in range(N_ATTRS):
        for j in range(N_ATTRS):
            val = mat[i, j]
            if val > 0:
                text_color = "white" if val >= threshold else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=font_size, color=text_color, fontweight="normal")

    for x in range(N_ATTRS + 1):
        ax.axvline(x - 0.5, color="lightgray", linewidth=0.3, zorder=3)
        ax.axhline(x - 0.5, color="lightgray", linewidth=0.3, zorder=3)

    for b in family_boundaries():
        ax.axvline(b - 0.5, color="black", linewidth=0.8, zorder=4)
        ax.axhline(b - 0.5, color="black", linewidth=0.8, zorder=4)

    ax.set_xticks(range(N_ATTRS))
    ax.set_yticks(range(N_ATTRS))
    tick_fs = max(5, min(7, figsize[0] * 0.75))
    ax.set_xticklabels(ORDERED_ATTRS, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(ORDERED_ATTRS, fontsize=tick_fs)

    for tick, attr in zip(ax.get_xticklabels(), ORDERED_ATTRS):
        fam = ATTR_TO_FAMILY.get(attr, "")
        tick.set_color(FAM_TICK_COLORS.get(fam, "black"))
    for tick, attr in zip(ax.get_yticklabels(), ORDERED_ATTRS):
        fam = ATTR_TO_FAMILY.get(attr, "")
        tick.set_color(FAM_TICK_COLORS.get(fam, "black"))

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=tick_fs + 1)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=tick_fs + 1)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Matplotlib 8×8 family heatmap (for paper PNG) ────────────────────────────
def plot_8x8_mpl(fam_mat: np.ndarray, figsize: tuple[float, float],
                 out_path: Path, dpi: int, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    cmap = plt.get_cmap("Greys")
    vmax = fam_mat.max() if fam_mat.max() > 0 else 1
    ax.imshow(fam_mat, cmap=cmap, vmin=0, vmax=vmax, aspect="equal",
              interpolation="nearest")

    threshold = 0.6 * vmax
    fs = max(6, min(9, figsize[0] * 1.2))
    for i in range(len(ORDERED_FAMS)):
        for j in range(len(ORDERED_FAMS)):
            val = fam_mat[i, j]
            if val > 0:
                text_color = "white" if val >= threshold else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=fs, color=text_color)

    n = len(ORDERED_FAMS)
    for x in range(n + 1):
        ax.axvline(x - 0.5, color="lightgray", linewidth=0.4, zorder=3)
        ax.axhline(x - 0.5, color="lightgray", linewidth=0.4, zorder=3)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    label_fs = max(6, min(8, figsize[0] * 0.9))
    ax.set_xticklabels(FAM_SHORT_LIST, rotation=45, ha="right", fontsize=label_fs)
    ax.set_yticklabels(FAM_SHORT_LIST, fontsize=label_fs)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Altair 8×8 (3-panel side-by-side HTML) ───────────────────────────────────
def make_altair_panel(fam_mat: np.ndarray, title: str,
                      max_val: int, width: int = 220) -> alt.Chart:
    n = len(ORDERED_FAMS)
    records = []
    for ri, rf in enumerate(ORDERED_FAMS):
        for ci, cf in enumerate(ORDERED_FAMS):
            records.append({
                "row_family": FAM_SHORT[rf],
                "col_family": FAM_SHORT[cf],
                "count": int(fam_mat[ri, ci]),
                "row_order": ri,
                "col_order": ci,
            })
    source = pd.DataFrame(records)

    # Use a shared max_val so all three charts have the same color scale
    color_scale = alt.Scale(domain=[0, max_val], scheme="greys")

    heatmap = alt.Chart(source, title=title).mark_rect(stroke="lightgray", strokeWidth=0.5).encode(
        x=alt.X("col_family:N",
                sort=FAM_SHORT_LIST,
                axis=alt.Axis(labelAngle=-45, title=None, labelFontSize=10)),
        y=alt.Y("row_family:N",
                sort=FAM_SHORT_LIST,
                axis=alt.Axis(title=None, labelFontSize=10)),
        color=alt.Color("count:Q",
                        scale=color_scale,
                        legend=alt.Legend(title="count", orient="right",
                                          gradientLength=80)),
    ).properties(width=width, height=width)

    text = alt.Chart(source).mark_text(fontSize=9).encode(
        x=alt.X("col_family:N", sort=FAM_SHORT_LIST),
        y=alt.Y("row_family:N", sort=FAM_SHORT_LIST),
        text=alt.Text("count:Q"),
        color=alt.condition(
            alt.datum.count > max_val * 0.6,
            alt.value("white"),
            alt.value("black"),
        ),
    )

    return (heatmap + text)


def save_altair_html(co_fam: np.ndarray, raw_fam: np.ndarray, ext_fam: np.ndarray,
                     out_path: Path) -> None:
    # Shared max across all three so the gradient is comparable
    max_val = int(max(co_fam.max(), raw_fam.max(), ext_fam.max()))

    chart_co  = make_altair_panel(co_fam,  "Co-leakage (ext→ext)",    max_val)
    chart_raw = make_altair_panel(raw_fam, "GT → raw output",          max_val)
    chart_ext = make_altair_panel(ext_fam, "GT → externalized",        max_val)

    combined = (chart_co | chart_raw | chart_ext).properties(
        title=alt.TitleParams(
            "Attribute-family flow matrices (8×8 collapsed)",
            fontSize=13, anchor="middle",
        )
    ).configure_title(
        fontSize=11
    ).configure_view(strokeWidth=0)

    combined.save(str(out_path))
    print(f"  Saved: {out_path}")


# ── Print summary ─────────────────────────────────────────────────────────────
def print_summary(mat_raw: np.ndarray, mat_ext: np.ndarray, total_items: int) -> None:
    print(f"\n=== Input Flow Summary (total items: {total_items:,}) ===")
    print(f"\nGT → Raw Output matrix max = {mat_raw.max()}")
    print(f"GT → Externalized matrix max = {mat_ext.max()}")

    print("\nTop 10 (GT attr → raw output attr) pairs:")
    raw_pairs = [(ORDERED_ATTRS[i], ORDERED_ATTRS[j], mat_raw[i, j])
                 for i in range(N_ATTRS) for j in range(N_ATTRS) if mat_raw[i, j] > 0]
    raw_pairs.sort(key=lambda x: -x[2])
    for a, b, cnt in raw_pairs[:10]:
        tag = " [same]" if a == b else ""
        print(f"  GT={a:<20s} → raw={b:<20s} : {cnt}{tag}")

    print("\nTop 10 (GT attr → externalized attr) pairs:")
    ext_pairs = [(ORDERED_ATTRS[i], ORDERED_ATTRS[j], mat_ext[i, j])
                 for i in range(N_ATTRS) for j in range(N_ATTRS) if mat_ext[i, j] > 0]
    ext_pairs.sort(key=lambda x: -x[2])
    for a, b, cnt in ext_pairs[:10]:
        tag = " [same]" if a == b else ""
        print(f"  GT={a:<20s} → ext={b:<20s} : {cnt}{tag}")

    # Diagonal (passthrough) vs off-diagonal (transformation)
    diag_raw = int(np.trace(mat_raw))
    diag_ext = int(np.trace(mat_ext))
    total_raw = int(mat_raw.sum())
    total_ext = int(mat_ext.sum())
    print(f"\nPassthrough (diagonal) raw: {diag_raw}/{total_raw} "
          f"({100*diag_raw/max(1,total_raw):.1f}%)")
    print(f"Passthrough (diagonal) ext: {diag_ext}/{total_ext} "
          f"({100*diag_ext/max(1,total_ext):.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading corpus...")
    df = load_corpus()
    print(f"  Corpus: {df['full_key'].nunique():,} items, "
          f"{df['dir_name'].nunique()} configs, {df['app'].nunique()} apps")

    print("Building input-flow matrices...")
    mat_raw, mat_ext, total = build_flow_matrices(df)

    print("Building co-leakage matrix (for collapsed chart)...")
    mat_co = build_co_matrix(df)

    print_summary(mat_raw, mat_ext, total)

    # ── 21×21 PNG plots ──────────────────────────────────────────────────────
    print("\nPlotting 21×21 heatmaps...")
    for mat, tag, ylabel, xlabel in [
        (mat_raw, "raw",
         "GT input attribute", "Raw output confirmed attribute"),
        (mat_ext, "ext",
         "GT input attribute", "Externalized confirmed attribute"),
    ]:
        plot_21x21(mat, (7.0, 7.0),
                   ATTACH_DIR / f"leakage_input_flow_{tag}_1x1.png",
                   dpi=150, ylabel=ylabel, xlabel=xlabel)
        plot_21x21(mat, (3.5, 3.5),
                   ATTACH_DIR / f"leakage_input_flow_{tag}_2x1.png",
                   dpi=200, ylabel=ylabel, xlabel=xlabel)

    # ── 8×8 collapsed matrices ───────────────────────────────────────────────
    print("\nCollapsing to 8×8 family matrices...")
    co_fam  = collapse_to_families(mat_co)
    raw_fam = collapse_to_families(mat_raw)
    ext_fam = collapse_to_families(mat_ext)

    print("\nPlotting 8×8 family heatmaps (matplotlib)...")
    for fam_mat, tag in [
        (co_fam,  "coleakage"),
        (raw_fam, "raw"),
        (ext_fam, "ext"),
    ]:
        plot_8x8_mpl(fam_mat, (3.5, 3.5),
                     ATTACH_DIR / f"leakage_family_collapsed_{tag}_1x1.png",
                     dpi=150)
        plot_8x8_mpl(fam_mat, (2.5, 2.5),
                     ATTACH_DIR / f"leakage_family_collapsed_{tag}_2x1.png",
                     dpi=200)

    print("\nGenerating Altair 3-panel HTML...")
    save_altair_html(co_fam, raw_fam, ext_fam,
                     ATTACH_DIR / "leakage_family_collapsed_altair.html")

    # ── Copy to paper/figures/ ───────────────────────────────────────────────
    print("\nCopying to paper/figures/...")
    for src_name, dst_name in [
        ("leakage_input_flow_raw_2x1.png",            "leakage_input_flow_raw_2x1.png"),
        ("leakage_input_flow_ext_2x1.png",            "leakage_input_flow_ext_2x1.png"),
        ("leakage_family_collapsed_coleakage_2x1.png","leakage_family_collapsed_coleakage_2x1.png"),
        ("leakage_family_collapsed_raw_2x1.png",      "leakage_family_collapsed_raw_2x1.png"),
        ("leakage_family_collapsed_ext_2x1.png",      "leakage_family_collapsed_ext_2x1.png"),
    ]:
        src = ATTACH_DIR / src_name
        dst = FIGURES_DIR / dst_name
        shutil.copy2(src, dst)
        print(f"  Copied: {dst_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
