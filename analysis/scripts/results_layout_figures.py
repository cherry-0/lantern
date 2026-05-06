"""
Paper-specific composite figures for the Results section layout.

This script intentionally produces only the figures whose layout differs from
the exploratory/headline scripts:
  - Fig. 7(a): aggregate + category stage gap in one shared-axis panel.
  - Fig. 11: family-level co-leakage / GT->Raw / GT->Ext matrices with
    stage-colored gradients and explicit axis labels.
  - Fig. 12(a): modality-pair stage matrices, one palette per modality pair.
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
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
LANTERN_ROOT = SCRIPT_DIR.parent.parent
FIG_DIR = LANTERN_ROOT / "paper" / "26_CCS_Lantern" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from _leakage_common import (  # noqa: E402
    load_data, ATTR_FAMILIES, ATTR_TO_FAMILY, MOD_PAIR_COLOR,
    S_INPUT, P_ORANGE, S_EXT, P_TEAL, GRAY,
)


def _stage_rates(sub: pd.DataFrame) -> dict[str, float]:
    out_known = sub[sub["output_has"] == 1]
    return {
        "raw": out_known["output_leak"].mean() if len(out_known) else 0.0,
        "any": sub["ext_leak"].mean(),
        "conf": sub["ext_conf"].mean(),
    }


def fig_stage_aggregate_category(df: pd.DataFrame) -> None:
    rows = [{"category": "Aggregated", **_stage_rates(df)}]
    for category, sub in df.groupby("category"):
        rows.append({"category": category, **_stage_rates(sub)})
    stage_df = pd.DataFrame(rows)

    order = ["Aggregated"] + (
        stage_df[stage_df["category"] != "Aggregated"]
        .sort_values("conf", ascending=False)["category"].tolist()
    )
    stage_df = stage_df.set_index("category").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(12.0, 4.0), facecolor="white")
    x = np.arange(len(stage_df))
    w = 0.24
    stages = [
        ("raw", "raw output", S_INPUT),
        ("any", "any-leak ext.", P_ORANGE),
        ("conf", "confirmed ext.", S_EXT),
    ]
    for j, (key, label, color) in enumerate(stages):
        vals = stage_df[key].values
        ax.bar(x + (j - 1) * w, vals, w, color=color, alpha=0.95,
               edgecolor="white", linewidth=0.6, label=label)
        for xi, val in zip(x + (j - 1) * w, vals):
            ax.text(xi, val + 0.007, f"{val:.0%}", ha="center",
                    va="bottom", fontsize=8.5, color="#333",
                    fontweight="bold" if key == "conf" else "normal")

    ax.axvline(0.5, color="#BDBDBD", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_df["category"], rotation=35, ha="right", fontsize=9.0)
    ax.set_ylabel("Rate over pairs")
    ax.set_ylim(0, max(stage_df[["raw", "any", "conf"]].max().max(), 0.05) * 1.22)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "leakage_headline_a_stage_category.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out.relative_to(LANTERN_ROOT)}")


FAMILY_ORDER = list(ATTR_FAMILIES.keys())
FAM_SHORT = {
    "Identity & Identifiability": "Identity",
    "Demographic": "Demographic",
    "Health & Medical": "Health",
    "Location & Spatial": "Location",
    "Religion & Cultural": "Religion",
    "Appearance & Body": "Appearance",
    "Attire, Role & Group": "Attire",
    "Activity & Lifestyle": "Activity",
}
FAM_LABELS = [FAM_SHORT[f] for f in FAMILY_ORDER]


def _family_item_sets(df: pd.DataFrame, col: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for _, row in df[df[col] == 1][["full_key", "family"]].drop_duplicates().iterrows():
        if row["family"] in FAMILY_ORDER:
            out[row["full_key"]].add(row["family"])
    return out


def _family_matrix(df: pd.DataFrame, mode: str) -> np.ndarray:
    idx = {f: i for i, f in enumerate(FAMILY_ORDER)}
    mat = np.zeros((len(FAMILY_ORDER), len(FAMILY_ORDER)), dtype=int)

    if mode == "coleak":
        item_fams = _family_item_sets(df, "ext_conf")
        for fams in item_fams.values():
            fam_list = sorted(fams, key=lambda f: idx[f])
            for i, rf in enumerate(fam_list):
                mat[idx[rf], idx[rf]] += 1
                for cf in fam_list[i + 1:]:
                    mat[idx[rf], idx[cf]] += 1
                    mat[idx[cf], idx[rf]] += 1
        return mat

    out_col = "output_conf" if mode == "raw" else "ext_conf"
    for _, grp in df.groupby("full_key"):
        gt_fams = sorted(set(grp[grp["input_label"] == 1]["family"]) & set(FAMILY_ORDER),
                         key=lambda f: idx[f])
        out_fams = sorted(set(grp[grp[out_col] == 1]["family"]) & set(FAMILY_ORDER),
                          key=lambda f: idx[f])
        for rf in gt_fams:
            for cf in out_fams:
                mat[idx[rf], idx[cf]] += 1
    return mat


def _white_to(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("white_to_color", ["#FFFFFF", color])


def fig_family_flow(df: pd.DataFrame) -> None:
    panels = [
        ("Externalized family", "Externalized family", "Ext→Ext", _family_matrix(df, "coleak"), S_EXT),
        ("Input GT family", "Raw-output family", "GT→Raw", _family_matrix(df, "raw"), S_EXT),
        ("Input GT family", "Externalized family", "GT→Ext", _family_matrix(df, "ext"), S_EXT),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.05), facecolor="white")
    for ax, (ylabel, xlabel, title, mat, color) in zip(axes, panels):
        vmax = mat.max() if mat.max() > 0 else 1
        ax.imshow(mat, cmap=_white_to(color), vmin=0, vmax=vmax, aspect="equal")
        threshold = 0.58 * vmax
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = int(mat[i, j])
                if val:
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=7.2, color="white" if val >= threshold else "#222")
        for x in range(len(FAMILY_ORDER) + 1):
            ax.axvline(x - 0.5, color="#D9D9D9", linewidth=0.35)
            ax.axhline(x - 0.5, color="#D9D9D9", linewidth=0.35)
        ax.set_title(title, fontsize=9.5, pad=5)
        ax.set_xticks(range(len(FAMILY_ORDER)))
        ax.set_yticks(range(len(FAMILY_ORDER)))
        ax.set_xticklabels(FAM_LABELS, rotation=35, ha="right", fontsize=7.1)
        ax.set_yticklabels(FAM_LABELS, fontsize=7.1)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    fig.tight_layout(w_pad=1.1)
    out = FIG_DIR / "leakage_family_collapsed_stagecolored.png"
    fig.savefig(out, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out.relative_to(LANTERN_ROOT)}")


def fig_modality_stage(df: pd.DataFrame) -> None:
    pairs = ["text→text", "text→image", "image→text", "image→image"]
    stages = [("input_label", "Input GT"), ("output_leak", "Raw output"), ("ext_conf", "Confirmed ext.")]

    fs = 1.3
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 3.9), facecolor="white", sharey=True)
    for ax, mp in zip(axes, pairs):
        sub_mp = df[df["modality_pair"] == mp]
        color = MOD_PAIR_COLOR.get(mp, P_TEAL)
        records = []
        for col, stage in stages:
            for fam in FAMILY_ORDER:
                sub = sub_mp[sub_mp["family"] == fam]
                if col == "output_leak":
                    sub = sub[sub["output_has"] == 1]
                records.append({
                    "stage": stage,
                    "family": fam,
                    "rate": sub[col].mean() * 100 if len(sub) else np.nan,
                })
        mat = (pd.DataFrame(records)
               .pivot(index="stage", columns="family", values="rate")
               .reindex([s for _, s in stages])[FAMILY_ORDER])
        ax.imshow(mat.values, cmap=_white_to(color), vmin=0, vmax=60, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=7.2 * fs, color="white" if val >= 36 else "#222")
        for x in range(len(FAMILY_ORDER) + 1):
            ax.axvline(x - 0.5, color="#E0E0E0", linewidth=0.35)
        for y in range(len(stages) + 1):
            ax.axhline(y - 0.5, color="#E0E0E0", linewidth=0.35)
        ax.set_title(f"{mp}\n(n={sub_mp['full_key'].nunique():,})", fontsize=9.2 * fs)
        ax.set_xticks(range(len(FAMILY_ORDER)))
        ax.set_xticklabels(
            [FAM_SHORT[f].replace("Identity", "Identity\n& ID") for f in FAMILY_ORDER],
            rotation=42, ha="right", fontsize=6.7 * fs,
        )
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels([s for _, s in stages], fontsize=8.2 * fs)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    fig.tight_layout(w_pad=1.35)
    out = FIG_DIR / "deep_fig13_modality_stage_panels_colored.png"
    fig.savefig(out, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out.relative_to(LANTERN_ROOT)}")


def main() -> None:
    df, _ = load_data()
    fig_stage_aggregate_category(df)
    fig_family_flow(df)
    fig_modality_stage(df)


if __name__ == "__main__":
    main()
