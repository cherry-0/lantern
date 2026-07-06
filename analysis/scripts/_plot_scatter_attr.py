#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from _leakage_common import load_data, CATEGORY_COLORS

ATTACH_DIR = Path(__file__).resolve().parent.parent / "attachments"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_MARKERS = {
    "Identity & Identifiability": "o",
    "Demographic":                 "s",
    "Health & Medical":            "^",
    "Location & Spatial":          "D",
    "Religion & Cultural":         "v",
    "Appearance & Body":           "P",
    "Attire, Role & Group":        "*",
    "Activity & Lifestyle":        "X",
}
FAMILY_SHORT = {
    "Identity & Identifiability": "Identity",
    "Demographic":                 "Demographic",
    "Health & Medical":            "Health",
    "Location & Spatial":          "Location",
    "Religion & Cultural":         "Religion",
    "Appearance & Body":           "Appearance",
    "Attire, Role & Group":        "Attire/Role",
    "Activity & Lifestyle":        "Activity",
}


def main():
    print("Loading data...")
    df, _ = load_data()
    print(f"  rows: {len(df)}")

    grp = df.groupby(["app", "family"]).agg(
        input_rate=("input_label", "mean"),
        confirmed_rate=("ext_conf", "mean"),
        category=("category", "first"),
    ).reset_index()
    print(f"  (app, family) points: {len(grp)}")

    BASE_FS = 13
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": BASE_FS,
        "axes.labelsize": BASE_FS * 1.15,
        "xtick.labelsize": BASE_FS,
        "ytick.labelsize": BASE_FS,
        "legend.fontsize": BASE_FS * 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    categories = [c for c in CATEGORY_COLORS if c in grp["category"].values]
    families   = list(FAMILY_MARKERS.keys())

    # Draw scatter points: color=category, marker=family
    for cat in categories:
        color = CATEGORY_COLORS[cat]
        sub = grp[grp["category"] == cat]
        for fam in families:
            fsub = sub[sub["family"] == fam]
            if fsub.empty:
                continue
            marker = FAMILY_MARKERS[fam]
            ax.scatter(fsub["input_rate"], fsub["confirmed_rate"],
                       color=color, marker=marker, s=65, alpha=0.5, zorder=3,
                       edgecolors="white", linewidths=0.4)

    lim_max = 0.60
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.02, 0.63)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_aspect("equal")

    diag = np.linspace(0, lim_max, 200)
    ax.plot(diag, diag, color="#CCCCCC", linewidth=1.2, linestyle="--", zorder=0)

    shade_x = np.linspace(0, lim_max, 200)
    ax.fill_between(shade_x, shade_x, np.full_like(shade_x, lim_max),
                    color="#FFB3C1", alpha=0.18, zorder=0)

    ax.set_xlabel("Fraction of scenarios where attribute is in input", fontsize=BASE_FS * 1.1)
    ax.set_ylabel("Fraction of scenarios where\nattribute is confirmed leaked", fontsize=BASE_FS * 1.1)

    # Attribute (family) legend — top, 2 rows (4 columns)
    fam_handles = [
        mlines.Line2D([0], [0], marker=FAMILY_MARKERS[f], color="none",
                      markerfacecolor="white", markeredgecolor="black",
                      markeredgewidth=1.0, linestyle="None", markersize=7,
                      label=FAMILY_SHORT[f])
        for f in families
    ]
    leg_fam = ax.legend(handles=fam_handles, title="Attribute Family",
                        bbox_to_anchor=(0.5, 1.02), loc="lower center",
                        ncol=4, borderaxespad=0, frameon=False,
                        fontsize=BASE_FS * 0.90, title_fontsize=BASE_FS * 0.90)

    out = ATTACH_DIR / "motivation_scatter_attr.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                bbox_extra_artists=[leg_fam])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
