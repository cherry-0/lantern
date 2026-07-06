#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

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


def lighten(hex_color: str, frac: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    return "#{:02X}{:02X}{:02X}".format(
        int((r+(1-r)*frac)*255), int((g+(1-g)*frac)*255), int((b+(1-b)*frac)*255))


def smooth_hull(pts, pad=0.018):
    if len(pts) < 3:
        return None, None
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    if x.max() - x.min() < 1e-6:
        x += np.linspace(-1e-5, 1e-5, len(x))
    if y.max() - y.min() < 1e-6:
        y += np.linspace(-1e-5, 1e-5, len(y))
    p2 = np.column_stack([x, y])
    try:
        hull = ConvexHull(p2)
    except Exception:
        return None, None
    hp = p2[hull.vertices]
    cx, cy = hp[:, 0].mean(), hp[:, 1].mean()
    ex = hp + pad * (hp - np.array([cx, cy]))
    ex = np.vstack([ex, ex[0]])
    try:
        tck, _ = splprep([ex[:, 0], ex[:, 1]], s=0, per=True, k=3)
        sx, sy = splev(np.linspace(0, 1, 300), tck)
        return sx, sy
    except Exception:
        return ex[:, 0], ex[:, 1]


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

    # Draw smooth hull regions
    for cat in categories:
        color = CATEGORY_COLORS[cat]
        sub = grp[grp["category"] == cat]
        pts = sub[["input_rate", "confirmed_rate"]].values
        if len(pts) < 3:
            continue
        sx, sy = smooth_hull(pts, pad=0.020)
        if sx is None:
            continue
        ax.fill(sx, sy, color=lighten(color, 0.72), alpha=0.35, zorder=1)
        ax.plot(sx, sy, color=lighten(color, 0.10), linewidth=1.2, alpha=1.0,
                linestyle="--", zorder=2)

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

    # Pink shade — above y=x diagonal, up to y=0.6
    shade_x = np.linspace(0, lim_max, 200)
    ax.fill_between(shade_x, shade_x, np.full_like(shade_x, lim_max),
                    color="#FFB3C1", alpha=0.18, zorder=0)

    ax.set_xlabel("Fraction of scenarios where attribute is in input", fontsize=BASE_FS * 1.1)
    ax.set_ylabel("Fraction of scenarios where\nattribute is confirmed leaked", fontsize=BASE_FS * 1.1)

    # Category legend — right side, vertical
    cat_handles = [mpatches.Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in categories]
    leg_cat = ax.legend(handles=cat_handles, title="App Category",
                        bbox_to_anchor=(0.5, 1.02), loc="lower center",
                        ncol=3, borderaxespad=0, frameon=False,
                        fontsize=BASE_FS * 0.90, title_fontsize=BASE_FS * 0.90)

    out = ATTACH_DIR / "motivation_scatter_region.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                bbox_extra_artists=[leg_cat])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
