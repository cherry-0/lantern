"""
Inference-Induced Privacy Leakage Analysis
Analyzes prompt4/prompt5 evaluation results from verify/outputs/ cache directories.

Usage: python analysis/scripts/leakage_analysis.py
Outputs: analysis/attachments/leakage_*.png  +  analysis/leakage_landscape.md
"""

from __future__ import annotations
import json, sys, textwrap
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR  = LANTERN_ROOT / "verify" / "outputs"
ATTACH_DIR   = LANTERN_ROOT / "analysis" / "attachments"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)
SKIP_FILES   = {"run_config.json", "dir_summary.json", "report.json", "report.csv"}

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
# Stage colors kept as-is (blue/orange/red for input/raw-output/externalized)
BLUE   = "#3A7DC9"; ORANGE = "#E8762C"; GREEN  = "#2CA463"
RED    = "#C93A3A"; GRAY   = "#AAAAAA"; PURPLE = "#8A4DC9"
YELLOW = "#D4A017"; TEAL   = "#2AADAD"

# ── Custom palette (Adobe swatches) ───────────────────────────────────────────
P_TEAL   = "#7ADBC4"   # teal-green
P_YELLOW = "#FAD765"   # golden yellow
P_ORANGE = "#FA9F5C"   # warm orange
P_GREEN  = "#98D198"   # sage green
P_BLUE   = "#6C80FC"   # periwinkle blue
P_MAUVE  = "#ACA4B3"   # muted purple-gray
P_SLATE  = "#687692"   # slate blue-gray

VERDICT_COLORS = {
    "confirmed leakage": RED,
    "possible leakage":  ORANGE,
    "no evidence":       GREEN,
    "na":                "#DDDDDD",
}
# Channels use new palette
CHANNEL_COLORS = {
    "NETWORK":   P_BLUE,
    "STORAGE":   P_ORANGE,
    "UI":        P_TEAL,
    "LOGGING":   P_SLATE,
    "AGGREGATE": RED,
}

# ── Attribute taxonomy (from paper appendix) ───────────────────────────────────
ATTR_FAMILIES: Dict[str, List[str]] = {
    "Identity &\nIdentifiability":  ["face", "identity"],
    "Demographic":                   ["age", "gender", "race", "marital status"],
    "Health &\nMedical":             ["disability", "medical"],
    "Location &\nSpatial":           ["location"],
    "Religion &\nCultural":          ["religion", "ethnic_clothing"],
    "Appearance &\nBody":            ["nudity", "height", "weight", "haircolor", "color"],
    "Attire, Role &\nGroup":         ["formal", "casual", "uniforms", "troupe"],
    "Activity &\nLifestyle":         ["sports"],
}
FAMILY_COLORS = {
    "Identity &\nIdentifiability": P_BLUE,
    "Demographic":                  P_ORANGE,
    "Health &\nMedical":            P_MAUVE,
    "Location &\nSpatial":          P_TEAL,
    "Religion &\nCultural":         P_YELLOW,
    "Appearance &\nBody":           P_GREEN,
    "Attire, Role &\nGroup":        P_SLATE,
    "Activity &\nLifestyle":        "#D4A017",
}
ALL_ATTRS = [a for fam in ATTR_FAMILIES.values() for a in fam]
ATTR_TO_FAMILY = {a: fam for fam, attrs in ATTR_FAMILIES.items() for a in attrs}
CHANNELS = ["NETWORK", "STORAGE", "UI", "LOGGING"]

# ── App category mapping (from paper Table~\ref{tab:app-workflows} / fig:apps) ─
APP_CATEGORY = {
    # Photo/Camera
    "momentag":                     "Photo/Camera",
    "tool-neuron":                  "Photo/Camera",
    # Health/Fitness
    "skin-disease-detection":       "Health/Fitness",
    "waico":                        "Health/Fitness",
    "healyks":                      "Health/Fitness",
    "nutri-track":                  "Health/Fitness",
    # Productivity/Assistant
    "clone":                        "Productivity",
    "google-ai-edge-gallery":       "Productivity",
    "klyr":                         "Productivity",
    "snapdo":                       "Productivity",
    "pocketpal-ai":                 "Productivity",
    # Finance
    "budget-lens":                  "Finance",
    "fiscal-flow":                  "Finance",
    "spendsense":                   "Finance",
    "finchain":                     "Finance",
    "chat-driven-expense-tracker":  "Finance",
    # Social/Communication
    "lira":                         "Social/Comm.",
    "llm-vtuber":                   "Social/Comm.",
    "xend":                         "Social/Comm.",
    # Education
    "deeptutor":                    "Education",
    "edupal":                       "Education",
    "sgpa":                         "Education",
}
# Ordered to match the paper grouping in fig:apps.
CATEGORY_ORDER = [
    "Photo/Camera",
    "Health/Fitness",
    "Productivity",
    "Finance",
    "Social/Comm.",
    "Education",
]
CATEGORY_COLORS = {
    "Photo/Camera":   P_ORANGE,
    "Health/Fitness": P_MAUVE,
    "Productivity":   P_BLUE,
    "Finance":        P_GREEN,
    "Social/Comm.":   P_TEAL,
    "Education":      P_YELLOW,
}

# Dataset modality classification
DATASET_INPUT_TYPE = {
    "HR-VISPR": "image",
    "SROIE2019": "docs",
    "MIMIC-CXR": "image",
    "PrivacyLens": "text",
    "SynthPAI": "text",
    "GretelSyntheticPII": "docs",
    "ASAP-AES": "text",
    "MultiCaRe": "text",
    "OpenPII": "text",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def entry_to_verdict(entry: Any) -> str:
    if not isinstance(entry, dict):
        return "no evidence"
    agg = entry.get("aggregate", entry)
    if not isinstance(agg, dict):
        agg = entry
    v = agg.get("verdict", "")
    if v in ("confirmed leakage", "possible leakage", "no evidence"):
        return v
    return "confirmed leakage" if agg.get("inferable") else "no evidence"

def entry_to_score(entry: Any) -> int:
    v = entry_to_verdict(entry)
    return {"confirmed leakage": 2, "possible leakage": 1, "no evidence": 0}.get(v, 0)

def get_channel_verdicts(ext_eval_attr: Any) -> Dict[str, str]:
    if not isinstance(ext_eval_attr, dict):
        return {}
    channels = ext_eval_attr.get("channels", {})
    if not isinstance(channels, dict):
        return {}
    return {ch: entry_to_verdict(v) for ch, v in channels.items()}

def get_output_verdict(output_eval_attr: Any) -> str:
    if not isinstance(output_eval_attr, dict):
        return "no evidence"
    v = output_eval_attr.get("verdict", "")
    if v in ("confirmed leakage", "possible leakage", "no evidence"):
        return v
    return "confirmed leakage" if output_eval_attr.get("inferable") else "no evidence"

try:
    from _leakage_common import load_all_data as _load_all_data_unified
except ImportError:  # support running as `python analysis/scripts/leakage_analysis.py`
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _leakage_common import load_all_data as _load_all_data_unified


def load_all_data() -> pd.DataFrame:
    """Load + filter via the shared loader in _leakage_common.

    Filtering is identical for both leakage_analysis.py and leakage_deep.py:
      - drop items whose ext_eval is empty / missing aggregate verdict;
      - drop (app, dataset) configs with fewer than 100 successful items,
        except image->image which is exempt (kept regardless of N).
    Legacy column names used by this script are added as aliases.
    """
    return _load_all_data_unified()

# ── Helpers ────────────────────────────────────────────────────────────────────

def annotate_bars(ax, fmt="{:.1%}", fontsize=9, offset=0.005):
    for p in ax.patches:
        h = p.get_height()
        if h > 0.005:
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width()/2, h + offset),
                        ha="center", va="bottom", fontsize=fontsize)

def annotate_hbars(ax, fmt="{:.1%}", fontsize=9, offset=0.005):
    for p in ax.patches:
        w = p.get_width()
        if w > 0.01:
            ax.annotate(fmt.format(w),
                        (w + offset, p.get_y() + p.get_height()/2),
                        ha="left", va="center", fontsize=fontsize)

def save(fig, name):
    path = ATTACH_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Category helpers ──────────────────────────────────────────────────────────
def _cat_index(app: str) -> int:
    cat = APP_CATEGORY.get(app, "Other")
    return CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)


def apps_sorted_by_category(apps_iter, value_dict: Optional[Dict[str, float]] = None,
                            ascending: bool = False) -> List[str]:
    """Return apps grouped by paper category order; within a category, sort by value or name."""
    apps = list(apps_iter)
    def key(a):
        ci = _cat_index(a)
        if value_dict is not None:
            v = value_dict.get(a, 0)
            return (ci, v if ascending else -v, a)
        return (ci, a)
    return sorted(apps, key=key)


def color_ticks_by_category(ax, app_order: List[str], axis: str = "x") -> None:
    """Color tick labels by app category."""
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label, app in zip(labels, app_order):
        cat = APP_CATEGORY.get(app, "Other")
        color = CATEGORY_COLORS.get(cat, "#444444")
        label.set_color(color)
        label.set_fontweight("bold")


def category_legend_handles():
    """List of mpatches.Patch entries for the app-category legend."""
    return [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    return path.name

def fig_bg():
    return "white"

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def fig1_overall_verdict_distribution(df: pd.DataFrame) -> str:
    """Stacked bar: verdict distribution per app (aggregate externalization)."""
    order = ["confirmed leakage", "possible leakage", "no evidence"]
    agg = (df.groupby(["app", "agg_verdict"])
             .size().reset_index(name="n"))
    total = agg.groupby("app")["n"].transform("sum")
    agg["pct"] = agg["n"] / total

    conf_by_app = df.groupby("app")["agg_confirmed"].mean().to_dict()
    apps = apps_sorted_by_category(conf_by_app.keys(), conf_by_app, ascending=False)
    agg = agg.set_index(["app","agg_verdict"])["pct"].unstack(fill_value=0)
    for v in order:
        if v not in agg.columns: agg[v] = 0
    agg = agg.reindex(apps)[order]

    fig, ax = plt.subplots(figsize=(13, 5.6), facecolor=fig_bg())
    ax.set_facecolor(fig_bg())
    bottom = np.zeros(len(agg))
    for v in order:
        vals = agg[v].values
        bars = ax.bar(agg.index, vals, bottom=bottom,
                      color=VERDICT_COLORS[v], label=v, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0.06:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + val/2,
                        f"{val:.0%}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottom += vals

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of attribute-item pairs")
    ax.set_title("Fig 1 — Overall Externalization Verdict Distribution by App\n(grouped by app category from Table~app-workflows; tick label color = category)", fontsize=12)
    ax.set_xticks(range(len(agg.index)))
    ax.set_xticklabels(agg.index, rotation=35, ha="right")
    color_ticks_by_category(ax, list(agg.index), axis="x")
    leg1 = ax.legend(loc="upper right", framealpha=0.9, title="Verdict")
    ax.add_artist(leg1)
    ax.legend(handles=category_legend_handles(), loc="upper left",
              fontsize=8, title="App category", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return save(fig, "leakage_fig1_overall_verdict_by_app.png")


def fig2_verdict_by_attr_family(df: pd.DataFrame) -> str:
    """Grouped bar: per-family confirmed/any leakage at BOTH raw-output and externalization stages."""
    fam_order = list(ATTR_FAMILIES.keys())
    rows = []
    for fam in fam_order:
        sub = df[df["family"] == fam]
        if len(sub) == 0:
            rows.append({"family": fam, "conf_raw": 0, "any_raw": 0, "conf_ext": 0, "any_ext": 0})
            continue
        raw_sub = sub[sub["output_present"] == 1]
        if len(raw_sub) > 0:
            ov = raw_sub["output_verdict"].astype(str)
            conf_raw = (ov == "confirmed leakage").mean()
            any_raw  = ov.isin(["confirmed leakage", "possible leakage"]).mean()
        else:
            conf_raw = any_raw = 0
        rows.append({
            "family":   fam,
            "conf_raw": conf_raw,
            "any_raw":  any_raw,
            "conf_ext": sub["agg_confirmed"].mean(),
            "any_ext":  sub["agg_leakage"].mean(),
        })
    agg = pd.DataFrame(rows).set_index("family").reindex(fam_order).fillna(0)

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=fig_bg())
    ax.set_facecolor(fig_bg())
    x = np.arange(len(fam_order))
    w = 0.26
    # Raw-output evaluator is binary (inferable / not), so confirmed_raw == any_raw — show one bar.
    metrics = [
        ("any_raw",  "Inferable in raw output\n(binary judge)",       PURPLE, 0.95),
        ("any_ext",  "Any leakage (externalization)",                  ORANGE, 0.85),
        ("conf_ext", "Confirmed leakage (externalization)",            RED,    1.0),
    ]
    for i, (col, label, color, alpha) in enumerate(metrics):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, agg[col], w, label=label,
                      color=color, alpha=alpha, edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, agg[col]):
            if v > 0.015:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.0%}", ha="center", fontsize=7, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("\n"," ") for f in fam_order], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Leakage rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Fig 2 — Leakage Rate by Attribute Family — Raw Output vs Externalization\n(Raw output: binary inferable judge from `output_eval`; Externalization: 3-class judge from `ext_eval` aggregate)", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return save(fig, "leakage_fig2_family_leakage_rate.png")


def fig3_attr_heatmap_by_app(df: pd.DataFrame) -> str:
    """Heatmap: confirmed leakage rate per (attribute, app)."""
    pivot = (df.pivot_table(index="attr", columns="app",
                            values="agg_confirmed", aggfunc="mean") * 100)
    attr_order = (pivot.mean(axis=1).sort_values(ascending=False).index.tolist())
    col_means  = pivot.mean(axis=0).to_dict()
    app_order  = apps_sorted_by_category(pivot.columns.tolist(), col_means, ascending=False)
    pivot = pivot.loc[attr_order, app_order]

    fig, ax = plt.subplots(figsize=(max(11, len(app_order)*1.1), max(8, len(attr_order)*0.45)), facecolor=fig_bg())
    cmap = LinearSegmentedColormap.from_list("leak", [GREEN, "#FFFACC", ORANGE, RED])
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".0f", linewidths=0.4,
                linecolor="#e0e0e0", vmin=0, vmax=100,
                cbar_kws={"label": "Confirmed leakage %", "shrink": 0.8})
    ax.set_title("Fig 3 — Confirmed Leakage Rate (%) per Attribute × App\n(apps grouped by paper category; tick label color = category)", fontsize=12)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    color_ticks_by_category(ax, app_order, axis="x")
    ax.legend(handles=category_legend_handles(), loc="upper left",
              bbox_to_anchor=(1.18, 1.0), fontsize=8, title="App category",
              framealpha=0.9)
    return save(fig, "leakage_fig3_attr_app_heatmap.png")


def fig4_channel_distribution(df: pd.DataFrame) -> str:
    """Grouped bar: confirmed vs any leakage rate per channel, side by side."""
    rows = []
    for ch in CHANNELS:
        col_leak = f"ch_{ch}_leakage"
        col_conf = f"ch_{ch}_confirmed"
        pres_col = f"ch_{ch}_present"
        if col_leak not in df.columns: continue
        sub = df[df[pres_col] == 1]
        if len(sub) == 0: continue
        rows.append({"channel": ch,
                     "Any leakage\n(conf+possible)": sub[col_leak].mean(),
                     "Confirmed leakage": sub[col_conf].mean(),
                     "n": len(sub)})

    cdf = pd.DataFrame(rows).set_index("channel")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=fig_bg())
    ax.set_facecolor(fig_bg())

    x = np.arange(len(cdf))
    w = 0.35
    ch_colors = [CHANNEL_COLORS.get(c, GRAY) for c in cdf.index]
    metrics = [("Any leakage\n(conf+possible)", 0.55), ("Confirmed leakage", 1.0)]
    for i, (metric, alpha) in enumerate(metrics):
        offset = (i - 0.5) * w
        for j, (ch, color) in enumerate(zip(cdf.index, ch_colors)):
            v = cdf.loc[ch, metric]
            bar = ax.bar(j + offset, v, w, color=color, alpha=alpha,
                         edgecolor="white", linewidth=0.8,
                         label=metric if j == 0 else "")
            ax.text(j + offset, v + 0.008, f"{v:.1%}",
                    ha="center", fontsize=8.5, va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cdf.index, fontsize=11)
    ax.set_ylabel("Leakage rate (conditioned on channel present)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Fig 4 — Privacy Leakage Rate by Externalization Channel\n(light = any leakage, dark = confirmed only; conditioned on channel being captured)", fontsize=11)
    legend_patches = [
        mpatches.Patch(color=GRAY, alpha=0.55, label="Any leakage (conf+possible)"),
        mpatches.Patch(color=GRAY, alpha=1.0, label="Confirmed leakage only"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Add n= labels below x-axis (well clear of the channel tick labels above)
    for j, ch in enumerate(cdf.index):
        ax.text(j, -0.14, f"n={cdf.loc[ch,'n']:,}", ha="center", fontsize=7.5,
                transform=ax.get_xaxis_transform(), color="#555")
    fig.subplots_adjust(bottom=0.18)
    return save(fig, "leakage_fig4_channel_distribution.png")


def fig5_channel_heatmap_by_app(df: pd.DataFrame) -> str:
    """Heatmap: per-channel confirmed leakage rate across apps."""
    rows = []
    for ch in CHANNELS:
        col = f"ch_{ch}_confirmed"
        pres = f"ch_{ch}_present"
        if col not in df.columns: continue
        for app, grp in df.groupby("app"):
            sub = grp[grp[pres] == 1]
            if len(sub) == 0: continue
            rows.append({"channel": ch, "app": app, "confirmed": sub[col].mean()*100, "n": len(sub)})

    cdf = pd.DataFrame(rows)
    if cdf.empty:
        return ""
    pivot = cdf.pivot_table(index="channel", columns="app", values="confirmed", aggfunc="mean").fillna(0)
    col_means = pivot.mean(axis=0).to_dict()
    app_order = apps_sorted_by_category(pivot.columns.tolist(), col_means, ascending=False)
    pivot = pivot[app_order]

    fig, ax = plt.subplots(figsize=(max(11, len(app_order)*1.1), 5), facecolor=fig_bg())
    cmap = LinearSegmentedColormap.from_list("ch", ["#FFFFFF", BLUE, RED])
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".0f", linewidths=0.5,
                linecolor="#e0e0e0", vmin=0, vmax=100,
                cbar_kws={"label": "Confirmed leakage %", "shrink": 0.7})
    ax.set_title("Fig 5 — Confirmed Leakage Rate (%) by Channel × App\n(apps grouped by paper category; tick label color = category)", fontsize=12)
    ax.set_xlabel(""); ax.set_ylabel("Channel")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    color_ticks_by_category(ax, app_order, axis="x")
    ax.legend(handles=category_legend_handles(), loc="upper left",
              bbox_to_anchor=(1.20, 1.0), fontsize=8, title="App category",
              framealpha=0.9)
    return save(fig, "leakage_fig5_channel_app_heatmap.png")


def fig6_background_vs_ui_leakage(df: pd.DataFrame) -> str:
    """Side-by-side stacked horizontal bars: confirmed vs any-leakage, background vs foreground per app."""
    def _rates(grp, channel_list, metric):
        suffix = "leakage" if metric == "any" else "confirmed"
        cols = [f"ch_{c}_{suffix}" for c in channel_list if f"ch_{c}_{suffix}" in grp.columns]
        pres = [f"ch_{c}_present" for c in channel_list if f"ch_{c}_present" in grp.columns]
        sub = grp[grp[pres].max(axis=1) == 1] if pres else grp.iloc[0:0]
        return sub[cols].max(axis=1).mean() if cols and len(sub) > 0 else 0

    rows = []
    for app, grp in df.groupby("app"):
        for metric in ("any", "confirmed"):
            rows.append({
                "app": app, "metric": metric,
                "Background\n(STORAGE+LOGGING)": _rates(grp, ["STORAGE","LOGGING"], metric),
                "Foreground\n(UI+NETWORK)":       _rates(grp, ["NETWORK","UI"], metric),
            })

    rdf = pd.DataFrame(rows)
    # Group by category (paper order); sort ascending within category so highest sits on top per group.
    conf_total = (rdf[rdf["metric"]=="confirmed"]
                  .assign(total=lambda d: d["Background\n(STORAGE+LOGGING)"] + d["Foreground\n(UI+NETWORK)"])
                  .set_index("app")["total"].to_dict())
    # ascending=True puts large values at top of each category band on the horizontal bar (matplotlib draws first item at bottom)
    sort_order = list(reversed(apps_sorted_by_category(conf_total.keys(), conf_total, ascending=False)))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(sort_order)*0.55)),
                             facecolor=fig_bg(), sharey=True)
    titles  = {"any": "Any Leakage (conf+possible)", "confirmed": "Confirmed Leakage Only"}
    alphas  = {"any": 0.6, "confirmed": 1.0}
    cols    = ["Background\n(STORAGE+LOGGING)", "Foreground\n(UI+NETWORK)"]
    colors  = [ORANGE, BLUE]

    for ax_idx, metric in enumerate(["any", "confirmed"]):
        ax = axes[ax_idx]; ax.set_facecolor(fig_bg())
        sub = rdf[rdf["metric"]==metric].set_index("app").reindex(sort_order)
        left = np.zeros(len(sub))
        for col, color in zip(cols, colors):
            vals = sub[col].fillna(0).values
            ax.barh(sort_order, vals, left=left,
                    color=color, alpha=alphas[metric], label=col, edgecolor="white", linewidth=0.5)
            for i, (v, l) in enumerate(zip(vals, left)):
                if v > 0.04:
                    ax.text(l + v/2, i, f"{v:.0%}",
                            ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
            left += vals
        ax.set_xlabel("Leakage rate"); ax.set_xlim(0, 1.05)
        ax.set_title(titles[metric], fontsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        color_ticks_by_category(ax, sort_order, axis="y")
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=8)
        if ax_idx == 1:
            ax.legend(handles=category_legend_handles(), loc="lower right",
                      fontsize=7.5, title="App category", framealpha=0.9)

    fig.suptitle("Fig 6 — Background vs Foreground Leakage by App\n(grouped by paper category; tick label color = category)", fontsize=11, y=1.02)
    plt.tight_layout()
    return save(fig, "leakage_fig6_background_vs_ui.png")


def fig7_three_stage_flow(df: pd.DataFrame) -> str:
    """Bar chart: input label presence → output inferred → externalized, by attribute family."""
    has_output = df["output_verdict"].notna() & (df["output_verdict"] != "")
    df2 = df[has_output].copy() if has_output.sum() > 100 else df.copy()

    fam_order = list(ATTR_FAMILIES.keys())
    rows = []
    for fam in fam_order:
        sub = df2[df2["family"] == fam]
        if len(sub) == 0: continue
        inp  = sub["input_label"].mean()
        outp_v = sub["output_verdict"].apply(lambda v: 1 if v in ("confirmed leakage","possible leakage") else 0)
        out  = outp_v.mean()
        ext  = sub["agg_leakage"].mean()
        rows.append({"family": fam.replace("\n"," "), "Input (GT)": inp, "Raw Output": out, "Externalized": ext})

    sdf = pd.DataFrame(rows)
    if len(sdf) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=fig_bg())
    ax.set_facecolor(fig_bg())
    x = np.arange(len(sdf))
    w = 0.26
    stage_colors = {"Input (GT)": BLUE, "Raw Output": ORANGE, "Externalized": RED}
    for i, (stage, color) in enumerate(stage_colors.items()):
        if stage not in sdf.columns: continue
        offset = (i-1)*w
        bars = ax.bar(x + offset, sdf[stage], w, label=stage, color=color, edgecolor="white", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(sdf["family"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of attribute-item pairs"); ax.set_ylim(0, 1.1)
    ax.set_title("Fig 7 — Input → Raw Output → Externalized: Stage-wise Leakage by Family\n(Input=GT label presence; Raw Output=inferred; Externalized=aggregate ext)", fontsize=11)
    ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return save(fig, "leakage_fig7_three_stage_flow.png")


def fig8_attribute_persistence(df: pd.DataFrame) -> str:
    """Scatter: input_rate vs externalized_rate per attribute, colored by family."""
    by_attr = df.groupby(["attr","family"]).agg(
        input_rate=("input_label","mean"),
        ext_rate=("agg_leakage","mean"),
        conf_rate=("agg_confirmed","mean"),
        n=("input_label","count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=fig_bg())
    ax.set_facecolor(fig_bg())
    fam_unique = by_attr["family"].unique()
    fam_color_map = {f: FAMILY_COLORS.get(f, GRAY) for f in fam_unique}

    for _, row in by_attr.iterrows():
        color = fam_color_map[row["family"]]
        ax.scatter(row["input_rate"], row["ext_rate"],
                   s=max(30, row["conf_rate"]*400), color=color, alpha=0.75, edgecolors="white", linewidth=0.7)
        ax.annotate(row["attr"], (row["input_rate"], row["ext_rate"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, alpha=0.85)

    ax.plot([0,1],[0,1], "k--", alpha=0.3, lw=1, label="y=x (no transformation)")
    ax.plot([0,1],[0,0.5], color=GRAY, alpha=0.2, lw=1, ls=":")
    ax.set_xlabel("Input label presence rate (GT)", fontsize=11)
    ax.set_ylabel("Any leakage rate in externalization", fontsize=11)
    ax.set_title("Fig 8 — Attribute Persistence: Input GT Rate vs Externalized Leakage Rate\n(bubble size = confirmed leakage rate; color = attribute family)", fontsize=11)
    patches = [mpatches.Patch(color=c, label=f.replace("\n"," ")) for f, c in FAMILY_COLORS.items() if f in fam_unique]
    ax.legend(handles=patches, fontsize=8, loc="upper left", framealpha=0.85)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return save(fig, "leakage_fig8_attribute_persistence.png")


def fig9_inference_expansion(df: pd.DataFrame) -> str:
    """Bar: per-item, new attributes in output that were NOT in input (inference expansion)."""
    rows = []
    for (app, fn), grp in df.groupby(["app","filename"]):
        in_set  = set(grp[grp["input_label"]==1]["attr"])
        ext_set = set(grp[grp["agg_leakage"]==1]["attr"])
        if len(grp) == 0: continue
        new_attrs = ext_set - in_set
        lost_attrs = in_set - ext_set
        rows.append({
            "app": app, "filename": fn,
            "n_input": len(in_set),
            "n_ext": len(ext_set),
            "n_new": len(new_attrs),
            "n_lost": len(lost_attrs),
            "n_shared": len(in_set & ext_set),
        })

    idf = pd.DataFrame(rows)
    if idf.empty: return ""

    # Per-app mean — group by paper category
    app_agg = idf.groupby("app").agg(
        mean_input=("n_input","mean"),
        mean_ext=("n_ext","mean"),
        mean_new=("n_new","mean"),
        mean_lost=("n_lost","mean"),
    )
    new_by_app = app_agg["mean_new"].to_dict()
    app_order = apps_sorted_by_category(app_agg.index.tolist(), new_by_app, ascending=False)
    app_agg = app_agg.reindex(app_order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=fig_bg())
    for ax in axes: ax.set_facecolor(fig_bg())

    # Left: mean attribute counts
    ax = axes[0]
    x = np.arange(len(app_agg))
    w = 0.3
    b1 = ax.bar(x-w/2, app_agg["mean_input"], w, color=BLUE, label="Input (GT)", edgecolor="white")
    b2 = ax.bar(x+w/2, app_agg["mean_ext"],   w, color=RED,  label="Externalized (any leakage)", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(app_agg.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean # attributes per item")
    ax.set_title("Mean Attribute Count:\nInput GT vs Externalized", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    color_ticks_by_category(ax, app_order, axis="x")

    # Right: new vs lost attributes
    ax = axes[1]
    b3 = ax.bar(x-w/2, app_agg["mean_new"],  w, color=ORANGE, label="New (not in input)", edgecolor="white")
    b4 = ax.bar(x+w/2, app_agg["mean_lost"], w, color=GRAY,   label="Lost (in input, not extern)", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(app_agg.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean # attributes per item")
    ax.set_title("Inference Transformation:\nNew vs Lost Attributes", fontsize=10)
    leg_data = ax.legend(loc="upper left", fontsize=8)
    ax.add_artist(leg_data)
    ax.legend(handles=category_legend_handles(), loc="upper left",
              bbox_to_anchor=(1.0, 1.0), fontsize=7.5, title="App category", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    color_ticks_by_category(ax, app_order, axis="x")

    fig.suptitle("Fig 9 — Inference Expansion: How Attribute Sets Transform from Input to Externalization\n(apps grouped by paper category; tick label color = category)", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "leakage_fig9_inference_expansion.png")


def fig10_leakage_by_dataset(df: pd.DataFrame) -> str:
    """Heatmap: confirmed leakage rate per (dataset, attribute)."""
    pivot = df.pivot_table(index="attr", columns="dataset",
                           values="agg_confirmed", aggfunc="mean").fillna(0) * 100
    attr_order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
    ds_order   = pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[attr_order, ds_order]

    fig, ax = plt.subplots(figsize=(max(8, len(ds_order)*1.4), max(7, len(attr_order)*0.5)), facecolor=fig_bg())
    cmap = LinearSegmentedColormap.from_list("ds", [GREEN, YELLOW, RED])
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".0f", linewidths=0.4,
                linecolor="#e0e0e0", vmin=0, vmax=100,
                cbar_kws={"label": "Confirmed leakage %", "shrink": 0.7})
    ax.set_title("Fig 10 — Confirmed Leakage Rate (%) by Attribute × Dataset", fontsize=12)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    return save(fig, "leakage_fig10_dataset_attr_heatmap.png")


def fig11_modality_comparison(df: pd.DataFrame) -> str:
    """Grouped bar: leakage rate by modality_pair and attribute family."""
    fam_order = list(ATTR_FAMILIES.keys())
    mod_pairs = df["modality_pair"].unique().tolist()

    rows = []
    for mp in mod_pairs:
        sub = df[df["modality_pair"] == mp]
        for fam in fam_order:
            fsub = sub[sub["family"] == fam]
            if len(fsub) == 0: continue
            rows.append({
                "modality_pair": mp, "family": fam.replace("\n"," "),
                "confirmed": fsub["agg_confirmed"].mean(),
                "any_leak":  fsub["agg_leakage"].mean(),
                "n": len(fsub)
            })

    mdf = pd.DataFrame(rows)
    if mdf.empty: return ""

    pairs_sorted = (df.groupby("modality_pair")["agg_confirmed"].mean()
                    .sort_values(ascending=False).index.tolist())
    pair_colors = {p: c for p, c in zip(pairs_sorted, [P_BLUE, P_ORANGE, P_TEAL, P_GREEN, P_MAUVE])}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=fig_bg())
    for ax in axes: ax.set_facecolor(fig_bg())

    families = [f.replace("\n"," ") for f in fam_order if f.replace("\n"," ") in mdf["family"].values]
    x = np.arange(len(families))
    n_pairs = len(pairs_sorted)
    w = 0.8 / n_pairs

    for ax_idx, metric in enumerate(["confirmed", "any_leak"]):
        ax = axes[ax_idx]
        for i, mp in enumerate(pairs_sorted):
            sub = mdf[mdf["modality_pair"]==mp].set_index("family")
            vals = [sub.loc[f, metric] if f in sub.index else 0 for f in families]
            ax.bar(x + (i-(n_pairs-1)/2)*w, vals, w,
                   label=mp, color=pair_colors[mp], edgecolor="white", alpha=0.87)
        ax.set_xticks(x); ax.set_xticklabels(families, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel(f"{'Confirmed' if metric=='confirmed' else 'Any'} leakage rate")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{'Confirmed' if metric=='confirmed' else 'Any (conf+possible)'} Leakage", fontsize=10)
        ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Fig 11 — Leakage Rate by Modality Pair × Attribute Family", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "leakage_fig11_modality_family.png")


def fig12_input_type_comparison(df: pd.DataFrame) -> str:
    """Three-way: docs vs image vs text input types."""
    fam_order = list(ATTR_FAMILIES.keys())
    rows = []
    for it in ["docs","image","text"]:
        sub = df[df["in_type"]==it]
        for fam in fam_order:
            fsub = sub[sub["family"]==fam]
            if len(fsub) == 0: continue
            rows.append({
                "input_type": it,
                "family": fam.replace("\n"," "),
                "confirmed": fsub["agg_confirmed"].mean(),
                "input_rate": fsub["input_label"].mean(),
                "n": len(fsub)
            })

    idf = pd.DataFrame(rows)
    if idf.empty: return ""

    families = [f.replace("\n"," ") for f in fam_order]
    x = np.arange(len(families))
    w = 0.25
    type_colors = {"docs": P_YELLOW, "image": P_BLUE, "text": P_TEAL}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=fig_bg())
    for ax in axes: ax.set_facecolor(fig_bg())

    for ax_idx, metric in enumerate(["input_rate","confirmed"]):
        ax = axes[ax_idx]
        for i, it in enumerate(["docs","image","text"]):
            sub = idf[idf["input_type"]==it].set_index("family")
            vals = [sub.loc[f, metric] if f in sub.index else 0 for f in families]
            ax.bar(x + (i-1)*w, vals, w, label=it.capitalize(),
                   color=type_colors[it], edgecolor="white", alpha=0.87)
        ax.set_xticks(x); ax.set_xticklabels(families, rotation=28, ha="right", fontsize=9)
        ax.set_ylabel(f"{'Input GT presence' if metric=='input_rate' else 'Confirmed leakage'} rate")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{'Input GT Presence' if metric=='input_rate' else 'Confirmed Leakage Rate'}\nby Input Type", fontsize=10)
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Fig 12 — Input Type Comparison (Docs / Image / Text) × Attribute Family\n(Docs = SROIE receipts; Image = HR-VISPR/MIMIC-CXR; Text = text datasets)", fontsize=11, y=1.02)
    plt.tight_layout()
    return save(fig, "leakage_fig12_input_type_comparison.png")


def fig13_semantic_crystallization(df: pd.DataFrame) -> str:
    """Scatter: input_type × attribute — shows image has richer input but text externalizes more."""
    by_attr_type = df.groupby(["attr","in_type","family"]).agg(
        input_rate=("input_label","mean"),
        ext_rate=("agg_leakage","mean"),
        conf_rate=("agg_confirmed","mean"),
        n=("input_label","count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=fig_bg(), sharey=True)
    type_colors = {"docs": P_YELLOW, "image": P_BLUE, "text": P_TEAL}
    for ax in axes: ax.set_facecolor(fig_bg())

    for ax_idx, (it, title) in enumerate([
        ("image","Image Input\n(HR-VISPR/MIMIC-CXR)"),
        ("docs", "Document Input\n(SROIE receipts)"),
        ("text", "Text Input\n(PrivacyLens/SynthPAI/etc.)")
    ]):
        ax = axes[ax_idx]
        sub = by_attr_type[by_attr_type["in_type"]==it]
        if sub.empty:
            ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes)
            continue

        color = type_colors[it]
        for _, row in sub.iterrows():
            fc = FAMILY_COLORS.get(row["family"], GRAY)
            ax.scatter(row["input_rate"], row["ext_rate"],
                       s=max(20, row["conf_rate"]*350),
                       color=fc, alpha=0.75, edgecolors="white", linewidth=0.6)
            if row["n"] > 5:
                ax.annotate(row["attr"], (row["input_rate"], row["ext_rate"]),
                            textcoords="offset points", xytext=(4,2), fontsize=7, alpha=0.8)
        ax.plot([0,1],[0,1], "k--", alpha=0.25, lw=1)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Input GT rate"); ax.set_title(title, fontsize=10)
        if ax_idx == 0: ax.set_ylabel("Externalized leakage rate")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Add diagonal region annotation
        n_above = (sub["ext_rate"] > sub["input_rate"]).sum()
        n_below = (sub["ext_rate"] <= sub["input_rate"]).sum()
        ax.text(0.97, 0.03, f"above y=x: {n_above}/{len(sub)}\n(inference expansion)",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
                color=ORANGE if n_above > n_below else BLUE)

    patches = [mpatches.Patch(color=c, label=f.replace("\n"," ")) for f, c in FAMILY_COLORS.items()]
    fig.legend(handles=patches, fontsize=7.5, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    fig.suptitle("Fig 13 — Semantic Crystallization: Input GT vs Externalized Leakage Rate by Input Type\n(Points above y=x: inference EXPANDS attributes; below: attributes ATTENUATE)", fontsize=11)
    plt.tight_layout()
    return save(fig, "leakage_fig13_semantic_crystallization.png")


def fig14_profile_consolidation(df: pd.DataFrame) -> str:
    """Distribution of simultaneous active (confirmed) attributes per item."""
    per_item = df.groupby(["app","filename"]).agg(
        n_confirmed=("agg_confirmed","sum"),
        n_input=("input_label","sum"),
        app=("app","first"),
        in_type=("in_type","first"),
    ).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=fig_bg())
    for ax in axes: ax.set_facecolor(fig_bg())

    # Left: distribution histogram
    ax = axes[0]
    bins = range(0, int(per_item["n_confirmed"].max())+2)
    ax.hist(per_item["n_confirmed"], bins=bins, color=RED, alpha=0.75, edgecolor="white")
    mean_v = per_item["n_confirmed"].mean()
    ax.axvline(mean_v, color="black", lw=1.5, ls="--", label=f"Mean = {mean_v:.1f}")
    ax.set_xlabel("# Simultaneously confirmed-leaked attributes per item")
    ax.set_ylabel("# Items")
    ax.set_title("Distribution of Co-Occurring\nConfirmed Leakage Attributes", fontsize=10)
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Right: input vs confirmed per app — grouped by paper category
    ax = axes[1]
    app_agg = per_item.groupby("app").agg(
        mean_in=("n_input","mean"),
        mean_conf=("n_confirmed","mean"),
    )
    conf_by_app = app_agg["mean_conf"].to_dict()
    app_order = apps_sorted_by_category(app_agg.index.tolist(), conf_by_app, ascending=False)
    app_agg = app_agg.reindex(app_order)
    x = np.arange(len(app_agg))
    w = 0.35
    b1 = ax.bar(x-w/2, app_agg["mean_in"],   w, color=BLUE,  label="Input GT attrs", edgecolor="white")
    b2 = ax.bar(x+w/2, app_agg["mean_conf"],  w, color=RED,   label="Confirmed leaked", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(app_agg.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean # attributes per item")
    ax.set_title("Mean Attribute Profile Density\n(Input vs Confirmed Leaked)", fontsize=10)
    leg_metric = ax.legend(loc="upper right", fontsize=8)
    ax.add_artist(leg_metric)
    ax.legend(handles=category_legend_handles(), loc="upper left",
              bbox_to_anchor=(1.0, 1.0), fontsize=7.5, title="App category", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    color_ticks_by_category(ax, app_order, axis="x")

    fig.suptitle("Fig 14 — Profile Consolidation: How Many Attributes Are Simultaneously Leaked\n(right panel: apps grouped by paper category)", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "leakage_fig14_profile_consolidation.png")


def fig15_attr_family_channel_heatmap(df: pd.DataFrame) -> str:
    """Heatmap: confirmed leakage rate per (attribute family, channel)."""
    rows = []
    for fam in ATTR_FAMILIES:
        sub = df[df["family"]==fam]
        for ch in CHANNELS:
            col  = f"ch_{ch}_confirmed"
            pres = f"ch_{ch}_present"
            if col not in sub.columns: continue
            s = sub[sub[pres]==1]
            if len(s) == 0: continue
            rows.append({"family": fam.replace("\n"," "), "channel": ch,
                         "confirmed": s[col].mean()*100, "n": len(s)})

    fdf = pd.DataFrame(rows)
    if fdf.empty: return ""
    pivot = fdf.pivot_table(index="family", columns="channel", values="confirmed", aggfunc="mean").fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=fig_bg())
    cmap = LinearSegmentedColormap.from_list("fch", ["#FFFFFF", BLUE, RED])
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".0f", linewidths=0.5,
                linecolor="#e0e0e0", vmin=0, vmax=100,
                cbar_kws={"label": "Confirmed leakage %", "shrink": 0.7})
    ax.set_title("Fig 15 — Confirmed Leakage (%) by Attribute Family × Channel", fontsize=12)
    ax.set_xlabel("Channel"); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    return save(fig, "leakage_fig15_family_channel_heatmap.png")


def fig16_transformation_cases(df: pd.DataFrame) -> str:
    """Case study: 3 apps — input vs externalized heatmaps for same attributes."""
    # Pick 3 interesting apps with sufficient data and different modalities
    candidate_apps = (df.groupby("app")["agg_confirmed"].mean()
                      .sort_values(ascending=False))
    text_apps   = df[df["in_mod"]=="text"]["app"].unique()
    image_apps  = df[(df["in_mod"]=="image") & (df["out_mod"]=="text")]["app"].unique()

    selected = []
    for lst in [image_apps, text_apps]:
        for a in candidate_apps.index:
            if a in lst and a not in selected and len(df[df["app"]==a]) >= 20:
                selected.append(a)
                if len(selected) >= 3:
                    break
        if len(selected) >= 3: break

    if len(selected) < 2:
        return ""

    attrs_to_show = [a for a in ALL_ATTRS
                     if df[(df["app"].isin(selected)) & (df["attr"]==a) & (df["agg_leakage"]==1)].shape[0] > 0]
    if not attrs_to_show:
        attrs_to_show = ALL_ATTRS[:12]

    VERDICT_TO_NUM = {"confirmed leakage": 2, "possible leakage": 1, "no evidence": 0, "na": -1}
    cmap = LinearSegmentedColormap.from_list("v", [GREEN, "#FFFACC", ORANGE, RED])

    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 3.5*len(selected)), facecolor=fig_bg())
    if len(selected) == 1:
        axes = [axes]

    for row_i, app in enumerate(selected):
        sub = df[df["app"]==app]
        ax_in  = axes[row_i][0]
        ax_ext = axes[row_i][1]

        # Input GT heatmap (binary)
        in_vals = sub.groupby("attr")["input_label"].mean().reindex(attrs_to_show).fillna(0)
        in_mat  = in_vals.values.reshape(1, -1) * 2
        im1 = ax_in.imshow(in_mat, aspect="auto", vmin=0, vmax=2, cmap=cmap)
        ax_in.set_xticks(range(len(attrs_to_show)))
        ax_in.set_xticklabels(attrs_to_show, rotation=45, ha="right", fontsize=7.5)
        ax_in.set_yticks([0]); ax_in.set_yticklabels([app], fontsize=9)
        ax_in.set_title(f"{app}: Input (GT)", fontsize=10)

        # Externalized heatmap
        ext_vals = sub.groupby("attr")["agg_score"].mean().reindex(attrs_to_show).fillna(0)
        ext_mat  = ext_vals.values.reshape(1, -1)
        im2 = ax_ext.imshow(ext_mat, aspect="auto", vmin=0, vmax=2, cmap=cmap)
        ax_ext.set_xticks(range(len(attrs_to_show)))
        ax_ext.set_xticklabels(attrs_to_show, rotation=45, ha="right", fontsize=7.5)
        ax_ext.set_yticks([0]); ax_ext.set_yticklabels([app], fontsize=9)
        ax_ext.set_title(f"{app}: Externalized (aggregate)", fontsize=10)

    fig.suptitle("Fig 16 — Privacy Transformation Case Studies: Input GT vs Externalized Attribute Heatmaps\n(0=no evidence; 1=possible; 2=confirmed; average across items)", fontsize=11)
    plt.tight_layout()
    return save(fig, "leakage_fig16_transformation_cases.png")


def fig17_modality_text_vs_image_radar(df: pd.DataFrame) -> str:
    """Radar/spider: comparing text vs image input confirmed leakage across families."""
    fam_order = list(ATTR_FAMILIES.keys())
    fam_labels = [f.replace("\n"," ") for f in fam_order]
    N = len(fam_order)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    type_data = {}
    type_colors2 = {"docs": P_YELLOW, "image": P_BLUE, "text": P_TEAL}
    for it in ["docs","image","text"]:
        sub = df[df["in_type"]==it]
        vals = []
        for fam in fam_order:
            fsub = sub[sub["family"]==fam]
            vals.append(fsub["agg_confirmed"].mean() if len(fsub) > 0 else 0)
        type_data[it] = vals

    fig, ax = plt.subplots(1, 1, figsize=(8, 7), subplot_kw={"projection": "polar"}, facecolor=fig_bg())
    ax.set_facecolor(fig_bg())
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(fam_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=7.5)
    ax.grid(True, alpha=0.4)

    for it, vals in type_data.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=type_colors2[it], linewidth=2, label=it.capitalize())
        ax.fill(angles, v, color=type_colors2[it], alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Fig 17 — Confirmed Leakage Radar:\nDocs vs Image vs Text Input Types", fontsize=12, pad=20)
    return save(fig, "leakage_fig17_modality_radar.png")


def fig18_app_leakage_summary(df: pd.DataFrame) -> str:
    """Per-app: confirmed/any leakage from raw output AND externalization (4 bars per app).

    Raw-output rates use `output_verdict` (raw model output stage); externalization
    rates use the aggregate over channels (`agg_*`). Per-(app, dataset) groups for
    which output_eval was not run yield raw rates of zero for those items, so the
    raw bars only reflect apps/datasets where output_eval is populated.
    """
    rows = []
    for app, sub in df.groupby("app"):
        n = len(sub)
        raw_sub = sub[sub["output_present"] == 1]
        n_raw = len(raw_sub)
        if n_raw > 0:
            ov = raw_sub["output_verdict"].astype(str)
            any_raw = ov.isin(["confirmed leakage","possible leakage"]).mean()
            conf_raw = (ov == "confirmed leakage").mean()
        else:
            any_raw = conf_raw = 0
        any_ext  = sub["agg_leakage"].mean()
        conf_ext = sub["agg_confirmed"].mean()
        rows.append({
            "app": app, "n": n, "n_raw": n_raw,
            "conf_raw": conf_raw, "any_raw": any_raw,
            "conf_ext": conf_ext, "any_ext": any_ext,
        })

    adf = pd.DataFrame(rows)
    if adf.empty:
        return ""
    conf_ext_dict = adf.set_index("app")["conf_ext"].to_dict()
    # ascending=True for horizontal bars: highest at top of each category band
    app_order = list(reversed(apps_sorted_by_category(adf["app"].tolist(), conf_ext_dict, ascending=False)))
    adf = adf.set_index("app").reindex(app_order)

    fig, ax = plt.subplots(figsize=(10, max(5.5, len(adf)*0.55)), facecolor=fig_bg())
    ax.set_facecolor(fig_bg())

    y = np.arange(len(adf))
    h = 0.26
    # Raw-output evaluator is binary (inferable / not), so a single "any_raw" bar is enough.
    metrics = [
        ("any_raw",  "Inferable in raw output (binary judge)",  PURPLE, 0.95),
        ("any_ext",  "Any leakage (externalization)",           ORANGE, 0.85),
        ("conf_ext", "Confirmed leakage (externalization)",     RED,    1.0),
    ]
    for i, (col, label, color, alpha) in enumerate(metrics):
        offset = (i - 1) * h
        vals = adf[col].values
        bars = ax.barh(y + offset, vals, h, color=color, alpha=alpha,
                       edgecolor="white", linewidth=0.5, label=label)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{v:.0%}", va="center", fontsize=7)

    ax.set_yticks(y)
    def _ylabel(a):
        n = int(adf.loc[a, "n"])
        nr = int(adf.loc[a, "n_raw"])
        marker = "" if nr > 0 else "  *"
        return f"{a}  (n={n:,}, raw n={nr:,}){marker}"
    ax.set_yticklabels([_ylabel(a) for a in adf.index], fontsize=9)
    color_ticks_by_category(ax, list(adf.index), axis="y")
    ax.set_xlabel("Leakage rate")
    ax.set_xlim(0, 1.05)
    ax.set_title("Fig 18 — Per-App Leakage Rate: Raw Output vs Externalization\n(solid = raw output stage; light = aggregate externalization;  * = no output_eval data)", fontsize=12)
    leg_metric = ax.legend(loc="lower right", fontsize=7.5, title="Metric / stage")
    ax.add_artist(leg_metric)
    ax.legend(handles=category_legend_handles(), loc="upper right", bbox_to_anchor=(1.02, 1.0),
              fontsize=7.5, title="App category", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return save(fig, "leakage_fig18_app_summary.png")


def fig19_channel_presence_by_app(df: pd.DataFrame) -> str:
    """Stacked bar: which channels are captured per app."""
    rows = []
    for app, grp in df.groupby("app"):
        for ch in CHANNELS:
            pres = f"ch_{ch}_present"
            if pres not in grp.columns: continue
            # Per-item level: fraction of items where this channel is present
            per_item = grp.groupby("filename")[pres].max()
            rows.append({"app": app, "channel": ch, "presence": per_item.mean()})

    cdf = pd.DataFrame(rows)
    if cdf.empty: return ""
    pivot = cdf.pivot_table(index="app", columns="channel", values="presence", aggfunc="mean").fillna(0)
    sum_by_app = pivot.sum(axis=1).to_dict()
    app_order = apps_sorted_by_category(pivot.index.tolist(), sum_by_app, ascending=False)
    pivot = pivot.loc[app_order]

    fig, ax = plt.subplots(figsize=(13, 5.4), facecolor=fig_bg())
    ax.set_facecolor(fig_bg())
    bottom = np.zeros(len(pivot))
    for ch in CHANNELS:
        if ch not in pivot.columns: continue
        vals = pivot[ch].values
        ax.bar(pivot.index, vals, bottom=bottom, label=ch,
               color=CHANNEL_COLORS[ch], edgecolor="white", linewidth=0.5, alpha=0.9)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.05:
                ax.text(i, b + v/2, f"{v:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottom += vals

    ax.set_ylabel("Fraction of items with channel captured")
    ax.set_title("Fig 19 — Channel Capture Rate per App\n(apps grouped by paper category; tick label color = category)", fontsize=11)
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    color_ticks_by_category(ax, list(pivot.index), axis="x")
    leg_ch = ax.legend(title="Channel", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.add_artist(leg_ch)
    ax.legend(handles=category_legend_handles(), bbox_to_anchor=(1.01, 0.45),
              loc="upper left", fontsize=7.5, title="App category", framealpha=0.9)
    ax.set_ylim(0, max(3.5, bottom.max()*1.1))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return save(fig, "leakage_fig19_channel_presence.png")


def fig20_most_consistently_leaked(df: pd.DataFrame) -> str:
    """Horizontal bar: top attributes by confirmed leakage rate + consistency across apps."""
    attr_agg = df.groupby("attr").agg(
        confirmed=("agg_confirmed","mean"),
        n_apps=("app", "nunique"),
        consistency=("agg_confirmed", lambda x: (x > 0).mean()),  # fraction of items with any confirmed
    ).sort_values("confirmed", ascending=True)

    # Color by family
    attr_colors = [FAMILY_COLORS.get(ATTR_TO_FAMILY.get(a, ""), GRAY) for a in attr_agg.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=fig_bg())
    for ax in axes: ax.set_facecolor(fig_bg())

    # Left: confirmed rate
    ax = axes[0]
    bars = ax.barh(attr_agg.index, attr_agg["confirmed"], color=attr_colors, edgecolor="white")
    for bar, v in zip(bars, attr_agg["confirmed"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.1%}",
                va="center", fontsize=8)
    ax.set_xlabel("Confirmed leakage rate"); ax.set_xlim(0, 1.1)
    ax.set_title("Confirmed Leakage Rate\n(all apps/datasets)", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Right: consistency (fraction of items with confirmed)
    ax = axes[1]
    bars2 = ax.barh(attr_agg.index, attr_agg["consistency"], color=attr_colors, edgecolor="white")
    for bar, v in zip(bars2, attr_agg["consistency"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.1%}",
                va="center", fontsize=8)
    ax.set_xlabel("Fraction of items with any confirmed leakage"); ax.set_xlim(0, 1.1)
    ax.set_title("Leakage Consistency\n(% items with ≥1 confirmed)", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color=c, label=f.replace("\n"," ")) for f, c in FAMILY_COLORS.items()]
    fig.legend(handles=patches, ncol=4, loc="lower center", bbox_to_anchor=(0.5,-0.04),
               fontsize=8.5, framealpha=0.9)
    fig.suptitle("Fig 20 — Most Consistently Leaked Attributes (color = attribute family)", fontsize=12)
    plt.tight_layout()
    return save(fig, "leakage_fig20_consistent_attrs.png")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def fig21_category_verdict_overview(df: pd.DataFrame) -> str:
    """Stacked bar: verdict distribution per app category."""
    order = ["confirmed leakage", "possible leakage", "no evidence"]
    cat_order = (df.groupby("category")["agg_confirmed"].mean()
                   .sort_values(ascending=False).index.tolist())

    agg = (df.groupby(["category","agg_verdict"])
             .size().reset_index(name="n"))
    total = agg.groupby("category")["n"].transform("sum")
    agg["pct"] = agg["n"] / total
    agg = agg.set_index(["category","agg_verdict"])["pct"].unstack(fill_value=0)
    for v in order:
        if v not in agg.columns: agg[v] = 0
    agg = agg.reindex(cat_order)[order]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=fig_bg())
    ax.set_facecolor(fig_bg())
    bottom = np.zeros(len(agg))
    for v in order:
        vals = agg[v].values
        bars = ax.bar(agg.index, vals, bottom=bottom,
                      color=VERDICT_COLORS[v], label=v, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0.06:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + val/2,
                        f"{val:.0%}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of attribute-item pairs")
    ax.set_title("Fig 21 — Externalization Verdict Distribution by App Category", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return save(fig, "leakage_fig21_category_verdict.png")


def fig22_category_family_heatmap(df: pd.DataFrame) -> str:
    """Heatmap: confirmed leakage rate per (attribute family, app category)."""
    fam_order = list(ATTR_FAMILIES.keys())
    fam_labels = [f.replace("\n"," ") for f in fam_order]
    cat_order = (df.groupby("category")["agg_confirmed"].mean()
                   .sort_values(ascending=False).index.tolist())

    rows = []
    for cat in cat_order:
        for fam, label in zip(fam_order, fam_labels):
            sub = df[(df["category"]==cat) & (df["family"]==fam)]
            rows.append({"category": cat, "family": label,
                         "confirmed": sub["agg_confirmed"].mean()*100 if len(sub) > 0 else 0,
                         "any_leak":  sub["agg_leakage"].mean()*100 if len(sub) > 0 else 0})

    hdf = pd.DataFrame(rows)
    pivot_c = hdf.pivot_table(index="category", columns="family", values="confirmed").fillna(0)
    pivot_c = pivot_c.reindex(cat_order)[fam_labels]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), facecolor=fig_bg())
    cmap = LinearSegmentedColormap.from_list("cf", [GREEN, "#FFFACC", ORANGE, RED])

    for ax_idx, (pivot, title) in enumerate([
        (pivot_c, "Confirmed Leakage Rate (%)"),
        (hdf.pivot_table(index="category", columns="family", values="any_leak").fillna(0).reindex(cat_order)[fam_labels],
         "Any Leakage Rate (%)"),
    ]):
        ax = axes[ax_idx]
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".0f",
                    linewidths=0.4, linecolor="#e0e0e0", vmin=0, vmax=100,
                    cbar_kws={"label": title, "shrink": 0.7})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle("Fig 22 — Leakage Rate by App Category × Attribute Family", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "leakage_fig22_category_family_heatmap.png")


def fig23_category_channel_breakdown(df: pd.DataFrame) -> str:
    """Grouped bar: per-category confirmed leakage rate broken down by channel."""
    cat_order = (df.groupby("category")["agg_confirmed"].mean()
                   .sort_values(ascending=False).index.tolist())

    rows = []
    for cat in cat_order:
        sub = df[df["category"]==cat]
        row = {"category": cat, "Aggregate": sub["agg_confirmed"].mean()}
        for ch in CHANNELS:
            col  = f"ch_{ch}_confirmed"
            pres = f"ch_{ch}_present"
            if col not in sub.columns: continue
            s = sub[sub[pres]==1]
            row[ch] = s[col].mean() if len(s) > 0 else 0
        rows.append(row)

    cdf = pd.DataFrame(rows).set_index("category")
    channels_present = [c for c in ["Aggregate"] + CHANNELS if c in cdf.columns]
    ch_colors = {"Aggregate": RED, **CHANNEL_COLORS}

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=fig_bg())
    ax.set_facecolor(fig_bg())
    x = np.arange(len(cdf))
    w = 0.75 / len(channels_present)
    for i, ch in enumerate(channels_present):
        if ch not in cdf.columns: continue
        offset = (i - (len(channels_present)-1)/2) * w
        bars = ax.bar(x + offset, cdf[ch].fillna(0), w,
                      label=ch, color=ch_colors.get(ch, GRAY), edgecolor="white", alpha=0.88)
        for bar, v in zip(bars, cdf[ch].fillna(0)):
            if v > 0.03:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                        f"{v:.0%}", ha="center", fontsize=7.5, va="bottom")

    ax.set_xticks(x); ax.set_xticklabels(cdf.index, fontsize=11)
    ax.set_ylabel("Confirmed leakage rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Fig 23 — Confirmed Leakage by App Category × Channel", fontsize=12)
    ax.legend(title="Channel", bbox_to_anchor=(1.01,1), loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return save(fig, "leakage_fig23_category_channel.png")


def fig24_category_modality_matrix(df: pd.DataFrame) -> str:
    """Scatter matrix: app categories colored, x=input_rate, y=ext_confirmed, faceted by attribute family."""
    fam_order = list(ATTR_FAMILIES.keys())[:6]  # top 6 families for readability
    cat_order = sorted(df["category"].unique())

    ncols = 3; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9), facecolor=fig_bg())
    axes = axes.flatten()

    for ax_idx, fam in enumerate(fam_order):
        ax = axes[ax_idx]
        ax.set_facecolor(fig_bg())
        for cat in cat_order:
            sub = df[(df["family"]==fam) & (df["category"]==cat)]
            if len(sub) == 0: continue
            x_val = sub["input_label"].mean()
            y_val = sub["agg_confirmed"].mean()
            color = CATEGORY_COLORS.get(cat, GRAY)
            ax.scatter(x_val, y_val, s=max(40, len(sub)*0.15),
                       color=color, alpha=0.82, edgecolors="white", linewidth=0.7,
                       label=cat, zorder=3)
            ax.annotate(cat[:4], (x_val, y_val),
                        textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=color)

        ax.plot([0,1],[0,1],"k--",alpha=0.2,lw=1)
        ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
        ax.set_title(fam.replace("\n"," "), fontsize=9, color=FAMILY_COLORS.get(fam,"#333"))
        ax.set_xlabel("Input GT rate", fontsize=8)
        ax.set_ylabel("Confirmed ext. rate", fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Legend in last subplot
    ax = axes[-1]
    ax.set_visible(True); ax.set_facecolor(fig_bg())
    ax.axis("off")
    handles = [mpatches.Patch(color=CATEGORY_COLORS.get(c, GRAY), label=c) for c in cat_order]
    ax.legend(handles=handles, title="App Category", fontsize=9, title_fontsize=10,
              loc="center", framealpha=0.9)

    fig.suptitle("Fig 24 — Input GT Rate vs Confirmed Leakage Rate:\nApp Category × Attribute Family", fontsize=12)
    plt.tight_layout()
    return save(fig, "leakage_fig24_category_family_scatter.png")


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    stats["total_rows"] = len(df)
    stats["n_items"]    = df.groupby(["app","filename"]).ngroups
    stats["n_apps"]     = df["app"].nunique()
    stats["n_datasets"] = df["dataset"].nunique()

    # Overall verdict distribution — externalization (post channel projection)
    vc = df["agg_verdict"].value_counts(normalize=True)
    stats["overall_confirmed"] = vc.get("confirmed leakage", 0)
    stats["overall_possible"]  = vc.get("possible leakage", 0)
    stats["overall_no_ev"]     = vc.get("no evidence", 0)
    stats["overall_any_leak"]  = stats["overall_confirmed"] + stats["overall_possible"]

    # Overall verdict distribution — raw model output stage (only over rows where output_eval was actually run)
    raw_sub = df[df["output_present"] == 1] if "output_present" in df.columns else df.iloc[0:0]
    if len(raw_sub) > 0:
        ov = raw_sub["output_verdict"].astype(str)
        vc_raw = ov.value_counts(normalize=True)
        stats["overall_confirmed_raw"] = vc_raw.get("confirmed leakage", 0)
        stats["overall_possible_raw"]  = vc_raw.get("possible leakage", 0)
        stats["overall_any_leak_raw"]  = stats["overall_confirmed_raw"] + stats["overall_possible_raw"]
    else:
        stats["overall_confirmed_raw"] = 0
        stats["overall_possible_raw"]  = 0
        stats["overall_any_leak_raw"]  = 0
    stats["n_rows_with_raw"] = int(len(raw_sub))

    # Per-attribute top confirmed
    attr_conf = df.groupby("attr")["agg_confirmed"].mean().sort_values(ascending=False)
    stats["top_attr"] = attr_conf.head(5).to_dict()
    stats["bot_attr"] = attr_conf.tail(5).to_dict()

    # Per-family leakage
    fam_conf = df.groupby("family").agg(
        confirmed=("agg_confirmed","mean"),
        any_leak=("agg_leakage","mean"),
        input_rate=("input_label","mean"),
    )
    stats["family_stats"] = fam_conf.to_dict()

    # Channel presence
    ch_stats = {}
    for ch in CHANNELS:
        col  = f"ch_{ch}_confirmed"
        pres = f"ch_{ch}_present"
        if col not in df.columns: continue
        sub = df[df[pres]==1]
        ch_stats[ch] = {
            "n_items": len(sub),
            "confirmed": sub[col].mean() if len(sub) > 0 else 0,
            "any_leak": sub[f"ch_{ch}_leakage"].mean() if len(sub) > 0 else 0,
        }
    stats["channel_stats"] = ch_stats

    # Per-app
    app_stats = df.groupby("app").agg(
        n=("agg_confirmed","count"),
        confirmed=("agg_confirmed","mean"),
        any_leak=("agg_leakage","mean"),
    ).sort_values("confirmed", ascending=False)
    stats["app_stats"] = app_stats.to_dict()

    # Modality comparison
    mod_stats = df.groupby("modality_pair").agg(
        n=("agg_confirmed","count"),
        confirmed=("agg_confirmed","mean"),
        any_leak=("agg_leakage","mean"),
    ).sort_values("confirmed", ascending=False)
    stats["modality_stats"] = mod_stats.to_dict()

    # Input type comparison
    intype_stats = df.groupby("in_type").agg(
        n=("agg_confirmed","count"),
        confirmed=("agg_confirmed","mean"),
        any_leak=("agg_leakage","mean"),
        input_rate=("input_label","mean"),
    ).sort_values("confirmed", ascending=False)
    stats["intype_stats"] = intype_stats.to_dict()

    # Inference expansion
    per_item = df.groupby(["app","filename"]).agg(
        n_input=("input_label","sum"),
        n_ext=("agg_leakage","sum"),
        n_confirmed=("agg_confirmed","sum"),
        app=("app","first"),
    ).reset_index(drop=True)
    per_item["n_new"] = (df.groupby(["app","filename"]).apply(
        lambda g: len(set(g[g["agg_leakage"]==1]["attr"]) - set(g[g["input_label"]==1]["attr"]))
    ).values)
    stats["mean_input_attrs"]  = per_item["n_input"].mean()
    stats["mean_ext_attrs"]    = per_item["n_ext"].mean()
    stats["mean_new_attrs"]    = per_item["n_new"].mean()
    stats["pct_expansion"]     = (per_item["n_new"] > 0).mean()

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════════════

REPORT_TEMPLATE = """\
# Inference-Induced Privacy Leakage Landscape

> **Status:** Analysis of prompt4/prompt5 evaluation results — {n_items:,} items across {n_apps} apps, {n_datasets} datasets.
> Generated: {gen_date}.

---

## 1. Experimental Setup

| App | Dataset | Input → Output | Prompt | Verdict Rows |
|-----|---------|----------------|--------|-------------|
{setup_table}

**Evaluation protocol:** Each item is evaluated with prompt4 or prompt5 (3-way leakage verdicts: *confirmed leakage*, *possible leakage*, *no evidence*). For each attribute, a judgment is given for the **aggregate** externalization as well as **per-channel** (NETWORK, STORAGE, UI, LOGGING). Rows with failed or missing `ext_eval` are excluded.

---

## 2. Metric Definitions

| Metric | Definition |
|--------|-----------|
| **Confirmed leakage** | Attribute is clearly and directly inferable from externalized output (score=2) |
| **Possible leakage** | Partial or indirect evidence; attribute may be inferable (score=1) |
| **No evidence** | No meaningful signal for this attribute (score=0) |
| **Any leakage rate** | (confirmed + possible) / total attribute-item pairs |
| **Confirmed leakage rate** | confirmed / total attribute-item pairs |
| **Raw-output stage** | Computed from `output_eval` — judges whether the raw model output makes the attribute *inferable*. The raw-output evaluator is **binary** (inferable / not), so a single "inferable rate" subsumes both confirmed and any-leakage. |
| **Externalization stage** | Computed from `ext_eval` — judges what is actually visible after the channel projection (NETWORK / STORAGE / UI / LOGGING). The aggregate (`agg_*`) judge is **3-class** (confirmed / possible / no evidence), so confirmed and any-leakage are reported separately and the default headline metric throughout. |
| **Inference expansion** | # attributes in externalized set but NOT in input GT set |
| **Background leakage** | Leakage through STORAGE or LOGGING (non-user-visible) |

---

## 3. Headline Results

| Metric | Value |
|--------|-------|
| Total attribute-item pairs evaluated | {total_rows:,} |
| Unique items | {n_items:,} |
| Apps covered | {n_apps} |
| Datasets covered | {n_datasets} |
| **Inferable in raw model output (binary judge)** | **{overall_any_leak_raw:.1%}** |
| **Confirmed leakage — externalization (aggregate, 3-class)** | **{overall_confirmed:.1%}** |
| **Any leakage — externalization (aggregate, 3-class)** | **{overall_any_leak:.1%}** |
| No-evidence rate (externalization) | {overall_no_ev:.1%} |
| Attribute-item pairs with `output_eval` populated | {n_rows_with_raw:,} |
| Mean GT input attributes per item | {mean_input_attrs:.2f} |
| Mean externalized (any-leak) attributes per item | {mean_ext_attrs:.2f} |
| Items with inference expansion (new attrs ≥1) | {pct_expansion:.1%} |

> The two judges report different metrics by design. The raw-output judge (`output_eval`) is **binary** — for each (item, attribute) it returns `inferable: true/false`, so confirmed and any-leakage collapse into a single "inferable" rate. The externalization judge (`ext_eval` aggregate) is **3-class** — it returns `confirmed leakage` / `possible leakage` / `no evidence`, so confirmed and any-leakage are reported separately. Externalization is the default metric throughout the rest of this report unless explicitly noted.

![Fig 1 — Overall Verdict Distribution by App](attachments/leakage_fig1_overall_verdict_by_app.png)

---

## 4. Key Findings

### Finding 1: Identity, Demographic, and Location Attributes Show the Highest Confirmed Leakage

Across all apps and datasets, the **Identity & Identifiability** family (face, identity), **Demographic** (gender, age), and **Location & Spatial** attributes show the highest confirmed leakage rates.

{top_attrs_text}

The top-5 confirmed attributes are: {top_attr_str}. These attributes appear consistently across input types, suggesting they are not merely passed through from input but are actively reconstructed by model inference.

**Implication:** Even when these attributes are not the primary purpose of the AI app, they leak through as inference by-products — a structural risk that cannot be patched by simple output filtering.

![Fig 2 — Leakage Rate by Attribute Family](attachments/leakage_fig2_family_leakage_rate.png)
![Fig 3 — Attribute × App Heatmap](attachments/leakage_fig3_attr_app_heatmap.png)

---

### Finding 2: UI Carries the Most Volume; STORAGE Creates Hidden Background Leakage at Comparable Rates

{channel_text}

![Fig 4 — Channel Distribution](attachments/leakage_fig4_channel_distribution.png)
![Fig 5 — Channel × App Heatmap](attachments/leakage_fig5_channel_app_heatmap.png)
![Fig 6 — Background vs Foreground Leakage](attachments/leakage_fig6_background_vs_ui.png)

---

### Finding 3: Apps Consistently Infer MORE Attributes Than Were Present in Input (Inference Expansion)

{expansion_text}

![Fig 7 — Three-Stage Flow](attachments/leakage_fig7_three_stage_flow.png)
![Fig 8 — Attribute Persistence Scatter](attachments/leakage_fig8_attribute_persistence.png)
![Fig 9 — Inference Expansion per App](attachments/leakage_fig9_inference_expansion.png)

---

### Finding 4: Image Input Activates More Attributes in Input, but Text Externalizes More Consistently

{modality_text}

![Fig 11 — Modality × Family Comparison](attachments/leakage_fig11_modality_family.png)
![Fig 12 — Input Type Comparison](attachments/leakage_fig12_input_type_comparison.png)
![Fig 13 — Semantic Crystallization](attachments/leakage_fig13_semantic_crystallization.png)

---

### Finding 5: Multiple Attributes Leak Simultaneously — Profile Consolidation in AI Output

{consolidation_text}

![Fig 14 — Profile Consolidation](attachments/leakage_fig14_profile_consolidation.png)
![Fig 20 — Consistently Leaked Attributes](attachments/leakage_fig20_consistent_attrs.png)

---

## 5. Cross-Group Synthesis

### RQ1: Where is leakage most prevalent? (by attribute family)
{rq1_text}

### RQ2: Which channels carry the most risk?
{rq2_text}

### RQ3: How does input type shape the leakage landscape?
{rq3_text}

![Fig 17 — Modality Radar Chart](attachments/leakage_fig17_modality_radar.png)
![Fig 18 — Per-App Leakage Summary](attachments/leakage_fig18_app_summary.png)
![Fig 19 — Channel Presence by App](attachments/leakage_fig19_channel_presence.png)

---

## 6. Special Section: Semantic Crystallization, Profile Consolidation, and Visual Re-encoding

This section tests the theoretical claims about how inference transforms attribute spaces across modalities.

### 6.1 Semantic Crystallization (Text→Text)
Text-to-text apps exhibit a phenomenon we term *semantic crystallization*: even when the input text contains only weak or implicit cues for a privacy attribute, the language model's output *consolidates* those cues into explicit, externalized statements. This is evidenced by the high ratio of "confirmed leakage" in text→text runs relative to the input GT label presence rate.

### 6.2 Profile Consolidation (Multi-Attribute Simultaneity)
When an AI system externalizes output, it rarely leaks just one attribute. The analysis shows that items with any confirmed leakage average **{mean_confirmed_per_leaking_item:.1f}** simultaneously confirmed attributes — meaning a single externalization event can expose a multi-dimensional privacy profile.

### 6.3 Visual Re-encoding (Image→Text)
Image-to-text apps translate visual attributes into textual form, which is then externalized to NETWORK and STORAGE channels. The confirmed leakage rate for visual attributes (face, race, age, gender) in image→text apps is substantial, demonstrating that **the text representation inherits and sometimes amplifies the privacy-sensitive content of the original image**.

### 6.4 Attribute Persistence vs. Transformation
Not all attributes persist from input to externalization. Some (e.g., fine-grained visual attributes like nudity, troupe) are present in the GT labels but rarely confirmed in externalized output — these *attenuate*. Others (e.g., identity, location) are externalized at rates **exceeding** their input GT presence — these *amplify* through inference.

### 6.5 Modality Split Analysis (Docs / Image / Text)
{modality_split_text}

![Fig 12 — Input Type 3-Way Comparison](attachments/leakage_fig12_input_type_comparison.png)
![Fig 16 — Transformation Case Studies](attachments/leakage_fig16_transformation_cases.png)

---

## 6b. Leakage by App Category

Apps are grouped into six functional categories: **Finance**, **Photo/Camera**, **Productivity**, **Education**, **Social/Communication**, and **Health/Fitness**.

{category_text}

![Fig 21 — Category Verdict Overview](attachments/leakage_fig21_category_verdict.png)
![Fig 22 — Category × Family Heatmap](attachments/leakage_fig22_category_family_heatmap.png)
![Fig 23 — Category × Channel Breakdown](attachments/leakage_fig23_category_channel.png)
![Fig 24 — Category × Family Scatter](attachments/leakage_fig24_category_family_scatter.png)

---

## 7. Recommendations

1. **Prioritize identity and location filtering** across all app types — these attributes leak at the highest rate regardless of input modality.
2. **Audit STORAGE channels** (databases, memory systems, on-device logs) — STORAGE leakage is non-visible to users and persists beyond session boundaries.
3. **Deploy inference-aware output filters for image→text apps** — the text description of an image is a semantic consolidation point for many visual attributes that were diffuse in the original image.
4. **Test for inference expansion** — apps should be evaluated not just for passthrough of GT-present attributes, but for attributes that appear in the output that were NOT in the input (implicit inference).
5. **Treat multi-attribute leakage as the norm** — privacy defenses that target individual attribute leakage miss the broader threat of simultaneous multi-attribute profile construction.

---

## Appendix

### A. Per-App Detailed Statistics

{app_detail_table}

### B. Per-Dataset Sample Counts

{dataset_table}

### C. Attribute Family → Member Mapping

| Family | Attributes |
|--------|-----------|
{family_table}

### D. Data Quality Notes

- prompt4/5 items with failed `ext_eval` (no verdict) are excluded from all analyses.
- Per-(app, dataset) groups with fewer than 5 verdict rows are excluded; only `tool-neuron|HR-VISPR` (1 item, image→image) and `spendsense|SROIE2019` (2 items) remain in the very-small-sample regime — interpret their per-app rates with caution.
- `output_eval` (raw app output stage) is only available for PrivacyLens text→text runs (deeptutor, waico, llm-vtuber, tool-neuron, xend).
- Channel-level statistics are conditioned on the channel being present in that item's externalization record.
"""


def render_report(df: pd.DataFrame, stats: dict, fig_names: dict) -> str:
    # Setup table
    setup_rows = df.groupby(["app","dataset","modality_pair","eval_prompt"]).size().reset_index(name="n")
    setup_lines = [
        f"| {r.app} | {r.dataset} | {r.modality_pair} | {r.eval_prompt} | {r.n:,} |"
        for _, r in setup_rows.iterrows()
    ]

    # Top attrs
    top_attr_str = ", ".join(f"**{a}** ({v:.1%})" for a,v in list(stats["top_attr"].items())[:5])
    top_attrs_text = ""
    for a, v in list(stats["top_attr"].items())[:5]:
        fam = ATTR_TO_FAMILY.get(a, "")
        top_attrs_text += f"- `{a}` ({fam.replace(chr(10),' ')}): {v:.1%} confirmed leakage rate\n"

    # Channel text
    ch = stats["channel_stats"]
    ch_lines = []
    for c, s in sorted(ch.items(), key=lambda x: -x[1]["confirmed"]):
        ch_lines.append(f"- **{c}**: {s['confirmed']:.1%} confirmed, {s['any_leak']:.1%} any leakage ({s['n_items']:,} attr-item pairs)")
    channel_text = "\n".join(ch_lines)

    # Expansion text
    expansion_text = (
        f"On average, items enter with **{stats['mean_input_attrs']:.1f}** GT-labeled attributes "
        f"but exit with **{stats['mean_ext_attrs']:.1f}** attributes showing any leakage in externalization. "
        f"**{stats['pct_expansion']:.1%}** of items show at least one *new* attribute (present in externalization "
        f"but absent from input GT) — evidence of active inference rather than simple passthrough."
    )

    # Modality text
    mod = stats["modality_stats"]
    mod_lines = []
    for mp, vals in sorted(mod.get("confirmed", {}).items(), key=lambda x: -x[1]):
        n = mod.get("n", {}).get(mp, 0)
        mod_lines.append(f"- **{mp}**: {vals:.1%} confirmed, {mod.get('any_leak',{}).get(mp,0):.1%} any leakage (n={n:,})")
    modality_text = "\n".join(mod_lines)

    # Consolidation text
    per_item_leaking = df[df["agg_confirmed"]==1].groupby(["app","filename"])["agg_confirmed"].sum()
    mean_conf_per = per_item_leaking.mean() if len(per_item_leaking) > 0 else 0
    consolidation_text = (
        f"Among items with at least one confirmed leakage event, "
        f"an average of **{mean_conf_per:.1f}** attributes are simultaneously confirmed. "
        f"This means a single AI interaction can expose a **multi-dimensional privacy profile** "
        f"spanning demographic, location, identity, and appearance attributes at once."
    )

    # RQ texts
    fam_c = stats["family_stats"]["confirmed"]
    top_fam = sorted(fam_c.items(), key=lambda x: -x[1])[:3]
    rq1_text = "; ".join(f"{f.replace(chr(10),' ')}: {v:.1%}" for f,v in top_fam)

    rq2_text = channel_text[:400] + "..."

    it = stats["intype_stats"]
    rq3_text = "; ".join(
        f"**{k}** input: {it['confirmed'][k]:.1%} confirmed (n={it['n'][k]:,})"
        for k in sorted(it.get("confirmed", {}), key=lambda x: -it["confirmed"][x])
    )

    # Modality split text
    it_conf = it.get("confirmed", {})
    it_inp  = it.get("input_rate", {})
    modality_split_text = ""
    for k in ["docs","image","text"]:
        if k in it_conf:
            modality_split_text += (
                f"- **{k.capitalize()} input**: input GT rate = {it_inp.get(k,0):.1%}, "
                f"confirmed leakage = {it_conf[k]:.1%}\n"
            )

    # App detail table
    app_tbl = stats["app_stats"]
    app_lines = []
    for app in sorted(app_tbl["confirmed"], key=lambda x: -app_tbl["confirmed"][x]):
        app_lines.append(
            f"| {app} | {app_tbl['n'][app]:,} | {app_tbl['confirmed'][app]:.1%} | {app_tbl['any_leak'][app]:.1%} |"
        )
    app_detail_table = (
        "| App | N pairs | Confirmed % | Any Leak % |\n"
        "|-----|---------|-------------|------------|\n" +
        "\n".join(app_lines)
    )

    # Dataset table
    ds_tbl = df.groupby(["dataset","in_type"]).agg(n=("agg_confirmed","count"), conf=("agg_confirmed","mean")).reset_index()
    dataset_table = (
        "| Dataset | Input Type | N pairs | Confirmed % |\n"
        "|---------|-----------|---------|-------------|\n" +
        "\n".join(f"| {r.dataset} | {r.in_type} | {r.n:,} | {r.conf:.1%} |" for _, r in ds_tbl.iterrows())
    )

    # Family table
    family_lines = []
    for fam, attrs in ATTR_FAMILIES.items():
        family_lines.append(f"| {fam.replace(chr(10),' ')} | {', '.join(attrs)} |")
    family_table = "\n".join(family_lines)

    # Category text
    cat_stats = df.groupby("category").agg(
        n_items=("filename","nunique"),
        confirmed=("agg_confirmed","mean"),
        any_leak=("agg_leakage","mean"),
    ).sort_values("confirmed", ascending=False)
    cat_lines = []
    for cat, row in cat_stats.iterrows():
        cat_lines.append(
            f"- **{cat}** ({row.n_items:,} items): {row.confirmed:.1%} confirmed, "
            f"{row.any_leak:.1%} any leakage"
        )
    category_text = "\n".join(cat_lines)

    from datetime import datetime
    text = REPORT_TEMPLATE.format(
        gen_date=datetime.now().strftime("%Y-%m-%d"),
        n_items=stats["n_items"],
        n_apps=stats["n_apps"],
        n_datasets=stats["n_datasets"],
        total_rows=stats["total_rows"],
        overall_confirmed=stats["overall_confirmed"],
        overall_any_leak=stats["overall_any_leak"],
        overall_any_leak_raw=stats["overall_any_leak_raw"],
        n_rows_with_raw=stats["n_rows_with_raw"],
        overall_no_ev=stats["overall_no_ev"],
        mean_input_attrs=stats["mean_input_attrs"],
        mean_ext_attrs=stats["mean_ext_attrs"],
        pct_expansion=stats["pct_expansion"],
        top_attr_str=top_attr_str,
        top_attrs_text=top_attrs_text.rstrip(),
        channel_text=channel_text,
        expansion_text=expansion_text,
        modality_text=modality_text,
        consolidation_text=consolidation_text,
        rq1_text=rq1_text,
        rq2_text=rq2_text,
        rq3_text=rq3_text,
        modality_split_text=modality_split_text.rstrip(),
        mean_confirmed_per_leaking_item=mean_conf_per,
        setup_table="\n".join(setup_lines),
        app_detail_table=app_detail_table,
        dataset_table=dataset_table,
        family_table=family_table,
        category_text=category_text,
    )
    return text


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    df = load_all_data()
    # Exclude tiny runs (n<5 verdict rows per app/dataset)
    counts = df.groupby(["app","dataset"])["agg_confirmed"].count()
    valid  = counts[counts >= 5].reset_index()[["app","dataset"]]
    df = df.merge(valid, on=["app","dataset"])
    print(f"  Loaded {len(df):,} attribute-item pairs, {df.groupby(['app','filename']).ngroups:,} unique items")
    print(f"  Apps: {sorted(df['app'].unique())}")

    print("\nGenerating figures...")
    figs = {}
    figs["fig1"]  = fig1_overall_verdict_distribution(df)
    figs["fig2"]  = fig2_verdict_by_attr_family(df)
    figs["fig3"]  = fig3_attr_heatmap_by_app(df)
    figs["fig4"]  = fig4_channel_distribution(df)
    figs["fig5"]  = fig5_channel_heatmap_by_app(df)
    figs["fig6"]  = fig6_background_vs_ui_leakage(df)
    figs["fig7"]  = fig7_three_stage_flow(df)
    figs["fig8"]  = fig8_attribute_persistence(df)
    figs["fig9"]  = fig9_inference_expansion(df)
    figs["fig10"] = fig10_leakage_by_dataset(df)
    figs["fig11"] = fig11_modality_comparison(df)
    figs["fig12"] = fig12_input_type_comparison(df)
    figs["fig13"] = fig13_semantic_crystallization(df)
    figs["fig14"] = fig14_profile_consolidation(df)
    figs["fig15"] = fig15_attr_family_channel_heatmap(df)
    figs["fig16"] = fig16_transformation_cases(df)
    figs["fig17"] = fig17_modality_text_vs_image_radar(df)
    figs["fig18"] = fig18_app_leakage_summary(df)
    figs["fig19"] = fig19_channel_presence_by_app(df)
    figs["fig20"] = fig20_most_consistently_leaked(df)
    figs["fig21"] = fig21_category_verdict_overview(df)
    figs["fig22"] = fig22_category_family_heatmap(df)
    figs["fig23"] = fig23_category_channel_breakdown(df)
    figs["fig24"] = fig24_category_modality_matrix(df)

    print("\nComputing statistics...")
    stats = compute_stats(df)

    print("\nRendering report...")
    md_text = render_report(df, stats, figs)
    out_path = LANTERN_ROOT / "analysis" / "leakage_landscape.md"
    out_path.write_text(md_text)
    print(f"  Report written: {out_path}")

    # Print key numbers
    print(f"\n=== KEY STATS ===")
    print(f"Total attr-item pairs: {stats['total_rows']:,}")
    print(f"Unique items:          {stats['n_items']:,}")
    print(f"Apps:                  {stats['n_apps']}")
    print(f"Confirmed leakage (ext, 3-class):  {stats['overall_confirmed']:.1%}")
    print(f"Any leakage       (ext, 3-class):  {stats['overall_any_leak']:.1%}")
    print(f"Inferable          (raw output, binary): {stats['overall_any_leak_raw']:.1%}")
    print(f"  (raw-output verdict populated for {stats['n_rows_with_raw']:,} of {stats['total_rows']:,} pairs)")
    print(f"Inference expansion:   {stats['pct_expansion']:.1%} of items")
    print(f"Top attrs: {list(stats['top_attr'].keys())[:5]}")
