"""
Headline figures for the paper's Results section.

Each figure is a single 1x1 panel intended to be placed 2-3 per row in a
LaTeX figure float. Outputs land in:

    paper/26_CCS_Lantern/figures/leakage_headline_<slug>.png

with the panel-letter convention (a) through (h) so the prose can refer to
``Fig.~\\ref{fig:leakage-headline}(a)`` etc.

This script is a thin orchestrator on top of the shared loader in
``_leakage_common.py`` so the filter (N >= 100 per (app, dataset) config,
image->image exempt; non-empty ext_eval items only) is identical to the
filters used by ``leakage_analysis.py`` and ``leakage_deep.py``.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorsys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _leakage_common import (
    load_data, ATTR_FAMILIES, ALL_ATTRS, ATTR_TO_FAMILY,
    CHANNELS, CHANNEL_COLORS, FAMILY_COLORS, CATEGORY_COLORS,
    APP_CATEGORY, MOD_PAIR_COLOR, INPUT_TYPE_COLOR,
    P_BLUE, P_ORANGE, P_GREEN, P_TEAL, P_YELLOW, P_MAUVE, P_SLATE,
    S_INPUT, S_OUTPUT, S_EXT, GRAY,
)

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR      = LANTERN_ROOT / "paper" / "26_CCS_Lantern" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Style (FIGURE.md §2: no plot titles, font_scale ~1.3 over the previous baseline) ──
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "axes.labelsize":     12.5,   # was 9.5
    "xtick.labelsize":    12,     # was 9
    "ytick.labelsize":    12,     # was 9
    "legend.fontsize":    11,     # was 8.5
    "font.family":        "DejaVu Sans",
})


def _save(fig, slug: str) -> None:
    out = FIG_DIR / f"leakage_headline_{slug}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out.relative_to(LANTERN_ROOT)}")


def _new_panel(figsize=(4.6, 3.6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    return fig, ax


def _blend_with_white(hex_color: str, white_frac: float) -> str:
    """Lighten a category color while preserving its hue."""
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    r = r * (1 - white_frac) + white_frac
    g = g * (1 - white_frac) + white_frac
    b = b * (1 - white_frac) + white_frac
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def _darken(hex_color: str, lightness_scale: float = 0.72) -> str:
    """Darken a pastel color enough for readable value labels."""
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(hue, max(0, lightness * lightness_scale), saturation)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


# ── (a) Stage gap: raw output inferability vs externalized vs confirmed ──────
def fig_a_stage_gap(df: pd.DataFrame) -> None:
    out_known = df[df["output_has"] == 1]
    rates = {
        "raw output":      out_known["output_leak"].mean() if len(out_known) else np.nan,
        "any-leak ext.":   df["ext_leak"].mean(),
        "confirmed ext.":  df["ext_conf"].mean(),
    }
    fig, ax = _new_panel()
    bars = ax.bar(list(rates), list(rates.values()),
                  color=[S_INPUT, P_ORANGE, S_EXT],
                  edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, rates.values()):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{v:.1%}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#333")
    ax.set_ylim(0, max(rates.values()) * 1.20)
    ax.set_ylabel("Rate over (item, attribute) pairs")
    _save(fig, "a_stage_gap")


# ── (b) Confirmed externalization rate per attribute family ──────────────────
def fig_b_family_rate(df: pd.DataFrame) -> None:
    fam_order = list(ATTR_FAMILIES.keys())
    rates = (df.groupby("family")["ext_conf"].mean()
               .reindex(fam_order).fillna(0).sort_values(ascending=True))
    colors = [FAMILY_COLORS.get(f, GRAY) for f in rates.index]

    fig, ax = _new_panel(figsize=(7.2, 3.6))
    y = np.arange(len(rates))
    ax.barh(y, rates.values, color=colors, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(rates.values):
        ax.text(max(v + 0.003, 0.005), i, f"{v:.1%}",
                va="center", fontsize=11, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace(" & ", "\n& ") for f in rates.index], fontsize=10)
    ax.set_xlim(0, rates.values.max() * 1.35 if rates.values.max() > 0 else 0.1)
    ax.set_xlabel("Confirmed externalization rate")
    _save(fig, "b_family_rate")


# ── (c) Stage gap per app category ───────────────────────────────────────────
def fig_c_category_rate(df: pd.DataFrame) -> None:
    rows = []
    for category, sub in df.groupby("category"):
        out_known = sub[sub["output_has"] == 1]
        rows.append({
            "category": category,
            "raw": out_known["output_leak"].mean() if len(out_known) else 0.0,
            "any": sub["ext_leak"].mean(),
            "conf": sub["ext_conf"].mean(),
            "n": sub["full_key"].nunique(),
        })
    cat_stage = (pd.DataFrame(rows)
                   .sort_values("conf", ascending=False)
                   .reset_index(drop=True))

    fig, ax = _new_panel()
    x = np.arange(len(cat_stage)); w = 0.24
    stages = [
        ("raw", "raw output", 0.50),
        ("any", "any-leak ext.", 0.28),
        ("conf", "confirmed ext.", 0.00),
    ]
    for j, (key, label, white_mix) in enumerate(stages):
        colors = [
            _blend_with_white(CATEGORY_COLORS.get(c, GRAY), white_mix)
            for c in cat_stage["category"]
        ]
        ax.bar(x + (j - 1) * w, cat_stage[key].values, w,
               color=colors, edgecolor="white", linewidth=0.6, label=label)

    y_max = max(cat_stage[["raw", "any", "conf"]].max().max(), 0.05)
    for i, row in cat_stage.iterrows():
        cat_color = CATEGORY_COLORS.get(row["category"], GRAY)
        ax.text(x[i] - w, row["raw"] + 0.018 * y_max,
                f"{row['raw']:.0%}", ha="center", va="bottom",
                fontsize=9, color=_darken(cat_color))
        ax.text(x[i] + w, row["conf"] + 0.018 * y_max,
                f"{row['conf']:.0%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=_darken(cat_color))
    ax.set_xticks(x)
    ax.set_xticklabels(cat_stage["category"], rotation=25, ha="right", fontsize=10.5)
    ax.set_ylim(0, y_max * 1.30)
    ax.set_ylabel("Rate over pairs")
    ax.legend(loc="upper right", framealpha=0.92)
    _save(fig, "c_category_rate")


# ── (d) Channel architecture: any-leak vs confirmed conditioning on capture ──
def fig_d_channel_rate(df: pd.DataFrame) -> None:
    rows = []
    for ch in CHANNELS:
        cap = df[df[f"ch_{ch}_pres"] == 1]
        if len(cap) == 0: continue
        rows.append({
            "ch": ch,
            "any":  cap[f"ch_{ch}_leak"].mean(),
            "conf": cap[f"ch_{ch}_conf"].mean(),
            "n":    cap["full_key"].nunique(),
        })
    sub = pd.DataFrame(rows).sort_values("any", ascending=False).reset_index(drop=True)

    fig, ax = _new_panel(figsize=(7.2, 3.6))
    x = np.arange(len(sub)); w = 0.36
    colors = [CHANNEL_COLORS.get(c, GRAY) for c in sub["ch"]]
    ax.bar(x - w/2, sub["any"],  w, color=colors, alpha=0.50,
           edgecolor="white", linewidth=0.6, label="any-leak")
    ax.bar(x + w/2, sub["conf"], w, color=colors,
           edgecolor="white", linewidth=0.6, label="confirmed")
    for i, v in enumerate(sub["conf"]):
        ax.text(x[i] + w/2, v + 0.002, f"{v:.1%}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["ch"], fontsize=11.5)
    ax.set_ylim(0, max(sub["any"].max(), 0.05) * 1.25)
    ax.set_ylabel("Rate (conditioned on capture)")
    ax.legend(loc="upper right", framealpha=0.92)
    _save(fig, "d_channel_rate")


# ── (e) Modality pair: confirmed externalization rate ────────────────────────
def fig_e_modality_pair(df: pd.DataFrame) -> None:
    order = ["text→text", "image→text", "image→image"]
    sub = []
    for mp in order:
        slc = df[df["modality_pair"] == mp]
        if len(slc) == 0: continue
        sub.append({
            "mp":   mp,
            "any":  slc["ext_leak"].mean(),
            "conf": slc["ext_conf"].mean(),
            "n":    slc["full_key"].nunique(),
        })
    sub = pd.DataFrame(sub)

    fig, ax = _new_panel()
    x = np.arange(len(sub)); w = 0.36
    colors = [MOD_PAIR_COLOR.get(m, GRAY) for m in sub["mp"]]
    ax.bar(x - w/2, sub["any"],  w, color=colors, alpha=0.50,
           edgecolor="white", linewidth=0.6, label="any-leak")
    ax.bar(x + w/2, sub["conf"], w, color=colors,
           edgecolor="white", linewidth=0.6, label="confirmed")
    y_max = max(sub["any"].max(), 0.05)
    for i in range(len(sub)):
        v_any  = sub["any"].iloc[i]
        v_conf = sub["conf"].iloc[i]
        ax.text(x[i], max(v_any, v_conf) + 0.018 * y_max,
                f"{v_conf:.1%}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333")
    # Append n=... to the x-tick label itself instead of as a separate
    # annotation, so it never collides with the tick label.
    labels = [f"{r['mp']}\n($n_{{\\text{{items}}}}{{=}}{r['n']:,}$)"
              for _, r in sub.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, y_max * 1.30)
    ax.set_ylabel("Externalization rate")
    ax.legend(loc="upper left", framealpha=0.92)
    _save(fig, "e_modality_pair")


# ── (f) Profile consolidation: # confirmed attrs per leaking item ────────────
def fig_f_profile_consolidation(df: pd.DataFrame) -> None:
    per_item = df.groupby("full_key")["ext_conf"].sum()
    leaking  = per_item[per_item >= 1]
    if len(leaking) == 0:
        return
    counts, bin_edges = np.histogram(leaking, bins=range(1, max(int(leaking.max()), 2) + 2))
    fig, ax = _new_panel()
    bars = ax.bar(bin_edges[:-1], counts, color=P_BLUE,
                  edgecolor="white", linewidth=0.6)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                str(int(c)), ha="center", va="bottom", fontsize=11, color="#333")
    mean_per = leaking.mean()
    ax.axvline(mean_per, color=S_EXT, linestyle="--", linewidth=1.4,
               label=f"mean = {mean_per:.2f}")
    ax.set_xlabel("# confirmed attrs per leaking item")
    ax.set_ylabel("# items")
    ax.legend(loc="upper right", framealpha=0.92)
    _save(fig, "f_profile_consolidation")


# ── (g) Per-app spread: top + bottom by confirmed rate ───────────────────────
def fig_g_per_app_spread(df: pd.DataFrame) -> None:
    per_app = (df.groupby("app").agg(conf=("ext_conf", "mean"),
                                       any=("ext_leak", "mean"),
                                       n=("full_key", "nunique"))
                  .sort_values("conf", ascending=True))
    colors = [CATEGORY_COLORS.get(APP_CATEGORY.get(a, "Other"), GRAY)
              for a in per_app.index]

    # Wider canvas + extra right padding so the percentage labels live
    # outside the bars and outside the y-axis labels.
    fig, ax = _new_panel(figsize=(14.4, 3.6))
    y = np.arange(len(per_app))
    any_vals  = per_app["any"].values
    conf_vals = per_app["conf"].values
    ax.barh(y, any_vals,  color=colors, alpha=0.45,
            edgecolor="white", linewidth=0.6, label="any-leak")
    ax.barh(y, conf_vals, color=colors,
            edgecolor="white", linewidth=0.6, label="confirmed")
    x_max = max(any_vals.max(), 0.05)
    for i, (a_v, c_v) in enumerate(zip(any_vals, conf_vals)):
        ax.text(a_v + 0.012 * x_max, i, f"{c_v:.1%}",
                va="center", ha="left",
                fontsize=10, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(per_app.index, fontsize=10)
    ax.set_xlim(0, x_max * 1.18)
    ax.set_xlabel("Externalization rate")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    _save(fig, "g_per_app_spread")


# ── (h) Coverage gap with Google Play Data Safety vocabulary ─────────────────
# Class assignment from results.tex §6.6:
#   (a) declarable, slot semantically aligned (5)  — location, identity, race, religion, medical
#   (b) declarable only via "Other" catch-all (5)   — age, gender, marital status, disability, nudity
#   (c) undeclarable (11)                           — face, height, weight, haircolor, color,
#                                                      ethnic_clothing, formal, casual, uniforms,
#                                                      troupe, sports
DATA_SAFETY_CLASS = {
    # (a)
    "location": "a", "identity": "a", "race": "a", "religion": "a", "medical": "a",
    # (b)
    "age": "b", "gender": "b", "marital status": "b", "disability": "b", "nudity": "b",
    # (c)
    "face": "c", "height": "c", "weight": "c", "haircolor": "c", "color": "c",
    "ethnic_clothing": "c", "formal": "c", "casual": "c", "uniforms": "c",
    "troupe": "c", "sports": "c",
}
DS_CLASS_COLOR = {"a": P_GREEN, "b": P_YELLOW, "c": P_ORANGE}
DS_CLASS_LABEL = {"a": "(a) slot exists", "b": "(b) catch-all only", "c": "(c) no slot"}


def fig_h_data_safety_gap(df: pd.DataFrame) -> None:
    df = df.copy()
    df["ds_class"] = df["attr"].map(DATA_SAFETY_CLASS).fillna("c")
    any_evt   = df[df["ext_leak"] == 1]
    conf_evt  = df[df["ext_conf"] == 1]

    def pct(events):
        if len(events) == 0:
            return {"a": 0, "b": 0, "c": 0}
        s = events["ds_class"].value_counts(normalize=True)
        return {k: float(s.get(k, 0)) for k in ("a", "b", "c")}

    a_pct = pct(any_evt); c_pct = pct(conf_evt)

    fig, ax = _new_panel()
    cats = ["any-leak", "confirmed"]
    bottom = [0, 0]
    for cls in ("a", "b", "c"):
        vals = [a_pct[cls], c_pct[cls]]
        ax.bar(cats, vals, bottom=bottom,
               color=DS_CLASS_COLOR[cls], edgecolor="white", linewidth=0.6,
               label=DS_CLASS_LABEL[cls])
        for i, v in enumerate(vals):
            if v > 0.04:
                ax.text(i, bottom[i] + v / 2, f"{v:.0%}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color="#222")
        bottom = [bottom[i] + vals[i] for i in (0, 1)]
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Share of externalization events")
    # Place the legend below the panel so it never overlaps the stacked bars.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, fontsize=10, framealpha=0.92, frameon=False)
    _save(fig, "h_data_safety")


# ── Driver ────────────────────────────────────────────────────────────────────
def main() -> None:
    df, _ = load_data()
    audit = df.attrs.get("filter", {})
    print(f"Headline figures: kept {audit.get('n_configs_kept')} configs / "
          f"{audit.get('n_items_kept')} items "
          f"(modality split: {audit.get('kept_modality')})")
    fig_a_stage_gap(df)
    fig_b_family_rate(df)
    fig_c_category_rate(df)
    fig_d_channel_rate(df)
    fig_e_modality_pair(df)
    fig_f_profile_consolidation(df)
    fig_g_per_app_spread(df)
    fig_h_data_safety_gap(df)


if __name__ == "__main__":
    main()
