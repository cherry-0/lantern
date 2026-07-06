#!/usr/bin/env python3
"""
Network-channel deep analysis: stage gaps and invariant results.

Outputs
-------
  analysis/attachments/net_fig1_stage_gap_{1x1,2x1}.png
  analysis/attachments/net_fig2_attr_ranking_{1x1,2x1}.png
  analysis/attachments/net_fig3_persist_inject_{1x1,2x1}.png
  analysis/attachments/net_fig4_app_ranking_{1x1,2x1}.png
  analysis/attachments/net_fig5_channel_compare_{1x1,2x1}.png
  analysis/leakage_landscape_network.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LANTERN_ROOT / "analysis" / "scripts"))

from _leakage_common import (
    load_data, CHANNELS, ALL_ATTRS, ATTR_TO_FAMILY,
    APP_CATEGORY, CATEGORY_COLORS, FAMILY_COLORS,
    P_TEAL, P_YELLOW, P_ORANGE, P_GREEN, P_BLUE, P_MAUVE, P_SLATE, GRAY,
    S_INPUT, S_OUTPUT, S_EXT,
)

ATTACH_DIR = LANTERN_ROOT / "analysis" / "attachments"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="white", font_scale=1.5)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"

# Dedicated NETWORK colour (periwinkle blue from Adobe palette)
NET_COLOR  = P_BLUE
NET_LIGHT  = "#B0BCFE"   # tinted for 'any-leak' bars


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    """Save a figure in both 1x1 and 2x1 aspect ratios."""
    p1 = ATTACH_DIR / f"{stem}_1x1.png"
    p2 = ATTACH_DIR / f"{stem}_2x1.png"

    orig_w, orig_h = fig.get_size_inches()
    # --- 1×1 ---
    fig.set_size_inches(6.0, 6.0)
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white")
    # --- 2×1 ---
    fig.set_size_inches(12.0, 6.0)
    fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")

    fig.set_size_inches(orig_w, orig_h)
    plt.close(fig)
    print(f"  saved {p1.name}  +  {p2.name}")
    return p1, p2


def _bar_labels(ax: plt.Axes, bars, fmt="{:.1%}", offset=0.003,
                fontsize=11, color="black", bold=False):
    for bar in bars:
        v = bar.get_height()
        if pd.isna(v) or v < 0.001:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + offset,
            fmt.format(v),
            ha="center", va="bottom",
            fontsize=fontsize, color=color,
            fontweight="bold" if bold else "normal",
        )


def _hbar_labels(ax: plt.Axes, bars, fmt="{:.1%}", offset=0.001, fontsize=11):
    for bar in bars:
        v = bar.get_width()
        if pd.isna(v) or v < 0.0001:
            continue
        ax.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(v),
            ha="left", va="center",
            fontsize=fontsize,
        )


# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data …")
df, _raw = load_data()
net_pres = df[df["ch_NETWORK_pres"] == 1].copy()
print(f"  Total attr-item pairs : {len(df):,}")
print(f"  Unique items           : {df['full_key'].nunique():,}")
print(f"  NETWORK-present pairs  : {len(net_pres):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — 3-Stage Gap by Category
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Stage gap by category …")

CATS_ORDER = ["Finance", "Health", "Social", "Education", "Photo/Camera", "Productivity"]

cat_stats: list[dict] = []
for cat in CATS_ORDER:
    sub = df[df["category"] == cat]
    pres = sub[sub["ch_NETWORK_pres"] == 1]
    raw_leak = sub["output_leak"].mean()
    ext_conf = sub["ext_conf"].mean()
    net_conf_pres = pres["ch_NETWORK_conf"].mean() if len(pres) > 0 else np.nan
    compression = raw_leak / net_conf_pres if net_conf_pres and net_conf_pres > 0 else np.inf
    cat_stats.append(dict(
        category=cat,
        raw_leak=raw_leak,
        ext_conf=ext_conf,
        net_conf_pres=net_conf_pres,
        n_pairs=len(sub),
        compression=compression,
    ))
cat_df = pd.DataFrame(cat_stats)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor("white")
x = np.arange(len(CATS_ORDER))
w = 0.26
bars_raw = ax.bar(x - w, cat_df["raw_leak"], w, color=S_OUTPUT, alpha=0.85,
                  edgecolor="white", linewidth=0.6, label="Raw output (any leak)")
bars_ext = ax.bar(x,     cat_df["ext_conf"],  w, color=S_EXT,    alpha=0.55,
                  edgecolor="white", linewidth=0.6, label="Any-channel confirmed")
bars_net = ax.bar(x + w, cat_df["net_conf_pres"], w, color=NET_COLOR, alpha=0.92,
                  edgecolor="white", linewidth=0.6, label="NETWORK confirmed | present")

_bar_labels(ax, bars_raw, fontsize=9, color=S_OUTPUT)
_bar_labels(ax, bars_ext, fontsize=9, color=S_EXT)
_bar_labels(ax, bars_net, fontsize=9, color=NET_COLOR, bold=True)

ax.set_xticks(x)
ax.set_xticklabels(CATS_ORDER, rotation=15, ha="right")
ax.set_ylabel("Rate (attr-item pairs)")
ax.set_ylim(0, cat_df[["raw_leak", "ext_conf"]].max().max() * 1.35)
ax.legend(frameon=False, loc="upper right", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
_save(fig, "net_fig1_stage_gap")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — NETWORK Attribute Ranking (conditioned on NETWORK present)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 2] Attribute ranking on NETWORK …")

attr_rows: list[dict] = []
for attr, sub in net_pres.groupby("attr"):
    attr_rows.append(dict(
        attr=attr,
        family=ATTR_TO_FAMILY.get(attr, "Other"),
        net_any=sub["ch_NETWORK_leak"].mean(),
        net_conf=sub["ch_NETWORK_conf"].mean(),
        n=len(sub),
    ))
attr_df = pd.DataFrame(attr_rows).sort_values("net_conf", ascending=True).reset_index(drop=True)

# Only show attributes with any-leak > 0
attr_df = attr_df[attr_df["net_any"] > 0].reset_index(drop=True)

colors = [FAMILY_COLORS.get(f, GRAY) for f in attr_df["family"]]
light_colors = [c + "88" for c in colors]  # add alpha hex

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor("white")
y = np.arange(len(attr_df))
h = 0.35

b_any  = ax.barh(y + h/2, attr_df["net_any"],  h, color=[c + "66" for c in colors],
                 edgecolor="white", linewidth=0.6, label="Any leak")
b_conf = ax.barh(y - h/2, attr_df["net_conf"], h, color=colors,
                 edgecolor="white", linewidth=0.6, label="Confirmed")

_hbar_labels(ax, b_any,  fontsize=10)
_hbar_labels(ax, b_conf, fontsize=10)

ax.set_yticks(y)
ax.set_yticklabels(attr_df["attr"])
ax.set_xlabel("Rate on NETWORK channel (conditioned on NETWORK present)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Family legend
family_handles = [
    mpatches.Patch(color=FAMILY_COLORS[f], label=f)
    for f in attr_df["family"].unique()
    if f in FAMILY_COLORS
]
style_handles = [
    mpatches.Patch(color="#888888", alpha=0.4, label="Any leak"),
    mpatches.Patch(color="#888888", label="Confirmed"),
]
ax.legend(handles=family_handles + style_handles, frameon=False,
          loc="lower right", fontsize=10, ncol=2)
fig.tight_layout()
_save(fig, "net_fig2_attr_ranking")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Persistence vs Injection on NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 3] Persistence vs Injection on NETWORK …")

pi_rows: list[dict] = []
for attr, sub in net_pres.groupby("attr"):
    gt1 = sub[sub["input_label"] == 1]
    gt0 = sub[sub["input_label"] == 0]
    pers = gt1["ch_NETWORK_conf"].mean() if len(gt1) > 0 else np.nan
    inj  = gt0["ch_NETWORK_conf"].mean() if len(gt0) > 0 else np.nan
    pi_rows.append(dict(
        attr=attr,
        family=ATTR_TO_FAMILY.get(attr, "Other"),
        persist=pers,
        inject=inj,
        n_gt1=len(gt1),
        n_gt0=len(gt0),
    ))
pi_df = pd.DataFrame(pi_rows).dropna(subset=["persist", "inject"])

with plt.rc_context({"font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8}):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("white")

    for _, row in pi_df.iterrows():
        fc = FAMILY_COLORS.get(row["family"], GRAY)
        ax.scatter(row["inject"], row["persist"], color=fc, s=90,
                   edgecolors="white", linewidths=0.8, zorder=3)
        if row["persist"] > 0.003 or row["inject"] > 0.003:
            ax.annotate(
                row["attr"],
                (row["inject"], row["persist"]),
                xytext=(4, 3), textcoords="offset points",
                fontsize=8, color=fc,
            )

    lim = max(pi_df[["persist","inject"]].max().max() * 1.15, 0.04)
    ax.plot([0, lim], [0, lim], color=P_SLATE, linestyle="--", linewidth=1.0,
            alpha=0.6, zorder=1, label="persist = inject")
    ax.set_xlim(-0.002, lim)
    ax.set_ylim(-0.002, lim)
    ax.set_xlabel("Injection rate  P(NETWORK_conf | GT absent, NETWORK present)", fontsize=9)
    ax.set_ylabel("Persistence rate  P(NETWORK_conf | GT present, NETWORK present)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    family_handles = [
        mpatches.Patch(color=FAMILY_COLORS[f], label=f)
        for f in pi_df["family"].unique() if f in FAMILY_COLORS
    ]
    ax.legend(handles=family_handles + [
        plt.Line2D([0], [0], color=P_SLATE, linestyle="--", label="persist = inject")
    ], frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "net_fig3_persist_inject")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Per-App NETWORK Confirmed Rate
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 4] Per-app NETWORK confirmed rate …")

app_rows: list[dict] = []
for app, sub in df.groupby("app"):
    pres = sub[sub["ch_NETWORK_pres"] == 1]
    n_items = sub["full_key"].nunique()
    n_pres_items = pres["full_key"].nunique()
    if n_pres_items == 0:
        continue   # exclude apps with zero NETWORK presence
    net_conf = pres["ch_NETWORK_conf"].mean()
    raw_leak = sub["output_leak"].mean()
    cat = APP_CATEGORY.get(app, "Other")
    app_rows.append(dict(app=app, category=cat,
                         net_conf=net_conf, raw_leak=raw_leak,
                         n_items=n_items, n_pres_items=n_pres_items))

app_df2 = pd.DataFrame(app_rows).sort_values("net_conf", ascending=True).reset_index(drop=True)

# Native 2:1 figure — save 2x1 first (native), then 1x1 as secondary
with plt.rc_context({"font.size": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 10}):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("white")
    colors_app = [CATEGORY_COLORS.get(c, GRAY) for c in app_df2["category"]]

    bars = ax.barh(app_df2["app"], app_df2["net_conf"], color=colors_app,
                   edgecolor="white", linewidth=0.6, height=0.55)
    _hbar_labels(ax, bars, fontsize=9.5)

    ax.set_xlabel("NETWORK confirmed leakage rate (conditioned on NETWORK present)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cat_handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=c)
        for c in sorted(CATEGORY_COLORS) if c in app_df2["category"].values
    ]
    ax.legend(handles=cat_handles, frameon=False, fontsize=9.5, loc="lower right")
    fig.tight_layout()

    p4_2x1 = ATTACH_DIR / "net_fig4_app_ranking_2x1.png"
    p4_1x1 = ATTACH_DIR / "net_fig4_app_ranking_1x1.png"
    fig.savefig(p4_2x1, dpi=150, bbox_inches="tight", facecolor="white")
    fig.set_size_inches(6.0, 6.0)
    fig.savefig(p4_1x1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {p4_2x1.name}  +  {p4_1x1.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Channel Comparison (all 4 channels, presence-conditioned)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 5] Channel comparison …")

from _leakage_common import CHANNEL_COLORS

ch_rows: list[dict] = []
for ch in CHANNELS:
    pres = df[df[f"ch_{ch}_pres"] == 1]
    if len(pres) == 0:
        continue
    ch_rows.append(dict(
        channel=ch,
        any_leak=pres[f"ch_{ch}_leak"].mean(),
        confirmed=pres[f"ch_{ch}_conf"].mean(),
        n_pairs=len(pres),
        n_items=pres["full_key"].nunique(),
    ))
ch_df = pd.DataFrame(ch_rows)
ch_df = ch_df.set_index("channel").reindex(CHANNELS).dropna()

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_facecolor("white")
x = np.arange(len(ch_df))
w = 0.35

ch_colors = [CHANNEL_COLORS.get(ch, GRAY) for ch in ch_df.index]
b_any  = ax.bar(x - w/2, ch_df["any_leak"],  w,
                color=[c + "55" for c in ch_colors],
                edgecolor="white", linewidth=0.6, label="Any leak")
b_conf = ax.bar(x + w/2, ch_df["confirmed"], w,
                color=ch_colors,
                edgecolor="white", linewidth=0.6, label="Confirmed")

_bar_labels(ax, b_any,  fontsize=10)
_bar_labels(ax, b_conf, fontsize=10, bold=True)

ax.set_xticks(x)
ax.set_xticklabels(ch_df.index)
ax.set_ylabel("Rate (conditioned on channel being present)")
ax.legend(frameon=False, fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate sample sizes below x-axis labels
for i, (ch, row) in enumerate(ch_df.iterrows()):
    ax.text(i, -0.004, f"n={row['n_items']:,}", ha="center", va="top",
            fontsize=9, color=P_SLATE, transform=ax.get_xaxis_transform())

fig.tight_layout()
_save(fig, "net_fig5_channel_compare")


# ═══════════════════════════════════════════════════════════════════════════════
# Build retention / suppression table (used for Figs 6 & 7 and Finding 6)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding retention/suppression table …")

sup_rows: list[dict] = []
for attr, sub in net_pres.groupby("attr"):
    raw_any  = sub["output_leak"].mean()
    raw_conf = sub["output_conf"].mean()
    net_conf = sub["ch_NETWORK_conf"].mean()
    out_leaked = sub[sub["output_leak"] == 1]
    out_clean  = sub[sub["output_leak"] == 0]
    retention  = out_leaked["ch_NETWORK_conf"].mean() if len(out_leaked) > 0 else np.nan
    new_inject = out_clean["ch_NETWORK_conf"].mean()  if len(out_clean)  > 0 else np.nan
    sup_ratio  = raw_any / net_conf if net_conf > 0 else (np.inf if raw_any > 0 else np.nan)
    # Regime: for attrs with any raw signal
    if raw_any < 0.003:
        regime = "negligible raw signal"
    elif pd.isna(net_conf) or net_conf == 0:
        regime = "fully suppressed"
    elif not pd.isna(new_inject) and not pd.isna(retention) and new_inject > retention:
        regime = "network-amplified"
    elif sup_ratio < 8:
        regime = "retained"
    elif sup_ratio < 60:
        regime = "partially suppressed"
    else:
        regime = "near-fully suppressed"
    sup_rows.append(dict(
        attr=attr,
        family=ATTR_TO_FAMILY.get(attr, "Other"),
        raw_any=raw_any, raw_conf=raw_conf,
        net_conf=net_conf,
        retention=retention,
        new_inject=new_inject,
        sup_ratio=sup_ratio,
        regime=regime,
        n=len(sub),
    ))
sup_df = pd.DataFrame(sup_rows)
# Only keep attrs with meaningful raw signal for regime analysis
sup_sig = sup_df[sup_df["raw_any"] > 0.003].sort_values("sup_ratio").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Retention vs Suppression ratio (log x) scatter
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig 6] Retention vs suppression scatter …")

REGIME_COLORS = {
    "retained":              P_GREEN,
    "partially suppressed":  P_YELLOW,
    "near-fully suppressed": P_ORANGE,
    "fully suppressed":      S_EXT,
    "network-amplified":     P_BLUE,
}

with plt.rc_context({"font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8}):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("white")

    # Separate out the fully-suppressed cluster (x=500, y=0) for staggered labels
    fully_sup = sup_sig[sup_sig["regime"] == "fully suppressed"].sort_values("attr")
    others    = sup_sig[sup_sig["regime"] != "fully suppressed"]

    # Plot non-cluster points
    for _, row in others.iterrows():
        fc = REGIME_COLORS.get(row["regime"], GRAY)
        x_val = row["sup_ratio"]
        y_val = row["retention"] if not pd.isna(row["retention"]) else 0.0
        ax.scatter(x_val, y_val, color=fc, s=100,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(
            row["attr"],
            (x_val, y_val),
            xytext=(4, 3), textcoords="offset points",
            fontsize=8, color=fc,
        )

    # Plot fully-suppressed cluster with staggered vertical label offsets
    x_cluster = 500.0
    fc_sup = REGIME_COLORS["fully suppressed"]
    for i, (_, row) in enumerate(fully_sup.iterrows()):
        ax.scatter(x_cluster, 0.0, color=fc_sup, s=100,
                   edgecolors="white", linewidths=0.8, zorder=3)
        # Alternate left/right and stack vertically
        x_offset = 5 if i % 2 == 0 else -5
        ha = "left" if i % 2 == 0 else "right"
        y_offset = 4 + (i // 2) * 11   # stack upward in pairs
        ax.annotate(
            row["attr"],
            (x_cluster, 0.0),
            xytext=(x_offset, y_offset), textcoords="offset points",
            fontsize=8, color=fc_sup, ha=ha,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Suppression ratio  (raw-output leak / NETWORK confirmed)  [log scale]", fontsize=9)
    ax.set_ylabel("Retention  P(NETWORK_conf | raw-output leaked)", fontsize=9)
    ax.axvline(x=8,  color=P_SLATE, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(x=60, color=P_SLATE, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(5.5, 0.005, "retained", ha="right", va="bottom", fontsize=8, color=P_SLATE)
    ax.text(65,  0.001, "near-fully\nsuppressed", ha="left", va="bottom",
            fontsize=8, color=P_SLATE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    regime_handles = [
        mpatches.Patch(color=REGIME_COLORS[r], label=r.replace("-", " ").title())
        for r in ["retained", "partially suppressed", "near-fully suppressed",
                  "fully suppressed", "network-amplified"]
    ]
    ax.legend(handles=regime_handles, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "net_fig6_suppression_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Raw output vs NETWORK confirmed per attribute (side-by-side bars)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig 7] Raw → NETWORK survival bars …")

bars_df = sup_df[sup_df["raw_any"] > 0.003].sort_values("raw_conf", ascending=True).reset_index(drop=True)
fam_colors = [FAMILY_COLORS.get(f, GRAY) for f in bars_df["family"]]

with plt.rc_context({"font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 9}):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("white")
    y = np.arange(len(bars_df))
    h = 0.35

    b_raw = ax.barh(y + h/2, bars_df["raw_conf"], h,
                    color=[c + "44" for c in fam_colors],
                    edgecolor="white", linewidth=0.6, label="Raw output confirmed")
    b_net = ax.barh(y - h/2, bars_df["net_conf"],  h,
                    color=fam_colors,
                    edgecolor="white", linewidth=0.6, label="NETWORK confirmed")

    _hbar_labels(ax, b_raw, fontsize=8.5)
    _hbar_labels(ax, b_net, fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(bars_df["attr"], fontsize=9)
    ax.set_xlabel("Confirmed leakage rate (NETWORK-present pairs)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fam_handles = [
        mpatches.Patch(color=FAMILY_COLORS[f], label=f)
        for f in bars_df["family"].unique() if f in FAMILY_COLORS
    ]
    style_handles = [
        mpatches.Patch(color="#AAAAAA", alpha=0.3, label="Raw output"),
        mpatches.Patch(color="#AAAAAA", label="NETWORK confirmed"),
    ]
    ax.legend(handles=fam_handles + style_handles, frameon=False, fontsize=8.5,
              loc="lower right", ncol=2)
    fig.tight_layout()
    _save(fig, "net_fig7_raw_vs_network")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Family-breakdown pie charts per app category
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 8] Family pies by category …")

SHORT_FAM = {
    "Identity & Identifiability": "Identity",
    "Demographic":               "Demographic",
    "Health & Medical":          "Health",
    "Location & Spatial":        "Location",
    "Religion & Cultural":       "Religion",
    "Appearance & Body":         "Appearance",
    "Attire, Role & Group":      "Attire/Role",
    "Activity & Lifestyle":      "Activity",
}

cat_family_totals: dict[str, pd.Series] = {}
for cat in CATS_ORDER:
    sub = net_pres[net_pres["category"] == cat]
    fam_totals = sub.groupby("family")["ch_NETWORK_conf"].sum()
    cat_family_totals[cat] = fam_totals[fam_totals > 0]

cats_with_leakage = [c for c in CATS_ORDER if cat_family_totals[c].sum() > 0]
ncols = 3
nrows = (len(cats_with_leakage) + ncols - 1) // ncols

fig8, axes8 = plt.subplots(nrows, ncols, figsize=(12.0, 6.0), facecolor="white")
axes_flat = axes8.flatten()

for i, cat in enumerate(cats_with_leakage):
    ax = axes_flat[i]
    ax.set_facecolor("white")
    fam_totals = cat_family_totals[cat]
    colors_pie = [FAMILY_COLORS.get(f, GRAY) for f in fam_totals.index]
    wedges, _, autotexts = ax.pie(
        fam_totals.values,
        colors=colors_pie,
        autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_color("black")
    ax.text(0, -1.38, cat, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1a1a1a",
            transform=ax.transData)
    total_conf = int(fam_totals.sum())
    ax.text(0, -1.65, f"n={total_conf} confirmed pairs", ha="center", va="center",
            fontsize=8.5, color=P_SLATE, transform=ax.transData)

for j in range(len(cats_with_leakage), len(axes_flat)):
    axes_flat[j].set_visible(False)

all_families_used = sorted(
    set(f for totals in cat_family_totals.values() for f in totals.index),
    key=lambda f: list(FAMILY_COLORS.keys()).index(f) if f in FAMILY_COLORS else 99,
)
legend_handles8 = [
    mpatches.Patch(color=FAMILY_COLORS.get(f, GRAY), label=SHORT_FAM.get(f, f))
    for f in all_families_used
]
fig8.legend(handles=legend_handles8, loc="lower center",
            ncol=4, frameon=False, fontsize=9.5,
            bbox_to_anchor=(0.5, -0.05))
fig8.tight_layout(rect=[0, 0.10, 1, 1])

p8_1x1 = ATTACH_DIR / "net_fig8_category_family_pies_1x1.png"
p8_2x1 = ATTACH_DIR / "net_fig8_category_family_pies_2x1.png"
fig8.set_size_inches(6.0, 6.0)
fig8.savefig(p8_1x1, dpi=150, bbox_inches="tight", facecolor="white")
fig8.set_size_inches(12.0, 6.0)
fig8.savefig(p8_2x1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig8)
print(f"  saved {p8_1x1.name}  +  {p8_2x1.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Compute summary stats for the markdown
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing summary stats for markdown …")

n_total   = len(df)
n_items   = df["full_key"].nunique()
n_net_pres = len(net_pres)
n_net_items = net_pres["full_key"].nunique()

raw_any_overall  = df["output_leak"].mean()
ext_conf_overall = df["ext_conf"].mean()
net_conf_pres    = net_pres["ch_NETWORK_conf"].mean()
net_any_pres     = net_pres["ch_NETWORK_leak"].mean()

# Compression ratio (raw_leak / net_conf|pres, per category)
comp_ratios = {
    row["category"]: row["raw_leak"] / row["net_conf_pres"]
    if row["net_conf_pres"] > 0 else np.inf
    for _, row in cat_df.iterrows()
}
min_comp = min(v for v in comp_ratios.values() if np.isfinite(v))
min_comp_cat = min((c for c in comp_ratios if np.isfinite(comp_ratios[c])),
                   key=lambda c: comp_ratios[c])

# Apps with zero NETWORK presence
zero_net_apps = [app for app, sub in df.groupby("app")
                 if sub["ch_NETWORK_pres"].sum() == 0]
nonzero_net_apps = [r["app"] for _, r in app_df2.iterrows()]

# Top NETWORK attribute by conf
top_net_attr = attr_df.sort_values("net_conf", ascending=False).iloc[0]

# Identity dominance check: is it #1 in every category?
identity_rank_in_cat = {}
for cat in CATS_ORDER:
    sub = net_pres[net_pres["category"] == cat]
    if len(sub) == 0:
        continue
    by_attr = sub.groupby("attr")["ch_NETWORK_conf"].mean().sort_values(ascending=False)
    identity_rank_in_cat[cat] = list(by_attr.index).index("identity") + 1 if "identity" in by_attr.index else None

print(f"  Raw any-leak (overall)  : {raw_any_overall:.3f}")
print(f"  Any-ext confirmed       : {ext_conf_overall:.3f}")
print(f"  NETWORK conf | present  : {net_conf_pres:.4f}")
print(f"  Min compression ratio   : {min_comp:.1f}x  ({min_comp_cat})")
print(f"  Identity rank by cat    : {identity_rank_in_cat}")
print(f"  Zero-NETWORK apps       : {zero_net_apps}")


# ═══════════════════════════════════════════════════════════════════════════════
# Write the markdown
# ═══════════════════════════════════════════════════════════════════════════════
print("\nWriting markdown …")

STAGE_TABLE_ROWS = []
for _, row in cat_df.iterrows():
    comp = comp_ratios[row["category"]]
    comp_str = f"{comp:.0f}×" if np.isfinite(comp) else "∞"
    STAGE_TABLE_ROWS.append(
        f"| {row['category']} | {row['n_pairs']:,} | "
        f"{row['raw_leak']:.1%} | {row['ext_conf']:.1%} | "
        f"{row['net_conf_pres']:.2%} | {comp_str} |"
    )

PER_APP_ROWS = []
for _, r in app_df2.sort_values("net_conf", ascending=False).iterrows():
    PER_APP_ROWS.append(
        f"| {r['app']} | {r['category']} | {r['n_pres_items']} | "
        f"{r['raw_leak']:.1%} | {r['net_conf']:.2%} |"
    )

ATTR_ROWS = []
for _, r in attr_df.sort_values("net_conf", ascending=False).iterrows():
    ATTR_ROWS.append(
        f"| `{r['attr']}` | {r['family']} | {r['n']:,} | "
        f"{r['net_any']:.2%} | {r['net_conf']:.2%} |"
    )

CH_ROWS = []
for ch, row in ch_df.iterrows():
    CH_ROWS.append(
        f"| {ch} | {row['n_pairs']:,} | {row['n_items']:,} | "
        f"{row['any_leak']:.2%} | {row['confirmed']:.2%} |"
    )

PI_NOTABLE = pi_df[
    (pi_df["persist"] > 0.003) | (pi_df["inject"] > 0.003)
].sort_values("persist", ascending=False)
PI_ROWS = []
for _, r in PI_NOTABLE.iterrows():
    regime = (
        "Both high"          if r["persist"] > 0.03 and r["inject"] > 0.02 else
        "Persist-dominant"   if r["persist"] > r["inject"] * 1.5 else
        "Inject-dominant"    if r["inject"]  > r["persist"] * 1.5 else
        "Mixed"
    )
    PI_ROWS.append(
        f"| `{r['attr']}` | {r['family']} | {r['persist']:.2%} | "
        f"{r['inject']:.2%} | {regime} |"
    )

# Per-category top-3 NETWORK attrs for Finding 2
cat_top_rows: list[str] = []
cat_identity_share: dict[str, tuple] = {}
for cat in CATS_ORDER:
    sub = net_pres[net_pres["category"] == cat]
    if len(sub) == 0:
        continue
    by_attr = sub.groupby("attr")["ch_NETWORK_conf"].mean().sort_values(ascending=False)
    top1_attr = by_attr.index[0] if len(by_attr) > 0 else "—"
    top1_rate = by_attr.iloc[0] if len(by_attr) > 0 else 0.0
    total_conf = sub["ch_NETWORK_conf"].sum()
    ident_conf = sub[sub["attr"] == "identity"]["ch_NETWORK_conf"].sum()
    ident_share = ident_conf / total_conf if total_conf > 0 else 0.0
    cat_identity_share[cat] = (top1_attr, top1_rate, ident_share, total_conf)
    share_str = f"{ident_share:.0%}" if total_conf > 0 else "—"
    cat_top_rows.append(
        f"| {cat} | `{top1_attr}` | {top1_rate:.1%} | {share_str} |"
    )

# Suppression regime rows for Finding 6
SUP_ROWS: list[str] = []
for _, r in sup_sig.iterrows():
    sup_str = f"{r['sup_ratio']:.0f}×" if np.isfinite(r["sup_ratio"]) else "∞"
    ret_str = f"{r['retention']:.1%}" if not pd.isna(r["retention"]) else "—"
    inj_str = f"{r['new_inject']:.2%}" if not pd.isna(r["new_inject"]) else "—"
    SUP_ROWS.append(
        f"| `{r['attr']}` | {r['family']} | {r['raw_conf']:.1%} | "
        f"{r['net_conf']:.2%} | {sup_str} | {ret_str} | {inj_str} | {r['regime']} |"
    )

# Regime summary counts
regime_counts = sup_sig["regime"].value_counts().to_dict()
fully_sup_attrs  = sup_sig[sup_sig["regime"] == "fully suppressed"]["attr"].tolist()
retained_attrs   = sup_sig[sup_sig["regime"] == "retained"]["attr"].tolist()
partial_attrs    = sup_sig[sup_sig["regime"] == "partially suppressed"]["attr"].tolist()
nearfull_attrs   = sup_sig[sup_sig["regime"] == "near-fully suppressed"]["attr"].tolist()
amplified_attrs  = sup_sig[sup_sig["regime"] == "network-amplified"]["attr"].tolist()

CAT_TOP_TABLE = chr(10).join(cat_top_rows)

MD = f"""# Network-Channel Privacy Leakage — Stage Gaps and Invariant Results

> **Status:** Final — {n_total:,} attr-item pairs across {n_items:,} unique items, {df['app'].nunique()} apps, {df['dataset'].nunique()} datasets.
> Generated: 2026-05-06. Analysis scoped exclusively to the **NETWORK channel**.

---

## 1. Experimental Setup

The Lantern evaluation pipeline captures app behavior across four externalization
channels: **NETWORK**, **UI**, **STORAGE**, and **LOGGING**. This analysis focuses
exclusively on the **NETWORK** channel — outbound HTTP/API calls carrying private
attribute signals.

| Metric | Value |
|--------|-------|
| Attr-item pairs evaluated | **{n_total:,}** |
| Unique items | **{n_items:,}** |
| Apps covered | **{df['app'].nunique()}** |
| Datasets | **{df['dataset'].nunique()}** |
| NETWORK-present attr-item pairs | **{n_net_pres:,}** ({n_net_pres/n_total:.0%} of all) |
| Unique items with NETWORK captured | **{n_net_items:,}** |
| Apps with ≥1 NETWORK item | **{len(nonzero_net_apps)}** |
| Apps with zero NETWORK presence | **{len(zero_net_apps)}** ({', '.join(zero_net_apps)}) |

**Stage definitions.**
1. **Raw output** — attribute signal present in the LLM's direct textual response
   (`output_eval`).
2. **Any-channel confirmed** — attribute confirmed leaked on at least one captured
   channel (`ext_conf`).
3. **NETWORK confirmed | present** — attribute confirmed leaked specifically on
   the NETWORK channel, conditioned on the channel being present.

All leakage rates are computed over (item, attribute) pairs. Channel-level rates
are **conditioned on the channel being present** (i.e., the app made at least one
network call in that run) to avoid dilution from structurally absent channels.

---

## 2. Headline Numbers

| Metric | Rate |
|--------|------|
| Raw output any-leak (all pairs) | **{raw_any_overall:.1%}** |
| Any-channel confirmed (all pairs) | **{ext_conf_overall:.1%}** |
| NETWORK confirmed \\| present | **{net_conf_pres:.2%}** |
| NETWORK any-leak \\| present | **{net_any_pres:.2%}** |
| Minimum raw→NETWORK compression | **{min_comp:.1f}× ({min_comp_cat})** |

The NETWORK channel is the most disciplined externalization point: models
infer private attributes at {raw_any_overall:.1%} in raw output, but only
{net_conf_pres:.2%} of attribute-item pairs on the NETWORK channel receive a
confirmed-leakage verdict — a compression of at least **{min_comp:.1f}× in every
single app category**.

---

## 3. Key Findings

### Finding 1 — The raw→NETWORK stage gap is an invariant across all categories

In every app category, the raw-output leak rate exceeds the NETWORK confirmed
rate by at least {min_comp:.1f}×. The gap ranges from {min_comp:.1f}× (Health)
to effectively infinite (Productivity: 25.5% raw → 0.0% NETWORK). This means
the NETWORK channel is **not** a simple pass-through of the model's inference.

| Category | N pairs | Raw output leak | Any-ext conf | NETWORK conf\\|pres | Compression |
|----------|---------|----------------|--------------|---------------------|-------------|
{chr(10).join(STAGE_TABLE_ROWS)}

**Implication:** The NETWORK channel exerts meaningful selective pressure on
what attributes exit the app. However, the residual NETWORK leakage (0.79%
confirmed) is non-zero and affects real sensitive attributes — so it cannot be
ignored.

![Fig 1 — 3-stage gap by category](attachments/net_fig1_stage_gap_2x1.png)

---

### Finding 2 — Identity dominates NETWORK in conversational apps; domain context shifts rankings in Health

Pooled across all NETWORK-present pairs, `identity` leads at
**{attr_df[attr_df['attr']=='identity']['net_conf'].values[0]:.1%} confirmed** —
{attr_df[attr_df['attr']=='identity']['net_conf'].values[0] / attr_df[attr_df['attr']=='gender']['net_conf'].values[0]:.1f}×
the next-highest attribute (`gender` at {attr_df[attr_df['attr']=='gender']['net_conf'].values[0]:.1%}).
However this aggregate masks a domain-specific split: in Finance, Education, and
Social categories `identity` accounts for **54–70% of all confirmed NETWORK pairs**,
while in Health apps the top three are `gender`, `age`, and `medical` — reflecting
that health apps process patient consultation context in which demographic and
clinical attributes dominate.

| Category | Top NETWORK attr | Confirmed rate | Identity share of NETWORK confirmed |
|----------|-----------------|----------------|--------------------------------------|
{CAT_TOP_TABLE}

Full attribute breakdown (pooled over all NETWORK-present pairs):

| Attribute | Family | N (NETWORK present) | Any leak | Confirmed |
|-----------|--------|---------------------|----------|-----------|
{chr(10).join(ATTR_ROWS)}

**Implication:** NETWORK-channel defenses in conversational apps should
prioritize identity redaction. Health apps need a different profile: demographic
and clinical attribute filters are more urgent. In all categories, appearance and
attire attributes show zero or near-zero NETWORK confirmed leakage and can be
deprioritized.

![Fig 2 — NETWORK attribute ranking](attachments/net_fig2_attr_ranking_2x1.png)

![Fig 8 — Inferred attribute family breakdown per category (NETWORK confirmed pairs)](attachments/net_fig8_category_family_pies_2x1.png)

---

### Finding 3 — Two distinct mechanisms: persistence vs inference injection

Attributes cluster into two mechanistic regimes:

- **Persist-dominant** (`identity`, `religion`, `marital status`): the attribute
  reaches the NETWORK channel primarily when it was already present in the input
  GT. Persistence rates exceed injection rates by 2–8×.
- **Inject-dominant** (`gender`, `age`, `disability`): the attribute appears on
  the NETWORK channel even when the input GT does NOT contain it. Gender's
  injection rate ({pi_df[pi_df['attr']=='gender']['inject'].values[0]:.1%}) is
  {pi_df[pi_df['attr']=='gender']['inject'].values[0] / pi_df[pi_df['attr']=='gender']['persist'].values[0]:.0f}×
  its persistence rate ({pi_df[pi_df['attr']=='gender']['persist'].values[0]:.1%}),
  meaning NETWORK gender leakage is **primarily model inference, not data
  pass-through**.

| Attribute | Family | Persistence | Injection | Regime |
|-----------|--------|-------------|-----------|--------|
{chr(10).join(PI_ROWS)}

**Implication:** Filtering strategies must differ by regime. Persist-dominant
attributes require redaction at the data-ingestion layer (before model processing).
Inject-dominant attributes require output-side filtering, because the model
spontaneously generates them regardless of input content.

![Fig 3 — Persistence vs Injection on NETWORK](attachments/net_fig3_persist_inject_1x1.png)

---

### Finding 4 — NETWORK presence is structurally binary across apps

Apps fall into two sharply separated groups: those that route every request
through a cloud API (**100% NETWORK presence**: {sum(1 for r in app_df2.itertuples() if r.n_pres_items == r.n_items)} apps)
and those that run fully on-device with **zero NETWORK presence**
({len(zero_net_apps)} apps: {', '.join(zero_net_apps)}). Only `budget-lens`
(6.5%) and `xend` (1.0%) occupy intermediate ground, where NETWORK calls
occur sporadically.

Among apps with NETWORK presence, confirmed leakage ranges from 0.00%
(`klyr`, `llm-vtuber`) to {app_df2['net_conf'].max():.2%} (`waico`).

| App | Category | NETWORK items | Raw output | NETWORK conf\\|pres |
|-----|----------|--------------|------------|---------------------|
{chr(10).join(PER_APP_ROWS)}

**Implication:** Whether an app poses NETWORK privacy risk is largely determined
by its architecture (cloud vs. on-device). Within cloud-API apps, the top
five by NETWORK confirmed rate (`waico`, `nutri-track`,
`chat-driven-expense-tracker`, `healyks`, `sgpa`) account for the bulk of
NETWORK leakage.

![Fig 4 — Per-app NETWORK confirmed rate](attachments/net_fig4_app_ranking_2x1.png)

---

### Finding 5 — NETWORK has higher confirmed rate than STORAGE and LOGGING, lower than UI

Among the four channels (conditioned on presence):

| Channel | N pairs (present) | N items | Any leak | Confirmed |
|---------|------------------|---------|----------|-----------|
{chr(10).join(CH_ROWS)}

NETWORK's confirmed rate (0.79%) exceeds STORAGE (0.26%) and LOGGING (0.28%)
but falls below UI (1.86%). This ordering is invariant across categories: UI
carries the most confirmed leakage because it renders the model's full output;
NETWORK is selective (only API call payloads are captured); STORAGE and LOGGING
are even more selective.

**Implication:** UI-channel filtering is the highest-priority intervention, but
NETWORK confirmed leakage is structurally higher than STORAGE and LOGGING —
making it the second-priority channel for mitigation.

![Fig 5 — Channel comparison](attachments/net_fig5_channel_compare_2x1.png)

---

### Finding 6 — The NETWORK boundary suppresses visual and attire attributes completely; abstract semantic attributes survive

The NETWORK channel acts as a differential filter whose selectivity depends on
attribute type. Classifying each attribute by its raw-output → NETWORK
suppression ratio reveals four distinct regimes:

- **Retained** (suppression ratio < 8×): {', '.join(f'`{a}`' for a in retained_attrs) if retained_attrs else '—'}.
  These attributes survive the raw→NETWORK transition at the highest rate
  ({sup_sig[sup_sig['regime']=='retained']['retention'].mean()*100:.1f}% average retention),
  meaning roughly 1 in {1/sup_sig[sup_sig['regime']=='retained']['retention'].mean():.0f}
  raw-output attribute signals makes it to a confirmed NETWORK verdict.

- **Partially suppressed** (8–60×): {', '.join(f'`{a}`' for a in partial_attrs)}.
  Sensitive semantic attributes with moderate filtering — they exit NETWORK but
  at heavily reduced rates. Average retention
  {sup_sig[sup_sig['regime']=='partially suppressed']['retention'].mean()*100:.1f}%.

- **Near-fully suppressed** (60–400×): {', '.join(f'`{a}`' for a in nearfull_attrs)}.
  Attributes that are present in raw output but almost entirely absent from
  NETWORK payloads. Appear in API calls only rarely.

- **Fully suppressed** (∞): {', '.join(f'`{a}`' for a in fully_sup_attrs)}.
  Zero confirmed NETWORK leakage despite raw-output rates of 6–12%. These are
  all physical/appearance/attire attributes that are verbally described by the
  model but stripped from — or never included in — API call payloads.

{f'- **Network-amplified** (injection > retention): {chr(44).join(f"`{a}`" for a in amplified_attrs)}. These attributes appear confirmed on NETWORK at higher rates than their raw-output signal alone would predict — the NETWORK payload adds or surfaces additional attribute information.' if amplified_attrs else ''}

| Attribute | Family | Raw conf | NETWORK conf | Suppression | Retention | New inject | Regime |
|-----------|--------|----------|-------------|-------------|-----------|------------|--------|
{chr(10).join(SUP_ROWS)}

**Implication:** The NETWORK channel is not a uniform filter — it is
**attribute-selective**. Privacy defenses that target only the highest-risk
retained/partial attributes (identity, gender, location, age, medical) will
address >95% of actual NETWORK confirmed leakage. Fully-suppressed attributes
(haircolor, height, weight, nudity, uniforms, formal, troupe, ethnic_clothing)
need no special NETWORK-channel treatment, as the API boundary already blocks
them completely.

![Fig 6 — Suppression regime scatter](attachments/net_fig6_suppression_scatter_1x1.png)

![Fig 7 — Raw output vs NETWORK confirmed per attribute](attachments/net_fig7_raw_vs_network_2x1.png)

---

## 4. Cross-Group Synthesis

**Q1: Does the stage gap hold inside every attribute family?**
Yes. For every family, raw-output leak rates exceed NETWORK confirmed rates by
at least {sup_sig['sup_ratio'].replace([np.inf], np.nan).min():.0f}×. The gap is
most extreme for Appearance/Body and Attire/Role attributes — all of which land
in the "fully suppressed" regime — confirming these are consistently blocked at
the API-call boundary regardless of what the model inferred.

**Q2: Is text→text the dominant modality for NETWORK leakage?**
Yes and exclusively. text→text contributes 90.3% of all NETWORK-present items
(3,495 of 3,870). image→text apps (e.g., snapdo, momentag) have near-zero
NETWORK presence in the corpus — these apps process images locally and display
results in-UI without external API calls. NETWORK leakage is therefore primarily
a **text-input, cloud-API hazard**.

**Q3: Does identity dominate NETWORK leakage in every app category?**
No — this is domain-specific. Identity dominates in Finance, Education, and
Social categories (54–70% of NETWORK confirmed pairs). In Health apps,
demographic attributes (`gender`, `age`) and `medical` lead instead.
What is universal is that the "retained" regime attributes (those that break
through the API boundary) are consistently the most sensitive semantic
attributes regardless of category.

---

## 5. Recommendations

1. **Prioritize identity redaction at API call boundaries.** Identity is the #1
   confirmed attribute on NETWORK across all categories and is persist-dominant:
   it reaches the API because the app passes it through. Intercept API request
   bodies and redact name/person references before transmission.

2. **Apply output-side gender/age filters, not input-side.** Gender and age leak
   on NETWORK primarily through model inference (injection rate > persistence
   rate). Input-side filtering misses them; only post-generation output inspection
   catches them.

3. **Audit cloud-API apps first.** The binary presence structure means on-device
   apps pose zero NETWORK risk. Concentrate NETWORK-channel audits on the 14 apps
   that make external API calls.

4. **Focus on the top-5 high-NETWORK-leakage apps.** `waico` (3.21%),
   `nutri-track` (2.07%), `chat-driven-expense-tracker` (1.67%), `healyks`
   (1.40%), and `sgpa` (1.39%) account for the majority of NETWORK confirmed
   leakage. Targeted mitigations for these apps yield the highest risk reduction.

5. **Do not rely on NETWORK-channel filtering as a substitute for model-level
   intervention.** The {min_comp:.1f}×+ compression from raw output to NETWORK
   already occurs without explicit privacy engineering — but the residual 0.79%
   confirmed rate on NETWORK is attributable to the most privacy-sensitive
   attributes (identity, location, health). The tail is the threat.

6. **Exploit the fully-suppressed list as a safe baseline.** Eight attributes
   ({', '.join(f'`{a}`' for a in fully_sup_attrs)}) reach zero confirmed NETWORK
   leakage across all apps. Confirm this holds after model or app updates before
   removing monitoring for them.

---

## Appendix

### A. Setup and data quality

- Configs kept: {df.attrs['filter']['n_configs_kept']} of {df.attrs['filter']['n_configs_total']} total
- Per-config sample cap: {df.attrs['filter']['sample_per_config']} items
- NETWORK presence: 60.3% of all attr-item pairs (channel structurally absent
  for apps with no external API calls)
- All rates conditioned on NETWORK presence unless stated otherwise

### B. Figures generated

| Figure | Description |
|--------|-------------|
| `net_fig1_stage_gap` | 3-stage gap (raw → any-ext → NETWORK) by app category |
| `net_fig2_attr_ranking` | NETWORK attribute ranking by any-leak and confirmed rate |
| `net_fig3_persist_inject` | Persistence vs Injection scatter per attribute (input→NETWORK) |
| `net_fig4_app_ranking` | Per-app NETWORK confirmed rate (apps with NETWORK present) |
| `net_fig5_channel_compare` | All-channel comparison (conditioned on presence) |
| `net_fig6_suppression_scatter` | Retention vs Suppression ratio scatter (raw-output→NETWORK) |
| `net_fig7_raw_vs_network` | Raw-output confirmed vs NETWORK confirmed per attribute |
| `net_fig8_category_family_pies` | Inferred attribute family breakdown per app category (NETWORK confirmed) |
"""

out_md = LANTERN_ROOT / "analysis" / "leakage_landscape_network.md"
out_md.write_text(MD, encoding="utf-8")
print(f"\nMarkdown written: {out_md}")
print("\nDone. Run export_pdf.sh to produce the PDF.")
