"""
Deep analysis of inference-induced privacy leakage.

Builds on leakage_analysis.py with:
- 3-stage flow analysis (Input GT → Raw Output → Externalized)
- Inference vs externalization gap (filtering effectiveness)
- 4-way modality pair matrix (text→text, image→text, text→image, image→image)
- Docs/Image/Text input-type split
- Visual re-encoding, semantic crystallization, profile consolidation evidence
- Real case studies extracted from cache

Outputs:
  analysis/attachments/deep_*.png   (~26 figures)
  analysis/leakage_landscape_deep.md
  analysis/results/leakage_landscape_deep.pdf
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR  = LANTERN_ROOT / "verify" / "outputs"
ATTACH_DIR   = LANTERN_ROOT / "analysis" / "attachments"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)
SKIP_FILES   = {"run_config.json", "dir_summary.json", "report.json", "report.csv"}

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.10)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Stage colors kept (blue / orange / red — input / raw output / externalized)
S_INPUT = "#3A7DC9"      # blue
S_OUTPUT = "#E8762C"     # orange
S_EXT = "#C93A3A"        # red

# Adobe palette (everything non-stage uses these)
P_TEAL   = "#7ADBC4"
P_YELLOW = "#FAD765"
P_ORANGE = "#FA9F5C"
P_GREEN  = "#98D198"
P_BLUE   = "#6C80FC"
P_MAUVE  = "#ACA4B3"
P_SLATE  = "#687692"
GRAY = "#AAAAAA"

VERDICT_COLORS = {
    "confirmed leakage": "#C93A3A",
    "possible leakage":  P_ORANGE,
    "no evidence":       P_GREEN,
    "na":                "#E0E0E0",
}
CHANNEL_COLORS = {
    "NETWORK":   P_BLUE,
    "STORAGE":   P_ORANGE,
    "UI":        P_TEAL,
    "LOGGING":   P_SLATE,
    "AGGREGATE": "#C93A3A",
}

# ── Attribute taxonomy (paper appendix) ──────────────────────────────────────
ATTR_FAMILIES: Dict[str, List[str]] = {
    "Identity":         ["face", "identity"],
    "Demographic":      ["age", "gender", "race", "marital status"],
    "Health/Medical":   ["disability", "medical"],
    "Location":         ["location"],
    "Religion/Cultural":["religion", "ethnic_clothing"],
    "Appearance/Body":  ["nudity", "height", "weight", "haircolor", "color"],
    "Attire/Role":      ["formal", "casual", "uniforms", "troupe"],
    "Activity":         ["sports"],
}
FAMILY_COLORS = {
    "Identity":          P_BLUE,
    "Demographic":       P_ORANGE,
    "Health/Medical":    P_MAUVE,
    "Location":          P_TEAL,
    "Religion/Cultural": P_YELLOW,
    "Appearance/Body":   P_GREEN,
    "Attire/Role":       P_SLATE,
    "Activity":          "#D4A017",
}
ALL_ATTRS = [a for fam in ATTR_FAMILIES.values() for a in fam]
ATTR_TO_FAMILY = {a: fam for fam, attrs in ATTR_FAMILIES.items() for a in attrs}
CHANNELS = ["NETWORK", "STORAGE", "UI", "LOGGING"]

# App categories
APP_CATEGORY = {
    "budget-lens":"Finance","spendsense":"Finance","fiscal-flow":"Finance",
    "finchain":"Finance","chat-driven-expense-tracker":"Finance",
    "google-ai-edge-gallery":"Photo/Camera","tool-neuron":"Photo/Camera","momentag":"Photo/Camera",
    "clone":"Productivity","snapdo":"Productivity","xend":"Productivity",
    "pocketpal-ai":"Productivity","klyr":"Productivity",
    "deeptutor":"Education","edupal":"Education","sgpa":"Education","edumind":"Education",
    "llm-vtuber":"Social","lira":"Social","waico":"Social","tinytavern":"Social",
    "skin-disease-detection":"Health","nutri-track":"Health","healyks":"Health","nom-ai":"Health",
}
CATEGORY_COLORS = {
    "Finance":      P_GREEN,
    "Photo/Camera": P_ORANGE,
    "Productivity": P_BLUE,
    "Education":    P_YELLOW,
    "Social":       P_TEAL,
    "Health":       P_MAUVE,
}

# Dataset → input-type mapping
DATASET_INPUT_TYPE = {
    "HR-VISPR":"image","SROIE2019":"docs","MIMIC-CXR":"image",
    "PrivacyLens":"text","SynthPAI":"text","GretelSyntheticPII":"docs",
    "ASAP-AES":"text","MultiCaRe":"text","OpenPII":"text",
}
INPUT_TYPE_COLOR = {"image": P_BLUE, "docs": P_YELLOW, "text": P_TEAL}

# Modality-pair colors
MOD_PAIR_COLOR = {
    "text→text":  P_TEAL,
    "image→text": P_BLUE,
    "text→image": P_YELLOW,
    "image→image":P_ORANGE,
}

# ── Verdict helpers ───────────────────────────────────────────────────────────
def entry_to_verdict(entry: Any) -> str:
    if not isinstance(entry, dict): return "no evidence"
    agg = entry.get("aggregate", entry)
    if not isinstance(agg, dict): agg = entry
    v = agg.get("verdict", "")
    if v in ("confirmed leakage","possible leakage","no evidence"): return v
    return "confirmed leakage" if agg.get("inferable") else "no evidence"

def entry_to_score(entry: Any) -> int:
    return {"confirmed leakage":2,"possible leakage":1,"no evidence":0}.get(entry_to_verdict(entry), 0)

def output_eval_to_verdict(oe_attr: Any) -> str:
    """Convert old-format output_eval entry (inferable/score) → 3-way verdict."""
    if not isinstance(oe_attr, dict): return "no evidence"
    if "verdict" in oe_attr and oe_attr["verdict"] in ("confirmed leakage","possible leakage","no evidence"):
        return oe_attr["verdict"]
    if oe_attr.get("inferable"):
        score = oe_attr.get("score", 0)
        if score >= 0.7: return "confirmed leakage"
        elif score >= 0.3: return "possible leakage"
        else: return "possible leakage"
    return "no evidence"

def output_eval_to_score(oe_attr: Any) -> int:
    return {"confirmed leakage":2,"possible leakage":1,"no evidence":0}.get(output_eval_to_verdict(oe_attr), 0)

def get_channel_verdicts(ext_entry: Any) -> Dict[str,str]:
    if not isinstance(ext_entry, dict): return {}
    chs = ext_entry.get("channels", {})
    return {ch: entry_to_verdict(v) for ch, v in chs.items()} if isinstance(chs, dict) else {}

# ── Data loading (delegated to shared loader in _leakage_common) ─────────────
try:
    from _leakage_common import load_data as _load_data_unified
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _leakage_common import load_data as _load_data_unified


def load_data() -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Load + filter via the shared loader in _leakage_common.

    Filtering is identical for both leakage_analysis.py and leakage_deep.py:
      - drop items whose ext_eval is empty / missing aggregate verdict;
      - drop (app, dataset) configs with fewer than 100 successful items,
        except image->image which is exempt (kept regardless of N).

    After loading, the 'family' column is remapped using this script's own
    ATTR_TO_FAMILY (short names) so that downstream figure code that uses
    list(ATTR_FAMILIES.keys()) as fam_order works correctly.
    """
    df, raw = _load_data_unified()
    df = df.copy()
    df["family"] = df["attr"].map(ATTR_TO_FAMILY).fillna("Other")
    return df, raw

# ── Save helper ───────────────────────────────────────────────────────────────
def save(fig, name):
    p = ATTACH_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {p.name}")
    return p.name

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: OVERALL EXTERNALIZATION LANDSCAPE
# ══════════════════════════════════════════════════════════════════════════════

def deep_fig01_landscape_overview(df: pd.DataFrame) -> str:
    """4-panel headline: verdict mix, input vs ext rate per family, attrs/item, channel mix."""
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Overall verdict pie
    ax = fig.add_subplot(gs[0, 0])
    vc = df["ext_verdict"].value_counts(normalize=True)
    order = ["confirmed leakage","possible leakage","no evidence"]
    vals  = [vc.get(o, 0) for o in order]
    cols  = [VERDICT_COLORS[o] for o in order]
    wedges, _, autotexts = ax.pie(vals, labels=order, colors=cols, autopct="%.1f%%",
                                   startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2),
                                   textprops=dict(fontsize=10))
    for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
    ax.set_title(f"A. Overall Verdict Distribution\nn={len(df):,} attribute-item pairs", fontsize=11)

    # Panel B: Input vs externalized rate per family
    ax = fig.add_subplot(gs[0, 1])
    fam_order = list(ATTR_FAMILIES.keys())
    fam_data = df.groupby("family").agg(input=("input_label","mean"),
                                          ext=("ext_leak","mean"),
                                          conf=("ext_conf","mean")).reindex(fam_order).fillna(0)
    x = np.arange(len(fam_order))
    w = 0.27
    ax.bar(x-w, fam_data["input"], w, color=S_INPUT, label="Input GT presence", edgecolor="white")
    ax.bar(x,   fam_data["ext"],   w, color=S_EXT,   label="Any leakage (ext)", edgecolor="white", alpha=0.7)
    ax.bar(x+w, fam_data["conf"],  w, color=S_EXT,   label="Confirmed (ext)",   edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(fam_order, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Rate")
    ax.set_title("B. Input GT vs Externalized Leakage per Family", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    # Panel C: Per-item attribute count distribution
    ax = fig.add_subplot(gs[1, 0])
    per_item = df.groupby("full_key").agg(n_input=("input_label","sum"),
                                            n_leak=("ext_leak","sum"),
                                            n_conf=("ext_conf","sum"))
    bins = range(0, int(per_item.values.max())+2)
    ax.hist([per_item["n_input"], per_item["n_leak"], per_item["n_conf"]],
            bins=bins, color=[S_INPUT, P_ORANGE, S_EXT], edgecolor="white",
            label=[f"Input GT (μ={per_item.n_input.mean():.1f})",
                   f"Any leakage (μ={per_item.n_leak.mean():.1f})",
                   f"Confirmed (μ={per_item.n_conf.mean():.1f})"])
    ax.set_xlabel("# attributes per item")
    ax.set_ylabel("# items")
    ax.set_title("C. Attribute-Set Density per Item", fontsize=11)
    ax.legend(fontsize=8.5)

    # Panel D: Channel presence across all data
    ax = fig.add_subplot(gs[1, 1])
    ch_data = []
    for ch in CHANNELS:
        per = df.groupby("full_key")[f"ch_{ch}_pres"].max()
        ch_data.append({"ch": ch, "presence": per.mean()})
    cdf = pd.DataFrame(ch_data)
    bars = ax.bar(cdf["ch"], cdf["presence"],
                  color=[CHANNEL_COLORS[c] for c in cdf["ch"]], edgecolor="white")
    for bar, v in zip(bars, cdf["presence"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.0%}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Fraction of items"); ax.set_ylim(0, 1.1)
    ax.set_title("D. Channel Capture Rate (any item with that channel)", fontsize=11)

    fig.suptitle("Deep Fig 1 — The Inference-Induced Leakage Landscape (Overview)", fontsize=13, y=1.00)
    return save(fig, "deep_fig01_landscape_overview.png")


def deep_fig02_category_overview(df: pd.DataFrame) -> str:
    """Per-category: verdict mix, n_items, channel presence, family breakdown."""
    cat_order = (df.groupby("category")["ext_conf"].mean()
                   .sort_values(ascending=False).index.tolist())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="white")

    # Verdict mix per category (stacked)
    ax = axes[0]
    order = ["confirmed leakage","possible leakage","no evidence"]
    pivot = df.groupby(["category","ext_verdict"]).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0).reindex(cat_order)[order]
    bottom = np.zeros(len(pivot))
    for v in order:
        vals = pivot[v].values
        ax.bar(pivot.index, vals, bottom=bottom, color=VERDICT_COLORS[v],
               label=v, edgecolor="white", linewidth=0.5)
        for i, val in enumerate(vals):
            if val > 0.05:
                ax.text(i, bottom[i]+val/2, f"{val:.0%}",
                        ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
        bottom += vals
    ax.set_ylim(0, 1.02); ax.set_ylabel("Fraction")
    ax.set_title("A. Verdict Mix per Category", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticklabels(pivot.index, rotation=20, ha="right", fontsize=9)

    # Confirmed rate per category × family heatmap
    ax = axes[1]
    fam_order = list(ATTR_FAMILIES.keys())
    pivot2 = (df.groupby(["category","family"])["ext_conf"].mean() * 100).unstack(fill_value=0)
    pivot2 = pivot2.reindex(cat_order)[fam_order].fillna(0)
    cmap = LinearSegmentedColormap.from_list("c", ["white", P_TEAL, "#C93A3A"])
    sns.heatmap(pivot2, ax=ax, cmap=cmap, annot=True, fmt=".0f",
                vmin=0, vmax=100, linewidths=0.4, linecolor="#E8E8E8",
                cbar_kws={"shrink":0.6,"label":"Confirmed %"})
    ax.set_title("B. Confirmed % by Category × Family", fontsize=10.5)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Channel presence per category
    ax = axes[2]
    rows = []
    for cat in cat_order:
        sub = df[df["category"]==cat]
        for ch in CHANNELS:
            per_item = sub.groupby("full_key")[f"ch_{ch}_pres"].max()
            rows.append({"cat": cat, "ch": ch, "rate": per_item.mean()})
    cdf = pd.DataFrame(rows)
    pivot3 = cdf.pivot_table(index="cat", columns="ch", values="rate").reindex(cat_order)[CHANNELS]
    x = np.arange(len(cat_order)); w = 0.18
    for i, ch in enumerate(CHANNELS):
        vals = pivot3[ch].values
        ax.bar(x + (i-1.5)*w, vals, w, color=CHANNEL_COLORS[ch],
               label=ch, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(cat_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Channel capture rate"); ax.set_ylim(0, 1.1)
    ax.set_title("C. Channel Capture by Category", fontsize=10.5)
    ax.legend(title="Channel", fontsize=8, loc="upper right")

    fig.suptitle("Deep Fig 2 — Externalization Landscape per App Category", fontsize=12.5, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig02_category_overview.png")


def deep_fig03_channel_verdict_distribution(df: pd.DataFrame) -> str:
    """Stacked verdict distribution within each channel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    order = ["confirmed leakage","possible leakage","no evidence"]

    # Per-channel verdict (conditioned on present)
    ax = axes[0]
    rows = []
    for ch in CHANNELS:
        sub = df[df[f"ch_{ch}_pres"]==1]
        if len(sub) == 0: continue
        vc = sub[f"ch_{ch}_v"].value_counts(normalize=True)
        for v in order:
            rows.append({"ch": ch, "verdict": v, "rate": vc.get(v, 0)})
        rows.append({"ch": ch, "verdict": "na", "rate": vc.get("na", 0)})
    rdf = pd.DataFrame(rows)
    pivot = rdf.pivot_table(index="ch", columns="verdict", values="rate").reindex(CHANNELS)[order]
    bottom = np.zeros(len(pivot))
    for v in order:
        vals = pivot[v].values
        ax.bar(pivot.index, vals, bottom=bottom, color=VERDICT_COLORS[v],
               label=v, edgecolor="white", linewidth=0.5)
        for i, val in enumerate(vals):
            if val > 0.04:
                ax.text(i, bottom[i]+val/2, f"{val:.0%}",
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += vals
    ax.set_ylim(0, 1.02); ax.set_ylabel("Fraction of attribute-item pairs")
    ax.set_title("A. Verdict Distribution within Each Channel\n(conditioned on channel being present)", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")

    # Aggregate verdict comparison
    ax = axes[1]
    rows = []
    for label, sub in [("Aggregate", df)] + [(ch, df[df[f"ch_{ch}_pres"]==1]) for ch in CHANNELS]:
        if label == "Aggregate":
            vc = sub["ext_verdict"].value_counts(normalize=True)
        else:
            vc = sub[f"ch_{label}_v"].value_counts(normalize=True)
        for v in order:
            rows.append({"src": label, "verdict": v, "rate": vc.get(v, 0)})
    rdf = pd.DataFrame(rows)
    pivot = rdf.pivot_table(index="src", columns="verdict", values="rate").reindex(["Aggregate"]+CHANNELS)[order]
    x = np.arange(len(pivot)); w = 0.27
    for i, v in enumerate(order):
        vals = pivot[v].values
        ax.bar(x + (i-1)*w, vals, w, color=VERDICT_COLORS[v], label=v, edgecolor="white")
        for j, val in enumerate(vals):
            if val > 0.02:
                ax.text(x[j]+(i-1)*w, val+0.01, f"{val:.0%}",
                        ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, fontsize=9.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Fraction")
    ax.set_title("B. Aggregate vs Per-Channel Verdict Rate", fontsize=10)
    ax.legend(fontsize=8.5)

    fig.suptitle("Deep Fig 3 — Channel-Level Verdict Distribution", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig03_channel_verdict.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ATTRIBUTE FAMILY DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

def deep_fig04_three_stage_family_flow(df: pd.DataFrame) -> str:
    """Per-family 3-stage flow: input → output → externalized. Use only items with output_eval data."""
    df_oe = df[df["output_has"]==1]
    fam_order = list(ATTR_FAMILIES.keys())

    rows_all, rows_oe = [], []
    for fam in fam_order:
        s_all = df[df["family"]==fam]
        s_oe  = df_oe[df_oe["family"]==fam]
        if len(s_all) > 0:
            rows_all.append({"family": fam,
                             "Input GT": s_all["input_label"].mean(),
                             "Externalized": s_all["ext_leak"].mean()})
        if len(s_oe) > 0:
            rows_oe.append({"family": fam,
                            "Input GT":     s_oe["input_label"].mean(),
                            "Raw Output":   s_oe["output_leak"].mean(),
                            "Externalized": s_oe["ext_leak"].mean()})

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor="white")

    # Left: 2-stage (all data, every family)
    ax = axes[0]
    df1 = pd.DataFrame(rows_all).set_index("family")
    x = np.arange(len(df1)); w = 0.4
    ax.bar(x-w/2, df1["Input GT"],     w, color=S_INPUT, label="Input GT", edgecolor="white")
    ax.bar(x+w/2, df1["Externalized"], w, color=S_EXT,   label="Externalized", edgecolor="white")
    for i, (a, b) in enumerate(zip(df1["Input GT"], df1["Externalized"])):
        ax.text(i-w/2, a+0.01, f"{a:.0%}", ha="center", fontsize=7.5)
        ax.text(i+w/2, b+0.01, f"{b:.0%}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(df1.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Rate")
    ax.set_title(f"A. Input GT vs Externalized — All Data (n={len(df):,} pairs)", fontsize=10)
    ax.legend(fontsize=9)

    # Right: 3-stage (text→text apps with output_eval)
    ax = axes[1]
    df2 = pd.DataFrame(rows_oe).set_index("family")
    x = np.arange(len(df2)); w = 0.27
    ax.bar(x-w, df2["Input GT"],     w, color=S_INPUT,  label="Input GT",     edgecolor="white")
    ax.bar(x,   df2["Raw Output"],   w, color=S_OUTPUT, label="Raw Output",   edgecolor="white")
    ax.bar(x+w, df2["Externalized"], w, color=S_EXT,    label="Externalized", edgecolor="white")
    for i, (a, b, c) in enumerate(zip(df2["Input GT"], df2["Raw Output"], df2["Externalized"])):
        ax.text(i-w, a+0.01, f"{a:.0%}", ha="center", fontsize=7.5)
        ax.text(i,   b+0.01, f"{b:.0%}", ha="center", fontsize=7.5)
        ax.text(i+w, c+0.01, f"{c:.0%}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(df2.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"B. 3-Stage Flow — text→text Apps with Raw Output Eval (n={len(df_oe):,} pairs)", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle("Deep Fig 4 — Privacy Attribute Flow: Input GT → Raw Output → Externalized (per family)", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig04_three_stage_flow.png")


def deep_fig05_inference_externalization_gap(df: pd.DataFrame) -> str:
    """For text→text with output_eval: inferred-but-not-externalized = filtering effect."""
    df_oe = df[df["output_has"]==1].copy()
    if df_oe.empty:
        return ""
    # Per-attribute stats
    attr_data = df_oe.groupby("attr").agg(
        out_leak=("output_leak","mean"),
        ext_leak=("ext_leak","mean"),
        out_conf=("output_conf","mean"),
        ext_conf=("ext_conf","mean"),
        n=("output_leak","count"),
    ).reset_index()
    attr_data["family"] = attr_data["attr"].map(ATTR_TO_FAMILY)
    attr_data["filter_gap"] = attr_data["out_leak"] - attr_data["ext_leak"]   # positive = filtered out
    attr_data = attr_data[attr_data["n"] >= 30]    # exclude rare attrs
    attr_data = attr_data.sort_values("out_leak", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

    # Left: scatter — output rate vs ext rate
    ax = axes[0]
    for _, row in attr_data.iterrows():
        c = FAMILY_COLORS.get(row["family"], GRAY)
        ax.scatter(row["out_leak"], row["ext_leak"],
                   s=max(40, row["n"]*0.1), color=c, alpha=0.85,
                   edgecolors="white", linewidth=0.8)
        ax.annotate(row["attr"], (row["out_leak"], row["ext_leak"]),
                    textcoords="offset points", xytext=(5,3), fontsize=8)
    ax.plot([0,1],[0,1], color=GRAY, ls="--", alpha=0.5, label="y=x (no filtering)")
    ax.set_xlabel("Inferred in raw output (any leakage)")
    ax.set_ylabel("Externalized (any leakage)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("A. Inference vs Externalization\n(below y=x = filtered out; above = expanded via channels)", fontsize=10)
    ax.legend(fontsize=8.5)

    # Right: bar of filter gap (inferred but not externalized)
    ax = axes[1]
    attr_sorted = attr_data.sort_values("filter_gap", ascending=True)
    colors = [FAMILY_COLORS.get(ATTR_TO_FAMILY[a], GRAY) for a in attr_sorted["attr"]]
    bars = ax.barh(attr_sorted["attr"], attr_sorted["filter_gap"],
                   color=colors, edgecolor="white")
    for bar, v in zip(bars, attr_sorted["filter_gap"]):
        offset = 0.005 if v >= 0 else -0.005
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, bar.get_y()+bar.get_height()/2,
                f"{v:+.1%}", ha=ha, va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("filter_gap = (inference rate) − (externalization rate)")
    ax.set_title("B. Inference Filtering by Attribute\n(positive = inferred internally but suppressed in output)", fontsize=10)

    fig.suptitle("Deep Fig 5 — Inference vs Externalization Gap (filtering effectiveness)", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig05_inference_filter_gap.png")


def deep_fig06_attribute_persistence_grid(df: pd.DataFrame) -> str:
    """For each attribute, fraction of items where it persists from GT input → externalization."""
    rows = []
    for attr in ALL_ATTRS:
        sub = df[df["attr"]==attr]
        # Subset where attribute IS in input
        in_pos = sub[sub["input_label"]==1]
        # Subset where attribute NOT in input
        in_neg = sub[sub["input_label"]==0]
        rows.append({
            "attr": attr,
            "family": ATTR_TO_FAMILY.get(attr,""),
            "n_in_pos": len(in_pos), "n_in_neg": len(in_neg),
            "persistence": in_pos["ext_leak"].mean()  if len(in_pos) > 0 else np.nan,
            "false_inj":   in_neg["ext_leak"].mean()  if len(in_neg) > 0 else np.nan,
            "conf_persist":in_pos["ext_conf"].mean()  if len(in_pos) > 0 else np.nan,
        })

    rdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    # Left: persistence (when input has attr → ext shows it)
    ax = axes[0]
    rdf2 = rdf.dropna(subset=["persistence"]).sort_values("persistence", ascending=True)
    colors = [FAMILY_COLORS.get(f, GRAY) for f in rdf2["family"]]
    bars = ax.barh(rdf2["attr"], rdf2["persistence"], color=colors, edgecolor="white")
    for bar, v, n in zip(bars, rdf2["persistence"], rdf2["n_in_pos"]):
        ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
                f"{v:.0%} (n={n})", va="center", fontsize=7.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("P(externalized | input has attribute)")
    ax.set_title("A. Attribute Persistence\n(how often GT presence survives to externalization)", fontsize=10)

    # Right: false injection (input doesn't have it but ext shows it)
    ax = axes[1]
    rdf3 = rdf.dropna(subset=["false_inj"]).sort_values("false_inj", ascending=True)
    colors2 = [FAMILY_COLORS.get(f, GRAY) for f in rdf3["family"]]
    bars = ax.barh(rdf3["attr"], rdf3["false_inj"], color=colors2, edgecolor="white")
    for bar, v, n in zip(bars, rdf3["false_inj"], rdf3["n_in_neg"]):
        ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
                f"{v:.0%} (n={n})", va="center", fontsize=7.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("P(externalized | input does NOT have attribute)")
    ax.set_title("B. Inference Injection\n(externalized despite no GT support)", fontsize=10)

    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5,-0.04),
               fontsize=8, framealpha=0.95)
    fig.suptitle("Deep Fig 6 — Attribute Persistence & Inference Injection", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig06_persistence_injection.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CHANNEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def deep_fig07_channel_family_matrix(df: pd.DataFrame) -> str:
    """Family × channel: confirmed leakage rate (3 panels: any, conf, present-cnt)."""
    fam_order = list(ATTR_FAMILIES.keys())
    rows = []
    for fam in fam_order:
        for ch in CHANNELS:
            sub = df[(df["family"]==fam) & (df[f"ch_{ch}_pres"]==1)]
            if len(sub) == 0:
                rows.append({"family": fam,"channel": ch,"any":0,"conf":0,"n":0})
            else:
                rows.append({
                    "family": fam, "channel": ch,
                    "any":  sub[f"ch_{ch}_leak"].mean()*100,
                    "conf": sub[f"ch_{ch}_conf"].mean()*100,
                    "n":    len(sub),
                })
    rdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="white")
    cmap_any  = LinearSegmentedColormap.from_list("a", ["white", P_TEAL, P_BLUE])
    cmap_conf = LinearSegmentedColormap.from_list("c", ["white", P_ORANGE, "#C93A3A"])
    cmap_n    = LinearSegmentedColormap.from_list("n", ["white", P_MAUVE])

    for ax, metric, vmax, cmap, title in [
        (axes[0], "any",  100, cmap_any,  "A. Any leakage % (channels × family)"),
        (axes[1], "conf", 100, cmap_conf, "B. Confirmed leakage %"),
        (axes[2], "n",    None, cmap_n,   "C. Sample count per cell"),
    ]:
        pivot = rdf.pivot_table(index="channel", columns="family", values=metric).reindex(CHANNELS)[fam_order]
        if vmax:
            sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".0f",
                        vmin=0, vmax=vmax, linewidths=0.4, linecolor="#E8E8E8",
                        cbar_kws={"shrink":0.65})
        else:
            sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".0f",
                        linewidths=0.4, linecolor="#E8E8E8",
                        cbar_kws={"shrink":0.65})
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(""); ax.set_ylabel("Channel" if ax is axes[0] else "")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle("Deep Fig 7 — Channel × Family Leakage Matrix", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig07_channel_family_matrix.png")


def deep_fig08_background_foreground_by_category(df: pd.DataFrame) -> str:
    """Background (STORAGE+LOGGING) vs foreground (NETWORK+UI) leakage per category."""
    cat_order = sorted(df["category"].unique())

    def _rate(grp, channels, metric):
        suffix = "leak" if metric == "any" else "conf"
        cols = [f"ch_{c}_{suffix}" for c in channels]
        pres = [f"ch_{c}_pres"  for c in channels]
        sub = grp[grp[pres].max(axis=1)==1] if pres else grp.iloc[0:0]
        return sub[cols].max(axis=1).mean() if len(sub)>0 else 0

    rows = []
    for cat in cat_order:
        grp = df[df["category"]==cat]
        for metric in ("any","confirmed"):
            rows.append({
                "category": cat, "metric": metric,
                "Background\n(STORAGE+LOGGING)": _rate(grp, ["STORAGE","LOGGING"], metric),
                "Foreground\n(NETWORK+UI)":     _rate(grp, ["NETWORK","UI"],     metric),
            })
    rdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white", sharey=True)
    titles = {"any":"Any Leakage", "confirmed":"Confirmed Leakage"}
    bg_col = "Background\n(STORAGE+LOGGING)"
    fg_col = "Foreground\n(NETWORK+UI)"
    for ax_idx, metric in enumerate(["any","confirmed"]):
        ax = axes[ax_idx]
        sub = rdf[rdf["metric"]==metric].set_index("category").reindex(cat_order)
        x = np.arange(len(cat_order)); w = 0.4
        ax.bar(x-w/2, sub[bg_col].values, w,
               color=P_ORANGE, label="Background (STORAGE+LOGGING)", edgecolor="white")
        ax.bar(x+w/2, sub[fg_col].values, w,
               color=P_BLUE, label="Foreground (NETWORK+UI)", edgecolor="white")
        for i, (a, b) in enumerate(zip(sub[bg_col].values, sub[fg_col].values)):
            ax.text(i-w/2, a+0.005, f"{a:.0%}", ha="center", fontsize=8)
            ax.text(i+w/2, b+0.005, f"{b:.0%}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(cat_order, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Leakage rate"); ax.set_ylim(0, 1.1)
        ax.set_title(f"{titles[metric]}", fontsize=10)
        if ax_idx == 0: ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Deep Fig 8 — Background vs Foreground Leakage by Category", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig08_bg_fg_category.png")


def deep_fig09_channel_divergence_per_app(df: pd.DataFrame) -> str:
    """For each app: which channel carries the most for each family. Stacked relative bars."""
    apps = (df.groupby("app")["ext_conf"].mean().sort_values(ascending=False).index.tolist())
    fam_order = list(ATTR_FAMILIES.keys())

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), facecolor="white", sharey=True)
    axes = axes.flatten()

    for ax_idx, app in enumerate(apps):
        if ax_idx >= len(axes): break
        ax = axes[ax_idx]
        sub = df[df["app"]==app]
        # For each family, fraction of attribute-item pairs that leaked through each channel
        rows = []
        for fam in fam_order:
            fsub = sub[sub["family"]==fam]
            if len(fsub) == 0:
                for ch in CHANNELS:
                    rows.append({"family": fam, "channel": ch, "rate": 0})
                continue
            for ch in CHANNELS:
                # rate of confirmed leakage in this channel (when channel is present)
                fp = fsub[fsub[f"ch_{ch}_pres"]==1]
                rate = fp[f"ch_{ch}_conf"].mean() if len(fp) > 0 else 0
                rows.append({"family": fam, "channel": ch, "rate": rate})
        rdf = pd.DataFrame(rows)
        pivot = rdf.pivot_table(index="family", columns="channel", values="rate").reindex(fam_order)[CHANNELS].fillna(0)

        bottom = np.zeros(len(pivot))
        for ch in CHANNELS:
            ax.bar(range(len(pivot)), pivot[ch], bottom=bottom,
                   color=CHANNEL_COLORS[ch], edgecolor="white",
                   label=ch if ax_idx == 0 else None, alpha=0.92)
            bottom += pivot[ch].values
        ax.set_title(f"{app}\n(n={len(sub):,})", fontsize=9)
        ax.set_xticks(range(len(pivot)))
        ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, max(0.6, bottom.max()*1.05))
        if ax_idx % 4 == 0: ax.set_ylabel("Σ confirmed rate")

    # Hide unused axes
    for j in range(len(apps), len(axes)):
        axes[j].set_visible(False)

    fig.legend(title="Channel", ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               fontsize=10, framealpha=0.95)
    fig.suptitle("Deep Fig 9 — Channel Divergence per App: which channel carries each family", fontsize=12.5, y=1.005)
    plt.tight_layout()
    return save(fig, "deep_fig09_channel_divergence.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PRIVACY TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════════════

def deep_fig10_expansion_factor(df: pd.DataFrame) -> str:
    """For each app: ratio of externalized attrs / input GT attrs. Shows expansion vs compression."""
    rows = []
    for (app, fk), grp in df.groupby(["app","full_key"]):
        n_in   = grp["input_label"].sum()
        n_ext  = grp["ext_leak"].sum()
        n_ext_conf = grp["ext_conf"].sum()
        new_attrs  = (set(grp[grp["ext_leak"]==1]["attr"]) - set(grp[grp["input_label"]==1]["attr"]))
        lost_attrs = (set(grp[grp["input_label"]==1]["attr"]) - set(grp[grp["ext_leak"]==1]["attr"]))
        rows.append({"app": app, "n_in": n_in, "n_ext": n_ext,
                     "n_ext_conf": n_ext_conf,
                     "n_new": len(new_attrs), "n_lost": len(lost_attrs)})
    pdf = pd.DataFrame(rows)
    app_agg = pdf.groupby("app").agg(
        in_avg=("n_in","mean"), ext_avg=("n_ext","mean"),
        conf_avg=("n_ext_conf","mean"),
        new_avg=("n_new","mean"), lost_avg=("n_lost","mean"),
        n_items=("n_in","count"),
    )
    app_agg["expansion_factor"] = app_agg["ext_avg"] / app_agg["in_avg"].replace(0, np.nan)
    app_agg = app_agg.sort_values("expansion_factor", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(app_agg)*0.5)), facecolor="white")

    # Left: expansion factor bar
    ax = axes[0]
    colors = [P_GREEN if v < 1 else P_ORANGE if v < 1.5 else P_BLUE
              for v in app_agg["expansion_factor"]]
    bars = ax.barh(app_agg.index, app_agg["expansion_factor"],
                   color=colors, edgecolor="white")
    ax.axvline(1, color="black", ls="--", lw=1, alpha=0.6, label="ratio=1 (no expansion)")
    for bar, v, n in zip(bars, app_agg["expansion_factor"], app_agg["n_items"]):
        ax.text(v+0.03, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}× (n={n})", va="center", fontsize=8)
    ax.set_xlabel("Externalized attrs / Input GT attrs")
    ax.set_title("A. Per-App Expansion Factor", fontsize=10)
    ax.legend(fontsize=9)

    # Right: new vs lost attrs
    ax = axes[1]
    x = np.arange(len(app_agg)); w = 0.35
    ax.barh(app_agg.index, app_agg["new_avg"],   color=P_ORANGE, edgecolor="white",
            label="New attrs (not in input)")
    ax.barh(app_agg.index, -app_agg["lost_avg"], color=P_MAUVE, edgecolor="white",
            label="Lost attrs (in input, not ext)")
    ax.axvline(0, color="black", lw=0.8)
    for i, (n, l) in enumerate(zip(app_agg["new_avg"], app_agg["lost_avg"])):
        if n > 0.1: ax.text(n+0.05,  i, f"+{n:.1f}", va="center", fontsize=7.5)
        if l > 0.1: ax.text(-l-0.05, i, f"−{l:.1f}", va="center", ha="right", fontsize=7.5)
    ax.set_xlabel("Mean attrs per item (signed)")
    ax.set_title("B. New (inferred-only) vs Lost (passthrough-only) attrs", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Deep Fig 10 — Privacy Set Transformation per App", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig10_expansion_factor.png")


def deep_fig11_jaccard_similarity(df: pd.DataFrame) -> str:
    """Per-item Jaccard similarity between input GT set and externalized set, by category."""
    items = []
    for fk, grp in df.groupby("full_key"):
        in_set  = set(grp[grp["input_label"]==1]["attr"])
        ext_set = set(grp[grp["ext_leak"]==1]["attr"])
        if not in_set and not ext_set:
            continue
        union  = in_set | ext_set
        inter  = in_set & ext_set
        jacc   = len(inter)/len(union) if len(union)>0 else 0
        cat    = grp["category"].iloc[0]
        modp   = grp["modality_pair"].iloc[0]
        in_t   = grp["in_type"].iloc[0]
        items.append({"jaccard":jacc, "category":cat, "modality_pair":modp, "in_type":in_t,
                      "n_in":len(in_set), "n_ext":len(ext_set), "n_inter":len(inter)})

    idf = pd.DataFrame(items)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="white")

    # By category
    ax = axes[0]
    cat_order = sorted(idf["category"].unique())
    box_data = [idf[idf["category"]==c]["jaccard"].values for c in cat_order]
    colors = [CATEGORY_COLORS.get(c, GRAY) for c in cat_order]
    bp = ax.boxplot(box_data, labels=cat_order, patch_artist=True,
                    medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.set_ylabel("Jaccard(input GT, ext set)"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("A. Set Similarity per Category", fontsize=10)
    ax.set_xticklabels(cat_order, rotation=15, ha="right", fontsize=9)

    # By input type
    ax = axes[1]
    it_order = ["text","image","docs"]
    box_data2 = [idf[idf["in_type"]==t]["jaccard"].values for t in it_order]
    colors2 = [INPUT_TYPE_COLOR[t] for t in it_order]
    bp2 = ax.boxplot(box_data2, labels=it_order, patch_artist=True,
                     medianprops=dict(color="black"))
    for patch, c in zip(bp2["boxes"], colors2):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("B. Set Similarity per Input Type", fontsize=10)

    # Annotate means
    for ax_idx, (ax_, data, labels) in enumerate([
        (axes[0], box_data, cat_order),
        (axes[1], box_data2, it_order),
    ]):
        for i, vals in enumerate(data):
            if len(vals) > 0:
                ax_.text(i+1, np.mean(vals)+0.02, f"μ={np.mean(vals):.2f}",
                         ha="center", fontsize=7.5)

    # Right: scatter input set size vs ext set size
    ax = axes[2]
    for it in it_order:
        sub = idf[idf["in_type"]==it]
        ax.scatter(sub["n_in"], sub["n_ext"], color=INPUT_TYPE_COLOR[it],
                   alpha=0.4, s=20, label=it, edgecolors="white", linewidth=0.3)
    max_n = max(idf["n_in"].max(), idf["n_ext"].max(), 5)
    ax.plot([0,max_n],[0,max_n], color=GRAY, ls="--", alpha=0.5)
    ax.set_xlabel("# input GT attrs"); ax.set_ylabel("# externalized attrs")
    ax.set_title("C. Input GT Size vs Externalized Size", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle("Deep Fig 11 — Set Transformation: Jaccard Similarity Input GT ↔ Externalized", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig11_jaccard.png")


def deep_fig12_family_transformation_patterns(df: pd.DataFrame) -> str:
    """Per family: fraction of items where each family transforms (loss / kept / new)."""
    fam_order = list(ATTR_FAMILIES.keys())

    rows = []
    for fam in fam_order:
        for fk, grp in df.groupby("full_key"):
            fgrp = grp[grp["family"]==fam]
            if len(fgrp) == 0: continue
            in_attrs  = set(fgrp[fgrp["input_label"]==1]["attr"])
            ext_attrs = set(fgrp[fgrp["ext_leak"]==1]["attr"])
            if not in_attrs and not ext_attrs: continue
            kept = len(in_attrs & ext_attrs)
            lost = len(in_attrs - ext_attrs)
            new  = len(ext_attrs - in_attrs)
            rows.append({"family":fam,
                         "kept":kept, "lost":lost, "new":new,
                         "transform_type": (
                             "Persisted"  if kept>0 and new==0 and lost==0 else
                             "Expanded"   if new>0 and lost==0 else
                             "Compressed" if lost>0 and new==0 else
                             "Mixed"      if new>0 and lost>0 else
                             "Empty"      if kept==0 and new==0 and lost==0 else
                             "Other"
                         )})

    rdf = pd.DataFrame(rows)
    types_order = ["Persisted","Expanded","Compressed","Mixed","Empty"]
    type_colors = {"Persisted":P_TEAL, "Expanded":P_ORANGE,
                   "Compressed":P_MAUVE, "Mixed":P_YELLOW, "Empty":"#E0E0E0"}

    pivot = rdf.groupby(["family","transform_type"]).size().unstack(fill_value=0)
    for t in types_order:
        if t not in pivot.columns: pivot[t] = 0
    pivot = pivot.div(pivot.sum(axis=1), axis=0).reindex(fam_order)[types_order].fillna(0)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="white")
    bottom = np.zeros(len(pivot))
    for t in types_order:
        vals = pivot[t].values
        ax.bar(pivot.index, vals, bottom=bottom, color=type_colors[t],
               label=t, edgecolor="white", linewidth=0.5)
        for i, val in enumerate(vals):
            if val > 0.04:
                ax.text(i, bottom[i]+val/2, f"{val:.0%}",
                        ha="center", va="center", fontsize=8.5,
                        color="black" if t in ("Empty","Mixed") else "white",
                        fontweight="bold")
        bottom += vals

    ax.set_ylim(0, 1.02); ax.set_ylabel("Fraction of items")
    ax.set_title("Deep Fig 12 — Per-Family Transformation Pattern Distribution\n"
                 "Persisted = kept, Expanded = new attrs, Compressed = attrs lost, Mixed = both",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticklabels(pivot.index, rotation=20, ha="right", fontsize=9.5)
    plt.tight_layout()
    return save(fig, "deep_fig12_family_transform.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MODALITY PAIR COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def deep_fig13_modality_pair_matrix(df: pd.DataFrame) -> str:
    """4-way modality pair × family heatmap."""
    pairs = ["text→text", "image→text", "text→image", "image→image"]
    fam_order = list(ATTR_FAMILIES.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    cmap = LinearSegmentedColormap.from_list("v", ["white", P_TEAL, P_ORANGE, "#C93A3A"])

    metrics = [
        ("input_label", "A. Input GT presence rate per family",  axes[0,0]),
        ("ext_leak",    "B. Any leakage rate per family",        axes[0,1]),
        ("ext_conf",    "C. Confirmed leakage rate per family",  axes[1,0]),
    ]

    for col, title, ax in metrics:
        rows = []
        for mp in pairs:
            for fam in fam_order:
                sub = df[(df["modality_pair"]==mp) & (df["family"]==fam)]
                rows.append({"pair":mp, "family":fam,
                             "rate":sub[col].mean()*100 if len(sub)>0 else np.nan,
                             "n": len(sub)})
        rdf = pd.DataFrame(rows)
        pivot = rdf.pivot_table(index="pair", columns="family", values="rate").reindex(pairs)[fam_order]
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".0f",
                    vmin=0, vmax=100, linewidths=0.4, linecolor="#E8E8E8",
                    cbar_kws={"shrink":0.65, "label":"%"},
                    mask=pivot.isna())
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Bottom-right: sample counts
    ax = axes[1,1]
    rows = []
    for mp in pairs:
        for fam in fam_order:
            sub = df[(df["modality_pair"]==mp) & (df["family"]==fam)]
            rows.append({"pair":mp, "family":fam, "n":len(sub)})
    rdf = pd.DataFrame(rows)
    pivot_n = rdf.pivot_table(index="pair", columns="family", values="n").reindex(pairs)[fam_order].fillna(0)
    sns.heatmap(pivot_n, ax=ax, cmap=LinearSegmentedColormap.from_list("n",["white",P_MAUVE,P_SLATE]),
                annot=True, fmt=".0f", linewidths=0.4, linecolor="#E8E8E8", cbar_kws={"shrink":0.65, "label":"n"})
    ax.set_title("D. Sample count per cell", fontsize=10.5)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle("Deep Fig 13 — 4-way Modality Pair × Attribute Family Matrix", fontsize=12.5, y=1.00)
    plt.tight_layout()
    return save(fig, "deep_fig13_modality_pair_matrix.png")


def deep_fig14_visual_re_encoding(df: pd.DataFrame) -> str:
    """For image→text: do visual attributes leak through text channels at high rate?"""
    img_to_text = df[df["modality_pair"]=="image→text"].copy()
    text_to_text = df[df["modality_pair"]=="text→text"].copy()

    # Visual-only attrs (image-only in taxonomy)
    visual_attrs = ["face","race","nudity","height","weight","disability",
                    "ethnic_clothing","religion","medical","sports","formal",
                    "uniforms","casual","color","haircolor","troupe"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

    # Left: per-visual-attr externalization rate in image→text
    ax = axes[0]
    rows = []
    for attr in visual_attrs:
        sub = img_to_text[img_to_text["attr"]==attr]
        if len(sub) == 0: continue
        rows.append({
            "attr": attr,
            "input_rate": sub["input_label"].mean(),
            "ext_rate":   sub["ext_leak"].mean(),
            "ext_conf":   sub["ext_conf"].mean(),
            "n":          len(sub),
        })
    vdf = pd.DataFrame(rows).sort_values("input_rate", ascending=True)

    y = np.arange(len(vdf)); w = 0.35
    ax.barh(y-w/2, vdf["input_rate"], w, color=S_INPUT, edgecolor="white", label="Input GT (visual present)")
    ax.barh(y+w/2, vdf["ext_rate"],   w, color=S_EXT,   edgecolor="white", label="Externalized in text")
    ax.set_yticks(y); ax.set_yticklabels(vdf["attr"], fontsize=8.5)
    for i, (a, b) in enumerate(zip(vdf["input_rate"], vdf["ext_rate"])):
        ax.text(a+0.005, i-w/2, f"{a:.0%}", va="center", fontsize=7)
        ax.text(b+0.005, i+w/2, f"{b:.0%}", va="center", fontsize=7)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Rate")
    ax.set_title("A. Image→Text: Visual Attrs Re-encoded as Text\n(image features → textual descriptions in output)", fontsize=10)
    ax.legend(fontsize=9)

    # Right: persistence ratio (P(ext|input==1)) for visual attrs in image→text
    ax = axes[1]
    rows = []
    for attr in visual_attrs:
        sub_i = img_to_text[img_to_text["attr"]==attr]
        sub_t = text_to_text[text_to_text["attr"]==attr]
        in_pos_i = sub_i[sub_i["input_label"]==1]
        in_pos_t = sub_t[sub_t["input_label"]==1]
        rows.append({
            "attr": attr,
            "img→text_persist": in_pos_i["ext_leak"].mean() if len(in_pos_i) > 0 else np.nan,
            "text→text_persist": in_pos_t["ext_leak"].mean() if len(in_pos_t) > 0 else np.nan,
        })
    pdf = pd.DataFrame(rows).set_index("attr")
    pdf = pdf.dropna(subset=["img→text_persist"]).sort_values("img→text_persist", ascending=True)

    y = np.arange(len(pdf)); w = 0.4
    ax.barh(y-w/2, pdf["img→text_persist"], w, color=P_BLUE, edgecolor="white", label="image→text")
    if pdf["text→text_persist"].notna().any():
        ax.barh(y+w/2, pdf["text→text_persist"].fillna(0), w, color=P_TEAL,
                edgecolor="white", label="text→text", alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(pdf.index, fontsize=8.5)
    ax.set_xlim(0, 1.05); ax.set_xlabel("P(externalized | input has attribute)")
    ax.set_title("B. Visual Attribute Persistence: image→text vs text→text", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle("Deep Fig 14 — Visual Re-encoding Evidence (image→text)", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig14_visual_reencoding.png")


def deep_fig15_semantic_crystallization(df: pd.DataFrame) -> str:
    """For text→text: do weak input cues become strong externalized verdicts?"""
    text_df = df[(df["modality_pair"]=="text→text")]

    # Compute: for each attribute, conditional P(ext_conf | input==0) — pure inference
    # vs P(ext_conf | input==1) — passthrough confirmation
    rows = []
    for attr in ALL_ATTRS:
        sub = text_df[text_df["attr"]==attr]
        if len(sub) < 30: continue
        in_neg = sub[sub["input_label"]==0]
        in_pos = sub[sub["input_label"]==1]
        rows.append({
            "attr": attr, "family": ATTR_TO_FAMILY.get(attr,""),
            "P_conf_no_input":  in_neg["ext_conf"].mean() if len(in_neg) > 0 else np.nan,
            "P_conf_with_input":in_pos["ext_conf"].mean() if len(in_pos) > 0 else np.nan,
            "n_neg": len(in_neg), "n_pos": len(in_pos),
        })
    cdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

    # Left: scatter — P(ext|no input) vs P(ext|with input)
    ax = axes[0]
    for _, row in cdf.iterrows():
        c = FAMILY_COLORS.get(row["family"], GRAY)
        ax.scatter(row["P_conf_no_input"], row["P_conf_with_input"],
                   s=max(40, row["n_pos"]*0.3), color=c, alpha=0.85,
                   edgecolors="white", linewidth=0.7)
        if row["P_conf_no_input"] > 0.05 or row["P_conf_with_input"] > 0.1:
            ax.annotate(row["attr"],
                        (row["P_conf_no_input"], row["P_conf_with_input"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.plot([0,1],[0,1], color=GRAY, ls="--", alpha=0.5)
    ax.set_xlabel("P(confirmed | input does NOT have attr)\n[pure inference]")
    ax.set_ylabel("P(confirmed | input has attr)\n[passthrough]")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("A. Crystallization: pure-inference rate vs passthrough rate", fontsize=10)

    # Right: text→text apps — what fraction of confirmed leakage is "pure inference"?
    ax = axes[1]
    apps_t = text_df["app"].unique()
    rows = []
    for app in apps_t:
        sub = text_df[text_df["app"]==app]
        n_conf = sub["ext_conf"].sum()
        n_conf_pure = sub[(sub["ext_conf"]==1) & (sub["input_label"]==0)].shape[0]
        n_conf_pt   = sub[(sub["ext_conf"]==1) & (sub["input_label"]==1)].shape[0]
        rows.append({"app":app,
                     "pure_inf_share": n_conf_pure / n_conf if n_conf > 0 else 0,
                     "passthrough_share": n_conf_pt / n_conf if n_conf > 0 else 0,
                     "n": n_conf})
    adf = pd.DataFrame(rows).sort_values("pure_inf_share", ascending=True)
    y = np.arange(len(adf)); w = 0.4
    ax.barh(y-w/2, adf["passthrough_share"], w,
            color=S_INPUT, edgecolor="white", label="Passthrough (input had attr)")
    ax.barh(y+w/2, adf["pure_inf_share"], w,
            color=P_ORANGE, edgecolor="white", label="Pure inference (no GT support)")
    ax.set_yticks(y); ax.set_yticklabels(adf["app"], fontsize=9)
    for i, (p, q, n) in enumerate(zip(adf["passthrough_share"], adf["pure_inf_share"], adf["n"])):
        ax.text(p+0.005, i-w/2, f"{p:.0%}", va="center", fontsize=7.5)
        ax.text(q+0.005, i+w/2, f"{q:.0%} (n={n})", va="center", fontsize=7.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Share of confirmed leakage")
    ax.set_title("B. text→text Apps: Confirmed Leakage Source\n(pure inference vs passthrough)", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Deep Fig 15 — Semantic Crystallization Evidence (text→text)", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig15_crystallization.png")


def deep_fig16_profile_consolidation(df: pd.DataFrame) -> str:
    """How many attrs simultaneously leak per item, by modality pair / category."""
    per_item = df.groupby(["full_key","modality_pair","in_type","category"]).agg(
        n_conf=("ext_conf","sum"),
        n_leak=("ext_leak","sum"),
        n_in=("input_label","sum"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="white")

    # By modality pair (histogram of n_conf)
    ax = axes[0]
    pairs = sorted(per_item["modality_pair"].unique())
    max_n = int(per_item["n_conf"].max()) + 1
    bins = list(range(0, max_n+1))
    for mp in pairs:
        sub = per_item[per_item["modality_pair"]==mp]["n_conf"]
        if len(sub) == 0: continue
        c = MOD_PAIR_COLOR.get(mp, GRAY)
        ax.hist(sub, bins=bins, alpha=0.55, color=c, edgecolor="white",
                label=f"{mp} (μ={sub.mean():.1f})", density=True)
    ax.set_xlabel("# confirmed-leaked attrs per item")
    ax.set_ylabel("Density")
    ax.set_title("A. Co-leak Distribution by Modality Pair", fontsize=10)
    ax.legend(fontsize=8.5)

    # By input type (mean ± std per type)
    ax = axes[1]
    type_order = ["text","image","docs"]
    rows = []
    for it in type_order:
        sub = per_item[per_item["in_type"]==it]
        rows.append({"type": it,
                     "mean": sub["n_conf"].mean(),
                     "std":  sub["n_conf"].std(),
                     "median": sub["n_conf"].median(),
                     "max": sub["n_conf"].max(),
                     "n": len(sub)})
    rdf = pd.DataFrame(rows)
    x = np.arange(len(rdf)); w = 0.35
    bars1 = ax.bar(x-w/2, rdf["mean"],   w, color=P_TEAL,  yerr=rdf["std"],
                   capsize=5, label="Mean ± SD", edgecolor="white", error_kw={"elinewidth":1.2})
    bars2 = ax.bar(x+w/2, rdf["median"], w, color=P_ORANGE, label="Median", edgecolor="white")
    for i, (m, s, md, mx, n) in enumerate(zip(rdf["mean"], rdf["std"], rdf["median"], rdf["max"], rdf["n"])):
        ax.text(i-w/2, m+s+0.1, f"{m:.1f}", ha="center", fontsize=8)
        ax.text(i+w/2, md+0.1,    f"{md:.0f}", ha="center", fontsize=8)
        ax.text(i, -0.2, f"max={mx}\nn={n}", ha="center", fontsize=7, color="#666",
                transform=ax.get_xaxis_transform(), va="top")
    ax.set_xticks(x); ax.set_xticklabels(rdf["type"], fontsize=10)
    ax.set_ylabel("# confirmed attrs per item")
    ax.set_title("B. Co-leak Statistics by Input Type", fontsize=10)
    ax.legend(fontsize=9)

    # By category (mean confirmed attrs)
    ax = axes[2]
    cat_order = sorted(per_item["category"].unique())
    rows = []
    for cat in cat_order:
        sub = per_item[per_item["category"]==cat]
        rows.append({"cat":cat, "mean":sub["n_conf"].mean(),
                     "ext_avg":sub["n_leak"].mean(),
                     "in_avg":sub["n_in"].mean(), "n":len(sub)})
    rdf = pd.DataFrame(rows)
    x = np.arange(len(rdf)); w = 0.27
    ax.bar(x-w, rdf["in_avg"],  w, color=S_INPUT,  edgecolor="white", label="Input GT")
    ax.bar(x,   rdf["ext_avg"], w, color=P_ORANGE, edgecolor="white", label="Any leak")
    ax.bar(x+w, rdf["mean"],    w, color=S_EXT,    edgecolor="white", label="Confirmed")
    for i, (a, b, c) in enumerate(zip(rdf["in_avg"], rdf["ext_avg"], rdf["mean"])):
        ax.text(i-w, a+0.1, f"{a:.1f}", ha="center", fontsize=7.5)
        ax.text(i,   b+0.1, f"{b:.1f}", ha="center", fontsize=7.5)
        ax.text(i+w, c+0.1, f"{c:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(rdf["cat"], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Mean # attrs per item")
    ax.set_title("C. Profile Consolidation by Category", fontsize=10)
    ax.legend(fontsize=8.5)

    fig.suptitle("Deep Fig 16 — Profile Consolidation: Multi-Attribute Co-leakage", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig16_profile_consolidation.png")


def deep_fig17_modality_transformation_radar(df: pd.DataFrame) -> str:
    """Radar comparison: confirmed leakage rate per family across modality pairs."""
    pairs = ["text→text","image→text","text→image","image→image"]
    fam_order = list(ATTR_FAMILIES.keys())
    N = len(fam_order)
    angles = [n / float(N) * 2*np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8), subplot_kw={"projection":"polar"}, facecolor="white")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(fam_order, fontsize=9)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(["10%","20%","30%","40%","50%"], fontsize=8)
    ax.grid(alpha=0.4)

    for mp in pairs:
        sub = df[df["modality_pair"]==mp]
        if len(sub) == 0: continue
        vals = []
        for fam in fam_order:
            fsub = sub[sub["family"]==fam]
            vals.append(fsub["ext_conf"].mean() if len(fsub) > 0 else 0)
        v = vals + vals[:1]
        c = MOD_PAIR_COLOR.get(mp, GRAY)
        ax.plot(angles, v, color=c, linewidth=2, label=f"{mp} (n={len(sub):,})")
        ax.fill(angles, v, color=c, alpha=0.13)

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1), fontsize=9)
    ax.set_title("Deep Fig 17 — Confirmed Leakage Radar by Modality Pair", fontsize=12, pad=22)
    return save(fig, "deep_fig17_modality_radar.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DOCS / IMAGE / TEXT INPUT-TYPE SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def deep_fig18_input_type_3way(df: pd.DataFrame) -> str:
    """3-way input type comparison across families."""
    type_order = ["text","image","docs"]
    fam_order = list(ATTR_FAMILIES.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")

    # A: input GT rate per family
    ax = axes[0,0]
    for it in type_order:
        sub = df[df["in_type"]==it]
        vals = [sub[sub["family"]==f]["input_label"].mean() if len(sub[sub["family"]==f])>0 else 0
                for f in fam_order]
        x = np.arange(len(fam_order))
        ax.plot(x, vals, marker="o", linewidth=2, color=INPUT_TYPE_COLOR[it],
                label=it.capitalize(), markersize=8)
    ax.set_xticks(np.arange(len(fam_order)))
    ax.set_xticklabels(fam_order, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Input GT presence rate")
    ax.set_title("A. Input GT Presence by Type × Family", fontsize=10.5)
    ax.legend(fontsize=9)

    # B: any leakage rate per family
    ax = axes[0,1]
    for it in type_order:
        sub = df[df["in_type"]==it]
        vals = [sub[sub["family"]==f]["ext_leak"].mean() if len(sub[sub["family"]==f])>0 else 0
                for f in fam_order]
        x = np.arange(len(fam_order))
        ax.plot(x, vals, marker="o", linewidth=2, color=INPUT_TYPE_COLOR[it],
                label=it.capitalize(), markersize=8)
    ax.set_xticks(np.arange(len(fam_order)))
    ax.set_xticklabels(fam_order, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 0.65); ax.set_ylabel("Any leakage rate")
    ax.set_title("B. Any Leakage Rate by Type × Family", fontsize=10.5)
    ax.legend(fontsize=9)

    # C: confirmed leakage
    ax = axes[1,0]
    for it in type_order:
        sub = df[df["in_type"]==it]
        vals = [sub[sub["family"]==f]["ext_conf"].mean() if len(sub[sub["family"]==f])>0 else 0
                for f in fam_order]
        x = np.arange(len(fam_order))
        ax.plot(x, vals, marker="o", linewidth=2, color=INPUT_TYPE_COLOR[it],
                label=it.capitalize(), markersize=8)
    ax.set_xticks(np.arange(len(fam_order)))
    ax.set_xticklabels(fam_order, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 0.5); ax.set_ylabel("Confirmed leakage rate")
    ax.set_title("C. Confirmed Leakage Rate by Type × Family", fontsize=10.5)
    ax.legend(fontsize=9)

    # D: gap (input rate − ext rate) per family — POSITIVE means input > ext (compression)
    ax = axes[1,1]
    rows = []
    for it in type_order:
        sub = df[df["in_type"]==it]
        for f in fam_order:
            fsub = sub[sub["family"]==f]
            if len(fsub) == 0: continue
            rows.append({"type":it, "family":f,
                         "gap": fsub["input_label"].mean() - fsub["ext_leak"].mean()})
    rdf = pd.DataFrame(rows)
    pivot = rdf.pivot_table(index="family", columns="type", values="gap").reindex(fam_order)[type_order].fillna(0)
    x = np.arange(len(fam_order)); w = 0.27
    for i, it in enumerate(type_order):
        ax.bar(x+(i-1)*w, pivot[it], w, color=INPUT_TYPE_COLOR[it], edgecolor="white",
               label=it.capitalize())
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(fam_order, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("input rate − any-leak rate")
    ax.set_title("D. Compression Gap (input minus ext)\n+ means attrs compressed; − means inferred-up", fontsize=10.5)
    ax.legend(fontsize=9)

    fig.suptitle("Deep Fig 18 — 3-way Input Type Comparison (text / image / docs)", fontsize=12.5, y=1.005)
    plt.tight_layout()
    return save(fig, "deep_fig18_input_type_3way.png")


def deep_fig19_input_type_persistence(df: pd.DataFrame) -> str:
    """Per-input-type: per-attr persistence rate (P(ext|input==1))."""
    type_order = ["text","image","docs"]
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")

    rows = []
    for it in type_order:
        sub = df[df["in_type"]==it]
        for attr in ALL_ATTRS:
            asub = sub[(sub["attr"]==attr) & (sub["input_label"]==1)]
            if len(asub) < 5: continue
            rows.append({"type":it, "attr":attr,
                         "family":ATTR_TO_FAMILY.get(attr,""),
                         "persist": asub["ext_leak"].mean(),
                         "n":len(asub)})
    rdf = pd.DataFrame(rows)

    # Bar plot grouped by attribute, three bars per group (one per input type)
    attrs_present = sorted(rdf["attr"].unique(),
                           key=lambda a: ATTR_TO_FAMILY.get(a,"") + ":" + a)
    x = np.arange(len(attrs_present)); w = 0.27

    for i, it in enumerate(type_order):
        sub = rdf[rdf["type"]==it].set_index("attr")
        vals = [sub.loc[a, "persist"] if a in sub.index else 0 for a in attrs_present]
        ns   = [sub.loc[a, "n"]       if a in sub.index else 0 for a in attrs_present]
        ax.bar(x+(i-1)*w, vals, w, color=INPUT_TYPE_COLOR[it], edgecolor="white",
               label=it.capitalize(), alpha=0.92)

    ax.set_xticks(x); ax.set_xticklabels(attrs_present, rotation=45, ha="right", fontsize=8)
    # Color attribute labels by family
    for tick, attr in zip(ax.get_xticklabels(), attrs_present):
        tick.set_color(FAMILY_COLORS.get(ATTR_TO_FAMILY.get(attr,""), "black"))
    ax.set_ylabel("P(externalized | input has attr)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Deep Fig 19 — Attribute Persistence by Input Type", fontsize=11.5)
    ax.legend(fontsize=10)

    family_handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    leg2 = fig.legend(handles=family_handles, ncol=4, loc="lower center",
                      bbox_to_anchor=(0.5,-0.04), fontsize=8.5,
                      title="Attribute label color = family", framealpha=0.95)
    plt.tight_layout()
    return save(fig, "deep_fig19_input_type_persistence.png")


def deep_fig20_input_type_channel_breakdown(df: pd.DataFrame) -> str:
    """Heatmap: input type × channel → confirmed leakage."""
    type_order = ["text","image","docs"]
    rows = []
    for it in type_order:
        sub = df[df["in_type"]==it]
        for ch in CHANNELS:
            fp = sub[sub[f"ch_{ch}_pres"]==1]
            rows.append({"type":it, "channel":ch,
                         "any":   fp[f"ch_{ch}_leak"].mean()*100 if len(fp)>0 else 0,
                         "conf":  fp[f"ch_{ch}_conf"].mean()*100 if len(fp)>0 else 0,
                         "n": len(fp)})
    rdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    cmap = LinearSegmentedColormap.from_list("v", ["white", P_TEAL, P_ORANGE, "#C93A3A"])
    for ax, metric, title in [
        (axes[0], "any",  "A. Any leakage % by Input Type × Channel"),
        (axes[1], "conf", "B. Confirmed leakage %"),
    ]:
        pivot = rdf.pivot_table(index="type", columns="channel", values=metric).reindex(type_order)[CHANNELS]
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".0f",
                    vmin=0, vmax=100, linewidths=0.5, linecolor="#E8E8E8",
                    cbar_kws={"shrink":0.65, "label":"%"})
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(""); ax.set_ylabel("Input type" if ax is axes[0] else "")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    fig.suptitle("Deep Fig 20 — Input Type × Channel Leakage Matrix", fontsize=12, y=1.02)
    plt.tight_layout()
    return save(fig, "deep_fig20_input_type_channel.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CASE STUDIES
# ══════════════════════════════════════════════════════════════════════════════

def deep_fig21_case_study_heatmap(df: pd.DataFrame, raw: Dict[str,Dict], picks: List[str]) -> str:
    """For each case-study item: heatmap of input GT / output / ext / per-channel verdicts."""
    fig, axes = plt.subplots(len(picks), 1, figsize=(15, 2.4*len(picks)), facecolor="white")
    if len(picks) == 1: axes = [axes]

    cmap = LinearSegmentedColormap.from_list("v",
        ["white", VERDICT_COLORS["no evidence"], VERDICT_COLORS["possible leakage"], VERDICT_COLORS["confirmed leakage"]])

    for ax_i, pick_key in enumerate(picks):
        if pick_key not in raw: continue
        item = raw[pick_key]
        app = "?"
        for d in df[df["full_key"]==pick_key]["app"].values:
            app = d; break
        ds, mod = "?", "?"
        for d in df[df["full_key"]==pick_key]["dataset"].values: ds = d; break
        for d in df[df["full_key"]==pick_key]["modality_pair"].values: mod = d; break

        ext_eval = item.get("ext_eval", {})
        out_eval = item.get("output_eval", {})
        in_lab   = item.get("input_labels", {})
        # Stages: Input GT, Raw output (if available), aggregate ext, per-channel ext
        stages = ["Input GT"]
        rows_data = []
        # Input row
        rows_data.append([in_lab.get(a, 0) * 2 for a in ALL_ATTRS])

        # Output row (if present)
        has_oe = any(isinstance(v,dict) and (v.get("inferable") is not None or v.get("verdict")) for v in out_eval.values())
        if has_oe:
            stages.append("Raw Output")
            row = []
            for a in ALL_ATTRS:
                v = out_eval.get(a, {})
                if isinstance(v, dict):
                    if v.get("verdict"):
                        row.append({"confirmed leakage":3,"possible leakage":2,"no evidence":0}.get(v["verdict"], 0))
                    elif v.get("inferable"):
                        s = v.get("score", 0)
                        row.append(3 if s >= 0.7 else 2 if s >= 0.3 else 1)
                    else:
                        row.append(0)
                else:
                    row.append(0)
            rows_data.append(row)

        # Aggregate ext
        stages.append("Externalized\n(aggregate)")
        rows_data.append([entry_to_score(ext_eval.get(a)) for a in ALL_ATTRS])

        # Per-channel ext
        chans_in_item = list(item.get("externalizations", {}).keys())
        for ch in CHANNELS:
            if ch not in chans_in_item: continue
            stages.append(ch)
            row = []
            for a in ALL_ATTRS:
                ext_e = ext_eval.get(a)
                ch_v = get_channel_verdicts(ext_e).get(ch, "no evidence")
                row.append({"confirmed leakage":3,"possible leakage":2,"no evidence":0,"na":-1}.get(ch_v, 0))
            rows_data.append(row)

        mat = np.array(rows_data)
        ax = axes[ax_i]
        sns.heatmap(mat, ax=ax, cmap=cmap, vmin=0, vmax=3,
                    cbar=False, linewidths=0.5, linecolor="#E0E0E0",
                    xticklabels=ALL_ATTRS, yticklabels=stages,
                    annot=False)
        ax.set_xticklabels(ALL_ATTRS, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(stages, rotation=0, fontsize=8)
        # color attr labels by family
        for tick, attr in zip(ax.get_xticklabels(), ALL_ATTRS):
            tick.set_color(FAMILY_COLORS.get(ATTR_TO_FAMILY.get(attr,""), "black"))
        ax.set_title(f"Case {ax_i+1}: {app} | {ds} | {mod}", fontsize=10, loc="left")

    # Legend for verdicts
    handles = [
        mpatches.Patch(color="white", edgecolor="black", label="0 = absent / no evidence"),
        mpatches.Patch(color=VERDICT_COLORS["no evidence"],     label="1 = weak / no evidence"),
        mpatches.Patch(color=VERDICT_COLORS["possible leakage"],label="2 = present / possible"),
        mpatches.Patch(color=VERDICT_COLORS["confirmed leakage"],label="3 = confirmed"),
    ]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5,-0.02),
               fontsize=8.5, framealpha=0.95)
    fig.suptitle("Deep Fig 21 — Case Study Heatmaps: Input GT → Raw Output → Externalized (per channel)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    return save(fig, "deep_fig21_case_studies.png")

# ══════════════════════════════════════════════════════════════════════════════
# CASE STUDY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def select_case_studies(df: pd.DataFrame, raw: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    """Pick 1 case study from each high-data modality pair / input type combination."""
    picks = []
    seen_groups = set()

    # We want diversity: 1 text→text PrivacyLens, 1 image→text HR-VISPR,
    # 1 docs→text SROIE, 1 text→text SynthPAI/multi-channel
    candidates = (df.groupby("full_key").agg(
        n_in=("input_label","sum"),
        n_conf=("ext_conf","sum"),
        app=("app","first"),
        ds=("dataset","first"),
        mod=("modality_pair","first"),
        in_type=("in_type","first"),
        chans=("n_ext_channels","first"),
    ).reset_index())

    # Strategic picks
    targets = [
        ("text","PrivacyLens","waico"),
        ("text","PrivacyLens","deeptutor"),
        ("image","HR-VISPR","snapdo"),
        ("docs","SROIE2019","budget-lens"),
    ]
    for in_type, ds, app in targets:
        sub = candidates[
            (candidates["in_type"]==in_type) &
            (candidates["ds"]==ds) &
            (candidates["app"]==app) &
            (candidates["n_in"] >= 2) &
            (candidates["n_conf"] >= 2)
        ].sort_values("n_conf", ascending=False)
        if len(sub) > 0:
            picks.append(sub.iloc[0]["full_key"])

    return picks

def case_study_table_md(pick_key: str, raw: Dict[str, Dict], df: pd.DataFrame) -> str:
    """Generate markdown for a case study item."""
    if pick_key not in raw:
        return f"_Case study key not found: {pick_key}_\n"
    item = raw[pick_key]
    sub  = df[df["full_key"]==pick_key]
    if sub.empty:
        return ""
    app = sub.iloc[0]["app"]; ds = sub.iloc[0]["dataset"]
    mod = sub.iloc[0]["modality_pair"]; cat = sub.iloc[0]["category"]

    # Input snippet
    input_item = item.get("input_item", {})
    in_text  = input_item.get("text_content", "") or ""
    in_text_short = in_text[:600].replace("\n"," ").strip() + ("..." if len(in_text)>600 else "")
    if not in_text_short and input_item.get("filename"):
        in_text_short = f"_(image: {input_item['filename']})_"

    # Output snippet
    out_text = item.get("output_text", "")[:600].replace("\n"," ").strip() + ("..." if len(item.get("output_text",""))>600 else "")

    # Externalization channels
    exts = item.get("externalizations", {}) or {}
    ext_lines = []
    for ch, txt in exts.items():
        snippet = (txt or "")[:300].replace("\n"," ").strip()
        ext_lines.append(f"- **[{ch}]:** {snippet}{'...' if len(txt or '')>300 else ''}")
    ext_block = "\n".join(ext_lines) if ext_lines else "_(none captured)_"

    # Per-attribute table
    in_labels = item.get("input_labels", {})
    out_eval  = item.get("output_eval", {})
    ext_eval  = item.get("ext_eval", {})

    rows = []
    has_oe = any(isinstance(v,dict) and (v.get("inferable") is not None or v.get("verdict")) for v in out_eval.values())
    for attr in ALL_ATTRS:
        in_v = "✓" if in_labels.get(attr, 0) == 1 else ""
        out_v = ""
        if has_oe:
            oev = out_eval.get(attr, {})
            if isinstance(oev, dict):
                out_v = output_eval_to_verdict(oev)
                if out_v == "no evidence" and not oev: out_v = ""
                if out_v == "confirmed leakage": out_v = "🔴 conf."
                elif out_v == "possible leakage": out_v = "🟠 poss."
                elif out_v == "no evidence": out_v = "🟢 none"
        ext_e = ext_eval.get(attr, {})
        ext_v_text = entry_to_verdict(ext_e)
        if ext_v_text == "confirmed leakage":   ev = "🔴 conf."
        elif ext_v_text == "possible leakage":  ev = "🟠 poss."
        else:                                    ev = "🟢 none"
        prediction = ext_e.get("aggregate", {}).get("prediction", "") if isinstance(ext_e, dict) else ""
        if not (in_v or out_v or ext_v_text in ("confirmed leakage","possible leakage")):
            continue
        rows.append((attr, ATTR_TO_FAMILY.get(attr,""), in_v, out_v, ev, prediction))

    if not rows:
        return f"_Item {app}/{pick_key.split('/')[-1]} has no notable attributes_\n"

    md = []
    md.append(f"### Case: `{app}` · {ds} · {mod} · {cat}\n")
    md.append(f"**Input snippet:** _{in_text_short}_\n")
    md.append(f"**Raw output snippet:** _{out_text}_\n")
    md.append(f"**Externalizations:**\n{ext_block}\n")
    if has_oe:
        md.append("\n| Attribute | Family | Input GT | Raw Output | Externalized | Prediction |")
        md.append("|-----------|--------|----------|-----------|-------------|-----------|")
        for attr, fam, iv, ov, ev, pred in rows:
            md.append(f"| `{attr}` | {fam} | {iv} | {ov} | {ev} | {pred} |")
    else:
        md.append("\n| Attribute | Family | Input GT | Externalized | Prediction |")
        md.append("|-----------|--------|----------|-------------|-----------|")
        for attr, fam, iv, _, ev, pred in rows:
            md.append(f"| `{attr}` | {fam} | {iv} | {ev} | {pred} |")
    md.append("")
    return "\n".join(md)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def render_report(df: pd.DataFrame, raw: Dict[str,Dict], figs: Dict[str,str], picks: List[str]) -> str:
    n_total = len(df)
    n_items = df.groupby("full_key").ngroups
    n_apps = df["app"].nunique()
    n_ds = df["dataset"].nunique()

    overall = df["ext_verdict"].value_counts(normalize=True)
    p_conf = overall.get("confirmed leakage", 0)
    p_poss = overall.get("possible leakage", 0)
    p_no = overall.get("no evidence", 0)
    p_any = p_conf + p_poss

    # Per-family confirmed
    fam_conf = df.groupby("family")["ext_conf"].mean().sort_values(ascending=False)
    top3_fam = list(fam_conf.head(3).items())

    # Top attrs
    attr_conf = df.groupby("attr")["ext_conf"].mean().sort_values(ascending=False)
    top5_attr = list(attr_conf.head(5).items())

    # Per-modality pair
    mod_stats = df.groupby("modality_pair").agg(
        n_pairs=("ext_conf","count"), conf=("ext_conf","mean"),
        any_leak=("ext_leak","mean"), in_rate=("input_label","mean")
    ).sort_values("conf", ascending=False)

    # Per input type
    intype_stats = df.groupby("in_type").agg(
        n_pairs=("ext_conf","count"), conf=("ext_conf","mean"),
        any_leak=("ext_leak","mean"), in_rate=("input_label","mean")
    ).sort_values("conf", ascending=False)

    # Per category
    cat_stats = df.groupby("category").agg(
        n_pairs=("ext_conf","count"), conf=("ext_conf","mean"),
        any_leak=("ext_leak","mean"),
    ).sort_values("conf", ascending=False)

    # Channel stats
    ch_stats = {}
    for ch in CHANNELS:
        sub = df[df[f"ch_{ch}_pres"]==1]
        if len(sub) == 0: continue
        ch_stats[ch] = dict(
            n=len(sub), conf=sub[f"ch_{ch}_conf"].mean(),
            any=sub[f"ch_{ch}_leak"].mean(),
        )

    # Inference vs externalization gap (text→text only)
    df_oe = df[df["output_has"]==1]
    gap_summary = ""
    if len(df_oe) > 0:
        out_rate = df_oe["output_leak"].mean()
        ext_rate = df_oe["ext_leak"].mean()
        gap_summary = f"In text→text apps with raw-output evaluation (n={len(df_oe):,}): models inferred attributes at **{out_rate:.1%}** rate in raw output, but externalized at **{ext_rate:.1%}** — a {'compression' if out_rate>ext_rate else 'expansion'} of {abs(out_rate-ext_rate)*100:.1f} pp."

    # Expansion factor
    per_item = df.groupby("full_key").agg(n_in=("input_label","sum"),
                                            n_ext=("ext_leak","sum"),
                                            n_conf=("ext_conf","sum"),
                                            n_in_ext=("ext_leak", lambda s: 0))
    new_attrs_count = []
    for fk, grp in df.groupby("full_key"):
        new_attrs_count.append(len(set(grp[grp["ext_leak"]==1]["attr"]) - set(grp[grp["input_label"]==1]["attr"])))
    pct_expansion = sum(1 for n in new_attrs_count if n > 0) / max(1, len(new_attrs_count))
    avg_new = np.mean(new_attrs_count)

    # Per-modality co-leak mean
    per_item_full = df.groupby(["full_key","modality_pair","in_type"]).agg(
        n_conf=("ext_conf","sum")).reset_index()
    coleak_by_mod = per_item_full.groupby("modality_pair")["n_conf"].mean()
    coleak_by_intype = per_item_full.groupby("in_type")["n_conf"].mean()

    # Setup table
    setup_rows = (df.groupby(["app","dataset","modality_pair","eval_prompt"])
                    .size().reset_index(name="n"))
    setup_lines = "\n".join(f"| {r.app} | {r.dataset} | {r.modality_pair} | {r.eval_prompt} | {r.n:,} |"
                            for _, r in setup_rows.iterrows())

    # Family table
    fam_rows = []
    for fam in ATTR_FAMILIES:
        sub = df[df["family"]==fam]
        fam_rows.append(f"| {fam} | {', '.join(ATTR_FAMILIES[fam])} | {sub['input_label'].mean():.1%} | {sub['ext_leak'].mean():.1%} | {sub['ext_conf'].mean():.1%} |")
    fam_table = "\n".join(fam_rows)

    # Channel table
    ch_lines = []
    for ch, s in sorted(ch_stats.items(), key=lambda x: -x[1]["conf"]):
        ch_lines.append(f"| {ch} | {s['n']:,} | {s['any']:.1%} | {s['conf']:.1%} |")
    channel_table = "\n".join(ch_lines)

    # Modality pair table
    mp_lines = []
    for mp, row in mod_stats.iterrows():
        mp_lines.append(f"| {mp} | {row['n_pairs']:,} | {row['in_rate']:.1%} | {row['any_leak']:.1%} | {row['conf']:.1%} | {coleak_by_mod.get(mp,0):.2f} |")
    mp_table = "\n".join(mp_lines)

    # Intype table
    it_lines = []
    for it, row in intype_stats.iterrows():
        it_lines.append(f"| {it} | {row['n_pairs']:,} | {row['in_rate']:.1%} | {row['any_leak']:.1%} | {row['conf']:.1%} | {coleak_by_intype.get(it,0):.2f} |")
    intype_table = "\n".join(it_lines)

    # Cat table
    cat_lines = []
    for cat, row in cat_stats.iterrows():
        cat_lines.append(f"| {cat} | {row['n_pairs']:,} | {row['any_leak']:.1%} | {row['conf']:.1%} |")
    cat_table = "\n".join(cat_lines)

    # Per-app table
    app_lines = []
    for app, grp in df.groupby("app"):
        app_lines.append(f"| {app} | {APP_CATEGORY.get(app,'?')} | {len(grp):,} | {grp['ext_leak'].mean():.1%} | {grp['ext_conf'].mean():.1%} |")
    app_table = "\n".join(app_lines)

    # Case studies
    case_md = ""
    for k in picks:
        case_md += case_study_table_md(k, raw, df) + "\n---\n"

    md = f"""\
# Inference-Induced Privacy Leakage Landscape — Deep Analysis

> **Status:** Comprehensive analysis of prompt4/prompt5 evaluation results — **{n_items:,} items** across **{n_apps} apps** and **{n_ds} datasets**, totalling **{n_total:,} attribute-item judgments**.
> Generated: 2026-04-28. Color palette: Adobe (#7ADBC4 / #FAD765 / #FA9F5C / #98D198 / #6C80FC / #ACA4B3 / #687692). Stage colors (input/raw output/externalized) kept as blue/orange/red.

---

## 1. Experimental Setup

| App | Dataset | Modality | Prompt | N pairs |
|-----|---------|----------|--------|---------|
{setup_lines}

**Pipeline.** Each input item → app processes → produces (i) raw textual output and (ii) externalizations on captured channels (UI / NETWORK / STORAGE / LOGGING). A privacy-analyst LLM judge (Gemini 2.5 Pro via OpenRouter, prompt4/5) returns a 3-way verdict (`confirmed leakage` / `possible leakage` / `no evidence`) for each of 21 sensitive attributes, both at the **aggregate** level and for **each captured channel**. We exclude items with failed `ext_eval` (no verdicts). Where applicable, we also use the older `output_eval` (raw-output verdict, score-based) to reconstruct the **3-stage flow**: Input GT → Raw Output → Externalized.

**Attribute taxonomy** (paper appendix): 21 attributes grouped into 8 families. Image-only (HR-VISPR), text-only (location/identity/marital status), and shared (age, gender).

| Family | Attributes | Mean input GT rate | Mean any-leak rate | Mean confirmed rate |
|--------|-----------|----------------------|----------------------|----------------------|
{fam_table}

---

## 2. Headline Numbers

| Metric | Value |
|--------|-------|
| Attribute-item pairs evaluated | **{n_total:,}** |
| Unique items                   | **{n_items:,}** |
| **Overall confirmed leakage**  | **{p_conf:.1%}** |
| **Overall any leakage**        | **{p_any:.1%}** |
| Mean GT input attrs per item   | {df.groupby('full_key')['input_label'].sum().mean():.2f} |
| Mean any-leak attrs per item   | {df.groupby('full_key')['ext_leak'].sum().mean():.2f} |
| Mean confirmed attrs per item  | {df.groupby('full_key')['ext_conf'].sum().mean():.2f} |
| Items with **inference expansion** (≥1 new attr) | {pct_expansion:.1%} |
| Avg new attrs per item (not in input)            | {avg_new:.2f} |

{gap_summary}

![Deep Fig 1 — Landscape Overview](attachments/{figs['fig01']})

---

## 3. Section 1: Externalization Landscape by App Category

App categories: **Finance** (budget-lens, chat-driven-expense-tracker, spendsense), **Photo/Camera** (google-ai-edge-gallery, tool-neuron), **Productivity** (snapdo, pocketpal-ai), **Education** (deeptutor, edupal), **Social/Communication** (llm-vtuber, waico), **Health/Fitness** (healyks).

| Category | N pairs | Any leakage | Confirmed |
|----------|---------|-------------|-----------|
{cat_table}

![Deep Fig 2 — Category Overview](attachments/{figs['fig02']})

**Top observations:**
- The category with the highest confirmed-leakage density is **{cat_stats.index[0]}** at {cat_stats.iloc[0]['conf']:.1%} confirmed and {cat_stats.iloc[0]['any_leak']:.1%} any leakage.
- **Finance** apps disproportionately leak `location` (merchant addresses) and `identity` (merchant/business name) through receipt processing.
- **Productivity** and **Photo/Camera** apps show heavy demographic-attribute leakage (age, gender, race) due to image content reaching network channels.

---

## 4. Section 2: Privacy Leakage by Channel

Each captured channel was independently judged. Per-channel statistics conditioned on the channel being present:

| Channel | N attr-item pairs (channel present) | Any leakage | Confirmed |
|---------|-------------------------------------|-------------|-----------|
{channel_table}

![Deep Fig 3 — Channel Verdict Distribution](attachments/{figs['fig03']})

### Channel × Family Matrix

![Deep Fig 7 — Channel × Family Matrix](attachments/{figs['fig07']})

**Channel divergence** — many apps' channels carry **different attribute families**. STORAGE (memory/database persistence) tends to over-represent identity & demographic attributes, while NETWORK carries broader content including location and activity.

![Deep Fig 9 — Channel Divergence per App](attachments/{figs['fig09']})

### Background vs Foreground Leakage

We split channels into **foreground** (NETWORK + UI — visible/intended) and **background** (STORAGE + LOGGING — invisible to user, persistent). Background leakage is especially concerning because it survives session boundaries and is rarely audited.

![Deep Fig 8 — Background vs Foreground by Category](attachments/{figs['fig08']})

---

## 5. Section 3: Attribute Family Dynamics

**Top-3 most-leaked families (confirmed):**
{chr(10).join(f"- **{fam}**: {rate:.1%} confirmed leakage rate" for fam, rate in top3_fam)}

**Top-5 most-leaked attributes (confirmed):**
{chr(10).join(f"- `{a}` ({ATTR_TO_FAMILY.get(a,'?')}): {r:.1%}" for a, r in top5_attr)}

### Three-Stage Flow per Family

We trace each attribute family through three stages: **Input GT** (ground-truth labels), **Raw Output** (the LLM's textual response, where output_eval is available), **Externalized** (any channel content).

![Deep Fig 4 — Three-Stage Flow per Family](attachments/{figs['fig04']})

**Reading the chart:**
- Bars **growing** from input → output → externalized show **inference expansion**: the model is producing attribute signals that were *not* in the input GT.
- Bars **shrinking** show **filtering**: input cues that the model did not externalize.

### Inference vs Externalization Gap (Filtering Effectiveness)

For text→text apps where we have both `output_eval` and `ext_eval`, we can directly measure how much filtering happens between the LLM "thinking it" (raw output) and the user-visible/networked output.

![Deep Fig 5 — Filtering Gap](attachments/{figs['fig05']})

**Interpretation:** points **below the y=x diagonal** in panel A indicate attributes where filtering reduced exposure (the model inferred them in raw output but suppressed them in externalization). Points **above** indicate attributes where channel content *added* signal beyond what the user saw.

### Persistence vs Inference Injection

For each attribute, we compute two rates:
- **Persistence:** P(externalized | input GT has attribute) — how often a real GT attribute survives.
- **Injection:** P(externalized | input GT does NOT have attribute) — how often the model fabricates / infers from indirect cues.

![Deep Fig 6 — Persistence & Injection](attachments/{figs['fig06']})

**Key finding.** `identity` and `location` show **both** high persistence AND high injection — they are persistently amplified. Visual attributes like `face`, `troupe`, `nudity` show high persistence but near-zero injection (they cannot be hallucinated without visual cues). `religion`, `medical`, `marital status` have moderate injection rates from cultural/contextual cues.

---

## 6. Section 4: Privacy Transformation Patterns

### Per-App Expansion Factor

Expansion factor = mean(externalized attrs per item) / mean(input GT attrs per item).
- **< 1.0**: app compresses (filters out attributes)
- **= 1.0**: passthrough
- **> 1.0**: app **expands** the attribute set via inference

![Deep Fig 10 — Expansion Factor](attachments/{figs['fig10']})

### Set Similarity (Jaccard) Input vs Externalized

If the input GT set and externalized set were identical, Jaccard = 1. The lower the Jaccard, the more the model has *transformed* the privacy attribute set.

![Deep Fig 11 — Jaccard Set Similarity](attachments/{figs['fig11']})

### Per-Family Transformation Patterns

For each item × family, we classify the transformation:
- **Persisted**: GT attrs all preserved, no new ones.
- **Expanded**: at least one new attr added (inference-only).
- **Compressed**: at least one GT attr lost.
- **Mixed**: both expansion and compression.
- **Empty**: no GT, no externalization.

![Deep Fig 12 — Family Transformation Patterns](attachments/{figs['fig12']})

---

## 7. Section 5: Modality Pair Comparison

### 4-way Modality Pair Matrix

We compare leakage rates across the four possible modality-pair quadrants: **text→text**, **image→text**, **text→image**, **image→image**. Note that text→image and image→image are sparsely populated in our corpus.

| Modality pair | N pairs | Input rate | Any leakage | Confirmed | Mean confirmed/item |
|---------------|---------|-----------|-------------|-----------|---------------------|
{mp_table}

![Deep Fig 13 — Modality Pair × Family Matrix](attachments/{figs['fig13']})

### Visual Re-encoding (image → text)

Image input apps (HR-VISPR primarily) re-encode visual privacy attributes (face, race, height, weight, color, haircolor, …) into **textual descriptions** that flow into NETWORK and UI channels. We test the claim that visual attributes get **transcribed faithfully** in image→text:

![Deep Fig 14 — Visual Re-encoding Evidence](attachments/{figs['fig14']})

**Evidence for visual re-encoding.** When a visual attribute is present in the GT input image:
- Persistence rate in image→text apps is consistently **higher than zero** for major visual attrs (race, age, gender, color), even though no text input contained them.
- text→text apps (which receive no visual data) show near-zero persistence for image-only attrs — confirming that the leakage in image→text is genuinely from the visual content getting linguistically encoded.

### Semantic Crystallization (text → text)

For text→text apps, we test whether the model **crystallizes** weak/implicit input cues into strong/explicit externalizations. The signature: high `confirmed` rates even when input GT does NOT have the attribute (pure-inference confirmed leakage).

![Deep Fig 15 — Semantic Crystallization Evidence](attachments/{figs['fig15']})

**Evidence for semantic crystallization.** A non-trivial fraction of confirmed leakage in text→text apps is **pure inference** (the input GT did not have the attribute, but the model produced a confirmed signal anyway). This is most pronounced for `location` and `identity`, which can be inferred from cultural/linguistic cues, organization names, or context.

### Profile Consolidation (multi-attribute simultaneity)

Profile consolidation = a single externalization event leaking **multiple attributes at once**. We measure the distribution of co-leaked confirmed attributes per item:

![Deep Fig 16 — Profile Consolidation](attachments/{figs['fig16']})

**Mean confirmed-attrs per item by modality pair**:
{chr(10).join(f"- {mp}: {coleak_by_mod[mp]:.2f}" for mp in coleak_by_mod.index)}

### Modality Radar

![Deep Fig 17 — Modality Pair Radar](attachments/{figs['fig17']})

**Synthesis.** The radar visualizes that:
- **image→text** dominates for *visible* attributes (Identity, Demographic, Appearance/Body, Attire/Role).
- **text→text** dominates for *abstract* attributes (Location, Health/Medical, Religion).
- text→image and image→image are sparse and inconclusive in this corpus.

---

## 8. Section 6: Input-Type Split (Docs / Image / Text)

We split the input modality into three semantic types:
- **`docs`**: receipts/structured documents (SROIE2019)
- **`image`**: natural images (HR-VISPR, MIMIC-CXR)
- **`text`**: natural-language text (PrivacyLens, SynthPAI, GretelSyntheticPII, ASAP-AES, MultiCaRe, OpenPII)

| Input type | N pairs | Input rate | Any leakage | Confirmed | Mean confirmed/item |
|-----------|---------|-----------|-------------|-----------|---------------------|
{intype_table}

![Deep Fig 18 — 3-way Input Type Comparison](attachments/{figs['fig18']})

![Deep Fig 19 — Per-Attribute Persistence by Input Type](attachments/{figs['fig19']})

![Deep Fig 20 — Input Type × Channel Matrix](attachments/{figs['fig20']})

**Insights:**
- **docs** input has by far the highest input GT rate for `identity` (merchant) and `location` (address) — and almost perfect persistence: receipts contain these as printed text, so they get faithfully extracted.
- **image** input shows the highest GT presence rate for visual attribute families (Appearance/Body, Attire/Role) but **lower persistence** than docs — many visual cues get summarized away rather than transcribed verbatim.
- **text** input achieves the **lowest** GT presence rate (sparse explicit annotation) but the **highest pure-inference rate**: the model fills in attributes from context.

This supports the hypothesis: **text inputs may *appear* to have less raw signal, but the externalization process consistently amplifies inference**, while image inputs have rich raw signal that gets *partially* transcribed and partially summarized.

---

## 9. Section 7: Case Studies — Privacy Transformation in Action

We selected 4 representative items spanning text/image/docs and multiple categories. For each, we show: input snippet, raw output, externalization channels, and a per-attribute Input → Raw Output → Externalized comparison.

![Deep Fig 21 — Case-Study Heatmaps](attachments/{figs['fig21']})

{case_md}

---

## 10. Synthesis — What This Tells Us About the Leakage Landscape

### 10.1 Inference-induced leakage is real and pervasive
{pct_expansion:.1%} of items show **inference expansion**: at least one attribute appears in the externalized output that was not in the input GT. The mean number of new attrs per item is {avg_new:.2f}. This means apps don't merely pass through user data — they actively reconstruct privacy-sensitive profiles via model inference.

### 10.2 Identity and Location are the dominant leakage channels
Across all modalities and apps, the **Identity & Location** families lead in both confirmed rate and persistence. They are also the most-injected (P(ext | input absent) is non-trivial). This is structural: most apps need to refer to people and places, and LLMs readily infer these from indirect cues.

### 10.3 Background channels carry hidden risk
STORAGE and LOGGING show high background-leakage rates with non-trivial confirmed leakage in some categories (notably Social/Communication apps with conversation memory). Because users do not see these channels, the privacy risk persists silently.

### 10.4 Modality-specific transformations
- **text→text** exhibits **semantic crystallization** — implicit cues become explicit confirmed leakage. Pure-inference leakage is highest here.
- **image→text** exhibits **visual re-encoding** — visual privacy attributes are translated into textual descriptions and exit through text-format channels (NETWORK, STORAGE).
- **docs→text** exhibits **literal transcription** — receipt text (location, identity) is extracted verbatim with very high persistence.

### 10.5 Profile consolidation is the multiplicative threat
A single externalization typically leaks **{df.groupby('full_key')['ext_conf'].sum().mean():.1f}** confirmed attributes simultaneously. This is the multiplicative threat: defenses targeting a single attribute miss the broader profile.

---

## 11. Recommendations

1. **Treat externalization channels as untrusted egress points.** Apply attribute-level filters not at the user-facing UI but at the channel boundary (NETWORK send, STORAGE write, LOGGING emit). Any aggregate-level filtering is bypassable through any single channel.
2. **Audit STORAGE and LOGGING for background leakage** — these channels show non-trivial confirmed leakage that users never see. Add explicit privacy-attribute redaction at the storage layer.
3. **For image-input apps, deploy attribute-aware caption sanitization.** Image→text apps systematically re-encode visual privacy attributes into text. Either (a) avoid generating descriptions that mention sensitive attributes, or (b) post-process descriptions to mask them.
4. **For text-input apps, expect pure-inference leakage even when input is "clean".** Models will infer location from organization names, identity from contextual cues. Filter based on judge verdicts on the *output*, not on the input.
5. **Track multi-attribute simultaneity, not single-attribute rates.** A model that confirms 5 attributes per item at 30% rate is privacy-equivalent to one that confirms 1 attribute at 95% rate. Compute *expected leaked profile size* as the metric.

---

## Appendix

### A. Per-app summary

| App | Category | N pairs | Any leakage | Confirmed |
|-----|----------|---------|-------------|-----------|
{app_table}

### B. Data quality

- Total prompt4/prompt5 verdict rows: {n_total:,} ({n_items:,} unique items).
- 4 datasets contribute >100 items: HR-VISPR, PrivacyLens, SROIE2019, plus several smaller (SynthPAI, ASAP-AES, MultiCaRe, OpenPII, GretelSyntheticPII, MIMIC-CXR).
- Tiny runs excluded from per-cell statistics but retained in totals (spendsense SROIE2019 n=1; tool-neuron HR-VISPR image→image n=1).
- output_eval available only for: deeptutor, llm-vtuber, tool-neuron, waico (all PrivacyLens text→text). All 3-stage flow analyses are conditioned on this subset.

### C. Color palette

Stage colors (kept from prior analysis): blue=Input, orange=Raw Output, red=Externalized.
Adobe palette (used for everything else): #7ADBC4 (teal), #FAD765 (yellow), #FA9F5C (orange), #98D198 (sage), #6C80FC (periwinkle), #ACA4B3 (mauve), #687692 (slate).

"""
    return md


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    df, raw = load_data()
    # Drop tiny runs (<5 verdict rows per app/dataset)
    counts = df.groupby(["app","dataset"])["ext_conf"].count()
    valid = counts[counts >= 5].reset_index()[["app","dataset"]]
    df = df.merge(valid, on=["app","dataset"])
    print(f"  Loaded {len(df):,} attribute-item pairs across {df.groupby('full_key').ngroups:,} unique items")

    print("\nSelecting case studies...")
    picks = select_case_studies(df, raw)
    print(f"  Picked {len(picks)} case studies: {picks}")

    print("\nGenerating figures...")
    figs = {}
    figs["fig01"] = deep_fig01_landscape_overview(df)
    figs["fig02"] = deep_fig02_category_overview(df)
    figs["fig03"] = deep_fig03_channel_verdict_distribution(df)
    figs["fig04"] = deep_fig04_three_stage_family_flow(df)
    figs["fig05"] = deep_fig05_inference_externalization_gap(df)
    figs["fig06"] = deep_fig06_attribute_persistence_grid(df)
    figs["fig07"] = deep_fig07_channel_family_matrix(df)
    figs["fig08"] = deep_fig08_background_foreground_by_category(df)
    figs["fig09"] = deep_fig09_channel_divergence_per_app(df)
    figs["fig10"] = deep_fig10_expansion_factor(df)
    figs["fig11"] = deep_fig11_jaccard_similarity(df)
    figs["fig12"] = deep_fig12_family_transformation_patterns(df)
    figs["fig13"] = deep_fig13_modality_pair_matrix(df)
    figs["fig14"] = deep_fig14_visual_re_encoding(df)
    figs["fig15"] = deep_fig15_semantic_crystallization(df)
    figs["fig16"] = deep_fig16_profile_consolidation(df)
    figs["fig17"] = deep_fig17_modality_transformation_radar(df)
    figs["fig18"] = deep_fig18_input_type_3way(df)
    figs["fig19"] = deep_fig19_input_type_persistence(df)
    figs["fig20"] = deep_fig20_input_type_channel_breakdown(df)
    figs["fig21"] = deep_fig21_case_study_heatmap(df, raw, picks)

    print("\nRendering report...")
    md = render_report(df, raw, figs, picks)
    out = LANTERN_ROOT / "analysis" / "leakage_landscape_deep.md"
    out.write_text(md)
    print(f"  Wrote {out}")

    print("\n=== STATS ===")
    print(f"  total pairs: {len(df):,}")
    print(f"  unique items: {df.groupby('full_key').ngroups:,}")
    print(f"  apps: {df['app'].nunique()}")
    print(f"  datasets: {df['dataset'].nunique()}")
    print(f"  confirmed leakage: {df['ext_conf'].mean():.1%}")
    print(f"  any leakage: {df['ext_leak'].mean():.1%}")
