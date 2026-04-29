"""
Judge Validation Analysis — figure regeneration.

Follows analysis/FIGURE.md:
  - Adobe pastel palette (P_*).
  - White figure & axes background.
  - One plot per PNG (no panel concatenation).
  - Two aspect ratios per plot: 1:1 (6"x6") and 2:1 (12"x6").

Data points are hard-coded from judge_analysis.pdf §3 (Headline Metrics) and
Appendix B (per-attribute breakdown for HR-VISPR / judge-image / Run E).

Usage:  python analysis/scripts/judge_analysis.py
Outputs:
  analysis/attachments/<slug>_1x1.png
  analysis/attachments/<slug>_2x1.png
  paper/26_CCS_Lantern/figures/judge_<slug>_{1x1,2x1}.png
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
ATTACH_DIR   = LANTERN_ROOT / "analysis" / "attachments"
PAPER_FIGS   = LANTERN_ROOT / "paper" / "26_CCS_Lantern" / "figures"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# ── Style (FIGURE.md §1, §2) ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
# Adobe pastel palette
P_TEAL   = "#7ADBC4"
P_YELLOW = "#FAD765"
P_ORANGE = "#FA9F5C"
P_GREEN  = "#98D198"
P_BLUE   = "#6C80FC"
P_MAUVE  = "#ACA4B3"
P_SLATE  = "#687692"
# Reserved stage colors (input / raw output / externalized) — for stage flow only.
STAGE_BLUE   = "#3A7DC9"
STAGE_ORANGE = "#E8762C"
STAGE_RED    = "#C93A3A"

# Aspect ratios required by FIGURE.md §4
ASPECTS = {
    "1x1": (6.0, 6.0),    # square
    "2x1": (12.0, 6.0),   # horizontal: width = 2 * height
}


def _apply_axes_chrome(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, slug):
    """Save the current figure under the named slug only (one aspect ratio at a time)."""
    out_path = ATTACH_DIR / f"{slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paper_path = PAPER_FIGS / f"judge_{slug}.png"
    paper_path.write_bytes(out_path.read_bytes())
    print(f"  Saved: {out_path.name}  ->  {paper_path.relative_to(LANTERN_ROOT)}")


def render_both_aspects(slug, draw_fn):
    """Run draw_fn once per required aspect ratio and save under <slug>_<aspect>.png."""
    for aspect_key, (w, h) in ASPECTS.items():
        fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
        _apply_axes_chrome(ax)
        draw_fn(ax, aspect=aspect_key)
        save(fig, f"{slug}_{aspect_key}")


# ── Fig 1 — Evaluator Configuration Comparison ────────────────────────────────
def _draw_fig1_modality_gap(ax, aspect: str):
    """Recall_lb and precision_confirmed for HR-VISPR & OpenPII under judge-text vs judge-image."""
    rows = [
        # (dataset, evaluator, recall_lb, precision)
        ("HR-VISPR\njudge-text / flash-lite",   "judge-text",  0.176, 0.698),
        ("HR-VISPR\njudge-image / flash",       "judge-image", 0.669, 0.762),
        ("OpenPII\njudge-text / flash-lite",    "judge-text",  0.836, 0.651),
        ("OpenPII\njudge-image / flash",        "judge-image", 0.841, 0.694),
    ]
    labels  = [r[0] for r in rows]
    evals   = [r[1] for r in rows]
    recalls = [r[2] for r in rows]
    precs   = [r[3] for r in rows]

    x = np.arange(len(labels))
    w = 0.36
    REC_HUE, PRC_HUE = P_BLUE, P_ORANGE
    rec_alphas = [1.0 if e == "judge-image" else 0.40 for e in evals]
    prc_alphas = [1.0 if e == "judge-image" else 0.40 for e in evals]

    for i in range(len(labels)):
        ax.bar(x[i] - w/2, recalls[i], w,
               color=REC_HUE, alpha=rec_alphas[i],
               edgecolor="white", linewidth=0.6)
        ax.bar(x[i] + w/2, precs[i], w,
               color=PRC_HUE, alpha=prc_alphas[i],
               edgecolor="white", linewidth=0.6)
    for i, v in enumerate(recalls):
        ax.text(x[i] - w/2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333")
    for i, v in enumerate(precs):
        ax.text(x[i] + w/2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333")

    # 3.8× gap callout (bars 0 → 1, recall_lb on HR-VISPR; 0.669 / 0.176 ≈ 3.8)
    x0 = x[0] - w/2; x1 = x[1] - w/2
    y0 = recalls[0]; y1 = recalls[1]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=P_SLATE, lw=1.8,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text((x0 + x1) / 2 - 0.02, (y0 + y1) / 2 + 0.10,
            "3.8× recall gap", color=P_SLATE, fontsize=10, fontweight="bold",
            ha="center")

    label_fs = 9.5 if aspect == "2x1" else 8.5
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_fs)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.12)
    ax.set_title("Evaluator Configuration Comparison\n"
                 "(dark = judge-image, light = judge-text)",
                 fontsize=11.5)

    legend_handles = [
        mpatches.Patch(color=P_BLUE,   alpha=1.0,  label="recall_lb (judge-image)"),
        mpatches.Patch(color=P_BLUE,   alpha=0.40, label="recall_lb (judge-text)"),
        mpatches.Patch(color=P_ORANGE, alpha=1.0,  label="precision_confirmed (judge-image)"),
        mpatches.Patch(color=P_ORANGE, alpha=0.40, label="precision_confirmed (judge-text)"),
    ]
    ncol = 2 if aspect == "2x1" else 1
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.5,
              ncol=ncol, framealpha=0.92)


def fig1_modality_gap():
    render_both_aspects("fig1_modality_gap", _draw_fig1_modality_gap)


# ── Fig 2 — Judge Discrimination: TPR(GT=1) vs FPR(GT=0) ──────────────────────
def _draw_fig2_discrimination(ax, aspect: str):
    """Per-config: recall on GT=1 stratum (TPR) vs false-fire rate on GT=0 stratum (FPR).

    The original "precision on GT=1 stratum" framing was structurally trivial —
    in a GT=1 stratum, FPs are impossible by definition, so precision = 1.000
    is mechanical. The non-trivial discrimination metric is the gap between
    TPR (rate at which the judge fires when attribute is present) and FPR
    (rate at which it fires when attribute is absent).
    """
    rows = [
        # (label, TPR_GT1, FPR_GT0, n_GT_pos, n_GT_neg)
        ("OpenPII\njudge-text / flash-lite",    0.836, 0.226,   67,  133),
        ("OpenPII\njudge-image / flash",        0.841, 0.183,  132,  268),
        ("HR-VISPR\njudge-text / flash-lite",   0.176, 0.055,  210,  290),
        ("HR-VISPR\njudge-image / flash",       0.669, 0.144,  734, 1065),
    ]
    labels   = [r[0] for r in rows]
    tpr      = [r[1] for r in rows]
    fpr      = [r[2] for r in rows]

    x = np.arange(len(labels))
    w = 0.36
    bars_t = ax.bar(x - w/2, tpr, w, color=P_GREEN, alpha=1.0,
                    edgecolor="white", linewidth=0.6,
                    label="TPR on GT=1  (judge fires when attribute is present)")
    bars_f = ax.bar(x + w/2, fpr, w, color=P_ORANGE, alpha=1.0,
                    edgecolor="white", linewidth=0.6,
                    label="FPR on GT=0  (judge fires when attribute is absent)")
    for bar, v in zip(bars_t, tpr):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#3F7A5A")
    for bar, v in zip(bars_f, fpr):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#9C5A2A")

    # Annotate the discrimination gap (TPR - FPR) inside each x cluster, between the bars
    for i in range(len(labels)):
        gap = tpr[i] - fpr[i]
        gap_y = max(tpr[i], fpr[i]) + 0.085
        ax.annotate(f"gap = {gap:+.2f}",
                    xy=(x[i], gap_y),
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=P_SLATE)

    label_fs = 10 if aspect == "2x1" else 9
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_fs)
    ax.set_ylabel("Rate at which the judge issues a 'confirmed' verdict")
    ax.set_ylim(0, 1.12)
    ax.set_title("Judge Discrimination: TPR on GT=1 vs FPR on GT=0\n"
                 "(large gap = the judge fires on present attributes far more often than on absent ones)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)


def fig2_precision_calibration():
    render_both_aspects("fig2_precision_calibration", _draw_fig2_discrimination)


# ── Fig 3 — Attribute Sensitivity Hierarchy ────────────────────────────────────
def _draw_fig3_attribute_sensitivity(ax, aspect: str):
    """Per-attribute recall_lb on HR-VISPR / judge-image / canonical run (n_ok=1799, GT+ only)."""
    data = [
        # (attribute, recall, n_GT_pos)
        ("disability",      1.000,   1),
        ("medical",         1.000,   8),
        ("sports",          1.000,   9),
        ("troupe",          1.000,   2),
        ("uniforms",        1.000,   4),
        ("face",            0.989,  88),
        ("casual",          0.958,  48),
        ("color",           0.935,  93),
        ("formal",          0.933,  15),
        ("haircolor",       0.786,  84),
        ("nudity",          0.727,  11),
        ("religion",        0.667,   3),
        ("race",            0.593,  91),
        ("ethnic_clothing", 0.500,   2),
        ("weight",          0.480,  50),
        ("gender",          0.457,  94),
        ("age",             0.366,  93),
        ("height",          0.026,  38),
    ]
    data_sorted = sorted(data, key=lambda r: r[1])
    attrs   = [r[0] for r in data_sorted]
    recalls = [r[1] for r in data_sorted]
    ns      = [r[2] for r in data_sorted]

    def tier_color(r):
        if r >= 0.90: return P_GREEN
        if r >= 0.60: return P_TEAL
        if r >= 0.30: return P_YELLOW
        return P_ORANGE

    colors = [tier_color(r) for r in recalls]

    y = np.arange(len(attrs))
    ax.barh(y, recalls, color=colors, edgecolor="white", linewidth=0.6)
    for i, (v, n) in enumerate(zip(recalls, ns)):
        ax.text(max(v + 0.015, 0.02), i,
                f"{v:.2f}  (n={n})", va="center", fontsize=9.5, color="#333")

    ax.axvline(1.0, color=P_SLATE, linestyle="--", linewidth=0.9, alpha=0.45)

    ax.set_yticks(y)
    ax.set_yticklabels(attrs, fontsize=10)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("recall_lb")
    ax.set_title("Attribute Sensitivity Hierarchy\n"
                 "HR-VISPR • judge-image • gemini-2.0-flash-001  "
                 "(n_ok=1,799, GT+ items only)",
                 fontsize=11)

    legend_handles = [
        mpatches.Patch(color=P_GREEN,  label="≥ 0.90  (very high)"),
        mpatches.Patch(color=P_TEAL,   label="0.60–0.89  (high)"),
        mpatches.Patch(color=P_YELLOW, label="0.30–0.59  (medium)"),
        mpatches.Patch(color=P_ORANGE, label="< 0.30  (low / zero)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              title="Recall tier", fontsize=8.5, title_fontsize=9,
              framealpha=0.92)


def fig3_attribute_sensitivity():
    render_both_aspects("fig3_attribute_sensitivity", _draw_fig3_attribute_sensitivity)


# ── Fig 4 — Per-attribute (GT × Verdict) Heatmaps ─────────────────────────────
# One heatmap per HR-VISPR attribute on Run E (judge-image / gemini-2.0-flash-001,
# n_ok=170).  Rows: GT label (absent / present).  Columns: judge verdict
# (confirmed / possible / no evidence).  Color intensity is row-normalized so
# each row reads as a probability distribution; the per-attribute hue varies
# through the primary palette so the figure family is visually distinguishable.
import json as _json
from matplotlib.colors import LinearSegmentedColormap as _LSC

JUDGE_RUN_E_DIR = (LANTERN_ROOT / "verify" / "outputs" / "judge_validation_runs"
                   / "judge_validation_20260429_224412_030164")

_PRIMARY_PALETTE = [P_TEAL, P_YELLOW, P_ORANGE, P_GREEN, P_BLUE, P_MAUVE, P_SLATE]
_VERDICT_LABELS  = ["confirmed", "possible", "none"]
_VERDICT_DISPLAY = ["confirmed", "possible", "no evidence"]


def _load_run_E_items():
    with open(JUDGE_RUN_E_DIR / "results.json") as fh:
        results = _json.load(fh)
    return [r for r in results if r.get("judge_ok")]


def _draw_attribute_heatmap(ax, attr, mat, color_hex, n_neg, n_pos, aspect):
    """mat shape (2, 3): rows = [GT=0, GT=1], cols = [confirmed, possible, none]."""
    cmap = _LSC.from_list(f"white_{attr}", ["#FFFFFF", color_hex])

    row_totals = mat.sum(axis=1, keepdims=True).astype(float)
    rates = np.divide(mat, row_totals,
                      where=row_totals > 0,
                      out=np.zeros_like(mat, dtype=float))

    ax.imshow(rates, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)

    row_labels = [f"absent\n(n={n_neg})", f"present\n(n={n_pos})"]
    ax.set_xticks(range(3))
    ax.set_xticklabels(_VERDICT_DISPLAY, fontsize=11)
    ax.set_yticks(range(2))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Judge verdict", fontsize=11)
    ax.set_ylabel("Ground truth", fontsize=11)
    ax.set_title(f"{attr}", fontsize=14, fontweight="bold")

    for i in range(2):
        for j in range(3):
            count = int(mat[i, j])
            rate  = float(rates[i, j])
            text_color = "#FFFFFF" if rate >= 0.60 else "#222"
            row_total = int(row_totals[i, 0])
            cell = f"{count}\n({rate:.0%})" if row_total > 0 else f"{count}\n(--)"
            ax.text(j, i, cell, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)


def fig4_attribute_heatmaps():
    items = _load_run_E_items()
    attrs = sorted({it["attribute"] for it in items})
    attr_color = {a: _PRIMARY_PALETTE[i % len(_PRIMARY_PALETTE)]
                  for i, a in enumerate(attrs)}

    print(f"  Per-attribute heatmaps for {len(attrs)} attributes (Run E, n_ok={len(items)})")
    for attr in attrs:
        sub = [it for it in items if it["attribute"] == attr]
        mat = np.zeros((2, 3), dtype=int)
        for it in sub:
            gt = int(it["ground_truth"])
            verdict = it.get("label")
            if verdict not in _VERDICT_LABELS:
                continue
            mat[gt, _VERDICT_LABELS.index(verdict)] += 1
        n_neg = int(mat[0].sum())
        n_pos = int(mat[1].sum())
        slug = f"fig4_attr_{attr}"
        for aspect_key, (w, h) in ASPECTS.items():
            fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
            _apply_axes_chrome(ax)
            _draw_attribute_heatmap(ax, attr, mat, attr_color[attr],
                                    n_neg, n_pos, aspect_key)
            save(fig, f"{slug}_{aspect_key}")


# ── Driver ─────────────────────────────────────────────────────────────────────
def main():
    print("Generating judge-validation figures (Adobe palette, white bg, 1x1 + 2x1)...")
    fig1_modality_gap()
    fig2_precision_calibration()
    fig3_attribute_sensitivity()
    fig4_attribute_heatmaps()
    print("Done.")


if __name__ == "__main__":
    main()
