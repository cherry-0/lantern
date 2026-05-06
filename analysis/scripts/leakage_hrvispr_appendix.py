"""
HR-VISPR appendix figures: stage gap and injection/retention.

Two figures for the HR-VISPR subset of the leakage corpus:

  Fig A: Per-app stage gap — raw-output inferability vs. any-leak vs. confirmed
          externalization for each of the 5 HR-VISPR apps. Mirrors Fig. 7(a)
          but restricted to image→text items.

  Fig B: Per-attribute retention and injection rates across all HR-VISPR apps.
          Retention = P(ext_conf | input_label=1).
          Injection  = P(ext_conf | input_label=0).
          Mirrors Table 1 in the paper but as a figure and over visual attributes.

Outputs (saved to paper/figures/ and analysis/attachments/):
  appx_hrvispr_stage_gap_{1x1,2x1}.png
  appx_hrvispr_retention_injection_{1x1,2x1}.png

Usage:
  conda run -n lantern python analysis/scripts/leakage_hrvispr_appendix.py
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

SCRIPT_DIR = Path(__file__).resolve().parent
LANTERN_ROOT = SCRIPT_DIR.parent.parent
FIG_DIR     = LANTERN_ROOT / "paper" / "26_CCS_Lantern" / "figures"
ATTACH_DIR  = LANTERN_ROOT / "analysis" / "attachments"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ATTACH_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from _leakage_common import load_data, S_INPUT, P_ORANGE, S_EXT  # noqa: E402

ASPECTS = {"1x1": (7.5, 2.5), "2x1": (13.0, 2.5)}

APP_LABEL = {
    "clone":                  "Clone",
    "google-ai-edge-gallery": "AI Edge Gallery",
    "momentag":               "Momentag",
    "snapdo":                 "Snapdo",
    "tool-neuron":            "Tool Neuron",
}

STAGE_PATCHES = [
    mpatches.Patch(color=S_INPUT,  label="Raw output inferable"),
    mpatches.Patch(color=P_ORANGE, label="Any-leak externalized"),
    mpatches.Patch(color=S_EXT,    label="Confirmed externalized"),
]


def _save(fig: plt.Figure, slug: str) -> None:
    for d in (ATTACH_DIR, FIG_DIR):
        path = d / f"{slug}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        print(f"  saved {path.relative_to(LANTERN_ROOT)}")
    plt.close(fig)


# ── Fig A: per-app stage gap ──────────────────────────────────────────────────

def fig_stage_gap(hrv: pd.DataFrame, aspect: str) -> None:
    apps = [a for a in APP_LABEL if a in hrv["app"].unique()]
    rows = []
    for app in apps:
        sub = hrv[hrv["app"] == app]
        out_known = sub[sub["output_has"] == 1]
        rows.append({
            "app":  app,
            "raw":  out_known["output_leak"].mean() if len(out_known) else 0.0,
            "any":  sub["ext_leak"].mean(),
            "conf": sub["ext_conf"].mean(),
        })
    adf = pd.DataFrame(rows)

    w, h = ASPECTS[aspect]
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    x  = np.arange(len(apps))
    bw = 0.24
    for j, (key, color) in enumerate([("raw", S_INPUT), ("any", P_ORANGE), ("conf", S_EXT)]):
        vals = adf[key].values
        bars = ax.bar(x + (j - 1) * bw, vals, bw, color=color,
                      alpha=0.92, edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.0%}", ha="center", va="bottom",
                    fontsize=8 if aspect == "2x1" else 7,
                    fontweight="bold" if key == "conf" else "normal",
                    color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([APP_LABEL[a] for a in apps], fontsize=10)
    ax.set_ylabel("Rate over (item, attribute) pairs")
    ax.set_ylim(0, adf[["raw", "any", "conf"]].max().max() * 1.28)
    ax.legend(handles=STAGE_PATCHES, loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("white")
    fig.tight_layout()
    _save(fig, f"appx_hrvispr_stage_gap_{aspect}")


# ── Fig B: per-attribute retention and injection ──────────────────────────────

RET_COLOR = "#4E79A7"
INJ_COLOR = "#E15759"

def fig_retention_injection(hrv: pd.DataFrame, aspect: str) -> None:
    attrs = sorted(hrv["attr"].unique())
    rows = []
    for attr in attrs:
        sub = hrv[hrv["attr"] == attr]
        gt1 = sub[sub["input_label"] == 1]
        gt0 = sub[sub["input_label"] == 0]
        n_gt1 = len(gt1)
        n_gt0 = len(gt0)
        rows.append({
            "attr":      attr,
            "retention": gt1["ext_conf"].mean() if n_gt1 else 0.0,
            "injection": gt0["ext_conf"].mean() if n_gt0 else 0.0,
            "n_gt1":     n_gt1,
        })
    rdf = pd.DataFrame(rows).sort_values("retention", ascending=False)
    # drop attrs with zero gt1 support (no input GT positive, retention undefined)
    rdf = rdf[rdf["n_gt1"] > 0].reset_index(drop=True)

    w, h = ASPECTS[aspect]
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    x  = np.arange(len(rdf))
    bw = 0.35

    ax.bar(x - bw / 2, rdf["retention"], bw, color=RET_COLOR, alpha=0.90,
           edgecolor="white", linewidth=0.5, label="Retention (GT=1 → confirmed ext.)")
    ax.bar(x + bw / 2, rdf["injection"], bw, color=INJ_COLOR, alpha=0.90,
           edgecolor="white", linewidth=0.5, label="Injection (GT=0 → confirmed ext.)")

    fs = 8.5 if aspect == "2x1" else 7.5
    ax.set_xticks(x)
    ax.set_xticklabels(rdf["attr"], rotation=38, ha="right", fontsize=fs)
    ax.set_ylabel("Confirmed externalization rate")
    ax.set_ylim(0, max(rdf[["retention", "injection"]].max().max(), 0.02) * 1.30)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("white")
    fig.tight_layout()
    _save(fig, f"appx_hrvispr_retention_injection_{aspect}")


def main() -> None:
    df, _ = load_data()
    hrv = df[df["dataset"] == "HR-VISPR"].copy()
    print(f"HR-VISPR: {hrv['filename'].nunique()} items, "
          f"{hrv['app'].nunique()} apps, {hrv['attr'].nunique()} attrs")

    for aspect in ASPECTS:
        fig_stage_gap(hrv, aspect)
        fig_retention_injection(hrv, aspect)

    print("Done.")


if __name__ == "__main__":
    main()
