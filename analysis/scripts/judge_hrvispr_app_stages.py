"""
HR-VISPR input-to-externalization stage comparison for two apps.

For google-ai-edge-gallery and snapdo (both image→text, HR-VISPR stimulus):
  Stage 1 (Input):        per-attribute confirmed rate from the HR-VISPR
                          judge-validation run (judge evaluated on raw images,
                          n≈100 items, randomly selected)
  Stage 2 (Raw output):   per-attribute output_conf from the leakage pipeline
  Stage 3 (Externalized): per-attribute ext_conf from the leakage pipeline

Purpose: verify that the patterns from §5 (F2 filter gap, F5 modality
transformation) hold at the per-app, per-attribute level.

Usage:  python analysis/scripts/judge_hrvispr_app_stages.py
Outputs (1x1 and 2x1 per app):
  analysis/attachments/judge_appx_hrvispr_stages_{app}_1x1.png
  paper/26_CCS_Lantern/figures/judge_appx_hrvispr_stages_{app}_1x1.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.scripts._leakage_common import load_data

ATTACH_DIR = ROOT / "analysis" / "attachments"
PAPER_FIGS = ROOT / "paper" / "26_CCS_Lantern" / "figures"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

VAL_RUN_DIR = (
    ROOT
    / "verify"
    / "outputs"
    / "judge_validation_runs"
    / "judge_validation_20260429_224412_030164"
)

sns.set_theme(style="white", font_scale=1.5)
plt.rcParams["axes.grid"] = False

STAGE_INPUT = "#3A7DC9"
STAGE_RAW   = "#E8762C"
STAGE_EXT   = "#C93A3A"

ASPECTS = {"1x1": (6.0, 3.0), "2x1": (12.0, 3.0)}

APPS = ["google-ai-edge-gallery", "snapdo"]
APP_LABEL = {
    "google-ai-edge-gallery": "Google AI Edge Gallery",
    "snapdo": "Snapdo",
}


def load_val_input_rates() -> dict[str, float]:
    """Per-attribute overall confirmed rate from the HR-VISPR validation run."""
    with open(VAL_RUN_DIR / "results.json") as f:
        results = json.load(f)
    by_attr: dict[str, list[int]] = {}
    for r in results:
        if not r.get("judge_ok"):
            continue
        attr = str(r.get("attribute") or "")
        confirmed = 1 if r.get("label") == "confirmed" else 0
        by_attr.setdefault(attr, []).append(confirmed)
    return {a: sum(v) / len(v) for a, v in by_attr.items() if v}


def save(fig, slug: str) -> None:
    path = ATTACH_DIR / f"{slug}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    dst = PAPER_FIGS / f"{slug}.png"
    dst.write_bytes(path.read_bytes())
    print(f"  saved {path.name}")


def draw_stages(ax, app: str, df_app, val_rates: dict[str, float], aspect: str) -> None:
    attrs_in_app = set(df_app["attr"].unique())
    # Order by descending input confirmed rate; drop near-zero input attrs
    attrs = [
        a for a in sorted(val_rates, key=lambda x: -val_rates[x])
        if a in attrs_in_app and val_rates[a] > 0.02
    ]

    x   = np.arange(len(attrs))
    w   = 0.26

    input_v = [val_rates[a] for a in attrs]
    raw_v   = [df_app[df_app["attr"] == a]["output_conf"].mean() for a in attrs]
    ext_v   = [df_app[df_app["attr"] == a]["ext_conf"].mean()    for a in attrs]

    ax.bar(x - w, input_v, w, color=STAGE_INPUT, alpha=0.90, label="Input (val run)")
    ax.bar(x,     raw_v,   w, color=STAGE_RAW,   alpha=0.90, label="Raw output")
    ax.bar(x + w, ext_v,   w, color=STAGE_EXT,   alpha=0.90, label="Externalized")

    fs = 7.8 if aspect == "2x1" else 7.0
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, rotation=38, ha="right", fontsize=fs)
    ax.set_ylabel("Confirmed rate")
    ax.set_ylim(0, 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")

    legend_handles = [
        mpatches.Patch(color=STAGE_INPUT, label="Input  (val run, n≈100 items)"),
        mpatches.Patch(color=STAGE_RAW,   label="Raw output  (n=100 items)"),
        mpatches.Patch(color=STAGE_EXT,   label="Externalized  (n=100 items)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7.5 if aspect == "1x1" else 8.5, framealpha=0.92)


N_ITEMS = 100
SEED    = 42


def main() -> None:
    val_rates = load_val_input_rates()
    print(f"Validation run: {len(val_rates)} attributes loaded")

    df, _ = load_data()
    import random

    for app in APPS:
        df_app = df[(df["app"] == app) & (df["dataset"] == "HR-VISPR")].copy()
        # subsample to N_ITEMS so pipeline bars match the val-run sample size
        items = sorted(df_app["filename"].unique())
        random.Random(SEED).shuffle(items)
        df_app = df_app[df_app["filename"].isin(items[:N_ITEMS])]
        n_items = df_app["filename"].nunique()
        print(f"\n{app}: {n_items} items | "
              f"raw_conf={df_app['output_conf'].mean():.3f} | "
              f"ext_conf={df_app['ext_conf'].mean():.3f}")

        slug = f"judge_appx_hrvispr_stages_{app.replace('-', '_')}"
        for aspect, (w, h) in ASPECTS.items():
            fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
            draw_stages(ax, app, df_app, val_rates, aspect)
            plt.tight_layout()
            save(fig, f"{slug}_{aspect}")

    print("\nDone.")


if __name__ == "__main__":
    main()
