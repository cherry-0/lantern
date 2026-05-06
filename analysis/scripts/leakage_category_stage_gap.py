"""
Category-stratified stage-gap plot.

This exploratory figure asks whether the headline stage gap
(raw-output inferability >> externalized leakage) holds inside each app
category. It intentionally writes to the repository-level attachments/
directory. The paper figure generator now uses the same visual encoding for
Fig. 7(b), but this script keeps an attachment copy plus the raw CSV.

Outputs:
  attachments/leakage_category_stage_gap.png
  attachments/leakage_category_stage_gap.csv
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _leakage_common import CATEGORY_COLORS, GRAY, load_data


LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
ATTACH_DIR = LANTERN_ROOT / "attachments"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "font.family": "DejaVu Sans",
})


STAGES = [
    ("raw_output", "Raw output"),
    ("any_ext", "Any ext."),
    ("confirmed_ext", "Confirmed ext."),
]


def _blend_with_white(hex_color: str, white_frac: float) -> str:
    """Return a lighter tone by mixing a hex color with white."""
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    r = r * (1 - white_frac) + white_frac
    g = g * (1 - white_frac) + white_frac
    b = b * (1 - white_frac) + white_frac
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def _darken(hex_color: str, lightness_scale: float = 0.72) -> str:
    """Darken a pastel category color enough for readable text/markers."""
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(hue, max(0, lightness * lightness_scale), saturation)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def _category_stage_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, sub in df.groupby("category"):
        out_known = sub[sub["output_has"] == 1]
        rows.append({
            "category": category,
            "input_gt": sub["input_label"].mean(),
            "raw_output": out_known["output_leak"].mean() if len(out_known) else np.nan,
            "any_ext": sub["ext_leak"].mean(),
            "confirmed_ext": sub["ext_conf"].mean(),
            "n_items": sub["full_key"].nunique(),
            "n_attr_pairs": len(sub),
            "n_output_known_attr_pairs": len(out_known),
        })
    rates = pd.DataFrame(rows)
    return rates.sort_values("confirmed_ext", ascending=False).reset_index(drop=True)


def plot_category_stage_gap(rates: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor="white")
    x = np.arange(len(rates))
    width = 0.24
    offsets = np.linspace(-width, width, len(STAGES))
    white_mix = {
        "raw_output": 0.48,
        "any_ext": 0.28,
        "confirmed_ext": 0.00,
    }

    for i, (key, label) in enumerate(STAGES):
        colors = [
            _blend_with_white(CATEGORY_COLORS.get(cat, GRAY), white_mix[key])
            for cat in rates["category"]
        ]
        bars = ax.bar(
            x + offsets[i],
            rates[key].fillna(0).values,
            width,
            color=colors,
            edgecolor="white",
            linewidth=0.7,
            label=label,
        )
        if key == "any_ext":
            continue
        for bar, value, cat in zip(bars, rates[key], rates["category"]):
            if pd.isna(value) or value < 0.002:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=_darken(CATEGORY_COLORS.get(cat, GRAY)),
                fontweight="bold" if key == "confirmed_ext" else "normal",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(rates["category"], rotation=20, ha="right")
    ax.set_ylabel("Rate over (item, attribute) pairs")
    ax.set_ylim(0, max(rates[[k for k, _ in STAGES]].max().max() * 1.25, 0.05))

    # Use neutral legend swatches because bar hue encodes category and tone
    # encodes stage.
    legend_colors = ["#E6E6E6", "#C8C8C8", "#A8A8A8", "#707070"]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, ec="white", lw=0.7)
        for color in legend_colors
    ]
    ax.legend(handles, [label for _, label in STAGES], loc="upper right", frameon=False)

    fig.subplots_adjust(bottom=0.22)
    out = ATTACH_DIR / "leakage_category_stage_gap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    df, _ = load_data()
    rates = _category_stage_rates(df)
    csv_out = ATTACH_DIR / "leakage_category_stage_gap.csv"
    rates.to_csv(csv_out, index=False)
    fig_out = plot_category_stage_gap(rates)

    audit = df.attrs.get("filter", {})
    print(
        f"Category stage-gap plot: {audit.get('n_configs_kept')} configs / "
        f"{audit.get('n_items_kept')} items"
    )
    print(f"  saved {fig_out.relative_to(LANTERN_ROOT)}")
    print(f"  saved {csv_out.relative_to(LANTERN_ROOT)}")


if __name__ == "__main__":
    main()
