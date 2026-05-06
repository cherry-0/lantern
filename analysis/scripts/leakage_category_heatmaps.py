"""
Per-category three-stage flow heatmaps (altair).

Six heatmaps --- one per app category --- arranged 3x2 in the paper. Each
heatmap has:
  - rows = three pipeline stages: input GT presence, raw output inferable,
    externalized any-leak;
  - columns = the 8 attribute families from the taxonomy;
  - cell value = the per-stage rate (computed from rows of the active
    sampled corpus emitted by ``_leakage_common.load_data``).

Color palette: each category uses its own white -> palette-color ramp drawn
from the same Adobe pastel palette as the rest of the paper, so the six
panels read as visually distinct categorical encodings while keeping the
absolute numeric scale comparable (all ramps share the same domain
[0, max_rate]).

Outputs:
  paper/26_CCS_Lantern/figures/leakage_cat_heatmap_<slug>.png
"""

from __future__ import annotations
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import vl_convert as vlc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _leakage_common import (
    load_data, ATTR_FAMILIES, CATEGORY_COLORS, CATEGORY_COLORS as _CC,
)

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR      = LANTERN_ROOT / "paper" / "26_CCS_Lantern" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Stage labels (rows of each heatmap) ──────────────────────────────────────
STAGES = ["Input GT", "Raw output", "Externalized"]

FAMILY_ORDER = list(ATTR_FAMILIES.keys())

# Per-category 2-stop ramp: white -> the category's palette color. Altair
# expects a list of hex values (interpolation is automatic).
def _ramp(hex_color: str) -> list[str]:
    return ["#FFFFFF", hex_color]

CATEGORY_RAMPS = {cat: _ramp(c) for cat, c in CATEGORY_COLORS.items()}


def _category_long_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Build the long-form (stage, family, rate) table for one category."""
    sub = df[df["category"] == category]
    out_known = sub[sub["output_has"] == 1]
    rows: list[dict] = []
    for fam in FAMILY_ORDER:
        fam_sub      = sub[sub["family"] == fam]
        fam_known    = out_known[out_known["family"] == fam]
        rows.append({"stage": STAGES[0], "family": fam,
                     "rate":  fam_sub["input_label"].mean() if len(fam_sub) else 0.0})
        rows.append({"stage": STAGES[1], "family": fam,
                     "rate":  fam_known["output_leak"].mean() if len(fam_known) else 0.0})
        rows.append({"stage": STAGES[2], "family": fam,
                     "rate":  fam_sub["ext_leak"].mean() if len(fam_sub) else 0.0})
    long = pd.DataFrame(rows)
    long["rate_label"] = long["rate"].map(lambda v: f"{v*100:.0f}%")
    return long


def _short_family(fam: str) -> str:
    """Shorten family name for column ticks."""
    return {
        "Identity & Identifiability": "Identity",
        "Demographic":                 "Demographic",
        "Health & Medical":            "Health",
        "Location & Spatial":          "Location",
        "Religion & Cultural":         "Religion",
        "Appearance & Body":           "Appearance",
        "Attire, Role & Group":        "Attire",
        "Activity & Lifestyle":        "Activity",
    }.get(fam, fam)


def _heatmap(category: str, long: pd.DataFrame) -> alt.Chart:
    """Render the per-category heatmap in altair.

    The color scale is local to the panel so each category fills the full
    white-to-color gradient regardless of how the absolute rate compares to
    other categories. ``rate == 0`` maps exactly to white via the
    ``[0, vmax]`` domain and ``[#FFFFFF, category color]`` range with linear
    RGB interpolation.
    """
    long = long.copy()
    long["family_short"] = long["family"].map(_short_family)
    ramp = CATEGORY_RAMPS.get(category, ["#FFFFFF", "#888888"])
    vmax = max(long["rate"].max(), 0.05)
    long["rate_label"] = long["rate"].map(
        lambda v: "0%" if v == 0 else f"{v*100:.0f}%"
    )

    # Heat cells. ``interpolate="rgb"`` keeps the gradient linear in sRGB so
    # rate = 0 lands at pure white, and ``clamp=True`` prevents tiny rounding
    # artifacts from dragging the low end off-white.
    heat = (
        alt.Chart(long)
        .mark_rect(stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X(
                "family_short:N",
                sort=[_short_family(f) for f in FAMILY_ORDER],
                title=None,
                axis=alt.Axis(labelAngle=-30, labelFontSize=11, labelLimit=120,
                              labelPadding=4, ticks=False, domain=False),
            ),
            y=alt.Y(
                "stage:N",
                sort=STAGES,
                title=None,
                axis=alt.Axis(labelFontSize=12, ticks=False, domain=False),
            ),
            color=alt.Color(
                "rate:Q",
                scale=alt.Scale(domain=[0, vmax], range=ramp,
                                interpolate="rgb", clamp=True),
                legend=None,
            ),
            tooltip=["stage", "family", alt.Tooltip("rate:Q", format=".1%")],
        )
    )
    # Cell labels: dark text on light cells, white text on dark cells.
    text = (
        alt.Chart(long)
        .mark_text(fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("family_short:N", sort=[_short_family(f) for f in FAMILY_ORDER]),
            y=alt.Y("stage:N", sort=STAGES),
            text="rate_label:N",
            color=alt.condition(
                f"datum.rate > {vmax * 0.55}",
                alt.value("#FFFFFF"),
                alt.value("#222222"),
            ),
        )
    )
    chart = (heat + text).properties(
        title=alt.TitleParams(text=category, fontSize=14, fontWeight="bold",
                              anchor="start", offset=4),
        width=320, height=130,
    )
    return chart


def main() -> None:
    df, _ = load_data()
    audit = df.attrs.get("filter", {})
    print(f"Per-category heatmaps: {audit.get('n_configs_kept')} configs / "
          f"{audit.get('n_items_kept')} items")

    cats_in_data = [c for c in CATEGORY_COLORS if c in set(df["category"])]

    for cat in cats_in_data:
        long = _category_long_df(df, cat)
        local_vmax = max(long["rate"].max(), 0.05)
        slug = cat.lower().replace("/", "_").replace(" ", "_")
        chart = _heatmap(cat, long)
        png = vlc.vegalite_to_png(chart.to_json(), scale=2.0)
        out = FIG_DIR / f"leakage_cat_heatmap_{slug}.png"
        out.write_bytes(png)
        print(f"  saved {out.relative_to(LANTERN_ROOT)}  (local vmax={local_vmax:.2f})")


if __name__ == "__main__":
    main()
