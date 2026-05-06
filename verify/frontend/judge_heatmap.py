"""Shared Streamlit renderers for judge validation heatmaps."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from verify.backend.judge.metrics import (
    LABEL_CONFIRMED,
    LABEL_NONE,
    LABEL_POSSIBLE,
)


DATASET_HEATMAP_COLORS = [
    "#7ADBC4",
    "#FAD765",
    "#FA9F5C",
    "#98D198",
    "#6C80FC",
    "#ACA4B3",
    "#687692",
]

DIFFICULTY_ORDER = ["explicit", "implicit", "none"]
LABEL_ORDER = [LABEL_CONFIRMED, LABEL_POSSIBLE, LABEL_NONE]


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    raw = hex_color.lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _mix_with_white(hex_color: str, intensity: float) -> str:
    intensity = max(0.0, min(1.0, intensity))
    r, g, b = _hex_to_rgb(hex_color)
    mixed = tuple(round(255 * (1 - intensity) + c * intensity) for c in (r, g, b))
    return f"#{mixed[0]:02X}{mixed[1]:02X}{mixed[2]:02X}"


def _text_color(bg_hex: str) -> str:
    r, g, b = _hex_to_rgb(bg_hex)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#111827" if luminance > 0.55 else "#FFFFFF"


def _dataset_color(dataset: str, datasets: List[str]) -> str:
    try:
        idx = datasets.index(dataset)
    except ValueError:
        idx = 0
    return DATASET_HEATMAP_COLORS[idx % len(DATASET_HEATMAP_COLORS)]


def _matrix_for_dataset(results: List[Dict[str, Any]], dataset: str) -> pd.DataFrame:
    rows = []
    for difficulty in DIFFICULTY_ORDER:
        row = {"input": difficulty}
        for label in LABEL_ORDER:
            row[label] = sum(
                1
                for r in results
                if r.get("dataset") == dataset
                and str(r.get("difficulty", "")).lower() == difficulty
                and r.get("label", LABEL_NONE) == label
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("input")


def _chart_rows_for_dataset(
    results: List[Dict[str, Any]],
    dataset: str,
    base_color: str,
) -> pd.DataFrame:
    matrix = _matrix_for_dataset(results, dataset)
    max_value = max(int(matrix.to_numpy().max()), 1)
    rows = []
    for difficulty in DIFFICULTY_ORDER:
        for label in LABEL_ORDER:
            value = int(matrix.loc[difficulty, label])
            bg = _mix_with_white(base_color, value / max_value if value > 0 else 0.0)
            rows.append({
                "input": difficulty,
                "judge": label,
                "count": value,
                "color": bg,
                "text_color": _text_color(bg),
            })
    return pd.DataFrame(rows)


def _render_dataset_heatmap(
    results: List[Dict[str, Any]],
    dataset: str,
    base_color: str,
) -> None:
    import altair as alt

    chart_df = _chart_rows_for_dataset(results, dataset, base_color)

    rect = (
        alt.Chart(chart_df)
        .mark_rect(stroke="#FFFFFF", strokeWidth=2)
        .encode(
            x=alt.X(
                "judge:N",
                sort=LABEL_ORDER,
                title="Judge verdict",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "input:N",
                sort=DIFFICULTY_ORDER,
                title="Input label",
            ),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("input:N", title="Input"),
                alt.Tooltip("judge:N", title="Judge"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
    )

    text = (
        alt.Chart(chart_df)
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(
            x=alt.X("judge:N", sort=LABEL_ORDER),
            y=alt.Y("input:N", sort=DIFFICULTY_ORDER),
            text=alt.Text("count:Q"),
            color=alt.Color("text_color:N", scale=None, legend=None),
        )
    )

    chart = (rect + text).properties(height=170)
    st.altair_chart(chart, use_container_width=True)


def render_judge_heatmaps(results: List[Dict[str, Any]]) -> None:
    """Render one difficulty x judge-label heatmap per dataset."""
    datasets = sorted({str(r.get("dataset", "")).strip() for r in results if r.get("dataset")})
    if not datasets:
        return

    st.subheader("Judge Validation Heatmaps")
    st.caption(
        "Rows are ground-truth input difficulty; columns are judge verdicts. "
        "Each dataset uses one palette color, interpolated with white by cell intensity."
    )

    legend = "  ".join(
        f"<span style='display:inline-block;width:0.85rem;height:0.85rem;"
        f"background:{_dataset_color(ds, datasets)};border-radius:2px;"
        f"margin-right:0.25rem'></span>{ds}"
        for ds in datasets
    )
    st.markdown(legend, unsafe_allow_html=True)

    for start in range(0, len(datasets), 2):
        cols = st.columns(min(2, len(datasets) - start))
        for col, dataset in zip(cols, datasets[start:start + 2]):
            base_color = _dataset_color(dataset, datasets)
            with col.container(border=True):
                st.markdown(f"**{dataset}**")
                _render_dataset_heatmap(results, dataset, base_color)
