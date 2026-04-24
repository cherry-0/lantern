"""
Input-Output Comparison — stage-wise privacy attribute exposure analysis.

Compares privacy attribute presence across three stages for each dataset item:
    1. Input  — binary labels from dataset annotations
    2. Raw inferred output  — evaluator assessment of the app's output text
    3. Externalized results — evaluator assessment of externalization channels

Does NOT apply perturbation.  Runs the original app pipeline only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st
from verify.backend.evaluation_method.evaluator import (
    get_aggregate_eval_entry,
    get_channel_eval_entries,
    is_channelwise_eval_entry,
)


# ─── Constants ────────────────────────────────────────────────────────────────

KNOWN_APPS = [
    "momentag", "clone", "snapdo", "xend", "budget-lens",
    "deeptutor", "llm-vtuber", "skin-disease-detection", "google-ai-edge-gallery",
    "tool-neuron",
    "chat-driven-expense-tracker",
    "photomath",
    "replika",
    "expensify",
]

STAGE_INPUT = "Input"
STAGE_OUTPUT = "Raw Output"
STAGE_EXT = "Externalized"
STAGES = [STAGE_INPUT, STAGE_OUTPUT, STAGE_EXT]

STAGE_COLORS = {
    STAGE_INPUT: "#5bc0de",
    STAGE_OUTPUT: "#d9534f",
    STAGE_EXT: "#f0ad4e",
}
CHANNEL_STAGE_ORDER = ["AGGREGATE", "UI", "NETWORK", "STORAGE", "LOGGING"]
CHANNEL_STAGE_LABELS = {
    "AGGREGATE": "Aggregate Ext",
    "UI": "UI",
    "NETWORK": "Network",
    "STORAGE": "Storage",
    "LOGGING": "Logging",
}
CHANNEL_STAGE_COLORS = {
    "AGGREGATE": "#f0ad4e",
    "UI": "#7f8c8d",
    "NETWORK": "#5b8def",
    "STORAGE": "#27ae60",
    "LOGGING": "#f39c12",
}

DISPLAY_PREVIEW_CHARS = 4000


# ─── Config loaders ───────────────────────────────────────────────────────────

@st.cache_data
def _load_unified_attrs() -> List[str]:
    path = VERIFY_ROOT / "config" / "attribute_list_unified.txt"
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@st.cache_resource
def _load_config():
    from verify.backend.utils.config import load_dataset_list, list_target_apps
    return {
        "datasets": load_dataset_list(),
        "apps": list_target_apps(),
    }


# ─── Pipeline runner ──────────────────────────────────────────────────────────

def _build_ext_text(externalizations: Dict[str, str]) -> str:
    """Join all externalization channels into a single evaluable string."""
    if not externalizations:
        return ""
    return "\n".join(
        f"[{channel.upper()}] {content}"
        for channel, content in externalizations.items()
    )


def _display_preview_text(text: str, area_key: str, *, empty_text: str, height: int = 200) -> None:
    """Show a truncated preview in the UI while preserving the full stored text."""
    show_full = st.session_state.get("show_full_externalizations", False)
    if not text.strip():
        value = empty_text
    elif show_full or len(text) <= DISPLAY_PREVIEW_CHARS:
        value = text
    else:
        value = text[:DISPLAY_PREVIEW_CHARS] + "\n\n[truncated for display]"

    st.text_area(
        area_key,
        value=value,
        height=height,
        disabled=True,
        label_visibility="collapsed",
        key=area_key,
    )

    if text.strip() and not show_full and len(text) > DISPLAY_PREVIEW_CHARS:
        st.caption(
            f"Display preview limited to {DISPLAY_PREVIEW_CHARS:,} chars. "
            "Enable `Show full externalizations` in the sidebar to inspect the full captured text."
        )


def _ioc_cache_dir(app_name: str, dataset_name: str, modality: str) -> "Path":
    """Return the IOC-specific cache directory (distinct from perturb-input caches)."""
    from verify.backend.utils import cache as cache_module
    # Use "ioc_comparison" as perturbation_method so the SHA256 key never
    # collides with any perturb-input cache (which always has a real method name).
    return cache_module.get_cache_dir(
        app_name, dataset_name, modality, [], "ioc_comparison"
    )


def _strip_nonserializable(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of an input_item safe to serialize to JSON.
    Removes PIL Image objects (stored under 'data') but keeps image_base64 strings.
    """
    return {k: v for k, v in item.items() if k != "data"}


def run_comparison_pipeline(
    app_name: str,
    dataset_name: str,
    modality: str,
    unified_attrs: List[str],
    max_items: Optional[int] = None,
    use_cache: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields one result dict per dataset item.

    Result dict schema:
        filename        str
        status          "success" | "failed"
        error           str | None
        modality        str
        input_item      dict              — full loader item
        input_labels    dict[str, int]    — binary, from label_mapper
        output_text     str
        externalizations dict[str, str]
        ext_text        str               — joined externalization channels
        output_eval     dict[str, {...}]  — evaluator results for raw output
        ext_eval        dict[str, {...}]  — evaluator results for externalizations
        output_eval_ok  bool
        ext_eval_ok     bool
        output_eval_error str | None
        ext_eval_error  str | None
        from_cache      bool
    """
    from verify.backend.adapters import get_adapter
    from verify.backend.datasets.loader import iter_dataset
    from verify.backend.datasets.label_mapper import get_input_labels
    from verify.backend.evaluation_method.evaluator import evaluate_inferability
    from verify.backend.utils import cache as cache_module

    adapter = get_adapter(app_name)
    if adapter is None:
        yield {"type": "error", "error": f"No adapter registered for '{app_name}'."}
        return

    cache_dir = _ioc_cache_dir(app_name, dataset_name, modality) if use_cache else None

    # Save run config on first use so the View IOC Results page can discover this cache
    if cache_dir is not None:
        run_config_path = cache_dir / "run_config.json"
        if not run_config_path.exists():
            cache_module.save_run_config(cache_dir, {
                "app_name": app_name,
                "dataset_name": dataset_name,
                "modality": modality,
                "unified_attrs": unified_attrs,
                "perturbation_method": "ioc_comparison",
                "evaluation_method": "openrouter",
            })

    for ok, item, err in iter_dataset(dataset_name, modality, max_items=max_items):
        filename = item.get("filename", "unknown")

        # ── Cache check ───────────────────────────────────────────────────────
        if cache_dir is not None:
            cached = cache_module.load_item_cache(cache_dir, filename)
            if cached is not None:
                # Restore PIL-less input_item if it was stripped before saving
                cached.setdefault("input_item", item)
                cached["from_cache"] = True
                yield cached
                continue

        if not ok:
            result = {
                "filename": filename,
                "status": "failed",
                "error": err or "Dataset load error",
                "modality": modality,
                "input_item": item,
                "input_labels": {},
                "output_text": "",
                "externalizations": {},
                "ext_text": "",
                "output_eval": {},
                "ext_eval": {},
                "output_eval_ok": False,
                "ext_eval_ok": False,
                "output_eval_error": None,
                "ext_eval_error": None,
                "from_cache": False,
            }
            yield result
            continue

        # 1. Map input labels
        input_labels = get_input_labels(item, unified_attrs)

        # 2. Run app pipeline (original only, no perturbation)
        try:
            from verify.backend.utils.config import set_current_app_context
            set_current_app_context(app_name)
            adapter._reset_openrouter_calls()
            pipeline_result = adapter.run_pipeline(item)
        except Exception as e:
            result = {
                "filename": filename,
                "status": "failed",
                "error": f"Pipeline error: {e}",
                "modality": modality,
                "input_item": item,
                "input_labels": input_labels,
                "output_text": "",
                "externalizations": {},
                "ext_text": "",
                "output_eval": {},
                "ext_eval": {},
                "output_eval_ok": False,
                "ext_eval_ok": False,
                "output_eval_error": None,
                "ext_eval_error": None,
                "from_cache": False,
            }
            yield result
            continue

        if not pipeline_result.success:
            result = {
                "filename": filename,
                "status": "failed",
                "error": pipeline_result.error or "Adapter returned failure.",
                "modality": modality,
                "input_item": item,
                "input_labels": input_labels,
                "output_text": "",
                "externalizations": {},
                "ext_text": "",
                "output_eval": {},
                "ext_eval": {},
                "output_eval_ok": False,
                "ext_eval_ok": False,
                "output_eval_error": None,
                "ext_eval_error": None,
                "from_cache": False,
            }
            yield result
            continue

        output_text = pipeline_result.output_text or ""
        externalizations = pipeline_result.externalizations or {}
        ext_text = _build_ext_text(externalizations)

        # 3. Evaluate raw output
        out_ok, out_eval, out_err = evaluate_inferability(output_text, unified_attrs)

        # 4. Evaluate externalized results (skip if empty)
        if ext_text.strip():
            ext_ok, ext_eval, ext_err = evaluate_inferability(ext_text, unified_attrs)
        else:
            ext_ok, ext_eval, ext_err = True, {}, None

        result = {
            "filename": filename,
            "status": "success",
            "error": None,
            "modality": modality,
            "input_item": item,
            "input_labels": input_labels,
            "output_text": output_text,
            "externalizations": externalizations,
            "ext_text": ext_text,
            "output_eval": out_eval,
            "ext_eval": ext_eval,
            "output_eval_ok": out_ok,
            "ext_eval_ok": ext_ok,
            "output_eval_error": out_err,
            "ext_eval_error": ext_err,
            "from_cache": False,
            "prompt_text": pipeline_result.metadata.get("prompt_text", ""),
        }

        # ── Save to cache (strip PIL objects before serializing) ──────────────
        if cache_dir is not None:
            saveable = {
                **result,
                "input_item": _strip_nonserializable(item),
            }
            cache_module.save_item_cache(cache_dir, filename, saveable)

        yield result


# ─── Visualization helpers ────────────────────────────────────────────────────

def _display_image(b64_str: str | None, data=None):
    try:
        if b64_str:
            import base64
            st.image(base64.b64decode(b64_str), use_container_width=True)
        elif data is not None:
            st.image(data, use_container_width=True)
        else:
            st.warning("No image available.")
    except Exception as e:
        st.error(f"Could not display image: {e}")



def _stage_table(
    input_labels: Dict[str, int],
    output_eval: Dict[str, Any],
    ext_eval: Dict[str, Any],
    unified_attrs: List[str],
):
    """
    Full-width HTML table: attribute × 3 stages.
    Cells with a positive result show ✅ on a light-green background.
    """
    _CHECK = "✅"
    _BG_YES = "background:#d4f5d4;"
    _BG_NO = "background:#fafafa;"
    _TD_BASE = (
        "text-align:center;padding:6px 10px;border:1px solid #e0e0e0;"
        "font-size:0.95em;width:18%;"
    )
    _TH_BASE = (
        "text-align:center;padding:7px 10px;border:1px solid #d0d0d0;"
        "background:#f0f0f0;font-size:0.9em;font-weight:600;"
    )
    _ATTR_TD = (
        "text-align:left;padding:6px 12px;border:1px solid #e0e0e0;"
        "font-size:0.95em;font-weight:500;width:46%;"
    )

    rows_html = []
    for attr in unified_attrs:
        in_val = input_labels.get(attr, 0) == 1
        out_entry = output_eval.get(attr, {})
        out_val = isinstance(out_entry, dict) and bool(out_entry.get("inferable"))
        ext_entry = get_aggregate_eval_entry(ext_eval.get(attr))
        ext_val = bool(ext_entry.get("inferable"))

        def cell(flag: bool) -> str:
            bg = _BG_YES if flag else _BG_NO
            content = _CHECK if flag else ""
            return f'<td style="{_TD_BASE}{bg}">{content}</td>'

        rows_html.append(
            f'<tr>'
            f'<td style="{_ATTR_TD}">{attr}</td>'
            f'{cell(in_val)}{cell(out_val)}{cell(ext_val)}'
            f'</tr>'
        )

    table_html = f"""
<table style="width:100%;border-collapse:collapse;margin-top:6px">
  <thead>
    <tr>
      <th style="{_TH_BASE}text-align:left;width:46%">Attribute</th>
      <th style="{_TH_BASE}width:18%">{STAGE_INPUT}</th>
      <th style="{_TH_BASE}width:18%">{STAGE_OUTPUT}</th>
      <th style="{_TH_BASE}width:18%">{STAGE_EXT}</th>
    </tr>
  </thead>
  <tbody>
    {"".join(rows_html)}
  </tbody>
</table>
"""
    st.markdown(table_html, unsafe_allow_html=True)


def _has_prompt3_channel_data(ext_eval: Dict[str, Any]) -> bool:
    return any(is_channelwise_eval_entry(entry) for entry in ext_eval.values())


def _render_attribute_heatmap(
    input_labels: Dict[str, int],
    output_eval: Dict[str, Any],
    ext_eval: Dict[str, Any],
    unified_attrs: List[str],
):
    import pandas as pd
    import altair as alt

    rows = [
        {"Stage": STAGE_INPUT, "Attribute": attr, "state": "yes" if input_labels.get(attr, 0) == 1 else "no"}
        for attr in unified_attrs
    ]
    rows += [
        {
            "Stage": STAGE_OUTPUT,
            "Attribute": attr,
            "state": "yes" if bool(output_eval.get(attr, {}).get("inferable")) else "no",
        }
        for attr in unified_attrs
    ]

    if _has_prompt3_channel_data(ext_eval):
        stage_keys = CHANNEL_STAGE_ORDER
    else:
        stage_keys = ["AGGREGATE"]

    for stage_key in stage_keys:
        label = CHANNEL_STAGE_LABELS[stage_key]
        for attr in unified_attrs:
            entry = ext_eval.get(attr)
            if stage_key == "AGGREGATE":
                inferable = bool(get_aggregate_eval_entry(entry).get("inferable"))
                state = "yes" if inferable else "no"
            else:
                channels = get_channel_eval_entries(entry)
                if stage_key in channels:
                    state = "yes" if bool(channels[stage_key].get("inferable")) else "no"
                else:
                    state = "na"
            rows.append({"Stage": label, "Attribute": attr, "state": state})

    df = pd.DataFrame(rows)
    stage_order = [STAGE_INPUT, STAGE_OUTPUT] + [CHANNEL_STAGE_LABELS[k] for k in stage_keys]

    rect = (
        alt.Chart(df)
        .mark_rect(stroke="#ffffff", strokeWidth=1)
        .encode(
            x=alt.X("Attribute:N", sort=unified_attrs, axis=alt.Axis(labelAngle=-40, labelFontSize=10)),
            y=alt.Y("Stage:N", sort=stage_order, title=None),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(
                    domain=["yes", "no", "na"],
                    range=["#c62828", "#ffffff", "#e6e6e6"],
                ),
                legend=None,
            ),
            tooltip=["Stage", "Attribute", alt.Tooltip("state:N", title="Status")],
        )
    )
    text = (
        alt.Chart(df[df["state"] == "yes"])
        .mark_text(text="✓", fontSize=12, fontWeight="bold", color="#2f4f2f")
        .encode(
            x=alt.X("Attribute:N", sort=unified_attrs),
            y=alt.Y("Stage:N", sort=stage_order),
        )
    )
    chart = (rect + text).properties(width=alt.Step(44), height=alt.Step(44))
    st.altair_chart(chart, use_container_width=True)
    st.caption("Red = inferable/present, white = not inferable, grey = channel not captured or not evaluated.")


def _render_channel_aggregated(all_results: List[Dict[str, Any]], unified_attrs: List[str]):
    import pandas as pd
    import altair as alt

    success = [r for r in all_results if r.get("status") == "success"]
    if not success:
        st.info("No successful items to aggregate.")
        return

    include_channelwise = any(_has_prompt3_channel_data(r.get("ext_eval", {})) for r in success)
    stage_keys = ["INPUT", "OUTPUT", "AGGREGATE"] + (CHANNEL_STAGE_ORDER[1:] if include_channelwise else [])
    stage_labels = {
        "INPUT": STAGE_INPUT,
        "OUTPUT": STAGE_OUTPUT,
        "AGGREGATE": CHANNEL_STAGE_LABELS["AGGREGATE"],
        "UI": CHANNEL_STAGE_LABELS["UI"],
        "NETWORK": CHANNEL_STAGE_LABELS["NETWORK"],
        "STORAGE": CHANNEL_STAGE_LABELS["STORAGE"],
        "LOGGING": CHANNEL_STAGE_LABELS["LOGGING"],
    }
    stage_colors = {
        "INPUT": STAGE_COLORS[STAGE_INPUT],
        "OUTPUT": STAGE_COLORS[STAGE_OUTPUT],
        "AGGREGATE": CHANNEL_STAGE_COLORS["AGGREGATE"],
        "UI": CHANNEL_STAGE_COLORS["UI"],
        "NETWORK": CHANNEL_STAGE_COLORS["NETWORK"],
        "STORAGE": CHANNEL_STAGE_COLORS["STORAGE"],
        "LOGGING": CHANNEL_STAGE_COLORS["LOGGING"],
    }

    rows = []
    for attr in unified_attrs:
        for stage_key in stage_keys:
            if stage_key == "INPUT":
                support = len(success)
                positive = sum(r.get("input_labels", {}).get(attr, 0) for r in success)
            elif stage_key == "OUTPUT":
                support = len(success)
                positive = sum(
                    1
                    for r in success
                    if isinstance(r.get("output_eval", {}).get(attr), dict)
                    and r["output_eval"][attr].get("inferable")
                )
            elif stage_key == "AGGREGATE":
                support = len(success)
                positive = sum(
                    1
                    for r in success
                    if bool(get_aggregate_eval_entry(r.get("ext_eval", {}).get(attr)).get("inferable"))
                )
            else:
                support = 0
                positive = 0
                for r in success:
                    channels = get_channel_eval_entries(r.get("ext_eval", {}).get(attr))
                    if stage_key in channels:
                        support += 1
                        if channels[stage_key].get("inferable"):
                            positive += 1
            if support == 0 and stage_key not in ("INPUT", "OUTPUT", "AGGREGATE"):
                continue
            rows.append(
                {
                    "Attribute": attr,
                    "Stage": stage_labels[stage_key],
                    "Positive Rate": (positive / support) if support else 0.0,
                    "Support": support,
                }
            )

    df = pd.DataFrame(rows)
    stage_order = [stage_labels[s] for s in stage_keys if stage_labels[s] in set(df["Stage"])]
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Attribute:N", sort=unified_attrs, axis=alt.Axis(labelAngle=-40, labelFontSize=10)),
            xOffset=alt.XOffset("Stage:N", sort=stage_order),
            y=alt.Y("Positive Rate:Q", scale=alt.Scale(domain=[0, 1]), title="Positive rate"),
            color=alt.Color(
                "Stage:N",
                sort=stage_order,
                scale=alt.Scale(domain=stage_order, range=[stage_colors[k] for k in stage_keys if stage_labels[k] in stage_order]),
            ),
            tooltip=["Attribute", "Stage", alt.Tooltip("Positive Rate:Q", format=".2f"), "Support"],
        )
        .properties(height=320, title=f"Attribute-wise positive rate across {len(success)} item(s)")
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Input/Raw Output/Aggregate use all successful items. Channel bars use only items where that channel was captured and evaluated."
    )


def _render_channel_aggregated_heatmap(all_results: List[Dict[str, Any]], unified_attrs: List[str]):
    import pandas as pd
    import altair as alt

    success = [r for r in all_results if r.get("status") == "success"]
    if not success:
        st.info("No successful items to aggregate.")
        return

    include_channelwise = any(_has_prompt3_channel_data(r.get("ext_eval", {})) for r in success)
    stage_keys = ["INPUT", "OUTPUT", "AGGREGATE"] + (CHANNEL_STAGE_ORDER[1:] if include_channelwise else [])
    stage_labels = {
        "INPUT": STAGE_INPUT,
        "OUTPUT": STAGE_OUTPUT,
        "AGGREGATE": CHANNEL_STAGE_LABELS["AGGREGATE"],
        "UI": CHANNEL_STAGE_LABELS["UI"],
        "NETWORK": CHANNEL_STAGE_LABELS["NETWORK"],
        "STORAGE": CHANNEL_STAGE_LABELS["STORAGE"],
        "LOGGING": CHANNEL_STAGE_LABELS["LOGGING"],
    }

    rows = []
    for attr in unified_attrs:
        for stage_key in stage_keys:
            if stage_key == "INPUT":
                support = len(success)
                positive = sum(r.get("input_labels", {}).get(attr, 0) for r in success)
            elif stage_key == "OUTPUT":
                support = len(success)
                positive = sum(
                    1
                    for r in success
                    if isinstance(r.get("output_eval", {}).get(attr), dict)
                    and r["output_eval"][attr].get("inferable")
                )
            elif stage_key == "AGGREGATE":
                support = len(success)
                positive = sum(
                    1
                    for r in success
                    if bool(get_aggregate_eval_entry(r.get("ext_eval", {}).get(attr)).get("inferable"))
                )
            else:
                support = 0
                positive = 0
                for r in success:
                    channels = get_channel_eval_entries(r.get("ext_eval", {}).get(attr))
                    if stage_key in channels:
                        support += 1
                        if channels[stage_key].get("inferable"):
                            positive += 1

            rows.append(
                {
                    "Attribute": attr,
                    "Stage": stage_labels[stage_key],
                    "Exposure Score": round((positive / support), 3) if support else None,
                    "Support": support,
                }
            )

    df = pd.DataFrame(rows)
    stage_order = [stage_labels[s] for s in stage_keys]

    df_valid = df[df["Exposure Score"].notna()]
    df_na = df[df["Exposure Score"].isna()]

    layers = []
    if not df_na.empty:
        layers.append(
            alt.Chart(df_na)
            .mark_rect(stroke="#ffffff", strokeWidth=1, color="#e6e6e6")
            .encode(
                x=alt.X("Attribute:N", sort=unified_attrs, axis=alt.Axis(labelAngle=-40, labelFontSize=10)),
                y=alt.Y("Stage:N", sort=stage_order, title=None),
                tooltip=[
                    alt.Tooltip("Stage:N"),
                    alt.Tooltip("Attribute:N"),
                    alt.Tooltip("Support:Q"),
                ],
            )
        )

    if not df_valid.empty:
        layers.append(
            alt.Chart(df_valid)
            .mark_rect(stroke="#ffffff", strokeWidth=1)
            .encode(
                x=alt.X("Attribute:N", sort=unified_attrs, axis=alt.Axis(labelAngle=-40, labelFontSize=10)),
                y=alt.Y("Stage:N", sort=stage_order, title=None),
                color=alt.Color(
                    "Exposure Score:Q",
                    scale=alt.Scale(domain=[0, 1], range=["#ffffff", "#c62828"]),
                    legend=alt.Legend(title="Exposure score"),
                ),
                tooltip=[
                    alt.Tooltip("Stage:N"),
                    alt.Tooltip("Attribute:N"),
                    alt.Tooltip("Exposure Score:Q", format=".2f"),
                    alt.Tooltip("Support:Q"),
                ],
            )
            .properties(width=alt.Step(44), height=alt.Step(44))
        )
        layers.append(
            alt.Chart(df_valid)
            .mark_text(fontSize=11, color="#1f1f1f")
            .encode(
                x=alt.X("Attribute:N", sort=unified_attrs),
                y=alt.Y("Stage:N", sort=stage_order),
                text=alt.Text("Exposure Score:Q", format=".1f"),
            )
        )

    if layers:
        chart = layers[0]
        for layer in layers[1:]:
            chart = chart + layer
        st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Heatmap cells show attribute-inferred accuracy / exposure rate by stage or channel. "
        "Absent channels remain blank light gray for layout consistency."
    )


def _reasoning_expander(
    output_eval: Dict[str, Any],
    ext_eval: Dict[str, Any],
    unified_attrs: List[str],
    idx: int = 0,
):
    """Collapsible reasoning panel: one row per attribute, side-by-side output vs ext."""
    if st.checkbox("Show reasoning details", value=False, key=f"reasoning_{idx}"):
        col_out, col_ext = st.columns(2)
        with col_out:
            st.markdown(f"**{STAGE_OUTPUT}**")
            for attr in unified_attrs:
                entry = output_eval.get(attr)
                if not isinstance(entry, dict):
                    continue
                icon = "🔴" if entry.get("inferable") else "🟢"
                reason = entry.get("reasoning", "—")
                st.markdown(
                    f'{icon} <span style="font-size:0.9em"><b>{attr}</b></span>',
                    unsafe_allow_html=True,
                )
                st.caption(reason)
        with col_ext:
            st.markdown(f"**{STAGE_EXT}**")
            if not ext_eval:
                st.caption("No externalizations captured.")
            else:
                for attr in unified_attrs:
                    entry = ext_eval.get(attr)
                    if not isinstance(entry, dict):
                        continue
                    agg = get_aggregate_eval_entry(entry)
                    icon = "🔴" if agg.get("inferable") else "🟢"
                    reason = agg.get("reasoning", "—")
                    st.markdown(
                        f'{icon} <span style="font-size:0.9em"><b>{attr}</b></span>',
                        unsafe_allow_html=True,
                    )
                    if is_channelwise_eval_entry(entry):
                        st.caption(f"Aggregate: {reason}")
                        channels = get_channel_eval_entries(entry)
                        for channel, channel_entry in channels.items():
                            st.caption(
                                f"[{channel}] "
                                f'{"inferable" if channel_entry.get("inferable") else "not inferable"}: '
                                f'{channel_entry.get("reasoning", "—")}'
                            )
                    else:
                        st.caption(reason)


def _render_item(result: Dict[str, Any], unified_attrs: List[str], idx: int):
    """Render one per-item expander."""
    filename = result["filename"]
    status = result["status"]
    modality = result["modality"]
    input_item = result.get("input_item", {})

    status_icon = {"success": "✅", "failed": "❌"}.get(status, "")
    from_cache = " (cached)" if result.get("from_cache") else ""

    # Build expander title suffix from input labels
    positives = [a for a in unified_attrs if result.get("input_labels", {}).get(a, 0) == 1]
    label_suffix = f"  —  🏷 {', '.join(positives)}" if positives else ""
    # Also show data_type for PrivacyLens items
    data_type = input_item.get("data_type", "")
    if data_type and not positives:
        label_suffix = f"  —  📄 {data_type}"

    with st.expander(f"{status_icon} {filename}{from_cache}{label_suffix}", expanded=(status == "failed")):
        if status == "failed":
            st.error(f"Error: {result.get('error', 'Unknown error')}")
            return

        detail_key = f"ioc_render_details_{idx}"
        show_details = bool(st.session_state.get(detail_key, False))
        if not show_details:
            if st.button("Load details", key=f"ioc_load_details_{idx}", use_container_width=True):
                st.session_state[detail_key] = True
                st.rerun()
            st.caption("Details are not rendered until requested. Click `Load details` to render this item.")
            return

        if st.button("Hide details", key=f"ioc_hide_details_{idx}", type="secondary"):
            st.session_state[detail_key] = False
            st.rerun()

        # ── Top row: Input (left) | Outputs (right) ──────────────────────
        col_in, col_out = st.columns([1, 2])

        with col_in:
            st.markdown("**Input**")
            if modality == "image":
                _display_image(input_item.get("image_base64"), input_item.get("data"))
                prompt_text = result.get("prompt_text", "").strip()
                if prompt_text:
                    st.caption("**Prompt sent to model:**")
                    st.text(prompt_text)
            elif modality == "text":
                text = input_item.get("text_content", "")
                st.text_area(
                    "input_text", value=text, height=220, disabled=True,
                    label_visibility="collapsed",
                    key=f"ioc_input_text_{idx}",
                )
            else:
                frames = input_item.get("data") or []
                if frames:
                    cols = st.columns(min(len(frames), 2))
                    for c, f in zip(cols, frames[:2]):
                        with c:
                            st.image(f, use_container_width=True)
                else:
                    st.info("No media available.")

            input_labels = result.get("input_labels", {})
            positives = [a for a in unified_attrs if input_labels.get(a, 0) == 1]
            if positives:
                st.caption("**Annotated attributes:**")
                st.markdown("  ".join(f"`{a}`" for a in positives))

        with col_out:
            out_col, ext_col = st.columns(2)

            with out_col:
                st.markdown(f"**{STAGE_OUTPUT}**")
                output_text = result.get("output_text", "")
                st.text_area(
                    "output_text", value=output_text, height=200, disabled=True,
                    label_visibility="collapsed",
                    key=f"ioc_out_{idx}",
                )
            with ext_col:
                st.markdown(f"**{STAGE_EXT}**")
                ext_text = result.get("ext_text", "")
                _display_preview_text(
                    ext_text,
                    f"ioc_ext_{idx}",
                    empty_text="No externalizations captured.",
                    height=200,
                )

        st.divider()

        # ── Heatmap: 3 stages × N attributes ─────────────────────────────
        st.markdown("**Stage-wise attribute presence**")

        if result.get("output_eval_error"):
            st.warning(f"Output evaluation error: {result['output_eval_error']}")
        if result.get("ext_eval_error"):
            st.warning(f"Externalization evaluation error: {result['ext_eval_error']}")

        if _has_prompt3_channel_data(result.get("ext_eval", {})):
            _render_attribute_heatmap(
                result.get("input_labels", {}),
                result.get("output_eval", {}),
                result.get("ext_eval", {}),
                unified_attrs,
            )
        else:
            _stage_table(
                result.get("input_labels", {}),
                result.get("output_eval", {}),
                result.get("ext_eval", {}),
                unified_attrs,
            )

        # ── Reasoning ─────────────────────────────────────────────────────
        _reasoning_expander(
            result.get("output_eval", {}),
            result.get("ext_eval", {}),
            unified_attrs,
            idx,
        )


def _render_aggregated(all_results: List[Dict[str, Any]], unified_attrs: List[str]):
    st.markdown("**Exposure Heatmap**")
    _render_channel_aggregated_heatmap(all_results, unified_attrs)
    st.markdown("**Grouped Bar Chart**")
    _render_channel_aggregated(all_results, unified_attrs)

# ─── Main UI ──────────────────────────────────────────────────────────────────

def _sync_app_modes() -> None:
    """Push per-app mode choices from session state into the config module."""
    from verify.backend.utils.config import set_app_mode_override
    for app_name, mode in st.session_state.get("app_modes", {}).items():
        set_app_mode_override(app_name, mode)


def main():
    st.title("🔬 Input-Output Comparison")
    st.markdown(
        "Stage-wise comparison of privacy attribute exposure: "
        "**Input** (dataset labels) → **Raw Output** (app inference) → "
        "**Externalized** (network/storage/logging channels)."
    )

    _sync_app_modes()

    config = _load_config()
    unified_attrs = _load_unified_attrs()
    datasets = config["datasets"]
    all_apps = config["apps"]

    recognized_apps = [a for a in all_apps if a in KNOWN_APPS]
    other_apps = [a for a in all_apps if a not in KNOWN_APPS]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        st.toggle(
            "Show full externalizations",
            key="show_full_externalizations",
            value=False,
            help="Display the full captured externalization text instead of a shortened preview.",
        )
        st.divider()

        # App
        st.subheader("Target App")
        app_options = recognized_apps + [f"{a} (unrecognized)" for a in other_apps]
        if not app_options:
            st.error("No target apps found in target-apps/.")
            return
        selected_app_display = st.selectbox("App", app_options, key="ioc_app")
        selected_app = selected_app_display.split(" (")[0]

        # Check adapter availability
        if selected_app in KNOWN_APPS:
            try:
                from verify.backend.adapters import get_adapter
                from verify.backend.utils.config import set_current_app_context
                adapter = get_adapter(selected_app)
                if adapter:
                    set_current_app_context(selected_app)
                    avail, msg = adapter.check_availability()
                    if avail:
                        st.success(f"Available: {msg}")
                    else:
                        st.error(f"Unavailable: {msg}")
                    app_available = avail
                else:
                    st.error("No adapter registered.")
                    app_available = False
            except Exception as e:
                st.error(f"Adapter check error: {e}")
                app_available = False
        else:
            app_available = False
            st.warning("Unrecognized app — no adapter available.")

        st.divider()

        # Dataset
        st.subheader("Dataset")
        if not datasets:
            st.error("No datasets in config/dataset_list.txt.")
            return
        selected_dataset = st.selectbox("Dataset", datasets, key="ioc_dataset")

        st.divider()

        # Modality
        st.subheader("Modality")
        selected_modality = st.selectbox(
            "Modality", ["image", "text", "video"], key="ioc_modality"
        )

        st.divider()

        # Unified attributes
        st.subheader("Attributes")
        if not unified_attrs:
            st.warning("attribute_list_unified.txt not found or empty.")
        else:
            st.caption(f"{len(unified_attrs)} unified attributes loaded.")
            st.markdown(", ".join(f"`{a}`" for a in unified_attrs))

        st.divider()

        # Item limit
        st.subheader("Item Limit")
        limit_enabled = st.checkbox("Limit items", value=True, key="ioc_limit_en")
        max_items = None
        if limit_enabled:
            max_items = int(
                st.number_input("Max items", min_value=1, value=1, step=1, key="ioc_max")
            )

        st.divider()

        # Cache option
        use_cache = st.checkbox(
            "Use cache (skip already-processed items)", value=False, key="ioc_use_cache"
        )

        run_clicked = st.button(
            "▶ Run Comparison",
            type="primary",
            disabled=not (app_available and unified_attrs),
            use_container_width=True,
        )
        if not app_available:
            st.caption(f"App '{selected_app}' is not available.")
        if not unified_attrs:
            st.caption("No attributes configured.")

    # ── Session state ─────────────────────────────────────────────────────────
    for key, default in [
        ("ioc_results", []),
        ("ioc_processing", False),
        ("ioc_items_processed", 0),
        ("ioc_items_total", 0),
        ("ioc_run_config", {}),
        ("ioc_last_run_config", {}),   # config that produced the stored results
        ("ioc_current_item", ""),
        ("ioc_error", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Start a new run
    if run_clicked:
        st.session_state.ioc_results = []
        st.session_state.ioc_processing = True
        st.session_state.ioc_items_processed = 0
        st.session_state.ioc_current_item = ""
        st.session_state.ioc_error = None
        run_cfg = {
            "app": selected_app,
            "dataset": selected_dataset,
            "modality": selected_modality,
            "max_items": max_items,
        }
        st.session_state.ioc_run_config = run_cfg
        st.session_state.ioc_last_run_config = run_cfg   # mark as authoritative

        from verify.frontend.utils import count_dataset_items
        total = count_dataset_items(selected_dataset, selected_modality)
        st.session_state.ioc_items_total = (
            min(total, max_items) if max_items else total
        )

        st.session_state["_ioc_generator"] = run_comparison_pipeline(
            app_name=selected_app,
            dataset_name=selected_dataset,
            modality=selected_modality,
            unified_attrs=unified_attrs,
            max_items=max_items,
            use_cache=use_cache,
        )
        st.rerun()

    # ── Display ───────────────────────────────────────────────────────────────
    # Rendered BEFORE the processing block so it is visible on every rerun
    # (st.rerun() stops execution, so anything after it is skipped).

    if st.session_state.ioc_error:
        st.error(st.session_state.ioc_error)

    if st.session_state.ioc_processing:
        # ── Active run: show ONLY the progress bar, nothing else ──────────────
        rc = st.session_state.ioc_run_config
        processed = st.session_state.ioc_items_processed
        total = st.session_state.ioc_items_total

        st.markdown(
            f"**{rc.get('app')}** · {rc.get('dataset')} · {rc.get('modality')}"
        )
        if total > 0:
            st.progress(
                min(processed / total, 1.0),
                text=f"Processed {processed} / {total}",
            )
        else:
            st.progress(0, text=f"Processed {processed} item{'s' if processed != 1 else ''}")

        next_num = processed + 1
        if total == 0 or next_num <= total:
            st.caption(f"⏳ Processing item {next_num}" + (f" of {total}" if total > 0 else "") + "…")

    else:
        # ── Idle: show results or instructions ────────────────────────────────
        results = st.session_state.ioc_results

        if results:
            last_cfg = st.session_state.ioc_last_run_config
            stale = bool(last_cfg) and (
                selected_app != last_cfg.get("app")
                or selected_dataset != last_cfg.get("dataset")
                or selected_modality != last_cfg.get("modality")
                or max_items != last_cfg.get("max_items")
            )

            if stale:
                st.info(
                    f"Results below are from a previous run "
                    f"(**{last_cfg.get('app', '?')}** / {last_cfg.get('dataset', '?')} "
                    f"/ {last_cfg.get('modality', '?')}). "
                    "Click **▶ Run Comparison** to run with the current configuration."
                )
            else:
                rc = st.session_state.ioc_run_config
                n = len(results)
                st.subheader(
                    f"Results — {rc.get('app', '')} / {rc.get('dataset', '')} "
                    f"/ {rc.get('modality', '')} ({n} item{'s' if n != 1 else ''})"
                )
                st.divider()
                st.subheader("Aggregated Attribute-wise Positive Rate")
                _render_aggregated(results, unified_attrs)
                st.divider()
                for idx, result in enumerate(results):
                    _render_item(result, unified_attrs, idx)

        elif not st.session_state.ioc_error:
            st.markdown(
                """
                ### How to use this page

                1. Select a **target app** and **dataset** in the sidebar.
                2. Choose the **modality** that matches the dataset.
                3. Optionally limit the number of items processed.
                4. Click **▶ Run Comparison**.

                For each item the page will show:
                - **Input** image or text with its annotated privacy labels
                - **Raw app output** and **externalized channel results**
                - A **heatmap** comparing attribute presence across all three stages
                - An **aggregated chart** across all items (positive rate per attribute)
                """
            )

    # ── Process one item per rerun ────────────────────────────────────────────
    # Kept AFTER the display block so progress/results are visible before this
    # triggers st.rerun().
    if st.session_state.ioc_processing and "_ioc_generator" in st.session_state:
        gen = st.session_state["_ioc_generator"]
        try:
            item = next(gen)

            if isinstance(item, dict) and item.get("type") == "error":
                st.session_state.ioc_error = f"Pipeline error: {item.get('error')}"
                st.session_state.ioc_processing = False
            else:
                st.session_state.ioc_results.append(item)
                st.session_state.ioc_items_processed += 1
                st.session_state.ioc_current_item = item.get("filename", "")
            st.rerun()

        except StopIteration:
            st.session_state.ioc_processing = False
            st.session_state.pop("_ioc_generator", None)
            st.rerun()
        except Exception as e:
            st.session_state.ioc_error = f"Unexpected error: {e}"
            st.session_state.ioc_processing = False
            st.rerun()


if __name__ == "__main__":
    main()
