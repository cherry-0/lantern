"""
View Input-Output Comparison Results — browse cached IOC runs.

Loads items directly from an IOC cache directory
(verify/outputs/cache_<hash>/ where run_config.json contains
perturbation_method == "ioc_comparison") and renders the same
stage-wise view as the live Input-Output Comparison page.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT  = Path(__file__).resolve().parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st
from verify.backend.evaluation_method.evaluator import (
    get_aggregate_eval_entry,
    get_channel_eval_entries,
    is_channelwise_eval_entry,
)


# ── Stage constants (mirrors 2_Input_Output_Comparison.py) ────────────────────

STAGE_INPUT  = "Input"
STAGE_OUTPUT = "Raw Output"
STAGE_EXT    = "Externalized"
STAGES       = [STAGE_INPUT, STAGE_OUTPUT, STAGE_EXT]
STAGE_COLORS = {
    STAGE_INPUT:  "#5bc0de",
    STAGE_OUTPUT: "#d9534f",
    STAGE_EXT:    "#f0ad4e",
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _display_image(b64_str: str | None, caption: str = ""):
    try:
        if b64_str:
            import base64
            st.image(base64.b64decode(b64_str), caption=caption, use_container_width=True)
        else:
            st.info("Image not available in cache.")
    except Exception as e:
        st.error(f"Could not display image: {e}")


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


def _find_image_path(filename: str, dataset_name: str) -> Path | None:
    """Return the filesystem path of an image in a dataset, or None."""
    try:
        from verify.backend.utils.config import get_dataset_path
        dataset_path = get_dataset_path(dataset_name)
        if dataset_path is None:
            return None
        _SPLITS = ["val2017", "test2017", "train2017"]
        for split in _SPLITS:
            p = dataset_path / split / filename
            if p.exists():
                return p
        p = dataset_path / filename
        if p.exists():
            return p
        return None
    except Exception:
        return None


def _load_original_image(filename: str, dataset_name: str) -> str | None:
    """Load original image from dataset and return as base64 string."""
    try:
        from PIL import Image as PILImage
        img_path = _find_image_path(filename, dataset_name)
        if img_path is None:
            return None
        img = PILImage.open(str(img_path)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        import base64
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def _stage_table(
    input_labels: Dict[str, int],
    output_eval:  Dict[str, Any],
    ext_eval:     Dict[str, Any],
    unified_attrs: List[str],
):
    _CHECK  = "✅"
    _BG_YES = "background:#d4f5d4;"
    _BG_NO  = "background:#fafafa;"
    _TD = "text-align:center;padding:6px 10px;border:1px solid #e0e0e0;font-size:0.95em;width:18%;"
    _TH = "text-align:center;padding:7px 10px;border:1px solid #d0d0d0;background:#f0f0f0;font-size:0.9em;font-weight:600;"
    _AT = "text-align:left;padding:6px 12px;border:1px solid #e0e0e0;font-size:0.95em;font-weight:500;width:46%;"

    def cell(flag: bool) -> str:
        bg = _BG_YES if flag else _BG_NO
        return f'<td style="{_TD}{bg}">{"✅" if flag else ""}</td>'

    rows = "".join(
        f"<tr>"
        f'<td style="{_AT}">{attr}</td>'
        f'{cell(input_labels.get(attr, 0) == 1)}'
        f'{cell(isinstance(output_eval.get(attr), dict) and bool(output_eval[attr].get("inferable")))}'
        f'{cell(bool(get_aggregate_eval_entry(ext_eval.get(attr)).get("inferable")))}'
        f"</tr>"
        for attr in unified_attrs
    )
    st.markdown(
        f"""
<table style="width:100%;border-collapse:collapse;margin-top:6px">
  <thead><tr>
    <th style="{_TH}text-align:left;width:46%">Attribute</th>
    <th style="{_TH}width:18%">{STAGE_INPUT}</th>
    <th style="{_TH}width:18%">{STAGE_OUTPUT}</th>
    <th style="{_TH}width:18%">{STAGE_EXT}</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>""",
        unsafe_allow_html=True,
    )


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
    output_eval:   Dict[str, Any],
    ext_eval:      Dict[str, Any],
    unified_attrs: List[str],
    idx: int = 0,
):
    """Collapsible reasoning panel using checkbox to avoid nested expanders."""
    if st.checkbox("Show reasoning details", value=False, key=f"reasoning_{idx}"):
        col_out, col_ext = st.columns(2)
        with col_out:
            st.markdown(f"**{STAGE_OUTPUT}**")
            for attr in unified_attrs:
                entry = output_eval.get(attr)
                if not isinstance(entry, dict):
                    continue
                icon   = "🔴" if entry.get("inferable") else "🟢"
                reason = entry.get("reasoning", "—")
                st.markdown(f'{icon} <span style="font-size:0.9em"><b>{attr}</b></span>',
                            unsafe_allow_html=True)
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
                    icon   = "🔴" if agg.get("inferable") else "🟢"
                    reason = agg.get("reasoning", "—")
                    st.markdown(f'{icon} <span style="font-size:0.9em"><b>{attr}</b></span>',
                                unsafe_allow_html=True)
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


def _delete_item_from_cache(cache_dir: Path, filename: str) -> bool:
    """Delete a single item JSON file from the cache directory."""
    try:
        # Construct the JSON filename from the original filename
        json_name = Path(filename).stem + ".json"
        item_path = cache_dir / json_name
        if item_path.exists():
            item_path.unlink()
            return True
        return False
    except Exception:
        return False


def _render_item(
    result: Dict[str, Any],
    unified_attrs: List[str],
    idx: int,
    cache_dir: Path | None = None,
    dataset_name: str = "unknown",
):
    filename    = result.get("filename", "unknown")
    status      = result.get("status", "")
    modality    = result.get("modality", "")
    input_item  = result.get("input_item", {})
    input_labels = result.get("input_labels", {})
    eval_prompt = result.get("eval_prompt", "prompt1")

    status_icon  = {"success": "✅", "failed": "❌"}.get(status, "")
    from_cache   = " (cached)" if result.get("from_cache") else ""
    positives    = [a for a in unified_attrs if input_labels.get(a, 0) == 1]
    label_suffix = f"  —  🏷 {', '.join(positives)}" if positives else ""
    data_type    = input_item.get("data_type", "")
    if data_type and not positives:
        label_suffix = f"  —  📄 {data_type}"

    with st.expander(f"{status_icon} {filename}{from_cache}{label_suffix}",
                     expanded=(status == "failed")):
        if status == "failed":
            st.error(f"Error: {result.get('error', 'Unknown error')}")
            # Show delete option even for failed items
            if cache_dir:
                st.divider()
                _render_item_delete_button(cache_dir, filename, idx)
            return

        detail_key = f"vioc_render_details_{idx}"
        show_details = bool(st.session_state.get(detail_key, False))
        if not show_details:
            if st.button("Load details", key=f"vioc_load_details_{idx}", use_container_width=True):
                st.session_state[detail_key] = True
                st.rerun()
            st.caption("Details are not rendered until requested. Click `Load details` to render this item.")
            if cache_dir:
                st.divider()
                _render_item_delete_button(cache_dir, filename, idx)
            return

        if st.button("Hide details", key=f"vioc_hide_details_{idx}", type="secondary"):
            st.session_state[detail_key] = False
            st.rerun()

        col_in, col_out = st.columns([1, 2])

        with col_in:
            st.markdown("**Input**")
            if modality == "image":
                # Try cache first (new results), fallback to dataset (legacy)
                image_b64 = input_item.get("image_base64")
                if not image_b64 and dataset_name != "unknown":
                    image_b64 = _load_original_image(filename, dataset_name)
                _display_image(image_b64, caption=filename)
                prompt_text = result.get("prompt_text", "").strip()
                if prompt_text:
                    st.caption("**Prompt sent to model:**")
                    st.text(prompt_text)
            elif modality == "text":
                st.text_area("Input text", value=input_item.get("text_content", ""),
                             height=220, disabled=True, label_visibility="collapsed",
                             key=f"vioc_text_{idx}")
            else:
                st.info("Media not stored in cache.")

            if positives:
                st.caption("**Annotated attributes:**")
                st.markdown("  ".join(f"`{a}`" for a in positives))

        with col_out:
            out_col, ext_col = st.columns(2)

            with out_col:
                st.markdown(f"**{STAGE_OUTPUT}**")
                st.text_area("Raw output text", value=result.get("output_text", ""),
                             height=200, disabled=True, label_visibility="collapsed",
                             key=f"vioc_out_{idx}")

            with ext_col:
                st.markdown(f"**{STAGE_EXT}**")
                ext_text = result.get("ext_text", "")
                _display_preview_text(
                    ext_text,
                    f"vioc_ext_{idx}",
                    empty_text="No externalizations captured.",
                    height=200,
                )

        st.divider()
        st.markdown("**Stage-wise attribute presence**")
        st.caption(
            "Externalized stage uses the aggregate externalized result."
            if eval_prompt == "prompt3"
            else "Externalized stage uses the stored ext_eval result."
        )

        if result.get("output_eval_error"):
            st.warning(f"Output evaluation error: {result['output_eval_error']}")
        if result.get("ext_eval_error"):
            st.warning(f"Externalization evaluation error: {result['ext_eval_error']}")

        if _has_prompt3_channel_data(result.get("ext_eval", {})):
            _render_attribute_heatmap(
                input_labels,
                result.get("output_eval", {}),
                result.get("ext_eval", {}),
                unified_attrs,
            )
        else:
            _stage_table(
                input_labels,
                result.get("output_eval", {}),
                result.get("ext_eval", {}),
                unified_attrs,
            )
        _reasoning_expander(
            result.get("output_eval", {}),
            result.get("ext_eval", {}),
            unified_attrs,
            idx,
        )

        # Per-item delete button
        if cache_dir:
            st.divider()
            _render_item_delete_button(cache_dir, filename, idx)


def _render_item_delete_button(cache_dir: Path, filename: str, idx: int):
    """Render a delete button for a single item inside an expander."""
    col1, col2 = st.columns([3, 1])
    with col1:
        confirm_key = f"vioc_del_confirm_{idx}"
        confirmed = st.checkbox("Confirm delete", key=confirm_key)
    with col2:
        if st.button("🗑️ Delete", key=f"vioc_del_btn_{idx}", type="secondary", disabled=not confirmed):
            if _delete_item_from_cache(cache_dir, filename):
                st.success(f"Deleted: {filename}")
                # Remove from session state items
                items_key = "vioc_items"
                if items_key in st.session_state:
                    st.session_state[items_key] = [
                        item for item in st.session_state[items_key]
                        if item.get("filename") != filename
                    ]
                st.rerun()
            else:
                st.error(f"Failed to delete: {filename}")


def _render_aggregated(all_results: List[Dict[str, Any]], unified_attrs: List[str]):
    st.markdown("**Exposure Heatmap**")
    _render_channel_aggregated_heatmap(all_results, unified_attrs)
    st.markdown("**Grouped Bar Chart**")
    _render_channel_aggregated(all_results, unified_attrs)


# ── Cache discovery ───────────────────────────────────────────────────────────

_IOC_ITEM_KEYS = {"ext_eval", "input_labels", "ext_text", "output_eval_ok"}


def _sniff_ioc_item(cache_dir: Path) -> Optional[dict]:
    """
    Read the first item JSON in cache_dir and return it if it looks like an IOC
    item (has ext_eval + input_labels keys).  Returns None otherwise.
    """
    for f in sorted(cache_dir.iterdir()):
        if f.suffix != ".json" or f.name == "run_config.json":
            continue
        try:
            item = json.loads(f.read_text())
            if _IOC_ITEM_KEYS.issubset(item.keys()):
                return item
        except Exception:
            pass
    return None


def _reconstruct_config(cache_dir: Path, sample_item: dict) -> dict:
    """Build a minimal run_config from a sample IOC item and save it for future use."""
    unified_attrs = (
        list(sample_item.get("output_eval", {}).keys())
        or list(sample_item.get("ext_eval", {}).keys())
    )
    cfg = {
        "app_name":          "unknown",
        "dataset_name":      "unknown",
        "modality":          sample_item.get("modality", "image"),
        "unified_attrs":     unified_attrs,
        "perturbation_method": "ioc_comparison",
        "evaluation_method": "openrouter",
    }
    # Persist so we don't have to sniff again next time
    try:
        (cache_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass
    return cfg


def _list_ioc_cache_dirs() -> list[tuple[Path, dict]]:
    """
    Return [(cache_dir, run_config), ...] for all IOC caches, newest first.

    Detection strategy (in order):
    1. run_config.json exists with perturbation_method == "ioc_comparison" → IOC cache.
    2. No run_config.json → sniff first item; if it has IOC-specific keys, treat as IOC
       cache and backfill run_config.json from the item data.
    3. run_config.json exists with any other perturbation_method → perturb-input cache,
       skip.
    """
    outputs_root = LANTERN_ROOT / "verify" / "outputs"
    if not outputs_root.exists():
        return []

    result = []
    for d in outputs_root.iterdir():
        if not d.is_dir() or not d.name.startswith("cache_"):
            continue

        config_path = d / "run_config.json"

        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
            except Exception:
                continue
            if cfg.get("perturbation_method") == "ioc_comparison":
                result.append((d, cfg))
        else:
            # No run_config — sniff the item schema
            sample = _sniff_ioc_item(d)
            if sample is not None:
                cfg = _reconstruct_config(d, sample)
                result.append((d, cfg))

    return sorted(result, key=lambda x: x[0].stat().st_mtime, reverse=True)


def _load_items_from_cache(cache_dir: Path) -> list[dict]:
    """Load all item JSON files from an IOC cache directory, newest first."""
    items = []
    for f in cache_dir.iterdir():
        if f.suffix == ".json" and f.name != "run_config.json":
            try:
                items.append(json.loads(f.read_text()))
            except Exception:
                pass
    return sorted(items, key=lambda r: r.get("filename", ""))


def _render_delete_section(cache_dir: Path):
    """Render a danger-zone delete button for the selected cache."""
    import shutil

    st.markdown("#### Danger Zone")
    st.caption(
        f"This will permanently delete the cache directory:\n\n"
        f"- `{cache_dir.name}`"
    )
    confirmed = st.checkbox("I understand this cannot be undone", key="vioc_delete_confirm")
    if st.button("🗑️ Delete this cache", type="primary", disabled=not confirmed):
        try:
            shutil.rmtree(cache_dir)
            st.success("Cache deleted.")
        except Exception as e:
            st.error(f"Deletion failed: {e}")
        for key in ("vioc_items", "vioc_config", "vioc_cache_dir", "vioc_delete_confirm"):
            st.session_state.pop(key, None)
        st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    st.title("🔬 View Input-Output Comparison Results")
    st.markdown("Browse cached IOC runs — stage-wise privacy attribute exposure analysis.")

    available = _list_ioc_cache_dirs()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Select Cache")
        st.toggle(
            "Show full externalizations",
            key="show_full_externalizations",
            value=False,
            help="Display the full captured externalization text instead of a shortened preview.",
        )
        st.divider()

        if available:
            def _label(cache_dir: Path, cfg: dict) -> str:
                app      = cfg.get("app_name", "?")
                dataset  = cfg.get("dataset_name", "?")
                modality = cfg.get("modality", "?")
                return f"{app} / {dataset} / {modality}  [{cache_dir.name}]"

            options = {_label(d, c): (d, c) for d, c in available}
            selected_label = st.selectbox(
                "Pick a cache",
                list(options.keys()),
                key="vioc_selector",
            )
            selected_dir, selected_cfg = options[selected_label]
        else:
            st.info("No IOC caches found in `verify/outputs/`.\nRun the **Input / Output Comparison** page first.")
            selected_dir = None
            selected_cfg = {}

        load_clicked = st.button("📂 Load", type="primary", use_container_width=True,
                                 disabled=selected_dir is None)

    # ── Load ──────────────────────────────────────────────────────────────────
    if load_clicked and selected_dir is not None:
        items = _load_items_from_cache(selected_dir)
        st.session_state["vioc_items"]     = items
        st.session_state["vioc_config"]    = selected_cfg
        st.session_state["vioc_cache_dir"] = str(selected_dir)

    # ── Render ────────────────────────────────────────────────────────────────
    items      = st.session_state.get("vioc_items")
    run_config = st.session_state.get("vioc_config", {})
    cache_dir_str = st.session_state.get("vioc_cache_dir", "")

    if items is None:
        st.info("Select a cache in the sidebar and click **📂 Load**.")
        return

    app_name      = run_config.get("app_name", "unknown")
    dataset_name  = run_config.get("dataset_name", "unknown")
    modality      = run_config.get("modality", "image")
    unified_attrs = run_config.get("unified_attrs", [])

    # Derive attrs from eval keys if not stored in config
    if not unified_attrs and items:
        for item in items:
            keys = list(item.get("output_eval", {}).keys()) or \
                   list(item.get("ext_eval", {}).keys())
            if keys:
                unified_attrs = keys
                break

    # ── Run info ──────────────────────────────────────────────────────────────
    st.subheader("Run Info")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("App",      app_name)
    c2.metric("Dataset",  dataset_name)
    c3.metric("Modality", modality)
    c4.metric("Items",    len(items))
    if unified_attrs:
        st.caption(f"**Attributes:** {', '.join(unified_attrs)}")
    st.caption(f"Cache: `{Path(cache_dir_str).name if cache_dir_str else '—'}`")

    # ── Aggregated chart ──────────────────────────────────────────────────────
    if unified_attrs:
        st.divider()
        st.subheader("Aggregated Attribute-wise Positive Rate")
        _render_aggregated(items, unified_attrs)

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Summary")
    statuses = [r.get("status", "") for r in items]
    m1, m2, m3 = st.columns(3)
    m1.metric("Total",      len(items))
    m2.metric("Successful", statuses.count("success"))
    m3.metric("Failed",     statuses.count("failed"))

    # ── Per-item results ──────────────────────────────────────────────────────
    st.divider()
    n = len(items)
    st.subheader(
        f"Results — {app_name} / {dataset_name} / {modality} "
        f"({n} item{'s' if n != 1 else ''})"
    )
    cache_dir_path = Path(cache_dir_str) if cache_dir_str else None
    for idx, result in enumerate(items):
        _render_item(result, unified_attrs, idx, cache_dir_path, dataset_name)

    # ── Delete ────────────────────────────────────────────────────────────────
    if cache_dir_str:
        st.divider()
        _render_delete_section(Path(cache_dir_str))


if __name__ == "__main__":
    main()
