"""
Verify — Perturb Input

Provides a unified UI to:
  1. Choose a target app, dataset, modality, and privacy attributes.
  2. Click Verify to run the pipeline progressively (item-by-item).
  3. View original vs. perturbed inputs and outputs side-by-side.
  4. See per-attribute privacy inferability results with bar charts.
  5. Download a structured JSON/CSV report.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

# Ensure the Lantern root is on the Python path so `verify.*` imports work
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

DISPLAY_PREVIEW_CHARS = 4000
CHANNEL_STAGE_ORDER = ["AGGREGATE", "UI", "NETWORK", "STORAGE", "LOGGING"]
CHANNEL_STAGE_LABELS = {
    "AGGREGATE": "Aggregate",
    "UI": "UI",
    "NETWORK": "Network",
    "STORAGE": "Storage",
    "LOGGING": "Logging",
}
CHANNEL_STAGE_COLORS = {
    "Aggregate": "#f0ad4e",
    "UI": "#7f8c8d",
    "Network": "#5b8def",
    "Storage": "#27ae60",
    "Logging": "#f39c12",
}

# ─── Page config ─────────────────────────────────────────────────────────────

# ─── Lazy imports (backend modules) ──────────────────────────────────────────

@st.cache_resource
def _load_config():
    from verify.backend.utils.config import (
        load_dataset_list,
        list_target_apps,
        load_perturbation_method_map,
    )
    return {
        "datasets": load_dataset_list(),
        "apps": list_target_apps(),
        "perturbation_map": load_perturbation_method_map(),
    }


@st.cache_data
def _load_attributes(modality: str) -> list[str]:
    from verify.backend.utils.config import load_attribute_list
    return load_attribute_list(modality)


def get_adapter_status(app_name: str) -> tuple[bool, str]:
    """Return (available, reason) for a given app adapter."""
    from verify.backend.adapters import get_adapter
    adapter = get_adapter(app_name)
    if adapter is None:
        return False, f"No adapter registered for '{app_name}'."
    return adapter.check_availability()


def get_perturbation_status(modality: str, method: str) -> tuple[bool, str]:
    from verify.backend.perturbation_method.interface import check_perturbation_availability
    return check_perturbation_availability(modality, method)


def list_perturbation_methods(modality: str) -> list[str]:
    from verify.backend.perturbation_method.interface import list_methods_for_modality
    return list_methods_for_modality(modality)


# ─── Helpers ─────────────────────────────────────────────────────────────────

KNOWN_APPS = [
    "momentag", "clone", "snapdo", "xend", "budget-lens",
    "deeptutor", "llm-vtuber", "skin-disease-detection", "google-ai-edge-gallery",
    "tool-neuron",
    "chat-driven-expense-tracker",
    "photomath",
    "replika",
    "expensify",
]
MODALITIES = ["image", "text", "video"]


def _display_image(b64_str: str | None, data=None, caption: str = ""):
    """Display a PIL image or base64 string."""
    try:
        if b64_str:
            import base64
            from PIL import Image as PILImage
            import io
            img_data = base64.b64decode(b64_str)
            img = PILImage.open(io.BytesIO(img_data))
            st.image(img, caption=caption, use_container_width=True)
        elif data is not None:
            st.image(data, caption=caption, use_container_width=True)
        else:
            st.warning("No image data available.")
    except Exception as e:
        st.error(f"Could not display image: {e}")


def _display_text(text: str, label: str = "", key_suffix: str = ""):
    """Display text content in a styled box."""
    if label:
        st.markdown(f"**{label}**")
    
    # Provide a unique label for accessibility, but keep it hidden.
    unique_label = f"text_area_{key_suffix}" if key_suffix else "text_area"
    st.text_area(unique_label, value=text, height=200, disabled=True, 
                 label_visibility="collapsed", key=f"text_{key_suffix}")


def _display_externalized_preview(text: str, key_suffix: str = ""):
    show_full = st.session_state.get("show_full_externalizations", False)
    if show_full or len(text) <= DISPLAY_PREVIEW_CHARS:
        value = text
    else:
        value = text[:DISPLAY_PREVIEW_CHARS] + "\n\n[truncated for display]"
    st.text_area(
        f"ext_area_{key_suffix}",
        value=value,
        height=140,
        disabled=True,
        label_visibility="collapsed",
        key=f"ext_{key_suffix}",
    )
    if not show_full and len(text) > DISPLAY_PREVIEW_CHARS:
        st.caption(
            f"Display preview limited to {DISPLAY_PREVIEW_CHARS:,} chars. "
            "Enable `Show full externalizations` in the sidebar to inspect the full captured text."
        )


def _display_frames(frames: list, caption_prefix: str = "Frame"):
    """Display video frames as a grid."""
    if not frames:
        st.warning("No frames available.")
        return
    cols = st.columns(min(len(frames), 4))
    for i, (col, frame) in enumerate(zip(cols, frames)):
        with col:
            st.image(frame, caption=f"{caption_prefix} {i+1}", use_container_width=True)


def _eval_chart(eval_results: dict, stage_label: str):
    """Render a bar chart of inferability scores for an evaluation result."""
    import pandas as pd

    if not eval_results:
        st.info("No evaluation results available.")
        return

    rows = []
    for attr, info in eval_results.items():
        score = info.get("score", 0.0) if isinstance(info, dict) else 0.0
        inferable = info.get("inferable", False) if isinstance(info, dict) else False
        rows.append({"Attribute": attr, "Inferability Score": score, "Inferable": inferable})

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Color text by inferability
    for _, row in df.iterrows():
        attr = row["Attribute"]
        score = row["Inferability Score"]
        inferable = row["Inferable"]
        icon = "🔴" if inferable else "🟢"
        st.markdown(f"{icon} **{attr}**: score={score:.2f} — {'Inferable' if inferable else 'Not inferable'}")

        reasoning = eval_results.get(attr, {}).get("reasoning", "")
        if reasoning:
            st.caption(reasoning)

    import altair as alt
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Attribute:N", sort=None),
            y=alt.Y("Inferability Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(
                alt.datum["Inferable"] == True,
                alt.value("#d9534f"),
                alt.value("#5cb85c"),
            ),
        )
        .properties(height=200)
    )
    st.altair_chart(chart, use_container_width=True)


def _has_channelwise_eval(eval_results: dict) -> bool:
    return any(is_channelwise_eval_entry(entry) for entry in eval_results.values())


def _render_eval_heatmap(eval_results: dict, stage_label: str):
    import pandas as pd
    import altair as alt

    attrs = list(eval_results.keys())
    rows = []
    for attr in attrs:
        entry = eval_results.get(attr)
        agg = get_aggregate_eval_entry(entry)
        rows.append(
            {
                "Stage": f"{stage_label} Aggregate",
                "Attribute": attr,
                "state": "yes" if agg.get("inferable") else "no",
            }
        )
        channels = get_channel_eval_entries(entry)
        for channel in CHANNEL_STAGE_ORDER[1:]:
            state = "na"
            if channel in channels:
                state = "yes" if channels[channel].get("inferable") else "no"
            rows.append(
                {
                    "Stage": f"{stage_label} {CHANNEL_STAGE_LABELS[channel]}",
                    "Attribute": attr,
                    "state": state,
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    stage_order = [f"{stage_label} Aggregate"] + [f"{stage_label} {CHANNEL_STAGE_LABELS[c]}" for c in CHANNEL_STAGE_ORDER[1:]]
    rect = (
        alt.Chart(df)
        .mark_rect(stroke="#ffffff", strokeWidth=1)
        .encode(
            x=alt.X("Attribute:N", sort=attrs, axis=alt.Axis(labelAngle=-40, labelFontSize=10)),
            y=alt.Y("Stage:N", sort=stage_order, title=None),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(domain=["yes", "no", "na"], range=["#d4f5d4", "#fafafa", "#e6e6e6"]),
                legend=None,
            ),
            tooltip=["Stage", "Attribute", alt.Tooltip("state:N", title="Status")],
        )
    )
    text = (
        alt.Chart(df[df["state"] == "yes"])
        .mark_text(text="✓", fontSize=12, fontWeight="bold", color="#2f4f2f")
        .encode(x=alt.X("Attribute:N", sort=attrs), y=alt.Y("Stage:N", sort=stage_order))
    )
    st.altair_chart((rect + text).properties(height=max(180, 34 * len(stage_order))), use_container_width=True)
    st.caption("Green = inferable, white = not inferable, grey = channel not available in this evaluation.")


def _render_channel_aggregated_chart(all_results: list[dict], attributes: list[str]):
    import pandas as pd
    import altair as alt

    success = [r for r in all_results if r.get("status") == "success"]
    if not success or not attributes:
        st.info("No successful items to aggregate.")
        return

    include_channelwise = any(
        _has_channelwise_eval((r.get("evaluation", {}) or {}).get(stage, {}))
        for r in success
        for stage in ("original", "perturbed")
    )
    rows = []
    for attr in attributes:
        for stage_name, key in [("Original", "original"), ("Perturbed", "perturbed")]:
            stage_eval_list = [(r.get("evaluation", {}) or {}).get(key, {}) for r in success]
            support = len(success)
            positive = sum(1 for ev in stage_eval_list if bool(get_aggregate_eval_entry(ev.get(attr)).get("inferable")))
            rows.append({"Attribute": attr, "Stage": f"{stage_name} Aggregate", "Positive Rate": positive / support, "Support": support})
            if include_channelwise:
                for channel in CHANNEL_STAGE_ORDER[1:]:
                    ch_support = 0
                    ch_positive = 0
                    for ev in stage_eval_list:
                        channels = get_channel_eval_entries(ev.get(attr))
                        if channel in channels:
                            ch_support += 1
                            if channels[channel].get("inferable"):
                                ch_positive += 1
                    if ch_support > 0:
                        rows.append(
                            {
                                "Attribute": attr,
                                "Stage": f"{stage_name} {CHANNEL_STAGE_LABELS[channel]}",
                                "Positive Rate": ch_positive / ch_support,
                                "Support": ch_support,
                            }
                        )
    df = pd.DataFrame(rows)
    stage_order = list(dict.fromkeys(df["Stage"].tolist()))
    color_range = []
    for label in stage_order:
        if label.endswith("Aggregate"):
            color_range.append(CHANNEL_STAGE_COLORS["Aggregate"] if label.startswith("Original") else "#b9770e")
        elif label.endswith("UI"):
            color_range.append(CHANNEL_STAGE_COLORS["UI"])
        elif label.endswith("Network"):
            color_range.append(CHANNEL_STAGE_COLORS["Network"])
        elif label.endswith("Storage"):
            color_range.append(CHANNEL_STAGE_COLORS["Storage"])
        elif label.endswith("Logging"):
            color_range.append(CHANNEL_STAGE_COLORS["Logging"])
        else:
            color_range.append("#999999")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Attribute:N", sort=attributes, axis=alt.Axis(labelAngle=-40, labelFontSize=10)),
            xOffset=alt.XOffset("Stage:N", sort=stage_order),
            y=alt.Y("Positive Rate:Q", scale=alt.Scale(domain=[0, 1]), title="Positive rate"),
            color=alt.Color("Stage:N", sort=stage_order, scale=alt.Scale(domain=stage_order, range=color_range)),
            tooltip=["Attribute", "Stage", alt.Tooltip("Positive Rate:Q", format=".2f"), "Support"],
        )
        .properties(height=320, title=f"Attribute-wise positive rate across {len(success)} item(s)")
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Aggregate bars use all successful items. Channel bars use only items where that channel exists in the saved evaluation.")


def _render_generated_task(result: dict):
    """If the result carries a snapdo-generated task, render it as an info banner."""
    generated_task = (
        result.get("original_output", {}).get("metadata", {}).get("generated_task")
        or result.get("perturbed_output", {}).get("metadata", {}).get("generated_task")
    )
    if not generated_task:
        return
    title = generated_task.get("title", "")
    description = generated_task.get("description", "")
    st.markdown("### Generated Task Context")
    st.info(
        f"**📋 Task title:** {title}"
        + (f"\n\n**Description:** {description}" if description else "")
    )


def _render_item_result(result: dict):
    """Render a single item result inside an expander."""
    filename = result.get("filename", "Unknown")
    status = result.get("status", "")
    modality = result.get("original_input", {}).get("modality", "")

    status_icon = {"success": "✅", "failed": "❌", "skipped": "⚠️"}.get(status, "")
    from_cache = " (cached)" if result.get("from_cache") else ""

    # Build label suffix for the expander title
    orig_input = result.get("original_input", {})
    _privacy_labels = orig_input.get("privacy_labels", [])
    _data_type = orig_input.get("data_type", "")
    _label_suffix = ""
    if _privacy_labels:
        _label_suffix = f"  —  🏷 {', '.join(_privacy_labels)}"
    elif _data_type:
        _label_suffix = f"  —  📄 {_data_type}"

    with st.expander(f"{status_icon} {filename}{from_cache}{_label_suffix}", expanded=(status == "failed")):
        if status == "failed":
            st.error(f"Error: {result.get('error', 'Unknown error')}")
            return

        if status == "skipped":
            st.warning(result.get("perturbation_warning", "Item was skipped."))
            # Still show original output if available
            orig_out = result.get("original_output", {})
            if orig_out and orig_out.get("success"):
                st.markdown("**Original Output (perturbation skipped):**")
                st.text(orig_out.get("output_text", ""))
            return

        orig_input = result.get("original_input", {})

        # ── Label metadata line ───────────────────────────────────────────
        _pl = orig_input.get("privacy_labels", [])
        _dt = orig_input.get("data_type", "")
        _dta = orig_input.get("data_type_attributes", [])
        if _pl:
            st.caption(f"🏷 **Privacy labels:** {', '.join(_pl)}")
        if _dt:
            _dta_str = f"  →  attributes: `{', '.join(_dta)}`" if _dta else "  →  *unmapped*"
            st.caption(f"📄 **Data type:** {_dt}{_dta_str}")
        pert_input = result.get("perturbed_input", {})
        orig_out = result.get("original_output", {})
        pert_out = result.get("perturbed_output", {})
        evaluation = result.get("evaluation", {})

        # ── Generated task (snapdo only) ──────────────────────────────────
        _render_generated_task(result)

        # ── Original vs Perturbed Input ──────────────────────────────────
        st.markdown("### Input Comparison")
        col_orig, col_pert = st.columns(2)

        with col_orig:
            st.markdown("**Original Input**")
            if modality == "image":
                _display_image(result.get("_original_image_b64"), result.get("_original_data"))
            elif modality == "text":
                _display_text(orig_input.get("text_content", ""), "", key_suffix=f"orig_{filename}")
            elif modality == "video":
                _display_frames(result.get("_original_frames", []), "Frame")

        with col_pert:
            st.markdown("**Perturbed Input**")
            pert_info = pert_input.get("perturbation_applied", {})
            if modality == "image":
                _display_image(result.get("_perturbed_image_b64"), result.get("_perturbed_data"))
            elif modality == "text":
                _display_text(pert_input.get("text_content", ""), "", key_suffix=f"pert_{filename}")
            elif modality == "video":
                _display_frames(result.get("_perturbed_frames", []), "Frame")
            if pert_info:
                method = pert_info.get("method", "")
                attrs = pert_info.get("attributes", [])
                st.caption(f"Method: {method} | Attributes: {', '.join(attrs)}")

        st.divider()

        # ── Inference Results ─────────────────────────────────────────────
        st.markdown("### Inference Results")
        col_orig_out, col_pert_out = st.columns(2)

        with col_orig_out:
            st.markdown("**Original Input → App Output**")
            if orig_out and orig_out.get("success"):
                st.text_area(
                    "Output",
                    value=orig_out.get("output_text", ""),
                    height=180,
                    disabled=True,
                    key=f"orig_out_{filename}",
                )

                # Render Externalizations
                exts = orig_out.get("externalizations", {})
                if exts:
                    st.markdown("**🌐 Captured Externalizations**")
                    with st.container():
                        for channel, content in exts.items():
                            st.markdown(f"**{channel}**")
                            _display_externalized_preview(content, f"orig_{filename}_{channel}")

                structured = orig_out.get("structured_output", {})
                if structured:
                    st.markdown("**Structured output**")
                    with st.container():
                        st.json(structured)
            else:
                st.error(orig_out.get("error", "Pipeline failed.") if orig_out else "No output.")

        with col_pert_out:
            st.markdown("**Perturbed Input → App Output**")
            if pert_out and pert_out.get("success"):
                st.text_area(
                    "Output",
                    value=pert_out.get("output_text", ""),
                    height=180,
                    disabled=True,
                    key=f"pert_out_{filename}",
                )

                # Render Externalizations
                exts = pert_out.get("externalizations", {})
                if exts:
                    st.markdown("**🌐 Captured Externalizations**")
                    with st.container():
                        # Handle phase-aware structure: {"DURING": {...}, "POST": {...}}
                        if "DURING" in exts or "POST" in exts:
                            for phase in ["DURING", "POST"]:
                                phase_data = exts.get(phase, {})
                                if not phase_data:
                                    continue
                                
                                label = "Step 1: Inference Process" if phase == "DURING" else "Step 2: Externalization / UI"
                                st.markdown(f"**{label}**")
                                for channel, content in phase_data.items():
                                    st.markdown(f"*{channel}*")
                                    _display_externalized_preview(content, f"pert_{filename}_{phase}_{channel}")
                                if phase == "DURING" and "POST" in exts:
                                    st.divider()
                        else:
                            # Fallback for old flat dictionary structure
                            for channel, content in exts.items():
                                st.markdown(f"**[{channel}]**")
                                _display_externalized_preview(content, f"pert_{filename}_{channel}")

                structured = pert_out.get("structured_output", {})
                if structured:
                    st.markdown("**Structured output**")
                    with st.container():
                        st.json(structured)
            else:
                st.error(pert_out.get("error", "Pipeline failed.") if pert_out else "No output.")

        st.divider()

        # ── Privacy Evaluation ────────────────────────────────────────────
        st.markdown("### Privacy Inferability")

        orig_eval = evaluation.get("original", {})
        pert_eval = evaluation.get("perturbed", {})
        orig_eval_ok = evaluation.get("original_success", False)
        pert_eval_ok = evaluation.get("perturbed_success", False)

        col_eval_orig, col_eval_pert = st.columns(2)

        with col_eval_orig:
            st.markdown("**From original output:**")
            if orig_eval_ok and orig_eval:
                if _has_channelwise_eval(orig_eval):
                    _render_eval_heatmap(orig_eval, "Original")
                else:
                    _eval_chart(orig_eval, "Original")
            elif evaluation.get("original_error"):
                st.error(f"Evaluation error: {evaluation['original_error']}")
            else:
                st.info("No evaluation results.")

        with col_eval_pert:
            st.markdown("**From perturbed output:**")
            if pert_eval_ok and pert_eval:
                if _has_channelwise_eval(pert_eval):
                    _render_eval_heatmap(pert_eval, "Perturbed")
                else:
                    _eval_chart(pert_eval, "Perturbed")
            elif evaluation.get("perturbed_error"):
                st.error(f"Evaluation error: {evaluation['perturbed_error']}")
            else:
                st.info("No evaluation results.")


def _render_aggregated_chart(all_results: list, attributes: list):
    """Render an aggregated bar chart across all processed items."""
    _render_channel_aggregated_chart(all_results, attributes)
    return
    import pandas as pd

    if not all_results or not attributes:
        return

    # Collect scores
    data = []
    for attr in attributes:
        orig_scores = []
        pert_scores = []
        for r in all_results:
            eval_r = r.get("evaluation", {})
            if eval_r:
                o = eval_r.get("original", {}).get(attr, {})
                p = eval_r.get("perturbed", {}).get(attr, {})
                if isinstance(o, dict) and "score" in o:
                    orig_scores.append(o["score"])
                if isinstance(p, dict) and "score" in p:
                    pert_scores.append(p["score"])

        data.append({
            "Attribute": attr,
            "Original (avg score)": sum(orig_scores) / len(orig_scores) if orig_scores else 0.0,
            "Perturbed (avg score)": sum(pert_scores) / len(pert_scores) if pert_scores else 0.0,
        })

    if not data:
        st.info("No aggregated data to display yet.")
        return

    import altair as alt
    df = pd.DataFrame(data)
    df_melted = df.melt("Attribute", var_name="Stage", value_name="Avg Inferability Score")
    chart = (
        alt.Chart(df_melted)
        .mark_bar()
        .encode(
            x=alt.X("Attribute:N", sort=None),
            y=alt.Y("Avg Inferability Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Stage:N", scale=alt.Scale(
                domain=["Original (avg score)", "Perturbed (avg score)"],
                range=["#d9534f", "#5bc0de"],
            )),
            xOffset="Stage:N",
        )
        .properties(height=250)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Average inferability score across all items (lower is better after perturbation).")


# ─── Main UI ─────────────────────────────────────────────────────────────────

def _sync_app_modes() -> None:
    """Push per-app mode choices from session state into the config module."""
    from verify.backend.utils.config import set_app_mode_override
    for app_name, mode in st.session_state.get("app_modes", {}).items():
        set_app_mode_override(app_name, mode)


def main():
    st.title("🔍 Perturb Input")
    st.markdown(
        "Evaluate whether privacy attributes survive through target app AI pipelines. "
        "Compare original and perturbed inputs/outputs side-by-side."
    )

    _sync_app_modes()

    config = _load_config()
    datasets = config["datasets"]
    all_apps = config["apps"]
    perturbation_map = config["perturbation_map"]

    # Keep a stable ordering for historically prominent apps, but still expose
    # every adapter discovered from the backend registry.
    recognized_apps = [a for a in all_apps if a in KNOWN_APPS]
    other_apps = [a for a in all_apps if a not in KNOWN_APPS]

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        st.toggle(
            "Show full externalizations",
            key="show_full_externalizations",
            value=False,
            help="Display the full captured externalization text instead of a shortened preview.",
        )
        st.divider()

        # App dropdown
        st.subheader("Target App")
        app_options = recognized_apps + [f"{a} (unrecognized)" for a in other_apps]
        if not app_options:
            st.error("No target apps found in target-apps/.")
            return

        selected_app_display = st.selectbox("Select app", app_options)
        selected_app = selected_app_display.split(" (")[0]  # strip " (unrecognized)"

        # Show adapter availability
        with st.spinner(f"Checking {selected_app} availability..."):
            try:
                from verify.backend.utils.config import set_current_app_context
                set_current_app_context(selected_app)
                app_available, app_msg = get_adapter_status(selected_app)
                if app_available:
                    st.success(f"Available: {app_msg}")
                else:
                    st.error(f"Unavailable: {app_msg}")
            except Exception as e:
                app_available = False
                st.error(f"Adapter error: {e}")

        st.divider()

        # Dataset dropdown
        st.subheader("Dataset")
        if not datasets:
            st.error("No datasets in config/dataset_list.txt.")
            return
        selected_dataset = st.selectbox("Select dataset", datasets)

        st.divider()

        # Modality dropdown
        st.subheader("Modality")
        selected_modality = st.selectbox("Select modality", MODALITIES)
        generation_task = "text"
        if selected_app == "tool-neuron" and selected_modality == "text":
            st.divider()
            st.subheader("ToolNeuron Task")
            generation_task = st.radio(
                "Generation task",
                ["text", "image"],
                key="perturb_toolneuron_generation_task",
                help="ToolNeuron can run either text generation or text-to-image generation from text inputs.",
            )

        # Load attributes for the selected modality
        all_attributes = _load_attributes(selected_modality)

        # Perturbation method dropdown
        available_methods = list_perturbation_methods(selected_modality)
        if available_methods:
            selected_pert_method = st.selectbox("Perturbation method", available_methods)
        else:
            selected_pert_method = perturbation_map.get(selected_modality, "")
            st.caption(f"Perturbation method: **{selected_pert_method or 'None configured'}**")
        try:
            pert_available, pert_msg = get_perturbation_status(selected_modality, selected_pert_method)
            if pert_available:
                st.success("Perturbation ready")
            else:
                st.warning(f"Perturbation unavailable: {pert_msg}")
        except Exception as e:
            pert_available = False
            st.warning(f"Could not check perturbation: {e}")

        # Imago Obscura mode selector (only shown when that method is selected)
        perturbation_kwargs: dict = {}
        if selected_pert_method == "Imago_Obscura":
            imago_mode = st.selectbox(
                "Obscura mode",
                ["blur", "pixelate", "fill"],
                help="blur: Gaussian blur · pixelate: mosaic · fill: solid grey",
            )
            perturbation_kwargs["mode"] = imago_mode

        st.divider()

        # Attribute checklist
        st.subheader("Privacy Attributes")
        if not all_attributes:
            st.warning("No attributes in config/attribute_list.txt.")
            selected_attributes = []
        else:
            selected_attributes = []
            for attr in all_attributes:
                if st.checkbox(attr.capitalize(), value=False, key=f"attr_{attr}"):
                    selected_attributes.append(attr)

        st.divider()

        # Max items
        st.subheader("Item Limit")
        limit_enabled = st.checkbox("Limit number of items", value=True)
        max_items = None
        if limit_enabled:
            max_items = int(st.number_input("Max items to process", min_value=1, value=1, step=1))

        st.divider()

        # Cache option
        use_cache = st.checkbox("Use cache (skip already-processed items)", value=False)

        # Verify button
        verify_clicked = st.button(
            "▶ Verify",
            type="primary",
            disabled=not (app_available and selected_attributes),
            use_container_width=True,
        )

        if not selected_attributes:
            st.caption("Select at least one attribute to enable Verify.")
        if not app_available:
            st.caption(f"App '{selected_app}' is not available.")

    # ── Main panel ────────────────────────────────────────────────────────────

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "run_config" not in st.session_state:
        st.session_state.run_config = {}
    if "last_run_config" not in st.session_state:
        st.session_state.last_run_config = {}   # config that produced stored results
    if "status_messages" not in st.session_state:
        st.session_state.status_messages = []
    if "items_processed" not in st.session_state:
        st.session_state.items_processed = 0
    if "items_total" not in st.session_state:
        st.session_state.items_total = 0
    if "current_item" not in st.session_state:
        st.session_state.current_item = ""

    # Start a new run
    if verify_clicked:
        st.session_state.results = []
        st.session_state.summary = None
        st.session_state.status_messages = []
        st.session_state.processing = True
        st.session_state.items_processed = 0
        st.session_state.current_item = ""
        run_cfg = {
            "app": selected_app,
            "dataset": selected_dataset,
            "modality": selected_modality,
            "generation_task": generation_task,
            "attributes": selected_attributes,
            "pert_method": selected_pert_method,
            "max_items": max_items,
        }
        st.session_state.run_config = {
            "app": selected_app,
            "dataset": selected_dataset,
            "modality": selected_modality,
            "generation_task": generation_task,
            "attributes": selected_attributes,
        }
        st.session_state.last_run_config = run_cfg   # mark as authoritative
        # Pre-count dataset size for the progress bar (capped by max_items if set)
        from verify.frontend.utils import count_dataset_items
        total = count_dataset_items(selected_dataset, selected_modality)
        st.session_state.items_total = min(total, max_items) if max_items else total

        from verify.backend.orchestrator import Orchestrator
        orch = Orchestrator(
            app_name=selected_app,
            dataset_name=selected_dataset,
            modality=selected_modality,
            attributes=selected_attributes,
            use_cache=use_cache,
            max_items=max_items,
            perturbation_method=selected_pert_method,
            perturbation_kwargs=perturbation_kwargs,
            adapter_kwargs={"generation_task": generation_task} if selected_app == "tool-neuron" else None,
        )
        st.session_state._generator = orch.run()
        st.rerun()

    # ── Display ────────────────────────────────────────────────────────────
    # Rendered BEFORE the processing block so it is visible on every rerun
    # (st.rerun() stops execution, so anything after it is skipped).

    if st.session_state.processing:
        # ── Active run: show ONLY the progress bar, nothing else ──────────
        rc = st.session_state.run_config
        processed = st.session_state.items_processed
        total = st.session_state.items_total

        st.markdown(
            f"**{rc.get('app')}** &nbsp;·&nbsp; {rc.get('dataset')} &nbsp;·&nbsp; "
            f"{rc.get('modality')}"
            + (
                f" &nbsp;·&nbsp; task=`{rc.get('generation_task')}`"
                if rc.get("app") == "tool-neuron" and rc.get("modality") == "text"
                else ""
            )
            + f" &nbsp;·&nbsp; `{'`, `'.join(rc.get('attributes', []))}`"
        )

        if total > 0:
            fraction = min(processed / total, 1.0)
            st.progress(fraction, text=f"Processed {processed} / {total}")
        else:
            st.progress(0, text=f"Processed {processed} item{'s' if processed != 1 else ''}")
            st.caption("Counting items…")

        next_num = processed + 1
        if total == 0 or next_num <= total:
            st.caption(f"⏳ Processing item {next_num}" + (f" of {total}" if total > 0 else "") + "…")

    else:
        # ── Idle: status messages, then results or instructions ───────────
        for msg in st.session_state.status_messages:
            if msg.startswith("❌"):
                st.error(msg)
            elif msg.startswith("⚠️"):
                st.warning(msg)
            else:
                st.info(msg)

        if st.session_state.results:
            last_cfg = st.session_state.last_run_config
            stale = bool(last_cfg) and (
                selected_app != last_cfg.get("app")
                or selected_dataset != last_cfg.get("dataset")
                or selected_modality != last_cfg.get("modality")
                or sorted(selected_attributes) != sorted(last_cfg.get("attributes") or [])
                or selected_pert_method != last_cfg.get("pert_method")
                or max_items != last_cfg.get("max_items")
            )

            if stale:
                st.info(
                    f"Results below are from a previous run "
                    f"(**{last_cfg.get('app', '?')}** / {last_cfg.get('dataset', '?')} "
                    f"/ {last_cfg.get('modality', '?')}). "
                    "Click **▶ Verify** to run with the current configuration."
                )
            else:
                n = len(st.session_state.results)
                rc = st.session_state.run_config
                st.subheader(
                    f"Results — {rc.get('app', '')} / {rc.get('dataset', '')} "
                    f"/ {rc.get('modality', '')} ({n} item{'s' if n != 1 else ''})"
                )

                for result in st.session_state.results:
                    _render_item_result(result)

                st.divider()
                st.subheader("Aggregated Privacy Analysis")
                attrs = st.session_state.run_config.get("attributes", [])
                _render_aggregated_chart(st.session_state.results, attrs)

    # Summary
    if st.session_state.summary and not st.session_state.processing:
        summary = st.session_state.summary
        st.divider()
        st.subheader("Run Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total items", summary.get("total", 0))
        col2.metric("Successful", summary.get("success", 0))
        col3.metric("Skipped", summary.get("skipped", 0))
        col4.metric("Failed", summary.get("failed", 0))

        run_dir = summary.get("run_dir", "")
        if run_dir:
            st.caption(f"Output directory: `{run_dir}`")

        # Aggregated scores table
        agg = summary.get("aggregated_scores", {})
        if agg:
            st.markdown("**Average inferability scores:**")
            import pandas as pd

            rows = []
            for attr in summary.get("attributes", []):
                orig_score = agg.get("original", {}).get(attr)
                pert_score = agg.get("perturbed", {}).get(attr)
                rows.append({
                    "Attribute": attr,
                    "Orig score (avg)": f"{orig_score:.3f}" if orig_score is not None else "N/A",
                    "Pert score (avg)": f"{pert_score:.3f}" if pert_score is not None else "N/A",
                    "Delta": (
                        f"{(pert_score - orig_score):.3f}"
                        if (orig_score is not None and pert_score is not None)
                        else "N/A"
                    ),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Download buttons
        st.subheader("Download Report")
        report_dir = Path(run_dir) if run_dir else None

        col_json, col_csv = st.columns(2)
        with col_json:
            json_path = report_dir / "report.json" if report_dir else None
            if json_path and json_path.exists():
                st.download_button(
                    "Download JSON report",
                    data=json_path.read_text(),
                    file_name="verify_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("JSON report not available.")

        with col_csv:
            csv_path = report_dir / "report.csv" if report_dir else None
            if csv_path and csv_path.exists():
                st.download_button(
                    "Download CSV report",
                    data=csv_path.read_text(),
                    file_name="verify_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("CSV report not available.")

    if not st.session_state.processing and not st.session_state.results and not st.session_state.summary:
        st.markdown(
            """
            ### How to use Verify

            1. **Select a target app** from the sidebar (e.g. `snapdo`, `momentag`).
            2. **Select a dataset** (e.g. `VISPR` for images, `PrivacyLens` for text).
            3. **Select the modality** that matches your dataset.
            4. **Check the privacy attributes** you want to test (location, identity, etc.).
            5. Click **▶ Verify** to run the full pipeline.

            Results will appear progressively as each dataset item is processed.
            """
        )


if __name__ == "__main__":
    main()
