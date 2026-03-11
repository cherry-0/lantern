"""
Verify — Streamlit frontend

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
LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
VERIFY_ROOT = Path(__file__).resolve().parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Verify — Privacy Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Lazy imports (backend modules) ──────────────────────────────────────────

@st.cache_resource
def _load_config():
    from verify.backend.utils.config import (
        load_dataset_list,
        load_attribute_list,
        list_target_apps,
        load_perturbation_method_map,
    )
    return {
        "datasets": load_dataset_list(),
        "attributes": load_attribute_list(),
        "apps": list_target_apps(),
        "perturbation_map": load_perturbation_method_map(),
    }


def get_adapter_status(app_name: str) -> tuple[bool, str]:
    """Return (available, reason) for a given app adapter."""
    from verify.backend.adapters import get_adapter
    adapter = get_adapter(app_name)
    if adapter is None:
        return False, f"No adapter registered for '{app_name}'."
    return adapter.check_availability()


def get_perturbation_status(modality: str) -> tuple[bool, str]:
    from verify.backend.perturbation_method.interface import check_perturbation_availability
    return check_perturbation_availability(modality)


# ─── Helpers ─────────────────────────────────────────────────────────────────

KNOWN_APPS = ["momentag", "clone", "snapdo", "xend"]
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


def _display_text(text: str, label: str = ""):
    """Display text content in a styled box."""
    if label:
        st.markdown(f"**{label}**")
    st.text_area("", value=text, height=200, disabled=True, label_visibility="collapsed")


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

    st.bar_chart(df.set_index("Attribute")["Inferability Score"])


def _render_item_result(result: dict):
    """Render a single item result inside an expander."""
    filename = result.get("filename", "Unknown")
    status = result.get("status", "")
    modality = result.get("original_input", {}).get("modality", "")

    status_icon = {"success": "✅", "failed": "❌", "skipped": "⚠️"}.get(status, "")
    from_cache = " (cached)" if result.get("from_cache") else ""

    with st.expander(f"{status_icon} {filename}{from_cache}", expanded=(status == "failed")):
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
        pert_input = result.get("perturbed_input", {})
        orig_out = result.get("original_output", {})
        pert_out = result.get("perturbed_output", {})
        evaluation = result.get("evaluation", {})

        # ── Original vs Perturbed Input ──────────────────────────────────
        st.markdown("### Input Comparison")
        col_orig, col_pert = st.columns(2)

        with col_orig:
            st.markdown("**Original Input**")
            if modality == "image":
                _display_image(result.get("_original_image_b64"), result.get("_original_data"))
            elif modality == "text":
                _display_text(orig_input.get("text_content", ""), "")
            elif modality == "video":
                _display_frames(result.get("_original_frames", []), "Frame")

        with col_pert:
            st.markdown("**Perturbed Input**")
            pert_info = pert_input.get("perturbation_applied", {})
            if pert_info:
                method = pert_info.get("method", "")
                attrs = pert_info.get("attributes", [])
                st.caption(f"Method: {method} | Attributes: {', '.join(attrs)}")

            if modality == "image":
                _display_image(result.get("_perturbed_image_b64"), result.get("_perturbed_data"))
            elif modality == "text":
                _display_text(pert_input.get("text_content", ""), "")
            elif modality == "video":
                _display_frames(result.get("_perturbed_frames", []), "Frame")

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
                structured = orig_out.get("structured_output", {})
                if structured:
                    with st.expander("Structured output"):
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
                structured = pert_out.get("structured_output", {})
                if structured:
                    with st.expander("Structured output"):
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
                _eval_chart(orig_eval, "Original")
            elif evaluation.get("original_error"):
                st.error(f"Evaluation error: {evaluation['original_error']}")
            else:
                st.info("No evaluation results.")

        with col_eval_pert:
            st.markdown("**From perturbed output:**")
            if pert_eval_ok and pert_eval:
                _eval_chart(pert_eval, "Perturbed")
            elif evaluation.get("perturbed_error"):
                st.error(f"Evaluation error: {evaluation['perturbed_error']}")
            else:
                st.info("No evaluation results.")


def _render_aggregated_chart(all_results: list, attributes: list):
    """Render an aggregated bar chart across all processed items."""
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

    df = pd.DataFrame(data).set_index("Attribute")
    st.bar_chart(df)
    st.caption("Average inferability score across all items (lower is better after perturbation).")


# ─── Main UI ─────────────────────────────────────────────────────────────────

def main():
    st.title("🔍 Verify — Privacy Analysis Framework")
    st.markdown(
        "Evaluate whether privacy attributes survive through target app AI pipelines. "
        "Compare original and perturbed inputs/outputs side-by-side."
    )

    config = _load_config()
    datasets = config["datasets"]
    all_attributes = config["attributes"]
    all_apps = config["apps"]
    perturbation_map = config["perturbation_map"]

    # Filter to only known/recognized apps (others may still appear if in target-apps/)
    recognized_apps = [a for a in all_apps if a in KNOWN_APPS]
    other_apps = [a for a in all_apps if a not in KNOWN_APPS]

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        # App dropdown
        st.subheader("Target App")
        app_options = recognized_apps + [f"{a} (unrecognized)" for a in other_apps]
        if not app_options:
            st.error("No target apps found in target-apps/.")
            return

        selected_app_display = st.selectbox("Select app", app_options)
        selected_app = selected_app_display.split(" (")[0]  # strip " (unrecognized)"

        # Show adapter availability
        if selected_app in KNOWN_APPS:
            with st.spinner(f"Checking {selected_app} availability..."):
                try:
                    app_available, app_msg = get_adapter_status(selected_app)
                    if app_available:
                        st.success(f"Available: {app_msg}")
                    else:
                        st.error(f"Unavailable: {app_msg}")
                except Exception as e:
                    app_available = False
                    st.error(f"Adapter error: {e}")
        else:
            app_available = False
            st.warning("Unrecognized app — no adapter available.")

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

        # Show perturbation method and availability
        pert_method = perturbation_map.get(selected_modality, "None configured")
        st.caption(f"Perturbation method: **{pert_method}**")
        try:
            pert_available, pert_msg = get_perturbation_status(selected_modality)
            if pert_available:
                st.success(f"Perturbation ready")
            else:
                st.warning(f"Perturbation unavailable: {pert_msg}")
        except Exception as e:
            pert_available = False
            st.warning(f"Could not check perturbation: {e}")

        st.divider()

        # Attribute checklist
        st.subheader("Privacy Attributes")
        if not all_attributes:
            st.warning("No attributes in config/attribute_list.txt.")
            selected_attributes = []
        else:
            selected_attributes = []
            for attr in all_attributes:
                if st.checkbox(attr.capitalize(), value=True, key=f"attr_{attr}"):
                    selected_attributes.append(attr)

        st.divider()

        # Max items
        st.subheader("Item Limit")
        limit_enabled = st.checkbox("Limit number of items", value=False)
        max_items = None
        if limit_enabled:
            max_items = int(st.number_input("Max items to process", min_value=1, value=5, step=1))

        st.divider()

        # Cache option
        use_cache = st.checkbox("Use cache (skip already-processed items)", value=True)

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
        st.session_state.run_config = {
            "app": selected_app,
            "dataset": selected_dataset,
            "modality": selected_modality,
            "attributes": selected_attributes,
        }
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
        )
        st.session_state._generator = orch.run()
        st.rerun()

    # Process next item from generator
    if st.session_state.processing and "_generator" in st.session_state:
        gen = st.session_state._generator
        try:
            item = next(gen)
            item_type = item.get("type", "item_result")

            if item_type == "adapter_status":
                icon = "✅" if item.get("available") else "❌"
                msg = f"{icon} Adapter [{item.get('app_name')}]: {item.get('message')}"
                st.session_state.status_messages.append(msg)
                st.rerun()

            elif item_type == "perturbation_status":
                icon = "✅" if item.get("available") else "⚠️"
                msg = f"{icon} Perturbation [{item.get('method')}]: {item.get('message')}"
                st.session_state.status_messages.append(msg)
                st.rerun()

            elif item_type == "error":
                st.session_state.status_messages.append(f"❌ Error: {item.get('error')}")
                st.session_state.processing = False
                st.rerun()

            elif item_type == "summary":
                st.session_state.summary = item
                st.session_state.processing = False
                if "_generator" in st.session_state:
                    del st.session_state._generator
                st.rerun()

            elif item_type == "item_result":
                st.session_state.results.append(item)
                st.session_state.items_processed += 1
                st.session_state.current_item = item.get("filename", "")
                st.rerun()

        except StopIteration:
            st.session_state.processing = False
            if "_generator" in st.session_state:
                del st.session_state._generator
            st.rerun()
        except Exception as e:
            st.session_state.status_messages.append(f"❌ Pipeline error: {e}")
            st.session_state.processing = False
            st.rerun()

    # ── Display ────────────────────────────────────────────────────────────

    # Status messages
    for msg in st.session_state.status_messages:
        if msg.startswith("❌"):
            st.error(msg)
        elif msg.startswith("⚠️"):
            st.warning(msg)
        else:
            st.info(msg)

    # Progress indicator
    if st.session_state.processing:
        rc = st.session_state.run_config
        processed = st.session_state.items_processed
        total = st.session_state.items_total
        current = st.session_state.current_item

        st.markdown(
            f"**{rc.get('app')}** &nbsp;·&nbsp; {rc.get('dataset')} &nbsp;·&nbsp; "
            f"{rc.get('modality')} &nbsp;·&nbsp; `{'`, `'.join(rc.get('attributes', []))}`"
        )

        if total > 0:
            fraction = min(processed / total, 1.0)
            st.progress(fraction, text=f"Item {processed} / {total}" + (f"  —  `{current}`" if current else ""))
        else:
            # Total unknown (e.g. flat file dataset not yet counted) — show indeterminate
            st.progress(0, text=f"{processed} item{'s' if processed != 1 else ''} processed" + (f"  —  `{current}`" if current else ""))
            st.caption("Counting items…")

    # Results
    if st.session_state.results:
        n = len(st.session_state.results)
        rc = st.session_state.run_config
        st.subheader(
            f"Results — {rc.get('app', '')} / {rc.get('dataset', '')} "
            f"/ {rc.get('modality', '')} ({n} item{'s' if n != 1 else ''})"
        )

        for result in st.session_state.results:
            _render_item_result(result)

        # Aggregated visualization
        if not st.session_state.processing:
            st.divider()
            st.subheader("Aggregated Privacy Analysis")
            attrs = st.session_state.run_config.get("attributes", [])
            _render_aggregated_chart(st.session_state.results, attrs)

    # Summary
    if st.session_state.summary:
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

    elif not st.session_state.processing and not st.session_state.results:
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
