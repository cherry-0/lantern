"""
View Results — load and visualize a saved Verify output directory.

Reads run_config.json and report.json from a completed run directory and
renders the same side-by-side view as the live inference page.
Original images are re-loaded from the dataset on the fly (they are not
stored in the report).  Perturbed images are reconstructed by re-applying
the stored perturbation parameters (e.g. blur regions + radius) — no API
call needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st

st.set_page_config(
    page_title="View Results — Verify",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helpers (mirrors app.py) ─────────────────────────────────────────────────

def _display_image(b64_str: str | None, data=None, caption: str = ""):
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


def _display_text(text: str, key_suffix: str = ""):
    st.text_area("", value=text, height=200, disabled=True,
                 label_visibility="collapsed", key=f"txt_{key_suffix}")


def _display_frames(frames: list, caption_prefix: str = "Frame"):
    if not frames:
        st.warning("No frames available.")
        return
    cols = st.columns(min(len(frames), 4))
    for i, (col, frame) in enumerate(zip(cols, frames)):
        with col:
            st.image(frame, caption=f"{caption_prefix} {i+1}", use_container_width=True)


def _eval_chart(eval_results: dict, stage_label: str, key_suffix: str = ""):
    import pandas as pd
    import altair as alt

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
    for _, row in df.iterrows():
        attr = row["Attribute"]
        score = row["Inferability Score"]
        inferable = row["Inferable"]
        icon = "🔴" if inferable else "🟢"
        st.markdown(f"{icon} **{attr}**: score={score:.2f} — {'Inferable' if inferable else 'Not inferable'}")
        reasoning = eval_results.get(attr, {}).get("reasoning", "")
        if reasoning:
            st.caption(reasoning)

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


def _pil_to_b64(img) -> str:
    import base64, io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_original_image(filename: str, dataset_name: str) -> str | None:
    """Load original image from dataset and return as base64 string."""
    try:
        from PIL import Image as PILImage
        img_path = _find_image_path(filename, dataset_name)
        if img_path is None:
            return None
        img = PILImage.open(str(img_path)).convert("RGB")
        return _pil_to_b64(img)
    except Exception:
        return None


def _load_perturbed_image(filename: str, run_dir: str, result: dict,
                           dataset_name: str) -> str | None:
    """
    Load the perturbed image for a result item.

    Strategy (in order):
    1. Use the saved file recorded in result["perturbed_image_file"] (relative to run_dir).
    2. Probe the conventional path: <run_dir>/perturbed_images/<stem>_perturbed.jpg
    3. Fall back to re-applying stored blur parameters to the original image.

    Returns a base64 JPEG string, or None if unavailable.
    """
    # ── Strategy 1: explicit path stored in result ──────────────────────────
    rel_path = result.get("perturbed_image_file")
    if rel_path and run_dir:
        candidate = Path(run_dir) / rel_path
        if candidate.exists():
            try:
                from PIL import Image as PILImage
                img = PILImage.open(str(candidate)).convert("RGB")
                return _pil_to_b64(img)
            except Exception:
                pass

    # ── Strategy 2: conventional path ───────────────────────────────────────
    if run_dir:
        stem = Path(filename).stem
        candidate = Path(run_dir) / "perturbed_images" / f"{stem}_perturbed.jpg"
        if candidate.exists():
            try:
                from PIL import Image as PILImage
                img = PILImage.open(str(candidate)).convert("RGB")
                return _pil_to_b64(img)
            except Exception:
                pass

    # ── Strategy 3: re-apply stored blur parameters ─────────────────────────
    pert_info = result.get("perturbed_input", {}).get("perturbation_applied", {})
    blur_radius = pert_info.get("blur_radius")
    regions = pert_info.get("regions")
    if not blur_radius:
        return None

    try:
        from PIL import Image as PILImage, ImageFilter
        img_path = _find_image_path(filename, dataset_name)
        if img_path is None:
            return None
        img = PILImage.open(str(img_path)).convert("RGB")

        if regions:
            w, h = img.size
            out = img.copy()
            for box in regions:
                x1 = max(0, int(box["x1"] * w))
                y1 = max(0, int(box["y1"] * h))
                x2 = min(w, int(box["x2"] * w))
                y2 = min(h, int(box["y2"] * h))
                if x2 <= x1 or y2 <= y1:
                    continue
                region = out.crop((x1, y1, x2, y2))
                out.paste(region.filter(ImageFilter.GaussianBlur(radius=blur_radius)), (x1, y1))
        else:
            out = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return _pil_to_b64(out)
    except Exception:
        return None


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


def _render_item_result(result: dict, dataset_name: str, modality: str, run_dir: str = ""):
    filename = result.get("filename", "Unknown")
    status = result.get("status", "")

    status_icon = {"success": "✅", "failed": "❌", "skipped": "⚠️"}.get(status, "")
    from_cache = " (cached)" if result.get("from_cache") else ""

    with st.expander(f"{status_icon} {filename}{from_cache}", expanded=(status == "failed")):
        if status == "failed":
            st.error(f"Error: {result.get('error', 'Unknown error')}")
            return

        if status == "skipped":
            st.warning(result.get("perturbation_warning", "Item was skipped."))
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

        # ── Generated task (snapdo only) ──────────────────────────────────
        _render_generated_task(result)

        # ── Input Comparison ──────────────────────────────────────────────
        st.markdown("### Input Comparison")
        col_orig, col_pert = st.columns(2)

        with col_orig:
            st.markdown("**Original Input**")
            if modality == "image":
                b64 = _load_original_image(filename, dataset_name)
                if b64:
                    _display_image(b64, caption=filename)
                else:
                    st.info("Original image not found in dataset path.")
            elif modality == "text":
                _display_text(orig_input.get("text_content", ""), key_suffix=f"orig_{filename}")
            else:
                st.info("Media not stored in report.")

        with col_pert:
            st.markdown("**Perturbed Input**")
            pert_info = pert_input.get("perturbation_applied", {})
            if modality == "image":
                pert_b64 = _load_perturbed_image(filename, run_dir, result, dataset_name)
                if pert_b64:
                    _display_image(pert_b64, caption=f"{filename} (perturbed)")
                else:
                    st.info("Perturbed image not found.")
            elif modality == "text":
                _display_text(pert_input.get("text_content", ""), key_suffix=f"pert_{filename}")
            else:
                st.info("Media not stored in report.")
            if pert_info:
                method = pert_info.get("method", "")
                attrs = pert_info.get("attributes", [])
                st.caption(f"Method: {method} | Attributes: {', '.join(attrs)}")

        st.divider()

        # ── Inference Results ──────────────────────────────────────────────
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
                exts = orig_out.get("externalizations", {})
                if exts:
                    with st.expander("🌐 Captured Externalizations", expanded=True):
                        for channel, content in exts.items():
                            st.markdown(f"**[{channel}]**")
                            st.caption(content)
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
                exts = pert_out.get("externalizations", {})
                if exts:
                    with st.expander("🌐 Captured Externalizations", expanded=True):
                        for channel, content in exts.items():
                            st.markdown(f"**[{channel}]**")
                            st.caption(content)
                structured = pert_out.get("structured_output", {})
                if structured:
                    with st.expander("Structured output"):
                        st.json(structured)
            else:
                st.error(pert_out.get("error", "Pipeline failed.") if pert_out else "No output.")

        st.divider()

        # ── Privacy Evaluation ─────────────────────────────────────────────
        st.markdown("### Privacy Inferability")

        orig_eval = evaluation.get("original", {})
        pert_eval = evaluation.get("perturbed", {})
        orig_eval_ok = evaluation.get("original_success", False)
        pert_eval_ok = evaluation.get("perturbed_success", False)

        col_eval_orig, col_eval_pert = st.columns(2)

        with col_eval_orig:
            st.markdown("**From original output:**")
            if orig_eval_ok and orig_eval:
                _eval_chart(orig_eval, "Original", key_suffix=f"orig_{filename}")
            elif evaluation.get("original_error"):
                st.error(f"Evaluation error: {evaluation['original_error']}")
            else:
                st.info("No evaluation results.")

        with col_eval_pert:
            st.markdown("**From perturbed output:**")
            if pert_eval_ok and pert_eval:
                _eval_chart(pert_eval, "Perturbed", key_suffix=f"pert_{filename}")
            elif evaluation.get("perturbed_error"):
                st.error(f"Evaluation error: {evaluation['perturbed_error']}")
            else:
                st.info("No evaluation results.")


def _render_aggregated_chart(items: list, attributes: list):
    import pandas as pd
    import altair as alt

    if not items or not attributes:
        return

    data = []
    for attr in attributes:
        orig_scores, pert_scores = [], []
        for r in items:
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
        st.info("No aggregated data to display.")
        return

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


def _get_cache_dir_for_run(run_config: dict) -> Path | None:
    """Reconstruct the cache directory path from a run config dict."""
    try:
        import hashlib
        parts = {
            "app": run_config.get("app_name", ""),
            "dataset": run_config.get("dataset_name", ""),
            "modality": run_config.get("modality", ""),
            "attributes": sorted(run_config.get("attributes", [])),
            "perturbation": run_config.get("perturbation_method", ""),
            "evaluation": run_config.get("evaluation_method", "openrouter"),
        }
        payload = json.dumps(parts, sort_keys=True)
        key = hashlib.sha256(payload.encode()).hexdigest()[:16]
        outputs_root = LANTERN_ROOT / "verify" / "outputs"
        candidate = outputs_root / f"cache_{key}"
        return candidate if candidate.exists() else None
    except Exception:
        return None


def _render_delete_section(run_dir: str, run_config: dict):
    """Render the 'Delete this record' button with confirmation."""
    import shutil

    st.markdown("#### Danger Zone")

    cache_dir = _get_cache_dir_for_run(run_config)
    cache_note = f"`{Path(cache_dir).name}`" if cache_dir else "_(no matching cache found)_"

    st.caption(
        f"This will permanently delete the run directory and its associated cache.\n\n"
        f"- Run directory: `{Path(run_dir).name}`\n"
        f"- Cache: {cache_note}"
    )

    confirmed = st.checkbox("I understand this cannot be undone", key="delete_confirm")
    if st.button("🗑️ Delete this record", type="primary", disabled=not confirmed):
        errors = []

        # Delete run directory
        run_path = Path(run_dir)
        if run_path.exists():
            try:
                shutil.rmtree(run_path)
            except Exception as e:
                errors.append(f"Run dir: {e}")

        # Delete cache directory
        if cache_dir and cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
            except Exception as e:
                errors.append(f"Cache dir: {e}")

        if errors:
            st.error("Deletion completed with errors:\n" + "\n".join(errors))
        else:
            st.success("Record deleted successfully.")

        # Clear session state so the page resets
        for key in ("vr_report", "vr_run_config", "vr_run_dir", "delete_confirm"):
            st.session_state.pop(key, None)
        st.rerun()


def _list_output_dirs() -> list[Path]:
    """Return all non-cache run directories under verify/outputs/, newest first."""
    outputs_root = LANTERN_ROOT / "verify" / "outputs"
    if not outputs_root.exists():
        return []
    dirs = [
        d for d in outputs_root.iterdir()
        if d.is_dir() and not d.name.startswith("cache_") and (d / "report.json").exists()
    ]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


# ─── Main UI ──────────────────────────────────────────────────────────────────

def main():
    st.title("📂 View Results")
    st.markdown("Load a saved Verify output directory and inspect results.")

    # ── Sidebar: directory picker ─────────────────────────────────────────
    with st.sidebar:
        st.header("Select Output Directory")

        available_dirs = _list_output_dirs()
        dir_options = [str(d) for d in available_dirs]

        if dir_options:
            st.markdown("**Recent runs:**")
            selected_from_list = st.selectbox(
                "Pick a run",
                options=["(enter path manually)"] + dir_options,
                format_func=lambda p: Path(p).name if p != "(enter path manually)" else p,
            )
        else:
            selected_from_list = "(enter path manually)"

        manual_path = st.text_input(
            "Or enter path manually",
            placeholder="/path/to/verify/outputs/run_dir",
        )

        if manual_path.strip():
            run_dir_str = manual_path.strip()
        elif selected_from_list != "(enter path manually)":
            run_dir_str = selected_from_list
        else:
            run_dir_str = ""

        load_clicked = st.button("📂 Load", type="primary", use_container_width=True)

    # ── Load on button click ───────────────────────────────────────────────
    if load_clicked:
        if not run_dir_str:
            st.error("Please select or enter an output directory path.")
            return

        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            st.error(f"Directory not found: `{run_dir}`")
            return

        report_path = run_dir / "report.json"
        config_path = run_dir / "run_config.json"

        if not report_path.exists():
            st.error(f"`report.json` not found in `{run_dir}`")
            return

        try:
            report = json.loads(report_path.read_text())
        except Exception as e:
            st.error(f"Failed to parse report.json: {e}")
            return

        run_config = {}
        if config_path.exists():
            try:
                run_config = json.loads(config_path.read_text())
            except Exception:
                pass

        # Merge: report.json may also embed run_config
        if not run_config and "run_config" in report:
            run_config = report["run_config"]

        st.session_state["vr_report"] = report
        st.session_state["vr_run_config"] = run_config
        st.session_state["vr_run_dir"] = str(run_dir)

    # ── Render loaded report ───────────────────────────────────────────────
    report = st.session_state.get("vr_report")
    run_config = st.session_state.get("vr_run_config", {})
    run_dir_loaded = st.session_state.get("vr_run_dir", "")

    if report is None:
        if not run_dir_str:
            st.info("Select an output directory in the sidebar and click **📂 Load**.")
        return

    items = report.get("items", [])
    # run_config can be nested inside report or at top level
    if not run_config:
        run_config = report.get("run_config", {})

    app_name = run_config.get("app_name", "unknown")
    dataset_name = run_config.get("dataset_name", "unknown")
    modality = run_config.get("modality", "image")
    attributes = run_config.get("attributes", [])
    pert_method = run_config.get("perturbation_method", "")
    started_at = run_config.get("started_at", "")
    generated_at = report.get("generated_at", "")

    # ── Run info banner ────────────────────────────────────────────────────
    st.subheader("Run Info")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("App", app_name)
    c2.metric("Dataset", dataset_name)
    c3.metric("Modality", modality)
    c4.metric("Items", len(items))

    info_cols = st.columns(3)
    info_cols[0].markdown(f"**Attributes:** {', '.join(attributes) or '—'}")
    info_cols[1].markdown(f"**Perturbation:** {pert_method or '—'}")
    info_cols[2].markdown(f"**Started:** {started_at[:19].replace('T', ' ') if started_at else '—'}")

    st.caption(f"Directory: `{run_dir_loaded}`")

    # ── Aggregated chart ───────────────────────────────────────────────────
    success_items = [r for r in items if r.get("status") == "success"]
    if success_items and attributes:
        st.divider()
        st.subheader("Aggregated Privacy Analysis")
        _render_aggregated_chart(success_items, attributes)

    # ── Summary metrics ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Summary")
    statuses = [r.get("status", "") for r in items]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", len(items))
    m2.metric("Successful", statuses.count("success"))
    m3.metric("Skipped", statuses.count("skipped"))
    m4.metric("Failed", statuses.count("failed"))

    # Average scores table
    if success_items and attributes:
        import pandas as pd
        rows = []
        for attr in attributes:
            orig_scores, pert_scores = [], []
            for r in success_items:
                ev = r.get("evaluation", {})
                o = ev.get("original", {}).get(attr, {})
                p = ev.get("perturbed", {}).get(attr, {})
                if isinstance(o, dict) and "score" in o:
                    orig_scores.append(o["score"])
                if isinstance(p, dict) and "score" in p:
                    pert_scores.append(p["score"])
            orig_avg = sum(orig_scores) / len(orig_scores) if orig_scores else None
            pert_avg = sum(pert_scores) / len(pert_scores) if pert_scores else None
            rows.append({
                "Attribute": attr,
                "Orig score (avg)": f"{orig_avg:.3f}" if orig_avg is not None else "N/A",
                "Pert score (avg)": f"{pert_avg:.3f}" if pert_avg is not None else "N/A",
                "Delta": (
                    f"{(pert_avg - orig_avg):.3f}"
                    if (orig_avg is not None and pert_avg is not None)
                    else "N/A"
                ),
            })
        if rows:
            st.markdown("**Average inferability scores:**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Download buttons
    report_dir = Path(run_dir_loaded) if run_dir_loaded else None
    if report_dir:
        st.divider()
        st.subheader("Download Report")
        col_json, col_csv = st.columns(2)
        with col_json:
            json_path = report_dir / "report.json"
            if json_path.exists():
                st.download_button(
                    "Download JSON report",
                    data=json_path.read_text(),
                    file_name="verify_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
        with col_csv:
            csv_path = report_dir / "report.csv"
            if csv_path.exists():
                st.download_button(
                    "Download CSV report",
                    data=csv_path.read_text(),
                    file_name="verify_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ── Per-item results ───────────────────────────────────────────────────
    st.divider()
    n = len(items)
    st.subheader(f"Results — {app_name} / {dataset_name} / {modality} ({n} item{'s' if n != 1 else ''})")

    for result in items:
        _render_item_result(result, dataset_name, modality, run_dir=run_dir_loaded)

    # ── Delete record ──────────────────────────────────────────────────────
    st.divider()
    _render_delete_section(run_dir_loaded, run_config)


if __name__ == "__main__":
    main()
