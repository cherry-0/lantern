"""
View Judge Validation Results — browse saved Judge Validator runs.

Loads results saved by 9_Judge_Validator.py from
verify/outputs/judge_validation_runs/<run>/ and renders the same metrics,
distribution, sample viewer, error analysis, and export layout.
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT  = Path(__file__).resolve().parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import pandas as pd
import streamlit as st

from verify.backend.evaluation_method.evaluator import (
    VERDICT_CONFIRMED,
    VERDICT_NO_EVIDENCE,
    VERDICT_POSSIBLE,
)
from verify.backend.judge.metrics import (
    LABEL_CONFIRMED,
    LABEL_NONE,
    LABEL_POSSIBLE,
    compute_distribution,
    compute_distribution_by_difficulty,
    compute_metrics,
    compute_metrics_by_group,
    false_negatives,
    false_positives,
)
from verify.backend.utils.config import load_color_palette


_RUNS_DIR = VERIFY_ROOT / "outputs" / "judge_validation_runs"
_PALETTE = load_color_palette()
_VERDICT_COLORS = {
    LABEL_CONFIRMED: _PALETTE["verdict"][VERDICT_CONFIRMED],
    LABEL_POSSIBLE: _PALETTE["verdict"][VERDICT_POSSIBLE],
    LABEL_NONE: _PALETTE["verdict"][VERDICT_NO_EVIDENCE],
}
_DIFFICULTY_ORDER = ["explicit", "implicit", "none"]


def _find_runs() -> List[Path]:
    if not _RUNS_DIR.exists():
        return []
    return sorted(
        [d for d in _RUNS_DIR.iterdir() if d.is_dir() and (d / "results.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load_run(run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results = json.loads((run_dir / "results.json").read_text())
    info_path = run_dir / "run_info.json"
    run_info = json.loads(info_path.read_text()) if info_path.exists() else {}
    run_info.setdefault("run_dir", str(run_dir))
    run_info.setdefault("n_samples", len(results))
    return results, run_info


def _load_uploaded_results(uploaded) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = json.loads(uploaded.getvalue().decode("utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"], payload.get("run_info", {})
    if isinstance(payload, list):
        return payload, {"source": uploaded.name, "n_samples": len(payload)}
    raise ValueError("Uploaded JSON must be a results list or an object with a results list.")


def _is_saved_run_dir(path: Path) -> bool:
    try:
        resolved = path.resolve()
        runs_root = _RUNS_DIR.resolve()
        return (
            resolved.parent == runs_root
            and resolved.is_dir()
            and (resolved / "results.json").exists()
        )
    except Exception:
        return False


def _render_delete_section(run_dir: Path) -> None:
    """Render a guarded delete button for the selected saved run directory."""
    st.divider()
    st.subheader("Delete")

    if not _is_saved_run_dir(run_dir):
        st.warning("Delete is only available for saved judge-validation run directories.")
        return

    st.warning(
        "This will permanently delete the selected judge-validation run directory:\n\n"
        f"`{run_dir}`"
    )
    confirmed = st.checkbox("I understand this cannot be undone", key="vjv_delete_confirm")
    if st.button("🗑️ Delete this run", type="primary", disabled=not confirmed):
        try:
            shutil.rmtree(run_dir)
            st.success("Judge-validation run deleted.")
            for key in (
                "vjv_saved_run",
                "vjv_delete_confirm",
                "vjv_attr_filter",
                "vjv_diff_filter",
                "vjv_label_filter",
            ):
                st.session_state.pop(key, None)
            st.rerun()
        except Exception as e:
            st.error(f"Delete failed: {e}")


def _verdict_badge(label: str) -> str:
    color = _VERDICT_COLORS.get(label, "#e6e6e6")
    return f'<span style="background:{color};padding:2px 8px;border-radius:4px;font-size:0.85em">{label}</span>'


def _gt_badge(gt: int) -> str:
    color = _PALETTE["binary"]["positive"] if gt else _PALETTE["binary"]["negative"]
    text = "positive" if gt else "negative"
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em">{text}</span>'


def _metric_card(col, label: str, value, fmt: str = ".1%", warn_none: str = "N/A") -> None:
    if value is None:
        col.metric(label, warn_none)
    else:
        col.metric(label, f"{value:{fmt}}")


def _render_summary(results: List[Dict[str, Any]]) -> None:
    metrics = compute_metrics(results)
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "Precision (confirmed)", metrics["precision_confirmed"])
    _metric_card(c2, "Coverage (confirmed)", metrics["coverage_confirmed"])
    _metric_card(c3, "Ambiguity rate", metrics["ambiguity_rate"])
    _metric_card(c4, "Recall lower bound", metrics["recall_lower_bound"])


def _render_distribution(results: List[Dict[str, Any]]) -> None:
    dist = compute_distribution(results)
    labels = [LABEL_CONFIRMED, LABEL_POSSIBLE, LABEL_NONE]
    colors = [_VERDICT_COLORS[l] for l in labels]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Overall distribution**")
        df_wide = pd.DataFrame({l: [dist[l]] for l in labels}, index=pd.Index(["samples"]))
        st.bar_chart(df_wide, color=colors, height=250)

    with col_b:
        st.markdown("**Breakdown by difficulty**")
        by_diff = compute_distribution_by_difficulty(results)
        rows = []
        for diff in _DIFFICULTY_ORDER:
            if diff in by_diff:
                rows.append({"difficulty": diff, **{l: by_diff[diff][l] for l in labels}})
        if rows:
            df_diff = pd.DataFrame(rows).set_index("difficulty")
            st.bar_chart(df_diff[labels], color=colors, height=250)


def _render_sample_viewer(results: List[Dict[str, Any]]) -> None:
    st.subheader(f"Sample Viewer  ({len(results)} samples)")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        attr_opts = sorted({r["attribute"] for r in results if r.get("attribute")})
        attr_sel = st.multiselect("Attribute", attr_opts, key="vjv_attr_filter")
    with filter_col2:
        diff_opts = sorted({r["difficulty"] for r in results if r.get("difficulty")})
        diff_sel = st.multiselect("Difficulty", diff_opts, key="vjv_diff_filter")
    with filter_col3:
        label_opts = [LABEL_CONFIRMED, LABEL_POSSIBLE, LABEL_NONE]
        label_sel = st.multiselect("Judge label", label_opts, key="vjv_label_filter")

    filtered = [
        r for r in results
        if (not attr_sel or r.get("attribute") in attr_sel)
        and (not diff_sel or r.get("difficulty") in diff_sel)
        and (not label_sel or r.get("label") in label_sel)
    ]
    st.caption(f"Showing {len(filtered)} / {len(results)}")

    for r in filtered[:200]:
        gt = int(r.get("ground_truth", 0))
        label = r.get("label", LABEL_NONE)
        icon = "✅" if label == LABEL_CONFIRMED else ("⚠️" if label == LABEL_POSSIBLE else "⭕")
        header = (
            f"{icon} [{r.get('dataset', 'unknown')}] **{r.get('attribute', 'unknown')}** — "
            f"GT: {'pos' if gt else 'neg'} | judge: **{label}** "
            f"(conf {float(r.get('confidence', 0) or 0):.2f}) | {r.get('difficulty', 'unknown')}"
        )
        with st.expander(header):
            c1, c2 = st.columns(2)
            with c1:
                if r.get("text_content"):
                    st.text_area(
                        "Content",
                        str(r["text_content"])[:2000],
                        height=160,
                        disabled=True,
                        key=f"vjv_content_{r.get('id', id(r))}",
                        label_visibility="collapsed",
                    )
                elif r.get("image_b64"):
                    try:
                        st.image(base64.b64decode(r["image_b64"]), width="stretch")
                    except Exception:
                        st.caption("Could not render image.")
            with c2:
                st.markdown(
                    f"**Dataset:** {r.get('dataset', 'unknown')}  \n"
                    f"**Attribute:** {r.get('attribute', 'unknown')}  \n"
                    f"**Ground truth:** {_gt_badge(gt)}  \n"
                    f"**Judge label:** {_verdict_badge(label)}  \n"
                    f"**Confidence:** {float(r.get('confidence', 0) or 0):.2f}  \n"
                    f"**Difficulty:** {r.get('difficulty', '-')}",
                    unsafe_allow_html=True,
                )
                st.markdown("**Explanation:**")
                st.caption(r.get("explanation") or "-")
                if r.get("prediction"):
                    st.markdown("**Prediction:**")
                    st.caption(str(r.get("prediction")))
                if not r.get("judge_ok", True):
                    st.error(f"Judge error: {r.get('judge_error')}")


def _render_error_analysis(results: List[Dict[str, Any]]) -> None:
    st.subheader("Error Analysis")
    fps = false_positives(results)
    fns = false_negatives(results)
    col_fp, col_fn = st.columns(2)

    with col_fp:
        st.markdown(f"**False positives** — confirmed but GT=0 ({len(fps)})")
        if fps:
            st.dataframe(pd.DataFrame([
                {
                    "id": r.get("id"),
                    "dataset": r.get("dataset"),
                    "attribute": r.get("attribute"),
                    "difficulty": r.get("difficulty"),
                    "conf": f"{float(r.get('confidence', 0) or 0):.2f}",
                }
                for r in fps[:100]
            ]), use_container_width=False)
        else:
            st.success("No false positives.")

    with col_fn:
        st.markdown(f"**False negatives** — GT=1 but not confirmed ({len(fns)})")
        if fns:
            st.dataframe(pd.DataFrame([
                {
                    "id": r.get("id"),
                    "dataset": r.get("dataset"),
                    "attribute": r.get("attribute"),
                    "difficulty": r.get("difficulty"),
                    "label": r.get("label", LABEL_NONE),
                    "conf": f"{float(r.get('confidence', 0) or 0):.2f}",
                }
                for r in fns[:100]
            ]), use_container_width=False)
        else:
            st.success("No false negatives.")


def _export_button(results: List[Dict[str, Any]], run_info: Dict[str, Any]) -> None:
    df = pd.DataFrame([
        {
            "id": r.get("id", ""),
            "dataset": r.get("dataset", ""),
            "attribute": r.get("attribute", ""),
            "difficulty": r.get("difficulty", ""),
            "ground_truth": r.get("ground_truth", ""),
            "label": r.get("label", ""),
            "confidence": r.get("confidence", ""),
            "prediction": r.get("prediction", ""),
            "explanation": r.get("explanation", ""),
            "judge_ok": r.get("judge_ok", False),
            "from_cache": r.get("from_cache", False),
        }
        for r in results
    ])
    bundle = {"run_info": run_info, "results": results}
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download CSV", df.to_csv(index=False).encode(), "judge_results.csv", "text/csv")
    c2.download_button(
        "⬇ Download JSON",
        json.dumps(bundle, ensure_ascii=False, indent=2).encode(),
        "judge_results.json",
        "application/json",
    )


def _render_results(results: List[Dict[str, Any]], run_info: Dict[str, Any]) -> None:
    n_cached = sum(1 for r in results if r.get("from_cache"))
    n_failed = sum(1 for r in results if not r.get("judge_ok", True))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", run_info.get("n_samples", len(results)))
    col2.metric("Cached", n_cached)
    col3.metric("Failed", n_failed)
    col4.metric("Model", run_info.get("model", "-"))

    successful = [r for r in results if r.get("judge_ok", True)]
    if not successful:
        st.error("All evaluations in this result set failed.")
        return

    st.divider()
    st.subheader("Summary Metrics")
    _render_summary(successful)

    with st.expander("Per-attribute metrics", expanded=False):
        by_attr = compute_metrics_by_group(successful, "attribute")
        rows = []
        for attr, m in sorted(by_attr.items()):
            rows.append({
                "attribute": attr,
                "precision_confirmed": f"{m['precision_confirmed']:.1%}" if m["precision_confirmed"] is not None else "N/A",
                "coverage_confirmed": f"{m['coverage_confirmed']:.1%}",
                "ambiguity_rate": f"{m['ambiguity_rate']:.1%}",
                "recall_lower_bound": f"{m['recall_lower_bound']:.1%}" if m["recall_lower_bound"] is not None else "N/A",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("attribute"), use_container_width=False)

    st.divider()
    st.subheader("Distribution")
    _render_distribution(successful)

    st.divider()
    _render_sample_viewer(successful)

    st.divider()
    _render_error_analysis(successful)

    st.divider()
    st.subheader("Export")
    _export_button(results, run_info)


def main() -> None:
    st.title("🎯 View Judge Validation Results")
    st.markdown("Browse saved Judge Validator runs without re-running the evaluator.")

    results: Optional[List[Dict[str, Any]]] = None
    run_info: Dict[str, Any] = {}
    selected_run_dir: Optional[Path] = None

    with st.sidebar:
        st.header("Result Source")
        runs = _find_runs()
        run_labels = [
            f"{p.name}  ({datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})"
            for p in runs
        ]
        selected_idx = None
        if runs:
            selected_label = st.selectbox("Saved run", run_labels, key="vjv_saved_run")
            selected_idx = run_labels.index(selected_label)
        else:
            st.info("No saved judge-validation runs found.")

        uploaded = st.file_uploader("Or upload judge_results.json", type=["json"], key="vjv_upload")

    try:
        if uploaded is not None:
            results, run_info = _load_uploaded_results(uploaded)
        elif selected_idx is not None:
            selected_run_dir = runs[selected_idx]
            results, run_info = _load_run(selected_run_dir)
    except Exception as e:
        st.error(f"Could not load judge validation results: {e}")
        return

    if results is None:
        st.info("Run the Judge Validator page first, or upload a downloaded judge results JSON.")
        return

    source = run_info.get("run_dir") or run_info.get("source")
    if source:
        st.caption(f"Source: `{source}`")
    _render_results(results, run_info)
    if selected_run_dir is not None and uploaded is None:
        _render_delete_section(selected_run_dir)


main()
