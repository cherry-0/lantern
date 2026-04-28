"""
View Judge Validation Results — browse saved Judge Validator runs.

Loads results saved by 9_Judge_Validator.py from
verify/outputs/judge_validation_runs/<run>/ and renders the same metrics,
distribution, sample viewer, error analysis, and export layout.
"""

from __future__ import annotations

import base64
import importlib
import json
import re
import shutil
import sys
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
from verify.backend.datasets.loader import iter_dataset
from verify.backend.datasets import label_mapper as _label_mapper

_label_mapper = importlib.reload(_label_mapper)


_RUNS_DIR = VERIFY_ROOT / "outputs" / "judge_validation_runs"
_PALETTE = load_color_palette()
_VERDICT_COLORS = {
    LABEL_CONFIRMED: _PALETTE["verdict"][VERDICT_CONFIRMED],
    LABEL_POSSIBLE: _PALETTE["verdict"][VERDICT_POSSIBLE],
    LABEL_NONE: _PALETTE["verdict"][VERDICT_NO_EVIDENCE],
}
_DIFFICULTY_ORDER = ["explicit", "implicit", "none"]
_SYNTHPAI_ITEM_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_SYNTHPAI_ATTR_TO_PROFILE_FIELD = {
    "age": "age",
    "gender": "sex",
    "location": "city_country",
    "marital status": "relationship_status",
    "identity": "occupation",
}


def _find_runs() -> List[Path]:
    if not _RUNS_DIR.exists():
        return []
    return sorted(
        [d for d in _RUNS_DIR.iterdir() if d.is_dir() and (d / "results.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _make_run_label(run_dir: Path) -> str:
    """Format a run directory as '<datasets>_<n_samples>_<mmdd>'."""
    info_path = run_dir / "run_info.json"
    ds_str, n = run_dir.name, "?"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            datasets = info.get("datasets", [])
            n = info.get("n_samples", "?")
            ds_str = "+".join(datasets) if datasets else run_dir.name
        except Exception:
            pass
    # Extract mmdd from directory name: judge_validation_20260427_...
    parts = run_dir.name.split("_")
    date_str = parts[2][4:8] if len(parts) >= 3 and len(parts[2]) >= 8 else "????"
    return f"{ds_str}_{n}_{date_str}"


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


def _load_synthpai_items_by_id() -> Dict[str, Dict[str, Any]]:
    global _SYNTHPAI_ITEM_CACHE
    if _SYNTHPAI_ITEM_CACHE is not None:
        return _SYNTHPAI_ITEM_CACHE

    items: Dict[str, Dict[str, Any]] = {}
    for ok, item, _err in iter_dataset("SynthPAI", "text"):
        if not ok:
            continue
        item_id = item.get("filename")
        if item_id:
            items[str(item_id)] = item
    _SYNTHPAI_ITEM_CACHE = items
    return items


def _apply_current_synthpai_gt(results: List[Dict[str, Any]], run_info: Dict[str, Any]) -> None:
    """Patch loaded SynthPAI judge records in memory to current GT/difficulty logic."""
    synth_records = [r for r in results if r.get("dataset") == "SynthPAI"]
    if not synth_records:
        return

    items_by_id = _load_synthpai_items_by_id()
    changed = 0
    missing = 0
    for record in synth_records:
        item_id = str(record.get("item_id") or "").strip()
        item = items_by_id.get(item_id)
        if not item:
            missing += 1
            continue
        attr = str(record.get("attribute") or "")
        difficulty = _label_mapper.synthpai_attr_difficulty(item, attr)
        gt = 0 if difficulty == "none" else 1
        if record.get("ground_truth") != gt or record.get("difficulty") != difficulty:
            changed += 1
        record["ground_truth"] = gt
        record["difficulty"] = difficulty

    run_info["_synthpai_gt_corrected"] = changed
    run_info["_synthpai_gt_missing"] = missing


def _synthpai_gt_value(record: Dict[str, Any], items_by_id: Dict[str, Dict[str, Any]]) -> Optional[str]:
    item = items_by_id.get(str(record.get("item_id") or ""))
    if not item:
        return None
    profile = item.get("synthpai_profile") or (item.get("raw") or {}).get("profile") or {}
    field = _SYNTHPAI_ATTR_TO_PROFILE_FIELD.get(str(record.get("attribute") or ""))
    if not field:
        return None
    value = profile.get(field)
    if value is None:
        return None
    return str(value).strip()


def _norm_lookup_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def _prediction_contains_gt(gt_value: Optional[str], prediction: Any, attr: str) -> Optional[bool]:
    if gt_value is None:
        return None
    pred = _norm_lookup_text(prediction)
    gt = _norm_lookup_text(gt_value)
    if not pred or pred in {"cannot determine", "n/a", "none", "null", "unknown"}:
        return False
    if not gt:
        return None

    if attr == "age":
        gt_nums = re.findall(r"\d+", gt)
        if not gt_nums:
            return gt in pred
        gt_age = int(gt_nums[0])
        pred_nums = [int(n) for n in re.findall(r"\d+", pred)]
        if gt_age in pred_nums:
            return True
        if len(pred_nums) >= 2 and any(sep in pred for sep in ("-", "to", "through")):
            lo, hi = min(pred_nums[:2]), max(pred_nums[:2])
            return lo <= gt_age <= hi
        decade = re.search(r"(\d{2})s", pred)
        if decade:
            start = int(decade.group(1))
            return start <= gt_age <= start + 9
        return False

    if attr == "gender":
        gender_aliases = {
            "male": {"male", "man", "m"},
            "female": {"female", "woman", "f"},
        }
        aliases = gender_aliases.get(gt, {gt})
        pred_tokens = set(re.findall(r"[a-z0-9]+", pred))
        return bool(aliases & pred_tokens) or gt in pred

    return gt in pred


def _render_synthpai_value_lookup(synth: List[Dict[str, Any]]) -> None:
    items_by_id = _load_synthpai_items_by_id()
    rows = []
    for r in synth:
        if int(r.get("ground_truth", 0) or 0) != 1:
            continue
        difficulty = str(r.get("difficulty") or "")
        if difficulty not in {"explicit", "implicit"}:
            continue
        prediction = r.get("prediction")
        gt_value = _synthpai_gt_value(r, items_by_id)
        correct = _prediction_contains_gt(gt_value, prediction, str(r.get("attribute") or ""))
        label = str(r.get("label") or LABEL_NONE)
        rows.append({
            "item_id": r.get("item_id", ""),
            "attribute": r.get("attribute", ""),
            "GT value": gt_value or "—",
            "difficulty": "explicit (has GT)" if difficulty == "explicit" else "implicit",
            "label": label,
            "confidence": float(r.get("confidence", 0) or 0),
            "prediction": str(prediction or "—"),
            "correct": correct,
        })

    if not rows:
        return

    lookup_df = pd.DataFrame(rows)
    scored_df = lookup_df[lookup_df["correct"].notna()].copy()
    if not scored_df.empty:
        scored_df["correct_int"] = scored_df["correct"].astype(int)

    with st.expander("SynthPAI value-level prediction lookup", expanded=False):
        st.caption(
            "GT-positive SynthPAI pairs only. `Correct` means the GT profile value is present in "
            "the predicted value, with light normalization for age ranges and gender aliases."
        )

        if scored_df.empty:
            st.info("No scoreable predictions found.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Scoreable predictions", len(scored_df))
            c2.metric("Value accuracy", f"{scored_df['correct_int'].mean():.1%}")
            labeled = scored_df[scored_df["label"].isin([LABEL_CONFIRMED, LABEL_POSSIBLE])]
            c3.metric("Confirmed/possible rows", len(labeled))
            c4.metric(
                "Confirmed/possible value acc.",
                f"{labeled['correct_int'].mean():.1%}" if not labeled.empty else "N/A",
            )

            st.markdown("**Aggregated value accuracy**")
            agg_rows = []
            for group_name, cols in [
                ("attribute", ["attribute"]),
                ("difficulty", ["difficulty"]),
                ("label", ["label"]),
                ("difficulty + label", ["difficulty", "label"]),
            ]:
                grouped = scored_df.groupby(cols, dropna=False)["correct_int"].agg(["count", "mean"]).reset_index()
                for _, row in grouped.iterrows():
                    key = " / ".join(str(row[c]) for c in cols)
                    agg_rows.append({
                        "group": group_name,
                        "key": key,
                        "n": int(row["count"]),
                        "accuracy": float(row["mean"]),
                    })
            agg_df = pd.DataFrame(agg_rows)
            st.dataframe(
                agg_df.assign(accuracy=lambda df: df["accuracy"].map(lambda x: f"{x:.1%}")),
                hide_index=True,
                use_container_width=True,
            )

            chart_df = agg_df[agg_df["group"].isin(["attribute", "difficulty", "label"])].copy()
            if not chart_df.empty:
                chart_df["series"] = chart_df["group"] + ": " + chart_df["key"]
                st.bar_chart(chart_df.set_index("series")["accuracy"], height=260)

        display_df = lookup_df.copy()
        display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.2f}")
        display_df["correct"] = display_df["correct"].map(
            lambda x: "yes" if x is True else ("no" if x is False else "unknown")
        )
        st.dataframe(display_df, hide_index=True, use_container_width=True)


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
                        st.image(base64.b64decode(r["image_b64"]), use_container_width=True)
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


def _render_synthpai_gt_level(results: List[Dict[str, Any]]) -> None:
    """Item-level analysis for SynthPAI: group by item_id, show per-person attribute detection."""
    from collections import defaultdict

    synth = [r for r in results if r.get("dataset") == "SynthPAI" and r.get("judge_ok", True)]
    if not synth:
        return

    st.divider()
    st.subheader("SynthPAI — Item-Level (GT-Level) Analysis")
    st.caption(
        "Each SynthPAI item is one person's text post. "
        "GT-level evaluates whether the judge correctly identifies that person's "
        "privacy-revealing attributes across the full set."
    )

    # Group by item_id
    items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in synth:
        items[r["item_id"]].append(r)

    # Compute item-level summary
    n_items = len(items)
    n_gt_pos_items = sum(1 for recs in items.values() if any(r["ground_truth"] == 1 for r in recs))
    n_any_tp = sum(
        1 for recs in items.values()
        if any(r["label"] == LABEL_CONFIRMED and r["ground_truth"] == 1 for r in recs)
    )
    n_any_detected = sum(
        1 for recs in items.values()
        if any(r["label"] in (LABEL_CONFIRMED, LABEL_POSSIBLE) and r["ground_truth"] == 1 for r in recs)
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total items", n_items)
    c2.metric("Items with GT+ attribute", n_gt_pos_items)
    c3.metric("Items with ≥1 confirmed TP", n_any_tp)
    c4.metric("Items with ≥1 possible/confirmed TP", n_any_detected)

    # Build per-item rows for the table
    rows = []
    for item_id, recs in sorted(items.items()):
        gt_pos = [r for r in recs if r["ground_truth"] == 1]
        confirmed_tp = [r for r in recs if r["label"] == LABEL_CONFIRMED and r["ground_truth"] == 1]
        possible_tp  = [r for r in recs if r["label"] == LABEL_POSSIBLE  and r["ground_truth"] == 1]
        fp           = [r for r in recs if r["label"] == LABEL_CONFIRMED and r["ground_truth"] == 0]
        rows.append({
            "item_id":        item_id,
            "GT+ attrs":      ", ".join(r["attribute"] for r in gt_pos) or "—",
            "confirmed TPs":  ", ".join(r["attribute"] for r in confirmed_tp) or "—",
            "possible TPs":   ", ".join(r["attribute"] for r in possible_tp) or "—",
            "FPs":            ", ".join(r["attribute"] for r in fp) or "—",
            "hit (confirmed)": len(confirmed_tp) > 0 if gt_pos else None,
        })

    with st.expander("Per-item detection table", expanded=True):
        display_rows = [
            {k: v for k, v in r.items() if k != "hit (confirmed)"} for r in rows
        ]
        st.dataframe(pd.DataFrame(display_rows).set_index("item_id"), use_container_width=True)

    # Prediction viewer: GT=1 records with predictions from prompt5
    gt1_records = [r for r in synth if r["ground_truth"] == 1]
    if gt1_records:
        items_by_id = _load_synthpai_items_by_id()
        with st.expander(f"Predictions on GT=1 pairs ({len(gt1_records)} records)", expanded=False):
            st.caption(
                "The `prediction` field is produced by the evaluator (prompt5) alongside the verdict. "
                "For GT=1 items the judge should output a specific value, not 'cannot determine'."
            )
            pred_rows = []
            for r in gt1_records:
                pred = r.get("prediction") or "—"
                pred_rows.append({
                    "item_id":   r["item_id"],
                    "attribute": r["attribute"],
                    "GT value":  _synthpai_gt_value(r, items_by_id) or "—",
                    "difficulty": r.get("difficulty", "—"),
                    "label":     r.get("label", "—"),
                    "confidence": f"{float(r.get('confidence', 0) or 0):.2f}",
                    "prediction": str(pred)[:80],
                    "text (truncated)": str(r.get("text_content", ""))[:120],
                })
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)

    _render_synthpai_value_lookup(synth)


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

    _render_synthpai_gt_level(successful)

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
        run_labels = [_make_run_label(p) for p in runs]
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

    _apply_current_synthpai_gt(results, run_info)

    source = run_info.get("run_dir") or run_info.get("source")
    if source:
        st.caption(f"Source: `{source}`")
    corrected = int(run_info.get("_synthpai_gt_corrected", 0) or 0)
    missing = int(run_info.get("_synthpai_gt_missing", 0) or 0)
    if corrected:
        st.caption(f"SynthPAI GT/difficulty corrected in memory for {corrected} saved records.")
    if missing:
        st.warning(f"Could not refresh SynthPAI GT for {missing} saved records because item_id was not found.")
    _render_results(results, run_info)
    if selected_run_dir is not None and uploaded is None:
        _render_delete_section(selected_run_dir)


main()
