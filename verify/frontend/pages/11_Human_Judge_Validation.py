"""
Human Judge Validation - compare human labels against saved LLM judge runs.

Loads judge-validation outputs from verify/outputs/judge_validation_runs and
collects independent human labels for the same (item, attribute) records. Human
decisions are saved separately under verify/outputs/human_judge_validation_runs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import pandas as pd
import streamlit as st

from verify.backend.evaluation_method.evaluator import (
    VERDICT_CONFIRMED,
    VERDICT_NO_EVIDENCE,
    VERDICT_POSSIBLE,
)
from verify.backend.judge.metrics import LABEL_CONFIRMED, LABEL_NONE, LABEL_POSSIBLE
from verify.backend.utils.config import load_color_palette


_JUDGE_RUNS_DIR = VERIFY_ROOT / "outputs" / "judge_validation_runs"
_HUMAN_RUNS_DIR = VERIFY_ROOT / "outputs" / "human_judge_validation_runs"
_LABEL_ORDER = [LABEL_CONFIRMED, LABEL_POSSIBLE, LABEL_NONE]
_SYNTHPAI_DATASET = "SynthPAI"
_PALETTE = load_color_palette()
_VERDICT_COLORS = {
    LABEL_CONFIRMED: _PALETTE["verdict"][VERDICT_CONFIRMED],
    LABEL_POSSIBLE: _PALETTE["verdict"][VERDICT_POSSIBLE],
    LABEL_NONE: _PALETTE["verdict"][VERDICT_NO_EVIDENCE],
}


def _find_judge_runs() -> List[Path]:
    if not _JUDGE_RUNS_DIR.exists():
        return []
    return sorted(
        [d for d in _JUDGE_RUNS_DIR.iterdir() if d.is_dir() and (d / "results.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load_judge_run(run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results = json.loads((run_dir / "results.json").read_text())
    info_path = run_dir / "run_info.json"
    run_info = json.loads(info_path.read_text()) if info_path.exists() else {}
    run_info.setdefault("run_dir", str(run_dir))
    run_info.setdefault("n_samples", len(results))
    return results, run_info


def _datasets_in_run(run_dir: Path) -> List[str]:
    try:
        results, _ = _load_judge_run(run_dir)
    except Exception:
        return []
    return sorted({str(r.get("dataset") or "").strip() for r in results if r.get("dataset")})


def _run_label(run_dir: Path) -> str:
    try:
        results, info = _load_judge_run(run_dir)
        datasets = "+".join(sorted({str(r.get("dataset") or "") for r in results if r.get("dataset")}))
        n_items = len({(r.get("dataset"), r.get("item_id")) for r in results})
        n_samples = len(results)
        model = str(info.get("model") or "model?")
    except Exception:
        datasets, n_items, n_samples, model = run_dir.name, "?", "?", "model?"

    parts = run_dir.name.split("_")
    date_str = parts[2][4:8] if len(parts) >= 3 and len(parts[2]) >= 8 else "????"
    time_str = parts[3] if len(parts) >= 4 else "??????"
    return f"{datasets} | {n_items} items / {n_samples} rows | {model} | {date_str}_{time_str}"


def _safe_key(value: str) -> str:
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()


def _format_label(label: Any) -> str:
    label = str(label or LABEL_NONE).strip().lower()
    return label if label in _LABEL_ORDER else LABEL_NONE


def _select_item_records(
    results: List[Dict[str, Any]],
    dataset: str,
    n_items: int,
    only_successful: bool,
) -> List[Dict[str, Any]]:
    filtered = [
        r
        for r in results
        if str(r.get("dataset") or "") == dataset
        and (not only_successful or bool(r.get("judge_ok", True)))
    ]

    selected_item_ids: List[str] = []
    seen = set()
    for r in filtered:
        item_id = str(r.get("item_id") or r.get("id") or "")
        if item_id and item_id not in seen:
            selected_item_ids.append(item_id)
            seen.add(item_id)
        if len(selected_item_ids) >= n_items:
            break

    selected = set(selected_item_ids)
    return [r for r in filtered if str(r.get("item_id") or r.get("id") or "") in selected]


def _human_runs_for_judge_run(run_dir: Path, dataset: str) -> List[Path]:
    if not _HUMAN_RUNS_DIR.exists():
        return []
    matches = []
    run_dir_str = str(run_dir.resolve())
    for d in _HUMAN_RUNS_DIR.iterdir():
        if not d.is_dir() or not (d / "human_decisions.json").exists():
            continue
        try:
            payload = json.loads((d / "human_decisions.json").read_text())
            info = payload.get("validation_info", {})
            if str(Path(info.get("judge_run_dir", "")).resolve()) == run_dir_str and info.get("dataset") == dataset:
                matches.append(d)
        except Exception:
            continue
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)


def _load_existing_decisions(human_dir: Path) -> Dict[str, Dict[str, Any]]:
    try:
        payload = json.loads((human_dir / "human_decisions.json").read_text())
    except Exception:
        return {}
    decisions = payload.get("decisions", [])
    return {
        str(d.get("id") or ""): d
        for d in decisions
        if d.get("id") and d.get("human_label") in _LABEL_ORDER
    }


def _cohens_kappa(human_labels: List[str], judge_labels: List[str]) -> Optional[float]:
    if len(human_labels) != len(judge_labels) or not human_labels:
        return None

    n = len(human_labels)
    observed = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / n
    human_counts = Counter(human_labels)
    judge_counts = Counter(judge_labels)
    expected = sum((human_counts[l] / n) * (judge_counts[l] / n) for l in _LABEL_ORDER)

    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def _save_human_decisions(
    records: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    validation_info: Dict[str, Any],
) -> Path:
    _HUMAN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    validation_id = datetime.now().strftime("human_judge_validation_%Y%m%d_%H%M%S_%f")
    run_dir = _HUMAN_RUNS_DIR / validation_id
    run_dir.mkdir(parents=True, exist_ok=False)

    payload = {
        "validation_info": {
            **validation_info,
            "validation_id": validation_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "human_run_dir": str(run_dir),
            "n_items_labeled": len({d["item_id"] for d in decisions}),
            "n_samples_labeled": len(decisions),
        },
        "decisions": decisions,
    }
    (run_dir / "human_decisions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    rows = []
    record_by_id = {r.get("id"): r for r in records}
    for d in decisions:
        r = record_by_id.get(d["id"], {})
        rows.append({
            "id": d["id"],
            "dataset": d["dataset"],
            "item_id": d["item_id"],
            "attribute": d["attribute"],
            "judge_label": d["judge_label"],
            "human_label": d["human_label"],
            "judge_prediction": r.get("prediction", ""),
            "human_candidate": d.get("human_candidate", ""),
            "human_notes": d.get("human_notes", ""),
        })
    pd.DataFrame(rows).to_csv(run_dir / "human_decisions.csv", index=False)
    return run_dir


def _render_content(record: Dict[str, Any], key_prefix: str) -> None:
    if record.get("text_content"):
        st.text_area(
            "Data item",
            str(record.get("text_content") or "")[:4000],
            height=170,
            disabled=True,
            key=f"{key_prefix}_content",
        )
    elif record.get("image_b64"):
        try:
            st.image(base64.b64decode(record["image_b64"]), use_container_width=True)
        except Exception:
            st.caption("Could not render image.")
    else:
        st.caption("No displayable data item is available for this record.")


def _seed_widget_state(records: List[Dict[str, Any]], existing: Dict[str, Dict[str, Any]]) -> None:
    for r in records:
        rid = str(r.get("id") or "")
        suffix = _safe_key(rid)
        loaded = existing.get(rid, {})
        label_key = f"hjv_label_{suffix}"
        candidate_key = f"hjv_candidate_{suffix}"
        notes_key = f"hjv_notes_{suffix}"
        if label_key not in st.session_state and loaded.get("human_label") in _LABEL_ORDER:
            st.session_state[label_key] = loaded.get("human_label")
        if candidate_key not in st.session_state:
            st.session_state[candidate_key] = loaded.get("human_candidate", "")
        if notes_key not in st.session_state:
            st.session_state[notes_key] = loaded.get("human_notes", "")


def _render_labeling_records(records: List[Dict[str, Any]]) -> None:
    st.subheader(f"Label Records ({len(records)} item-attribute rows)")
    st.caption("Choose the human verdict independently from the judge result. Judge output is hidden until after submission.")

    item_count = len({str(r.get("item_id") or r.get("id") or "") for r in records})
    progress = sum(
        1
        for r in records
        if st.session_state.get(f"hjv_label_{_safe_key(str(r.get('id') or ''))}") in _LABEL_ORDER
    )
    c1, c2 = st.columns(2)
    c1.metric("Dataset items", item_count)
    c2.metric("Labeled rows", f"{progress}/{len(records)}")

    for idx, r in enumerate(records, start=1):
        rid = str(r.get("id") or f"record_{idx}")
        suffix = _safe_key(rid)
        attr = str(r.get("attribute") or "unknown")
        item_id = str(r.get("item_id") or rid)
        label_key = f"hjv_label_{suffix}"
        candidate_key = f"hjv_candidate_{suffix}"
        notes_key = f"hjv_notes_{suffix}"

        current_label = st.session_state.get(label_key)
        status = current_label if current_label in _LABEL_ORDER else "unlabeled"
        header = f"{idx}. {item_id} | {attr} | {status}"
        with st.expander(header, expanded=current_label not in _LABEL_ORDER):
            left, right = st.columns([1.15, 0.85])
            with left:
                _render_content(r, f"hjv_{suffix}")
            with right:
                st.markdown(f"**Attribute:** `{attr}`")
                st.radio(
                    "Human label",
                    _LABEL_ORDER,
                    key=label_key,
                    index=None,
                    horizontal=True,
                )
                chosen = st.session_state.get(label_key)
                if r.get("dataset") == _SYNTHPAI_DATASET and chosen in (LABEL_CONFIRMED, LABEL_POSSIBLE):
                    st.text_input(
                        "Candidate value",
                        key=candidate_key,
                        placeholder="e.g., Japan, software engineer, 20s",
                    )
                st.text_area("Notes", key=notes_key, height=95, placeholder="Optional")


def _collect_decisions(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    decisions: List[Dict[str, Any]] = []
    errors: List[str] = []
    for r in records:
        rid = str(r.get("id") or "")
        suffix = _safe_key(rid)
        label = st.session_state.get(f"hjv_label_{suffix}")
        candidate = str(st.session_state.get(f"hjv_candidate_{suffix}", "") or "").strip()
        notes = str(st.session_state.get(f"hjv_notes_{suffix}", "") or "").strip()
        attr = str(r.get("attribute") or "")
        item_id = str(r.get("item_id") or rid)

        if label not in _LABEL_ORDER:
            errors.append(f"{item_id} / {attr}: missing human label")
            continue
        if r.get("dataset") == _SYNTHPAI_DATASET and label in (LABEL_CONFIRMED, LABEL_POSSIBLE) and not candidate:
            errors.append(f"{item_id} / {attr}: candidate value is required for SynthPAI {label}")
            continue

        decisions.append({
            "id": rid,
            "dataset": r.get("dataset", ""),
            "item_id": item_id,
            "attribute": attr,
            "human_label": label,
            "human_candidate": candidate,
            "human_notes": notes,
            "judge_label": _format_label(r.get("label")),
            "judge_confidence": r.get("confidence"),
            "judge_prediction": r.get("prediction"),
            "judge_explanation": r.get("explanation"),
            "judge_ok": bool(r.get("judge_ok", True)),
        })
    return decisions, errors


def _comparison_rows(decisions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        rows.append({
            "item_id": d.get("item_id", ""),
            "attribute": d.get("attribute", ""),
            "judge": d.get("judge_label", LABEL_NONE),
            "human": d.get("human_label", LABEL_NONE),
            "match": d.get("judge_label") == d.get("human_label"),
            "judge_prediction": d.get("judge_prediction") or "",
            "human_candidate": d.get("human_candidate") or "",
            "notes": d.get("human_notes") or "",
        })
    return pd.DataFrame(rows)


def _render_agreement_heatmap(decisions: List[Dict[str, Any]]) -> None:
    import altair as alt

    counts = Counter((d.get("human_label", LABEL_NONE), d.get("judge_label", LABEL_NONE)) for d in decisions)
    chart_rows = []
    max_count = max(counts.values()) if counts else 1
    for human in _LABEL_ORDER:
        for judge in _LABEL_ORDER:
            count = counts.get((human, judge), 0)
            chart_rows.append({"human": human, "judge": judge, "count": count, "intensity": count / max_count})
    df = pd.DataFrame(chart_rows)

    rect = (
        alt.Chart(df)
        .mark_rect(stroke="#FFFFFF", strokeWidth=2)
        .encode(
            x=alt.X("judge:N", sort=_LABEL_ORDER, title="Judge verdict", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("human:N", sort=_LABEL_ORDER, title="Human verdict"),
            color=alt.Color(
                "count:Q",
                scale=alt.Scale(scheme="teals"),
                legend=alt.Legend(title="Count"),
            ),
            tooltip=[
                alt.Tooltip("human:N", title="Human"),
                alt.Tooltip("judge:N", title="Judge"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
    )
    text = (
        alt.Chart(df)
        .mark_text(fontSize=17, fontWeight="bold")
        .encode(
            x=alt.X("judge:N", sort=_LABEL_ORDER),
            y=alt.Y("human:N", sort=_LABEL_ORDER),
            text=alt.Text("count:Q"),
            color=alt.condition(alt.datum.intensity > 0.55, alt.value("white"), alt.value("#111827")),
        )
    )
    st.altair_chart((rect + text).properties(height=260), use_container_width=True)


def _render_comparison(decisions: List[Dict[str, Any]], saved_dir: Optional[Path] = None) -> None:
    if not decisions:
        return

    st.divider()
    st.subheader("Human vs. Judge")
    if saved_dir is not None:
        st.success(f"Saved human decisions to `{saved_dir}`")

    human_labels = [d.get("human_label", LABEL_NONE) for d in decisions]
    judge_labels = [d.get("judge_label", LABEL_NONE) for d in decisions]
    kappa = _cohens_kappa(human_labels, judge_labels)
    agreement = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / len(decisions)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows compared", len(decisions))
    c2.metric("Agreement", f"{agreement:.1%}")
    c3.metric("Cohen's kappa", "N/A" if kappa is None else f"{kappa:.3f}")

    df = _comparison_rows(decisions)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Agreement Heatmap")
    _render_agreement_heatmap(decisions)


def _clear_labeling_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("hjv_label_") or key.startswith("hjv_candidate_") or key.startswith("hjv_notes_"):
            st.session_state.pop(key, None)
    for key in ("hjv_records", "hjv_run_info", "hjv_config", "hjv_existing_decisions", "hjv_saved_decisions", "hjv_saved_dir"):
        st.session_state.pop(key, None)


def main() -> None:
    st.title("Human Judge Validation")
    st.markdown("Cross-validate saved LLM judge verdicts with independent human labels.")

    runs = _find_judge_runs()
    if not runs:
        st.info("No saved judge-validation runs found. Run the Judge Validator page first.")
        return

    all_datasets = sorted({ds for run in runs for ds in _datasets_in_run(run)})
    if not all_datasets:
        st.info("No datasets found in saved judge-validation runs.")
        return

    with st.sidebar:
        st.header("Configuration")
        dataset = st.selectbox("Dataset", all_datasets, key="hjv_dataset")
        matching_runs = [run for run in runs if dataset in _datasets_in_run(run)]
        run_labels = [_run_label(run) for run in matching_runs]
        selected_label = st.selectbox("Judge run directory", run_labels, key="hjv_run")
        selected_run = matching_runs[run_labels.index(selected_label)]

        n_items = st.number_input(
            "Number of dataset items",
            min_value=1,
            max_value=5000,
            value=10,
            step=1,
            key="hjv_n_items",
        )
        only_successful = st.toggle(
            "Only judge-successful rows",
            value=True,
            key="hjv_only_successful",
        )

        existing_runs = _human_runs_for_judge_run(selected_run, dataset)
        existing_labels = ["Do not preload"] + [
            f"{p.name} ({datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})"
            for p in existing_runs
        ]
        existing_choice = st.selectbox("Preload human labels", existing_labels, key="hjv_existing_run")

        st.divider()
        start = st.button("Start Labeling", type="primary", use_container_width=True)
        clear = st.button("Clear Page State", use_container_width=True)

    if clear:
        _clear_labeling_state()
        st.rerun()

    if start:
        _clear_labeling_state()
        try:
            results, run_info = _load_judge_run(selected_run)
        except Exception as e:
            st.error(f"Could not load judge run: {e}")
            return

        records = _select_item_records(
            results,
            dataset=dataset,
            n_items=int(n_items),
            only_successful=only_successful,
        )
        if not records:
            st.warning("No records matched that configuration.")
            return

        existing: Dict[str, Dict[str, Any]] = {}
        if existing_choice != "Do not preload" and existing_runs:
            existing_idx = existing_labels.index(existing_choice) - 1
            existing = _load_existing_decisions(existing_runs[existing_idx])

        st.session_state["hjv_records"] = records
        st.session_state["hjv_run_info"] = run_info
        st.session_state["hjv_config"] = {
            "dataset": dataset,
            "n_items_requested": int(n_items),
            "only_successful": only_successful,
            "judge_run_dir": str(selected_run.resolve()),
            "judge_run_name": selected_run.name,
        }
        st.session_state["hjv_existing_decisions"] = existing
        st.rerun()

    records = st.session_state.get("hjv_records", [])
    if not records:
        st.info("Choose a dataset, judge run, and item count in the sidebar, then click **Start Labeling**.")
        return

    config = st.session_state.get("hjv_config", {})
    run_info = st.session_state.get("hjv_run_info", {})
    existing = st.session_state.get("hjv_existing_decisions", {})
    _seed_widget_state(records, existing)

    st.caption(
        f"Judge source: `{config.get('judge_run_dir', '')}` | "
        f"model: `{run_info.get('model', '-')}`"
    )
    _render_labeling_records(records)

    if st.button("Submit Human Decisions", type="primary", use_container_width=True):
        decisions, errors = _collect_decisions(records)
        if errors:
            st.error("Please finish the required labels before submitting.")
            with st.expander(f"Missing required fields ({len(errors)})", expanded=True):
                for err in errors[:100]:
                    st.write(f"- {err}")
                if len(errors) > 100:
                    st.caption(f"... and {len(errors) - 100} more")
            return

        validation_info = {
            **config,
            "judge_model": run_info.get("model"),
            "judge_evaluator": run_info.get("evaluator"),
        }
        saved_dir = _save_human_decisions(records, decisions, validation_info)
        st.session_state["hjv_saved_decisions"] = decisions
        st.session_state["hjv_saved_dir"] = str(saved_dir)
        st.rerun()

    saved_decisions = st.session_state.get("hjv_saved_decisions")
    if saved_decisions:
        saved_dir_raw = st.session_state.get("hjv_saved_dir")
        _render_comparison(saved_decisions, Path(saved_dir_raw) if saved_dir_raw else None)


main()
