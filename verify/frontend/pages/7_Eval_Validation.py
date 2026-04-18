"""
Evaluation Validation — compare evaluator predictions against SynthPAI ground truth.

Loads SynthPAI text-modality IOC results and displays, for each item:
  1. Binary accuracy table  — does ext_eval[attr].inferable agree with reviews.human certainty?
  2. Profile prediction table — does ext_eval[attr].prediction match the person's actual
     SynthPAI profile value (age, gender, location, marital status)?

Green  = evaluator agrees with ground truth
Red    = evaluator disagrees
Grey   = data not available (prediction requires re-evaluation with prompt2)

Only IOC-comparison runs (perturbation_method == "ioc_comparison") on the SynthPAI
dataset are shown.  No new experiments are executed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT  = Path(__file__).resolve().parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Evaluation Validation — Verify",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SynthPAI content-label helper ────────────────────────────────────────────

# Mirrors _SYNTHPAI_REVIEW_MAP in label_mapper.py
_REVIEW_MAP: Dict[str, str] = {
    "age":                 "age",
    "sex":                 "gender",
    "city_country":        "location",
    "relationship_status": "marital status",
    "occupation":          "identity",
}


def _content_labels(result: Dict[str, Any]) -> Dict[str, int]:
    """
    Return binary content-revealed labels for a SynthPAI cached result.

    Uses reviews.human[field].certainty > 0 instead of the stored input_labels
    (which were set from profile completeness and are always 1 for all attributes).
    Falls back to stored input_labels for non-SynthPAI items.
    """
    input_item = result.get("input_item", {})
    if input_item.get("label_source") != "synthpai":
        return result.get("input_labels", {})

    stored   = result.get("input_labels", {})
    derived  = {attr: 0 for attr in stored}
    raw      = input_item.get("raw") or {}
    human    = (raw.get("reviews") or {}).get("human") or {}

    for review_field, attr in _REVIEW_MAP.items():
        if attr not in derived:
            continue
        review    = human.get(review_field) or {}
        estimate  = str(review.get("estimate", "") or "").strip()
        certainty = float(review.get("certainty", 0) or 0)
        if estimate and estimate not in ("None", "null") and certainty > 0:
            derived[attr] = 1

    return derived


# ── SynthPAI profile → attribute mapping ─────────────────────────────────────
# Maps our evaluation attribute name → SynthPAI profile field name
SYNTHPAI_ATTR_MAP: Dict[str, str] = {
    "age":           "age",
    "gender":        "sex",
    "location":      "city_country",
    "marital status": "relationship_status",
    "identity":      "username",
}

# SynthPAI profile fields displayed even without an MCQ prediction
SYNTHPAI_EXTRA_FIELDS = ["occupation", "income_level", "education"]

# Age-bin mapping (must match prompt2.yaml choices)
_AGE_BINS: List[Tuple[int, str]] = [
    (18,  "under 18"),
    (30,  "18-29"),
    (40,  "30-39"),
    (50,  "40-49"),
    (60,  "50-59"),
    (70,  "60-69"),
    (80,  "70-79"),
]
_AGE_MAX = "80 or older"


def _age_to_bin(age: int) -> str:
    for threshold, label in _AGE_BINS:
        if age < threshold:
            return label
    return _AGE_MAX


def _profile_truth(attr: str, profile: Dict[str, Any]) -> Optional[str]:
    """Return the human-readable ground-truth value for an attribute from a SynthPAI profile."""
    field = SYNTHPAI_ATTR_MAP.get(attr)
    if field is None:
        return None
    val = profile.get(field)
    if val is None:
        return None
    if attr == "age":
        try:
            return _age_to_bin(int(val))
        except (ValueError, TypeError):
            return str(val)
    return str(val).strip()


def _prediction_matches(prediction: Optional[str], truth: Optional[str], attr: str) -> Optional[bool]:
    """True/False/None (no data)."""
    if prediction is None or truth is None:
        return None
    p = prediction.strip().lower()
    t = truth.strip().lower()
    if p in ("cannot determine", "n/a", ""):
        return None
    if attr == "location":
        # Partial match: at least one token overlaps (city or country name)
        p_tokens = set(p.replace(",", " ").split())
        t_tokens = set(t.replace(",", " ").split())
        return bool(p_tokens & t_tokens)
    if attr == "identity":
        return p in t or t in p
    return p == t


# ── HTML table helpers ────────────────────────────────────────────────────────

_TH = (
    "text-align:center;padding:7px 10px;border:1px solid #d0d0d0;"
    "background:#f0f0f0;font-size:0.88em;font-weight:600;"
)
_TH_L = (
    "text-align:left;padding:7px 10px;border:1px solid #d0d0d0;"
    "background:#f0f0f0;font-size:0.88em;font-weight:600;"
)
_TD_ATTR = (
    "text-align:left;padding:6px 12px;border:1px solid #e0e0e0;"
    "font-size:0.92em;font-weight:500;"
)
_TD = (
    "text-align:center;padding:6px 10px;border:1px solid #e0e0e0;"
    "font-size:0.9em;"
)

_GREEN = "background:#d4f5d4;"
_RED   = "background:#fde8e8;"
_GREY  = "background:#f5f5f5;color:#aaa;"
_PLAIN = "background:#fafafa;"


def _match_cell(match: Optional[bool]) -> str:
    if match is True:
        return f'<td style="{_TD}{_GREEN}">✅</td>'
    if match is False:
        return f'<td style="{_TD}{_RED}">❌</td>'
    return f'<td style="{_TD}{_GREY}">—</td>'


def _text_cell(text: Optional[str], *, highlight: Optional[bool] = None) -> str:
    if text is None:
        return f'<td style="{_TD}{_GREY}">—</td>'
    bg = ""
    if highlight is True:
        bg = _GREEN
    elif highlight is False:
        bg = _RED
    else:
        bg = _PLAIN
    safe = str(text).replace("<", "&lt;").replace(">", "&gt;")
    return f'<td style="{_TD}{bg}">{safe}</td>'


# ── Binary accuracy table ─────────────────────────────────────────────────────

def _binary_table(
    input_labels: Dict[str, int],
    ext_eval:     Dict[str, Any],
    unified_attrs: List[str],
):
    """Evaluator binary inferable vs. ground-truth input_labels."""
    rows = ""
    for attr in unified_attrs:
        gt_val     = input_labels.get(attr, 0)
        gt_flag    = gt_val == 1
        entry      = ext_eval.get(attr)
        pred_flag  = isinstance(entry, dict) and bool(entry.get("inferable"))
        match      = gt_flag == pred_flag

        gt_cell    = f'<td style="{_TD}{"background:#d4f5d4;" if gt_flag else _PLAIN}">{"✅" if gt_flag else ""}</td>'
        pred_cell  = f'<td style="{_TD}{"background:#d4f5d4;" if pred_flag else _PLAIN}">{"✅" if pred_flag else ""}</td>'
        match_cell = f'<td style="{_TD}{_GREEN if match else _RED}">{"✅" if match else "❌"}</td>'
        reasoning  = ""
        if isinstance(entry, dict):
            r = entry.get("reasoning", "")
            if r:
                safe_r = r.replace("<", "&lt;").replace(">", "&gt;")
                reasoning = f'<span style="font-size:0.8em;color:#666">{safe_r}</span>'

        rows += (
            f"<tr>"
            f'<td style="{_TD_ATTR}">{attr}</td>'
            f"{gt_cell}{pred_cell}{match_cell}"
            f'<td style="text-align:left;padding:6px 10px;border:1px solid #e0e0e0;font-size:0.85em;color:#555">{reasoning}</td>'
            f"</tr>"
        )

    st.markdown(
        f"""
<table style="width:100%;border-collapse:collapse;margin-top:4px">
  <thead><tr>
    <th style="{_TH_L}width:22%">Attribute</th>
    <th style="{_TH}width:13%">Ground Truth<br><small>(input label)</small></th>
    <th style="{_TH}width:13%">Evaluator<br><small>(ext_eval)</small></th>
    <th style="{_TH}width:10%">Match</th>
    <th style="{_TH_L}width:42%">Reasoning</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>""",
        unsafe_allow_html=True,
    )


# ── Profile prediction table ──────────────────────────────────────────────────

def _profile_table(
    ext_eval:      Dict[str, Any],
    profile:       Dict[str, Any],
):
    """MCQ prediction vs. actual SynthPAI profile values (text-focused attributes)."""
    rows = ""
    shown = False
    for attr, field in SYNTHPAI_ATTR_MAP.items():
        truth  = _profile_truth(attr, profile)
        entry  = ext_eval.get(attr)
        pred   = entry.get("prediction") if isinstance(entry, dict) else None
        match  = _prediction_matches(pred, truth, attr)

        truth_cell = _text_cell(truth)
        pred_cell  = _text_cell(pred, highlight=match)
        mch_cell   = _match_cell(match)

        attr_label = f"{attr} ({field})"
        rows += (
            f"<tr>"
            f'<td style="{_TD_ATTR}">{attr_label}</td>'
            f"{pred_cell}{truth_cell}{mch_cell}"
            f"</tr>"
        )
        shown = True

    if not shown:
        return

    has_predictions = any(
        isinstance(ext_eval.get(a), dict) and ext_eval[a].get("prediction") is not None
        for a in SYNTHPAI_ATTR_MAP
    )

    if not has_predictions:
        st.caption(
            "ℹ️ No MCQ predictions found — run `python verify/reeval.py --model <model> "
            "--prompt2 --dataset SynthPAI` to populate the `prediction` field."
        )

    st.markdown(
        f"""
<table style="width:100%;border-collapse:collapse;margin-top:4px">
  <thead><tr>
    <th style="{_TH_L}width:26%">Attribute (profile field)</th>
    <th style="{_TH}width:27%">Evaluator Prediction</th>
    <th style="{_TH}width:27%">Ground Truth<br><small>(SynthPAI profile)</small></th>
    <th style="{_TH}width:10%">Match</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>""",
        unsafe_allow_html=True,
    )


# ── Per-item renderer ─────────────────────────────────────────────────────────

def _render_item(result: Dict[str, Any], unified_attrs: List[str], idx: int):
    input_item   = result.get("input_item", {})
    input_labels = _content_labels(result)          # reviews.human-based for SynthPAI
    profile      = input_item.get("synthpai_profile", {})
    filename     = result.get("filename", "unknown")
    status       = result.get("status", "")
    ext_eval     = result.get("ext_eval", {})

    # Expander label
    positives = [a for a in unified_attrs if input_labels.get(a, 0) == 1]
    gt_snippet = ", ".join(positives[:5]) + ("…" if len(positives) > 5 else "")
    status_icon = {"success": "✅", "failed": "❌"}.get(status, "⬜")
    label = f"{status_icon} {filename}"
    if gt_snippet:
        label += f"  —  🏷 {gt_snippet}"
    if profile:
        age_raw = profile.get("age", "?")
        sex     = profile.get("sex", "?")
        loc     = profile.get("city_country", "?")
        label += f"  |  {sex}, {age_raw}, {loc}"

    with st.expander(label, expanded=False):
        if status == "failed":
            st.error(f"Error: {result.get('error', 'Unknown error')}")
            return

        # ── Input text + profile summary ──────────────────────────────────
        col_in, col_profile = st.columns([3, 2])

        with col_in:
            st.markdown("**Reddit post / input text**")
            text_content = input_item.get("text_content", "")
            st.text_area(
                "text", value=text_content,
                height=130, disabled=True, label_visibility="collapsed",
                key=f"eval_val_text_{idx}",
            )
            if result.get("output_text"):
                st.markdown("**App output**")
                st.text_area(
                    "output", value=result.get("output_text", ""),
                    height=90, disabled=True, label_visibility="collapsed",
                    key=f"eval_val_out_{idx}",
                )

        with col_profile:
            st.markdown("**SynthPAI profile (ground truth)**")
            if profile:
                display_fields = [
                    ("age",               profile.get("age")),
                    ("sex",               profile.get("sex")),
                    ("city / country",    profile.get("city_country")),
                    ("relationship",      profile.get("relationship_status")),
                    ("occupation",        profile.get("occupation")),
                    ("income level",      profile.get("income_level")),
                    ("education",         profile.get("education")),
                ]
                rows_prof = "".join(
                    f'<tr><td style="padding:3px 8px;font-size:0.85em;color:#555;font-weight:600">{k}</td>'
                    f'<td style="padding:3px 8px;font-size:0.85em">{v if v is not None else "—"}</td></tr>'
                    for k, v in display_fields
                )
                st.markdown(
                    f'<table style="border-collapse:collapse">{rows_prof}</table>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No profile data available.")

        st.divider()

        # ── Binary accuracy ───────────────────────────────────────────────
        st.markdown("**Inferability accuracy** — evaluator vs. ground-truth labels")
        _binary_table(input_labels, ext_eval, unified_attrs)

        # ── Profile prediction (MCQ) ──────────────────────────────────────
        if profile:
            st.markdown(
                "**Profile prediction** — MCQ value prediction vs. actual profile",
                help=(
                    "Populated after running:\n"
                    "python verify/reeval.py --model <model> --prompt2 --dataset SynthPAI\n"
                    "Shows '—' until predictions are available."
                ),
            )
            _profile_table(ext_eval, profile)


# ── Aggregate accuracy section ────────────────────────────────────────────────

def _safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den > 0 else None


def _fmt(val: Optional[float], pct: bool = True) -> str:
    if val is None:
        return "—"
    return f"{val:.1%}" if pct else f"{val:.3f}"


def _aggregate_section(items: List[Dict[str, Any]], unified_attrs: List[str]):
    import pandas as pd

    success = [r for r in items if r.get("status") == "success"]
    if not success:
        st.info("No successful items.")
        return

    n = len(success)

    # Compute TP / TN / FP / FN per attribute
    rows = []
    for attr in unified_attrs:
        tp = tn = fp = fn = 0
        for r in success:
            gt   = _content_labels(r).get(attr, 0) == 1
            entry = r.get("ext_eval", {}).get(attr)
            pred  = isinstance(entry, dict) and bool(entry.get("inferable"))
            if gt and pred:
                tp += 1
            elif not gt and not pred:
                tn += 1
            elif not gt and pred:
                fp += 1
            else:
                fn += 1

        precision = _safe_div(tp, tp + fp)
        recall    = _safe_div(tp, tp + fn)
        f1        = _safe_div(
            2 * precision * recall,
            precision + recall,
        ) if precision is not None and recall is not None else None
        accuracy  = (tp + tn) / n

        rows.append({
            "Attribute": attr,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": precision,
            "Recall":    recall,
            "F1":        f1,
            "Accuracy":  accuracy,
        })

    df = pd.DataFrame(rows)

    # ── Chart tab / table tab ─────────────────────────────────────────────
    tab_chart, tab_table = st.tabs(["Chart", "Table"])

    with tab_chart:
        try:
            import altair as alt

            # ── Stacked TP/FP/FN/TN bar (counts) ─────────────────────────
            count_rows = []
            for row in rows:
                for seg, val, color in [
                    ("TP", row["TP"], "#5cb85c"),
                    ("FP", row["FP"], "#d9534f"),
                    ("FN", row["FN"], "#f0ad4e"),
                    ("TN", row["TN"], "#aec7e8"),
                ]:
                    count_rows.append({"Attribute": row["Attribute"], "Segment": seg,
                                       "Count": val, "Color": color})

            df_counts = pd.DataFrame(count_rows)
            seg_order = ["TP", "FP", "FN", "TN"]

            bar = (
                alt.Chart(df_counts)
                .mark_bar()
                .encode(
                    x=alt.X("Attribute:N", sort=unified_attrs,
                            axis=alt.Axis(labelAngle=-40, labelFontSize=10)),
                    y=alt.Y("Count:Q", title="Item count"),
                    color=alt.Color(
                        "Segment:N",
                        sort=seg_order,
                        scale=alt.Scale(
                            domain=seg_order,
                            range=["#5cb85c", "#d9534f", "#f0ad4e", "#aec7e8"],
                        ),
                        legend=alt.Legend(title=""),
                    ),
                    order=alt.Order("Segment:N", sort="ascending"),
                    tooltip=["Attribute", "Segment", "Count"],
                )
                .properties(height=260, title=f"TP / FP / FN / TN counts  (n = {n})")
            )

            # ── Precision / Recall / F1 lines ─────────────────────────────
            metric_rows = []
            for row in rows:
                for metric, val in [
                    ("Precision", row["Precision"]),
                    ("Recall",    row["Recall"]),
                    ("F1",        row["F1"]),
                ]:
                    metric_rows.append({
                        "Attribute": row["Attribute"],
                        "Metric":    metric,
                        "Value":     val if val is not None else 0.0,
                        "Defined":   val is not None,
                    })

            df_metrics = pd.DataFrame(metric_rows)

            lines = (
                alt.Chart(df_metrics)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Attribute:N", sort=unified_attrs,
                            axis=alt.Axis(labelAngle=-40, labelFontSize=10)),
                    y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]),
                            title="Score"),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(
                            domain=["Precision", "Recall", "F1"],
                            range=["#1f77b4", "#ff7f0e", "#2ca02c"],
                        ),
                    ),
                    strokeDash=alt.StrokeDash(
                        "Metric:N",
                        scale=alt.Scale(
                            domain=["Precision", "Recall", "F1"],
                            range=[[4, 2], [1, 0], [6, 2]],
                        ),
                    ),
                    tooltip=["Attribute", "Metric", alt.Tooltip("Value:Q", format=".3f")],
                )
                .properties(height=220, title="Precision / Recall / F1 per attribute")
            )

            st.altair_chart(bar,   width="stretch")
            st.altair_chart(lines, width="stretch")
            st.caption(
                "Stacked bar: **green** = TP · **red** = FP · **amber** = FN · **blue-grey** = TN  |  "
                "Lines: **blue** = Precision · **orange** = Recall · **green** = F1  |  "
                "Undefined metrics (no positives) plotted as 0."
            )

        except ImportError:
            st.info("Install altair for charts.")

    with tab_table:
        display_df = pd.DataFrame([{
            "Attribute": r["Attribute"],
            "TP": r["TP"],
            "TN": r["TN"],
            "FP": r["FP"],
            "FN": r["FN"],
            "Precision": _fmt(r["Precision"]),
            "Recall":    _fmt(r["Recall"]),
            "F1":        _fmt(r["F1"]),
            "Accuracy":  _fmt(r["Accuracy"]),
        } for r in rows])

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Macro-average summary (skip undefined)
        valid_p  = [r["Precision"] for r in rows if r["Precision"] is not None]
        valid_r  = [r["Recall"]    for r in rows if r["Recall"]    is not None]
        valid_f1 = [r["F1"]        for r in rows if r["F1"]        is not None]
        macro_p  = sum(valid_p)  / len(valid_p)  if valid_p  else None
        macro_r  = sum(valid_r)  / len(valid_r)  if valid_r  else None
        macro_f1 = sum(valid_f1) / len(valid_f1) if valid_f1 else None
        macro_acc = df["Accuracy"].mean()

        st.divider()
        st.markdown("**Macro-average** (attributes with at least one GT positive)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", _fmt(macro_p))
        m2.metric("Recall",    _fmt(macro_r))
        m3.metric("F1",        _fmt(macro_f1))
        m4.metric("Accuracy",  _fmt(macro_acc))


# ── Directory discovery ───────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _list_synthpai_ioc_dirs() -> List[Tuple[str, str, str, str]]:
    """
    Scan verify/outputs/ for SynthPAI + text + ioc_comparison runs.
    Returns list of (dir_path_str, app_name, dir_name, label).
    """
    outputs_root = LANTERN_ROOT / "verify" / "outputs"
    if not outputs_root.exists():
        return []

    found = []
    for d in outputs_root.iterdir():
        if not d.is_dir():
            continue
        cfg_path = d / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if (
            cfg.get("dataset_name") == "SynthPAI"
            and cfg.get("modality") == "text"
            and cfg.get("perturbation_method") == "ioc_comparison"
        ):
            app   = cfg.get("app_name", "unknown")
            label = f"{app}  [{d.name}]"
            found.append((str(d), app, d.name, label))

    return sorted(found, key=lambda x: x[2], reverse=True)


def _load_items(dir_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load items + run_config from a directory."""
    d   = Path(dir_path)
    cfg = {}
    try:
        cfg = json.loads((d / "run_config.json").read_text())
    except Exception:
        pass

    items = []
    for f in sorted(d.iterdir()):
        if f.suffix == ".json" and f.name not in ("run_config.json", "dir_summary.json"):
            try:
                items.append(json.loads(f.read_text()))
            except Exception:
                pass
    return items, cfg


# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    st.title("🎯 Evaluation Validation")
    st.markdown(
        "Compare the evaluator's predictions against **SynthPAI ground truth** "
        "(text-modality IOC runs only)."
    )

    available = _list_synthpai_ioc_dirs()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Select Run")

        if not available:
            st.warning(
                "No SynthPAI + text + IOC runs found in `verify/outputs/`.\n\n"
                "Run the **Input / Output Comparison** page with the SynthPAI dataset first."
            )
            return

        # App filter
        all_apps  = sorted({a for _, a, _, _ in available})
        app_filter = st.multiselect("Filter by app", all_apps, default=all_apps)

        filtered = [(p, a, n, l) for p, a, n, l in available if a in app_filter]

        if not filtered:
            st.info("No matching runs.")
            return

        options = {l: p for p, a, n, l in filtered}
        selected_label = st.selectbox("Pick a run", list(options.keys()))
        selected_path  = options[selected_label]

        load_btn = st.button("📂 Load", type="primary", width="stretch")

        st.divider()
        st.caption(
            "**Prediction column** is populated after running:\n\n"
            "```\npython verify/reeval.py \\\n"
            "  --model google/gemini-2.0-flash-001 \\\n"
            "  --prompt2 \\\n"
            "  --dataset SynthPAI \\\n"
            "  --workers 4\n```\n\n"
            "Or for a specific cache directory:\n\n"
            "```\npython verify/reeval.py \\\n"
            "  --model google/gemini-2.0-flash-001 \\\n"
            "  --prompt2 \\\n"
            "  --dir verify/outputs/<cache_dir>\n```"
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    if load_btn:
        items, cfg = _load_items(selected_path)
        st.session_state["ev_val_items"]  = items
        st.session_state["ev_val_cfg"]    = cfg
        st.session_state["ev_val_dir"]    = selected_path

    items  = st.session_state.get("ev_val_items")
    cfg    = st.session_state.get("ev_val_cfg", {})
    dirstr = st.session_state.get("ev_val_dir", "")

    if items is None:
        st.info("Select a run in the sidebar and click **📂 Load**.")
        return

    app_name      = cfg.get("app_name", "unknown")
    dataset_name  = cfg.get("dataset_name", "SynthPAI")
    modality      = cfg.get("modality", "text")
    unified_attrs = cfg.get("unified_attrs", [])
    if not unified_attrs and items:
        for item in items:
            keys = list(item.get("ext_eval", {}).keys())
            if keys:
                unified_attrs = keys
                break

    # ── Run info ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("App",      app_name)
    c2.metric("Dataset",  dataset_name)
    c3.metric("Modality", modality)
    c4.metric("Items",    len(items))
    if dirstr:
        st.caption(f"Cache: `{Path(dirstr).name}`")

    # ── Aggregate accuracy ────────────────────────────────────────────────────
    if unified_attrs:
        st.divider()
        st.subheader("Aggregate Binary Accuracy")
        _aggregate_section(items, unified_attrs)

    # ── Text filter ───────────────────────────────────────────────────────────
    st.divider()
    n = len(items)
    st.subheader(f"Per-item Results  ({n} item{'s' if n != 1 else ''})")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        text_filter = st.text_input(
            "Filter items by filename or profile value",
            placeholder="e.g. 'married', 'male', '0Bb'",
            key="ev_val_filter",
        )
    with col_f2:
        status_filter = st.selectbox(
            "Status", ["all", "success", "failed"], key="ev_val_status"
        )

    def _item_matches(r: Dict[str, Any]) -> bool:
        if status_filter != "all" and r.get("status") != status_filter:
            return False
        if text_filter:
            needle = text_filter.lower()
            if needle in r.get("filename", "").lower():
                return True
            profile = r.get("input_item", {}).get("synthpai_profile", {})
            for v in profile.values():
                if isinstance(v, (str, int)) and needle in str(v).lower():
                    return True
            return False
        return True

    filtered_items = [r for r in items if _item_matches(r)]
    st.caption(f"Showing {len(filtered_items)} / {n} items")

    for idx, result in enumerate(filtered_items):
        _render_item(result, unified_attrs, idx)


if __name__ == "__main__":
    main()
