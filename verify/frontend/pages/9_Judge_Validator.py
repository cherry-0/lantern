"""
Judge Validator — evaluate the LLM-based privacy leakage evaluator against
ground-truth labels from SynthPAI, HR-VISPR, and OpenPII.

For each (content, attribute) sample the existing evaluate_inferability_v5
evaluator is used as the judge.  Its 3-way verdict is compared against the
dataset's binary ground-truth label to produce precision, coverage, ambiguity,
and recall metrics.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT  = Path(__file__).resolve().parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import pandas as pd
import streamlit as st

from verify.frontend.judge_heatmap import render_judge_heatmaps
from verify.backend.evaluation_method.evaluator import (
    EVAL_MODEL,
    VERDICT_CONFIRMED,
    VERDICT_NO_EVIDENCE,
    VERDICT_POSSIBLE,
    _PROMPT5_SYSTEM,
    _build_eval_prompt_v4,
    _extract_channels_from_text,
    _normalize_verdict_result,
    evaluate_inferability_v5,
    get_aggregate_eval_entry,
)
from verify.backend.judge.data_loader import (
    DATASET_ATTRS,
    available_datasets,
    load_judge_samples,
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
    normalise_verdict,
)
from verify.backend.utils.config import get_openrouter_api_key, load_color_palette

# ── Constants ─────────────────────────────────────────────────────────────────

_CACHE_DIR  = VERIFY_ROOT / "outputs" / "judge_validation_cache"
_RUNS_DIR   = VERIFY_ROOT / "outputs" / "judge_validation_runs"
_EVALUATOR_VERSION = "v5-vision"
_PALETTE    = load_color_palette()
_VERDICT_COLORS = {
    LABEL_CONFIRMED: _PALETTE["verdict"][VERDICT_CONFIRMED],
    LABEL_POSSIBLE:  _PALETTE["verdict"][VERDICT_POSSIBLE],
    LABEL_NONE:      _PALETTE["verdict"][VERDICT_NO_EVIDENCE],
}

_DIFFICULTY_ORDER = ["explicit", "implicit", "none"]


# ── Model helpers ─────────────────────────────────────────────────────────────

def _vision_model_for_family(model: str) -> str:
    """
    Pick a vision-capable model from the same provider/family as *model*.

    HR-VISPR judge samples are images, so text-only evaluator models must be
    upgraded to a same-family VLM. Models already known to support images are
    kept unchanged.
    """
    m = (model or EVAL_MODEL).strip()
    lower = m.lower()

    if lower.startswith("google/"):
        return m if "gemini" in lower else "google/gemini-2.0-flash-001"

    if lower.startswith("openai/"):
        if any(name in lower for name in ("gpt-4o", "gpt-4.1", "gpt-4.5")):
            return m
        return "openai/gpt-4o-mini"

    if lower.startswith("anthropic/"):
        return m if "claude-3" in lower else "anthropic/claude-3.5-haiku"

    if lower.startswith("meta-llama/") or lower.startswith("meta/"):
        return m if "vision" in lower else "meta-llama/llama-3.2-11b-vision-instruct"

    if lower.startswith("qwen/"):
        return m if "vl" in lower else "qwen/qwen2.5-vl-72b-instruct"

    if lower.startswith("mistral") or lower.startswith("mistralai/"):
        return m if "pixtral" in lower else "mistralai/pixtral-12b"

    return "google/gemini-2.0-flash-001"


def _sample_model(sample: Dict[str, Any], requested_model: str) -> str:
    if sample.get("dataset") == "HR-VISPR" or sample.get("image_b64"):
        return _vision_model_for_family(requested_model)
    return requested_model


# ── Caching helpers ───────────────────────────────────────────────────────────

def _cache_key(sample_id: str, model: str) -> str:
    raw = f"{sample_id}||{model}||{_EVALUATOR_VERSION}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.json"


def _load_cached(sample_id: str, model: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(_cache_key(sample_id, model))
    if p.exists():
        try:
            cached = json.loads(p.read_text())
            if str(cached.get("model") or "") != model:
                return None
            cached_version = cached.get("evaluator_version")
            if cached_version and cached_version != _EVALUATOR_VERSION:
                return None
            return cached
        except Exception:
            return None
    return None


def _save_cached(sample_id: str, model: str, result: Dict[str, Any]) -> None:
    try:
        payload = {
            **result,
            "model": model,
            "evaluator_version": _EVALUATOR_VERSION,
        }
        _cache_path(_cache_key(sample_id, model)).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
    except Exception:
        pass


def _save_run_results(results: List[Dict[str, Any]], run_info: Dict[str, Any]) -> Optional[Path]:
    """Persist a completed judge-validation run so the viewer page can reload it."""
    try:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("judge_validation_%Y%m%d_%H%M%S_%f")
        run_dir = _RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        saved_info = {**run_info, "run_dir": str(run_dir)}
        (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        (run_dir / "run_info.json").write_text(json.dumps(saved_info, ensure_ascii=False, indent=2))
        return run_dir
    except Exception:
        return None


# ── Judge call (wraps evaluator) ──────────────────────────────────────────────

def _evaluate_inferability_v5_image(
    image_b64: str,
    attr: str,
    api_key: Optional[str],
    model: str,
) -> tuple[bool, Dict[str, Any], Optional[str]]:
    """Run the v5 judge prompt with an actual image content part."""
    key = api_key or get_openrouter_api_key()
    if not key or key.startswith("your_"):
        return False, {}, "No valid OpenRouter API key available for evaluation."
    if not image_b64:
        return False, {}, "No image data provided for HR-VISPR evaluation."

    import requests

    output_text = "[IMAGE] HR-VISPR sample image attached to this message."
    prompt = _build_eval_prompt_v4(output_text, [attr], include_prediction=True)
    detected_channels = _extract_channels_from_text(output_text)
    last_error = ""

    for _attempt in range(5):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Verify",
                    "X-Title": "Verify",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _PROMPT5_SYSTEM},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                    },
                                },
                            ],
                        },
                    ],
                    "max_tokens": 4096,
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"]["content"].strip()
            raw_content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_content)

            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if not match:
                    last_error = f"Could not parse JSON from v5 vision evaluator: {raw_content[:200]}"
                    continue
                parsed = json.loads(match.group())

            entry = parsed.get(attr, {})
            aggregate = _normalize_verdict_result(
                entry.get("aggregate", entry if isinstance(entry, dict) else {})
            )
            channel_results: Dict[str, Dict[str, Any]] = {}
            raw_channels = entry.get("channels", {}) if isinstance(entry, dict) else {}
            if isinstance(raw_channels, dict):
                for channel in detected_channels.keys():
                    if channel in raw_channels and isinstance(raw_channels[channel], dict):
                        channel_results[channel] = _normalize_verdict_result(raw_channels[channel])

            return True, {attr: {"aggregate": aggregate, "channels": channel_results}}, None
        except Exception as e:
            last_error = f"v5 vision evaluation API call failed: {e}"

    return False, {}, last_error

def _judge_sample(
    sample: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluate_inferability_v5 on a single sample and return a result dict.

    For text samples the text_content is passed directly.
    For image samples (HR-VISPR), a same-family vision model is selected and the
    base64 image is sent as an OpenRouter image_url content part.
    """
    effective_model = _sample_model(sample, model)
    if use_cache:
        cached = _load_cached(sample["id"], effective_model)
        if cached is not None:
            return {**sample, **cached, "from_cache": True}

    attr   = sample["attribute"]
    text   = sample.get("text_content", "")
    img_b64 = sample.get("image_b64", "")

    if img_b64:
        ok, results, error = _evaluate_inferability_v5_image(
            img_b64, attr, api_key=api_key, model=effective_model
        )
    else:
        output_text = text
        ok, results, error = evaluate_inferability_v5(
            output_text, [attr], api_key=api_key, model=effective_model
        )

    if not ok or attr not in results:
        result_fields = {
            "label":       LABEL_NONE,
            "confidence":  0.0,
            "explanation": error or "Evaluation failed.",
            "prediction":  None,
            "judge_ok":    False,
            "judge_error": error,
            "model":       effective_model,
        }
    else:
        agg     = get_aggregate_eval_entry(results[attr])
        verdict = agg.get("verdict", VERDICT_NO_EVIDENCE)
        result_fields = {
            "label":       normalise_verdict(verdict),
            "confidence":  agg.get("score", 0) / 2.0,   # score is 0/1/2 → 0/0.5/1.0
            "explanation": agg.get("reasoning", ""),
            "prediction":  agg.get("prediction"),
            "judge_ok":    True,
            "judge_error": None,
            "model":       effective_model,
        }

    if result_fields["judge_ok"]:
        _save_cached(sample["id"], effective_model, result_fields)
    return {**sample, **result_fields, "from_cache": False}


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_evaluation(
    samples: List[Dict[str, Any]],
    model: str,
    api_key: Optional[str],
    workers: int = 4,
    progress_bar=None,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Run judge on all samples with a thread pool.  Progress via Streamlit bar."""
    results: List[Optional[Dict[str, Any]]] = [None] * len(samples)
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_judge_sample, s, model, api_key, use_cache): i
            for i, s in enumerate(samples)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {
                    **samples[idx],
                    "label":       LABEL_NONE,
                    "confidence":  0.0,
                    "explanation": str(e),
                    "judge_ok":    False,
                    "judge_error": str(e),
                    "from_cache":  False,
                }
            done += 1
            if progress_bar is not None:
                progress_bar.progress(done / len(samples), text=f"Evaluated {done}/{len(samples)}")

    return [r for r in results if r is not None]


# ── UI helpers ────────────────────────────────────────────────────────────────

def _verdict_badge(label: str) -> str:
    color = _VERDICT_COLORS.get(label, "#e6e6e6")
    return f'<span style="background:{color};padding:2px 8px;border-radius:4px;font-size:0.85em">{label}</span>'


def _gt_badge(gt: int) -> str:
    color = _PALETTE["binary"]["positive"] if gt else _PALETTE["binary"]["negative"]
    text  = "positive" if gt else "negative"
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em">{text}</span>'


def _metric_card(col, label: str, value, fmt: str = ".1%", warn_none: str = "N/A") -> None:
    if value is None:
        col.metric(label, warn_none)
    else:
        col.metric(label, f"{value:{fmt}}")


# ── Section renderers ─────────────────────────────────────────────────────────

def _render_summary(results: List[Dict[str, Any]]) -> None:
    metrics = compute_metrics(results)
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "Precision (confirmed)",   metrics["precision_confirmed"])
    _metric_card(c2, "Coverage (confirmed)",    metrics["coverage_confirmed"])
    _metric_card(c3, "Ambiguity rate",          metrics["ambiguity_rate"])
    _metric_card(c4, "Recall lower bound",      metrics["recall_lower_bound"])


def _render_distribution(results: List[Dict[str, Any]]) -> None:
    dist = compute_distribution(results)
    labels = [LABEL_CONFIRMED, LABEL_POSSIBLE, LABEL_NONE]
    colors = [_VERDICT_COLORS[l] for l in labels]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Overall distribution**")
        # Wide format: one column per label so color list length matches columns
        df_wide = pd.DataFrame(
            {l: [dist[l]] for l in labels},
            index=pd.Index(["samples"]),
        )
        st.bar_chart(df_wide, color=colors, height=250)

    with col_b:
        st.markdown("**Breakdown by difficulty**")
        by_diff = compute_distribution_by_difficulty(results)
        rows = []
        for diff in _DIFFICULTY_ORDER:
            if diff not in by_diff:
                continue
            d = by_diff[diff]
            rows.append({"difficulty": diff, **{l: d[l] for l in labels}})
        if rows:
            df_diff = pd.DataFrame(rows).set_index("difficulty")
            st.bar_chart(df_diff[labels], color=colors, height=250)

    render_judge_heatmaps(results)


def _render_sample_viewer(results: List[Dict[str, Any]]) -> None:
    st.subheader(f"Sample Viewer  ({len(results)} samples)")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        attr_opts  = sorted({r["attribute"] for r in results})
        attr_sel   = st.multiselect("Attribute", attr_opts, key="jv_attr_filter")
    with filter_col2:
        diff_opts  = sorted({r["difficulty"] for r in results})
        diff_sel   = st.multiselect("Difficulty", diff_opts, key="jv_diff_filter")
    with filter_col3:
        label_opts = [LABEL_CONFIRMED, LABEL_POSSIBLE, LABEL_NONE]
        label_sel  = st.multiselect("Judge label", label_opts, key="jv_label_filter")

    filtered = [
        r for r in results
        if (not attr_sel  or r["attribute"]  in attr_sel)
        and (not diff_sel  or r["difficulty"]  in diff_sel)
        and (not label_sel or r.get("label")   in label_sel)
    ]
    st.caption(f"Showing {len(filtered)} / {len(results)}")

    for r in filtered[:200]:   # cap display at 200 rows
        gt    = int(r.get("ground_truth", 0))
        label = r.get("label", LABEL_NONE)
        icon  = "✅" if label == LABEL_CONFIRMED else ("⚠️" if label == LABEL_POSSIBLE else "⭕")
        header = (
            f"{icon} [{r['dataset']}] **{r['attribute']}** — "
            f"GT: {'pos' if gt else 'neg'} | judge: **{label}** "
            f"(conf {r.get('confidence', 0):.2f}) | {r['difficulty']}"
        )
        with st.expander(header):
            c1, c2 = st.columns(2)
            with c1:
                if r.get("text_content"):
                    st.text_area(
                        "Content",
                        r["text_content"][:2000],
                        height=160,
                        disabled=True,
                        key=f"jv_content_{r['id']}",
                        label_visibility="collapsed",
                    )
                elif r.get("image_b64"):
                    import base64
                    try:
                        st.image(base64.b64decode(r["image_b64"]), width="stretch")
                    except Exception:
                        st.caption("Could not render image.")
            with c2:
                st.markdown(
                    f"**Dataset:** {r['dataset']}  \n"
                    f"**Attribute:** {r['attribute']}  \n"
                    f"**Ground truth:** {_gt_badge(gt)}  \n"
                    f"**Judge label:** {_verdict_badge(label)}  \n"
                    f"**Confidence:** {r.get('confidence', 0):.2f}  \n"
                    f"**Difficulty:** {r.get('difficulty', '—')}  \n"
                    f"**Model:** {r.get('model', '—')}",
                    unsafe_allow_html=True,
                )
                st.markdown("**Explanation:**")
                st.caption(r.get("explanation") or "—")
                if r.get("prediction"):
                    st.markdown("**Prediction:**")
                    st.caption(str(r.get("prediction")))
                if not r.get("judge_ok"):
                    st.error(f"Judge error: {r.get('judge_error')}")


def _render_error_analysis(results: List[Dict[str, Any]]) -> None:
    st.subheader("Error Analysis")

    fps = false_positives(results)
    fns = false_negatives(results)

    col_fp, col_fn = st.columns(2)

    with col_fp:
        st.markdown(f"**False positives** — confirmed but GT=0 ({len(fps)})")
        if fps:
            df = pd.DataFrame([
                {
                    "id":        r["id"],
                    "dataset":   r["dataset"],
                    "attribute": r["attribute"],
                    "difficulty":r["difficulty"],
                    "conf":      f"{r.get('confidence',0):.2f}",
                }
                for r in fps[:100]
            ])
            st.dataframe(df, width="content")
        else:
            st.success("No false positives.")

    with col_fn:
        st.markdown(f"**False negatives** — GT=1 but not confirmed ({len(fns)})")
        if fns:
            df = pd.DataFrame([
                {
                    "id":        r["id"],
                    "dataset":   r["dataset"],
                    "attribute": r["attribute"],
                    "difficulty":r["difficulty"],
                    "label":     r.get("label", LABEL_NONE),
                    "conf":      f"{r.get('confidence',0):.2f}",
                }
                for r in fns[:100]
            ])
            st.dataframe(df, width="content")
        else:
            st.success("No false negatives.")


# ── Export ────────────────────────────────────────────────────────────────────

def _export_button(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    # CSV
    df = pd.DataFrame([
        {
            "id":           r["id"],
            "dataset":      r["dataset"],
            "attribute":    r["attribute"],
            "difficulty":   r["difficulty"],
            "ground_truth": r["ground_truth"],
            "label":        r.get("label", ""),
            "confidence":   r.get("confidence", ""),
            "prediction":   r.get("prediction", ""),
            "explanation":  r.get("explanation", ""),
            "judge_ok":     r.get("judge_ok", False),
            "model":        r.get("model", ""),
            "from_cache":   r.get("from_cache", False),
        }
        for r in results
    ])
    csv_bytes = df.to_csv(index=False).encode()
    json_bytes = json.dumps(results, ensure_ascii=False, indent=2).encode()

    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download CSV",  csv_bytes,  "judge_results.csv",  "text/csv")
    c2.download_button("⬇ Download JSON", json_bytes, "judge_results.json", "application/json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🎯 Judge Validator")
    st.markdown(
        "Evaluate the **privacy leakage evaluator** against ground-truth labels from "
        "SynthPAI, HR-VISPR, and OpenPII."
    )

    datasets_on_disk = available_datasets()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        if not datasets_on_disk:
            st.error("No supported datasets found on disk.")
            st.stop()

        selected_datasets = st.multiselect(
            "Datasets",
            datasets_on_disk,
            default=datasets_on_disk[:1],
            key="jv_datasets",
        )

        # Derive available attributes from selected datasets
        all_attrs = []
        for ds in selected_datasets:
            for a in DATASET_ATTRS.get(ds, []):
                if a not in all_attrs:
                    all_attrs.append(a)

        attr_filter = st.multiselect(
            "Attribute filter (empty = all)",
            all_attrs,
            key="jv_attr_filter_sidebar",
        )

        diff_filter = st.multiselect(
            "Difficulty filter (empty = all)",
            ["explicit", "implicit", "none"],
            key="jv_diff_filter_sidebar",
        )

        max_items = st.number_input(
            "Max items per dataset",
            min_value=1,
            max_value=5000,
            value=50,
            step=10,
            key="jv_max_items",
        )

        max_samples = st.number_input(
            "Max samples total (after attribute expansion)",
            min_value=1,
            max_value=50000,
            value=500,
            step=50,
            key="jv_max_samples",
        )

        model = st.text_input("Model", value=EVAL_MODEL, key="jv_model")
        if "HR-VISPR" in selected_datasets:
            hrvispr_model = _vision_model_for_family(model)
            if hrvispr_model != model:
                st.caption(f"HR-VISPR will use vision model: `{hrvispr_model}`")
            else:
                st.caption(f"HR-VISPR will use selected vision-capable model: `{hrvispr_model}`")

        workers = st.slider("Parallel workers", 1, 8, 4, key="jv_workers")
        use_cache = st.toggle("Use cache", value=True, key="jv_use_cache",
                              help="Off: re-runs all samples and overwrites cached results")

        api_key_input = st.text_input(
            "OpenRouter API key (leave blank to use env var)",
            type="password",
            key="jv_api_key",
        )
        api_key = api_key_input.strip() or get_openrouter_api_key()

        st.divider()
        run_btn = st.button("▶ Run evaluation", type="primary", width="stretch")
        clear_btn = st.button("🗑 Clear results", width="stretch")

        if clear_btn:
            for key in ("jv_results", "jv_samples_loaded", "jv_run_info"):
                st.session_state.pop(key, None)
            st.rerun()

    # ── Run evaluation ────────────────────────────────────────────────────────
    if run_btn:
        if not selected_datasets:
            st.warning("Select at least one dataset.")
            st.stop()
        if not api_key:
            st.warning("No OpenRouter API key found. Set OPENROUTER_API_KEY or enter it above.")
            st.stop()

        all_samples: List[Dict[str, Any]] = []
        with st.spinner("Loading samples…"):
            for ds in selected_datasets:
                ds_samples = load_judge_samples(
                    ds,
                    attribute_filter=attr_filter or None,
                    difficulty_filter=diff_filter or None,
                    max_items=int(max_items),
                    max_samples=int(max_samples),
                )
                all_samples.extend(ds_samples)

        if not all_samples:
            st.warning("No samples loaded. Try relaxing the filters or increasing max items.")
            st.stop()

        st.info(f"Loaded **{len(all_samples)}** samples. Starting evaluation…")
        pbar = st.progress(0.0, text="Starting…")
        t0   = time.time()

        results = run_evaluation(all_samples, model, api_key, workers=int(workers), progress_bar=pbar, use_cache=use_cache)
        elapsed = time.time() - t0

        pbar.empty()
        run_info = {
            "datasets": selected_datasets,
            "n_samples": len(results),
            "model":     model,
            "effective_models": sorted({r.get("model", model) for r in results}),
            "elapsed":   elapsed,
            "evaluator": _EVALUATOR_VERSION,
        }
        run_dir = _save_run_results(results, run_info)
        if run_dir is not None:
            run_info["run_dir"] = str(run_dir)

        st.session_state["jv_results"]   = results
        st.session_state["jv_run_info"]  = run_info
        st.rerun()

    # ── Display results ───────────────────────────────────────────────────────
    results  = st.session_state.get("jv_results")
    run_info = st.session_state.get("jv_run_info", {})

    if results is None:
        st.info("Configure options in the sidebar and click **▶ Run evaluation**.")
        return

    # Run info bar
    n_cached   = sum(1 for r in results if r.get("from_cache"))
    n_failed   = sum(1 for r in results if not r.get("judge_ok", True))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples",    run_info.get("n_samples", len(results)))
    col2.metric("Cached",     n_cached)
    col3.metric("Failed",     n_failed)
    col4.metric("Model",      run_info.get("model", "—"))

    successful = [r for r in results if r.get("judge_ok", True)]

    if not successful:
        st.error("All evaluations failed. Check your API key and model name.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Summary Metrics")
    _render_summary(successful)

    # ── Per-attribute metrics table ───────────────────────────────────────────
    with st.expander("Per-attribute metrics", expanded=False):
        by_attr = compute_metrics_by_group(successful, "attribute")
        rows = []
        for attr, m in sorted(by_attr.items()):
            rows.append({
                "attribute":            attr,
                "precision_confirmed":  f"{m['precision_confirmed']:.1%}" if m["precision_confirmed"] is not None else "N/A",
                "coverage_confirmed":   f"{m['coverage_confirmed']:.1%}",
                "ambiguity_rate":       f"{m['ambiguity_rate']:.1%}",
                "recall_lower_bound":   f"{m['recall_lower_bound']:.1%}" if m["recall_lower_bound"] is not None else "N/A",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("attribute"), width="content")

    # ── Distribution plots ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Distribution")
    _render_distribution(successful)

    # ── Sample viewer ─────────────────────────────────────────────────────────
    st.divider()
    _render_sample_viewer(successful)

    # ── Error analysis ────────────────────────────────────────────────────────
    st.divider()
    _render_error_analysis(successful)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export")
    _export_button(results)

if __name__ == "__main__":
    main()
