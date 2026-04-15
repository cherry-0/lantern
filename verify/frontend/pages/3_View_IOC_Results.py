"""
View Input-Output Comparison Results — browse cached IOC runs.

Loads items directly from an IOC cache directory
(verify/outputs/cache_<hash>/ where run_config.json contains
perturbation_method == "ioc_comparison") and renders the same
stage-wise view as the live Input-Output Comparison page.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT  = Path(__file__).resolve().parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st

st.set_page_config(
    page_title="View IOC Results — Verify",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _display_image(b64_str: str | None, caption: str = ""):
    try:
        if b64_str:
            import base64, io
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(base64.b64decode(b64_str)))
            st.image(img, caption=caption, width="stretch")
        else:
            st.info("Image not available in cache.")
    except Exception as e:
        st.error(f"Could not display image: {e}")


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
        f'{cell(isinstance(ext_eval.get(attr),   dict) and bool(ext_eval[attr].get("inferable")))}'
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


def _reasoning_expander(
    output_eval:   Dict[str, Any],
    ext_eval:      Dict[str, Any],
    unified_attrs: List[str],
):
    with st.expander("Reasoning details", expanded=False):
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
                    icon   = "🔴" if entry.get("inferable") else "🟢"
                    reason = entry.get("reasoning", "—")
                    st.markdown(f'{icon} <span style="font-size:0.9em"><b>{attr}</b></span>',
                                unsafe_allow_html=True)
                    st.caption(reason)


def _render_item(result: Dict[str, Any], unified_attrs: List[str], idx: int):
    filename    = result.get("filename", "unknown")
    status      = result.get("status", "")
    modality    = result.get("modality", "")
    input_item  = result.get("input_item", {})
    input_labels = result.get("input_labels", {})

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
            return

        col_in, col_out = st.columns([1, 2])

        with col_in:
            st.markdown("**Input**")
            if modality == "image":
                _display_image(input_item.get("image_base64"), caption=filename)
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
                st.text_area(
                    "Externalized text",
                    value=ext_text if ext_text.strip() else "No externalizations captured.",
                    height=200, disabled=True, label_visibility="collapsed",
                    key=f"vioc_ext_{idx}",
                )

        st.divider()
        st.markdown("**Stage-wise attribute presence**")

        if result.get("output_eval_error"):
            st.warning(f"Output evaluation error: {result['output_eval_error']}")
        if result.get("ext_eval_error"):
            st.warning(f"Externalization evaluation error: {result['ext_eval_error']}")

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
        )


def _render_aggregated(all_results: List[Dict[str, Any]], unified_attrs: List[str]):
    import pandas as pd
    import altair as alt

    success = [r for r in all_results if r.get("status") == "success"]
    if not success:
        st.info("No successful items to aggregate.")
        return

    n = len(success)
    rows = []
    for attr in unified_attrs:
        in_rate  = sum(r.get("input_labels",  {}).get(attr, 0) for r in success) / n
        out_rate = sum(
            1 if (isinstance(r.get("output_eval", {}).get(attr), dict)
                  and r["output_eval"][attr].get("inferable")) else 0
            for r in success
        ) / n
        ext_rate = sum(
            1 if (isinstance(r.get("ext_eval", {}).get(attr), dict)
                  and r["ext_eval"][attr].get("inferable")) else 0
            for r in success
        ) / n
        rows += [
            {"Attribute": attr, "Stage": STAGE_INPUT,  "Positive Rate": in_rate},
            {"Attribute": attr, "Stage": STAGE_OUTPUT, "Positive Rate": out_rate},
            {"Attribute": attr, "Stage": STAGE_EXT,    "Positive Rate": ext_rate},
        ]

    df    = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Attribute:N", sort=unified_attrs,
                    axis=alt.Axis(labelAngle=-40, labelFontSize=10)),
            xOffset=alt.XOffset("Stage:N", sort=STAGES),
            y=alt.Y("Positive Rate:Q", scale=alt.Scale(domain=[0, 1]),
                    title="Positive rate"),
            color=alt.Color(
                "Stage:N", sort=STAGES,
                scale=alt.Scale(domain=STAGES,
                                range=[STAGE_COLORS[s] for s in STAGES]),
            ),
            tooltip=["Attribute", "Stage",
                     alt.Tooltip("Positive Rate:Q", format=".2f")],
        )
        .properties(height=280,
                    title=f"Attribute-wise positive rate across {n} item(s)")
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Blue = Input annotation positive rate · "
        "Red = Raw output inferability rate · "
        "Amber = Externalized result inferability rate"
    )


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

        load_clicked = st.button("📂 Load", type="primary", width="stretch",
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
    for idx, result in enumerate(items):
        _render_item(result, unified_attrs, idx)

    # ── Delete ────────────────────────────────────────────────────────────────
    if cache_dir_str:
        st.divider()
        _render_delete_section(Path(cache_dir_str))


if __name__ == "__main__":
    main()
