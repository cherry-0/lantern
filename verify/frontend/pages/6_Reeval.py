"""
Re-evaluate — Re-run inferability evaluation with a different model.

Lists every output directory, shows which eval model produced its current
ext_eval scores, and lets you re-evaluate selected directories with any
OpenRouter-compatible model.

Workflow:
  1. Click "Initialize labels" once — stamps each cached result with the
     name of the default eval model (no API calls).
  2. Pick a new model in the sidebar.
  3. Select the directories you want to re-evaluate and click "Re-evaluate".
"""
from __future__ import annotations

import html as _html
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT  = Path(__file__).resolve().parent.parent.parent

if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st


_REEVAL_SCRIPT = VERIFY_ROOT / "reeval.py"
_OUTPUTS_DIR   = VERIFY_ROOT / "outputs"
_DIR_SUMMARY   = "dir_summary.json"

_SUGGESTED_MODELS = [
    "google/gemini-2.0-flash-001",         # current default eval model
    "google/gemini-2.5-flash-preview-05-20",
    "google/gemini-2.5-pro",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-sonnet-4-5",
]


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def _scan_dirs() -> List[Dict]:
    """
    Scan all output directories.  Returns list of dicts with:
      dir, dir_name, app_name, dataset_name, modality, method,
      total, success, failed, eval_model, last_reeval, has_summary
    """
    if not _OUTPUTS_DIR.exists():
        return []
    rows: List[Dict] = []
    for d in sorted(_OUTPUTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            app     = cfg.get("app_name",     "").strip()
            dataset = cfg.get("dataset_name", "").strip()
            if not app or not dataset:
                continue
        except Exception:
            continue

        summary: Dict = {}
        summary_path = d / _DIR_SUMMARY
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                pass

        rows.append({
            "dir":          str(d),
            "dir_name":     d.name,
            "app_name":     app,
            "dataset_name": dataset,
            "modality":     cfg.get("modality", ""),
            "method":       cfg.get("perturbation_method", "") or "ioc",
            "total":        summary.get("total",      0),
            "success":      summary.get("success",    0),
            "failed":       summary.get("failed",     0),
            "eval_model":   summary.get("eval_model", ""),
            "last_reeval":  summary.get("last_reeval",""),
            "has_summary":  bool(summary),
        })
    return rows


# ── Background subprocess ──────────────────────────────────────────────────────

def _run_subprocess(cmd: List[str], state: Dict) -> None:
    """Run *cmd* in a daemon thread; stream stdout into state["log"]."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        state["pid"] = proc.pid
        for line in proc.stdout:
            state["log"].append(line.rstrip())
        proc.wait()
        state["returncode"] = proc.returncode
    except Exception as exc:
        state["log"].append(f"[RUNNER ERROR] {exc}")
        state["returncode"] = -1
    finally:
        state["running"]       = False
        state["just_finished"] = True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🔬 Re-evaluate")
    st.markdown(
        "Re-run the inferability evaluator on already-cached output results "
        "using a different model, without re-executing any target app.  \n"
        "Run **Initialize labels** once first to record which model produced "
        "the current `ext_eval` scores."
    )

    # ── Session state ─────────────────────────────────────────────────────────
    if "reeval_state" not in st.session_state:
        st.session_state["reeval_state"] = {
            "running": False, "log": [], "pid": None,
            "returncode": None, "just_finished": False,
        }
    rs      = st.session_state["reeval_state"]
    running = rs["running"]

    if rs.get("just_finished"):
        rs["just_finished"] = False
        st.cache_data.clear()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        model_choice = st.selectbox(
            "Model", _SUGGESTED_MODELS,
            index=0,
            disabled=running,
            help="OpenRouter model ID for re-evaluation",
        )
        custom_model = st.text_input(
            "Custom model ID (overrides dropdown)",
            placeholder="e.g. meta-llama/llama-3.3-70b-instruct",
            disabled=running,
        )
        model: str = custom_model.strip() if custom_model.strip() else model_choice
        st.caption(f"Active model: `{model}`")

        st.divider()
        workers = st.slider("Parallel workers", 1, 8, 4, disabled=running,
                            help="Concurrent API calls per directory")
        verbose = st.toggle("Verbose output", value=False, disabled=running)
        dry_run = st.toggle("Dry run (no writes)", value=False, disabled=running)
        st.divider()

        if st.button("🔄 Refresh table", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption(
            "**Eval Model column**\n\n"
            "Shows the model that produced current `ext_eval` scores.  \n"
            "Blank = not yet labeled.  \n"
            "🟩 = matches active model.  \n"
            "🟨 = different model."
        )

    # ── Load directories ──────────────────────────────────────────────────────
    dirs = _scan_dirs()

    if not dirs:
        st.info("No output directories found. Run the batch pipeline first.")
        return

    # ── Initialize labels ─────────────────────────────────────────────────────
    unlabeled = sum(1 for d in dirs if not d["eval_model"])
    init_col, status_col = st.columns([3, 5])

    with init_col:
        init_tip = (
            "Stamp existing ext_eval results with the default model name. "
            "No API calls are made."
        )
        if st.button(
            f"🏷 Initialize labels  ({unlabeled} unlabeled dirs)",
            disabled=running or unlabeled == 0,
            use_container_width=True,
            help=init_tip,
        ):
            cmd = [sys.executable, "-u", str(_REEVAL_SCRIPT), "--init"]
            if verbose:  cmd.append("--verbose")
            if dry_run:  cmd.append("--dry-run")
            rs.update({"running": True, "log": [f"$ {' '.join(cmd)}", ""],
                       "pid": None, "returncode": None})
            threading.Thread(target=_run_subprocess, args=(cmd, rs), daemon=True).start()
            st.rerun()

    with status_col:
        if running:
            st.info("Running…", icon="🔄")
        elif rs["returncode"] is not None:
            rc = rs["returncode"]
            if rc == 0:
                st.success("Completed.", icon="✅")
            else:
                st.error(f"Exited with code {rc}.", icon="❌")

    st.divider()

    # ── Directory table ───────────────────────────────────────────────────────
    st.subheader("Output Directories")

    sa_col, da_col, _, filt_col = st.columns([1, 1, 2, 3])
    if sa_col.button("Select all",   disabled=running, use_container_width=True):
        for i in range(len(dirs)):
            st.session_state[f"reeval_row_{i}"] = True
    if da_col.button("Deselect all", disabled=running, use_container_width=True):
        for i in range(len(dirs)):
            st.session_state[f"reeval_row_{i}"] = False

    # Optional app/dataset filter
    with filt_col:
        search = st.text_input("Filter rows", placeholder="app or dataset substring",
                               label_visibility="collapsed")

    # Header
    hdr_cols = st.columns([0.4, 2, 1.2, 2, 1, 1, 3, 2])
    for col, label in zip(hdr_cols, ["", "App", "Modality", "Dataset", "Method",
                                      "Items", "Eval Model", "Last Re-eval"]):
        col.markdown(f"**{label}**")
    st.divider()

    selected_dirs: List[str] = []

    for i, row in enumerate(dirs):
        # Text filter
        if search:
            haystack = f"{row['app_name']} {row['dataset_name']}".lower()
            if search.lower() not in haystack:
                continue

        cols = st.columns([0.4, 2, 1.2, 2, 1, 1, 3, 2])

        checked = cols[0].checkbox(
            "select",
            value=st.session_state.get(f"reeval_row_{i}", False),
            key=f"reeval_row_{i}",
            label_visibility="collapsed",
            disabled=running,
        )
        if checked:
            selected_dirs.append(row["dir"])

        # Modality badge
        mod_color = "#4a90d9" if row["modality"] == "image" else "#e07b2a"
        mod_badge = (
            f'<span style="background:{mod_color};color:#fff;'
            f'padding:2px 8px;border-radius:10px;font-size:0.82em">'
            f'{row["modality"]}</span>'
        )

        # Eval model cell
        ev = row["eval_model"]
        if not ev:
            ev_cell = '<span style="color:#aaa;font-size:0.84em">not labeled</span>'
        elif ev == model:
            ev_cell = (
                f'<span style="color:#155724;font-weight:600;font-size:0.84em">'
                f'✓ {_html.escape(ev)}</span>'
            )
        else:
            ev_cell = (
                f'<span style="color:#856404;font-size:0.84em">'
                f'{_html.escape(ev)}</span>'
            )

        items_str  = f"{row['success']} / {row['total']}" if row["total"] else "—"
        last_re    = row["last_reeval"][:10] if row["last_reeval"] else "—"

        cols[1].write(row["app_name"])
        cols[2].markdown(mod_badge, unsafe_allow_html=True)
        cols[3].write(row["dataset_name"])
        cols[4].write(row["method"])
        cols[5].write(items_str)
        cols[6].markdown(ev_cell, unsafe_allow_html=True)
        cols[7].write(last_re)

    n_sel = len(selected_dirs)
    st.caption(f"{n_sel} of {len(dirs)} directories selected")

    # ── Re-evaluate / Stop ────────────────────────────────────────────────────
    st.divider()
    btn_col, stop_col, stat_col = st.columns([2.5, 1.5, 4])

    with btn_col:
        if not running:
            label = f"▶ Re-evaluate selected ({n_sel})"
            if dry_run:
                label += " — DRY RUN"
            run_clicked = st.button(
                label,
                type="primary",
                disabled=(n_sel == 0 or not model),
                use_container_width=True,
            )
            if run_clicked:
                cmd = [
                    sys.executable, "-u", str(_REEVAL_SCRIPT),
                    "--model", model,
                    "--workers", str(workers),
                    "--dir",
                ] + selected_dirs
                if verbose:  cmd.append("--verbose")
                if dry_run:  cmd.append("--dry-run")
                rs.update({"running": True, "log": [f"$ {' '.join(cmd)}", ""],
                           "pid": None, "returncode": None})
                threading.Thread(target=_run_subprocess, args=(cmd, rs), daemon=True).start()
                st.rerun()
        else:
            st.button("⏳ Running…", disabled=True, use_container_width=True)

    with stop_col:
        if running:
            if st.button("⏹ Stop", type="secondary", use_container_width=True):
                pid = rs.get("pid")
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass

    with stat_col:
        if running:
            done = sum(1 for line in rs["log"] if "success=" in line or "labeled=" in line)
            st.info(f"Running… ({done} dir(s) reported)", icon="🔄")
        elif rs["returncode"] is not None and rs["log"]:
            rc = rs["returncode"]
            if rc == 0:
                st.success("Re-evaluation complete.", icon="✅")
            else:
                st.error(f"Exited with code {rc}.", icon="❌")

    # ── Log output ────────────────────────────────────────────────────────────
    if rs["log"] or running:
        st.subheader("Output")
        log_content = _html.escape("\n".join(rs["log"][-400:]))
        st.components.v1.html(
            f"""<!DOCTYPE html>
<html>
<head>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: transparent; overflow: hidden; }}
#log {{
    height: 380px;
    overflow-y: auto;
    background: #0e1117;
    color: #d6d6d6;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 12.5px;
    line-height: 1.55;
    padding: 12px 16px;
    border-radius: 6px;
    white-space: pre-wrap;
    word-break: break-all;
}}
</style>
</head>
<body>
<div id="log">{log_content}</div>
<script>
  var el = document.getElementById("log");
  el.scrollTop = el.scrollHeight;
</script>
</body>
</html>""",
            height=400,
        )

        if not running and rs["log"]:
            if st.button("Clear log"):
                rs["log"]        = []
                rs["returncode"] = None
                st.rerun()

    # ── Auto-refresh while running ────────────────────────────────────────────
    if running:
        import time
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
