"""
Re-evaluate — Re-run inferability evaluation with a different model.

Lists output run groups, shows which eval model produced their current
ext_eval scores, and lets you re-evaluate all directories in selected groups with any
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
from verify.backend.utils.cache import normalize_eval_prompt
from verify.backend.utils.config import EVAL_PROMPT_CHOICES, get_default_eval_prompt


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
    Scan all output directories and aggregate them into run groups keyed by:
      (app_name, dataset_name, modality, perturbation_method)

    Returns list of dicts with:
      dirs, dir_count, app_name, dataset_name, modality, method,
      total, success, failed, eval_model, eval_prompt, last_reeval, has_summary
    """
    if not _OUTPUTS_DIR.exists():
        return []

    def _merge_item_state(
        existing: Optional[tuple[str, bool]],
        incoming_status: str,
        incoming_stale: bool,
    ) -> tuple[str, bool]:
        if existing is None:
            return incoming_status, incoming_stale
        status_rank = {"success": 3, "failed": 2, "error": 1}
        existing_status, existing_stale = existing
        if status_rank.get(incoming_status, 0) > status_rank.get(existing_status, 0):
            return incoming_status, incoming_stale
        if incoming_status == existing_status == "success":
            return "success", existing_stale or incoming_stale
        return existing_status, existing_stale

    groups: Dict[tuple, Dict] = {}

    for d in sorted(_OUTPUTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            app = cfg.get("app_name", "").strip()
            dataset = cfg.get("dataset_name", "").strip()
            if not app or not dataset:
                continue
            modality = cfg.get("modality", "").strip()
            method = cfg.get("perturbation_method", "") or "ioc"
            expected_eval_prompt = normalize_eval_prompt(cfg.get("eval_prompt")) if method == "ioc_comparison" else None
        except Exception:
            continue

        key = (app, dataset, modality, method)
        group = groups.setdefault(
            key,
            {
                "dirs": [],
                "app_name": app,
                "dataset_name": dataset,
                "modality": modality,
                "method": method,
                "items": {},
                "summary_total": 0,
                "summary_success": 0,
                "summary_failed": 0,
                "has_summary": False,
                "eval_models": set(),
                "eval_prompts": set(),
                "last_reeval": "",
            },
        )
        group["dirs"].append(str(d))

        summary: Dict = {}
        summary_path = d / _DIR_SUMMARY
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                pass
        if summary:
            group["has_summary"] = True
            group["summary_total"] += int(summary.get("total", 0) or 0)
            group["summary_success"] += int(summary.get("success", 0) or 0)
            group["summary_failed"] += int(summary.get("failed", 0) or 0)
            if summary.get("eval_model"):
                group["eval_models"].add(str(summary["eval_model"]))
            if summary.get("eval_prompt"):
                group["eval_prompts"].add(str(summary["eval_prompt"]))
            last_reeval = str(summary.get("last_reeval", "") or "")
            if last_reeval and last_reeval > group["last_reeval"]:
                group["last_reeval"] = last_reeval

        saw_item_json = False
        for f in d.iterdir():
            if f.suffix != ".json" or f.name in ("run_config.json", _DIR_SUMMARY):
                continue
            saw_item_json = True
            try:
                data = json.loads(f.read_text())
                if expected_eval_prompt is not None:
                    item_prompt = normalize_eval_prompt(data.get("eval_prompt"))
                    if item_prompt != expected_eval_prompt:
                        continue
                item_name = str(data.get("filename") or f.stem)
                group["items"][item_name] = _merge_item_state(
                    group["items"].get(item_name),
                    str(data.get("status", "")),
                    bool(data.get("ext_eval_stale", False)),
                )
            except Exception:
                item_name = f.stem
                group["items"][item_name] = _merge_item_state(
                    group["items"].get(item_name),
                    "error",
                    False,
                )

        if not saw_item_json and not summary:
            group["items"][d.name] = _merge_item_state(
                group["items"].get(d.name),
                "error",
                False,
            )

    rows: List[Dict] = []
    for group in groups.values():
        if group["items"]:
            total = len(group["items"])
            success = sum(1 for st, _ in group["items"].values() if st == "success")
            failed = sum(1 for st, _ in group["items"].values() if st == "failed")
        else:
            total = group["summary_total"]
            success = group["summary_success"]
            failed = group["summary_failed"]

        eval_models = sorted(group["eval_models"])
        eval_prompts = sorted(group["eval_prompts"])
        rows.append(
            {
                "dirs": group["dirs"],
                "dir_count": len(group["dirs"]),
                "app_name": group["app_name"],
                "dataset_name": group["dataset_name"],
                "modality": group["modality"],
                "method": group["method"],
                "total": total,
                "success": success,
                "failed": failed,
                "eval_model": eval_models[0] if len(eval_models) == 1 else ("mixed" if eval_models else ""),
                "eval_prompt": eval_prompts[0] if len(eval_prompts) == 1 else ("mixed" if eval_prompts else ""),
                "last_reeval": group["last_reeval"],
                "has_summary": group["has_summary"],
            }
        )
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
        "Rows are aggregated run groups keyed by app / dataset / modality / method.  \n"
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

        prompt_mode = st.radio(
            "Evaluation prompt",
            list(EVAL_PROMPT_CHOICES),
            index=list(EVAL_PROMPT_CHOICES).index(get_default_eval_prompt()),
            disabled=running,
            help=(
                "prompt1: binary inferability. "
                "prompt2: binary inferability + prediction. "
                "prompt3: aggregate externalized result + per-channel threat. "
                "prompt4: prompt3 reasoning with confirmed/possible/no-evidence leakage verdicts."
            ),
        )

        st.divider()
        workers = st.slider("Parallel workers", 1, 8, 4, disabled=running,
                            help="Concurrent API calls per directory")
        dir_workers = st.slider(
            "Directory workers",
            1,
            8,
            1,
            disabled=running,
            help="Number of output directories to process concurrently",
        )
        verbose = st.toggle("Verbose output", value=False, disabled=running)
        dry_run = st.toggle("Dry run (no writes)", value=False, disabled=running)
        st.divider()

        if st.button("🔄 Refresh table", width="stretch"):
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

    # Separate into IOC and Perturbation sections
    ioc_dirs = [d for d in dirs if d["method"] == "ioc_comparison"]
    perturb_dirs = [d for d in dirs if d["method"] != "ioc_comparison"]

    if not ioc_dirs and not perturb_dirs:
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
            width="stretch",
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

    selected_dirs: List[str] = []

    def _render_section(section_dirs: List[Dict], section_title: str) -> None:
        """Render a section of directory groups."""
        nonlocal selected_dirs

        if not section_dirs:
            return

        st.subheader(section_title)

        sa_col, da_col, _, filt_col = st.columns([1, 1, 2, 3])
        section_key = section_title.lower().replace(" ", "_")

        if sa_col.button("Select all", key=f"sa_{section_key}", disabled=running, width="stretch"):
            for d in section_dirs:
                st.session_state[f"reeval_row_{d['_orig_idx']}"] = True
            st.rerun()
        if da_col.button("Deselect all", key=f"da_{section_key}", disabled=running, width="stretch"):
            for d in section_dirs:
                st.session_state[f"reeval_row_{d['_orig_idx']}"] = False
            st.rerun()

        # Optional app/dataset filter
        with filt_col:
            search = st.text_input("Filter rows", key=f"filt_{section_key}",
                                   placeholder="app or dataset substring",
                                   label_visibility="collapsed")

        # Header
        hdr_cols = st.columns([0.4, 2, 1.1, 2, 1, 0.8, 1, 1.2, 2.7, 1.6])
        for col, label in zip(hdr_cols, ["", "App", "Modality", "Dataset", "Method",
                                          "Dirs", "Items", "Prompt", "Eval Model", "Last Re-eval"]):
            col.markdown(f"**{label}**")
        st.divider()

        for row in section_dirs:
            orig_idx = row["_orig_idx"]

            # Text filter
            if search:
                haystack = f"{row['app_name']} {row['dataset_name']}".lower()
                if search.lower() not in haystack:
                    continue

            cols = st.columns([0.4, 2, 1.1, 2, 1, 0.8, 1, 1.2, 2.7, 1.6])

            checked = cols[0].checkbox(
                "select",
                value=st.session_state.get(f"reeval_row_{orig_idx}", False),
                key=f"reeval_row_{orig_idx}",
                label_visibility="collapsed",
                disabled=running,
            )
            if checked:
                selected_dirs.extend(row["dirs"])

            # Display row data
            cols[1].markdown(f"`{row['app_name']}`")
            cols[2].markdown(f"`{row['modality']}`")
            cols[3].markdown(row['dataset_name'])
            cols[4].markdown(f"`{row['method']}`")
            cols[5].markdown(str(row['dir_count']))
            cols[6].markdown(f"{row['total']}")
            cols[7].markdown(f"`{row['eval_prompt'] or '—'}`")

            # Eval model with indicator
            model = row['eval_model']
            if not model:
                cols[8].markdown("—")
            elif model == model_choice or model == model:
                cols[8].markdown(f"🟩 `{model}`")
            else:
                cols[8].markdown(f"🟨 `{model}`")

            cols[9].markdown(row['last_reeval'] or "—")

        st.divider()

    # Add original index to each row for state management
    for i, d in enumerate(ioc_dirs):
        d["_orig_idx"] = i
    for i, d in enumerate(perturb_dirs):
        d["_orig_idx"] = len(ioc_dirs) + i

    # Render IOC section
    _render_section(ioc_dirs, "Input-Output Comparison")

    # Render Perturbation section
    _render_section(perturb_dirs, "Perturbation Analysis")

    n_sel = len(selected_dirs)
    n_selected_groups = sum(1 for i in range(len(dirs)) if st.session_state.get(f"reeval_row_{i}", False))
    st.caption(f"{n_selected_groups} of {len(dirs)} groups selected ({n_sel} directories)")

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
                width="stretch",
            )
            if run_clicked:
                cmd = [
                    sys.executable, "-u", str(_REEVAL_SCRIPT),
                    "--model", model,
                    "--workers", str(workers),
                    "--dir-workers", str(dir_workers),
                    "--dir",
                ] + selected_dirs
                if prompt_mode == "prompt1":
                    cmd.append("--prompt1")
                elif prompt_mode == "prompt2":
                    cmd.append("--prompt2")
                elif prompt_mode == "prompt3":
                    cmd.append("--prompt3")
                elif prompt_mode == "prompt4":
                    cmd.append("--prompt4")
                if verbose:  cmd.append("--verbose")
                if dry_run:  cmd.append("--dry-run")
                rs.update({"running": True, "log": [f"$ {' '.join(cmd)}", ""],
                           "pid": None, "returncode": None})
                threading.Thread(target=_run_subprocess, args=(cmd, rs), daemon=True).start()
                st.rerun()
        else:
            st.button("⏳ Running…", disabled=True, width="stretch")

    with stop_col:
        if running:
            if st.button("⏹ Stop", type="secondary", width="stretch"):
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
