"""
Batch Runner — trigger run_batch.py from the Streamlit UI.

Reads batch_config.csv, shows each row as a selectable config, and streams
live log output from run_batch.py while it runs.
"""

from __future__ import annotations

import csv
import html as _html
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT  = Path(__file__).resolve().parent.parent.parent

if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st


_BATCH_SCRIPT = VERIFY_ROOT / "run_batch.py"
_BATCH_CONFIG = VERIFY_ROOT / "batch_config.csv"
_TEMP_CONFIG  = VERIFY_ROOT / "_batch_config_ui_temp.csv"


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Parse batch_config.csv, skipping comment and blank lines."""
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#"))
        )
        for row in reader:
            if not any(v.strip() for v in row.values()):
                continue
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def _write_temp_csv(rows: List[Dict[str, str]], path: Path) -> None:
    fieldnames = ["enabled", "app_name", "modality", "dataset_name",
                  "perturbation_method", "max_items"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ── Log parsing for per-task progress ────────────────────────────────────────

_RE_LOG = re.compile(r'^\[\d{2}:\d{2}:\d{2}\]\s+\w+\s+\[([^\]]+)\]\s+(.*)')


def _parse_progress_line(line: str, progress: Dict) -> None:
    """Update per-task progress dict from a single log line (called in thread)."""
    m = _RE_LOG.match(line)
    if not m:
        return
    tag = m.group(1)
    msg = m.group(2).strip()

    p = progress.get(tag)
    if p is None:
        return

    if msg.startswith("Starting"):
        p["status"] = "running"
    elif msg.startswith("Done —"):
        # "Done — success=N cached=N failed=N"
        for key in ("success", "cached", "failed"):
            m2 = re.search(rf'{key}=(\d+)', msg)
            if m2:
                p[key] = int(m2.group(1))
        p["done"]   = p["success"] + p["cached"] + p["failed"]
        p["status"] = "done"
    elif (
        msg.startswith("✓")           # IOC item success
        or "[success]" in msg         # perturb item (live or cached)
        or "[failed]"  in msg         # perturb item failed
        or msg.startswith("load error")
        or msg.startswith("pipeline error")
        or msg.startswith("pipeline failed")
    ):
        p["done"] += 1


# ── Background subprocess ─────────────────────────────────────────────────────

def _run_subprocess(cmd: List[str], temp_csv: Path, state: Dict) -> None:
    """Daemon thread: execute cmd and stream stdout into the shared state dict.

    Writes only to `state` (a plain dict held in session_state) — never
    touches st.session_state directly, which is not thread-safe.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(VERIFY_ROOT),
        )
        state["pid"] = proc.pid
        for line in proc.stdout:
            stripped = line.rstrip()
            state["log"].append(stripped)
            _parse_progress_line(stripped, state.get("progress", {}))
        proc.wait()
        state["returncode"] = proc.returncode
    except Exception as exc:
        state["log"].append(f"[RUNNER ERROR] {exc}")
        state["returncode"] = -1
    finally:
        state["running"]        = False
        state["just_finished"]  = True   # triggers one extra rerun to flush final log
        try:
            temp_csv.unlink(missing_ok=True)
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("⚡ Batch Runner")
    st.markdown("Select configs from `batch_config.csv` and run the batch evaluation pipeline.")

    all_rows = _load_csv_rows(_BATCH_CONFIG)

    if not all_rows:
        st.error(
            f"`{_BATCH_CONFIG}` not found or empty. "
            "Expected at `verify/batch_config.csv`."
        )
        return

    # `batch_state` is a plain dict mutated in-place by the background thread.
    # Never access st.session_state from the thread — pass this ref instead.
    if "batch_state" not in st.session_state:
        st.session_state["batch_state"] = {"running": False, "log": [],
                                            "pid": None, "returncode": None,
                                            "progress": {}}
    bs = st.session_state["batch_state"]
    running: bool = bs["running"]

    # ── Sidebar: run settings ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("Run Settings")
        mode = st.radio(
            "Mode", ["both", "ioc", "perturb"],
            horizontal=True,
            disabled=running,
            help="**both** runs IOC + perturbation analysis. **ioc** runs input-output comparison only. **perturb** runs perturbation analysis only.",
        )
        workers = st.slider(
            "Parallel workers", min_value=1, max_value=8, value=4,
            disabled=running,
        )
        max_items_val = st.number_input(
            "Max items per run (0 = all)", min_value=0, value=0, step=1,
            disabled=running,
            help="Global item cap. Row-level `max_items` in the CSV overrides this.",
        )
        use_cache = st.toggle("Use cache", value=True, disabled=running)
        dry_run   = st.toggle(
            "Dry run (print plan only)", value=False, disabled=running,
        )

        st.divider()

        # Select-all / deselect-all
        sa_col, da_col = st.columns(2)
        if sa_col.button("Select all", use_container_width=True, disabled=running):
            for i in range(len(all_rows)):
                st.session_state[f"batch_row_{i}"] = True
        if da_col.button("Deselect all", use_container_width=True, disabled=running):
            for i in range(len(all_rows)):
                st.session_state[f"batch_row_{i}"] = False

    # ── Config checklist ──────────────────────────────────────────────────────
    st.subheader("Config Checklist")

    hdr = st.columns([0.5, 2.5, 1.5, 2, 2.5, 1.5])
    for col, label in zip(hdr, ["", "App", "Modality", "Dataset", "Perturbation", "Max items"]):
        col.markdown(f"**{label}**")
    st.divider()

    selected_rows: List[Dict[str, str]] = []

    for i, row in enumerate(all_rows):
        default = row.get("enabled", "true").lower() not in ("false", "0", "no")
        cols = st.columns([0.5, 2.5, 1.5, 2, 2.5, 1.5])

        checked = cols[0].checkbox(
            "Include row",
            value=st.session_state.get(f"batch_row_{i}", default),
            key=f"batch_row_{i}",
            label_visibility="collapsed",
            disabled=running,
        )

        app_name = row.get("app_name", "")
        modality = row.get("modality", "")
        tag_color = "#4a90d9" if modality == "image" else "#e07b2a"
        mod_badge = (
            f'<span style="background:{tag_color};color:#fff;'
            f'padding:2px 8px;border-radius:10px;font-size:0.82em">'
            f'{modality}</span>'
        )

        cols[1].write(app_name)
        cols[2].markdown(mod_badge, unsafe_allow_html=True)
        cols[3].write(row.get("dataset_name", ""))
        cols[4].write(row.get("perturbation_method", "") or "—")
        cols[5].write(row.get("max_items", "") or "—")

        if checked:
            selected_rows.append({**row, "enabled": "true"})

    n_sel = len(selected_rows)
    st.caption(f"{n_sel} of {len(all_rows)} configs selected")

    # ── Run / Stop controls ───────────────────────────────────────────────────
    st.divider()
    btn_col, status_col = st.columns([2, 5])

    with btn_col:
        if not running:
            run_label = "▶ Dry Run" if dry_run else "▶ Run Batch"
            run_clicked = st.button(
                run_label, type="primary",
                disabled=(n_sel == 0),
                use_container_width=True,
            )
            if bs["log"]:
                if st.button("Clear log", use_container_width=True):
                    bs["log"] = []
                    bs["returncode"] = None
                    st.rerun()
        else:
            run_clicked = False
            if st.button("⏹ Stop", type="secondary", use_container_width=True):
                pid = bs.get("pid")
                if pid:
                    try:
                        import os
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass

    with status_col:
        if running:
            done = sum(1 for l in bs["log"] if "Done —" in l)
            st.info(f"Running…  ({done} task(s) finished so far)", icon="🔄")
        elif bs["returncode"] is not None:
            rc = bs["returncode"]
            if rc == 0:
                st.success("Batch completed successfully.", icon="✅")
            elif rc == -1:
                st.error("Batch runner encountered an internal error.", icon="❌")
            else:
                st.error(f"Batch exited with code {rc}.", icon="❌")

    # ── Launch ────────────────────────────────────────────────────────────────
    if run_clicked and n_sel > 0 and not running:
        _write_temp_csv(selected_rows, _TEMP_CONFIG)

        cmd: List[str] = [
            sys.executable, "-u", str(_BATCH_SCRIPT),
            "--config", str(_TEMP_CONFIG),
            "--mode", mode,
            "--workers", str(workers),
        ]
        if max_items_val > 0:
            cmd += ["--max-items", str(int(max_items_val))]
        if not use_cache:
            cmd.append("--no-cache")
        if dry_run:
            cmd.append("--dry-run")

        # Pre-populate per-task progress (pending) so the UI can render
        # progress bars immediately, before the subprocess emits any logs.
        from verify.backend.datasets.loader import count_dataset_items, detect_modality as _detect_modality

        global_max = int(max_items_val) if max_items_val > 0 else None
        task_progress: Dict[str, Dict] = {}
        modes_to_run = (["ioc", "perturb"] if mode == "both"
                        else [mode])
        for task_mode in modes_to_run:
            for row in selected_rows:
                tag = f"{task_mode}/{row['app_name']}/{row['dataset_name']}"
                row_max_str = row.get("max_items", "")
                total: Optional[int] = (
                    int(row_max_str) if row_max_str
                    else global_max
                )
                # If total is still unknown, count the dataset directly.
                # count_dataset_items is fast (reads metadata files, doesn't load data).
                if total is None:
                    try:
                        modality = row.get("modality") or _detect_modality(row["dataset_name"]) or "text"
                        n = count_dataset_items(row["dataset_name"], modality)
                        if n > 0:
                            total = n
                    except Exception:
                        pass
                task_progress[tag] = {
                    "status":   "pending",
                    "done":     0,
                    "total":    total,
                    "success":  0,
                    "failed":   0,
                    "cached":   0,
                    "app":      row["app_name"],
                    "dataset":  row["dataset_name"],
                    "modality": row["modality"],
                    "mode":     task_mode,
                }

        bs["running"]    = True
        bs["log"]        = [f"$ {' '.join(cmd)}", ""]
        bs["pid"]        = None
        bs["returncode"] = None
        bs["progress"]   = task_progress

        threading.Thread(
            target=_run_subprocess,
            args=(cmd, _TEMP_CONFIG, bs),
            daemon=True,
        ).start()
        st.rerun()

    # ── Per-task progress ─────────────────────────────────────────────────────
    task_progress: Dict = bs.get("progress", {})
    if task_progress:
        st.divider()
        st.subheader("Progress")

        for tag, p in task_progress.items():
            status  = p["status"]
            done    = p["done"]
            total   = p["total"]
            app     = p["app"]
            dataset = p["dataset"]
            mode_lbl = "IOC" if p["mode"] == "ioc" else "Perturb"

            if status == "pending":
                status_icon = "⏳"
            elif status == "running":
                status_icon = "🔄"
            elif status == "done":
                status_icon = "❌" if p["failed"] > 0 else "✅"
            else:
                status_icon = "❓"

            label_md = (
                f"{status_icon} &nbsp; **[{mode_lbl}]** &nbsp; "
                f"`{app}` / `{dataset}`"
            )

            if status == "pending":
                bar_text = "Waiting…"
                fraction = 0.0
            elif total and total > 0:
                fraction = min(done / total, 1.0)
                s = p["success"]
                c = p["cached"]
                f_cnt = p["failed"]
                if status == "done":
                    bar_text = (
                        f"{done} / {total}  ·  "
                        f"{s} new  {c} cached  {f_cnt} failed"
                    )
                else:
                    bar_text = f"{done} / {total}"
            else:
                fraction = 0.0 if status == "pending" else None
                s = p["success"]
                c = p["cached"]
                f_cnt = p["failed"]
                if status == "done":
                    bar_text = (
                        f"{done} processed  ·  "
                        f"{s} new  {c} cached  {f_cnt} failed"
                    )
                else:
                    bar_text = f"{done} processed" if done > 0 else "Starting…"

            st.markdown(label_md, unsafe_allow_html=True)

            if fraction is not None:
                st.progress(fraction, text=bar_text)
            else:
                # Total unknown — use full bar while running, show text only
                st.progress(
                    1.0 if status == "done" else 0.0,
                    text=bar_text,
                )

    # ── Live log ──────────────────────────────────────────────────────────────
    log_lines: List[str] = bs["log"]
    if log_lines or running:
        st.subheader("Log Output")
        log_content = _html.escape("\n".join(log_lines))
        st.components.v1.html(
            f"""<!DOCTYPE html>
<html>
<head>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: transparent; overflow: hidden; }}
#log {{
    height: 440px;
    overflow-y: scroll;
    background: #0e1117;
    color: #e6eaf0;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.82em;
    line-height: 1.55;
    padding: 12px 14px;
    border-radius: 6px;
    border: 1px solid #2e3440;
    white-space: pre-wrap;
    word-break: break-all;
}}
</style>
</head>
<body>
<div id="log">{log_content}</div>
<script>
    const el = document.getElementById('log');
    el.scrollTop = el.scrollHeight;
</script>
</body>
</html>""",
            height=460,
            scrolling=False,
        )

        if running:
            time.sleep(0.8)
            st.rerun()
        elif bs.pop("just_finished", False):
            # One extra rerun after the process exits to flush the final log lines.
            time.sleep(0.2)
            st.rerun()


if __name__ == "__main__":
    main()
