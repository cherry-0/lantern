"""
Verbose inference logger for the Verify framework.

Activated by setting the environment variable VERBOSE=true (case-insensitive)
before launching the app:

    VERBOSE=true streamlit run verify/frontend/app.py

Output is written to stdout so it appears in the terminal running Streamlit.
The format is an AppWorld-style agent log: each inference call is wrapped in a
bordered block that shows input, inference module, and output at a glance.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, Optional

# ── Width of the log block ────────────────────────────────────────────────────
_W = 70
_BAR = "━" * _W
_SEP = "─" * _W

# Column widths
_TAG_W = 8   # "INPUT   " / "MODULE  " / "OUTPUT  "


def is_verbose() -> bool:
    """Return True if VERBOSE env var is set to a truthy value."""
    return os.environ.get("VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def _emit(line: str = "") -> None:
    """Write a line directly to stdout (bypasses Streamlit capture)."""
    print(line, flush=True)


def _wrap(text: str, indent: int) -> list[str]:
    """Wrap text to fit inside the log block with a given indent."""
    available = _W - indent
    return textwrap.wrap(text, width=max(available, 20)) or [""]


def _fmt_input(modality: str, input_item: Dict[str, Any]) -> list[str]:
    """
    Format the input field of the log block.
    - image  → [image] <path>  (or <filename> if path unavailable)
    - text   → first ~200 chars of the text, wrapped
    - video  → [video] <path>  (<N> frames)
    """
    if modality == "image":
        path = input_item.get("path") or input_item.get("filename", "<unknown>")
        return [f"[image] {path}"]
    elif modality == "text":
        raw = (
            input_item.get("text_content")
            or input_item.get("data")
            or input_item.get("filename", "")
        )
        if not isinstance(raw, str):
            raw = str(raw)
        preview = raw[:300].replace("\n", " ").strip()
        return _wrap(f"[text] {preview}", indent=_TAG_W + 2)
    elif modality == "video":
        path = input_item.get("path") or input_item.get("filename", "<unknown>")
        n_frames = len(input_item.get("frames", []))
        return [f"[video] {path}  ({n_frames} frames)"]
    else:
        return [f"[{modality}] {input_item.get('filename', '<unknown>')}"]


def _fmt_output(output_text: str) -> list[str]:
    """Format the output field: up to 400 chars, wrapped."""
    preview = (output_text or "").strip()[:400]
    if not preview:
        return ["<empty>"]
    lines = []
    for raw_line in preview.split("\n"):
        lines.extend(_wrap(raw_line, indent=_TAG_W + 2) if raw_line.strip() else [""])
    return lines or ["<empty>"]


def _row(tag: str, lines: list[str]) -> None:
    """Print a labeled row. First sub-line gets the tag; subsequent lines are indented."""
    pad = " " * (_TAG_W + 2)
    for i, line in enumerate(lines):
        if i == 0:
            label = f"{tag:<{_TAG_W}}  "
            _emit(f"  {label}{line}")
        else:
            _emit(f"  {pad}{line}")


def log_setup_start(app_name: str, python: str) -> None:
    """
    Always-on banner printed when one-time conda env setup begins.

    Example:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚙ SETUP   skin-disease-detection   (one-time)
    ──────────────────────────────────────────────────────────────────────
      Creating conda env with Python 3.10 ...
    """
    _emit()
    _emit(_BAR)
    _emit(f"⚙ SETUP   {app_name}   (one-time)")
    _emit(_SEP)
    _row("ENV", [f"Creating conda env  python={python}  ..."])


def log_setup_step(cmd: list[str]) -> None:
    """Print a single install step inside an open setup block."""
    label = " ".join(cmd[:4])          # e.g. "pip install tflite-runtime pillow"
    _row("INSTALL", [f"{label}  ..."])


def log_setup_done(ok: bool, error: str = "") -> None:
    """
    Always-on footer that closes the setup block.

    Example (success):
      STATUS    ✔  Done — env ready.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    if ok:
        _row("STATUS", ["✔  Done — env ready."])
    else:
        _row("STATUS", [f"✘  FAILED"])
        if error:
            _row("ERROR", _wrap(error[:400], indent=_TAG_W + 2))
    _emit(_BAR)


def log_availability(
    app_name: str,
    available: bool,
    message: str,
    traceback_str: str = "",
) -> None:
    """
    Always-on (not VERBOSE-gated) log block printed when an adapter is checked.
    Shows OK/FAIL status, the reason message, and — on failure — the full
    traceback from the native import attempt so config problems are obvious.

    Example (failure):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✦ AVAILABILITY   snapdo
    ──────────────────────────────────────────────────────────────────────
      STATUS    FAIL
      REASON    Native snapdo pipeline unavailable (No module named
                'django'); using OpenRouter fallback.
      TRACE     Traceback (most recent call last):
                  File ".../adapters/snapdo.py", line 74, in _check_native
                    import django
                ModuleNotFoundError: No module named 'django'
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    icon = "✔" if available else "✘"
    status_word = "OK  " if available else "FAIL"

    _emit()
    _emit(_BAR)
    _emit(f"{'✦'} AVAILABILITY   {app_name}")
    _emit(_SEP)
    _row("STATUS", [f"{icon}  {status_word}"])
    _row("REASON", _wrap(message, indent=_TAG_W + 2))
    if not available and traceback_str.strip():
        tb_lines = traceback_str.strip().splitlines()
        _row("TRACE", tb_lines[:1])
        pad = " " * (_TAG_W + 2)
        for line in tb_lines[1:]:
            _emit(f"  {pad}{line}")
    _emit(_BAR)


def log_inference(
    *,
    stage: str,                      # "original" | "perturbed"
    app_name: str,
    filename: str,
    modality: str,
    input_item: Dict[str, Any],
    result: Any,                     # AdapterResult or None
) -> None:
    """
    Emit one inference log block to stdout.

    Example output:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ▶ INFERENCE   snapdo :: original   —   img001.jpg
    ──────────────────────────────────────────────────────────────────────
      INPUT     [image] /datasets/hr-vipr/img001.jpg
      MODULE    native_vlmservice
      OUTPUT    Task: Go for a morning run at the park
                Verdict: PASSED  |  Confidence: 0.87
                Explanation: The photo shows a running track...
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    if not is_verbose():
        return

    method = "—"
    output_text = ""
    ok = False
    error_msg: Optional[str] = None

    if result is not None:
        ok = getattr(result, "success", False)
        output_text = getattr(result, "output_text", "") or ""
        meta = getattr(result, "metadata", {}) or {}
        method = meta.get("method", "unknown")
        if not ok:
            error_msg = getattr(result, "error", None) or "unknown error"

    header = f"▶ INFERENCE   {app_name} :: {stage}   —   {filename}"

    _emit()
    _emit(_BAR)
    _emit(header)
    _emit(_SEP)
    _row("INPUT",  _fmt_input(modality, input_item))
    _row("MODULE", [method])
    if ok:
        _row("OUTPUT", _fmt_output(output_text))
    else:
        _row("STATUS", [f"FAILED — {error_msg}"])
    _emit(_BAR)
