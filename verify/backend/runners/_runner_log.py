"""
Minimal logging helpers for Verify runner scripts.

Runners execute inside conda envs and cannot import from the main verify
package, so this file is a self-contained copy of the relevant formatting
logic from verbose_log.py.  It must only use the Python standard library.

All output goes to sys.stderr so it is relayed to the terminal in real time
by CondaRunner.run() without interfering with the JSON result on stdout.
"""
from __future__ import annotations

import sys
import textwrap

_W = 70
_BAR = "━" * _W
_SEP = "─" * _W
_TAG_W = 8


def _row(tag: str, lines: list[str]) -> None:
    pad = " " * (_TAG_W + 2)
    for i, line in enumerate(lines):
        if i == 0:
            print(f"  {tag:<{_TAG_W}}  {line}", file=sys.stderr)
        else:
            print(f"  {pad}{line}", file=sys.stderr)


def log_input(app: str, modality: str, content: str) -> None:
    """
    Print a bordered input block to stderr.

    For image modality, `content` should be the file path (or "<base64>" if
    no path is available).  For text modality, `content` is the raw text,
    truncated and wrapped to fit the block.

    Example (text):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ▶ INPUT   xend
    ──────────────────────────────────────────────────────────────────────
      TYPE      text
      CONTENT   Hello everyone, I hope this email finds you well. I am
                writing to share a brief update on some of the work...
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Example (image):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ▶ INPUT   snapdo
    ──────────────────────────────────────────────────────────────────────
      TYPE      image
      PATH      /datasets/VISPR/val2017/2017_00123.jpg
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    label = "PATH" if modality == "image" else "CONTENT"

    # For text: truncate to 300 chars and wrap; for image: show path as-is
    if modality == "text":
        preview = content.replace("\n", " ").strip()[:300]
        content_lines = textwrap.wrap(preview, width=_W - _TAG_W - 2) or ["<empty>"]
    else:
        content_lines = [content or "<base64>"]

    print(_BAR, file=sys.stderr)
    print(f"▶ INPUT   {app}", file=sys.stderr)
    print(_SEP, file=sys.stderr)
    _row("TYPE", [modality])
    _row(label, content_lines)
    print(_BAR, file=sys.stderr, flush=True)


def log_output(app: str, task: str, content: str, extra: dict | None = None) -> None:
    """
    Print a bordered output block to stderr.

    app:     adapter name (e.g. "tool-neuron")
    task:    pipeline stage (e.g. "text_generation", "expense_parsing")
    content: main result text — truncated and wrapped to fit the block
    extra:   optional {label: value} pairs appended as extra rows

    Example:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ◀ OUTPUT  tool-neuron
    ──────────────────────────────────────────────────────────────────────
      TASK      text_generation
      RESULT    Sure! The capital of France is Paris, which has been the
                country's capital since the 10th century...
      TOKENS    87
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    preview = content.replace("\n", " ").strip()[:300]
    content_lines = textwrap.wrap(preview, width=_W - _TAG_W - 2) or ["<empty>"]

    print(_BAR, file=sys.stderr)
    print(f"◀ OUTPUT  {app}", file=sys.stderr)
    print(_SEP, file=sys.stderr)
    _row("TASK", [task])
    _row("RESULT", content_lines)
    if extra:
        for k, v in extra.items():
            _row(k.upper()[:_TAG_W], [str(v)])
    print(_BAR, file=sys.stderr, flush=True)
