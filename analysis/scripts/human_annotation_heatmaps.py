"""
Generate the paper figure for human-vs-LLM judge consistency.

The figure expects three validation datasets: OpenPII, HR-VISPR, and SynthPAI.
If a dataset has no human annotation run yet, its panel is left as an all-zero
heatmap so the paper layout remains stable while annotation is in progress.

Input:
  verify/outputs/human_judge_validation_runs/*/human_decisions.json

Output:
  analysis/attachments/human_annotation_heatmaps.png
  paper/26_CCS_Lantern/figures/human_annotation_heatmaps.png
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
HUMAN_RUNS_DIR = LANTERN_ROOT / "verify" / "outputs" / "human_judge_validation_runs"
ATTACH_DIR = LANTERN_ROOT / "analysis" / "attachments"
PAPER_FIG_DIR = LANTERN_ROOT / "paper" / "26_CCS_Lantern" / "figures"

DATASETS = ["OpenPII", "HR-VISPR", "SynthPAI"]
LABELS = ["confirmed", "possible", "none"]

P_TEAL = "#7ADBC4"
GRID = "#FFFFFF"
TEXT_DARK = "#1F2933"
TEXT_MUTED = "#687692"


def normalise_dataset(value: Any) -> str:
    raw = str(value or "").strip()
    low = raw.lower().replace("_", "").replace("-", "")
    if low in {"openpii", "openpiiopenpii"}:
        return "OpenPII"
    if low in {"hrvispr", "vispr", "hrvisprivacy"}:
        return "HR-VISPR"
    if low in {"synthpai", "synthpailite", "synthpaiflash"}:
        return "SynthPAI"
    return raw


def normalise_label(value: Any) -> str:
    low = str(value or "none").strip().lower()
    if "confirm" in low:
        return "confirmed"
    if "possible" in low:
        return "possible"
    return "none"


def _is_autosave(run_dir: Path) -> bool:
    try:
        payload = json.loads((run_dir / "human_decisions.json").read_text())
        return bool(payload.get("validation_info", {}).get("autosave", False))
    except Exception:
        return True


def load_latest_decisions() -> list[dict[str, Any]]:
    """Load decisions from the most recent completed (non-autosave) run per dataset.

    Uses the most recent autosave as fallback only when no completed run exists
    for a dataset.  Never merges decisions across sessions for the same dataset,
    because session 1 and session 2 may cover different items and different item
    counts; merging would inflate n and mix annotation contexts.
    """
    if not HUMAN_RUNS_DIR.exists():
        return []

    all_dirs = sorted(
        [p for p in HUMAN_RUNS_DIR.iterdir() if p.is_dir() and (p / "human_decisions.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Pick the newest completed run per dataset; fall back to newest autosave.
    best_run: dict[str, Path] = {}
    for run_dir in all_dirs:
        try:
            payload = json.loads((run_dir / "human_decisions.json").read_text())
        except Exception:
            continue
        is_auto = bool(payload.get("validation_info", {}).get("autosave", False))
        for decision in payload.get("decisions", []):
            ds = normalise_dataset(decision.get("dataset"))
            if not ds:
                continue
            if ds not in best_run:
                best_run[ds] = run_dir
            elif not is_auto and _is_autosave(best_run[ds]):
                best_run[ds] = run_dir
        # Stop early once we have a completed run for every target dataset.
        if all(ds in best_run and not _is_autosave(best_run[ds]) for ds in DATASETS):
            break

    decisions: list[dict[str, Any]] = []
    seen_runs: set[Path] = set()
    for ds, run_dir in best_run.items():
        if run_dir in seen_runs:
            continue
        seen_runs.add(run_dir)
        try:
            payload = json.loads((run_dir / "human_decisions.json").read_text())
        except Exception:
            continue
        for decision in payload.get("decisions", []):
            dataset = normalise_dataset(decision.get("dataset"))
            row_id = str(decision.get("id") or "").strip()
            if not dataset or not row_id:
                continue
            decisions.append({
                **decision,
                "dataset": dataset,
                "human_label": normalise_label(decision.get("human_label")),
                "judge_label": normalise_label(decision.get("judge_label")),
                "source_run": run_dir.name,
            })
    return decisions


def matrix_for_dataset(decisions: list[dict[str, Any]], dataset: str) -> np.ndarray:
    counts = Counter(
        (d["human_label"], d["judge_label"])
        for d in decisions
        if d.get("dataset") == dataset
    )
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for i, human in enumerate(LABELS):
        for j, judge in enumerate(LABELS):
            mat[i, j] = counts.get((human, judge), 0)
    return mat


def draw() -> None:
    decisions = load_latest_decisions()
    cmap = LinearSegmentedColormap.from_list("white_to_teal", ["#FFFFFF", P_TEAL])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.45), facecolor="white")
    global_max = max(
        [matrix_for_dataset(decisions, dataset).max() for dataset in DATASETS] + [1]
    )

    for ax, dataset in zip(axes, DATASETS):
        mat = matrix_for_dataset(decisions, dataset)
        n = int(mat.sum())
        agreement = (np.trace(mat) / n) if n else 0.0

        ax.imshow(mat, cmap=cmap, vmin=0, vmax=global_max, aspect="equal")
        threshold = 0.55 * global_max
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = int(mat[i, j])
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white" if val > threshold else TEXT_DARK,
                )

        for edge in range(len(LABELS) + 1):
            ax.axhline(edge - 0.5, color=GRID, linewidth=1.8)
            ax.axvline(edge - 0.5, color=GRID, linewidth=1.8)

        ax.set_title(f"{dataset}\nn={n}, agree={agreement:.0%}", fontsize=10.5, pad=8)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=32, ha="right", fontsize=9)
        ax.set_yticklabels(LABELS, fontsize=9)
        ax.set_xlabel("LLM judge", fontsize=9.5)
        ax.set_ylabel("Human annotator", fontsize=9.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if n == 0:
            ax.text(
                1,
                1.72,
                "annotation pending",
                ha="center",
                va="center",
                fontsize=8.5,
                color=TEXT_MUTED,
            )

    fig.tight_layout(w_pad=1.5)
    ATTACH_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    attach_out = ATTACH_DIR / "human_annotation_heatmaps.png"
    paper_out = PAPER_FIG_DIR / "human_annotation_heatmaps.png"
    fig.savefig(attach_out, dpi=220, bbox_inches="tight", facecolor="white")
    paper_out.write_bytes(attach_out.read_bytes())
    plt.close(fig)
    print(f"saved {attach_out.relative_to(LANTERN_ROOT)}")
    print(f"saved {paper_out.relative_to(LANTERN_ROOT)}")


if __name__ == "__main__":
    draw()
