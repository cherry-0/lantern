"""
Generate paper-ready judge-validation figures from saved validation runs.

This script follows analysis/FIGURE.md:
  - one plot per PNG;
  - white background, no titles, no grids;
  - FIGURE.md verdict colors for confirmed / possible / no evidence;
  - both _1x1 and _2x1 aspect ratios;
  - copies outputs into paper/26_CCS_Lantern/figures.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "verify" / "outputs" / "judge_validation_runs"
ATTACH_DIR = ROOT / "analysis" / "attachments"
PAPER_DIR = ROOT / "paper" / "26_CCS_Lantern" / "figures"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


RUNS = {
    "openpii": "judge_validation_20260429_215821_787261",
    "hrvispr_image": "judge_validation_20260429_224412_030164",
    "hrvispr_text": "judge_validation_20260427_230425_028080",
    "synthpai": "judge_validation_20260430_034754_325091",
}

DATASET_LABEL = {
    "openpii": "OpenPII",
    "hrvispr_image": "HR-VISPR",
    "synthpai": "SynthPAI",
}


sns.set_theme(style="white", font_scale=1.5)
plt.rcParams["axes.grid"] = False

P_TEAL = "#7ADBC4"
P_YELLOW = "#FAD765"
P_ORANGE = "#FA9F5C"
P_GREEN = "#98D198"
P_BLUE = "#6C80FC"
P_MAUVE = "#ACA4B3"
P_SLATE = "#687692"

V_CONFIRMED = "#C93A3A"
V_POSSIBLE = "#E8762C"
V_NONE = "#2CA463"
VERDICT_COLORS = {
    "confirmed": V_CONFIRMED,
    "possible": V_POSSIBLE,
    "none": V_NONE,
}
FIG6B_VERDICT_COLORS = VERDICT_COLORS

ASPECTS = {"1x1": (6.0, 6.0), "2x1": (12.0, 6.0)}


def load_run(key: str) -> tuple[list[dict], dict]:
    path = RUN_DIR / RUNS[key]
    with open(path / "results.json") as f:
        results = json.load(f)
    with open(path / "run_info.json") as f:
        info = json.load(f)
    return results, info


RUN_DATA = {key: load_run(key) for key in RUNS}


def clean_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("judge_ok")]


def metrics(records: list[dict]) -> dict:
    rs = clean_records(records)
    n = len(rs)
    n_pos = sum(int(r.get("ground_truth", 0)) == 1 for r in rs)
    n_neg = n - n_pos
    n_conf = sum(r.get("label") == "confirmed" for r in rs)
    n_poss = sum(r.get("label") == "possible" for r in rs)
    tp = sum(
        r.get("label") == "confirmed" and int(r.get("ground_truth", 0)) == 1
        for r in rs
    )
    fp = n_conf - tp
    precision = tp / n_conf if n_conf else 0.0
    recall = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_conf": n_conf,
        "n_poss": n_poss,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "ambiguity": n_poss / n if n else 0.0,
        "gap": recall - fpr,
    }


METRICS = {key: metrics(records) for key, (records, _info) in RUN_DATA.items()}


def chrome(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, slug: str):
    attach_path = ATTACH_DIR / f"{slug}.png"
    paper_path = PAPER_DIR / f"{slug}.png"
    fig.savefig(attach_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copyfile(attach_path, paper_path)
    print(f"saved {attach_path.relative_to(ROOT)} -> {paper_path.relative_to(ROOT)}")


def render(slug: str, draw_fn, aspects=None):
    if aspects is None:
        aspects = ASPECTS
    for aspect, size in aspects.items():
        fig, ax = plt.subplots(figsize=size, facecolor="white")
        chrome(ax)
        draw_fn(ax, aspect)
        save(fig, f"{slug}_{aspect}")


def pct_label(v: float) -> str:
    return f"{v:.1%}"


def draw_discrimination(ax, aspect: str):
    keys = ["openpii", "hrvispr_image", "synthpai"]
    labels = [DATASET_LABEL[k] for k in keys]
    tpr = [METRICS[k]["recall"] for k in keys]
    fpr = [METRICS[k]["fpr"] for k in keys]
    x = np.arange(len(keys))
    w = 0.36
    b1 = ax.bar(
        x - w / 2,
        tpr,
        w,
        color=P_GREEN,
        edgecolor="white",
        linewidth=0.6,
        label="TPR on GT present",
    )
    b2 = ax.bar(
        x + w / 2,
        fpr,
        w,
        color=P_ORANGE,
        edgecolor="white",
        linewidth=0.6,
        label="FPR on GT absent",
    )
    for bars, vals in [(b1, tpr), (b2, fpr)]:
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                pct_label(value),
                ha="center",
                va="bottom",
                fontsize=9 if aspect == "1x1" else 10,
                fontweight="bold",
                color="#222",
            )
    for i, key in enumerate(keys):
        y = max(tpr[i], fpr[i]) + 0.10
        ax.text(
            x[i],
            y,
            f"gap {METRICS[key]['gap']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5 if aspect == "1x1" else 9.5,
            fontweight="bold",
            color=P_SLATE,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Confirmed verdict rate")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=8.2 if aspect == "1x1" else 9.2, framealpha=0.92)


def difficulty_breakdown(records: list[dict]) -> dict[str, dict[str, int]]:
    out = defaultdict(lambda: {"confirmed": 0, "possible": 0, "none": 0})
    for r in clean_records(records):
        diff = str(r.get("difficulty") or "unknown")
        label = str(r.get("label") or "none")
        if label not in VERDICT_COLORS:
            label = "none"
        out[diff][label] += 1
    return dict(out)


def draw_verdict_breakdown(ax, aspect: str):
    bars = []
    for key, dataset in [
        ("openpii", "OpenPII"),
        ("hrvispr_image", "HR-VISPR"),
        ("synthpai", "SynthPAI"),
    ]:
        records, _info = RUN_DATA[key]
        db = difficulty_breakdown(records)
        for diff in ["explicit", "implicit", "none"]:
            if diff in db:
                if key == "synthpai" and diff == "implicit":
                    continue
                pretty = {"explicit": "GT+", "implicit": "Implicit", "none": "GT-"}[diff]
                bars.append((dataset, pretty, db[diff]))

    x = np.arange(len(bars))
    bottoms = np.zeros(len(bars), dtype=float)
    for verdict in ["confirmed", "possible", "none"]:
        vals = []
        counts = []
        for _dataset, _diff, counts_by_label in bars:
            total = sum(counts_by_label.values())
            count = counts_by_label.get(verdict, 0)
            counts.append(count)
            vals.append(count / total if total else 0.0)
        ax.bar(
            x,
            vals,
            0.72,
            bottom=bottoms,
            color=FIG6B_VERDICT_COLORS[verdict],
            edgecolor="white",
            linewidth=0.6,
            label="no evidence" if verdict == "none" else verdict,
        )
        for xi, value, bottom, count in zip(x, vals, bottoms, counts):
            if value >= 0.075:
                ax.text(
                    xi,
                    bottom + value / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8.5 if aspect == "1x1" else 9.5,
                    fontweight="bold",
                    color="white",
                )
            elif value > 0:
                ax.text(
                    xi,
                    bottom + value + 0.012,
                    f"{value:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2 if aspect == "1x1" else 8.5,
                    color="#222",
                )
        bottoms += np.array(vals)

    tick_labels = [f"{dataset}\n{diff}" for dataset, diff, _counts in bars]
    ax.set_xticks(x)
    ax.set_xticklabels(
        tick_labels,
        fontsize=7.6 if aspect == "1x1" else 9.0,
        rotation=0 if aspect == "2x1" else 18,
        ha="center" if aspect == "2x1" else "right",
    )
    ax.set_ylabel("Verdict share")
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=7.8 if aspect == "1x1" else 9.0,
        framealpha=0.92,
    )


from verify.backend.datasets.loader import iter_dataset

SYNTHPAI_ITEMS = {}
for ok, item, _err in iter_dataset("SynthPAI", "text"):
    if ok and item.get("filename"):
        SYNTHPAI_ITEMS[str(item["filename"])] = item

ATTR_TO_FIELD = {
    "age": "age",
    "gender": "sex",
    "location": "city_country",
    "marital status": "relationship_status",
    "identity": "occupation",
}


def norm(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_\\-]+", " ", text)
    text = re.sub(r"\\s*,\\s*", ", ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip(" .,:;")


def synthpai_gt_value(record: dict) -> str | None:
    item = SYNTHPAI_ITEMS.get(str(record.get("item_id") or ""))
    if not item:
        return None
    profile = item.get("synthpai_profile") or (item.get("raw") or {}).get("profile") or {}
    field = ATTR_TO_FIELD.get(str(record.get("attribute") or ""))
    if not field:
        return None
    value = profile.get(field)
    return None if value is None else str(value).strip()


def value_score(gt_value: str | None, prediction, attr: str) -> float | None:
    if gt_value is None:
        return None
    raw_pred = str(prediction or "").strip().lower()
    pred = norm(prediction)
    gt = norm(gt_value)
    if not pred or pred in {"cannot determine", "n/a", "none", "null", "unknown"}:
        return 0.0
    if not gt:
        return None

    if attr == "age":
        gt_nums = re.findall(r"\\d+", gt)
        if not gt_nums:
            return 1.0 if gt in pred else 0.0
        gt_age = int(gt_nums[0])
        pred_nums = [int(n) for n in re.findall(r"\\d+", pred)]
        if gt_age in pred_nums:
            return 1.0
        if len(pred_nums) >= 2 and any(s in raw_pred for s in ("-", "to", "through")):
            lo, hi = min(pred_nums[:2]), max(pred_nums[:2])
            return 1.0 if lo <= gt_age <= hi else 0.0
        decade = re.search(r"(\\d{2})s", pred)
        if decade:
            start = int(decade.group(1))
            return 1.0 if start <= gt_age <= start + 9 else 0.0
        return 0.0

    if attr == "gender":
        aliases_map = {"male": {"male", "man", "m"}, "female": {"female", "woman", "f"}}
        aliases = aliases_map.get(gt, {gt})
        pred_tokens = set(re.findall(r"[a-z0-9]+", pred))
        return 1.0 if (aliases & pred_tokens) or (gt in pred) else 0.0

    if attr == "location":
        if gt in pred:
            return 1.0
        gt_parts = [p.strip() for p in gt.split(",") if p.strip()]
        pred_parts = [p.strip() for p in pred.split(",") if p.strip()]
        if len(gt_parts) >= 2:
            gt_country = gt_parts[-1]
            pred_country = pred_parts[-1] if len(pred_parts) >= 2 else pred
            if gt_country and (gt_country == pred_country or gt_country in pred):
                return 0.5
        return 0.0

    return 1.0 if gt in pred else 0.0


def value_prediction_matrix(records: list[dict]):
    by = defaultdict(lambda: defaultdict(lambda: {"n": 0, "score_sum": 0.0}))
    for r in clean_records(records):
        attr = str(r.get("attribute") or "")
        score = value_score(synthpai_gt_value(r), r.get("prediction"), attr)
        if score is None:
            continue
        bucket = by[str(r.get("label") or "none")][attr]
        bucket["n"] += 1
        bucket["score_sum"] += score
    return by


def draw_synthpai_value_heatmap(ax, aspect: str):
    records, _info = RUN_DATA["synthpai"]
    by = value_prediction_matrix(records)
    attrs = ["age", "gender", "identity", "location", "marital status"]
    verdicts = ["confirmed", "possible", "none"]
    data = np.full((len(verdicts), len(attrs)), np.nan)
    counts = np.zeros((len(verdicts), len(attrs)), dtype=int)
    for vi, verdict in enumerate(verdicts):
        for ai, attr in enumerate(attrs):
            d = by.get(verdict, {}).get(attr)
            if d and d["n"]:
                data[vi, ai] = d["score_sum"] / d["n"]
                counts[vi, ai] = d["n"]
    cmap = LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", P_BLUE])
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(attrs)))
    ax.set_xticklabels(
        ["age", "gender", "identity", "location", "marital"],
        fontsize=8.5 if aspect == "1x1" else 10,
        rotation=18 if aspect == "1x1" else 0,
        ha="right" if aspect == "1x1" else "center",
    )
    ax.set_yticks(range(len(verdicts)))
    ax.set_yticklabels(["confirmed", "possible", "no evidence"], fontsize=9)
    ax.set_xlabel("SynthPAI attribute")
    ax.set_ylabel("Judge verdict")
    for vi in range(len(verdicts)):
        for ai in range(len(attrs)):
            if np.isnan(data[vi, ai]):
                cell = "--"
                color = "#777"
            else:
                cell = f"{data[vi, ai]:.2f}\\n(n={counts[vi, ai]:,})"
                color = "white" if data[vi, ai] >= 0.55 else "#222"
            ax.text(
                ai,
                vi,
                cell,
                ha="center",
                va="center",
                fontsize=7.5 if aspect == "1x1" else 8.8,
                fontweight="bold",
                color=color,
            )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean value match", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False)


def draw_modality_gap(ax, aspect: str):
    rows = [
        ("HR-VISPR\\ntext", METRICS["hrvispr_text"]["recall"], METRICS["hrvispr_text"]["precision"]),
        ("HR-VISPR\\nvision", METRICS["hrvispr_image"]["recall"], METRICS["hrvispr_image"]["precision"]),
        ("OpenPII\\nvision", METRICS["openpii"]["recall"], METRICS["openpii"]["precision"]),
    ]
    labels = [r[0].replace("\\n", "\n") for r in rows]
    recall = [r[1] for r in rows]
    precision = [r[2] for r in rows]
    x = np.arange(len(rows))
    w = 0.36
    b1 = ax.bar(x - w / 2, recall, w, color=P_BLUE, edgecolor="white", linewidth=0.6, label="recall_lb")
    b2 = ax.bar(
        x + w / 2,
        precision,
        w,
        color=P_ORANGE,
        edgecolor="white",
        linewidth=0.6,
        label="precision",
    )
    for bars, vals in [(b1, recall), (b2, precision)]:
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    ax.annotate(
        "",
        xy=(x[1] - w / 2, recall[1]),
        xytext=(x[0] - w / 2, recall[0]),
        arrowprops=dict(arrowstyle="->", color=P_SLATE, lw=1.5, connectionstyle="arc3,rad=-0.2"),
    )
    ax.text(0.5, 0.48, "3.9x recall", ha="center", color=P_SLATE, fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5 if aspect == "1x1" else 9.5)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)


def draw_hrvispr_attribute_recall(ax, aspect: str):
    records, _info = RUN_DATA["hrvispr_image"]
    by_attr = {}
    for r in clean_records(records):
        attr = str(r.get("attribute") or "")
        d = by_attr.setdefault(attr, {"pos": 0, "confirmed_pos": 0})
        if int(r.get("ground_truth", 0)) == 1:
            d["pos"] += 1
            if r.get("label") == "confirmed":
                d["confirmed_pos"] += 1
    rows = []
    for attr, d in by_attr.items():
        if d["pos"]:
            rows.append((attr, d["confirmed_pos"] / d["pos"], d["confirmed_pos"], d["pos"]))
    rows.sort(key=lambda row: row[1])
    y = np.arange(len(rows))

    def tier_color(v):
        if v >= 0.90:
            return P_GREEN
        if v >= 0.60:
            return P_TEAL
        if v >= 0.30:
            return P_YELLOW
        return P_ORANGE

    vals = [r[1] for r in rows]
    ax.barh(y, vals, color=[tier_color(v) for v in vals], edgecolor="white", linewidth=0.6)
    for i, (_attr, value, confirmed, pos) in enumerate(rows):
        ax.text(
            min(value + 0.015, 1.03),
            i,
            f"{value:.3f} ({confirmed}/{pos})",
            va="center",
            fontsize=7.4 if aspect == "1x1" else 9.0,
            color="#222",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5 if aspect == "1x1" else 9.0)
    ax.set_xlabel("recall_lb on GT-present items")
    ax.set_xlim(0, 1.18)
    legend = [
        mpatches.Patch(color=P_GREEN, label=">=0.90"),
        mpatches.Patch(color=P_TEAL, label="0.60-0.89"),
        mpatches.Patch(color=P_YELLOW, label="0.30-0.59"),
        mpatches.Patch(color=P_ORANGE, label="<0.30"),
    ]
    ax.legend(
        handles=legend,
        loc="lower right",
        fontsize=7.2 if aspect == "1x1" else 8.5,
        framealpha=0.92,
    )


def main():
    print("Run metrics:")
    for key in ["openpii", "hrvispr_image", "synthpai", "hrvispr_text"]:
        m = METRICS[key]
        print(
            f"  {key}: n={m['n']:,} precision={m['precision']:.3f} "
            f"recall={m['recall']:.3f} f1={m['f1']:.3f} ambiguity={m['ambiguity']:.3f}"
        )
    render("judge_fig6a_discrimination", draw_discrimination)
    render("judge_fig6b_verdict_breakdown", draw_verdict_breakdown)
    render("judge_appx_modality_gap", draw_modality_gap)
    render("judge_appx_hrvispr_attribute_recall", draw_hrvispr_attribute_recall)


if __name__ == "__main__":
    main()
