"""
Deep analysis for judge_analysis.md update — judge-image (canonical) results
across OpenPII / HR-VISPR / SynthPAI, plus SynthPAI value-prediction analysis
and model-wise comparison vs text-only judge configurations.

Compliant with analysis/FIGURE.md:
  §1.1  Adobe pastel primary palette for non-verdict, non-stage data.
  §1.3  Verdict colors only for the verdict axis (confirmed/possible/none).
  §1.4  Different metrics on the same axis use different primary-palette hues.
  §2    White figure & axes background; whitegrid; hide top/right spines.
  §3    One plot, one PNG (no panel concatenation).
  §4    Each figure saved in both _1x1 (6×6) and _2x1 (12×6) aspect ratios.
  §5    150 dpi, bbox_inches="tight", explicit white facecolor.

Outputs:
  analysis/attachments/judge_deep_*_{1x1,2x1}.png
  analysis/judge_analysis_deep.json   (raw computed numbers)
"""
from __future__ import annotations
import json, sys, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

LANTERN = Path("/Users/sieun/Research/Lantern/lantern")
sys.path.insert(0, str(LANTERN))
RUNS  = LANTERN / "verify/outputs/judge_validation_runs"
OUT   = LANTERN / "analysis/attachments"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style (FIGURE.md §1, §2) ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
# Adobe pastel primary palette (§1.1)
P_TEAL, P_YELLOW, P_ORANGE, P_GREEN, P_BLUE, P_MAUVE, P_SLATE = (
    "#7ADBC4", "#FAD765", "#FA9F5C", "#98D198", "#6C80FC", "#ACA4B3", "#687692"
)
# Reserved verdict colors (§1.3)
V_CONFIRMED = "#C93A3A"
V_POSSIBLE  = "#E8762C"
V_NONE      = "#2CA463"
VERDICT_COLOR = {"confirmed": V_CONFIRMED, "possible": V_POSSIBLE, "none": V_NONE}

ASPECTS = {"1x1": (6.0, 6.0), "2x1": (12.0, 6.0)}


def chrome(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, slug):
    p = OUT / f"{slug}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {p.name}")


def render_both_aspects(slug, draw_fn):
    """Run draw_fn(ax, aspect, fig) for each required aspect ratio (FIGURE.md §4)."""
    for aspect_key, (w, h) in ASPECTS.items():
        fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
        chrome(ax)
        draw_fn(ax, aspect_key, fig)
        save(fig, f"{slug}_{aspect_key}")


# ── Run inventory ────────────────────────────────────────────────────────────
def load_run(rid):
    rd = RUNS / rid
    info = json.load(open(rd / "run_info.json"))
    res  = json.load(open(rd / "results.json"))
    return res, info


RUNS_INDEX = {
    # Canonical judge-image runs (updated 2026-04-29 to use n=1800 HR-VISPR / n=400 OpenPII)
    "HR-VISPR_judge-image_flash001":    "judge_validation_20260429_224412_030164",  # n=1800, n_ok=1799
    "HR-VISPR_judge-text_flash001":     "judge_validation_20260427_230425_028080",
    "OpenPII_judge-image_flash001":     "judge_validation_20260429_215821_787261",  # n=400, n_ok=400
    "SynthPAI_judge-image_flash":       "judge_validation_20260428_204037_791595",
    "SynthPAI_judge-image_flashLite":   "judge_validation_20260428_214703_485379",
    # Earlier (smaller-n) HR-VISPR judge-image runs, kept for historical comparison only
    "HR-VISPR_judge-image_flash001_C":  "judge_validation_20260427_234152_815659",
    "HR-VISPR_judge-image_flash001_D":  "judge_validation_20260428_000725_921062",
    "HR-VISPR_judge-image_flash001_E":  "judge_validation_20260428_002834_655781",
}


# ── Metric primitives ────────────────────────────────────────────────────────
def compute_metrics(records, restrict_attr=None):
    rs = [r for r in records if r.get("judge_ok")]
    if restrict_attr is not None:
        rs = [r for r in rs if r["attribute"] == restrict_attr]
    n = len(rs)
    if n == 0:
        return None
    n_pos = sum(1 for r in rs if int(r.get("ground_truth", 0)) == 1)
    n_neg = n - n_pos
    n_conf = sum(1 for r in rs if r["label"] == "confirmed")
    n_poss = sum(1 for r in rs if r["label"] == "possible")
    n_none = n - n_conf - n_poss
    n_conf_tp = sum(1 for r in rs if r["label"] == "confirmed" and int(r.get("ground_truth", 0)) == 1)
    n_conf_fp = n_conf - n_conf_tp
    n_poss_pos = sum(1 for r in rs if r["label"] == "possible" and int(r.get("ground_truth", 0)) == 1)
    return {
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "n_conf": n_conf, "n_poss": n_poss, "n_none": n_none,
        "n_conf_tp": n_conf_tp, "n_conf_fp": n_conf_fp,
        "n_poss_pos": n_poss_pos,
        "precision_confirmed": (n_conf_tp / n_conf) if n_conf else None,
        "coverage_confirmed":  n_conf / n,
        "ambiguity_rate":      n_poss / n,
        "recall_lb":           (n_conf_tp / n_pos) if n_pos else None,
        "recall_lb_with_poss": ((n_conf_tp + n_poss_pos) / n_pos) if n_pos else None,
        "tpr":                 (n_conf_tp / n_pos) if n_pos else None,
        "fpr":                 (n_conf_fp / n_neg) if n_neg else None,
        "f1":                  None,
        "specificity":         None,
    }


def with_f1(m):
    if m and m.get("precision_confirmed") and m.get("recall_lb"):
        p, r = m["precision_confirmed"], m["recall_lb"]
        m["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    if m and m.get("n_neg") and m["n_neg"] > 0:
        m["specificity"] = (m["n_neg"] - m["n_conf_fp"]) / m["n_neg"]
    return m


# ── Load runs ────────────────────────────────────────────────────────────────
print("Loading runs...")
runs = {}
for k, rid in RUNS_INDEX.items():
    res, info = load_run(rid)
    runs[k] = (res, info)
    n_total = len(res)
    n_ok = sum(1 for r in res if r.get("judge_ok"))
    print(f"  {k}: total={n_total} ok={n_ok} model={info.get('model')}")


# ── §3 Headline ───────────────────────────────────────────────────────────────
print("\n=== Headline metrics ===")
headline = {}
for label, key in [("OpenPII", "OpenPII_judge-image_flash001"),
                   ("HR-VISPR", "HR-VISPR_judge-image_flash001"),
                   ("SynthPAI (flash)", "SynthPAI_judge-image_flash"),
                   ("SynthPAI (flash-lite, n=25k)", "SynthPAI_judge-image_flashLite")]:
    res, info = runs[key]
    m = with_f1(compute_metrics(res))
    headline[label] = m


# ── Per-attr & difficulty breakdowns ─────────────────────────────────────────
def per_attr_metrics(records):
    out = {}
    rs = [r for r in records if r.get("judge_ok")]
    attrs = sorted({r["attribute"] for r in rs})
    for a in attrs:
        out[a] = with_f1(compute_metrics(rs, restrict_attr=a))
    return out


def difficulty_breakdown(records):
    out = defaultdict(lambda: {"confirmed": 0, "possible": 0, "none": 0})
    for r in records:
        if not r.get("judge_ok"):
            continue
        d = r.get("difficulty", "unknown")
        out[d][r["label"]] += 1
    return dict(out)


per_attr = {}
diff_brk = {}
for label, key in [("OpenPII", "OpenPII_judge-image_flash001"),
                   ("HR-VISPR", "HR-VISPR_judge-image_flash001"),
                   ("SynthPAI (flash)", "SynthPAI_judge-image_flash"),
                   ("SynthPAI (flash-lite, n=25k)", "SynthPAI_judge-image_flashLite")]:
    res, _ = runs[key]
    per_attr[label] = per_attr_metrics(res)
    diff_brk[label] = difficulty_breakdown(res)


# ── §4 SynthPAI value-prediction analysis ────────────────────────────────────
print("\n=== Loading SynthPAI profiles for value-prediction analysis ===")
from verify.backend.datasets.loader import iter_dataset
items_by_id = {}
for ok, item, _err in iter_dataset("SynthPAI", "text"):
    if ok and item.get("filename"):
        items_by_id[str(item["filename"])] = item
print(f"  Loaded {len(items_by_id)} SynthPAI profile items")

ATTR_TO_FIELD = {
    "age": "age",
    "gender": "sex",
    "location": "city_country",
    "marital status": "relationship_status",
    "identity": "occupation",
}


def norm_lookup(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def synthpai_gt_value(record):
    item = items_by_id.get(str(record.get("item_id") or ""))
    if not item:
        return None
    profile = item.get("synthpai_profile") or (item.get("raw") or {}).get("profile") or {}
    field = ATTR_TO_FIELD.get(str(record.get("attribute") or ""))
    if not field:
        return None
    val = profile.get(field)
    return None if val is None else str(val).strip()


def value_score(gt_value, prediction, attr):
    if gt_value is None:
        return None
    raw_pred = str(prediction or "").strip().lower()
    pred = norm_lookup(prediction)
    gt = norm_lookup(gt_value)
    if not pred or pred in {"cannot determine", "n/a", "none", "null", "unknown"}:
        return 0.0
    if not gt:
        return None

    if attr == "age":
        gt_nums = re.findall(r"\d+", gt)
        if not gt_nums:
            return 1.0 if gt in pred else 0.0
        gt_age = int(gt_nums[0])
        pred_nums = [int(n) for n in re.findall(r"\d+", pred)]
        if gt_age in pred_nums:
            return 1.0
        if len(pred_nums) >= 2 and any(s in raw_pred for s in ("-", "to", "through")):
            lo, hi = min(pred_nums[:2]), max(pred_nums[:2])
            return 1.0 if lo <= gt_age <= hi else 0.0
        decade = re.search(r"(\d{2})s", pred)
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


val_results = {}
for tag, key in [("SynthPAI (flash-lite, n=25k)", "SynthPAI_judge-image_flashLite"),
                 ("SynthPAI (flash)", "SynthPAI_judge-image_flash")]:
    res, _ = runs[key]
    by_label_attr = defaultdict(lambda: defaultdict(lambda: {"n": 0, "score_sum": 0.0, "exact": 0, "partial": 0}))
    by_label_overall = defaultdict(lambda: {"n": 0, "score_sum": 0.0, "exact": 0, "partial": 0})
    n_scored = 0
    for r in res:
        if not r.get("judge_ok"):
            continue
        attr = r.get("attribute")
        gtv  = synthpai_gt_value(r)
        s    = value_score(gtv, r.get("prediction"), attr)
        if s is None:
            continue
        n_scored += 1
        verdict = r["label"]
        d = by_label_attr[verdict][attr]
        o = by_label_overall[verdict]
        for bucket in (d, o):
            bucket["n"] += 1
            bucket["score_sum"] += s
            if s == 1.0:
                bucket["exact"] += 1
            if 0.0 < s < 1.0:
                bucket["partial"] += 1
    val_results[tag] = {
        "n_scored": n_scored,
        "by_label_overall": {k: dict(v) for k, v in by_label_overall.items()},
        "by_label_attr":   {k: {a: dict(b) for a, b in v.items()} for k, v in by_label_attr.items()},
    }


# ── Model comparison ────────────────────────────────────────────────────────
model_compare = {}
for label, key in [
    ("HR-VISPR / judge-text / flash-001",        "HR-VISPR_judge-text_flash001"),
    ("HR-VISPR / judge-image / flash-001",       "HR-VISPR_judge-image_flash001"),
    ("OpenPII / judge-image / flash-001",         "OpenPII_judge-image_flash001"),
    ("SynthPAI / judge-image / flash",            "SynthPAI_judge-image_flash"),
    ("SynthPAI / judge-image / flash-lite",       "SynthPAI_judge-image_flashLite"),
]:
    res, info = runs[key]
    m = with_f1(compute_metrics(res))
    model_compare[label] = {
        "model":     info.get("model"),
        "evaluator": info.get("evaluator"),
        "dataset":   info.get("datasets", [None])[0],
        "metrics":   m,
    }


# ── Persist ─────────────────────────────────────────────────────────────────
def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list): return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o
JSON_OUT = LANTERN / "analysis/judge_analysis_deep.json"
JSON_OUT.write_text(json.dumps(_clean({
    "headline": headline, "per_attr": per_attr,
    "difficulty_breakdown": diff_brk, "value_prediction": val_results,
    "model_compare": model_compare,
}), indent=2))


# ── Cleanup older non-compliant figures ─────────────────────────────────────
print("\n=== Cleaning up older judge_deep_* files ===")
for old in OUT.glob("judge_deep_*"):
    old.unlink()


# ── Figures ──────────────────────────────────────────────────────────────────
print("\n=== Generating figures (FIGURE.md compliant) ===")

# F1: Headline metrics — different metrics → primary palette (§1.4)
HEAD_DATASETS = [
    ("OpenPII\n(judge-image / flash-001, n=400)",       headline["OpenPII"]),
    ("HR-VISPR\n(judge-image / flash-001, n=1.8k)",     headline["HR-VISPR"]),
    ("SynthPAI\n(judge-image / flash, n=250)",          headline["SynthPAI (flash)"]),
    ("SynthPAI\n(judge-image / flash-lite, n=25k)",     headline["SynthPAI (flash-lite, n=25k)"]),
]
HEAD_METRICS = [
    ("precision_confirmed", "precision",  P_BLUE),
    ("recall_lb",           "recall_lb",  P_GREEN),
    ("f1",                  "F1",         P_ORANGE),
    ("ambiguity_rate",      "ambiguity",  P_MAUVE),
    ("coverage_confirmed",  "coverage",   P_TEAL),
]


def _draw_headline(ax, aspect, fig):
    labels = [a for a, _ in HEAD_DATASETS]
    x = np.arange(len(labels))
    w = 0.16
    for i, (mk, pretty, color) in enumerate(HEAD_METRICS):
        vals = [(HEAD_DATASETS[j][1] or {}).get(mk) or 0 for j in range(len(labels))]
        offset = (i - 2) * w
        bars = ax.bar(x + offset, vals, w, color=color,
                      edgecolor="white", linewidth=0.6, label=pretty)
        for b, v in zip(bars, vals):
            if v > 0.005:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8 if aspect == "1x1" else 8.5)
    label_fs = 8.5 if aspect == "1x1" else 9.5
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=label_fs)
    ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.10)
    ax.set_title("Headline metrics — judge-image (canonical) on three validation datasets",
                 fontsize=11.5)
    ax.legend(loc="upper right", ncol=5 if aspect == "2x1" else 2, fontsize=8.5,
              framealpha=0.92)


render_both_aspects("judge_deep_fig1_headline", _draw_headline)


# F2: Difficulty stacked bars — verdict colors (§1.3)
def _make_difficulty_drawer(label):
    def draw(ax, aspect, fig):
        db = diff_brk[label]
        diffs_present = [d for d in ["explicit", "implicit", "none"] if d in db]
        x = np.arange(len(diffs_present))
        bottoms = np.zeros(len(diffs_present))
        for verdict in ["confirmed", "possible", "none"]:
            vals = [db[d].get(verdict, 0) for d in diffs_present]
            ax.bar(x, vals, 0.6, bottom=bottoms,
                   color=VERDICT_COLOR[verdict], edgecolor="white", linewidth=0.6,
                   label=verdict)
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 0:
                    ax.text(xi, b + v / 2, f"{v:,}", ha="center", va="center",
                            color="white" if v > 5 else "#222", fontsize=10, fontweight="bold")
            bottoms += np.array(vals)
        ax.set_xticks(x); ax.set_xticklabels(diffs_present, fontsize=11)
        ax.set_ylabel("Count"); ax.set_title(f"Verdict breakdown by difficulty — {label}", fontsize=11.5)
        ax.legend(loc="upper right", ncol=3, framealpha=0.92, fontsize=10)
    return draw


for label, slug in [("OpenPII",                       "judge_deep_fig2_difficulty_openpii"),
                    ("HR-VISPR",                      "judge_deep_fig2_difficulty_hrvispr"),
                    ("SynthPAI (flash)",              "judge_deep_fig2_difficulty_synthpai_flash"),
                    ("SynthPAI (flash-lite, n=25k)",  "judge_deep_fig2_difficulty_synthpai_lite")]:
    render_both_aspects(slug, _make_difficulty_drawer(label))


# F3: Per-attribute recall_lb — tier ramp (§1.4)
def _tier_color(r):
    if r >= 0.90: return P_GREEN
    if r >= 0.60: return P_TEAL
    if r >= 0.30: return P_YELLOW
    return P_ORANGE


def _make_perattr_drawer(label):
    pa = per_attr[label]
    rows = []
    for a, m in pa.items():
        if m and m["n"] > 0:
            rows.append((a, m["recall_lb"] if m["recall_lb"] is not None else 0.0,
                         m["n_pos"], m["n"]))
    rows.sort(key=lambda r: r[1])

    def draw(ax, aspect, fig):
        y = np.arange(len(rows))
        recalls = [r[1] for r in rows]
        colors = [_tier_color(r) for r in recalls]
        ax.barh(y, recalls, color=colors, edgecolor="white", linewidth=0.6)
        for i, (a, r, npos, n) in enumerate(rows):
            ax.text(max(r + 0.012, 0.02), i,
                    f"{r:.3f}  (GT+={npos:,}, n={n:,})", va="center",
                    fontsize=9 if aspect == "1x1" else 10, color="#222")
        ax.axvline(1.0, color=P_SLATE, linestyle="--", linewidth=0.9, alpha=0.45)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
        ax.set_xlim(0, 1.18); ax.set_xlabel("recall_lb")
        ax.set_title(f"Per-attribute recall_lb — {label}", fontsize=11.5)
        legend = [
            mpatches.Patch(color=P_GREEN,  label="≥ 0.90"),
            mpatches.Patch(color=P_TEAL,   label="0.60–0.89"),
            mpatches.Patch(color=P_YELLOW, label="0.30–0.59"),
            mpatches.Patch(color=P_ORANGE, label="< 0.30"),
        ]
        ax.legend(handles=legend, loc="lower right", title="Recall tier",
                  fontsize=8.5, title_fontsize=9, framealpha=0.92)
    return draw


for label, slug in [("OpenPII",                       "judge_deep_fig3_perattr_openpii"),
                    ("HR-VISPR",                      "judge_deep_fig3_perattr_hrvispr"),
                    ("SynthPAI (flash)",              "judge_deep_fig3_perattr_synthpai_flash"),
                    ("SynthPAI (flash-lite, n=25k)",  "judge_deep_fig3_perattr_synthpai_lite")]:
    render_both_aspects(slug, _make_perattr_drawer(label))


# F4: Per-attribute (GT × verdict) heatmaps — ONE PNG PER ATTRIBUTE (§3)
PRIMARY_CYCLE = [P_TEAL, P_YELLOW, P_ORANGE, P_GREEN, P_BLUE, P_MAUVE, P_SLATE]
LABELS = ["confirmed", "possible", "none"]
LABELS_DISPLAY = ["confirmed", "possible", "no evidence"]


def _make_attr_heatmap_drawer(records, attr, color_hex):
    rs = [r for r in records if r.get("judge_ok") and r["attribute"] == attr]
    mat = np.zeros((2, 3), dtype=int)
    for r in rs:
        gt = int(r.get("ground_truth", 0))
        v = r["label"]
        if v in LABELS:
            mat[gt, LABELS.index(v)] += 1
    n_neg = int(mat[0].sum()); n_pos = int(mat[1].sum())
    rowt = mat.sum(axis=1, keepdims=True).astype(float)
    rates = np.divide(mat, rowt, where=rowt > 0, out=np.zeros_like(mat, dtype=float))
    cmap = LinearSegmentedColormap.from_list(f"w_{attr}", ["#FFFFFF", color_hex])

    def draw(ax, aspect, fig):
        ax.imshow(rates, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        row_labels = [f"absent\n(n={n_neg:,})", f"present\n(n={n_pos:,})"]
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS_DISPLAY, fontsize=11)
        ax.set_yticks(range(2)); ax.set_yticklabels(row_labels, fontsize=11)
        ax.set_xlabel("Judge verdict", fontsize=11)
        ax.set_ylabel("Ground truth", fontsize=11)
        ax.set_title(f"{attr}", fontsize=14, fontweight="bold")
        for i in range(2):
            for j in range(3):
                count = int(mat[i, j]); rate = float(rates[i, j])
                text_color = "#FFFFFF" if rate >= 0.60 else "#222"
                rt = int(rowt[i, 0])
                cell = f"{count:,}\n({rate:.1%})" if rt > 0 else f"{count:,}\n(--)"
                ax.text(j, i, cell, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_color)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(left=False, bottom=False)
    return draw


for ds_label, ds_key, slug_root in [
    ("OpenPII",                      "OpenPII_judge-image_flash001",     "openpii"),
    ("HR-VISPR",                     "HR-VISPR_judge-image_flash001",    "hrvispr"),
    ("SynthPAI (flash-lite, n=25k)", "SynthPAI_judge-image_flashLite",   "synthpai_lite"),
]:
    res, _ = runs[ds_key]
    rs = [r for r in res if r.get("judge_ok")]
    attrs = sorted({r["attribute"] for r in rs})
    for i, attr in enumerate(attrs):
        attr_slug = re.sub(r"[^a-z0-9]+", "_", attr.lower()).strip("_")
        color = PRIMARY_CYCLE[i % len(PRIMARY_CYCLE)]
        slug = f"judge_deep_fig4_heatmap_{slug_root}_{attr_slug}"
        render_both_aspects(slug, _make_attr_heatmap_drawer(res, attr, color))


# F5: Value-prediction accuracy heatmap (single 2D heatmap, one PNG per dataset)
def _make_valuepred_drawer(tag):
    vr = val_results[tag]["by_label_attr"]
    attrs = sorted({a for v in vr.values() for a in v.keys()})
    verdicts = ["confirmed", "possible", "none"]
    M = np.full((len(verdicts), len(attrs)), np.nan)
    N = np.zeros((len(verdicts), len(attrs)), dtype=int)
    for vi, v in enumerate(verdicts):
        for ai, a in enumerate(attrs):
            d = vr.get(v, {}).get(a)
            if d and d["n"]:
                M[vi, ai] = d["score_sum"] / d["n"]
                N[vi, ai] = d["n"]

    def draw(ax, aspect, fig):
        cmap = LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", P_BLUE])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(attrs))); ax.set_xticklabels(attrs, fontsize=10, rotation=20)
        ax.set_yticks(range(len(verdicts))); ax.set_yticklabels(verdicts, fontsize=11)
        ax.set_title(f"SynthPAI value-prediction accuracy — {tag}\n(mean score, location partial credit)",
                     fontsize=11)
        for vi in range(len(verdicts)):
            for ai in range(len(attrs)):
                if not np.isnan(M[vi, ai]):
                    ax.text(ai, vi, f"{M[vi, ai]:.3f}\n(n={N[vi, ai]:,})",
                            ha="center", va="center", fontsize=9,
                            color="white" if M[vi, ai] >= 0.55 else "#222",
                            fontweight="bold")
                else:
                    ax.text(ai, vi, "—", ha="center", va="center",
                            fontsize=10, color="#888")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(left=False, bottom=False)
    return draw


for tag, slug in [("SynthPAI (flash-lite, n=25k)", "judge_deep_fig5_valuepred_synthpai_lite"),
                  ("SynthPAI (flash)",             "judge_deep_fig5_valuepred_synthpai_flash")]:
    render_both_aspects(slug, _make_valuepred_drawer(tag))


# F6: Model comparison — different metrics → primary palette (§1.4)
def _draw_model_compare(ax, aspect, fig):
    rows = list(model_compare.items())
    labels = [r[0] for r in rows]
    p   = [(r[1]["metrics"] or {}).get("precision_confirmed") or 0 for r in rows]
    rec = [(r[1]["metrics"] or {}).get("recall_lb") or 0 for r in rows]
    f1  = [(r[1]["metrics"] or {}).get("f1") or 0 for r in rows]
    x = np.arange(len(labels))
    w = 0.26
    ax.bar(x - w, p,   w, color=P_BLUE,   edgecolor="white", linewidth=0.6, label="precision")
    ax.bar(x,     rec, w, color=P_GREEN,  edgecolor="white", linewidth=0.6, label="recall_lb")
    ax.bar(x + w, f1,  w, color=P_ORANGE, edgecolor="white", linewidth=0.6, label="F1")
    for i in range(len(labels)):
        for off, v in [(-w, p[i]), (0, rec[i]), (w, f1[i])]:
            if v > 0.005:
                ax.text(x[i] + off, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5 if aspect == "1x1" else 9, rotation=15, ha="right")
    ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.10)
    ax.set_title("Model-wise comparison — judge-text vs judge-image, across model tiers",
                 fontsize=11)
    ax.legend(loc="upper right", ncol=3, framealpha=0.92, fontsize=9.5)


render_both_aspects("judge_deep_fig6_model_compare", _draw_model_compare)


# F7: Precision calibration TPR/FPR — different metrics → primary palette (§1.4)
def _draw_calibration(ax, aspect, fig):
    rows = []
    for label, mc in model_compare.items():
        m = mc["metrics"]
        if m and m.get("tpr") is not None and m.get("fpr") is not None:
            rows.append((label, m["tpr"], m["fpr"]))
    x = np.arange(len(rows))
    w = 0.36
    tpr = [r[1] for r in rows]; fpr = [r[2] for r in rows]
    ax.bar(x - w/2, tpr, w, color=P_GREEN,  edgecolor="white", linewidth=0.6,
           label="TPR (judge fires when GT=1)")
    ax.bar(x + w/2, fpr, w, color=P_ORANGE, edgecolor="white", linewidth=0.6,
           label="FPR (judge fires when GT=0)")
    for i, (label, t, f) in enumerate(rows):
        ax.text(x[i] - w/2, t + 0.012, f"{t:.3f}", ha="center", va="bottom",
                fontsize=9, color="#3F7A5A", fontweight="bold")
        ax.text(x[i] + w/2, f + 0.012, f"{f:.3f}", ha="center", va="bottom",
                fontsize=9, color="#9C5A2A", fontweight="bold")
        gap = t - f
        ax.text(x[i], max(t, f) + 0.10, f"gap={gap:+.2f}", ha="center", fontsize=9,
                fontweight="bold", color=P_SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5 if aspect == "1x1" else 9,
                      rotation=15, ha="right")
    ax.set_ylabel("Rate of 'confirmed' verdict"); ax.set_ylim(0, 1.15)
    ax.set_title("Precision calibration — TPR (GT=1) vs FPR (GT=0)", fontsize=11.5)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9.5)


render_both_aspects("judge_deep_fig7_calibration", _draw_calibration)

print("\nDone.")
