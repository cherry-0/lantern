"""
Judge Validator — metrics computation.

Each result dict is a sample extended with evaluator output fields:
    label       — "confirmed" | "possible" | "none"
    confidence  — float 0–1
    explanation — str
    judge_ok    — bool (False if evaluator call failed)
    judge_error — str | None

The 4 headline metrics:

    precision_confirmed  = TP_confirmed / n_confirmed
    coverage_confirmed   = n_confirmed / n_total
    ambiguity_rate       = n_possible / n_total
    recall_lower_bound   = TP_confirmed / n_gt_positive

where TP_confirmed = samples that are both "confirmed" AND ground_truth == 1.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


# ── Label constants (normalised from evaluator verdict strings) ───────────────
LABEL_CONFIRMED = "confirmed"
LABEL_POSSIBLE  = "possible"
LABEL_NONE      = "none"


def normalise_verdict(verdict: str) -> str:
    """Map evaluator verdict strings to short label constants."""
    v = str(verdict).lower()
    if "confirmed" in v:
        return LABEL_CONFIRMED
    if "possible" in v:
        return LABEL_POSSIBLE
    return LABEL_NONE


# ── Core metric computation ───────────────────────────────────────────────────

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute the 4 headline metrics over a flat list of result dicts.

    Returns a dict with keys:
        precision_confirmed, coverage_confirmed, ambiguity_rate, recall_lower_bound
    Each value is a float in [0, 1] or None when the denominator is 0.
    """
    if not results:
        return {
            "precision_confirmed":  None,
            "coverage_confirmed":   0.0,
            "ambiguity_rate":       0.0,
            "recall_lower_bound":   None,
        }

    n_total       = len(results)
    n_confirmed   = 0
    n_possible    = 0
    tp_confirmed  = 0   # confirmed AND GT=1
    n_gt_positive = 0

    for r in results:
        gt    = int(r.get("ground_truth", 0))
        label = r.get("label", LABEL_NONE)
        if gt == 1:
            n_gt_positive += 1
        if label == LABEL_CONFIRMED:
            n_confirmed += 1
            if gt == 1:
                tp_confirmed += 1
        elif label == LABEL_POSSIBLE:
            n_possible += 1

    precision  = tp_confirmed / n_confirmed  if n_confirmed   > 0 else None
    coverage   = n_confirmed  / n_total
    ambiguity  = n_possible   / n_total
    recall_lb  = tp_confirmed / n_gt_positive if n_gt_positive > 0 else None

    return {
        "precision_confirmed":  precision,
        "coverage_confirmed":   coverage,
        "ambiguity_rate":       ambiguity,
        "recall_lower_bound":   recall_lb,
    }


def compute_metrics_by_group(
    results: List[Dict[str, Any]],
    group_key: str,
) -> Dict[str, Dict[str, float]]:
    """Return compute_metrics() for each unique value of *group_key*."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[str(r.get(group_key, "unknown"))].append(r)
    return {k: compute_metrics(v) for k, v in sorted(groups.items())}


def compute_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return {confirmed, possible, none} counts."""
    counts: Dict[str, int] = {LABEL_CONFIRMED: 0, LABEL_POSSIBLE: 0, LABEL_NONE: 0}
    for r in results:
        label = r.get("label", LABEL_NONE)
        if label in counts:
            counts[label] += 1
        else:
            counts[LABEL_NONE] += 1
    return counts


def compute_distribution_by_difficulty(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Return label distribution broken down by difficulty."""
    by_diff: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_diff[r.get("difficulty", "unknown")].append(r)
    return {k: compute_distribution(v) for k, v in sorted(by_diff.items())}


def false_positives(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return samples where label == confirmed AND ground_truth == 0."""
    return [r for r in results if r.get("label") == LABEL_CONFIRMED and int(r.get("ground_truth", 0)) == 0]


def false_negatives(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return samples where ground_truth == 1 AND label != confirmed."""
    return [r for r in results if int(r.get("ground_truth", 0)) == 1 and r.get("label") != LABEL_CONFIRMED]
