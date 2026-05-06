"""
Judge Validator — dataset sample loader.

Loads (text_or_image, attribute, ground_truth, difficulty) tuples from
SynthPAI, HR-VISPR, and OpenPII for evaluator validation runs.

Each returned sample dict:
    id              — unique sample identifier
    dataset         — source dataset name
    text_content    — text to evaluate (empty string for image-only)
    image_b64       — base64-encoded JPEG (HR-VISPR only; empty string otherwise)
    attribute       — target privacy attribute
    ground_truth    — int 0 or 1
    difficulty      — "explicit" | "implicit" | "none"
"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

from verify.backend.datasets.loader import iter_dataset, get_dataset_path
from verify.backend.datasets import label_mapper as _label_mapper

_label_mapper = importlib.reload(_label_mapper)
get_input_labels = _label_mapper.get_input_labels
_SYNTHPAI_REVIEW_MAP = _label_mapper._SYNTHPAI_REVIEW_MAP

# ── Attribute sets per dataset ────────────────────────────────────────────────

SYNTHPAI_ATTRS: List[str] = list(_SYNTHPAI_REVIEW_MAP.values())   # age, gender, location, marital status, identity

HRVISPR_ATTRS: List[str] = [
    "age", "face", "color", "haircolor", "gender", "race", "nudity",
    "height", "weight", "disability", "ethnic_clothing", "formal",
    "uniforms", "medical", "troupe", "sports", "casual", "religion",
]

OPENPII_ATTRS: List[str] = ["age", "gender", "location", "identity"]

DATASET_ATTRS: Dict[str, List[str]] = {
    "SynthPAI": SYNTHPAI_ATTRS,
    "HR-VISPR": HRVISPR_ATTRS,
    "OpenPII":  OPENPII_ATTRS,
}

# ── Difficulty helpers ────────────────────────────────────────────────────────

def _hrvispr_difficulty(gt: int) -> str:
    return "explicit" if gt == 1 else "none"


def _openpii_difficulty(gt: int) -> str:
    return "explicit" if gt == 1 else "none"


# ── Per-dataset sample generators ─────────────────────────────────────────────

def _synthpai_samples(
    max_items: Optional[int],
    attrs: List[str],
    seed: int = 42,
) -> Generator[Dict[str, Any], None, None]:
    # Collect all valid items first, then shuffle to avoid ordering bias
    # (the SynthPAI dataset is sorted such that the first N items are
    # heavily skewed toward a single attribute being positive).
    import random
    all_items = []
    for ok, item, _err in iter_dataset("SynthPAI", "text"):
        if not ok:
            continue
        text = item.get("text_content", "") or ""
        if not text.strip():
            continue
        all_items.append(item)

    random.Random(seed).shuffle(all_items)
    if max_items is not None:
        all_items = all_items[:max_items]

    for count, item in enumerate(all_items):
        item_id = item.get("filename", f"sp_{count:05d}")
        text = item.get("text_content", "") or ""
        for attr in attrs:
            difficulty = _label_mapper.synthpai_attr_difficulty(item, attr)
            gt = 0 if difficulty == "none" else 1
            yield {
                "id":           f"{item_id}__{attr}",
                "dataset":      "SynthPAI",
                "text_content": text,
                "image_b64":    "",
                "attribute":    attr,
                "ground_truth": gt,
                "difficulty":   difficulty,
                "item_id":      item_id,
            }



def _hrvispr_samples(
    max_items: Optional[int],
    attrs: List[str],
) -> Generator[Dict[str, Any], None, None]:
    count = 0
    for ok, item, _err in iter_dataset("HR-VISPR", "image", max_items=max_items):
        if not ok:
            continue
        image_b64 = item.get("image_base64", "") or ""
        if not image_b64:
            continue
        labels = get_input_labels(item, attrs)
        item_id = item.get("filename", f"hv_{count:05d}")
        for attr in attrs:
            gt = labels.get(attr, 0)
            yield {
                "id":           f"{item_id}__{attr}",
                "dataset":      "HR-VISPR",
                "text_content": "",
                "image_b64":    image_b64,
                "attribute":    attr,
                "ground_truth": gt,
                "difficulty":   _hrvispr_difficulty(gt),
                "item_id":      item_id,
            }
        count += 1


def _openpii_samples(
    max_items: Optional[int],
    attrs: List[str],
) -> Generator[Dict[str, Any], None, None]:
    if get_dataset_path("OpenPII") is None:
        return
    count = 0
    for ok, item, _err in iter_dataset("OpenPII", "text", max_items=max_items):
        if not ok:
            continue
        text = item.get("text_content", "") or ""
        if not text.strip():
            continue
        labels = get_input_labels(item, attrs)
        item_id = item.get("filename", f"op_{count:05d}")
        for attr in attrs:
            gt = labels.get(attr, 0)
            yield {
                "id":           f"{item_id}__{attr}",
                "dataset":      "OpenPII",
                "text_content": text,
                "image_b64":    "",
                "attribute":    attr,
                "ground_truth": gt,
                "difficulty":   _openpii_difficulty(gt),
                "item_id":      item_id,
            }
        count += 1


# ── Public API ────────────────────────────────────────────────────────────────

def load_judge_samples(
    dataset_name: str,
    attribute_filter: Optional[List[str]] = None,
    difficulty_filter: Optional[List[str]] = None,
    max_items: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load validation samples from *dataset_name*.

    Args:
        dataset_name:      "SynthPAI", "HR-VISPR", or "OpenPII"
        attribute_filter:  if set, keep only these attributes
        difficulty_filter: if set, keep only samples with these difficulty values
        max_items:         maximum *items* (posts/images/texts) to load before
                           expanding to per-attribute samples
        max_samples:       cap on total returned samples (after filtering)

    Returns a list of sample dicts (see module docstring).
    """
    all_attrs = DATASET_ATTRS.get(dataset_name, [])
    attrs = [a for a in all_attrs if not attribute_filter or a in attribute_filter]
    if not attrs:
        return []

    if dataset_name == "SynthPAI":
        gen = _synthpai_samples(max_items, attrs)
    elif dataset_name == "HR-VISPR":
        gen = _hrvispr_samples(max_items, attrs)
    elif dataset_name == "OpenPII":
        gen = _openpii_samples(max_items, attrs)
    else:
        return []

    samples: List[Dict[str, Any]] = []
    for sample in gen:
        if difficulty_filter and sample["difficulty"] not in difficulty_filter:
            continue
        samples.append(sample)
        if max_samples and len(samples) >= max_samples:
            break

    return samples


def available_datasets() -> List[str]:
    """Return the list of datasets that exist on disk."""
    names = ["SynthPAI", "HR-VISPR", "OpenPII"]
    return [n for n in names if get_dataset_path(n) is not None]
