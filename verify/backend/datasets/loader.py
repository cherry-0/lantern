"""
Dataset loading utilities for Verify.

Loads items from dataset/<dataset_name>/ directories.
Supports image, text, and video modalities.
"""

import base64
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from verify.backend.utils.config import get_dataset_path

# Supported file extensions by modality
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".json", ".csv", ".md"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Number of frames to sample from a video
VIDEO_FRAME_COUNT = 4


# HR-VISPR 18-class label names (index → attribute)
_HRVISPR_18_CLASSES = [
    "age", "face", "color", "haircolor", "gender", "race", "nudity",
    "height", "weight", "disability", "ethnic_clothing", "formal",
    "uniforms", "medical", "troupe", "sports", "casual", "religion",
]

# Preferred split order for HR-VISPR (pick the first one that exists)
_HRVISPR_SPLIT_PREFERENCE = ["val2017", "test2017", "train2017"]


def _is_subdir_image_dataset(dataset_path: Path) -> bool:
    """Return True if the dataset stores images inside named subdirectories."""
    return any(
        sub.is_dir() and any(f.suffix.lower() in IMAGE_EXTENSIONS for f in sub.iterdir())
        for sub in dataset_path.iterdir()
        if sub.is_dir() and not sub.name.endswith("_labels") and not sub.name.endswith("_pkl_labels")
    )


def _load_hrvispr_pkl_labels(dataset_path: Path, split: str) -> Dict[str, List[str]]:
    """
    Load 18-class PKL labels for a given split.
    Returns {image_id: [label_name, ...]} with only positive labels.
    """
    pkl_path = dataset_path / "18_class_pkl_labels" / f"{split}_labels.pkl"
    if not pkl_path.exists():
        return {}
    try:
        import pickle
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        result: Dict[str, List[str]] = {}
        for img_id, vec in data.items():
            labels = [_HRVISPR_18_CLASSES[i] for i, v in enumerate(vec) if v > 0]
            result[img_id] = labels
        return result
    except Exception:
        return {}


def _iter_subdir_image_dataset(
    dataset_path: Path,
) -> Generator[Tuple[bool, Dict[str, Any], Optional[str]], None, None]:
    """
    Load an HR-VISPR-style dataset: images in split subdirs, labels in JSON or PKL.
    Yields one image item per file, with privacy labels attached as metadata.
    """
    # Pick the first available split
    split = next(
        (s for s in _HRVISPR_SPLIT_PREFERENCE if (dataset_path / s).is_dir()),
        None,
    )
    if split is None:
        yield False, {}, "No recognised split directory found in dataset."
        return

    image_dir = dataset_path / split
    label_dir = dataset_path / f"{split}_labels"
    has_json_labels = label_dir.is_dir() and any(label_dir.iterdir())

    # Load PKL labels as fallback (always available for all splits)
    pkl_labels = _load_hrvispr_pkl_labels(dataset_path, split)

    image_files = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_files:
        img_id = img_path.stem  # e.g. "2017_67135519"

        # Load labels: prefer per-image JSON, fall back to PKL
        privacy_labels: List[str] = []
        label_source = "none"

        if has_json_labels:
            json_path = label_dir / f"{img_id}.json"
            if json_path.exists():
                try:
                    raw = json.loads(json_path.read_text())
                    # Strip the "a{N}_" prefix to get clean attribute names
                    privacy_labels = [
                        lbl.split("_", 1)[1] if "_" in lbl else lbl
                        for lbl in raw.get("labels", [])
                    ]
                    label_source = "json"
                except Exception:
                    pass

        if not privacy_labels and img_id in pkl_labels:
            privacy_labels = pkl_labels[img_id]
            label_source = "pkl_18class"

        # Load image
        ok, item, err = _load_image_item(img_path, {
            "modality": "image",
            "path": str(img_path),
            "filename": img_path.name,
        })

        if ok:
            item["privacy_labels"] = privacy_labels
            item["label_source"] = label_source
            item["split"] = split
            item["image_id"] = img_id

        yield ok, item, err


def _is_hf_dataset(dataset_path: Path) -> bool:
    """Return True if the directory is a HuggingFace dataset saved to disk."""
    return (dataset_path / "dataset_dict.json").exists() or any(
        list(sub.glob("*.arrow"))
        for sub in dataset_path.iterdir()
        if sub.is_dir()
    )


def detect_modality(dataset_name: str) -> Optional[str]:
    """
    Heuristically detect the modality of a dataset by inspecting its files.
    Returns "image", "text", "video", or None if ambiguous/empty.
    """
    dataset_path = get_dataset_path(dataset_name)
    if dataset_path is None:
        return None

    # HuggingFace disk datasets are always treated as text
    if _is_hf_dataset(dataset_path):
        return "text"

    # Subdirectory image datasets (e.g. HR-VISPR)
    if _is_subdir_image_dataset(dataset_path):
        return "image"

    counts = {"image": 0, "text": 0, "video": 0}
    for f in dataset_path.iterdir():
        ext = f.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            counts["image"] += 1
        elif ext in TEXT_EXTENSIONS:
            counts["text"] += 1
        elif ext in VIDEO_EXTENSIONS:
            counts["video"] += 1

    # Return the dominant modality
    dominant = max(counts, key=lambda k: counts[k])
    return dominant if counts[dominant] > 0 else None


def list_dataset_items(dataset_name: str, modality: str) -> List[Path]:
    """
    List all dataset item file paths for the given dataset and modality.
    Returns sorted list of file paths.
    HuggingFace Arrow datasets are not file-per-item; use iter_dataset instead.
    """
    dataset_path = get_dataset_path(dataset_name)
    if dataset_path is None:
        return []

    if _is_hf_dataset(dataset_path):
        return []  # rows are accessed via iter_dataset

    ext_set = {
        "image": IMAGE_EXTENSIONS,
        "text": TEXT_EXTENSIONS,
        "video": VIDEO_EXTENSIONS,
    }.get(modality, set())

    items = [
        f
        for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in ext_set
    ]
    return sorted(items, key=lambda p: p.name)


def load_item(
    path: Path, modality: str
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Load a single dataset item.

    Returns:
        (success, item_dict, error_message)

    item_dict keys:
        - modality: str
        - path: str
        - filename: str
        - data: loaded data (PIL.Image | str | list[PIL.Image])
        - image_base64: str (for image modality, convenience)
        - text_content: str (for text modality)
        - frames: list[PIL.Image] (for video modality)
        - raw: original parsed content (for text)
    """
    item: Dict[str, Any] = {
        "modality": modality,
        "path": str(path),
        "filename": path.name,
    }

    try:
        if modality == "image":
            return _load_image_item(path, item)
        elif modality == "text":
            return _load_text_item(path, item)
        elif modality == "video":
            return _load_video_item(path, item)
        else:
            return False, item, f"Unsupported modality: {modality}"
    except Exception as e:
        return False, item, str(e)


def _load_image_item(
    path: Path, item: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    try:
        from PIL import Image as PILImage
        import io

        img = PILImage.open(str(path)).convert("RGB")
        item["data"] = img

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        item["image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        return True, item, None
    except ImportError:
        return False, item, "Pillow (PIL) is required to load images. Please install it."
    except Exception as e:
        return False, item, f"Failed to load image: {e}"


def _load_text_item(
    path: Path, item: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    ext = path.suffix.lower()

    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, item, f"Failed to read file: {e}"

    if ext == ".json":
        try:
            data = json.loads(raw_text)
            item["raw"] = data
            # Extract the most relevant text content
            item["text_content"] = _extract_text_from_json(data)
            item["data"] = item["text_content"]
        except json.JSONDecodeError as e:
            return False, item, f"Invalid JSON in {path.name}: {e}"

    elif ext == ".csv":
        try:
            import csv
            import io

            reader = csv.DictReader(io.StringIO(raw_text))
            rows = list(reader)
            item["raw"] = rows
            # Flatten CSV into text
            lines = []
            for row in rows:
                lines.append(", ".join(f"{k}: {v}" for k, v in row.items()))
            item["text_content"] = "\n".join(lines)
            item["data"] = item["text_content"]
        except Exception as e:
            return False, item, f"Failed to parse CSV: {e}"

    elif ext in {".txt", ".md"}:
        item["raw"] = raw_text
        item["text_content"] = raw_text
        item["data"] = raw_text

    else:
        return False, item, f"Unsupported text file extension: {ext}"

    return True, item, None


def _extract_text_from_json(data: Any) -> str:
    """
    Extract a useful text representation from a JSON object.
    Handles the PrivacyLens dataset format and generic dicts/lists.
    """
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        # PrivacyLens-specific: use Vignette + Tool-Use Agent Trajectory
        parts = []
        for key in ["Vignette", "Tool-Use Agent Trajectory", "Seed"]:
            if key in data and data[key]:
                parts.append(f"[{key}]\n{data[key]}")
        if parts:
            return "\n\n".join(parts)

        # Generic: join all string values
        parts = []
        for k, v in data.items():
            if isinstance(v, str):
                parts.append(f"{k}: {v}")
            elif isinstance(v, (int, float, bool)):
                parts.append(f"{k}: {v}")
        return "\n".join(parts)

    if isinstance(data, list):
        return "\n".join(_extract_text_from_json(item) for item in data)

    return str(data)


def _load_video_item(
    path: Path, item: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    try:
        import cv2
        import numpy as np
        from PIL import Image as PILImage

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False, item, f"Cannot open video file: {path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return False, item, "Video has no frames."

        # Sample VIDEO_FRAME_COUNT evenly spaced frames
        indices = [
            int(i * total_frames / VIDEO_FRAME_COUNT)
            for i in range(VIDEO_FRAME_COUNT)
        ]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # cv2 → PIL (BGR → RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(PILImage.fromarray(frame_rgb))

        cap.release()

        if not frames:
            return False, item, "Could not extract any frames from video."

        item["frames"] = frames
        item["data"] = frames
        return True, item, None

    except ImportError:
        return (
            False,
            item,
            "opencv-python (cv2) and Pillow are required to load videos.",
        )
    except Exception as e:
        return False, item, f"Failed to load video: {e}"


def _hf_row_to_item(row: Dict[str, Any], source: str, idx: int) -> Dict[str, Any]:
    """
    Convert a single HuggingFace dataset row into a Verify item dict.

    Handles the PrivacyLens schema (seed / vignette / trajectory structs) and
    falls back to a generic JSON dump for other HF datasets.
    """
    item: Dict[str, Any] = {
        "modality": "text",
        "path": source,
        "filename": f"row_{idx:05d}",
        "raw": row,
    }

    # PrivacyLens schema: nested seed / vignette / trajectory dicts
    if "vignette" in row and "trajectory" in row:
        seed = row.get("seed") or {}
        vignette = row.get("vignette") or {}
        trajectory = row.get("trajectory") or {}

        seed_str = (
            f"(Data Type: {seed.get('data_type', '')}, "
            f"Data Subject: {seed.get('data_subject', '')}, "
            f"Data Sender: {seed.get('data_sender', '')}, "
            f"Data Recipient: {seed.get('data_recipient', '')}, "
            f"Transmission Principle: {seed.get('transmission_principle', '')})"
        )

        toolkits = trajectory.get("toolkits") or []
        sensitive_items = trajectory.get("sensitive_info_items") or []

        trajectory_str = (
            f"User Name: {trajectory.get('user_name', '')}\n\n"
            f"User Email: {trajectory.get('user_email', '')}\n\n"
            f"User Instruction: {trajectory.get('user_instruction', '')}\n\n"
            f"Toolkits: {','.join(toolkits)}\n\n"
            f"Executable Trajectory:\n{trajectory.get('executable_trajectory', '')}\n\n"
            f"Final Action:\n{trajectory.get('final_action', '')}"
        )

        parts = [
            f"[Seed]\n{seed_str}",
            f"[Vignette]\n{vignette.get('story', '')}",
            f"[Tool-Use Agent Trajectory]\n{trajectory_str}",
        ]
        if sensitive_items:
            parts.append(f"[Sensitive Info Items]\n" + "\n".join(f"- {s}" for s in sensitive_items))

        text_content = "\n\n".join(parts)
        item["name"] = row.get("name", f"item_{idx}")
        item["seed"] = seed
        item["vignette"] = vignette
        item["trajectory"] = trajectory
    else:
        # Generic fallback: dump everything as JSON text
        text_content = json.dumps(row, ensure_ascii=False, indent=2)

    item["text_content"] = text_content
    item["data"] = text_content
    return item


def _iter_hf_dataset(
    dataset_path: Path,
) -> Generator[Tuple[bool, Dict[str, Any], Optional[str]], None, None]:
    """Load a HuggingFace dataset saved to disk and yield one item per row."""
    try:
        from datasets import load_from_disk  # type: ignore
    except ImportError:
        yield False, {}, "The 'datasets' library is required. Install with: pip install datasets"
        return

    try:
        ds_dict = load_from_disk(str(dataset_path))
    except Exception as e:
        yield False, {}, f"Failed to load HuggingFace dataset from {dataset_path}: {e}"
        return

    # Iterate over all splits (train / validation / test)
    splits = ds_dict.keys() if hasattr(ds_dict, "keys") else ["train"]
    for split in splits:
        split_ds = ds_dict[split] if hasattr(ds_dict, "__getitem__") else ds_dict
        source = str(dataset_path / split)
        for idx, row in enumerate(split_ds):
            try:
                item = _hf_row_to_item(dict(row), source, idx)
                yield True, item, None
            except Exception as e:
                yield False, {"modality": "text", "path": source, "filename": f"row_{idx:05d}"}, str(e)


def count_dataset_items(dataset_name: str, modality: str) -> int:
    """
    Return the total number of items in a dataset without loading all data.
    For HuggingFace datasets, reads num_examples from dataset_info.json.
    For flat file datasets, counts matching files.
    """
    dataset_path = get_dataset_path(dataset_name)
    if dataset_path is None:
        return 0

    if _is_hf_dataset(dataset_path):
        total = 0
        for split_dir in dataset_path.iterdir():
            if not split_dir.is_dir():
                continue
            info_file = split_dir / "dataset_info.json"
            if info_file.exists():
                try:
                    info = json.loads(info_file.read_text())
                    for split_info in info.get("splits", {}).values():
                        total += split_info.get("num_examples", 0)
                except Exception:
                    pass
        return total

    if _is_subdir_image_dataset(dataset_path):
        split = next(
            (s for s in _HRVISPR_SPLIT_PREFERENCE if (dataset_path / s).is_dir()),
            None,
        )
        if split:
            return sum(
                1 for f in (dataset_path / split).iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            )
        return 0

    return len(list_dataset_items(dataset_name, modality))


def iter_dataset(
    dataset_name: str, modality: str, max_items: Optional[int] = None
) -> Generator[Tuple[bool, Dict[str, Any], Optional[str]], None, None]:
    """
    Generator that yields (success, item_dict, error) for each item in the dataset.
    Handles both flat file datasets and HuggingFace Arrow datasets.

    Args:
        max_items: if set, stop after yielding this many items.
    """
    dataset_path = get_dataset_path(dataset_name)
    if dataset_path is None:
        return

    count = 0
    if _is_hf_dataset(dataset_path):
        source = _iter_hf_dataset(dataset_path)
    elif _is_subdir_image_dataset(dataset_path):
        source = _iter_subdir_image_dataset(dataset_path)
    else:
        source = (load_item(path, modality) for path in list_dataset_items(dataset_name, modality))
    for item in source:
        yield item
        count += 1
        if max_items is not None and count >= max_items:
            break
