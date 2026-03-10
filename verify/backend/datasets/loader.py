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


def detect_modality(dataset_name: str) -> Optional[str]:
    """
    Heuristically detect the modality of a dataset by inspecting its files.
    Returns "image", "text", "video", or None if ambiguous/empty.
    """
    dataset_path = get_dataset_path(dataset_name)
    if dataset_path is None:
        return None

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
    """
    dataset_path = get_dataset_path(dataset_name)
    if dataset_path is None:
        return []

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


def iter_dataset(
    dataset_name: str, modality: str
) -> Generator[Tuple[bool, Dict[str, Any], Optional[str]], None, None]:
    """
    Generator that yields (success, item_dict, error) for each item in the dataset.
    """
    items = list_dataset_items(dataset_name, modality)
    if not items:
        return

    for path in items:
        yield load_item(path, modality)
