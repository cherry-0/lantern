"""
Dataset loading utilities for Verify.

Loads items from dataset/<dataset_name>/ directories.
Supports image, text, and video modalities.
"""

import base64
import functools
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

# SROIE2019 entity fields and their corresponding generic privacy attributes
SROIE_ENTITY_TO_ATTR = {
    "company": "identity",
    "address": "location",
    "date": "date",
    "total": "total",
}

# Preferred split order for SROIE2019
_SROIE_SPLIT_PREFERENCE = ["test", "train"]


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


def _is_sroie_dataset(dataset_path: Path) -> bool:
    """Return True if the directory follows the SROIE2019 layout (split/img/)."""
    return any(
        (dataset_path / split / "img").is_dir() and any(
            f.suffix.lower() in IMAGE_EXTENSIONS
            for f in (dataset_path / split / "img").iterdir()
        )
        for split in _SROIE_SPLIT_PREFERENCE
    )


def _load_sroie_entities(entities_dir: Path, stem: str) -> Dict[str, Any]:
    """
    Load the SROIE entities JSON for a given image stem.
    Returns a dict with keys: company, date, address, total (empty strings if missing).
    Returns an empty dict if the file doesn't exist.
    """
    entity_path = entities_dir / f"{stem}.txt"
    if not entity_path.exists():
        return {}
    try:
        return json.loads(entity_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _iter_sroie_dataset(
    dataset_path: Path,
) -> Generator[Tuple[bool, Dict[str, Any], Optional[str]], None, None]:
    """
    Load a SROIE2019-style dataset.

    Structure:
        <split>/img/<id>.jpg       — receipt image
        <split>/entities/<id>.txt  — JSON ground truth: {company, date, address, total}

    Yields one image item per receipt. Entity fields are stored as:
        - privacy_labels: list of entity field names that are non-empty
                          (e.g. ["company", "address", "date", "total"])
        - sroie_entities: full dict {"company": ..., "date": ..., "address": ..., "total": ...}
        - sroie_entity_attrs: mapped generic attribute names (e.g. ["identity", "location"])
    """
    split = next(
        (s for s in _SROIE_SPLIT_PREFERENCE if (dataset_path / s / "img").is_dir()),
        None,
    )
    if split is None:
        yield False, {}, "No recognised split directory (test/, train/) found in SROIE dataset."
        return

    img_dir = dataset_path / split / "img"
    entities_dir = dataset_path / split / "entities"

    image_files = sorted(
        f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_files:
        stem = img_path.stem

        # Load ground-truth entities
        entities = _load_sroie_entities(entities_dir, stem)

        # Derive privacy labels from non-empty entity fields
        privacy_labels = [
            field for field in ("company", "date", "address", "total")
            if entities.get(field, "").strip()
        ]
        entity_attrs = [
            SROIE_ENTITY_TO_ATTR[field]
            for field in privacy_labels
            if field in SROIE_ENTITY_TO_ATTR
        ]

        ok, item, err = _load_image_item(img_path, {
            "modality": "image",
            "path": str(img_path),
            "filename": img_path.name,
        })

        if ok:
            item["privacy_labels"] = privacy_labels
            item["sroie_entities"] = entities
            item["sroie_entity_attrs"] = entity_attrs
            item["label_source"] = "sroie_entities"
            item["split"] = split
            item["image_id"] = stem

        yield ok, item, err


def _is_hf_dataset(dataset_path: Path) -> bool:
    """Return True if the directory is a HuggingFace dataset saved to disk."""
    return (dataset_path / "dataset_dict.json").exists() or any(
        list(sub.glob("*.arrow"))
        for sub in dataset_path.iterdir()
        if sub.is_dir()
    )


def _is_hf_image_dataset(dataset_path: Path) -> bool:
    """Return True if any HF split in the directory has an Image feature column."""
    for sub in dataset_path.iterdir():
        if not sub.is_dir():
            continue
        info_file = sub / "dataset_info.json"
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
                for feat_info in info.get("features", {}).values():
                    if isinstance(feat_info, dict) and feat_info.get("_type") == "Image":
                        return True
            except Exception:
                pass
    return False


def _is_openpii_dataset(dataset_path: Path) -> bool:
    """Return True if the directory contains the OpenPII JSONL nested structure."""
    for sub in dataset_path.iterdir():
        if not sub.is_dir():
            continue
        data_dir = sub / "data"
        if data_dir.is_dir():
            for split_dir in data_dir.iterdir():
                if split_dir.is_dir() and any(split_dir.glob("*.jsonl")):
                    return True
    return False


def _is_mimicxr_dataset(dataset_path: Path) -> bool:
    """Return True if the directory contains a MIMIC-CXR layout (CSV + official_data_iccv_final/)."""
    for sub in dataset_path.iterdir():
        if sub.is_dir() and (sub / "official_data_iccv_final").is_dir():
            return any(sub.glob("*.csv"))
    return False


def _is_multicare_dataset(dataset_path: Path) -> bool:
    """Return True if the directory contains a MultiCaRe cases.parquet file."""
    return (dataset_path / "cases.parquet").exists()


def _is_asapaes_dataset(dataset_path: Path) -> bool:
    """Return True if the directory contains ASAP-AES TSV files in a subdirectory."""
    for sub in dataset_path.iterdir():
        if sub.is_dir() and any(sub.glob("*.tsv")):
            return True
    return False


def _parse_stringified_list(s: str) -> List[str]:
    """Parse a stringified Python list (e.g. \"['a', 'b']\") back to a Python list."""
    import ast
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return [str(x) for x in result]
    except Exception:
        pass
    return []


def _extract_study_id(img_path: str) -> str:
    """
    Extract the study ID (e.g. 's50414267') from a MIMIC-CXR image path.
    Path format: files/p{10}/p{subject_id}/s{study_id}/{image}.jpg
    """
    for part in img_path.replace("\\", "/").split("/"):
        if part.startswith("s") and part[1:].isdigit():
            return part
    return ""


def detect_modality(dataset_name: str) -> Optional[str]:
    """
    Heuristically detect the modality of a dataset by inspecting its files.
    Returns "image", "text", "video", or None if ambiguous/empty.
    """
    dataset_path = get_dataset_path(dataset_name)
    if dataset_path is None:
        return None

    # MIMIC-CXR: image
    if _is_mimicxr_dataset(dataset_path):
        return "image"

    # MultiCaRe: text
    if _is_multicare_dataset(dataset_path):
        return "text"

    # OpenPII: text
    if _is_openpii_dataset(dataset_path):
        return "text"

    # ASAP-AES: text
    if _is_asapaes_dataset(dataset_path):
        return "text"

    # HuggingFace disk datasets — image if any split has an Image feature, else text
    if _is_hf_dataset(dataset_path):
        if _is_hf_image_dataset(dataset_path):
            return "image"
        return "text"

    # SROIE2019-style datasets (split/img/)
    if _is_sroie_dataset(dataset_path):
        return "image"

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


@functools.lru_cache(maxsize=1)
def _load_privacylens_mapping() -> Dict[str, Any]:
    """Load the pre-built data_type → attribute mapping PKL (cached after first load)."""
    pkl_path = Path(__file__).resolve().parent / "PrivacyLens" / "data_type_mapping.pkl"
    if not pkl_path.exists():
        return {}
    try:
        import pickle
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _hf_row_to_item(row: Dict[str, Any], source: str, idx: int) -> Dict[str, Any]:
    """
    Convert a single HuggingFace dataset row into a Verify item dict.

    Handles the PrivacyLens schema (seed / vignette / trajectory structs),
    the SynthPAI schema (text + profile + id), and falls back to a generic
    JSON dump for other HF datasets.
    """
    item: Dict[str, Any] = {
        "modality": "text",
        "path": source,
        "filename": f"row_{idx:05d}",
        "raw": row,
    }

    # GretelSyntheticPII schema: synthetic document with PII spans
    if "generated_text" in row and "pii_spans" in row:
        generated_text = str(row.get("generated_text") or "")
        pii_spans_raw = row.get("pii_spans", "[]")
        if isinstance(pii_spans_raw, list):
            pii_spans = pii_spans_raw
        elif isinstance(pii_spans_raw, str):
            try:
                pii_spans = json.loads(pii_spans_raw)
            except Exception:
                pii_spans = []
        else:
            pii_spans = []
        item["filename"] = str(row.get("index", idx))
        item["text_content"] = generated_text
        item["data"] = generated_text
        item["gretel_pii_spans"] = pii_spans
        item["label_source"] = "gretel_pii"
        item["document_type"] = str(row.get("document_type") or "")
        item["domain"] = str(row.get("domain") or "")
        return item

    # MultiPriv schema: only an 'image' field (PIL image decoded by HF)
    pil_img = row.get("image")
    if pil_img is not None and hasattr(pil_img, "mode"):
        import io as _io
        item["modality"] = "image"
        item["filename"] = f"img_{idx:05d}"
        item["label_source"] = "multiprivate"
        try:
            rgb = pil_img.convert("RGB")
            item["data"] = rgb
            buf = _io.BytesIO()
            rgb.save(buf, format="JPEG", quality=85)
            item["image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            item["image_base64"] = ""
            item["data"] = None
        return item

    # SynthPAI schema: each row is a Reddit-style post from a synthetic persona.
    # Key fields: text (post content), profile (ground-truth private attributes), id.
    profile = row.get("profile")
    if "text" in row and isinstance(profile, dict) and "sex" in profile:
        post_text = row.get("text", "")
        persona = row.get("username") or row.get("author") or ""
        thread = row.get("thread_id", "")
        item["filename"] = row.get("id") or f"row_{idx:05d}"
        item["synthpai_profile"] = profile
        item["synthpai_thread"] = thread
        item["synthpai_author"] = row.get("author", "")
        item["synthpai_username"] = persona
        item["label_source"] = "synthpai"
        item["text_content"] = post_text
        item["data"] = post_text
        return item

    # PrivacyLens schema: nested seed / vignette / trajectory dicts
    if "vignette" in row and "trajectory" in row:
        seed = row.get("seed") or {}
        vignette = row.get("vignette") or {}
        # Attach mapped privacy attributes from the pre-built PKL
        _mapping = _load_privacylens_mapping()
        _dt2attrs = _mapping.get("data_type_to_attrs", {})
        _data_type = (seed.get("data_type") or "").strip()
        item["data_type"] = _data_type
        item["data_type_attributes"] = _dt2attrs.get(_data_type, [])
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


def _iter_openpii_dataset(
    dataset_path: Path,
) -> Generator[Tuple[bool, Dict[str, Any], Optional[str]], None, None]:
    """
    Load the OpenPII (ai4privacy) dataset from nested JSONL files.

    Structure:
        <name>/data/train/train.jsonl
        <name>/data/validation/test.jsonl

    Each row: source_text (PII text), masked_text, privacy_mask (span list),
              uid, language, split.
    """
    jsonl_files: List[Path] = []
    for sub in sorted(dataset_path.iterdir()):
        if not sub.is_dir():
            continue
        data_dir = sub / "data"
        if not data_dir.is_dir():
            continue
        for split_dir in sorted(data_dir.iterdir()):
            if split_dir.is_dir():
                jsonl_files.extend(sorted(split_dir.glob("*.jsonl")))

    if not jsonl_files:
        yield False, {}, "No JSONL files found in OpenPII dataset."
        return

    for jf in jsonl_files:
        with open(jf, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as e:
                    yield False, {"modality": "text", "path": str(jf), "filename": "?"}, str(e)
                    continue

                uid = str(row.get("uid", ""))
                source_text = str(row.get("source_text") or "")
                privacy_mask = row.get("privacy_mask", [])

                item: Dict[str, Any] = {
                    "modality": "text",
                    "path": str(jf),
                    "filename": uid or f"uid_{uid}",
                    "text_content": source_text,
                    "data": source_text,
                    "raw": row,
                    "openpii_spans": privacy_mask if isinstance(privacy_mask, list) else [],
                    "label_source": "openpii",
                    "language": str(row.get("language") or ""),
                    "openpii_uid": uid,
                }
                yield True, item, None


def _iter_mimicxr_dataset(
    dataset_path: Path,
) -> Generator[Tuple[bool, Dict[str, Any], Optional[str]], None, None]:
    """
    Load MIMIC-CXR: one item per image file, with the study's text_augment attached.

    Structure:
        <name>/mimic_cxr_aug_train.csv  (or _validate.csv)
        <name>/official_data_iccv_final/files/p{10}/p{subject}/s{study}/{img}.jpg

    CSV columns used: subject_id, image (list of paths), text_augment (list of reports).
    Image paths in CSV are relative to official_data_iccv_final/.
    """
    import ast
    import csv as csv_module

    ds_dir: Optional[Path] = None
    for sub in dataset_path.iterdir():
        if sub.is_dir() and (sub / "official_data_iccv_final").is_dir():
            ds_dir = sub
            break
    if ds_dir is None:
        yield False, {}, "MIMIC-CXR: official_data_iccv_final/ subdirectory not found."
        return

    img_root = ds_dir / "official_data_iccv_final"

    # Prefer train CSV, then validate
    csv_files = sorted(ds_dir.glob("*.csv"), key=lambda p: (0 if "train" in p.name else 1))
    if not csv_files:
        yield False, {}, "MIMIC-CXR: no CSV files found."
        return

    for csv_path in csv_files:
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                subject_id = str(row.get("subject_id", ""))

                image_paths = _parse_stringified_list(row.get("image", "[]"))
                text_augments = _parse_stringified_list(row.get("text_augment", "[]"))

                # Build study_id → text_augment mapping (studies in order of first appearance)
                study_ids_ordered: List[str] = []
                seen: set = set()
                for p in image_paths:
                    sid = _extract_study_id(p)
                    if sid and sid not in seen:
                        study_ids_ordered.append(sid)
                        seen.add(sid)

                study_to_text: Dict[str, str] = {
                    sid: (text_augments[i] if i < len(text_augments) else "")
                    for i, sid in enumerate(study_ids_ordered)
                }

                for img_path_str in image_paths:
                    full_path = img_root / img_path_str
                    if not full_path.exists():
                        continue

                    sid = _extract_study_id(img_path_str)
                    text_content = study_to_text.get(sid, "")

                    base_item: Dict[str, Any] = {
                        "modality": "image",
                        "path": str(full_path),
                        "filename": full_path.name,
                        "text_content": text_content,
                        "label_source": "mimicxr",
                        "subject_id": subject_id,
                        "study_id": sid,
                    }

                    ok, item, err = _load_image_item(full_path, base_item)
                    if ok:
                        # Ensure text_content survives _load_image_item (it modifies in place)
                        item["text_content"] = text_content
                    yield ok, item, err


def _iter_multicare_dataset(
    dataset_path: Path,
) -> Generator[Tuple[bool, Dict[str, Any], Optional[str]], None, None]:
    """
    Load MultiCaRe clinical case reports from cases.parquet.

    Columns used: article_id, case_id, case_text.
    All labels are all-zeros (no pre-labelled PII attributes).
    """
    parquet_path = dataset_path / "cases.parquet"
    try:
        import pandas as pd
        df = pd.read_parquet(str(parquet_path))
    except ImportError:
        yield False, {}, "pandas + pyarrow required for MultiCaRe. Install: pip install pandas pyarrow"
        return
    except Exception as e:
        yield False, {}, f"MultiCaRe: failed to load cases.parquet: {e}"
        return

    for _, row in df.iterrows():
        case_id = str(row.get("case_id", ""))
        article_id = str(row.get("article_id", ""))
        case_text = str(row.get("case_text") or "")

        item: Dict[str, Any] = {
            "modality": "text",
            "path": str(parquet_path),
            "filename": case_id,
            "text_content": case_text,
            "data": case_text,
            "raw": {"case_id": case_id, "article_id": article_id},
            "label_source": "multicare",
            "article_id": article_id,
            "case_id": case_id,
        }
        yield True, item, None


def _iter_asapaes_dataset(
    dataset_path: Path,
) -> Generator[Tuple[bool, Dict[str, Any], Optional[str]], None, None]:
    """
    Load ASAP-AES student essays from TSV files.

    Structure: <name>/asap-aes/training_set_rel3.tsv  (and valid_set.tsv)
    Columns used: essay_id, essay_set, essay, domain1_score.
    All labels are all-zeros (no pre-labelled PII attributes).
    """
    import csv as csv_module

    tsv_files: List[Path] = []
    for sub in sorted(dataset_path.iterdir()):
        if sub.is_dir():
            tsv_files.extend(sorted(sub.glob("*.tsv")))

    # Prefer training set first
    training = [f for f in tsv_files if "training" in f.name]
    other = [f for f in tsv_files if "training" not in f.name]
    ordered = training + other

    if not ordered:
        yield False, {}, "ASAP-AES: no TSV files found."
        return

    for tsv_path in ordered:
        with open(tsv_path, encoding="utf-8", errors="replace") as f:
            reader = csv_module.DictReader(f, delimiter="\t")
            for row in reader:
                essay_id = str(row.get("essay_id", ""))
                essay_set = str(row.get("essay_set", ""))
                essay = str(row.get("essay") or "").strip()
                if not essay:
                    continue
                score = str(row.get("domain1_score") or "")

                item: Dict[str, Any] = {
                    "modality": "text",
                    "path": str(tsv_path),
                    "filename": f"essay_{essay_id}",
                    "text_content": essay,
                    "data": essay,
                    "raw": dict(row),
                    "label_source": "asap_aes",
                    "essay_id": essay_id,
                    "essay_set": essay_set,
                    "domain1_score": score,
                }
                yield True, item, None


def count_dataset_items(dataset_name: str, modality: str) -> int:
    """
    Return the total number of items in a dataset without loading all data.
    For HuggingFace datasets, reads num_examples from dataset_info.json.
    For flat file datasets, counts matching files.
    """
    dataset_path = get_dataset_path(dataset_name)
    if dataset_path is None:
        return 0

    if _is_mimicxr_dataset(dataset_path):
        import csv as _csv
        for sub in dataset_path.iterdir():
            if sub.is_dir() and (sub / "official_data_iccv_final").is_dir():
                csv_files = sorted(sub.glob("*.csv"), key=lambda p: (0 if "train" in p.name else 1))
                total = 0
                for csv_path in csv_files:
                    try:
                        with open(csv_path, encoding="utf-8", errors="replace") as f:
                            for row in _csv.DictReader(f):
                                total += len(_parse_stringified_list(row.get("image", "[]")))
                    except Exception:
                        pass
                return total
        return 0

    if _is_multicare_dataset(dataset_path):
        try:
            import pandas as _pd
            return len(_pd.read_parquet(str(dataset_path / "cases.parquet")))
        except Exception:
            return 0

    if _is_openpii_dataset(dataset_path):
        total = 0
        for sub in dataset_path.iterdir():
            if not sub.is_dir():
                continue
            data_dir = sub / "data"
            if data_dir.is_dir():
                for split_dir in data_dir.iterdir():
                    if split_dir.is_dir():
                        for jf in split_dir.glob("*.jsonl"):
                            try:
                                total += sum(1 for line in open(jf, encoding="utf-8", errors="replace") if line.strip())
                            except Exception:
                                pass
        return total

    if _is_asapaes_dataset(dataset_path):
        import csv as _csv
        total = 0
        for sub in dataset_path.iterdir():
            if sub.is_dir():
                for tsv_path in sub.glob("*.tsv"):
                    try:
                        with open(tsv_path, encoding="utf-8", errors="replace") as f:
                            total += sum(1 for _ in _csv.DictReader(f, delimiter="\t"))
                    except Exception:
                        pass
        return total

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

    if _is_sroie_dataset(dataset_path):
        split = next(
            (s for s in _SROIE_SPLIT_PREFERENCE if (dataset_path / s / "img").is_dir()),
            None,
        )
        if split:
            return sum(
                1 for f in (dataset_path / split / "img").iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            )
        return 0

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
    if _is_mimicxr_dataset(dataset_path):
        source = _iter_mimicxr_dataset(dataset_path)
    elif _is_multicare_dataset(dataset_path):
        source = _iter_multicare_dataset(dataset_path)
    elif _is_openpii_dataset(dataset_path):
        source = _iter_openpii_dataset(dataset_path)
    elif _is_asapaes_dataset(dataset_path):
        source = _iter_asapaes_dataset(dataset_path)
    elif _is_hf_dataset(dataset_path):
        source = _iter_hf_dataset(dataset_path)
    elif _is_sroie_dataset(dataset_path):
        source = _iter_sroie_dataset(dataset_path)
    elif _is_subdir_image_dataset(dataset_path):
        source = _iter_subdir_image_dataset(dataset_path)
    else:
        source = (load_item(path, modality) for path in list_dataset_items(dataset_name, modality))
    for item in source:
        yield item
        count += 1
        if max_items is not None and count >= max_items:
            break
