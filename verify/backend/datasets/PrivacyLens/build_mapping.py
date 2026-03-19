"""
Build a data_type → privacy attribute mapping for the PrivacyLens dataset.

Loads all 493 rows, extracts seed.data_type, maps each to a subset of Verify's
attribute list, and saves the result as data_type_mapping.pkl.

Run from the verify/ directory:
    python -m backend.datasets.PrivacyLens.build_mapping

The saved PKL contains:
    {
        "version": 1,
        "data_type_to_attrs": {data_type_str: [attr, ...]},
        "rows": [{"name": str, "data_type": str, "attributes": [attr, ...]}, ...],
        "stats": {data_type_str: count},
        "unmapped": [data_type_str, ...],   # types with no matched attribute
    }
"""

import pickle
import re
from collections import Counter
from pathlib import Path

# ── Keyword-based mapping ────────────────────────────────────────────────────
# Each entry maps an attribute name to a list of keyword fragments (lowercased).
# A data_type string is assigned an attribute if ANY keyword is found in it.

_ATTR_KEYWORDS: dict[str, list[str]] = {
    "location": [
        "location", "address", "timetable", "itinerary", "travel", "trip",
        "route", "gps", "schedule", "commute", "transit", "destination",
        "check-in", "check in", "flight", "hotel", "navigation", "map",
        "whereabouts", "place", "venue", "airport",
    ],
    "identity": [
        "identity", "name", "personal information", "account", "profile",
        "credential", "contact", "email", "phone", "id ", " id", "username",
        "user info", "biometric", "passport", "driver", "ssn", "national id",
        "social security", "identification",
    ],
    "age": [
        "age", "birth", "birthday", "date of birth", "dob", "generational",
        "retirement", "graduation", "school year", "born",
    ],
    "gender": [
        "gender", " sex ", "sexual", "pronouns", "transgender", "non-binary",
        "male", "female",
    ],
    "marital status": [
        "marital", "marriage", "married", "divorce", "divorced", "widowed",
        "widow", "spouse", "wedding", "relationship status", "civil",
    ],
}


def _map_data_type(data_type: str) -> list[str]:
    """Return list of matching attribute names for a data_type string."""
    dt_lower = data_type.lower()
    matched = []
    for attr, keywords in _ATTR_KEYWORDS.items():
        if any(kw in dt_lower for kw in keywords):
            matched.append(attr)
    return matched


def build_and_save(output_path: Path | None = None) -> dict:
    """Load PrivacyLens, build mapping, save PKL, return summary dict."""
    here = Path(__file__).resolve().parent
    dataset_path = here  # same dir as this script

    if output_path is None:
        output_path = here / "data_type_mapping.pkl"

    print("Loading PrivacyLens dataset …")
    try:
        from datasets import load_from_disk
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    ds_dict = load_from_disk(str(dataset_path))
    splits = ds_dict.keys() if hasattr(ds_dict, "keys") else ["train"]

    rows_out = []
    data_type_counter: Counter = Counter()

    for split in splits:
        split_ds = ds_dict[split]
        for idx, row in enumerate(split_ds):
            seed = row.get("seed") or {}
            data_type = (seed.get("data_type") or "").strip()
            name = row.get("name") or f"row_{idx:05d}"
            attrs = _map_data_type(data_type) if data_type else []
            rows_out.append({"name": name, "data_type": data_type, "attributes": attrs})
            if data_type:
                data_type_counter[data_type] += 1

    # Build unique data_type → attrs mapping
    data_type_to_attrs: dict[str, list[str]] = {}
    for data_type in data_type_counter:
        data_type_to_attrs[data_type] = _map_data_type(data_type)

    unmapped = [dt for dt, attrs in data_type_to_attrs.items() if not attrs]

    payload = {
        "version": 1,
        "data_type_to_attrs": data_type_to_attrs,
        "rows": rows_out,
        "stats": dict(data_type_counter),
        "unmapped": unmapped,
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved → {output_path}")
    return payload


def print_stats(payload: dict) -> None:
    stats = payload["stats"]
    data_type_to_attrs = payload["data_type_to_attrs"]
    unmapped = payload["unmapped"]

    print(f"\n{'='*60}")
    print(f"PrivacyLens data_type statistics  ({len(stats)} unique types, {sum(stats.values())} rows total)")
    print(f"{'='*60}")

    # Sort by count descending
    for dt, count in sorted(stats.items(), key=lambda x: -x[1]):
        attrs = data_type_to_attrs.get(dt, [])
        attr_str = ", ".join(attrs) if attrs else "— unmapped —"
        print(f"  {count:4d}x  {dt!r:50s}  →  {attr_str}")

    print(f"\nMapped:   {len(stats) - len(unmapped)} / {len(stats)} unique types")
    print(f"Unmapped: {len(unmapped)} types")
    if unmapped:
        print("  Unmapped data types:")
        for dt in unmapped:
            print(f"    - {dt!r}")

    # Per-attribute coverage
    print(f"\nAttribute coverage (rows whose data_type maps to the attribute):")
    attr_counts: dict[str, int] = {}
    for row in payload["rows"]:
        for attr in row["attributes"]:
            attr_counts[attr] = attr_counts.get(attr, 0) + 1
    total_rows = len(payload["rows"])
    for attr in ["location", "identity", "age", "gender", "marital status"]:
        n = attr_counts.get(attr, 0)
        print(f"  {attr:20s}: {n:4d} / {total_rows} rows ({100*n/total_rows:.1f}%)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # lantern root
    payload = build_and_save()
    print_stats(payload)
