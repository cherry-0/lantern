"""
Shared loader, taxonomy, and config-filtering for leakage_analysis.py and leakage_deep.py.

The loader applies two filtering steps:

1. **Per-item success.** A row is "successfully evaluated" iff its ``ext_eval``
   contains at least one attribute entry with an aggregate verdict (i.e. the
   LLM judge returned a non-empty result). Rows that fail this check are
   dropped before any other filter sees them.

2. **Per-config admission.** A (app, dataset) cache directory is admitted iff
   it has at least ``MIN_CONFIG_ITEMS`` successfully-evaluated items, with two
   small-n exemptions:
   - image->image modality pair: exempted because the I->I sample is
     structurally small but analytically important (currently only tool-neuron
     on HR-VISPR, n=19).
   - apps in ``SMALL_N_EXEMPT_APPS``: exempted regardless of modality so that
     all their configs are included with small-n flagging. Currently covers
     tool-neuron, whose T->T (PrivacyLens, n=42) config would otherwise be
     excluded by the 100-item threshold.

3. **Per-config sampling.** Within each admitted config, if more than
   ``SAMPLE_PER_CONFIG`` items are available, exactly ``SAMPLE_PER_CONFIG``
   items are kept --- sampled deterministically (seed
   ``SAMPLE_SEED``, items first sorted by full_key for reproducibility). Configs
   with fewer items are kept at their actual N and surfaced in the audit dict.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR  = LANTERN_ROOT / "verify" / "outputs"
ATTACH_DIR   = LANTERN_ROOT / "analysis" / "attachments"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)
SKIP_FILES   = {"run_config.json", "dir_summary.json", "report.json", "report.csv"}

# ── Filter knobs ─────────────────────────────────────────────────────────────
MIN_CONFIG_ITEMS  = 100
SAMPLE_PER_CONFIG = 200   # cap retained items per (app, dataset) config
SAMPLE_SEED       = 42    # reproducible per-config sampling
I2I_EXEMPT        = True  # keep image->image configs even if N < MIN_CONFIG_ITEMS
# Apps whose *all* configs are admitted regardless of N (small-n exempt).
# tool-neuron has T->T (PrivacyLens, n=42) and I->I (HR-VISPR, n=19); both
# are structurally important as the sole representatives of their modality
# configurations in the corpus.
SMALL_N_EXEMPT_APPS: set = {"tool-neuron"}

# ── Adobe palette (re-exported so figure scripts can share the same colors) ──
P_TEAL   = "#7ADBC4"
P_YELLOW = "#FAD765"
P_ORANGE = "#FA9F5C"
P_GREEN  = "#98D198"
P_BLUE   = "#6C80FC"
P_MAUVE  = "#ACA4B3"
P_SLATE  = "#687692"
S_INPUT  = "#3A7DC9"
S_OUTPUT = "#E8762C"
S_EXT    = "#C93A3A"
GRAY     = "#AAAAAA"

VERDICT_COLORS = {
    "confirmed leakage": S_EXT,
    "possible leakage":  P_ORANGE,
    "no evidence":       P_GREEN,
    "na":                "#E0E0E0",
}
CHANNELS = ["NETWORK", "STORAGE", "UI", "LOGGING"]
CHANNEL_COLORS = {
    "NETWORK":   P_BLUE,
    "STORAGE":   P_ORANGE,
    "UI":        P_TEAL,
    "LOGGING":   P_SLATE,
    "AGGREGATE": S_EXT,
}

# ── Attribute taxonomy (paper appendix) ──────────────────────────────────────
ATTR_FAMILIES: Dict[str, List[str]] = {
    "Identity & Identifiability":  ["face", "identity"],
    "Demographic":                  ["age", "gender", "race", "marital status"],
    "Health & Medical":             ["disability", "medical"],
    "Location & Spatial":           ["location"],
    "Religion & Cultural":          ["religion", "ethnic_clothing"],
    "Appearance & Body":            ["nudity", "height", "weight", "haircolor", "color"],
    "Attire, Role & Group":         ["formal", "casual", "uniforms", "troupe"],
    "Activity & Lifestyle":         ["sports"],
}
ALL_ATTRS = [a for fam_attrs in ATTR_FAMILIES.values() for a in fam_attrs]
ATTR_TO_FAMILY = {a: fam for fam, attrs in ATTR_FAMILIES.items() for a in attrs}

FAMILY_COLORS = {
    "Identity & Identifiability": P_BLUE,
    "Demographic":                 P_ORANGE,
    "Health & Medical":            P_MAUVE,
    "Location & Spatial":          P_TEAL,
    "Religion & Cultural":         P_YELLOW,
    "Appearance & Body":           P_GREEN,
    "Attire, Role & Group":        P_SLATE,
    "Activity & Lifestyle":        S_OUTPUT,
}

APP_CATEGORY = {
    "budget-lens":"Finance","spendsense":"Finance","fiscal-flow":"Finance",
    "finchain":"Finance","chat-driven-expense-tracker":"Finance",
    "google-ai-edge-gallery":"Photo/Camera","tool-neuron":"Photo/Camera","momentag":"Photo/Camera",
    "clone":"Productivity","snapdo":"Productivity","xend":"Productivity",
    "pocketpal-ai":"Productivity","klyr":"Productivity",
    "deeptutor":"Education","edupal":"Education","sgpa":"Education","edumind":"Education",
    "llm-vtuber":"Social","lira":"Social","waico":"Social","tinytavern":"Social",
    "skin-disease-detection":"Health","nutri-track":"Health","healyks":"Health","nom-ai":"Health",
}
CATEGORY_COLORS = {
    "Finance":      P_GREEN,
    "Photo/Camera": P_ORANGE,
    "Productivity": P_BLUE,
    "Education":    P_YELLOW,
    "Social":       P_TEAL,
    "Health":       P_MAUVE,
}

DATASET_INPUT_TYPE = {
    "HR-VISPR":"image","SROIE2019":"docs","MIMIC-CXR":"image",
    "PrivacyLens":"text","SynthPAI":"text","GretelSyntheticPII":"docs",
    "ASAP-AES":"text","MultiCaRe":"text","OpenPII":"text",
    "MultiPriv":"docs","VISPR":"image",
}
INPUT_TYPE_COLOR = {"image": P_BLUE, "docs": P_YELLOW, "text": P_TEAL}

MOD_PAIR_COLOR = {
    "text→text":  P_TEAL,
    "image→text": P_BLUE,
    "text→image": P_YELLOW,
    "image→image":P_ORANGE,
}

# ── Verdict helpers ──────────────────────────────────────────────────────────
def entry_to_verdict(entry: Any) -> str:
    if not isinstance(entry, dict): return "no evidence"
    agg = entry.get("aggregate", entry)
    if not isinstance(agg, dict): agg = entry
    v = agg.get("verdict", "")
    if v in ("confirmed leakage","possible leakage","no evidence"): return v
    return "confirmed leakage" if agg.get("inferable") else "no evidence"


def entry_to_score(entry: Any) -> int:
    return {"confirmed leakage":2, "possible leakage":1, "no evidence":0}.get(
        entry_to_verdict(entry), 0)


def output_eval_to_verdict(oe_attr: Any) -> str:
    if not isinstance(oe_attr, dict): return "no evidence"
    if oe_attr.get("verdict") in ("confirmed leakage","possible leakage","no evidence"):
        return oe_attr["verdict"]
    if oe_attr.get("inferable"):
        return "confirmed leakage" if oe_attr.get("score", 0) >= 0.7 else "possible leakage"
    return "no evidence"


def output_eval_to_score(oe_attr: Any) -> int:
    return {"confirmed leakage":2, "possible leakage":1, "no evidence":0}.get(
        output_eval_to_verdict(oe_attr), 0)


def get_channel_verdicts(ext_entry: Any) -> Dict[str, str]:
    if not isinstance(ext_entry, dict): return {}
    chs = ext_entry.get("channels", {})
    return {ch: entry_to_verdict(v) for ch, v in chs.items()} if isinstance(chs, dict) else {}


# ── Item success predicate (non-empty evaluation) ────────────────────────────
def _is_item_success(data: Dict[str, Any]) -> bool:
    """Item passes iff its ``ext_eval`` carries at least one aggregate verdict."""
    ext = data.get("ext_eval") or {}
    return any(
        isinstance(v, dict) and "aggregate" in v and "verdict" in v.get("aggregate", {})
        for v in ext.values()
    )


# ── Loader ────────────────────────────────────────────────────────────────────
def load_data(min_items: int = MIN_CONFIG_ITEMS,
              i2i_exempt: bool = I2I_EXEMPT,
              small_n_exempt_apps: set = SMALL_N_EXEMPT_APPS,
              sample_per_config: int = SAMPLE_PER_CONFIG,
              seed: int = SAMPLE_SEED,
              eval_prompts: Tuple[str, ...] = ("prompt4", "prompt5"),
              ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Walk OUTPUTS_DIR, parse run_config.json + per-item JSONs, build a flat
    (item, attr) DataFrame and a parallel raw-item dict.

    Apply the following filters / transforms:
      - per item: ``ext_eval`` must contain at least one aggregate verdict
        (i.e. the LLM judge returned a non-empty result for the item);
      - per (app, dataset) config (= one ``cache_*`` directory): success-item
        count >= ``min_items``, except (a) image->image configs which are kept
        regardless of count when ``i2i_exempt`` is True, and (b) any config
        whose app is in ``small_n_exempt_apps`` (admitted with small-n
        flagging regardless of modality or count);
      - per-config sampling: if a kept config has more than
        ``sample_per_config`` items, exactly that many are retained, sampled
        deterministically with ``seed`` (items are first sorted by full_key
        before sampling so the result is reproducible across runs).

    Returns:
        df: one row per (item, attribute) pair after filtering and sampling.
        raw: dict mapping ``"<dir_name>/<filename>"`` -> raw item JSON, for
             case-study lookups by downstream scripts.
    """
    pending_items: Dict[str, List[Tuple[str, List[dict]]]] = defaultdict(list)
    pending_raw:   Dict[str, Dict[str, Dict]] = defaultdict(dict)
    config_n_succ: Dict[str, int] = defaultdict(int)
    dir_modality:  Dict[str, str] = {}
    dir_app_ds:    Dict[str, Tuple[str, str]] = {}

    for d in sorted(OUTPUTS_DIR.iterdir()):
        cfg_f = d / "run_config.json"
        if not cfg_f.exists(): continue
        try: cfg = json.loads(cfg_f.read_text())
        except Exception: continue

        app   = cfg.get("app_name", "?")
        ds    = cfg.get("dataset_name", "?")
        in_m  = cfg.get("input_modality",  "?")
        out_m = cfg.get("output_modality", "?")
        modality_pair = f"{in_m}→{out_m}"
        dir_modality[d.name] = modality_pair
        dir_app_ds[d.name]   = (app, ds)
        in_type  = DATASET_INPUT_TYPE.get(ds, in_m)
        category = APP_CATEGORY.get(app, "Other")

        for r in sorted(d.glob("*.json")):
            if r.name in SKIP_FILES: continue
            try: data = json.loads(r.read_text())
            except Exception: continue
            if data.get("eval_prompt", "") not in eval_prompts: continue
            if not _is_item_success(data): continue

            ext_eval     = data.get("ext_eval", {})        or {}
            output_eval  = data.get("output_eval", {})     or {}
            input_labels = data.get("input_labels", {})    or {}
            exts         = data.get("externalizations",{}) or {}
            filename     = data.get("filename", r.stem)
            full_key     = f"{d.name}/{filename}"

            base = dict(
                app=app, dataset=ds, in_mod=in_m, out_mod=out_m,
                in_type=in_type, modality_pair=modality_pair,
                category=category,
                eval_prompt=data.get("eval_prompt", ""),
                filename=filename, dir_name=d.name, full_key=full_key,
                n_ext_channels=len(exts),
            )

            item_rows: List[dict] = []
            for attr in ALL_ATTRS:
                ext_e = ext_eval.get(attr)
                out_e = output_eval.get(attr)
                ext_v = entry_to_verdict(ext_e)
                out_v = output_eval_to_verdict(out_e)
                ch_v  = get_channel_verdicts(ext_e)
                inp   = int(input_labels.get(attr, 0))

                rec = {**base,
                    "attr":           attr,
                    "family":         ATTR_TO_FAMILY.get(attr, "Other"),
                    "input_label":    inp,
                    "output_verdict": out_v,
                    "output_leak":    1 if out_v in ("confirmed leakage","possible leakage") else 0,
                    "output_conf":    1 if out_v == "confirmed leakage" else 0,
                    "output_has":     1 if isinstance(out_e, dict) and len(out_e) > 0 else 0,
                    "ext_verdict":    ext_v,
                    "ext_score":      entry_to_score(ext_e),
                    "ext_leak":       1 if ext_v in ("confirmed leakage","possible leakage") else 0,
                    "ext_conf":       1 if ext_v == "confirmed leakage" else 0,
                }
                for ch in CHANNELS:
                    cv = ch_v.get(ch)
                    rec[f"ch_{ch}_v"]    = cv if cv else "na"
                    rec[f"ch_{ch}_leak"] = 1 if cv in ("confirmed leakage","possible leakage") else 0
                    rec[f"ch_{ch}_conf"] = 1 if cv == "confirmed leakage" else 0
                    rec[f"ch_{ch}_pres"] = 1 if ch in exts else 0
                item_rows.append(rec)

            pending_items[d.name].append((full_key, item_rows))
            pending_raw[d.name][full_key] = data
            config_n_succ[d.name] += 1

    # Per-config admission filter (N >= min_items, with I->I and small-n-app exceptions).
    keep: List[str] = []
    for dname, n in config_n_succ.items():
        is_i2i   = (dir_modality.get(dname) == "image→image")
        app_name = dir_app_ds.get(dname, ("?", "?"))[0]
        is_small_n_exempt = app_name in (small_n_exempt_apps or set())
        if (i2i_exempt and is_i2i) or is_small_n_exempt or n >= min_items:
            keep.append(dname)
    keep_set = set(keep)

    # Per-config sampling: cap each kept config at sample_per_config items,
    # sampled deterministically. Configs with fewer items are kept whole.
    rng = random.Random(seed)
    rows: List[dict] = []
    raw:  Dict[str, Dict] = {}
    n_items_per_config: Dict[str, int] = {}
    under_floor: List[Dict[str, Any]] = []   # configs with fewer items than the cap
    for dname in keep:
        items = sorted(pending_items[dname], key=lambda t: t[0])  # deterministic order
        if len(items) > sample_per_config:
            items = rng.sample(items, sample_per_config)
            items = sorted(items, key=lambda t: t[0])             # restore order in df
        n_kept = len(items)
        n_items_per_config[dname] = n_kept
        if n_kept < sample_per_config:
            app, ds = dir_app_ds.get(dname, ("?", "?"))
            under_floor.append({
                "dir_name":      dname,
                "app":           app,
                "dataset":       ds,
                "modality_pair": dir_modality.get(dname, "?"),
                "n":             n_kept,
            })
        for fk, item_rows in items:
            rows.extend(item_rows)
            raw[fk] = pending_raw[dname][fk]

    df = pd.DataFrame(rows)

    # Append a small audit table to df.attrs so downstream scripts can report it.
    df.attrs["filter"] = {
        "min_items":              min_items,
        "i2i_exempt":             i2i_exempt,
        "small_n_exempt_apps":    sorted(small_n_exempt_apps or []),
        "sample_per_config":      sample_per_config,
        "seed":                   seed,
        "n_configs_total":        len(config_n_succ),
        "n_configs_kept":         len(keep_set),
        "n_configs_drop":         len(config_n_succ) - len(keep_set),
        "kept_modality":          {
            mp: sum(1 for d, m in dir_modality.items() if m == mp and d in keep_set)
            for mp in sorted(set(dir_modality.values()))
        },
        "n_items_kept":           sum(n_items_per_config.values()),
        "n_items_per_config":     n_items_per_config,
        "under_floor":            sorted(under_floor, key=lambda r: r["n"]),
    }
    return df, raw


# ── Legacy column aliases (for leakage_analysis.py) ──────────────────────────
# leakage_analysis.py historically used a different naming convention than
# leakage_deep.py. ``load_all_data`` emits both sets so the legacy script keeps
# working without invasive edits.
_LEGACY_ANALYSIS_ALIASES = {
    "output_verdict":  "output_verdict",   # same in both
    "output_leak":     "out_leakage",
    "output_conf":     "out_confirmed",
    "output_has":      "output_present",
    "ext_verdict":     "agg_verdict",
    "ext_leak":        "agg_leakage",
    "ext_conf":        "agg_confirmed",
    "ext_score":       "agg_score",
}


def _add_legacy_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for new_name, legacy_name in _LEGACY_ANALYSIS_ALIASES.items():
        if new_name in df.columns and legacy_name != new_name:
            df[legacy_name] = df[new_name]
    for ch in CHANNELS:
        for short, legacy in (("v", "verdict"), ("leak", "leakage"),
                              ("conf", "confirmed"), ("pres", "present")):
            new_col   = f"ch_{ch}_{short}"
            legacy_col = f"ch_{ch}_{legacy}"
            if new_col in df.columns and legacy_col != new_col:
                df[legacy_col] = df[new_col]
    if "n_ext_channels" in df.columns and "ext_channels" not in df.columns:
        # downstream just needs presence; full channel-list reconstruction not required
        df["ext_channels"] = [[] for _ in range(len(df))]
    return df


# ── Convenience wrapper for legacy callers (leakage_analysis.py used a
#    DataFrame-only signature) ──────────────────────────────────────────────────
def load_all_data(min_items: int = MIN_CONFIG_ITEMS,
                  i2i_exempt: bool = I2I_EXEMPT,
                  small_n_exempt_apps: set = SMALL_N_EXEMPT_APPS) -> pd.DataFrame:
    df, _ = load_data(min_items=min_items, i2i_exempt=i2i_exempt,
                      small_n_exempt_apps=small_n_exempt_apps)
    return _add_legacy_aliases(df)
