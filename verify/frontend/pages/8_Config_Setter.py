"""
Config Setter — edit verify/batch_config.csv from a checkbox matrix.

Each matrix cell represents one batch config row:
  app + input/output workflow + dataset + perturbation method.

Checked cells mean "this config row exists in batch_config.csv". Existing rows
are checked by default. Adding and deletion are separate actions so an accidental
uncheck does not immediately remove rows.
"""

from __future__ import annotations

import csv
import html as _html
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERIFY_ROOT = Path(__file__).resolve().parent.parent.parent

if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st

from verify.backend.utils.config import (
    list_target_apps,
    load_dataset_list,
    load_perturbation_method_map,
)

_BATCH_CONFIG = VERIFY_ROOT / "batch_config.csv"
_CSV_FIELDNAMES = [
    "enabled",
    "app_name",
    "input_modality",
    "output_modality",
    "dataset_name",
    "perturbation_method",
    "max_items",
]

_Identity = Tuple[str, str, str, str, str]
_Workflow = Tuple[str, str, str, str]


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in reader:
            cleaned = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if not any(cleaned.values()):
                continue
            rows.append(_normalize_row(cleaned))
    return rows


def _normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    input_modality = row.get("input_modality") or row.get("modality") or ""
    output_modality = (
        row.get("output_modality")
        or row.get("generation_task")
        or row.get("modality")
        or ""
    )
    return {
        "enabled": row.get("enabled", "true") or "true",
        "app_name": row.get("app_name", ""),
        "input_modality": input_modality,
        "output_modality": output_modality,
        "dataset_name": row.get("dataset_name", ""),
        "perturbation_method": row.get("perturbation_method", ""),
        "max_items": row.get("max_items", ""),
    }


def _identity(row: Dict[str, str]) -> _Identity:
    return (
        row.get("app_name", "").strip(),
        row.get("input_modality", "").strip(),
        row.get("output_modality", "").strip(),
        row.get("dataset_name", "").strip(),
        row.get("perturbation_method", "").strip(),
    )


def _workflow(row: Dict[str, str]) -> _Workflow:
    return (
        row.get("app_name", "").strip(),
        row.get("input_modality", "").strip(),
        row.get("output_modality", "").strip(),
        row.get("perturbation_method", "").strip(),
    )


def _workflow_sort_key(workflow: _Workflow) -> Tuple[str, str, str, str]:
    app, input_modality, output_modality, method = workflow
    modality_rank = {"image": "0", "text": "1", "video": "2"}.get(input_modality, "9")
    return (modality_rank, app, output_modality, method)


def _load_registered_app_modalities() -> Dict[str, List[str]]:
    try:
        from verify.backend.adapters import ADAPTER_REGISTRY
    except Exception:
        return {}

    modalities: Dict[str, List[str]] = {}
    for app_name, cls in ADAPTER_REGISTRY.items():
        try:
            adapter = cls()
            supported = list(getattr(adapter, "supported_modalities", []) or [])
        except Exception:
            supported = []
        if supported:
            modalities[app_name] = supported
    return modalities


def _default_outputs_for(app_name: str, input_modality: str) -> List[str]:
    if app_name == "tool-neuron":
        if input_modality == "image":
            return ["image"]
        if input_modality == "text":
            return ["text", "image"]
    if input_modality == "video":
        return ["text"]
    return ["text"]


def _build_workflows(rows: List[Dict[str, str]]) -> List[_Workflow]:
    perturbation_map = load_perturbation_method_map()
    workflows: Set[_Workflow] = {_workflow(row) for row in rows}

    app_modalities = _load_registered_app_modalities()
    for app_name in list_target_apps():
        app_modalities.setdefault(app_name, [])

    for app_name, modalities in app_modalities.items():
        for input_modality in modalities:
            method = perturbation_map.get(input_modality, "")
            for output_modality in _default_outputs_for(app_name, input_modality):
                workflows.add((app_name, input_modality, output_modality, method))

    return sorted(workflows, key=_workflow_sort_key)


def _row_from_identity(identity: _Identity) -> Dict[str, str]:
    app_name, input_modality, output_modality, dataset_name, method = identity
    return {
        "enabled": "true",
        "app_name": app_name,
        "input_modality": input_modality,
        "output_modality": output_modality,
        "dataset_name": dataset_name,
        "perturbation_method": method,
        "max_items": "",
    }


def _write_batch_config(path: Path, rows: List[Dict[str, str]]) -> None:
    deduped: Dict[_Identity, Dict[str, str]] = {}
    for row in rows:
        normalized = _normalize_row(row)
        key = _identity(normalized)
        if all(key[:4]):
            deduped.setdefault(key, normalized)

    sorted_rows = sorted(
        deduped.values(),
        key=lambda row: (
            _workflow_sort_key(_workflow(row)),
            row.get("dataset_name", ""),
            row.get("enabled", ""),
        ),
    )

    lines: List[str] = [",".join(_CSV_FIELDNAMES) + "\n"]
    current_group: Tuple[str, str] | None = None
    for row in sorted_rows:
        group = (row["input_modality"], row["output_modality"])
        if group != current_group:
            current_group = group
            lines.append(
                f"# {group[0]} -> {group[1]} pipelines\n"
            )
        lines.append(_render_csv_row(row))
    path.write_text("".join(lines))


def _render_csv_row(row: Dict[str, str]) -> str:
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
    writer.writerow({name: row.get(name, "") for name in _CSV_FIELDNAMES})
    return buf.getvalue()


def _cell_key(identity: _Identity) -> str:
    return "config_setter__" + "__".join(identity).replace(" ", "_").replace("/", "_")


def _clear_batch_runner_selection_state() -> None:
    """Batch Runner keys are row-index based; clear them after CSV edits."""
    for key in list(st.session_state.keys()):
        if str(key).startswith("batch_row_") or key == "batch_config_signature":
            st.session_state.pop(key, None)


def _render_styles() -> None:
    st.markdown(
        """
<style>
.cfg-app-cell {
    padding: 0.35rem 0.15rem 0.15rem 0.1rem;
    font-weight: 600;
    font-size: 0.94rem;
}
.cfg-workflow-cell {
    padding: 0.12rem 0.15rem 0.15rem 0.1rem;
    font-size: 0.8rem;
    color: #666;
}
.cfg-method-cell {
    padding: 0.08rem 0.15rem 0.15rem 0.1rem;
    font-size: 0.76rem;
    color: #777;
}
.cfg-cell-label {
    color: #777;
    font-size: 0.74rem;
    min-height: 1.1rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Config Setter")
    st.markdown(
        "Edit `verify/batch_config.csv`. Checked cells represent config rows that should exist."
    )

    existing_rows = _load_csv_rows(_BATCH_CONFIG)
    if not existing_rows and not _BATCH_CONFIG.exists():
        st.error("`verify/batch_config.csv` does not exist.")
        return

    datasets = load_dataset_list()
    workflows = _build_workflows(existing_rows)
    existing_by_id = {_identity(row): row for row in existing_rows}
    existing_ids = set(existing_by_id)

    if not datasets:
        st.error("No datasets found in `verify/config/dataset_list.txt`.")
        return
    if not workflows:
        st.error("No workflows found from adapters or `batch_config.csv`.")
        return

    with st.sidebar:
        st.header("Filters")
        apps = sorted({workflow[0] for workflow in workflows})
        selected_apps = st.multiselect("Apps", apps, default=apps)
        selected_inputs = st.multiselect(
            "Input modalities",
            sorted({workflow[1] for workflow in workflows}),
            default=sorted({workflow[1] for workflow in workflows}),
        )
        selected_datasets = st.multiselect("Datasets", datasets, default=datasets)
        show_existing_only = st.toggle(
            "Show existing rows only",
            value=False,
            help="Hide workflow rows that have no checked cells in the current CSV.",
        )

    visible_workflows = [
        workflow for workflow in workflows
        if workflow[0] in selected_apps and workflow[1] in selected_inputs
    ]
    if show_existing_only:
        visible_workflows = [
            workflow for workflow in visible_workflows
            if any(
                (workflow[0], workflow[1], workflow[2], dataset, workflow[3]) in existing_ids
                for dataset in selected_datasets
            )
        ]

    _render_styles()

    header_cols = st.columns([1.55] + [1.25] * len(selected_datasets))
    header_cols[0].markdown("**App / Workflow**")
    for idx, dataset in enumerate(selected_datasets, start=1):
        header_cols[idx].markdown(f"**{dataset}**")
    st.divider()

    checked_ids: Set[_Identity] = set()

    for workflow in visible_workflows:
        app_name, input_modality, output_modality, method = workflow
        workflow_label = f"{input_modality}->{output_modality}"
        row_cols = st.columns([1.55] + [1.25] * len(selected_datasets))
        row_cols[0].markdown(
            f'<div class="cfg-app-cell">{_html.escape(app_name)}</div>'
            f'<div class="cfg-workflow-cell">{_html.escape(workflow_label)}</div>'
            f'<div class="cfg-method-cell">{_html.escape(method or "-")}</div>',
            unsafe_allow_html=True,
        )

        for dataset_idx, dataset_name in enumerate(selected_datasets, start=1):
            identity = (app_name, input_modality, output_modality, dataset_name, method)
            default_checked = identity in existing_ids
            with row_cols[dataset_idx].container(border=True):
                checked = st.checkbox(
                    f"{app_name}/{workflow_label}/{dataset_name}",
                    value=st.session_state.get(_cell_key(identity), default_checked),
                    key=_cell_key(identity),
                    label_visibility="collapsed",
                )
                st.markdown(
                    '<div class="cfg-cell-label">'
                    + ("existing" if default_checked else "new")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                if checked:
                    checked_ids.add(identity)

    add_ids = checked_ids - existing_ids
    delete_ids = {
        identity
        for identity in existing_ids
        if identity not in checked_ids
        and identity[0] in selected_apps
        and identity[1] in selected_inputs
        and identity[3] in selected_datasets
    }

    st.divider()
    st.caption(
        f"{len(existing_ids)} rows currently in CSV. "
        f"{len(add_ids)} checked new rows. {len(delete_ids)} unchecked existing rows in the visible matrix."
    )

    add_col, delete_col = st.columns(2)
    with add_col:
        if st.button("Add checked new configs", type="primary", disabled=not add_ids, width="stretch"):
            updated_rows = existing_rows + [_row_from_identity(identity) for identity in sorted(add_ids)]
            _write_batch_config(_BATCH_CONFIG, updated_rows)
            _clear_batch_runner_selection_state()
            st.success(f"Added {len(add_ids)} config row(s).")
            st.rerun()

    with delete_col:
        confirm_delete = st.checkbox(
            "Confirm deletion of unchecked visible existing rows",
            value=False,
            key="config_setter_confirm_delete",
        )
        if st.button(
            "Delete unchecked existing configs",
            type="secondary",
            disabled=not delete_ids or not confirm_delete,
            width="stretch",
        ):
            updated_rows = [
                row for row in existing_rows
                if _identity(row) not in delete_ids
            ]
            _write_batch_config(_BATCH_CONFIG, updated_rows)
            _clear_batch_runner_selection_state()
            st.success(f"Deleted {len(delete_ids)} config row(s).")
            st.rerun()


if __name__ == "__main__":
    main()
