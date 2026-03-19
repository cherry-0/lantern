"""
Verify orchestrator — coordinates the full pipeline for one run:

For each dataset item:
  1. Load item (modality-aware)
  2. Run target app pipeline on original input
  3. Perturb the input using the selected attributes + configured method
  4. Run target app pipeline on perturbed input
  5. Evaluate privacy inferability for original and perturbed outputs
  6. Save and cache results

Yields item results progressively as a generator so the Streamlit UI
can display them item-by-item without waiting for the whole dataset.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from verify.backend.utils.config import (
    ensure_outputs_dir,
    load_perturbation_method_map,
)
from verify.backend.utils import cache as cache_module
from verify.backend.adapters import get_adapter
from verify.backend.datasets.loader import iter_dataset
from verify.backend.perturbation_method.interface import (
    check_perturbation_availability,
    run_perturbation,
)
from verify.backend.evaluation_method.evaluator import evaluate_both


# ─── Result type definitions ────────────────────────────────────────────────

# Status constants for item results
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def _make_item_result(
    filename: str,
    status: str = STATUS_SUCCESS,
    **kwargs,
) -> Dict[str, Any]:
    """Construct a uniform item result dict."""
    return {
        "filename": filename,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }


# ─── Orchestrator ────────────────────────────────────────────────────────────


class Orchestrator:
    """
    Runs the full Verify pipeline for a given configuration and yields
    per-item results progressively.
    """

    def __init__(
        self,
        app_name: str,
        dataset_name: str,
        modality: str,
        attributes: List[str],
        use_cache: bool = True,
        max_items: Optional[int] = None,
        perturbation_method: Optional[str] = None,
        perturbation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.app_name = app_name
        self.dataset_name = dataset_name
        self.modality = modality
        self.attributes = attributes
        self.use_cache = use_cache
        self.max_items = max_items
        self.perturbation_kwargs = perturbation_kwargs or {}

        # Resolve perturbation method: use explicit override or fall back to config
        if perturbation_method:
            self.perturbation_method = perturbation_method
        else:
            method_map = load_perturbation_method_map()
            self.perturbation_method = method_map.get(modality, "")

        self._adapter = get_adapter(app_name)
        self._cache_dir: Optional[Path] = None
        self._run_dir: Optional[Path] = None

    def _setup_dirs(self) -> None:
        """Prepare cache and output directories for this run."""
        outputs_dir = ensure_outputs_dir()

        # Unique run directory with timestamp
        attr_slug = "_".join(sorted(self.attributes)) or "none"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.app_name}_{self.modality}_{attr_slug}_{ts}"
        self._run_dir = outputs_dir / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        if self.use_cache:
            self._cache_dir = cache_module.get_cache_dir(
                self.app_name,
                self.dataset_name,
                self.modality,
                self.attributes,
                self.perturbation_method,
            )

    def _save_item_result(self, filename: str, result: Dict[str, Any]) -> None:
        """Persist an item result to the run directory and cache."""
        if self._run_dir:
            out_file = self._run_dir / f"{filename}.json"
            try:
                out_file.write_text(json.dumps(result, indent=2, default=str))
            except Exception:
                pass

        if self._cache_dir and self.use_cache:
            cache_module.save_item_cache(self._cache_dir, filename, result)

    def _check_cache(self, filename: str) -> Optional[Dict[str, Any]]:
        """Return a cached result if available, else None."""
        if self._cache_dir and self.use_cache:
            return cache_module.load_item_cache(self._cache_dir, filename)
        return None

    def _save_run_config(self) -> None:
        """Save the run configuration for reproducibility."""
        config = {
            "app_name": self.app_name,
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "attributes": self.attributes,
            "perturbation_method": self.perturbation_method,
            "evaluation_method": "openrouter",
            "use_cache": self.use_cache,
            "run_dir": str(self._run_dir),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._run_dir:
            try:
                (self._run_dir / "run_config.json").write_text(
                    json.dumps(config, indent=2)
                )
            except Exception:
                pass

        if self._cache_dir and self.use_cache:
            cache_module.save_run_config(self._cache_dir, config)

    def _generate_report(self, all_results: List[Dict[str, Any]]) -> None:
        """Generate JSON and CSV summary reports in the run directory."""
        if not self._run_dir:
            return

        # JSON report
        report = {
            "run_config": {
                "app_name": self.app_name,
                "dataset_name": self.dataset_name,
                "modality": self.modality,
                "attributes": self.attributes,
                "perturbation_method": self.perturbation_method,
            },
            "items": all_results,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            (self._run_dir / "report.json").write_text(
                json.dumps(report, indent=2, default=str)
            )
        except Exception:
            pass

        # CSV summary
        try:
            import csv
            import io

            rows = []
            for item in all_results:
                row = {
                    "filename": item.get("filename", ""),
                    "status": item.get("status", ""),
                    "original_output": item.get("original_output", {}).get(
                        "output_text", ""
                    )[:200],
                    "perturbed_output": item.get("perturbed_output", {}).get(
                        "output_text", ""
                    )[:200],
                }
                eval_result = item.get("evaluation", {})
                orig_eval = eval_result.get("original", {})
                pert_eval = eval_result.get("perturbed", {})
                for attr in self.attributes:
                    row[f"orig_{attr}_inferable"] = orig_eval.get(attr, {}).get(
                        "inferable", ""
                    )
                    row[f"orig_{attr}_score"] = orig_eval.get(attr, {}).get("score", "")
                    row[f"pert_{attr}_inferable"] = pert_eval.get(attr, {}).get(
                        "inferable", ""
                    )
                    row[f"pert_{attr}_score"] = pert_eval.get(attr, {}).get("score", "")
                rows.append(row)

            if rows:
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
                (self._run_dir / "report.csv").write_text(buf.getvalue())
        except Exception:
            pass

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """
        Execute the pipeline and yield per-item result dicts progressively.

        Each yielded dict has the shape defined by _make_item_result().
        The final yielded item has "type": "summary" with aggregated stats.
        """
        self._setup_dirs()
        self._save_run_config()

        # Check adapter availability
        if self._adapter is None:
            yield _make_item_result(
                filename="__config__",
                status=STATUS_FAILED,
                error=f"No adapter registered for app '{self.app_name}'.",
                type="error",
            )
            return

        adapter_ok, adapter_msg = self._adapter.check_availability()
        yield {
            "type": "adapter_status",
            "app_name": self.app_name,
            "available": adapter_ok,
            "message": adapter_msg,
        }

        if not adapter_ok:
            yield _make_item_result(
                filename="__config__",
                status=STATUS_FAILED,
                error=f"Adapter '{self.app_name}' is not available: {adapter_msg}",
                type="error",
            )
            return

        # Check perturbation availability
        pert_ok, pert_msg = check_perturbation_availability(self.modality, self.perturbation_method or None)
        yield {
            "type": "perturbation_status",
            "modality": self.modality,
            "method": self.perturbation_method,
            "available": pert_ok,
            "message": pert_msg,
        }

        all_results: List[Dict[str, Any]] = []

        for load_ok, item, load_err in iter_dataset(self.dataset_name, self.modality, max_items=self.max_items):
            filename = item.get("filename", "unknown")

            # ── Attribute-based filtering ──────────────────────────────────
            # Keep only items whose label set overlaps with the selected attributes.
            # Items with no labels are always included.
            # The UI already presents modality-appropriate attributes (HR-VISPR labels
            # for image, text attributes for text), so the comparison is always
            # within the same vocabulary — a direct set intersection suffices.
            item_labels: List[str] = (
                item.get("privacy_labels")           # HR-VISPR / image datasets
                or item.get("data_type_attributes")  # PrivacyLens text dataset
                or []
            )
            if item_labels and not (set(item_labels) & set(self.attributes)):
                continue  # no overlap — skip

            # Check cache first
            cached = self._check_cache(filename)
            if cached:
                cached["from_cache"] = True
                all_results.append(cached)
                yield {**cached, "type": "item_result"}
                continue

            if not load_ok:
                result = _make_item_result(
                    filename=filename,
                    status=STATUS_FAILED,
                    error=f"Failed to load item: {load_err}",
                    type="item_result",
                )
                all_results.append(result)
                self._save_item_result(filename, result)
                yield result
                continue

            # --- Step 2: Run pipeline on original input ---
            try:
                orig_result = self._adapter.run_pipeline(item)
            except Exception as e:
                orig_result = None
                orig_error = str(e)
            else:
                orig_error = orig_result.error if not orig_result.success else None

            if not orig_result or not orig_result.success:
                result = _make_item_result(
                    filename=filename,
                    status=STATUS_FAILED,
                    error=f"Original pipeline failed: {orig_error}",
                    type="item_result",
                )
                all_results.append(result)
                self._save_item_result(filename, result)
                yield result
                continue

            # --- Step 3: Perturb the input ---
            if not pert_ok:
                # Skip perturbation, still record original result
                result = _make_item_result(
                    filename=filename,
                    status=STATUS_SKIPPED,
                    original_output=orig_result.to_dict(),
                    perturbed_output=None,
                    evaluation=None,
                    perturbation_warning=f"Perturbation skipped: {pert_msg}",
                    type="item_result",
                )
                all_results.append(result)
                self._save_item_result(filename, result)
                yield result
                continue

            pert_ok_item, perturbed_item, pert_err = run_perturbation(
                item, self.modality, self.attributes, self.perturbation_method or None,
                **self.perturbation_kwargs,
            )

            if not pert_ok_item:
                result = _make_item_result(
                    filename=filename,
                    status=STATUS_SKIPPED,
                    original_output=orig_result.to_dict(),
                    perturbed_output=None,
                    evaluation=None,
                    perturbation_warning=f"Perturbation failed: {pert_err}",
                    type="item_result",
                )
                all_results.append(result)
                self._save_item_result(filename, result)
                yield result
                continue

            # --- Step 4: Run pipeline on perturbed input ---
            try:
                pert_pipeline_result = self._adapter.run_pipeline(perturbed_item)
            except Exception as e:
                pert_pipeline_result = None
                pert_pipeline_error = str(e)
            else:
                pert_pipeline_error = (
                    pert_pipeline_result.error
                    if not pert_pipeline_result.success
                    else None
                )

            if not pert_pipeline_result or not pert_pipeline_result.success:
                result = _make_item_result(
                    filename=filename,
                    status=STATUS_FAILED,
                    original_output=orig_result.to_dict(),
                    perturbed_item_info={
                        "perturbation_applied": perturbed_item.get("perturbation_applied", {})
                    },
                    perturbed_output=None,
                    evaluation=None,
                    error=f"Perturbed pipeline failed: {pert_pipeline_error}",
                    type="item_result",
                )
                all_results.append(result)
                self._save_item_result(filename, result)
                yield result
                continue

            # --- Step 5: Evaluate inferability ---
            eval_results = evaluate_both(
                original_output=orig_result.output_text,
                perturbed_output=pert_pipeline_result.output_text,
                attributes=self.attributes,
            )

            # --- Build complete item result ---
            result = _make_item_result(
                filename=filename,
                status=STATUS_SUCCESS,
                original_input={
                    "modality": self.modality,
                    "text_content": item.get("text_content", ""),
                    # Images are not serialized to JSON; use filename reference
                    "has_image": "image_base64" in item,
                    "has_frames": "frames" in item,
                    "frame_count": len(item.get("frames", [])),
                    # Label metadata (for display)
                    "privacy_labels": item.get("privacy_labels", []),
                    "data_type": item.get("data_type", ""),
                    "data_type_attributes": item.get("data_type_attributes", []),
                },
                original_output=orig_result.to_dict(),
                perturbed_input={
                    "modality": self.modality,
                    "text_content": perturbed_item.get("text_content", ""),
                    "has_image": "image_base64" in perturbed_item,
                    "has_frames": "frames" in perturbed_item,
                    "perturbation_applied": perturbed_item.get("perturbation_applied", {}),
                },
                perturbed_output=pert_pipeline_result.to_dict(),
                evaluation=eval_results,
                type="item_result",
            )

            # Attach PIL objects for UI (not serialized to disk)
            result["_original_data"] = item.get("data")
            result["_original_image_b64"] = item.get("image_base64")
            result["_original_frames"] = item.get("frames", [])
            result["_perturbed_data"] = perturbed_item.get("data")
            result["_perturbed_image_b64"] = perturbed_item.get("image_base64")
            result["_perturbed_frames"] = perturbed_item.get("frames", [])

            # Save to disk (without the _ prefixed PIL fields)
            saveable = {k: v for k, v in result.items() if not k.startswith("_")}
            all_results.append(saveable)
            self._save_item_result(filename, saveable)

            yield result

        # --- Generate final report ---
        self._generate_report(all_results)

        # --- Yield summary ---
        total = len(all_results)
        success_count = sum(1 for r in all_results if r.get("status") == STATUS_SUCCESS)
        skipped_count = sum(1 for r in all_results if r.get("status") == STATUS_SKIPPED)
        failed_count = sum(1 for r in all_results if r.get("status") == STATUS_FAILED)

        # Aggregate inferability scores across items
        agg: Dict[str, Dict[str, List[float]]] = {
            "original": {attr: [] for attr in self.attributes},
            "perturbed": {attr: [] for attr in self.attributes},
        }
        for item_result in all_results:
            eval_r = item_result.get("evaluation") or {}
            for stage in ("original", "perturbed"):
                stage_eval = eval_r.get(stage, {})
                for attr in self.attributes:
                    score = stage_eval.get(attr, {}).get("score")
                    if isinstance(score, (int, float)):
                        agg[stage][attr].append(float(score))

        def _avg(lst):
            return sum(lst) / len(lst) if lst else None

        agg_summary = {
            stage: {attr: _avg(scores) for attr, scores in attrs.items()}
            for stage, attrs in agg.items()
        }

        yield {
            "type": "summary",
            "total": total,
            "success": success_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "run_dir": str(self._run_dir),
            "attributes": self.attributes,
            "aggregated_scores": agg_summary,
        }
