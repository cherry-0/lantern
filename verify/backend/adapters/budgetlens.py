"""
Adapter for the budget-lens app.

Core pipeline: receipt image → OpenAI vision analysis → structured expense data
               (category, date, amount, currency).

Primary strategy: import process_receipt() from the budget-lens Django app.
Fallback: use OpenRouter vision model to replicate the same receipt extraction,
          but also captures merchant name and address for richer privacy analysis.

Note: process_receipt() requires a file path (not base64). For native runs we
write a temporary JPEG if only in-memory data is available.

Upcoming datasets: SROIE2019, receipt private-attribute dataset.
"""

import io
import json
import os
import re
import sys
import tempfile
import base64
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key

# Budget-lens Django project root (the dir that contains budgetlens/settings.py)
BUDGETLENS_DJANGO_ROOT = TARGET_APPS_DIR / "budget-lens" / "budgetlens"

# Expense categories as defined in core/models.py
BASE_CATEGORIES = [
    "Housing", "Utilities", "Transportation", "Groceries", "Dining Out",
    "Healthcare", "Debt Payments", "Insurance", "Clothing", "Entertainment",
    "Education", "Childcare", "Pet Care", "Subscriptions", "Miscellaneous",
]


def _encode_image_b64(data_or_path) -> str:
    """Encode a PIL Image or file path to base64 JPEG."""
    from PIL import Image as PILImage

    try:
        if isinstance(data_or_path, (str, Path)):
            img = PILImage.open(str(data_or_path)).convert("RGB")
        else:
            img = data_or_path.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to encode image: {e}")


def _image_to_temp_file(data_or_path) -> str:
    """
    Write a PIL Image or file path to a temporary JPEG on disk and return the
    path. The caller is responsible for deleting the file.
    """
    from PIL import Image as PILImage

    if isinstance(data_or_path, (str, Path)):
        return str(data_or_path)

    img = data_or_path.convert("RGB")
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG", quality=85)
    tmp.close()
    return tmp.name


class BudgetLensAdapter(BaseAdapter):
    """
    Wraps the budget-lens receipt scanning pipeline.

    Given a receipt image the app extracts:
      - category  (one of 15 predefined spending categories)
      - date      (ISO-format transaction date)
      - amount    (decimal amount)
      - currency  (ISO 4217 three-letter code)

    The OpenRouter fallback additionally attempts to extract merchant name and
    address, which are privacy-sensitive fields present in SROIE annotations
    and the upcoming receipt private-attribute dataset.
    """

    name = "budget-lens"
    supported_modalities = ["image"]

    def __init__(self):
        self._native_available: Optional[bool] = None
        self._native_error: str = ""

    # ── Django / native availability ─────────────────────────────────────────

    def _check_native(self) -> Tuple[bool, str]:
        if self._native_available is not None:
            return self._native_available, self._native_error

        # core/views.py creates `client = OpenAI()` at module level — the
        # env vars must be set before the first import so the client picks up
        # the correct API key and base URL.
        self._inject_openrouter_env()

        django_root = str(BUDGETLENS_DJANGO_ROOT)
        if django_root not in sys.path:
            sys.path.insert(0, django_root)

        try:
            import django
            from django.apps import apps as django_apps
            from django import conf as django_conf
            from django.utils.functional import empty

            already_configured = (
                django_conf.settings.configured
                and os.environ.get("DJANGO_SETTINGS_MODULE") == "budgetlens.settings"
                and django_apps.ready
            )

            if not already_configured:
                # Purge any stale module cache from other adapters
                for _mod in list(sys.modules):
                    if _mod in ("budgetlens", "core", "accounts") or \
                       _mod.startswith("budgetlens.") or \
                       _mod.startswith("core.") or \
                       _mod.startswith("accounts."):
                        del sys.modules[_mod]

                if django_apps.ready or getattr(django_apps, "_loading", False):
                    from collections import defaultdict
                    django_apps.app_configs = {}
                    django_apps.all_models = defaultdict(dict)
                    django_apps.ready = False
                    django_apps.loading = False # changed here
                    # Use Django's own sentinel value so LazySettings.configured
                    # returns False and _setup() is re-run on the next attribute
                    # access. Setting _wrapped = None (not empty) would make
                    # .configured return True while the wrapped object is None,
                    # causing 'NoneType has no attribute LOGGING_CONFIG' errors.
                    django_conf.settings._wrapped = empty

                os.environ["DJANGO_SETTINGS_MODULE"] = "budgetlens.settings"
                django.setup()

            from core.views import process_receipt  # noqa: F401

            self._native_available = True
            self._native_error = ""
        except Exception as e:
            self._native_available = False
            self._native_error = str(e)

        return self._native_available, self._native_error

    def check_availability(self) -> Tuple[bool, str]:
        """Available if either native pipeline or OpenRouter fallback works."""
        native_ok, native_err = self._check_native()
        if native_ok:
            return True, "Native budget-lens process_receipt() available."

        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return (
                True,
                f"Native pipeline unavailable ({native_err}); using OpenRouter vision fallback.",
            )

        return (
            False,
            f"budget-lens native pipeline unavailable ({native_err}) and no valid OpenRouter API key found.",
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "image":
            return AdapterResult(
                success=False,
                error=f"budget-lens only supports 'image' modality, got '{input_item.get('modality')}'.",
            )

        try:
            native_ok, _ = self._check_native()
            if native_ok:
                return self._run_native(input_item)
            else:
                return self._run_openrouter_fallback(input_item)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # ── Native strategy ───────────────────────────────────────────────────────

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        """
        Call budget-lens's process_receipt() directly.

        process_receipt() requires a filesystem path, so we write a temp file
        when only in-memory PIL data is provided.
        """
        from core.views import process_receipt

        data = input_item.get("data")
        path = input_item.get("path", "")

        temp_path: Optional[str] = None
        if path and Path(path).exists():
            image_path = path
        else:
            image_path = _image_to_temp_file(data)
            temp_path = image_path

        try:
            category, expense_date, amount, currency = process_receipt(image_path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

        structured = {
            "category": category,
            "date": str(expense_date),
            "amount": float(amount) if amount is not None else None,
            "currency": currency,
        }
        output_text = (
            f"Category: {category}\n"
            f"Date: {expense_date}\n"
            f"Amount: {amount} {currency}"
        )

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output=structured,
            structured_output=structured,
            metadata={"method": "native_process_receipt"},
        )

    # ── OpenRouter fallback ───────────────────────────────────────────────────

    def _run_openrouter_fallback(self, input_item: Dict[str, Any]) -> AdapterResult:
        """
        Use OpenRouter vision model to replicate budget-lens receipt extraction.

        Extracts the same fields as the native pipeline (category, date, amount,
        currency) and additionally attempts to surface merchant name and address,
        which are relevant for upcoming receipt-specific privacy datasets (SROIE,
        receipt private-attribute dataset).
        """
        data = input_item.get("data")
        path = input_item.get("path", "")
        image_b64 = input_item.get("image_base64")

        if image_b64 is None:
            image_b64 = _encode_image_b64(data if data is not None else path)

        prompt = (
            "You are a receipt analysis assistant. Analyze the provided receipt image and extract:\n\n"
            "1. category: The expense category from this list: "
            f"{', '.join(BASE_CATEGORIES)}. Choose 'Miscellaneous' if none fits.\n"
            "2. date: Transaction date in YYYY-MM-DD format.\n"
            "3. amount: Total expense amount as a decimal number (no currency symbol). "
            "Note: commas may be thousand separators or decimal separators depending on locale.\n"
            "4. currency: ISO 4217 three-letter currency code (e.g. USD, EUR, KRW).\n"
            "5. merchant_name: The store or business name as printed on the receipt "
            "(empty string if not visible).\n"
            "6. merchant_address: The store address as printed on the receipt "
            "(empty string if not visible).\n\n"
            "Respond ONLY with a JSON object (no markdown, no extra text):\n"
            '{"category": "...", "date": "...", "amount": 0.00, "currency": "...", '
            '"merchant_name": "...", "merchant_address": "..."}'
        )

        try:
            raw_response = self._call_openrouter(prompt, image_b64=image_b64, max_tokens=512)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        structured = self._parse_receipt_json(raw_response)

        output_text = (
            f"Category: {structured.get('category', 'Unknown')}\n"
            f"Date: {structured.get('date', 'Unknown')}\n"
            f"Amount: {structured.get('amount', 'Unknown')} {structured.get('currency', '')}"
        )
        merchant_name = structured.get("merchant_name", "")
        merchant_address = structured.get("merchant_address", "")
        if merchant_name:
            output_text += f"\nMerchant: {merchant_name}"
        if merchant_address:
            output_text += f"\nAddress: {merchant_address}"

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"raw_response": raw_response, "parsed": structured},
            structured_output=structured,
            metadata={"method": "openrouter_vision_fallback"},
        )

    @staticmethod
    def _parse_receipt_json(text: str) -> Dict[str, Any]:
        """Extract and parse the JSON object from a model response."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                cat = data.get("category", "Miscellaneous")
                if cat not in BASE_CATEGORIES:
                    cat = "Miscellaneous"
                data["category"] = cat
                date_str = str(data.get("date", ""))
                data["date"] = date_str.replace("/", "-")
                curr = data.get("currency", "")
                if isinstance(curr, str):
                    data["currency"] = curr.upper()
                return data
            except json.JSONDecodeError:
                pass

        return {
            "category": "Miscellaneous",
            "date": "",
            "amount": None,
            "currency": "",
            "merchant_name": "",
            "merchant_address": "",
        }
