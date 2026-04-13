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

import atexit
import tempfile
import io
import json
import re
import base64
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

BUDGETLENS_DJANGO_ROOT = TARGET_APPS_DIR / "budget-lens" / "budgetlens"

_ENV_SPEC = EnvSpec(
    name="budget-lens",
    python="3.10",
    install_cmds=[
        ["pip", "install", "-r", str(TARGET_APPS_DIR / "budget-lens" / "requirements.txt")],
        ["pip", "install", "fastapi", "uvicorn", "pydantic", "requests"]
    ],
)
_RUNNER = Path(__file__).parent.parent / "runners" / "budgetlens_runner.py"

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
    env_spec = _ENV_SPEC

    def __init__(self):
        self._server_process = None
        self._server_port = None
        atexit.register(self._cleanup_server)

    def _cleanup_server(self):
        if self._server_process is not None:
            import sys
            print(f"[budget-lens] Shutting down local API server (port {self._server_port})...", file=sys.stderr, flush=True)
            self._server_process.terminate()
            self._server_process.wait()
            self._server_process = None

    def _start_server(self):
        if self._server_process is not None and self._server_process.poll() is None:
            return

        import socket
        import subprocess
        import time
        import requests
        import sys

        s = socket.socket()
        s.bind(("", 0))
        self._server_port = s.getsockname()[1]
        s.close()

        conda = CondaRunner.find_conda()
        server_script = Path(__file__).parent.parent / "runners" / "budgetlens_server.py"

        print(f"[budget-lens] Starting local API server on port {self._server_port}...", file=sys.stderr, flush=True)
        self._server_process = subprocess.Popen(
            [conda, "run", "-n", _ENV_SPEC.name, "python", str(server_script), "--port", str(self._server_port)],
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )

        start_time = time.time()
        while time.time() - start_time < 30:
            try:
                resp = requests.get(f"http://127.0.0.1:{self._server_port}/health")
                if resp.status_code == 200:
                    print("[budget-lens] Server is ready.", file=sys.stderr, flush=True)
                    return
            except Exception:
                time.sleep(1)
        
        raise RuntimeError("budgetlens_server failed to start within 30 seconds.")

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter vision fallback for budget-lens."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "image":
            return AdapterResult(
                success=False,
                error=f"budget-lens only supports 'image' modality, got '{input_item.get('modality')}'.",
            )
        if use_app_servers():
            return self._run_native(input_item)
        return self._run_openrouter_fallback(input_item)

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Run budget-lens's process_receipt() inside the 'budget-lens' conda env via HTTP server."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        import requests
        import sys

        try:
            self._start_server()
        except Exception as e:
            return AdapterResult(success=False, error=f"Failed to start server: {e}")

        image_b64 = input_item.get("image_base64") or _encode_image_b64(
            input_item.get("data") or input_item.get("path", "")
        )

        try:
            payload = {
                "image_base64": image_b64,
                "openrouter_api_key": get_openrouter_api_key() or "",
            }
            print("[budget-lens] Sending inference request to local server...", file=sys.stderr, flush=True)
            resp = requests.post(f"http://127.0.0.1:{self._server_port}/infer", json=payload, timeout=90)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"HTTP request to server failed: {e}")

        if not result.get("success"):
            return AdapterResult(success=False, error=result.get("error"))

        structured = {k: result.get(k) for k in ("category", "date", "amount", "currency")}
        externalizations = result.get("externalizations", {})

        output_text = (
            f"Category: {structured.get('category', 'Unknown')}\n"
            f"Date: {structured.get('date', 'Unknown')}\n"
            f"Amount: {structured.get('amount', 'Unknown')} {structured.get('currency', '')}"
        )
        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output=result,
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "native_process_receipt_server"},
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

        if not image_b64:
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

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    f"[OpenRouter Fallback] Sending receipt image (base64) for extraction. \n"
                    f"[Merchant Search] Searching for {merchant_name} at {merchant_address}."
                ),
                "UI": f"Dashboard display: {output_text}",
            }
        )

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"raw_response": raw_response, "parsed": structured},
            structured_output=structured,
            externalizations=externalizations,
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
