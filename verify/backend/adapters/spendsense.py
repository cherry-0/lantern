"""
Adapter for the spendsense app.

Two AI pipelines, both calling Google Gemini 2.5 Flash directly from the browser:

  1. Receipt scanning (image → text)
     services/geminiService.ts: scanReceipt(base64Image)
     Sends the receipt image + extraction prompt to Gemini multimodal API.
     Returns JSON: {amount, date, title, category}

  2. Spending insights (text → text)
     services/geminiService.ts: getSpendingInsights(transactions)
     Sends up to 50 recent expense records as JSON to Gemini.
     Returns ~150-word markdown advice.

The app is a pure frontend PWA (Vite/React) — there is no backend server.
All Gemini calls are made client-side with a build-time API key.  There is
therefore no native HTTP mode; both pipelines are replicated via OpenRouter.

Externalizations per request:
  NETWORK  — POST to Google Gemini API
             (generativelanguage.googleapis.com, gemini-2.5-flash):
             Receipt images (pipeline 1) or full transaction list with titles,
             amounts, dates, and categories (pipeline 2) are sent in plaintext.

Execution mode:
  Both USE_APP_SERVERS=true and false use OpenRouter — native mode is not
  applicable because the app has no server component.

Configuration (.env)
--------------------
OPENROUTER_API_KEY  — required for both modes
"""

import json
import sys
from datetime import date
from typing import Any, Dict, List, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_openrouter_api_key, use_app_servers

# ── Prompts (mirrors geminiService.ts exactly) ────────────────────────────────

_SCAN_RECEIPT_PROMPT = (
    "Analyze this receipt. Extract the total amount, the date (ISO format YYYY-MM-DD), "
    "the merchant name as 'title', and guess the best category from this list: "
    "Food, Transport, Shopping, Bills, Entertainment, Health, Education, Others. "
    "Return JSON with keys: amount (number), date (string), title (string), category (string)."
)

_INSIGHTS_PROMPT_TMPL = (
    "You are a financial advisor. Analyze these expenses: {expenses_json}. "
    "Provide 3 short, actionable bullet points about my spending habits. "
    "Be friendly and encouraging. in less than 150 words "
    "Format the output as a simple Markdown list."
)


def _format_receipt_output(result: Dict[str, Any]) -> str:
    parts = []
    if result.get("title"):
        parts.append(f"Merchant: {result['title']}")
    if result.get("amount") is not None:
        parts.append(f"Amount: {result['amount']}")
    if result.get("date"):
        parts.append(f"Date: {result['date']}")
    if result.get("category"):
        parts.append(f"Category: {result['category']}")
    return "\n".join(parts) if parts else str(result)


def _build_synthetic_transactions(text: str) -> List[Dict[str, Any]]:
    """Wrap a text item as a single expense transaction for the insights pipeline."""
    return [
        {
            "date": date.today().isoformat(),
            "title": text[:200],
            "amount": 0.0,
            "category": "Others",
            "type": "expense",
        }
    ]


class SpendSenseAdapter(BaseAdapter):
    """
    Wraps the spendsense Gemini AI pipelines.

    Image modality  → receipt scanning pipeline (scanReceipt)
    Text modality   → spending insights pipeline (getSpendingInsights),
                      with the text item embedded as a transaction title.

    No native mode: the app is a browser-only PWA with no HTTP server.
    """

    name = "spendsense"
    supported_modalities = ["image", "text"]
    env_spec = None

    # ── Availability ──────────────────────────────────────────────────────────

    def check_availability(self) -> Tuple[bool, str]:
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            mode = "NATIVE (serverless equivalent)" if use_app_servers() else "SERVERLESS"
            return True, (
                f"[{mode}] Using OpenRouter to replicate Gemini 2.5 Flash calls. "
                "spendsense is a browser-only PWA — no server to call natively."
            )
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Pipeline dispatch ─────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        modality = input_item.get("modality", "")
        if modality == "image":
            return self._run_receipt_scan(input_item)
        if modality == "text":
            return self._run_insights(input_item)
        return AdapterResult(
            success=False,
            error=f"spendsense supports 'image' and 'text' modalities, got '{modality}'.",
        )

    # ── Pipeline 1: receipt scanning (image → text) ───────────────────────────

    def _run_receipt_scan(self, input_item: Dict[str, Any]) -> AdapterResult:
        image_b64 = input_item.get("image_base64", "")
        if not image_b64:
            return AdapterResult(success=False, error="No image_base64 in input item.")

        print(
            f"[spendsense] Calling OpenRouter (scanReceipt) image_size={len(image_b64)} chars",
            file=sys.stderr, flush=True,
        )

        try:
            raw_response = self._call_openrouter(
                prompt=_SCAN_RECEIPT_PROMPT,
                image_b64=image_b64,
                max_tokens=256,
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        print(f"[spendsense] Receipt scan response: {raw_response[:200]!r}",
              file=sys.stderr, flush=True)

        # Parse JSON from response
        result: Dict[str, Any] = {}
        try:
            text = raw_response.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
        except Exception:
            start, end = raw_response.find("{"), raw_response.rfind("}")
            if start != -1 and end != -1:
                try:
                    result = json.loads(raw_response[start:end + 1])
                except Exception:
                    pass

        output_text = _format_receipt_output(result) if result else raw_response

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    "[Google Gemini API] POST https://generativelanguage.googleapis.com/"
                    "v1beta/models/gemini-2.5-flash:generateContent — "
                    "receipt image (base64 JPEG) sent for data extraction"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"llm_response": raw_response, "parsed": result},
            structured_output=result,
            externalizations=externalizations,
            metadata={"method": "openrouter_scan_receipt"},
        )

    # ── Pipeline 2: spending insights (text → text) ───────────────────────────

    def _run_insights(self, input_item: Dict[str, Any]) -> AdapterResult:
        data = input_item.get("data", "") or input_item.get("text_content", "")
        text = str(data).strip() if data else ""
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        transactions = _build_synthetic_transactions(text)
        prompt = _INSIGHTS_PROMPT_TMPL.format(
            expenses_json=json.dumps(transactions)
        )

        print(
            f"[spendsense] Calling OpenRouter (getSpendingInsights) text={text[:80]!r}",
            file=sys.stderr, flush=True,
        )

        try:
            answer = self._call_openrouter(prompt=prompt, max_tokens=256)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        print(f"[spendsense] Insights: {answer[:120]!r}", file=sys.stderr, flush=True)

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    f"[Google Gemini API] POST https://generativelanguage.googleapis.com/"
                    f"v1beta/models/gemini-2.5-flash:generateContent — "
                    f"expense records including title={text[:120]!r}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=answer,
            raw_output={"text": text, "transactions": transactions, "answer": answer},
            structured_output={"answer": answer},
            externalizations=externalizations,
            metadata={"method": "openrouter_spending_insights"},
        )
