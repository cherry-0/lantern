"""
Adapter for the chat-driven-expense-tracker (FinChain) app.

Core pipeline: natural language expense text → Groq llama3-8b-8192 (via LangChain)
               → structured JSON (item, amount, category) → MongoDB + Pinecone.

The app exposes a FastAPI server on port 8000.  The main endpoint is:
  POST /parse_expense  {"entry": "<expense text>"}
  → {"status": "saved", "parsed": [{"item": ..., "amount": ..., "category": ...}]}

Externalizations the app makes per request:
  NETWORK  — POST to Groq API (llama3-8b-8192): sends the raw expense text to a cloud LLM
  STORAGE  — MongoDB insert: stores {user, raw, parsed, timestamp} in finchain.expenses
  STORAGE  — Pinecone upsert: stores 384-dim embedding of raw text with parsed metadata

Execution mode is controlled by USE_APP_SERVERS in .env:

  USE_APP_SERVERS=true  (NATIVE mode)
    Calls the running FastAPI backend directly via HTTP.
    Start the server first:
      cd target-apps/chat-driven-expense-tracker/backend
      pip install -r requirements.txt
      uvicorn main:app --reload
    Requires in the server's environment:
      GROQ_API_KEY       — Groq API key (llama3-8b-8192)
      MONGO_URI          — MongoDB Atlas connection string
      PINECONE_API_KEY   — Pinecone API key
      PINECONE_INDEX     — Pinecone index name (default: expenses-index)

  USE_APP_SERVERS=false  (SERVERLESS mode)
    Replicates the Groq expense-parsing call via OpenRouter using the same
    prompt template. Fakes MongoDB + Pinecone + Groq externalizations.

Configuration (.env)
--------------------
USE_APP_SERVERS                — "true" / "false"         (default: false)
CHAT_EXPENSE_TRACKER_HOST      — FastAPI base URL          (default: http://localhost:8000)
"""

import json
import re
import sys
from typing import Any, Dict, List, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

# Expense categories as defined by the app
_CATEGORIES = [
    "Food", "Transportation", "Utilities", "Entertainment",
    "Shopping", "Healthcare", "Other",
]

# Prompt template — exactly as in backend/langchain_prompt.py
_PARSE_PROMPT = """\
You are an expense parser. Extract all expense items from the given sentence.

Sentence: "{entry}"

Return ONLY a valid JSON array in this exact format:
[{{"item": "item_name", "amount": number, "category": "category_name"}}]

Categories should be one of: Food, Transportation, Utilities, Entertainment, Shopping, Healthcare, Other

Example: [{{"item": "groceries", "amount": 50, "category": "Food"}}]"""

_DEFAULT_HOST = "http://localhost:8000"


def _parse_items_from_llm(raw: str) -> List[Dict[str, Any]]:
    """Extract the JSON array from a raw LLM response string."""
    # Use greedy match to capture the full array including nested objects
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


def _format_output(parsed: List[Dict[str, Any]], raw_entry: str) -> str:
    """Build a human-readable summary of parsed expense items."""
    if not parsed:
        return f"No expense items could be parsed from: {raw_entry!r}"
    lines = [f"Parsed {len(parsed)} expense item(s) from: {raw_entry!r}"]
    for i, item in enumerate(parsed, 1):
        name = item.get("item", "unknown")
        amount = item.get("amount", 0)
        category = item.get("category", "Other")
        lines.append(f"  {i}. {name} — {amount} ({category})")
    return "\n".join(lines)


class ChatExpenseTrackerAdapter(BaseAdapter):
    """
    Wraps the chat-driven-expense-tracker (FinChain) NLP expense parsing pipeline.

    For Verify: given a natural language text item, run the expense parsing
    pipeline and evaluate whether the raw input and structured output reveal
    private attributes (financial amounts, health-related purchases, location,
    spending behaviour, etc.).

    NATIVE mode     : HTTP POST to the running FastAPI server.
    SERVERLESS mode : OpenRouter call replicating the Groq llama3-8b-8192 parse.
    """

    name = "chat-driven-expense-tracker"
    supported_modalities = ["text"]
    env_spec = None  # No CondaRunner — native mode calls the HTTP server directly

    def __init__(self):
        self._host: str = (get_env("CHAT_EXPENSE_TRACKER_HOST") or _DEFAULT_HOST).rstrip("/")

    # ── Availability ──────────────────────────────────────────────────────────

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            try:
                import requests
                resp = requests.get(f"{self._host}/", timeout=5)
                if resp.ok:
                    return True, f"[NATIVE] Server reachable at {self._host}"
                return False, f"[NATIVE] Server returned {resp.status_code} at {self._host}"
            except Exception as e:
                return False, (
                    f"[NATIVE] Cannot reach server at {self._host}: {e}\n"
                    "Start with: cd target-apps/chat-driven-expense-tracker/backend && "
                    "uvicorn main:app --reload"
                )
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter to replicate Groq expense parsing."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"chat-driven-expense-tracker only supports 'text' modality, "
                      f"got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        entry = str(data).strip() if data else ""
        if not entry:
            return AdapterResult(success=False, error="Empty text input.")

        if use_app_servers():
            return self._run_native(entry)
        return self._run_serverless(entry)

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, entry: str) -> AdapterResult:
        """POST to the running FastAPI server at /parse_expense."""
        try:
            import requests
        except ImportError:
            return AdapterResult(
                success=False,
                error="requests library not installed. Run: pip install requests",
            )

        print(
            f"[chat-expense-tracker] POST {self._host}/parse_expense  "
            f"entry={entry[:80]!r}",
            file=sys.stderr, flush=True,
        )

        try:
            resp = requests.post(
                f"{self._host}/parse_expense",
                json={"entry": entry},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"HTTP request failed: {e}")

        if data.get("status") != "saved":
            return AdapterResult(
                success=False,
                error=f"Server returned unexpected status: {data}",
            )

        parsed: List[Dict[str, Any]] = data.get("parsed", [])
        print(
            f"[chat-expense-tracker] Parsed {len(parsed)} item(s): "
            + ", ".join(
                f"{i.get('item','?')} ${i.get('amount','?')} ({i.get('category','?')})"
                for i in parsed
            ),
            file=sys.stderr, flush=True,
        )
        output_text = _format_output(parsed, entry)

        # The server externalized to Groq, MongoDB, and Pinecone on every request
        externalizations = {
            "NETWORK": (
                f"[Groq API] POST https://api.groq.com/openai/v1/chat/completions — "
                f"model=llama3-8b-8192, expense_text={(entry[:120])!r}"
            ),
            "STORAGE": (
                f"[MongoDB] finchain.expenses insert — "
                f"{{user: 'demo', raw: {(entry[:80])!r}, parsed: {parsed}}}; "
                f"[Pinecone] expenses-index upsert — 384-dim embedding of raw text"
            ),
        }

        structured = {
            "raw_entry": entry,
            "parsed": parsed,
            "status": "saved",
        }

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output=data,
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "native_http", "host": self._host},
        )

    # ── SERVERLESS mode ───────────────────────────────────────────────────────

    def _run_serverless(self, entry: str) -> AdapterResult:
        """
        Replicate the Groq llama3-8b-8192 LangChain call via OpenRouter.
        Uses the exact same prompt template as langchain_prompt.py.
        """
        prompt = _PARSE_PROMPT.format(entry=entry)

        print(
            f"[chat-expense-tracker] Calling OpenRouter (groq/llama3-8b-8192)  "
            f"entry={entry[:80]!r}",
            file=sys.stderr, flush=True,
        )

        try:
            raw_response = self._call_openrouter(
                prompt=prompt,
                max_tokens=512,
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        print(
            f"[chat-expense-tracker] Raw LLM response: {raw_response[:200]!r}",
            file=sys.stderr, flush=True,
        )

        parsed = _parse_items_from_llm(raw_response)
        print(
            f"[chat-expense-tracker] Parsed {len(parsed)} item(s): "
            + ", ".join(
                f"{i.get('item','?')} ${i.get('amount','?')} ({i.get('category','?')})"
                for i in parsed
            ),
            file=sys.stderr, flush=True,
        )
        output_text = _format_output(parsed, entry)

        total_amount = sum(
            item.get("amount", 0) for item in parsed
            if isinstance(item.get("amount"), (int, float))
        )
        categories_hit = list({item.get("category", "Other") for item in parsed})

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    f"[Groq API Fallback] POST https://api.groq.com/openai/v1/chat/completions — "
                    f"model=llama3-8b-8192, entry={(entry[:120])!r}"
                ),
                "STORAGE": (
                    f"[MongoDB Fallback] finchain.expenses insert — "
                    f"{{user: 'demo', raw: {(entry[:80])!r}, "
                    f"parsed: {parsed}, total: {total_amount}}}; "
                    f"[Pinecone Fallback] expenses-index upsert — "
                    f"384-dim embedding, categories={categories_hit}"
                ),
            }
        )

        structured = {
            "raw_entry": entry,
            "parsed": parsed,
            "total_amount": total_amount,
            "categories": categories_hit,
            "status": "parsed",
        }

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"entry": entry, "llm_response": raw_response, "parsed": parsed},
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "serverless_openrouter", "model": "groq/llama3-8b-8192"},
        )
