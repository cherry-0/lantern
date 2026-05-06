"""
Adapter for the EduMind app.

Core backend pipeline (target-apps/edumind/backend):
  POST /api/ai/summarize     -> buildPrompt("summarize", ...)
  POST /api/ai/tutor         -> buildPrompt("tutor", ...)
  POST /api/ai/generate-quiz -> buildPrompt("quiz", ...)
  -> aiService.chatCompletion() -> Gemini primary provider
  -> parsed response returned to the user.

For Verify we target the guest-accessible text AI tools. Each text item is
embedded as the content/question sent to EduMind's AI service.

Configuration (.env)
--------------------
USE_APP_SERVERS  - "true" / "false" (default: false)
EDUMIND_HOST     - backend base URL for native mode (default: http://localhost:5000)
EDUMIND_TOOL     - summarize | tutor | quiz (default: summarize)
OPENROUTER_API_KEY - required for serverless mode
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_DEFAULT_HOST = "http://localhost:5000"
_DEFAULT_MAX_TOKENS = 2048
_SUPPORTED_TOOLS = {"summarize", "tutor", "quiz"}


def _clean_tool(value: str | None) -> str:
    tool = (value or "summarize").strip().lower()
    return tool if tool in _SUPPORTED_TOOLS else "summarize"


def _build_prompt(tool: str, text: str) -> str:
    """
    Mirror buildPrompt() in target-apps/edumind/backend/controllers/ai.js for
    the three guest-accessible text tools.
    """
    if tool == "tutor":
        return (
            "You are a helpful AI tutor. Answer the following question clearly and "
            "educationally:\n\n"
            "Question: What important facts, entities, or study-relevant details are "
            "present in the provided material?\n\n"
            f"Context: {text}"
        )

    if tool == "quiz":
        return (
            "Generate exactly 5 medium-level multiple choice quiz questions from "
            "the following content. Return ONLY a valid JSON array with no extra "
            'text. Each item must have: "question" (string), "options" (array of '
            '4 strings), "correctAnswer" (0-indexed number).\n\n'
            f"Content:\n{text}"
        )

    return (
        "Transform the following content into a professional, high-fidelity medium "
        "synthesis. \n"
        "      Synthesis Type: general\n"
        "      \n"
        f"      Content:\n\n{text}"
    )


def _parse_output(tool: str, raw_text: str) -> Dict[str, Any]:
    clean = raw_text.replace("```json", "").replace("```", "").strip()
    if tool == "summarize":
        return {"summary": clean}
    if tool == "tutor":
        return {"answer": clean}
    try:
        parsed = json.loads(clean)
    except Exception:
        return {"questions": [], "raw": clean}
    return {"questions": parsed}


class EduMindAdapter(BaseAdapter):
    """
    Wraps EduMind's text AI tools.

    NATIVE mode     : POSTs to a running EduMind backend. Start it with:
                      cd target-apps/edumind/backend && npm install && npm run dev
    SERVERLESS mode : Replicates EduMind's prompt construction via OpenRouter.
    """

    name = "edumind"
    supported_modalities = ["text"]
    env_spec = None

    def __init__(self):
        self._host = (get_env("EDUMIND_HOST") or _DEFAULT_HOST).rstrip("/")
        self._tool = _clean_tool(get_env("EDUMIND_TOOL"))
        self._max_tokens = int(get_env("EDUMIND_MAX_TOKENS") or _DEFAULT_MAX_TOKENS)

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            try:
                import requests

                resp = requests.get(f"{self._host}/api/health", timeout=5)
                if resp.ok:
                    return True, (
                        f"[NATIVE] EduMind backend reachable at {self._host} "
                        f"(tool={self._tool})."
                    )
                return False, f"[NATIVE] EduMind backend returned {resp.status_code}."
            except Exception as e:
                return False, (
                    f"[NATIVE] Cannot reach EduMind backend at {self._host}: {e}\n"
                    "Start with: cd target-apps/edumind/backend && npm install && npm run dev"
                )

        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, (
                f"[SERVERLESS] Using OpenRouter to replicate EduMind {self._tool} flow."
            )
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"edumind only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        text = str(data).strip() if data else ""
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        if use_app_servers():
            return self._run_native(text)
        return self._run_serverless(text)

    def _run_native(self, text: str) -> AdapterResult:
        try:
            import requests
        except ImportError:
            return AdapterResult(
                success=False,
                error="requests library not installed. Run: pip install requests",
            )

        endpoint, payload = self._native_request(text)
        print(
            f"[edumind] POST {self._host}{endpoint}  tool={self._tool} text={text[:80]!r}",
            file=sys.stderr,
            flush=True,
        )

        try:
            resp = requests.post(f"{self._host}{endpoint}", json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"HTTP request failed: {e}")

        if data.get("success") is False:
            return AdapterResult(success=False, error=f"Server error: {data}")

        structured = data.get("data") or {}
        output_text = self._output_text_from_structured(structured)
        if not output_text:
            output_text = json.dumps(structured, ensure_ascii=False)

        externalizations = {
            "NETWORK": (
                "[Google Gemini API via EduMind] POST "
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-flash:generateContent - tool={self._tool}, "
                f"prompt includes input={text[:120]!r}"
            ),
            "STORAGE": (
                "[MongoDB] AIUsage record written by GeminiService with provider='gemini', "
                f"toolType={self._tool!r}"
            ),
        }

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output=data,
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "native_http", "host": self._host, "tool": self._tool},
        )

    def _run_serverless(self, text: str) -> AdapterResult:
        prompt = _build_prompt(self._tool, text)
        print(
            f"[edumind] Calling OpenRouter (EduMind {self._tool})  text={text[:80]!r}",
            file=sys.stderr,
            flush=True,
        )

        try:
            response = self._call_openrouter(
                prompt=prompt,
                max_tokens=self._max_tokens,
                model="google/gemini-2.0-flash-001",
                extra_body={"temperature": 0.7},
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        structured = _parse_output(self._tool, response)
        output_text = self._output_text_from_structured(structured) or response

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    "[Google Gemini API via EduMind] POST "
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-1.5-flash:generateContent - tool={self._tool}, "
                    f"prompt includes input={text[:120]!r}"
                ),
                "STORAGE": (
                    "[MongoDB] AIUsage record would be written by GeminiService with "
                    f"provider='gemini', toolType={self._tool!r}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"text": text, "response": response, "tool": self._tool},
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "serverless_openrouter", "tool": self._tool},
        )

    def _native_request(self, text: str) -> Tuple[str, Dict[str, Any]]:
        if self._tool == "tutor":
            return "/api/ai/tutor", {
                "question": (
                    "What important facts, entities, or study-relevant details are "
                    "present in this material?"
                ),
                "context": text,
            }
        if self._tool == "quiz":
            return "/api/ai/generate-quiz", {
                "text": text,
                "numQuestions": 5,
                "difficulty": "medium",
            }
        return "/api/ai/summarize", {
            "text": text,
            "type": "general",
            "length": "medium",
        }

    @staticmethod
    def _output_text_from_structured(structured: Dict[str, Any]) -> str:
        for key in ("summary", "answer"):
            value = structured.get(key)
            if value:
                return str(value)
        if structured.get("questions"):
            return json.dumps(structured["questions"], ensure_ascii=False)
        return ""
