"""
Adapter for the SGPA (Study Guide & Personal Assistant) app.

Core pipeline: study question / topic text → Gemini 2.5 Flash → academic
explanation, summary, or quiz response.

SGPA is a Streamlit web app with three modes (Explainer, Summarizer, Quizzer)
all backed by Google Gemini 2.5 Flash.  The Streamlit UI is not automatable,
so both execution modes use the OpenRouter serverless path replicating the
Explainer workflow (the most privacy-relevant mode: user supplies personal
study context and the model generates a personalised academic response).
"""

import sys
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import AdapterResult, BaseAdapter
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_MODEL = "google/gemini-2.5-flash"
_DEFAULT_MAX_TOKENS = 2048

_SYSTEM = (
    "You are Study Buddy, SGPA's AI-powered academic explainer. "
    "When given a concept or question, provide a clear explanation with: "
    "a simple definition or analogy, a step-by-step breakdown or key characteristics, "
    "any common misconceptions, and 2-3 Key Takeaways for revision. "
    "Use concise Markdown formatting."
)


class SGPAAdapter(BaseAdapter):
    """
    Replicates SGPA's Explainer workflow via OpenRouter.

    SGPA is Streamlit-based and not automatable natively; both
    USE_APP_SERVERS=true and false fall through to the serverless path.
    """

    name = "sgpa"
    supported_modalities = ["text"]
    env_spec = None

    def __init__(self):
        self._max_tokens = int(get_env("SGPA_MAX_TOKENS") or _DEFAULT_MAX_TOKENS)

    def check_availability(self) -> Tuple[bool, str]:
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            mode = "NATIVE (serverless equivalent)" if use_app_servers() else "SERVERLESS"
            return True, (
                f"[{mode}] Using OpenRouter to replicate SGPA's Gemini 2.5 Flash "
                f"Explainer pipeline ({_MODEL}). Streamlit UI is not automatable."
            )
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"sgpa only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        text = str(data).strip() if data else ""
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        return self._run_serverless(text)

    def _run_serverless(self, text: str) -> AdapterResult:
        prompt = f"{_SYSTEM}\n\nUser: {text}"

        print(
            f"[sgpa] Calling OpenRouter ({_MODEL}) text={text[:80]!r}",
            file=sys.stderr,
            flush=True,
        )

        try:
            raw_response = self._call_openrouter(
                prompt=prompt,
                model=_MODEL,
                max_tokens=self._max_tokens,
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    f"[SGPA / Gemini API] POST https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.5-flash:generateContent "
                    f"model=gemini-2.5-flash, input={text[:160]!r}"
                ),
                "UI": (
                    f"Explainer mode renders response: {raw_response!r}"
                ),
            }
        )
        # STORAGE always fires: log_usage() appends to logs/usage_log.csv after every response
        externalizations["STORAGE"] = (
            f"[SGPA / usage log] Appended to logs/usage_log.csv: "
            f"session_id=<uuid4>, mode='💡 Explainer', sub_mode='', "
            f"topic={text[:50]!r}, had_pdf=0, "
            f"prompt_chars={len(text)}, response_chars={len(raw_response)}, "
            f"visuals_enabled=1, visuals_detected=<types>, visuals_used=<0|1>"
        )

        return AdapterResult(
            success=True,
            output_text=raw_response,
            raw_output={"input": text, "raw_response": raw_response},
            structured_output={"response": raw_response},
            externalizations=externalizations,
            metadata={
                "method": "serverless_openrouter",
                "workflow": "explainer",
                "model": _MODEL,
            },
        )
