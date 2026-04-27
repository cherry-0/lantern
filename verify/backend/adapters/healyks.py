"""
Adapter for the healyks app.

Core pipeline: symptom text -> /api/symptoms/analyze -> Gemini 2.0 Flash
health report with condition, recommendation, and home remedies.

Healyks is an Android app. Its Analyze screen sends a GeminiBody containing a
single "symptoms" string to the backend endpoint api/symptoms/analyze with a
Firebase auth token. Native Android/backend automation is not wired here, so
Verify uses an OpenRouter serverless equivalent that mirrors that text-to-text
symptom analysis flow.
"""

import json
import sys
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import AdapterResult, BaseAdapter
from verify.backend.utils.config import get_openrouter_api_key, use_app_servers

_MODEL = "google/gemini-2.0-flash-001"
_BACKEND_BASE_URL = "http://ec2-13-232-188-167.ap-south-1.compute.amazonaws.com:5000/"
_ANALYZE_ENDPOINT = "api/symptoms/analyze"


class HealyksAdapter(BaseAdapter):
    """Replicate Healyks' symptom analysis workflow."""

    name = "healyks"
    supported_modalities = ["text"]
    env_spec = None

    def check_availability(self) -> Tuple[bool, str]:
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            mode = "NATIVE (serverless equivalent)" if use_app_servers() else "SERVERLESS"
            return True, (
                f"[{mode}] Using OpenRouter to replicate Healyks symptom analysis "
                f"with Gemini 2.0 Flash ({_MODEL}). Native Android/backend auth is not wired."
            )
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"healyks only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        symptoms = str(data).strip() if data else ""
        if not symptoms:
            return AdapterResult(success=False, error="Empty text input.")

        return self._run_serverless(symptoms)

    def _run_serverless(self, symptoms: str) -> AdapterResult:
        prompt = self._build_symptom_prompt(symptoms)

        print(
            f"[healyks] Calling OpenRouter ({_MODEL} symptom analysis) symptoms={symptoms[:80]!r}",
            file=sys.stderr,
            flush=True,
        )

        try:
            raw_response = self._call_openrouter(
                prompt=prompt,
                model=_MODEL,
                max_tokens=550,
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        report = self._parse_report(raw_response)
        output_text = self._format_report(report, raw_response)
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    f"[Healyks Backend] POST {_BACKEND_BASE_URL}{_ANALYZE_ENDPOINT} "
                    f"Authorization=<Firebase ID token>, body={{'symptoms': {symptoms[:160]!r}}}; "
                    f"[Gemini API] model=gemini-2.0-flash, symptoms={symptoms[:160]!r}"
                ),
                "UI": (
                    "Analyze screen displays "
                    f"condition={report.get('condition', '')[:120]!r}; "
                    f"recommendation={report.get('recommendation', '')[:160]!r}; "
                    f"homeRemedies={report.get('homeRemedies', '')[:160]!r}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={
                "symptoms": symptoms,
                "raw_response": raw_response,
            },
            structured_output=report,
            externalizations=externalizations,
            metadata={
                "method": "serverless_openrouter",
                "workflow": "symptom_analysis",
                "backend_endpoint": _ANALYZE_ENDPOINT,
            },
        )

    @staticmethod
    def _build_symptom_prompt(symptoms: str) -> str:
        return (
            "You are powering Healyks, a rural healthcare assistant symptom checker. "
            "Analyze the user's symptoms and return a JSON object with exactly these keys: "
            '"condition", "recommendation", "homeRemedies". '
            "The condition should be a cautious possible condition or category, not a definitive diagnosis. "
            "The recommendation should include practical next steps and urgent warning signs when relevant. "
            "The homeRemedies field should contain safe, general self-care suggestions and should tell the "
            "user to seek professional medical advice for serious, worsening, or persistent symptoms. "
            f"Symptoms: {symptoms!r}"
        )

    @staticmethod
    def _parse_report(raw_response: str) -> Dict[str, str]:
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {
                "condition": "",
                "recommendation": raw_response.strip(),
                "homeRemedies": "",
            }

        return {
            "condition": str(parsed.get("condition", "")).strip(),
            "recommendation": str(parsed.get("recommendation", "")).strip(),
            "homeRemedies": str(parsed.get("homeRemedies", "")).strip(),
        }

    @staticmethod
    def _format_report(report: Dict[str, str], raw_response: str) -> str:
        if not any(report.values()):
            return raw_response.strip()

        lines = []
        if report.get("condition"):
            lines.append(f"Condition: {report['condition']}")
        if report.get("recommendation"):
            lines.append(f"Recommendation: {report['recommendation']}")
        if report.get("homeRemedies"):
            lines.append(f"Home remedies: {report['homeRemedies']}")
        return "\n".join(lines)
