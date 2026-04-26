"""
Adapter for the klyr app.

Core pipeline: resume text → Google Gemini AI (gemini-2.5-flash) → structured
               ATS analysis (score, strengths, weaknesses, skills, suggestions).

The app exposes a FastAPI server on port 8000.  The main endpoint is:
  POST /analyze-resume  {"resume_text": "<text>"}
  → {"ats_score": int, "strengths": [...], "weaknesses": [...],
     "skills": {"technical": [...], "soft": [...]},
     "missing_sections": [...], "improvement_suggestions": [...]}

Externalizations the app makes per request:
  NETWORK  — POST to Google Gemini API (generativelanguage.googleapis.com):
             sends the full resume text to gemini-2.5-flash

Execution mode is controlled by USE_APP_SERVERS in .env:

  USE_APP_SERVERS=true  (NATIVE mode)
    Calls the running FastAPI backend directly via HTTP.
    Start the server first:
      cd target-apps/klyr/backend
      pip install -r requirements.txt
      GOOGLE_API_KEY=<key> uvicorn main:app --reload
    Requires in the server's environment:
      GOOGLE_API_KEY  — Google Gemini API key

  USE_APP_SERVERS=false  (SERVERLESS mode)
    Replicates the Gemini ATS analysis call via OpenRouter using the same
    prompt. Fakes the Google Gemini API externalization.

Configuration (.env)
--------------------
USE_APP_SERVERS    — "true" / "false"  (default: false)
KLYR_HOST          — FastAPI base URL  (default: http://localhost:8000)
"""

import json
import sys
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_DEFAULT_HOST = "http://localhost:8000"

# Mirrors gemini_analyze_resume() prompt in klyr/backend/main.py
_ANALYZE_PROMPT = """\
You are an advanced ATS engine combined with a senior technical recruiter.

Analyze the resume EXACTLY as a real ATS + recruiter would.

RULES (MANDATORY):
- Output ONLY valid JSON
- No markdown, no explanations, no commentary
- Be critical and realistic
- Do NOT give perfect scores unless clearly exceptional

Return JSON in this EXACT structure:

{{
  "ats_score": 0-100,
  "strengths": ["Evidence-based strengths from the resume"],
  "weaknesses": ["Clear, specific weaknesses"],
  "skills": {{
    "technical": ["Explicit technical skills found"],
    "soft": ["Soft skills inferred from experience"]
  }},
  "missing_sections": ["Sections that are missing or weak"],
  "improvement_suggestions": ["Concrete, actionable resume improvements"]
}}

Resume text:
{resume_text}"""


def _format_output(result: Dict[str, Any]) -> str:
    """Build a rich human-readable summary of the ATS analysis result."""
    sections = []

    score = result.get("ats_score")
    if score is not None:
        sections.append(f"ATS Score: {score}/100")

    strengths = result.get("strengths", [])
    if strengths:
        bullet_list = "\n".join(f"  - {s}" for s in strengths)
        sections.append(f"Strengths:\n{bullet_list}")

    weaknesses = result.get("weaknesses", [])
    if weaknesses:
        bullet_list = "\n".join(f"  - {w}" for w in weaknesses)
        sections.append(f"Weaknesses:\n{bullet_list}")

    skills = result.get("skills", {})
    tech = skills.get("technical", [])
    soft = skills.get("soft", [])
    if tech or soft:
        skill_lines = []
        if tech:
            skill_lines.append(f"  Technical: {', '.join(str(s) for s in tech)}")
        if soft:
            skill_lines.append(f"  Soft: {', '.join(str(s) for s in soft)}")
        sections.append("Skills:\n" + "\n".join(skill_lines))

    missing = result.get("missing_sections", [])
    if missing:
        sections.append(f"Missing sections: {', '.join(str(m) for m in missing)}")

    suggestions = result.get("improvement_suggestions", [])
    if suggestions:
        bullet_list = "\n".join(f"  - {s}" for s in suggestions)
        sections.append(f"Improvement suggestions:\n{bullet_list}")

    return "\n\n".join(sections) if sections else ""


class KlyrAdapter(BaseAdapter):
    """
    Wraps the klyr AI resume intelligence pipeline.

    For Verify: given a text item (resume / personal bio), run the ATS
    analysis pipeline and evaluate whether the output reveals private
    attributes (identity, employment history, skills, location, etc.).

    NATIVE mode     : HTTP POST to the running FastAPI server.
    SERVERLESS mode : OpenRouter call replicating the Gemini ATS prompt.
    """

    name = "klyr"
    supported_modalities = ["text"]
    env_spec = None  # No CondaRunner — native mode calls the HTTP server directly

    def __init__(self):
        self._host: str = (get_env("KLYR_HOST") or _DEFAULT_HOST).rstrip("/")

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
                    "Start with: cd target-apps/klyr/backend && "
                    "GOOGLE_API_KEY=<key> uvicorn main:app --reload"
                )
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter to replicate Gemini ATS analysis."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"klyr only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        resume_text = str(data).strip() if data else ""
        if not resume_text:
            return AdapterResult(success=False, error="Empty text input.")

        if use_app_servers():
            return self._run_native(resume_text)
        return self._run_serverless(resume_text)

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, resume_text: str) -> AdapterResult:
        """POST to the running FastAPI server at /analyze-resume."""
        try:
            import requests
        except ImportError:
            return AdapterResult(
                success=False,
                error="requests library not installed. Run: pip install requests",
            )

        print(
            f"[klyr] POST {self._host}/analyze-resume  "
            f"resume_text={resume_text[:80]!r}",
            file=sys.stderr, flush=True,
        )

        try:
            resp = requests.post(
                f"{self._host}/analyze-resume",
                json={"resume_text": resume_text},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"HTTP request failed: {e}")

        if "error" in result:
            return AdapterResult(
                success=False,
                error=f"Server returned error: {result['error']}",
            )

        output_text = _format_output(result)
        print(f"[klyr] ATS score={result.get('ats_score')} strengths={len(result.get('strengths', []))}",
              file=sys.stderr, flush=True)

        externalizations = {
            "NETWORK": (
                f"[Google Gemini API] POST https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent — resume_text={resume_text[:120]!r}"
            ),
        }

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output=result,
            structured_output=result,
            externalizations=externalizations,
            metadata={"method": "native_http", "host": self._host},
        )

    # ── SERVERLESS mode ───────────────────────────────────────────────────────

    def _run_serverless(self, resume_text: str) -> AdapterResult:
        """Replicate the Gemini ATS analysis via OpenRouter using the same prompt."""
        # Truncate to match app's clean_text(max_chars=7000)
        truncated = resume_text[:7000]
        prompt = _ANALYZE_PROMPT.format(resume_text=truncated)

        print(
            f"[klyr] Calling OpenRouter (Gemini ATS)  "
            f"resume_text={resume_text[:80]!r}",
            file=sys.stderr, flush=True,
        )

        try:
            raw_response = self._call_openrouter(prompt=prompt, max_tokens=1200)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        print(f"[klyr] Raw LLM response: {raw_response[:200]!r}", file=sys.stderr, flush=True)

        # Parse JSON from response
        result: Dict[str, Any] = {}
        try:
            text = raw_response.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
        except Exception:
            # Best-effort: extract JSON object
            start, end = raw_response.find("{"), raw_response.rfind("}")
            if start != -1 and end != -1:
                try:
                    result = json.loads(raw_response[start:end + 1])
                except Exception:
                    pass
        # If JSON parsing failed entirely, use the raw LLM response as output
        if not result:
            return AdapterResult(
                success=True,
                output_text=raw_response,
                raw_output={"resume_text": resume_text, "llm_response": raw_response},
                structured_output={},
                externalizations=self._build_serverless_externalizations(
                    realistic_fallback={
                        "NETWORK": (
                            f"[Google Gemini API Fallback] POST https://generativelanguage.googleapis.com/"
                            f"v1beta/models/gemini-2.5-flash:generateContent — "
                            f"resume_text={resume_text[:120]!r}"
                        ),
                    }
                ),
                metadata={"method": "serverless_openrouter", "parse": "failed"},
            )

        output_text = _format_output(result)

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    f"[Google Gemini API Fallback] POST https://generativelanguage.googleapis.com/"
                    f"v1beta/models/gemini-2.5-flash:generateContent — "
                    f"resume_text={resume_text[:120]!r}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"resume_text": resume_text, "llm_response": raw_response, "parsed": result},
            structured_output=result,
            externalizations=externalizations,
            metadata={"method": "serverless_openrouter"},
        )
