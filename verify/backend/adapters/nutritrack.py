"""
Adapter for the nutri-track app.

Core pipeline: user text -> NutriCoach Gemini prompt -> AI nutrition advice.

NutriTrack is an Android app whose AI surfaces are implemented in
GenAIViewModel:
  - sendRequest(): end-user NutriCoach chat
  - sendAnalysisRequest(): clinician dashboard pattern/trend analysis
  - sendClinicianQuestion(): clinician Q&A over aggregate patient stats

For Verify we target sendRequest(), because it externalizes the user's
free-text nutrition question plus profile/nutrition context to Gemini.

Execution mode:
  Native Android automation is not implemented for this app yet.
  Both USE_APP_SERVERS modes use the serverless equivalent: an OpenRouter call
  that mirrors NutriCoach's Gemini request shape.
"""

import sys
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_DEFAULT_MODEL = "google/gemini-2.5-flash"
_DEFAULT_MAX_TOKENS = 2048
_FOLLOW_UPS_DELIMITER = "SUGGESTED_FOLLOW_UPS:"

_NUTRITION_GUIDELINES_SUMMARY = """
Use Australian adult nutrition guidance covering discretionary foods, vegetables,
fruit, grains and wholegrains, meat and alternatives, dairy and alternatives,
water, saturated and unsaturated fats, sodium, added sugars, and alcohol.
Prefer practical, food-level suggestions and avoid medical diagnosis.
""".strip()


class NutriTrackAdapter(BaseAdapter):
    """Replicate NutriTrack's NutriCoach text chat workflow."""

    name = "nutri-track"
    supported_modalities = ["text"]
    env_spec = None

    def __init__(self):
        self._model = get_env("NUTRITRACK_MODEL") or _DEFAULT_MODEL
        self._max_tokens = int(get_env("NUTRITRACK_MAX_TOKENS") or _DEFAULT_MAX_TOKENS)

    def check_availability(self) -> Tuple[bool, str]:
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            mode = "NATIVE (serverless equivalent)" if use_app_servers() else "SERVERLESS"
            return True, (
                f"[{mode}] Using OpenRouter to replicate NutriTrack NutriCoach "
                f"Gemini chat ({self._model}). Native Android UI automation is not wired."
            )
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"nutri-track only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        text = str(data).strip() if data else ""
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        user_stats_json = self._build_user_stats_context(input_item)
        return self._run_serverless(text, user_stats_json)

    def _run_serverless(self, text: str, user_stats_json: str) -> AdapterResult:
        prompt = self._build_nutricoach_prompt(text, user_stats_json)

        print(
            f"[nutri-track] Calling OpenRouter ({self._model} NutriCoach) text={text[:80]!r}",
            file=sys.stderr,
            flush=True,
        )

        try:
            raw_response = self._call_openrouter(
                prompt=prompt,
                model=self._model,
                max_tokens=self._max_tokens,
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

        main_answer, follow_ups = self._parse_follow_ups(raw_response)
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    "[Gemini API] generateContentStream — "
                    f"model={self._model}, "
                    f"user_query={text[:160]!r}, user_stats={user_stats_json[:240]!r}"
                ),
                "STORAGE": (
                    "[Room DB] chat_messages insert — "
                    f"user message={text[:120]!r}; ai response={main_answer!r}"
                ),
                "UI": (
                    f"NutriCoach displays answer={main_answer!r}; "
                    f"follow_ups={follow_ups}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=main_answer,
            raw_output={
                "query": text,
                "user_stats_json": user_stats_json,
                "raw_response": raw_response,
            },
            structured_output={
                "answer": main_answer,
                "suggested_follow_ups": follow_ups,
            },
            externalizations=externalizations,
            metadata={
                "method": "serverless_openrouter",
                "workflow": "nutricoach_chat",
                "model": self._model,
            },
        )

    @staticmethod
    def _build_nutricoach_prompt(user_query: str, user_stats_json: str) -> str:
        stats_info = (
            f" User's current stats (in JSON): {user_stats_json}."
            if user_stats_json and user_stats_json != "{}"
            else ""
        )
        return (
            "You are NutriCoach, a helpful and friendly AI nutrition assistant."
            f"{stats_info} Refer to these nutritional guidelines for your answer: "
            f"{_NUTRITION_GUIDELINES_SUMMARY}. "
            f"Please provide a concise and informative answer to the user's query: {user_query!r}. "
            "Your answer should be grounded in the provided nutritional guidelines. "
            f"After your main answer, on a new separate line, write '{_FOLLOW_UPS_DELIMITER}' "
            "followed by a comma-separated list of 2-3 brief follow-up questions."
        )

    @staticmethod
    def _build_user_stats_context(input_item: Dict[str, Any]) -> str:
        """
        Build lightweight app-style profile context when dataset metadata exists.

        Text datasets usually provide only free text. If a loader supplies extra
        fields, include them because NutriTrack's real prompt includes user stats.
        """
        context: Dict[str, Any] = {}
        for key in (
            "age",
            "gender",
            "location",
            "health_condition",
            "dietary_preferences",
            "persona",
            "biggestMealTime",
            "sleepTime",
            "wakeUpTime",
        ):
            if key in input_item and input_item[key] not in (None, ""):
                context[key] = input_item[key]

        labels = input_item.get("labels")
        if isinstance(labels, dict) and labels:
            context["inputLabels"] = labels

        if not context:
            return "{}"

        import json

        return json.dumps(context, ensure_ascii=True, default=str)

    @staticmethod
    def _parse_follow_ups(raw_response: str) -> Tuple[str, list[str]]:
        if _FOLLOW_UPS_DELIMITER not in raw_response:
            return raw_response.strip(), []
        main, follow_up_text = raw_response.split(_FOLLOW_UPS_DELIMITER, 1)
        follow_ups = [item.strip() for item in follow_up_text.split(",") if item.strip()]
        return main.strip(), follow_ups
