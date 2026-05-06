"""
Adapter for the NomAI app.

Core client pipeline (target-apps/nom-ai):
  - Image/text chat: ChatController.sendMessage()
    -> optional Firebase Storage image upload
    -> POST /api/v1/chat/messages with text, user_id, image_url/image_data,
       dietary_preferences, allergies, selected_goals
    -> backend agent may call nutrition tools and returns ai_answer + nutrition_data.

  - Direct nutrition analysis: AiRepository.getNutritionData()
    -> POST /api/v1/nutrition/analyze with imageUrl or food_description plus
       dietary preferences, allergies, and goals.

The full NomAI backend is a separate repo/service. Verify therefore defaults to
a serverless OpenRouter replica of the nutrition/chat AI surfaces.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_DEFAULT_HOST = "http://localhost:8080"
_DEFAULT_MODEL = "google/gemini-2.0-flash-001"
_DEFAULT_MAX_TOKENS = 2048


def _encode_image_b64(input_item: Dict[str, Any]) -> str:
    if input_item.get("image_base64"):
        return str(input_item["image_base64"])

    data = input_item.get("data")
    path = input_item.get("path")

    if data is not None:
        try:
            buf = io.BytesIO()
            data.convert("RGB").save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass

    if path:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")

    return ""


def _input_text(input_item: Dict[str, Any]) -> str:
    data = input_item.get("data", "") or input_item.get("text_content", "")
    if isinstance(data, str):
        return data.strip()
    return str(data).strip() if data else ""


def _context_from_item(input_item: Dict[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "user_id": input_item.get("user_id") or "verify-user",
        "dietary_preferences": input_item.get("dietary_preferences") or [],
        "allergies": input_item.get("allergies") or [],
        "selected_goals": input_item.get("selected_goals") or [],
    }
    labels = input_item.get("labels")
    if isinstance(labels, dict) and labels:
        context["input_labels"] = labels
    return context


def _nutrition_prompt(text: str, context: Dict[str, Any], has_image: bool) -> str:
    source = "the attached meal image" if has_image else "the user's food description"
    description = text or ("Analyse this image" if has_image else "")
    return (
        "You are NomAI, an AI nutrition assistant. Analyse the user's meal from "
        f"{source}. Use the user's dietary preferences, allergies, and goals as "
        "context. Estimate nutrition carefully and return a concise response plus "
        "structured nutrition facts.\n\n"
        f"User text: {description}\n"
        f"User context JSON: {json.dumps(context, ensure_ascii=True)}\n\n"
        "Return JSON with keys: message, foodName, portion, portionSize, "
        "confidenceScore, ingredients, primaryConcerns, overallHealthScore, "
        "overallHealthComments. Each ingredient should include name, calories, "
        "protein, carbs, fiber, fat, healthScore, and healthComments."
    )


def _chat_prompt(text: str, context: Dict[str, Any]) -> str:
    return (
        "You are NomAI's conversational nutrition assistant. Answer the user's "
        "food, health, and lifestyle question in a friendly but practical way. "
        "Use dietary preferences, allergies, and selected goals when relevant.\n\n"
        f"User message: {text}\n"
        f"User context JSON: {json.dumps(context, ensure_ascii=True)}"
    )


def _parse_jsonish(text: str) -> Dict[str, Any]:
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
        return parsed if isinstance(parsed, dict) else {"items": parsed}
    except Exception:
        pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(clean[start : end + 1])
            return parsed if isinstance(parsed, dict) else {"items": parsed}
        except Exception:
            pass
    return {"message": clean}


class NomAIAdapter(BaseAdapter):
    """
    Wraps NomAI's nutrition analysis and chat surfaces.

    SERVERLESS mode: OpenRouter replica of the backend AI gateway.
    NATIVE mode: POSTs to a running NomAI backend at NOMAI_HOST. This requires
    the separate backend, Firebase/Remote Config, and provider env vars.
    """

    name = "nom-ai"
    supported_modalities = ["text", "image"]
    env_spec = None

    def __init__(self):
        self._host = (get_env("NOMAI_HOST") or _DEFAULT_HOST).rstrip("/")
        self._model = get_env("NOMAI_MODEL") or _DEFAULT_MODEL
        self._max_tokens = int(get_env("NOMAI_MAX_TOKENS") or _DEFAULT_MAX_TOKENS)
        self._workflow = (get_env("NOMAI_WORKFLOW") or "nutrition").strip().lower()

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            try:
                import requests

                resp = requests.get(f"{self._host}/", timeout=5)
                if resp.status_code < 500:
                    return True, f"[NATIVE] NomAI backend reachable at {self._host}."
                return False, f"[NATIVE] NomAI backend returned {resp.status_code}."
            except Exception as e:
                return False, (
                    f"[NATIVE] Cannot reach NomAI backend at {self._host}: {e}. "
                    "Run the separate NomAI backend or switch this app to serverless mode."
                )

        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, (
                f"[SERVERLESS] Using OpenRouter to replicate NomAI "
                f"{self._workflow} flow ({self._model})."
            )
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        modality = input_item.get("modality")
        if modality not in self.supported_modalities:
            return AdapterResult(
                success=False,
                error=f"nom-ai supports text/image modalities, got {modality!r}.",
            )

        text = _input_text(input_item)
        image_b64 = _encode_image_b64(input_item) if modality == "image" else ""
        if modality == "text" and not text:
            return AdapterResult(success=False, error="Empty text input.")
        if modality == "image" and not image_b64:
            return AdapterResult(success=False, error="Empty image input.")

        if use_app_servers():
            return self._run_native(input_item, text, image_b64)
        return self._run_serverless(input_item, text, image_b64)

    def _run_native(self, input_item: Dict[str, Any], text: str, image_b64: str) -> AdapterResult:
        try:
            import requests
        except ImportError:
            return AdapterResult(success=False, error="requests library not installed.")

        context = _context_from_item(input_item)
        modality = input_item.get("modality")
        endpoint = "/api/v1/chat/messages" if self._workflow == "chat" else "/api/v1/nutrition/analyze"

        if endpoint.endswith("/chat/messages"):
            body = {
                "text": text or "Analyse this image",
                "user_id": context["user_id"],
                "local_message_id": f"verify_{int(datetime.now().timestamp())}",
                "image_url": input_item.get("image_url"),
                "image_data": image_b64 or None,
                "local_time": datetime.now(timezone.utc).isoformat(),
                "dietary_preferences": context["dietary_preferences"],
                "allergies": context["allergies"],
                "selected_goals": context["selected_goals"],
            }
        else:
            body = {
                "imageUrl": input_item.get("image_url"),
                "image_data": image_b64 or None,
                "food_description": text if modality == "text" else None,
                "dietaryPreferences": context["dietary_preferences"],
                "allergies": context["allergies"],
                "selectedGoals": context["selected_goals"],
            }

        print(f"[nom-ai] POST {self._host}{endpoint}", file=sys.stderr, flush=True)
        try:
            resp = requests.post(
                f"{self._host}{endpoint}",
                headers={"Content-Type": "application/json", "Accept": "*/*"},
                json=body,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"NomAI native request failed: {e}")

        output = (
            data.get("ai_answer")
            or data.get("message")
            or json.dumps(data.get("response", data), ensure_ascii=False)
        )
        return AdapterResult(
            success=True,
            output_text=output,
            raw_output=data,
            structured_output=data if isinstance(data, dict) else {},
            externalizations={
                "NETWORK": (
                    f"[NomAI backend] POST {self._host}{endpoint} - "
                    f"payload={json.dumps(body, ensure_ascii=True)[:800]}"
                )
            },
            metadata={"method": "native_http", "host": self._host, "workflow": self._workflow},
        )

    def _run_serverless(self, input_item: Dict[str, Any], text: str, image_b64: str) -> AdapterResult:
        context = _context_from_item(input_item)
        modality = input_item.get("modality")
        has_image = modality == "image"
        workflow = "chat" if self._workflow == "chat" and not has_image else "nutrition"
        prompt = _chat_prompt(text, context) if workflow == "chat" else _nutrition_prompt(text, context, has_image)

        print(
            f"[nom-ai] Calling OpenRouter ({workflow}, {self._model}) "
            f"modality={modality} text={text[:80]!r}",
            file=sys.stderr,
            flush=True,
        )
        try:
            answer = self._call_openrouter(
                prompt=prompt,
                image_b64=image_b64 or None,
                model=self._model,
                max_tokens=self._max_tokens,
                extra_body={"temperature": 0.4},
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        structured = _parse_jsonish(answer) if workflow == "nutrition" else {"ai_answer": answer}
        output = structured.get("message") or structured.get("ai_answer") or answer

        externalizations = self._build_serverless_externalizations()
        extra_network = (
            "[NomAI backend AI gateway replica] Original app POSTs to "
            f"{'/api/v1/chat/messages' if workflow == 'chat' else '/api/v1/nutrition/analyze'} "
            "with text/image_url/image_data, dietary_preferences, allergies, selected_goals. "
            "Backend then uses Gemini/OpenRouter plus web search grounding."
        )
        if "NETWORK" in externalizations:
            externalizations["NETWORK"] += "\n" + extra_network
        else:
            externalizations["NETWORK"] = extra_network
        if has_image:
            externalizations["STORAGE:FIREBASE"] = (
                "[Firebase Storage] Client uploads selected meal image under uploads/<timestamp>.png "
                "and sends the resulting download URL to the backend."
            )
        externalizations["STORAGE"] = (
            "[Firestore/Supabase] NomAI stores chat messages, diet plans, nutrition records, "
            f"user context={json.dumps(context, ensure_ascii=True)}"
        )
        externalizations["NETWORK:SEARCH"] = (
            "[Exa/DuckDuckGo] Backend may issue food/nutrition web searches for USDA/FDA/brand facts."
        )

        return AdapterResult(
            success=True,
            output_text=str(output),
            raw_output={
                "text": text,
                "workflow": workflow,
                "response": answer,
                "structured": structured,
                "has_image": has_image,
            },
            structured_output=structured,
            externalizations=externalizations,
            metadata={
                "method": "serverless_openrouter",
                "workflow": workflow,
                "model": self._model,
                "modality": modality,
            },
        )
