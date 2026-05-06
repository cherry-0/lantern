"""
Adapter for the TinyTavern app.

Core app pipeline (target-apps/tinytavern):
  ChatScreen.sendMessage()
  -> CharacterCardService.generateSystemPrompt(selectedCharacter, userName)
  -> OpenRouterService.sendMessage(model, messages, systemPrompt?)
  -> POST https://openrouter.ai/api/v1/chat/completions
  -> assistant reply stored in local AsyncStorage message history.

For Verify we replicate the default Assistant character chat flow as text->text.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers

_DEFAULT_MAX_TOKENS = 1000
_DEFAULT_MODEL = "google/gemini-2.0-flash-001"
_DEFAULT_USER_NAME = "User"
_DEFAULT_CHARACTER = {
    "name": "Assistant",
    "description": (
        "A helpful AI assistant ready to chat, answer questions, and help with "
        "various tasks. I'm knowledgeable, friendly, and always eager to assist you."
    ),
    "personality": (
        "Helpful, knowledgeable, friendly, and professional. I enjoy having "
        "conversations and helping users with their questions and tasks."
    ),
    "scenario": (
        "You are chatting with an AI assistant in a casual conversation setting. "
        "Feel free to ask questions, discuss topics, or just have a friendly chat."
    ),
    "first_mes": (
        "Hello! I'm your AI assistant. I'm here to help with any questions you "
        "might have or just to have a friendly conversation. What would you like "
        "to talk about today?"
    ),
    "mes_example": (
        "<START>\n"
        "{{user}}: Hello, can you help me?\n"
        "{{char}}: Of course! I'd be happy to help you. What do you need assistance with?\n"
        "{{user}}: I need help with planning my day.\n"
        "{{char}}: I'd be glad to help you plan your day! Let's start by talking "
        "about what you need to accomplish. What are your main priorities or tasks for today?"
    ),
}


def _replace_variables(text: str, char_name: str, user_name: str) -> str:
    return (
        text.replace("{{char}}", char_name)
        .replace("{{CHAR}}", char_name)
        .replace("{{Char}}", char_name)
        .replace("{{user}}", user_name)
        .replace("{{USER}}", user_name)
        .replace("{{User}}", user_name)
    )


def _build_default_character_messages(user_name: str) -> List[Dict[str, str]]:
    """
    Mirror CharacterCardService.generateSystemPrompt() for TinyTavern's default
    Assistant character from characterStorage.ts.
    """
    data = _DEFAULT_CHARACTER
    char_name = data["name"]

    def repl(value: str) -> str:
        return _replace_variables(value, char_name, user_name)

    return [
        {
            "role": "system",
            "content": (
                f"You play a role as {char_name}.\n"
                f"Write {char_name}'s next reply in a fictional conversation between "
                f"you and {user_name}."
            ),
        },
        {"role": "system", "content": f"Description:\n{repl(data['description'])}"},
        {"role": "system", "content": f"{char_name}'s personality:\n{repl(data['personality'])}"},
        {"role": "system", "content": f"Scenario:\n{repl(data['scenario'])}"},
        {"role": "system", "content": f"{char_name}'s message example:\n{repl(data['mes_example'])}"},
        {"role": "system", "content": "[Start a new Chat]"},
        {"role": "assistant", "content": repl(data["first_mes"])},
    ]


class TinyTavernAdapter(BaseAdapter):
    """
    Wraps TinyTavern's default character chat pipeline.

    TinyTavern has no separate backend server; the mobile app calls OpenRouter,
    OpenAI, or Ollama directly. Verify therefore uses a serverless OpenRouter
    replica of the OpenRouter branch.
    """

    name = "tinytavern"
    supported_modalities = ["text"]
    env_spec = None

    def __init__(self):
        self._model = get_env("TINYTAVERN_MODEL") or _DEFAULT_MODEL
        self._max_tokens = int(get_env("TINYTAVERN_MAX_TOKENS") or _DEFAULT_MAX_TOKENS)
        self._user_name = get_env("TINYTAVERN_USER_NAME") or _DEFAULT_USER_NAME

    def check_availability(self) -> Tuple[bool, str]:
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            mode = "NATIVE (serverless equivalent)" if use_app_servers() else "SERVERLESS"
            return True, (
                f"[{mode}] Using OpenRouter to replicate TinyTavern default "
                f"character chat (model={self._model})."
            )
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"tinytavern only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        text = str(data).strip() if data else ""
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        return self._run_serverless(text)

    def _run_serverless(self, text: str) -> AdapterResult:
        api_messages = _build_default_character_messages(self._user_name)
        api_messages.append({"role": "user", "content": text})

        # BaseAdapter._call_openrouter supports a single user prompt, so flatten
        # TinyTavern's message array while preserving roles and ordering.
        prompt = "\n\n".join(
            f"{msg['role'].upper()}:\n{msg['content']}" for msg in api_messages
        )

        print(
            f"[tinytavern] Calling OpenRouter default character chat  text={text[:80]!r}",
            file=sys.stderr,
            flush=True,
        )

        try:
            answer = self._call_openrouter(
                prompt=prompt,
                model=self._model,
                max_tokens=self._max_tokens,
                extra_body={"temperature": 0.7},
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": (
                    "[OpenRouter API] POST https://openrouter.ai/api/v1/chat/completions - "
                    f"model={self._model}, messages include TinyTavern Assistant "
                    f"character prompt and user={text[:120]!r}"
                ),
                "STORAGE": (
                    "[AsyncStorage] per-character chat history stores user message "
                    f"{text[:80]!r} and assistant response {answer[:120]!r}"
                ),
            }
        )

        return AdapterResult(
            success=True,
            output_text=answer,
            raw_output={
                "text": text,
                "response": answer,
                "messages": api_messages,
                "character": _DEFAULT_CHARACTER["name"],
            },
            structured_output={"response": answer, "character": _DEFAULT_CHARACTER["name"]},
            externalizations=externalizations,
            metadata={
                "method": "serverless_openrouter",
                "provider": "openrouter",
                "model": self._model,
                "character": _DEFAULT_CHARACTER["name"],
            },
        )
