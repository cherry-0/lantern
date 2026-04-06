"""
Adapter for the deeptutor app.

Core pipeline: user question (+ optional document context) → RAG-based tutor response.

Primary strategy: use deeptutor's ChatOrchestrator directly — bypassing the
facade's session store — to drive a single "chat" capability turn.  LLM calls
go through deeptutor's own LLM service layer (litellm), routed to OpenRouter via
env-var injection before any deeptutor module is imported.

The async orchestrator is driven synchronously via _run_async(), which runs the
coroutine in a dedicated thread so it works even when Streamlit's event loop is
already running.

Fallback: direct OpenRouter chat call if the deeptutor package cannot be imported
or if the orchestrator raises an unrecoverable error.
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key

DEEPTUTOR_ROOT = TARGET_APPS_DIR / "deeptutor"

# System prompt matching deeptutor's tutoring persona
_TUTOR_SYSTEM = (
    "You are DeepTutor, an expert AI tutor specialising in academic and technical subjects. "
    "Your role is to help students understand difficult concepts clearly and thoroughly. "
    "When given a question or a passage of text, provide a structured, educational response "
    "that explains the core ideas, highlights key concepts, and offers relevant examples. "
    "Be precise, insightful, and pedagogically clear."
)


def _inject_deeptutor_env(api_key: str) -> None:
    """
    Set the env vars deeptutor's EnvStore and LLM service read at startup.
    deeptutor uses LLM_API_KEY / LLM_HOST (its own naming) in addition to the
    standard OPENAI_API_KEY / OPENAI_BASE_URL that litellm reads.
    """
    base_url = "https://openrouter.ai/api/v1"
    for var, val in [
        ("LLM_BINDING", "openai"),
        ("LLM_API_KEY", api_key),
        ("LLM_HOST", base_url),
        ("LLM_MODEL", OPENROUTER_DEFAULT_MODEL),
        ("OPENAI_API_KEY", api_key),
        ("OPENAI_BASE_URL", base_url),
    ]:
        os.environ.setdefault(var, val)


class DeepTutorAdapter(BaseAdapter):
    """
    Wraps the deeptutor RAG-based tutoring pipeline.

    For Verify: given a text item (question or document excerpt), produce a tutor
    response via deeptutor's ChatOrchestrator → ChatCapability → AgenticChatPipeline.
    The output is evaluated for privacy attribute inferability.
    """

    name = "deeptutor"
    supported_modalities = ["text"]

    def __init__(self):
        self._native_available: Optional[bool] = None
        self._native_error: str = ""

    # ── Native availability ───────────────────────────────────────────────────

    def _check_native(self) -> Tuple[bool, str]:
        if self._native_available is not None:
            return self._native_available, self._native_error

        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            _inject_deeptutor_env(api_key)

        deeptutor_path = str(DEEPTUTOR_ROOT)
        if deeptutor_path not in sys.path:
            sys.path.insert(0, deeptutor_path)

        try:
            from deeptutor.runtime.orchestrator import ChatOrchestrator  # noqa: F401
            from deeptutor.core.context import UnifiedContext  # noqa: F401
            from deeptutor.core.stream import StreamEventType  # noqa: F401
            self._native_available = True
            self._native_error = ""
        except Exception as e:
            self._native_available = False
            self._native_error = str(e)

        return self._native_available, self._native_error

    def check_availability(self) -> Tuple[bool, str]:
        native_ok, native_err = self._check_native()
        if native_ok:
            return True, "Native deeptutor ChatOrchestrator available; routing via OpenRouter."

        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return (
                True,
                f"Native deeptutor pipeline unavailable ({native_err}); using OpenRouter chat fallback.",
            )
        return (
            False,
            f"deeptutor native pipeline unavailable ({native_err}) and no valid OpenRouter API key.",
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"deeptutor only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        native_ok, _ = self._check_native()
        try:
            if native_ok:
                return self._run_native(input_item)
            else:
                return self._run_openrouter_fallback(input_item)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # ── Native path: ChatOrchestrator ─────────────────────────────────────────

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        """
        Drive a single deeptutor chat turn through ChatOrchestrator.

        ChatOrchestrator.handle(context) is an async generator of StreamEvent
        objects.  We collect all CONTENT events (the streaming text chunks) and
        join them into the final response.  Any other event types (STAGE_START,
        TOOL_CALL, DONE, …) are ignored for the output text but preserved in
        raw_output for debugging.
        """
        from deeptutor.runtime.orchestrator import ChatOrchestrator
        from deeptutor.core.context import UnifiedContext
        from deeptutor.core.stream import StreamEventType

        data = input_item.get("data", "")
        if not isinstance(data, str):
            data = str(data)
        text = data.strip()
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        context = UnifiedContext(
            session_id=str(uuid.uuid4()),
            user_message=text,
            active_capability="chat",
            language="en",
        )

        orchestrator = ChatOrchestrator()

        async def _collect():
            content_parts = []
            all_events = []
            async for event in orchestrator.handle(context):
                all_events.append(event.to_dict())
                if event.type == StreamEventType.CONTENT and event.content:
                    content_parts.append(event.content)
            return "".join(content_parts), all_events

        response_text, all_events = self._run_async(_collect())

        if not response_text:
            # No CONTENT events — fall back gracefully
            return self._run_openrouter_fallback(input_item)

        structured = {
            "student_input": text[:500],
            "tutor_response": response_text,
        }

        return AdapterResult(
            success=True,
            output_text=response_text,
            raw_output={"events": all_events},
            structured_output=structured,
            metadata={
                "method": "native_chat_orchestrator",
                "workflow": "rag_tutoring",
                "model": OPENROUTER_DEFAULT_MODEL,
            },
        )

    # ── OpenRouter fallback ───────────────────────────────────────────────────

    def _run_openrouter_fallback(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Direct OpenRouter chat call when deeptutor package cannot be imported."""
        data = input_item.get("data", "")
        if not isinstance(data, str):
            data = str(data)
        text = data.strip()
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        prompt = (
            f"{_TUTOR_SYSTEM}\n\n"
            f"A student has submitted the following question or document excerpt:\n\n"
            f"---\n{text}\n---\n\n"
            "Please provide a thorough tutoring response with:\n"
            "1. A concise summary of the core topic\n"
            "2. Key concepts explained clearly\n"
            "3. Relevant examples or analogies\n"
            "4. Any important caveats or follow-up points"
        )

        try:
            response = self._call_openrouter(prompt=prompt, max_tokens=1024)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        structured = {
            "student_input": text[:500],
            "tutor_response": response,
        }

        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"prompt": prompt, "response": response},
            structured_output=structured,
            metadata={"method": "openrouter_chat_fallback", "workflow": "rag_tutoring"},
        )
