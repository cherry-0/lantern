"""
Adapter for the xend app.

Core pipeline: original email/text content + style → revised email (subject + body).

Primary strategy: import xend's LangChain chains (requires Django + Poetry deps).
Fallback: use OpenRouter to rewrite the text as an email drafting assistant would.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key  # used in check_availability

XEND_BACKEND = TARGET_APPS_DIR / "xend" / "backend"


class XendAdapter(BaseAdapter):
    """
    Wraps the xend email revision pipeline.

    For Verify: given a text item (email content or scenario text), produce
    a revised email draft. The output is evaluated for privacy attribute inferability.
    """

    name = "xend"
    supported_modalities = ["text"]

    def __init__(self):
        self._native_available: Optional[bool] = None
        self._native_error: str = ""

    def _check_native(self) -> Tuple[bool, str]:
        if self._native_available is not None:
            return self._native_available, self._native_error

        xend_path = str(XEND_BACKEND)
        if xend_path not in sys.path:
            sys.path.insert(0, xend_path)

        try:
            import django
            import os
            import environ

            # Manually load xend's .env (config/utils.py uses match syntax, Python 3.10+ only)
            env_file = XEND_BACKEND / ".env"
            if not env_file.exists():
                env_example = XEND_BACKEND / ".env_example"
                if env_example.exists():
                    import shutil
                    shutil.copyfile(str(env_example), str(env_file))

            if env_file.exists():
                _env = environ.Env(DEBUG=(bool, True), overwrite=True)
                environ.Env.read_env(env_file=str(env_file))

            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.local")
            # Purge any cached 'config' package from a previous adapter to
            # avoid sys.modules collision (momentag and xend both use a
            # top-level package named 'config').
            for _mod in list(sys.modules):
                if _mod == "config" or _mod.startswith("config."):
                    del sys.modules[_mod]
            try:
                django.setup()
            except RuntimeError:
                pass

            # Try importing the chain — this will fail if Django isn't configured correctly
            from apps.ai.services.chains import body_chain, subject_chain  # noqa: F401

            self._native_available = True
            self._native_error = ""
        except Exception as e:
            self._native_available = False
            self._native_error = str(e)

        return self._native_available, self._native_error

    def check_availability(self) -> Tuple[bool, str]:
        native_ok, native_err = self._check_native()
        if native_ok:
            return True, "Native xend LangChain pipeline available."

        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return (
                True,
                f"Native xend pipeline unavailable ({native_err}); using OpenRouter fallback.",
            )

        return (
            False,
            f"xend native pipeline unavailable ({native_err}) and no valid OpenRouter API key.",
        )

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"xend only supports 'text' modality, got '{input_item.get('modality')}'.",
            )

        native_ok, _ = self._check_native()
        if native_ok:
            try:
                result = self._run_native(input_item)
                # If native produced empty output, fall through to OpenRouter
                if result.success and not (result.output_text or "").strip().replace("Subject: \n\nBody:", "").strip():
                    raise RuntimeError("Native chain returned empty output.")
                return result
            except Exception as native_err:
                # Native failed at inference time — fall back to OpenRouter
                fallback = self._run_openrouter_fallback(input_item)
                if fallback.success:
                    fallback.metadata = {**( fallback.metadata or {}), "native_error": str(native_err)}
                return fallback

        return self._run_openrouter_fallback(input_item)

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Attempt to use xend's actual chain (unlikely to work without full Django setup)."""
        from apps.ai.services.chains import body_chain, subject_chain

        text_content = input_item.get("text_content", "")
        if not text_content:
            return AdapterResult(success=False, error="No text content provided.")

        # Build minimal inputs (xend chains expect many context fields; use defaults)
        inputs = {
            "body": text_content,
            "subject": "",
            "language": "en",
            "recipients": "",
            "group_name": "",
            "group_description": "",
            "prompt_text": "",
            "sender_role": "",
            "recipient_role": "",
            "plan_text": "",
            "analysis": None,
            "fewshots": None,
            "profile": "",
            "attachments": [],
            "locked_subject": "",
        }

        subject = (subject_chain.invoke(inputs) or "").strip()
        inputs["locked_subject"] = subject
        body = (body_chain.invoke(inputs) or "").strip()

        output_text = f"Subject: {subject}\n\nBody:\n{body}"

        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"subject": subject, "body": body},
            structured_output={"subject": subject, "body": body},
            metadata={"method": "native_langchain"},
        )

    def _run_openrouter_fallback(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Use OpenRouter to rewrite text content as an email draft."""
        text_content = input_item.get("text_content", "")
        if not text_content:
            return AdapterResult(success=False, error="No text content provided.")

        prompt = (
            "You are an AI email drafting assistant (like the xend app). "
            "Given the following content/scenario, compose a professional email.\n\n"
            f"Content:\n{text_content}\n\n"
            "Respond in this exact format:\n"
            "Subject: <email subject>\n\n"
            "Body:\n<email body>"
        )

        try:
            response_text = self._call_openrouter(prompt, max_tokens=1024)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        # Parse subject and body
        subject, body = "", response_text
        lines = response_text.splitlines()
        for i, line in enumerate(lines):
            if line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
            elif line.lower().startswith("body:"):
                body = "\n".join(lines[i + 1 :]).strip()
                break

        return AdapterResult(
            success=True,
            output_text=response_text,
            raw_output={"raw_response": response_text},
            structured_output={"subject": subject, "body": body},
            metadata={"method": "openrouter_fallback"},
        )
