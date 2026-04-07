"""
Adapter for the xend app.

Core pipeline: original email/text content + style → revised email (subject + body).

USE_APP_SERVERS=true  : runs xend's LangChain chains inside the 'xend' conda env.
USE_APP_SERVERS=false : uses OpenRouter to rewrite text as an email drafting assistant would.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

XEND_BACKEND = TARGET_APPS_DIR / "xend" / "backend"

_ENV_SPEC = EnvSpec(
    name="xend",
    python="3.12",
    install_cmds=[["pip", "install", "poetry"], ["poetry", "install", "--no-root"]],
    cwd=XEND_BACKEND,
)
_RUNNER = Path(__file__).parent.parent / "runners" / "xend_runner.py"


class XendAdapter(BaseAdapter):
    """
    Wraps the xend email revision pipeline.

    NATIVE mode     : xend's LangChain chains (subject_chain + body_chain) in conda env.
    SERVERLESS mode : OpenRouter replicating the same subject/body output structure.
    """

    name = "xend"
    supported_modalities = ["text"]

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter fallback for xend."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"xend only supports 'text' modality, got '{input_item.get('modality')}'.",
            )
        if use_app_servers():
            return self._run_native(input_item)
        return self._run_openrouter_fallback(input_item)

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Run xend's LangChain chains inside the 'xend' conda env via subprocess."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        text_content = input_item.get("text_content", "")
        if not text_content:
            return AdapterResult(success=False, error="No text content provided.")

        ok, result, err = CondaRunner.run(
            _ENV_SPEC.name,
            _RUNNER,
            {
                "text_content": text_content,
                "openrouter_api_key": get_openrouter_api_key() or "",
                "model": OPENROUTER_DEFAULT_MODEL,
            },
            timeout=90,
        )
        if not ok:
            return AdapterResult(success=False, error=err)

        subject = result.get("subject", "")
        body = result.get("body", "")
        externalizations = result.get("externalizations", {})
        output_text = f"Subject: {subject}\n\nBody:\n{body}"
        return AdapterResult(
            success=result.get("success", False),
            output_text=output_text,
            raw_output=result,
            structured_output={"subject": subject, "body": body},
            externalizations=externalizations,
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

        subject, body = "", response_text
        lines = response_text.splitlines()
        for i, line in enumerate(lines):
            if line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
            elif line.lower().startswith("body:"):
                body = "\n".join(lines[i + 1:]).strip()
                break

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": "[OpenRouter Fallback] Direct email generation request.",
                "UI": f"Rendering email draft with subject: {subject}",
            }
        )

        return AdapterResult(
            success=True,
            output_text=response_text,
            raw_output={"raw_response": response_text},
            structured_output={"subject": subject, "body": body},
            externalizations=externalizations,
            metadata={"method": "serverless"},
        )
