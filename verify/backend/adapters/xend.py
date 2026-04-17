"""
Adapter for the xend app.

Core pipeline: original email/text content + style → revised email (subject + body).

USE_APP_SERVERS=true  : runs xend's LangChain chains inside the 'xend' conda env.
USE_APP_SERVERS=false : uses OpenRouter to rewrite text as an email drafting assistant would.
"""

import atexit
from pathlib import Path
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

XEND_BACKEND = TARGET_APPS_DIR / "xend" / "backend"

_ENV_SPEC = EnvSpec(
    name="xend",
    python="3.12",
    install_cmds=[
        ["pip", "install", "poetry"],
        ["poetry", "install", "--no-root"],
        ["pip", "install", "fastapi", "uvicorn", "pydantic", "requests"]
    ],
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
    env_spec = _ENV_SPEC

    def __init__(self):
        self._server_process = None
        self._server_port = None
        atexit.register(self._cleanup_server)

    def _cleanup_server(self):
        if self._server_process is not None:
            import sys
            print(f"[xend] Shutting down local API server (port {self._server_port})...", file=sys.stderr, flush=True)
            self._server_process.terminate()
            self._server_process.wait()
            self._server_process = None

    def _start_server(self):
        if self._server_process is not None and self._server_process.poll() is None:
            return

        import socket
        import subprocess
        import time
        import requests
        import sys

        s = socket.socket()
        s.bind(("", 0))
        self._server_port = s.getsockname()[1]
        s.close()

        conda = CondaRunner.find_conda()
        server_script = Path(__file__).parent.parent / "runners" / "xend_server.py"

        print(f"[xend] Starting local API server on port {self._server_port}...", file=sys.stderr, flush=True)
        self._server_process = subprocess.Popen(
            [conda, "run", "-n", _ENV_SPEC.name, "python", str(server_script), "--port", str(self._server_port)],
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )

        start_time = time.time()
        while time.time() - start_time < 30:
            try:
                resp = requests.get(f"http://127.0.0.1:{self._server_port}/health")
                if resp.status_code == 200:
                    print("[xend] Server is ready.", file=sys.stderr, flush=True)
                    return
            except Exception:
                time.sleep(1)
        
        raise RuntimeError("xend_server failed to start within 30 seconds.")

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
        """Run xend's LangChain chains inside the 'xend' conda env via HTTP server."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        text_content = input_item.get("text_content", "")
        if not text_content:
            return AdapterResult(success=False, error="No text content provided.")

        import requests
        import sys

        try:
            self._start_server()
        except Exception as e:
            return AdapterResult(success=False, error=f"Failed to start server: {e}")

        try:
            payload = {
                "text_content": text_content,
                "openrouter_api_key": get_openrouter_api_key() or "",
                "model": OPENROUTER_DEFAULT_MODEL,
            }
            print("[xend] Sending inference request to local server...", file=sys.stderr, flush=True)
            resp = requests.post(f"http://127.0.0.1:{self._server_port}/infer", json=payload, timeout=90)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"HTTP request to server failed: {e}")

        if not result.get("success"):
            return AdapterResult(success=False, error=result.get("error"))

        subject = result.get("subject", "")
        body = result.get("body", "")
        externalizations = result.get("externalizations", {})
        output_text = f"Subject: {subject}\n\nBody:\n{body}"
        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output=result,
            structured_output={"subject": subject, "body": body},
            externalizations=externalizations,
            metadata={"method": "native_langchain_server"},
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
