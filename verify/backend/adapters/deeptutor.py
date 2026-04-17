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

import atexit
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult, OPENROUTER_DEFAULT_MODEL
from verify.backend.utils.config import TARGET_APPS_DIR, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

DEEPTUTOR_ROOT = TARGET_APPS_DIR / "deeptutor"

_ENV_SPEC = EnvSpec(
    name="deeptutor",
    python="3.10",
    install_cmds=[["pip", "install", "-e", str(DEEPTUTOR_ROOT), "fastapi", "uvicorn", "pydantic", "requests"]],
)
_RUNNER = Path(__file__).parent.parent / "runners" / "deeptutor_runner.py"

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
    env_spec = _ENV_SPEC

    def __init__(self):
        self._server_process = None
        self._server_port = None
        atexit.register(self._cleanup_server)

    def _cleanup_server(self):
        if self._server_process is not None:
            import sys
            print(f"[deeptutor] Shutting down local API server (port {self._server_port})...", file=sys.stderr, flush=True)
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
        server_script = Path(__file__).parent.parent / "runners" / "deeptutor_server.py"

        print(f"[deeptutor] Starting local API server on port {self._server_port}...", file=sys.stderr, flush=True)
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
                    print("[deeptutor] Server is ready.", file=sys.stderr, flush=True)
                    return
            except Exception:
                time.sleep(1)
        
        raise RuntimeError("deeptutor_server failed to start within 30 seconds.")

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            return CondaRunner.probe(_ENV_SPEC)
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter chat fallback for deeptutor."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        if input_item.get("modality") != "text":
            return AdapterResult(
                success=False,
                error=f"deeptutor only supports 'text' modality, got '{input_item.get('modality')}'.",
            )
        if use_app_servers():
            return self._run_native(input_item)
        return self._run_openrouter_fallback(input_item)

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, input_item: Dict[str, Any]) -> AdapterResult:
        """Run deeptutor's ChatOrchestrator inside the 'deeptutor' conda env via HTTP server."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        data = input_item.get("data", "")
        text = str(data).strip() if data else ""
        if not text:
            return AdapterResult(success=False, error="Empty text input.")

        import requests
        import sys

        try:
            self._start_server()
        except Exception as e:
            return AdapterResult(success=False, error=f"Failed to start server: {e}")

        try:
            payload = {
                "text_content": text,
                "openrouter_api_key": get_openrouter_api_key() or "",
                "model": OPENROUTER_DEFAULT_MODEL,
            }
            print("[deeptutor] Sending inference request to local server...", file=sys.stderr, flush=True)
            resp = requests.post(f"http://127.0.0.1:{self._server_port}/infer", json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            return AdapterResult(success=False, error=f"HTTP request to server failed: {e}")

        if not result.get("success"):
            return AdapterResult(success=False, error=result.get("error"))

        response = result.get("tutor_response", "")
        externalizations = result.get("externalizations", {})
        structured = {"student_input": text[:500], "tutor_response": response}
        return AdapterResult(
            success=True,
            output_text=response,
            raw_output=result,
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "native_chat_orchestrator_server", "workflow": "rag_tutoring"},
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

        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "NETWORK": "[OpenRouter Fallback] Sending query + persona prompt to API.",
                "UI": f"Tutor response displayed: {response[:100]}...",
            }
        )

        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"prompt": prompt, "response": response},
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "openrouter_chat_fallback", "workflow": "rag_tutoring"},
        )
