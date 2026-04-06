"""
Base adapter interface for Verify.

All target-app adapters must subclass BaseAdapter and implement the three required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Default model used for all OpenRouter fallback calls.
# Change this one constant to switch every adapter's fallback model at once.
OPENROUTER_DEFAULT_MODEL = "google/gemini-2.5-pro"


@dataclass
class AdapterResult:
    """Normalized result from a target-app pipeline run."""

    success: bool
    # Human-readable output (text summary of what the app produced)
    output_text: str = ""
    # Raw output dict as returned by the app's pipeline
    raw_output: Dict[str, Any] = field(default_factory=dict)
    # Optional structured output fields (tags, verdict, subject/body, etc.)
    structured_output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_text": self.output_text,
            "raw_output": self.raw_output,
            "structured_output": self.structured_output,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseAdapter(ABC):
    """
    Abstract base class for all target-app adapters.

    Subclasses wrap the specific AI pipeline of each app and expose
    a uniform interface to the Verify orchestrator.
    """

    # Human-readable app name
    name: str = "base"

    # Modalities this adapter can process: "image", "text", "video"
    supported_modalities: List[str] = []

    @abstractmethod
    def check_availability(self) -> Tuple[bool, str]:
        """
        Check whether the adapter's required dependencies are available.

        Returns:
            (available: bool, reason: str)
            If available is False, reason explains what is missing.
        """

    @abstractmethod
    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        """
        Run the target app's main AI pipeline on the given input item.

        Args:
            input_item: dict with at least:
                - "modality": "image" | "text" | "video"
                - "path": str — absolute path to the source file
                - "filename": str — display name
                - "data": loaded data (PIL.Image, str, list[PIL.Image] for video frames)
                - optional app-specific extra keys

        Returns:
            AdapterResult with success flag and output.
        """

    def get_display_label(self) -> str:
        """Return a short display label (used in UI dropdowns)."""
        return self.name

    def supports_modality(self, modality: str) -> bool:
        return modality in self.supported_modalities

    # ── Shared env-injection helpers ─────────────────────────────────────────

    @staticmethod
    def _inject_openrouter_env() -> None:
        """
        Inject OpenRouter credentials into the process environment so that
        target-app code that reads standard OpenAI-style env vars at import
        time will route to OpenRouter.

        Sets (only if not already set):
          OPENAI_API_KEY      → OPENROUTER_API_KEY
          OPENAI_BASE_URL     → https://openrouter.ai/api/v1
          LLM_API_KEY         → OPENROUTER_API_KEY   (deeptutor EnvStore)
          LLM_HOST            → https://openrouter.ai/api/v1   (deeptutor)
          VLM_API_KEY         → OPENROUTER_API_KEY   (snapdo Django settings)
          VLM_API_URL         → https://openrouter.ai/api/v1   (snapdo)
        """
        import os
        from verify.backend.utils.config import get_openrouter_api_key

        api_key = get_openrouter_api_key()
        if not api_key or api_key.startswith("your_"):
            return

        base_url = "https://openrouter.ai/api/v1"

        for var, val in [
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_BASE_URL", base_url),
            ("LLM_API_KEY", api_key),
            ("LLM_HOST", base_url),
            ("VLM_API_KEY", api_key),
            ("VLM_API_URL", base_url),
        ]:
            os.environ.setdefault(var, val)

    @staticmethod
    def _run_async(coro) -> Any:
        """
        Run an async coroutine synchronously, safe even when an event loop is
        already running (e.g. inside Streamlit or Jupyter).

        Uses a dedicated thread with its own event loop so the caller never
        needs to worry about 'cannot run nested event loop' errors.
        """
        import asyncio
        import concurrent.futures

        def _target():
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_target).result()

    # ── Shared OpenRouter helper ──────────────────────────────────────────────

    def _call_openrouter(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> str:
        """
        Send a prompt to OpenRouter and return the response text.

        Args:
            prompt:     User message text.
            image_b64:  Optional base64-encoded JPEG image. If provided, the
                        message becomes a vision request (text + image_url).
            model:      Model override. Defaults to OPENROUTER_DEFAULT_MODEL.
            max_tokens: Maximum tokens in the response.
            timeout:    Request timeout in seconds.

        Returns:
            The model's response string.

        Raises:
            RuntimeError: if no API key is configured or the request fails.
        """
        import requests
        from verify.backend.utils.config import get_openrouter_api_key

        api_key = get_openrouter_api_key()
        if not api_key or api_key.startswith("your_"):
            raise RuntimeError(
                "No valid OPENROUTER_API_KEY found. "
                "Set it in the .env file at the Lantern root."
            )

        content: Any
        if image_b64:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        else:
            content = prompt

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Verify",
                "X-Title": "Verify",
            },
            json={
                "model": model or OPENROUTER_DEFAULT_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
