"""
DeepTutor runner — executed inside the 'deeptutor' conda env.

Input JSON keys:
  text_content       str   student question or document excerpt
  openrouter_api_key str
  model              str   (optional)

Output JSON:
  success        bool
  tutor_response str
  student_input  str
  error          str | null
"""
import asyncio
import json
import os
import sys
import traceback
import uuid


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    text: str = data.get("text_content", "").strip()
    api_key: str = data.get("openrouter_api_key", "")
    model: str = data.get("model", "google/gemini-2.5-pro")

    # ── Inject env vars before deeptutor imports ───────────────────────────────
    base_url = "https://openrouter.ai/api/v1"
    for var, val in [
        ("LLM_BINDING", "openai"),
        ("LLM_API_KEY", api_key),
        ("LLM_HOST", base_url),
        ("LLM_MODEL", model),
        ("OPENAI_API_KEY", api_key),
        ("OPENAI_BASE_URL", base_url),
    ]:
        os.environ.setdefault(var, val)

    # ── Bootstrap deeptutor ────────────────────────────────────────────────────
    runners_dir = os.path.dirname(os.path.abspath(__file__))
    deeptutor_root = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps", "deeptutor"
    ))
    sys.path.insert(0, deeptutor_root)
    sys.path.insert(0, runners_dir)
    import _runtime_capture
    _runtime_capture.install()

    from _runner_log import log_input
    log_input("deeptutor", "text", text)
    print("[deeptutor] Loading ChatOrchestrator ...", file=sys.stderr, flush=True)
    from deeptutor.runtime.orchestrator import ChatOrchestrator
    from deeptutor.core.context import UnifiedContext
    from deeptutor.core.stream import StreamEventType

    context = UnifiedContext(
        session_id=str(uuid.uuid4()),
        user_message=text,
        active_capability="chat",
        language="en",
    )

    orchestrator = ChatOrchestrator()

    async def _collect():
        parts = []
        async for event in orchestrator.handle(context):
            if event.type == StreamEventType.CONTENT and event.content:
                parts.append(event.content)
        return "".join(parts)

    print("[deeptutor] Running ChatOrchestrator ...", file=sys.stderr, flush=True)
    response = asyncio.run(_collect())
    print("[deeptutor] Inference complete.", file=sys.stderr, flush=True)

    externalizations = _runtime_capture.finalize()

    print(json.dumps({
        "success": True,
        "tutor_response": response,
        "student_input": text[:500],
        "externalizations": externalizations,
        "error": None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"success": False, "tutor_response": "", "error": traceback.format_exc()}))
        sys.exit(1)
