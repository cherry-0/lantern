"""
LLM-VTuber runner — executed inside the 'llm-vtuber' conda env.

Input JSON keys:
  text_content       str
  openrouter_api_key str
  model              str   (optional)

Output JSON:
  success            bool
  character_response str
  user_message       str
  error              str | null
"""
import asyncio
import json
import os
import sys
import traceback

_VTUBER_SYSTEM = (
    "You are Shizuku, a cheerful and curious AI VTuber. You speak in a warm, "
    "expressive style and are genuinely interested in what the user shares with you. "
    "You respond naturally to conversation as a VTuber streamer would — engaging, "
    "friendly, and occasionally playful. Keep responses concise and conversational, "
    "as they will be spoken aloud via text-to-speech."
)


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    user_message: str = data.get("text_content", "").strip()
    api_key: str = data.get("openrouter_api_key", "")
    model: str = data.get("model", "google/gemini-2.5-pro")

    # ── Bootstrap llm-vtuber ───────────────────────────────────────────────────
    runners_dir = os.path.dirname(os.path.abspath(__file__))
    llmvtuber_src = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps", "llm-vtuber", "src"
    ))
    sys.path.insert(0, llmvtuber_src)
    sys.path.insert(0, runners_dir)
    import _runtime_capture
    _runtime_capture.install()

    from _runner_log import log_input
    log_input("llm-vtuber", "text", user_message)
    print("[llm-vtuber] Loading AsyncLLM ...", file=sys.stderr, flush=True)
    from open_llm_vtuber.agent.stateless_llm.openai_compatible_llm import AsyncLLM

    llm = AsyncLLM(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        llm_api_key=api_key,
        temperature=1.0,
    )

    messages = [{"role": "user", "content": user_message}]

    async def _collect() -> str:
        parts = []
        async for chunk in llm.chat_completion(messages, system=_VTUBER_SYSTEM):
            if isinstance(chunk, str):
                parts.append(chunk)
        return "".join(parts)

    print("[llm-vtuber] Running chat_completion ...", file=sys.stderr, flush=True)
    response = asyncio.run(_collect())
    print("[llm-vtuber] Inference complete.", file=sys.stderr, flush=True)

    externalizations = _runtime_capture.finalize()

    print(json.dumps({
        "success": True,
        "character_response": response,
        "user_message": user_message[:500],
        "externalizations": externalizations,
        "error": None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"success": False, "character_response": "", "error": traceback.format_exc()}))
        sys.exit(1)
