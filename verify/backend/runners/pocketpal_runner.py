"""
PocketPal AI runner — executed inside the 'pocketpal-ai' conda env.

Replicates PocketPal AI's LocalCompletionEngine (LlamaContext via llama.rn) in Python:
  llama-cpp-python wraps the same llama.cpp C++ library as llama.rn, so running the
  same GGUF model file here reproduces the on-device inference faithfully.

Post-inference: writes the chat session to a local JSON store, simulating
WatermelonDB's ChatMessage + ChatSession persistence — captured as STORAGE.

Input JSON keys:
  text_content   str   user message
  model_path     str   absolute path to .gguf model file
  system_prompt  str   (optional) system message
  max_tokens     int   (optional, default 512)
  ctx_size       int   (optional, default 4096)

Output JSON:
  success          bool
  user_message     str
  ai_response      str
  tokens_predicted int
  externalizations dict
  error            str | null
"""
import json
import sys
import traceback
from pathlib import Path

# ── sys.path setup ─────────────────────────────────────────────────────────────
_RUNNERS_DIR = Path(__file__).parent
_VERIFY_ROOT = _RUNNERS_DIR.parent.parent
sys.path.insert(0, str(_RUNNERS_DIR))   # _runtime_capture, _runner_log
sys.path.insert(0, str(_VERIFY_ROOT))   # verify.backend.utils.config

import _runtime_capture
_runtime_capture.install()

from _runner_log import log_input

# Chat store — simulates WatermelonDB's local SQLite persistence.
# pathlib.Path.write_text is patched by _runtime_capture → captured as STORAGE.
_CHAT_STORE = _RUNNERS_DIR / "pocketpal_chats.json"

_DEFAULT_SYSTEM = (
    "You are a helpful AI assistant running entirely on-device. "
    "Respond clearly and concisely to the user's questions."
)


def _record_text_externalizations(
    text: str,
    response: str,
    *,
    model_path: str,
    tokens_predicted: int,
    ctx_size: int,
    max_tokens: int,
) -> None:
    model_name = Path(model_path).name or model_path
    _runtime_capture.record_ui_event("STREAMING_TEXT", response)
    _runtime_capture.record_ui_event("DISPLAY_TEXT", response)
    _runtime_capture.record_ui_event(
        "MESSAGE_METRICS",
        f"model={model_name} prompt_chars={len(text)} tokens={tokens_predicted} ctx={ctx_size} max_tokens={max_tokens}",
    )
    _runtime_capture.record_storage_event(
        "WATERMELONDB_PUT",
        f"ChatMessage(role=user) model={model_name} content={text}",
    )
    _runtime_capture.record_storage_event(
        "WATERMELONDB_PUT",
        f"ChatMessage(role=assistant) model={model_name} content={response}",
    )
    _runtime_capture.record_storage_event(
        "WATERMELONDB_PUT",
        "ChatSession updated lastMessageAt + messageCount",
    )


def _save_chat(text: str, response: str) -> None:
    """
    Simulate WatermelonDB ChatSession + ChatMessage inserts.
    pathlib.Path.write_text is patched by _runtime_capture → STORAGE externalization.
    """
    sessions: list = []
    if _CHAT_STORE.exists():
        try:
            sessions = json.loads(_CHAT_STORE.read_text())
        except Exception:
            sessions = []
    sessions.append({
        "messages": [
            {"role": "user",      "content": text},
            {"role": "assistant", "content": response},
        ]
    })
    _CHAT_STORE.write_text(json.dumps(sessions, ensure_ascii=False))


def main(data: dict) -> dict:
    text: str        = data.get("text_content", "").strip()
    model_path: str  = data.get("model_path", "")
    system: str      = data.get("system_prompt", "") or _DEFAULT_SYSTEM
    max_tokens: int  = int(data.get("max_tokens", 512))
    ctx_size: int    = int(data.get("ctx_size", 4096))

    log_input("pocketpal-ai", "text", text)

    if not text:
        return {"success": False, "error": "Empty text_content.", "user_message": "", "ai_response": "", "tokens_predicted": 0, "externalizations": {}}
    if not model_path or not Path(model_path).exists():
        return {"success": False, "error": f"GGUF model not found: {model_path}", "user_message": text, "ai_response": "", "tokens_predicted": 0, "externalizations": {}}

    print(f"[pocketpal-ai] Loading GGUF model: {model_path}", file=sys.stderr, flush=True)

    from llama_cpp import Llama  # noqa: PLC0415
    llm = Llama(
        model_path=model_path,
        n_ctx=ctx_size,
        n_gpu_layers=-1,   # use Metal/CUDA if available (mirrors llama.rn behaviour)
        verbose=False,
    )
    print("[pocketpal-ai] Model loaded. Running inference...", file=sys.stderr, flush=True)

    # Inference — DURING phase; captured by _runtime_capture but filtered from externalizations.
    output = llm.create_chat_completion(
        messages=[
            {"role": "system",    "content": system},
            {"role": "user",      "content": text},
        ],
        max_tokens=max_tokens,
        stream=False,
    )

    response: str = output["choices"][0]["message"]["content"] or ""
    tokens_predicted: int = output.get("usage", {}).get("completion_tokens", 0)
    print(f"[pocketpal-ai] Inference complete ({tokens_predicted} tokens).", file=sys.stderr, flush=True)

    # Switch to POST phase — everything below is captured as post-inference externalizations.
    _runtime_capture.set_phase("POST")

    # Simulate WatermelonDB write (ChatSession + ChatMessage rows → local SQLite).
    _save_chat(text, response)
    _record_text_externalizations(
        text,
        response,
        model_path=model_path,
        tokens_predicted=tokens_predicted,
        ctx_size=ctx_size,
        max_tokens=max_tokens,
    )
    print("[pocketpal-ai] Chat session saved to local store.", file=sys.stderr, flush=True)

    externalizations = _runtime_capture.finalize()
    return {
        "success": True,
        "user_message": text,
        "ai_response": response,
        "tokens_predicted": tokens_predicted,
        "externalizations": externalizations,
        "error": None,
    }


if __name__ == "__main__":
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)
    try:
        result = main(data)
    except Exception:
        result = {
            "success": False,
            "error": traceback.format_exc(),
            "user_message": "",
            "ai_response": "",
            "tokens_predicted": 0,
            "externalizations": {},
        }
    print(json.dumps(result))
