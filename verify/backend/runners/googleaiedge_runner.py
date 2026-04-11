"""
Google AI Edge Gallery runner — executed inside the 'google-ai-edge-gallery' conda env.
Requires: pip install transformers accelerate torch pillow

Input JSON keys:
  modality       str          "text" | "image"
  text_content   str          user text (for text modality)
  image_base64   str          base64-encoded JPEG (for image modality; empty for text)
  model_id       str          HuggingFace model ID override (empty = auto-select)
  max_tokens     int

Model auto-selection (when model_id is empty)
---------------------------------------------
  image_base64 non-empty → DEFAULT_VLM_MODEL  (vision-language model)
  image_base64 empty     → DEFAULT_TEXT_MODEL (text-only model)

Set GOOGLE_AI_EDGE_MODEL_ID in .env to override for both cases.

Output JSON:
  success        bool
  ai_response    str
  model_id       str
  error          str | null
"""
import base64
import io
import json
import os
import sys
import traceback

DEFAULT_VLM_MODEL  = "Qwen/Qwen2-VL-2B-Instruct"     # vision-language model (smallest available VLM)
DEFAULT_TEXT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"    # text-only model

_CHAT_SYSTEM = (
    "You are a helpful AI assistant running locally on a mobile device via the Google AI Edge "
    "Gallery app. You have multimodal capabilities: you can process both text and images. "
    "Respond helpfully and concisely to the user's input."
)


def _decode_image(image_b64: str):
    """Decode a base64 JPEG string to a PIL Image."""
    from PIL import Image as PILImage
    return PILImage.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _run_image(model_id: str, image_b64: str, text_content: str, max_tokens: int) -> str:
    """
    Run a vision-language model on an image (+ optional text prompt).
    Uses the "image-text-to-text" pipeline task.
    """
    import torch
    from transformers import pipeline as hf_pipeline

    img = _decode_image(image_b64)
    user_text = text_content.strip() if text_content.strip() else "Please analyze this image."

    device = 0 if torch.cuda.is_available() else -1
    print(f"[google-ai-edge] Loading VLM {model_id} (device={'cuda' if device >= 0 else 'cpu'}) ...",
          file=sys.stderr, flush=True)

    pipe = hf_pipeline(
        "image-text-to-text",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device >= 0 else torch.float32,
        trust_remote_code=True,
    )
    print("[google-ai-edge] VLM loaded.", file=sys.stderr, flush=True)

    # Override generation_config to avoid deprecation warnings from mixing
    # a pre-set generation_config with explicit generation kwargs.
    # Setting max_length=None removes the model's default max_length=20 which
    # conflicts with max_new_tokens.
    pipe.model.generation_config.max_new_tokens = max_tokens
    pipe.model.generation_config.max_length = None

    # The image-text-to-text pipeline iterates over message["content"] for every
    # message to extract visuals, so all content fields must be lists of dicts —
    # a plain string content causes "string indices must be integers".
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _CHAT_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": user_text},
            ],
        },
    ]

    print("[google-ai-edge] Running image-text-to-text inference ...", file=sys.stderr, flush=True)
    outputs = pipe(messages)
    # Pipeline returns list of dicts; last generated turn is the assistant reply
    generated = outputs[0]["generated_text"]
    if isinstance(generated, list):
        response = generated[-1].get("content", "")
    else:
        response = str(generated)

    print("[google-ai-edge] Inference complete.", file=sys.stderr, flush=True)
    return response


def _run_text(model_id: str, user_message: str, max_tokens: int) -> str:
    """Run a text-generation model on a plain text prompt."""
    import torch
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"[google-ai-edge] Loading model {model_id} (device={'cuda' if device >= 0 else 'cpu'}) ...",
          file=sys.stderr, flush=True)

    pipe = hf_pipeline(
        "text-generation",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device >= 0 else torch.float32,
        trust_remote_code=True,
    )
    print("[google-ai-edge] Model loaded.", file=sys.stderr, flush=True)

    # Override generation_config to avoid deprecation warnings from mixing
    # a pre-set generation_config with explicit generation kwargs.
    pipe.model.generation_config.max_new_tokens = max_tokens
    pipe.model.generation_config.max_length = None
    pipe.model.generation_config.do_sample = True
    pipe.model.generation_config.temperature = 1.0
    pipe.model.generation_config.top_k = 64
    pipe.model.generation_config.top_p = 0.95

    messages = [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "user",   "content": user_message},
    ]

    print("[google-ai-edge] Running text-generation inference ...", file=sys.stderr, flush=True)
    outputs = pipe(messages)
    response = outputs[0]["generated_text"][-1]["content"]
    print("[google-ai-edge] Inference complete.", file=sys.stderr, flush=True)
    return response


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    modality:     str = data.get("modality", "text")
    max_tokens:   int = int(data.get("max_tokens", 512))
    image_b64:    str = data.get("image_base64", "")
    text_content: str = data.get("text_content", "").strip()

    # Auto-select model based on whether image input is present.
    # An explicit override from GOOGLE_AI_EDGE_MODEL_ID takes priority.
    model_id: str = data.get("model_id", "") or (
        DEFAULT_VLM_MODEL if image_b64 else DEFAULT_TEXT_MODEL
    )

    runners_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, runners_dir)
    import _runtime_capture
    _runtime_capture.install()
    from _runner_log import log_input

    if modality == "image":
        log_input("google-ai-edge", modality, f"<image_base64 len={len(image_b64)}>")
        if not image_b64:
            print(json.dumps({"success": False, "error": "image_base64 is empty."}))
            sys.exit(1)
        response = _run_image(model_id, image_b64, text_content, max_tokens)
    else:
        if not text_content:
            print(json.dumps({"success": False, "error": "text_content is empty."}))
            sys.exit(1)
        log_input("google-ai-edge", modality, text_content)
        response = _run_text(model_id, text_content, max_tokens)

    externalizations = _runtime_capture.finalize()
    print(json.dumps({
        "success": True,
        "ai_response": response,
        "model_id": model_id,
        "externalizations": externalizations,
        "error": None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"success": False, "ai_response": "", "error": traceback.format_exc()}))
        sys.exit(1)
