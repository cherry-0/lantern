"""
Google AI Edge Gallery runner — executed inside the 'google-ai-edge-gallery' conda env.
Requires: pip install transformers accelerate torch

Input JSON keys:
  text_content   str   (for text modality)
  image_base64   str   (for image modality; describes image via text prompt)
  modality       str   "text" | "image"
  model_id       str   HuggingFace model ID
  max_tokens     int

Output JSON:
  success    bool
  ai_response str
  model_id   str
  error      str | null
"""
import json
import os
import sys
import traceback

_CHAT_SYSTEM = (
    "You are a helpful AI assistant running locally on a mobile device via the Google AI Edge "
    "Gallery app. You have multimodal capabilities: you can process both text and images. "
    "Respond helpfully and concisely to the user's input. If an image is provided, analyze it "
    "and incorporate your visual understanding into the response."
)


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    modality: str = data.get("modality", "text")
    model_id: str = data.get("model_id", "Qwen/Qwen2.5-1.5B-Instruct")
    max_tokens: int = int(data.get("max_tokens", 512))
    image_desc: str = data.get("image_description", "")  # pre-described by adapter

    if modality == "image":
        user_message = (
            f"The user shared an image. Visual content: {image_desc}\n\n"
            "Please analyze this and provide a helpful response."
        ) if image_desc else "The user shared an image. Please describe what you see."
    else:
        user_message = data.get("text_content", "").strip()

    if not user_message:
        print(json.dumps({"success": False, "error": "Empty input."}))
        sys.exit(1)

    runners_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, runners_dir)
    from _runner_log import log_input
    content = image_desc if modality == "image" else user_message
    log_input("google-ai-edge", modality, content)

    import torch
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"[google-ai-edge] Loading model {model_id} (device={'cuda' if device >= 0 else 'cpu'}) ...", file=sys.stderr, flush=True)
    pipe = hf_pipeline(
        "text-generation",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device >= 0 else torch.float32,
        trust_remote_code=True,
    )
    print(f"[google-ai-edge] Model loaded.", file=sys.stderr, flush=True)

    messages = [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "user",   "content": user_message},
    ]

    print(f"[google-ai-edge] Running text-generation ...", file=sys.stderr, flush=True)
    outputs = pipe(
        messages,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
    )
    response = outputs[0]["generated_text"][-1]["content"]
    print(f"[google-ai-edge] Inference complete.", file=sys.stderr, flush=True)

    # --- Capture Externalizations as identified in analysis/google-ai-edge-gallery.md ---
    externalizations = {
        "UI": f"[WebView Log] JS Skill: Processing visual request. Result: {response[:100]}...",
        "INTENT": f"[Android Intent] Action: ACTION_SENDTO. Data: mailto:?body={response[:50]}",
        "ANALYTICS": f"[Firebase] Event: GALLERY_GENERATION_COMPLETE. Model: {model_id}"
    }

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
