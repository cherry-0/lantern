"""
ToolNeuron runner — executed inside the 'tool-neuron' conda env.

Requires:
  pip install llama-cpp-python --prefer-binary
  pip install diffusers transformers accelerate torch pillow

For Apple Silicon Metal acceleration (faster text generation), install with:
  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

Input JSON keys
---------------
  task              str    "text" | "image"

  --- task = "text" ---
  text_content      str    user prompt
  model_path        str    absolute path to .gguf model file
  max_tokens        int    max new tokens              (default: 512)
  ctx_size          int    context window size         (default: 4096)
  system_prompt     str    system message              (default: generic assistant)

  --- task = "image" ---
  image_prompt      str    text prompt for image generation
  sd_model_id       str    HuggingFace SD model ID    (default: runwayml/stable-diffusion-v1-5)
  steps             int    diffusion steps             (default: 20)
  cfg_scale         float  CFG guidance scale          (default: 7.5)
  seed              int    RNG seed (-1 = random)      (default: -1)
  width             int    image width                 (default: 512)
  height            int    image height                (default: 512)

Output JSON
-----------
  success           bool
  error             str | null
  externalizations  dict

  --- task = "text" ---
  response          str    generated text
  tokens_predicted  int

  --- task = "image" ---
  image_base64      str    base64-encoded PNG
  image_prompt      str    prompt that was used
  width             int
  height            int
  seed              int    seed actually used (resolved from -1 if random)
"""

import base64
import io
import json
import os
import sys
import traceback

_DEFAULT_SYSTEM = (
    "You are a helpful, knowledgeable AI assistant running entirely on-device via the "
    "ToolNeuron app, powered by llama.cpp with GGUF model inference. Respond clearly "
    "and helpfully to the user's queries."
)


# ── Text generation (llama-cpp-python / llama.cpp) ────────────────────────────

def _run_text(data: dict) -> dict:
    model_path: str = data.get("model_path", "")
    text_content: str = data.get("text_content", "").strip()
    max_tokens: int = int(data.get("max_tokens", 512))
    ctx_size: int = int(data.get("ctx_size", 4096))
    system_prompt: str = data.get("system_prompt", _DEFAULT_SYSTEM)

    if not model_path or not os.path.exists(model_path):
        return {"success": False, "error": f"GGUF model not found: {model_path!r}"}
    if not text_content:
        return {"success": False, "error": "text_content is empty."}

    try:
        from llama_cpp import Llama
    except ImportError:
        return {
            "success": False,
            "error": (
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python --prefer-binary"
            ),
        }

    print(f"[tool-neuron] Loading GGUF model: {os.path.basename(model_path)} ...",
          file=sys.stderr, flush=True)

    llm = Llama(
        model_path=model_path,
        n_ctx=ctx_size,
        n_threads=0,       # 0 = auto-detect CPU cores (matches app default)
        n_gpu_layers=-1,   # offload all layers to GPU/Metal if available
        flash_attn=True,
        verbose=False,
    )

    print("[tool-neuron] Running text generation ...", file=sys.stderr, flush=True)

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_content},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        repeat_penalty=1.1,
    )

    response: str = output["choices"][0]["message"]["content"] or ""
    tokens_predicted: int = output.get("usage", {}).get("completion_tokens", 0)

    print("[tool-neuron] Text generation complete.", file=sys.stderr, flush=True)
    return {
        "success": True,
        "response": response,
        "tokens_predicted": tokens_predicted,
        "error": None,
    }


# ── Image generation (diffusers / Stable Diffusion 1.5) ──────────────────────

def _run_image(data: dict) -> dict:
    image_prompt: str = data.get("image_prompt", "").strip()
    sd_model_id: str = data.get("sd_model_id", "runwayml/stable-diffusion-v1-5")
    steps: int = int(data.get("steps", 20))
    cfg_scale: float = float(data.get("cfg_scale", 7.5))
    seed: int = int(data.get("seed", -1))
    width: int = int(data.get("width", 512))
    height: int = int(data.get("height", 512))

    if not image_prompt:
        return {"success": False, "error": "image_prompt is empty."}

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as e:
        return {
            "success": False,
            "error": f"Required package not installed: {e}. Run: pip install diffusers torch",
        }

    # Select device (CUDA > MPS/Metal > CPU)
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"[tool-neuron] Loading SD model {sd_model_id} on {device} ...",
          file=sys.stderr, flush=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=dtype,
        safety_checker=None,       # match app's configurable safetyMode=false default
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # Enable memory-efficient attention when available
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    print("[tool-neuron] Running image generation ...", file=sys.stderr, flush=True)

    generator = None
    if seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        import random
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=image_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        width=width,
        height=height,
        generator=generator,
    )
    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    print("[tool-neuron] Image generation complete.", file=sys.stderr, flush=True)
    return {
        "success": True,
        "image_base64": image_b64,
        "image_prompt": image_prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "error": None,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    task: str = data.get("task", "text")

    runners_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, runners_dir)
    import _runtime_capture
    _runtime_capture.install()
    from _runner_log import log_input

    if task == "image":
        image_prompt = data.get("image_prompt", "")
        log_input("tool-neuron", "image_generation", image_prompt)
        result = _run_image(data)
    else:
        text_content = data.get("text_content", "")
        log_input("tool-neuron", "text_generation", text_content)
        result = _run_text(data)

    externalizations = _runtime_capture.finalize()
    result["externalizations"] = externalizations
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({
            "success": False,
            "response": "",
            "image_base64": "",
            "error": traceback.format_exc(),
            "externalizations": {},
        }))
        sys.exit(1)
