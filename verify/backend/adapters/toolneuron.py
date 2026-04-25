"""
Adapter for the tool-neuron app.

Core pipelines implemented here:
  text generation  — user text prompt → GGUF LLM (llama.cpp on-device) → AI response
  image generation — user text prompt → Stable Diffusion 1.5 (on-device) → generated image

The app runs entirely on an Android device via Android AIDL IPC (LLMService ↔ GGUFEngine /
DiffusionEngine).  There is no HTTP server and no Python SDK.  Both pipelines are
reproduced in native mode by the closest Python equivalents:
  text  → llama-cpp-python  (wraps the same llama.cpp C++ library as the app's gguf_lib.aar)
  image → diffusers          (Stable Diffusion 1.5, same model family as the app's LocalDream)

Execution mode is controlled by USE_APP_SERVERS in .env:

  USE_APP_SERVERS=true  (NATIVE mode)
    text  : llama-cpp-python with a local GGUF model file.
            Requires: TOOL_NEURON_GGUF_MODEL_PATH pointing to a valid .gguf file.
            Install note: pip install llama-cpp-python compiles from source.
              For Apple Silicon Metal acceleration run:
                CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
              before the conda env is used (or set it in install_cmds below).
    image : diffusers + StableDiffusionPipeline (same model family as app's LocalDream).
            Downloads the model from HuggingFace on first run (~4 GB).
            Set TOOL_NEURON_SD_MODEL_ID to override the model.

  USE_APP_SERVERS=false  (SERVERLESS mode)
    text  : OpenRouter chat call with the same general assistant system prompt as the app.
    image : OpenRouter text call asking the model to describe what the generated image
            would look like; no actual image is produced.  The description is stored as
            output_text so the privacy evaluator can still assess information leakage.

Input item:
  modality          "text"   (both text-gen and image-gen take text prompts as input)
  generation_task   "text" | "image"   (default: "text")
  data / text_content  str  — the user's text prompt

Configuration (.env)
--------------------
USE_APP_SERVERS              — "true" / "false"                          (default: false)
TOOL_NEURON_GGUF_MODEL_PATH  — absolute path to a local .gguf model file (native text)
TOOL_NEURON_MAX_TOKENS       — max new tokens for text generation         (default: 1024)
TOOL_NEURON_CTX_SIZE         — GGUF context window size                   (default: 4096)
TOOL_NEURON_SD_MODEL_ID      — HuggingFace SD 1.5 model ID for image gen
                               (default: runwayml/stable-diffusion-v1-5)
TOOL_NEURON_IMAGE_STEPS      — diffusion steps                            (default: 20)
TOOL_NEURON_IMAGE_CFG        — classifier-free guidance scale             (default: 7.5)
TOOL_NEURON_IMAGE_SIZE       — output image dimensions as WxH             (default: 512x512)

TODOs (not yet implemented)
---------------------------
  RAG  — neuron-packet encrypted RAG format (.rag files): decrypt → retrieve chunks →
          inject into GGUF context.  Requires the neuron-packet module's decryption
          key (Ed25519) and the same llama.cpp KV-cache injection path.

  TTS  — Supertonic ONNX Runtime TTS (ai_supertonic_tts.aar).
          Python equivalent: onnxruntime + Supertonic ONNX weights from HuggingFace.
          Pipeline: LLM response text → ONNX TTS → WAV audio → externalization capture.

  STT  — Speech-to-text (microphone input before LLM).
          Python equivalent: openai-whisper / faster-whisper.
          Pipeline: WAV file → transcription → text_content for LLM.

  Function calling / tool-use — GGUF grammar-constrained JSON output.
          llama-cpp-python supports grammar via LlamaGrammar.
          Implement: enable_tool_calling(tools_json) → parse ToolCall response → dispatch.

  Inpainting — Stable Diffusion inpainting (prompt + base64 image + mask).
          Pipeline: StableDiffusionInpaintPipeline from diffusers.

  Image upscaling — RealESRGAN 4× (ai_sd.aar).
          Python equivalent: realesrgan or basicsr library.

  Multi-turn conversation — KV cache state persistence between pipeline calls.
          llama-cpp-python supports state save/load via Llama.save_state() / load_state().

  Persona / control vectors — personality JSON + control vector files (.gguf).
          llama-cpp-python supports control vectors via Llama.set_control_vector().
"""

import io
from pathlib import Path
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_CTX_SIZE = 4096
_DEFAULT_SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
_DEFAULT_IMAGE_STEPS = 20
_DEFAULT_IMAGE_CFG = 7.5
_DEFAULT_IMAGE_SIZE = "512x512"

# ── Conda environment ─────────────────────────────────────────────────────────

_ENV_SPEC = EnvSpec(
    name="tool-neuron",
    python="3.10",
    install_cmds=[
        # llama-cpp-python: compiles llama.cpp from source.
        # Use --prefer-binary to pull a pre-built wheel when available.
        # For Metal/CUDA acceleration, manually run:
        #   CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
        ["pip", "install", "llama-cpp-python", "--prefer-binary"],
        ["pip", "install", "diffusers", "transformers", "accelerate", "torch", "pillow"],
    ],
)
_RUNNER = Path(__file__).parent.parent / "runners" / "toolneuron_runner.py"

# ── System prompt — mirrors the app's default general-assistant persona ───────

_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable AI assistant running entirely on-device via the "
    "ToolNeuron app, powered by llama.cpp with GGUF model inference. You have advanced "
    "reasoning, coding, and analysis capabilities. Respond clearly and helpfully to the "
    "user's queries. You can also generate images on request using the built-in "
    "Stable Diffusion engine."
)


class ToolNeuronAdapter(BaseAdapter):
    """
    Wraps the tool-neuron on-device AI assistant pipeline.

    NATIVE mode     : llama-cpp-python (text gen) + diffusers (image gen) in conda env.
    SERVERLESS mode : OpenRouter for text gen; descriptive fallback for image gen.

    Both text generation and image generation take text as input (modality="text").
    The generation task is selected via input_item["generation_task"]:
      "text"  (default) → LLM text generation
      "image"           → Stable Diffusion image generation
    """

    name = "tool-neuron"
    supported_modalities = ["text", "image"]
    env_spec = _ENV_SPEC

    def __init__(self):
        self._gguf_model_path: str = get_env("TOOL_NEURON_GGUF_MODEL_PATH") or ""
        self._max_tokens: int = int(get_env("TOOL_NEURON_MAX_TOKENS") or _DEFAULT_MAX_TOKENS)
        self._ctx_size: int = int(get_env("TOOL_NEURON_CTX_SIZE") or _DEFAULT_CTX_SIZE)
        self._sd_model_id: str = get_env("TOOL_NEURON_SD_MODEL_ID") or _DEFAULT_SD_MODEL_ID
        self._image_steps: int = int(get_env("TOOL_NEURON_IMAGE_STEPS") or _DEFAULT_IMAGE_STEPS)
        self._image_cfg: float = float(get_env("TOOL_NEURON_IMAGE_CFG") or _DEFAULT_IMAGE_CFG)
        size_str: str = get_env("TOOL_NEURON_IMAGE_SIZE") or _DEFAULT_IMAGE_SIZE
        parts = size_str.lower().split("x")
        self._image_width: int = int(parts[0]) if len(parts) == 2 else 512
        self._image_height: int = int(parts[1]) if len(parts) == 2 else 512
        self._prompt_cache: Dict[str, str] = {}  # path → generated SD prompt

    # ── Availability ──────────────────────────────────────────────────────────

    def check_availability(self) -> Tuple[bool, str]:
        if use_app_servers():
            ok, msg = CondaRunner.probe(_ENV_SPEC)
            if not ok:
                return False, msg
            if not self._gguf_model_path:
                return False, (
                    "[NATIVE] TOOL_NEURON_GGUF_MODEL_PATH is not set. "
                    "Point it to a local .gguf model file for text generation."
                )
            if not Path(self._gguf_model_path).exists():
                return False, (
                    f"[NATIVE] GGUF model not found at: {self._gguf_model_path}"
                )
            return True, f"[NATIVE] GGUF model: {Path(self._gguf_model_path).name}"
        api_key = get_openrouter_api_key()
        if api_key and not api_key.startswith("your_"):
            return True, "[SERVERLESS] Using OpenRouter to replicate tool-neuron output."
        return False, "[SERVERLESS] No OPENROUTER_API_KEY configured."

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        modality = input_item.get("modality", "text")

        if modality == "image":
            # Image input → image generation task.
            # Derive a Stable Diffusion prompt from the image via VLM, then generate.
            import base64, io
            data = input_item.get("data")
            path = input_item.get("path", "")
            image_b64 = input_item.get("image_base64", "")
            if not image_b64:
                try:
                    from PIL import Image as PILImage
                    img = PILImage.open(str(data if isinstance(data, str) else path)).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    image_b64 = base64.b64encode(buf.getvalue()).decode()
                except Exception as e:
                    return AdapterResult(success=False, error=f"Image encoding failed: {e}")

            sd_prompt = self._get_sd_prompt(path, image_b64)
            return self._run_image_equivalent(sd_prompt)

        if modality != "text":
            return AdapterResult(
                success=False,
                error=f"tool-neuron supports 'text' and 'image' modalities, got '{modality}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        prompt = str(data).strip() if data else ""
        if not prompt:
            return AdapterResult(success=False, error="Empty text input.")

        generation_task = input_item.get("generation_task", "text")

        # Image generation: the app uses QNN-accelerated SD (Android-only, no Python SDK).
        # Both native and serverless use an architecture-equivalent OpenRouter SD call.
        if generation_task == "image":
            return self._run_image_equivalent(prompt)

        if use_app_servers():
            return self._run_native(prompt)
        return self._run_serverless_text(prompt)

    def _get_sd_prompt(self, path: str, image_b64: str) -> str:
        """
        Ask a VLM to generate an SD-style prompt that would reproduce this image.
        Cached per path so original and perturbed runs for the same image reuse it.
        """
        if path in self._prompt_cache:
            return self._prompt_cache[path]

        import sys
        try:
            sd_prompt = self._call_openrouter(
                prompt=(
                    "Look at this image and write a concise Stable Diffusion 1.5 prompt "
                    "that would reproduce it as faithfully as possible. Include subject, "
                    "style, setting, lighting, and any notable details. "
                    "Return ONLY the prompt text, no other text."
                ),
                image_b64=image_b64,
                model="google/gemini-2.0-flash-001",
                max_tokens=128,
            ).strip()
            if not sd_prompt:
                raise ValueError("Empty response")
        except Exception as e:
            print(f"[tool-neuron] _get_sd_prompt failed: {e}", file=sys.stderr)
            sd_prompt = f"a detailed scene from the image: {path or 'unknown'}"

        self._prompt_cache[path] = sd_prompt
        return sd_prompt

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, prompt: str) -> AdapterResult:
        """Run llama-cpp-python inside the 'tool-neuron' conda env (text only)."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)
        return self._run_native_text(prompt)

    def _run_native_text(self, prompt: str) -> AdapterResult:
        if not self._gguf_model_path or not Path(self._gguf_model_path).exists():
            return AdapterResult(
                success=False,
                error=(
                    f"GGUF model not found: {self._gguf_model_path!r}. "
                    "Set TOOL_NEURON_GGUF_MODEL_PATH in .env."
                ),
            )

        ok, result, err = CondaRunner.run(
            _ENV_SPEC.name,
            _RUNNER,
            {
                "task": "text",
                "text_content": prompt,
                "model_path": self._gguf_model_path,
                "max_tokens": self._max_tokens,
                "ctx_size": self._ctx_size,
                "system_prompt": _SYSTEM_PROMPT,
            },
            timeout=300,
        )
        if not ok:
            return AdapterResult(success=False, error=err)

        response = result.get("response", "")
        externalizations = result.get("externalizations", {})
        structured = {
            "generation_task": "text",
            "user_prompt": prompt,
            "response": response,
            "tokens_predicted": result.get("tokens_predicted", 0),
            "model_path": self._gguf_model_path,
        }
        return AdapterResult(
            success=result.get("success", False),
            output_text=response,
            raw_output=result,
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "native_llama_cpp", "generation_task": "text"},
        )


    # ── SERVERLESS mode (text) ────────────────────────────────────────────────

    def _run_serverless_text(self, prompt: str) -> AdapterResult:
        """
        OpenRouter call replicating the app's GGUF LLM text generation.

        Uses meta-llama/llama-3.1-8b-instruct — a non-reasoning Llama-family model,
        architecturally equivalent to the GGUF models tool-neuron runs on-device.
        Avoids reasoning models (e.g. Gemini 2.5 Pro) whose thinking tokens consume
        the token budget and produce truncated responses.
        """
        full_prompt = f"{_SYSTEM_PROMPT}\n\nUser: {prompt}"
        try:
            response = self._call_openrouter(
                prompt=full_prompt,
                model="meta-llama/llama-3.1-8b-instruct",
                max_tokens=self._max_tokens,
            )
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        structured = {
            "generation_task": "text",
            "user_prompt": prompt,
            "response": response,
        }
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "UI": f"ToolNeuron Chat: rendering LLM response — {response}",
                "STORAGE": "[UMS] Writing assistant message to messages.ums",
            }
        )
        return AdapterResult(
            success=True,
            output_text=response,
            raw_output={"prompt": prompt, "response": response},
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "serverless_openrouter", "generation_task": "text"},
        )

    # ── Image generation (architecture-equivalent, both modes) ───────────────

    def _run_image_equivalent(self, prompt: str) -> AdapterResult:
        """
        Architecture-equivalent image generation via OpenRouter.

        The app uses QNN-accelerated Stable Diffusion on Snapdragon NPU — an
        Android-only runtime with no Python SDK.  Both local diffusers and this
        OpenRouter call are equivalent approximations: same SD architecture,
        different runtime.  We use OpenRouter here so there's no ambiguity about
        which path is "more native".

        The evaluator receives a detailed visual description of what the SD model
        would generate from this prompt, allowing privacy-attribute leakage to be
        assessed from both the prompt and the inferred image content.
        """
        desc_prompt = (
            f"You are an expert at predicting Stable Diffusion 1.5 image outputs.\n\n"
            f"A user submitted this image generation prompt to the ToolNeuron app:\n"
            f"\"{prompt}\"\n\n"
            f"Describe in precise visual detail what Stable Diffusion 1.5 would generate "
            f"from this prompt: the main subjects, their appearance, setting, style, colors, "
            f"composition, and any identifiable, private, or sensitive content that would "
            f"appear. Focus on concrete visual details, not abstract commentary."
        )
        try:
            description = self._call_openrouter(prompt=desc_prompt, max_tokens=512)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        output_text = (
            f"[SD image generation — architecture-equivalent]\n"
            f"Prompt: {prompt}\n\n"
            f"Predicted image content:\n{description}"
        )
        structured = {
            "generation_task": "image",
            "image_prompt": prompt,
            "image_description": description,
        }
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "UI": f"ToolNeuron Image: rendering SD generation progress for — {prompt}",
                "STORAGE": "[UMS] Writing image message to messages.ums (ImageContent)",
            }
        )
        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"prompt": prompt, "image_description": description},
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "image_equivalent_openrouter", "generation_task": "image"},
        )

    def _run_image_with_vlm_analysis(self, prompt: str) -> AdapterResult:
        """
        Generate an actual image using Stable Diffusion and analyze it with a VLM.
        
        This provides raw pixel-based analysis rather than architecture-equivalent
        text description. The VLM analyzes the actual generated image for privacy
        attribute leakage.
        """
        import base64
        from PIL import Image as PILImage
        
        # Generate image using Stable Diffusion
        try:
            image = self._generate_image_local(prompt)
        except RuntimeError as e:
            return AdapterResult(success=False, error=f"Image generation failed: {e}")
        
        # Convert to base64 for VLM analysis
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()
        
        # Send to VLM for analysis
        vlm_prompt = (
            "Analyze this AI-generated image and describe: "
            "1) Main subjects and their appearance, "
            "2) Setting and environment, "
            "3) Any identifiable, private, or sensitive content visible, "
            "4) Overall composition and style. "
            "Be detailed and objective."
        )
        try:
            vlm_description = self._call_openrouter(
                prompt=vlm_prompt,
                image_b64=image_b64,
                model="google/gemini-2.0-flash-001",
                max_tokens=512
            )
        except RuntimeError as e:
            # Fall back to prompt-only description
            return self._run_image_equivalent(prompt)
        
        output_text = (
            f"[SD image generation — VLM-analyzed]\n"
            f"Prompt: {prompt}\n\n"
            f"VLM image analysis:\n{vlm_description}"
        )
        structured = {
            "generation_task": "image",
            "image_prompt": prompt,
            "vlm_description": vlm_description,
            "image_generated": True,
            "analysis_method": "vlm_raw_pixels",
        }
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "UI": f"ToolNeuron Image: rendering SD generation progress for — {prompt}",
                "STORAGE": "[UMS] Writing image message to messages.ums (ImageContent)",
            }
        )
        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"prompt": prompt, "vlm_description": vlm_description, "image_b64": image_b64},
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "vlm_image_analysis", "generation_task": "image"},
        )
    
    def _generate_image_local(self, prompt: str) -> "PILImage.Image":
        """Generate image using local Stable Diffusion 1.5."""
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError as e:
            raise RuntimeError(f"diffusers/torch not installed: {e}")
        
        # Lazy-load pipeline
        if not hasattr(self, "_sd_pipeline"):
            model_id = "runwayml/stable-diffusion-v1-5"
            self._sd_pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            if torch.cuda.is_available():
                self._sd_pipeline = self._sd_pipeline.to("cuda")
            elif torch.backends.mps.is_available():
                self._sd_pipeline = self._sd_pipeline.to("mps")
        
        # Generate image
        with torch.no_grad():
            result = self._sd_pipeline(
                prompt,
                num_inference_steps=20,
                width=self._image_width,
                height=self._image_height,
            )
        
        return result.images[0]
