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
TOOL_NEURON_MAX_TOKENS       — max new tokens for text generation         (default: 512)
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

from pathlib import Path
from typing import Any, Dict, Tuple

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.utils.config import get_env, get_openrouter_api_key, use_app_servers
from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_MAX_TOKENS = 512
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
    supported_modalities = ["text"]
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
        if modality != "text":
            return AdapterResult(
                success=False,
                error=f"tool-neuron only supports 'text' modality, got '{modality}'.",
            )

        data = input_item.get("data", "") or input_item.get("text_content", "")
        prompt = str(data).strip() if data else ""
        if not prompt:
            return AdapterResult(success=False, error="Empty text input.")

        generation_task = input_item.get("generation_task", "text")

        if use_app_servers():
            return self._run_native(prompt, generation_task)
        return self._run_serverless(prompt, generation_task)

    # ── NATIVE mode ───────────────────────────────────────────────────────────

    def _run_native(self, prompt: str, generation_task: str) -> AdapterResult:
        """Run llama-cpp-python or diffusers inside the 'tool-neuron' conda env."""
        ok, msg = CondaRunner.ensure(_ENV_SPEC)
        if not ok:
            return AdapterResult(success=False, error=msg)

        if generation_task == "image":
            return self._run_native_image(prompt)
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
            "user_prompt": prompt[:500],
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

    def _run_native_image(self, prompt: str) -> AdapterResult:
        ok, result, err = CondaRunner.run(
            _ENV_SPEC.name,
            _RUNNER,
            {
                "task": "image",
                "image_prompt": prompt,
                "sd_model_id": self._sd_model_id,
                "steps": self._image_steps,
                "cfg_scale": self._image_cfg,
                "width": self._image_width,
                "height": self._image_height,
                "seed": -1,
            },
            timeout=600,  # first run downloads ~4 GB model
        )
        if not ok:
            return AdapterResult(success=False, error=err)

        image_b64 = result.get("image_base64", "")
        used_seed = result.get("seed", -1)
        output_text = (
            f"[Image generated] Prompt: {prompt}\n"
            f"Model: {self._sd_model_id} | "
            f"Steps: {self._image_steps} | CFG: {self._image_cfg} | Seed: {used_seed} | "
            f"Size: {self._image_width}x{self._image_height}"
        )
        structured = {
            "generation_task": "image",
            "image_prompt": prompt,
            "image_base64": image_b64,
            "sd_model_id": self._sd_model_id,
            "steps": self._image_steps,
            "cfg_scale": self._image_cfg,
            "width": self._image_width,
            "height": self._image_height,
            "seed": used_seed,
        }
        externalizations = result.get("externalizations", {})
        return AdapterResult(
            success=result.get("success", False),
            output_text=output_text,
            raw_output=result,
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "native_stable_diffusion", "generation_task": "image"},
        )

    # ── SERVERLESS mode ───────────────────────────────────────────────────────

    def _run_serverless(self, prompt: str, generation_task: str) -> AdapterResult:
        if generation_task == "image":
            return self._run_serverless_image(prompt)
        return self._run_serverless_text(prompt)

    def _run_serverless_text(self, prompt: str) -> AdapterResult:
        """OpenRouter chat call replicating the app's GGUF LLM text generation."""
        full_prompt = f"{_SYSTEM_PROMPT}\n\nUser: {prompt}"
        try:
            response = self._call_openrouter(prompt=full_prompt, max_tokens=self._max_tokens)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        structured = {
            "generation_task": "text",
            "user_prompt": prompt[:500],
            "response": response,
        }
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "UI": f"ToolNeuron Chat: rendering LLM response — {response[:100]}...",
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

    def _run_serverless_image(self, prompt: str) -> AdapterResult:
        """
        OpenRouter call describing what the Stable Diffusion image would contain.
        Full image generation is not available in serverless mode — the description
        is returned as output_text so the privacy evaluator can still assess
        information leakage from the prompt and generated content.
        """
        desc_prompt = (
            f"You are simulating a Stable Diffusion 1.5 image generation system.\n\n"
            f"A user has submitted this image generation prompt:\n"
            f"\"{prompt}\"\n\n"
            f"Describe in detail what the generated image would visually contain: "
            f"the subjects, setting, style, colors, and any identifiable or private "
            f"information that might be depicted. Be specific and thorough."
        )
        try:
            description = self._call_openrouter(prompt=desc_prompt, max_tokens=512)
        except RuntimeError as e:
            return AdapterResult(success=False, error=str(e))

        output_text = (
            f"[Image generation — serverless description]\n"
            f"Prompt: {prompt}\n\n"
            f"Generated image would contain:\n{description}"
        )
        structured = {
            "generation_task": "image",
            "image_prompt": prompt,
            "image_description": description,
            "note": "Serverless mode: description only, no actual image generated.",
        }
        externalizations = self._build_serverless_externalizations(
            realistic_fallback={
                "UI": f"ToolNeuron Image: rendering SD generation progress for — {prompt[:80]}...",
                "STORAGE": "[UMS] Writing image message to messages.ums (ImageContent)",
            }
        )
        return AdapterResult(
            success=True,
            output_text=output_text,
            raw_output={"prompt": prompt, "description": description},
            structured_output=structured,
            externalizations=externalizations,
            metadata={"method": "serverless_image_description", "generation_task": "image"},
        )
