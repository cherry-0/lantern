# AI Inference Privacy Audit: tool-neuron

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | UI Rendering (Text) | `toolneuron_runner.py` | 67-72 | `_record_text_externalizations` | Generated LLM response text streamed and displayed to user. | `_runtime_capture.record_ui_event("STREAMING_TEXT", response)`, `_runtime_capture.record_ui_event("DISPLAY_TEXT", response)` | High |
| 2 | UI Rendering (Image) | `toolneuron_runner.py` | 95-106 | `_record_image_externalizations` | Image generation progress, final image display, metrics (prompt, steps, seed, model). | `_runtime_capture.record_ui_event("IMAGE_GENERATION_PROGRESS", ...)`, `_runtime_capture.record_ui_event("DISPLAY_IMAGE", ...)` | High |
| 3 | Storage (Text) | `toolneuron_runner.py` | 73-80 | `_record_text_externalizations` | Text messages written to UMS (Unified Message Storage), chat metadata updated. | `_runtime_capture.record_storage_event("UMS_PUT", "messages contentType=Text...")`, `_runtime_capture.record_storage_event("UMS_PUT", "chats update lastMessageAt...")` | High |
| 4 | Storage (Image) | `toolneuron_runner.py` | 107-114 | `_record_image_externalizations` | Image messages written to UMS with ImageContent type, chat metadata updated. | `_runtime_capture.record_storage_event("UMS_PUT", "messages contentType=Image...")`, `_runtime_capture.record_storage_event("UMS_PUT", "chats update lastMessageAt...")` | High |
| 5 | Network (Image Description) | `toolneuron.py` | 233-242, 374 | `_get_sd_prompt`, `_run_image_equivalent` | Original image sent to OpenRouter VLM (Gemini) to generate SD reproduction prompt. | `self._call_openrouter(prompt=..., image_b64=..., model="google/gemini-2.0-flash-001")` | High |
| 6 | Network (Architecture-Equivalent) | `toolneuron.py` | 350-401 | `_run_image_equivalent` | SD prompt sent to OpenRouter to get visual description of generated image content. | `self._call_openrouter(prompt=desc_prompt, max_tokens=512)` | High |

## B. Main AI Inference Workflows

### Workflow 1: On-Device LLM Text Generation
- **Purpose**: Generate text responses using on-device GGUF model via llama.cpp.
- **Input**: User text prompt.
- **Processing**: Text tokenized, fed to Llama.cpp GGUF inference engine.
- **Inference**: Local inference on CPU/GPU (no network call in NATIVE mode).
- **Externalization**:
    - UI: Streaming text displayed to user (`toolneuron_runner.py:67-68`).
    - Storage: Message persisted to UMS database (`toolneuron_runner.py:73-80`).
- **Episode path**: User Input → Llama.cpp Tokenization → GGUF Inference → Streaming Output → UI Display → UMS Storage
- **Key files**: `toolneuron_runner.py`, `toolneuron.py`
- **Confidence**: High

### Workflow 2: Image-to-Image Generation (Perturbed Privacy Evaluation)
- **Purpose**: Generate privacy-preserving variants of input images using Stable Diffusion.
- **Input**: Source image (as base64).
- **Processing**:
    1. VLM analyzes image to generate SD-style reproduction prompt (`toolneuron.py:233-242`).
    2. SD pipeline generates new image from prompt.
- **Inference**: 
    - Step 1: Remote VLM call to Gemini via OpenRouter.
    - Step 2: Local Stable Diffusion inference (diffusers library).
- **Externalization**:
    - Network: Original image sent to OpenRouter for prompt generation (`toolneuron.py:240`).
    - UI: Generation progress displayed (`toolneuron_runner.py:95-98`).
    - UI: Final image displayed (`toolneuron_runner.py:99-102`).
    - Storage: Image message persisted to UMS (`toolneuron_runner.py:107-110`).
- **Episode path**: Input Image → VLM Prompt Generation (Network) → SD Inference → UI Progress → UI Display → UMS Storage
- **Key files**: `toolneuron_runner.py`, `toolneuron.py`
- **Confidence**: High

### Workflow 3: Text-to-Image Generation
- **Purpose**: Generate images from text prompts using Stable Diffusion.
- **Input**: User text prompt for image generation.
- **Processing**: Text prompt fed to SD pipeline (no VLM step).
- **Inference**: Local Stable Diffusion inference (architecture-equivalent approximation via OpenRouter description for evaluation).
- **Externalization**:
    - Network: Prompt sent to OpenRouter for visual description (architecture-equivalent evaluation) (`toolneuron.py:374`).
    - UI: Generation progress displayed (`toolneuron_runner.py:95-98`).
    - UI: Final image displayed (`toolneuron_runner.py:99-102`).
    - Storage: Image message persisted to UMS (`toolneuron_runner.py:107-110`).
- **Episode path**: Text Prompt → SD Inference → OpenRouter Description (Network) → UI Display → UMS Storage
- **Key files**: `toolneuron_runner.py`, `toolneuron.py`
- **Confidence**: High

## C. Architecture-Equivalent Approximations

The ToolNeuron app runs on Android using Qualcomm NPU (QNN) for both LLM and SD inference. Since there is no Python SDK for QNN:

| App Component | Approximation Used | Rationale |
|---|---|---|
| LLM Text Generation | llama-cpp-python with GGUF models | Same llama.cpp C++ library as app's `gguf_lib.aar` |
| SD Image Generation | diffusers library (Stable Diffusion 1.5) | Same model architecture as app's LocalDream/QNN SD |
| Image Evaluation | OpenRouter VLM description | Evaluator receives textual description instead of actual pixels |

## Final Summary

- **Total number of distinct externalization sites found**: 6
- **Total number of main AI inference workflows found**: 3
- **Top 3 highest-risk workflows or channels**:
    1. **VLM Image Analysis (Workflow 2)**: Original input images are sent to OpenRouter (Gemini) to generate SD reproduction prompts, potentially leaking visual content to third-party AI.
    2. **OpenRouter Description (Workflow 3)**: SD prompts are sent to OpenRouter for visual description, revealing intended image content.
    3. **UMS Storage (All Workflows)**: All generated content (text and images with metadata) persisted to local database with full content type and model information.
