# Local Model Setup Guide — Verify Pipeline

This guide covers running the Verify privacy-evaluation pipeline entirely offline using a local Ollama model instead of OpenRouter.

---

## 1. What `INFER_LOCAL` controls

| `INFER_LOCAL` | Perturbation (text) | Perturbation (vision) | Evaluation (LLM-as-judge) |
|---|---|---|---|
| `false` (default) | OpenRouter | OpenRouter | OpenRouter |
| `true` | Local Ollama | Local Ollama | Local Ollama |

When `INFER_LOCAL=false` a valid `OPENROUTER_API_KEY` is required.  
When `INFER_LOCAL=true` no API key is needed; Ollama must be running locally.

---

## 2. Hardware requirements

### Minimum (gemma4:e4b — auto-selected when RAM < 8 GB)

| | Spec |
|---|---|
| RAM / VRAM | 6 GB free |
| Storage | ~4 GB for model weights |
| CPU | Any modern x86-64 or Apple Silicon |
| GPU | Optional — CPU inference works, ~4–8 tok/s |

Expected throughput (CPU, Apple M-series or modern x86):
- Perturbation: ~45–90 s per item
- Evaluation: ~60–120 s per item

### Recommended (gemma4:26b MoE — auto-selected when RAM ≥ 8 GB)

| | Spec |
|---|---|
| RAM / VRAM | 8 GB free (unified or discrete) |
| Storage | ~17 GB for model weights |
| CPU / GPU | Apple Silicon M2 Pro+ / NVIDIA RTX 3080+ |

Expected throughput (Apple M2 Pro 16 GB or NVIDIA RTX 3080):
- Perturbation: ~15–30 s per item
- Evaluation: ~20–40 s per item

### Why gemma4:26b at only 8 GB?

gemma4:26b is a **Mixture-of-Experts** model that activates only 3.8B parameters per token.  
Inference speed and memory footprint are close to a 4B dense model, while quality approaches 31B.

---

## 3. Model selection logic

```
INFER_LOCAL=true
      │
      ├─ LOCAL_MODEL_NAME set?  ──yes──► use that model
      │
      └─ no ──► check available RAM/VRAM
                    ≥ 8 192 MB ──► gemma4:26b   (recommended)
                    <  8 192 MB ──► gemma4:e4b   (fallback)
```

Per-pipeline overrides (`PERTURBATION_TEXT_MODEL`, `PERTURBATION_VISION_MODEL`, `EVAL_MODEL`) take precedence over `LOCAL_MODEL_NAME`.

---

## 4. Installation

### Step 1 — Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:
```bash
ollama --version
```

### Step 2 — Pull the model(s)

```bash
# Recommended (≥ 8 GB free RAM — ~17 GB download)
ollama pull gemma4:26b

# Fallback (< 8 GB free RAM — ~4 GB download)
ollama pull gemma4:e4b

# Pull both if you want auto-select to work without a network connection
ollama pull gemma4:26b
ollama pull gemma4:e4b
```

### Step 3 — Start the Ollama server

```bash
ollama serve
```

Ollama listens on `http://localhost:11434` by default.  
Leave this terminal open while running the pipeline, or run it as a background service:

```bash
# macOS — run as a launchd service (starts on login)
brew services start ollama
```

### Step 4 — Configure the pipeline

Edit (or create) `<repo-root>/.env`:

```dotenv
# Enable local inference
INFER_LOCAL=true

# Optional: pin a specific model instead of auto-select
# LOCAL_MODEL_NAME=gemma4:26b

# Optional: per-pipeline overrides
# PERTURBATION_TEXT_MODEL=gemma4:26b
# PERTURBATION_VISION_MODEL=gemma4:26b
# EVAL_MODEL=gemma4:26b

# Optional: increase timeout for CPU inference (seconds)
# EVAL_TIMEOUT=300
```

### Step 5 — Verify the setup

```bash
# Confirm Ollama is reachable
curl http://localhost:11434/v1/models

# Quick smoke test through the pipeline config
cd <repo-root>
python - <<'EOF'
from verify.backend.utils.config import is_infer_local
from verify.backend.utils.llm_client import pick_local_model, _get_available_memory_mb
print("INFER_LOCAL   :", is_infer_local())
print("Available RAM :", _get_available_memory_mb(), "MB")
print("Selected model:", pick_local_model())
EOF
```

Expected output (16 GB Apple Silicon):
```
INFER_LOCAL   : True
Available RAM : 8960 MB
Selected model: gemma4:26b
```

---

## 5. Per-pipeline env vars reference

| Variable | Default | Description |
|---|---|---|
| `INFER_LOCAL` | `false` | Master toggle: `true` routes all LLM calls to local Ollama |
| `LOCAL_MODEL_URL` | `http://localhost:11434/v1` | Ollama base URL (change for remote Ollama or vLLM) |
| `LOCAL_MODEL_NAME` | *(auto)* | Override auto VRAM-based selection for all pipelines |
| `PERTURBATION_TEXT_MODEL` | *(auto)* | Model for PrivacyLens text rewriting |
| `PERTURBATION_VISION_MODEL` | *(auto)* | Model for Simple_Blur region detection |
| `EVAL_MODEL` | *(auto)* | Model for LLM-as-judge evaluation |
| `EVAL_TIMEOUT` | `300` (local) / `60` (OpenRouter) | HTTP timeout in seconds for evaluation calls |

---

## 6. Switching back to OpenRouter

```dotenv
INFER_LOCAL=false
OPENROUTER_API_KEY=sk-or-...
```

No other changes needed. The pipeline falls back to `google/gemini-2.0-flash-001` on OpenRouter.

---

## 7. Using a custom local endpoint (vLLM, LM Studio)

Any OpenAI-compatible `/v1/chat/completions` endpoint works:

```dotenv
INFER_LOCAL=true
LOCAL_MODEL_URL=http://localhost:8000/v1     # vLLM
LOCAL_MODEL_NAME=google/gemma-4-27b-it      # model name as served
```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: LLM call failed (http://localhost:11434/v1/chat/completions)` | Ollama not running | `ollama serve` |
| `model "gemma4:26b" not found` | Model not pulled | `ollama pull gemma4:26b` |
| Evaluation times out | CPU inference too slow | Set `EVAL_TIMEOUT=600` or `LOCAL_MODEL_NAME=gemma4:e4b` |
| JSON parse retries | Small model inconsistency | Use `gemma4:26b`; it has native structured-output support |
| Wrong model auto-selected | VRAM detection failed on your hardware | Set `LOCAL_MODEL_NAME` explicitly |
