# Verify Framework Model Usage

## `google/gemini-2.5-pro`

`google/gemini-2.5-pro` is the framework-wide default OpenRouter model for app inference paths that call `BaseAdapter._call_openrouter()` without passing an explicit model.

Primary definition:

- `verify/backend/adapters/base.py`
  - `OPENROUTER_DEFAULT_MODEL = "google/gemini-2.5-pro"`

Primary call path:

- `BaseAdapter._call_openrouter(..., model=None, ...)`
  - if no model override is provided, it sends `OPENROUTER_DEFAULT_MODEL` to `https://openrouter.ai/api/v1/chat/completions`.

Adapters that import or use this default include:

- `clone`
- `deeptutor`
- `xend`
- `llm-vtuber`
- `snapdo`
- `pocketpal-ai`
- `waico`
- `oxproxion`

Some runner/server wrappers also default to `google/gemini-2.5-pro`, including clone, deeptutor, snapdo, xend, and llm-vtuber runners.

## Evaluation Models

Inferability evaluation is separate from app inference. The evaluator uses `VERIFY_EVAL_MODEL` or `EVAL_MODEL` from `.env` when it is a valid OpenRouter model id. If the configured value looks like a local/Ollama model name, the evaluator falls back to `google/gemini-2.0-flash-001` so OpenRouter does not receive an invalid local model id.

You can explicitly re-evaluate with Gemini 2.5 Pro:

```bash
python verify/reeval.py --model google/gemini-2.5-pro
```

## ToolNeuron Image Pipelines

ToolNeuron image generation does not use `google/gemini-2.5-pro` by default.

Current defaults:

- text-to-image / image-to-image generation: `google/gemini-3-pro-image-preview`
- image-generation fallbacks: `google/gemini-3.1-flash-image-preview`, `google/gemini-2.5-flash-image`
- image edit prompt writing: `google/gemini-2.0-flash-001`
- generated image description helper: `google/gemini-2.0-flash-001`

The image model can be changed with `TOOL_NEURON_IMAGE_MODEL` or `TOOL_NEURON_REAL_IMAGE_MODEL`.
