# LOCAL MODEL NOTES — Evaluation Pipeline (`evaluator.py`)

Author: planner-2  
Target file: `/Users/sieun/Research/Lantern/lantern/verify/backend/evaluation_method/evaluator.py`  
Config file: `/Users/sieun/Research/Lantern/lantern/verify/backend/utils/config.py`

## 1. How `evaluate_inferability()` / `evaluate_both()` call OpenRouter

Call site: `evaluator.py:217-237` (inside a 5-attempt `for attempt in range(5)` retry loop).

- **URL (hardcoded)**: `https://openrouter.ai/api/v1/chat/completions` (line 220).
- **HTTP method**: `requests.post(...)`.
- **Headers** (lines 221-226):
  - `Authorization: Bearer {key}` — key from `api_key` kwarg, else `get_openrouter_api_key()` (`config.py:95`), which reads `OPENROUTER_API_KEY` or `OPENROUTER_KEY` from env / `LANTERN_ROOT/.env`.
  - `Content-Type: application/json`
  - `HTTP-Referer: https://github.com/Verify` — OpenRouter-specific, optional for attribution.
  - `X-Title: Verify` — OpenRouter-specific, optional for attribution.
- **Body** (lines 227-235):
  ```json
  {
    "model": "<model>",
    "messages": [
      {"role": "system", "content": EVAL_SYSTEM_PROMPT},
      {"role": "user",   "content": prompt}
    ],
    "max_tokens": 4096,
    "response_format": {"type": "json_object"}
  }
  ```
- **Timeout**: `60` seconds (line 236).
- **Auth failure sentinel**: if the key starts with `your_` (placeholder), the function short-circuits with `"No valid OpenRouter API key available for evaluation."` (line 208-209).
- **Response handling**: `raw_content = resp.json()["choices"][0]["message"]["content"].strip()`, then strips ASCII control chars (`\x00-\x08\x0b\x0c\x0e-\x1f\x7f`) — this pattern exists to tolerate garbage from Gemini responses, but is harmless for local models. JSON parse falls back to `re.search(r"\{.*\}", ...)` greedy match on failure (lines 244-253).

`evaluate_both()` (lines 281-313) just fan-outs two parallel calls to `evaluate_inferability` via a `ThreadPoolExecutor(max_workers=2)` — one for `original_output`, one for `perturbed_output`. It forwards the same `model` kwarg to both and does not re-open a session or do any OpenRouter-specific wiring of its own.

## 2. `EVAL_MODEL` constant and model parameter threading

- `EVAL_MODEL = "google/gemini-2.0-flash-001"` — declared at `evaluator.py:21`.
- Threading is **clean and already parameterized**:
  - `evaluate_inferability(..., model: str = EVAL_MODEL)` — `evaluator.py:182`.
  - `evaluate_both(..., model: str = EVAL_MODEL)` — `evaluator.py:285`; forwards via `_pool.submit(evaluate_inferability, ..., None, model)` twice (lines 301-302).
- **Callers and whether they pass `model=`**:
  | Caller | Passes `model`? | Notes |
  |---|---|---|
  | `verify/backend/orchestrator.py:385` (`evaluate_both(...)`) | No — uses default `EVAL_MODEL` | Main batch orchestrator path. |
  | `verify/run_batch.py:433,435` (`evaluate_inferability(...)`) | No — default | IOC headless pipeline. |
  | `verify/frontend/pages/2_Input_Output_Comparison.py:257,261` | No — default | Streamlit UI. |
  | `verify/patch_ext_ui.py:158` | No — default | |
  | `verify/reeval.py:181` | **Yes** — `evaluate_inferability(ext_text, attrs, model=model)` | Re-evaluation CLI explicitly takes `--model MODEL`. |
- **Good news**: swapping the default `EVAL_MODEL` to a local model ID or adding an env-var override in this single module would immediately flip all callers that rely on the default. `reeval.py` already accepts per-run model overrides on the command line.

## 3. `response_format: {"type": "json_object"}` — local model compatibility

- OpenRouter accepts this verbatim; many backends honor it, others silently ignore it. The evaluator is already **defensive** about this — when JSON parsing fails it retries and falls back to `re.search(r"\{.*\}", raw_content, re.DOTALL)` (lines 245-253), then loops up to 5 times (line 217). So a local model that ignores `response_format` would still often work via the fallback, provided it puts some JSON-looking blob in the response.
- **Ollama (`/v1/chat/completions` OpenAI-compatible endpoint)**:
  - Ollama's OpenAI-compat layer **does not honor `response_format={"type":"json_object"}`** the same way the native `/api/chat` endpoint does — the native endpoint uses `format: "json"` (or a JSON schema) as a top-level field. In recent Ollama versions (≥0.5), the OpenAI-compat endpoint does accept `response_format` with `{"type":"json_object"}` and internally maps it to `format: "json"`, but behavior varies by version. Older versions will silently ignore it.
  - **Practical recommendation**: when targeting Ollama, either (a) rely on the existing regex fallback and the strong "Return a JSON object ONLY" instruction in `EVAL_SYSTEM_PROMPT` (`prompts/prompt1.yaml`), or (b) send an extra payload key `"format": "json"` when the base URL points to Ollama. The current prompt already says `Return a valid JSON object ONLY, with no additional text.` so fallback parsing should cover most cases.
- **vLLM / llama.cpp / LM Studio / text-generation-webui (OpenAI-compat servers)**:
  - vLLM supports `response_format={"type":"json_object"}` and `{"type":"json_schema", ...}` (guided decoding). Works out of the box.
  - llama.cpp `llama-server` supports `response_format` since v0.0.3+.
  - LM Studio supports it.
- **Conclusion**: leave `response_format: {"type":"json_object"}` in the body — worst case, backends ignore the field and the regex fallback handles it. No need to branch the call site.

## 4. Minimum config changes to swap in a local model

All changes are localized; the hardcoded URL at line 220 is the only real code hotspot.

**Minimum surface**: two env-var overrides plumbed through `config.py` and `evaluator.py`:

- Add in `verify/backend/utils/config.py` (mirroring `get_openrouter_api_key()` at line 95):
  ```python
  def get_eval_base_url() -> Optional[str]:
      """Return OpenAI-compatible base URL for the judge LLM (e.g. http://localhost:11434/v1)."""
      return get_env("EVAL_BASE_URL") or get_env("LOCAL_MODEL_URL")

  def get_eval_model() -> Optional[str]:
      """Return judge LLM model id override (e.g. llama3.1:8b-instruct)."""
      return get_env("EVAL_MODEL")

  def get_eval_api_key() -> Optional[str]:
      """Return API key for judge LLM. Local servers usually accept any non-empty string."""
      return get_env("EVAL_API_KEY") or get_openrouter_api_key() or "local"
  ```

- Patch `evaluator.py` to resolve URL / model / key from config when overrides are set:
  - Replace `EVAL_MODEL = "google/gemini-2.0-flash-001"` with a lazy default that respects `EVAL_MODEL` env var.
  - Build the endpoint as `f"{base_url.rstrip('/')}/chat/completions"` where `base_url` defaults to `https://openrouter.ai/api/v1`.
  - Only set OpenRouter-specific headers (`HTTP-Referer`, `X-Title`) when the base URL is OpenRouter. For local servers those headers are unused but harmless.

This keeps every caller (`orchestrator.py`, `run_batch.py`, `reeval.py`, the two Streamlit pages, `patch_ext_ui.py`) unchanged — they keep calling `evaluate_inferability(...)` / `evaluate_both(...)` with no kwargs and pick up the local backend automatically.

**Bonus**: `get_env()` (`config.py:66-92`) already reads `LANTERN_ROOT/.env`, so users can set these in one place.

## 5. Retry/timeout appropriateness for local models

Current values (lines 217, 233, 236):
- `for attempt in range(5)` — 5 retries.
- `max_tokens=4096`
- `timeout=60` (seconds)

Assessment for local inference:
- **`timeout=60`**: potentially too tight. On CPU-only Ollama with a 7-8B model (e.g. `llama3.1:8b-instruct-q4_K_M`), first-token latency alone can exceed 20-30 s cold, and generating 300-500 tokens of JSON can easily exceed 60 s. On a single GPU with a 7B q4 model this is usually fine; on a 14B-30B model it may not be. **Recommendation**: make the timeout configurable via `EVAL_TIMEOUT` env var (default 60 for OpenRouter, bump to 180-300 for local). 60 s works; 180 s is safer.
- **`max_tokens=4096`**: fine. JSON output for ~20 attributes is usually <800 tokens; 4096 is a generous ceiling and doesn't slow generation (it's just a cap).
- **`for attempt in range(5)`**: 5 retries is **too aggressive** for slow local models. If each call takes 60-180 s and times out, a failing call can eat 5-15 minutes before giving up. The current loop also doesn't sleep between attempts, so it hammers the server. **Recommendation**: reduce to 2-3 retries and add a small backoff when the backend is local. Or leave it — most local failures are parse errors, which are near-instant to retry, so 5 is only costly on network timeouts. Minimum-touch fix: leave as-is.
- **`ThreadPoolExecutor(max_workers=2)` in `evaluate_both`**: two concurrent calls is fine for OpenRouter but will **halve throughput and double VRAM pressure** on a single local GPU with one model loaded. For Ollama (which serializes by default unless `OLLAMA_NUM_PARALLEL>1`), the two calls just queue — no harm, but no speedup either. No code change needed; just be aware.

## 6. Prompt structure compatibility

- `EVAL_SYSTEM_PROMPT` (loaded from `prompts/prompt1.yaml` via `_load_system_prompt()` at lines 27-43) is short, plain English, and instructs `Return a valid JSON object ONLY, with no additional text.` — this works across Llama-3.x, Qwen-2.5, Mistral, Gemma-2, Phi-3. No model-specific formatting (no `<|im_start|>` etc.), so the OpenAI-compatible `messages` array handles role templating server-side. Good.
- A `channel_guidance` paragraph is appended (lines 34-39) explaining the `[UI]/[NETWORK]/[STORAGE]/[LOGGING]` labels — also plain English, no compatibility issues.
- The user prompt from `_build_eval_prompt()` (lines 142-175) embeds:
  - a bullet list of attribute names with per-attribute guidance from `ATTRIBUTE_GUIDANCE` (lines 48-139),
  - the AI system output between `---` fences,
  - a literal JSON schema example generated by `json.dumps(schema_example, indent=2)`.
- **Potential issues with small local models (≤7B)**:
  - Long prompts. With all 20+ attributes in `ATTRIBUTE_GUIDANCE` enumerated, the user prompt can reach 1.5-3k tokens before the output text is appended. Target-app output may add another few hundred-thousand tokens. Context of 4-8k is typically enough but leaves tight margin with small-context models. Ensure the local model is configured with ≥8k context (`num_ctx` for Ollama).
  - Strict JSON emission. Small instruction-tuned models sometimes add prose like `Here is the JSON:` before the object. The existing regex fallback `re.search(r"\{.*\}", raw_content, re.DOTALL)` handles this. No change needed.
  - Boolean keys. The schema uses Python-style `True`/`False` in the `schema_example` (because it's a Python `dict`). `json.dumps` correctly emits lowercase `true`/`false`. Verified — no bug here.
- **Conclusion**: prompt is portable. Keep as-is.

## 7. `config.py` hooks for a clean `LOCAL_MODEL_URL` addition

`config.py` already has the right shape — `get_env()` (lines 66-92) handles env + `.env` with quote stripping, and there are precedent helpers (`get_openrouter_api_key`, `get_openai_api_key`, `is_debug`, `is_malicious_prompt_mode`). Adding judge-model helpers fits the existing pattern cleanly.

Suggested additions (insert around line 97, next to `get_openrouter_api_key`):

```python
def get_eval_base_url() -> Optional[str]:
    """
    Return OpenAI-compatible base URL for the judge LLM used by
    verify.backend.evaluation_method.evaluator. Set to e.g.
    http://localhost:11434/v1 (Ollama) or http://localhost:8000/v1 (vLLM).
    Falls back to OpenRouter when unset.
    """
    return get_env("EVAL_BASE_URL") or get_env("LOCAL_MODEL_URL")


def get_eval_model_id() -> Optional[str]:
    """Override the judge model id (e.g. 'llama3.1:8b-instruct-q4_K_M')."""
    return get_env("EVAL_MODEL")


def get_eval_api_key() -> Optional[str]:
    """
    Return an API key for the judge LLM endpoint. Local OpenAI-compat
    servers (Ollama, vLLM, LM Studio) usually accept any non-empty string,
    so we default to 'local' when neither EVAL_API_KEY nor OPENROUTER_API_KEY
    is set.
    """
    return get_env("EVAL_API_KEY") or get_openrouter_api_key() or "local"


def get_eval_timeout() -> int:
    """Request timeout (seconds) for the judge LLM. Local models need longer."""
    raw = get_env("EVAL_TIMEOUT", "60") or "60"
    try:
        return int(raw)
    except ValueError:
        return 60
```

## 8. Recommended minimal patch to `evaluator.py`

Pseudo-diff for the lowest-risk swap:

```python
# Top of evaluator.py
from verify.backend.utils.config import (
    get_openrouter_api_key,
    get_eval_base_url,
    get_eval_model_id,
    get_eval_api_key,
    get_eval_timeout,
)

# Module-level default — respects EVAL_MODEL env var if set, else OpenRouter Gemini.
EVAL_MODEL = get_eval_model_id() or "google/gemini-2.0-flash-001"

# Inside evaluate_inferability(), replace the request block:
base_url = (get_eval_base_url() or "https://openrouter.ai/api/v1").rstrip("/")
endpoint = f"{base_url}/chat/completions"
key = api_key or get_eval_api_key()

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type":  "application/json",
}
if "openrouter.ai" in base_url:
    headers["HTTP-Referer"] = "https://github.com/Verify"
    headers["X-Title"]      = "Verify"

resp = requests.post(
    endpoint,
    headers=headers,
    json={
        "model": model,
        "messages": [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    },
    timeout=get_eval_timeout(),
)
```

Net effect: unchanged behavior on OpenRouter, but setting `EVAL_BASE_URL=http://localhost:11434/v1` + `EVAL_MODEL=llama3.1:8b-instruct` + (optionally) `EVAL_TIMEOUT=180` in `.env` flips the judge to a local Ollama instance without touching any caller. `reeval.py --model <local-id>` continues to work because its explicit `model=` still wins over the env-var default.

## 9. Open questions / risks

- **Does Ollama serialize calls?** `evaluate_both()` fires 2 requests in parallel. On a single-GPU Ollama with default `OLLAMA_NUM_PARALLEL=1` they will queue — functionally correct, throughput-neutral. If we want true parallelism, recommend users set `OLLAMA_NUM_PARALLEL=2` or move to vLLM.
- **Stream mode**: none of the current code uses `stream=True`, so no changes needed.
- **Structured output**: if we later want stricter JSON guarantees on Ollama, switch from `response_format: {"type":"json_object"}` to the OpenAI-compat `response_format: {"type":"json_schema","json_schema":{...}}` using the same schema as `schema_example` in `_build_eval_prompt`. vLLM, LM Studio and recent Ollama all support this. Not needed for v1.
- **Cost of 5 retries**: leave as-is for v1; revisit if local timeouts become common.
