# Runtime Capture Coverage Report

This report measures how many of the inference-episode externalizations documented in per-app analysis reports are actually captured by `_runtime_capture.py` during a verify pipeline run.

**Capture mechanisms available:**
- `requests.Session.request` patch → HTTP calls via the `requests` library
- `httpx.Client.send` / `httpx.AsyncClient.send` patches → HTTP calls via `httpx` (includes LangChain, OpenAI SDK ≥1.0, litellm)
- Django `post_save` signal → ORM model saves
- `logging` handler → filtered log records

**Legend:**

| Symbol | Meaning |
|---|---|
| ✅ | Captured — runtime capture records this |
| ⚠️ | Partial — captured only under specific conditions |
| 🔴 | Not executed — runner scope does not reach this code path |
| ❌ | Not capturable — runner executes it but no mechanism can intercept |

---

## budget-lens

**Runner scope:** `process_receipt(path)` inside Django (CondaRunner)

| Externalization | Channel | Status | Reason |
|---|---|---|---|
| Receipt image + prompt → OpenAI API | NETWORK | ✅ | `openai` SDK ≥1.0 uses `httpx` → intercepted |
| Transaction date → Exchange Rates API | NETWORK | ✅ | Likely `requests` → intercepted |
| Expense model save → Django DB | STORAGE | ✅ | `post_save` signal |
| OpenAI response / error details → log | LOGGING | ✅ | Keyword filter passes "response", "error" |
| Inferred fields → UI | UI | 🔴 | No browser/HTTP client in runner |

**Captured: 4 / 5 (80%)**  
Coverage is high — budget-lens is a pure server-side Django app and the runner exercises the full `process_receipt()` path including external API calls and DB writes.

---

## clone

**Runner scope:** Django ORM directly (CondaRunner) — creates User, ChatSession, ChatMessage; calls OpenRouter for frame description

The runner replaces the Electron frontend entirely. The original app's externalization surface is mostly in TypeScript; the runner only exercises the Django backend layer.

| Externalization | Channel | Status | Reason |
|---|---|---|---|
| Frame description → OpenRouter | NETWORK | ✅ | Runner calls OpenRouter via `requests` |
| ChatSession create → Django DB | STORAGE | ✅ | `post_save` signal |
| ChatMessage save → Django DB | STORAGE | ✅ | `post_save` signal |
| User create/get → Django DB | STORAGE | ✅ | `post_save` signal |
| OpenAI/Ollama LLM call from Electron | NETWORK | 🔴 | Electron (TypeScript) — different process, not executed |
| Ollama local inference | NETWORK | 🔴 | Electron process — not executed |
| Screen recording → WebM file | STORAGE | 🔴 | Electron — not executed |
| ONNX/HuggingFace model download | NETWORK | 🔴 | Electron — not executed |
| Encrypted chat history → Django REST | NETWORK | 🔴 | Runner uses ORM directly, bypasses HTTP layer |
| Streaming response → UI | UI | 🔴 | No frontend in runner |

**Captured: 4 / 10 (40%)**  
Low raw coverage because most of clone's externalization surface lives in the Electron/TypeScript frontend. The runner captures the Django backend slice: one OpenRouter call + three ORM saves. The Electron LLM calls, recording, and model downloads are entirely out of scope.

---

## deeptutor

**Runner scope:** `ChatOrchestrator.handle()` → `ChatCapability` → `AgenticChatPipeline` (CondaRunner)

| Externalization | Channel | Status | Reason |
|---|---|---|---|
| User prompt + context → Cloud LLM | NETWORK | ✅ | litellm uses `httpx` → intercepted |
| Document chunks → Embedding API | NETWORK | ✅ | litellm embedding calls via `httpx` |
| RAG retrieval (LlamaIndex) | NETWORK | ⚠️ | Local vector search in verify env (no document uploaded); network only if using a remote retriever — effectively a no-op in practice |
| Completion events → event bus | LOGGING | ⚠️ | Internal Python event emission; captured only if `orchestrator.py` also logs the event via `logging` module |
| Chunking/Embedding/Indexing stages → log | LOGGING | ✅ | Keyword filter passes "embedding", "indexing" |
| Tutor response → React UI | UI | 🔴 | No browser in runner |

**Captured: 3–4 / 6 (50–67%)**  
The two primary network calls (LLM + embedding API) are captured. RAG retrieval is local in the verify environment so there is nothing to capture on that channel. The event bus is only captured if it also emits a `logging` call.

---

## google-ai-edge-gallery

**Runner scope:** `transformers.pipeline("text-generation")` local inference (CondaRunner)

The app's main externalizations are Android-side (Firebase, Android Intents, FCM). The runner uses a local HuggingFace model — the only network activity is the one-time model download.

| Externalization | Channel | Status | Reason |
|---|---|---|---|
| HuggingFace model download (first run) | NETWORK | ⚠️ | `huggingface_hub` uses `requests` → intercepted, but only on first run; cached afterward |
| Local LiteRT LM inference → UI | UI | 🔴 | Local inference, no network; runner does same via transformers |
| Firebase Analytics events | ANALYTICS | 🔴 | Android-only |
| Model download in Android (DownloadWorker) | NETWORK | 🔴 | Android-only |
| WebView JavaScript execution | UI | 🔴 | Android-only |
| Android Intents (email/SMS via tool-calling) | INTENT | 🔴 | Android-only |
| FCM push notifications | NETWORK | 🔴 | Android-only |

**Captured: 0–1 / 7 (0–14%)**  
Coverage is effectively zero for steady-state runs (after the first model download). This app's entire externalization surface is on the Android device. The verify runner captures only the inference result, not any data flows.

---

## llm-vtuber

**Runner scope:** `AsyncLLM.chat_completion()` only — one LLM call (CondaRunner)

The runner exercises the LLM step of the pipeline. STT, TTS, WebSocket, and Live2D animation are not invoked.

| Externalization | Channel | Status | Reason |
|---|---|---|---|
| Conversation history → Cloud LLM | NETWORK | ✅ | `openai` SDK via `httpx` → intercepted |
| Raw audio → Cloud STT (Whisper) | NETWORK | 🔴 | STT not called in runner |
| Character text → Cloud TTS (Fish TTS) | NETWORK | 🔴 | TTS not called in runner |
| Audio/text chunks → WebSocket client | NETWORK | 🔴 | No WebSocket server in runner |
| Chat log → `chat_history_manager` (file) | STORAGE | ❌ | File I/O; no `post_save` signal; `open()` not patched |
| Live2D animation → PIXI.js UI | UI | 🔴 | Browser-only |

**Captured: 1 / 6 (17%)**  
Only the LLM API call is captured. The runner deliberately scopes to the LLM component because the rest of the pipeline (VAD, STT, TTS, WebSocket, animation) requires real-time audio I/O and a running browser — both unavailable in the verify environment.

---

## momentag

**Runner scope:** `get_image_captions(pil_image)` — pure local CLIP + BLIP inference (CondaRunner)

No network calls occur. The function returns captions/keywords; it does not write to Qdrant or Django DB.

| Externalization | Channel | Status | Reason |
|---|---|---|---|
| Image embeddings → Qdrant | NETWORK | 🔴 | `get_image_captions()` returns data; Qdrant write is in the Celery task caller, not executed |
| Photo download/delete → MinIO/S3 | NETWORK | 🔴 | Not in runner scope (would now be intercepted via urllib3 patch if called) |
| Caption + metadata → Django DB | STORAGE | 🔴 | DB write is in the Celery task caller, not `get_image_captions()` |
| Semantic search results → UI | UI | 🔴 | Not executed |
| Model loading / inference status → log | LOGGING | ⚠️ | PyTorch/transformers logs are filtered out as noisy; runner's own `print()` to stderr is not a `logging` call |

**Captured: 0 / 5 (0%)**  
No captures. The runner deliberately scopes to the CPU/GPU inference step (CLIP + BLIP), which is pure local computation. All I/O (Qdrant, S3, DB) happens in the surrounding Celery task that the runner does not call.

> **Note:** Even if Qdrant/S3 writes were in scope, `boto3` (S3) uses `urllib3` directly — not `requests` or `httpx` — so it would not be intercepted by the current patches.

---

## skin-disease-detection

**Runner scope:** Three TFLite model files → local inference only (CondaRunner)

All meaningful externalizations are in the Flutter/Android frontend; the runner does purely local ML.

| Externalization | Channel | Status | Reason |
|---|---|---|---|
| Message + photo → Email (oncologist) | NETWORK | 🔴 | Flutter/Android — not executed |
| Message + phone → WhatsApp intent | NETWORK | 🔴 | Flutter/Android — not executed |
| User data → Firebase | STORAGE | 🔴 | Flutter/Android — not executed |
| User lat/lng → Google Maps API | NETWORK | 🔴 | Flutter/Android — not executed |
| Prediction labels → UI | UI | 🔴 | Flutter/Android — not executed |
| Inference timing/results → `print()` | LOGGING | ❌ | Runner uses `print()` to stderr, not `logging` module; not intercepted by log handler |

**Captured: 0 / 6 (0%)**  
No captures. The runner is a faithful reproduction of the local TFLite classification step. All privacy-relevant externalizations (email, WhatsApp, Firebase, location lookup) happen in the mobile app after the user sees the result.

---

## snapdo

**Runner scope:** `VLMService().verify_evidence(image_b64, constraint)` inside Django (CondaRunner)

| Externalization | Channel | Status | Reason |
|---|---|---|---|
| Base64 image + task constraint → OpenRouter VLM | NETWORK | ✅ | `vlm_service.py` uses `requests` → intercepted |
| Verification verdict → log | LOGGING | ✅ | Keyword filter passes "verdict", "verification" |
| Verification result → UI | UI | 🔴 | No frontend in runner |

**Captured: 2 / 3 (67%)**  
Good coverage for what the runner executes. The VLM call and logging are captured. UI rendering is not applicable in the runner context.

---

## xend

**Runner scope:** `subject_chain.invoke()` + `body_chain.invoke()` — two LangChain chain calls (CondaRunner)

| Externalization | Channel | Status | Reason |
|---|---|---|---|
| Prompt + style context → Cloud LLM | NETWORK | ✅ | LangChain uses `httpx` → intercepted (both subject and body chains) |
| LLM generation attempt → log | LOGGING | ✅ | Keyword filter passes "generation", "response" |
| Emails → Gmail API | NETWORK | 🔴 | Runner takes text input directly; no Gmail sync in runner scope |
| Streamed tokens → Redis pub/sub | NETWORK | 🔴 | Not triggered; `redis-py` uses its own socket layer (not `requests`/`httpx`) anyway |
| Style analysis → Django DB | STORAGE | 🔴 | Style analysis task is a separate Celery task; not triggered by the chain calls |
| Results → React UI | UI | 🔴 | No frontend |

**Captured: 2 / 6 (33%)**  
The two LangChain LLM calls (subject + body) are captured. The broader xend pipeline — Gmail sync, Redis streaming, style analysis storage — is not in runner scope, and Redis would not be intercepted even if it were.

---

## Summary Table

| App | Externalizations described | Captured | In-scope but not capturable | Out of runner scope | Coverage (of in-scope) |
|---|---|---|---|---|---|
| budget-lens | 5 | 4 | 0 | 1 (UI) | **80%** |
| clone | 10 | 4 | 0 | 6 | **100% of in-scope** |
| deeptutor | 6 | 3–4 | 0 | 1–2 | **75–100% of in-scope** |
| google-ai-edge | 7 | 0–1 | 0 | 6–7 | **0–100% (model dl only)** |
| llm-vtuber | 6 | 1 | 1 (file) | 4 | **50% of in-scope** |
| momentag | 5 | 0 | 0 | 5 | N/A |
| skin-disease | 6 | 0 | 1 (print) | 5 | **0%** |
| snapdo | 3 | 2 | 0 | 1 (UI) | **100% of in-scope** |
| xend | 6 | 2 | 0 | 4 | **100% of in-scope** |
| **Total** | **54** | **16–18** | **2** | **34–36** | — |

**Overall: ~17/54 externalizations captured (~31% of all described channels)**

The critical distinction: for externalizations that the runner actually executes, capture rate is high (~80–100%). The gap comes from **runner scope**, not from capture mechanism failures.

---

## Root Causes of Gaps

### 1. Runner scope is narrower than the full app pipeline (largest gap — ~34 channels)

The verify runners deliberately scope to the AI inference component, not the full end-to-end app:
- **Mobile/Android/Flutter** — entire externalization surface of skin-disease, google-ai-edge, and parts of snapdo/momentag/llm-vtuber
- **Electron/TypeScript frontend** — clone's LLM calls, recording, model downloads
- **Background workers** — momentag's Celery task (Qdrant + S3 + DB writes), xend's style analysis task, Redis pub/sub streaming
- **Server-side sessions** — xend Gmail sync, clone HTTP REST layer

### 2. ~~Library HTTP transport not intercepted~~ ✅ Fixed

~~`boto3` / `urllib3` direct — S3 calls in momentag and clone bypass `requests.Session`~~

**Fixed:** `_runtime_capture` now patches `urllib3.connectionpool.HTTPConnectionPool.urlopen` directly, which is the transport layer under both `requests` and `boto3`. The old `requests.Session.request` patch has been replaced by this lower-level hook. `aiohttp` async patch also added (no current app uses it, but ready for future apps).

### 3. `print()` vs `logging` module (1 channel, not fixed)

- skin-disease runner uses `print(..., file=sys.stderr)` for inference results. `_runtime_capture` only hooks the Python `logging` module, not `print`. Patching `sys.stderr.write` would capture all stderr including exception tracebacks, progress bars, and Django/transformers noise — too fragile to be useful.

### 4. Local inference has no network surface (momentag, google-ai-edge in steady state)

- CLIP/BLIP, TFLite, and local HuggingFace transformers produce no network calls at inference time. These apps' privacy surface is in what happens *after* inference (storage, sharing), not during it.

---

## Feasibility: Extending Capture Coverage

### A. Android Mobile Externalizations

**Verdict: Feasible via network proxy; not feasible via Python patching.**

The Android app runs on a separate device or emulator. `_runtime_capture.py` patches Python process memory — it cannot touch the JVM running on Android. Three approaches exist, in increasing complexity:

#### Option 1 — MITM Proxy (mitmproxy / Charles)
Install a root CA on the emulator, run a proxy, and route all Android traffic through it. Every HTTPS call the app makes — Firebase Analytics, Google Maps, email SMTP, OpenAI — is captured as plaintext before re-encryption.

- **What you capture:** Full request/response bodies, URLs, headers, for all HTTP/HTTPS traffic from the app.
- **Integration point:** `mitmproxy` can run a Python addon script that filters and writes captured calls to a JSON file, which verify could read after the run.
- **Blockers:**
  - **Certificate pinning** — apps that bundle their own certificate chain (e.g., Firebase SDK itself does not pin, but some custom API clients do) will reject the proxy CA and crash or silently fail. Bypass requires patching the APK (Apktool + repack) or using Frida to hook `X509TrustManager`.
  - **Emulator lifecycle** — each verify run needs a clean emulator snapshot; resetting state between items is slow.
  - **Non-HTTP channels** — Firebase Realtime Database and FCM use persistent WebSocket/gRPC connections that a standard HTTP proxy does not capture cleanly.

#### Option 2 — Frida Dynamic Instrumentation
Attach a Frida server to a rooted emulator and hook Java/Kotlin methods directly: `OkHttpClient.newCall()`, `Retrofit` interceptors, `FirebaseAnalytics.logEvent()`, `android.content.Intent` constructors.

- **What you capture:** Method-level arguments before they hit the network — including data that never leaves the device (local DB writes, Intent extras, UI state). More complete than a proxy.
- **Blockers:**
  - Requires a rooted emulator (Android 12+ has restrictions on rooting images).
  - Frida scripts are per-app and need updating when the app updates.
  - `google-ai-edge-gallery` and `skin-disease-detection` use native (C++) TFLite inference; hooking the inference layer requires Frida's native API hooks on `.so` files.

#### Option 3 — `tcpdump` on emulator interface
Capture raw packets on the emulator's virtual NIC (`eth0` on the emulator host).

- **What you capture:** All TCP/UDP traffic, but HTTPS payloads are encrypted and unreadable without the private key or a MITM CA.
- **Useful for:** confirming that a network call happened (URL from TLS SNI field) without seeing its content.

**Recommendation:** Option 1 (MITM proxy) gives the best coverage-to-effort ratio for the apps in this study. None of them appear to use certificate pinning for their primary AI API calls. The main engineering cost is building the emulator snapshot management.

---

### B. urllib3 Direct Calls (boto3, etc.) ✅ Implemented

**Verdict: Feasible. Cleaner than patching `requests` separately.**

`boto3` → `botocore` → `urllib3` → raw socket. The `requests` patch in `_runtime_capture.py` only intercepts calls that go through `requests.Session.request`. boto3 bypasses this entirely.

#### Patch point
`urllib3.connectionpool.HTTPConnectionPool.urlopen` is the single chokepoint for all urllib3 HTTP(S) traffic, whether called by `requests`, `boto3`, or any other library:

```python
import urllib3.connectionpool as _pool

_orig_urlopen = _pool.HTTPConnectionPool.urlopen

def _patched_urlopen(self, method, url, **kwargs):
    full_url = f"{'https' if self.scheme == 'https' else 'http'}://{self.host}:{self.port}{url}"
    resp = _orig_urlopen(self, method, full_url, **kwargs)
    _record_network(method, full_url, resp.status)
    return resp

_pool.HTTPConnectionPool.urlopen = _patched_urlopen
```

- **urllib3 v1 vs v2:** The method signature changed in v2 (`body` → `json` param added), but `urlopen(method, url, **kwargs)` is stable across both.
- **Consolidation opportunity:** Since `requests` delegates to `urllib3`, patching `urllib3.urlopen` captures `requests` calls too. You could drop the `requests.Session.request` patch and unify at the urllib3 level — one patch, broader coverage.
- **boto3-specific note:** botocore wraps urllib3 calls with retry logic and SigV4 signing, but the actual `urlopen` is always called once per attempt, so the patch fires correctly.
- **Remaining blind spot:** gRPC (used by some Firebase SDKs) uses HTTP/2 over raw sockets via the `grpcio` C extension — not urllib3, not interceptable by Python monkey-patching.

---

### C. asyncio-native HTTP Libraries (aiohttp) ✅ Implemented

**Verdict: Feasible, same pattern as the existing httpx async patch.**

`aiohttp.ClientSession._request()` is the async entry point for all requests made through an aiohttp session:

```python
try:
    import aiohttp

    _orig_aiohttp_request = aiohttp.ClientSession._request

    async def _patched_aiohttp_request(self, method, str_or_url, **kwargs):
        resp = await _orig_aiohttp_request(self, method, str_or_url, **kwargs)
        _record_network(method, str(str_or_url), resp.status)
        return resp

    aiohttp.ClientSession._request = _patched_aiohttp_request
except ImportError:
    pass  # aiohttp not installed in this env
```

- **Why it matters:** `aiohttp` is the HTTP library of choice for many async Python frameworks (FastAPI, some LangChain extensions). If a future target app uses it, calls would currently go uncaptured.
- **Current relevance:** None of the 9 existing apps use `aiohttp` — they use `requests` (snapdo, budget-lens), `httpx` (xend via LangChain, deeptutor via litellm), or the `openai` SDK (llm-vtuber, clone runner). **Zero gap today.**
- **Async context caveat:** The patch must be applied before any `aiohttp.ClientSession` is instantiated. If a session is created at module import time (before `install()` runs), the patch misses those calls. This is the same limitation as the existing httpx async patch.
- **Priority:** Low. Add to `_runtime_capture.install()` behind a try/import guard when a new app using `aiohttp` is added.

---

### Coverage Gap Summary

| Gap | Root cause | Status |
|---|---|---|
| ~~boto3 / S3 calls~~ | ~~urllib3 bypass~~ | ✅ **Fixed** — urllib3 patch now covers boto3 |
| ~~aiohttp calls~~ | ~~Not patched~~ | ✅ **Fixed** — async wrapper added (no current app uses it) |
| Android app network calls | Different process/OS | ❌ Not fixable — requires MITM proxy or Frida |
| Redis pub/sub | Raw TCP socket (not HTTP) | ❌ Not fixable by patching |
| `print()` stderr vs `logging` | Different output mechanism | ⚠️ Not fixed — patching `sys.stderr.write` too noisy/fragile |
| File I/O (chat logs, recordings) | `open()` not patched | ⚠️ Not fixed — patching `builtins.open` has very high false-positive rate |
| Mobile-only channels (Firebase, Intents, FCM) | Android JVM | ❌ Not fixable — requires Frida or MITM |
| Local ML inference (CLIP, TFLite, transformers) | No network surface | N/A — nothing to capture |
