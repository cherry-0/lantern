# AI Inference Privacy Audit: lira

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM) | `backend/app/services/llm.py` | 27–43 | `LLMService.generate_reply` | User message + up to last 10 conversation turns + Lira grandmother system prompt sent to OpenRouter (`mistral-7b-instruct` by default). Full message history included via `request.conversation[-10:]`. | `await self._client.post("/v1/chat/completions", json=payload)` where `payload["messages"]` contains system + history + user turn | High |
| 2 | Network Request (Cloud STT) — optional | `backend/app/routers/stt.py` | — | STT router | Audio file (base64) transcribed locally via OpenAI Whisper model running on-device. No network call for STT in the default configuration (model runs locally). | `openai-whisper==20231117` in `requirements.txt`; local inference | Medium |
| 3 | Local TTS synthesis — no externalization | `backend/app/routers/tts.py` | — | TTS router | Bot reply synthesized locally via Coqui TTS. Audio never leaves the device. | `TTS==0.22.0` in `requirements.txt`; local inference | High |

## B. Main AI Inference Workflows

### Workflow 1: Agentic Chat (`POST /chat/`)
- **Purpose**: Generate empathetic conversational responses as "Lira", an Ethiopian grandmother AI companion.
- **Input**: Ordered list of conversation `Message` objects (role + content), locale, and optional model override.
- **Processing**:
  - `_build_payload()` constructs a messages array: system prompt prepended, then the last 10 conversation turns from `request.conversation[-10:]`.
  - `temperature=0.7` applied; model defaults to `openrouter/mistral-7b-instruct` (overridable via `LLM_MODEL` env var or per-request `model` field).
  - Response parsed from `choices[0].message.content`; optional `reasoning_content` captured if the model provides it.
- **Inference**: OpenRouter routes to `mistral-7b-instruct` (or caller-specified model) and returns the reply.
- **Externalization**:
  - Conversation history (up to 10 turns) + user message sent to `openrouter.ai` (Channel 1). Personal disclosures accumulate across turns and all re-leave the device on each new request.
- **Episode path**: User message + history → `_build_payload()` → POST `/v1/chat/completions` (OpenRouter) → `reply` string → Flutter client
- **Key files**: `backend/app/services/llm.py` (`LLMService`, `generate_reply`, `_build_payload`), `backend/app/routers/chat.py`
- **Confidence**: High

### Workflow 2: Speech-to-Text (`POST /stt/`)
- **Purpose**: Transcribe voice input from the Flutter client into text for the chat pipeline.
- **Input**: Base64-encoded audio + optional Whisper model size (`tiny` / `base` / `small`) and language hint.
- **Processing**: Audio decoded and fed into a locally-running OpenAI Whisper model.
- **Inference**: Whisper runs on the server machine — no network request to a third party in the default configuration.
- **Externalization**: None in default setup; audio processed locally (Channel 2 marked Medium because a custom `LLM_API_BASE_URL` deployment could route this differently).
- **Episode path**: Device audio → base64 encode → POST `/stt/` → local Whisper → transcript text → chat pipeline
- **Key files**: `backend/app/routers/stt.py`, `backend/app/schemas.py` (`STTRequest`, `STTSettings`)
- **Confidence**: Medium

### Workflow 3: Text-to-Speech (`POST /tts/`)
- **Purpose**: Convert Lira's text reply into audio for playback on the Flutter client.
- **Input**: Reply text + voice name (`grandma`) + speed parameter.
- **Processing**: Coqui TTS model runs locally on the server; audio returned as base64 MP3.
- **Inference**: Local — no third-party network call.
- **Externalization**: None (Channel 3).
- **Episode path**: Reply text → POST `/tts/` → local Coqui TTS → base64 audio → Flutter audio player
- **Key files**: `backend/app/routers/tts.py`, `backend/app/schemas.py` (`TTSRequest`, `TTSResponse`)
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 1 (network); 2 others are local
- **Total number of main AI inference workflows found**: 3
- **Top 3 highest-risk workflows or channels**:
    1. **Conversation history sent to OpenRouter (Channel 1 / Workflow 1)**: The chat endpoint includes up to 10 prior turns on every request. Lira's empathetic persona actively elicits personal disclosures (emotional state, health concerns, family situations) — all of which accumulate in the conversation and are repeatedly transmitted to the OpenRouter cloud API, potentially to a third-party model provider.
    2. **Model override per request (Workflow 1)**: The `ChatRequest.model` field allows the caller to override the LLM destination at runtime. A malicious or misconfigured client could route conversation history to an arbitrary model endpoint without server-side validation.
    3. **STT local-only assumption (Workflow 2)**: STT runs locally by default, which is privacy-positive. However, the `LLM_API_BASE_URL` environment variable and the lack of STT-specific network enforcement mean a deployment misconfiguration could silently route audio transcription to a cloud endpoint.
