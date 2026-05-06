# AI Inference Privacy Audit: llm-vtuber

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM) | `src/open_llm_vtuber/agent/stateless_llm/openai_compatible_llm.py` | 105 | `chat_completion` | Full conversation history, system prompts, and tool definitions sent to OpenAI-compatible APIs. | `await self.client.chat.completions.create(...)` | High |
| 2 | Network Request (Cloud STT) | `src/open_llm_vtuber/asr/openai_whisper_asr.py` | 35 | `transcribe` | Raw user audio bytes sent to OpenAI Whisper API. | `await self.client.audio.transcriptions.create(...)` | High |
| 3 | Network Request (Cloud TTS) | `src/open_llm_vtuber/tts/fish_api_tts.py` | 49 | `async_generate_audio` | Character-generated text sent to Fish TTS API. | `self.session.tts(TTSRequest(...))` | High |
| 4 | WebSocket | `src/open_llm_vtuber/routes.py` | 201 | `tts_endpoint` | Real-time generated audio bytes and text response chunks streamed to the client. | `@router.websocket("/tts-ws")` | High |
| 5 | Storage Write (Logs) | `src/open_llm_vtuber/chat_history_manager.py` | - | `save_history` | Persistent storage of chat logs, including user inputs and model outputs. | Standard loguru/json logging usage. | High |
| 6 | UI Rendering | `frontend/` | - | - | Live2D model animations, subtitles, and real-time audio playback based on model output. | Integration with PIXI.js and Live2D SDK. | High |

## B. Main AI Inference Workflows

### Workflow 1: Real-time Multimodal VTuber Interaction
- **Purpose**: Enable users to talk to an AI-powered VTuber with voice, text, and visual feedback.
- **Input**: User voice (via WebSocket) or text input.
- **Processing**: VAD (Silero) detects speech; STT (Whisper/Azure) transcribes audio; Chat history is retrieved and formatted into a prompt.
- **Inference**: LLM (OpenAI/Claude/Ollama) generates a response chunk-by-chunk.
- **Externalization**: 
    - Transcription sent to UI as subtitles.
    - Text response sent to TTS (Piper/Fish/Cartesia) to generate audio.
    - Audio and Live2D animation parameters streamed to frontend via WebSocket (`routes.py:201`).
- **Episode path**: User Audio -> VAD -> STT API -> LLM API -> TTS API -> WebSocket -> UI (Live2D + Audio)
- **Key files**: `websocket_handler.py`, `asr_factory.py`, `agent_factory.py`, `tts_factory.py`, `routes.py`
- **Confidence**: High

### Workflow 2: Agentic Tool-Calling (MCPP)
- **Purpose**: Allow the VTuber to perform external actions (e.g., search, system control).
- **Input**: User prompt requiring a tool (e.g., "What's the weather?").
- **Processing**: LLM detects a tool call requirement based on provided function schemas.
- **Inference**: LLM generates a tool call object.
- **Externalization**: 
    - Tool call results (which may involve further network requests) are fed back into the LLM.
    - Final status reported to the user via voice/UI.
- **Episode path**: User Prompt -> LLM API -> Tool Execution -> LLM API -> VTuber Response
- **Key files**: `openai_compatible_llm.py`, `agent/agents/`, `mcpp/`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 6
- **Total number of main AI inference workflows found**: 2
- **Top 3 highest-risk workflows or channels**:
    1. **Cloud LLM/STT/TTS API Calls (Workflow 1)**: Transmits raw biometric data (voice) and personal conversations to multiple third-party AI providers (OpenAI, Microsoft, Cartesia, etc.).
    2. **WebSocket Audio Streaming (Channel 4)**: Real-time externalization of inferred "personality" and voice, which could be intercepted if the connection is not properly secured (WSS).
    3. **Persistent Chat History (Channel 5)**: Stores the entire interaction history locally; if the host machine is compromised, all private user-AI conversations are exposed.
