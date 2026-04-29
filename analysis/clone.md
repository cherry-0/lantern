# AI Inference Privacy Audit: clone


## clone/

A desktop Electron app that records digital activities and enables chat against a personal knowledge base.

### Architecture

- **`frontend/`** — Electron app (main + renderer processes)
  - `src/main.ts` — Electron main process; manages IPC, Ollama/OpenAI integration, model downloads
  - `src/renderer.tsx` — React entry point
  - `src/recording/` — Screen/audio capture providers
  - `src/llm/` — LLM abstraction layer (Ollama local models, OpenAI, embedding manager)
  - `src/services/` — API clients for auth, chat, collection, memory, embeddings
  - `src/components/` — React UI components (each paired with a `.test.tsx`)
  - `src/embedding/` — ONNX-based local embedding worker (runs in a separate worker thread via `embedding-worker.ts`)
- **`server/`** — Django REST API (MySQL)
  - Apps: `user/`, `chat/`, `collection/`
  - Settings split: `config/settings/base.py`, `local.py`, `test.py`
- **`vectordb/`** — Separate Django service with pluggable VectorDB backends (`vectordb/vectordb/`: `naive_vectordb.py`, `milvus_vectordb.py`)

### Commands

```bash
# Frontend (Electron)
cd clone/frontend
npm install
npm run dev            # Start dev app
npm run lint           # ESLint
npm run test:unit      # Vitest unit tests
npm run test:e2e       # Playwright e2e tests
npm run test:all       # Both unit + e2e

# Run a single Vitest test file
npm run test -- src/components/ChatInterface.test.tsx

# Server (Django)
cd clone/server
pip install -r requirements.txt
python manage.py migrate --settings=config.settings.local
python manage.py runserver --settings=config.settings.local

# Run tests
pytest                            # All tests with coverage
pytest user/tests/                # Specific app
pytest -m unit                    # Only unit tests
pytest -m integration             # Only integration tests
pytest -n auto                    # Parallel execution
pytest user/tests/test_models.py::TestUserModel::test_create_user_with_email_and_username
```



## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (OpenAI) | `frontend/src/llm/openai-manager.ts` | 114, 133 | `chat`, `streamChat` | User messages, conversation history, and base64 video frames/images. | `this.client.chat.completions.create({model: "gpt-5-mini-2025-08-07", ...})` | High |
| 2 | Local Network Request (Ollama) | `frontend/src/llm/ollama-manager.ts` | 193, 274 | `chat`, `streamChat` | User messages, conversation history, and base64 video frames/images. | `this.ollama.chat({model: "gemma3:4b", ...})` | High |
| 3 | Storage Write (Recording) | `frontend/src/main.ts` | 1083 | `rec:save-file` | Screen/window recordings (WebM format). | `fsp.writeFile(filePath, data)` | High |
| 4 | Network Request (AWS S3 / HF) | `frontend/src/main.ts` | 11, 41-118 | `isEmbeddingModelDownloaded`, `model:start-download` | Model file downloads (ONNX, config, tokens). | `downloadFile(mainWindow, {downloadUrl: file.url, ...})` | High |
| 5 | Network Request (Django Server) | `frontend/src/services/chat.ts` | 23-102 | `fetchSessions`, `createSession`, `sendMessage` | **Encrypted** chat history (titles, messages). | `apiRequestWithAuth<ChatSession>('/api/chat/sessions/create/', ...)` | High |
| 6 | UI Rendering | `frontend/src/services/llm.ts` | 74 | `streamMessage` | Real-time AI response chunks. | `onChunk(chunk.chunk)` | High |
| 7 | Logging | `frontend/src/llm/ollama-manager.ts` | 199, 200, 264, 265 | `chat`, `streamChat` | Image counts and base64 image sizes. | `console.log('[Ollama] Processing message with ...')` | High |

## B. Main AI Inference Workflows

### Workflow 1: Multimodal Chat (OpenAI)
- **Purpose**: Provide a real-time, vision-enabled chat assistant.
- **Input**: User text messages and optionally video frames (extracted from recorded videos at 1 fps).
- **Processing**: Video frames are extracted, converted to base64; prompt is built with system prompt and history.
- **Inference**: Remote call to OpenAI `gpt-5-mini-2025-08-07` via `OpenAI` client in the main process.
- **Externalization**: 
    - Full multimodal payload sent to OpenAI API (`openai-manager.ts:114, 133`).
    - Response chunks sent to renderer and displayed in UI (`llm.ts:74`).
- **Episode path**: User Input / Video Capture -> Frame Extraction -> OpenAI API -> Streaming Response -> UI
- **Key files**: `openai-manager.ts`, `main.ts`, `llm.ts`
- **Confidence**: High

### Workflow 2: Multimodal Chat (Ollama - Local)
- **Purpose**: Provide a real-time, vision-enabled chat assistant running locally.
- **Input**: User text and video frames.
- **Processing**: Same as Workflow 1.
- **Inference**: Local call to `Ollama` running `gemma3:4b` on the same machine.
- **Externalization**: 
    - Full multimodal payload sent to local Ollama server (`ollama-manager.ts:193, 274`).
    - Response chunks sent to renderer and displayed in UI (`llm.ts:74`).
- **Episode path**: User Input / Video Capture -> Frame Extraction -> Local Ollama API -> Streaming Response -> UI
- **Key files**: `ollama-manager.ts`, `main.ts`, `llm.ts`
- **Confidence**: High

### Workflow 3: Query & Context Embedding
- **Purpose**: Convert text to vectors for semantic search/RAG (local).
- **Input**: User query or session context.
- **Processing**: Tokenization using CLIP/DRAGON tokenizer.
- **Inference**: Local ONNX model inference using `EmbeddingManager`.
- **Externalization**: 
    - Embeddings used for local vector search.
- **Episode path**: Text Input -> EmbeddingManager (ONNX) -> Local Vector Store
- **Key files**: `embedding.ts`, `main.ts`
- **Confidence**: High

### Workflow 4: Conversation Title Generation
- **Purpose**: Automatically generate a title for a chat session.
- **Input**: First user message and initial assistant response.
- **Processing**: Prompt construction for a concise title (max 5 words).
- **Inference**: LLM call (OpenAI or Ollama) to generate the title.
- **Externalization**: 
    - Title sent to backend server (encrypted) for persistent storage.
    - Title displayed in the session list in UI.
- **Episode path**: Chat Start -> Title Generation Prompt -> LLM API -> Encrypted Backend Storage -> UI
- **Key files**: `llm.ts`, `chat.ts`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 7
- **Total number of main AI inference workflows found**: 4
- **Top 3 highest-risk workflows or channels**:
    1. **OpenAI API Call (Workflow 1)**: Sends potentially sensitive user text and **video frame captures** of their screen to OpenAI cloud.
    2. **Local Recording Storage (Channel 3)**: Saves raw recordings of the user's screen/windows to their local video folder; if poorly managed, these could be accessed by other local processes.
    3. **Cloud Backend Storage (Channel 5)**: While the content is encrypted before upload, any vulnerability in the encryption/decryption logic would expose the entire chat history (messages, titles) on the remote server.
