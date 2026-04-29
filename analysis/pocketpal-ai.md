# AI Inference Privacy Audit: pocketpal-ai

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | On-Device Inference (no network) | `src/api/completionEngines.ts` | 15–54 | `LocalCompletionEngine.completion` | User message + full chat history + system prompt fed to a local `LlamaContext` (llama.rn / llama.cpp) running entirely on-device. No bytes leave the device for inference. | `class LocalCompletionEngine implements CompletionEngine { constructor(private context: LlamaContext) {} … this.context.completion(params, …)` | High |
| 2 | Local File Read (model load) | `src/store/ModelStore.ts` | 10, 1546–1625 | `ModelStore.initContext` → `initLlama` | Loads a local GGUF model file from device storage into a `LlamaContext`, then wraps it in `LocalCompletionEngine`. Model bytes are read from local FS only. | `import {ContextParams, LlamaContext, initLlama} from 'llama.rn'; … const ctx = await initLlama(…); this.engine = new LocalCompletionEngine(ctx);` | High |
| 3 | Local Storage (WatermelonDB / SQLite) | `src/repositories/ChatSessionRepository.ts` | 245–318, 414–472 | `createSession`, `addMessageToSession` | Every user prompt and assistant response is persisted as a `messages` row (`author`, `text`, `metadata`, `createdAt`, `position`) inside a SQLite-backed WatermelonDB on the device. Sessions, completion settings, and active pal IDs are also written. | `database.collections.get('messages').create((record:any) => { record.text = msg.text; record.metadata = JSON.stringify(metadata); … })` against `appSchema(version: 6)` defined in `src/database/schema.ts` (tables `chat_sessions`, `messages`, `completion_settings`, `global_settings`). | High |
| 4 | Local File Write (legacy migration / export) | `src/repositories/ChatSessionRepository.ts` | 28–168, 401–412 | `checkAndMigrateFromJSON`, `exportSessions` | One-time read of legacy `session-metadata.json` / `global-completion-settings.json` from `RNFS.DocumentDirectoryPath` and a migration flag file are written to disk; `exportSessions` writes chat sessions to local files via `exportChatSession`. | `RNFS.readFile(oldDataPath); … RNFS.writeFile(migrationFlagPath, 'true');` and `const {exportChatSession} = await import('../utils/exportUtils');` | High |
| 5 | UI Rendering (Streaming Chat Bubbles) | `src/hooks/useChatSession.ts`, `src/screens/ChatScreen/ChatScreen.tsx` | 290–334 (hook), 13–170 (screen) | `handleSendPress` → `engine.completion(callback)`, `<ChatView … messages=… />` | Streaming token deltas (`data.token`, accumulated `content`, `reasoning_content`) are pushed back to the React renderer chat bubble in real time and the full conversation is displayed via `chatSessionStore.currentSessionMessages`. | `engine.completion(cleanCompletionParams, data => { … updateMessage(messageInfo.id, { text: data.content, … }) })`; `<ChatView messages={chatSessionStore.currentSessionMessages} renderBubble={renderBubble} />` | High |
| 6 | Console Logging (debug) | `src/hooks/useChatSession.ts` | 387–394, 439, 489 | `handleSendPress` (catch + result branches) | Full completion result objects, raw `result`, and error stacks are logged to JS console (`console.log('Completion result:', …)`, `console.log('result', result)`, `console.error('Completion error:', error)`); on debug builds these can include user-visible response content. | `console.log('Completion result:', { … }); console.log('result', result); console.error('Completion error:', error);` | Medium |
| 7 | Network Request (HuggingFace — model download, user-triggered, non-inference) | `src/services/downloads/DownloadManager.ts`, `src/api/hf.ts` | 277–332 (DM), 27–222 (hf.ts) | `RNFS.downloadFile`, `fetchModels`, `fetchModelFilesDetails`, `fetchModelInfo`, `fetchGGUFSpecs` | Outbound HTTPS to `huggingface.co` to (a) list/search GGUF models, (b) read repo trees and GGUF specs, (c) download the model file to local storage. Sends optional `Authorization: Bearer <HF token>` header for gated models. No chat content sent. One-time setup flow, not per-inference. | `axios.get(nextPageUrl || urls.modelsList(), {params: {search, author, …}, headers})` and `RNFS.downloadFile({fromUrl: model.downloadUrl!, toFile: destinationPath, …})` | High |
| 8 | Network Request (Firebase Functions — feedback / benchmark, user-triggered) | `src/api/feedback.ts`, `src/api/benchmark.ts` | 96–112, 178–195, 273–290 (feedback); 60–68 (benchmark) | `submitContentReport`, `submitFeedback`, `submitModelLoadErrorReport`, `submitBenchmark` | Posts feedback text, content reports, model-load error reports (incl. error message, optional model id/url/size, optional device model/OS/memory/CPU), or benchmark + device info to a Firebase Functions endpoint, gated by Firebase App Check token. Triggered only when the user explicitly taps "Submit". | `axios.post(urls.feedbackSubmit(), { ...feedbackData, appFeedbackId, appVersion, appBuild }, { headers: {'X-Firebase-AppCheck': appCheckToken} })` and `axios.post(urls.benchmarkSubmit(), {deviceInfo, benchmarkResult}, {headers: {'X-Firebase-AppCheck': …}})` | High |
| 9 | Network Request (Supabase PalsHub — optional cloud sync, user-triggered) | `src/services/palshub/supabase.ts`, `AuthService.ts`, `PalsHubApiService.ts`, `SyncService.ts` | 1–67 (supabase), 89–461 (auth), 161+ (api) | `supabase.auth.signIn*`, `supabase.from('profiles').upsert`, `fetch(url, {…})` with bearer auth | Optional Supabase-backed "PalsHub" feature: signs the user in (email/password or OAuth idToken), upserts user profile, fetches/syncs Pal definitions (system prompts, model settings, categories, tags) to the local `cached_pals` / `user_library` / `sync_status` tables. No chat messages or user prompts are sent — only Pal metadata + auth identity. Disabled when `isSupabaseConfigured` is false. | `export const supabase = isSupabaseConfigured ? createClient(…) : null;` `await supabase.auth.signInWithIdToken({…})`; `await supabase!.from('profiles').upsert({…})`; `fetch(url, {headers: await getAuthHeaders()})` | High |
| 10 | Network Request (OpenAI-compatible remote LLM — opt-in alternate engine) | `src/api/openai.ts`, `src/api/completionEngines.ts` | 60–96 (engine), 208–528 (stream), 79–198 (models) | `OpenAICompletionEngine.completion`, `streamChatCompletion`, `fetchModelsWithHeaders` | If the user explicitly configures a remote OpenAI-compatible server (e.g. self-hosted llama.cpp, LM Studio, Ollama, or OpenAI itself) and selects it as the active "model", chat messages + system prompt are POSTed to `<serverUrl>/v1/chat/completions` with `stream: true` and optional `Authorization: Bearer <apiKey>`. Default flow uses local llama.rn instead. | `xhr.open('POST', url); xhr.setRequestHeader('Authorization', 'Bearer …'); xhr.send(JSON.stringify({model, messages, stream:true, …}))` and `class OpenAICompletionEngine implements CompletionEngine` | High |

## B. Main AI Inference Workflows

### Workflow 1: On-Device Chat Completion (default — `LocalCompletionEngine`)
- **Purpose**: Generate an assistant reply to the user's chat message using a small language model that runs entirely on the phone.
- **Input**: User's text message (optionally with image URIs for multimodal models), prior session messages from WatermelonDB, system prompt resolved from the active Pal / session settings, and `completionSettings` (temperature, top_p, n_predict, stop, etc.).
- **Processing**:
  - `useChatSession.handleSendPress` resolves system messages (`resolveSystemMessages`), converts prior messages with `convertToChatMessages`, optionally strips thinking parts, and builds an OpenAI-style `messages` array.
  - `prepareCompletion` writes an empty assistant message row to WatermelonDB, then calls `engine.completion(cleanCompletionParams, callback)`.
  - `LocalCompletionEngine` delegates 1:1 to a `LlamaContext` that was created earlier by `ModelStore.initContext` via `initLlama(...)` against a local GGUF file on disk.
  - llama.rn (native binding for llama.cpp) runs token-by-token inference on-device; each token fires the callback, which updates the assistant message text in the MobX store and persists incremental updates to the `messages` table.
- **Inference**: Local GGUF model (Qwen / Phi / Gemma / Danube etc.) — fully on-device, no network call per chat message.
- **Externalization**:
  - Model file read from local storage into native context (Channel 2).
  - User message + assistant response streamed token-by-token to ChatScreen bubbles (Channel 5).
  - Both the user message row and assistant message row (with `text`, `metadata`, `position`, `createdAt`) written to the local SQLite-backed WatermelonDB (Channel 3).
  - Debug-build console logs may dump `result` and error stacks (Channel 6).
  - **No network externalization** in the default configuration.
- **Episode path**: User text in chat input → `prepareCompletion` builds messages → `LocalCompletionEngine.completion` → `LlamaContext.completion` (llama.rn → llama.cpp) → streamed token callback → `chatSessionStore.updateMessage` + WatermelonDB `messages.update` → ChatScreen bubble re-renders.
- **Key files**: `src/api/completionEngines.ts` (`LocalCompletionEngine`, lines 15–54); `src/store/ModelStore.ts` (`initContext`/`initLlama`, lines 1546–1625); `src/hooks/useChatSession.ts` (`handleSendPress`, lines 191–500); `src/repositories/ChatSessionRepository.ts` (`addMessageToSession`, lines 414–472); `src/database/schema.ts` (lines 17–28).
- **Confidence**: High

### Workflow 2: Structured-Output Inference (utility — JSON tasks like Pal-prompt generation)
- **Purpose**: Run a one-shot local inference that must return JSON conforming to a schema (e.g. for AI-generated system prompts when creating a "Pal").
- **Input**: Prompt text + JSON schema, dispatched via `useStructuredOutput`.
- **Processing**:
  - Calls `modelStore.context.completion({...})` directly on the loaded `LlamaContext`, then parses the returned text as JSON.
  - Same on-device llama.cpp path as Workflow 1, just without streaming to the chat UI.
- **Inference**: Local GGUF model.
- **Externalization**:
  - Model bytes read locally (Channel 2).
  - Generated structured output is consumed by the calling screen (Pal creation form) — the resulting Pal definition is written to the `local_pals` table (Channel 3, schema lines 99–130) and, if PalsHub sync is enabled by the user, may later be uploaded to Supabase (Channel 9).
  - **No network externalization** during inference itself.
- **Episode path**: Prompt + schema → `modelStore.context.completion(...)` → JSON parse → form fields → `local_pals` insert (WatermelonDB).
- **Key files**: `src/hooks/useStructuredOutput.ts` (lines 47–65); `src/store/ModelStore.ts` (context held at line 152).
- **Confidence**: High

### Workflow 3: Remote Chat Completion (opt-in `OpenAICompletionEngine`)
- **Purpose**: When the user has configured a remote OpenAI-compatible server and selected it as the active model, route chat completions to that endpoint instead of the local llama.rn context.
- **Input**: Same OpenAI-style `messages` array as Workflow 1, plus user-configured `serverUrl`, `modelId`, and optional `apiKey`.
- **Processing**:
  - `OpenAICompletionEngine.completion` calls `streamChatCompletion`, which opens an `XMLHttpRequest` POST to `${serverUrl}/v1/chat/completions` with `stream: true`.
  - SSE events are parsed by `SSEParser`; deltas update the assistant message via the same `useChatSession` callback path used for local inference.
- **Inference**: Remote (OpenAI / LM Studio / Ollama / self-hosted llama.cpp etc.) — fully off-device.
- **Externalization**:
  - **Network**: full system + chat message history + user input sent to `<serverUrl>` with optional bearer API key (Channel 10).
  - Streaming response rendered to ChatScreen bubbles (Channel 5).
  - Final assistant message row written to WatermelonDB (Channel 3).
- **Episode path**: User text → `OpenAICompletionEngine.completion` → `streamChatCompletion` → POST `/v1/chat/completions` → SSE deltas → `chatSessionStore.updateMessage` → ChatScreen.
- **Key files**: `src/api/completionEngines.ts` (`OpenAICompletionEngine`, lines 60–96); `src/api/openai.ts` (`streamChatCompletion`, lines 208–528); `src/store/ModelStore.ts` (`Set a remote model as the active model`, ~line 1890).
- **Confidence**: High (only triggered if user explicitly adds and selects a remote server)

## Final Summary
- **Total number of distinct externalization sites found**: 10
- **Total number of main AI inference workflows found**: 3 (one default on-device chat, one on-device structured-output utility, one opt-in remote chat)
- **Top 3 highest-risk workflows or channels**:
    1. **Local Chat Persistence — WatermelonDB / SQLite (Channel 3, Workflow 1)**: Every user prompt, assistant reply, attached image URI (in `metadata.imageUris`), session title, active Pal ID, and completion settings are persisted indefinitely in the on-device SQLite database. While the data never leaves the device automatically, it is the richest pool of personal content in the app — anyone with device access (forensics, malware, an unlocked phone, or a future opt-in PalsHub export) can read full conversational history, including any sensitive disclosures the user made to what they believed was a fully private model.
    2. **Opt-in Remote OpenAI-compatible Engine (Channel 10, Workflow 3)**: Although the app's headline promise is "your conversations never leave your phone," users can configure an arbitrary remote `serverUrl` and bearer API key, after which complete chat history (system prompt + all prior turns + new user input) is POSTed to that server every message. The UX still routes through the same chat surface, so a user who switched to a remote engine days earlier may not realize subsequent prompts are leaving the device.
    3. **HuggingFace Model Download with Auth Token (Channel 7) + Firebase Feedback/Benchmark Submissions (Channel 8)**: The HF download flow attaches the user's personal HF access token to outbound requests (granting access to gated repos and identifying the account); the feedback / model-load-error / benchmark endpoints can carry detailed device fingerprinting (`model`, `systemName`, `systemVersion`, `totalMemory`, `cpuArch`, `isEmulator`), the offending model id/url/size, free-text user feedback, and a persistent UUID `appFeedbackId` from `FeedbackStore`. None are per-inference, but together they are the only paths that reliably leave the device, and the persistent feedback UUID enables linkage of multiple submissions from the same install.
