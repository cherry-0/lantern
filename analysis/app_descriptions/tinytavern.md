# AI Inference Privacy Audit: tinytavern

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (OpenRouter) | `src/services/openrouter.ts` | 33-58 | `sendMessage` | Chat messages, character/system prompts, conversation history, selected model, temperature, and max token settings sent to OpenRouter chat completions. | `fetch(\`${OPENROUTER_BASE_URL}/chat/completions\`, { body: JSON.stringify({ model, messages: allMessages, ... }) })` | High |
| 2 | Network Request (OpenAI) | `src/services/openai.ts` | 39-62 | `sendMessage` | Same chat messages, character/system prompts, and conversation history sent to OpenAI chat completions when OpenAI provider is selected. | `fetch(\`${OPENAI_BASE_URL}/chat/completions\`, { body: JSON.stringify({ model, messages: allMessages, ... }) })` | High |
| 3 | Local or Remote Network Request (Ollama) | `src/services/ollama.ts` | 43-73 | `sendMessage` | Chat messages and optional system prompt sent to configured Ollama host, which may be local or a remote server depending on settings. | `fetch(\`${this.baseUrl}/api/chat\`, { body: JSON.stringify({ model, messages: ollamaMessages, stream: false }) })` | High |
| 4 | Network Request (OpenRouter model list) | `src/services/openrouter.ts` | 12-26 | `getModels` | API key used to fetch available OpenRouter models. | `fetch(\`${OPENROUTER_BASE_URL}/models\`, { Authorization: Bearer ... })` | Medium |
| 5 | Network Request (OpenAI model list) | `src/services/openai.ts` | 84-90 | `testConnection` / model listing | API key used to test/fetch OpenAI models. | `fetch(\`${OPENAI_BASE_URL}/models\`, { Authorization: Bearer ... })` | Medium |
| 6 | Local Storage (AsyncStorage settings) | `src/utils/storage.ts` | 4-44, 95-107 | `saveSettings`, `getSettings`, `saveUserProfile` | Provider settings, API keys, selected model, selected character, system prompt, and user profile name/avatar stored on device. | `AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))` and `AsyncStorage.setItem(USER_PROFILE_KEY, ...)` | High |
| 7 | Local Storage (AsyncStorage chat history) | `src/utils/storage.ts` | 51-80 | `saveMessages`, `getMessages`, `clearMessages` | Per-character chat history including user messages and assistant replies stored on device. | `AsyncStorage.setItem(key, JSON.stringify(messages))` where key is `chat_messages_<characterId>` | High |
| 8 | UI Rendering | `src/screens/ChatScreen.tsx` | 292-302 | `sendMessage` | Assistant reply displayed and persisted in chat UI after provider response. | `setMessages(finalMessages); await saveMessages(finalMessages)` | High |
| 9 | Server / Console Logging | `src/services/openrouter.ts`, `src/services/openai.ts`, `src/screens/ChatScreen.tsx` | 35-37, 60-69, 41-43, 64-73, 304-307 | Various | Messages, system prompt, API response metadata/body, provider errors, and UI errors logged to console. | `console.log('Messages:', messages)`, `console.log('API Response:', data)`, `console.error(...)` | High |
| 10 | Network Request (Stable Diffusion WebUI-compatible image service) | `src/services/imageGeneration.ts` | 38-91 | `generateImage` | Image prompt, dimensions, generation settings, and optional authorization key sent to configured `/txt2img` endpoint. | `fetch(\`${config.baseUrl}/txt2img\`, { body: JSON.stringify(requestData) })` | Medium |

## B. Main AI Inference Workflows

### Workflow 1: Character Chat via OpenRouter
- **Purpose**: Let the user chat with an imported or default character using an OpenRouter-hosted LLM.
- **Input**: User chat message, selected model, selected character card, user profile name, and conversation history.
- **Processing**:
    - `ChatScreen.sendMessage()` creates a user message and appends it to existing history (`src/screens/ChatScreen.tsx:161-172`).
    - If a character is selected, `CharacterCardService.generateSystemPrompt()` builds multiple system messages from the character card fields: description, personality, scenario, message examples, first message, and `{{char}}`/`{{user}}` variable replacement (`src/services/characterCard.ts:688-733`).
    - Character prompts and user messages are concatenated and sent through `OpenRouterService.sendMessage()` (`src/screens/ChatScreen.tsx:189-213`).
- **Inference**: Remote OpenRouter chat-completions model selected in app settings.
- **Externalization**:
    - Character persona, conversation history, and current user message sent to OpenRouter (Channel 1).
    - Full messages and system prompt logged to console (Channel 9).
    - Final user/assistant history stored in AsyncStorage (Channel 7).
- **Episode path**: Text input -> character prompt construction -> OpenRouter `/chat/completions` -> assistant reply -> UI render -> AsyncStorage.
- **Key files**: `src/screens/ChatScreen.tsx`, `src/services/openrouter.ts`, `src/services/characterCard.ts`, `src/utils/storage.ts`
- **Confidence**: High

### Workflow 2: Character Chat via OpenAI
- **Purpose**: Provide the same character chat flow using OpenAI directly instead of OpenRouter.
- **Input**: User message, selected model, optional system prompt, selected character card, user profile, and chat history.
- **Processing**:
    - The OpenAI branch mirrors the OpenRouter branch by building character messages when a character is selected (`src/screens/ChatScreen.tsx:221-255`).
    - `OpenAIService.sendMessage()` sends `model`, `messages`, `temperature`, and `max_tokens` to OpenAI (`src/services/openai.ts:39-62`).
- **Inference**: Remote OpenAI chat-completions model selected in app settings.
- **Externalization**:
    - Full chat and character context sent to OpenAI (Channel 2).
    - Console logs include messages, system prompt, response status, and response body (Channel 9).
    - Chat history stored locally after response (Channel 7).
- **Episode path**: Text input -> character prompt construction -> OpenAI `/chat/completions` -> assistant reply -> UI render -> AsyncStorage.
- **Key files**: `src/screens/ChatScreen.tsx`, `src/services/openai.ts`, `src/services/characterCard.ts`
- **Confidence**: High

### Workflow 3: Character Chat via Ollama
- **Purpose**: Let the user chat with a local or self-hosted Ollama model.
- **Input**: User message, selected model, optional character/system prompt, configured Ollama host/port, and chat history.
- **Processing**:
    - For character chats, character system messages are concatenated into a single system prompt (`src/screens/ChatScreen.tsx:273-285`).
    - `OllamaService.sendMessage()` converts messages to Ollama format and POSTs to `/api/chat` (`src/services/ollama.ts:43-73`).
- **Inference**: Ollama model running at the configured host. This may be local to the device/network or externally hosted.
- **Externalization**:
    - If Ollama host is remote, full chat and character context leave the device/network (Channel 3).
    - Chat history still stored in AsyncStorage after response (Channel 7).
- **Episode path**: Text input -> character prompt construction -> Ollama `/api/chat` -> assistant reply -> UI render -> AsyncStorage.
- **Key files**: `src/screens/ChatScreen.tsx`, `src/services/ollama.ts`, `src/services/characterCard.ts`
- **Confidence**: High

### Workflow 4: AI Image Generation
- **Purpose**: Generate images for gallery/story illustration features using a Stable Diffusion WebUI-compatible service.
- **Input**: User prompt, orientation, configured image generation base URL, optional auth key.
- **Processing**:
    - `generateImage()` appends an LCM LoRA token to the prompt and sets width, height, steps, CFG scale, sampler, and face restoration (`src/services/imageGeneration.ts:38-53`).
    - The request is POSTed to `${baseUrl}/txt2img` with optional authorization (`src/services/imageGeneration.ts:55-76`).
- **Inference**: Configured Stable Diffusion WebUI-compatible image backend.
- **Externalization**:
    - Prompt and generation parameters sent to the configured image service (Channel 10).
    - Generated base64 image returned to the app and can be stored in gallery flows.
- **Episode path**: Image prompt -> configured `/txt2img` endpoint -> base64 image -> app gallery/story UI.
- **Key files**: `src/services/imageGeneration.ts`, `src/services/imageStorage.ts`, `src/screens/ImageGenerationScreen.tsx`
- **Confidence**: Medium

## Final Summary
- **Total number of distinct externalization sites found**: 10
- **Total number of main AI inference workflows found**: 4
- **Top 3 highest-risk workflows or channels**:
    1. **Character chat provider calls (Channels 1-3)**: TinyTavern sends the full character persona, conversation history, user profile name substitutions, and the latest user message to the selected provider. OpenRouter and OpenAI are third-party cloud APIs; Ollama may also be remote if configured that way.
    2. **Local persistence of settings and chat history (Channels 6-7)**: API keys, selected model, user profile, system prompt, selected character, and per-character chat histories are stored in AsyncStorage on the device.
    3. **Verbose console logging (Channel 9)**: The OpenRouter/OpenAI service wrappers log message arrays, system prompts, response status, and full API responses, which can expose private chat content in development logs, device logs, or debugging sessions.
