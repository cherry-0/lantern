# AI Inference Privacy Audit: oxproxion

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM — streaming) | `app/src/main/java/io/github/stardomains3/oxproxion/ChatViewModel.kt` | 3157–3172 | streaming OpenRouter call inside `sendMessage` flow | Full chat history (`messages`) — including the just-typed user prompt, the active system message, and any prior turns — POSTed as JSON to `https://openrouter.ai/api/v1/chat/completions` with `Authorization: Bearer <openrouter_api_key>`. Response is consumed as a Server-Sent-Event stream (`ByteReadChannel`) so each delta chunk is also returned over the network. | `httpClient.preparePost(activeChatUrl) { header("Authorization", "Bearer $activeChatApiKey"); header("HTTP-Referer", "https://github.com/stardomains3/oxproxion/"); header("X-Title", "oxproxion"); contentType(ContentType.Application.Json); setBody(chatRequest) }.execute { httpResponse -> ... }` with `activeChatUrl = "https://openrouter.ai/api/v1/chat/completions"` (line 382, 718, 837). | High |
| 2 | Network Request (Cloud LLM — non-streaming) | `app/src/main/java/io/github/stardomains3/oxproxion/ChatViewModel.kt` | 3681–3708 | non-streaming OpenRouter completion | Same `ChatRequest` payload (system + full message history + tool definitions + reasoning options + optional `image_url` parts) POSTed once; `ChatResponse` body returned synchronously. Used when streaming is disabled in Settings or when the model is invoked for tool/image-generation completion. | `val response = httpClient.post(activeChatUrl) { header("Authorization", "Bearer $activeChatApiKey"); header("HTTP-Referer", "https://github.com/stardomains3/oxproxion/"); header("X-Title", "oxproxion"); contentType(ContentType.Application.Json); setBody(chatRequest) }` then `response.body<ChatResponse>()`. | High |
| 3 | Network Request (Multimodal image upload) | `app/src/main/java/io/github/stardomains3/oxproxion/ChatViewModel.kt` | ~700–714, 4598, 5550–5567 | image attachment path through `attachmentButton` / camera | When a user attaches a gallery photo or camera capture, the URI is read, base64-encoded, and embedded as an `image_url` content part in the same OpenRouter POST. The full image bytes (faces, location backdrops, EXIF-implied context) are sent to `openrouter.ai`. | `userMessage = userMessage.copy(imageUri = uriStr)` (line 714); content parts of `type=image_url` are constructed from the URI and shipped in the chat body. `import java.util.Base64` (line 102) and `android.util.Base64.encodeToString(baos.toByteArray(), NO_WRAP)` (line 5567) confirm in-app base64 encoding for the wire payload. | High |
| 4 | Network Request (Cloud LLM — auxiliary "title suggestion") | `app/src/main/java/io/github/stardomains3/oxproxion/LlmService.kt` | 35–98 | `getSuggestedChatTitle` | Concatenated chat content (`"Chat Contents: \`\`\`$chatContent\`\`\`"`) — i.e. a verbatim copy of the user's conversation — is POSTed to OpenRouter (or the configured LAN endpoint) to ask the LLM to invent a 1–8-word title that will then be persisted to Room as `ChatSession.title`. | `val prompt = "Respond only with a 1 to 8 word title ... Chat Contents: \`\`\`$chatContent\`\`\`"`; `activeClient.post(endpoint) { header("Authorization", authHeader); ... setBody(chatRequest) }`. Endpoint is hard-coded for OpenRouter at `ChatViewModel.kt:4043` (`"https://openrouter.ai/api/v1/chat/completions"`). | High |
| 5 | Network Request (Account telemetry / credit lookup) | `app/src/main/java/io/github/stardomains3/oxproxion/LlmService.kt` | 111–133 | `getRemainingCredits` | OpenRouter API key is sent in the `Authorization` header to `GET https://openrouter.ai/api/v1/credits`. No user content goes out, but the key + device IP do, and the request reveals app/account usage to OpenRouter. | `httpClient.get("https://openrouter.ai/api/v1/credits") { header("Authorization", "Bearer $apiKey") }`. | High |
| 6 | Network Request (LAN fallback) | `app/src/main/java/io/github/stardomains3/oxproxion/ChatViewModel.kt` | 2832–2837, 3486–3487 | streaming/non-streaming POST to user-configured LAN endpoint (Ollama / LM Studio / llama.cpp / MLX LM / Hermes Agent) | Same `ChatRequest` body POSTed to a LAN URL stored in SharedPreferences (`KEY_LAN_ENDPOINT`). Traffic stays on the user's LAN, but full prompt + system message + image content parts still leave the device over plaintext HTTP unless the user has TLS configured locally. | `lanHttpClient.preparePost(activeChatUrl) { header("Authorization", "Bearer $activeChatApiKey"); contentType(ContentType.Application.Json); setBody(chatRequest) }.execute { ... }` and `lanHttpClient.post(activeChatUrl) { ... }`. | High |
| 7 | Local Storage (Room / SQLite) — chat persistence | `app/src/main/java/io/github/stardomains3/oxproxion/ChatRepository.kt`, `ChatDao.kt`, `ChatViewModel.kt` | repo: 17–19; dao: 22–61; vm: 521, 546 | `ChatRepository.insertSessionAndMessages` | Every saved session writes a `ChatSession(title, modelUsed, timestamp)` row plus N `ChatMessage(sessionId, role, content)` rows into the unencrypted Room DB `chat_database`. `content` is the JSON-serialized message (user text, assistant reply, embedded `image_url` parts are stripped only when `hasImagesInChat()` is true; otherwise stored verbatim). | `Room.databaseBuilder(context.applicationContext, AppDatabase::class.java, "chat_database").build()` (`AppDatabase.kt:19–23`); `repository.insertSessionAndMessages(session, chatMessages)` (`ChatViewModel.kt:521, 546`); `content = json.encodeToString(JsonElement.serializer(), it.content)` (lines 518, 543). README confirms: "Chats are stored unencrypted on device only." | High |
| 8 | Local Storage (SharedPreferences — encrypted API key) | `app/src/main/java/io/github/stardomains3/oxproxion/SharedPreferencesHelper.kt` | 509–528, 532–548, 550–588 | `saveApiKey` / `getApiKeyFromPrefs` | OpenRouter API key (and Brave / LAN keys) are encrypted with an AES-256-GCM key from the AndroidKeyStore and stored as Base64 in SharedPreferences file `ApiKeysPrefsStore`. Plaintext key is decrypted on demand on the same device. Confidentiality depends on AndroidKeyStore + lockscreen; no biometric or user-authenticated key binding is used. | `apiKeysPrefs.edit { putString("${alias}_encrypted", encryptedKeyString); putString("${alias}_iv", ivString) }`; key alias `"openrouter_api_key"` is the canonical alias (`ChatViewModel.kt:439, 718, 837, 4002, 4013, 4404, 4519`). Set from UI in `SaveApiDialogFragment.kt:42` (`sharedPreferencesHelper.saveApiKey("openrouter_api_key", apiKey)`). | High |
| 9 | Local Storage (SharedPreferences — settings + history fragments) | `app/src/main/java/io/github/stardomains3/oxproxion/SharedPreferencesHelper.kt` | 385–391, 736–755, 662–717 | `saveLastAiResponseForChannel`, `setLanEndpoint`, `saveSelectedSystemMessage`, etc. | The last AI response per notification channel is cached in plaintext SharedPreferences (`KEY_LAST_AI_RESPONSE_CHANNEL_*`); LAN endpoint URL, selected/custom system messages (which often encode user persona instructions), and presets are also persisted in clear. | `mainPrefs.edit { putString("${KEY_LAST_AI_RESPONSE_CHANNEL}${channelId}", responseText) }` (line 386); `mainPrefs.edit { putString(KEY_LAN_ENDPOINT, url) }` (line 740); `mainPrefs.edit { putString(KEY_SELECTED_SYSTEM_MESSAGE, jsonString) }` (line 664). | High |
| 10 | Local Storage (Foreground notification cache) | `app/src/main/java/io/github/stardomains3/oxproxion/ChatViewModel.kt` | 3699–3704 | error / TTS notification path while `ForegroundService.isRunningForeground` | When the foreground service is active, the latest assistant response (or formatted error message) is written into SharedPreferences via `saveLastAiResponseForChannel(2, openRouterError)` so the system notification can read it back. Means assistant text leaves the in-memory chat surface and lands in another on-device store. | `sharedPreferencesHelper.saveLastAiResponseForChannel(2, openRouterError)` plus `ForegroundService.updateNotificationStatus(displayName, "Error!")`. | Medium |
| 11 | Local Storage (PDF export) | `app/src/main/java/io/github/stardomains3/oxproxion/PdfGenerator.kt`, `ChatViewModel.kt` | pdf: 532–650, 4598–4645; vm: 4598+ | "Make PDF" button / "PDF this response" tool | `generatePdf(messages, modelName)` lays out the entire conversation (user prompts + AI responses + any embedded images base64-decoded into bitmaps) and writes the file via `FileOutputStream(file).use { document.writeTo(it) }` to app-visible storage. The exported PDF is then reachable via the Android share sheet. | `FileOutputStream(file).use { document.writeTo(it) }` (`PdfGenerator.kt:496, 643`); `embedGeneratedImage` writes generated assistant images into the document. | High |
| 12 | UI Rendering (chatRecyclerView) | `app/src/main/res/layout/fragment_chat.xml`, `app/src/main/res/layout/item_message_ai.xml`, `item_message_user.xml`; `ChatFragment.kt` | fragment: ~145, 172–176, 435–441; vm: 3216–3275 | `ChatAdapter` `AssistantViewHolder`, real-time delta render | Every streamed delta (`accumulatedResponse`) is pushed back to the LiveData `_chatMessages`, which the adapter binds into `messageTextView` inside `chatRecyclerView`. Anyone with line-of-sight to the device sees the model's output character-by-character — including any leaked PII the model echoes back. | `chatRecyclerView = view.findViewById(R.id.chatRecyclerView)` (`ChatFragment.kt:435`); `messageTextView` is the AI bubble text view targeted by the verify adapter (`verify/backend/adapters/oxproxion.py:163, 173–179`). | High |
| 13 | UI Rendering (Conversation Mode TTS) | `app/src/main/java/io/github/stardomains3/oxproxion/ChatFragment.kt` | 255, 901, 504–537 | conversation-mode auto-speech | When `getConversationModeEnabled()` is true, the assistant's textual response is fed into Android TTS and played out loud; the synthesized audio file can also be written to `MediaStore.Downloads/oxproxion/*.wav`. Anyone in earshot or with access to Downloads can recover the model's reply. | `if (sharedPreferencesHelper.getConversationModeEnabled()) { ... }` and `MediaStore` insert path with `MediaColumns.RELATIVE_PATH = "${Environment.DIRECTORY_DOWNLOADS}/oxproxion"`. | Medium |
| 14 | Clipboard (avatar press to copy) | `app/src/main/java/io/github/stardomains3/oxproxion/ChatFragment.kt` | imports line 9 (`ClipData`) | "Press any avatar to copy the corresponding response to the clipboard" (README §Effortless Copying) | The full assistant response (or its Markdown source on long-press) is placed on the system clipboard, which is readable by any app with clipboard access on older Android versions and exposed by ClipboardManager events on newer ones. | `import android.content.ClipData` (`ChatFragment.kt:9`); README line 56: "Press any avatar to copy ... to the clipboard. Long-press the response avatar to copy in Markdown." | Medium |
| 15 | Logging (logcat) | `app/src/main/java/io/github/stardomains3/oxproxion/ChatViewModel.kt`, `PdfGenerator.kt` | vm: 1826, 1843, 2118, 2432, 2536; pdf: scattered (mostly commented) | `Log.e("ToolCall", ...)`, `Log.e("EditFile", ...)`, `Log.e("CopyFile", ...)` | Live `Log.e` calls in tool-call execution paths emit error metadata (file paths the user asked to read/edit/copy, exception messages) to logcat, which is readable by the user via ADB and persisted by some Android OEMs to crash-report channels. The vast majority of `Log.*` calls in the codebase are commented out, but five active error-logs remain in `ChatViewModel.kt`. | `Log.e("ToolCall", "Error executing edit_file", e)` (line 1826); `Log.e("EditFile", "Error editing file", e)` (line 2432); etc. | Medium |
| 16 | Outbound Headers (app-identifying telemetry) | `app/src/main/java/io/github/stardomains3/oxproxion/ChatViewModel.kt` | 3159–3160, 3683–3684 | every OpenRouter POST | Every chat request advertises the app to OpenRouter via `HTTP-Referer: https://github.com/stardomains3/oxproxion/` and `X-Title: oxproxion`, allowing OpenRouter to attribute traffic to this specific app and to the user's account when correlated with the Bearer token. | `header("HTTP-Referer", "https://github.com/stardomains3/oxproxion/"); header("X-Title", "oxproxion")` (lines 3159, 3683). | High |

## B. Main AI Inference Workflows

### Workflow 1: Cloud Chat (OpenRouter, streaming)
- **Purpose**: Default text/image chat via OpenRouter when an `openrouter_api_key` is configured and streaming is enabled in Settings.
- **Input**: Text typed into `chatEditText` (R.id.chatEditText) plus optional gallery/camera attachment from `attachmentButton`. Optional system message and reasoning settings.
- **Processing**:
  - `ChatFragment` collects the input on `sendChatButton.performClick()` (line ~256, 998) and pushes it to `ChatViewModel`.
  - The view model assembles a `ChatRequest` (system + history + new user message; image content as base64 `image_url` parts) and prepares an SSE POST.
  - Ktor `httpClient.preparePost(activeChatUrl)` opens an HTTP/1.1 connection to `openrouter.ai`; deltas come back as `data: {...}` SSE chunks (`ChatViewModel.kt:3157–3311`).
- **Inference**: OpenRouter routes the prompt to the chosen model (`activeChatModel.value`, configurable per-chat); streamed `delta.content` is accumulated into `accumulatedResponse`.
- **Externalization**:
  - Full prompt + history + system message + image bytes leave the device over TLS to `openrouter.ai/api/v1/chat/completions` (Channel 1, Channel 3).
  - App-identifying headers are attached on every request (Channel 16).
  - Each delta is rendered into `chatRecyclerView`/`messageTextView` (Channel 12) and, when conversation mode is on, spoken aloud (Channel 13).
  - On user-initiated save, the full conversation (with stripped image parts) is persisted into Room (Channel 7).
  - Any error path may write the error string to SharedPreferences for the foreground notification (Channel 10).
- **Episode path**: ChatFragment input → ChatViewModel `ChatRequest` → Ktor SSE POST to OpenRouter → streamed `delta.content` → chatRecyclerView render → optional Room save / TTS / clipboard.
- **Key files**: `ChatFragment.kt` (input + render), `ChatViewModel.kt:3157–3450` (streaming POST + delta loop), `ApiData.kt` (`ChatRequest`/`StreamedChatResponse`), `LlmService.kt` (auxiliary calls), `fragment_chat.xml`, `item_message_ai.xml`.
- **Confidence**: High

### Workflow 2: Cloud Chat (OpenRouter, non-streaming + tools / image-generation)
- **Purpose**: Same as Workflow 1 but used when streaming is disabled, when the model is a tool-calling model performing function calls, or when the model returns generated images.
- **Input**: Identical text/image input from `ChatFragment`, optionally augmented with a tool definition payload.
- **Processing**:
  - `ChatViewModel` builds the `ChatRequest` with `tools = buildTools()` and `toolChoice = "auto"` if tools are enabled; for image-generation models it sets `modalities = ["image", "text"]` and `imageConfig = ImageConfig(aspectRatio)`.
  - One-shot Ktor POST: `httpClient.post(activeChatUrl) { ... }`; the entire response is parsed as `ChatResponse` (`ChatViewModel.kt:3681–3708`).
  - Returned image URLs (`delta.images[].image_url.url`) are downloaded by `downloadImages` and re-embedded as on-device URIs.
- **Inference**: OpenRouter returns either a normal completion, a tool-call directive (which the app re-invokes with tool results), or a base64/URL image payload.
- **Externalization**:
  - Full prompt + history + image bytes to OpenRouter (Channels 2, 3).
  - On-device tool execution writes/reads files (`Read File`, `Make File`, `Make Calendar Event`, etc.) and emits error logs to logcat (Channel 15).
  - Generated images are embedded into chat bubbles (Channel 12) and may be exported to PDF later (Channel 11).
  - Same Room persistence path on save (Channel 7).
- **Episode path**: ChatFragment input → ChatRequest with tool/imageConfig → Ktor POST → ChatResponse → tool dispatch / image download → chatRecyclerView render → Room save.
- **Key files**: `ChatViewModel.kt:3500–3850` (non-streaming send + tool plumbing), `ToolItem.kt`, `ToolsFragment.kt`, `PdfGenerator.kt`.
- **Confidence**: High

### Workflow 3: Auto Title Suggestion
- **Purpose**: Generate a 1–8-word save title for a chat by asking an LLM to summarize the chat contents.
- **Input**: The full text of the current chat (`chatContent`) constructed by the view model when the user taps "Save Chat".
- **Processing**:
  - `LlmService.getSuggestedChatTitle(chatContent, apiKey, modelId, endpoint, ...)` builds a `ChatRequest` containing the entire chat verbatim inside the prompt.
  - `withTimeout(SUGGESTION_TIMEOUT_MS = 15_000L) { activeClient.post(endpoint) { header("Authorization", "Bearer $apiKey"); ... setBody(chatRequest) } }`.
- **Inference**: OpenRouter (or LAN backend) returns a short title string (`chatResponse.choices[0].message.content`).
- **Externalization**:
  - The user's full chat content is POSTed to `openrouter.ai/api/v1/chat/completions` a second time (Channel 4) — even after the chat session has ended.
  - The returned title is then written to Room as `ChatSession.title` (Channel 7).
- **Episode path**: User taps Save → ChatViewModel builds chatContent → `LlmService.getSuggestedChatTitle` → POST → title returned → `repository.insertSessionAndMessages` writes session row.
- **Key files**: `LlmService.kt:35–98`, `ChatViewModel.kt:4040–4070` (call site), `SaveChatDialogFragment.kt`.
- **Confidence**: High

### Workflow 4: LAN Local-Model Chat
- **Purpose**: Chat against a self-hosted Ollama / LM Studio / llama.cpp / MLX LM / Hermes Agent endpoint on the user's network.
- **Input**: Same text/image input as Workflows 1–2.
- **Processing**:
  - `lanHttpClient` (separate Ktor `HttpClient(OkHttp)` instance, `ChatViewModel.kt:213, 216`) is used instead of `httpClient`.
  - URL is taken from `SharedPreferencesHelper.getLanEndpoint()` (line 752); auth header uses `getLanApiKey()` which defaults to `"any-non-empty-string"` if the user never set one.
- **Inference**: Local model on the LAN responds; same SSE/non-SSE branches.
- **Externalization**:
  - Prompt + image content parts leave the device to the LAN endpoint (Channel 6) — and unlike the cloud path, the connection is plaintext HTTP unless the user provisioned TLS.
  - All other downstream channels (Room save, UI render, TTS, PDF, clipboard) behave identically (Channels 7, 11, 12, 13, 14).
- **Episode path**: ChatFragment input → ChatViewModel (LAN branch) → Ktor `lanHttpClient` POST to LAN URL → streamed/one-shot response → render + persist.
- **Key files**: `ChatViewModel.kt:2820–3180, 3486–3500, 213–216` (`createLanHttpClient`), `SharedPreferencesHelper.kt:736–760`, `LanModelsFragment.kt`, `SaveLANDialogFragment.kt`.
- **Confidence**: High

### Workflow 5: OpenRouter Credits / Model Catalog Refresh
- **Purpose**: Long-press the API-key icon to display remaining credits, and pull the live OpenRouter model catalog.
- **Input**: Stored OpenRouter API key.
- **Processing**:
  - `LlmService.getRemainingCredits(apiKey)` issues `GET https://openrouter.ai/api/v1/credits`.
  - Model fetch issues `GET https://openrouter.ai/api/v1/models` (`ChatViewModel.kt:4986`).
- **Inference**: None on-device; OpenRouter responds with usage / model JSON which is then displayed and (for the catalog) cached via `saveOpenRouterModels()`.
- **Externalization**:
  - API key + device IP + User-Agent sent to `openrouter.ai` (Channel 5).
  - Cached model list lands in `MainAppPrefs` SharedPreferences (Channel 9 family).
- **Episode path**: Long-press → `getRemainingCredits` → OpenRouter → numeric remaining credit displayed in UI.
- **Key files**: `LlmService.kt:111–133`, `ChatViewModel.kt:4980–5000` (catalog), `OpenRouterModelsFragment.kt`.
- **Confidence**: High

### Workflow 6: Save / Export / Share Chat
- **Purpose**: Persist a chat to disk for later reload, export as PDF, or share via Android intents.
- **Input**: Current `_chatMessages` list, optional generated-image URIs, current model identifier.
- **Processing**:
  - `repository.insertSessionAndMessages(session, chatMessages)` writes the conversation into Room (`chat_database`, unencrypted).
  - `PdfGenerator.generatePdf(messages, modelName)` lays out the conversation (with embedded images) and writes via `FileOutputStream`.
  - The avatar tap copies the rendered response into the system clipboard.
- **Externalization**:
  - Full conversation text (with image content parts stripped, image URIs preserved) saved into Room (Channel 7).
  - Full conversation rendered into a PDF on disk (Channel 11) — recoverable by file managers, share sheet, or backup tools.
  - Assistant response copied to system clipboard (Channel 14).
- **Episode path**: User triggers Save / PDF / Copy → ChatViewModel collects messages → Room insert OR PdfGenerator write OR ClipData paste.
- **Key files**: `ChatRepository.kt`, `ChatDao.kt:22–61`, `AppDatabase.kt:8–28`, `PdfGenerator.kt`, `ChatBackup.kt`, `SaveChatDialogFragment.kt`.
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 16
- **Total number of main AI inference workflows found**: 6
- **Top 3 highest-risk workflows or channels**:
    1. **Workflow 1 / Channels 1, 3 — Cloud streaming chat with image attachments to OpenRouter**: Every keystroke that the user sends, plus the entire prior conversation and any attached gallery/camera image (base64-encoded), is shipped to a third-party API gateway (`openrouter.ai`) which then forwards to the chosen upstream provider (OpenAI, Anthropic, Google, xAI, etc.). The Bearer-token + identifying `HTTP-Referer`/`X-Title` headers (Channel 16) tie every request to the user's OpenRouter account, and the image path inherits faces, location backdrops, and any visible text from the photo. There is no PII redaction layer between `ChatFragment` and the network call.
    2. **Workflow 3 / Channel 4 — Auto title-suggestion re-POST**: When the user saves a chat, the *entire* chat content (including anything the user might have considered private once they decided to leave the conversation) is POSTed *again* to OpenRouter inside a synthetic prompt asking for a title. This effectively duplicates the privacy footprint of the conversation and creates a second log entry on OpenRouter's side, often using a different model than the chat itself. Many users would not anticipate that "Save Chat" is itself a network exfiltration step.
    3. **Channel 7 — Unencrypted Room database (`chat_database`)**: All saved sessions, including original user prompts and assistant replies, are persisted in a plain SQLite DB inside the app's private data dir with no field-level encryption. The README explicitly warns "Chats are stored unencrypted on device only", so any device backup, root-shell access, ADB pull on a debuggable build, or post-compromise forensic image trivially recovers full chat history. Combined with the foreground-service SharedPreferences cache (Channel 10) and the PDF export path (Channel 11), the on-device blast radius of a single device compromise is large.
