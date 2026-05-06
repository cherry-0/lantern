# AI Inference Privacy Audit: Waico (Wellbeing AI Companion)

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | UI Rendering (Flutter) | `features/counselor/counselor_agent.dart` | — | Counselor chat stream | Full AI response streamed token-by-token into the chat interface; rendered in real-time via MediaPipe LLM Inference streaming API | Flutter `StreamBuilder` on Gemma 3n token stream | High |
| 2 | Storage Write (ObjectBox — Conversation) | `core/ai_agent/conversation_processor.dart` | 405 | `ConversationRepository.save()` | Entire conversation saved: AI-generated `summary`, `observations` (clinical-grade notes), message history | `_conversationRepository.save(conversation)` after `ConversationProcessor.process()` | High |
| 3 | Storage Write (ObjectBox — Episodic Memory) | `core/ai_agent/conversation_processor.dart` | 419–422 | `ConversationMemoryRepository.save()` | Extracted episodic memories converted to dense vector embeddings (Qwen3-Embedding-0.6B) and written to ObjectBox for RAG retrieval | `_conversationMemoryRepository.save(conversationMemory)` with `embeddings` list | High |
| 4 | Storage Write (ObjectBox — User Profile) | `core/repositories/user_repository.dart` | 45, 78, 116 | `UserRepository.updateUserInfo()` | User profile updated: name, goals, preferences, contact information of healthcare professionals (therapist email/phone, doctor email) | `save(user)` after extracting new info via LLM in `ConversationProcessor._userInfoPrompt` | High |
| 5 | Network (Email — tool-triggered) | `core/ai_agent/tools.dart` / `core/services/communication_service.dart` | 85–112 / 9 | `ReportTool.call()` → `CommunicationService.sendEmail()` | Clinical wellbeing report (last 5 conversation summaries + observations) emailed via `flutter_mailer` to recipient email provided by user (typically therapist/doctor) | `FlutterMailer.send(MailOptions(body: reportContent, ...))` — triggers device email client or direct send | High |
| 6 | Network (Phone call — tool-triggered) | `core/ai_agent/tools.dart` / `core/services/communication_service.dart` | 42–51 / 5 | `PhoneCallTool.call()` → `CommunicationService.makePhoneCall()` | Phone number dialled via `flutter_phone_direct_caller` — initiates direct call to health professional without user confirmation step | `FlutterPhoneDirectCaller.callNumber(phoneNumber)` | High |
| 7 | Network (Model download — one-time setup) | `lib/ai_models_init_page.dart` | 149–161 | `FileDownloader` / `background_downloader` | Gemma 3n `.task` file, STT (nemo-fast-conformer), optional TTS models downloaded from `huggingface.co/sitatech/waico-models` on first launch | `FileDownloader().download(task)` with HuggingFace URL | High |

## B. Main AI Inference Workflows

### Workflow 1: Counselor Chat (Text mode)
- **Purpose**: Empathetic mental health support via Gemma 3n on-device Counselor agent with tool-calling.
- **Input**: User message text.
- **Processing**:
  - System prompt dynamically enriched with: available tool definitions, user profile from `UserRepository`, usage examples.
  - `AiAgent` runs multi-iteration tool-execution loop (up to N iterations).
  - Each iteration: Gemma 3n generates response → `ToolParser` detects `tool_call` blocks → executes matched `Tool.call()` → appends `tool_output` to history → re-runs if more tools needed.
  - After final response: `ConversationProcessor.process()` runs three parallel LLM passes (summary, user info extraction, memory extraction) using the same on-device model.
- **Inference**: Gemma 3n E2B or E4B via MediaPipe LLM Inference (fully on-device).
- **Post-inference externalizations**:
  - Conversation + summary + clinical observations → ObjectBox (Channel 2).
  - Episodic memories → ObjectBox with Qwen3 embeddings (Channel 3).
  - User profile update → ObjectBox (Channel 4).
  - If `ReportTool` invoked: email sent with report (Channel 5).
  - If `PhoneCallTool` invoked: direct phone call placed (Channel 6).
- **Episode path**: User text → Gemma 3n (on-device) → tool loop → response → ConversationProcessor → ObjectBox saves → (optionally) email / phone
- **Key files**: `core/ai_agent/ai_agent.dart`, `core/ai_agent/tools.dart`, `core/ai_agent/conversation_processor.dart`, `core/repositories/`
- **Confidence**: High

### Workflow 2: Counselor Chat (Voice mode)
- **Same as Workflow 1** with additional pipeline stages:
  - VAD (Silero v4) → STT (nemo-fast-conformer, on-device) → text → Gemma 3n → text response → chunked by punctuation → TTS (Kokoro or Piper, on-device) → audio playback.
  - No additional network externalizations — STT and TTS run fully on-device.
- **Confidence**: High

### Workflow 3: Guided Meditation Generator
- **Input**: User's current mood/need.
- **Processing**: Gemma 3n generates a personalised meditation script with pause markers → TTS converts chunks to audio → audio plays with background music.
- **Post-inference**: Conversation saved via `ConversationProcessor` (Channel 2, 3, 4). No network calls.
- **Confidence**: Medium (inferred from architecture; Counselor agent handles this as a sub-feature)

### Workflow 4: Workout Coach (Pose detection)
- **Input**: Live camera frames.
- **Processing**: MediaPipe Pose Landmark Detection (on-device) → exercise classifier (holistic algorithms) → rep counting → Gemma 3n generates form feedback text.
- **Post-inference**: No ObjectBox saves for workout feedback; workout plan saved to `UserRepository` if generated (Channel 4).
- **Network**: None (fully on-device).
- **Confidence**: High

## C. Verify Adapter Coverage

| Channel | Real app behaviour | Serverless adapter captures? |
|---|---|---|
| UI | Token-by-token streaming response in counselor chat | ✅ Yes — `realistic_fallback["UI"]` shows truncated response preview |
| STORAGE | ObjectBox saves: Conversation (summary + observations), ConversationMemory (embeddings), User profile | ✅ Yes (patched) — `realistic_fallback["STORAGE"]` describes all three ObjectBox writes |
| NETWORK (tool-triggered email) | `flutter_mailer` sends wellbeing report to therapist/doctor email | ❌ Not captured — tool invocation depends on LLM deciding to call `ReportTool`; serverless path does not simulate tool calling |
| NETWORK (tool-triggered phone) | `flutter_phone_direct_caller` dials health professional | ❌ Not captured — same reason as email; tool-triggered conditional |
| NETWORK (model download) | One-time download of Gemma 3n + STT from `huggingface.co/sitatech/waico-models` | ✅ Noted in metadata — excluded from per-inference externalizations as it is a one-time setup event, not a per-inference channel |

## Final Summary
- **Distinct externalization sites**: 7 (4 always-on, 2 conditional tool-triggered, 1 one-time setup)
- **Main inference workflows**: 4
- **Top risks**:
  1. **ConversationProcessor (Channels 2–4)**: After every conversation, Gemma 3n runs three additional LLM passes to extract clinical observations, episodic memories, and personal user profile updates — all of which are stored in ObjectBox. This creates a richly detailed, persistent local profile including mental health history, personal relationships, and contact details for healthcare professionals.
  2. **ReportTool (Channel 5)**: The counselor agent can independently decide to email a clinical wellbeing report (based on last 5 conversation summaries + clinical observations) to any email address the user has mentioned — without requiring an additional confirmation step. The full `reportContent` string embeds multi-session summaries.
  3. **PhoneCallTool (Channel 6)**: The agent can initiate a direct phone call (bypassing the standard dial confirmation) to a number extracted from conversation context. The tool definition explicitly tells the model this action is "completely safe" and it is "encouraged" to use it.
