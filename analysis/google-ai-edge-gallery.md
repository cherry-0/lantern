# AI Inference Privacy Audit: google-ai-edge-gallery

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Analytics (Firebase) | `Analytics.kt` | 33, 44 | `firebaseAnalytics` | User events: capability selection, model downloads, generation actions, and button clicks. | `firebaseAnalytics?.logEvent(GalleryEvent.MODEL_DOWNLOAD.id, ...)` | High |
| 2 | Network Request (HTTP) | `worker/DownloadWorker.kt` | 133 | `doWork` | Model download requests (URLs, access tokens for gated models). | `url.openConnection() as HttpURLConnection` | High |
| 3 | UI Rendering (WebView) | `customtasks/agentchat/AgentChatScreen.kt` | 239 | `AgentChatScreen` | JavaScript execution results, potential for skills to render external web content. | `webViewRef?.evaluateJavascript(script, null)` | High |
| 4 | Notification / Intent | `customtasks/agentchat/IntentHandler.kt` | 51, 75 | `handleAction` | User data (email, SMS) triggered by AI tool-calling. | `context.startActivity(intent)` with `ACTION_SEND`, `ACTION_SENDTO` | High |
| 5 | Logging | `customtasks/agentchat/AgentChatScreen.kt` | 243-261 | `AgentChatScreen` | WebView console messages (logs, errors, warnings) from JS skills. | `Log.d(TAG, "${curConsoleMessage.message()} ...")` | High |
| 6 | FCM (Firebase Cloud Messaging) | `FcmMessagingService.kt` | 33-54 | `onMessageReceived` | Incoming push notification payloads and potentially downloaded image URLs. | `remoteMessage.notification?.let { ... }` | High |

## B. Main AI Inference Workflows

### Workflow 1: Local Multimodal Chat (LiteRT LM)
- **Purpose**: Provide a fully local, multimodal AI chat assistant (text, image, audio).
- **Input**: User text messages, local image bitmaps, and recorded audio clips.
- **Processing**: Content is converted to `com.google.ai.edge.litertlm.Contents`; images are converted to PNG byte arrays.
- **Inference**: Local inference using LiteRT (TFLite) `Engine` and `Conversation` in the `LlmChatModelHelper`.
- **Externalization**: 
    - Results rendered in the UI (`LlmChatScreen`).
    - Status events logged via Firebase Analytics (`Analytics.kt`).
- **Episode path**: User Input -> Byte Conversion -> LiteRT LM Engine (Local) -> UI Rendering
- **Key files**: `LlmChatModelHelper.kt`, `LlmChatViewModel.kt`, `LlmChatScreen.kt`
- **Confidence**: High

### Workflow 2: Agent Skill Execution (Tool Calling)
- **Purpose**: Extend the AI's capabilities by allowing it to call "Skills" (JavaScript-based tools).
- **Input**: User prompt requiring external actions (e.g., "Send an email").
- **Processing**: LLM identifies the need for a tool and generates parameters for `runJs` or `runIntent`.
- **Inference**: Local LLM identifies the tool call; then JavaScript is executed in a `WebView`.
- **Externalization**: 
    - JavaScript can perform network requests or trigger Android Intents (Email/SMS via `IntentHandler`).
    - Progress and result data displayed in UI via `actionChannel`.
- **Episode path**: User Prompt -> Local LLM (Tool Call) -> AgentTools (JS/Intent) -> External App/Network -> UI
- **Key files**: `AgentTools.kt`, `AgentChatScreen.kt`, `IntentHandler.kt`, `SkillManagerViewModel.kt`
- **Confidence**: High

### Workflow 3: Model Gating & Authentication (Hugging Face)
- **Purpose**: Authenticate users to download gated models from Hugging Face.
- **Input**: User's Hugging Face OAuth token or secret.
- **Processing**: OAuth flow handled via `ProjectConfig.kt` and `DownloadAndTryButton.kt`.
- **Inference**: N/A (Pre-inference setup).
- **Externalization**: 
    - Access tokens transmitted to Hugging Face servers for model retrieval.
- **Episode path**: User Auth -> Hugging Face OAuth -> Token Retrieval -> Model Download (HTTP)
- **Key files**: `ProjectConfig.kt`, `DownloadAndTryButton.kt`, `DownloadWorker.kt`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 6
- **Total number of main AI inference workflows found**: 3
- **Top 3 highest-risk workflows or channels**:
    1. **Intent Triggering (Workflow 2)**: The AI can autonomously trigger system intents to send Emails and SMS messages containing potentially sensitive user data to arbitrary recipients.
    2. **JavaScript Skill Execution (Workflow 2)**: Skills loaded from external URLs (e.g., Hugging Face) can execute arbitrary JavaScript in a WebView, potentially leaking local context or secrets via network requests.
    3. **Firebase Analytics (Channel 1)**: Tracks fine-grained user interaction with AI models and capabilities, which could be used to profile user behavior and preferences.
