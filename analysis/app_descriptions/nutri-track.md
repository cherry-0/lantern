# AI Inference Privacy Audit: nutri-track

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM — text) | `data/viewmodel/GenAIViewModel.kt` | 438–516 | `sendRequest` (NutriCoach end-user chat) | User free-text nutrition query plus a JSON blob of the user's stats — `persona`, `biggestMealTime`, `sleepTime`, `wakeUpTime`, all HEIFA `nutritionScores`, and `foodPreferences` — embedded in a system-prompted Gemini call. | `genAIModel.generateContentStream(themedPrompt)` where `themedPrompt = buildThemedPrompt(prompt, userStatsJson, NUTRITIONAL_GUIDELINES_JSON)` and `genAIModel = GenerativeModel(modelName = "gemini-2.5-flash-preview-05-20", apiKey = BuildConfig.MAPS_API_KEY)` | High |
| 2 | Network Request (Cloud LLM — text) | `data/viewmodel/GenAIViewModel.kt` | 275–381 | `sendAnalysisRequest` (clinician dashboard analysis) | Aggregate patient JSON dataset (per-user `userId`, `userName`, `gender`, `heifaScore`, plus `maleAverageScore` / `femaleAverageScore`) sent to Gemini under a clinician-analyst system instruction asking for PATTERNS / TRENDS. | `clinicianModel = createModelWithSystemInstruction(systemPrompt); val responseFlow = clinicianModel.generateContentStream(analysisPrompt)` where `analysisPrompt = "Analyze this nutrition data and provide insights: $userStatsJson..."` | High |
| 3 | Network Request (Cloud LLM — text) | `data/viewmodel/GenAIViewModel.kt` | 387–436 | `sendClinicianQuestion` (clinician Q&A) | Clinician's free-text query plus the same aggregate patient stats JSON, sent to Gemini under a clinician-QA system instruction. | `val qaPrompt = "Query: $query\nAvailable data: $userStatsJson"; clinicianModel.generateContentStream(qaPrompt)` | High |
| 4 | Local Storage (Room DB — `chat_messages`) | `data/viewmodel/GenAIViewModel.kt` → `data/model/repository/ChatRepository.kt` → `chat_messages` table | 466–470, 504–508 (and equivalent calls in the clinician methods) | `chatRepository.saveUserMessage` / `chatRepository.saveAiResponse` | Every user prompt and parsed AI main-answer is persisted to the on-device `chat_messages` Room table, keyed by `userId` and `sessionId`, with a `timestamp`. Clinician analysis/QA messages are stored with `userId = null` under fixed sessionIds `clinician-analysis` / `clinician-qa`. | `chatRepository.saveUserMessage(message = prompt, userId = requestUserId, sessionId = requestSessionId)` and `chatRepository.saveAiResponse(response = mainAnswer, userId = requestUserId, sessionId = requestSessionId)` | High |
| 5 | UI Rendering (NutriCoach chat panel) | `view/screens/CoachScreen.kt` | 395–475, 478–520 | `AiChatPanel`, `AiMessageBubble`, `UserMessageBubble`, follow-up `AssistChip` row | Full conversation history (user + AI) rendered as Markdown bubbles in a `LazyColumn`; streaming partial answers shown as they arrive; parsed `SUGGESTED_FOLLOW_UPS:` chips displayed; "All AI Responses" dialog dumps every persisted AI reply for the user. | `MarkdownText(message.message)` inside `items(filteredHistory)`; `AiMessageBubble(text = (genAiUiState as UiState.Streaming).currentMessageContent, ...)`; `AssistChip(...) { Text(question, ...) }` over `currentUiState.suggestedFollowUps` | High |
| 6 | UI Rendering (Clinician dashboard) | `view/screens/ClinicianDashboardScreen.kt` (consumes `ClinicianDashboardViewModel`) | — | Patterns / Trends tab and "Ask AI" tab | AI-generated `_patterns` and `_trends` lists (parsed from PATTERNS:/TRENDS: bullets) plus the live Q&A streaming response are rendered alongside per-patient `UserStatsDisplay` rows (`userId`, `userName`, `gender`, `heifaScore`). | `_patterns.value = patternsList; _trends.value = trendsList; _qaResponse.value = state.finalMessageContent` consumed by the dashboard composable | High |
| 7 | Application Logging (Logcat / stdout) | `data/viewmodel/GenAIViewModel.kt`, `data/viewmodel/ClinicianDashboardViewModel.kt` | GenAI: 286, 501, 552, 575, 577; Clinician: 149, 156, 158, 170, 200, 217, 221, 228, 242, 297, 330 | `Log.d` / `Log.w` / `Log.e` / `println` | Parsed AI main answer and follow-up questions printed via `println(...)`; current `userId` printed on session events; clinician dashboard logs each `userId`, `userGender`, `heifaScore`, the full `userStatsJson` payload, and the raw AI response to Logcat. | `println("GenAIViewModel: Parsed Main Answer: '$mainAnswer', Follow-ups: $finalFollowUpQuestions")`; `Log.d(TAG, "convertUserStatsToJson: Generated JSON: $jsonString")`; `Log.d(TAG, "generateAIAnalysis: Got AI response: $response")` | High |


## B. Main AI Inference Workflows

### Workflow 1: NutriCoach End-User Chat (`sendRequest`)
- **Purpose**: Answer the end user's free-text nutrition question with personalized advice grounded in Australian adult nutrition guidelines (HEIFA).
- **Input**: User's typed query in the NutriCoach chat box; resolved `currentUserId` from `SharedPreferencesManager`.
- **Processing**:
  - `userStatsViewModel.getUserStats(userId)` is called to fetch the user's `persona`, `biggestMealTime`, `sleepTime`, `wakeUpTime`, every HEIFA nutrition score, and per-category food preferences.
  - `userStatsToJson(stats)` serializes those fields into a compact JSON object.
  - `buildThemedPrompt` concatenates a NutriCoach role description, the user-stats JSON, the embedded `NUTRITIONAL_GUIDELINES_JSON` (food group max scores, criteria, terminology) and the user query, ending with a `SUGGESTED_FOLLOW_UPS:` formatting instruction.
  - The user's raw prompt is persisted to Room (`chat_messages`) under `(userId, sessionId)` *before* the model call.
- **Inference**: `GenerativeModel(modelName = "gemini-2.5-flash-preview-05-20", apiKey = BuildConfig.MAPS_API_KEY).generateContentStream(themedPrompt)` streams chunks; chunks are reduced to `internalAccumulatedResponse` and to a per-character animated `uiDisplayResponse`. On completion the response is split on `SUGGESTED_FOLLOW_UPS:` into `mainAnswer` and a comma-separated follow-up list.
- **Externalization**:
  - User query + full `userStatsJson` (persona, sleep/wake/meal times, all nutrition scores, food preferences) sent to `generativelanguage.googleapis.com` (Channel 1).
  - User prompt and parsed AI main-answer written to `chat_messages` Room table (Channel 4).
  - Markdown-rendered AI bubble, streaming partial text, and follow-up `AssistChip` row shown in `AiChatPanel` (Channel 5).
  - `println("GenAIViewModel: Parsed Main Answer: '$mainAnswer', Follow-ups: ...")` and session-id/user-id `println` calls (Channel 7).
  - Gemini API key shipped inside the APK (Channel 8).
- **Episode path**: User types query → `sendRequest(prompt, userId)` → `getUserStats` → `userStatsToJson` → `buildThemedPrompt` → `saveUserMessage` (Room) → `generateContentStream` (Gemini) → split on `SUGGESTED_FOLLOW_UPS:` → `saveAiResponse` (Room) → `UiState.Success(mainAnswer, followUps)` → `MarkdownText` bubble + `AssistChip` row.
- **Key files**: `data/viewmodel/GenAIViewModel.kt` (`sendRequest`, `buildThemedPrompt`, `userStatsToJson`), `view/screens/CoachScreen.kt` (`AiChatPanel`), `data/model/repository/ChatRepository.kt`.
- **Confidence**: High

### Workflow 2: Clinician Dashboard Pattern/Trend Analysis (`sendAnalysisRequest`)
- **Purpose**: Auto-generate three "patterns" and 5–7 "trends" describing dietary habits, demographics and HEIFA scoring correlations across the entire registered-user cohort, for display in the clinician dashboard.
- **Input**: The list of all registered `UserEntity` rows; per-user total HEIFA score (`ScoreTypes.TOTAL`); a static system prompt describing the clinician-analyst role.
- **Processing**:
  - `loadAllUserStats()` queries every user (`userId`, `userName`, `userGender`) and their HEIFA total score, then computes `maleAverageScore` and `femaleAverageScore`.
  - `convertUserStatsToJson` packs all per-user records plus the gender averages into a single JSON object (`{maleAverageScore, femaleAverageScore, users:[{userId,userName,gender,heifaScore},...]}`).
  - `sendAnalysisRequest` builds a Gemini model with `systemInstruction = clinicianAnalysisPrompt` and submits an analysis prompt of the form `"Analyze this nutrition data and provide insights: $userStatsJson\n\nFORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:\nPATTERNS:\n- ...\nTRENDS:\n- ..."`.
  - Empty-dataset short-circuit returns a hardcoded fallback string without contacting Gemini.
  - Response is streamed character-by-character into the UI; `ensureCorrectAnalysisFormat` re-shapes it; `parseAIAnalysisResponse` splits on `PATTERNS:` / `TRENDS:` to populate `_patterns` / `_trends` StateFlows.
- **Inference**: `clinicianModel.generateContentStream(analysisPrompt)` against `gemini-2.5-flash-preview-05-20`.
- **Externalization**:
  - Aggregate cohort JSON (every patient's `userId`, `userName`, `gender`, `heifaScore` and gender averages) sent to Gemini (Channel 2).
  - Both the placeholder request `"Generate analysis of user nutrition data"` and the formatted response are written to `chat_messages` with `userId = null` and `sessionId = "clinician-analysis"` (Channel 4).
  - Patterns and trends rendered in the dashboard tabs alongside per-patient rows (Channel 6).
  - `Log.d(TAG, "convertUserStatsToJson: Generated JSON: $jsonString")` and `Log.d(TAG, "generateAIAnalysis: Got AI response: $response")` write the entire prompt+answer to Logcat (Channel 7).
- **Episode path**: Clinician opens dashboard → `loadAllUserStats` → per-user HEIFA query → `convertUserStatsToJson` → `sendAnalysisRequest(userStatsJson, clinicianAnalysisPrompt)` → `saveUserMessage` (Room, `userId=null`) → `generateContentStream` (Gemini) → `ensureCorrectAnalysisFormat` → `saveAiResponse` (Room) → `parseAIAnalysisResponse` → `_patterns` / `_trends` populate dashboard.
- **Key files**: `data/viewmodel/GenAIViewModel.kt` (`sendAnalysisRequest`, `createModelWithSystemInstruction`, `ensureCorrectAnalysisFormat`), `data/viewmodel/ClinicianDashboardViewModel.kt` (`loadAllUserStats`, `convertUserStatsToJson`, `generateAIAnalysis`, `parseAIAnalysisResponse`).
- **Confidence**: High

### Workflow 3: Clinician Q&A Over Aggregate Patient Stats (`sendClinicianQuestion`)
- **Purpose**: Let a clinician ask free-text questions about cohort-level nutritional patterns and receive evidence-based clinical interpretations.
- **Input**: Clinician's free-text query (`_clinicianQuery`); the same `userStatsJson` produced by `convertUserStatsToJson`; the `clinicianQAPrompt` system instruction enumerating HEIFA categories.
- **Processing**:
  - `askQuestion(query)` updates state and dispatches to `genAIViewModel.sendClinicianQuestion(query, userStatsJson, clinicianQAPrompt)`.
  - A model is constructed with `systemInstruction = clinicianQAPrompt` and prompted with `"Query: $query\nAvailable data: $userStatsJson"`.
  - The query is persisted to `chat_messages` (`userId = null`, `sessionId = "clinician-qa"`) before the call; the streamed response is animated character-by-character and the final text is saved.
- **Inference**: `clinicianModel.generateContentStream(qaPrompt)` against `gemini-2.5-flash-preview-05-20`.
- **Externalization**:
  - Clinician query + cohort JSON (per-patient `userId`, `userName`, `gender`, `heifaScore` and gender averages) sent to Gemini (Channel 3).
  - Query and response written to Room `chat_messages` (Channel 4).
  - Streaming + final response surfaced via `_qaResponse` to the dashboard's "Ask AI" tab (Channel 6).
  - Logcat traces of dashboard state and AI response (Channel 7).
- **Episode path**: Clinician types question → `askQuestion` → `convertUserStatsToJson` → `sendClinicianQuestion` (Gemini) → `saveUserMessage` / `saveAiResponse` (Room, `userId=null`, `sessionId="clinician-qa"`) → streaming Q&A view via `observeGenAIState`.
- **Key files**: `data/viewmodel/GenAIViewModel.kt` (`sendClinicianQuestion`, `createModelWithSystemInstruction`), `data/viewmodel/ClinicianDashboardViewModel.kt` (`askQuestion`, `clinicianQAPrompt`, `observeGenAIState`).
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 8
- **Total number of main AI inference workflows found**: 3
- **Top 3 highest-risk workflows or channels**:
    1. **NutriCoach End-User Chat (Workflow 1 / Channel 1)**: Each request to `gemini-2.5-flash-preview-05-20` carries a structured profile of the user — persona type, biggest-meal time, sleep/wake schedule, every HEIFA nutrition score, and per-category food preferences — bundled with the free-text question. This is a rich behavioral and dietary fingerprint per call, sent in the clear to a third-party LLM with no PII redaction or pseudonymization in the request payload.
    2. **Clinician Dashboard Analysis & Q&A (Workflows 2 & 3 / Channels 2, 3)**: `convertUserStatsToJson` ships the entire registered-user cohort to Gemini in one prompt, including each patient's `userId`, `userName`, `gender`, and total HEIFA score, plus gender-averaged scores. The clinician QA path then re-sends this same cohort JSON with every clinician question. The system prompt instructs the model to "respect patient privacy" but no privacy controls are enforced client-side — the raw identifiers and names leave the device.
    3. **Logcat exposure of full prompts and responses (Channel 7) combined with the embedded API key (Channel 8)**: `ClinicianDashboardViewModel` writes the full `userStatsJson` payload and the full Gemini response to Logcat (`Log.d(TAG, "convertUserStatsToJson: Generated JSON: $jsonString")`, `Log.d(TAG, "generateAIAnalysis: Got AI response: $response")`), and `GenAIViewModel` `println`s parsed answers and user/session ids. Combined with the Gemini key fallback hardcoded as a comment in `createModelWithSystemInstruction` and the `BuildConfig.MAPS_API_KEY` baked into the APK, an attacker with logcat access (rooted device, ADB, crash report aggregator) or APK reverse-engineering capability can both read every chat and abuse the inference quota.
