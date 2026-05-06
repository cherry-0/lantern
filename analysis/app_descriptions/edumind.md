# AI Inference Privacy Audit: edumind

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Gemini) | `backend/services/geminiService.js` | 18-37 | `generateContent` | Full AI prompt for summarization, quiz generation, tutoring, study planning, or flashcard generation sent to Google Gemini. | `model.generateContent(prompt)` after `getGenerativeModel({ model: ... })` | High |
| 2 | Network Request (OpenAI fallback / explicit provider) | `backend/services/aiService.js` | 76-87 | `callOpenAI` | OpenAI-style `messages` payload, including user study content and generated prompts, sent to OpenAI if explicitly selected. | `this.openai.chat.completions.create(options)` | Medium |
| 3 | Network Request (Legacy direct OpenAI-compatible path) | `backend/controllers/ai.js` | 140-167 | `callOpenAI` | Prompt text and optional uploaded image encoded as base64 sent to a configured OpenAI-compatible chat-completions endpoint. | `axios.post(\`${service.baseUrl}/chat/completions\`, { model, messages, ... })` | Medium |
| 4 | Network Request (Legacy direct Gemini path) | `backend/controllers/ai.js` | 179-204 | `callGemini` | Prompt text and optional uploaded file content sent to Gemini `generateContent`. | `axios.post(\`${service.baseUrl}/models/${service.model}:generateContent?key=...\`, { contents: [{ parts }] })` | Medium |
| 5 | Network Request (Grok) | `backend/controllers/ai.js` | 220-235 | `callGrok` | Prompt text sent to an OpenAI-compatible Grok chat-completions endpoint. | `axios.post(\`${service.baseUrl}/chat/completions\`, { model, messages: ... })` | Medium |
| 6 | Cloud Database Write (MongoDB) | `backend/services/geminiService.js` | 22-55, 65-74 | `generateContent` | AI usage record containing user ID, tool type, provider, model name, token counts, latency, status, and error message on failures. | `await AIUsage.create(usageRecord)` and failure `AIUsage.create(...)` | High |
| 7 | Cloud Database Write (MongoDB) | `backend/controllers/ai.js` | 422-454 | `summarizeText` | Logged-in users' generated summaries plus source metadata, token counts, provider, IP address, and user agent. | `Summary.create(summaryData)` and `UserActivity.create(...)` | High |
| 8 | Cloud Database Write (MongoDB) | `backend/controllers/ai.js` | 496-520 | `generateQuiz` | Logged-in users' generated quiz questions plus title, difficulty, source metadata, IP address, and user agent. | `Quiz.create(...)` and `UserActivity.create(...)` | High |
| 9 | Cloud Database Write (MongoDB) | `backend/controllers/ai.js` | 561-571 | `aiTutor` | Logged-in users' AI tutor activity metadata, including a truncated question, IP address, and user agent. | `UserActivity.create({ ..., metadata: { question: question.slice(0, 50) } })` | High |
| 10 | Cloud Database Write (MongoDB) | `backend/middlewares/aiUsageGuard.js` | 34-62, 82-92 | `aiUsageGuard` | Quota and rate-limit failures written to `AIUsage`, including user ID or guest status, tool type, error code, and plan. | `AIUsage.create({ ..., errorMessage: 'QUOTA_EXCEEDED' })` | High |
| 11 | Network Request (YouTube transcript) | `backend/controllers/ai.js` | 373-386 | `summarizeText` | YouTube video ID/URL used to fetch captions; transcript becomes AI input for summarization. | `YoutubeTranscript.fetchTranscript(videoId)` then `contentToSummarize = ...join(' ')` | Medium |
| 12 | UI / HTTP Response | `backend/controllers/ai.js` | 457-463, 523-529, 574-579, 622-627, 680-685 | AI route handlers | Generated summary, quiz, tutor answer, study plan, or flashcards returned to the client. | `res.status(200).json({ success: true, data: ... })` | High |
| 13 | Server Logging | `backend/controllers/ai.js`, `backend/services/geminiService.js`, `backend/services/aiService.js`, `backend/middlewares/aiUsageGuard.js` | 131-136, 76, 116, 35, 83, 107 | Various | Parse failures, provider failures, quota/rate-limit events, and transcript errors logged server-side. | `console.error(...)`, `logger.error(...)`, `logger.warn(...)` | Medium |

## B. Main AI Inference Workflows

### Workflow 1: Smart Summarizer (`POST /api/ai/summarize`)
- **Purpose**: Summarize user text, uploaded files, notes, or YouTube transcripts into a requested synthesis style.
- **Input**: `text`, optional uploaded file, optional `youtubeUrl`, `type`, and `length`.
- **Processing**:
    - Text or transcript is selected as `contentToSummarize`.
    - `buildPrompt('summarize', ...)` embeds the full content into a high-fidelity synthesis prompt (`backend/controllers/ai.js:45-91`).
    - `aiService.chatCompletion()` routes primarily to Gemini (`backend/services/aiService.js:23-50`).
- **Inference**: Google Gemini `gemini-1.5-flash` by default.
- **Externalization**:
    - Full study material or transcript sent to Gemini (Channel 1).
    - Logged-in users' summary and source metadata saved to MongoDB (Channel 7).
    - Response returned to UI/API caller (Channel 12).
- **Episode path**: User text/file/YouTube URL -> prompt construction -> Gemini API -> parsed summary -> optional MongoDB storage -> HTTP response/UI.
- **Key files**: `backend/controllers/ai.js`, `backend/services/aiService.js`, `backend/services/geminiService.js`, `backend/routes/ai.js`
- **Confidence**: High

### Workflow 2: Quiz Generation (`POST /api/ai/generate-quiz`)
- **Purpose**: Generate multiple-choice questions from supplied study content.
- **Input**: `text`, `numQuestions`, `difficulty`, optional source metadata.
- **Processing**:
    - `buildPrompt('quiz', ...)` asks the model for a JSON array of questions (`backend/controllers/ai.js:92-93`).
    - Response is parsed with JSON fallback extraction (`backend/controllers/ai.js:113-136`).
- **Inference**: Google Gemini through `aiService.chatCompletion()`.
- **Externalization**:
    - Full source text sent to Gemini (Channel 1).
    - Logged-in users' generated quiz and activity metadata stored in MongoDB (Channel 8).
    - Quiz JSON returned to the client (Channel 12).
- **Episode path**: Study text -> quiz prompt -> Gemini API -> JSON parse -> optional MongoDB storage -> HTTP response/UI.
- **Key files**: `backend/controllers/ai.js`, `backend/services/geminiService.js`, `backend/models/Quiz.js`
- **Confidence**: High

### Workflow 3: AI Tutor (`POST /api/ai/tutor`)
- **Purpose**: Answer a student's question using optional contextual study material.
- **Input**: `question`, optional `context`, optional `sourceId`.
- **Processing**:
    - `buildPrompt('tutor', ...)` embeds the question and context into a tutor prompt (`backend/controllers/ai.js:94-95`).
    - `parseResponse('tutor', ...)` returns direct text as `answer`.
- **Inference**: Google Gemini through `aiService.chatCompletion()`.
- **Externalization**:
    - Student question and context sent to Gemini (Channel 1).
    - Logged-in users' tutor activity metadata stored in MongoDB (Channel 9).
    - Answer returned to the client (Channel 12).
- **Episode path**: Question/context -> tutor prompt -> Gemini API -> answer parse -> optional activity logging -> HTTP response/UI.
- **Key files**: `backend/controllers/ai.js`, `backend/services/aiService.js`, `backend/services/geminiService.js`
- **Confidence**: High

### Workflow 4: Study Planner and Flashcard Generation
- **Purpose**: Generate weekly study schedules and flashcards from user-provided subjects/goals or study text.
- **Input**: `subjects`, `timeAvailable`, `goals` for study plans; `text`, `numCards`, source metadata for flashcards.
- **Processing**:
    - `buildPrompt('study-planner', ...)` and `buildPrompt('flashcards', ...)` request JSON outputs (`backend/controllers/ai.js:96-99`).
    - Responses are parsed into structured plan/flashcard objects.
- **Inference**: Google Gemini through `aiService.chatCompletion()`.
- **Externalization**:
    - Study goals, subject list, and/or source text sent to Gemini (Channel 1).
    - Activity records and flashcard decks stored for logged-in users (Channels 6 and 10; flashcard persistence at `backend/controllers/ai.js:659-678`).
    - Generated plans/flashcards returned to the client (Channel 12).
- **Episode path**: Study inputs -> structured generation prompt -> Gemini API -> JSON parse -> optional MongoDB storage -> HTTP response/UI.
- **Key files**: `backend/controllers/ai.js`, `backend/models/Flashcard.js`, `backend/models/UserActivity.js`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 13
- **Total number of main AI inference workflows found**: 4
- **Top 3 highest-risk workflows or channels**:
    1. **Gemini prompt submission (Channel 1)**: User study notes, uploaded file contents, YouTube transcripts, tutor questions, goals, and source text are inserted directly into prompts and sent to Google's Gemini API.
    2. **MongoDB persistence for logged-in users (Channels 7-10)**: Generated summaries, quiz questions, flashcards, activity records, IP address, user agent, usage metadata, and partial question text create a durable learning profile.
    3. **Multi-provider legacy/fallback surfaces (Channels 2-5)**: The codebase includes OpenAI, Gemini, and Grok-compatible request paths; if enabled, the same educational content can be routed to multiple third-party AI providers.
