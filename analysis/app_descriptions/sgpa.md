# AI Inference Privacy Audit: SGPA (Study Guide & Personal Assistant)

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM) | `utils/gemini_helper.py` | 26–31 | `generate_response` | Full prompt including user's topic, notes, exam questions, or pasted PDF text sent to Google Gemini 2.5 Flash API. | `model.generate_content(prompt)` where prompt embeds raw user-supplied text across all three modes | High |
| 2 | Storage Write (CSV analytics log) | `utils/logger.py` | 64–88 | `log_usage` | Per-request row appended to `logs/usage_log.csv`: `session_id` (UUID), `mode`, `sub_mode`, `topic` (first 50 chars of prompt), `had_pdf`, `prompt_chars`, `response_chars`, `visuals_enabled`, `visuals_detected`, `visuals_used` | `writer.writerow(row)` in `log_usage`; called from `components/chat_ui.py:105` after every inference | High |
| 3 | UI Rendering (Streamlit) | `components/chat_ui.py` | — | chat response handler | Full AI response rendered in Streamlit chat interface; also includes the user's original prompt as a chat bubble | Streamlit `st.chat_message` + `st.write` with `response` content | High |

## B. Main AI Inference Workflows

### Workflow 1: Explainer (`💡 Explainer` mode)
- **Purpose**: Explain an academic concept with definition, breakdown, misconceptions, and key takeaways.
- **Input**: Free-text concept or question typed by the user.
- **Processing**: `explain_concept(concept, previous_context)` in `core/explainer.py` builds a Study Buddy prompt embedding the user text and optional previous chat context.
- **Inference**: Google Gemini 2.5 Flash (`models/gemini-2.5-flash`) via `generate_response()`.
- **Externalization**:
  - User concept text sent to Gemini API (Channel 1).
  - Usage row (topic snippet, prompt/response lengths) appended to `logs/usage_log.csv` (Channel 2).
  - Response rendered in Streamlit chat (Channel 3).
- **Episode path**: User text → `explain_concept` → `generate_response` → Gemini API → Streamlit chat + CSV log
- **Key files**: `core/explainer.py`, `utils/gemini_helper.py`, `utils/logger.py`, `components/chat_ui.py`
- **Confidence**: High

### Workflow 2: Summarizer (`📄 Summarizer` mode)
- **Purpose**: Produce an exam-ready summary from pasted notes or an uploaded PDF.
- **Input**: Plain text or PDF file (extracted via PyPDF2).
- **Processing**: `summarize_text(text, ...)` in `core/summarizer.py`; PDF extraction happens locally. The full extracted text (no truncation shown in code) is embedded into the prompt.
- **Inference**: Gemini 2.5 Flash.
- **Externalization**:
  - Full note/PDF text sent to Gemini API (Channel 1). `had_pdf=1` flag logged.
  - Usage row appended to `logs/usage_log.csv` (Channel 2).
  - Summary rendered in Streamlit chat (Channel 3).
- **Episode path**: Notes / PDF → PyPDF2 extract → `summarize_text` → Gemini API → Streamlit chat + CSV log
- **Key files**: `core/summarizer.py`, `core/pdf_handler.py`, `utils/gemini_helper.py`
- **Confidence**: High

### Workflow 3: Quizzer — Generate Questions
- **Purpose**: Generate MCQ / T-F / fill-in-the-blank / descriptive questions with an answer key.
- **Input**: Topic, chapter, or passage text.
- **Processing**: `generate_questions(text, previous_context)` in `core/quizzer.py`.
- **Inference**: Gemini 2.5 Flash.
- **Externalization**: Topic text → Gemini API (Channel 1); CSV log (Channel 2); quiz rendered in chat (Channel 3).
- **Confidence**: High

### Workflow 4: Quizzer — Solve Questions
- **Purpose**: Generate concise exam-ready answers for pasted questions.
- **Input**: Exam question text, optional word-limit constraint.
- **Processing**: `solve_questions(user_questions, ...)` in `core/quizzer.py`.
- **Externalization**: Question text → Gemini API (Channel 1); CSV log (Channel 2); answers rendered (Channel 3).
- **Confidence**: High

### Workflow 5: Quizzer — Evaluate Answers
- **Purpose**: Score user-submitted answers and provide detailed feedback.
- **Input**: Questions and user answers (separated by `---`).
- **Processing**: Evaluate sub-mode in `core/quizzer.py`; both questions and answers embedded in prompt.
- **Externalization**: Full Q&A text → Gemini API (Channel 1); CSV log (Channel 2); feedback rendered (Channel 3).
- **Confidence**: High

## C. Verify Adapter Coverage

| Channel | Real app behaviour | Serverless adapter captures? |
|---|---|---|
| NETWORK | POST to `generativelanguage.googleapis.com` with full user prompt | ✅ Yes — `realistic_fallback["NETWORK"]` describes Gemini API call with truncated input |
| STORAGE | Appends row to `logs/usage_log.csv` with session_id, topic snippet, prompt/response lengths | ✅ Yes (patched) — `realistic_fallback["STORAGE"]` describes CSV append with actual `len(text)` and `len(raw_response)` |
| UI | Streamlit chat renders full response | ✅ Yes — `realistic_fallback["UI"]` contains truncated response preview |

## Final Summary
- **Distinct externalization sites**: 3
- **Main inference workflows**: 5
- **Top risks**:
  1. **Summarizer (Workflow 2)**: Sends full PDF / notes text (potentially dozens of pages of sensitive academic/personal content) to Gemini on every request.
  2. **Quizzer Evaluate (Workflow 5)**: Both the exam questions and the user's own written answers are sent to Gemini — capturing academic performance data.
  3. **CSV analytics log (Channel 2)**: Persists a topic snippet (first 50 chars of every prompt), session UUID, mode, and size metadata to a local CSV for every inference — lightweight but continuous local accumulation tied to a stable session ID.
