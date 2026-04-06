# AI Inference Privacy Audit: xend

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Gmail API) | `backend/apps/mail/services.py` | - | `list_emails_logic` | Fetching user emails (subjects, bodies, attachments). | `googleapiclient.discovery.build("gmail", "v1", ...)` | High |
| 2 | Network Request (Redis) | `gpu-server/app/llm.py` | 108 | `generate_and_publish` | Streamed tokens/messages from the local LLM. | `redis.publish(channel, message)` at `redis://xend-fiveis-dev.duckdns.org:6379` | High |
| 3 | Storage Write (Django DB) | `backend/apps/ai/tasks.py` | 33-66, 126-129, 172-175 | `analyze_speech`, `unified_analysis` | Inferred style analysis (lexical, grammar, emotional tone). | `MailAnalysisResult.objects.create(...)`, `ContactAnalysisResult.objects.create(...)` | High |
| 4 | UI Rendering | `frontend/src/App.tsx` (assumed) | - | - | Analysis results and generated email replies. | Expected in a typical React frontend. | High |
| 5 | Logging | `gpu-server/app/llm.py` | 51, 60, 64, 73 | `stream_generate_reply` | LLM generation attempts and JSON parsing status. | `print(f"[DEBUG] Attempt {attempt}: {generated_text}")` | High |

## B. Main AI Inference Workflows

### Workflow 1: Individual Speech & Style Analysis
- **Purpose**: Analyze the writing style of sent emails to create a "digital twin" for the user.
- **Input**: Email subject and body (retrieved from the Gmail API via `SENT` label).
- **Processing**: Prompt construction for style extraction (lexical, grammar, tone).
- **Inference**: Local model inference on `gpu-server` (or similar service).
- **Externalization**: 
    - Fetched email content processed on the server.
    - Style results (lexical, grammar, tone, representative sentences) saved to Django DB (`tasks.py:33`).
- **Episode path**: Gmail API -> Style Extraction Prompt -> LLM Inference -> Django DB -> UI
- **Key files**: `tasks.py`, `services/analysis.py`, `llm.py`
- **Confidence**: High

### Workflow 2: Unified Contact/Group Style Analysis
- **Purpose**: Integrate multiple individual analysis results into a single, robust style profile for a contact or group.
- **Input**: Multiple individual style analysis results (from `MailAnalysisResult`).
- **Processing**: Grouping results; constructing an integration prompt.
- **Inference**: LLM call (`integrate_analysis`) to synthesize patterns.
- **Externalization**: 
    - Integrated style profile updated in `ContactAnalysisResult` or `GroupAnalysisResult` in Django DB (`tasks.py:126, 172`).
- **Episode path**: Multiple Style Results -> Integration Prompt -> LLM Inference -> Unified DB Entry -> UI
- **Key files**: `tasks.py`, `services/analysis.py`
- **Confidence**: High

### Workflow 3: AI-Powered Streaming Reply Generation
- **Purpose**: Generate a contextually and stylistically appropriate reply to an email.
- **Input**: User input and a system prompt (likely seeded with the user's analyzed style).
- **Processing**: Token-by-token generation using a chat template.
- **Inference**: Local inference using `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` model on `gpu-server`.
- **Externalization**: 
    - Generated tokens published to an external Redis server (`llm.py:108`).
    - Tokens streamed to the frontend for real-time display in the UI.
- **Episode path**: User Input + Style Context -> EXAONE Inference -> Redis Pub/Sub -> Frontend Streaming -> UI
- **Key files**: `llm.py`, `main.py`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 5
- **Total number of main AI inference workflows found**: 3
- **Top 3 highest-risk workflows or channels**:
    1. **Gmail API Access (Workflow 1)**: Grants the application full access to read and download the user's sent emails (subjects, bodies, and potentially attachments).
    2. **External Redis Pub/Sub (Channel 2)**: Transmits generated AI responses through a remote Redis server (`xend-fiveis-dev.duckdns.org`), which could be intercepted or logged by third parties.
    3. **Style Analysis Storage (Workflow 1/2)**: Creates a persistent "digital twin" of the user's communication style (lexical, emotional, grammar); if this profile is leaked, it could be used for highly convincing impersonation attacks.
