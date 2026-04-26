# AI Inference Privacy Audit: klyr

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM) | `klyr/backend/main.py` | 17–20, 97 | `ask_gemini_json`, `gemini_analyze_resume` | Full resume text (up to 7,000 chars), skills list, job description, and bullet points sent to Google Gemini API (`gemini-2.5-flash`). | `model.generate_content(prompt)` where prompt embeds the raw user-supplied text | High |
| 2 | Network Request (LLM Self-repair retry) | `klyr/backend/main.py` | 113–132 | `ask_gemini_json` | Same resume/skill/JD content re-sent to Gemini in a second call if the first response is not valid JSON. | `retry = model.generate_content(repair_prompt)` — repair prompt includes the broken output plus the original data | High |
| 3 | Network Request (Bullet rewrite retry) | `klyr/backend/main.py` | 152–169 | `ask_gemini_bullet` | Bullet point text and role metadata re-sent to Gemini when the first rewrite is unparseable. | `retry = model.generate_content(repair_prompt)` in `ask_gemini_bullet` | High |
| 4 | UI Rendering (Android) | `klyr/android/` | — | — | ATS score, strengths, weaknesses, extracted skills, and improvement suggestions displayed in Jetpack Compose UI. | Android client consumes the FastAPI JSON response and renders it directly. | High |

## B. Main AI Inference Workflows

### Workflow 1: Resume ATS Analysis (`/analyze-resume`, `/analyze-resume-pdf`)
- **Purpose**: Grade a resume as an ATS engine + senior recruiter, surface strengths, weaknesses, and missing sections.
- **Input**: Raw resume text (string) or uploaded PDF file.
- **Processing**:
  - PDF path: `extract_text_from_pdf` (PyPDF2, local) → plain text.
  - `clean_text` truncates to 7,000 chars and collapses whitespace.
  - Full cleaned text embedded into `gemini_analyze_resume` prompt.
- **Inference**: Google Gemini (`gemini-2.5-flash`) returns structured JSON with `ats_score`, `strengths`, `weaknesses`, `skills`, `missing_sections`, `improvement_suggestions`.
- **Externalization**:
  - Resume text sent to `generativelanguage.googleapis.com` (Channel 1).
  - If JSON parse fails, a repair call re-sends the content (Channel 2).
  - Result displayed in Android UI (Channel 4).
- **Episode path**: Resume text/PDF → `clean_text` → Gemini API → JSON parse → (retry if needed) → Android UI
- **Key files**: `backend/main.py` (`gemini_analyze_resume`, `/analyze-resume`, `/analyze-resume-pdf`)
- **Confidence**: High

### Workflow 2: Skill Gap Analysis (`/skill-gap`)
- **Purpose**: Evaluate how well a candidate's skills match a target role and produce a learning roadmap.
- **Input**: List of resume skills (strings) + target role name.
- **Processing**: Skills joined with `, ` and embedded into `SKILL_GAP_PROMPT` template alongside the target role.
- **Inference**: Gemini returns match percentage, role readiness, matched/missing skills, and learning recommendations.
- **Externalization**:
  - Skills list and target role sent to Gemini API (Channel 1).
  - Repair retry if JSON is invalid (Channel 2).
  - Result rendered in Android UI (Channel 4).
- **Episode path**: Skills + role → `SKILL_GAP_PROMPT` → Gemini API → JSON parse → (retry) → Android UI
- **Key files**: `backend/main.py` (`ask_gemini_json`, `/skill-gap`, `SKILL_GAP_PROMPT`)
- **Confidence**: High

### Workflow 3: Job Description Match (`/jd-match`)
- **Purpose**: Score how well a resume aligns with a specific job description and surface missing ATS keywords.
- **Input**: Resume text + job description text (both truncated to 7,000 chars).
- **Processing**: Both texts embedded into `JD_MATCH_PROMPT`.
- **Inference**: Gemini returns match percentage, matched/missing keywords, ATS risks, and improvement tips.
- **Externalization**:
  - Both resume and JD text sent to Gemini API (Channel 1).
  - Repair retry if needed (Channel 2).
- **Episode path**: Resume + JD → `JD_MATCH_PROMPT` → Gemini API → JSON parse → Android UI
- **Key files**: `backend/main.py` (`ask_gemini_json`, `/jd-match`, `JD_MATCH_PROMPT`)
- **Confidence**: High

### Workflow 4: Resume Bullet Rewriter (`/rewrite-bullet`)
- **Purpose**: Rewrite a single resume bullet to be ATS-optimized for a target role.
- **Input**: Original bullet text + target role + experience level.
- **Processing**: All three fields embedded into `BULLET_REWRITE_PROMPT`.
- **Inference**: Gemini returns a rewritten bullet and explanation.
- **Externalization**:
  - Bullet, role, and experience level sent to Gemini API (Channel 1).
  - Repair retry via `ask_gemini_bullet` if JSON is unparseable (Channel 3).
- **Episode path**: Bullet + role + level → `BULLET_REWRITE_PROMPT` → Gemini API → (retry) → Android UI
- **Key files**: `backend/main.py` (`ask_gemini_bullet`, `/rewrite-bullet`, `BULLET_REWRITE_PROMPT`)
- **Confidence**: High

### Workflow 5: Resume Section Generator (`/generate-section`)
- **Purpose**: Generate a professional resume section (Summary, Experience, Skills, etc.) from structured inputs.
- **Input**: Section type + target role + experience level + skills list.
- **Processing**: All fields embedded into `SECTION_GENERATOR_PROMPT`.
- **Inference**: Gemini returns the generated section text as JSON.
- **Externalization**:
  - Role, experience level, and skills list sent to Gemini API (Channel 1).
  - Repair retry if needed (Channel 2).
- **Episode path**: Section params → `SECTION_GENERATOR_PROMPT` → Gemini API → (retry) → Android UI
- **Key files**: `backend/main.py` (`ask_gemini_json`, `/generate-section`, `SECTION_GENERATOR_PROMPT`)
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 4
- **Total number of main AI inference workflows found**: 5
- **Top 3 highest-risk workflows or channels**:
    1. **Resume ATS Analysis (Workflow 1)**: Sends the full resume — containing name, contact details, employment history, education, and personal skills — to Google Gemini on every request, plus a potential second call on JSON parse failure.
    2. **Job Description Match (Workflow 3)**: Transmits both the full resume and the target job description to Gemini, doubling the personal data exposure surface.
    3. **LLM Self-repair retry (Channel 2)**: The `ask_gemini_json` fallback embeds the raw original data inside a repair prompt and sends a second API call unconditionally when parsing fails, meaning sensitive content may be transmitted twice per request.
