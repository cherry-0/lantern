# AI Inference Privacy Audit: edupal

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM) | `firebase_functions/functions/speech_utils.py` | 125–129 | `generate_bot_response` | User text + last N conversation turns + AI character system prompt sent to OpenAI GPT-3.5-turbo. Conversation history (up to all past messages) included via `all_msgs`. | `openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[system_msg, *past_msgs, user_msg])` | High |
| 2 | Network Request (Cloud STT) | `firebase_functions/functions/speech_utils.py` | 74–76 | `get_transcript` | Raw audio file (WAV converted from device recording) sent to OpenAI Whisper API for speech-to-text transcription. | `openai.Audio.transcribe("whisper-1", audio_file)` | High |
| 3 | Cloud Storage Write (Firestore) | `firebase_functions/functions/main.py` | 28–34, 71, 75 | `save_message_to_firestore` | Every user message and bot response written to the Firestore `messages` collection with fields: `message`, `side` (user/bot), `session_id`, `character`, `timestamp`. | `db.collection('messages').add(message_data)` — called after both transcription (line 71) and bot response (line 75) | High |
| 4 | Network Request (Cloud TTS) | `firebase_functions/functions/speech_utils.py` | 137–155 | `generate_speech` | Bot response text sent to ElevenLabs TTS API (`eleven_multilingual_v2`) to synthesize audio. Response returned as streaming audio chunks. | `requests.post("https://api.elevenlabs.io/v1/text-to-speech/<voice_id>", ...)` with `text=speech_content` in body | High |

## B. Main AI Inference Workflows

### Workflow 1: Voice-to-Bot-Response Pipeline (main conversation loop)
- **Purpose**: Power a child-facing AI character chatbot with voice input and audio output.
- **Input**: Audio recording from the child's device (AAC/PCM format).
- **Processing**:
  - Audio converted to WAV in memory (`aac_to_wav_in_memory`, `pcm16_to_wav`).
  - `get_transcript()` sends WAV to OpenAI Whisper (Channel 2); returns text transcript.
  - Transcript + character system prompt + conversation history fed into `generate_bot_response()` (Channel 1).
  - Bot response text fed into `generate_speech()` for ElevenLabs TTS (Channel 4).
  - Both user transcript and bot response written to Firestore (Channel 3).
- **Inference**: OpenAI GPT-3.5-turbo generates the character reply.
- **Externalization**:
  - Audio (potentially containing child's voice and statements) sent to Whisper API (Channel 2).
  - Transcript + prior conversation history sent to OpenAI GPT-3.5-turbo (Channel 1).
  - Both messages persisted in Firestore (Channel 3).
  - Bot reply text transmitted to ElevenLabs (Channel 4).
- **Episode path**: Device audio → Whisper STT → transcript text → GPT-3.5-turbo → bot reply → ElevenLabs TTS → audio; both messages → Firestore
- **Key files**: `firebase_functions/functions/speech_utils.py`, `firebase_functions/functions/main.py`
- **Confidence**: High

### Workflow 2: Character Persona Switching
- **Purpose**: Adjust the AI character's personality and language based on the `character` and `language` parameters.
- **Input**: `character` string (e.g., "Shiba Inu", "cat") + `language` string.
- **Processing**: `generate_bot_response()` selects a character-specific `system_prompt` and appends a language instruction if non-English.
- **Inference**: Same GPT-3.5-turbo call; system prompt content varies by character.
- **Externalization**:
  - Chosen character and language preference included in the system message sent to OpenAI (Channel 1).
  - Character and session metadata stored in each Firestore message document (Channel 3).
- **Episode path**: Character/language params → system prompt construction → GPT-3.5-turbo call → Firestore write
- **Key files**: `firebase_functions/functions/speech_utils.py` (lines 99–129)
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 4
- **Total number of main AI inference workflows found**: 2
- **Top 3 highest-risk workflows or channels**:
    1. **Voice-to-text via Whisper (Channel 2)**: Children's raw audio — which may include names, household sounds, location references, or emotionally sensitive statements — is sent to OpenAI's cloud STT API on every turn. Audio is among the highest-sensitivity personal data categories under children's privacy regulations (COPPA, GDPR-K).
    2. **Conversation history sent to GPT-3.5-turbo (Channel 1)**: The full accumulated conversation — including everything the child has said across prior turns — is transmitted to OpenAI on each request. Personal disclosures build up over sessions and all of them re-leave the device with every reply.
    3. **Firestore persistence (Channel 3)**: Every user turn and bot reply is stored indefinitely in a cloud Firestore collection with session and character metadata. This creates a long-term, queryable record of children's conversations in a third-party cloud database with no visible retention or deletion controls.
