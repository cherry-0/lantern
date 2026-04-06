# AI Inference Privacy Audit: snapdo

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (OpenAI) | `snapdo/services/vlm_service.py` | 101, 133, 226 | `generate_evidence`, `verify_evidence`, `infer_location` | Base64 encoded photos, task descriptions, and prompt instructions. | `ChatOpenAI(model="openai/gpt-4o-mini", ...).invoke(...)` | High |
| 2 | UI Rendering | `snapdo/views.py` (assumed) | - | - | Verification results (passed/failed), location inference details (country, city, POI, lat/lon), and generated action items. | Expected in a typical Django view. | Medium |
| 3 | Logging | `snapdo/services/vlm_service.py` | 104, 137, 138, 229 | `generate_evidence`, `verify_evidence`, `infer_location` | Evidence chunks, verification status, and error details. | `print('evidence', msg.evidence)`, `logger.error(...)` | High |

## B. Main AI Inference Workflows

### Workflow 1: Evidence Generation
- **Purpose**: Create a concrete, verifiable "Photo of..." action item for a given task.
- **Input**: User-provided task description (e.g., "Wash the dishes").
- **Processing**: Prompt construction to generate exactly one, single-line action (≤18 words).
- **Inference**: Remote call to OpenAI `gpt-4o-mini` via `LangChain`.
- **Externalization**: 
    - Task description sent to OpenAI API (`vlm_service.py:101`).
    - Generated evidence logged and displayed in UI.
- **Episode path**: User Task -> OpenAI API -> Evidence Generation -> UI
- **Key files**: `vlm_service.py`, `views.py`
- **Confidence**: High

### Workflow 2: Proof-of-Work Photo Verification
- **Purpose**: Automatically verify if a user's uploaded photo matches the task's completion constraints.
- **Input**: User's proof photo (base64) and the previously generated task constraint.
- **Processing**: Photo + constraint formatted into a vision prompt.
- **Inference**: Remote call to OpenAI `gpt-4o-mini` (Vision) via `LangChain`.
- **Externalization**: 
    - Photo sent to OpenAI API (`vlm_service.py:133`).
    - Verification verdict (PASSED/FAILED), confidence, and explanation displayed in UI.
- **Episode path**: User Photo -> OpenAI API (Vision) -> Verification Result -> UI
- **Key files**: `vlm_service.py`, `views.py`
- **Confidence**: High

### Workflow 3: Privacy-Aware Location Inference
- **Purpose**: Infer the location of a photo for additional verification context.
- **Input**: User's photo (base64) and optional textual hint.
- **Processing**: System instructions emphasize privacy and robust visual cues (signs, landmarks, vegetation).
- **Inference**: Remote call to OpenAI `gpt-4o-mini` (Vision) via `LangChain`.
- **Externalization**: 
    - Photo sent to OpenAI API (`vlm_service.py:226`).
    - Inferred location fields (country, city, POI, lat/lon, evidence cues, reasoning) displayed in UI.
- **Episode path**: User Photo -> OpenAI API (Vision) -> Location Inference -> UI
- **Key files**: `vlm_service.py`, `views.py`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 3
- **Total number of main AI inference workflows found**: 3
- **Top 3 highest-risk workflows or channels**:
    1. **Location Inference (Workflow 3)**: Automatically extracts precise location data (GPS coordinates, specific POIs) from user photos and sends them to a third-party AI provider.
    2. **Photo Verification (Workflow 2)**: Transmits full photos (which may contain PII or sensitive surroundings) to OpenAI for content analysis.
    3. **UI Rendering (Inferred Data)**: Displays potentially sensitive inferred location data; if this information is stored insecurely or shared, it could compromise user privacy.
