# AI Inference Privacy Audit: nom-ai

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (NomAI backend chat/agent) | `lib/app/modules/Chat/Controllers/ChatController.dart` | 438-462 | `_sendToApi` | User message, user ID, local message ID, image URL, timestamp, dietary preferences, allergies, and selected health goals sent to `/api/v1/chat/messages`. | `http.post(url, body: jsonEncode(body))` where body includes `text`, `user_id`, `image_url`, `dietary_preferences`, `allergies`, `selected_goals` | High |
| 2 | Network Request (NomAI backend nutrition analysis) | `lib/app/repo/meal_ai_repo.dart` | 10-25 | `getNutritionData` | Food image URL or food description plus dietary preferences, allergies, and goals sent to `/api/v1/nutrition/analyze`. | `http.post(Uri.parse(baseUrl), body: jsonEncode(inputQuery.toJsonForMealAIBackend()))` | High |
| 3 | Cloud Storage Write (Firebase Storage) | `lib/app/repo/storage_service.dart` | 9-27 | `uploadImage` | Selected meal photo uploaded to Firebase Storage under `uploads/<timestamp>.png`; backend receives the resulting download URL. | `ref.putData(bytes)` / `ref.putFile(File(imageFile.path))` then `getDownloadURL()` | High |
| 4 | Device Image Acquisition | `lib/app/modules/Chat/Controllers/ChatController.dart` | 122-158 | `pickImageFromGallery`, `pickImageFromCamera`, `_processSelectedImage` | User-selected or camera-captured meal image, optionally downscaled before upload. | `ImagePicker().pickImage(...)`, `ImageUtility.downscaleImage(...)`, `uploadImage(selectedImage.value!)` | High |
| 5 | UI Rendering and Local State | `lib/app/modules/Chat/Controllers/ChatController.dart` | 166-227 | `sendMessage` | User message, image URL, assistant answer, and structured nutrition data stored in in-memory chat state and rendered by the chat UI. | `messages.add(ChatMessage(...))` for user and assistant messages | High |
| 6 | Network Request (Diet planning) | `lib/app/repo/diet_repo.dart` | 9-31 | `createWeeklyDiet` | User ID, macro targets, dietary preferences, allergies, goals, and diet constraints sent to `/api/v1/diet`. | `http.post(... ApiPath.createDiet, body: jsonEncode(input.toJson()))` | High |
| 7 | Network Request (Diet history / active diet) | `lib/app/repo/diet_repo.dart` | 41-93 | `getActiveDiet`, `getDietHistory` | User ID sent to fetch active and historical diet plans from the backend. | `http.get(... /api/v1/diet/$userId...)` | High |
| 8 | Network Request (Meal alternatives) | `lib/app/repo/diet_repo.dart` | 96-180 | `suggestAlternate`, `suggestAlternatives` | Current meal nutrition object, meal type, prompt, dietary preferences, allergies, diseases, and goals sent for AI meal alternatives. | `http.post(... suggest-alternate(s), body: jsonEncode({...}))` | High |
| 9 | Network Request (Backend internal AI providers) | `README.md` | 130-158, 186-188 | Backend AI gateway | Backend forwards image/text food context to Gemini or OpenRouter and performs Exa/DuckDuckGo web searches for grounding. | README architecture lists `LLM Provider (Gemini / OpenRouter)` and `Exa / DuckDuckGo`; 3-step pipeline includes web grounding and multimodal synthesis | Medium |
| 10 | Cloud Database Persistence (Firestore / backend DB) | `README.md`, `lib/app/modules/Chat/Controllers/ChatController.dart` | 22-24, 93-124, 260-313 | Backend persistence, `addToLog` | Chat history, diet plans, and nutrition records persisted by backend/Firestore; client can add AI nutrition output to daily logs. | README states Firestore persistence; `addToLog` builds `NutritionRecord` and saves via `NutritionRecordRepo.saveNutritionData(...)` | High |
| 11 | Logging | `lib/app/modules/Chat/Controllers/ChatController.dart`, `lib/app/repo/diet_repo.dart`, `lib/app/repo/meal_ai_repo.dart`, `lib/app/repo/storage_service.dart` | 82, 149, 230, 27-39, 11-21, 156-175, 28-29 | Various | Backend errors, image downscaling errors, failed HTTP bodies, diet input JSON, diet response bodies, alternative prompts, and upload errors printed/debug-logged. | `print(...)`, `debugPrint(...)` across request and error paths | Medium |

## B. Main AI Inference Workflows

### Workflow 1: Multimodal Nutrition Chat (`POST /api/v1/chat/messages`)
- **Purpose**: Let the user ask nutrition/health questions and optionally attach a meal photo for analysis by the backend agent.
- **Input**: User text, optional Firebase Storage image URL, user ID, timestamp, dietary preferences, allergies, and selected health goals.
- **Processing**:
    - User chooses or captures an image with `ImagePicker` (`ChatController.dart:122-130`).
    - Mobile image is downscaled and uploaded to Firebase Storage (`ChatController.dart:136-158`, `storage_service.dart:9-27`).
    - `sendMessage()` creates a local user message, then `_sendToApi()` posts the full request body to `/api/v1/chat/messages` (`ChatController.dart:166-200`, `438-462`).
    - Backend agent may route to image analysis, food-description analysis, web search, and LLM synthesis according to the README architecture.
- **Inference**: Backend LLM provider: Gemini or OpenRouter, with tool use and web-grounded nutrition analysis.
- **Externalization**:
    - Meal photo leaves the device through Firebase Storage (Channel 3).
    - Text, image URL, dietary preferences, allergies, and goals sent to backend (Channel 1).
    - Backend may send derived image/text prompts to LLM provider and search engine (Channel 9).
    - AI answer and nutrition data displayed in chat UI and stored by backend (Channels 5, 10).
- **Episode path**: camera/gallery image + text -> Firebase Storage URL -> `/api/v1/chat/messages` -> backend ReAct agent/tools -> Gemini/OpenRouter + web search -> `ai_answer` + `nutrition_data` -> chat UI/logs.
- **Key files**: `ChatController.dart`, `storage_service.dart`, `urls.dart`
- **Confidence**: High

### Workflow 2: Direct Nutrition Analysis (`POST /api/v1/nutrition/analyze`)
- **Purpose**: Analyze a food image or text food description and return structured nutrition facts.
- **Input**: `imageUrl`, `food_description`, dietary preferences, allergies, and selected goals.
- **Processing**:
    - `NutritionInputQuery.toJsonForMealAIBackend()` builds the backend request body.
    - `AiRepository.getNutritionData()` posts that body to `/api/v1/nutrition/analyze`.
    - README describes a 3-step pipeline: food identification, web grounding, and multimodal synthesis.
- **Inference**: Gemini/OpenRouter backend model plus Exa/DuckDuckGo grounding.
- **Externalization**:
    - Image URL or food description sent to backend (Channel 2).
    - Backend forwards relevant image/text and web facts to LLM provider (Channel 9).
    - Structured nutrition result returned to client and can be stored as a daily nutrition record (Channel 10).
- **Episode path**: food image URL or description -> `/api/v1/nutrition/analyze` -> food extraction -> search grounding -> LLM synthesis -> nutrition JSON -> UI/log.
- **Key files**: `meal_ai_repo.dart`, `nutrition_input.dart`, `nutrition_output.dart`, `README.md`
- **Confidence**: High

### Workflow 3: Weekly Diet Planning (`POST /api/v1/diet`)
- **Purpose**: Generate a personalized weekly meal plan using macro targets, diet preferences, allergies, and goals.
- **Input**: `DietInput` containing user ID and nutrition planning parameters.
- **Processing**:
    - `DietRepo.createWeeklyDiet()` logs input JSON and posts it to `/api/v1/diet`.
    - README describes carb-cycling logic and a 7-day generation loop with LLM calls.
- **Inference**: Backend LLM provider for daily meal generation and variety tracking.
- **Externalization**:
    - User profile/macros/preferences/goals sent to backend (Channel 6).
    - Backend may forward diet context to LLM provider (Channel 9).
    - Diet plan stored and retrieved through backend/Firestore (Channels 7, 10).
- **Episode path**: profile/macro targets -> `/api/v1/diet` -> backend target calculator + day prompts -> LLM provider -> weekly plan -> Firestore/backend DB -> UI.
- **Key files**: `diet_repo.dart`, `diet_input.dart`, `diet_output.dart`, `README.md`
- **Confidence**: High

### Workflow 4: Meal Alternatives and Log Updates
- **Purpose**: Suggest substitute meals and update nutrition logs from chat/diet outputs.
- **Input**: Current meal object, meal type, optional user prompt, dietary preferences, allergies, diseases, and goals.
- **Processing**:
    - `suggestAlternate()` and `suggestAlternatives()` POST current meal context and user constraints to backend endpoints.
    - `addToLog()` converts AI nutrition output into a `NutritionRecord`, merges it into the user's daily records, and saves through `NutritionRecordRepo`.
- **Inference**: Backend LLM provider for alternative meal suggestions.
- **Externalization**:
    - Current meal and health constraints sent to backend (Channel 8).
    - Alternative generation may call Gemini/OpenRouter (Channel 9).
    - Daily meal record persisted through backend/Firestore (Channel 10).
- **Episode path**: current meal + constraints -> alternatives endpoint -> LLM provider -> alternatives -> optional daily log update.
- **Key files**: `diet_repo.dart`, `ChatController.dart`, `nutrition_record.dart`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 11
- **Total number of main AI inference workflows found**: 4
- **Top 3 highest-risk workflows or channels**:
    1. **Meal image and profile context to backend/LLM (Workflows 1-2)**: Meal photos, food descriptions, allergies, dietary preferences, and weight/health goals are transmitted to NomAI's backend and then potentially to Gemini/OpenRouter for analysis.
    2. **Firebase Storage and backend persistence (Channels 3 and 10)**: Uploaded food images, chat history, diet plans, and nutrition logs become durable cloud records tied to the user.
    3. **Web-grounded analysis (Channel 9)**: Backend search queries to Exa/DuckDuckGo can reveal inferred foods, brands, dietary constraints, and health goals even before the LLM synthesis step.
