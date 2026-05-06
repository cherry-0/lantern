# AI Inference Privacy Audit: skin-disease-detection

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Email) | `Screens/Online_email.dart` | 31-36 | `send` | User message, subject, recipients (oncologists), and photo attachments. | `FlutterEmailSender.send(email)` | High |
| 2 | Network Request (WhatsApp) | `Screens/Online_whatsapp.dart` | 94 | `onPressed` | User message and recipient phone numbers (oncologists). | `FlutterOpenWhatsapp.sendSingleMessage(number, message)` | High |
| 3 | Storage Write (Firebase) | `Screens/database.dart` | 134-138 | `handleSubmit` | User name, email/contact number, and potentially message content. | `itemRef.push().set(item.toJson())` | High |
| 4 | Network Request (Google Maps API) | `Screens/Offline.dart` | 47-53 | `_onLocationChanged` (assumed) | User's latitude/longitude and nearby search parameters. | `http.get("https://maps.googleapis.com/maps/api/place/nearbysearch/json", ...)` | High |
| 5 | UI Rendering | `Screens/imgpick.dart` | 338-380 | `pickimagehere` | Local TFLite model prediction labels and confidence scores. | `Text(category1 != null ? category1.label : '...')` | High |
| 6 | Logging (Print) | `Screens/classifier.dart`, `imgpick.dart` | 94, 100, 301, 312, 323 | Various | Prediction status, inference timing, and result labels/probabilities. | `print('Time to run inference: $run ms')`, `print("Prediction 1 Running....")` | High |

## B. Main AI Inference Workflows

### Workflow 1: Local Skin Lesion Classification (TFLite)
- **Purpose**: Detect potential skin diseases (Acne, Melanoma, Cancer, etc.) from a user's photo.
- **Input**: User-provided photo (Camera or Gallery) and a response to the "ABCDE" mole characteristic form.
- **Processing**: Photo is decoded into an `img.Image`; input is preprocessed for TFLite; `Classifier` selection is based on the user's "Yes/No" response.
- **Inference**: Local inference using `tflite_flutter` on several different models (`quantized_pruned_model_tflite.tflite`, `quantized_pruned_model_melanoma.tflite`, etc.).
- **Externalization**: 
    - Results displayed in the UI (`imgpick.dart:338`).
    - Timing and status logged via `print()` statements.
- **Episode path**: User Photo -> ABCDE Form -> Classifier Selection -> TFLite Inference (Local) -> UI Rendering
- **Key files**: `classifier.dart`, `classifier1.dart`, `classifier2.dart`, `classifier3.dart`, `imgpick.dart`
- **Confidence**: High

### Workflow 2: Medical Appointment Assistance (Email/WhatsApp)
- **Purpose**: Connect users with skin disease specialists for further diagnosis.
- **Input**: User's descriptive message, contact info, and optionally their lesion photo.
- **Processing**: User selects an oncologist from a hardcoded list; data is bundled for external communication.
- **Inference**: N/A (Post-inference action).
- **Externalization**: 
    - Full payload (message + photo) sent to external platforms (Gmail, WhatsApp).
- **Episode path**: User Message -> Oncologist Selection -> Email/WhatsApp Intent -> External Platforms
- **Key files**: `Online_email.dart`, `Online_whatsapp.dart`, `imgpick.dart`
- **Confidence**: High

### Workflow 3: Location-Based Specialist Search (Google Maps API)
- **Purpose**: Find nearby specialists or clinics based on the user's current location.
- **Input**: User's latitude and longitude coordinates.
- **Processing**: Location data used in a "nearby search" query for Google Places API.
- **Inference**: N/A (Pre-inference search).
- **Externalization**: 
    - Precise GPS coordinates sent to Google Maps servers via an HTTP GET request.
- **Episode path**: Location Change -> Google Maps API Call -> Nearby Results List -> UI
- **Key files**: `Offline.dart`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 6
- **Total number of main AI inference workflows found**: 3
- **Top 3 highest-risk workflows or channels**:
    1. **Medical Data Sharing (Workflow 2)**: Transmits highly sensitive personal information (name, health concerns) and photos of potentially diseased skin to external third-party communication platforms (Gmail, WhatsApp).
    2. **GPS Coordinate Leakage (Workflow 3)**: Automatically sends the user's precise geolocation to Google Maps API for a nearby search, which could be used to track the user's movements.
    3. **Firebase Realtime Database (Channel 3)**: Stores user contact info and inquiry history in a cloud database; if not properly secured, this centralizes PII on a remote server.
