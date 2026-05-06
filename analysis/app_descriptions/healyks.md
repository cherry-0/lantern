# AI Inference Privacy Audit: healyks

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Backend HTTP POST, Firebase-authenticated) | `app/src/main/java/com/healyks/app/data/remote/AnalyzeApi.kt` | 11–17 | `analyzeSymptom` | Free-text symptom description (`GeminiBody.symptoms`) sent as JSON body to `http://ec2-13-232-188-167.ap-south-1.compute.amazonaws.com:5000/api/symptoms/analyze` with a Firebase ID token in the `Authorization` header — directly identifying the user. | `@POST(Constants.ANALYZE_ENDPOINT) suspend fun analyzeSymptom(@Header("Authorization") accessToken: String, @Body geminiBody: GeminiBody)` where `ANALYZE_ENDPOINT = "api/symptoms/analyze"` (`Constants.kt:7`) and the URL is taken from `BuildConfig.BASE_URL` (`AppModule.kt:56`). | High |
| 2 | Network Request (Cloud LLM — relayed by backend) | (backend, AWS EC2 Express.js service) | — | Backend `POST /api/symptoms/analyze` handler | The `symptoms` string is forwarded by the AWS-hosted Express.js backend to Google Gemini 2.0 Flash for symptom-to-condition inference, returning `{condition, recommendation, homeRemedies}`. | README: "AI Integration: Google Gemini API"; UI label `"powered by Gemini 2.0 flash"` (`AnalyzeScreen.kt:314`); response model `GeminiResponse(condition, recommendation, homeRemedies)` (`GeminiResponse.kt:5–13`); adapter description `model="google/gemini-2.0-flash-001"` (`verify/backend/adapters/healyks.py:21`). | High |
| 3 | Network Request (User Profile / Health Body — Firebase-authenticated) | `app/src/main/java/com/healyks/app/data/remote/UserApi.kt` | 22–26 | `postUserBody` / `getUserDetails` | Full medical profile `UserDetails` (age, gender, blood group, height, weight, allergies list, chronic diseases list, medications list, lifestyle: alcohol/smoking/physicalActivity, email) POSTed to `api/user/postUserBody` and re-fetched via `api/user/details` with Firebase ID token. | `UserDetails` fields (`UserDetails.kt:5–35`) include `chronicDiseases`, `medications`, `allergies`, `bloodGroup`; `Lifestyle.kt:5–14` adds `alcohol`, `smoking`, `physicalActivity`; `userApi.postUserBody(accessToken, postUserBody)` in `UserRepo.kt:91–92`. | High |
| 4 | Authentication / Identity (Firebase Auth) | `app/src/main/java/com/healyks/app/data/repo/UserRepo.kt` | 21–36 | `getIdToken` | Firebase ID token (signed JWT containing Firebase UID, email, issuer, audience) is minted and attached to every analyze/profile request, tying every symptom string to a Google-Firebase-identified user. | `currentUser.getIdToken(true).await()`; `"Bearer $token"`; `providesFirebaseAuth(): FirebaseAuth = FirebaseAuth.getInstance()` (`AppModule.kt:81`). | High |
| 5 | Logging (Logcat — symptom payload + response + ID token) | `app/src/main/java/com/healyks/app/vm/AnalyzeViewModel.kt`; `app/src/main/java/com/healyks/app/data/repo/UserRepo.kt` | `AnalyzeViewModel.kt:33,36`; `UserRepo.kt:27` | `postSymptom`, `getIdToken` | Full Gemini API response (condition / recommendation / homeRemedies) and the raw Firebase ID token are written to Android Logcat, which is readable by other privileged apps / debuggers and may be picked up by crash-reporting SDKs in production builds. | `Log.d("AnalyzeViewModel", "API Response: $response")`, `Log.d("AnalyzeViewModel", "Symptom analysis successful: ${response.data}")`, `Log.d("AuthRepo", token)`. | High |
| 6 | Logging (HTTP body interceptor — request + response bytes) | `app/src/main/java/com/healyks/app/di/AppModule.kt` | 32–48 | `provideOkHttpClient` | OkHttp `HttpLoggingInterceptor` set to `Level.BODY` in DEBUG builds writes every analyze/user-profile request and response (symptoms text, Authorization header, full medical profile) to Logcat. | `level = if (BuildConfig.DEBUG) HttpLoggingInterceptor.Level.BODY else HttpLoggingInterceptor.Level.NONE`. | High |
| 7 | UI Rendering (Analyze screen) | `app/src/main/java/com/healyks/app/view/screens/AnalyzeScreen.kt` | 95–174, 281–289 | `AnalyzeScreen` | The user's symptom text (input field) and the Gemini-returned `condition`, `recommendation`, `homeRemedies` are rendered as Composable `Text` cards on screen — visible to anyone shoulder-surfing or to screen-recorder/accessibility services. | `CustomTextField(value = symptoms.value, ...)`, `Text(text = postSymptomState.data.data?.condition ?: "")`, similar `Text` blocks for `recommendation` and `homeRemedies`. | High |
| 8 | Local Persistence (Room — period cycle entries) | `app/src/main/java/com/healyks/app/data/local/periods/CycleDatabase.kt`; `app/src/main/java/com/healyks/app/data/model/Cycle.kt` | `CycleDatabase.kt:12–35`; `Cycle.kt:7–13` | `CycleDatabase`, `Cycle` entity | Menstrual `cycle_table` (start/end dates, cycle length) stored in Room SQLite DB `cycle_database` on device — sensitive reproductive-health data persisted unencrypted in the app sandbox. | `@Entity(tableName = "cycle_table") data class Cycle(... startDate: Date, endDate: Date, cycleLength: Int)`. | High |
| 9 | Local Persistence (Room — medication reminders) | `app/src/main/java/com/healyks/app/data/local/reminder/ReminderDatabase.kt`; `app/src/main/java/com/healyks/app/data/model/Reminder.kt` | `ReminderDatabase.kt:9–31`; `Reminder.kt:7–14` | `ReminderDatabase`, `Reminder` entity | Medication name, dosage, time, repeat flag stored in Room SQLite DB `reminder_database` on device — drug-name list reveals chronic conditions. | `@Entity(tableName = "reminders") data class Reminder(... name: String, dosage: String, timeInMillis: Long, ...)`. | High |
| 10 | Local Persistence (SharedPreferences `user_details`) | `app/src/main/java/com/healyks/app/di/AppModule.kt` | 84–86 | `provideSharedPreferences` | A `user_details` SharedPreferences file is provisioned for caching user info on device (readable by the app process and any backup mechanism). | `context.getSharedPreferences("user_details", Context.MODE_PRIVATE)`. | Medium |
| 11 | Cleartext HTTP Transport | `app/src/main/AndroidManifest.xml`; `app/src/main/res/xml/network_security_config.xml`; README | Manifest `:10`; README:66 | Network security config | The configured `BASE_URL` is `http://ec2-13-232-188-167.ap-south-1.compute.amazonaws.com:5000/` — plain HTTP, allowed by the app's `networkSecurityConfig`. Symptom text + Firebase ID token + full medical profile are transmitted over an unencrypted channel and exposed to any on-path observer (carrier, Wi-Fi, ISP). | `android:networkSecurityConfig="@xml/network_security_config"`; README "BASE_URL = \"http://ec2-13-232-188-167...\"". | High |
| 12 | Backup / Data-Extraction surface | `app/src/main/AndroidManifest.xml`; `app/src/main/res/xml/backup_rules.xml`, `data_extraction_rules.xml` | Manifest:13–15 | Application backup config | `android:allowBackup="true"` plus `fullBackupContent`/`dataExtractionRules` allow the Room cycle DB, reminder DB, and `user_details` prefs to be exfiltrated via `adb backup` or device-transfer flows on debuggable builds. | `android:allowBackup="true" android:dataExtractionRules="@xml/data_extraction_rules" android:fullBackupContent="@xml/backup_rules"`. | Medium |

## B. Main AI Inference Workflows

### Workflow 1: Symptom Analysis (`AnalyzeViewModel.postSymptom` → `/api/symptoms/analyze` → Gemini 2.0 Flash)
- **Purpose**: Take a user-typed natural-language description of symptoms and return a structured health report (`condition`, `recommendation`, `homeRemedies`) for a rural / under-served user who may not have local clinical access.
- **Input**: A single string typed into the `CustomTextField` on `AnalyzeScreen` (e.g. "fever, headache, vomiting since yesterday"), wrapped as `GeminiBody(symptoms = ...)`.
- **Processing**:
  - `AnalyzeScreen` calls `analyzeViewModel.postSymptom(GeminiBody(symptoms.value))` (`AnalyzeScreen.kt:295`).
  - `AnalyzeViewModel` first calls `userRepo.getIdToken()` — forces a Firebase token refresh — then `analyzeRepo.analyzeSymptom(token, geminiBody)` (`AnalyzeViewModel.kt:32`).
  - `AnalyzeRepo` invokes the Retrofit `AnalyzeApi.analyzeSymptom` with `Authorization: Bearer <Firebase ID token>` and a Gson-serialized `{"symptoms": "..."}` body (`AnalyzeRepo.kt:25–26`).
  - Retrofit/OkHttp pipeline (with `HttpLoggingInterceptor` set to `BODY` in DEBUG) sends the request to `BuildConfig.BASE_URL` + `api/symptoms/analyze`.
  - The AWS EC2 Express.js backend forwards the symptoms string to Google Gemini 2.0 Flash and returns a `CustomResponse<GeminiResponse>`.
- **Inference**: Gemini 2.0 Flash returns JSON `{condition, recommendation, homeRemedies}` (typed as `GeminiResponse`, `GeminiResponse.kt:5–13`).
- **Externalization**:
  - Symptom string + Firebase ID token (identifying the user) leave the device over plain HTTP to the AWS backend (Channels 1, 4, 11).
  - Backend then transmits the symptom text to Google's Gemini API (Channel 2).
  - Full request/response, including the symptom text, condition, remedies, and the bearer token, is written to Logcat (Channels 5, 6).
  - Returned `condition`, `recommendation`, `homeRemedies` are rendered as on-screen text cards on the Analyze screen (Channel 7).
- **Episode path**: User types symptoms → `AnalyzeScreen` → `AnalyzeViewModel.postSymptom` → `UserRepo.getIdToken` (Firebase) → `AnalyzeRepo.analyzeSymptom` → `AnalyzeApi` (Retrofit) → HTTP POST to EC2 backend → Gemini 2.0 Flash → JSON report → `UiState.Success` → Compose `Text` cards on Analyze screen.
- **Key files**:
  - `app/src/main/java/com/healyks/app/view/screens/AnalyzeScreen.kt` (`AnalyzeScreen`, lines 49–323)
  - `app/src/main/java/com/healyks/app/vm/AnalyzeViewModel.kt` (`postSymptom`, lines 28–44)
  - `app/src/main/java/com/healyks/app/data/repo/AnalyzeRepo.kt` (`analyzeSymptom`, lines 17–41)
  - `app/src/main/java/com/healyks/app/data/remote/AnalyzeApi.kt` (`analyzeSymptom`, lines 11–17)
  - `app/src/main/java/com/healyks/app/data/model/GeminiBody.kt`, `GeminiResponse.kt`
  - `app/src/main/java/com/healyks/app/data/util/Constants.kt` (`ANALYZE_ENDPOINT`, line 7)
  - `app/src/main/java/com/healyks/app/data/repo/UserRepo.kt` (`getIdToken`, lines 21–36)
  - `app/src/main/java/com/healyks/app/di/AppModule.kt` (Retrofit + OkHttp + Firebase wiring, lines 30–81)
- **Confidence**: High

### Workflow 2: Medical Profile Submission (`UserViewModel.postUser` → `/api/user/postUserBody`)
- **Purpose**: Persist a per-user medical profile on the backend (used by the dashboard and intended to enrich the symptom-analysis context server-side).
- **Input**: A `UserDetails` object built on the `PostUserBodyScreen` containing `age`, `gender`, `bloodGroup`, `height`, `weight`, `allergies: List<String>`, `chronicDiseases: List<String>`, `medications: List<String>`, `email`, and a `Lifestyle` substructure (`alcohol: Boolean`, `smoking: Boolean`, `physicalActivity: String?`).
- **Processing**:
  - `UserViewModel.postUser(postUserBody)` (`UserViewModel.kt:59–75`) calls `userRepo.getIdToken()` and then `userRepo.postUserBody(token, postUserBody)`.
  - `UserRepo.postUserBody` invokes Retrofit `userApi.postUserBody(accessToken, postUserBody)` to POST `api/user/postUserBody` (`UserRepo.kt:91–92`, `UserApi.kt:22–26`).
  - The same `getUserDetails` GET (`api/user/details`) re-fetches the saved profile (`UserApi.kt:17–20`).
- **Inference**: Not strictly an LLM call, but the profile is the standing context the symptom-analysis endpoint can read on the server side, and is therefore part of the AI workflow's data plane.
- **Externalization**:
  - Full medical profile transmitted over plain HTTP with Firebase ID token (Channels 3, 4, 11).
  - Request/response logged via OkHttp body interceptor in DEBUG (Channel 6) and stored in `user_details` SharedPreferences locally (Channel 10).
  - Profile fields rendered on `ProfileScreen` and `DashboardDetailScreen` (Channel 7).
- **Episode path**: User fills `PostUserBodyScreen` → `UserViewModel.postUser(UserDetails)` → `UserRepo.getIdToken` (Firebase) → `UserRepo.postUserBody` → `UserApi.postUserBody` → HTTP POST to EC2 backend → MongoDB persistence → re-fetched via `getUserDetails`.
- **Key files**:
  - `app/src/main/java/com/healyks/app/view/screens/PostUserBodyScreen.kt`
  - `app/src/main/java/com/healyks/app/vm/UserViewModel.kt` (`postUser`, lines 59–75)
  - `app/src/main/java/com/healyks/app/data/repo/UserRepo.kt` (`postUserBody`, `getUserDetails`, lines 61–106)
  - `app/src/main/java/com/healyks/app/data/remote/UserApi.kt` (lines 11–26)
  - `app/src/main/java/com/healyks/app/data/model/UserDetails.kt`, `Lifestyle.kt`
  - `app/src/main/java/com/healyks/app/data/util/Constants.kt` (`POSTUSERBODY_ENDPOINT`, `USERDETAILS_ENDPOINT`)
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 12
- **Total number of main AI inference workflows found**: 2
- **Top 3 highest-risk workflows or channels**:
    1. **Symptom Analysis over plain HTTP (Workflow 1 / Channels 1, 2, 11)**: Free-text symptom descriptions — among the most sensitive personal data (mental-health complaints, sexual / reproductive symptoms, substance issues, infectious-disease exposure) — are sent unencrypted (HTTP, port 5000) to a hard-coded EC2 host along with a Firebase ID token that resolves to a specific Google identity. Any on-path observer (carrier, public Wi-Fi, ISP) can read the full symptom string and bind it to the user, and the backend then re-emits the symptom text to Google Gemini 2.0 Flash for inference. The combination of plaintext transport, identifying auth header, and third-party LLM hand-off concentrates several severe medical-privacy failures into one round-trip.
    2. **Standing Medical Profile (Channel 3 / Workflow 2)**: The app submits the entire `UserDetails` profile — chronic diseases, current medications, allergies, blood group, lifestyle (alcohol / smoking / activity) — over the same plain-HTTP Firebase-authenticated channel. This profile, persisted in MongoDB on the EC2 instance, is far more revealing than any single symptom string and creates a long-lived server-side dossier that the symptom-analysis endpoint can correlate with each Gemini call.
    3. **Verbose Logging of Symptom Payloads, Gemini Responses, and Firebase ID Tokens (Channels 5, 6)**: `AnalyzeViewModel` writes the full Gemini response to Logcat, `UserRepo` logs the raw Firebase bearer token, and `AppModule` enables OkHttp `HttpLoggingInterceptor` at `Level.BODY` for any DEBUG build. Combined with `android:allowBackup="true"` (Channel 12) and unencrypted local Room/Prefs stores (Channels 8–10), this means the user's symptoms, conditions, remedies, and authentication credentials can be captured by other privileged processes, debuggers, ADB backup, or any crash-reporting SDK that scrapes Logcat.
