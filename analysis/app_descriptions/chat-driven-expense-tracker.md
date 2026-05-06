# AI Inference Privacy Audit: chat-driven-expense-tracker

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM) | `backend/langchain_prompt.py` | 27–34, 44 | `parse_expense` | Raw user expense text (free-form natural language, any language) embedded in a prompt and sent to Groq's `llama3-8b-8192` cloud LLM via LangChain for structured parsing. | `llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama3-8b-8192", temperature=0.1, max_tokens=1000)` and `response = chain.run({"entry": entry})` | High |
| 2 | Cloud Database Write (MongoDB Atlas) | `backend/db.py` | 32–44 | `save_expense` | Document `{user: "demo", raw: <full original text>, parsed: [{item, amount, category}, ...], timestamp}` inserted into the `finchain.expenses` collection on MongoDB Atlas (connection uses `MONGO_URI` with TLS via `certifi`). | `result = expenses.insert_one(doc)` where `doc` includes `raw` (verbatim user input) and `parsed` (LLM output) | High |
| 3 | Cloud Vector DB Upsert (Pinecone) | `backend/db.py` | 46–62 | `save_expense` | 384-dim embedding of the raw expense text plus metadata `{user, raw, parsed (stringified), timestamp}` upserted into the Pinecone index `expenses-index` keyed by Mongo `_id`. | `index.upsert([(expense_id, embedding.tolist(), {"user": user, "raw": raw, "parsed": str(parsed), "timestamp": ...})])` | High |
| 4 | Local Embedding Model (No Network) | `backend/vectorstore.py` | 17–20 | `embed_text` | Raw expense text encoded locally with `sentence-transformers/all-MiniLM-L6-v2` to produce a 384-dim vector. The vector itself leaves the box via Channel 3, but inference happens on-host. | `EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")` ; `EMBEDDING_MODEL.encode([text])[0]` | High |
| 5 | Server Logging (stdout/stderr) | `backend/main.py` | 49, 53, 57, 69, 82 | `parse_expense_api`, `get_expenses`, `query_expenses` | Full request body (`req.entry`), parsed LLM result, and semantic-search query strings logged at `INFO` level via Python `logging`. Logs go wherever the FastAPI process is run (terminal, Docker stdout, GCP logs). | `logger.info(f"Received request: {req.entry}")`, `logger.info(f"Parsed data: {parsed_data}")`, `logger.info(f"Semantic search for query: {query}")` | High |
| 6 | Server Logging (stdout) | `backend/langchain_prompt.py` | 41, 45, 56, 59, 62, 66 | `parse_expense` | Verbatim user entry, raw Groq API response, and parsed JSON printed to stdout via `print()` on every parse call. | `print(f"Parsing entry: {entry}")`, `print("RAW GROQ API RESPONSE:", response)`, `print("Successfully parsed JSON:", parsed_data)` | High |
| 7 | Server Logging (Mongo URI) | `backend/db.py` | 15 | module init | The `MONGO_URI` is partially logged at startup (last 20 chars exposed); a misconfigured short URI would fully leak the connection string and embedded credentials. | `logger.info(f"Connecting to MongoDB with URI: {'*' * (len(MONGO_URI) - 20) + MONGO_URI[-20:]}")` | Medium |
| 8 | Network Request (Cloud Vector DB query) | `backend/main.py` | 79–97 | `query_expenses` | User search query is embedded locally then the 384-dim vector is sent to Pinecone (`index.query`) for semantic similarity search; matching IDs are then fetched from MongoDB. | `embedding = embed_text(query); result = index.query(vector=embedding.tolist(), top_k=top_k, include_metadata=True)` | High |
| 9 | UI Rendering (Web Browser) | `frontend/src/app/page.tsx` | 384–415 | `FinancialDashboard` (Latest Transactions) | Raw expense text (`exp.raw`), parsed amount, parsed category, and timestamp from MongoDB are rendered in plain text in the dashboard transaction feed. | `<p ...>{exp.raw}</p>` and `{exp.parsed && exp.parsed[0]?.category}` and ``-${formatCurrency(exp.parsed[0]?.amount)}৳`` | High |
| 10 | UI Rendering (Charts) | `frontend/src/app/page.tsx` | 332–365, 256–280 | `FinancialDashboard` (Spending categories, Monthly totals) | Aggregated category totals and monthly totals (from `/analytics/category` and `/analytics/monthly`) rendered in pie/line/bar charts via Recharts. | `<Pie data={categoryData} ... label={({ name, value }) => \`${name}: ৳${value}\`} ...>` | High |
| 11 | Cross-Origin HTTP (CORS open to localhost) | `backend/main.py` | 34–40 | app middleware | `CORSMiddleware` allows `http://localhost:3000` with credentials, all methods, all headers — any locally-running web page on that origin can drive the parse endpoint. | `app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])` | Medium |
| 12 | Unauthenticated REST API | `backend/main.py` | 46–158 | all endpoints | No auth layer: `/parse_expense`, `/expenses`, `/query_expenses`, `/analytics/*` accept a `user` query param (default `"demo"`) and return data for that user with no token check — anyone on the network can read another user's transactions or post on their behalf. | All endpoints take `user: str = "demo"` and perform Mongo `find({"user": user})` directly | High |
| 13 | Test Script Logging (credential surface) | `backend/test_env.py` | 11, 15–16 | module top-level | Diagnostic script prints `GROQ_API_KEY` length and first 20 chars of `MONGO_URI` to stdout, then performs a live `llm.invoke("Say hello")` to Groq and prints the response. | `print(f"GROQ_API_KEY length: {len(...)}")`, `print(f"MONGO_URI starts with: {mongo_uri[:20]}...")`, `response = llm.invoke("Say hello")` | Medium |

## B. Main AI Inference Workflows

### Workflow 1: Natural-Language Expense Parsing (`POST /parse_expense`)
- **Purpose**: Convert free-form, multi-language expense text (e.g., "Spent 670 on burgers today", "বার্গার খাইসি ১০০০ টাকা দিয়া", "Uber ride last night: 450") into structured items `{item, amount, category}` with category drawn from {Food, Transportation, Utilities, Entertainment, Shopping, Healthcare, Other}.
- **Input**: Raw text from the dashboard textarea (`entry` state in `page.tsx`), POSTed as `{"entry": "..."}` to FastAPI.
- **Processing**:
  - FastAPI deserializes into `ExpenseRequest`.
  - `parse_expense(entry)` builds a `PromptTemplate` that interpolates the verbatim text into an "expense parser" prompt instructing the model to return a JSON array.
  - LangChain `LLMChain` calls Groq's hosted `llama3-8b-8192` with `temperature=0.1`, `max_tokens=1000`.
  - Response is regex-scanned for the first `[...]` array and `json.loads`-parsed.
- **Inference**: Cloud LLM (Groq `llama3-8b-8192`) — extraction, normalization, and categorization happen entirely on Groq's servers.
- **Externalization**:
  - Raw expense text → Groq API (Channel 1).
  - `{user, raw, parsed, timestamp}` → MongoDB Atlas `finchain.expenses` (Channel 2).
  - 384-dim local embedding + metadata (incl. raw text) → Pinecone `expenses-index` (Channel 3, after local embed via Channel 4).
  - Verbatim entry, raw LLM response, and parsed JSON written to server logs (Channels 5, 6).
- **Episode path**: textarea text → `postExpense()` in frontend → `POST /parse_expense` → `parse_expense()` (Groq llama3-8b-8192) → JSON array → `save_expense()` → MongoDB insert + `embed_text()` (MiniLM-L6-v2) + Pinecone upsert → response `{status, parsed}` → `loadAll()` re-fetch → Latest Transactions list rendered (Channel 9) and aggregate charts updated (Channel 10).
- **Key files**: `backend/main.py` (`parse_expense_api`, lines 46–64), `backend/langchain_prompt.py` (`parse_expense`, lines 36–67), `backend/db.py` (`save_expense`, lines 32–67), `backend/vectorstore.py` (`embed_text`, lines 19–20), `frontend/src/app/page.tsx` (`handleSubmit`, lines 70–87).
- **Confidence**: High

### Workflow 2: Semantic Expense Search (`GET /query_expenses`)
- **Purpose**: Find past expenses similar in meaning to a free-form query (e.g., "Show me all Uber rides this year") even when the literal words don't match stored entries.
- **Input**: `query` query-string parameter; optional `user` (default `"demo"`) and `top_k` (default 5).
- **Processing**:
  - Server logs the query string at INFO.
  - `embed_text(query)` runs locally with `all-MiniLM-L6-v2` to produce a 384-dim vector.
  - `index.query(vector=..., top_k=top_k, include_metadata=True)` is sent to Pinecone.
  - Matching IDs are converted to `ObjectId` and fetched from MongoDB; documents are returned to the caller.
- **Inference**: Local sentence-transformer embedding + remote Pinecone ANN search (no LLM).
- **Externalization**:
  - 384-dim query embedding → Pinecone (Channel 8).
  - Mongo read of matching expense docs (containing raw text + parsed items) returned over HTTP — same payload that Channel 9 would render if displayed.
  - Query string written to server logs (Channel 5).
- **Episode path**: client query → `GET /query_expenses?query=...` → `embed_text()` (local MiniLM) → Pinecone `index.query()` → match IDs → `expenses.find({_id: {$in: ...}})` → JSON list of expense docs returned.
- **Key files**: `backend/main.py` (`query_expenses`, lines 79–102), `backend/vectorstore.py` (`embed_text`, lines 19–20).
- **Confidence**: High

### Workflow 3: Aggregated Spending Analytics (`GET /analytics/{category,monthly,total}`)
- **Purpose**: Produce category, monthly, and grand-total spending summaries from previously parsed expenses to drive the dashboard charts and stat cards.
- **Input**: `user` query parameter (default `"demo"`).
- **Processing**:
  - Each endpoint runs a MongoDB aggregation pipeline over `finchain.expenses` matching the user, unwinding the `parsed` array, and grouping by category, by `(year, month)`, or globally.
- **Inference**: No model; deterministic Mongo aggregation. Results inherit the LLM-derived `category` field from Workflow 1, so any prior LLM mis-categorization propagates into the visualization.
- **Externalization**:
  - Aggregated totals returned over HTTP and rendered in `page.tsx`: total spending card, monthly line/bar chart, category pie chart with labels (Channel 10).
- **Episode path**: dashboard mount → `Promise.all([fetchExpenses, fetchCategoryAnalytics, fetchMonthlyAnalytics, fetchTotalAnalytics])` → Mongo `$match`/`$unwind`/`$group` pipelines → JSON → Recharts components.
- **Key files**: `backend/main.py` (`category_summary`, `monthly_summary`, `total_summary`, lines 105–158), `frontend/src/app/page.tsx` (`loadAll`, lines 36–63; chart blocks, lines 256–280, 332–365).
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 13
- **Total number of main AI inference workflows found**: 3
- **Top 3 highest-risk workflows or channels**:
    1. **Natural-Language Expense Parsing → Groq API (Workflow 1 / Channel 1)**: Every expense submission ships the user's verbatim free-form text — which can include merchants, locations, people, health-related purchases, alcohol, religious activity, multilingual context, and time-of-day — to a third-party cloud LLM (`llama3-8b-8192` on Groq). The text is interpolated directly into the prompt with no PII masking or redaction, so any side-channel content the user typed is exposed to the provider. Because every submission triggers this call, Groq accumulates a complete, time-stamped, high-fidelity diary of the user's spending behaviour.
    2. **MongoDB + Pinecone Persistence of Raw Text (Channels 2 + 3)**: `save_expense` stores the raw user message in MongoDB Atlas (cloud) and *also* upserts a 384-dim embedding plus the raw text and `str(parsed)` as metadata into Pinecone. This means the same sensitive text is persisted in two cloud datastores with different security postures, and Pinecone retains a vector representation that supports nearest-neighbour reconstruction of the user's spending themes. Combined with the unauthenticated REST API (Channel 12) that defaults `user="demo"`, any caller who can reach the FastAPI host can read or pollute another user's record.
    3. **Verbose Server-Side Logging of User Input (Channels 5 + 6 + 7)**: `main.py` logs `req.entry`, parsed output, and semantic-search queries at INFO; `langchain_prompt.py` additionally `print()`s the raw entry and the raw Groq response on every call; `db.py` logs a partially-masked `MONGO_URI`. Wherever this server runs (local dev terminal, Docker stdout, GCP Cloud Logging), the full plaintext of every user expense — including any incidental PII the LLM was supposed to abstract away — is mirrored into log sinks that typically have looser retention and access controls than the primary database.
