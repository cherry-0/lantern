# AI Inference Privacy Audit: fiscal-flow

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM) | `src/ai/flows/financial-qa.ts` | 53–122 | `financialQAFlow` / `prompt` | Full accounts array, expenses array (with transaction descriptions, amounts, dates, categories), income array, and user question sent to Google Gemini (`gemini-2.0-flash`) via Genkit. | `const { output } = await prompt(formattedInput)` where `formattedInput` embeds all financial context | High |
| 2 | Network Request (Cloud LLM) | `src/ai/flows/spending-insights.ts` | 41–83 | `spendingInsightsFlow` / `prompt` | Transaction records including amounts, dates, categories, and descriptions sent to Gemini for insights generation. | `ai.definePrompt` + `ai.defineFlow` with full transaction JSON inlined in the Handlebars template | High |
| 3 | Network Request (Cloud LLM) | `src/ai/flows/transaction-categorization.ts` | 25–63 | `transactionCategorizationFlow` / `prompt` | Individual transaction description and amount sent to Gemini to predict a spending category. | `ai.definePrompt` + `ai.defineFlow` with description field embedded in prompt | High |
| 4 | UI Rendering (Web) | `src/app/` | — | Dashboard / AI query response | Model answer displayed directly in the Next.js web UI; transaction data visible on-screen. | Next.js client renders the `response` field from `/api/ai-query` | High |

## B. Main AI Inference Workflows

### Workflow 1: Financial Q&A (`/api/ai-query` → `financialQAFlow`)
- **Purpose**: Answer natural-language questions about a user's personal financial situation.
- **Input**: User question + full accounts, expenses, and income arrays.
- **Processing**:
  - Genkit `financialQAPrompt` (Handlebars) inlines all financial data into a structured prompt.
  - Today's date injected via `currentDate`.
  - Data formatted to human-readable lines before embedding.
- **Inference**: Google Gemini (`gemini-2.0-flash`) receives the combined prompt and returns a plain-text answer.
- **Externalization**:
  - All account names, balances, transaction descriptions, amounts, and dates sent to `generativelanguage.googleapis.com` (Channel 1).
  - Answer displayed in web UI (Channel 4).
- **Episode path**: User question + financial data → `financialQAPrompt` → Gemini API → plain-text answer → web UI
- **Key files**: `src/ai/flows/financial-qa.ts` (`financialQAFlow`, `prompt`), `src/app/api/ai-query/route.ts`
- **Confidence**: High

### Workflow 2: Spending Insights (`spendingInsightsFlow`)
- **Purpose**: Generate actionable bullet-point advice from a user's recent spending history.
- **Input**: Array of transaction records (up to 50 items).
- **Processing**: Full transaction JSON serialized and embedded in the Genkit prompt template.
- **Inference**: Gemini returns a ~150-word markdown bullet list of spending observations.
- **Externalization**:
  - Complete transaction list (titles, amounts, dates, categories) sent to Gemini API (Channel 2).
  - Result rendered in web UI (Channel 4).
- **Episode path**: Transaction records → `spendingInsightsFlow` prompt → Gemini API → markdown insights → web UI
- **Key files**: `src/ai/flows/spending-insights.ts`
- **Confidence**: High

### Workflow 3: Transaction Auto-Categorization (`transactionCategorizationFlow`)
- **Purpose**: Automatically assign a spending category to a transaction based on its description and amount.
- **Input**: Single transaction description string + amount.
- **Processing**: Description and amount embedded into the Genkit categorization prompt.
- **Inference**: Gemini returns a predicted category label (e.g., Food, Transport, Shopping).
- **Externalization**:
  - Transaction description sent to Gemini API (Channel 3).
- **Episode path**: Transaction description + amount → `transactionCategorizationFlow` prompt → Gemini API → category label
- **Key files**: `src/ai/flows/transaction-categorization.ts`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 4
- **Total number of main AI inference workflows found**: 3
- **Top 3 highest-risk workflows or channels**:
    1. **Financial Q&A (Workflow 1)**: Sends the complete financial picture — all account names, balances, and every transaction record (descriptions, amounts, dates) — to Google Gemini on each query. A single request can expose the user's full financial identity.
    2. **Spending Insights (Workflow 2)**: Transmits up to 50 transaction records in one call. Transaction titles often encode location, merchant, and personal habits (e.g., medical expenses, subscription services), making this a high-volume personal data leak.
    3. **Transaction Categorization (Workflow 3)**: While smaller in scope, this runs on every new transaction and continuously streams individual expense descriptions to Gemini, creating a persistent data exposure channel over time.
