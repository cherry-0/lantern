# AI Inference Privacy Audit: spendsense

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM — multimodal) | `services/geminiService.ts` | 7–48 | `scanReceipt` | Base64-encoded receipt image sent to Google Gemini 2.5 Flash for OCR + structured data extraction (amount, date, merchant, category). | `ai.models.generateContent({ model: 'gemini-2.5-flash', contents: [{ parts: [{ text: prompt }, { inlineData: { mimeType: 'image/jpeg', data: base64Image } }] }] })` | High |
| 2 | Network Request (Cloud LLM) | `services/geminiService.ts` | 50–75 | `getSpendingInsights` | Up to 50 recent transaction records (title, amount, date, category, type) serialized as JSON and sent to Gemini 2.5 Flash for financial advice generation. | `ai.models.generateContent({ model: 'gemini-2.5-flash', contents: [{ parts: [{ text: prompt }] }] })` where prompt embeds `JSON.stringify(transactions)` | High |
| 3 | Client-side API Key Exposure | `services/geminiService.ts` | 4 | Module initialization | `process.env.API_KEY` (Google GenAI key) baked into the browser bundle at Vite build time and visible to any client. | `const ai = new GoogleGenAI({ apiKey: process.env.API_KEY })` | High |
| 4 | UI Rendering (Web PWA) | `components/`, `views/` | — | `AddTransactionModal`, `Analytics`, `Dashboard` | Extracted receipt fields and AI-generated spending insights displayed directly in the browser UI. | React components consume the `ReceiptData` / insights string returned from `geminiService.ts` | High |

## B. Main AI Inference Workflows

### Workflow 1: Receipt Scanning (`scanReceipt`)
- **Purpose**: Extract structured expense data (merchant name, total amount, date, category) from a photo of a receipt.
- **Input**: Camera or gallery image captured in the browser, base64-encoded as JPEG.
- **Processing**:
  - Image converted to base64 in the browser.
  - Extraction prompt + base64 image packed into a multimodal `generateContent` request.
- **Inference**: Gemini 2.5 Flash returns JSON `{ amount, date, title, category }`.
- **Externalization**:
  - Full receipt image (containing merchant name, itemized purchases, prices, address) sent to `generativelanguage.googleapis.com` (Channel 1).
  - Extracted fields shown in `AddTransactionModal` (Channel 4).
- **Episode path**: Receipt photo → base64 encode → Gemini multimodal API → JSON parse → pre-filled transaction form
- **Key files**: `services/geminiService.ts` (`scanReceipt`, lines 7–48)
- **Confidence**: High

### Workflow 2: Spending Insights (`getSpendingInsights`)
- **Purpose**: Generate short, actionable financial advice from the user's recent transaction history.
- **Input**: Array of up to 50 `Transaction` objects from local storage.
- **Processing**:
  - Transactions serialized to JSON and embedded in a prompt asking for 3 bullet-point observations.
  - Prompt sent as a plain-text `generateContent` request.
- **Inference**: Gemini 2.5 Flash returns a ≤150-word Markdown bullet list.
- **Externalization**:
  - Full transaction list (titles, amounts, dates, categories) sent to Gemini API (Channel 2).
  - Markdown advice rendered in the `Analytics` / `Dashboard` views (Channel 4).
- **Episode path**: localStorage transactions → JSON.stringify → Gemini text API → Markdown insights → `Analytics` view
- **Key files**: `services/geminiService.ts` (`getSpendingInsights`, lines 50–75)
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 4
- **Total number of main AI inference workflows found**: 2
- **Top 3 highest-risk workflows or channels**:
    1. **Receipt Scanning (Workflow 1 / Channel 1)**: Transmits a full receipt photo to Google's cloud API. Receipts contain merchant location, itemized purchases (potentially medical, personal-care, or financial products), exact transaction time, and printed payment card details — a uniquely rich source of personal information per image.
    2. **Spending Insights (Workflow 2 / Channel 2)**: Serializes the user's complete recent transaction history as JSON. Transaction titles encode daily routines, locations, medical activity, and social patterns, all sent in a single API call with no server-side mediation.
    3. **Client-side API Key Exposure (Channel 3)**: The Google GenAI API key is embedded in the browser bundle via `process.env.API_KEY` at Vite build time. Any visitor can extract it from the JavaScript bundle, enabling unauthorized use of the key and potential exfiltration of data under the app owner's quota.
