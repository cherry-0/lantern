# AI Inference Privacy Audit: budget-lens

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (OpenAI) | `core/views.py` | 65 | `process_receipt` | Base64 encoded receipt image and prompt for extraction. | `client.chat.completions.create(model="gpt-4o-mini", ...)` | High |
| 2 | Network Request (Exchange Rates) | `core/views.py` | 134 | `get_exchange_rate` | Transaction date to fetch exchange rates. | `requests.get(url, params=params, timeout=10)` | High |
| 3 | UI Rendering | `core/views.py` | 231, 241 | `dashboard`, `expense` | Inferred category, date, amount, currency, and converted totals. | `render(request, "dashboard.html", ...)` | High |
| 4 | Logging | `core/views.py` | 100, 101, 104, 106, 114, 115, 122, 178 | Various | OpenAI response content, exchange rate URLs, and error details. | `log.debug(content)`, `log.error(...)` | High |

## B. Main AI Inference Workflows

### Workflow 1: Receipt Data Extraction
- **Purpose**: Automatically extract expense details (category, date, amount, currency) from an uploaded receipt image.
- **Input**: User-uploaded receipt image file.
- **Processing**: Image is base64 encoded; prompt is constructed with predefined categories.
- **Inference**: Remote call to OpenAI `gpt-4o-mini` via `OpenAI` client.
- **Externalization**:
    - Image sent to OpenAI API (`core/views.py:65`).
    - Extracted data logged (`core/views.py:101`).
    - Extracted data saved to Django DB and displayed in `dashboard` and `expense` views.
- **Episode path**: User Upload -> Base64 Encoding -> OpenAI API -> Extraction -> Storage/UI
- **Key files**: `core/views.py`, `core/models.py`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 4
- **Total number of main AI inference workflows found**: 1
- **Top 3 highest-risk workflows or channels**:
    1. **OpenAI API Call (Workflow 1)**: Transmits full receipt images (containing PII like merchant, date, items, and potentially payment info) to a third-party AI provider.
    2. **UI Rendering (Inferred Data)**: Displays extracted financial information to the user; if incorrect or leaked via session, could compromise privacy.
    3. **Logging (OpenAI Response)**: In debug mode, raw extraction results (including amounts and categories) are written to server logs.
