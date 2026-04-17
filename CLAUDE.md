# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Adding a new app to the `verify/` privacy evaluation framework?**
> See [`analysis/verify_report.md`](/Users/sieun/Research/Lantern/lantern/analysis/verify_report.md) — Section 7 ("How to Add a New App") has the step-by-step guide with code templates for the adapter, runner, Django settings shim, and adapter registration.

> **Encountering an error in the `verify/` framework?**
> See [`TROUBLESHOOTING.md`](/Users/sieun/Research/Lantern/lantern/TROUBLESHOOTING.md) — covers known runner failures, phase-drift bugs, Django config issues, conda env problems, and frontend quirks with fixes.

## Overview

This repository contains four independent applications, each in its own subdirectory:

| App | Description | Stack |
|-----|-------------|-------|
| `clone/` | Desktop AI personal assistant (records screen activity) | Electron + React/TS + Django + VectorDB |
| `momentag/` | AI-powered photo search & tagging app | Django + Celery + Qdrant + Android |
| `snapdo/` | Todo app with VLM-based evidence verification | Django + OpenRouter VLM + Android |
| `xend/` | AI email drafting assistant with streaming | Django Channels + LangGraph + Android |

---

## clone/

A desktop Electron app that records digital activities and enables chat against a personal knowledge base.

### Architecture

- **`frontend/`** — Electron app (main + renderer processes)
  - `src/main.ts` — Electron main process; manages IPC, Ollama/OpenAI integration, model downloads
  - `src/renderer.tsx` — React entry point
  - `src/recording/` — Screen/audio capture providers
  - `src/llm/` — LLM abstraction layer (Ollama local models, OpenAI, embedding manager)
  - `src/services/` — API clients for auth, chat, collection, memory, embeddings
  - `src/components/` — React UI components (each paired with a `.test.tsx`)
  - `src/embedding/` — ONNX-based local embedding worker (runs in a separate worker thread via `embedding-worker.ts`)
- **`server/`** — Django REST API (MySQL)
  - Apps: `user/`, `chat/`, `collection/`
  - Settings split: `config/settings/base.py`, `local.py`, `test.py`
- **`vectordb/`** — Separate Django service with pluggable VectorDB backends (`vectordb/vectordb/`: `naive_vectordb.py`, `milvus_vectordb.py`)

### Commands

```bash
# Frontend (Electron)
cd clone/frontend
npm install
npm run dev            # Start dev app
npm run lint           # ESLint
npm run test:unit      # Vitest unit tests
npm run test:e2e       # Playwright e2e tests
npm run test:all       # Both unit + e2e

# Run a single Vitest test file
npm run test -- src/components/ChatInterface.test.tsx

# Server (Django)
cd clone/server
pip install -r requirements.txt
python manage.py migrate --settings=config.settings.local
python manage.py runserver --settings=config.settings.local

# Run tests
pytest                            # All tests with coverage
pytest user/tests/                # Specific app
pytest -m unit                    # Only unit tests
pytest -m integration             # Only integration tests
pytest -n auto                    # Parallel execution
pytest user/tests/test_models.py::TestUserModel::test_create_user_with_email_and_username
```

### Commit style
`[$task_id] description` (e.g., `[P9] fix: dependency errors`)

---

## momentag/

Photo management app with AI semantic search, tag recommendations, and Celery-based GPU processing.

### Architecture

- **`backend/`** — Django REST API (uv, Python ≥3.13, MySQL + Qdrant + Redis)
  - `accounts/` — JWT auth (sign up/in/out, token refresh)
  - `gallery/` — Photo upload (batch 8), GPS metadata, Qdrant embedding storage, S3 storage, Celery tasks (`tasks.py` CPU, `gpu_tasks.py` GPU)
  - `search/` — Semantic & hybrid search via Qdrant; `search_strategies.py` implements strategy pattern
  - `config/` — Django settings, Celery config (`compose.yml` for local dev)
- **`android/`** — Android client
- **`tag-search/`** — Standalone ML scripts: `Image_Preprocessing/`, `Image_Recommendation/`, `Tag_Recommendation/`, `NL_Search/`

### Commands

```bash
cd momentag/backend

# Install dependencies
uv sync

# Migrate & run CPU server
uv run manage.py migrate
uv run manage.py runserver 0.0.0.0:8080

# Run Celery workers (GPU server, separate terminals)
uv run celery -A config worker -Q gpu -l info --pool=threads -c4
uv run celery -A config worker -Q interactive -l info --pool=threads -c4

# Run tests
uv run pytest
uv run coverage run manage.py test
```

### Required env vars
`SECRET_KEY`, `QDRANT_CLIENT_URL`, `QDRANT_API_KEY`, `DJANGO_ALLOWED_HOSTS`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `REDIS_URL`

---

## snapdo/

Django todo app where tasks are verified by submitting photo evidence analyzed by a Vision Language Model.

### Architecture

- **`server/`** — Django REST API (Python 3.10, conda, SQLite default)
  - `snapdo/` app — models, views, serializers, permissions
  - `snapdo/services/vlm_service.py` — calls OpenRouter VLM API to verify evidence images
  - Local file storage (`media/`) by default; S3 via `USE_AWS=true`
- **`client/`** — Android client (MVVM architecture, see `client/MVVM.md`)

### Commands

```bash
conda create -n django python=3.10 && conda activate django
pip install -r requirements.txt
cd server
python manage.py migrate
python manage.py runserver

# Tests
python manage.py test snapdo.tests
python manage.py test new_challengers.test_aws -v 2   # AWS/S3 tests (mocked with moto)
```

### Env file: `server/snapdo/.env`
`API_KEY`, `USE_AWS`, `S3_BUCKET`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `VLM_API_URL`, `VLM_API_KEY`, `VLM_API_TIMEOUT`

---

## xend/

AI email drafting assistant integrated with Gmail, featuring real-time streaming via WebSocket (draft composition) and SSE (smart replies).

### Architecture

- **`backend/`** — Django + Django Channels (poetry, Python ≥3.12, PostgreSQL + Redis)
  - `apps/ai/` — LangChain/LangGraph chains for email generation; `consumers.py` handles WebSocket; `services/` contains `chains.py`, `graph.py`, `mail_generation.py`, `reply.py`, `analysis.py`, `pii_masker.py`
  - `apps/mail/` — Gmail OAuth2 sync, incremental fetch, email CRUD
  - `apps/contact/` — Contact & group management with custom AI prompt rules
  - `apps/user/` — JWT auth + Gmail OAuth, user profile (language preference)
  - `apps/core/` — Shared mixins, base models, renderers
  - Celery for background sync tasks; Django Channels + Redis for WebSocket
- **`gpu-server/`** — FastAPI app (`app/main.py`, `app/llm.py`, `app/models.py`) for LLM inference; runs separately with uvicorn
- **`frontend/`** — Android client

### Commands

```bash
# Backend
cd xend/backend
cp .env_example .env   # fill in credentials
poetry install
python manage.py migrate
python manage.py runserver

# GPU server (optional, for AI features)
cd xend/gpu-server
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Docker (dev)
docker compose -f docker-compose_dev.yml up -d

# Android: open xend/frontend in Android Studio
# Create local.properties with sdk.dir, base.url, ws.url
```
