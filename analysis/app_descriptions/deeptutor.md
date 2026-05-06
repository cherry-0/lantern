# AI Inference Privacy Audit: deeptutor

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Cloud LLM) | `deeptutor/services/provider_registry.py` | 169, 176 | N/A | User prompts, session context, and system instructions sent to providers (OpenAI, Anthropic, DeepSeek, etc.). | `default_api_base="https://api.anthropic.com/v1"` | High |
| 2 | Network Request (Embedding) | `deeptutor/services/rag/components/embedders/openai.py` | 57 | `embed_batch` | Document chunks and user queries sent to OpenAI-compatible embedding APIs. | `embeddings = await client.embed(texts)` | High |
| 3 | Network Request (RAG Retrieval) | `deeptutor/services/rag/pipelines/llamaindex.py` | - | `run` | Metadata and query context used for LlamaIndex retrieval. | Usage of `llama-index` library. | High |
| 4 | UI Rendering | `web/` | - | - | Tutor responses, citations from documents, and RAG-retrieved context displayed in the web UI. | React-based frontend integration. | High |
| 5 | Event Bus / Analytics | `deeptutor/runtime/orchestrator.py` | 101 | `_publish_completion` | Completion events including `user_input` and `task_id` published to a global event bus. | `await bus.publish(Event(type=EventType.CAPABILITY_COMPLETE, ...))` | High |
| 6 | Logging | `deeptutor/services/rag/pipeline.py` | 155 | `run` | Detailed execution stages (Chunking, Embedding, Indexing) and potential metadata. | `self.logger.info("Stage 3: Embedding...")` | High |

## B. Main AI Inference Workflows

### Workflow 1: RAG-Based Document Tutoring
- **Purpose**: Answer user questions based on private uploaded documents using Retrieval-Augmented Generation.
- **Input**: User text query and previously uploaded/processed documents.
- **Processing**: 
    - Documents are chunked and embedded via Cloud API (`Workflow 2`).
    - Query is embedded and used to retrieve relevant chunks via LlamaIndex.
    - Retrieved context + User Query + System Prompt are combined.
- **Inference**: Cloud LLM (GPT-4, Claude 3.5, etc.) generates a tutor response.
- **Externalization**: 
    - Text chunks sent to Embedding API (`openai.py:57`).
    - Full RAG prompt sent to LLM API (`provider_registry.py`).
    - Response rendered in UI.
- **Episode path**: User Query -> Embedding API -> Vector Search -> Contextual Prompt -> LLM API -> UI
- **Key files**: `rag/pipeline.py`, `rag/pipelines/llamaindex.py`, `runtime/orchestrator.py`
- **Confidence**: High

### Workflow 2: Persistent Document Indexing
- **Purpose**: Prepare uploaded documents for future tutoring sessions.
- **Input**: User-provided documents (PDF, Docx, etc.).
- **Processing**: File parsing; content chunking; batch embedding.
- **Inference**: Embedding model (remote) calculates vectors for each chunk.
- **Externalization**: 
    - Chunk content sent to remote Embedding API.
    - Vectors and metadata stored in a local/remote vector store.
- **Episode path**: Document Upload -> Parsing -> Chunking -> Embedding API -> Vector Store
- **Key files**: `rag/components/embedders/openai.py`, `rag/factory.py`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 6
- **Total number of main AI inference workflows found**: 2
- **Top 3 highest-risk workflows or channels**:
    1. **Document Embedding (Workflow 2)**: Sends the **entire content** of user-uploaded private documents to a third-party embedding provider (e.g., OpenAI) in plaintext chunks.
    2. **Contextual LLM Prompts (Workflow 1)**: Transmits both the user's question and relevant snippets from their private documents to a remote LLM provider.
    3. **Global Event Bus (Channel 5)**: Publishes user inputs and task metadata to a central bus, which could be monitored by other internal services or exported to external analytics if misconfigured.
