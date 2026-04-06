# AI Inference Privacy Audit: momentag

## A. Externalization Channels

| ID | Channel Type | File | Line(s) | Function | What is externalized | Evidence / code clue | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Network Request (Qdrant) | `backend/gallery/gpu_tasks.py` | 236 | `process_and_embed_photo` | Image embeddings and metadata (user_id, filename, lat, lng, created_at). | `client.upsert(collection_name=IMAGE_COLLECTION_NAME, points=[point_to_upsert], ...)` | High |
| 2 | Network Request (MinIO/S3) | `backend/gallery/gpu_tasks.py` | 211, 258 | `process_and_embed_photo` | Original photo download and deletion. | `download_photo(storage_key)`, `delete_photo(storage_key)` | High |
| 3 | Storage Write (Django DB) | `backend/gallery/gpu_tasks.py` | 244-252 | `process_and_embed_photo` | Inferred photo captions (keywords) and photo metadata. | `Caption.objects.get_or_create(user=user, caption=word)`, `Photo_Caption.objects.create(...)` | High |
| 4 | UI Rendering | `backend/search/views.py` (assumed) | - | - | Semantic search results (matching photos) and captions. | Expected in a typical Django view. | Medium |
| 5 | Logging | `backend/gallery/gpu_tasks.py` | 134, 151, 230, 255 | Various | Model loading info, inference status, and task exceptions. | `print(f"[INFO] Loading CLIP image model...")`, `print(f"[DONE] Finished image embedding
")` | High |

## B. Main AI Inference Workflows

### Workflow 1: Image Embedding & Captioning (Ingestion)
- **Purpose**: Automatically generate searchable embeddings and textual keywords for uploaded photos.
- **Input**: User-uploaded photo (retrieved from shared storage via `storage_key`).
- **Processing**: Photo downloaded to memory; converted to RGB; input preprocessed for CLIP and BLIP models.
- **Inference**:
    - **CLIP** (`clip-ViT-B-32`) generates a dense vector embedding.
    - **BLIP** (`blip-image-captioning-base`) generates multiple captions/keywords.
- **Externalization**: 
    - Embedding + metadata (user_id, lat/lng, filename) sent to Qdrant vector database (`gpu_tasks.py:236`).
    - Filtered keywords saved to Django PostgreSQL/SQLite DB (`gpu_tasks.py:252`).
- **Episode path**: Storage Trigger -> Photo Retrieval -> CLIP/BLIP Inference -> Qdrant/Django DB -> UI
- **Key files**: `gpu_tasks.py`, `qdrant_utils.py`, `models.py`
- **Confidence**: High

### Workflow 2: Semantic Query Embedding (Search)
- **Purpose**: Convert user text queries into vectors for semantic similarity matching.
- **Input**: User text search query.
- **Processing**: Query text is embedded using CLIP's multilingual text encoder.
- **Inference**: CLIP (`clip-ViT-B-32-multilingual-v1`) generates the query embedding.
- **Externalization**: 
    - Query vector sent to Qdrant for matching against stored image embeddings.
    - Matching photos retrieved and rendered in the UI.
- **Episode path**: Text Query -> CLIP Inference -> Qdrant Search -> Results Retrieval -> UI
- **Key files**: `embedding_service.py`, `views.py`
- **Confidence**: High

## Final Summary
- **Total number of distinct externalization sites found**: 5
- **Total number of main AI inference workflows found**: 2
- **Top 3 highest-risk workflows or channels**:
    1. **Qdrant DB Upsert (Workflow 1)**: Stores highly personal metadata (GPS coordinates, user IDs, original filenames) alongside sensitive biometric/visual embeddings in a vector database.
    2. **Caption Storage (Workflow 1)**: Extracts keywords from photos (e.g., "dog", "beach", "person") and stores them in a searchable Django DB; if the photos contain sensitive subjects, these keywords could be leaked.
    3. **MinIO/Storage Access (Channel 2)**: Centralized storage of original photos; any misconfiguration in `storage_service.py` could lead to unauthorized access to the entire gallery.
