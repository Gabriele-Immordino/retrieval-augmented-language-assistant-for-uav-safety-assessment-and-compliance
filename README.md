# Retrieval‑Augmented Language Assistant for Unmanned Aircraft Safety Assessment & Regulatory Compliance

This repository contains a Retrieval-Augmented Generation (RAG) assistant focused on **UAS (drone) safety assessment and regulatory compliance** workflows.

It consists of:
- a **Python/Flask backend** that performs retrieval (FAISS + BM25 + optional reranking) and calls an **OpenAI-compatible LLM API** (typically **Ollama** running locally)
- a **PostgreSQL-backed chat history store** (or SQLite for local/dev)
- an optional **Web GUI** (Express + vanilla JS) that talks to the backend

The default document corpus in this repo is a chunked JSON derived from UAS regulatory material (see `Documents/`).

---

## Table of contents

- [Quick start (Docker Compose)](#quick-start-docker-compose)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Configuration](#configuration)
- [Backend API](#backend-api)
- [Rebuilding the FAISS index](#rebuilding-the-faiss-index)
- [Local development (without Docker)](#local-development-without-docker)
- [Troubleshooting](#troubleshooting)

---

## Quick start (Docker Compose)

### Prerequisites

- **Docker Desktop** (Windows/macOS) or Docker Engine (Linux)
- **An OpenAI-compatible LLM endpoint** accessible from the backend container
  - Recommended: **Ollama** running on the host machine
  - The default `docker-compose.yml` expects Ollama at `http://host.docker.internal:11434/v1`

### 1) Start (backend + database + web GUI)

From the repository root:

```bash
docker compose up --build
```

Services:
- Backend API: `http://localhost:8080`
- Web GUI: `http://localhost:3000`

### 2) Ensure your LLM endpoint is running (Ollama)

The backend uses the **OpenAI Python client** configured with:
- `LLM_BASE_URL` (default in Docker Compose): `http://host.docker.internal:11434/v1`
- `LLM_API_KEY` (default): `ollama` (Ollama ignores this value but the client requires it)

On the host machine, start Ollama and pull the models referenced in code:

```bash
ollama pull gpt-oss:20b
ollama pull qwen3:8b
```

Notes:
- The chatbot and classification engine default to `gpt-oss:20b`.
- The query-preprocessor defaults to `qwen3:8b`.

### 3) Health check

```bash
curl http://localhost:8080/health
```

A healthy backend returns:

```json
{"status":"ok"}
```

---

## How it works

### Retrieval + generation (chat)

1. A user sends chat history and a question to the backend.
2. The backend loads a **FAISS index** + metadata from:
   - `PreProcessing/ProcessedFiles/index/faiss.index`
   - `PreProcessing/ProcessedFiles/index/docs.json`
3. The backend retrieves relevant chunks using `RAG/ragv2.py`:
   - Dense retrieval (FAISS) with optional MMR
   - BM25 retrieval (lexical)
   - Hybrid scoring + optional reranking
4. Retrieved chunks are formatted into a context block and passed to the LLM.
5. The response is returned along with **human-readable source references**.

### Query preprocessing (optional)

If `preprocess_query=true`, the backend uses `LLM/LLM_openAI_QueryCall.py` to split a user question into a small plan of focused sub-queries (up to 3). Each sub-query is answered separately and the answers are concatenated.

### Classification endpoint

`/api/v1/classification` computes initial “operation indicators” (e.g., likely regulatory pathway, ground/air risk orientation) using:
- a constrained prompt that requests a **single JSON object** per indicator
- RAG context drawn from the same indexed corpus

### Chat history persistence

The backend stores user/assistant messages using SQLAlchemy models in `db.py`.
- In Docker Compose, chat history is stored in **PostgreSQL**.
- If `DATABASE_URL` is not set, the backend falls back to **SQLite** at `/app/chat.db`.

---

## Repository structure

High-level map:

- `server.py` – Flask API server (chat, streaming, history, classification)
- `db.py` – SQLAlchemy models + `DatabaseService`
- `config.yml` – Retrieval/generation configuration (used mainly for classification)
- `Documents/` – Input documents / chunked JSON corpus
- `PreProcessing/` – Embedding + FAISS indexing utilities
  - `ProcessedFiles/index/` – persisted FAISS index + metadata (`faiss.index`, `docs.json`)
- `RAG/` – retrieval stack (dense, BM25, hybrid, rerank)
- `LLM/` – OpenAI-compatible client wrappers + prompt/rules files
- `Web-GUI/` – optional web interface (Express + static frontend)

---

## Configuration

### Environment variables (backend)

The backend reads:
- `DATABASE_URL`
  - default: `sqlite:////app/chat.db`
  - Docker Compose sets: `postgresql+psycopg2://chatbot:chatbot@db:5432/chatbot`
- `LLM_BASE_URL`
  - default: `http://host.docker.internal:11434/v1`
- `LLM_API_KEY`
  - default: `ollama`

### `config.yml`

`config.yml` contains knobs used by the **classification** endpoint and (optionally) retrieval settings:

- `path_fname`, `fname`: where to load chunks if `docs.json` isn’t available
- `output_dir`: where the FAISS index and `docs.json` live
- `rag_mode`: `vector` | `bm25` | `hybrid`
- `reranker_mode`: `colbert` | `ce`
- `top_k`, `ce_keep_k`: retrieval parameters
- generation parameters: `temperature`, `top_p`, `max_new_tokens`, penalties
- `reasoning_effort`: `none` | `low` | `medium` | `high`

Important:
- `server.py` attempts to parse YAML via `import yaml`. If **PyYAML is not installed**, it silently falls back to an empty config. If you run outside Docker, install `pyyaml`. If you rely on `config.yml` inside Docker, add `PyYAML` to the backend image.
- The chat endpoint currently loads its index from the fixed location `PreProcessing/ProcessedFiles/` (see `INDEX_DIR` in `server.py`). The classification endpoint uses `config.yml`’s `output_dir`. Keep these aligned.

### Web GUI configuration

The Web GUI container uses:
- `PORT` (default `3000`)
- `BACKEND_URL` (Docker Compose sets `http://app:8080`)

The GUI proxies all `/api/*` calls to the backend.

---

## Backend API

Base URL (Docker): `http://localhost:8080`

### Health

- `GET /health`

### Model preload

- `GET /preload_models`

Useful to warm up embeddings/reranker models and the LLM clients.

### Chat (non-streaming)

- `POST /api/v1/chat`

Request body:

```json
{
  "user_id": "user123",
  "user_name": "Alice",
  "preprocess_query": false,
  "reasoning_effort": "medium",
  "chat_history": [
    {"role": "user", "content": "What is SORA?"}
  ]
}
```

Response:

```json
{
  "user_id": "user123",
  "user_name": "Alice",
  "original_question": "What is SORA?",
  "answer": "...",
  "sources": ["[0] CS-UAS.pdf: ..."],
  "reasoning": "..."
}
```

If `preprocess_query=true`, the response also includes `generated_queries` and the answer is the concatenation of each sub-query’s answer.

### Chat (streaming via Server-Sent Events)

- `POST /api/v1/chat/stream`

This endpoint returns `text/event-stream` where each event is:

```
data: {"type": "token", "token": "..."}

```

Event types you may receive:
- `sources` – contains `sources: string[]`
- `token` – incremental answer tokens
- `reasoning` – incremental reasoning tokens (if the model provides them)
- `done` – final payload for single-query mode
- `subquery_start`, `subquery_done` – multi-query mode metadata
- `error` – terminal error

In multi-query mode (`preprocess_query=true` and multiple queries are generated), events include:
- `subquery_index`, `total_subqueries`, `subquery_text`

### Chat history

- `GET /api/v1/history/<external_user_id>`
- `POST /api/v1/history` with body `{ "user_id": "..." }`
- `DELETE /api/v1/history/<external_user_id>`

History response schema:

```json
{
  "user_id": "user123",
  "user_name": "Alice",
  "messages": [
    {"role": "user", "content": "...", "ts": "2026-02-11T..."},
    {"role": "assistant", "content": "...", "ts": "2026-02-11T..."}
  ]
}
```

### Classification

- `POST /api/v1/classification`

Request body:

```json
{
  "operation_input": {
    "maximum_takeoff_mass_category": "lt_25kg",
    "vlos_or_bvlos": "VLOS",
    "ground_environment": "sparsely_populated",
    "airspace_type": "uncontrolled",
    "maximum_altitude_category": "gt_50m_le_120m"
  },
  "indicators": [
    "likely_regulatory_pathway",
    "initial_ground_risk_orientation",
    "initial_air_risk_orientation",
    "expected_assessment_depth"
  ]
}
```

Response body:

```json
{
  "operation": {"maximum_takeoff_mass_category": "lt_25kg", "vlos_or_bvlos": "VLOS", "ground_environment": "sparsely_populated", "airspace_type": "uncontrolled", "maximum_altitude_category": "gt_50m_le_120m"},
  "indicators": {
    "likely_regulatory_pathway": {"name": "...", "value": "...", "explanation": "..."}
  },
  "sources": {
    "likely_regulatory_pathway": ["[0] CS-UAS.pdf: ..."]
  }
}
```

Allowed categorical values are validated server-side:
- `vlos_or_bvlos`: `VLOS` | `BVLOS`
- `ground_environment`: `controlled_area` | `sparsely_populated` | `populated`
- `airspace_type`: `controlled` | `uncontrolled`
- `maximum_takeoff_mass_category`: `lt_25kg` | `gte_25kg`
- `maximum_altitude_category`: `le_50m` | `gt_50m_le_120m` | `gt_120m_le_150m` | `gt_150m`

---

## Rebuilding the FAISS index

The backend requires a FAISS index at:
- `PreProcessing/ProcessedFiles/index/faiss.index`
- `PreProcessing/ProcessedFiles/index/docs.json`

This repo already includes an index under `PreProcessing/ProcessedFiles/index/`. Rebuild it if you change the corpus.

### Option A: Build from the provided chunk JSON

PowerShell (Windows) from the repository root:

```powershell
python -c "import json; from pathlib import Path; from PreProcessing.embeddingToolsFAISSv2 import EmbeddingToolFAISS; chunks=json.load(open('Documents/SORA_chunks_cleaned_manual.json','r',encoding='utf-8')); tool=EmbeddingToolFAISS(output_dir=Path('PreProcessing/ProcessedFiles')); tool.build_index(chunks)"
```

Bash (Linux/macOS):

```bash
python -c "import json; from pathlib import Path; from PreProcessing.embeddingToolsFAISSv2 import EmbeddingToolFAISS; chunks=json.load(open('Documents/SORA_chunks_cleaned_manual.json','r',encoding='utf-8')); tool=EmbeddingToolFAISS(output_dir=Path('PreProcessing/ProcessedFiles')); tool.build_index(chunks)"
```

### Expected chunk fields

Retrieval and source formatting uses these keys when present:
- `chunk_title`, `chunk_text`, `chunk_index`, `page`, `source_file`
- optional: `chunk_summary`, `chunk_keywords`

`EmbeddingToolFAISS.build_index()` also stores a title-weighted retrieval document in `_retrieval_doc`.

---

## Local development (without Docker)

Docker is strongly recommended because the backend has heavy dependencies (PyTorch, Transformers, sentence-transformers, FAISS).

If you still want to run locally:

1) Create an environment and install dependencies (approximate list; keep in sync with the root `Dockerfile`):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install flask werkzeug numpy faiss-cpu sentence-transformers rank-bm25 transformers openai gunicorn bitsandbytes accelerate SQLAlchemy psycopg2-binary pyyaml
```

2) Start PostgreSQL (optional) or use SQLite.

3) Run the backend:

```bash
set DATABASE_URL=sqlite:///chat.db
set LLM_BASE_URL=http://localhost:11434/v1
set LLM_API_KEY=ollama
python server.py
```

4) Start the Web GUI (optional):

```bash
cd Web-GUI
npm install
set BACKEND_URL=http://localhost:8080
npm start
```

---

## Troubleshooting

### Backend says: “Index directory not found”

The chat endpoint requires `PreProcessing/ProcessedFiles/index/` to exist and contain `faiss.index` + `docs.json`.
- Rebuild the index (see [Rebuilding the FAISS index](#rebuilding-the-faiss-index))
- Or restore the `PreProcessing/ProcessedFiles/index/` folder

### Backend cannot reach the LLM

- If using Docker Compose, the backend calls `host.docker.internal:11434`.
  - Ensure Ollama is running on the host and listening on port `11434`.
- If running locally (no Docker), set `LLM_BASE_URL=http://localhost:11434/v1`.

### `config.yml` seems ignored

`server.py` loads YAML only if `PyYAML` is installed. Install `pyyaml` or add it to the Docker image.

### Streaming doesn’t work

The streaming endpoint uses SSE. If you’re calling it from a client:
- do not buffer the response
- ensure proxies (if any) do not disable streaming

---

## License / citation

This project is released under the **MIT License**. See [LICENSE](LICENSE).

If you use this project in academic work, please cite the (placeholder) paper below.

BibTeX (arXiv preprint; fill in the arXiv identifier/URL):

```bibtex
@misc{immordino2026retrievalaugmented,
  title         = {A Retrieval-Augmented Language Assistant for Unmanned Aircraft Safety Assessment and Regulatory Compliance},
  author        = {Immordino, Gabriele and Vaiuso, Andrea and Righi, Marcello},
  year          = {2026},
  howpublished  = {arXiv preprint},
  eprint        = {arXiv:XXXX.XXXXX},
  url           = {REPLACE_WITH_ARXIV_URL}
}
```
