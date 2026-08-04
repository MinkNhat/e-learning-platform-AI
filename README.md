# E-learning Agentic RAG

A e-learning RAG chatbot built with LangGraph, Portkey Gateway, MongoDB Atlas, and Qdrant

## Key Features

- **Intent Routing**: The planner separates conversational, general knowledge, course recommendation, and authorized lesson Q&A.
- **Guardrails**: NeMo Guardrails uses a dedicated Portkey-routed model to block off-topic, jailbreak, and injection inputs before any retrieval.
- **LLM Gateway**: Portkey routes guardrail, RAG generation, and embedding calls with centralized provider config.
- **MongoDB Course Indexing**: Published courses and active modules/lessons are joined through a read-only Atlas connection.
- **Scoped Retrieval**: Protected lesson chunks are always filtered by backend-verified `allowed_course_ids`.
- **Source Provenance**: Responses contain structured file, course, module, and lesson metadata.
- **Local Document Parsing**: PDF, HTML, TXT, DOCX, PPTX parsed entirely on-device — no external OCR service.
- **Observability**: Full trace nesting with **Pydantic Logfire** and **LangSmith** across every agent node.

---

## Project Structure

```text
├── app/
│   ├── agents/
│   │   └── nodes/       # Planner, Retriever, Responder LangGraph nodes
│   ├── gateway/         # Portkey LLM gateway — primary/secondary/guardrail routing
│   ├── guardrails/      # NeMo Guardrails input/output filtering
│   ├── ingestion/
│   │   ├── chunking/    # Normalized recursive splitter (1600 char max)
│   │   ├── loaders/     # Local parsers — PDF (pypdf), HTML, TXT, DOCX, PPTX
│   │   ├── processor.py # Full MongoDB + file rebuild into Qdrant
│   │   ├── mongodb_reader.py
│   │   └── course_chunks.py
│   ├── services/
│   │   └── retrieval/   # Portkey embeddings + Qdrant search + FlashRank reranking
│   ├── config.py        # Centralized environment variable management
│   └── main.py          # FastAPI entrypoint — authenticated SSE query endpoint
├── DATA/                # Sample documents for ingestion
├── MONGODB_IMPORT_GUIDE.md
├── REFACTOR.md
├── Dockerfile           # Container definition (retained for reference)
└── requirements.txt     # Project dependencies
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain + LangGraph |
| LLMs | Primary, secondary, and guardrail tiers via **Portkey** gateway configs |
| Guardrails | NeMo Guardrails + Portkey-routed LLM |
| Vector DB | Qdrant Cloud |
| Course source | MongoDB Atlas through a read-only account |
| Reranking | FlashRank (local, zero-latency) |
| Embeddings | Dedicated Portkey embedding config |
| Document Parsing | pypdf + local HTML/Office parsers |
| Observability | Pydantic Logfire + LangSmith |

---

## Run locally

### 1. Create venv and Install dependencies

```powershell
python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file with the following keys:

```env
# LLM Gateway
PORTKEY_API_KEY = "..."
PORTKEY_PRIMARY_CONFIG_ID = "..."
PORTKEY_SECONDARY_CONFIG_ID = "..."
PORTKEY_GUARDRAIL_CONFIG_ID = "..."
PORTKEY_EMBEDDING_CONFIG_ID = "..."

# Vector DB
QDRANT_API_KEY = "..."
QDRANT_CLUSTER_ENDPOINT = "https://your-cluster.cloud.qdrant.io:6333"
QDRANT_COLLECTION = "elearning_rag"

# MongoDB read-only indexing connection
MONGODB_READ_URI = "mongodb+srv://rag_reader:...@cluster/learning"
MONGODB_DATABASE = "learning"

# Backend-signed RAG JWT public key (base64-encoded PEM)
RAG_JWT_PUBLIC_KEY = "..."

# Observability
LOGFIRE_TOKEN = "..."
LANGSMITH_API_KEY = "..."
LANGSMITH_PROJECT = "elearning_rag"
LANGSMITH_TRACING = true
LANGSMITH_ENDPOINT = https://api.smith.langchain.com
```

### 3. Run data ingestion

The default command indexes both general files in `DATA/` and published
MongoDB course content. It deletes and recreates `QDRANT_COLLECTION`, rebuilds
all embeddings, and validates the total point count.

```powershell
python -m app.ingestion.processor DATA
```

### 4. Backend query contract

The frontend calls `POST /api/v1/rag/query/stream` on the platform backend with the
normal user access token:

```json
{
  "q": "Tóm tắt phần này",
  "conversationId": "conversation-id",
  "courseId": "course-id",
  "moduleId": "module-id",
  "lessonId": "lesson-id"
}
```

The backend validates the hierarchy and active enrollment, signs the authorized
scope into a two-minute RS256 JWT, persists the conversation, and streams the AI
service response as SSE. The AI service only keeps the public key and never
queries enrollment data at request time.

### 5. Launch the app

```powershell
uv run uvicorn app.main:app --reload --port 8001
```