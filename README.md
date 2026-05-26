# 🌍 Sector AI Agent
### RAG-based Domain Knowledge Agent — World Bank ITSEF Use Case

A production-ready Retrieval-Augmented Generation (RAG) system that retrieves operational insights and lessons from sector documents, simulating the World Bank ITSEF (Independent Evaluation Group) knowledge agent architecture.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sector AI Agent Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Documents (PDF/TXT)                                            │
│       ↓                                                         │
│  DocumentIngestor  →  RecursiveCharacterTextSplitter            │
│       ↓                                                         │
│  OpenAIEmbeddings (text-embedding-3-small)                      │
│       ↓                                                         │
│  FAISS VectorStore  (persisted to disk)                         │
│       ↓                                                         │
│  MMR Retriever  (Maximal Marginal Relevance, k=5)               │
│       ↓                                                         │
│  ConversationalRetrievalChain + ConversationBufferMemory        │
│       ↓                                                         │
│  ChatOpenAI (gpt-4o-mini)  +  Domain System Prompt             │
│       ↓                                                         │
│  Grounded Answer + Source Citations                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

- **RAG Pipeline**: LangChain + FAISS + OpenAI embeddings
- **MMR Retrieval**: Maximal Marginal Relevance for diverse, non-redundant sources
- **Conversational Memory**: Multi-turn dialogue with history awareness
- **Dynamic Ingestion**: Add documents at runtime without rebuilding the index
- **Domain Prompt**: ITSEF-style sector knowledge agent persona
- **REST API**: FastAPI server exposing all agent capabilities
- **Web UI**: Dark-themed dashboard with source citations

---

## Quickstart

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. (Optional) Add your own documents

Drop `.pdf` or `.txt` files into `data/sample_docs/`. If empty, the agent seeds 5 built-in ITSEF-style documents automatically.

### 4. Start the backend API

```bash
cd backend
uvicorn server:app --reload --port 8000
```

### 5. Open the UI

Open `frontend/index.html` in your browser (or serve it):

```bash
cd frontend
python -m http.server 3000
# → http://localhost:3000
```

### 6. Initialize and query

1. Enter your OpenAI API key in the sidebar
2. Click **Initialize Agent**
3. Ask sector questions!

---

## CLI Mode

Skip the UI and query directly from terminal:

```bash
cd backend
OPENAI_API_KEY=sk-... python rag_agent.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/init` | Initialize agent with config |
| `POST` | `/query` | RAG query with conversation memory |
| `POST` | `/add-document` | Add document text dynamically |
| `POST` | `/upload` | Upload .txt file |
| `POST` | `/reset` | Reset conversation history |
| `GET` | `/health` | Health check |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are key lessons on road maintenance?"}'
```

### Example: Init

```bash
curl -X POST http://localhost:8000/init \
  -H "Content-Type: application/json" \
  -d '{
    "openai_api_key": "sk-...",
    "llm_model": "gpt-4o-mini",
    "retriever_k": 5
  }'
```

---

## Built-in Knowledge Base

The agent ships with 5 seed documents (real-format, synthetic content):

| Document | Sector | Type |
|----------|--------|------|
| Rural Roads ICR (P123456) | Transport | ICR |
| Digital Agriculture Sector Note | Agriculture | ITSEF Note |
| Urban WASH PAD (P198765) | WASH | PAD |
| Education ISR — West Africa | Education | ISR |
| FCS Portfolio Synthesis | Cross-Cutting | Synthesis |

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_model` | `gpt-4o-mini` | OpenAI chat model |
| `embedding_model` | `text-embedding-3-small` | Embedding model |
| `chunk_size` | `800` | Characters per chunk |
| `chunk_overlap` | `150` | Overlap between chunks |
| `retriever_k` | `5` | Documents retrieved per query |
| `temperature` | `0.2` | LLM temperature |

---

## Project Structure

```
sector-ai-agent/
├── backend/
│   ├── rag_agent.py        # Core RAG pipeline (agent, chain, ingestor)
│   ├── server.py           # FastAPI REST server
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Single-file dashboard UI
├── data/
│   ├── sample_docs/        # Drop PDF/TXT documents here
│   └── vectorstore/        # FAISS index (auto-created)
├── .env.example            # Environment template
└── README.md
```

---

## Extending the Agent

### Add a new sector

Edit the `SECTOR_AGENT_PROMPT` in `rag_agent.py` to include domain-specific instructions.

### Swap the vector store

Replace `FAISS` with `Chroma`, `Pinecone`, or `Weaviate` — the `VectorStoreManager` class isolates this concern.

### Add PDF support

Ensure `pypdf` is installed (included in requirements), then drop PDFs into `data/sample_docs/`.

### Multi-agent routing

Wrap `SectorAgent` instances (one per sector) and add a router agent that classifies the query and dispatches.

---

## Cost Estimate

| Operation | Model | Approx Cost |
|-----------|-------|------------|
| Embedding 5 seed docs | text-embedding-3-small | ~$0.0001 |
| Per query (retrieval + generation) | gpt-4o-mini | ~$0.001 |
| Full session (20 queries) | gpt-4o-mini | ~$0.02 |
