# 🌍 Sector AI Agent

**A RAG-based knowledge retrieval tool with real-time observability — inspired by how development organizations like the World Bank manage and surface institutional knowledge.**

---

## Why I Built This

One thing I've noticed working in international development is how much knowledge gets buried — in project completion reports, sector notes, implementation reviews — documents that exist but are rarely surfaced when they're actually needed.

I wanted to explore what it would look like to build an AI system that could actually *read* those documents and answer questions from them in real time. Not just search keywords, but understand context, retrieve relevant passages, and synthesize insights the way a knowledgeable colleague would.

This project is my attempt at that. It simulates the kind of knowledge agent the World Bank's ITSEF (Independent Evaluation Group) could use to help teams learn from past operations without having to manually dig through hundreds of reports.

After getting the agent working, I took it a step further — I wired real-time observability into it so every query gets logged, scored, and visualized on a live monitoring dashboard. That turned it from a prototype into something closer to a production system.

---

## What It Does

You can ask it questions like:

- *"What are the common failure modes in fragile state projects?"*
- *"How have water tariff reforms worked in MENA?"*
- *"What lessons exist on community ownership in rural roads?"*
- *"What monitoring approaches work best in conflict-affected states?"*

It searches a knowledge base of sector documents, retrieves the most relevant passages, and generates a grounded answer — with citations showing exactly which documents it pulled from. Every query is automatically logged to a Supabase database with quality scores and latency metrics, which feed into a live performance dashboard.

---

## Live Links

| | |
|---|---|
| 🤖 **Agent UI** | https://peps143.github.io/sector-ai-agent/frontend/index.html |
| 📊 **Live Dashboard** | https://peps143.github.io/sector-ai-agent/dashboard.html |
| ⚙️ **Backend API** | https://sector-ai-agent.onrender.com |
| 💻 **API Docs** | https://sector-ai-agent.onrender.com/docs |
| 🗄️ **Database** | Supabase · https://snlyyitkpyvmsbfmddkk.supabase.co |

---

## System Architecture

```
User Query
     ↓
FastAPI Backend (Render)
     ↓                    ↓
LangChain RAG         Query Logger
     ↓                    ↓
FAISS Vector Index    Supabase PostgreSQL
     ↓                    ↓
GPT-4o-mini          Live Dashboard
     ↓               (GitHub Pages)
Grounded Answer
+ Source Citations
```

### RAG Pipeline Detail

```
Sector Documents (PDF / TXT)
        ↓
Text Chunking (800-char chunks, 150 overlap)
        ↓
OpenAI Embeddings (text-embedding-3-small) → FAISS Vector Index
        ↓
MMR Retrieval (diverse, relevant chunks — k=5)
        ↓
GPT-4o-mini generates answer with citations
        ↓
Response logged to Supabase with TRACe quality scores
```

**MMR (Maximal Marginal Relevance)** — instead of returning the five most similar chunks, it balances relevance *and* diversity, so you get a richer set of sources rather than five versions of the same paragraph.

---

## Real-Time Observability

Every query the agent receives is automatically logged to a Supabase PostgreSQL database with:

| Field | Description |
|-------|-------------|
| `question` | The user's query |
| `answer` | The agent's response (trimmed to 1000 chars) |
| `domain` | Auto-detected sector (Transport, WASH, Agriculture, etc.) |
| `latency_sec` | Response time in seconds |
| `relevance_score` | TRACe relevance score (0–100) |
| `grounding_score` | TRACe grounding score (0–100) |
| `completeness_score` | TRACe completeness score (0–100) |
| `avg_trace_score` | Average across all TRACe dimensions |
| `hallucination_flag` | True if avg score < 72 or no sources retrieved |
| `sources_count` | Number of documents retrieved |
| `session_id` | Session identifier |
| `model_used` | LLM model used for generation |

The `/stats` endpoint aggregates this data and serves it to the dashboard in real time.

---

## Live Dashboard

The performance dashboard at `https://peps143.github.io/sector-ai-agent/dashboard.html` reads directly from the `/stats` endpoint and shows:

- **6 KPI cards** — total queries, avg latency, avg TRACe score, hallucination rate, active domains, sources per query
- **Query volume chart** — queries per day over time
- **Domain distribution** — which knowledge sectors are being queried most
- **Latency trend** — response time per query
- **TRACe quality scores** — relevance, grounding, completeness averaged across all sessions
- **Hallucination rate** — percentage of flagged responses per day
- **Weekly heatmap** — activity by day over 4 weeks
- **Live query feed** — last 6 real queries with scores
- **Period filter** — toggle between 7D / 30D / 90D views
- **Auto-refresh** — updates every 60 seconds

Quality scoring methodology is based on the **RAGBench TRACe evaluation framework** (Rungalileo, 2024).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| RAG pipeline | LangChain 0.3.25 |
| Vector store | FAISS (local) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-4o-mini |
| API backend | FastAPI + Uvicorn |
| Observability | Supabase PostgreSQL |
| Frontend | HTML / CSS / Chart.js |
| Backend hosting | Render (free tier) |
| Frontend hosting | GitHub Pages |

---

## Built-in Knowledge Base

The agent comes seeded with five synthetic but realistic documents modeled on real World Bank formats:

| Document | Sector | Type |
|----------|--------|------|
| Rural Roads ICR (P123456) | Transport | ICR |
| Digital Agriculture Sector Note | Agriculture | ITSEF Note |
| Urban WASH PAD (P198765) | WASH | PAD |
| Education ISR — West Africa | Education | ISR |
| FCS Portfolio Synthesis (45 projects) | Cross-Cutting | Synthesis |

You can upload your own `.txt` or `.pdf` documents through the upload panel in the UI.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/init` | Initialize agent with config |
| `POST` | `/query` | RAG query with conversation memory + Supabase logging |
| `GET` | `/stats` | Live aggregated metrics from Supabase |
| `POST` | `/add-document` | Add document text dynamically |
| `POST` | `/upload` | Upload .txt file to knowledge base |
| `POST` | `/reset` | Reset conversation history |
| `GET` | `/health` | Health check (agent ready + logging enabled) |

---

## Project Structure

```
sector-ai-agent/
├── backend/
│   ├── rag_agent.py        # Core RAG pipeline
│   ├── server.py           # FastAPI server + Supabase logging
│   └── requirements.txt
├── frontend/
│   └── index.html          # Agent UI
├── dashboard.html          # Live performance dashboard
├── data/
│   └── sample_docs/        # Drop PDF/TXT documents here
├── .env.example
└── README.md
```

---


## What's Next

- Add real World Bank ICR documents from the public open data repository
- Build a multi-sector router that dispatches queries to specialist sub-agents
- Experiment with Chroma or Pinecone for persistent cloud vector storage
- Add user feedback (thumbs up/down) to replace heuristic TRACe scoring with real signal
- Add a document comparison feature ("how did this project's approach differ from similar ones?")

---

*Built by Perpetual T. Adu*
