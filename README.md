# 🌍 Sector AI Agent

**A RAG-based knowledge retrieval tool inspired by how development organizations like the World Bank manage and surface institutional knowledge.**

---

## Why I Built This

One thing I've noticed working in international development is how much knowledge gets buried — in project completion reports, sector notes, implementation reviews — documents that exist but are rarely surfaced when they're actually needed.

I wanted to explore what it would look like to build an AI system that could actually *read* those documents and answer questions from them in real time. Not just search keywords, but understand context, retrieve relevant passages, and synthesize insights the way a knowledgeable colleague would.

This project is my attempt at that. It simulates the kind of knowledge agent the World Bank's ITSEF (Independent Evaluation Group) could use to help teams learn from past operations without having to manually dig through hundreds of reports.

---

## What It Does

You can ask it questions like:

- *"What are the common failure modes in fragile state projects?"*
- *"How have water tariff reforms worked in MENA?"*
- *"What lessons exist on community ownership in rural roads?"*

It searches a knowledge base of sector documents, retrieves the most relevant passages, and generates a grounded answer — with citations showing exactly which documents it pulled from.

---

## How It Works

The core architecture is a **Retrieval-Augmented Generation (RAG)** pipeline:

```
Sector Documents (PDF / TXT)
        ↓
Text Chunking (800-char chunks, 150 overlap)
        ↓
OpenAI Embeddings → FAISS Vector Index
        ↓
MMR Retrieval (finds diverse, relevant chunks)
        ↓
GPT-4o-mini generates answer with citations
```

**MMR (Maximal Marginal Relevance)** is the part I found most interesting to work with — instead of just returning the most similar chunks, it balances relevance *and* diversity, so you get a richer set of sources rather than five versions of the same paragraph.

---

## Tech Stack

- **LangChain** — RAG pipeline and conversational memory
- **FAISS** — local vector store for fast similarity search
- **OpenAI** — embeddings (`text-embedding-3-small`) and chat (`gpt-4o-mini`)
- **FastAPI** — REST API backend
- **GitHub Pages** — frontend hosting
- **Render** — backend deployment

---

## Built-in Knowledge Base

The agent comes seeded with five synthetic but realistic documents modeled on real World Bank formats:

| Document | Sector |
|----------|--------|
| Rural Roads ICR (P123456) | Transport |
| Digital Agriculture Sector Note | Agriculture |
| Urban WASH PAD (P198765) | WASH |
| Education ISR — West Africa | Education |
| FCS Portfolio Synthesis (45 projects) | Cross-Cutting |



---

## Try It

🌐 **Live Demo:** [sector-ai-agent on GitHub Pages](https://peps143.github.io/sector-ai-agent/frontend/index.html)

 Just type a question.

---

## Run It Locally

```bash
git clone https://github.com/peps143/sector-ai-agent.git
cd sector-ai-agent/backend
pip install -r requirements.txt

# Add your OpenAI key
cp .env.example .env

uvicorn server:app --reload --port 8000
```

Then open `frontend/index.html` in your browser.

---

## What I Learned

This project pushed me to understand things I hadn't worked with before — vector embeddings, semantic search, how LLMs use retrieved context to ground their answers. I also had to figure out deployment, CORS, environment variables, and why Python version mismatches can ruin your whole afternoon.

The part that surprised me most was how much the *quality of the prompt* affects the output. The domain-specific system prompt — framing the model as an ITSEF sector knowledge agent — made a significant difference in how structured and useful the answers were.

---

## What's Next to Do

- Add real World Bank ICR documents from the public repository
- Build a multi-sector router that dispatches queries to specialist sub-agents
- Experiment with Chroma or Pinecone for persistent cloud vector storage
- Add a document comparison feature ("how did this project's approach differ from similar ones?")

---

