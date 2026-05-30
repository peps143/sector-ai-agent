"""
FastAPI server for the Sector AI Agent
Auto-initializes on startup and logs every query to Supabase
"""

import os
import time
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from rag_agent import AgentConfig, SectorAgent

# ── Supabase client ───────────────────────────────────────────────────────────
def get_supabase() -> Client | None:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if url and key:
        return create_client(url, key)
    print("[WARN] Supabase not configured — logging disabled")
    return None

supabase: Client | None = None

# ── Domain detector ───────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "Transport":    ["road","transport","infrastructure","bridge","highway","procurement","contractor"],
    "Agriculture":  ["agri","farm","crop","food","irrigation","extension","digital agri","climate-smart"],
    "WASH":         ["water","sanitation","wash","tariff","nrw","hygiene","utility"],
    "Education":    ["school","teacher","learning","education","literacy","curriculum","reading"],
    "FCS":          ["fragile","conflict","fcs","post-conflict","humanitarian","displacement"],
}

def detect_domain(question: str) -> str:
    q = question.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k in q for k in keywords):
            return domain
    return "General"

# ── Quality auto-scorer ───────────────────────────────────────────────────────
def auto_score(answer: str, sources: list) -> dict:
    """
    Lightweight heuristic scoring based on RAGBench TRACe framework.
    In production you'd call GPT to score — this keeps latency low.
    """
    has_sources    = len(sources) > 0
    answer_len     = len(answer.split())
    has_numbers    = any(c.isdigit() for c in answer)
    has_caveats    = any(w in answer.lower() for w in ["however","but","note","caveat","uncertain"])
    source_variety = len(set(s.get("source","") for s in sources))

    relevance    = min(100, 70 + (answer_len // 20) + (10 if has_sources else 0))
    grounding    = min(100, 60 + (source_variety * 10) + (15 if has_sources else 0))
    completeness = min(100, 65 + (answer_len // 15) + (5 if has_numbers else 0))
    avg          = round((relevance + grounding + completeness) / 3, 1)
    hallucination_flag = avg < 72 or not has_sources

    return {
        "relevance_score":     round(relevance, 1),
        "grounding_score":     round(grounding, 1),
        "completeness_score":  round(completeness, 1),
        "avg_trace_score":     avg,
        "hallucination_flag":  hallucination_flag,
    }

# ── Logger ────────────────────────────────────────────────────────────────────
def log_query(question: str, answer: str, sources: list,
              latency: float, session_id: str, model: str):
    if not supabase:
        return
    try:
        scores = auto_score(answer, sources)
        supabase.table("query_logs").insert({
            "question":           question,
            "answer":             answer[:1000],   # trim very long answers
            "domain":             detect_domain(question),
            "sources_count":      len(sources),
            "latency_sec":        round(latency, 2),
            "relevance_score":    scores["relevance_score"],
            "grounding_score":    scores["grounding_score"],
            "completeness_score": scores["completeness_score"],
            "avg_trace_score":    scores["avg_trace_score"],
            "hallucination_flag": scores["hallucination_flag"],
            "session_id":         session_id,
            "model_used":         model,
        }).execute()
    except Exception as e:
        print(f"[WARN] Supabase log failed: {e}")

# ── Agent singleton ───────────────────────────────────────────────────────────
_agent: SectorAgent | None = None

def get_agent() -> SectorAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    return _agent

# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, supabase
    supabase = get_supabase()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        print("[INFO] Auto-initializing agent...")
        config = AgentConfig(openai_api_key=api_key)
        _agent = SectorAgent(config)
        _agent.initialize()
        print("[INFO] Agent ready!")
    else:
        print("[WARN] No OPENAI_API_KEY — agent not initialized.")
    yield

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sector AI Agent API",
    description="World Bank ITSEF-style RAG agent with Supabase observability",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ────────────────────────────────────────────────────────────
class InitRequest(BaseModel):
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 800
    retriever_k: int = 5
    force_rebuild: bool = False

class QueryRequest(BaseModel):
    question: str
    session_id: str = ""

class AddDocRequest(BaseModel):
    text: str
    title: str
    sector: str
    doc_type: str = "Manual Entry"

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/init")
def initialize(req: InitRequest):
    global _agent
    config = AgentConfig(
        openai_api_key=req.openai_api_key,
        llm_model=req.llm_model,
        chunk_size=req.chunk_size,
        retriever_k=req.retriever_k,
    )
    _agent = SectorAgent(config)
    return _agent.initialize(force_rebuild=req.force_rebuild)


@app.post("/query")
def query(req: QueryRequest):
    agent = get_agent()
    session_id = req.session_id or str(uuid.uuid4())[:8]
    t0 = time.time()
    try:
        result = agent.query(req.question)
        latency = time.time() - t0

        # Log to Supabase in background
        log_query(
            question=req.question,
            answer=result["answer"],
            sources=result.get("sources", []),
            latency=latency,
            session_id=session_id,
            model=agent.config.llm_model,
        )

        result["session_id"] = session_id
        result["latency_sec"] = round(latency, 2)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-document")
def add_document(req: AddDocRequest):
    agent = get_agent()
    meta = {"source": req.title, "sector": req.sector, "type": req.doc_type, "page": "N/A"}
    return agent.add_document_text(req.text, meta)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    agent = get_agent()
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    meta = {"source": file.filename, "sector": "Uploaded", "type": "Upload", "page": "N/A"}
    result = agent.add_document_text(text, meta)
    return {"filename": file.filename, **result}


@app.post("/reset")
def reset_conversation():
    get_agent().reset_conversation()
    return {"status": "conversation reset"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent_ready": _agent is not None and _agent.chain is not None,
        "logging_enabled": supabase is not None,
    }


@app.get("/stats")
def stats():
    """Return live stats from Supabase for the dashboard."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        result = supabase.table("query_logs").select(
            "id, created_at, domain, latency_sec, avg_trace_score, hallucination_flag, sources_count"
        ).order("created_at", desc=True).limit(500).execute()
        rows = result.data
        total = len(rows)
        if total == 0:
            return {"total_queries": 0, "rows": []}

        avg_latency   = round(sum(r["latency_sec"] or 0 for r in rows) / total, 2)
        avg_trace     = round(sum(r["avg_trace_score"] or 0 for r in rows) / total, 1)
        hall_count    = sum(1 for r in rows if r["hallucination_flag"])
        hall_rate     = round(hall_count / total * 100, 1)

        return {
            "total_queries":       total,
            "avg_latency_sec":     avg_latency,
            "avg_trace_score":     avg_trace,
            "hallucination_rate":  hall_rate,
            "rows":                rows[-100:],  # last 100 for charting
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Sector AI Agent API v2.0 — visit /docs for Swagger UI"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
