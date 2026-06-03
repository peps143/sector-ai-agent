"""
FastAPI server for the Sector AI Agent — Multi-Agent Edition
Orchestrates a 5-agent LangGraph pipeline with real FAISS retrieval
Auto-initializes on startup and logs every query to Supabase
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from rag_agent import AgentConfig, SectorAgent
from pipeline import run_pipeline, get_vectorstore

# ── Supabase ──────────────────────────────────────────────────────────────────
supabase: Client | None = None

def get_supabase() -> Client | None:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if url and key:
        return create_client(url, key)
    return None

# ── Quality scorer ────────────────────────────────────────────────────────────
def auto_score(answer: str, sources: list) -> dict:
    has_sources    = len(sources) > 0
    answer_len     = len(answer.split())
    source_variety = len(set(s.get("source", "") for s in sources))
    relevance      = min(100, 70 + (answer_len // 20) + (10 if has_sources else 0))
    grounding      = min(100, 60 + (source_variety * 10) + (15 if has_sources else 0))
    completeness   = min(100, 65 + (answer_len // 15))
    avg            = round((relevance + grounding + completeness) / 3, 1)
    return {
        "relevance_score":    round(relevance, 1),
        "grounding_score":    round(grounding, 1),
        "completeness_score": round(completeness, 1),
        "avg_trace_score":    avg,
        "hallucination_flag": avg < 72 or not has_sources,
    }

def detect_domain(question: str) -> str:
    from pipeline import detect_domain as _dd
    return _dd(question)

def log_query(question, answer, sources, latency, session_id, model, domain):
    if not supabase:
        return
    try:
        scores = auto_score(answer, sources)
        supabase.table("query_logs").insert({
            "question":           question,
            "answer":             answer[:1000],
            "domain":             domain,
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

# ── Agent singleton (legacy RAG for uploads) ──────────────────────────────────
_agent: SectorAgent | None = None

# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, supabase
    supabase = get_supabase()
    api_key  = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        print("[INFO] Pre-loading FAISS vector store…")
        get_vectorstore()
        config  = AgentConfig(openai_api_key=api_key)
        _agent  = SectorAgent(config)
        _agent.initialize()
        print("[INFO] Sector AI Agent (multi-agent edition) ready!")
    yield

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sector AI Agent API — Multi-Agent Edition",
    description="LangGraph 5-agent pipeline with real FAISS retrieval + Supabase observability",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: str = ""
    use_pipeline: bool = True   # True = multi-agent, False = legacy RAG

class AddDocRequest(BaseModel):
    text: str
    title: str
    sector: str
    doc_type: str = "Manual Entry"

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/query")
def query(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())[:8]
    t0 = time.time()

    try:
        if req.use_pipeline:
            # ── Multi-agent LangGraph pipeline ──
            result  = run_pipeline(req.question)
            latency = time.time() - t0

            sources = [
                {"source": d["source"], "page": d["page"], "snippet": d["snippet"]}
                for d in result["wbs_docs"]
            ]

            log_query(req.question, result["wbs_answer"], sources,
                      latency, session_id, "gpt-4o-mini", result["wbs_domain"])

            return {
                "answer":        result["wbs_answer"],
                "sources":       sources,
                "domain":        result["wbs_domain"],
                "risks":         result["wbs_risks"],
                "agent_trace":   result["wbs_trace"],
                "session_id":    session_id,
                "latency_sec":   round(latency, 2),
                "pipeline_mode": "multi-agent",
            }
        else:
            # ── Legacy single RAG chain ──
            if not _agent or not _agent.chain:
                raise HTTPException(status_code=503, detail="Agent not initialized")
            result  = _agent.query(req.question)
            latency = time.time() - t0
            log_query(req.question, result["wbs_answer"], result.get("wbs_docs", []),
                      latency, session_id, "gpt-4o-mini", detect_domain(req.question))
            result["session_id"]    = session_id
            result["latency_sec"]   = round(latency, 2)
            result["pipeline_mode"] = "single-rag"
            return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-document")
def add_document(req: AddDocRequest):
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    meta   = {"source": req.title, "sector": req.sector, "type": req.doc_type, "page": "N/A"}
    result = _agent.add_document_text(req.text, meta)
    # Reload vectorstore for pipeline
    global _vectorstore
    from pipeline import _vectorstore as pv
    _vectorstore = None
    return result


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    content = await file.read()
    text    = content.decode("utf-8", errors="replace")
    meta    = {"source": file.filename, "sector": "Uploaded", "type": "Upload", "page": "N/A"}
    result  = _agent.add_document_text(text, meta)
    return {"filename": file.filename, **result}


@app.post("/reset")
def reset():
    if _agent:
        _agent.reset_conversation()
    return {"status": "reset"}


@app.get("/health")
def health():
    return {
        "status":          "ok",
        "agent_ready":     _agent is not None,
        "logging_enabled": supabase is not None,
        "pipeline_mode":   "multi-agent (LangGraph)",
    }


@app.get("/stats")
def stats():
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        result = supabase.table("query_logs").select(
            "id, created_at, domain, latency_sec, avg_trace_score, hallucination_flag, sources_count"
        ).order("created_at", desc=True).limit(500).execute()
        rows  = result.data
        total = len(rows)
        if total == 0:
            return {"total_queries": 0, "rows": []}
        avg_latency = round(sum(r["latency_sec"] or 0 for r in rows) / total, 2)
        avg_trace   = round(sum(r["avg_trace_score"] or 0 for r in rows) / total, 1)
        hall_rate   = round(sum(1 for r in rows if r["hallucination_flag"]) / total * 100, 1)
        return {
            "total_queries":      total,
            "avg_latency_sec":    avg_latency,
            "avg_trace_score":    avg_trace,
            "hallucination_rate": hall_rate,
            "rows":               rows[-100:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class FeedbackRequest(BaseModel):
    rating: int        # 1 = thumbs up, -1 = thumbs down
    message_id: str = ""
    comment: str = ""


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Log user feedback (thumbs up/down) to Supabase."""
    if supabase:
        try:
            supabase.table("feedback_logs").insert({
                "rating":     req.rating,
                "message_id": req.message_id,
                "comment":    req.comment,
            }).execute()
        except Exception as e:
            print(f"[WARN] Feedback log failed: {e}")
    return {"status": "recorded", "rating": req.rating}


@app.get("/")
def root():
    return {"message": "Sector AI Agent API v3.0 — Multi-Agent Edition · /docs for Swagger"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
