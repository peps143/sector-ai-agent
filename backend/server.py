"""
FastAPI server for the Sector AI Agent
Exposes REST endpoints consumed by the frontend UI
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_agent import AgentConfig, SectorAgent

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sector AI Agent API",
    description="World Bank ITSEF-style RAG knowledge agent",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Agent singleton ───────────────────────────────────────────────────────────
_agent: SectorAgent | None = None


def get_agent() -> SectorAgent:
    global _agent
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. POST /init first.")
    return _agent


# ── Request / Response models ─────────────────────────────────────────────────
class InitRequest(BaseModel):
    openai_api_key: str
    docs_dir: str = "./data/sample_docs"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 800
    chunk_overlap: int = 150
    retriever_k: int = 5
    force_rebuild: bool = False


class QueryRequest(BaseModel):
    question: str


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
        docs_dir=req.docs_dir,
        llm_model=req.llm_model,
        embedding_model=req.embedding_model,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        retriever_k=req.retriever_k,
    )
    _agent = SectorAgent(config)
    result = _agent.initialize(force_rebuild=req.force_rebuild)
    return result


@app.post("/query")
def query(req: QueryRequest):
    agent = get_agent()
    try:
        result = agent.query(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-document")
def add_document(req: AddDocRequest):
    agent = get_agent()
    meta = {
        "source": req.title,
        "sector": req.sector,
        "type": req.doc_type,
        "page": "N/A",
    }
    result = agent.add_document_text(req.text, meta)
    return result


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a .txt file to the docs directory."""
    agent = get_agent()
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    meta = {
        "source": file.filename,
        "sector": "Uploaded",
        "type": "Upload",
        "page": "N/A",
    }
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
    }


@app.get("/")
def root():
    return {"message": "Sector AI Agent API — visit /docs for Swagger UI"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
