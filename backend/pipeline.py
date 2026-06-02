"""
Sector AI Agent — Multi-Agent Pipeline (LangGraph)
5 specialized agents orchestrated by LangGraph, using the real FAISS vector store
for grounded retrieval from actual sector documents.

Pipeline:
  Router → Retrieval → Reasoning → Risk → Synthesis
"""

import os
import re
import json
import time
from typing import TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from rag_agent import AgentConfig, VectorStoreManager, DocumentIngestor, _seed_sample_knowledge


# ── Shared state ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query:           str
    domain:          str
    retrieved_docs:  list[dict]
    analysis:        str
    risks:           list[dict]
    final_answer:    str
    agent_trace:     list[dict]
    current_agent:   str
    error:           str


# ── Domain detection ──────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "Transport":     ["road","transport","bridge","highway","procurement","maintenance","rural road","contractor"],
    "Agriculture":   ["agri","farm","crop","food","irrigation","extension","digital agri","climate-smart","smallholder"],
    "WASH":          ["water","sanitation","wash","tariff","nrw","hygiene","utility","non-revenue"],
    "Education":     ["school","teacher","learning","education","literacy","curriculum","reading","grade"],
    "FCS":           ["fragile","conflict","fcs","post-conflict","humanitarian","displacement","adaptive"],
    "Cross-Cutting": ["gender","climate","carbon","monitoring","capacity","procurement","sustainability"],
}


def detect_domain(query: str) -> str:
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )


def trace_entry(agent: str, input_summary: str, output_summary: str, duration_ms: int) -> dict:
    return {
        "agent":          agent,
        "timestamp":      datetime.now().isoformat(),
        "input_summary":  input_summary,
        "output_summary": output_summary,
        "duration_ms":    duration_ms,
        "status":         "success",
    }


# ── FAISS store loader ────────────────────────────────────────────────────────
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    api_key = os.getenv("OPENAI_API_KEY", "")
    config = AgentConfig(openai_api_key=api_key)
    vs_manager = VectorStoreManager(config)

    store = vs_manager.load()
    if store is None:
        # Build from seed documents
        ingestor = DocumentIngestor(config)
        docs = ingestor.load_documents()
        if not docs:
            docs = _seed_sample_knowledge()
        chunks = ingestor.ingest(docs)
        store = vs_manager.build(chunks)

    _vectorstore = store
    return _vectorstore


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Router
# ══════════════════════════════════════════════════════════════════════════════
def router_agent(state: AgentState) -> AgentState:
    t0 = time.time()
    query  = state["query"]
    domain = detect_domain(query)
    llm    = get_llm()

    response = llm.invoke([
        SystemMessage(content="""You are a Router Agent for a World Bank ITSEF knowledge system.
        Classify the query domain and identify what the retrieval agent should focus on.
        Be concise — 2 sentences max."""),
        HumanMessage(content=f"Query: {query}\nDetected domain: {domain}\n\nConfirm domain and state retrieval focus.")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Router Agent",
        f"Query: {query[:80]}",
        f"Domain: {domain} — {response.content[:100]}",
        duration,
    )
    return {**state, "domain": domain, "current_agent": "retrieval", "agent_trace": state.get("agent_trace", []) + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Retrieval (uses real FAISS vector store)
# ══════════════════════════════════════════════════════════════════════════════
def retrieval_agent(state: AgentState) -> AgentState:
    t0    = time.time()
    query = state["query"]

    try:
        store     = get_vectorstore()
        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )
        docs = retriever.invoke(query)

        retrieved = []
        for doc in docs:
            retrieved.append({
                "source":  doc.metadata.get("source", "Unknown"),
                "sector":  doc.metadata.get("sector", state["domain"]),
                "page":    doc.metadata.get("page", "N/A"),
                "content": doc.page_content,
                "snippet": doc.page_content[:200] + "…",
            })

        duration = int((time.time() - t0) * 1000)
        trace = trace_entry(
            "Retrieval Agent",
            f"MMR search: '{query[:60]}' in FAISS index",
            f"Retrieved {len(retrieved)} chunks from {len(set(d['source'] for d in retrieved))} documents",
            duration,
        )
        return {**state, "retrieved_docs": retrieved, "current_agent": "reasoning", "agent_trace": state["agent_trace"] + [trace]}

    except Exception as e:
        duration = int((time.time() - t0) * 1000)
        trace = trace_entry("Retrieval Agent", query, f"Error: {str(e)}", duration)
        trace["status"] = "error"
        return {**state, "retrieved_docs": [], "error": str(e), "current_agent": "reasoning", "agent_trace": state["agent_trace"] + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Reasoning
# ══════════════════════════════════════════════════════════════════════════════
def reasoning_agent(state: AgentState) -> AgentState:
    t0  = time.time()
    llm = get_llm()

    docs_text = "\n\n".join(
        f"[Source: {d['source']} | Sector: {d['sector']}]\n{d['content']}"
        for d in state["retrieved_docs"]
    ) if state["retrieved_docs"] else "No documents retrieved."

    response = llm.invoke([
        SystemMessage(content="""You are a Reasoning Agent for a World Bank ITSEF knowledge system.
        Analyze retrieved sector documents and extract structured operational insights.
        Focus on: key lessons learned, success factors, patterns across projects, and evidence-based findings.
        Structure your response with clear numbered points. Cite specific sources where possible."""),
        HumanMessage(content=f"""Query: {state['query']}
Domain: {state['domain']}

Retrieved Documents:
{docs_text}

Extract 4-5 key analytical insights relevant to the query, with evidence from the documents.""")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Reasoning Agent",
        f"Analyzed {len(state['retrieved_docs'])} retrieved chunks",
        f"Generated {len(response.content.split(chr(10)))} lines of structured analysis",
        duration,
    )
    return {**state, "analysis": response.content, "current_agent": "risk", "agent_trace": state["agent_trace"] + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Risk
# ══════════════════════════════════════════════════════════════════════════════
def risk_agent(state: AgentState) -> AgentState:
    t0  = time.time()
    llm = get_llm(temperature=0.1)

    response = llm.invoke([
        SystemMessage(content="""You are a Risk Agent for a World Bank ITSEF knowledge system.
        Identify operational risks based on the sector context and analysis provided.
        Return ONLY a JSON array with this exact structure:
        [{"risk": "risk name", "severity": "High/Medium/Low", "mitigation": "brief mitigation strategy"}]
        No other text — just the JSON array."""),
        HumanMessage(content=f"""Query: {state['query']}
Domain: {state['domain']}
Analysis summary: {state['analysis'][:600]}

Identify 3-4 key operational risks relevant to this query and context.""")
    ])

    try:
        raw   = response.content.strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        risks = json.loads(match.group()) if match else []
    except Exception:
        risks = [
            {"risk": "Implementation capacity gap",      "severity": "High",   "mitigation": "Early capacity assessment and phased rollout plan"},
            {"risk": "Stakeholder coordination failure", "severity": "Medium", "mitigation": "Establish inter-agency working group at project start"},
            {"risk": "Sustainability post-completion",   "severity": "Medium", "mitigation": "Government ownership and budget integration from Year 1"},
        ]

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Risk Agent",
        f"Assessed {state['domain']} context for operational risks",
        f"Identified {len(risks)} risks: {', '.join(r['risk'] for r in risks[:2])}…",
        duration,
    )
    return {**state, "risks": risks, "current_agent": "synthesis", "agent_trace": state["agent_trace"] + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Synthesis
# ══════════════════════════════════════════════════════════════════════════════
def synthesis_agent(state: AgentState) -> AgentState:
    t0  = time.time()
    llm = get_llm()

    sources_text = "\n".join(
        f"• {d['source']} (p.{d['page']})" for d in state["retrieved_docs"]
    )
    risks_text = "\n".join(
        f"[{r['severity']}] {r['risk']}: {r['mitigation']}"
        for r in state["risks"]
    )

    response = llm.invoke([
        SystemMessage(content="""You are a Synthesis Agent for a World Bank ITSEF knowledge system.
        You combine insights from multiple specialized agents into a clear, actionable, grounded response.
        Write for a development professional audience. Be specific and cite source documents.
        Structure: direct answer → key insights (numbered) → recommendations."""),
        HumanMessage(content=f"""Query: {state['query']}
Domain: {state['domain']}

Reasoning Agent Analysis:
{state['analysis']}

Risk Agent Findings:
{risks_text}

Source Documents Used:
{sources_text}

Write a comprehensive, grounded final response to the query.""")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Synthesis Agent",
        "Combining reasoning + risk analysis + source citations",
        f"Generated final response ({len(response.content)} chars) from {len(state['retrieved_docs'])} sources",
        duration,
    )
    return {**state, "final_answer": response.content, "current_agent": "complete", "agent_trace": state["agent_trace"] + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline():
    graph = StateGraph(AgentState)

    graph.add_node("router",    router_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("reasoning", reasoning_agent)
    graph.add_node("risk",      risk_agent)
    graph.add_node("synthesis", synthesis_agent)

    graph.set_entry_point("router")
    graph.add_edge("router",    "retrieval")
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "risk")
    graph.add_edge("risk",      "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


def run_pipeline(query: str) -> AgentState:
    pipeline = build_pipeline()
    initial: AgentState = {
        "query":          query,
        "domain":         "",
        "retrieved_docs": [],
        "analysis":      "",
        "risks":          [],
        "final_answer":   "",
        "agent_trace":    [],
        "current_agent":  "router",
        "error":          "",
    }
    return pipeline.invoke(initial)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What are the key lessons on community ownership in rural road projects?"

    print(f"\n🔍 Query: {query}\n{'='*60}")
    result = run_pipeline(query)

    print("\n📋 AGENT TRACE:")
    for step in result["agent_trace"]:
        print(f"  [{step['agent']}] {step['duration_ms']}ms — {step['output_summary'][:80]}")

    print(f"\n⚠️  RISKS:")
    for r in result["risks"]:
        print(f"  [{r['severity']}] {r['risk']}")

    print(f"\n✅ FINAL ANSWER:\n{result['final_answer']}")

    print(f"\n📄 SOURCES:")
    for d in result["retrieved_docs"]:
        print(f"  • {d['source']} (p.{d['page']})")
