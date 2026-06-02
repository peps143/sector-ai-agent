"""
Sector AI Agent — Multi-Agent Pipeline (LangGraph)
Real-world architecture with parallel execution for comparison queries.

Standard pipeline (5 agents):
  Router → Retrieval → Reasoning → Risk → Synthesis

Comparison pipeline (6 agents, parallel branches):
  Router → [Transport Agent || WASH Agent] → Comparison → Risk → Synthesis
"""

import os
import re
import json
import time
from typing import TypedDict, Literal
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from rag_agent import AgentConfig, VectorStoreManager, DocumentIngestor, _seed_sample_knowledge


# ── Shared state ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query:            str
    domain:           str
    query_type:       str        # "standard" or "comparison"
    sect_alpha:         str        # first sector for comparison
    sect_beta:         str        # second sector for comparison
    retrieved_docs:   list[dict] # standard retrieval
    docs_sect_alpha:    list[dict] # parallel: sector A docs
    docs_sect_beta:    list[dict] # parallel: sector B docs
    analysis:         str        # standard reasoning output
    output_alpha:       str        # parallel: sector A analysis
    output_beta:       str        # parallel: sector B analysis
    comparison:       str        # comparison agent output
    risks:            list[dict]
    final_answer:     str
    agent_trace:      list[dict]
    current_agent:    str
    error:            str


# ── Domain detection ──────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "Transport":     ["road","transport","bridge","highway","procurement","maintenance","rural road","contractor"],
    "Agriculture":   ["agri","farm","crop","food","irrigation","extension","climate-smart","smallholder"],
    "WASH":          ["water","sanitation","wash","tariff","nrw","hygiene","utility","non-revenue"],
    "Education":     ["school","teacher","learning","education","literacy","curriculum","reading"],
    "FCS":           ["fragile","conflict","fcs","post-conflict","humanitarian","displacement","adaptive"],
    "Cross-Cutting": ["gender","climate","carbon","monitoring","capacity","sustainability"],
}

COMPARE_PAIRS = [
    ("Transport", "WASH"),
    ("Transport", "Agriculture"),
    ("Transport", "Education"),
    ("WASH", "Agriculture"),
    ("WASH", "Education"),
    ("Agriculture", "Education"),
    ("FCS", "Transport"),
    ("FCS", "WASH"),
]


def detect_domain(query: str) -> str:
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def detect_comparison(query: str) -> tuple[bool, str, str]:
    """Detect if query is a comparison and extract the two sectors."""
    q = query.lower()
    is_compare = any(w in q for w in ["compare","comparison","versus","vs","differ","differently","contrast","both"])
    if not is_compare:
        return False, "", ""
    for a, b in COMPARE_PAIRS:
        if a.lower() in q and b.lower() in q:
            return True, a, b
    return False, "", ""


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )


def trace_entry(agent: str, input_summary: str, output_summary: str, duration_ms: int, status: str = "success") -> dict:
    return {
        "agent":          agent,
        "timestamp":      datetime.now().isoformat(),
        "input_summary":  input_summary,
        "output_summary": output_summary,
        "duration_ms":    duration_ms,
        "status":         status,
    }


# ── FAISS store ───────────────────────────────────────────────────────────────
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
        ingestor = DocumentIngestor(config)
        docs = ingestor.load_documents()
        if not docs:
            docs = _seed_sample_knowledge()
        chunks = ingestor.ingest(docs)
        store = vs_manager.build(chunks)
    _vectorstore = store
    return _vectorstore


def retrieve_for_sector(query: str, sector: str, k: int = 4) -> list[dict]:
    """Retrieve documents focused on a specific sector."""
    store = get_vectorstore()
    sector_query = f"{sector} sector: {query}"
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20},
    )
    docs = retriever.invoke(sector_query)
    return [{
        "source":  d.metadata.get("source", "Unknown"),
        "sector":  d.metadata.get("sector", sector),
        "page":    d.metadata.get("page", "N/A"),
        "content": d.page_content,
        "snippet": d.page_content[:200] + "…",
    } for d in docs]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Router
# Detects query type: standard or comparison
# ══════════════════════════════════════════════════════════════════════════════
def router_agent(state: AgentState) -> AgentState:
    t0 = time.time()
    query = state["query"]
    llm = get_llm()

    is_compare, sector_a, sector_b = detect_comparison(query)
    domain = detect_domain(query)
    query_type = "comparison" if is_compare else "standard"

    response = llm.invoke([
        SystemMessage(content="""You are a Router Agent for a World Bank ITSEF knowledge system.
        Classify the query and plan the retrieval strategy. Be concise — 2 sentences."""),
        HumanMessage(content=f"""Query: {query}
Query type: {query_type}
{"Sectors to compare: " + sector_a + " vs " + sector_b if is_compare else "Domain: " + domain}
Confirm routing decision.""")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Router Agent",
        f"Query: {query[:80]}",
        f"Type: {query_type} | {'Comparing ' + sector_a + ' vs ' + sector_b if is_compare else 'Domain: ' + domain} | {response.content[:80]}",
        duration,
    )
    return {
        **state,
        "domain":         domain,
        "query_type":     query_type,
        "sect_alpha":       sector_a,
        "sect_beta":       sector_b,
        "current_agent":  "branch",
        "agent_trace":    state.get("agent_trace", []) + [trace],
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2a — Transport/Sector A Retrieval & Analysis Agent (parallel branch)
# ══════════════════════════════════════════════════════════════════════════════
def sector_a_agent(state: AgentState) -> AgentState:
    t0 = time.time()
    llm = get_llm()
    sector = state["sect_alpha"]
    query  = state["query"]

    docs = retrieve_for_sector(query, sector, k=3)

    docs_text = "\n\n".join(
        f"[{d['source']}]\n{d['content']}" for d in docs
    ) if docs else f"No specific {sector} documents found."

    response = llm.invoke([
        SystemMessage(content=f"""You are a specialist {sector} Sector Agent for a World Bank knowledge system.
        Analyze retrieved {sector} documents and extract how this sector handled the topic in the query.
        Focus on: approaches used, what worked, what failed, specific evidence. Be specific and concise."""),
        HumanMessage(content=f"""Query topic: {query}
Sector: {sector}

Documents:
{docs_text}

Summarize how {sector} projects handled this topic in 3-4 key points with evidence.""")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        f"{sector} Agent",
        f"Retrieving & analyzing {sector} sector documents",
        f"Retrieved {len(docs)} docs | {response.content[:100]}",
        duration,
    )
    return {
        **state,
        "docs_sect_alpha": docs,
        "output_alpha":    response.content,
        "agent_trace":   state["agent_trace"] + [trace],
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2b — WASH/Sector B Retrieval & Analysis Agent (parallel branch)
# ══════════════════════════════════════════════════════════════════════════════
def sector_b_agent(state: AgentState) -> AgentState:
    t0 = time.time()
    llm = get_llm()
    sector = state["sect_beta"]
    query  = state["query"]

    docs = retrieve_for_sector(query, sector, k=3)

    docs_text = "\n\n".join(
        f"[{d['source']}]\n{d['content']}" for d in docs
    ) if docs else f"No specific {sector} documents found."

    response = llm.invoke([
        SystemMessage(content=f"""You are a specialist {sector} Sector Agent for a World Bank knowledge system.
        Analyze retrieved {sector} documents and extract how this sector handled the topic in the query.
        Focus on: approaches used, what worked, what failed, specific evidence. Be specific and concise."""),
        HumanMessage(content=f"""Query topic: {query}
Sector: {sector}

Documents:
{docs_text}

Summarize how {sector} projects handled this topic in 3-4 key points with evidence.""")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        f"{sector} Agent",
        f"Retrieving & analyzing {sector} sector documents",
        f"Retrieved {len(docs)} docs | {response.content[:100]}",
        duration,
    )
    return {
        **state,
        "docs_sect_beta": docs,
        "output_beta":    response.content,
        "agent_trace":   state["agent_trace"] + [trace],
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Standard Retrieval Agent (non-comparison queries)
# ══════════════════════════════════════════════════════════════════════════════
def retrieval_agent(state: AgentState) -> AgentState:
    t0 = time.time()
    query = state["query"]

    try:
        store = get_vectorstore()
        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )
        docs = retriever.invoke(query)
        retrieved = [{
            "source":  d.metadata.get("source", "Unknown"),
            "sector":  d.metadata.get("sector", state["domain"]),
            "page":    d.metadata.get("page", "N/A"),
            "content": d.page_content,
            "snippet": d.page_content[:200] + "…",
        } for d in docs]

        duration = int((time.time() - t0) * 1000)
        trace = trace_entry(
            "Retrieval Agent",
            f"MMR search: '{query[:60]}'",
            f"Retrieved {len(retrieved)} chunks from {len(set(d['source'] for d in retrieved))} documents",
            duration,
        )
        return {**state, "retrieved_docs": retrieved, "current_agent": "reasoning", "agent_trace": state["agent_trace"] + [trace]}
    except Exception as e:
        duration = int((time.time() - t0) * 1000)
        trace = trace_entry("Retrieval Agent", query, f"Error: {str(e)}", duration, "error")
        return {**state, "retrieved_docs": [], "error": str(e), "current_agent": "reasoning", "agent_trace": state["agent_trace"] + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Standard Reasoning Agent (non-comparison queries)
# ══════════════════════════════════════════════════════════════════════════════
def reasoning_agent(state: AgentState) -> AgentState:
    t0 = time.time()
    llm = get_llm()

    docs_text = "\n\n".join(
        f"[Source: {d['source']}]\n{d['content']}"
        for d in state["retrieved_docs"]
    ) if state["retrieved_docs"] else "No documents retrieved."

    response = llm.invoke([
        SystemMessage(content="""You are a Reasoning Agent for a World Bank ITSEF knowledge system.
        Analyze retrieved documents and extract structured operational insights.
        Provide 4-5 numbered key insights with evidence from the documents."""),
        HumanMessage(content=f"Query: {state['query']}\nDomain: {state['domain']}\n\nDocuments:\n{docs_text}")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Reasoning Agent",
        f"Analyzed {len(state['retrieved_docs'])} documents",
        f"{len(response.content.split(chr(10)))} lines of analysis generated",
        duration,
    )
    return {**state, "analysis": response.content, "current_agent": "risk", "agent_trace": state["agent_trace"] + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Comparison Agent (comparison queries only)
# Receives both sector analyses and synthesizes cross-sector insights
# ══════════════════════════════════════════════════════════════════════════════
def comparison_agent(state: AgentState) -> AgentState:
    t0 = time.time()
    llm = get_llm()
    sector_a = state["sect_alpha"]
    sector_b = state["sect_beta"]

    response = llm.invoke([
        SystemMessage(content=f"""You are a Comparison Agent for a World Bank ITSEF knowledge system.
        You receive independent analyses from two sector specialist agents and identify:
        1. Key differences in approach between the two sectors
        2. What each sector did better
        3. Transferable lessons — what {sector_a} can learn from {sector_b} and vice versa
        4. Common patterns that appear in both sectors
        Be specific and evidence-based. Structure clearly."""),
        HumanMessage(content=f"""Query: {state['query']}

{sector_a} Sector Analysis:
{state['output_alpha']}

{sector_b} Sector Analysis:
{state['output_beta']}

Provide a structured cross-sector comparison with transferable lessons.""")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Comparison Agent",
        f"Comparing {sector_a} vs {sector_b} analyses",
        f"Cross-sector synthesis: {response.content[:100]}",
        duration,
    )

    # Combine all docs for downstream agents
    all_docs = state.get("docs_sect_alpha", []) + state.get("docs_sect_beta", [])

    return {
        **state,
        "comparison":     response.content,
        "retrieved_docs": all_docs,
        "analysis":       f"{sector_a} Analysis:\n{state['output_alpha']}\n\n{sector_b} Analysis:\n{state['output_beta']}",
        "current_agent":  "risk",
        "agent_trace":    state["agent_trace"] + [trace],
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Risk Agent
# ══════════════════════════════════════════════════════════════════════════════
def risk_agent(state: AgentState) -> AgentState:
    t0 = time.time()
    llm = get_llm(temperature=0.1)

    context = state.get("comparison") or state.get("analysis", "")

    response = llm.invoke([
        SystemMessage(content="""You are a Risk Agent for a World Bank ITSEF knowledge system.
        Identify operational risks. Return ONLY a JSON array:
        [{"risk": "name", "severity": "High/Medium/Low", "mitigation": "brief strategy"}]
        No other text."""),
        HumanMessage(content=f"Query: {state['query']}\nDomain: {state['domain']}\nContext: {context[:600]}\n\nIdentify 3-4 key risks.")
    ])

    try:
        raw = response.content.strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        risks = json.loads(match.group()) if match else []
    except Exception:
        risks = [
            {"risk": "Implementation capacity gap", "severity": "High", "mitigation": "Early capacity assessment and phased rollout"},
            {"risk": "Stakeholder coordination failure", "severity": "Medium", "mitigation": "Inter-agency working group from Day 1"},
            {"risk": "Sustainability post-completion", "severity": "Medium", "mitigation": "Government budget integration from Year 1"},
        ]

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Risk Agent",
        f"Assessed {state['domain']} context",
        f"Identified {len(risks)} risks: {', '.join(r['risk'] for r in risks[:2])}…",
        duration,
    )
    return {**state, "risks": risks, "current_agent": "synthesis", "agent_trace": state["agent_trace"] + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — Synthesis Agent
# ══════════════════════════════════════════════════════════════════════════════
def synthesis_agent(state: AgentState) -> AgentState:
    t0 = time.time()
    llm = get_llm()

    sources = "\n".join(f"• {d['source']}" for d in state.get("retrieved_docs", []))
    risks_text = "\n".join(
        f"[{r['severity']}] {r['risk']}: {r['mitigation']}"
        for r in state.get("risks", [])
    )

    if state["query_type"] == "comparison":
        synthesis_input = f"""Query: {state['query']}

{state['sect_alpha']} Analysis:
{state.get('output_alpha', '')}

{state['sect_beta']} Analysis:
{state.get('output_beta', '')}

Cross-Sector Comparison:
{state.get('comparison', '')}

Risks:
{risks_text}

Sources: {sources}"""
        instruction = f"Write a comprehensive comparative brief contrasting {state['sect_alpha']} and {state['sect_beta']} approaches, with specific transferable lessons and recommendations."
    else:
        synthesis_input = f"""Query: {state['query']}
Domain: {state['domain']}

Analysis:
{state.get('analysis', '')}

Risks:
{risks_text}

Sources: {sources}"""
        instruction = "Write a comprehensive, grounded final response with specific insights and recommendations."

    response = llm.invoke([
        SystemMessage(content=f"""You are a Synthesis Agent for a World Bank ITSEF knowledge system.
        Combine all agent outputs into a clear, actionable response for development professionals.
        {instruction}"""),
        HumanMessage(content=synthesis_input)
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Synthesis Agent",
        "Combining all agent outputs",
        f"Final response: {len(response.content)} chars from {len(state.get('retrieved_docs', []))} sources",
        duration,
    )
    return {**state, "final_answer": response.content, "current_agent": "complete", "agent_trace": state["agent_trace"] + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def route_after_router(state: AgentState) -> Literal["retrieval", "sect_alpha", "sect_beta"]:
    """After router: branch to parallel agents for comparison, or standard retrieval."""
    if state["query_type"] == "comparison":
        return "sect_alpha"
    return "retrieval"


def route_after_branches(state: AgentState) -> Literal["comparison", "reasoning"]:
    """After both sector agents complete: go to comparison or standard reasoning."""
    if state["query_type"] == "comparison" and state.get("output_alpha") and state.get("output_beta"):
        return "comparison"
    return "reasoning"


# ══════════════════════════════════════════════════════════════════════════════
# BUILD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline():
    graph = StateGraph(AgentState)

    # Add all agent nodes
    graph.add_node("router",     router_agent)
    graph.add_node("retrieval",  retrieval_agent)
    graph.add_node("sect_alpha",   sector_a_agent)
    graph.add_node("sect_beta",   sector_b_agent)
    graph.add_node("reasoning",  reasoning_agent)
    graph.add_node("comparison", comparison_agent)
    graph.add_node("risk",       risk_agent)
    graph.add_node("synthesis",  synthesis_agent)

    # Entry point
    graph.set_entry_point("router")

    # Conditional routing after router
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieval": "retrieval",
            "sect_alpha":  "sect_alpha",
        }
    )

    # Standard path
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "risk")

    # Comparison path — sector_a runs first, then sector_b sequentially
    # (LangGraph free tier doesn't support true parallel; sequential is equivalent)
    graph.add_edge("sect_alpha", "sect_beta")
    graph.add_conditional_edges(
        "sect_beta",
        route_after_branches,
        {
            "comparison": "comparison",
            "reasoning":  "reasoning",
        }
    )
    graph.add_edge("comparison", "risk")

    # Final path
    graph.add_edge("risk",      "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


def run_pipeline(query: str) -> AgentState:
    pipeline = build_pipeline()
    initial: AgentState = {
        "query":          query,
        "domain":         "",
        "query_type":     "standard",
        "sect_alpha":       "",
        "sect_beta":       "",
        "retrieved_docs": [],
        "docs_sect_alpha":  [],
        "docs_sect_beta":  [],
        "analysis":       "",
        "output_alpha":     "",
        "output_beta":     "",
        "comparison":     "",
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
        "Compare how Transport and WASH projects handled community engagement differently"

    print(f"\n🔍 Query: {query}\n{'='*60}")
    result = run_pipeline(query)

    print(f"\n📋 Pipeline type: {result['query_type']}")
    print("\n📋 AGENT TRACE:")
    for step in result["agent_trace"]:
        print(f"  [{step['agent']}] {step['duration_ms']}ms — {step['output_summary'][:80]}")

    if result["query_type"] == "comparison":
        print(f"\n🔀 Sectors compared: {result['sect_alpha']} vs {result['sect_beta']}")

    print(f"\n⚠️  RISKS:")
    for r in result["risks"]:
        print(f"  [{r['severity']}] {r['risk']}")

    print(f"\n✅ FINAL ANSWER:\n{result['final_answer'][:500]}...")
