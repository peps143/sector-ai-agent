"""
Sector AI Agent — Multi-Agent Pipeline (LangGraph)
Standard queries: 5 agents
Comparison queries: 6 agents with branching

State keys use unique prefixed names to avoid LangGraph internal conflicts.
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


# ── State ─────────────────────────────────────────────────────────────────────
# ALL keys use unique prefixes (wbs_ = World Bank Sector) to avoid LangGraph conflicts
class AgentState(TypedDict):
    wbs_query:          str
    wbs_domain:         str
    wbs_query_type:     str           # "standard" or "compare"
    wbs_sector_one:     str           # first sector in comparison
    wbs_sector_two:     str           # second sector in comparison
    wbs_docs:           list[dict]    # standard retrieval docs
    wbs_docs_one:       list[dict]    # comparison: sector one docs
    wbs_docs_two:       list[dict]    # comparison: sector two docs
    wbs_insights:       str           # standard reasoning output
    wbs_insights_one:   str           # comparison: sector one insights
    wbs_insights_two:   str           # comparison: sector two insights
    wbs_cross:          str           # cross-sector comparison output
    wbs_risks:          list[dict]
    wbs_answer:         str
    wbs_trace:          list[dict]
    wbs_current:        str
    wbs_error:          str


# ── Domain detection ──────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "Transport":     ["road","transport","bridge","highway","procurement","maintenance","rural road"],
    "Agriculture":   ["agri","farm","crop","food","irrigation","extension","climate-smart","smallholder"],
    "WASH":          ["water","sanitation","wash","tariff","nrw","hygiene","utility"],
    "Education":     ["school","teacher","learning","education","literacy","curriculum","reading"],
    "FCS":           ["fragile","conflict","fcs","post-conflict","humanitarian","displacement","adaptive"],
    "Cross-Cutting": ["gender","climate","carbon","monitoring","capacity","sustainability"],
}

COMPARE_PAIRS = [
    ("Transport","WASH"),("Transport","Agriculture"),("Transport","Education"),
    ("WASH","Agriculture"),("WASH","Education"),("Agriculture","Education"),
    ("FCS","Transport"),("FCS","WASH"),
]


def detect_domain(query: str) -> str:
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def detect_compare(query: str) -> tuple[bool, str, str]:
    q = query.lower()
    is_cmp = any(w in q for w in ["compare","versus","vs","differ","differently","contrast","both sectors"])
    if not is_cmp:
        return False, "", ""
    for a, b in COMPARE_PAIRS:
        if a.lower() in q and b.lower() in q:
            return True, a, b
    return False, "", ""


def get_llm(temp: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=temp, api_key=os.getenv("OPENAI_API_KEY",""))


def make_trace(agent: str, inp: str, out: str, ms: int, status: str = "success") -> dict:
    return {"agent": agent, "timestamp": datetime.now().isoformat(),
            "input_summary": inp, "output_summary": out, "duration_ms": ms, "status": status}


# ── FAISS ─────────────────────────────────────────────────────────────────────
_store = None

def get_store():
    global _store
    if _store:
        return _store
    api_key = os.getenv("OPENAI_API_KEY","")
    config  = AgentConfig(openai_api_key=api_key)
    mgr     = VectorStoreManager(config)
    s       = mgr.load()
    if s is None:
        ing  = DocumentIngestor(config)
        docs = ing.load_documents() or _seed_sample_knowledge()
        chunks = ing.ingest(docs)
        s = mgr.build(chunks)
    _store = s
    return _store


def retrieve(query: str, sector: str = "", k: int = 5) -> list[dict]:
    store = get_store()
    q = f"{sector} sector: {query}" if sector else query
    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 20})
    docs = retriever.invoke(q)
    return [{"source": d.metadata.get("source","Unknown"),
             "sector": d.metadata.get("sector", sector or "General"),
             "page":   d.metadata.get("page","N/A"),
             "content": d.page_content,
             "snippet": d.page_content[:200] + "…"} for d in docs]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Router
# ══════════════════════════════════════════════════════════════════════════════
def router_agent(state: AgentState) -> AgentState:
    t0    = time.time()
    query = state["wbs_query"]
    llm   = get_llm()
    is_cmp, s1, s2 = detect_compare(query)
    domain     = detect_domain(query)
    query_type = "compare" if is_cmp else "standard"

    resp = llm.invoke([
        SystemMessage(content="You are a Router Agent for a World Bank ITSEF knowledge system. Classify the query and confirm routing. 2 sentences max."),
        HumanMessage(content=f"Query: {query}\nType: {query_type}\n{'Sectors: '+s1+' vs '+s2 if is_cmp else 'Domain: '+domain}\nConfirm.")
    ])
    ms = int((time.time()-t0)*1000)
    trace = make_trace("Router Agent", f"Query: {query[:80]}", f"Type: {query_type} | {'Compare: '+s1+' vs '+s2 if is_cmp else 'Domain: '+domain}", ms)
    return {**state, "wbs_domain": domain, "wbs_query_type": query_type,
            "wbs_sector_one": s1, "wbs_sector_two": s2,
            "wbs_current": "branch_a" if is_cmp else "retriever",
            "wbs_trace": state.get("wbs_trace",[]) + [trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Standard Retrieval
# ══════════════════════════════════════════════════════════════════════════════
def retrieval_agent(state: AgentState) -> AgentState:
    t0   = time.time()
    try:
        docs = retrieve(state["wbs_query"], state["wbs_domain"])
        ms   = int((time.time()-t0)*1000)
        trace = make_trace("Retrieval Agent", f"MMR search: '{state['wbs_query'][:60]}'",
                           f"Retrieved {len(docs)} chunks from {len(set(d['source'] for d in docs))} docs", ms)
        return {**state, "wbs_docs": docs, "wbs_current": "reasoner", "wbs_trace": state["wbs_trace"]+[trace]}
    except Exception as e:
        ms = int((time.time()-t0)*1000)
        trace = make_trace("Retrieval Agent", state["wbs_query"], f"Error: {e}", ms, "error")
        return {**state, "wbs_docs": [], "wbs_error": str(e), "wbs_current": "reasoner", "wbs_trace": state["wbs_trace"]+[trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3a — Sector One Specialist (comparison branch)
# ══════════════════════════════════════════════════════════════════════════════
def branch_a_agent(state: AgentState) -> AgentState:
    t0     = time.time()
    sector = state["wbs_sector_one"]
    llm    = get_llm()
    docs   = retrieve(state["wbs_query"], sector, k=3)
    docs_text = "\n\n".join(f"[{d['source']}]\n{d['content']}" for d in docs) or f"No {sector} docs found."

    resp = llm.invoke([
        SystemMessage(content=f"You are a specialist {sector} Sector Agent. Analyze how {sector} projects handled the query topic. 3-4 key points with evidence."),
        HumanMessage(content=f"Query: {state['wbs_query']}\n\nDocuments:\n{docs_text}")
    ])
    ms = int((time.time()-t0)*1000)
    trace = make_trace(f"{sector} Agent", f"Retrieving & analyzing {sector} docs",
                       f"Retrieved {len(docs)} docs | {resp.content[:100]}", ms)
    return {**state, "wbs_docs_one": docs, "wbs_insights_one": resp.content, "wbs_trace": state["wbs_trace"]+[trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3b — Sector Two Specialist (comparison branch)
# ══════════════════════════════════════════════════════════════════════════════
def branch_b_agent(state: AgentState) -> AgentState:
    t0     = time.time()
    sector = state["wbs_sector_two"]
    llm    = get_llm()
    docs   = retrieve(state["wbs_query"], sector, k=3)
    docs_text = "\n\n".join(f"[{d['source']}]\n{d['content']}" for d in docs) or f"No {sector} docs found."

    resp = llm.invoke([
        SystemMessage(content=f"You are a specialist {sector} Sector Agent. Analyze how {sector} projects handled the query topic. 3-4 key points with evidence."),
        HumanMessage(content=f"Query: {state['wbs_query']}\n\nDocuments:\n{docs_text}")
    ])
    ms = int((time.time()-t0)*1000)
    trace = make_trace(f"{sector} Agent", f"Retrieving & analyzing {sector} docs",
                       f"Retrieved {len(docs)} docs | {resp.content[:100]}", ms)
    return {**state, "wbs_docs_two": docs, "wbs_insights_two": resp.content, "wbs_trace": state["wbs_trace"]+[trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Standard Reasoning
# ══════════════════════════════════════════════════════════════════════════════
def reasoner_agent(state: AgentState) -> AgentState:
    t0  = time.time()
    llm = get_llm()
    docs_text = "\n\n".join(f"[{d['source']}]\n{d['content']}" for d in state["wbs_docs"]) or "No documents retrieved."

    resp = llm.invoke([
        SystemMessage(content="You are a Reasoning Agent for a World Bank ITSEF system. Extract 4-5 numbered operational insights with evidence."),
        HumanMessage(content=f"Query: {state['wbs_query']}\nDomain: {state['wbs_domain']}\n\nDocuments:\n{docs_text}")
    ])
    ms = int((time.time()-t0)*1000)
    trace = make_trace("Reasoning Agent", f"Analyzed {len(state['wbs_docs'])} docs",
                       f"{len(resp.content.split(chr(10)))} lines of analysis", ms)
    return {**state, "wbs_insights": resp.content, "wbs_current": "riskagent", "wbs_trace": state["wbs_trace"]+[trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Cross-Sector Comparator
# ══════════════════════════════════════════════════════════════════════════════
def comparator_agent(state: AgentState) -> AgentState:
    t0  = time.time()
    llm = get_llm()
    s1, s2 = state["wbs_sector_one"], state["wbs_sector_two"]

    resp = llm.invoke([
        SystemMessage(content=f"""You are a Cross-Sector Comparison Agent for World Bank ITSEF.
Compare {s1} and {s2} approaches. Identify:
1. Key differences in approach
2. What each sector did better
3. Transferable lessons between sectors
4. Common patterns in both
Be specific and evidence-based."""),
        HumanMessage(content=f"""Query: {state['wbs_query']}

{s1} Analysis:
{state['wbs_insights_one']}

{s2} Analysis:
{state['wbs_insights_two']}

Provide structured cross-sector comparison.""")
    ])
    ms    = int((time.time()-t0)*1000)
    trace = make_trace("Comparison Agent", f"Comparing {s1} vs {s2}",
                       f"Cross-sector synthesis: {resp.content[:100]}", ms)
    all_docs = state.get("wbs_docs_one",[]) + state.get("wbs_docs_two",[])
    return {**state, "wbs_cross": resp.content, "wbs_docs": all_docs,
            "wbs_insights": f"{s1}:\n{state['wbs_insights_one']}\n\n{s2}:\n{state['wbs_insights_two']}",
            "wbs_current": "riskagent", "wbs_trace": state["wbs_trace"]+[trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Risk Agent
# ══════════════════════════════════════════════════════════════════════════════
def risk_agent(state: AgentState) -> AgentState:
    t0  = time.time()
    llm = get_llm(0.1)
    ctx = state.get("wbs_cross") or state.get("wbs_insights","")

    resp = llm.invoke([
        SystemMessage(content='You are a Risk Agent. Return ONLY a JSON array: [{"risk":"name","severity":"High/Medium/Low","mitigation":"strategy"}]'),
        HumanMessage(content=f"Query: {state['wbs_query']}\nContext: {ctx[:600]}\n\nIdentify 3-4 key risks.")
    ])
    try:
        m = re.search(r'\[.*\]', resp.content.strip(), re.DOTALL)
        risks = json.loads(m.group()) if m else []
    except Exception:
        risks = [
            {"risk":"Implementation capacity gap","severity":"High","mitigation":"Early capacity assessment"},
            {"risk":"Coordination failure","severity":"Medium","mitigation":"Inter-agency working group from Day 1"},
            {"risk":"Sustainability post-completion","severity":"Medium","mitigation":"Budget integration from Year 1"},
        ]
    ms = int((time.time()-t0)*1000)
    trace = make_trace("Risk Agent", f"Assessed {state['wbs_domain']} context",
                       f"Identified {len(risks)} risks", ms)
    return {**state, "wbs_risks": risks, "wbs_current": "synthesizer", "wbs_trace": state["wbs_trace"]+[trace]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — Synthesis Agent
# ══════════════════════════════════════════════════════════════════════════════
def synthesis_agent(state: AgentState) -> AgentState:
    t0  = time.time()
    llm = get_llm()
    sources   = "\n".join(f"• {d['source']}" for d in state.get("wbs_docs",[]))
    risks_txt = "\n".join(f"[{r['severity']}] {r['risk']}: {r['mitigation']}" for r in state.get("wbs_risks",[]))

    if state["wbs_query_type"] == "compare":
        s1, s2 = state["wbs_sector_one"], state["wbs_sector_two"]
        content = f"""Query: {state['wbs_query']}

{s1} Analysis:
{state.get('wbs_insights_one','')}

{s2} Analysis:
{state.get('wbs_insights_two','')}

Cross-Sector Findings:
{state.get('wbs_cross','')}

Risks:
{risks_txt}

Sources: {sources}"""
        instruction = f"Write a comprehensive comparative brief contrasting {s1} and {s2} with specific transferable lessons."
    else:
        content = f"Query: {state['wbs_query']}\nDomain: {state['wbs_domain']}\n\nAnalysis:\n{state.get('wbs_insights','')}\n\nRisks:\n{risks_txt}\n\nSources: {sources}"
        instruction = "Write a comprehensive grounded response with insights and recommendations."

    resp = llm.invoke([
        SystemMessage(content=f"You are a Synthesis Agent for World Bank ITSEF. {instruction}"),
        HumanMessage(content=content)
    ])
    ms = int((time.time()-t0)*1000)
    trace = make_trace("Synthesis Agent", "Combining all agent outputs",
                       f"Final response: {len(resp.content)} chars", ms)
    return {**state, "wbs_answer": resp.content, "wbs_current": "complete", "wbs_trace": state["wbs_trace"]+[trace]}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def route_from_router(state: AgentState) -> Literal["retriever","branch_a"]:
    return "branch_a" if state["wbs_query_type"] == "compare" else "retriever"

def route_from_branch_b(state: AgentState) -> Literal["comparator","reasoner"]:
    return "comparator" if state["wbs_query_type"] == "compare" and state.get("wbs_insights_one") and state.get("wbs_insights_two") else "reasoner"


# ══════════════════════════════════════════════════════════════════════════════
# BUILD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline():
    g = StateGraph(AgentState)
    g.add_node("router",     router_agent)
    g.add_node("retriever",  retrieval_agent)
    g.add_node("branch_a",   branch_a_agent)
    g.add_node("branch_b",   branch_b_agent)
    g.add_node("reasoner",   reasoner_agent)
    g.add_node("comparator", comparator_agent)
    g.add_node("riskagent",  risk_agent)
    g.add_node("synthesizer",synthesis_agent)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_from_router, {"retriever":"retriever","branch_a":"branch_a"})
    g.add_edge("retriever", "reasoner")
    g.add_edge("branch_a",  "branch_b")
    g.add_conditional_edges("branch_b", route_from_branch_b, {"comparator":"comparator","reasoner":"reasoner"})
    g.add_edge("reasoner",   "riskagent")
    g.add_edge("comparator", "riskagent")
    g.add_edge("riskagent",  "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def run_pipeline(query: str) -> AgentState:
    pipeline = build_pipeline()
    initial: AgentState = {
        "wbs_query":        query,
        "wbs_domain":       "",
        "wbs_query_type":   "standard",
        "wbs_sector_one":   "",
        "wbs_sector_two":   "",
        "wbs_docs":         [],
        "wbs_docs_one":     [],
        "wbs_docs_two":     [],
        "wbs_insights":     "",
        "wbs_insights_one": "",
        "wbs_insights_two": "",
        "wbs_cross":        "",
        "wbs_risks":        [],
        "wbs_answer":       "",
        "wbs_trace":        [],
        "wbs_current":      "router",
        "wbs_error":        "",
    }
    return pipeline.invoke(initial)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv)>1 else \
        "Compare how Transport and WASH projects handled community engagement differently"
    print(f"\n🔍 Query: {query}\n{'='*60}")
    result = run_pipeline(query)
    print(f"\n📋 Type: {result['wbs_query_type']}")
    for step in result["wbs_trace"]:
        print(f"  [{step['agent']}] {step['duration_ms']}ms")
    print(f"\n✅ ANSWER:\n{result['wbs_answer'][:400]}...")
