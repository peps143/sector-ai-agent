"""
Sector AI Agent - RAG-based Domain Knowledge Agent
World Bank ITSEF-style operational insights retrieval system
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

# ── LangChain core ──────────────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.schema import Document


# ── Configuration ────────────────────────────────────────────────────────────
@dataclass
class AgentConfig:
    openai_api_key: str
    docs_dir: str = "./data/sample_docs"
    vectorstore_dir: str = "./data/vectorstore"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 800
    chunk_overlap: int = 150
    retriever_k: int = 5
    temperature: float = 0.2


# ── Domain-specific system prompt ────────────────────────────────────────────
SECTOR_AGENT_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are a World Bank ITSEF Sector Knowledge Agent — an expert AI assistant 
specializing in operational insights, lessons learned, and sector knowledge from 
World Bank projects, ICRs (Implementation Completion Reports), and sector documents.

Your role is to:
1. Surface actionable operational insights from the knowledge base
2. Identify patterns, risks, and success factors across projects  
3. Connect sector-specific lessons to the user's query
4. Cite specific documents or projects when possible
5. Flag gaps where the knowledge base may be insufficient

CONTEXT FROM KNOWLEDGE BASE:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUERY: {question}

Respond with structured, evidence-based insights. When referencing documents, 
note the source. If the knowledge base lacks sufficient information, say so clearly
and suggest what additional documents would be helpful.

RESPONSE:""",
)


# ── Document ingestion ───────────────────────────────────────────────────────
class DocumentIngestor:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )

    def load_documents(self) -> list[Document]:
        docs_path = Path(self.config.docs_dir)
        all_docs: list[Document] = []

        if not docs_path.exists():
            print(f"[WARN] Docs directory not found: {docs_path}")
            return []

        # PDF loader
        pdf_loader = DirectoryLoader(
            str(docs_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True,
        )
        # TXT / markdown loader
        txt_loader = DirectoryLoader(
            str(docs_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            silent_errors=True,
        )

        for loader in [pdf_loader, txt_loader]:
            try:
                docs = loader.load()
                all_docs.extend(docs)
                print(f"[INFO] Loaded {len(docs)} documents via {loader.__class__.__name__}")
            except Exception as e:
                print(f"[WARN] Loader error: {e}")

        return all_docs

    def ingest(self, docs: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        chunks = self.splitter.split_documents(docs)
        print(f"[INFO] Split into {len(chunks)} chunks")
        return chunks


# ── Vector store manager ─────────────────────────────────────────────────────
class VectorStoreManager:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.openai_api_key,
        )
        self.store_path = Path(config.vectorstore_dir)
        self.store_path.mkdir(parents=True, exist_ok=True)

    def _fingerprint(self, chunks: list[Document]) -> str:
        content = "".join(c.page_content for c in chunks[:20])
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def build(self, chunks: list[Document]) -> FAISS:
        print("[INFO] Building FAISS vectorstore…")
        store = FAISS.from_documents(chunks, self.embeddings)
        store.save_local(str(self.store_path))
        print(f"[INFO] Vectorstore saved → {self.store_path}")
        return store

    def load(self) -> Optional[FAISS]:
        index_file = self.store_path / "index.faiss"
        if index_file.exists():
            print("[INFO] Loading existing vectorstore…")
            return FAISS.load_local(
                str(self.store_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return None

    def add_texts(self, store: FAISS, texts: list[str], metadatas: list[dict]) -> FAISS:
        store.add_texts(texts, metadatas=metadatas)
        store.save_local(str(self.store_path))
        return store


# ── RAG Chain ────────────────────────────────────────────────────────────────
class SectorRAGChain:
    def __init__(self, config: AgentConfig, vectorstore: FAISS):
        self.config = config
        self.vectorstore = vectorstore
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            api_key=config.openai_api_key,
        )
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={"k": config.retriever_k, "fetch_k": 20},
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": SECTOR_AGENT_PROMPT},
            return_source_documents=True,
            verbose=False,
        )

    def query(self, question: str) -> dict:
        result = self.chain.invoke({"question": question})
        sources = []
        for doc in result.get("source_documents", []):
            meta = doc.metadata
            sources.append(
                {
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", "N/A"),
                    "snippet": doc.page_content[:200] + "…",
                }
            )
        return {
            "answer": result["answer"],
            "sources": sources,
            "query": question,
        }

    def reset_memory(self):
        self.memory.clear()


# ── Agent orchestrator ───────────────────────────────────────────────────────
class SectorAgent:
    """Top-level agent that wires ingestion, indexing, and retrieval."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.vs_manager = VectorStoreManager(config)
        self.ingestor = DocumentIngestor(config)
        self.chain: Optional[SectorRAGChain] = None

    def initialize(self, force_rebuild: bool = False) -> dict:
        """Load or build the vectorstore, then wire the RAG chain."""
        store = None if force_rebuild else self.vs_manager.load()

        if store is None:
            docs = self.ingestor.load_documents()
            if not docs:
                # Seed with sample knowledge if no docs found
                docs = _seed_sample_knowledge()
            chunks = self.ingestor.ingest(docs)
            store = self.vs_manager.build(chunks)

        self.chain = SectorRAGChain(self.config, store)
        return {"status": "ready", "vectorstore": str(self.vs_manager.store_path)}

    def query(self, question: str) -> dict:
        if not self.chain:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.chain.query(question)

    def add_document_text(self, text: str, metadata: dict) -> dict:
        """Dynamically add a document to the live vectorstore."""
        if not self.chain:
            raise RuntimeError("Agent not initialized.")
        chunks = self.ingestor.splitter.split_text(text)
        metadatas = [metadata] * len(chunks)
        self.chain.vectorstore = self.vs_manager.add_texts(
            self.chain.vectorstore, chunks, metadatas
        )
        # Rebuild retriever
        self.chain.retriever = self.chain.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.config.retriever_k, "fetch_k": 20},
        )
        return {"status": "added", "chunks": len(chunks)}

    def reset_conversation(self):
        if self.chain:
            self.chain.reset_memory()


# ── Sample knowledge seed ────────────────────────────────────────────────────
def _seed_sample_knowledge() -> list[Document]:
    """Returns built-in seed documents when no files are present."""
    samples = [
        Document(
            page_content="""ICR Review – Rural Roads Infrastructure Project (P123456)
Sector: Transport | Country: Sub-Saharan Africa | Rating: Moderately Satisfactory

KEY LESSONS LEARNED:
1. Community ownership is critical. Projects that established local road committees 
   achieved 40% better maintenance outcomes over 5 years post-completion.
2. Climate resilience design was underweighted. 3 of 7 road segments suffered 
   significant damage in the second rainy season, requiring emergency rehabilitation.
3. Procurement delays averaged 8.3 months due to single-source bidding in remote areas.
   Future projects should pre-qualify contractors 18 months before effectiveness.
4. Gender-disaggregated beneficiary data was not collected until Year 3, limiting 
   ability to demonstrate impact on women's market access.

OPERATIONAL INSIGHTS:
- Works supervision must be contracted separately from design to avoid conflicts of interest
- Environmental management plans were rarely enforced; dedicated environmental officer needed
- Beneficiary satisfaction surveys showed 78% satisfaction but qualitative data revealed 
  significant concerns about dust and road safety not captured in quantitative metrics""",
            metadata={"source": "ICR-P123456-Transport.txt", "sector": "Transport", "type": "ICR"},
        ),
        Document(
            page_content="""World Bank ITSEF Sector Note – Digital Agriculture & Extension Services
Thematic Area: Agriculture Technology | Region: South & East Asia

OPERATIONAL LESSONS FROM 12 PROJECTS (2018-2024):
1. Last-mile connectivity remains the binding constraint. Of 12 projects, 9 reported 
   that digital tools failed to reach the bottom 40% of farmers due to connectivity gaps.
2. Digital literacy programs need 6-12 months before platform deployment, not concurrent.
3. Public-private partnerships for agronomic data sharing often collapse at Year 2 
   when private partners seek to monetize data that farmers consider their own.
4. SMS-based systems outperformed app-based in low-literacy contexts (85th percentile 
   adoption vs 34th percentile).

RISK FLAGS FOR FUTURE OPERATIONS:
- Vendor lock-in risk: 4 projects now dependent on single proprietary platform
- Data sovereignty concerns unresolved in 7 of 12 project legal frameworks
- Sustainability: Only 2 of 12 projects had government-funded continuation plans

DESIGN RECOMMENDATIONS:
Use open-source platforms (DHIS2, ODK, CommCare), embed data governance in legal 
agreements from Day 1, and pilot SMS fallbacks regardless of connectivity assumptions.""",
            metadata={"source": "ITSEF-DigitalAg-SectorNote.txt", "sector": "Agriculture", "type": "Sector Note"},
        ),
        Document(
            page_content="""Project Appraisal Document Lessons – Urban Water & Sanitation (P198765)
Sector: WASH | Country: Middle East & North Africa | Status: Active

LESSONS INCORPORATED FROM PRIOR OPERATIONS:
1. Tariff reform sequencing: Prior project (P154321) failed because tariff increases 
   preceded service quality improvements. This project inverts the sequence — quality 
   first, then cost recovery over 36 months.
2. Non-Revenue Water (NRW): Baseline NRW of 52% is above regional average of 38%. 
   Root cause analysis identified illegal connections (60%) and aging pipes (40%).
   Targeting illegal connections first yields faster wins with community trust.
3. Women in governance: Mandatory 30% female representation on Water User Committees 
   increased complaint resolution rates by 55% in comparable projects in Morocco.
4. Utility reform fatigue: Staff interviews revealed 4 prior reform programs in 8 years 
   created resistance. Change management budget set at 8% of total project cost.

OPERATIONAL RISKS IDENTIFIED:
- Political economy of water pricing highly sensitive in election years
- Cross-ministerial coordination between MoF and MoW has historically caused 6-month delays
- Groundwater over-extraction creating long-term supply sustainability risks""",
            metadata={"source": "PAD-P198765-WASH.txt", "sector": "WASH", "type": "PAD"},
        ),
        Document(
            page_content="""Implementation Status Report – Education Sector Strengthening Project
Sector: Education | Country: West Africa | ISR Rating: Moderately Unsatisfactory

IMPLEMENTATION CHALLENGES (Mid-Term Review Findings):
1. Teacher deployment bottleneck: 2,300 trained teachers remain undeployed due to 
   Ministry of Finance hiring freeze. Project design did not include fiscal space analysis.
2. School construction: 47% of schools built in flood-prone areas despite climate 
   screening requirements — supervision consultant did not apply the checklist.
3. Learning outcomes: Reading scores improved by 0.3 SDs in Grades 1-3 but showed 
   no improvement in Grade 4-6, suggesting the pedagogical model needs adaptation 
   for older learners.

COURSE CORRECTIONS UNDERWAY:
- Restructuring $12M from construction to remedial learning programs
- Adding climate resilience retrofitting to 34 highest-risk schools
- Piloting mother-tongue instruction in 3 regions based on evidence from Kenya/Tanzania

SYSTEMIC LESSONS:
The project confirmed findings from 8 other education projects: capacity substitution 
(using project staff instead of government staff) delays institutionalization. 
Transition plans must be built in from Year 1, not Year 4.""",
            metadata={"source": "ISR-Education-WestAfrica.txt", "sector": "Education", "type": "ISR"},
        ),
        Document(
            page_content="""ITSEF Knowledge Synthesis – Fragile & Conflict-Affected States (FCS) Operations
Cross-Cutting Theme: FCS Project Design | Portfolio Review: 45 Projects

KEY PATTERNS FROM FCS PORTFOLIO:
1. Flexibility > Precision: Projects with adaptive design mechanisms (restructuring 
   triggers, contingency components) achieved 71% of PDO targets vs 43% for 
   traditionally designed projects.
2. Speed of disbursement correlates with trust: In post-conflict settings, first 
   disbursements arriving within 90 days of effectiveness increased community 
   participation in subsequent phases by 34%.
3. Remote monitoring has improved but gaps persist: Satellite imagery + community 
   scorecards + third-party monitoring = most reliable triangulation. Single-method 
   monitoring failed in 18 of 22 FCS projects reviewed.
4. The "NGO dependency trap": Heavy reliance on international NGOs for implementation 
   hollowed out government capacity in 12 projects. Hybrid models (government + NGO 
   with explicit capacity transfer plans) performed better.

CRITICAL SUCCESS FACTORS:
- Political economy analysis updated annually (not just at appraisal)  
- Country Director engagement personally tracked by Operations Manager
- Procurement thresholds raised (National Shopping up to $500K recommended)
- Do-No-Harm analysis updated at each ISR""",
            metadata={"source": "ITSEF-FCS-Synthesis.txt", "sector": "Cross-Cutting", "type": "Knowledge Synthesis"},
        ),
    ]
    print(f"[INFO] Seeded {len(samples)} sample knowledge documents")
    return samples


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    config = AgentConfig(openai_api_key=api_key)
    agent = SectorAgent(config)
    print(agent.initialize())

    print("\n🌍 Sector AI Agent ready. Type your query (or 'quit' to exit):\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        result = agent.query(q)
        print(f"\nAgent: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"][:3]:
                print(f"  📄 {Path(s['source']).name} (p.{s['page']})")
        print()
