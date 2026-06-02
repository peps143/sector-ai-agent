# AI Opportunity Assessment
## World Bank ITSEF — Sector Knowledge Operations

**Prepared by:** Perpetual T. Adu  
**Date:** June 2026  
**Purpose:** Use case prioritization, ROI framing, and adoption strategy for AI deployment in World Bank sector knowledge operations

---

## The Problem

World Bank task teams and sector specialists collectively produce thousands of documents every year — Implementation Completion Reports (ICRs), Implementation Status Reports (ISRs), Project Appraisal Documents (PADs), and sector notes. These documents contain hard-won operational knowledge: what worked, what failed, what risks emerged, and why.

The problem is that this knowledge is effectively invisible at the point of decision-making. A transport specialist designing a new rural roads project in West Africa has no practical way to quickly surface lessons from the 40 similar projects already completed. The knowledge exists — it just cannot be retrieved fast enough to be useful.

**The opportunity:** AI agents that can read, retrieve, and reason over this document corpus in real time, surfacing grounded insights in seconds rather than days.

---

## Use Case Prioritization Matrix

Five use cases were assessed across five dimensions: feasibility, sector fit, user demand, ROI clarity, and risk level. Each scored 1–10 and weighted equally.

| Use Case | Sector | Effort | Impact | Est. ROI | Priority |
|----------|--------|--------|--------|----------|----------|
| Sector Knowledge Agent | Cross-cutting | Low | Very High | 4.2× | **High** |
| Multi-Agent Workflow | Operations | Medium | Very High | 3.8× | **High** |
| Auto Risk Flagging | FCS/Operations | Medium | High | 3.1× | Medium |
| WASH Policy Advisor | WASH | High | Medium | 2.5× | Medium |
| M&E Report Generator | Education/Health | Very High | Medium | 2.1× | Low |

### Why Sector Knowledge Agent ranks first

Three factors converge: (1) the problem is acute and widely felt across TTLs, (2) the technical components are mature (RAG + vector search), and (3) the ROI is measurable — average manual document search takes 45 minutes; the agent reduces this to under 3 minutes. At 1,800 queries/month across a team, that is 630 hours of analyst time recovered monthly.

---

## ROI Framework

### Time savings model

| Metric | Baseline (manual) | With AI Agent | Saving |
|--------|-----------------|---------------|--------|
| Time per knowledge query | 45 min | 3 min | 42 min |
| Queries per analyst per month | 40 | 40 | — |
| Time saved per analyst/month | — | — | 28 hrs |
| Cost per analyst hour (est.) | $85 | $85 | — |
| Monthly saving per analyst | — | — | $2,380 |
| Annual saving (10 analysts) | — | — | **$285,600** |

### Quality uplift (harder to quantify but significant)

- Reduced risk of repeating known project failures
- Earlier surfacing of cross-sector lessons during project design
- More consistent use of evidence in sector policy documents

---

## Responsible AI Considerations

Any AI deployment in this context must address four non-negotiable principles:

**1. Human-in-the-loop by default**  
The agent should flag low-confidence responses and require human review before insights are acted upon. A confidence threshold (avg TRACe score < 75) triggers a visible warning: *"Low confidence — recommend human verification."*

**2. Source transparency**  
Every response must cite the specific documents it retrieved from. Users should be able to verify the source before trusting the insight. The Sector AI Agent displays source citations with document name and page number.

**3. Hallucination monitoring**  
A live observability dashboard tracks hallucination flag rates per session, per domain, and over time. Sustained hallucination rates above 5% trigger a review of the knowledge base and retrieval configuration.

**4. Scope limitation**  
The agent must be clear about what it does not know. If the knowledge base lacks relevant documents for a query, the agent says so explicitly rather than generating a plausible-sounding but unsourced answer.

---

## Adoption Strategy

### Phase 1 — Proof of concept (months 1–3)
- Deploy Sector Knowledge Agent with 3 pilot TTL teams across Transport, WASH, and Education
- Measure: query volume, satisfaction scores, time savings reported
- Success threshold: >80% satisfaction, >30 min average time saving per query

### Phase 2 — Pilot expansion (months 4–6)
- Expand to 20 TTLs across 5 sectors
- Add Multi-Agent Workflow for project preparation use cases
- Run structured adoption workshops: *"How to ask good questions of an AI agent"*

### Phase 3 — Institutionalization (months 7–12)
- Integrate with existing document management systems (OpenDocs, DMSF)
- Establish governance: AI Steering Group, quarterly model reviews
- Publish internal impact story: quantified time savings, user testimonials

### Adoption barriers and mitigations

| Barrier | Mitigation |
|---------|-----------|
| Trust in AI outputs | Source citations + hallucination flags + human-in-the-loop |
| Change fatigue | Embed in existing workflows, not as separate tool |
| Data quality concerns | Curated knowledge base with version control |
| Equity of access | Web-based, no local install required |

---

## Impact Story — Early Adopter Scenario

*"A TTL preparing a new irrigation project in the Sahel spent 3 minutes asking the Sector AI Agent: 'What climate data infrastructure failures have affected agriculture projects in West Africa?' The agent retrieved 4 relevant ICR passages, identified a recurring pattern of Year 0-1 climate data gaps, and flagged 2 high-severity risks specific to the Sahel context. The TTL used this directly in the project appraisal, avoiding a known failure mode that had affected 3 previous projects in the region. Estimated time saving: 4 hours of manual ICR review."*

---

## What's Next

1. Connect the Sector Knowledge Agent to World Bank's OpenDocs public repository (500,000+ documents)
2. Build sector-specific sub-agents (WASH, FCS, Climate) with domain-tuned prompts
3. Add multilingual support (French, Spanish, Arabic) for regional office teams
4. Develop a formal evaluation framework: pre/post adoption surveys, time-tracking integration

---

*This assessment was prepared as part of a portfolio project demonstrating AI product strategy competency for international development contexts.*
