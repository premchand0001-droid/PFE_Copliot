# Track B - 12-page report and 10-minute demo outline (PFE-Copilot)

Grading: technical depth & rigour 30% | punctuality & consistent progress 25% | working deployed system 20% | report & demo 25%.
The template says "What did we do to compensate for a small model?" is a **graded section** - give it a full page.

## Report (12 pages) - section -> where the evidence comes from
| # | Section | Pages | Evidence (notebook section / output file) |
|---|---|---|---|
| 1 | Problem, users, what the copilot does / refuses; **related work** (JPM COIN, ISDA Ark 51 / CDM, Morgan Stanley Debrief, vendor collateral tools: what they output and how this project follows the same artefact-first, human-approved pattern); team roles you covered (template assumes 6 people) | 1 | Section 2 spec; README |
| 2 | Evaluation design (written before agent evaluation): 100 queries (24 hard paraphrased), gold from deterministic references, categories, strict vs lenient judge | 1 | Section 6; `outputs/eval_set.json` |
| 3 | Data: real EDGAR CSAs (count!), BCBS-IOSCO, Yahoo Finance, illustrative portfolio and why | 1 | Section 3; corpus stats table |
| 4 | Prompting strategies (few-shot JSON vs CoT vs decomposition) and **how Qwen3-8B breaks** (JSON wrapping, think-mode, wrong args, rounding numbers) | 1.5 | Section 7 table + 5-8 real failure examples from `eval_results.csv` |
| 5 | RAG: chunking experiment, BM25/dense/hybrid/rerank, **human-labelled** recall@K/MRR/nDCG next to regex labels, pooling caveat | 2 | Section 4, `retrieval_eval_human.csv` |
| 6 | Agent design: planner -> executor -> critic, 9 tools, memory, guards (10 steps, token budget, loop guard, retries, JSON repair, endpoint limits) | 1.5 | Architecture diagram; Section 1 |
| 7 | Usable outputs + dashboard: CSA term extraction with verification gate, daily pack, notices, dispute waterfall, stress tests (screenshots of real files and dashboard tabs) | 1.5 | Sections 5b-5d; `outputs/deliverables/` |
| 6b | **Why AI?** Ablation rules-only vs with-LLM per category (Section 9a); where AI adds value (paraphrase, contracts, free-form scenarios) and where deterministic code is better; governance card (owner, tier, controls, kill switch) | (in 8) | Section 9a; `GOVERNANCE_CARD.md` |
| 7b | Validation: closed-form benchmark table (24 checks), SA-CCR cross-check and what the ratios mean, artifact manifest | (in 8) | Section 5g; `artifact_manifest.csv` |
| 8 | Results: accuracy (strict/lenient), completion, tool calls, latency, tokens/cost, **failure taxonomy**, extraction VERIFIED-gate precision | 1.5 | Section 9; `eval_results.csv`, `extraction_review.csv` |
| 9 | Observability: trace dashboard screenshots, latency p50/p95, tool usage, flags | 0.5 | App "Observability" tab; `traces.jsonl` |
| 10 | Responsible AI: injection test in corpus, PII, refusals, HITL boundary, audit trail | 1 | Section 10 |
| 11 | **What we did to compensate for a small model** | 1 | Section 12 |
| 12 | Limitations and future work (honest). Include the planned **user-selectable PFE models and multi-model comparison/summary** in the copilot | 0.5 | README sections 8 and 8b |

Must-state caveats: synthetic/illustrative portfolio; linear forwards plus optional vanilla swaps (Hull-White parameters are assumptions, swaps are industry test structures, rate factor independent of equity/FX); `VERIFIED` != legally correct; pooled-label bias; stressed-vol assumptions; Qwen3 non-thinking mode used.

## Demo (10 minutes) - run of show
| Time | Show | Why it scores |
|---|---|---|
| 0:00-0:45 | One slide: problem + principle "LLM plans/extracts, code computes/verifies, output is a reviewable file" | clarity |
| 0:45-2:00 | Dashboard Overview -> daily pack download (open the xlsx) | working system |
| 2:00-3:30 | Future risk: move MPOR 10 -> 20 and vol x1.5; point at limit line | depth |
| 3:30-4:45 | Crisis scenarios: COVID replay -> liquidity need, breach before/after; reverse stress | domain relevance |
| 4:45-6:30 | Copilot: "Prepare the margin call notice for CP-00X" (open docx); dispute reconciliation prompt; follow-up "its PFE?" (memory) | agentic + usable output |
| 6:30-7:30 | CSA extraction sheet: VERIFIED vs REVIEW with evidence quote | verification gate |
| 7:30-8:30 | Refusal ("send the margin call") + poisoned-document test result | responsible AI |
| 8:00-8:30 | Overview tab: run engine validation (24/24) and show the SA-CCR cross-check | rigour |
| 8:30-9:30 | Observability tab: trace, tool usage, latency; results table | observability |
| 9:30-10:00 | Lessons: what the 8B model got wrong and how you compensated | graded section |

**Demo safety:** warm the cache first (run the demo prompts once - responses are cached in `outputs/llm_cache.jsonl`), keep screenshots as backup, never run the notebook and the demo on the same group key at the same time (1 active request per key -> 429), and check no `SYNTHETIC` banner shows.
