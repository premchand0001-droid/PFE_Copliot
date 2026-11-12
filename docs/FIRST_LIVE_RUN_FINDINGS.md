# First live Colab run - findings and what was fixed

Source: `PFE_Copilot_Capstone_collab_ran.ipynb` (all cells executed, no errors; live Qwen3-8B endpoint, real Yahoo Finance and SEC EDGAR data). Numbers below are from that run **before** the fixes listed in section 3 - re-run to get the post-fix figures for the report.

## 1. What worked (real data, real endpoint)
- Endpoint smoke test, pre-flight, Yahoo Finance (SPY live), SEC EDGAR download (60 documents) and the BCBS-IOSCO PDF (d499 copy) all worked.
- Planning is easy for the model: route accuracy on 36 sampled queries - chain-of-thought 97.2%, few-shot 94.4%, decomposition 94.4%; JSON valid first try 100%; no fallbacks.
- Volatility backtest on real Yahoo history: EWMA + mean reversion beat flat 2-year volatility on 8 of 8 tickers (mean RMSE improvement +12.2%) - so the notebook selected `ewma` on evidence.
- End-to-end evaluation (86 queries): strict accuracy 94.2%, completion 100%, average 0.91 tool calls, average latency 2.24 s (p95 10.2 s), about 551 tokens per query. Every category scored 100% except CSA clause questions (50%).
- Engine validation passed; swap gates 4 of 5 passed (see fix 2).

## 2. What went wrong (all five evaluation failures were CSA clause questions)
| Finding | Evidence |
|---|---|
| Corpus quality | only 25 of 60 downloaded "CSA" exhibits contain genuine CSA terms; many are servicing/pooling agreements, and several are near-copies from the same company |
| Retrieval weak | recall on regex labels only 0.4-0.6 (hybrid+rerank best: 0.5 / 0.5 / 0.6 at k = 3 / 5 / 10); top results were duplicates from one company |
| Planner mislabels conceptual questions | "How much exposure can there be before collateral has to be posted?" was sent to a numeric tool with no counterparty -> tool error (2 cases) |
| Answer contradicts itself | "Meridian is within its PFE limit ... status BREACH" - numbers right, sentence wrong, and the critic passed it |
| Composer rambles about absent fields | "The status is not specified in the tool output ... No files were created" on a haircut answer |
| Extraction gate too strict | LLM VERIFIED share 36% vs regex 86%, mostly "quote not found verbatim" caused by curly quotes/ellipses. (The regex figure is **not** a quality score: its quotes are copied from the text, so it passes by construction; only human review shows which extractor is right) |
| Scenario translation 75% | model flipped the sign for "rates fall 50bp" and ignored "long bonds lose 10%" |
| Swap symmetry gate failed | ATM swap EPE/ENE ratio 0.81 on the real upward-sloping curve - symmetry only holds on a flat curve, so the test (not the model) was wrong |
| No margin calls to show | collateral was sized to today's exposure, so all six counterparties were NO_ACTION and no notice could be generated |

## 3. Fixes made in the current build
1. **Corpus:** `csa_score` (0-10) at download (weak documents rejected) and for indexing (`index_docs()`, default score >= 5, relaxed to 3 if fewer than 10 qualify); the notebook prints genuine vs indexed counts - report both.
2. **Swap gate:** the symmetry test now runs on a flat curve; the other gates are unchanged.
3. **Verification gate:** typography-insensitive containment (curly quotes, dashes, spacing) and ellipsis-aware quotes; an invented quote is still rejected (tested).
4. **Planner:** explicit rule and few-shot example that conceptual CSA/regulation questions go to the clause retriever; executor reroutes any numeric tool called without a counterparty to clause search (flag `rerouted_to_clause_search`).
5. **Retrieval:** results diversified (one chunk per document, at most two per company, no duplicate text).
6. **Composer and critic:** composer prompt now built per tool and forbids mentioning absent fields; critic rejects "within limit" when the status is BREACH (and the reverse) and rejects remarks about missing fields.
7. **Scenarios:** explicit figures in the text override the model's number when they conflict (sign flips and large mismatches), with the adjustment reported (tested on the two real failures).
8. **Margin demo:** collateral now reflects the last margin call settled 10 business days ago (`COLLATERAL_LAG_DAYS`), so real market moves since then create calls and returns; in my offline test three of six counterparties produced actions.
9. **Ported from your copy:** BIS d499 download with fallbacks and PDF validation, the clean-start cell, genuine-CSA-first selection.

## 4. What this says about "why AI"
The model is nearly redundant for routing templated questions (rules score about the same). The hard, genuinely language-dependent parts are **reading contracts** (extraction, clause retrieval) - where this run shows the real difficulty. Any redesign that strengthens the case for AI should put contract and filing understanding at the centre, with the deterministic engine as the quantifier.

## 5. Re-run after updating
Clean-start cell (`CLEAN_RUN = True`; set `CLEAN_CORPUS = True` once to re-download with the quality filter) -> Sections 3, 4, 5b (re-label extraction), 6-10. Then fill the README results table with the post-fix numbers.
