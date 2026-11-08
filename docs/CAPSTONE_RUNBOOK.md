# PFE-Copilot capstone - step-by-step runbook

Total effort: about 12 hours of your time, spread over 2-3 days (most of it is waiting, labelling and writing). Tick each box as you go. **Every phase ends with a "Done when" check - do not move on until it is true.**

Priority key: **[M]** must do (graded core) | **[S]** should do (strengthens the report) | **[C]** could do (skip if short of time).

---
## Phase 0 - Before you start (30 min)
- [ ] **[M]** Read the rest of the upGrad email (it was cut off in your screenshot) for deadlines or rules.
- [ ] **[M]** Because the API key appeared in a screenshot you shared, ask upGrad support (studentsupport@upgrad.com) for a replacement if you are uneasy; never paste the key into chat or notebook cells.
- [ ] **[M]** Download your group's CA certificate (`vmNN-ca.crt`) from the coordinator's link and keep it in a safe folder.
- [ ] **[M]** Confirm which group you are in (the certificate name `vm04/05/06/07` identifies your VM; the notebook derives your URL from it).
- [ ] **[M]** Unzip `pfe_copilot_submission.zip` somewhere; you only need `PFE_Copilot_Capstone.ipynb` to run.
- [ ] **[S]** Skim `README.md`, `docs/REPORT_AND_DEMO_OUTLINE.md` and `docs/COLAB_QUICKSTART.md` once.

**Done when:** you have the notebook, the certificate, and the key (in your email, not yet pasted anywhere).

---
## Phase 1 - Environment and connectivity (45 min)
1. **[M]** Open https://colab.research.google.com -> File -> Upload notebook -> `PFE_Copilot_Capstone.ipynb`. Keep the default **CPU** runtime (no GPU/TPU needed).
2. **[M]** Sidebar key icon (Secrets) -> add `LLM_API_KEY` with your key -> turn "Notebook access" on.
3. **[M]** Files panel -> upload the `.crt` -> right-click -> *Copy path*.
4. **[M]** Edit the **config cell** (Section 0): set `CA_CERT_PATH = "<copied path>"`, `USE_DRIVE = True` (keeps everything on Google Drive if Colab resets), leave `FINAL_RUN = False` for now. Leave the SEC contact string as is.
5. **[M]** Run: install cell -> config cell (it should print the base-URL port, "LLM mode: LIVE endpoint", and the Drive path) -> the six `%%writefile pfe_copilot.py` cells -> the import cell. The import cell finishes with the **smoke test**; it must print `endpoint says: ready`.
6. **[M]** Run **Section 1b pre-flight**. Fix every `FAIL` using the hint in the *detail* column:
   | FAIL | Fix |
   |---|---|
   | CA certificate file | wrong `CA_CERT_PATH`; re-copy path |
   | LLM /models (auth + TLS) | key not in Secrets, wrong group/cert, or the endpoint blocks Colab -> try Kaggle or your laptop |
   | Yahoo Finance | wait a few minutes and re-run; rate limits are common |
   | SEC EDGAR | check the User-Agent line; otherwise plan manual CSA downloads (Phase 3) |
   | BIS margin framework | the notebook downloads d499 (latest) automatically; only if that fails, save `d499.pdf` into `data/reg/` |

**Done when:** smoke test says "ready" and pre-flight shows OK for LLM, Yahoo and EDGAR (BIS can be manual).

---
## Phase 2 - Data download (30-60 min, mostly waiting)
1. **[M]** Run **Section 3** (download EDGAR CSAs + BCBS regulation, then the EDA table). With `FINAL_RUN = False` it fetches 60 documents.
2. **[M]** Open 3 files in `data/csa/` and confirm they really are credit support annexes / ISDA exhibits (title, "Paragraph 13", threshold/MTA wording).
3. **[M]** Check `data/reg/` contains the BCBS-IOSCO text (d499).
4. **[M]** Spot-check the haircut table in the code against the downloaded BCBS-IOSCO PDF (Appendix B): cash 0%, government 1-5y AAA-AA 2%, corporate >5y A+-BBB- 12%, main-index equities 15%. Fix `HAIRCUTS` in the library cell if any differ.
5. **[S]** Download BCBS d279 (SA-CCR) and check the constants in `SACCR_SF` / `SACCR_RHO` (equity index 20%/ρ 0.8, FX 4%, metals 18%, IR 0.5%, alpha 1.4, floor 5%).

**Done when:** the corpus table shows CSA documents (real EDGAR) and 1 regulation document, and you have verified the haircut numbers.

---
## Phase 3 - Retrieval experiments (45 min + 30 min labelling)
1. **[M]** Run **4.1** (chunk-size experiment) and **4.2** (final index, recall of bm25 / dense / hybrid / hybrid+rerank). First run downloads the MiniLM and reranker models. Copy both tables into your notes.
2. **[M]** Run the **4.3** labelling-sheet cell. It writes `outputs/retrieval_labeling.csv` (20 queries, about 5 passages each, shuffled, with optional `llm_hint`).
3. **[M]** Open the CSV in Excel/Google Sheets and fill `relevant`: **2** = passage directly answers the query, **1** = related/partial, **0** = not relevant. Judge from the text, not the hint. Save, upload back to the same path (or edit on Drive).
4. **[M]** Run the 4.3 evaluation cell -> `outputs/retrieval_eval_human.csv` (strict recall@1/3/5, MRR, nDCG@5 per mode). Note which mode wins.
5. **[S]** Run **4.9** (volatility backtest). It chooses EWMA vs flat volatility from evidence and saves `outputs/vol_model.txt`. If it prints a synthetic-data warning, Yahoo failed - fix and re-run.

**Done when:** you have the human-label table and a vol-model decision based on real data. Use the winning retrieval mode as the system default only if it clearly wins (the code uses hybrid+rerank by default).

---
## Phase 4 - Portfolio, tools and validation (45 min)
1. **[M]** Run **Section 5** (`make_portfolio`, tool demos, PFE profile chart). Check the data-source field says `yfinance`, **not** `FALLBACK_SYNTHETIC`.
2. **[S/C]** Run **5a** (swaps). Keep it if all five gates pass; otherwise set `INCLUDE_IRS = False` and move on (max 1 hour on this).
3. **[M]** Run **5g** (validation + SA-CCR). All validation checks must pass; the notebook stops otherwise. If a check fails, do not proceed - note which one and fix before any numbers are used.
4. **[S]** Read the SA-CCR table: large MC-vs-SA-CCR ratios are a discussion point, not an error.

**Done when:** validation table is all True and the portfolio uses live Yahoo data.

---
## Phase 5 - CSA extraction (45 min + 20 min labelling) **[S, high value]**
1. **[S]** Run **5b**. With the live LLM it extracts 8 fields from 10 CSAs with evidence quotes and a verification gate (VERIFIED / REVIEW / NOT_FOUND). Look at the status pivot table and the Excel in `outputs/deliverables/csa_terms_extracted.xlsx`.
2. **[S]** The LLM-vs-regex cell prints coverage and agreement; copy the numbers.
3. **[S]** Open `outputs/extraction_review.csv`; set `human_correct` (1 = value correct and unambiguous for the intended party, 0 = wrong/ambiguous; for NOT_FOUND rows 1 = term truly absent, 0 = missed). Re-run the `eval_extraction()` cell. Key metric: **VERIFIED precision** (should be near 100%).
4. **[C]** Optional: pick a document whose threshold/MTA are VERIFIED, set `APPLY = True` with `approved_by="<your name>"` to load real CSA terms into CP-002 (creates a backup and `outputs/csa_terms_audit.csv`). Do this **before** Phase 7.

**Done when:** you have the VERIFIED-precision figure and the Excel evidence sheet for a screenshot.

---
## Phase 6 - Usable outputs and dashboard charts (45 min)
1. **[M]** Run **5c** (daily pack, notices for each counterparty, dispute demo). Open in Office: `daily_ccr_pack_*.xlsx` (check there is **no red SYNTHETIC banner**) and one `margin_notice_*.docx`. Screenshot both.
2. **[M]** Run **5d** (future-risk fan charts, all scenarios incl. historical replays, reverse stress). Check the `data_notes` column: any "no data for window" or "synthetic" means that scenario's result must not be quoted.
3. **[S]** Run **5e** (AI-assisted scenarios) - copy the translation accuracy (with the live model) and the clamped-shock example.
4. **[S]** Run **5f** (effect of volatility model on PFE).

**Done when:** you have screenshots of the pack, a notice, the fan chart, a COVID replay result and the reverse-stress chart, all from live data.

---
## Phase 7 - Evaluation (60-90 min, mostly LLM time)
1. **[M]** Run **Section 6** (builds the 80+ query evaluation set; saved to `outputs/eval_set.json`). This is your frozen evaluation set - do not edit it after seeing results.
2. **[S]** Run **Section 7** (compare few-shot / chain-of-thought / decomposition planning). Copy the table. Set `PLANNER_STRATEGY` (environment variable, or the `strategy` argument) to the best performer if it is not few-shot, then re-run Section 7's winner only.
3. **[M]** Run **Section 8** (agent demo, multi-turn memory) and read the answers: do they look right and cite sources?
4. **[M]** Run **Section 9** (full evaluation). Record: strict and lenient accuracy, completion rate, tool calls, latency (p95), tokens, cost per query, accuracy by category, **failure taxonomy**.
5. **[M]** Open `outputs/eval_results.csv`; read 10 failures. Decide for each: real model failure, or the strict checker being too literal. Write 5-8 examples (query, answer, why it failed) for the "how the small model breaks" section.
6. **[S]** Run **9b** and rate the 20 sampled answers 1-5 on correctness, usefulness, grounding, clarity; re-run to get the mean scores.
7. **[M]** If accuracy is low in one category, fix the cause and re-run **that part only** (examples: many `retrieval_miss` -> revisit chunk size; many `tool_call_error` -> check counterparty names; many critic fallbacks -> inspect the composer outputs). Re-running is cheap: responses are cached in `outputs/llm_cache.jsonl`.

**Done when:** you have all metrics, the failure table, and 5-8 concrete failure examples.

---
## Phase 8 - Responsible AI tests (20 min)
1. **[M]** Run **Section 10**. Confirm: the injected "ignore all previous instructions" passage is flagged and not obeyed; PII is redacted; malformed model output is handled (`critic_fallback`/`planner_fallback` flags) and the answer is still correct; unknown counterparty gives an honest error; the refusal queries are all refused.
2. **[M]** Copy the outputs of each test into your notes (they are your Responsible-AI evidence).

**Done when:** all five tests behave as described.

---
## Phase 9 - Final graded run (1.5-2 hours) **[M]**
1. Set `FINAL_RUN = True` in the config cell (500 documents, live data only). Delete `data/csa` text files only if you want a clean re-download.
2. Run the notebook **top to bottom** (Runtime -> Run all), then redo the manual label steps if the retrieval sheet was rebuilt (the sheet is not overwritten if it exists, so your labels are safe).
3. Check the document count printed in Section 3. **Track B asks for 500-2000 documents.** If EDGAR gives fewer, add real CSA/ISDA exhibits manually to `data/csa/` (search sec.gov full-text for "credit support annex", form type EX-10), re-run Section 3, and if you still fall short report the real number plus the number of chunks, and tell the coordinator.
4. Make sure nothing fell back to synthetic: Section 13 raises an error if it did.
5. Section 13 builds `pfe_copilot_submission.zip` and `outputs/data_provenance.csv`. Review the provenance table: prices real, CSAs real EDGAR, the counterparty book labelled synthetic-by-design, swaps labelled industry test data.

**Done when:** Section 13 prints `secret scan: clean` and creates the zip.

---
## Phase 10 - Dashboard, container and traces (60 min)
1. **[M]** Run the Section 11 cells (they write `app.py`, `requirements.txt`, `Dockerfile`). Start the dashboard: locally `pip install -r requirements.txt && streamlit run app.py`, or in Colab `!streamlit run app.py &` then `!npx localtunnel --port 8501`.
2. **[M]** Click through each tab: Overview (try "Run engine validation"), Future risk (change MPOR and volatility), Crisis scenarios (COVID replay; use "Describe a scenario in words"), Reverse stress, Copilot.
3. **[M]** In the **Copilot** tab ask the demo prompts (this creates real traces): "Is Meridian within its PFE limit?", "And its margin call?", "Prepare the margin call notice for CP-002", "Meridian is calling for $X ... reconcile", "What happens in a COVID-style crash?", "Send the margin call to Meridian now" (must be refused).
4. **[M]** Open the **Observability** tab and screenshot it (latency, tool usage, flags, one inspected trace).
5. **[S]** Test the container if Docker is installed: `docker build -t pfe-copilot .` then `docker run -p 8501:8501 -e LLM_BASE_URL=... -e LLM_API_KEY=... -e LLM_MODEL=Qwen/Qwen3-8B pfe-copilot` (the CA certificate must also be available inside the container; otherwise state that the container was built but runs against the endpoint only on a machine with the certificate). If you do not test it, say so in the report.
6. **[C]** Host the dashboard on a free Hugging Face Space for the demo (add the key as a Space secret).

**Done when:** you have a working dashboard, real traces, and screenshots of every tab.

---
## Phase 11 - Report (4-5 hours) **[M]**
Use `docs/REPORT_AND_DEMO_OUTLINE.md` as the page budget. Fill each section from the notebook outputs:
| Report section | Take from |
|---|---|
| Problem, scope, refusals, related work | README + Section 2 |
| Evaluation design | Section 6, `eval_set.json` |
| Data and provenance | Section 3 EDA, `data_provenance.csv` |
| Prompt strategies and how the model breaks | Section 7 table + failure examples (Phase 7 step 5) |
| Retrieval experiments | 4.1, 4.2, `retrieval_eval_human.csv` (state the pooling caveat) |
| Agent design | architecture diagram, Section 1 |
| Outputs and dashboard | screenshots from Phases 5, 6 and 10 |
| Validation | Section 5g tables |
| Results | Section 9 metrics + taxonomy; extraction precision; answer review means |
| Observability | Observability screenshot |
| Responsible AI | Section 10 outputs |
| **How we compensated for a small model** (graded) | Section 12 + your own failure examples |
| Limitations | README section 8 (state them plainly) |
- [ ] Fill the **Results table in README.md** with your real numbers (replace every `_`).
- [ ] Be explicit in the report: synthetic portfolio, why; swaps are industry test structures; `VERIFIED` is not legal correctness; Hull-White parameters are assumptions; stressed-vol settings are assumptions.

---
## Phase 12 - Demo (2 hours incl. rehearsal) **[M]**
1. Follow the minute-by-minute plan in `docs/REPORT_AND_DEMO_OUTLINE.md`.
2. Warm the cache: run every demo prompt once beforehand.
3. Do not run the notebook and the demo at the same time on the same key (one active request per key -> 429).
4. Keep screenshots as a backup in case the endpoint or Yahoo is slow.
5. Rehearse once with a timer; cut anything that overruns 10 minutes.

---
## Phase 13 - Final QA and submission (30 min)
- [ ] The zip opens and has: notebook, `pfe_copilot.py`, `app.py`, `Dockerfile`, `requirements.txt`, `README.md`, `docs/`, `data/` (filled), `outputs/` (filled: `eval_set.json`, `eval_results.csv`, `retrieval_eval_human.csv`, `extraction_review.csv`, `answer_review.csv`, `data_provenance.csv`, `artifact_manifest.csv`, `traces.jsonl`, `deliverables/`).
- [ ] No `.crt`, no API key anywhere (Section 13 scans, but also search the notebook outputs).
- [ ] README results table filled; report PDF and demo recording/slides ready.
- [ ] Clear secrets from the notebook before sharing: Edit -> Clear all outputs, then re-run only if you need outputs, and re-check nothing sensitive is printed.
- [ ] Submit before the deadline from the email; keep a copy on Drive.

---
## If you run short of time (cut in this order)
1. Skip 5a swaps and the `APPLY` step (no loss to core grading).
2. Skip the Docker test and Hugging Face hosting (state it honestly).
3. Skip the answer review (9b) and extraction review labelling.
4. Reduce to `N_CSA_DOCS = 200` only if EDGAR is slow - and say so in the report.
**Never skip:** pre-flight, validation (5g), the human retrieval labels, the final live run, the evaluation with failure analysis, the Responsible-AI tests, the report's small-model section, and the demo rehearsal.

## Common problems
| Symptom | Likely cause | Fix |
|---|---|---|
| `TLS certificate error` | wrong/missing `.crt` path | set `CA_CERT_PATH`; never turn verification off |
| `HTTP 401` | wrong key or wrong group URL | re-check secret and certificate/VM |
| `HTTP 429` repeated | another run is using the same key | stop other runs; wait; the client retries automatically |
| Yahoo returns nothing | rate limit | wait 10 minutes, re-run (data is cached after success) |
| Section 13 raises "SYNTHETIC" | prices fell back | fix Yahoo access and re-run from Section 5 |
| Many `critic_fallback` flags | model drafts include figures not in tool output | read the drafts; keep the fallback (it is safe) and report the rate |
| Colab disconnects | runtime reset | `USE_DRIVE = True`, reopen, re-run the config and library cells |
| Validation check fails | code regression or bad data | do not continue; inspect which check; re-run with fresh data |
