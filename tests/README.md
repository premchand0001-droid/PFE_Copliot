# PFE-Copilot test suite

187 tests, ~4 seconds, **no internet / no API key / no GPU needed** (market prices are deterministic synthetic series; the model's replies are scripted).

## Run it (Colab or local), from the repo root (next to `pfe_copilot.py`)
```bash
pip install rank_bm25            # only needed for the retrieval tests (already in requirements.txt)
python -m unittest discover -s tests -v      # or: pip install pytest && pytest tests -v
```
Colab cell: `!cd /content/<your-repo> && python -m unittest discover -s tests 2>&1 | tail -30`
Tests use a temporary folder, so your real `data/` and `outputs/` are never touched.

## Live model tests (optional, uses your real key - not while the notebook/demo is running)
```bash
RUN_LIVE_TESTS=1 LLM_API_KEY=... LLM_BASE_URL=https://103.42.50.41:1034/v1 LLM_CA_CERT=/content/vm04-ca.crt \
  python -m unittest discover -s tests -p "test_08*.py" -v
# add EMBED_API_KEY=... EMBED_CA_CERT=/content/vm02-ca.crt to also test the hosted bge embeddings
```
Never paste the key into the notebook or chat; export it as an environment variable (or Colab secret).

## What each file proves (maps to the Track B rubric)
| File | Tests | Proves |
|---|---|---|
| `test_01_finance_engine` | 33 | Margin-call arithmetic (hand-computed), haircuts, MTA/rounding, CSA collateral model properties, PFE reproducibility, stress signs |
| `test_02_parsing_and_validation` | 27 | JSON extraction from messy model output, repair-retry loop, tool-call schema validation, plan validation, scenario parsing + clamping |
| `test_03_guardrails` | 19 | Responsible AI: injection, PII, scope, human-in-the-loop actions, legal refusal, poisoned documents, rate limit, kill switch |
| `test_04_critic` | 14 | Anti-hallucination critic: invented numbers, wrong status, missing citations, contradictions; fallback text always passes the critic |
| `test_05_retrieval` | 29 | Chunking, per-tier caps, diversity, hosted-embedding client (batching/retry/cache/prefix), hybrid retriever, graceful degradation to BM25 |
| `test_06_llm_client_and_small_model` | 32 | Endpoint client (429/503 retry, 401, TLS, clamp, cache, one request at a time) and the agent when the 8B model misbehaves |
| `test_07_agent_end_to_end` | 22 | 17 "golden" queries -> expected tool and arguments, multi-turn memory, files created, traces written |
| `test_08_live_endpoint` | 11 | Real Qwen3-8B: planner routing accuracy, 3 prompting strategies, grounded answers, refusals, injection resistance (skipped by default) |

## Findings the suite surfaced (status)
1. FIXED - `parse_amount("$1.5 million")` returned 1,000,000 (decimals dropped); the amount regexes now keep decimals.
2. FIXED - if the model endpoint failed *after* the tools ran, the user saw "Internal error"; `compose()` now returns the deterministic answer (flag `composer_unavailable`).
3. FIXED - with a live model the keyword scope filter no longer refuses before the planner can read the question (prefilter scope check only in rules-only mode); this is the over-refusal seen on "shave off BBB bonds" / "repeat of March 2020".
4. OPEN (kept as `expectedFailure`) - the rules-fallback planner reuses the last counterparty for a conceptual question; the LLM planner handles it. Delete that test's `@unittest.expectedFailure` line when fixed.
