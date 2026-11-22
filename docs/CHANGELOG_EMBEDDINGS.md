# Change: hosted embedding model (bge-small-en-v1.5 on VM02)

**What changed**
- `pfe_copilot.py`: new `RemoteEmbedder` (POST /v1/embeddings, 16 texts per call, retries on 429/5xx, shrinks over-long chunks, unit vectors, query-vector disk cache), `LocalEmbedder` (MiniLM), `make_embedder()`; `HybridRetriever(chunks, embed, rerank, backend=None|"remote"|"local")`.
- bge convention: documents are embedded as-is, queries get the prefix "Represent this sentence for searching relevant passages: ".
- Vector cache is keyed by model + chunk text, so switching models never reuses wrong vectors. Cached vectors are used with no network call.
- Failure ladder: cached vectors -> hosted endpoint -> local MiniLM (if indexing fails) -> BM25 only; a failed query embedding degrades that one query to BM25 instead of crashing the agent.
- Notebook: config cell (`EMBED_BACKEND`, `EMBED_API_KEY` Colab secret, `vm02-ca.crt`), bootstrap uploads two different certificates, new **Section 4.2b** (hosted bge vs local MiniLM -> `outputs/embedding_comparison.csv`), pre-flight check, secret scan also looks for the exact key values.
- `app.py` sidebar and `cli.py index` show the active embedding model. `.env.example`, Dockerfile comment, guides, governance card, architecture diagram updated.

**Settings** `EMBED_API_KEY` (secret), `EMBED_CA_CERT` (path to vm02-ca.crt), `EMBED_BASE_URL` (default https://103.42.50.41:1035/v1), `EMBED_MODEL`, `EMBED_BACKEND` = auto | remote | local.

**Tested** against a mock of the endpoint only (batching, ordering, 503 retry, over-long text, query cache, cache reuse, outage handling). NOT yet tested against the real VM02 server or with real CSAs.

**After switching models, re-do** the 4.3 human retrieval labels (the pooled candidates change) and the recall tables.

---
# Change 2: corpus expansion (Track B asks for 500-2,000 documents) and the 100-query cap

- New tiers besides CSAs: `data/contracts` (ISDA schedules, collateral / margin / netting agreements from EDGAR exhibits; a hit with real CSA terms goes to `data/csa`), `data/disclosures` (collateral and rating-trigger passages extracted from Form 10-Ks, not whole filings), and extra regulation PDFs in `data/reg` (SA-CCR d279, BCBS-IOSCO 2015, US prudential and CFTC margin rules - URLs unverified from the build sandbox; a miss is reported, never fatal).
- Searches run year by year (past EDGAR's 1,000-hit cap), per-company caps hold across all tiers, rejected documents are remembered (`data/rejected_stems.txt`) so re-runs are fast.
- Long contracts keep only their most collateral-relevant chunks (`CHUNK_CAP`: contract 60, disclosure 40, csa/regulation 400) so the hosted-embedding step is a few thousand calls once, then cached.
- `pfe.corpus_report()` prints downloaded vs indexed counts per tier and says plainly if the total is below 500. Provenance report lists the new tiers.
- Config cell: `EXPAND_CORPUS`, `N_CONTRACT_DOCS` (40 / final 300), `N_DISCLOSURE_DOCS` (20 / final 150). CLI: `python cli.py download --docs 60 --contracts 300 --disclosures 150`.
- Evaluation set: `trim_eval` drops queries from the largest unprotected category until there are at most 100 (hard paraphrased, refusal, clause and multi-turn are protected); an assertion enforces 60-100.

**Tested** with a mocked EDGAR (tier routing, caps, idempotent re-runs, excerpt extraction, chunk caps, report). NOT tested against the real SEC full-text search: the `forms`, `startdt`, `enddt` parameters and the yield (how many documents survive the filters) are unknown until you run it. If the total is still below 500, raise `N_CONTRACT_DOCS` / `N_DISCLOSURE_DOCS`, relax `per_company`, or add documents by hand into the folders (PDF/HTML are converted by `ingest_local_files`, but drop manual files into data/csa, data/reg only - or copy the text + json into data/contracts).
