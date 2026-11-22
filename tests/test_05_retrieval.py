"""RAG layer: chunking, per-tier chunk caps, diversity, hosted-embedding client (mocked HTTP), hybrid retriever + graceful degradation.
Tests that need the `rank_bm25` package are skipped automatically if it is not installed (it is in requirements.txt / Colab)."""
import unittest, re, json, hashlib
from unittest import mock
import numpy as np
from _env import pfe

try:
    import rank_bm25  # noqa: F401
    HAVE_BM25 = True
except ImportError:
    HAVE_BM25 = False


class TestChunking(unittest.TestCase):
    def test_empty_and_short(self):
        self.assertEqual(pfe.chunk_text(""), [])
        self.assertEqual(pfe.chunk_text("   \n\n  "), [])
        self.assertEqual(pfe.chunk_text("One short paragraph."), ["One short paragraph."])

    def test_no_chunk_exceeds_size_and_none_is_empty(self):
        text = "\n\n".join(f"Paragraph {i}. " + "collateral threshold " * (i * 7 % 60) for i in range(80)) + "\n\n" + "x" * 5000
        for size in (300, 900):
            with self.subTest(size=size):
                ch = pfe.chunk_text(text, size=size, overlap=50)
                self.assertTrue(ch)
                self.assertTrue(all(0 < len(c) <= size for c in ch), max(len(c) for c in ch))

    def test_no_text_is_lost(self):
        words = [f"w{i:04d}" for i in range(600)]
        text = "\n\n".join(" ".join(words[i:i + 12]) for i in range(0, 600, 12))
        joined = " ".join(pfe.chunk_text(text, size=400, overlap=80))
        self.assertTrue(all(w in joined for w in words))

    def test_overlong_single_paragraph_is_split_with_overlap(self):
        ch = pfe.chunk_text("a" * 2500, size=900, overlap=150)
        self.assertGreaterEqual(len(ch), 3)
        self.assertTrue(all(len(c) <= 900 for c in ch))

    def test_smaller_size_gives_more_chunks(self):                          # the chunking experiment (Track B requirement)
        text = "\n\n".join("Eligible Collateral means cash. " * 8 for _ in range(60))
        self.assertGreater(len(pfe.chunk_text(text, 300, 50)), len(pfe.chunk_text(text, 1200, 50)))


class TestBuildChunks(unittest.TestCase):
    def _doc(self, cat, n, hot_every=5):
        paras = [("Eligible collateral margin threshold netting downgrade " if i % hot_every == 0 else "Boilerplate filler text about nothing. ") * 12 for i in range(n)]
        return {"doc_id": f"{cat}1", "category": cat, "title": f"{cat} doc | x", "url": "u", "text": "\n\n".join(paras)}

    def test_caps_per_tier(self):
        for cat, cap in pfe.CHUNK_CAP.items():
            with self.subTest(cat=cat):
                out = pfe.build_chunks([self._doc(cat, 800)])
                self.assertLessEqual(len(out), cap)

    def test_capped_documents_keep_the_collateral_relevant_chunks(self):
        out = pfe.build_chunks([self._doc("disclosure", 400)])
        self.assertTrue(all("collateral" in c["text"].lower() for c in out))

    def test_chunk_ids_and_metadata(self):
        out = pfe.build_chunks([self._doc("csa", 10)])
        self.assertEqual([c["chunk_id"] for c in out], [f"csa1#{i}" for i in range(len(out))])
        self.assertTrue(all(c["category"] == "csa" and c["url"] == "u" for c in out))


class TestDiversify(unittest.TestCase):
    def _c(self, i, company, text=None):
        return {"doc_id": f"d{i}", "title": f"{company} | CSA", "text": text or f"unique text {i} " * 5, "chunk_id": f"d{i}#0"}

    def test_per_company_cap_and_k(self):
        res = [self._c(i, "BigBank") for i in range(6)] + [self._c(10 + i, f"Other{i}") for i in range(4)]
        out = pfe._diversify(res, 5, per_company=2)
        self.assertEqual(len(out), 5)
        self.assertLessEqual(sum(c["title"].startswith("BigBank") for c in out), 2)

    def test_duplicate_text_and_doc_removed(self):
        res = [self._c(1, "A", "same text"), self._c(2, "B", "same text"), {**self._c(3, "C"), "doc_id": "d1"}]
        self.assertEqual(len(pfe._diversify(res, 5)), 1)

    def test_never_returns_empty_when_input_present(self):
        res = [self._c(1, "A", "t")] * 3
        self.assertTrue(pfe._diversify(res, 2))


class _Resp:
    def __init__(self, code, data=None, text=""):
        self.status_code, self._d, self.text = code, data, text
    def json(self):
        return self._d


class FakeSession:
    """Pretends to be the VM02 /v1/embeddings server. `script` = list of status codes to return first; then 200s."""
    def __init__(self, script=(), dim=8, max_chars=None, shuffle=False, raise_exc=None):
        self.script, self.dim, self.max_chars, self.shuffle, self.raise_exc = list(script), dim, max_chars, shuffle, raise_exc
        self.sent = []
    def post(self, url, json=None, timeout=None, headers=None, verify=None):
        if self.raise_exc:
            raise self.raise_exc
        self.sent.append(json["input"])
        if self.script:
            code = self.script.pop(0)
            if code != 200:
                return _Resp(code, text=f"error {code}")
        if self.max_chars and any(len(t) > self.max_chars for t in json["input"]):
            return _Resp(400, text="too long")
        data = []
        for i, t in enumerate(json["input"]):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v = np.random.default_rng(h % (2 ** 32)).normal(size=self.dim) * 3.0          # NOT unit length on purpose
            data.append({"index": i, "embedding": v.tolist()})
        if self.shuffle:
            data = data[::-1]
        return _Resp(200, {"data": data})


class TestRemoteEmbedder(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(pfe, "EMBED_API_KEY", "test-key"), mock.patch.object(pfe, "EMBED_BASE_URL", "https://vm02.test/v1"),
                  mock.patch.object(pfe, "EMBED_MODEL", f"bge-test-{self.id()[-12:]}"), mock.patch.object(pfe.time, "sleep")):
            p.start(); self.addCleanup(p.stop)

    def _emb(self, **kw):
        e = pfe.RemoteEmbedder(); e.http = FakeSession(**kw); return e

    def test_batches_of_16_and_order_preserved_even_if_server_shuffles(self):
        e = self._emb(shuffle=True)
        texts = [f"text number {i}" for i in range(40)]
        v = e.encode(texts)
        self.assertEqual([len(b) for b in e.http.sent], [16, 16, 8])
        solo = self._emb().encode([texts[5]])[0]
        self.assertTrue(np.allclose(v[5], solo, atol=1e-5), "row 5 must still be the embedding of text 5")

    def test_vectors_are_unit_length(self):
        v = self._emb().encode(["a", "b", "c"])
        self.assertTrue(np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5))

    def test_blank_text_is_replaced_not_rejected(self):
        e = self._emb(); e.encode(["", "   "])
        self.assertEqual(e.http.sent[0], [".", "."])

    def test_retries_503_then_succeeds(self):
        e = self._emb(script=[503, 429])
        e.encode(["hello"]); self.assertEqual((e.retries, e.n_calls), (2, 1))

    def test_gives_up_after_persistent_503(self):
        e = self._emb(script=[503] * 10)
        with self.assertRaisesRegex(RuntimeError, "busy"):
            e.encode(["hello"])

    def test_overlong_text_is_truncated_then_halved(self):
        e = self._emb(max_chars=700)
        v = e.encode(["x" * 5000])
        lens = [len(b[0]) for b in e.http.sent]
        self.assertEqual(lens[0], 1600)                         # first try: truncated to MAX_CHARS
        self.assertEqual(lens, sorted(lens, reverse=True))      # then halved on each 400 ...
        self.assertLessEqual(lens[-1], 700)                     # ... until the server accepts
        self.assertEqual(v.shape[0], 1)

    def test_401_names_the_key_setting(self):
        with self.assertRaisesRegex(RuntimeError, "EMBED_API_KEY"):
            self._emb(script=[401]).encode(["x"])

    def test_tls_error_names_the_certificate_setting(self):
        with self.assertRaisesRegex(RuntimeError, "EMBED_CA_CERT"):
            self._emb(raise_exc=Exception("SSL certificate verify failed")).encode(["x"])

    def test_query_gets_bge_prefix_documents_do_not_and_queries_are_cached(self):
        e = self._emb()
        e.encode_docs(["a document"])
        self.assertEqual(e.http.sent[-1], ["a document"])
        q1 = e.encode_query("what is the threshold?")
        self.assertEqual(e.http.sent[-1], [pfe.BGE_QUERY_PREFIX + "what is the threshold?"])
        calls = len(e.http.sent)
        q2 = e.encode_query("what is the threshold?")
        self.assertEqual(len(e.http.sent), calls, "second identical query must not hit the network")
        self.assertEqual(e.cache_hits, 1); self.assertTrue(np.allclose(q1, q2, atol=1e-5))

    def test_make_embedder_remote_requires_key(self):
        with mock.patch.object(pfe, "EMBED_API_KEY", ""):
            with self.assertRaises(RuntimeError):
                pfe.make_embedder("remote")


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (so dense retrieval is meaningful without a network)."""
    kind, dim = "fake", 128
    def __init__(self, name="fake-bow", fail_queries=False):
        self.name, self.fail_queries, self.doc_calls = name, fail_queries, 0
    def _v(self, t):
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / max(np.linalg.norm(v), 1e-9)
    def encode_docs(self, texts):
        self.doc_calls += 1
        return np.stack([self._v(t) for t in texts])
    def encode_query(self, q):
        if self.fail_queries:
            raise RuntimeError("endpoint down")
        return self._v(q)


CORPUS = {
    "mta": "Minimum Transfer Amount means USD 250,000 for each party.",
    "threshold": "Threshold means zero for Party A and USD 5,000,000 for Party B.",
    "valuation": "Valuation Agent means Party A in all circumstances.",
    "eligible": "Eligible Collateral is cash in USD and US Treasury securities.",
    "dispute": "If a party disputes a Valuation Agent calculation it must notify the other party and the dispute resolution procedure applies.",
    "notification": "Notification Time means 1:00 pm New York time on a Local Business Day.",
    "interest": "Interest Rate for cash collateral is the overnight federal funds effective rate.",
    "substitution": "Substitution of collateral requires consent and a substitute Eligible Credit Support of equal value.",
}
CHUNKS = [{"chunk_id": f"{k}#0", "doc_id": k, "category": "csa", "title": f"Co {k} | CSA", "url": "u", "text": t} for k, t in CORPUS.items()]
QUERIES = {"minimum transfer amount": "mta", "valuation agent party": "valuation", "notification time new york": "notification",
           "interest rate cash collateral": "interest", "substitute collateral consent": "substitution", "dispute resolution procedure": "dispute"}


@unittest.skipUnless(HAVE_BM25, "rank_bm25 not installed")
class TestHybridRetriever(unittest.TestCase):
    def _ret(self, **kw):
        return pfe.HybridRetriever(CHUNKS, embed=kw.pop("embed", True), rerank=False, backend=kw.pop("backend", FakeEmbedder()))

    def test_every_mode_returns_k_ranked_results(self):
        r = self._ret()
        for mode in ("bm25", "dense", "hybrid", "hybrid+rerank"):          # no reranker loaded here -> hybrid+rerank behaves like hybrid
            with self.subTest(mode=mode):
                res = r.search("minimum transfer amount", k=3, mode=mode)
                self.assertEqual([x["rank"] for x in res], [1, 2, 3])

    def test_right_clause_comes_first_in_bm25_dense_and_hybrid(self):
        r = self._ret()
        for mode in ("bm25", "dense", "hybrid"):
            for q, exp in QUERIES.items():
                with self.subTest(mode=mode, q=q):
                    self.assertEqual(r.search(q, k=1, mode=mode)[0]["doc_id"], exp)

    def test_recall_at_k_function(self):
        class Fake:
            def __init__(self, good): self.good = good
            def search(self, q, k=5, mode=None):
                return [{"text": "minimum transfer amount threshold valuation agent eligible collateral valuation percentage independent amount dispute notification time substitut interest rate"}] if self.good else [{"text": "nothing"}]
        self.assertEqual(pfe.recall_at_k(Fake(True)), 1.0)
        self.assertEqual(pfe.recall_at_k(Fake(False)), 0.0)

    def test_query_embedding_outage_degrades_to_bm25_not_a_crash(self):
        r = self._ret(backend=FakeEmbedder(fail_queries=True))
        res = r.search("minimum transfer amount", k=2, mode="hybrid")
        self.assertEqual(res[0]["doc_id"], "mta"); self.assertEqual(r.embed_fallbacks, 1)
        r.search("valuation agent", k=2, mode="dense"); self.assertEqual(r.embed_fallbacks, 2)

    def test_no_embedder_means_bm25_only_and_still_works(self):
        r = self._ret(embed=False)
        self.assertEqual(r.embed_name, "none (BM25 only)")
        self.assertEqual(r.search("notification time", k=1, mode="hybrid")[0]["doc_id"], "notification")

    def test_indexing_failure_degrades_to_bm25(self):
        class Boom(FakeEmbedder):
            def encode_docs(self, texts): raise RuntimeError("down")
        r = self._ret(backend=Boom("boom-model"))
        self.assertEqual(r.embed_name, "none (BM25 only)")
        self.assertTrue(r.search("threshold", k=1))

    def test_vector_cache_is_reused_and_keyed_by_model(self):
        e1 = FakeEmbedder("cache-model-A"); self._ret(backend=e1)
        e2 = FakeEmbedder("cache-model-A"); self._ret(backend=e2)
        self.assertEqual((e1.doc_calls, e2.doc_calls), (1, 0), "same model + same text -> cached vectors, no re-embedding")
        e3 = FakeEmbedder("cache-model-B"); self._ret(backend=e3)
        self.assertEqual(e3.doc_calls, 1, "a different model must never reuse another model's vectors")

    def test_retriever_tool_returns_citations_the_critic_accepts(self):
        old = pfe.RETRIEVER
        try:
            pfe.set_retriever(self._ret())
            out = pfe.csa_clause_retriever_tool("minimum transfer amount", k=3)
        finally:
            pfe.set_retriever(old)
        self.assertEqual([c["cite"] for c in out["chunks"]], ["S1", "S2", "S3"])
        ans = "The Minimum Transfer Amount is USD 250,000 [S1]."
        self.assertEqual(pfe.critic(ans, [{"tool": "csa_clause_retriever", "ok": True, "output": out}])["issues"], [])


if __name__ == "__main__":
    unittest.main()
