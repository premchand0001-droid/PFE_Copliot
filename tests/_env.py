"""Shared test setup. Import this FIRST in every test module:  from _env import pfe

* Runs fully offline: no LLM key, no Yahoo, no EDGAR, no embedding endpoint.
* Uses a temporary data/output folder, so your real data/ and outputs/ are never touched.
* Market prices are deterministic synthetic series (so numbers are repeatable on any machine).
* Live-endpoint credentials (if you export RUN_LIVE_TESTS=1) are set aside here and only used by test_08.
"""
import os, sys, tempfile, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# keep credentials away from the module while it imports (it would go "live" otherwise)
LIVE_CREDS = {k: os.environ.pop(k, "") for k in
              ("LLM_API_KEY", "LLM_BASE_URL", "LLM_CA_CERT", "LLM_MODEL", "EMBED_API_KEY", "EMBED_CA_CERT", "EMBED_BASE_URL")}
RUN_LIVE = os.getenv("RUN_LIVE_TESTS") == "1" and bool(LIVE_CREDS["LLM_API_KEY"] and LIVE_CREDS["LLM_BASE_URL"])

TMP = pathlib.Path(tempfile.mkdtemp(prefix="pfe_tests_"))
os.environ.update({"PFE_DATA_DIR": str(TMP / "data"), "PFE_OUT_DIR": str(TMP / "outputs"),
                   "ALLOW_SYNTHETIC": "1", "N_PATHS": "600", "LLM_CACHE": "0"})
for k in ("PFE_LLM_OFF", "PFE_KILL_SWITCH"):
    os.environ.pop(k, None)

import pfe_copilot as pfe  # noqa: E402

TICKERS = ["SPY", "QQQ", "EEM", "GLD", "TLT", "LQD", "SHY", "SHV", "EURUSD=X", "USDJPY=X", "USDINR=X"]
for _t in TICKERS:                                    # deterministic prices; stops yfinance from being called
    pfe._PX[_t] = (pfe._synthetic_series(_t), "TEST_SYNTHETIC")
    pfe._LONG[_t] = (pfe._synthetic_series(_t), "TEST_SYNTHETIC")

_built = False
def ensure_portfolio():
    """Build the 6-counterparty book once per test run."""
    global _built
    if not _built:
        pfe.make_portfolio(force=True)
        _built = True


class NoRateLimit:
    def allow(self, key):
        return True


class FakeLLM:
    """Scriptable stand-in for pfe.llm (a 'live' small model that we control)."""
    live = True
    def __init__(self, replies):
        self.replies, self.calls, self.tokens, self.n_calls = list(replies), [], 0, 0
        self.cache_hits, self.cached_tokens, self.cached_secs, self.cache_unknown = 0, 0, 0.0, 0     # cache accounting the agent now reads
    def chat(self, messages, max_tokens=600, temperature=0.0, json_mode=False):
        self.calls.append(messages); self.n_calls += 1
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r
