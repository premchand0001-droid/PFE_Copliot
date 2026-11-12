"""cli.py - run the long steps from a terminal / IDE (same code as the notebook).
Usage:  python cli.py preflight | download [--docs N] | index | portfolio [--irs] | validate | pack | dashboard
Settings come from environment variables or a .env file next to this script (LLM_API_KEY, GROUP_ID, CA_CERT_PATH, SEC_USER_AGENT, ...)."""
import argparse, os, subprocess, sys
from pathlib import Path

VM_PORT = {"vm07": 1031, "vm06": 1032, "vm05": 1033, "vm04": 1034}
GROUP_PORT = [(1, 15, 1031), (16, 30, 1032), (31, 45, 1033), (46, 65, 1034)]


def load_env():
    f = Path(".env")
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    os.environ.setdefault("SEC_USER_AGENT", "PFE-Capstone-Student premchand0001@gmail.com")
    cert = os.environ.get("CA_CERT_PATH", "")
    if cert:
        os.environ["LLM_CA_CERT"] = cert
        vm = os.path.basename(cert)[:4]
        if vm in VM_PORT:
            os.environ.setdefault("LLM_BASE_URL", f"https://103.42.50.41:{VM_PORT[vm]}/v1")
    g = int(os.environ.get("GROUP_ID", "0") or 0)
    for lo, hi, port in GROUP_PORT:
        if lo <= g <= hi:
            os.environ.setdefault("LLM_BASE_URL", f"https://103.42.50.41:{port}/v1")
    os.environ.setdefault("LLM_MODEL", "Qwen/Qwen3-8B")
    if os.environ.get("FINAL_RUN") == "1":
        os.environ["ALLOW_SYNTHETIC"] = "0"


def main():
    load_env()                                           # must run before pfe_copilot is imported (it reads the environment at import)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight", help="check certificate, LLM, Yahoo, SEC, BIS, FRED, Hugging Face")
    d = sub.add_parser("download", help="download SEC EDGAR CSAs and BCBS regulation"); d.add_argument("--docs", type=int, default=int(os.environ.get("N_CSA_DOCS", "60")))
    sub.add_parser("index", help="chunk the corpus, build embeddings/BM25 and print recall of each retrieval mode")
    p = sub.add_parser("portfolio", help="create the illustrative portfolio (data/portfolio)"); p.add_argument("--irs", action="store_true", help="also add the optional swap netting set")
    sub.add_parser("validate", help="run the engine validation suite (exit code 1 on failure)")
    sub.add_parser("pack", help="generate the daily exposure & margin pack")
    sub.add_parser("dashboard", help="start the Streamlit dashboard")
    a = ap.parse_args()
    if a.cmd == "dashboard":
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"]))
    import pandas as pd
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
    import pfe_copilot as pfe
    if a.cmd == "preflight":
        df = pfe.preflight_checks(); print(df.to_string(index=False)); sys.exit(1 if (df.status == "FAIL").any() else 0)
    if a.cmd == "download":
        pfe.ingest_local_files(); n = pfe.download_edgar_csas(n_docs=a.docs); pfe.download_regs(); pfe.ingest_local_files()
        docs = pfe.load_corpus(); print(f"{len(docs)} documents ({sum(d['category'] == 'csa' for d in docs)} CSA, {sum(d['category'] == 'regulation' for d in docs)} regulation)")
    elif a.cmd == "index":
        chunks = pfe.build_chunks(pfe.index_docs(), 900, 150); print(len(chunks), "chunks; building index (first run downloads models and embeds the corpus - several minutes on CPU)")
        r = pfe.HybridRetriever(chunks, embed=True, rerank=True)
        print(pd.DataFrame([{"mode": m, **{f"recall@{k}": pfe.recall_at_k(r, k, m) for k in (3, 5, 10)}} for m in ["bm25", "dense", "hybrid", "hybrid+rerank"]]).to_string(index=False))
    elif a.cmd == "portfolio":
        pfe.make_portfolio()
        if a.irs:
            pfe.make_irs_portfolio()
        print(pfe.cp_table()[["counterparty_id", "counterparty_name", "credit_limit_usd"]].to_string(index=False))
    elif a.cmd == "validate":
        v = pfe.validate_engine(); print(v.to_string(index=False)); ok = bool(v.passed.all())
        if (pfe.PORT_DIR / "swaps.csv").exists():
            w = pfe.validate_irs(); print(w.to_string(index=False)); ok = ok and bool(w.passed.all())
        print("ALL PASSED" if ok else "FAILURES"); sys.exit(0 if ok else 1)
    elif a.cmd == "pack":
        print(pfe.generate_daily_pack_tool())


if __name__ == "__main__":
    main()
