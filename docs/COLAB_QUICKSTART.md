# Run the capstone on free compute - step by step (about 3-4 hours the first time)

**You do not need a GPU or TPU.** The project is CPU-friendly (numpy Monte-Carlo, small MiniLM embeddings). Use Colab's default CPU runtime; a free T4 only speeds up embeddings slightly. Never paste your Google password or the API key into any chat or notebook cell.

## One-time setup (10 min)
1. Open https://colab.research.google.com -> File -> Upload notebook -> `PFE_Copilot_Capstone.ipynb`.
2. Left sidebar -> key icon (Secrets) -> add `LLM_API_KEY` = your group key; switch "Notebook access" on.
3. Download your group's CA certificate from the coordinator's link; Files panel (folder icon) -> upload it -> right-click -> Copy path.
4. In the config cell set: `GROUP_ID` (or just `CA_CERT_PATH` - the certificate name identifies your VM), `CA_CERT_PATH = "<copied path>"`, `USE_DRIVE = True` (keeps everything on your Drive if Colab resets).

## Run order
5. Run the config cell, then the install cell, then Section 1 (library) and the smoke test (should print "ready").
6. Run **Section 1b pre-flight** and fix every FAIL (certificate path, key, Yahoo/SEC/BIS reachability). If the course endpoint refuses connections from Colab, run the same notebook on your own laptop instead.
7. Dry run first: `FINAL_RUN = False` (60 CSA documents). Read the outputs; fix anything odd.
8. Manual steps (about 1 hour): Section 4.3 label `outputs/retrieval_labeling.csv` (0/1/2); Section 5b label `outputs/extraction_review.csv`; Section 9b rate `outputs/answer_review.csv`. Download each CSV from the Files panel (or open from Drive), label in Excel/Sheets, upload back to the same path, re-run the cell.
9. Final run: set `FINAL_RUN = True` (500 documents, live data only), run everything top to bottom, then Section 13 builds `pfe_copilot_submission.zip`.
10. Dashboard demo: `!streamlit run app.py &` then `!npx localtunnel --port 8501` (or run locally with `pip install -r requirements.txt && streamlit run app.py`).

## Free platforms (details change - check before relying on them)
| Platform | Use it for | Notes |
|---|---|---|
| **Google Colab (free)** | Main run | CPU by default; GPU/TPU optional and not guaranteed; sessions up to about 12 h; secrets and Drive built in |
| **Kaggle Notebooks** | Backup if Colab is unavailable | Published session limits of 12 h (CPU/GPU) and 9 h (TPU); internet access must be switched on in notebook settings; secrets supported |
| **Lightning AI Studio** | Persistent IDE alternative | Free plan has one Studio that restarts every 4 hours and a small monthly credit allowance; reported numbers vary |
| **Hugging Face Spaces (free CPU)** | Host the Streamlit dashboard for the demo | Add the endpoint key as a Space secret; no GPU needed |
| **Your laptop** | Fallback | Python 3.10+, `pip install -r requirements.txt` |

Your first live run is where EDGAR, Yahoo Finance, FRED, BIS, the Hugging Face models and the course endpoint are first reached - the pre-flight cell tells you exactly what to fix.
