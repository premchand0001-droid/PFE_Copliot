# Run the capstone in a local IDE (no GPU, no Colab)

The project is CPU-only: a numpy Monte-Carlo, small MiniLM embeddings, and calls to the course LLM endpoint. A normal laptop with **8 GB RAM (16 GB comfortable)**, about 5 GB free disk and a stable internet connection is enough. **Use a personal laptop and network if you can**: corporate proxies and TLS inspection often break calls to the course endpoint (an IP address with a private certificate) and to sec.gov / Yahoo.

## 1. Install (15 min)
Use **Python 3.10-3.12** (3.11 is the safe choice; very new Python versions may lack wheels for some libraries).

**Windows (PowerShell)**
```powershell
cd C:\path\to\pfe_copilot_submission
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu    # small CPU-only build (avoids a ~2 GB CUDA download)
pip install -r requirements.txt ipykernel jupyter
```
**macOS / Linux**
```bash
cd /path/to/pfe_copilot_submission
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu    # Linux; on macOS plain `pip install torch` is fine
pip install -r requirements.txt ipykernel jupyter
```

## 2. Configure secrets (5 min)
1. Copy `.env.example` to `.env` (same folder) and fill in:
   ```
   LLM_API_KEY=<your group key>
   CA_CERT_PATH=C:\certs\vm04-ca.crt        # or ./certs/vm04-ca.crt - the file name identifies your VM and your URL
   GROUP_ID=0                                # optional if the certificate name is recognised
   ```
2. `.env` is already in `.gitignore`. Never commit it, never paste the key into a notebook cell or chat.
3. Keep the `.crt` file outside the project folder, or make sure it is not zipped for submission (the packaging step also scans for key-like strings).

## 3. Open in your IDE
**VS Code (recommended):** install the *Python* and *Jupyter* extensions -> File -> Open Folder -> the project folder -> open `PFE_Copilot_Capstone.ipynb` -> top-right *Select Kernel* -> `.venv`. Run cells with Shift+Enter. The notebook's working folder is the notebook's own folder, so `data/` and `outputs/` appear next to it.
**PyCharm:** notebooks need PyCharm Professional; with the free Community edition run `jupyter lab` in the activated environment and open the notebook in the browser (or use the terminal commands below).
**JupyterLab:** `jupyter lab` from the project folder (virtual environment active).

## 4. Terminal commands for the long steps (optional but handy)
The same code as the notebook, runnable with logs in a terminal while you do something else:
```
python cli.py preflight            # certificate, LLM, Yahoo, SEC, BIS, FRED, Hugging Face - fix every FAIL
python cli.py download --docs 60   # SEC EDGAR CSAs + BCBS regulation (use --docs 500 for the final run)
python cli.py index                # chunk + embed + recall table (first run downloads two small models and embeds the corpus: several minutes on CPU)
python cli.py portfolio --irs      # illustrative portfolio (+ optional swap netting set)
python cli.py validate             # 24+ engine checks; exit code 1 if any fails
python cli.py pack                 # daily Excel pack
python cli.py dashboard            # starts the Streamlit dashboard at http://localhost:8501
```
Set `FINAL_RUN=1` in `.env` for the graded run (live data only; synthetic fallback disabled). The notebook is still where the evaluation, labelling and report evidence live; the CLI only saves time on downloads and checks.

## 5. What to expect on a laptop
| Step | Typical time |
|---|---|
| First `index` (models + embeddings, 500 documents) | 5-15 min, about 2-4 GB RAM |
| One counterparty Monte-Carlo (4,000 paths) | seconds; a 10-year swap set about 10-20 s |
| Dashboard stress scenario with PFE re-simulation | 30-60 s the first time, cached after |
| Full evaluation (80+ queries, live model) | 30-60 min (endpoint limits: one request at a time, 60/min) |
Everything is cached under `data/market/` and `outputs/` (including LLM responses in `outputs/llm_cache.jsonl`), so re-runs are fast. Keep the laptop awake and plugged in during long cells.

## 6. Docker (optional)
`docker build -t pfe-copilot .` then
`docker run -p 8501:8501 -v <folder with the .crt>:/certs -e LLM_API_KEY=... -e LLM_BASE_URL=https://103.42.50.41:<port>/v1 -e LLM_CA_CERT=/certs/vm04-ca.crt pfe-copilot`
(build the portfolio and download data first so `data/` is inside the image). If you cannot run Docker, say in the report that the image was built but not executed.

## 7. Local troubleshooting
| Symptom | Fix |
|---|---|
| `pip` fails building a package | use Python 3.11; upgrade pip; on Windows install Microsoft C++ Build Tools only if a package asks for it |
| `ssl` / certificate errors to sec.gov or Yahoo | you are behind a corporate proxy: switch to a personal network |
| TLS error to the course endpoint | wrong `CA_CERT_PATH`, or a proxy; verification must stay on |
| Notebook cannot find `pfe_copilot` | run the `%%writefile pfe_copilot.py` cells first (or use the `pfe_copilot.py` already in the folder) and make sure the kernel's working folder is the project folder |
| `Activate.ps1` blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Streamlit page blank | check the terminal for tracebacks; the portfolio must exist (`python cli.py portfolio`) |
