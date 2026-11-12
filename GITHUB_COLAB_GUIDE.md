# Run the capstone in Google Colab via GitHub - step by step

You need: a free GitHub account, a Google account (for Colab), your upGrad key and certificate. No GPU is needed: use Colab's default CPU runtime. **Never paste the API key or a GitHub token into a notebook cell, the repo, or any chat.**

## A. Put the project on GitHub (20 min, once)
1. github.com -> **New repository** -> name it (for example `pfe-copilot-capstone`) -> **Private** (recommended) -> do **not** tick "Add a README" -> Create.
2. Unzip `pfe_copilot_submission.zip` on your computer.
3. Edit two placeholders before uploading (any text editor; `Ctrl+H` find/replace):
   - In `PFE_Copilot_Capstone.ipynb` replace `premchand0001-droid/PFE_Copliot` with, for example, `premchand0001-droid/PFE_Copliot` (appears in the `REPO_URL` line and in one explanatory markdown cell).
   - In `README.md` replace `premchand0001-droid/PFE_Copliot` the same way (the Open-in-Colab badge).
4. Upload. **Option 1 (no git installed):** on the new repo page -> *uploading an existing file* -> drag in **all** files and folders from the unzipped project (including `docs/`, `data/`, `outputs/`, `.gitignore`, `.env.example`) -> Commit. **Option 2 (git):**
   ```
   cd pfe_copilot_submission
   git init && git add . && git commit -m "initial project"
   git branch -M main
   git remote add origin https://github.com/<USER>/<REPO>.git
   git push -u origin main        # sign in via the browser prompt or a token
   ```
5. **Check the repo page:** you must **not** see `.env`, any `.crt`, or any key. (`.gitignore` already excludes them; the packaging step also scans for key-like text.)

## B. Colab secrets (5 min, once per Google account)
Open any Colab notebook -> left sidebar **key icon (Secrets)** -> add:
| Secret | Value | When |
|---|---|---|
| `LLM_API_KEY` | your group key | always (switch **Notebook access** on) |
| `GITHUB_TOKEN` | a *fine-grained* token | only if the repo is private. GitHub -> Settings -> Developer settings -> Personal access tokens -> Fine-grained -> repository access: **only this repo**; permission **Contents: Read** (add **Write** only if you will use the push-back cell in Section 14); short expiry (30 days) |

## C. Open and run
1. Open `https://colab.research.google.com/github/<USER>/<REPO>/blob/main/PFE_Copilot_Capstone.ipynb` (or click the badge in the README). For a private repo, Colab asks you to authorise GitHub access once; if that is awkward, download the notebook and use *File -> Upload notebook* - the first cell still clones the repo.
2. Runtime -> Change runtime type -> **CPU** (default).
3. Run the first code cell (**bootstrap**). It clones your repo into `/content/<repo>`, switches into it, never stores the token, and - if no certificate is found - shows an **Upload** button: choose your group's `vmNN-ca.crt`. Its file name identifies your VM and your base URL.
4. In the config cell leave `GROUP_ID = 0` (the certificate decides), set `USE_DRIVE = True` (keeps data, labels and caches on your Google Drive), keep `FINAL_RUN = False` for the first pass.
5. Run on in order: config -> install -> the library cells -> import + smoke test (`endpoint says: ready`) -> **Section 1b pre-flight** (fix every FAIL; see the table below).
6. Long downloads without freezing the notebook: in a new cell run `!python cli.py download --docs 60` (or 500 for the final run); `!python cli.py preflight` and `!python cli.py validate` also work.
7. Continue with the **runbook phases** (`docs/CAPSTONE_RUNBOOK.md`): retrieval labelling, extraction, evaluation, Responsible-AI tests, final run with `FINAL_RUN = True`, Section 13 (builds the zip and downloads it to your computer).

## D. Keep your work (Colab runtimes are temporary)
- `USE_DRIVE = True` keeps `data/` and `outputs/` (including your label sheets) in `MyDrive/pfe_capstone/`.
- Optional: set `PUSH_RESULTS = True` in **Section 14** to commit the small result files (`outputs/*.csv|json|jsonl|txt`, `outputs/deliverables/`) back to the repo. The LLM cache, embeddings and secrets are never committed.
- Label files: download the CSV from the Files panel (or open it on Drive), label it in Excel/Sheets, upload it back to the same path, re-run the cell.

## E. Dashboard demo from Colab
Streamlit needs a tunnel. In separate cells:
```
!streamlit run app.py --server.port 8501 --server.headless true > /content/streamlit.log 2>&1 &
!npm install -g localtunnel -s
!curl -s ipv4.icanhazip.com        # this IP is the tunnel password
!npx localtunnel --port 8501       # open the printed https URL; paste the IP when asked
```
Alternative with no tunnel: create a free **Hugging Face Space** (Streamlit, CPU), push the same repo files, add `LLM_API_KEY` as a Space secret.

## F. Problems
| Symptom | Fix |
|---|---|
| `git clone failed: ... Authentication failed` | private repo: add the `GITHUB_TOKEN` secret (Contents: Read, this repo) and switch Notebook access on |
| `Edit REPO_URL ...` printed | you did not replace `premchand0001-droid/PFE_Copliot` in the notebook |
| `TLS certificate error` | the upload step was skipped or the wrong `.crt`; re-run the bootstrap cell and upload again |
| `LLM /models` fails in pre-flight | wrong certificate/VM or key; if the course endpoint rejects Colab's network, run locally (`docs/../LOCAL_SETUP.md`) |
| Yahoo returns nothing | wait 10 minutes and re-run (data is cached after success) |
| Colab disconnected | reopen the notebook, re-run bootstrap + config + library cells; with `USE_DRIVE = True` nothing is lost |
| `HTTP 429` | another run is using the same key; the client retries automatically |
| GitHub rejects a big file | do not commit `data/` downloads (>100 MB files are refused); commit results only |
