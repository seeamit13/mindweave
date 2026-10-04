---
title: Notebook Clone (RAG)
emoji: 📓
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# 📓 Notebook Clone (NotebookLM / Gemini Notebook style)

A full-stack RAG application: create multiple notebooks, add sources (PDF,
PPTX, TXT, or a web URL), ask questions about a notebook's sources with
cited answers, and generate a report or quiz from the notebook's content.

## What it does

- **Notebook management** — create, rename, delete, and switch between
  notebooks. Each notebook's sources, chat history, and generated artifacts
  are stored completely separately.
- **Source ingestion** — upload PDF / PPTX / TXT files or paste a URL. Text
  is extracted, chunked (sentence-aware or fixed-size), embedded locally
  with `sentence-transformers/all-MiniLM-L6-v2`, and stored in a per-notebook
  ChromaDB collection.
- **RAG chat** — ask a question about a notebook; relevant chunks are
  retrieved, passed to an LLM (via the Groq API) along with the question, and
  the answer is returned with citations back to the specific source chunk(s)
  used. Two retrieval strategies are selectable in the UI (plain vector
  search, and vector search + cross-encoder reranking) — see
  `eval/RAG_EVALUATION.md` for the comparison.
- **Chat history** — every question/answer is persisted per notebook and
  reloaded when you return to it.
- **Artifact generation** — generate a Markdown **Report** or a Markdown
  **Quiz (with answer key)** from a notebook's sources; both are saved to
  disk and viewable/downloadable from the UI.
- **RAG evaluation** — `eval/eval_rag.py` runs a set of test questions
  against both retrieval strategies and records retrieved chunks, response
  time, and answer quality. See `eval/RAG_EVALUATION.md` for the write-up.

## Running locally

```bash
git clone <this-repo-url>
cd notebooklm-clone
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

export GROQ_API_KEY=gsk_...   # required, see below

python app.py
```

Then open the URL Gradio prints (default `http://localhost:7860`).

The first run will download the local embedding model (`all-MiniLM-L6-v2`,
~90MB) and, if you use the reranking retrieval strategy, the cross-encoder
model (`ms-marco-MiniLM-L-6-v2`) — both from Hugging Face, no API key needed
for those.

## Required environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Used for RAG answer generation and report/quiz generation (via the Groq API). Get a free key at https://console.groq.com/keys. |
| `LLM_MODEL` | No | Overrides the Groq model used (default: `llama-3.3-70b-versatile`). |
| `EMBEDDING_MODEL_NAME` | No | Overrides the sentence-transformers embedding model. |
| `RERANKER_MODEL_NAME` | No | Overrides the cross-encoder reranking model. |
| `APP_DATA_DIR` | No | Overrides where notebook/chat/artifact/vector-store data is written (default: `./data`). |

No API keys are committed to this repository. Locally, set them as shell
environment variables (or a `.env` file loaded by your shell — this repo
does not commit one, see `.gitignore`). On Hugging Face, set them as **Space
secrets** (Settings → Variables and secrets). In GitHub Actions, the
deploy workflow only needs `HF_TOKEN` and `HF_SPACE_REPO` as **repository
secrets** (see below) — it does not need `GROQ_API_KEY`, since that's
configured directly on the Space, not baked into the repo.

## How the application is structured

```
app.py                    # Gradio UI - wires notebook mgmt, ingestion, chat, artifacts together
src/
  config.py                # paths, model names, chunking/retrieval parameters (env-var overridable)
  storage.py                # notebook CRUD, chat history, artifact registry (filesystem-backed)
  ingestion.py               # PDF/PPTX/TXT/URL text extraction + two chunking strategies
  rag_engine.py               # embeddings, per-notebook Chroma collections, retrieval strategies, cited generation
  artifacts.py                # report/quiz generation (LLM prompts -> saved .md)
  llm_client.py                # thin Groq API wrapper
eval/
  eval_rag.py                # evaluation harness comparing retrieval strategies
  eval_questions.json         # representative test questions
  RAG_EVALUATION.md            # write-up: methodology, results, tradeoffs, conclusion
ARCHITECTURE.md              # architecture diagram + explanation (assignment section 10)
.github/workflows/deploy.yml # CI/CD: push to main -> deploy to Hugging Face Space
```

## How data is stored

Everything lives under `data/` (path overridable via `APP_DATA_DIR`):

```
data/
  notebooks/
    index.json                 # [{id, name, created_at}, ...]
    <notebook_id>/
      meta.json                 # notebook name + list of source records
      chat_history.json          # full Q&A history for this notebook
      raw_sources/                # original uploaded files
      artifacts/                   # generated report_*.md / quiz_*.md files
  chroma/
    <notebook_id>/                # persistent Chroma collection for this notebook's chunks
```

Each notebook has a unique ID; sources, chat history, and artifacts are all
namespaced under that ID, and everything is written to disk (not kept only
in memory), so restarting the app reloads all existing notebooks, sources,
conversations, and artifacts.

### Storage & persistence limitations (Hugging Face Spaces)

The **free tier of Hugging Face Spaces uses an ephemeral filesystem**: any
files written at runtime (everything under `data/`) are lost whenever the
Space restarts, sleeps and wakes up, or is rebuilt. This is a platform
limitation, not a bug in this app. Documenting it, per the assignment's
guidance, rather than solving it with unnecessary infrastructure:

- For a graded demo/local run, this is not an issue — data persists for the
  life of the running process, which is all that's needed to show
  notebook creation, ingestion, chat, and artifact generation end-to-end.
- For real persistence across restarts on Spaces, the supported options are
  (a) a paid Space with **Persistent Storage** enabled (mounts a real disk
  at a fixed path — just point `APP_DATA_DIR` at it), or (b) swapping the
  filesystem-backed `NotebookStore` for an external store (e.g. a hosted
  Postgres/SQLite-over-network + a hosted vector DB). Neither is required
  for this assignment, so neither is implemented here.

## How to use the application

1. **Create a notebook** — type a name and click "➕ Create".
2. **Add sources** — in the *Sources* tab, upload a PDF/PPTX/TXT file or
   paste a URL and click "Add". Watch the sources table update with the
   chunk count.
3. **Chat** — in the *Chat* tab, pick a retrieval strategy and ask a
   question. The answer includes citations back to the specific source
   file and chunk used.
4. **Generate artifacts** — in the *Artifacts* tab, click "Generate Report"
   or "Generate Quiz + Answer Key"; select a saved artifact from the
   dropdown to view or download it.
5. **Switch/rename/delete notebooks** — use the controls at the top of the
   page; each notebook's data stays isolated.

## Deploying to Hugging Face Spaces + CI/CD

1. Create a new Space on Hugging Face (SDK: **Gradio**), e.g.
   `https://huggingface.co/spaces/<your-username>/notebook-clone`.
2. In the Space's **Settings → Variables and secrets**, add `GROQ_API_KEY`
   as a secret.
3. In this GitHub repository's **Settings → Secrets and variables → Actions**,
   add two repository secrets:
   - `HF_TOKEN` — a Hugging Face access token with write access to the Space
     (create one at huggingface.co/settings/tokens).
   - `HF_SPACE_REPO` — the Space's git URL, e.g.
     `https://huggingface.co/spaces/<your-username>/notebook-clone`.
4. Push to `main`. `.github/workflows/deploy.yml` mirrors the repository to
   the Space, which triggers Hugging Face to rebuild and restart the app
   automatically. No secrets are ever committed to this repo — the workflow
   reads them from GitHub Secrets at deploy time only.

## RAG evaluation

See [`eval/RAG_EVALUATION.md`](eval/RAG_EVALUATION.md) for the full
methodology, results, and tradeoffs comparing plain vector search vs.
vector search + cross-encoder reranking (and notes on comparing chunking
strategies too). Run it yourself with:

```bash
python -m eval.eval_rag --notebook-id <your-notebook-id>
```

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the component diagram and an
explanation of major modules, data flow, storage, the RAG pipeline,
artifact generation, and deployment architecture.

## Testing

`tests/test_pipeline_smoke.py` exercises the full ingest → chunk → embed →
store → retrieve pipeline (plus notebook CRUD, source enable/disable
filtering, chat history, and artifact saving) with a mocked embedder, so it
runs offline with no network access or `GROQ_API_KEY` required:

```bash
pip install -r requirements-dev.txt
pytest tests/test_pipeline_smoke.py -v
```

This also runs automatically on every push/PR via `.github/workflows/test.yml`.

## Deployment & demo checklist

- [`DEPLOYMENT_AND_DEMO_GUIDE.md`](DEPLOYMENT_AND_DEMO_GUIDE.md) — exact
  steps to create the Hugging Face Space, wire up CI/CD, and record the
  required demo video.
- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) — the assignment's
  full checklist, with each item marked as already done in this repo or
  pending one of the steps above.
