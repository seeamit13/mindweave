# Architecture

## Diagram

```
                              ┌─────────────────────┐
                              │      Gradio UI       │
                              │      (app.py)        │
                              └──────────┬───────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            ▼                            ▼                            ▼
   ┌────────────────┐          ┌──────────────────┐          ┌──────────────────┐
   │ Notebook Mgmt   │          │     RAG Chat       │          │    Artifacts       │
   │ (src/storage.py)│          │ (src/rag_engine.py)│          │ (src/artifacts.py) │
   └────────┬────────┘          └─────────┬──────────┘          └─────────┬──────────┘
            │                             │                              │
            │                   ┌─────────┴─────────┐                    │
            │                   ▼                    ▼                    │
            │           ┌───────────────┐   ┌────────────────┐            │
            │           │  Ingestion     │   │   Retrieval      │            │
            │           │(src/ingestion.py)│  │ vector | rerank  │            │
            │           └───────┬───────┘   └────────┬─────────┘            │
            │                   │                     │                     │
            │                   ▼                     │                     │
            │           ┌───────────────┐              │                     │
            │           │  Embeddings    │             │                     │
            │           │(all-MiniLM-L6) │             │                     │
            │           └───────┬───────┘              │                     │
            │                   ▼                     ▼                     ▼
            │           ┌────────────────────────────────┐         ┌───────────────┐
            │           │      ChromaDB (per notebook)     │         │  Claude (LLM)  │
            │           │   data/chroma/<notebook_id>/      │         │   (Groq API)   │
            │           └────────────────────────────────┘         └───────────────┘
            ▼
   ┌───────────────────────────────────────────┐
   │        Filesystem (data/notebooks/)          │
   │  meta.json · chat_history.json · artifacts/  │
   └───────────────────────────────────────────┘
```

## Major modules

| Module | Responsibility |
|---|---|
| `app.py` | Gradio UI: layout, event wiring, and translating UI actions into calls on the modules below. Contains no business logic itself. |
| `src/storage.py` | Notebook CRUD, chat history persistence, artifact registry. Everything that is *not* the vector database. Filesystem-backed so the app can restart and reload state. |
| `src/ingestion.py` | Text extraction from PDF (`pypdf`), PPTX (`python-pptx`), TXT, and web URLs (`requests` + `BeautifulSoup`); two chunking strategies (fixed-size and sentence-aware). |
| `src/rag_engine.py` | Owns embeddings (`sentence-transformers`) and the per-notebook Chroma collections; implements the ingest pipeline (extract → chunk → embed → store) and both retrieval strategies (plain vector search, vector + cross-encoder reranking); formats retrieved chunks into a prompt and calls the LLM for a cited answer. |
| `src/artifacts.py` | Prompts the LLM to turn a notebook's aggregated chunk content into a Report or a Quiz+AnswerKey, saved as Markdown via `storage.py`. |
| `src/llm_client.py` | Thin wrapper around the Groq API (`groq` SDK, OpenAI-compatible chat completions) used by both chat and artifact generation, with centralized error handling. |
| `eval/eval_rag.py` | Standalone evaluation harness: runs a question set through both retrieval strategies against a real notebook and records chunks/timings/answers for the write-up in `eval/RAG_EVALUATION.md`. |

## Data flow

**Ingestion:** UI upload/URL → `ingestion.extract_text()` → `ingestion.chunk_sentences()` / `chunk_fixed()` → `rag_engine.embed_texts()` → Chroma collection `add()` (embeddings + text + metadata: source id/filename/chunk index/enabled flag) → a `SourceRecord` is also appended to that notebook's `meta.json` via `storage.py`.

**Chat:** UI question → `storage.append_chat()` (user turn) → `rag_engine.answer_question()` → retrieval strategy (`vector` or `rerank`) queries the notebook's Chroma collection (filtered to `enabled: True` chunks) → retrieved chunks formatted into a system+user prompt → `llm_client.generate()` calls Claude → answer + citations returned → `storage.append_chat()` (assistant turn, with citation metadata) so history survives restarts.

**Artifacts:** UI button → `artifacts.generate_report()` / `generate_quiz()` → `rag_engine.get_all_chunks_text()` pulls a broad sample of the notebook's enabled chunks (not a single retrieval query, since a report should cover the whole notebook) → `llm_client.generate()` → Markdown saved via `storage.save_artifact()` → listed/viewable/downloadable in the UI.

## Notebook storage

Each notebook gets a UUID-based ID at creation time. All of a notebook's
data — its metadata/source registry, chat history, raw artifacts, and
vector embeddings — is namespaced under that ID on disk (`data/notebooks/<id>/...`
and `data/chroma/<id>/`), so notebooks are fully isolated from one another
and deleting a notebook is a matter of removing its two directories. See
`README.md` → "How data is stored" and "Storage & persistence limitations"
for the full layout and the documented Hugging Face Spaces ephemeral-storage
caveat.

## RAG pipeline

Two retrieval strategies are implemented behind a common interface
(`RETRIEVAL_STRATEGIES` in `rag_engine.py`) so they can be swapped from the
UI or exercised identically by the evaluation script:

- **`vector`** — embed the query, cosine-similarity top-K search directly.
- **`rerank`** — embed the query, retrieve a wider top-N candidate pool,
  then rescore every candidate with a cross-encoder (`ms-marco-MiniLM-L-6-v2`)
  that looks at the query and passage together, and keep the top-K by that
  score.

Retrieved chunks are always tagged with their originating source filename
and chunk index, which is what powers the citations shown in the chat UI
and recorded in chat history.

## Artifact generation

Reports and quizzes are generated from a broad text sample of the
notebook (via `get_all_chunks_text`) rather than a single retrieval query,
because these artifacts are meant to synthesize the whole notebook rather
than answer one narrow question. Both use dedicated system prompts
(`REPORT_SYSTEM_PROMPT`, `QUIZ_SYSTEM_PROMPT` in `artifacts.py`) instructing
the LLM to stick to the provided material and, for quizzes, to always
include a numbered answer key.

## Deployment architecture

- **Runtime**: a single Gradio process (`app.py`) serving the UI and running
  all RAG logic in-process — no separate backend/frontend split, which
  keeps the Hugging Face Space deployment simple (one `app.py` entrypoint).
- **Models**: the embedding model and (optional) reranker run locally,
  downloaded from Hugging Face on first use — no API key needed for those.
  Only answer/artifact generation calls out to the Groq API, using
  `GROQ_API_KEY` from the environment (a Space secret in production,
  never committed to the repo).
- **CI/CD**: `.github/workflows/deploy.yml` runs on every push to `main`,
  and force-pushes the repository to the Hugging Face Space's git remote
  using a `HF_TOKEN` GitHub Secret, which causes the Space to rebuild and
  restart with the new code automatically.
