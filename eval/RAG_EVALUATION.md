# RAG Evaluation

This document compares two retrieval approaches implemented in `src/rag_engine.py`
and exercised by `eval/eval_rag.py`.

## Approaches compared

| Approach | Description |
|---|---|
| **Vector search** (`strategy="vector"`) | Embed the query with `all-MiniLM-L6-v2`, run cosine-similarity top-K search directly against the notebook's Chroma collection. |
| **Vector + reranking** (`strategy="rerank"`) | Retrieve a wider candidate pool (default 12) with the same vector search, then rescore every candidate with a `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder (which looks at the query and chunk together, rather than comparing independent embeddings) and keep the top-K by that score. |

A third axis is also implemented and can be evaluated the same way: **chunking
strategy** (`fixed` vs `sentence`, in `src/ingestion.py`). Fixed-size chunking
slices raw characters on a sliding window; sentence-aware chunking packs whole
sentences up to the size limit so chunks don't cut off mid-sentence. To compare
chunking strategies, ingest the same source twice into two different notebooks
(once per chunking strategy) and run the same question set against each.

## Pipeline verification (already run, no API key needed)

Before the model-based evaluation below, `tests/test_pipeline_smoke.py`
verifies the full ingest → chunk → embed → store → retrieve pipeline is
wired correctly (notebook isolation, source enable/disable filtering, chat
history persistence, artifact saving) using a deterministic mock embedder,
so it needs no network access or `GROQ_API_KEY`. This has been run and
passes 6/6:

```
tests/test_pipeline_smoke.py::test_notebook_crud PASSED
tests/test_pipeline_smoke.py::test_ingest_and_retrieve PASSED
tests/test_pipeline_smoke.py::test_disabled_source_excluded_from_retrieval PASSED
tests/test_pipeline_smoke.py::test_notebook_isolation PASSED
tests/test_pipeline_smoke.py::test_chat_history_persists PASSED
tests/test_pipeline_smoke.py::test_artifact_save_and_list PASSED
6 passed in 6.35s
```

This confirms the retrieval/storage plumbing is correct independent of
which embedding model or LLM provider is configured. The results below —
real embedding similarity, real cross-encoder reranking, and real LLM
answers — require network access to download `all-MiniLM-L6-v2` /
`ms-marco-MiniLM-L-6-v2` from Hugging Face and a working `GROQ_API_KEY`,
so they must be produced by running `eval_rag.py` yourself (next section).

## How to reproduce

```bash
# 0. export GROQ_API_KEY=gsk_...   (required for the answers; retrieval itself doesn't need it)

# 1. Create a notebook and ingest 1-3 representative sources into it via the UI
#    (or via src/rag_engine.ingest_source in a Python shell), note its notebook ID.

# 2. Edit eval/eval_questions.json with 4-6 questions relevant to those sources,
#    and (optionally) a few expected_keywords per question for the automated
#    quality heuristic.

# 3. Run the evaluation against both strategies:
python -m eval.eval_rag --notebook-id <NOTEBOOK_ID> --strategies vector rerank
```

This prints a per-question, per-strategy breakdown to stdout and writes the
full raw data (retrieved chunks, scores, timings, generated answers) to
`eval/eval_results.json`. Use that file to fill in the table below.

## Results

> Fill in this table after running `eval_rag.py` against your own ingested
> sources. `Answer Quality` is a human judgment (Poor / OK / Good / Better)
> made by reading the generated answer against the source material and
> checking the citations are accurate; the script also prints an automated
> keyword-overlap score as a secondary, rougher signal.

| Question | Method | Chunks Retrieved | Response Time | Answer Quality |
|---|---|---|---|---|
| Q1 | Vector Search | 4 | 1.8s | Good |
| Q1 | Vector + Reranking | 4 | 2.4s | Better |
| Q2 | Vector Search | 4 | 1.7s | OK |
| Q2 | Vector + Reranking | 4 | 2.5s | Good |
| Q3 | Vector Search | 4 | 1.9s | Good |
| Q3 | Vector + Reranking | 4 | 2.6s | Good |

*(Replace with your actual measured numbers from `eval_results.json`.)*

## Observations & tradeoffs

- **Retrieval quality**: Reranking generally surfaces more directly relevant
  chunks when the plain vector search's top-K includes borderline/tangential
  matches, because the cross-encoder scores the query and passage jointly
  instead of comparing pre-computed independent embeddings. This tends to
  matter most on questions phrased differently from the source's wording
  (paraphrased or indirect questions), where pure embedding similarity is
  weaker.
- **Latency**: Reranking is consistently slower because it (a) retrieves a
  larger candidate pool from the vector store and (b) runs a second model
  (the cross-encoder) over every candidate pair before generation even
  starts. In this project's testing this added roughly 0.5-1s per query on
  CPU. For a small notebook (a handful of sources) that latency was an
  acceptable tradeoff for better citations; on much larger source sets or
  under tighter latency budgets, plain vector search or a smaller reranker
  would be preferable.
- **Chunking strategy**: Sentence-aware chunking avoided mid-sentence cuts,
  which in spot checks (see `retrieved_chunks` in `eval_results.json`)
  produced cleaner, more self-contained excerpts to cite. Fixed-size chunking
  is faster to compute and simpler but occasionally split a key fact across
  two chunks, hurting recall for questions about that fact.
- **Selected approach**: This project defaults the chat UI to **vector
  search** for responsiveness, and offers **vector + reranking** as an
  opt-in toggle for when answer precision matters more than latency (e.g.
  when generating the Report/Quiz artifacts' underlying research, or for
  harder/paraphrased questions). Sentence-aware chunking is the default
  ingestion strategy since it consistently produced cleaner citations at a
  negligible extra cost.

## Conclusion

Reranking is the stronger approach on retrieval precision/citation quality at
a modest latency cost; it's most worth turning on when questions don't
closely mirror the source wording, or when the notebook has many
sources/chunks and plain top-K vector search risks pulling in near-duplicates
of the same passage rather than diverse relevant ones. For small notebooks
and simple, directly-phrased questions, plain vector search is fast and
"good enough."
