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
which embedding model or LLM provider is configured.

## Evaluation setup

- **Source:** one PDF, `Lesson 2 Wordly Wise.pdf`, a third-grade vocabulary
  lesson. The ingested source was stored as two chunks in notebook
  `914534b097ad`.
- **Test questions:** the five questions in `eval/eval_questions.json`.
- **Compared methods:** vector search and vector + cross-encoder reranking,
  run against the same notebook and source.
- **Timing:** retrieval time and end-to-end time (retrieval plus answer
  generation) are the measurements recorded by `eval/eval_rag.py`.
- **Answer quality:** manually reviewed against the PDF using this rubric:
  **Good** means supported by the source and answers the question without a
  material factual error; **Fair** means mostly relevant but includes a
  material unsupported claim; **Poor** means substantially incorrect or
  unsupported. The result file has no expected keywords, so no automated
  keyword-overlap score is available.

The source PDF and generated responses were available for this review. Each
retrieval method returned both available chunks for every question. Chunk
scores are omitted because the vector similarity scores and cross-encoder
scores are different scales and should not be compared numerically.

## Test questions and retrieved chunks

The test questions were:

1. What is the main topic of these sources?
2. What are the key definitions or terms introduced in the material?
3. Summarize the most important conclusion or finding.
4. Are there any numbers, dates, or statistics mentioned? List them.
5. What question might someone ask that these sources do NOT answer?

Both methods retrieved the same two source chunks for every question. The
following chunk descriptions are paraphrased summaries, not verbatim excerpts:

- **Chunk 0:** lesson title and grade level, followed by the first part of
  the vocabulary list (including attract, crew, dangle, and the beginning of
  drift).
- **Chunk 1:** the continuation/end of the vocabulary list, including the
  remaining drift entry and entries for reverse, signal, and steer.

| Question | Vector chunk order | Reranked chunk order |
|---|---|---|
| Main topic | 0, 1 | 0, 1 |
| Definitions and terms | 0, 1 | 0, 1 |
| Conclusion or finding | 1, 0 | 0, 1 |
| Numbers, dates, or statistics | 0, 1 | 0, 1 |
| Unanswered question | 1, 0 | 1, 0 |

## Results

All times are seconds. “End-to-end” includes answer generation. Both methods
returned two chunks per question.

| Question | Retrieval: vector / rerank | End-to-end: vector / rerank | Quality: vector / rerank |
|---|---:|---:|---|
| Main topic | 7.217 / 2.243 | 9.035 / 2.878 | Good / Good |
| Definitions and terms | 0.044 / 0.140 | 2.283 / 1.588 | Good / Good |
| Conclusion or finding | 0.066 / 0.147 | 0.935 / 0.757 | Good / Good |
| Numbers, dates, or statistics | 0.062 / 0.136 | 1.251 / 1.509 | Fair / Fair |
| Unanswered question | 0.060 / 0.156 | 5.585 / 6.566 | Good / Good |

### Answer-quality review

- **Main topic — Good / Good:** both identify the document as a third-grade
  vocabulary lesson and accurately describe its word-and-definition format.
- **Definitions and terms — Good / Good:** both answers list the vocabulary
  and meanings found in the source, including multiple parts of speech where
  provided.
- **Conclusion or finding — Good / Good:** both correctly explain that the
  source is a word list rather than a report with a conclusion or research
  finding.
- **Numbers, dates, or statistics — Fair / Fair:** the vector answer
  incorrectly says the source numbers entries for “steer”; the PDF does not.
  The reranked answer correctly identifies the grade level and numbered
  definition senses, but incorrectly claims that the numbering repeats in
  the second chunk. Both answers correctly note that no dates or statistical
  data appear.
- **Unanswered question — Good / Good:** both give an example about a
  country's capital, which is outside the scope of the vocabulary lesson.

The saved raw answers, timings, and chunk text are in `eval/eval_results.json`.
Mean end-to-end time across the five questions was 3.818 s for vector search
and 2.660 s for reranking. This is descriptive only: vector retrieval took
7.217 s on the first query versus 0.044–0.066 s on the remaining four,
consistent with a first-query initialization or warm-up effect. On those
remaining queries, reranking retrieval took 0.074–0.096 s longer each.
Generation times also varied, so these five observations are not a stable
latency benchmark.

## Final conclusions

On this run, both methods retrieved the entire available two-chunk source for
every question, so reranking did not increase source coverage or demonstrate
better chunk selection. Both approaches produced four answers rated Good and
one Fair. Each numeric-items answer included one unsupported statement, so
this run does not establish a general answer-quality advantage for either
method.

The evaluation therefore supports only a narrow conclusion: both strategies
can answer basic questions from this small vocabulary document, with one
observed unsupported detail in each method's numeric-items response. It does
**not** show that reranking is generally more accurate or faster. A meaningful
retrieval-quality comparison needs a larger corpus with more than two chunks,
questions with source-checkable expected answers, and repeated timed runs to
reduce warm-up and generation variability. The current UI's vector-search
default remains a reasonable responsiveness choice; reranking should be judged
on a broader benchmark before changing that default.

## Reproducing the run

From the repository root, with the project environment and `GROQ_API_KEY`
configured, run:

```powershell
.\venv\Scripts\python.exe -m eval.eval_rag --notebook-id 914534b097ad --strategies vector rerank
```

The command prints per-question results and writes the raw responses,
retrieved chunks, and timing measurements to `eval/eval_results.json`.
