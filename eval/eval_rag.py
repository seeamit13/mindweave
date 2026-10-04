"""
RAG evaluation harness (assignment section 6).

Compares two retrieval approaches on a small set of representative questions
against a given, already-populated notebook:

  1. "vector"  - plain vector similarity search (top-K)
  2. "rerank"  - vector search over a wider candidate pool, reranked with a
                 cross-encoder before taking the top-K

For each (question, strategy) pair it records:
  - the retrieved chunks (source + snippet)
  - response time (retrieval time and total time including generation)
  - the generated answer
  - a simple automated relevance/quality heuristic (keyword-overlap score),
    intended as a starting point — the accompanying RAG_EVALUATION.md
    should be filled in with human judgment on top of this raw data.

Usage:
    python -m eval.eval_rag --notebook-id <id> --questions eval/eval_questions.json

Output:
    Prints a results table to stdout and writes eval/eval_results.json with
    the full raw data (retrieved chunks, timings, answers) for write-up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rag_engine  # noqa: E402
from src.llm_client import LLMError  # noqa: E402


def keyword_overlap_score(answer: str, expected_keywords: list[str]) -> float:
    """Very simple heuristic: fraction of expected keywords that appear
    (case-insensitively) in the generated answer. Not a substitute for
    human judgment, but useful as a quick, reproducible signal."""
    if not expected_keywords:
        return float("nan")
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(hits / len(expected_keywords), 2)


def run_eval(notebook_id: str, questions_path: str, strategies: list[str]):
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        question_text = q["question"]
        expected_keywords = q.get("expected_keywords", [])

        for strategy in strategies:
            print(f"\n--- Q: {question_text!r}  |  strategy: {strategy} ---")
            start_total = time.perf_counter()
            try:
                rag_answer = rag_engine.answer_question(notebook_id, question_text, strategy=strategy)
                error = None
            except LLMError as e:
                rag_answer = None
                error = str(e)
            total_time = time.perf_counter() - start_total

            if rag_answer is None:
                print(f"  ERROR: {error}")
                results.append({
                    "question": question_text,
                    "strategy": strategy,
                    "error": error,
                })
                continue

            quality_score = keyword_overlap_score(rag_answer.answer, expected_keywords)
            print(f"  Retrieved {len(rag_answer.citations)} chunks in {rag_answer.retrieval_time_secs:.2f}s")
            print(f"  Generation time: {rag_answer.generation_time_secs:.2f}s | Total: {total_time:.2f}s")
            print(f"  Keyword-overlap quality score: {quality_score}")
            for i, c in enumerate(rag_answer.citations, start=1):
                snippet = c.text[:120].replace("\n", " ")
                print(f"    [{i}] {c.source_filename} #{c.chunk_index} (score={c.score:.3f}): {snippet}...")

            results.append({
                "question": question_text,
                "strategy": strategy,
                "retrieval_time_secs": round(rag_answer.retrieval_time_secs, 3),
                "generation_time_secs": round(rag_answer.generation_time_secs, 3),
                "total_time_secs": round(total_time, 3),
                "num_chunks_retrieved": len(rag_answer.citations),
                "retrieved_chunks": [
                    {
                        "source_filename": c.source_filename,
                        "chunk_index": c.chunk_index,
                        "score": round(c.score, 4),
                        "snippet": c.text[:300],
                    }
                    for c in rag_answer.citations
                ],
                "answer": rag_answer.answer,
                "keyword_overlap_quality_score": quality_score,
            })

    out_path = Path(__file__).resolve().parent / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval strategies on a notebook.")
    parser.add_argument("--notebook-id", required=True, help="ID of an existing notebook with ingested sources.")
    parser.add_argument(
        "--questions",
        default=str(Path(__file__).resolve().parent / "eval_questions.json"),
        help="Path to a JSON file of {question, expected_keywords} objects.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["vector", "rerank"],
        choices=list(rag_engine.RETRIEVAL_STRATEGIES.keys()),
    )
    args = parser.parse_args()
    run_eval(args.notebook_id, args.questions, args.strategies)


if __name__ == "__main__":
    main()
