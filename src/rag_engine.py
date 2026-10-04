"""
Core RAG engine.

Responsibilities:
  1. Embeddings   - local sentence-transformers model (no API key required)
  2. Vector store - one persistent ChromaDB collection per notebook
  3. Ingestion    - extract -> chunk -> embed -> store, tagged with notebook_id + source_id
  4. Retrieval    - two selectable strategies (used by both the app and eval/eval_rag.py):
                      "vector"  -> plain cosine similarity top-K
                      "rerank"  -> wider vector recall pool, then cross-encoder rerank to top-K
  5. Generation   - retrieved chunks are passed to the LLM with instructions to cite sources;
                     citations are returned alongside the answer for display in the UI.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder

from . import config
from . import ingestion
from . import llm_client
from .storage import store, SourceRecord


@dataclass
class RetrievedChunk:
    text: str
    source_filename: str
    source_id: str
    chunk_index: int
    score: float


@dataclass
class RAGAnswer:
    answer: str
    citations: list = field(default_factory=list)  # list[RetrievedChunk]
    retrieval_time_secs: float = 0.0
    generation_time_secs: float = 0.0


# ---------------------------------------------------------------------------
# Model loading (cached - loaded once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(config.RERANKER_MODEL_NAME)


@lru_cache(maxsize=None)
def _get_chroma_collection(notebook_id: str):
    client = chromadb.PersistentClient(
        path=str(store.chroma_dir(notebook_id)),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=f"nb_{notebook_id}",
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    embedder = _get_embedder()
    return embedder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False).tolist()


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_source(
    notebook_id: str,
    filename: str,
    source_type: str,
    path_or_url: str,
    chunking_strategy: str = "sentence",
) -> SourceRecord:
    """Extract -> chunk -> embed -> store a single source into the notebook's collection.
    Returns the SourceRecord that was persisted to notebook metadata."""
    if not store.notebook_exists(notebook_id):
        raise ValueError(f"Notebook '{notebook_id}' does not exist.")

    text = ingestion.extract_text(source_type, path_or_url)
    chunk_fn = ingestion.CHUNKING_STRATEGIES.get(chunking_strategy, ingestion.chunk_sentences)
    chunks = chunk_fn(text)
    if not chunks:
        raise ingestion.IngestionError("No usable text chunks were produced from this source.")

    source_id = uuid.uuid4().hex[:10]
    embeddings = embed_texts([c.text for c in chunks])

    collection = _get_chroma_collection(notebook_id)
    ids = [f"{source_id}_{c.chunk_index}" for c in chunks]
    metadatas = [
        {
            "source_id": source_id,
            "source_filename": filename,
            "source_type": source_type,
            "chunk_index": c.chunk_index,
            "enabled": True,
        }
        for c in chunks
    ]
    documents = [c.text for c in chunks]
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    record = SourceRecord(
        id=source_id,
        filename=filename,
        source_type=source_type,
        origin=path_or_url,
        added_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        chunk_count=len(chunks),
        enabled=True,
    )
    store.add_source_record(notebook_id, record)
    return record


def set_source_enabled(notebook_id: str, source_id: str, enabled: bool) -> None:
    """Enable/disable a source: updates both the metadata registry and every
    chunk's metadata in Chroma so disabled sources are filtered out of retrieval."""
    store.set_source_enabled(notebook_id, source_id, enabled)
    collection = _get_chroma_collection(notebook_id)
    existing = collection.get(where={"source_id": source_id})
    if not existing["ids"]:
        return
    updated_metas = []
    for md in existing["metadatas"]:
        md = dict(md)
        md["enabled"] = enabled
        updated_metas.append(md)
    collection.update(ids=existing["ids"], metadatas=updated_metas)


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def retrieve_vector(notebook_id: str, query: str, top_k: int = None) -> tuple[list[RetrievedChunk], float]:
    """Strategy 1: plain vector similarity search."""
    top_k = top_k or config.TOP_K
    collection = _get_chroma_collection(notebook_id)
    start = time.perf_counter()
    query_emb = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where={"enabled": True},
    )
    elapsed = time.perf_counter() - start
    return _to_retrieved_chunks(results), elapsed


def retrieve_rerank(notebook_id: str, query: str, top_k: int = None, candidate_k: int = None) -> tuple[list[RetrievedChunk], float]:
    """Strategy 2: retrieve a wider candidate pool via vector search, then
    rerank with a cross-encoder that scores (query, chunk) pairs directly,
    which is more accurate than embedding cosine similarity alone."""
    top_k = top_k or config.TOP_K
    candidate_k = candidate_k or config.RERANK_CANDIDATE_K
    collection = _get_chroma_collection(notebook_id)

    start = time.perf_counter()
    query_emb = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=candidate_k,
        where={"enabled": True},
    )
    candidates = _to_retrieved_chunks(results)
    if not candidates:
        return [], time.perf_counter() - start

    reranker = _get_reranker()
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c.score = float(s)
    candidates.sort(key=lambda c: c.score, reverse=True)
    elapsed = time.perf_counter() - start
    return candidates[:top_k], elapsed


RETRIEVAL_STRATEGIES = {
    "vector": retrieve_vector,
    "rerank": retrieve_rerank,
}


def _to_retrieved_chunks(results: dict) -> list[RetrievedChunk]:
    chunks = []
    if not results.get("ids") or not results["ids"][0]:
        return chunks
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results.get("distances", [[None] * len(ids)])[0]
    for _id, doc, meta, dist in zip(ids, docs, metas, dists):
        score = 1 - dist if dist is not None else 0.0  # cosine distance -> similarity
        chunks.append(RetrievedChunk(
            text=doc,
            source_filename=meta.get("source_filename", "unknown"),
            source_id=meta.get("source_id", ""),
            chunk_index=meta.get("chunk_index", -1),
            score=score,
        ))
    return chunks


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a research assistant answering questions strictly using the
provided source excerpts from the user's notebook. Rules:
- Only use information contained in the excerpts below. Do not use outside knowledge.
- If the excerpts do not contain enough information to answer, say so plainly.
- After each claim, cite the excerpt it came from using the bracketed tag shown before
  that excerpt, e.g. [Source 1].
- Be concise and directly answer the question first, then add supporting detail."""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[Source {i}] (from \"{c.source_filename}\", chunk #{c.chunk_index})\n{c.text}")
    return "\n\n".join(blocks)


def answer_question(
    notebook_id: str,
    question: str,
    strategy: str = "vector",
    top_k: int = None,
) -> RAGAnswer:
    retrieve_fn = RETRIEVAL_STRATEGIES.get(strategy, retrieve_vector)
    chunks, retrieval_time = retrieve_fn(notebook_id, question, top_k=top_k)

    if not chunks:
        return RAGAnswer(
            answer=("I couldn't find any relevant content in this notebook's sources to answer "
                    "that question. Try adding more sources or rephrasing your question."),
            citations=[],
            retrieval_time_secs=retrieval_time,
            generation_time_secs=0.0,
        )

    context = _format_context(chunks)
    user_prompt = f"Source excerpts:\n\n{context}\n\nQuestion: {question}"

    gen_start = time.perf_counter()
    answer_text = llm_client.generate(RAG_SYSTEM_PROMPT, user_prompt)
    gen_time = time.perf_counter() - gen_start

    return RAGAnswer(
        answer=answer_text,
        citations=chunks,
        retrieval_time_secs=retrieval_time,
        generation_time_secs=gen_time,
    )


def get_all_chunks_text(notebook_id: str, max_chars: int = 12000) -> str:
    """Pull a broad sample of the notebook's content for artifact generation
    (report/quiz), rather than relying on a single question's retrieval."""
    collection = _get_chroma_collection(notebook_id)
    data = collection.get(where={"enabled": True})
    docs = data.get("documents", [])
    combined = "\n\n".join(docs)
    return combined[:max_chars]
