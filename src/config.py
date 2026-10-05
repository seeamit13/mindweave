"""
Central configuration for Mindweave.

All paths are relative to DATA_DIR so the whole app's state (notebooks,
sources, chat history, artifacts, vector DB) lives under one directory that
can be mounted as a persistent volume, or simply documented as ephemeral
when running on a storage-less deployment target like the free HF Spaces
tier (see README "Storage & Persistence Limitations").
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", BASE_DIR / "data"))

NOTEBOOKS_DIR = DATA_DIR / "notebooks"          # per-notebook metadata + sources + chat + artifacts
CHROMA_DIR = DATA_DIR / "chroma"                # persistent vector store (one collection per notebook)

NOTEBOOKS_INDEX_FILE = NOTEBOOKS_DIR / "index.json"  # list of {id, name, created_at}

for d in (DATA_DIR, NOTEBOOKS_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM / Embeddings
# ---------------------------------------------------------------------------
# Chat/generation model - served via the Groq API (OpenAI-compatible, fast,
# free-tier available at https://console.groq.com/keys).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1500"))

# Local, free, no-API-key-required embedding model (downloaded on first run).
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Cross-encoder used only by the "reranking" retrieval strategy in the RAG evaluation.
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))       # characters
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))  # characters

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = int(os.environ.get("TOP_K", "4"))
RERANK_CANDIDATE_K = int(os.environ.get("RERANK_CANDIDATE_K", "12"))  # pool size before reranking

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
MAX_URL_FETCH_CHARS = 200_000  # safety cap on raw HTML text extracted from a URL
REQUEST_TIMEOUT_SECS = 15
