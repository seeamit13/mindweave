"""
Notebook storage manager.

Responsible for everything that is NOT the vector database:
  - notebook CRUD (create / rename / delete / list)
  - per-notebook source metadata (filename, type, added_at, chunk_count)
  - per-notebook chat history
  - per-notebook generated artifacts registry

Layout on disk:

  data/
    notebooks/
      index.json                     <- [{id, name, created_at}, ...]
      <notebook_id>/
        meta.json                    <- {id, name, created_at, sources: [...]}
        chat_history.json            <- [{role, content, citations, ts}, ...]
        raw_sources/                 <- original uploaded files (for re-ingestion/debug)
        artifacts/
          report_<ts>.md
          quiz_<ts>.md
    chroma/
      <notebook_id>/                 <- Chroma persistent collection storage

This keeps every notebook's data fully isolated and enables an "on restart,
reload everything" model: nothing lives only in memory.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


@dataclass
class SourceRecord:
    id: str
    filename: str
    source_type: str  # "pdf" | "pptx" | "txt" | "url"
    origin: str        # file name or URL
    added_at: str
    chunk_count: int
    enabled: bool = True


@dataclass
class Notebook:
    id: str
    name: str
    created_at: str
    sources: list = field(default_factory=list)  # list[dict] (SourceRecord as dict)


class NotebookStore:
    """Filesystem-backed notebook manager. All methods are safe to call repeatedly
    (idempotent reload) so app restarts pick up existing state automatically."""

    def __init__(self):
        config.NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
        if not config.NOTEBOOKS_INDEX_FILE.exists():
            _write_json(config.NOTEBOOKS_INDEX_FILE, [])

    # -- index -------------------------------------------------------------
    def _load_index(self) -> list:
        return _read_json(config.NOTEBOOKS_INDEX_FILE, [])

    def _save_index(self, index: list) -> None:
        _write_json(config.NOTEBOOKS_INDEX_FILE, index)

    def list_notebooks(self) -> list[dict]:
        return sorted(self._load_index(), key=lambda n: n["created_at"], reverse=True)

    # -- paths ---------------------------------------------------------------
    def notebook_dir(self, notebook_id: str) -> Path:
        return config.NOTEBOOKS_DIR / notebook_id

    def raw_sources_dir(self, notebook_id: str) -> Path:
        return self.notebook_dir(notebook_id) / "raw_sources"

    def artifacts_dir(self, notebook_id: str) -> Path:
        return self.notebook_dir(notebook_id) / "artifacts"

    def meta_path(self, notebook_id: str) -> Path:
        return self.notebook_dir(notebook_id) / "meta.json"

    def chat_path(self, notebook_id: str) -> Path:
        return self.notebook_dir(notebook_id) / "chat_history.json"

    def chroma_dir(self, notebook_id: str) -> Path:
        return config.CHROMA_DIR / notebook_id

    # -- CRUD ------------------------------------------------------------
    def create_notebook(self, name: str) -> dict:
        name = (name or "").strip() or "Untitled Notebook"
        notebook_id = uuid.uuid4().hex[:12]
        created_at = _now_iso()

        nb_dir = self.notebook_dir(notebook_id)
        self.raw_sources_dir(notebook_id).mkdir(parents=True, exist_ok=True)
        self.artifacts_dir(notebook_id).mkdir(parents=True, exist_ok=True)

        nb = Notebook(id=notebook_id, name=name, created_at=created_at, sources=[])
        _write_json(self.meta_path(notebook_id), asdict(nb))
        _write_json(self.chat_path(notebook_id), [])

        index = self._load_index()
        index.append({"id": notebook_id, "name": name, "created_at": created_at})
        self._save_index(index)
        return asdict(nb)

    def rename_notebook(self, notebook_id: str, new_name: str) -> dict:
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("New notebook name cannot be empty.")
        meta = self.get_meta(notebook_id)
        meta["name"] = new_name
        _write_json(self.meta_path(notebook_id), meta)

        index = self._load_index()
        for entry in index:
            if entry["id"] == notebook_id:
                entry["name"] = new_name
        self._save_index(index)
        return meta

    def delete_notebook(self, notebook_id: str) -> None:
        nb_dir = self.notebook_dir(notebook_id)
        if nb_dir.exists():
            shutil.rmtree(nb_dir, ignore_errors=True)
        chroma_dir = self.chroma_dir(notebook_id)
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir, ignore_errors=True)
        index = [e for e in self._load_index() if e["id"] != notebook_id]
        self._save_index(index)

    def get_meta(self, notebook_id: str) -> dict:
        meta = _read_json(self.meta_path(notebook_id), None)
        if meta is None:
            raise KeyError(f"Notebook '{notebook_id}' does not exist.")
        return meta

    def notebook_exists(self, notebook_id: str) -> bool:
        return self.meta_path(notebook_id).exists()

    # -- sources -----------------------------------------------------------
    def add_source_record(self, notebook_id: str, source: SourceRecord) -> None:
        meta = self.get_meta(notebook_id)
        meta["sources"].append(asdict(source))
        _write_json(self.meta_path(notebook_id), meta)

    def list_sources(self, notebook_id: str) -> list[dict]:
        return self.get_meta(notebook_id).get("sources", [])

    def set_source_enabled(self, notebook_id: str, source_id: str, enabled: bool) -> None:
        meta = self.get_meta(notebook_id)
        for s in meta["sources"]:
            if s["id"] == source_id:
                s["enabled"] = enabled
        _write_json(self.meta_path(notebook_id), meta)

    # -- chat history --------------------------------------------------------
    def get_chat_history(self, notebook_id: str) -> list[dict]:
        return _read_json(self.chat_path(notebook_id), [])

    def append_chat(self, notebook_id: str, role: str, content: str, citations: Optional[list] = None) -> None:
        history = self.get_chat_history(notebook_id)
        history.append({
            "role": role,
            "content": content,
            "citations": citations or [],
            "ts": _now_iso(),
        })
        _write_json(self.chat_path(notebook_id), history)

    # -- artifacts -----------------------------------------------------------
    def save_artifact(self, notebook_id: str, kind: str, markdown_text: str) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"{kind}_{ts}.md"
        path = self.artifacts_dir(notebook_id) / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        return path

    def list_artifacts(self, notebook_id: str) -> list[Path]:
        d = self.artifacts_dir(notebook_id)
        if not d.exists():
            return []
        return sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)


store = NotebookStore()
