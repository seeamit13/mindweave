"""
NotebookLM / Gemini Notebook clone - Gradio application entrypoint.

Run locally:
    pip install -r requirements.txt
    export GROQ_API_KEY=gsk_...
    python app.py

See README.md for full setup, environment variables, and architecture notes.
"""

from __future__ import annotations

import traceback
import logging

import gradio as gr

from src.storage import store
from src import ingestion
from src import rag_engine
from src import artifacts
from src.llm_client import LLMError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _notebook_choices():
    notebooks = store.list_notebooks()
    return [(f'{n["name"]}  ({n["id"]})', n["id"]) for n in notebooks]


def _err(msg: str) -> str:
    return f"⚠️ **Error:** {msg}"


def _format_sources_table(notebook_id: str):
    if not notebook_id:
        return []
    rows = []
    for s in store.list_sources(notebook_id):
        rows.append([s["id"], s["filename"], s["source_type"], s["chunk_count"], "Yes" if s.get("enabled", True) else "No"])
    return rows


def _format_chat_history(notebook_id: str):
    if not notebook_id:
        return []
    history = store.get_chat_history(notebook_id)
    pairs = []
    pending_user = None
    for msg in history:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant":
            content = msg["content"]
            if msg.get("citations"):
                cite_lines = "\n".join(
                    f"- *{c['source_filename']}* (chunk #{c['chunk_index']})" for c in msg["citations"]
                )
                content += f"\n\n**Citations:**\n{cite_lines}"
            pairs.append((pending_user or "", content))
            pending_user = None
    return pairs


def _artifact_dropdown_update(notebook_id: str, selected: str | None = None):
    choices = [str(p) for p in store.list_artifacts(notebook_id)] if notebook_id else []
    if selected not in choices:
        selected = None
    return gr.update(choices=choices, value=selected)


# ---------------------------------------------------------------------------
# Notebook management callbacks
# ---------------------------------------------------------------------------

def create_notebook(name):
    try:
        nb = store.create_notebook(name)
        choices = _notebook_choices()
        return (
            gr.update(choices=choices, value=nb["id"]),
            f"✅ Created notebook **{nb['name']}**.",
            "",
        )
    except Exception as e:
        return gr.update(), _err(str(e)), name


def rename_notebook(notebook_id, new_name):
    if not notebook_id:
        return gr.update(), _err("Select a notebook first.")
    try:
        store.rename_notebook(notebook_id, new_name)
        return gr.update(choices=_notebook_choices(), value=notebook_id), f"✅ Renamed to **{new_name}**."
    except Exception as e:
        return gr.update(), _err(str(e))


def delete_notebook(notebook_id):
    if not notebook_id:
        return gr.update(), _err("Select a notebook first."), [], [], gr.update(), "", None
    try:
        store.delete_notebook(notebook_id)
        choices = _notebook_choices()
        new_val = choices[0][1] if choices else None
        return (
            gr.update(choices=choices, value=new_val),
            "🗑️ Notebook deleted.",
            _format_sources_table(new_val),
            _format_chat_history(new_val),
            _artifact_dropdown_update(new_val),
            "",
            None,
        )
    except Exception as e:
        return gr.update(), _err(str(e)), [], [], gr.update(), gr.update(), gr.update()


def switch_notebook(notebook_id):
    return (
        _format_sources_table(notebook_id),
        _format_chat_history(notebook_id),
        _artifact_dropdown_update(notebook_id),
        "",
        None,
    )


# ---------------------------------------------------------------------------
# Ingestion callbacks
# ---------------------------------------------------------------------------

def upload_file(notebook_id, file_obj, chunking_strategy):
    if not notebook_id:
        return _err("Select or create a notebook first."), _format_sources_table(notebook_id)
    if file_obj is None:
        return _err("No file selected."), _format_sources_table(notebook_id)
    try:
        source_type = ingestion.infer_source_type(file_obj.name)
        record = rag_engine.ingest_source(
            notebook_id=notebook_id,
            filename=file_obj.name.split("/")[-1],
            source_type=source_type,
            path_or_url=file_obj.name,
            chunking_strategy=chunking_strategy,
        )
        return (
            f"✅ Ingested **{record.filename}** ({record.chunk_count} chunks).",
            _format_sources_table(notebook_id),
        )
    except ingestion.IngestionError as e:
        return _err(str(e)), _format_sources_table(notebook_id)
    except Exception:
        return _err("Unexpected error during ingestion. Check the file and try again."), _format_sources_table(notebook_id)


def add_url(notebook_id, url, chunking_strategy):
    if not notebook_id:
        return _err("Select or create a notebook first."), _format_sources_table(notebook_id), url
    if not url or not url.strip():
        return _err("Enter a URL first."), _format_sources_table(notebook_id), url
    try:
        record = rag_engine.ingest_source(
            notebook_id=notebook_id,
            filename=url.strip(),
            source_type="url",
            path_or_url=url.strip(),
            chunking_strategy=chunking_strategy,
        )
        return (
            f"✅ Ingested **{record.filename}** ({record.chunk_count} chunks).",
            _format_sources_table(notebook_id),
            "",
        )
    except ingestion.IngestionError as e:
        return _err(str(e)), _format_sources_table(notebook_id), url
    except Exception:
        return _err("Unexpected error fetching/ingesting that URL."), _format_sources_table(notebook_id), url


# ---------------------------------------------------------------------------
# Chat callback
# ---------------------------------------------------------------------------

def ask_question(notebook_id, question, strategy):
    if not notebook_id:
        return _format_chat_history(notebook_id), _err("Select a notebook first."), ""
    if not question or not question.strip():
        return _format_chat_history(notebook_id), "Enter a question first.", question
    try:
        store.append_chat(notebook_id, "user", question)
        result = rag_engine.answer_question(notebook_id, question, strategy=strategy)
        citations = [
            {"source_filename": c.source_filename, "chunk_index": c.chunk_index, "score": round(c.score, 3)}
            for c in result.citations
        ]
        store.append_chat(notebook_id, "assistant", result.answer, citations)
        status = (
            f"Retrieved {len(result.citations)} chunk(s) in {result.retrieval_time_secs:.2f}s, "
            f"generated answer in {result.generation_time_secs:.2f}s using **{strategy}** retrieval."
        )
        return _format_chat_history(notebook_id), status, ""
    except LLMError as e:
        return _format_chat_history(notebook_id), _err(str(e)), question
    except Exception:
        return _format_chat_history(notebook_id), _err("Unexpected error answering the question."), question


# ---------------------------------------------------------------------------
# Artifact callbacks
# ---------------------------------------------------------------------------

def gen_artifact(notebook_id, kind):
    if not notebook_id:
        return _err("Select a notebook first."), gr.update(), gr.update(), gr.update()
    try:
        fn = artifacts.ARTIFACT_GENERATORS[kind]
        path = fn(notebook_id)
        return (
            f"✅ Generated **{kind}**: `{path.name}`",
            _artifact_dropdown_update(notebook_id, str(path)),
            view_artifact(str(path)),
            str(path),
        )
    except (ValueError, LLMError) as e:
        return _err(str(e)), gr.update(), gr.update(), gr.update()
    except Exception as e:
        logger.exception("Unexpected error generating %s for notebook %s", kind, notebook_id)
        return (
            _err(f"Unexpected error generating {kind}: {e}"),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def view_artifact(path):
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return _err(f"Could not read artifact: {e}")


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Mindweave") as demo:
    gr.Markdown("# 📓 Mindweave — Turn your sources into insight")
    gr.Markdown(
        "Create a notebook, add sources (PDF / PPTX / TXT / URL), ask questions with cited "
        "answers, and generate a report or quiz from the notebook's content."
    )

    with gr.Row():
        notebook_dropdown = gr.Dropdown(choices=_notebook_choices(), label="Active Notebook", scale=3)
        new_notebook_name = gr.Textbox(label="New notebook name", placeholder="e.g. Biology 101", scale=2)
        create_btn = gr.Button("➕ Create", scale=1)

    with gr.Row():
        rename_input = gr.Textbox(label="Rename active notebook to", scale=3)
        rename_btn = gr.Button("✏️ Rename", scale=1)
        delete_btn = gr.Button("🗑️ Delete active notebook", scale=1, variant="stop")

    notebook_status = gr.Markdown()

    with gr.Tabs():
        # -- Sources tab ------------------------------------------------
        with gr.Tab("📁 Sources"):
            chunking_choice = gr.Radio(
                choices=[("Sentence-aware (recommended)", "sentence"), ("Fixed-size", "fixed")],
                value="sentence",
                label="Chunking strategy",
            )
            with gr.Row():
                file_upload = gr.File(label="Upload PDF / PPTX / TXT", file_types=[".pdf", ".pptx", ".txt"])
                upload_btn = gr.Button("Add file")
            with gr.Row():
                url_input = gr.Textbox(label="Or add a web URL", placeholder="https://example.com/article")
                url_btn = gr.Button("Add URL")
            ingest_status = gr.Markdown()
            sources_table = gr.Dataframe(
                headers=["Source ID", "Filename", "Type", "Chunks", "Enabled"],
                label="Sources in this notebook",
                interactive=False,
            )

        # -- Chat tab -----------------------------------------------------
        with gr.Tab("💬 Chat"):
            strategy_choice = gr.Radio(
                choices=[("Vector search", "vector"), ("Vector + reranking", "rerank")],
                value="vector",
                label="Retrieval strategy",
            )
            chatbot = gr.Chatbot(label="Conversation", height=420)
            question_input = gr.Textbox(label="Ask a question about this notebook's sources", placeholder="What are the main points?")
            ask_btn = gr.Button("Ask")
            chat_status = gr.Markdown()

        # -- Artifacts tab --------------------------------------------------
        with gr.Tab("📄 Artifacts"):
            with gr.Row():
                report_btn = gr.Button("Generate Report (.md)")
                quiz_btn = gr.Button("Generate Quiz + Answer Key (.md)")
            artifact_status = gr.Markdown()
            artifact_dropdown = gr.Dropdown(label="Saved artifacts (select to view/download)", choices=[])
            artifact_viewer = gr.Markdown()
            artifact_file = gr.File(label="Download")

    # -- wiring ------------------------------------------------------------
    create_btn.click(create_notebook, [new_notebook_name], [notebook_dropdown, notebook_status, new_notebook_name])
    rename_btn.click(rename_notebook, [notebook_dropdown, rename_input], [notebook_dropdown, notebook_status])
    delete_btn.click(
        delete_notebook,
        [notebook_dropdown],
        [
            notebook_dropdown,
            notebook_status,
            sources_table,
            chatbot,
            artifact_dropdown,
            artifact_viewer,
            artifact_file,
        ],
    )
    notebook_dropdown.change(
        switch_notebook,
        [notebook_dropdown],
        [sources_table, chatbot, artifact_dropdown, artifact_viewer, artifact_file],
    )

    upload_btn.click(upload_file, [notebook_dropdown, file_upload, chunking_choice], [ingest_status, sources_table])
    url_btn.click(add_url, [notebook_dropdown, url_input, chunking_choice], [ingest_status, sources_table, url_input])

    ask_btn.click(ask_question, [notebook_dropdown, question_input, strategy_choice], [chatbot, chat_status, question_input])
    question_input.submit(ask_question, [notebook_dropdown, question_input, strategy_choice], [chatbot, chat_status, question_input])

    artifact_outputs = [artifact_status, artifact_dropdown, artifact_viewer, artifact_file]
    report_btn.click(lambda nb: gen_artifact(nb, "report"), [notebook_dropdown], artifact_outputs)
    quiz_btn.click(lambda nb: gen_artifact(nb, "quiz"), [notebook_dropdown], artifact_outputs)
    artifact_dropdown.change(view_artifact, [artifact_dropdown], [artifact_viewer])
    artifact_dropdown.change(lambda p: p, [artifact_dropdown], [artifact_file])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
