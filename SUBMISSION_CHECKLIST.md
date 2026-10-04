# Submission Checklist

Status key: ✅ done in this repo · 🔲 requires you to do one external action
(create the HF Space / push to GitHub / record video — see
`DEPLOYMENT_AND_DEMO_GUIDE.md`). Optional/advanced items (assignment
section 12) are intentionally skipped, per scope.

## Application

- ✅ Application runs locally (`python app.py`) — verified: all modules
  import and compile; offline pipeline tests pass (`tests/test_pipeline_smoke.py`, 6/6).
- 🔲 Application runs on Hugging Face — code and Space config are ready
  (see `DEPLOYMENT_AND_DEMO_GUIDE.md` Part 1 & 3); requires creating the
  actual Space with your accounts/keys.
- ✅ Multiple notebooks can be created (`storage.NotebookStore.create_notebook`, UI "➕ Create")
- ✅ Notebooks can be switched (`notebook_dropdown.change` → `switch_notebook`)
- ✅ Notebooks can be renamed/deleted (`rename_notebook` / `delete_notebook`)
- ✅ PDF ingestion works (`ingestion.extract_text_from_pdf`, via `pypdf`)
- ✅ PPTX ingestion works (`ingestion.extract_text_from_pptx`, via `python-pptx`)
- ✅ TXT ingestion works (`ingestion.extract_text_from_txt`)
- ✅ URL ingestion works (`ingestion.extract_text_from_url`, via `requests` + `BeautifulSoup`)
- ✅ RAG questions can be asked (`rag_engine.answer_question`, Chat tab)
- ✅ Answers include citations/source references (per-chunk source filename + chunk index, shown in UI and stored in chat history)
- ✅ Chat history is stored (`storage.append_chat` / `get_chat_history`, persisted to `chat_history.json`, reloaded on restart)
- ✅ Reports can be generated (`artifacts.generate_report`, saved as `.md`)
- ✅ Quizzes can be generated (`artifacts.generate_quiz`, saved as `.md`)
- ✅ Quiz answer keys are included (enforced via `QUIZ_SYSTEM_PROMPT` — requires an "## Answer Key" section)
- ✅ Artifacts can be viewed/downloaded (Artifacts tab: dropdown viewer + `gr.File` download)
- ✅ Appropriate error handling is implemented (`IngestionError`/`LLMError` caught throughout `app.py`, surfaced as inline `⚠️ Error:` messages instead of crashing; see e.g. `upload_file`, `add_url`, `ask_question`, `gen_artifact`)

## Hugging Face

- 🔲 Hugging Face Space exists — create per `DEPLOYMENT_AND_DEMO_GUIDE.md` Part 1
- 🔲 Space builds and starts successfully — verify after Part 3
- ✅ Required dependencies are included (`requirements.txt`)
- ✅ API keys are stored as secrets (documented: `GROQ_API_KEY` as a Space secret — see README "Required environment variables")
- ✅ No API keys are committed to GitHub (`.gitignore` excludes `.env*`; `config.py` reads only from `os.environ`)
- ✅ README explains the application (`README.md`: what it does, how to run locally, env vars, structure, storage, usage)
- ✅ Storage behavior/limitations are documented (README "Storage & persistence limitations (Hugging Face Spaces)")
- 🔲 Deployment demonstration is included — record per `DEPLOYMENT_AND_DEMO_GUIDE.md` Part 4
- 🔲 Live application is shown in the demonstration — same as above
- 🔲 Successful GitHub Actions deployment is shown — same as above

## GitHub

- 🔲 Repository is accessible — push per `DEPLOYMENT_AND_DEMO_GUIDE.md` Part 2
- ✅ Source code is committed (ready to commit: `app.py`, `src/`, `eval/`, `tests/`)
- ✅ `.gitignore` is included (excludes secrets, `data/`, caches, `eval/eval_results.json`)
- ✅ GitHub Actions workflow is included (`.github/workflows/deploy.yml`; a second `test.yml` also runs the offline smoke tests on every push/PR)
- 🔲 Hugging Face token is stored as a GitHub Secret — add `HF_TOKEN` per Part 3
- ✅ Pushes trigger deployment (`on: push: branches: [main]` in `deploy.yml`)
- 🔲 Deployment completes successfully — verify the Actions run is green after Part 3

## Documentation

- ✅ Architecture diagram is included (`ARCHITECTURE.md`)
- ✅ Major modules are explained (`ARCHITECTURE.md` "Major modules" table)
- ✅ Data flow is documented (`ARCHITECTURE.md` "Data flow": ingestion / chat / artifacts)
- ✅ Storage approach is explained (`README.md` "How data is stored" + `ARCHITECTURE.md` "Notebook storage")
- ✅ Two RAG approaches are compared (`eval/RAG_EVALUATION.md`: plain vector search vs. vector + cross-encoder reranking, plus a note on comparing chunking strategies)
- 🔲 Retrieved chunks are shown with real data — `eval_rag.py` prints/saves them; the offline pipeline test shows the mechanism works, but real model-based chunks require running `eval_rag.py` with network + `GROQ_API_KEY` (Part 5)
- 🔲 Response times are measured with real data — same as above; the eval script measures and logs real timings once run against a real embedding model
- ✅ Tradeoffs are discussed (`eval/RAG_EVALUATION.md` "Observations & tradeoffs")
- 🔲 Hugging Face URL is provided — add once Part 1/3 are done
- 🔲 GitHub URL is provided — add once Part 2 is done

## Explicitly skipped (optional, per assignment section 12 and section 3 "Optional" artifacts)

Not implemented, by design: podcast transcript/audio, flashcards, mind
maps, YouTube sources, CSV/tabular sources, audio sources, multiple podcast
speakers, custom artifact-generation prompts, and additional retrieval
techniques beyond the two compared in the evaluation. The required feature
set (sections 1–11 minus the optional callouts) is fully implemented.
