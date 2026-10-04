# Mindweave — Deployment & Demo Guide

This repo is deployment-ready: the application code, `requirements.txt`,
`.github/workflows/deploy.yml`, and HF Spaces config (front-matter in
`README.md`) are all in place. Two deliverables require actions on
accounts this project can't act on for you — a live Hugging Face Space and
a screen-recorded demo — so this is the exact runbook to produce both in
about 10-15 minutes.

## Part 1 — Create the Hugging Face Space

Already deployed under another name? Open the Space's **Settings** tab and
rename the repository there. Hugging Face redirects the old URL to the new
one. Then update the GitHub `HF_SPACE_REPO` secret to the new Space URL
(see Part 3) so future deployments target the renamed Space.

1. Go to https://huggingface.co/new-space.
2. Owner: your account. Space name: e.g. `mindweave`. SDK: **Gradio**.
   Visibility: your choice (Public makes the demo link shareable).
3. Click **Create Space**. Hugging Face gives you an empty git repo at
   `https://huggingface.co/spaces/<your-username>/mindweave`.
4. In the Space, go to **Settings → Variables and secrets → New secret**
   and add:
   - `GROQ_API_KEY` = your key from https://console.groq.com/keys (free tier available)
5. Leave the Space as-is for now — step 3 below will push real code to it.

## Part 2 — Push this repo to GitHub

```bash
cd notebooklm-clone
git init
git add .
git commit -m "Initial commit: Mindweave RAG app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Part 3 — Wire up GitHub Actions → Hugging Face CI/CD

1. On GitHub: **Settings → Secrets and variables → Actions → New repository secret**, add:
   - `HF_TOKEN` — a Hugging Face token with **write** access
     (create at https://huggingface.co/settings/tokens).
   - `HF_SPACE_REPO` — `https://huggingface.co/spaces/<your-username>/mindweave`
2. Push any change to `main` (or re-run the `Deploy to Hugging Face Space`
   workflow manually from the **Actions** tab — it supports
   `workflow_dispatch`). Watch the run go green.
3. Open your Space URL — it should build and the Gradio app should load.
   First build can take a few minutes (it downloads the embedding model).

At this point both required checklist items are satisfied:
- ✅ Hugging Face Space exists, builds, and starts successfully
- ✅ GitHub Actions workflow automatically deploys on push, using secrets (no keys committed)

## Part 4 — Record the 1–2 minute demo

Record your screen (QuickTime, OBS, Loom, or similar) showing, in order:

1. **Live Space** — open your Hugging Face Space URL in a browser tab, show it's live.
2. **Create/select a notebook** — click "➕ Create", give it a name.
3. **Add a source** — upload a PDF/PPTX/TXT, or paste a URL, in the Sources tab; show the sources table update with a chunk count.
4. **Ask a RAG question** — go to the Chat tab, ask a question about the source you just added.
5. **Show a citation** — point out the "Citations" section under the answer, naming the source file/chunk it came from.
6. **Generate an artifact** — click "Generate Report" or "Generate Quiz + Answer Key", then select it from the dropdown to show the rendered Markdown.
7. **GitHub Actions workflow** — switch to the GitHub repo's **Actions** tab and show the `Deploy to Hugging Face Space` run completing successfully (green check).

Save/export the recording (e.g. `demo.mp4`) and include it or a link to it
(YouTube unlisted, Google Drive, Loom link, etc.) in your submission.

## Part 5 — Fill in the real RAG evaluation numbers

With the Space (or a local run) up and `GROQ_API_KEY` set:

```bash
python -m eval.eval_rag --notebook-id <id-of-a-notebook-with-sources>
```

Copy the printed table and `eval/eval_results.json` contents into the
results table and observations in `eval/RAG_EVALUATION.md`, replacing the
illustrative placeholder numbers with your real measurements.

## Submission checklist cross-reference

See `SUBMISSION_CHECKLIST.md` for the full assignment checklist with each
item marked as either already satisfied in this repository, or requiring
one of the five steps above.
