# DocuMind — RAG Document Q&A

A small, deployable document question-answering app built with Flask, HTML, CSS and JavaScript. Upload PDFs, plain-text notes or Markdown files, then ask questions in natural language and inspect the passages used to answer.

## API-free by design

- **No external AI service, API key, or model download.** Processing happens inside the Flask app.
- Documents are split into overlapping passages and represented as sparse, local TF-IDF vectors (words and adjacent phrases). Inverse-document-frequency weighting and cosine similarity retrieve the closest passages.
- Because there is no generative model or API, the answer step is **extractive**: it selects relevant sentences from the retrieved evidence instead of inventing a free-form response. Each response includes expandable source passages, filenames, page numbers when available, and match scores.
- Uploaded originals are not retained. The app stores extracted passage text and document metadata so it can answer follow-up questions.
- A signed, anonymous browser session isolates one visitor's library from other visitors. There is no account or team-sharing feature.

TF-IDF is a lightweight statistical text representation, not a neural language embedding. It is fast and self-contained, but may miss some paraphrases that a large language model would understand. Scanned/image-only PDFs are not OCR'd.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open <http://localhost:5000>. The first visit creates a local `data/` directory for the session-signing key and processed documents; it is ignored by Git. The app makes no outbound requests for fonts, models, embeddings, or answers.

Optional configuration:

| Variable | Default | Description |
| --- | --- | --- |
| `SECRET_KEY` | generated and stored in `data/.session_key` | Stable Flask cookie-signing key. Set this in production. |
| `DOCUMIND_DATA_DIR` | `./data` | Directory for the signing key and per-session document indexes. |
| `MAX_UPLOAD_MB` | `25` | Maximum individual upload size. The frontend displays the default limit. |
| `PORT` | `5000` | Listening port (Render supplies this automatically). |

## Render Blueprint deploy

`render.yaml` defines a Python web service, a health check, one Gunicorn worker with four threads, a generated Flask secret, and a 1 GB persistent disk mounted at `/var/data`. In Render, create a new **Blueprint** from this repository and review the plan before confirming deployment.

The blueprint uses Render's **Starter** web-service plan because persistent disks require a paid service. The disk keeps processed documents and session cookies working across deploys/restarts. If you intentionally change the service to a free plan, remove the `disk` block and point `DOCUMIND_DATA_DIR` to a writable temporary path (for example `/tmp/documind-data`); free-instance filesystems are ephemeral, so uploaded libraries will not survive a restart or redeploy. Check Render's current pricing and limits before deploying.

Render sets the service port; Gunicorn binds to it automatically. Health is checked at `/api/health`.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Supported documents and limits

- `.pdf`, `.txt`, `.md` and `.markdown`
- Text PDFs only (no OCR); password-locked PDFs must be unlocked first.
- 25 MB per file by default, 20 documents and 4,500 passages per browser library.
- Text extraction is capped per file to keep processing responsive and memory use bounded.

## Project structure

```text
app.py              Flask routes, session isolation, ingestion, TF-IDF retrieval and extractive answers
static/app.js       Upload, document library, chat, sources and theme interactions
static/style.css    Responsive interface, themes, animation and reduced-motion support
templates/index.html
render.yaml         Render Blueprint configuration
requirements.txt    Minimal runtime dependencies
```
