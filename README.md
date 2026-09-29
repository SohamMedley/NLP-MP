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

`render.yaml` is configured for Render's **free** Python web-service plan, with a health check, one Gunicorn worker/four threads, a generated Flask secret, and temporary storage at `/tmp/documind-data`. It does not request a persistent disk or a paid instance. Create a new **Blueprint** from this repository and confirm that the service plan is shown as Free before deploying; no payment card should be needed for this configuration.

Free compute has trade-offs: the instance can spin down after inactivity, so the first visit after idle time may be slower. Its filesystem is ephemeral, so uploaded document libraries and extracted passages can disappear after a restart, spin-down, or redeploy. Re-upload documents when that happens. Do not add a persistent disk or change the plan to Starter if keeping the deployment strictly cost-free is required.

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
