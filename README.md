# DocuRAG — RAG-Based Document Question Answering System Using NLP

DocuRAG is a web app that lets you upload PDF documents and ask questions about them in plain English. It finds the passages that match your question and sends **only those passages** to a Groq-hosted LLM. The model answers from them, and every answer links back to the document and page it came from.

> Stack: HTML/CSS/Vanilla JS · Python + Flask · PyMuPDF · all-MiniLM-L6-v2 (local, via FastEmbed/ONNX) · FAISS · Groq API · Gunicorn · Render Blueprint

---

## 1. Problem Statement

Students and professionals collect large PDFs such as lecture notes, research papers and manuals. Searching them by keyword misses answers that use different wording. General chatbots have their own problems: they don't know what's in your files, and they may **hallucinate** answers that sound right but aren't. We need a system that understands questions semantically and answers **only from the user's own documents**, with verifiable citations.

## 2. Purpose

This project shows the full **Retrieval-Augmented Generation (RAG)** pipeline as a working application. It applies core NLP topics: preprocessing, segmentation, embeddings, semantic similarity, information retrieval, language generation and hallucination control.

## 3. Features

- Upload several PDFs by drag and drop or file browser, with type and size checks
- Live processing stages from the server: *Reading PDF → Extracting text → Creating chunks → Generating embeddings → Building semantic index → Ready*
- Text extraction that keeps page numbers, plus text cleaning
- Sentence-aware chunking with overlap. Chunks never cross page boundaries, so citations are exact.
- **Local** embeddings (no document text is sent to an embedding API)
- FAISS cosine-similarity search across all documents
- **Relevance threshold**: if no chunk is similar enough, the app says so and **does not call Groq**
- Grounded answers from Groq with a strict "context-only" system prompt and inline `[Source n]` citations
- Source panel showing document, page, similarity score and an expandable excerpt
- **Retrieval debug mode**: see every retrieved chunk, its score, and whether it was sent to the LLM
- **Suggested questions** after upload
- **Bounded chat memory**: follow-ups like "Explain it with an example" work, and only the last 3 exchanges are sent
- Document management: remove one document (its vectors and metadata are deleted), clear all, rebuild the index
- Duplicate detection (SHA-256): uploading the same file again skips re-embedding
- Statistics: documents, pages, chunks and index status
- Safe Markdown rendering (HTML is escaped before formatting, which prevents XSS)
- Responsive layout for desktop and mobile, with visible focus states, keyboard support, and loading, empty and error states

## 4. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | HTML5, CSS3 (CSS variables), Vanilla JS | Required; no frameworks |
| Backend | Python 3.11, Flask | REST API and serves the UI |
| Production server | Gunicorn | WSGI server for Render |
| PDF extraction | PyMuPDF (`pymupdf`) | Fast, free, extracts page by page |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` run by **FastEmbed** (ONNX Runtime) | Free, local, 384-dim, small |
| Vector index | FAISS (`IndexIDMap2(IndexFlatIP)`) | Exact cosine search; supports deleting by id |
| LLM | Groq API (default `openai/gpt-oss-20b`, configurable) | Fast; free tier available |

**Why FastEmbed instead of the `sentence-transformers` package?** It runs the **same all-MiniLM-L6-v2 model** through ONNX Runtime instead of PyTorch. PyTorch adds roughly 1–2 GB to the install and several hundred MB of RAM. That can exceed the 512 MB memory of Render's free instance. FastEmbed keeps the same model quality at a fraction of the footprint. Vectors are L2-normalized, so FAISS inner product equals cosine similarity.

## 5. RAG Architecture

```
PDF ──► Extraction (PyMuPDF, per page) ──► Cleaning ──► Sentence-aware Chunking
                                                            │  (doc, page, chunk id, text)
                                                            ▼
                                         Local Embeddings (MiniLM, 384-d, normalized)
                                                            ▼
                                         FAISS index  +  metadata map (id → doc/page/text)

Question ──► (follow-up resolution) ──► Query embedding ──► Top-K cosine search
            ──► threshold filter ──► Context block with [Source n] + doc + page
            ──► Groq LLM (grounding system prompt, bounded history)
            ──► Answer + citations + scores
```

## 6. System Workflow (and where it lives in the code)

| Step | What happens | File |
|---|---|---|
| 1. Upload | Checks the extension **and** the `%PDF-` magic bytes, cleans the filename, saves under a random temp name, starts a background job | `app.py` (`/api/upload`) |
| 2. Extraction | `page.get_text()` for each page; stores the page number | `rag/pdf_processor.py` |
| 3. Cleaning | Normalizes line breaks, removes control characters and ligatures, rejoins hyphenated words, drops page-number lines, keeps paragraphs and punctuation | `rag/pdf_processor.py` |
| 4. Chunking | Splits into sentences, then groups them into ~`CHUNK_SIZE` characters. The overlap is made of whole trailing sentences. Chunks never span pages. | `rag/chunker.py` |
| 5. Embedding | Loads the MiniLM ONNX model **once per process** and embeds in batches | `rag/embeddings.py` |
| 6. Indexing | `add_with_ids` into FAISS, with metadata stored under the same ids | `rag/vector_store.py` |
| 7. Retrieval | Embeds the question (with follow-up resolution) and runs a top-K search | `rag/retriever.py` |
| 8. Context | `[Source n] Document / Page / Content` blocks | `rag/retriever.py` |
| 9. Generation | Grounding system prompt + last 3 exchanges + context → Groq | `rag/generator.py` |
| 10. Response | Answer, sources (doc, page, score, excerpt) and debug retrieval data | `app.py` (`/api/ask`) |

### Chunk size note
`CHUNK_SIZE` is measured in **characters** (default 1000 ≈ 180–220 words, overlap 150). all-MiniLM-L6-v2 truncates input at **256 word-piece tokens**. A 1000-word chunk would be silently cut to about its first 200 words before embedding, so most of its text would never be searchable. About 1000 characters fits the model's window, so the whole chunk is represented by its vector.

### Hallucination control
1. **Retrieval threshold** (`MIN_SIMILARITY`, default 0.25). If no chunk passes, the app returns *"I couldn't find sufficiently relevant information in the uploaded documents…"* and Groq is **never called**.
2. **Grounding prompt.** The model must answer only from the context, must not use outside knowledge, and must reply *"I couldn't find this information in the uploaded documents."* when the context doesn't support an answer.
3. **Low temperature** (0.1).
4. **Honest UI.** "Not found" answers are shown with a warning style and **no source list**, so an unsupported answer never looks document-backed.
5. **Chat history is only used to resolve references** such as "it". Facts must still come from the retrieved context.

## 7. NLP Concepts → Implementation

| NLP concept | Where it appears |
|---|---|
| Text preprocessing / normalization | `clean_text()`: whitespace and line-break normalization, de-hyphenation, artifact removal |
| Text segmentation | Sentence splitting and overlapping chunking in `chunker.py` |
| Embeddings / distributed semantics | MiniLM sentence embeddings (384-d) in `embeddings.py` |
| Semantic similarity | Cosine similarity (inner product of normalized vectors) in FAISS |
| Information retrieval | Top-K dense retrieval + relevance threshold (`retriever.py`) |
| Question understanding | Query embedding; follow-up resolution with pronouns/short questions (`build_search_query`) |
| Context retrieval / construction | `build_context()` with source metadata |
| Language generation | Groq LLM chat completion (`generator.py`) |
| Context-aware generation | Bounded conversation memory (last 3 turns) |
| Hallucination reduction | Threshold gating, grounding prompt, not-found detection |
| Source attribution | `[Source n]` citations; document + page + score in the UI |

## 8. Project Structure

```
NLP-MP/
├── app.py                  # Flask app: routes, upload jobs, error handling
├── rag/
│   ├── __init__.py
│   ├── config.py           # Environment-based configuration
│   ├── pdf_processor.py    # Extraction + cleaning
│   ├── chunker.py          # Sentence-aware overlapping chunking
│   ├── embeddings.py       # Local MiniLM embeddings (loaded once)
│   ├── vector_store.py     # FAISS index + metadata, delete/clear/rebuild
│   ├── retriever.py        # Query building, top-K search, context block
│   ├── generator.py        # Groq call with grounding prompt
│   └── pipeline.py         # Ingestion pipeline orchestration
├── templates/index.html
├── static/css/style.css
├── static/js/app.js
├── uploads/.gitkeep        # Temp PDFs (deleted right after processing)
├── data/.gitkeep           # Saved FAISS index + metadata (best effort)
├── requirements.txt
├── .env.example
├── .gitignore
├── render.yaml
└── README.md
```

## 9. API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | `{"status":"ok","service":"DocuRAG", ...stats}` |
| POST | `/api/upload` | multipart `files` (1..n PDFs) → `202 {job_id}` |
| GET | `/api/jobs/<job_id>` | Processing stage, current file, per-file results |
| GET | `/api/documents` | Documents + stats |
| DELETE | `/api/documents/<id>` | Removes the document's vectors and metadata |
| POST | `/api/clear` | Clears everything |
| POST | `/api/rebuild` | Re-embeds all stored chunks into a fresh index |
| POST | `/api/ask` | `{question, history}` → `{success, answer, grounded, sources, retrieval}` |

Errors always look like `{"success": false, "error": "..."}` with a matching HTTP status (400, 404, 413, 502, 500).

---

## 10. Installation (Windows)

Requirements: **Python 3.11 or newer** (tested with 3.11) and an internet connection for the first run. The embedding model (~90 MB) downloads once and is then cached in `model_cache/`.

```powershell
git clone https://github.com/SohamMedley/NLP-MP.git
cd NLP-MP
```

### Python virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
If PowerShell blocks the script, run this once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Dependencies
```powershell
pip install -r requirements.txt
```

### Groq API key
1. Sign in at <https://console.groq.com> (a free tier is available; limits are set by Groq and may change).
2. Go to **API Keys → Create API Key** and copy it.

### Environment variables
```powershell
Copy-Item .env.example .env
notepad .env
```
Set at least `GROQ_API_KEY=gsk_...`. The key is read only on the server and is never sent to the browser. `.env` is listed in `.gitignore`.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | Required for answer generation |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Any Groq chat model; change it without touching code (e.g. `openai/gpt-oss-120b`); if it is unavailable the app automatically falls back to `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, then `llama-3.1-8b-instant` |
| `CHUNK_SIZE` | `1000` | Chunk length in characters |
| `CHUNK_OVERLAP` | `150` | Overlap in characters (whole sentences) |
| `TOP_K` | `5` | Chunks retrieved per question |
| `MIN_SIMILARITY` | `0.25` | Cosine threshold; below it Groq isn't called |
| `MAX_UPLOAD_MB` | `50` | Maximum size of one upload request |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | FastEmbed model name |
| `EMBEDDING_MODEL_PATH` | *(empty)* | Optional local folder with ONNX model files (offline use) |

### Running locally
```powershell
python app.py
```
Open <http://localhost:5000>. The first start downloads the embedding model, which takes about a minute.

### Production command
Render uses Gunicorn:
```
gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT
```
Gunicorn doesn't run on Windows, so use `python app.py` locally. **Keep `--workers 1`**: the vector index lives in process memory, and several workers would each have their own index. Threads let the UI poll upload progress while embeddings are being generated.

---

## 11. Render Deployment (Blueprint)

1. Push this repository to GitHub.
2. In the Render Dashboard choose **New → Blueprint** and connect the repo. Render reads `render.yaml`.
3. When prompted, enter a value for **`GROQ_API_KEY`**. It is marked `sync: false`, so it is never stored in the repo. You can also add or change it later under **Service → Environment**.
4. Click **Apply**. The build runs `pip install -r requirements.txt && python -m rag.embeddings`. The second command downloads the embedding model during the build, so the first request doesn't have to.
5. Render checks `/api/health`. Once it's live, open the service URL.

To change the LLM model, chunking or threshold, edit the environment variables in the Render dashboard and redeploy. No code changes are needed.

---

## 12. Limitations (be aware)

- **Ephemeral storage on Render Free.** The service's local disk is not persistent. The FAISS index and metadata are saved to `data/` as a convenience, but they are lost on every redeploy, restart or spin-down. After that, **users must re-upload their PDFs.** A persistent disk needs a paid plan, which this project deliberately avoids.
- **Spin-down.** Free web services sleep after a period of inactivity. The next request can take tens of seconds (cold start plus model load).
- **Memory and CPU.** 512 MB RAM and a shared CPU. Large PDFs (hundreds of pages) take a while to embed and may approach memory limits.
- **Shared index.** There are no user accounts. Everyone using the same deployment sees the same document collection. That's fine for a demo or a single user, but not for multi-user privacy.
- **Scanned PDFs.** Pages that are only images have no extractable text (no OCR). They are reported as "not enough extractable text".
- **Retrieval limits.** A single-vector dense retriever can miss exact keyword matches, and broad requests like "Summarize this document" only see the top-K chunks, not the whole file.
- **Groq free tier.** Rate limits and available models are set by Groq and may change. If a model is retired, set `GROQ_MODEL` to a current one.
- **Relevance threshold** is a heuristic. You may need to tune `MIN_SIMILARITY` for your documents.

## 13. Future Enhancements

- Hybrid retrieval (BM25 + dense) and cross-encoder re-ranking
- OCR (Tesseract) for scanned PDFs
- Per-user sessions / isolated indexes
- Streaming answers (Server-Sent Events)
- Map-reduce summarization of whole documents
- Support for DOCX, TXT and HTML
- Evaluation set with retrieval metrics (Recall@K, MRR) and answer faithfulness scores
- Optional persistent storage for self-hosted deployments

## 14. Conclusion

DocuRAG shows how NLP techniques combine into a working question answering system. Documents are cleaned and segmented, embedded into a semantic vector space and searched by meaning. Only the most relevant evidence is passed to a large language model. Grounding comes from three layers: similarity thresholds, strict prompting and visible source attribution. Together they make answers verifiable and reduce hallucination. The project runs on free, open-source components plus the Groq API, and it deploys to Render with a single Blueprint.
