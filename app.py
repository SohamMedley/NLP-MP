"""DocuRAG - Flask entry point (API routes + frontend)."""
import logging
import re
import os
import tempfile
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from rag import config
from rag.embeddings import embed_texts, get_model
from rag.extractive import extractive_answer
from rag.generator import NOT_FOUND_MESSAGE, GenerationError, active_model, generate_answer
from rag.pipeline import ingest_document
from rag.retriever import build_context, is_broad_question, retrieve
from rag.vector_store import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("docurag")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

MAX_QUESTION_CHARS = 2000
LOW_RELEVANCE_MESSAGE = ("I couldn't find sufficiently relevant information in the uploaded "
                         "documents to answer this question.")

# Background ingestion jobs so the UI can poll real processing stages.
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def error(message, status):
    return jsonify({"success": False, "error": message}), status


def _update_job(job_id, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)


def _prune_jobs(max_age=3600):
    cutoff = time.time() - max_age
    with jobs_lock:
        for jid in [j for j, v in jobs.items() if v["done"] and v["created"] < cutoff]:
            del jobs[jid]


def _run_ingestion(job_id, saved_files):
    results = []
    for position, (temp_path, filename) in enumerate(saved_files, start=1):
        _update_job(job_id, current_file=filename, file_index=position)
        try:
            result = ingest_document(temp_path, filename,
                                on_stage=lambda stage: _update_job(job_id, stage=stage))
        except ValueError as exc:
            result = {"filename": filename, "status": "error", "message": str(exc)}
        except Exception:
            logger.exception("Failed to process %s", filename)
            result = {"filename": filename, "status": "error",
                      "message": "Unexpected error while processing this file."}
        finally:
            # Uploaded PDFs are only needed during processing.
            try:
                os.remove(temp_path)
            except OSError:
                pass
        results.append(result)
        _update_job(job_id, results=list(results))
    _update_job(job_id, done=True, stage="Ready", stats=store.stats())


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    return render_template("index.html", max_upload_mb=config.MAX_UPLOAD_MB)


# ------------------------------------------------------------------ API
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "DocuRAG",
                    "groq_configured": bool(config.groq_api_key()),
                    "llm_provider": "Groq", "model": active_model(), "embedding_model": config.EMBEDDING_MODEL,
                    **store.stats()})


@app.post("/api/upload")
def upload():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return error("Please choose at least one PDF or TXT file.", 400)

    saved, rejected = [], []
    for file in files:
        filename = secure_filename(file.filename) or "document"
        extension = os.path.splitext(filename)[1].lower()
        if extension not in (".pdf", ".txt"):
            rejected.append({"filename": file.filename, "status": "error",
                             "message": "Only PDF and TXT files are supported."})
            continue
        # Check the content, not just the extension.
        head = file.stream.read(4096)
        file.stream.seek(0)
        if extension == ".pdf" and not head.startswith(b"%PDF-"):
            rejected.append({"filename": filename, "status": "error",
                             "message": "This file is not a valid PDF."})
            continue
        if extension == ".txt" and b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
            rejected.append({"filename": filename, "status": "error",
                             "message": "This file is not a plain-text document."})
            continue
        # Random temp name inside uploads/ - user input never becomes a path.
        fd, temp_path = tempfile.mkstemp(suffix=extension, dir=config.UPLOAD_DIR)
        with os.fdopen(fd, "wb") as out:
            file.save(out)
        saved.append((temp_path, filename))

    if not saved:
        return jsonify({"success": False, "error": rejected[0]["message"],
                        "results": rejected}), 400

    _prune_jobs()
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "done": False, "stage": "Queued", "current_file": None,
                        "file_index": 0, "total_files": len(saved), "results": [],
                        "rejected": rejected, "created": time.time()}
    threading.Thread(target=_run_ingestion, args=(job_id, saved), daemon=True).start()
    return jsonify({"success": True, "job_id": job_id, "rejected": rejected}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return error("Upload job not found.", 404)
        return jsonify({"success": True, **job})


@app.get("/api/documents")
def documents():
    return jsonify({"success": True, "documents": store.list_documents(), "stats": store.stats()})


@app.delete("/api/documents/<doc_id>")
def delete_document(doc_id):
    if not store.remove_document(doc_id):
        return error("Document not found.", 404)
    return jsonify({"success": True, "stats": store.stats()})


@app.post("/api/clear")
def clear():
    store.clear()
    return jsonify({"success": True, "stats": store.stats()})


@app.post("/api/rebuild")
def rebuild():
    count = store.rebuild(embed_texts)
    return jsonify({"success": True, "chunks": count, "stats": store.stats()})


@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    # "llm" = generative RAG via Groq, "local" = extractive NLP in Python (no API).
    mode = "local" if payload.get("mode") == "local" else "llm"

    if not question:
        return error("Please enter a question.", 400)
    if len(question) > MAX_QUESTION_CHARS:
        return error(f"Questions are limited to {MAX_QUESTION_CHARS} characters.", 400)
    if not store.stats()["index_ready"]:
        return error("Please upload at least one PDF or TXT file before asking a question.", 400)

    relevant, candidates, search_query = retrieve(question, history)
    retrieval = [{"document": c["document"], "page": c["page"], "score": c["score"],
                  "unit": c.get("unit", "Page"),
                  "text": c["text"], "used": any(c["doc_id"] == r["doc_id"] and c["chunk_index"] == r["chunk_index"] for r in relevant)} for c in candidates]

    # Hallucination control #1: skip the LLM entirely when nothing is relevant enough.
    if not relevant:
        return jsonify({"success": True, "answer": LOW_RELEVANCE_MESSAGE, "grounded": False,
                        "status": "not_found", "checked": [], "mode": mode,
                        "sources": [], "retrieval": retrieval, "search_query": search_query})

    sources = [{"number": i, "document": c["document"], "page": c["page"],
                "unit": c.get("unit", "Page"),
                "score": c["score"], "excerpt": c["text"]}
               for i, c in enumerate(relevant, start=1)]

    if mode == "local":
        answer, found, cited_chunks = extractive_answer(question, relevant,
                                                        broad=is_broad_question(question))
        if not found:
            return jsonify({"success": True, "answer": LOW_RELEVANCE_MESSAGE, "status": "not_found",
                            "grounded": False, "sources": [], "checked": sources, "mode": mode,
                            "retrieval": retrieval, "search_query": search_query})
        scores = {(c["doc_id"], c["chunk_index"]): c["score"] for c in candidates}
        local_sources = [{"number": i, "document": c["document"], "page": c["page"],
                          "unit": c.get("unit", "Page"),
                          "score": scores.get((c["doc_id"], c["chunk_index"]), 0.0), "excerpt": c["text"]}
                         for i, c in enumerate(cited_chunks, start=1)]
        return jsonify({"success": True, "answer": answer, "status": "grounded", "grounded": True,
                        "sources": local_sources, "checked": [], "mode": mode,
                        "retrieval": retrieval, "search_query": search_query})

    try:
        answer = generate_answer(question, build_context(relevant), history,
                                 broad=is_broad_question(question))
    except GenerationError as exc:
        return jsonify({"success": False, "error": str(exc), "retrieval": retrieval}), 502

    # Grounding status: "grounded", "partial" (not-found sentence + useful cited detail),
    # or "not_found" (the model only said the answer is not in the documents).
    mentions_not_found = NOT_FOUND_MESSAGE.lower().rstrip(".") in answer.lower()
    cites = bool(re.search(r"source\s*\d", answer, re.I))
    if not mentions_not_found:
        status = "grounded"
    elif cites or len(answer) > len(NOT_FOUND_MESSAGE) + 120:
        status = "partial"
    else:
        status = "not_found"
    return jsonify({"success": True, "answer": answer, "status": status, "mode": mode,
                    "grounded": status != "not_found",
                    "sources": [] if status == "not_found" else sources,
                    "checked": sources if status == "not_found" else [],
                    "retrieval": retrieval, "search_query": search_query})


# ------------------------------------------------------------------ errors
@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return error(f"Upload too large. The limit is {config.MAX_UPLOAD_MB} MB per request.", 413)


@app.errorhandler(HTTPException)
def http_error(exc):
    if request.path.startswith("/api/"):
        return error(exc.description or exc.name, exc.code)
    return exc


@app.errorhandler(Exception)
def unexpected(exc):
    logger.exception("Unhandled error: %s", exc)
    return error("Something went wrong on the server. Please try again.", 500)


# Load the embedding model once at startup (not per request).
try:
    get_model()
    logger.info("Embedding model loaded: %s", config.EMBEDDING_MODEL)
except Exception:
    logger.exception("Could not load embedding model at startup; will retry on first use.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
