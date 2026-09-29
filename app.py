"""DocuMind: API-free, extractive RAG for private document collections.

Documents are chunked and represented as locally calculated TF-IDF vectors. A
cosine-similarity retriever finds evidence, and the answer step selects the best
sentences from that evidence. Nothing is sent to a third-party model or API.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import secrets
import shutil
import threading
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, session
from pypdf import PdfReader
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {"pdf", "txt", "md", "markdown"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_TEXT_CHARS = 1_500_000
MAX_DOCUMENTS_PER_SESSION = 20
MAX_CHUNKS_PER_SESSION = 4_500
CHUNK_WORDS = 220
CHUNK_OVERLAP = 35
CACHE_OWNERS = 2

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Keep negation words such as "not" and "no": removing them can reverse a fact.
STOP_WORDS = frozenset(
    "a about above after again against all am an and any are as at be because been "
    "before being below between both but by can could did do does doing down during "
    "each few for from further had has have having he her here hers herself him himself "
    "his how i if in into is it its itself just me more most my myself of off on once "
    "only or other our ours ourselves out over own same she should so some such than "
    "that the their theirs them themselves then there these they this those through "
    "to too under until up very was we were what when where which while who whom why "
    "will with would you your yours yourself yourselves"
    .split()
)

def _local_secret_key() -> str:
    """Keep a stable local signing key across Gunicorn workers and restarts."""
    data_dir = Path(os.environ.get("DOCUMIND_DATA_DIR", BASE_DIR / "data")).resolve()
    key_path = data_dir / ".session_key"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        return key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        generated = secrets.token_hex(32)
        try:
            with key_path.open("x", encoding="utf-8") as output:
                output.write(generated)
        except FileExistsError:
            pass
        return key_path.read_text(encoding="utf-8").strip()
    except OSError:
        # The Render blueprint sets SECRET_KEY explicitly; this is only a
        # fallback for read-only local filesystems.
        return secrets.token_hex(32)


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or _local_secret_key(),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")),
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    DATA_DIR=Path(os.environ.get("DOCUMIND_DATA_DIR", BASE_DIR / "data")).resolve(),
    JSON_AS_ASCII=False,
)
# Render terminates TLS before forwarding requests to Gunicorn.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("documind")

# Each index contains only one owner's chunks. The bounded cache keeps repeated
# questions quick without allowing anonymous sessions to grow memory forever.
_search_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_search_cache_lock = threading.RLock()


def _current_owner_id() -> str:
    """Get or create the opaque ID used to isolate each browser's library."""
    owner_id = session.get("owner_id")
    if not isinstance(owner_id, str) or not re.fullmatch(r"[a-f0-9]{32}", owner_id):
        owner_id = secrets.token_hex(16)
        session["owner_id"] = owner_id
    session.permanent = True
    return owner_id


def _owner_dir(owner_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", owner_id):
        abort(400, description="Invalid library session")
    return Path(app.config["DATA_DIR"]) / owner_id


def _document_dir(owner_id: str, doc_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{16}", doc_id):
        abort(404, description="Document not found")
    return _owner_dir(owner_id) / doc_id


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _list_documents(owner_id: str) -> list[dict[str, Any]]:
    owner_path = _owner_dir(owner_id)
    if not owner_path.exists():
        return []

    documents: list[dict[str, Any]] = []
    for folder in owner_path.iterdir():
        metadata_path = folder / "metadata.json"
        if not folder.is_dir() or not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as source:
                metadata = json.load(source)
            if metadata.get("doc_id") == folder.name:
                documents.append(metadata)
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable document metadata at %s", metadata_path)

    return sorted(documents, key=lambda item: item.get("created_at", ""), reverse=True)


def _extract_pdf_pages(raw_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(raw_bytes), strict=False)
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:  # pypdf can raise different errors by version.
            raise ValueError("This PDF is password-protected. Please upload an unlocked PDF.") from exc
        if not decrypted:
            raise ValueError("This PDF is password-protected. Please upload an unlocked PDF.")

    pages: list[tuple[int, str]] = []
    remaining_chars = MAX_TEXT_CHARS
    for page_number, page in enumerate(reader.pages, start=1):
        if remaining_chars <= 0:
            break
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.info("Could not extract text from PDF page %s: %s", page_number, exc)
            continue
        text = text[:remaining_chars]
        remaining_chars -= len(text)
        if text.strip():
            pages.append((page_number, text))
    return pages


def _extract_file_text(filename: str, raw_bytes: bytes) -> tuple[list[tuple[int | None, str]], bool]:
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a PDF, TXT, or Markdown file.")

    if extension == "pdf":
        pages = _extract_pdf_pages(raw_bytes)
        return pages, sum(len(text) for _, text in pages) >= MAX_TEXT_CHARS

    text = raw_bytes.decode("utf-8-sig", errors="replace")
    truncated = len(text) > MAX_TEXT_CHARS
    text = text[:MAX_TEXT_CHARS]
    return [(None, text)], truncated


def chunk_text(text: str, page: int | None = None) -> list[dict[str, Any]]:
    """Create modest, overlapping word chunks for retrieval and citations."""
    words = re.sub(r"\s+", " ", text).strip().split()
    if not words:
        return []

    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(words), step):
        excerpt = " ".join(words[start : start + CHUNK_WORDS]).strip()
        if excerpt:
            chunks.append({"text": excerpt, "page": page})
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def _features(text: str) -> list[str]:
    """Build a small sparse local embedding from unigrams and adjacent bigrams."""
    words = [word.lower() for word in TOKEN_RE.findall(text)]
    terms = [word for word in words if word not in STOP_WORDS and (len(word) > 1 or word.isdigit())]
    if not terms:
        return []
    features = list(terms)
    features.extend(f"{left}\x1f{right}" for left, right in zip(terms, terms[1:]))
    return features


def _weighted_vector(term_counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    weighted = {
        term: (1.0 + math.log(count)) * idf[term]
        for term, count in term_counts.items()
        if term in idf
    }
    norm = math.sqrt(sum(weight * weight for weight in weighted.values()))
    if not norm:
        return {}
    return {term: weight / norm for term, weight in weighted.items()}


def _make_search_index(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts_by_record: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()

    for record in records:
        counts = Counter(_features(record["text"]))
        counts_by_record.append(counts)
        document_frequency.update(counts.keys())

    document_count = max(len(records), 1)
    idf = {
        term: 1.0 + math.log((document_count + 1.0) / (frequency + 1.0))
        for term, frequency in document_frequency.items()
    }

    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record_index, counts in enumerate(counts_by_record):
        for term, weight in _weighted_vector(counts, idf).items():
            postings[term].append((record_index, weight))

    return {"records": records, "idf": idf, "postings": dict(postings)}


def _corpus_signature(owner_id: str, documents: list[dict[str, Any]]) -> tuple[Any, ...]:
    signature = []
    for document in documents:
        chunks_path = _document_dir(owner_id, document["doc_id"]) / "chunks.json"
        try:
            stat = chunks_path.stat()
            signature.append((document["doc_id"], stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append((document["doc_id"], 0, 0))
    return tuple(signature)


def _get_search_index(owner_id: str) -> dict[str, Any]:
    documents = _list_documents(owner_id)
    signature = _corpus_signature(owner_id, documents)

    with _search_cache_lock:
        cached = _search_cache.get(owner_id)
        if cached is not None and cached["signature"] == signature:
            _search_cache.move_to_end(owner_id)
            return cached["index"]

        records: list[dict[str, Any]] = []
        for document in documents:
            chunks_path = _document_dir(owner_id, document["doc_id"]) / "chunks.json"
            try:
                with chunks_path.open("r", encoding="utf-8") as source:
                    chunks = json.load(source)
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable chunks for %s", document["doc_id"])
                continue

            for chunk_number, chunk in enumerate(chunks, start=1):
                text = chunk.get("text", "").strip()
                if not text:
                    continue
                records.append(
                    {
                        "text": text,
                        "doc_id": document["doc_id"],
                        "filename": document["filename"],
                        "page": chunk.get("page"),
                        "chunk_index": chunk_number,
                    }
                )

        index = _make_search_index(records)
        _search_cache[owner_id] = {"signature": signature, "index": index}
        _search_cache.move_to_end(owner_id)
        while len(_search_cache) > CACHE_OWNERS:
            _search_cache.popitem(last=False)
        return index


def _retrieve(query: str, index: dict[str, Any], doc_id: str | None = None, top_k: int = 5) -> tuple[list[dict[str, Any]], dict[str, float]]:
    query_vector = _weighted_vector(Counter(_features(query)), index["idf"])
    if not query_vector or not index["records"]:
        return [], query_vector

    scores = [0.0] * len(index["records"])
    for term, query_weight in query_vector.items():
        for record_index, document_weight in index["postings"].get(term, ()):
            record = index["records"][record_index]
            if doc_id is None or record["doc_id"] == doc_id:
                scores[record_index] += query_weight * document_weight

    ranked = sorted(
        (record_index for record_index, score in enumerate(scores) if score > 0),
        key=lambda record_index: (-scores[record_index], record_index),
    )[:top_k]
    hits = [dict(index["records"][record_index], score=scores[record_index]) for record_index in ranked]
    return hits, query_vector


def _generate_extractive_answer(
    query: str,
    hits: list[dict[str, Any]],
    index: dict[str, Any],
    query_vector: dict[str, float],
) -> str:
    """Select source sentences instead of inventing unsupported text."""
    if not hits:
        return (
            "I couldn’t find a matching passage in this library. Try rephrasing the question "
            "or include a keyword that appears in the documents."
        )

    sentences: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    for hit_number, hit in enumerate(hits):
        candidates = [sentence.strip() for sentence in SENTENCE_RE.split(hit["text"]) if sentence.strip()]
        if not candidates:
            candidates = [hit["text"]]
        for sentence in candidates:
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            sentence_vector = _weighted_vector(Counter(_features(sentence)), index["idf"])
            relevance = sum(weight * query_vector.get(term, 0.0) for term, weight in sentence_vector.items())
            # Keep a useful fallback sentence from a clearly relevant retrieved chunk.
            relevance += hit["score"] * (0.08 if hit_number == 0 else 0.03)
            sentences.append((relevance, hit_number, sentence))

    sentences.sort(key=lambda item: (-item[0], item[1]))
    is_summary = any(term in query.lower() for term in ("summary", "summarize", "overview", "key points", "main ideas"))
    answer_limit = 4 if is_summary else 3
    selected: list[str] = []
    total_chars = 0
    for relevance, _hit_number, sentence in sentences:
        if relevance <= 0 or len(selected) >= answer_limit:
            continue
        if total_chars + len(sentence) > 850 and selected:
            continue
        selected.append(sentence)
        total_chars += len(sentence)

    if not selected:
        selected = [hits[0]["text"][:850].rstrip()]
    return "Here’s what I found in your documents:\n\n" + " ".join(selected)


@app.after_request
def set_response_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error: RequestEntityTooLarge):
    return jsonify({"error": f"That file is too large. The limit is {MAX_UPLOAD_MB} MB."}), 413


@app.route("/")
def home():
    _current_owner_id()
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "DocuMind"})


@app.route("/api/documents", methods=["GET"])
def get_documents():
    return jsonify({"documents": _list_documents(_current_owner_id())})


@app.route("/api/upload", methods=["POST"])
def upload_document():
    owner_id = _current_owner_id()
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a file to upload."}), 400

    filename = secure_filename(uploaded.filename)
    if not filename or Path(filename).suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": "Upload a PDF, TXT, or Markdown file."}), 400

    raw_bytes = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if not raw_bytes:
        return jsonify({"error": "The selected file is empty."}), 400
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"That file is too large. The limit is {MAX_UPLOAD_MB} MB."}), 413

    try:
        pages, truncated = _extract_file_text(filename, raw_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.info("Document extraction failed for %s: %s", filename, exc)
        return jsonify({"error": "We couldn’t read this file. Check that it is a valid, text-based PDF or UTF-8 text file."}), 400

    chunks: list[dict[str, Any]] = []
    for page_number, page_text in pages:
        chunks.extend(chunk_text(page_text, page_number))
    if not chunks:
        return jsonify({"error": "No readable text was found. Scanned/image-only PDFs are not supported."}), 400

    documents = _list_documents(owner_id)
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    destination = _document_dir(owner_id, digest)
    existing_metadata = destination / "metadata.json"
    if existing_metadata.is_file():
        try:
            with existing_metadata.open("r", encoding="utf-8") as source:
                metadata = json.load(source)
            return jsonify(
                {"message": "This document is already in your library.", "document": metadata, "duplicate": True}
            )
        except (OSError, json.JSONDecodeError):
            shutil.rmtree(destination, ignore_errors=True)

    if len(documents) >= MAX_DOCUMENTS_PER_SESSION:
        return jsonify({"error": f"Your library is at its limit of {MAX_DOCUMENTS_PER_SESSION} documents."}), 400
    existing_chunks = sum(int(document.get("chunks", 0)) for document in documents)
    if existing_chunks + len(chunks) > MAX_CHUNKS_PER_SESSION:
        return jsonify({"error": "This file would exceed your library’s processing limit. Remove a document or upload a smaller file."}), 400

    metadata = {
        "doc_id": digest,
        "filename": filename,
        "chunks": len(chunks),
        "words": sum(len(chunk["text"].split()) for chunk in chunks),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "truncated": truncated,
    }
    try:
        destination.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(destination / "chunks.json", chunks)
        _atomic_write_json(destination / "metadata.json", metadata)
    except OSError as exc:
        logger.exception("Could not save processed document")
        shutil.rmtree(destination, ignore_errors=True)
        return jsonify({"error": "Couldn’t save this document. Check the available storage and try again."}), 500

    with _search_cache_lock:
        _search_cache.pop(owner_id, None)

    return jsonify({"message": "Document processed and indexed locally.", "document": metadata, "duplicate": False}), 201


@app.route("/api/ask", methods=["POST"])
def ask_question():
    owner_id = _current_owner_id()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send your question as JSON."}), 400

    query = str(payload.get("query", "")).strip()
    doc_id = payload.get("doc_id") or None
    if not query:
        return jsonify({"error": "Type a question first."}), 400
    if len(query) > 2_000:
        return jsonify({"error": "Questions must be 2,000 characters or fewer."}), 400
    if doc_id is not None and (not isinstance(doc_id, str) or not re.fullmatch(r"[a-f0-9]{16}", doc_id)):
        return jsonify({"error": "That document is not available in this library."}), 404

    if doc_id and not any(document["doc_id"] == doc_id for document in _list_documents(owner_id)):
        return jsonify({"error": "That document is not available in this library."}), 404

    index = _get_search_index(owner_id)
    if not index["records"]:
        return jsonify({"error": "Upload a document first, then ask away."}), 400

    try:
        top_k = max(1, min(int(payload.get("top_k", 5)), 8))
    except (TypeError, ValueError):
        top_k = 5
    hits, query_vector = _retrieve(query, index, doc_id=doc_id, top_k=top_k)
    answer = _generate_extractive_answer(query, hits, index, query_vector)

    sources = [
        {
            "filename": hit["filename"],
            "page": hit["page"],
            "chunk_index": hit["chunk_index"],
            "score": round(hit["score"], 4),
            "text": hit["text"][:360] + ("…" if len(hit["text"]) > 360 else ""),
        }
        for hit in hits
    ]
    return jsonify({"query": query, "answer": answer, "sources": sources, "mode": "local-extractive-rag"})


@app.route("/api/delete/<doc_id>", methods=["DELETE"])
def delete_document(doc_id: str):
    owner_id = _current_owner_id()
    path = _document_dir(owner_id, doc_id)
    if not path.is_dir():
        return jsonify({"error": "Document not found."}), 404
    shutil.rmtree(path)
    with _search_cache_lock:
        _search_cache.pop(owner_id, None)
    return jsonify({"message": "Document removed from your library."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
