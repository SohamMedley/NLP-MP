"""Ingestion pipeline: PDF -> extraction -> cleaning -> chunking -> embeddings -> index."""
import hashlib
import time

from . import config
from .chunker import chunk_document
from .embeddings import embed_texts
from .pdf_processor import extract_pages
from .vector_store import store


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdf(path, filename: str, on_stage=lambda stage: None) -> dict:
    """Process one PDF and add it to the vector store. Returns a result summary."""
    on_stage("Reading PDF...")
    doc_id = file_sha256(path)[:16]
    if store.has_document(doc_id):
        # Same bytes already indexed this session: skip re-embedding.
        return {"filename": filename, "status": "duplicate", "doc_id": doc_id,
                "message": "Already indexed - skipped."}

    on_stage("Extracting text...")
    pages, total_pages = extract_pages(str(path))

    on_stage("Creating chunks...")
    chunks = chunk_document(pages, doc_id, filename, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("This PDF does not contain enough extractable text.")

    on_stage("Generating embeddings...")
    vectors = embed_texts([c["text"] for c in chunks])

    on_stage("Building semantic index...")
    doc_info = {
        "id": doc_id,
        "filename": filename,
        "pages": total_pages,
        "text_pages": len(pages),
        "chunks": len(chunks),
        "characters": sum(len(p["text"]) for p in pages),
        "uploaded_at": time.time(),
    }
    store.add_document(doc_info, chunks, vectors)
    return {"filename": filename, "status": "indexed", "doc_id": doc_id,
            "pages": total_pages, "chunks": len(chunks)}
