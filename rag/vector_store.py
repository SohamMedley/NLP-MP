"""Step 6: FAISS vector index + metadata store with per-document deletion."""
import json
import threading

import faiss
import numpy as np

from . import config

INDEX_PATH = config.DATA_DIR / "index.faiss"
META_PATH = config.DATA_DIR / "metadata.json"


class VectorStore:
    """IndexFlatIP over normalized vectors (= exact cosine search), wrapped in
    IndexIDMap2 so every vector has a stable integer id that maps to chunk metadata
    and can be removed when its document is deleted."""

    def __init__(self):
        self._lock = threading.RLock()
        self.index = None
        self.chunks = {}      # vector id -> chunk metadata (document, page, text...)
        self.documents = {}   # doc_id -> document info
        self.next_id = 0
        self._load()

    # ---------- persistence (best effort; disk is ephemeral on Render free) ----------
    def _load(self):
        try:
            if INDEX_PATH.exists() and META_PATH.exists():
                meta = json.loads(META_PATH.read_text(encoding="utf-8"))
                self.index = faiss.read_index(str(INDEX_PATH))
                self.chunks = {int(k): v for k, v in meta["chunks"].items()}
                self.documents = meta["documents"]
                self.next_id = meta["next_id"]
        except Exception:
            self.index, self.chunks, self.documents, self.next_id = None, {}, {}, 0

    def _save(self):
        if self.index is None or not self.chunks:
            INDEX_PATH.unlink(missing_ok=True)
            META_PATH.unlink(missing_ok=True)
            return
        faiss.write_index(self.index, str(INDEX_PATH))
        META_PATH.write_text(json.dumps({"chunks": self.chunks, "documents": self.documents,
                                         "next_id": self.next_id}), encoding="utf-8")

    # ---------- mutations ----------
    def has_document(self, doc_id):
        with self._lock:
            return doc_id in self.documents

    def add_document(self, doc_info: dict, chunks: list[dict], vectors: np.ndarray):
        with self._lock:
            if self.index is None:
                self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
            ids = np.arange(self.next_id, self.next_id + len(chunks), dtype="int64")
            self.index.add_with_ids(vectors, ids)
            for vector_id, chunk in zip(ids.tolist(), chunks):
                self.chunks[vector_id] = chunk
            self.next_id += len(chunks)
            self.documents[doc_info["id"]] = doc_info
            self._save()

    def remove_document(self, doc_id) -> bool:
        """Delete a document's vectors AND metadata so no stale citations remain."""
        with self._lock:
            if doc_id not in self.documents:
                return False
            ids = [vid for vid, c in self.chunks.items() if c["doc_id"] == doc_id]
            if ids and self.index is not None:
                self.index.remove_ids(np.array(ids, dtype="int64"))
            for vid in ids:
                del self.chunks[vid]
            del self.documents[doc_id]
            if not self.chunks:
                self.index = None
            self._save()
            return True

    def clear(self):
        with self._lock:
            self.index, self.chunks, self.documents, self.next_id = None, {}, {}, 0
            self._save()

    def rebuild(self, vectors_for):
        """Re-create the FAISS index from stored chunk text (compacts ids)."""
        with self._lock:
            items = sorted(self.chunks.items())
            if not items:
                self.clear()
                return 0
            texts = [c["text"] for _, c in items]
            vectors = vectors_for(texts)
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
            self.index.add_with_ids(vectors, np.arange(len(items), dtype="int64"))
            self.chunks = {i: c for i, (_, c) in enumerate(items)}
            self.next_id = len(items)
            self._save()
            return len(items)

    # ---------- queries ----------
    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            k = min(k, self.index.ntotal)
            scores, ids = self.index.search(query_vector.reshape(1, -1).astype("float32"), k)
            results = []
            for score, vid in zip(scores[0].tolist(), ids[0].tolist()):
                if vid == -1 or vid not in self.chunks:
                    continue
                results.append({**self.chunks[vid], "score": round(float(score), 4)})
            return results

    def list_documents(self):
        with self._lock:
            return sorted(self.documents.values(), key=lambda d: d["uploaded_at"])

    def stats(self):
        with self._lock:
            return {
                "documents": len(self.documents),
                "pages": sum(d["pages"] for d in self.documents.values()),
                "chunks": len(self.chunks),
                "index_ready": self.index is not None and self.index.ntotal > 0,
            }


store = VectorStore()
