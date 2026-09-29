import io
import tempfile
import unittest
from pathlib import Path

from app import app, _make_search_index, _retrieve, chunk_text, _search_cache, _search_cache_lock


class RagCoreTests(unittest.TestCase):
    def test_chunk_text_overlaps_adjacent_passages(self):
        words = [f"word{number}" for number in range(400)]
        chunks = chunk_text(" ".join(words), page=3)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page"], 3)
        first = chunks[0]["text"].split()
        second = chunks[1]["text"].split()
        self.assertEqual(first[-35:], second[:35])

    def test_local_tfidf_retrieves_the_relevant_passage(self):
        records = [
            {"text": "The rechargeable battery reaches 18 degrees during normal use.", "doc_id": "a", "filename": "manual.txt", "page": None, "chunk_index": 1},
            {"text": "The coastal railway connects the eastern and western stations.", "doc_id": "b", "filename": "travel.txt", "page": None, "chunk_index": 1},
        ]
        index = _make_search_index(records)
        hits, _ = _retrieve("What temperature does the battery reach?", index)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["filename"], "manual.txt")
        self.assertGreater(hits[0]["score"], 0)


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_data_dir = app.config["DATA_DIR"]
        self.old_secret = app.config["SECRET_KEY"]
        app.config.update(TESTING=True, DATA_DIR=Path(self.temp_dir.name), SECRET_KEY="test-secret-key")
        with _search_cache_lock:
            _search_cache.clear()
        self.client = app.test_client()
        self.client.get("/")  # Establish the anonymous browser library session.

    def tearDown(self):
        with _search_cache_lock:
            _search_cache.clear()
        app.config.update(DATA_DIR=self.old_data_dir, SECRET_KEY=self.old_secret, TESTING=False)
        self.temp_dir.cleanup()

    def upload_text(self, client=None, filename="research-notes.txt"):
        client = client or self.client
        body = (
            "The rechargeable battery reaches 18 degrees during normal use. "
            "The battery health indicator turns blue while charging. "
            "The coastal railway connects the eastern and western stations."
        )
        return client.post(
            "/api/upload",
            data={"file": (io.BytesIO(body.encode("utf-8")), filename)},
            content_type="multipart/form-data",
        )

    def test_home_and_health_are_available(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["status"], "ok")

    def test_upload_retrieve_sources_and_delete(self):
        uploaded = self.upload_text()
        self.assertEqual(uploaded.status_code, 201)
        document = uploaded.get_json()["document"]
        self.assertEqual(document["filename"], "research-notes.txt")

        answer = self.client.post(
            "/api/ask",
            json={"query": "What temperature does the battery reach?", "doc_id": document["doc_id"]},
        )
        self.assertEqual(answer.status_code, 200)
        result = answer.get_json()
        self.assertIn("18 degrees", result["answer"])
        self.assertEqual(result["sources"][0]["filename"], "research-notes.txt")

        removed = self.client.delete(f"/api/delete/{document['doc_id']}")
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(self.client.get("/api/documents").get_json()["documents"], [])

    def test_library_is_isolated_between_browser_sessions(self):
        uploaded = self.upload_text()
        document_id = uploaded.get_json()["document"]["doc_id"]
        other_client = app.test_client()
        self.assertEqual(other_client.get("/api/documents").get_json()["documents"], [])
        forbidden = other_client.post(
            "/api/ask", json={"query": "battery temperature", "doc_id": document_id}
        )
        self.assertEqual(forbidden.status_code, 404)

    def test_unsupported_file_is_rejected(self):
        response = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"hello"), "notes.docx")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    def test_unknown_words_do_not_return_unrelated_passages(self):
        self.upload_text()
        response = self.client.post("/api/ask", json={"query": "intergalactic quantum telescope"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["sources"], [])
        self.assertIn("couldn’t find", response.get_json()["answer"])


if __name__ == "__main__":
    unittest.main()
