"""Step 5: local sentence embeddings (all-MiniLM-L6-v2 via FastEmbed / ONNX Runtime).

FastEmbed runs the *same* sentence-transformers model as the `sentence-transformers`
library, but through ONNX Runtime instead of PyTorch. That cuts install size by
~1.5 GB and RAM by several hundred MB, which matters on Render's 512 MB free tier.
No text ever leaves the server for embedding.
"""
import threading

import numpy as np

from . import config

_model = None
_lock = threading.Lock()


def get_model():
    """Load the model once per process and reuse it for every request."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding
                kwargs = {"cache_dir": config.MODEL_CACHE_DIR, "threads": 1}
                if config.EMBEDDING_MODEL_PATH:
                    # Optional offline mode: load ONNX files from a local folder.
                    kwargs["specific_model_path"] = config.EMBEDDING_MODEL_PATH
                _model = TextEmbedding(model_name=config.EMBEDDING_MODEL, **kwargs)
    return _model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


def embed_texts(texts: list[str], batch_size: int = 4) -> np.ndarray:
    """Return L2-normalized float32 vectors so inner product == cosine similarity.

    Small batches keep peak memory low (~260 MB total), which matters on
    Render's 512 MB free instance; speed is similar on a single CPU thread."""
    vectors = np.array(list(get_model().embed(texts, batch_size=batch_size)), dtype="float32")
    return _normalize(vectors)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Used by the Render build step to download the model ahead of time.
    print("Embedding dimension:", embed_query("warm up").shape[0])
