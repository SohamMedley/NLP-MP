"""Central configuration read from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


GROQ_MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR") or str(BASE_DIR / "model_cache")

CHUNK_SIZE = max(200, _int("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = min(max(0, _int("CHUNK_OVERLAP", 150)), CHUNK_SIZE // 2)
TOP_K = min(max(1, _int("TOP_K", 5)), 20)
MIN_SIMILARITY = _float("MIN_SIMILARITY", 0.25)
MAX_UPLOAD_MB = max(1, _int("MAX_UPLOAD_MB", 50))

UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def groq_api_key():
    """Read the key lazily so it is never stored in module globals or logged."""
    return os.getenv("GROQ_API_KEY", "").strip()
