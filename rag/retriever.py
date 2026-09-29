"""Steps 7-8: question understanding, semantic retrieval and context construction."""
import re

from . import config
from .embeddings import embed_query
from .vector_store import store

# Words that usually mean a follow-up refers back to an earlier topic.
_FOLLOW_UP = re.compile(r"\b(it|its|this|that|these|those|they|them|their|he|she|"
                        r"above|previous|same|more|example|elaborate|explain further)\b", re.I)


def build_search_query(question: str, history: list[dict]) -> str:
    """Resolve simple follow-ups ("Explain it with an example") by prepending the
    previous user question, so the embedding carries the missing topic."""
    previous = [m["content"] for m in history if m.get("role") == "user"]
    if previous and (len(question.split()) <= 12 or _FOLLOW_UP.search(question)):
        return f"{previous[-1]} {question}"
    return question


def retrieve(question: str, history: list[dict], top_k: int | None = None):
    """Embed the query, run cosine-similarity search, and keep chunks above the
    relevance threshold. Returns (relevant_chunks, all_candidates, search_query)."""
    search_query = build_search_query(question, history)
    candidates = store.search(embed_query(search_query), top_k or config.TOP_K)
    relevant = [c for c in candidates if c["score"] >= config.MIN_SIMILARITY]
    return relevant, candidates, search_query


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Source {number}]\nDocument: {chunk['document']}\n"
                      f"Page: {chunk['page']}\nContent:\n{chunk['text']}")
    return "\n\n".join(blocks)
