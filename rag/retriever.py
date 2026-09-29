"""Steps 7-8: question understanding, semantic retrieval and context construction."""
import re

from . import config
from .embeddings import embed_query
from .vector_store import store

# Words that usually mean a follow-up refers back to an earlier topic.
_FOLLOW_UP = re.compile(r"\b(it|its|this|that|these|those|they|them|their|he|she|"
                        r"above|previous|same|more|example|elaborate|explain further)\b", re.I)


# Document-level requests ("Summarize this document", "What are the main concepts?")
# don't resemble any single passage, so their similarity scores are naturally low.
_BROAD = re.compile(r"\b(summar\w*|overview|main (concepts?|points?|ideas?|topics?|themes?)|key "
                    r"(concepts?|points?|findings?|ideas?|takeaways?|topics?)|conclusions?|"
                    r"what is (this|the) (document|pdf|paper|file) about|outline|highlights?)\b", re.I)


def is_broad_question(question: str) -> bool:
    return bool(_BROAD.search(question))


def _representative_chunks(limit: int) -> list[dict]:
    """Spread picks across every document: first chunks (intro) + last chunk (conclusion)."""
    by_doc = {}
    for chunk in store.all_chunks():
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)
    picks = []
    for chunks in by_doc.values():
        chunks.sort(key=lambda c: c["chunk_index"])
        step = max(1, len(chunks) // max(1, limit // max(1, len(by_doc))))
        picks.extend(chunks[::step][: max(2, limit // max(1, len(by_doc)))])
        if chunks[-1] not in picks:
            picks.append(chunks[-1])
    return picks


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
    k = top_k or config.TOP_K
    candidates = store.search(embed_query(search_query), k)
    relevant = [c for c in candidates if c["score"] >= config.MIN_SIMILARITY]

    if is_broad_question(question):
        # Broad question: skip the threshold and add representative chunks
        # from each document so the LLM sees an overview, not random fragments.
        seen, merged = set(), []
        for chunk in candidates + [dict(c, score=0.0) for c in _representative_chunks(k + 3)]:
            key = (chunk["doc_id"], chunk["chunk_index"])
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
        relevant = merged[: k + 3]
        relevant.sort(key=lambda c: (c["document"], c["chunk_index"]))
        used = {(c["doc_id"], c["chunk_index"]) for c in relevant}
        candidates = relevant + [c for c in candidates if (c["doc_id"], c["chunk_index"]) not in used]
    return relevant, candidates, search_query


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Source {number}]\nDocument: {chunk['document']}\n"
                      f"Page: {chunk['page']}\nContent:\n{chunk['text']}")
    return "\n\n".join(blocks)
