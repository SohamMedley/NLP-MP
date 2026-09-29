"""Local NLP answer engine (no LLM / no API).

Builds an *extractive* answer from the retrieved chunks using classic NLP:

1. Sentence segmentation      - split retrieved chunks into sentences
2. Tokenization               - lowercase word tokens
3. Stop-word removal          - drop function words ("the", "is", ...)
4. Stemming                   - light suffix stripping ("embeddings" -> "embed")
5. TF-IDF + cosine similarity - lexical match between question and sentences
6. Sentence embeddings        - semantic match (all-MiniLM-L6-v2, local)
7. MMR                        - pick relevant but non-redundant sentences
8. TextRank                   - graph-based extractive summarization for
                                "summarize / main concepts / conclusion" requests
"""
import math
import re
from collections import Counter

import numpy as np

from .chunker import split_sentences
from .embeddings import embed_texts

STOP_WORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not of off on once only or other our
ours ourselves out over own same she should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves also may might must shall us via per e g ie etc explain describe tell give
please document documents pdf file notes
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SUFFIXES = ("ational", "ization", "fulness", "ousness", "iveness", "ations", "ation", "ments", "ment",
             "ings", "ing", "edly", "ies", "ied", "ers", "er", "ed", "ly", "es", "s")


def stem(word: str) -> str:
    """Light rule-based stemmer (a simplified Porter-style suffix stripper)."""
    if len(word) <= 4 or word.isdigit():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if suffix in ("ies", "ied"):
                word += "y"
            elif len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeioulsz":
                word = word[:-1]  # undouble: "runn" -> "run", "embedd" -> "embed"
            break
    return word


def preprocess(text: str) -> list[str]:
    """Tokenize -> lowercase -> remove stop words -> stem."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP_WORDS and len(t) > 1]


def tfidf_similarities(query_tokens: list[str], docs_tokens: list[list[str]]) -> np.ndarray:
    """Cosine similarity between the query and each sentence in TF-IDF space."""
    n_docs = len(docs_tokens)
    df = Counter(term for tokens in docs_tokens for term in set(tokens))
    idf = {term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()}

    def vector(tokens):
        tf = Counter(tokens)
        total = max(1, len(tokens))
        return {term: (count / total) * idf.get(term, math.log(1 + n_docs) + 1) for term, count in tf.items()}

    q_vec = vector(query_tokens)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
    scores = []
    for tokens in docs_tokens:
        d_vec = vector(tokens)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(q_vec[t] * d_vec[t] for t in q_vec if t in d_vec)
        scores.append(dot / (q_norm * d_norm))
    return np.array(scores, dtype="float32")


def textrank(vectors: np.ndarray, damping: float = 0.85, iterations: int = 50) -> np.ndarray:
    """PageRank over the sentence-similarity graph (TextRank summarization)."""
    sim = np.clip(vectors @ vectors.T, 0, None)
    np.fill_diagonal(sim, 0)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = sim / row_sums
    n = len(vectors)
    rank = np.full(n, 1.0 / n, dtype="float32")
    for _ in range(iterations):
        rank = (1 - damping) / n + damping * (transition.T @ rank)
    return rank


def _collect_sentences(chunks: list[dict]) -> list[dict]:
    """Split chunks into unique, reasonably sized sentences, remembering their source."""
    seen, sentences = set(), []
    for number, chunk in enumerate(chunks, start=1):
        for position, sentence in enumerate(split_sentences(chunk["text"])):
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            words = len(sentence.split())
            if key in seen or words < 5 or words > 80:
                continue  # skip duplicates from chunk overlap, fragments and walls of text
            seen.add(key)
            sentences.append({"text": sentence, "source": number, "chunk": chunk, "position": position})
    return sentences


def _mmr(relevance: np.ndarray, vectors: np.ndarray, k: int, diversity: float = 0.3) -> list[int]:
    """Maximal Marginal Relevance: relevant sentences that don't repeat each other."""
    chosen, candidates = [], list(np.argsort(-relevance))
    while candidates and len(chosen) < k:
        best, best_score = None, -1e9
        for idx in candidates[:30]:
            redundancy = max((float(vectors[idx] @ vectors[j]) for j in chosen), default=0.0)
            score = (1 - diversity) * relevance[idx] - diversity * redundancy
            if score > best_score:
                best, best_score = idx, score
        chosen.append(best)
        candidates.remove(best)
    return chosen


def _document_order(sentences: list[dict], indices: list[int]) -> list[int]:
    return sorted(indices, key=lambda i: (sentences[i]["chunk"]["document"], sentences[i]["chunk"]["chunk_index"],
                                          sentences[i]["position"]))


def extractive_answer(question: str, chunks: list[dict], broad: bool = False) -> tuple[str, bool]:
    """Return (markdown answer with [Source n] citations, found?)."""
    sentences = _collect_sentences(chunks)
    if not sentences:
        return "", False
    vectors = embed_texts([s["text"] for s in sentences])

    if broad:
        # Extractive summary: most central sentences in the similarity graph.
        ranks = textrank(vectors)
        # Round-robin over documents so every uploaded file is represented.
        by_doc = {}
        for i in np.argsort(-ranks):
            by_doc.setdefault(sentences[i]["chunk"]["document"], []).append(int(i))
        per_doc = max(2, 6 // len(by_doc))
        picked = []
        for indices in by_doc.values():
            doc_ranks = ranks[indices] / (ranks[indices].max() or 1)
            local_picks = _mmr(doc_ranks, vectors[indices], k=min(per_doc, len(indices)))
            picked += [indices[j] for j in local_picks]
        picked = _document_order(sentences, picked[:10])
        lines = [f"- {sentences[i]['text']} [Source {sentences[i]['source']}]" for i in picked]
        return ("**Key points extracted from your documents**\n\n" + "\n".join(lines)
                + "\n\n*Generated locally with TextRank extractive summarization (no LLM).*"), True

    q_tokens = preprocess(question)
    semantic = vectors @ embed_texts([question])[0]
    lexical = tfidf_similarities(q_tokens, [preprocess(s["text"]) for s in sentences]) if q_tokens else 0
    relevance = 0.7 * semantic + 0.3 * lexical  # hybrid semantic + lexical score

    best = int(np.argmax(relevance))
    if float(semantic[best]) < 0.3 and float(np.max(lexical if q_tokens else [0])) < 0.15:
        return "", False

    picked = _mmr(relevance, vectors, k=min(4, len(sentences)))
    answer = [f"**Most relevant answer:** {sentences[best]['text']} [Source {sentences[best]['source']}]"]
    support = [i for i in _document_order(sentences, picked) if i != best and relevance[i] >= 0.6 * relevance[best]]
    if support:
        answer.append("\n**Supporting details**\n")
        answer += [f"- {sentences[i]['text']} [Source {sentences[i]['source']}]" for i in support]
    answer.append("\n*Extracted locally with TF-IDF + sentence-embedding ranking (no LLM).*")
    return "\n".join(answer), True
