"""Local NLP answer engine (no LLM / no API).

Builds *extractive* answers using classic NLP, entirely in Python:

1. Sentence segmentation        split passages into sentences
2. Noise filtering              drop tables, code, diagrams and fragments
3. Tokenization                 lowercase word tokens
4. Stop-word removal            drop function words ("the", "is", ...)
5. Stemming                     rule-based suffix stripping ("embeddings" -> "embed")
6. TF-IDF + cosine similarity   lexical match between question and sentences
7. Sentence embeddings          semantic match (all-MiniLM-L6-v2, local)
8. Question-type analysis       definition / number / person / list / reason cues
9. MMR                          relevant but non-redundant sentence selection
10. TextRank                    graph-based extractive summarization
11. Keyphrase extraction        TF-IDF ranked unigrams and bigrams ("key terms")
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

from .chunker import split_sentences
from .embeddings import embed_texts
from .vector_store import store

STOP_WORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not of off on once only or other our
ours ourselves out over own same she should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves also may might must shall us via per eg ie etc explain describe tell give
please document documents pdf file notes use used using one two three like get gets make makes many much well
""".split())

GENERIC_TERMS = set("""
never always couldn find found note see step steps
answer answers question questions model models app application user users system data text way thing things
time example page pages file files document documents chunk chunks result results value values section sections
work works need needs new first second main key important default see note
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SUFFIXES = ("ational", "ization", "fulness", "ousness", "iveness", "ations", "ation", "ments", "ment",
             "ings", "ing", "edly", "ies", "ied", "ers", "er", "ed", "ly", "es", "s")
_NOISE_CHARS = set("│┃─━►▼▲◄→←↑↓═║╔╗╚╝├┤┬┴┼|{}<>=`~^\\·")
_SUMMARY_CUES = re.compile(r"\b(in (summary|conclusion)|to (conclude|summari[sz]e)|overall|in short|therefore|"
                           r"thus|we (conclude|found|show|demonstrate)|this (paper|project|study|system|report))\b", re.I)
_CONCLUSION_CUES = re.compile(r"\b(conclu\w*|in summary|overall|finally|to sum up|future (work|scope)|"
                              r"demonstrates?|shows? that|limitations?)\b", re.I)
_FINDING_CUES = re.compile(r"\b(found|finding|result\w*|show\w*|achiev\w*|improv\w*|outperform\w*|"
                           r"increas\w*|reduc\w*|accura\w*|\d+(\.\d+)?\s*%)\b", re.I)
_DEFINITION = re.compile(r"\b(is|are|was|refers to|means|defined as|is called|stands for|represents?)\b", re.I)


# ----------------------------------------------------------------- text basics
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


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def preprocess(text: str) -> list[str]:
    """Tokenize -> lowercase -> remove stop words -> stem."""
    return [stem(t) for t in tokenize(text) if t not in STOP_WORDS and len(t) > 1]


def clean_sentence(sentence: str) -> str:
    """Strip leftover formatting symbols while keeping the words."""
    s = re.sub(r"(\*\*|__|`|#{1,6}\s)", "", sentence)
    s = re.sub(r"^\s*(?:[-*•>]|\d+[.)])\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—;|")
    return s


def is_informative(sentence: str) -> bool:
    """Reject tables, code, diagrams, headings and fragments."""
    words = sentence.split()
    if not 6 <= len(words) <= 60:
        return False
    if sum(ch in _NOISE_CHARS for ch in sentence) > 1:
        return False
    letters = sum(ch.isalpha() for ch in sentence)
    if letters / max(1, len(sentence.replace(" ", ""))) < 0.7:
        return False
    if not re.search(r"[a-z]{3,}", sentence):      # all caps / numbers only
        return False
    if sentence.rstrip()[-1:] not in ".!\"')":      # headings, questions, cut-off fragments
        return False
    return True


def is_table_row(sentence: str) -> bool:
    """Rows converted from Markdown tables look like 'Step: 3; What happens: ...'."""
    return bool(re.search(r";\s[^;:]{1,40}:\s", sentence)) and sentence.count(": ") >= 1


_VERBS = re.compile(r"\b(is|are|was|were|has|have|had|can|will|would|should|does|do|did|lets|allows?|uses?|"
                    r"provides?|makes?|shows?|helps?|runs?|finds?|gives?|returns?|needs?|takes?|works?|means?)\b", re.I)


def is_heading(raw: str) -> bool:
    """Short verb-less lines are titles/headings ("Limitations (be aware).")."""
    words = raw.split()
    return 1 <= len(words) <= 10 and raw.rstrip().endswith(":") and not _VERBS.search(raw)


# ----------------------------------------------------------------- scoring tools
def tfidf_matrix(docs_tokens: list[list[str]]):
    """Return (idf dict, list of L2-normalised sparse TF-IDF vectors)."""
    n_docs = len(docs_tokens)
    df = Counter(term for tokens in docs_tokens for term in set(tokens))
    idf = {term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()}
    vectors = []
    for tokens in docs_tokens:
        tf = Counter(tokens)
        vec = {t: (c / max(1, len(tokens))) * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})
    return idf, vectors


def tfidf_similarities(query_tokens: list[str], docs_tokens: list[list[str]]) -> np.ndarray:
    """Cosine similarity between the query and each sentence in TF-IDF space."""
    if not query_tokens or not docs_tokens:
        return np.zeros(len(docs_tokens), dtype="float32")
    idf, vectors = tfidf_matrix(docs_tokens)
    tf = Counter(query_tokens)
    q = {t: (c / len(query_tokens)) * idf.get(t, 1.0) for t, c in tf.items()}
    q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
    return np.array([sum(q[t] * vec.get(t, 0.0) for t in q) / q_norm for vec in vectors], dtype="float32")


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
    return rank / (rank.max() or 1)


def mmr(relevance: np.ndarray, vectors: np.ndarray, k: int, diversity: float = 0.35,
        pool: list[int] | None = None) -> list[int]:
    """Maximal Marginal Relevance: relevant sentences that don't repeat each other."""
    candidates = pool if pool is not None else list(np.argsort(-relevance))
    candidates = [int(i) for i in candidates]
    chosen = []
    while candidates and len(chosen) < k:
        best, best_score = None, -1e9
        for idx in candidates[:40]:
            redundancy = max((float(vectors[idx] @ vectors[j]) for j in chosen), default=0.0)
            if redundancy > 0.88:
                continue  # near-duplicate
            score = (1 - diversity) * relevance[idx] - diversity * redundancy
            if score > best_score:
                best, best_score = idx, score
        if best is None:
            break
        chosen.append(best)
        candidates.remove(best)
    return chosen


def keyphrases(sentences: list[dict], top_n: int = 8) -> list[str]:
    """Keyphrase extraction: rank unigrams/bigrams by TF-IDF across sentences,
    boosted for phrases that occur often; return their most common surface form."""
    docs, surface = [], defaultdict(Counter)
    for s in sentences:
        if is_table_row(s["text"]):
            continue
        original = [w for w in re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", s["text"]) if len(w) > 2]
        cased = {w.lower(): w for w in original}
        words = [w.lower() for w in original]
        terms = []
        for i, w in enumerate(words):
            if w in STOP_WORDS or w.isdigit():
                continue
            if w not in GENERIC_TERMS:
                terms.append(stem(w))
                surface[stem(w)][cased[w]] += 1
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt and nxt not in STOP_WORDS and not nxt.isdigit() and not (w in GENERIC_TERMS and nxt in GENERIC_TERMS):
                key = f"{stem(w)} {stem(nxt)}"
                terms.append(key)
                surface[key][f"{cased[w]} {cased[nxt]}"] += 1
        docs.append(terms)
    if not docs:
        return []
    idf, _ = tfidf_matrix(docs)
    freq = Counter(t for terms in docs for t in terms)
    scores = {t: freq[t] * idf[t] * (2.2 if " " in t else 1.0) for t in freq
              if freq[t] >= (2 if " " in t else 3)}
    picked = []
    for term, _ in sorted(scores.items(), key=lambda kv: -kv[1]):
        if any(term in p or p in term for p in picked):
            continue  # skip "embed" when "sentence embed" is already chosen (and vice versa)
        picked.append(term)
        if len(picked) == top_n:
            break
    def display(term):
        form = surface[term].most_common(1)[0][0]
        return form if any(ch.isupper() for ch in form) else form[:1].upper() + form[1:]
    return [display(t) for t in picked]


# ----------------------------------------------------------------- sentence pool
class SentencePool:
    """Informative, de-duplicated sentences with their source chunk and position."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.items = []
        seen = set()
        doc_lengths = Counter(c["doc_id"] for c in chunks)
        section = {}
        for chunk in sorted(chunks, key=lambda c: (c["document"], c["chunk_index"])):
            for position, raw in enumerate(split_sentences(chunk["text"])):
                text = clean_sentence(raw)
                # Headings end with ":" and may be glued to the next sentence by chunking.
                glued = re.match(r"^([^:.!?;]{2,90}:)\s+(?=[A-Z])(.+)$", text)
                if glued and is_heading(glued.group(1)) and not is_table_row(text):
                    section[chunk["doc_id"]] = glued.group(1).rstrip(":").strip()
                    text = glued.group(2)
                if is_heading(text):
                    section[chunk["doc_id"]] = re.sub(r"^[\d.\s]+", "", text).rstrip(".:").strip()
                    continue
                key = re.sub(r"\W+", " ", text.lower()).strip()
                if key in seen or not is_informative(text):
                    continue
                seen.add(key)
                self.items.append({
                    "text": text, "chunk": chunk, "position": position,
                    "section": section.get(chunk["doc_id"], ""), "table": is_table_row(text),
                    # Relative location inside the document (0 = start, 1 = end).
                    "rel": chunk["chunk_index"] / max(1, doc_lengths[chunk["doc_id"]] - 1),
                })
        self._vectors = None

    def __len__(self):
        return len(self.items)

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = embed_texts([s["text"] for s in self.items])
        return self._vectors

    def ordered(self, indices):
        return sorted(indices, key=lambda i: (self.items[i]["chunk"]["document"],
                                              self.items[i]["chunk"]["chunk_index"], self.items[i]["position"]))


class Citations:
    """Assigns [Source n] numbers to the chunks an answer actually uses."""

    def __init__(self):
        self.chunks, self._index = [], {}

    def cite(self, chunk: dict) -> int:
        key = (chunk["doc_id"], chunk["chunk_index"])
        if key not in self._index:
            self.chunks.append(chunk)
            self._index[key] = len(self.chunks)
        return self._index[key]

    def line(self, item: dict) -> str:
        return f"{item['text']} [Source {self.cite(item['chunk'])}]"


# ----------------------------------------------------------------- question types
def question_type(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(summar\w*|overview|what is (this|the) (document|pdf|paper|file) about|outline|tl;?dr)\b", q):
        return "summary"
    if re.search(r"\b(main|key|important|core)\s+(concepts?|topics?|ideas?|terms?|themes?|points?)\b|\bkeywords?\b", q):
        return "concepts"
    if re.search(r"\bconclu\w*\b|\bfinal (thoughts|remarks)\b", q):
        return "conclusion"
    if re.search(r"\b(key |main )?(findings?|results?|outcomes?)\b", q):
        return "findings"
    if re.search(r"\b(how many|how much|what percentage|how long|when|what year|what date|number of)\b", q):
        return "number"
    if re.search(r"^\s*who\b|\bwhich (person|author|company)\b", q):
        return "person"
    if re.search(r"^\s*(list|name|what are the)\b|\btypes of\b|\bsteps\b|\bfeatures\b", q):
        return "list"
    if re.search(r"^\s*(what|who) (is|are|was|were)\b|\bdefin\w*\b|\bmeaning of\b|\bstands? for\b", q):
        return "definition"
    if re.search(r"^\s*why\b|\breason\b|\badvantages?\b|\bbenefits?\b", q):
        return "reason"
    if re.search(r"^\s*(list|name|what are the)\b|\btypes of\b|\bsteps\b|\bfeatures\b", q):
        return "list"
    return "general"


def _subject_terms(question: str) -> list[str]:
    """Content words of the question (for definition matching / coverage)."""
    return [t for t in preprocess(question) if t not in {"what", "who", "defin", "mean"}]


# ----------------------------------------------------------------- answer builders
def _scope_chunks(chunks: list[dict], max_per_doc: int = 60) -> list[dict]:
    """For document-level requests, read every chunk of the documents in scope
    (evenly sampled for very long documents), not only the retrieved ones."""
    doc_ids = {c["doc_id"] for c in chunks}
    by_doc = defaultdict(list)
    for c in store.all_chunks():
        if c["doc_id"] in doc_ids:
            by_doc[c["doc_id"]].append(c)
    scoped = []
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda c: c["chunk_index"])
        step = max(1, math.ceil(len(doc_chunks) / max_per_doc))
        scoped += doc_chunks[::step]
    return scoped or chunks


def _summary(pool: SentencePool, cites: Citations, mode: str) -> str:
    vectors = pool.vectors
    centrality = textrank(vectors)
    rel = np.array([s["rel"] for s in pool.items], dtype="float32")
    text = [s["text"] for s in pool.items]
    cue = np.array([1.0 if (_CONCLUSION_CUES if mode == "conclusion" else
                            _FINDING_CUES if mode == "findings" else _SUMMARY_CUES).search(t) else 0.0
                    for t in text], dtype="float32")

    prose = np.array([0.0 if s["table"] else 1.0 for s in pool.items], dtype="float32")
    pronoun = np.array([1.0 if re.match(r"^(It|This|These|They|Its|Such|He|She)\b", t) else 0.0 for t in text],
                       dtype="float32")
    # Section-aware: sentences under a heading like "Conclusion" / "Results".
    heading_re = {"conclusion": r"conclu|summary|final|closing", "findings": r"result|finding|evaluation|discussion",
                  "summary": r"abstract|introduction|overview|description|about|summary"}[mode]
    in_section = np.array([1.0 if re.search(heading_re, s["section"], re.I) else 0.0 for s in pool.items],
                          dtype="float32")

    if mode == "conclusion":
        score = 0.4 * centrality + 0.25 * rel + 0.3 * cue + 0.8 * in_section   # late + cues + section
    elif mode == "findings":
        score = 0.5 * centrality + 0.4 * cue + 0.5 * in_section
    else:
        score = 0.7 * centrality + 0.25 * (1 - rel) + 0.15 * cue + 0.2 * in_section   # lead bias
    score = score - 0.15 * pronoun - 10 * (1 - prose)               # prose only, self-contained

    by_doc = defaultdict(list)
    for i in np.argsort(-score):
        if not pool.items[i]["table"]:
            by_doc[pool.items[i]["chunk"]["document"]].append(int(i))
    multi = len(by_doc) > 1
    per_doc = 3 if multi else 6
    title = {"summary": "Summary", "conclusion": "Conclusion", "findings": "Key findings"}[mode]

    parts = []
    for doc_name, ranked in by_doc.items():
        picks = mmr(score, vectors, k=min(per_doc + 1, len(ranked)), pool=ranked)
        if not picks:
            continue
        if mode == "summary":
            # Overview = the strongest early sentence; key points follow in document order.
            # Overview = best self-contained sentence among the document's opening sentences.
            # Overview = first self-contained, descriptive sentence ("X is ...") near the start.
            opening = [i for i in pool.ordered(ranked)[:8] if not pronoun[i]]
            descriptive = [i for i in opening if _DEFINITION.search(pool.items[i]["text"])
                           and len(pool.items[i]["text"].split()) >= 8]
            lead = (descriptive or opening or picks)[0]
            picks = [i for i in mmr(score, vectors, k=min(per_doc + 1, len(ranked)), pool=[i for i in ranked if i != lead])
                     if float(vectors[i] @ vectors[lead]) < 0.85]
            rest = pool.ordered(picks)
            block = [f"**Overview:** {cites.line(pool.items[lead])}"]
            if rest:
                block.append("")
                block += [f"- {cites.line(pool.items[i])}" for i in rest[:per_doc]]
        else:
            block = [f"- {cites.line(pool.items[i])}" for i in pool.ordered(picks[:per_doc])]
        parts.append((f"#### {doc_name}\n" if multi else "") + "\n".join(block))

    body = f"#### {title}\n" + "\n\n".join(parts) if not multi else f"**{title} of {len(by_doc)} documents**\n\n" + "\n\n".join(parts)
    if mode == "summary":
        terms = keyphrases(pool.items)
        if terms:
            body += "\n\n**Key terms:** " + ", ".join(terms)
    method = {"summary": "TextRank summarization with lead bias + TF-IDF keyphrases",
              "conclusion": "TextRank + position (end of document) + conclusion cue words",
              "findings": "TextRank + result cue words (found, improved, %, ...)"}[mode]
    return body + f"\n\n*Extracted locally: {method}. No LLM used.*"


def _concepts(pool: SentencePool, cites: Citations) -> str:
    terms = keyphrases(pool.items, top_n=7)
    if not terms:
        return ""
    vectors = pool.vectors
    centrality = textrank(vectors)
    lines, used = [], set()
    for term in terms:
        stems = [stem(w) for w in tokenize(term)]
        best, best_score = None, -1.0
        for i, item in enumerate(pool.items):
            if i in used:
                continue
            item_stems = {stem(w) for w in tokenize(item["text"])}
            if item["table"] or not all(s in item_stems for s in stems):
                continue
            head = [stem(w) for w in tokenize(item["text"])[:7]]
            if True:
                score = (centrality[i] + (0.5 if _DEFINITION.search(item["text"]) else 0)
                         + (0.6 if all(s in head for s in stems) else 0))
                if score > best_score:
                    best, best_score = i, score
        if best is None:
            lines.append(f"- **{term}**")
        else:
            used.add(best)
            lines.append(f"- **{term}**: {cites.line(pool.items[best])}")
    return ("#### Main concepts\n" + "\n".join(lines)
            + "\n\n*Extracted locally: TF-IDF keyphrase extraction + TextRank sentence selection. No LLM used.*")


def _answer_question(question: str, pool: SentencePool, cites: Citations) -> str:
    qtype = question_type(question)
    subject = _subject_terms(question)
    texts = [s["text"] for s in pool.items]
    tokens = [preprocess(t) for t in texts]

    semantic = pool.vectors @ embed_texts([question])[0]
    lexical = tfidf_similarities(preprocess(question), tokens)
    coverage = np.array([len(set(subject) & set(tk)) / max(1, len(set(subject))) for tk in tokens], dtype="float32")

    # Question-type boosts (simple answer-type detection).
    boost = np.zeros(len(texts), dtype="float32")
    for i, t in enumerate(texts):
        if qtype == "definition" and _DEFINITION.search(t) and coverage[i] > 0:
            head = " ".join(tokenize(t)[:8])
            boost[i] += 0.12 if any(stem(w) in subject for w in head.split()) else 0.05
        elif qtype == "number" and re.search(r"\d", t):
            boost[i] += 0.1
        elif qtype == "person" and re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", t):
            boost[i] += 0.08
        elif qtype == "reason" and re.search(r"\b(because|since|so that|therefore|due to|allows?|helps?)\b", t, re.I):
            boost[i] += 0.08

    # Section-aware: a question about "limitations" favours sentences under a "Limitations" heading.
    for i, s in enumerate(pool.items):
        sec = set(preprocess(s["section"]))
        if sec and subject and len(sec & set(subject)) / len(set(subject)) >= 0.5:
            boost[i] += 0.15
        if s["table"]:
            boost[i] -= 0.3

    relevance = 0.6 * semantic + 0.2 * lexical + 0.2 * coverage + boost
    best = int(np.argmax(relevance))

    # If the question names a section ("limitations", "features", "future enhancements"),
    # answer with that section's sentences in order.
    if subject and qtype in ("list", "general", "definition"):
        def section_match(item):
            sec = set(preprocess(item["section"]))
            return bool(sec) and len(sec & set(subject)) / len(set(subject)) >= 0.99
        section_items = [i for i, it in enumerate(pool.items) if section_match(it) and not it["table"]]
        if len(section_items) >= 2:
            name = pool.items[section_items[0]]["section"]
            lines = [f"**{name}**", ""] + [f"- {cites.line(pool.items[i])}" for i in pool.ordered(section_items)[:8]]
            lines.append(f"\n*Extracted locally: matched the document section \"{name}\" to your question "
                         "(section-aware extraction). No LLM used.*")
            return "\n".join(lines)
    if semantic.max() < 0.28 and coverage.max() < 0.5:
        return ""

    lines = [f"**Answer:** {cites.line(pool.items[best])}"]
    # Coherence: if the next sentence continues the thought ("It ...", "This ..."), include it.
    nxt = best + 1
    if (nxt < len(pool.items) and pool.items[nxt]["chunk"] is pool.items[best]["chunk"]
            and re.match(r"^(It|This|These|They|Its|Such|Therefore|Thus|However|For example)\b", texts[nxt])):
        lines[0] += " " + texts[nxt]

    k = 5 if qtype == "list" else 3
    support = [i for i in mmr(relevance, pool.vectors, k=k + 1)
               if i not in (best, nxt) and relevance[i] >= 0.72 * relevance[best]]
    if support:
        lines.append("\n**Related details**\n")
        lines += [f"- {cites.line(pool.items[i])}" for i in pool.ordered(support[:k])]
    lines.append(f"\n*Extracted locally: question type \"{qtype}\", ranked by 0.6 × embedding cosine + "
                 "0.2 × TF-IDF cosine + 0.2 × term coverage (+ section and answer-type cues). No LLM used.*")
    return "\n".join(lines)


def extractive_answer(question: str, chunks: list[dict], broad: bool = False) -> tuple[str, bool, list[dict]]:
    """Return (markdown answer with [Source n] citations, found?, cited chunks in order)."""
    qtype = question_type(question)
    document_level = qtype in ("summary", "concepts", "conclusion", "findings") or broad
    if document_level and qtype not in ("summary", "concepts", "conclusion", "findings"):
        qtype = "summary"
    pool = SentencePool(_scope_chunks(chunks) if document_level or qtype == "list" else chunks)
    if not len(pool):
        return "", False, []

    cites = Citations()
    if qtype == "concepts":
        answer = _concepts(pool, cites)
    elif document_level:
        answer = _summary(pool, cites, qtype)
    else:
        answer = _answer_question(question, pool, cites)
    return (answer, True, cites.chunks) if answer else ("", False, [])


def keyword_search(question: str, chunks: list[dict], limit: int = 4) -> list[dict]:
    """Lexical fallback: chunks containing most of the question's content terms.
    Catches acronyms and rare names (e.g. "FAISS") that dense embeddings can miss."""
    terms = set(_subject_terms(question))
    if not terms:
        return []
    hits = []
    for chunk in chunks:
        tokens = preprocess(chunk["text"])
        present = terms & set(tokens)
        coverage = len(present) / len(terms)
        if coverage >= (1.0 if len(terms) <= 2 else 0.6):
            hits.append((coverage, sum(tokens.count(t) for t in present), chunk))
    hits.sort(key=lambda h: (-h[0], -h[1]))
    return [h[2] for h in hits[:limit]]
