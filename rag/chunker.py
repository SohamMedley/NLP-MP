"""Step 4: sentence-aware overlapping chunking that keeps page numbers."""
import re

# Split after sentence-ending punctuation followed by whitespace + capital/digit/quote.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in text.split("\n\n"):
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip())
    return sentences


def _hard_split(sentence: str, size: int) -> list[str]:
    """Very long 'sentences' (tables, code) are split on word boundaries."""
    words, parts, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + len(word) + 1 > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_page(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Group whole sentences into chunks of ~chunk_size characters.

    The overlap is built from the trailing sentences of the previous chunk, so a
    sentence is never cut in half and context flows between neighbouring chunks.
    """
    units = []
    for sentence in split_sentences(text):
        units.extend(_hard_split(sentence, chunk_size) if len(sentence) > chunk_size else [sentence])

    chunks, current = [], []
    for unit in units:
        if current and len(" ".join(current + [unit])) > chunk_size:
            chunks.append(" ".join(current))
            # Carry trailing sentences forward as overlap.
            carry, length = [], 0
            for prev in reversed(current):
                if length + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                length += len(prev) + 1
            current = carry
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(pages: list[dict], doc_id: str, filename: str,
                   chunk_size: int, overlap: int) -> list[dict]:
    """Chunks never cross page boundaries, so every citation has an exact page."""
    chunks = []
    for page in pages:
        for text in chunk_page(page["text"], chunk_size, overlap):
            if len(text) < 20:  # skip meaningless fragments
                continue
            chunks.append({
                "doc_id": doc_id,
                "document": filename,
                "page": page["page"],
                "chunk_index": len(chunks),
                "text": text,
            })
    return chunks
