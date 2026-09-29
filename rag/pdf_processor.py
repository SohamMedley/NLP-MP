"""Step 2 + 3: page-wise text extraction (PyMuPDF) and NLP text cleaning."""
import re

import pymupdf

MIN_DOCUMENT_CHARS = 50


def clean_text(raw: str) -> str:
    """Normalize extracted PDF text while keeping punctuation and paragraphs intact."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control characters / null bytes that PDF extraction sometimes emits.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]", " ", text)
    # Normalize ligatures and odd whitespace.
    text = (text.replace("\ufb01", "fi").replace("\ufb02", "fl")
                .replace("\u00a0", " ").replace("\u00ad", ""))
    # Re-join words hyphenated across line breaks: "embed-\nding" -> "embedding".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Drop lines that are only page numbers (e.g. "12" or "Page 12 of 40").
    text = re.sub(r"(?im)^\s*(page\s+)?\d+(\s+of\s+\d+)?\s*$", "", text)
    # Collapse spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)
    # Blank lines mark paragraph boundaries; single newlines inside a paragraph become spaces.
    text = re.sub(r"\n\s*\n+", "\u2029", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = text.replace("\u2029", "\n\n")
    return text.strip()


def extract_pages(pdf_path: str) -> tuple[list[dict], int]:
    """Return (pages, total_pages); pages = [{"page": 1-based number, "text": cleaned text}]."""
    pages = []
    try:
        with pymupdf.open(pdf_path) as doc:
            if doc.needs_pass:
                raise ValueError("This PDF is password-protected and cannot be read.")
            for index, page in enumerate(doc):
                text = clean_text(page.get_text("text"))
                if text:
                    pages.append({"page": index + 1, "text": text})
            total_pages = doc.page_count
    except ValueError:
        raise
    except Exception as exc:  # corrupted / not a real PDF
        raise ValueError("The file could not be read as a valid PDF.") from exc

    if sum(len(p["text"]) for p in pages) < MIN_DOCUMENT_CHARS:
        raise ValueError("This PDF does not contain enough extractable text "
                         "(it may be a scanned image).")
    return pages, total_pages
