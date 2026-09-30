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


TXT_SECTION_CHARS = 3000  # plain text has no pages, so it is split into numbered sections


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            if encoding == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue  # only trust UTF-16 when a BOM is present
            return text
        except UnicodeDecodeError:
            continue
    raise ValueError("The text file could not be decoded.")


def _end_sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?:;" else text + "."


def normalize_markdown(text: str) -> str:
    """Turn Markdown-style plain text (READMEs, notes) into clean prose paragraphs.

    Headings, list items and table rows become their own sentences, so they don't
    merge into neighbouring sentences during segmentation; formatting symbols
    (#, **, `, >, |---|) are removed while the words are kept.
    """
    out, table_header, in_code = [], None, False
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append("")
            continue
        if in_code:
            out.append(line)          # keep code/diagrams as their own block
            continue
        if not stripped or re.fullmatch(r"[-*_=]{3,}", stripped):
            out.append("")
            table_header = None
            continue
        # Tables: "| a | b |" rows -> "header: value; header: value."
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue              # separator row
            if table_header is None:
                table_header = cells
                continue
            pairs = [f"{h}: {c}" if h else c for h, c in zip(table_header, cells) if c]
            out += ["", _end_sentence("; ".join(pairs)), ""]
            continue
        table_header = None
        stripped = re.sub(r"^>\s*", "", stripped)                        # blockquote
        stripped = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", stripped)     # links/images
        stripped = re.sub(r"(\*\*|__|`)", "", stripped)                  # bold / code marks
        heading = re.match(r"^#{1,6}\s+(.*)", stripped)
        item = re.match(r"^(?:[-*+•]|\d+[.)])\s+(.*)", stripped)
        if heading:
            out += ["", re.sub(r"^\d+(\.\d+)*\.?\s+", "", heading.group(1)).rstrip(".:") + ":", ""]
        elif item:
            out += ["", _end_sentence(item.group(1)), ""]
        else:
            out.append(stripped)
    return "\n".join(out)


def extract_text_sections(txt_path: str) -> tuple[list[dict], int]:
    """Read a .txt file and split it into ~3000-character sections on paragraph
    boundaries. Sections play the role of pages for citations ("Section 2")."""
    with open(txt_path, "rb") as fh:
        raw = fh.read()
    # Form feeds (\f) are real page breaks in some exported text files.
    text = normalize_markdown(_decode_text(raw))
    sections, current = [], ""
    for block in re.split(r"\f|\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if current and len(current) + len(block) > TXT_SECTION_CHARS:
            sections.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        sections.append(current)
    pages = [{"page": i + 1, "text": clean_text(s)} for i, s in enumerate(sections)]
    pages = [p for p in pages if p["text"]]
    if sum(len(p["text"]) for p in pages) < MIN_DOCUMENT_CHARS:
        raise ValueError("This text file does not contain enough text.")
    return pages, len(pages)
