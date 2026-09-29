"""Steps 9-10: grounded answer generation with the Groq chat API."""
import logging

from . import config

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded documents."

SYSTEM_PROMPT = f"""You are DocuRAG, a document-grounded question answering assistant.

Rules:
1. Answer the user's question using ONLY the supplied document context. Never use outside knowledge for facts.
2. Do not fabricate or guess information. If the context does not contain enough information, reply exactly:
   "{NOT_FOUND_MESSAGE}" and optionally mention briefly what related information the context does contain.
3. If only part of the question is answered by the context, answer that part and clearly state what is missing.
4. Cite sources inline using their numbers, e.g. [Source 2], whenever you use them.
5. Be concise but useful. Preserve important technical terminology exactly as written in the documents.
6. Format with Markdown when helpful: short paragraphs, bullet or numbered lists, **bold** key terms, code blocks for code.
7. Earlier conversation is provided only to resolve references such as "it" or "that"; facts must still come from the context."""

HISTORY_TURNS = 3          # keep chat memory bounded (last 3 question/answer pairs)
HISTORY_CHARS = 1200       # truncate long previous answers


class GenerationError(Exception):
    pass


_client = None


def _get_client():
    global _client
    api_key = config.groq_api_key()
    if not api_key:
        raise GenerationError("GROQ_API_KEY is not configured on the server.")
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=api_key, timeout=60, max_retries=2)
    return _client


def _bounded_history(history: list[dict]) -> list[dict]:
    cleaned = [
        {"role": m["role"], "content": str(m["content"])[:HISTORY_CHARS]}
        for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return cleaned[-HISTORY_TURNS * 2:]


def generate_answer(question: str, context: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += _bounded_history(history)
    messages.append({
        "role": "user",
        "content": f"Document context:\n\n{context}\n\n---\nQuestion: {question}\n\n"
                   "Answer using only the document context above.",
    })
    try:
        response = _get_client().chat.completions.create(
            model=config.GROQ_MODEL, messages=messages, temperature=0.1, max_tokens=1024,
        )
        answer = (response.choices[0].message.content or "").strip()
    except GenerationError:
        raise
    except Exception as exc:
        # Log the error type/message only - never the request headers or key.
        logger.error("Groq API call failed: %s: %s", type(exc).__name__, exc)
        raise GenerationError("Unable to generate an answer right now. Please try again.") from exc
    if not answer:
        raise GenerationError("Unable to generate an answer right now. Please try again.")
    return answer
