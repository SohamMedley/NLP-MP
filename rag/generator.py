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
7. For overview requests (summarize, main concepts, key findings, conclusion), synthesise the answer from the
   supplied passages. If the document has no explicit conclusion or findings section, say so in one short clause and then
   give a concluding summary of what the passages cover. Never answer an overview request with only the not-found sentence.
8. Earlier conversation is provided only to resolve references such as "it" or "that"; facts must still come from the context."""

HISTORY_TURNS = 3          # keep chat memory bounded (last 3 question/answer pairs)
HISTORY_CHARS = 1200       # truncate long previous answers


class GenerationError(Exception):
    pass


def _friendly_error(exc) -> str:
    """Turn Groq SDK errors into actionable messages (no secrets included)."""
    name = type(exc).__name__
    text = str(exc).lower()
    if name == "AuthenticationError" or "invalid api key" in text:
        return "Groq rejected the API key. Check GROQ_API_KEY in your environment settings."
    if "decommissioned" in text or "model_not_found" in text or name == "NotFoundError":
        return (f"The Groq model '{config.GROQ_MODEL}' is unavailable. "
                "Set GROQ_MODEL to a current model (e.g. openai/gpt-oss-20b).")
    if name == "RateLimitError":
        return "Groq rate limit reached. Please wait a minute and try again."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Could not reach the Groq API. Please try again shortly."
    return "Unable to generate an answer right now. Please try again."


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


# Tried in order if the configured model is unavailable on the account.
FALLBACK_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
_working_model = None  # remembered after the first successful fallback


def _is_model_error(exc) -> bool:
    text = str(exc).lower()
    return (type(exc).__name__ in ("NotFoundError", "PermissionDeniedError")
            or "model_not_found" in text or "decommissioned" in text
            or "does not exist" in text or "not have access" in text)


def _complete(model: str, messages: list[dict]) -> str:
    params = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 2048}
    if model.startswith("openai/gpt-oss"):
        # Reasoning models: keep thinking short and out of the returned content.
        params.update(reasoning_effort="low", include_reasoning=False)
    response = _get_client().chat.completions.create(**params)
    return (response.choices[0].message.content or "").strip()


def active_model() -> str:
    return _working_model or config.GROQ_MODEL


def generate_answer(question: str, context: str, history: list[dict], broad: bool = False) -> str:
    global _working_model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += _bounded_history(history)
    messages.append({
        "role": "user",
        "content": f"Document context:\n\n{context}\n\n---\nQuestion: {question}\n\n"
                   "Answer using only the document context above."
                   + (" This is an overview request: synthesise across the passages (rule 7)." if broad else ""),
    })
    candidates = [active_model()] + [m for m in FALLBACK_MODELS if m != active_model()]
    last_exc = None
    for model in candidates:
        try:
            answer = _complete(model, messages)
            if model != config.GROQ_MODEL and _working_model != model:
                logger.warning("GROQ_MODEL '%s' unavailable; using fallback '%s'.",
                               config.GROQ_MODEL, model)
            _working_model = model
            break
        except GenerationError:
            raise
        except Exception as exc:
            # Log the error type/message only - never the request headers or key.
            logger.error("Groq API call failed (%s): %s: %s", model, type(exc).__name__, exc)
            last_exc = exc
            if not _is_model_error(exc):
                raise GenerationError(_friendly_error(exc)) from exc
    else:
        raise GenerationError(_friendly_error(last_exc)) from last_exc
    if not answer:
        raise GenerationError("Unable to generate an answer right now. Please try again.")
    return answer
