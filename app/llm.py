from groq import Groq
import logfire

from app.config import settings
from app.guardrails import sanitize_input, validate_output

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def ask_groq(question: str, context: list[dict]) -> str:
    question = sanitize_input(question)
    with logfire.span("ask_groq", model=settings.groq_model):
        context_str = "\n\n".join(
            f"From {h.get('doc_name', 'unknown')}: {h['text']}"
            for h in context
        )

        system_prompt = (
            "You are a helpful RAG assistant. Answer the user's question based solely "
            "on the provided context. If the context doesn't contain enough information, "
            "say so. Keep answers natural and conversational — do NOT include source "
            "filenames or chunk numbers in your response."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {question}",
            },
        ]

        response = _get_client().chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content or ""
        answer = validate_output(answer)

        logfire.info(
            "Groq response received",
            tokens=response.usage.total_tokens if response.usage else 0,
        )

        return answer
