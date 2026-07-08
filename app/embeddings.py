from google import genai
from google.genai import types
import logfire

from app.config import settings

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _embed_config():
    return types.EmbedContentConfig(output_dimensionality=settings.embedding_dim)


def embed_text(text: str) -> list[float]:
    client = _get_client()
    with logfire.span("embed_text", model=settings.gemini_model):
        result = client.models.embed_content(
            model=settings.gemini_model,
            contents=text,
            config=_embed_config(),
        )
        return result.embeddings[0].values


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    with logfire.span("embed_texts", count=len(texts), model=settings.gemini_model):
        result = client.models.embed_content(
            model=settings.gemini_model,
            contents=texts,
            config=_embed_config(),
        )
        return [e.values for e in result.embeddings]
