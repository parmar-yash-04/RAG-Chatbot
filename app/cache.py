from cachetools import TTLCache
import logfire

from app.config import settings
from app.models import ChatResponse

_chat_cache: TTLCache = TTLCache(
    maxsize=settings.cache_maxsize, ttl=settings.cache_ttl
)


def get_cached_chat(question: str) -> ChatResponse | None:
    with logfire.span("cache_get", key=question[:50]):
        return _chat_cache.get(question)


def set_cached_chat(question: str, response: ChatResponse):
    with logfire.span("cache_set", key=question[:50]):
        _chat_cache[question] = response


def clear_cache():
    _chat_cache.clear()
    logfire.info("Chat cache cleared")


def get_or_compute(question: str, fn) -> ChatResponse:
    cached = get_cached_chat(question)
    if cached is not None:
        logfire.info("Cache hit", question=question[:50])
        return cached
    logfire.info("Cache miss", question=question[:50])
    result = fn(question)
    set_cached_chat(question, result)
    return result
