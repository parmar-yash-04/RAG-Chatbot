import re

_MAX_INPUT_LENGTH = 4096
_MAX_OUTPUT_LENGTH = 8192
_BLOCKED_PATTERNS = [
    re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
    re.compile(r"ignore\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
]


def sanitize_input(text: str) -> str:
    text = text.strip()
    if len(text) > _MAX_INPUT_LENGTH:
        text = text[:_MAX_INPUT_LENGTH]
    for pattern in _BLOCKED_PATTERNS:
        text = pattern.sub("", text)
    return text


def validate_output(text: str) -> str:
    if not text or len(text.strip()) == 0:
        return "I couldn't generate a response. Please try rephrasing your question."
    if len(text) > _MAX_OUTPUT_LENGTH:
        text = text[:_MAX_OUTPUT_LENGTH] + "..."
    return text
