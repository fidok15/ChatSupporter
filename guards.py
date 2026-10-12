"""Input and output guards. Model-backed checks are passed in as callables, so tests run without models."""
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

MAX_INPUT_CHARS = 1000
INJECTION_THRESHOLD = 0.5
REFUSAL = "Mogę pomóc tylko w sprawach sklepu. / I can only help with store matters."

# Chat-template tokens and our own wrapper tags: a customer must not be able to forge a role or close a tag.
ROLE_MARKUP = re.compile(
    r"<\|[^|>]{0,30}\|>"
    r"|</?\s*(?:user_message|data|system|assistant|tool|think)\s*>"
    r"|\[/?INST\]"
    r"|^[ \t]*#{2,}[ \t]*(?:system|assistant|user)\b[ \t]*:?",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Verdict:
    ok: bool
    layer: str = ""
    reason: str = ""


OK = Verdict(True)


def sanitize(text: str) -> str:
    text = text[: MAX_INPUT_CHARS * 4]  # bounds CPU before normalization; anything this long is rejected as too_long anyway
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf"))
    previous = None
    while previous != text:  # repeat: removing one token can join the pieces of another
        previous, text = text, ROLE_MARKUP.sub("", text)
    return text.strip()


def check_input(
    text: str,
    injection_score: Callable[[str], float],
    moderate: Callable[[list[dict]], bool],
) -> Verdict:
    """text must already be sanitized."""
    if not text:
        return Verdict(False, "input", "empty")
    if len(text) > MAX_INPUT_CHARS:
        return Verdict(False, "input", "too_long")
    score = injection_score(text)
    if score >= INJECTION_THRESHOLD:
        return Verdict(False, "prompt_guard", f"injection score {score:.2f}")
    if not moderate([{"role": "user", "content": text}]):
        return Verdict(False, "llama_guard", "unsafe input")
    return OK
