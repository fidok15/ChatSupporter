"""Input and output guards. Model-backed checks are passed in as callables, so tests run without models."""
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from tools import ORDER_ID_RE

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


NGRAM = 8
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def instruction_ngrams(instructions: str) -> set[tuple[str, ...]]:
    """8-grams of the system prompt, minus the refusal text the model is told to repeat."""
    return ngrams(instructions) - ngrams(REFUSAL)


def check_output(
    reply: str,
    *,
    canary: str,
    prompt_ngrams: set[tuple[str, ...]],
    allowed_text: str,
    user_text: str,
    moderate: Callable[[list[dict]], bool],
) -> Verdict:
    """allowed_text: everything this session's customer typed, plus orders they created and shop contacts."""
    if canary.lower() in reply.lower():
        return Verdict(False, "output", "canary")
    if ngrams(reply) & prompt_ngrams:
        return Verdict(False, "output", "prompt_leak")
    allowed_lower, allowed_upper = allowed_text.lower(), allowed_text.upper()
    if any(email.lower() not in allowed_lower for email in EMAIL_RE.findall(reply)):
        return Verdict(False, "output", "foreign_email")
    if any(oid not in allowed_upper for oid in ORDER_ID_RE.findall(reply.upper())):
        return Verdict(False, "output", "foreign_order")
    if not moderate([{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]):
        return Verdict(False, "output", "llama_guard")
    return OK
