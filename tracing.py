"""LangSmith tracing. Off unless LANGSMITH_TRACING=true; PII is masked before anything leaves the machine."""
import os

from langsmith import Client, get_current_run_tree, traceable
from langsmith.anonymizer import create_anonymizer
from langsmith.wrappers import wrap_openai

ENABLED = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"

RULES = [
    {"pattern": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "replace": "<EMAIL>"},
    {"pattern": r"(?i)GPU-[A-Z0-9]{4}-[A-Z0-9]{4}", "replace": "<ORDER>"},
    {"pattern": r'\\?"(?:name|address|email)\\?"\s*:\s*\\?"[^"\\]*\\?"', "replace": '"<FIELD>": "<PII>"'},
    {"pattern": r"(?i)\b(?:ul|al|os|pl)\.\s*[^\n,]{2,60}", "replace": "<ADDR>"},
    {"pattern": r"\b\d{2}-\d{3}\b", "replace": "<POSTCODE>"},
]
ANONYMIZE = create_anonymizer(RULES)
_client = Client(anonymizer=ANONYMIZE) if ENABLED else None


def wrap(openai_client):
    """Trace every model call (token usage, latency) when tracing is on."""
    return wrap_openai(openai_client, tracing_extra={"client": _client}) if ENABLED else openai_client


def traced(fn):
    return traceable(name="chat_turn", client=_client)(fn) if ENABLED else fn


def tag_blocked(reason: str) -> None:
    """Tag the current chat_turn run as blocked. Best-effort: tracing must never break a reply."""
    if not ENABLED:
        return
    try:
        run = get_current_run_tree()
        if run is not None:
            run.add_tags(["blocked"])
            run.add_metadata({"block_reason": reason})
    except Exception:
        pass
