"""LangSmith tracing. Off unless LANGSMITH_TRACING=true; PII is masked before anything leaves the machine."""
import os
import re

from langsmith import Client, get_current_run_tree, traceable
from langsmith.anonymizer import create_anonymizer
from langsmith.wrappers import wrap_openai

ENABLED = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"

RULES = [
    {"pattern": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "replace": "<EMAIL>"},
    {"pattern": r"(?i)GPU-[A-Z0-9]{4}-[A-Z0-9]{4}", "replace": "<ORDER>"},
    {"pattern": r'\\?"(?:name|address|email)\\?"\s*:\s*\\?"[^"\\]*\\?"', "replace": '"<FIELD>": "<PII>"'},
    {"pattern": r"(?i)\b(?:ul|al|os|pl)\.\s*[^\n,]{2,60}", "replace": "<ADDR>"},
    {"pattern": r"(?i)\b(?:ulica|ulicy|aleja|aleje|alei|plac|osiedle)\s+[^\n,]{2,60}", "replace": "<ADDR>"},
    {"pattern": r"\b\d{2}-\d{3}\b", "replace": "<POSTCODE>"},
]
_PII_KEY = re.compile(r"(?:^|_)(?:name|email|address)$")


def _replace(value: str, path: list) -> str:
    keys = [k.lower() for k in path if isinstance(k, str)]
    if keys and _PII_KEY.search(keys[-1]) and "function" not in keys:  # function.name is the tool name, keep it
        return "<PII>"
    for r in RULES:
        value = re.sub(r["pattern"], r["replace"], value)
    return value


# Known limits: free-text names ("nazywam się Jan Kowalski") and free-text addresses without a street prefix
# from RULES are not masked; names/addresses are masked only under name/email/address keys or in JSON arguments.
ANONYMIZE = create_anonymizer(_replace)
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
