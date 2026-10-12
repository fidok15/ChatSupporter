"""Agent loop: input guard -> tool-calling LLM -> output guard. Model clients are injected (fakes in tests)."""
import hashlib
import json
import re
import secrets
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import guards
import tools

INSTRUCTIONS_PATH = Path(__file__).parent / "instructions.md"
MAX_STEPS = 5
HISTORY_MESSAGES = 20  # 10 turns
SHOP_EMAIL = "support@gpu-warehouse.example"
FALLBACK_REPLY = (
    "Nie udało mi się tego załatwić. Mogę przekazać sprawę konsultantowi. / "
    "I could not complete this. I can pass it to a human agent."
)
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

# llm(messages, tool_schemas) -> {"content": str | None, "tool_calls": [{"id", "name", "arguments"}], "usage": {...}}
LLM = Callable[[list[dict], list[dict]], dict]


def log_incident(session_id: str, verdict: guards.Verdict, text: str) -> None:
    """Log a blocked message by hash only: the incident log must not become a PII store."""
    tools.LOG_DIR.mkdir(exist_ok=True)
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "session": session_id,
        "layer": verdict.layer,
        "reason": verdict.reason,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    with open(tools.LOG_DIR / "incidents.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


@dataclass
class Bot:
    llm: LLM
    injection_score: Callable[[str], float]
    moderate: Callable[[list[dict]], bool]
    db_path: str | Path = tools.DB_PATH
    instructions: str = field(default_factory=lambda: INSTRUCTIONS_PATH.read_text())
    canary: str = field(default_factory=lambda: secrets.token_hex(8))
    pages: dict = field(default_factory=tools.wiki_pages)
    on_block: Callable[[str], None] = lambda reason: None

    def __post_init__(self):
        self.system_prompt = f"{self.instructions}\n\nInternal reference id: {self.canary}"
        self.prompt_ngrams = guards.instruction_ngrams(self.instructions)

    def _blocked(self, s: tools.Session, verdict: guards.Verdict, text: str) -> str:
        log_incident(s.id, verdict, text)
        self.on_block(f"{verdict.layer}:{verdict.reason}")
        return guards.REFUSAL

    def reply(self, s: tools.Session, raw_text: str) -> str:
        text = guards.sanitize(raw_text)
        verdict = guards.check_input(text, self.injection_score, self.moderate)
        if not verdict.ok:
            return self._blocked(s, verdict, raw_text)

        s.turn += 1
        s.last_user_text = text
        s.user_texts.append(text)
        s.history.append({"role": "user", "content": f"<user_message>{text}</user_message>"})
        messages = [{"role": "system", "content": self.system_prompt}, *s.history[-HISTORY_MESSAGES:]]

        answer = FALLBACK_REPLY
        with closing(tools.connect(self.db_path)) as db:
            ctx = tools.Ctx(s, db, self.pages)
            for _ in range(MAX_STEPS):
                msg = self.llm(messages, tools.TOOL_SCHEMAS)
                if msg["tool_calls"]:
                    messages.append({
                        "role": "assistant",
                        "content": msg["content"] or "",
                        "tool_calls": [
                            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                            for c in msg["tool_calls"]
                        ],
                    })
                    for c in msg["tool_calls"]:
                        result = tools.dispatch(ctx, c["name"], c["arguments"])
                        messages.append({"role": "tool", "tool_call_id": c["id"], "content": f"<data>{result}</data>"})
                    continue
                answer = THINK_RE.sub("", msg["content"] or "").strip() or FALLBACK_REPLY
                verdict = guards.check_output(
                    answer,
                    canary=self.canary,
                    prompt_ngrams=self.prompt_ngrams,
                    allowed_text=" ".join([*s.user_texts, *s.created_orders, SHOP_EMAIL]),
                    user_text=text,
                    moderate=self.moderate,
                )
                if not verdict.ok:
                    answer = self._blocked(s, verdict, answer)
                break

        s.history.append({"role": "assistant", "content": answer})
        return answer
