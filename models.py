"""Real model clients: chat model and Llama Guard via Ollama's OpenAI-compatible API, Prompt Guard via transformers."""
import os

from openai import OpenAI

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "chatsupporter-qwen3")
GUARD_MODEL = os.environ.get("GUARD_MODEL", "llama-guard3:1b")
PROMPT_GUARD_ID = "meta-llama/Llama-Prompt-Guard-2-86M"


def make_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_URL, api_key="ollama", timeout=120)


def chat_llm(client: OpenAI):
    def llm(messages: list[dict], tool_schemas: list[dict]) -> dict:
        r = client.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=tool_schemas, temperature=0.2)
        m = r.choices[0].message
        return {
            "content": m.content,
            "tool_calls": [
                {"id": c.id, "name": c.function.name, "arguments": c.function.arguments} for c in (m.tool_calls or [])
            ],
            "usage": {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens} if r.usage else {},
        }

    return llm


def llama_guard(client: OpenAI):
    """True = safe. Llama Guard answers 'safe' or 'unsafe\\nS<category>'."""
    def moderate(messages: list[dict]) -> bool:
        r = client.chat.completions.create(model=GUARD_MODEL, messages=messages, temperature=0)
        return (r.choices[0].message.content or "").strip().lower().startswith("safe")

    return moderate


def prompt_guard():
    """Probability that text is a prompt injection / jailbreak (class index 1 = malicious)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(PROMPT_GUARD_ID)
    model = AutoModelForSequenceClassification.from_pretrained(PROMPT_GUARD_ID).eval()

    def score(text: str) -> float:
        with torch.no_grad():
            logits = model(**tokenizer(text, return_tensors="pt", truncation=True, max_length=512)).logits
        return torch.softmax(logits, dim=-1)[0, 1].item()

    return score
