# ChatSupporter

Local customer-support and tech-support chatbot for a sample NVIDIA graphics-card shop. Runs fully on a MacBook (Apple Silicon, 16 GB RAM) with Ollama. Built as a portfolio project, a prototype for a real shop, and a university project on prompt-injection defense.

## What it does

- Answers technical questions about 6 cards (RTX 5090, 5070, 4090, 4070, 3080, 3060 Ti) only from a reviewed markdown wiki.
- Answers shop-policy questions (shipping, returns, warranty, contact).
- Checks order status after verifying order number + email, creates orders and cancels pending ones, always with a confirmation the code checks.
- Hands anything else (returns, complaints, refunds) over to a human.
- Talks Polish and English.

## Architecture

```
browser ─▶ FastAPI ─▶ input guard ─▶ agent loop (qwen3:8b + 7 tools) ─▶ output guard ─▶ reply
                       │                      │                            │
                       │                      └─ SQLite, wiki/ (whitelist) │
                       └──── incidents.jsonl ◀──────────────────────────────┘
                                         LangSmith (optional, PII masked)
```

| File | Role |
|---|---|
| `app.py` | HTTP, sessions, rate limit |
| `agent.py` | Guarded tool-calling loop |
| `guards.py` | Input and output guards |
| `tools.py` | The 7 tools, SQLite, PII masking |
| `models.py` | Ollama chat model, Llama Guard 3, Prompt Guard 2 |
| `tracing.py` | LangSmith with anonymizer |
| `instructions.md` | System prompt (no secrets) |
| `wiki/` | The bot's only knowledge |
| `seed.py` | Builds `shop.db` from wiki frontmatter + sample orders |
| `evals/` | Red-team and accuracy suites, `report.md` |

## Security model

The model is treated as untrusted. A successful prompt injection must not give the attacker anything:

1. **Capabilities** — only 7 tools. No refunds, discounts, price or address changes, no order search by email. Prices always come from the DB.
2. **Input guard** — Unicode NFKC, zero-width and control characters removed, chat-template tokens and wrapper tags stripped, Llama Prompt Guard 2 (injection), Llama Guard 3 (harmful content), 1000-character limit.
3. **Prompt structure** — customer text in `<user_message>`, tool output in `<data>`, both declared as data; a random canary in the system prompt.
4. **Tools** — pydantic validation; actions bound to the verified order in the session; 5 failed verifications lock the session; order and cancel need a bare "tak"/"yes" as the customer's very next message; full email, name and address never reach the model.
5. **Output guard** — canary, any 8-word sequence from the system prompt, any email or order number the customer did not type, Llama Guard 3.
6. **Session** — 20 messages/minute, 30-minute TTL, last 10 turns of history.
7. **Monitoring** — `logs/incidents.jsonl` (hash only, no text) and `blocked` tag in LangSmith.

Results: run `uv run python evals/run.py` to generate `evals/report.md` (needs the Hugging Face login from Setup).

## Setup

Requirements: macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), a Hugging Face account.

```bash
uv sync
ollama pull qwen3:8b
ollama pull llama-guard3:1b
ollama create chatsupporter-qwen3 -f Modelfile
# accept the license at https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M, then:
uv run hf auth login
uv run python seed.py
```

Optional tracing (token usage, latency):

```bash
export LANGSMITH_TRACING=true LANGSMITH_API_KEY=... LANGSMITH_PROJECT=chatsupporter
```

## Run

```bash
uv run uvicorn app:main_app --factory --port 8000
```

Open http://localhost:8000. Demo order numbers and emails are in `tests/fixtures.md`.

## Test

```bash
uv run pytest                          # deterministic tests, no models (P1)
LIVE=1 uv run pytest tests/test_models_live.py   # model smoke tests
uv run python evals/run.py             # red team + accuracy (P2, P3) → evals/report.md
```

## Knowledge sources

The wiki was compiled from NVIDIA's official guides (not committed; download into `sources/pdf/`):

- RTX 5090: https://www.nvidia.com/content/geforce-gtx/geforce-rtx-5090-user-guide-r2.pdf
- RTX 5070: https://www.nvidia.com/content/geforce-gtx/geforce-rtx-5070-user-guide-r2.pdf
- RTX 4090: https://www.nvidia.com/content/geforce-gtx/GeForce_RTX_4090_QSG_Rev1.pdf
- RTX 4070: https://www.nvidia.com/content/geforce-gtx/GeForce_RTX_4070_QSG_Rev1.pdf
- RTX 3080: https://www.nvidia.com/content/geforce-gtx/GEFORCE_RTX_3080_USER_GUIDE_v03.pdf
- RTX 3060 Ti: https://www.nvidia.com/content/geforce-gtx/GEFORCE_RTX_3060_Ti_USER_GUIDE_v02.pdf

Memory, CUDA cores, power and bus width come from NVIDIA's product pages. Prices, stock, policies and orders are invented sample data.

## Known limits

- Prompt injection cannot be prevented fully at the model level; the design limits what a successful injection can do.
- Rate limiting is per session only; add a per-IP limit before exposing the app beyond localhost.
- Free-text names typed in chat are not masked in LangSmith traces; emails, order numbers, addresses and tool-argument names are.
- Sessions live in memory and reset on restart.
- Llama Prompt Guard 2 class index (1 = malicious) is verified by `LIVE=1 uv run pytest tests/test_models_live.py` after the Hugging Face login.
