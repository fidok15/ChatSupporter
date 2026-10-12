"""HTTP layer: sessions, rate limit, chat endpoint, static UI.

Run: uv run uvicorn app:main_app --factory --port 8000
"""
import secrets
import threading
import time
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import tools

SESSION_TTL = 30 * 60
# ponytail: per-session limit only; a client that drops session_id gets a new session each time.
# Add a per-IP limit before exposing the app beyond localhost.
RATE_LIMIT = 20  # messages per minute per session
STATIC = Path(__file__).parent / "static"
UNAVAILABLE = "Usługa chwilowo niedostępna. / Service temporarily unavailable."


class ChatIn(BaseModel):
    session_id: str | None = Field(default=None, max_length=64)
    message: str = Field(max_length=4000)  # the guard enforces 1000 after sanitizing


class ChatOut(BaseModel):
    session_id: str
    reply: str


def create_app(bot) -> FastAPI:
    app = FastAPI(title="GPU Warehouse support")
    sessions: dict[str, tools.Session] = {}
    locks: dict[str, threading.Lock] = {}  # one per session: a session handles one message at a time
    sessions_lock = threading.Lock()  # guards sessions and locks (sync endpoints run in a threadpool)
    app.state.sessions = sessions

    def get_session(session_id: str | None) -> tuple[tools.Session, threading.Lock]:
        now = time.time()
        with sessions_lock:
            for sid in [sid for sid, s in sessions.items() if now - s.last_seen > SESSION_TTL]:
                del sessions[sid]
                del locks[sid]
            session = sessions.get(session_id or "")
            if session is None:  # never trust a client-chosen id: unknown ids get a fresh server-made session
                session = tools.Session(id=secrets.token_urlsafe(16))
                sessions[session.id] = session
                locks[session.id] = threading.Lock()
            session.last_seen = now
            return session, locks[session.id]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "chat.html")

    @app.post("/chat", response_model=ChatOut)
    def chat(body: ChatIn):
        session, lock = get_session(body.session_id)
        with lock:  # same session: rate check and reply run one at a time; other sessions run in parallel
            now = time.time()
            session.message_times = [t for t in session.message_times if now - t < 60]
            if len(session.message_times) >= RATE_LIMIT:
                raise HTTPException(429, "Za dużo wiadomości, odczekaj minutę. / Too many messages, wait a minute.")
            session.message_times.append(now)
            try:
                reply = bot.reply(session, body.message)
            except openai.APIConnectionError:  # includes APITimeoutError
                raise HTTPException(503, UNAVAILABLE)
        return ChatOut(session_id=session.id, reply=reply)

    return app


def main_app() -> FastAPI:
    import models
    import tracing
    from agent import Bot

    client = tracing.wrap(models.make_client())
    bot = Bot(
        llm=models.chat_llm(client),
        injection_score=models.prompt_guard(),
        moderate=models.llama_guard(client),
        on_block=tracing.tag_blocked,
    )
    bot.reply = tracing.traced(bot.reply)
    return create_app(bot)
