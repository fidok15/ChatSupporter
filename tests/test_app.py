import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import openai
from fastapi.testclient import TestClient

import app as app_module


class FakeBot:
    def __init__(self, exc=None):
        self.exc, self.seen = exc, []

    def reply(self, session, text):
        if self.exc:
            raise self.exc
        self.seen.append((session.id, text))
        return f"echo:{text}"


def client(bot=None):
    application = app_module.create_app(bot or FakeBot())
    return application, TestClient(application)


def test_new_session_then_reuse():
    _, c = client()
    first = c.post("/chat", json={"message": "hej"}).json()
    assert first["reply"] == "echo:hej"
    second = c.post("/chat", json={"session_id": first["session_id"], "message": "znowu"}).json()
    assert second["session_id"] == first["session_id"]


def test_unknown_session_id_gets_a_new_session():
    _, c = client()
    out = c.post("/chat", json={"session_id": "made-up", "message": "hej"}).json()
    assert out["session_id"] != "made-up"


def test_rate_limit():
    _, c = client()
    sid = c.post("/chat", json={"message": "1"}).json()["session_id"]
    for i in range(app_module.RATE_LIMIT - 1):
        assert c.post("/chat", json={"session_id": sid, "message": str(i)}).status_code == 200
    r = c.post("/chat", json={"session_id": sid, "message": "too many"})
    assert r.status_code == 429


def test_ollama_down_returns_503():
    err = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"))
    _, c = client(FakeBot(exc=err))
    r = c.post("/chat", json={"message": "hej"})
    assert r.status_code == 503
    assert "niedostępna" in r.json()["detail"]


def test_expired_session_is_replaced():
    application, c = client()
    sid = c.post("/chat", json={"message": "hej"}).json()["session_id"]
    application.state.sessions[sid].last_seen -= app_module.SESSION_TTL + 1
    assert c.post("/chat", json={"session_id": sid, "message": "hej"}).json()["session_id"] != sid


def test_oversized_http_body_rejected():
    _, c = client()
    assert c.post("/chat", json={"message": "a" * 4001}).status_code == 422


def test_index_serves_chat_page_without_html_injection():
    _, c = client()
    page = c.get("/").text
    assert "textContent" in page and "innerHTML" not in page


class SlowBot:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.max_active = 0

    def reply(self, session, text):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return "ok"


def test_parallel_requests_on_one_session_run_one_at_a_time():
    bot = SlowBot()
    _, c = client(bot)
    sid = c.post("/chat", json={"message": "1"}).json()["session_id"]
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: c.post("/chat", json={"session_id": sid, "message": "x"}), range(2)))
    assert bot.max_active == 1


def test_concurrent_mixed_load_no_server_errors_and_rate_limit_holds():
    _, c = client()
    shared = c.post("/chat", json={"message": "seed"}).json()["session_id"]
    jobs = [{"message": f"new {i}"} for i in range(40)]
    jobs += [{"session_id": shared, "message": f"s {i}"} for i in range(30)]
    with ThreadPoolExecutor(16) as pool:
        responses = list(pool.map(lambda j: c.post("/chat", json=j), jobs))
    statuses = [r.status_code for r in responses]
    assert set(statuses) <= {200, 429}
    shared_ok = [r for r, j in zip(responses, jobs) if j.get("session_id") == shared and r.status_code == 200]
    assert len(shared_ok) <= app_module.RATE_LIMIT - 1  # the seed message counts too
