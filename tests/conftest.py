import json
import random

import pytest

import seed
import tools


@pytest.fixture
def db():
    conn = tools.connect(":memory:")
    seed.seed(conn, random.Random(42))
    yield conn
    conn.close()


def pick(db, status, exclude_email=""):
    """First seeded order with this status whose email differs from exclude_email."""
    return db.execute(
        "SELECT * FROM orders WHERE status = ? AND email != ? ORDER BY created_at LIMIT 1",
        (status, exclude_email),
    ).fetchone()


@pytest.fixture
def ctx(db, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path / "logs")
    return tools.Ctx(tools.Session(id="test"), db, tools.wiki_pages())


def run(ctx, name, **args):
    """Call a tool the way the agent does: JSON in, JSON out."""
    return json.loads(tools.dispatch(ctx, name, json.dumps(args)))


def user_says(ctx, text):
    """Simulate a new customer message reaching the agent (the agent does this in Bot.reply)."""
    ctx.session.turn += 1
    ctx.session.last_user_text = text
