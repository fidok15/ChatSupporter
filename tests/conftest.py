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
