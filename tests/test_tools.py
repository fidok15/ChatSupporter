import random
import sqlite3

import pytest

import seed
import tools


def test_seed_products_come_from_wiki(db):
    rows = {r["product_id"]: r for r in db.execute("SELECT * FROM products")}
    assert len(rows) == 6
    assert rows["rtx-5090"]["price_grosze"] == 1099900
    assert rows["rtx-3060-ti"]["price_grosze"] == 149900
    assert rows["rtx-4090"]["in_stock"] == 0


def test_seed_orders_are_deterministic():
    runs = []
    for _ in range(2):
        conn = tools.connect(":memory:")
        seed.seed(conn, random.Random(42))
        runs.append([r[0] for r in conn.execute("SELECT order_id FROM orders ORDER BY created_at")])
    assert runs[0] == runs[1]
    assert len(runs[0]) == 20
    assert all(tools.ORDER_ID_RE.fullmatch(i) for i in runs[0])


def test_seed_covers_every_status(db):
    statuses = {r[0] for r in db.execute("SELECT status FROM orders")}
    assert statuses == {"pending", "paid", "shipped", "delivered", "cancelled"}
    shipped = db.execute("SELECT tracking FROM orders WHERE status = 'shipped'").fetchall()
    assert all(r[0] for r in shipped)


def test_db_constraints_reject_bad_rows(db):
    oid = db.execute("SELECT order_id FROM orders LIMIT 1").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO order_items VALUES (?, 'rtx-5090', 99, 100)", (oid,))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE orders SET status = 'refunded' WHERE order_id = ?", (oid,))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO order_items VALUES (?, 'rtx-9999', 1, 100)", (oid,))


def test_order_ids_avoid_ambiguous_chars():
    assert len(tools.ORDER_ALPHABET) == 32
    rng = random.Random(1)
    for _ in range(200):
        oid = tools.new_order_id(rng)
        assert tools.ORDER_ID_RE.fullmatch(oid)
        assert not set(oid[4:].replace("-", "")) & set("01OI")
