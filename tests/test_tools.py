import json
import random
import sqlite3

import pytest

import seed
import tools
from conftest import pick, run, user_says


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


def test_masking_and_money():
    assert tools.mask_text("ul. Marszałkowska 12, 00-950 Warszawa") == "ul. M*** 12, 00-950 W***"
    assert tools.mask_text("Jan Kowalski") == "J*** K***"
    assert tools.mask_email("jan.kowalski@example.com") == "j***@example.com"
    assert tools.fmt_pln(289900) == "2899,00 zł"
    assert tools.fmt_pln(1499) == "14,99 zł"


def test_verify_order_binds_order_and_masks_pii(ctx, db):
    a = pick(db, "pending")
    out = run(ctx, "verify_order", order_id=a["order_id"], email=a["email"])
    assert out["verified"] is True
    assert ctx.session.verified_order == a["order_id"]
    assert a["email"] not in json.dumps(out)
    assert a["ship_name"] not in json.dumps(out, ensure_ascii=False)
    assert a["ship_address"] not in json.dumps(out, ensure_ascii=False)
    assert out["address"].startswith("ul. ")


def test_verify_order_accepts_sloppy_input(ctx, db):
    a = pick(db, "pending")
    out = run(ctx, "verify_order", order_id=f"  {a['order_id'].lower()} ", email=a["email"].upper())
    assert out["verified"] is True
    assert ctx.session.verified_order == a["order_id"]


def test_verify_order_wrong_email_counts_and_locks(ctx, db):
    a = pick(db, "pending")
    b = pick(db, "shipped", exclude_email=a["email"])
    for _ in range(5):
        assert run(ctx, "verify_order", order_id=b["order_id"], email=a["email"]) == {"error": tools.NOT_FOUND}
    assert run(ctx, "verify_order", order_id=b["order_id"], email=b["email"]) == {"error": tools.LOCKED}
    assert ctx.session.verified_order is None


def test_order_status_requires_verification(ctx):
    assert run(ctx, "order_status") == {"error": tools.NOT_VERIFIED}


def test_order_status_ignores_order_id_argument(ctx, db):
    a = pick(db, "pending")
    b = pick(db, "shipped", exclude_email=a["email"])
    run(ctx, "verify_order", order_id=a["order_id"], email=a["email"])
    out = run(ctx, "order_status", order_id=b["order_id"])
    assert out["order_id"] == a["order_id"]


def test_read_wiki_whitelist(ctx):
    assert "rtx-5090" in run(ctx, "read_wiki", page="index")["content"]
    assert "1000 W" in run(ctx, "read_wiki", page="gpu/rtx-5090.md")["content"]
    for bad in ["../app", "/etc/passwd", "../../instructions", "gpu/../../seed"]:
        assert "error" in run(ctx, "read_wiki", page=bad)


def test_dispatch_unknown_tool_and_bad_json(ctx):
    assert "error" in json.loads(tools.dispatch(ctx, "drop_tables", "{}"))
    assert json.loads(tools.dispatch(ctx, "verify_order", "{not json")) == {"error": "arguments are not valid JSON"}
    assert json.loads(tools.dispatch(ctx, "verify_order", "[1, 2]"))["error"] == "invalid arguments"
