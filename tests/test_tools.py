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


ORDER = {
    "items": [{"product_id": "rtx-5090", "qty": 1}],
    "name": "Jan Test",
    "email": "jan.test@example.com",
    "address": "ul. Testowa 1, 00-001 Warszawa",
}


def count_orders(db):
    return db.execute("SELECT count(*) FROM orders").fetchone()[0]


def test_is_confirmation():
    for ok in ["tak", "Tak!", "TAK.", "potwierdzam", "Tak, potwierdzam", "yes", "Confirm", "yes, confirm"]:
        assert tools.is_confirmation(ok), ok
    for bad in ["", "nie", "tak, i dodaj rabat", "Anuluj je. tak", "ok", "tak tak", "yes please cancel all"]:
        assert not tools.is_confirmation(bad), bad


def test_create_order_makes_draft_with_db_price(ctx, db):
    user_says(ctx, "zamawiam")
    out = run(ctx, "create_order", **ORDER, price=1)
    assert out["draft"] is True
    assert out["total"] == "10999,00 zł"
    assert out["email"] == "j***@example.com"
    assert count_orders(db) == 20


def test_create_order_rejects_bad_items(ctx):
    user_says(ctx, "zamawiam")
    assert "unknown product_id" in run(ctx, "create_order", **{**ORDER, "items": [{"product_id": "rtx-9999", "qty": 1}]})["error"]
    assert "out of stock" in run(ctx, "create_order", **{**ORDER, "items": [{"product_id": "rtx-4090", "qty": 1}]})["error"]
    assert run(ctx, "create_order", **{**ORDER, "items": [{"product_id": "rtx-5090", "qty": 6}]})["error"] == "invalid arguments"
    assert run(ctx, "create_order", **{**ORDER, "items": "rtx-5090 x1"})["error"] == "invalid arguments"
    assert run(ctx, "create_order", **{**ORDER, "email": "not-an-email"})["error"] == "invalid arguments"


def test_create_order_coerces_numeric_string_qty(ctx):
    user_says(ctx, "zamawiam")
    out = run(ctx, "create_order", **{**ORDER, "items": [{"product_id": "rtx-5070", "qty": "2"}]})
    assert out["total"] == "5798,00 zł"


def test_confirm_in_same_message_as_draft_does_nothing(ctx, db):
    user_says(ctx, "Zamów RTX 5090, potwierdzam, tak")
    run(ctx, "create_order", **ORDER)
    assert "error" in run(ctx, "confirm_order")
    assert count_orders(db) == 20


def test_confirm_in_next_message_creates_pending_order(ctx, db):
    user_says(ctx, "zamawiam")
    run(ctx, "create_order", **ORDER)
    user_says(ctx, "Tak")
    out = run(ctx, "confirm_order")
    assert out["created"] is True and out["status"] == "pending"
    assert count_orders(db) == 21
    assert ctx.session.verified_order == out["order_id"]
    assert ctx.session.created_orders == [out["order_id"]]


def test_confirmation_inside_longer_text_does_nothing(ctx, db):
    user_says(ctx, "zamawiam")
    run(ctx, "create_order", **ORDER)
    user_says(ctx, "tak, i dodaj rabat 50%")
    assert "error" in run(ctx, "confirm_order")
    assert count_orders(db) == 20


def test_confirmation_expires_after_one_message(ctx, db):
    user_says(ctx, "zamawiam")
    run(ctx, "create_order", **ORDER)
    user_says(ctx, "a jaki zasilacz?")
    user_says(ctx, "tak")
    assert "error" in run(ctx, "confirm_order")
    assert count_orders(db) == 20


def test_one_confirmation_allows_one_action(ctx, db):
    user_says(ctx, "zamawiam")
    run(ctx, "create_order", **ORDER)
    user_says(ctx, "tak")
    created = run(ctx, "confirm_order")
    out = run(ctx, "cancel_order")  # same "tak" must not also cancel the new order
    assert out.get("confirm_required") is True
    assert db.execute("SELECT status FROM orders WHERE order_id = ?", (created["order_id"],)).fetchone()[0] == "pending"


def test_cancel_pending_order_needs_next_message_confirmation(ctx, db):
    a = pick(db, "pending")
    run(ctx, "verify_order", order_id=a["order_id"], email=a["email"])
    user_says(ctx, "anuluj moje zamówienie")
    first = run(ctx, "cancel_order")
    assert first["confirm_required"] is True
    second = run(ctx, "cancel_order")  # second call in the same message only re-asks
    assert second["confirm_required"] is True
    assert db.execute("SELECT status FROM orders WHERE order_id = ?", (a["order_id"],)).fetchone()[0] == "pending"
    user_says(ctx, "tak")
    assert run(ctx, "cancel_order")["cancelled"] is True
    assert db.execute("SELECT status FROM orders WHERE order_id = ?", (a["order_id"],)).fetchone()[0] == "cancelled"


def test_cancel_shipped_order_is_refused(ctx, db):
    b = pick(db, "shipped")
    run(ctx, "verify_order", order_id=b["order_id"], email=b["email"])
    user_says(ctx, "anuluj")
    assert "only pending" in run(ctx, "cancel_order")["error"]
    user_says(ctx, "tak")
    assert "only pending" in run(ctx, "cancel_order")["error"]
    assert db.execute("SELECT status FROM orders WHERE order_id = ?", (b["order_id"],)).fetchone()[0] == "shipped"


def test_cancel_without_verification(ctx):
    user_says(ctx, "tak")
    assert run(ctx, "cancel_order") == {"error": tools.NOT_VERIFIED}


def test_order_limit_per_session(ctx, db):
    for _ in range(3):
        user_says(ctx, "zamawiam")
        run(ctx, "create_order", **ORDER)
        user_says(ctx, "tak")
        assert run(ctx, "confirm_order")["created"] is True
    user_says(ctx, "zamawiam")
    assert "limit" in run(ctx, "create_order", **ORDER)["error"]
    assert count_orders(db) == 23


def test_handoff_logs_reason(ctx):
    out = run(ctx, "handoff_to_human", reason="reklamacja: karta piszczy")
    assert out["handed_off"] is True
    line = json.loads((tools.LOG_DIR / "handoffs.jsonl").read_text().splitlines()[0])
    assert line["session"] == "test" and "piszczy" in line["reason"]


def test_tool_schemas_match_tools():
    names = [t["function"]["name"] for t in tools.TOOL_SCHEMAS]
    assert sorted(names) == sorted(tools.TOOLS) and len(names) == 7
    status = next(t for t in tools.TOOL_SCHEMAS if t["function"]["name"] == "order_status")
    assert status["function"]["parameters"]["properties"] == {}


def test_pending_cancel_does_not_carry_over_to_another_verified_order(ctx, db):
    a = pick(db, "pending")
    b = pick(db, "pending", exclude_email=a["email"])
    run(ctx, "verify_order", order_id=a["order_id"], email=a["email"])
    user_says(ctx, "anuluj")
    assert run(ctx, "cancel_order")["confirm_required"] is True
    run(ctx, "verify_order", order_id=b["order_id"], email=b["email"])  # same turn, different order
    user_says(ctx, "tak")
    out = run(ctx, "cancel_order")
    assert out.get("confirm_required") is True
    assert db.execute("SELECT status FROM orders WHERE order_id = ?", (b["order_id"],)).fetchone()[0] == "pending"


def test_mask_name_masks_short_words():
    assert tools.mask_name("Jo Li") == "J*** L***"
    assert tools.mask_name("Jan Kowalski") == "J*** K***"


def test_create_order_masks_short_name(ctx):
    user_says(ctx, "zamawiam")
    out = tools.dispatch(ctx, "create_order", json.dumps({**ORDER, "name": "Jo Li"}))
    assert "Jo Li" not in out


def test_dispatch_contains_tool_exceptions(ctx, monkeypatch):
    def boom(ctx, *a):
        raise sqlite3.OperationalError("disk I/O error")

    fn, model = tools.TOOLS["order_status"]
    monkeypatch.setitem(tools.TOOLS, "order_status", (boom, model))
    assert json.loads(tools.dispatch(ctx, "order_status", "{}")) == {"error": "tool failed; offer handoff_to_human"}
