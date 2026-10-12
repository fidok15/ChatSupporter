"""Bot tools: SQLite access, session-bound order actions, wiki reads."""
import json
import logging
import re
import secrets
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

ROOT = Path(__file__).parent
WIKI_DIR = ROOT / "wiki"
DB_PATH = ROOT / "shop.db"
LOG_DIR = ROOT / "logs"

ORDER_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # 32 chars, no 0/O/1/I
ORDER_ID_RE = re.compile(r"GPU-[A-Z0-9]{4}-[A-Z0-9]{4}")

SCHEMA = """
CREATE TABLE IF NOT EXISTS products(
  product_id   TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  price_grosze INTEGER NOT NULL CHECK (price_grosze > 0),
  in_stock     INTEGER NOT NULL CHECK (in_stock IN (0,1))
);
CREATE TABLE IF NOT EXISTS orders(
  order_id     TEXT PRIMARY KEY,
  email        TEXT NOT NULL,
  ship_name    TEXT NOT NULL,
  ship_address TEXT NOT NULL,
  status       TEXT NOT NULL CHECK (status IN ('pending','paid','shipped','delivered','cancelled')),
  tracking     TEXT,
  total_grosze INTEGER NOT NULL,
  created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS order_items(
  order_id          TEXT NOT NULL REFERENCES orders(order_id),
  product_id        TEXT NOT NULL REFERENCES products(product_id),
  qty               INTEGER NOT NULL CHECK (qty BETWEEN 1 AND 5),
  unit_price_grosze INTEGER NOT NULL
);
"""


def wiki_pages(wiki_dir: Path = WIKI_DIR) -> dict[str, Path]:
    """Whitelist of readable wiki pages: 'gpu/rtx-5090' -> path."""
    return {p.relative_to(wiki_dir).with_suffix("").as_posix(): p for p in wiki_dir.rglob("*.md")}


def connect(path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def init_db(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA)


def new_order_id(rng=secrets) -> str:
    """rng: the secrets module in production, random.Random in seed/tests (both have .choice)."""
    chars = "".join(rng.choice(ORDER_ALPHABET) for _ in range(8))
    return f"GPU-{chars[:4]}-{chars[4:]}"


MAX_FAILED_VERIFICATIONS = 5
NOT_FOUND = "order not found: check the order number and email"
LOCKED = "too many failed attempts in this conversation; offer handoff_to_human"
NOT_VERIFIED = "no verified order: ask for the order number and email, then call verify_order"


@dataclass
class Session:
    id: str
    turn: int = 0
    last_user_text: str = ""
    user_texts: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    verified_order: str | None = None
    failed_verifications: int = 0
    draft: dict | None = None
    pending: str | None = None  # action waiting for the customer's "tak" in their next message
    pending_turn: int = -1
    created_orders: list[str] = field(default_factory=list)
    message_times: list[float] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


@dataclass
class Ctx:
    session: Session
    db: sqlite3.Connection
    pages: dict[str, Path]


def mask_text(text: str) -> str:
    """'ul. Marszałkowska 12, Kraków' -> 'ul. M*** 12, K***'. Keeps numbers and words under 3 letters."""
    return re.sub(r"[^\W\d_]{3,}", lambda m: m.group()[0] + "***", text)


def mask_name(name: str) -> str:
    """'Jo Li' -> 'J*** L***'. Masks every letter run, however short."""
    return re.sub(r"[^\W\d_]+", lambda m: m.group()[0] + "***", name)


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def fmt_pln(grosze: int) -> str:
    zl, gr = divmod(grosze, 100)
    return f"{zl},{gr:02d} zł"


def _order_view(db: sqlite3.Connection, order_id: str) -> dict:
    """What the model may see about an order: no full email, name or address."""
    o = db.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    items = db.execute(
        "SELECT p.name, i.qty FROM order_items i JOIN products p USING (product_id) WHERE i.order_id = ?",
        (order_id,),
    ).fetchall()
    return {
        "order_id": o["order_id"],
        "status": o["status"],
        "tracking": o["tracking"],
        "items": [{"name": r["name"], "qty": r["qty"]} for r in items],
        "total": fmt_pln(o["total_grosze"]),
        "created": o["created_at"][:10],
        "email": mask_email(o["email"]),
        "name": mask_name(o["ship_name"]),
        "address": mask_text(o["ship_address"]),
    }


class WikiArgs(BaseModel):
    page: str = Field(max_length=80)


class VerifyArgs(BaseModel):
    order_id: str = Field(max_length=20)
    email: str = Field(max_length=120)


def read_wiki(ctx: Ctx, args: WikiArgs) -> dict:
    key = args.page.strip().removesuffix(".md")
    path = ctx.pages.get(key)  # whitelist lookup: no path is ever built from model input
    if path is None:
        return {"error": "unknown page; read 'index' for the list of pages"}
    return {"page": key, "content": path.read_text()}


def verify_order(ctx: Ctx, args: VerifyArgs) -> dict:
    s = ctx.session
    if s.failed_verifications >= MAX_FAILED_VERIFICATIONS:
        return {"error": LOCKED}
    order_id = args.order_id.strip().upper()
    row = ctx.db.execute(
        "SELECT 1 FROM orders WHERE order_id = ? AND lower(email) = lower(?)", (order_id, args.email.strip())
    ).fetchone()
    if row is None:
        s.failed_verifications += 1
        return {"error": NOT_FOUND}
    if order_id != s.verified_order:
        s.pending = None  # a pending confirmation was for the previous order, never carry it over
    s.verified_order = order_id
    return {"verified": True, **_order_view(ctx.db, order_id)}


def order_status(ctx: Ctx) -> dict:
    if not ctx.session.verified_order:
        return {"error": NOT_VERIFIED}
    return _order_view(ctx.db, ctx.session.verified_order)


MAX_ORDERS_PER_SESSION = 3
CONFIRMATIONS = {"tak", "potwierdzam", "tak potwierdzam", "yes", "confirm", "yes confirm"}
EMAIL_PATTERN = r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$"


def is_confirmation(text: str) -> bool:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split()) in CONFIRMATIONS


def _ask_confirmation(s: Session, action: str) -> None:
    s.pending, s.pending_turn = action, s.turn


def _confirmed(s: Session, action: str) -> bool:
    """True only if the customer's very next message after the summary is a bare confirmation.
    Consumes the confirmation, so one "tak" allows exactly one action."""
    ok = s.pending == action and s.pending_turn == s.turn - 1 and is_confirmation(s.last_user_text)
    if ok:
        s.pending = None
    return ok


def _clean(value: str) -> str:
    return "".join(ch for ch in value if unicodedata.category(ch)[0] != "C").strip()


class Item(BaseModel):
    product_id: str = Field(max_length=40)
    qty: int = Field(ge=1, le=5)


class CreateOrderArgs(BaseModel):
    items: list[Item] = Field(min_length=1, max_length=6)
    name: str = Field(min_length=3, max_length=80)
    email: str = Field(max_length=120, pattern=EMAIL_PATTERN)
    address: str = Field(min_length=10, max_length=200)

    @field_validator("name", "address")
    @classmethod
    def strip_control_chars(cls, v: str) -> str:
        return _clean(v)


class HandoffArgs(BaseModel):
    reason: str = Field(max_length=300)


def create_order(ctx: Ctx, args: CreateOrderArgs) -> dict:
    s = ctx.session
    if len(s.created_orders) >= MAX_ORDERS_PER_SESSION:
        return {"error": "order limit reached for this conversation; offer handoff_to_human"}
    lines, total = [], 0
    for item in args.items:
        p = ctx.db.execute(
            "SELECT name, price_grosze, in_stock FROM products WHERE product_id = ?", (item.product_id,)
        ).fetchone()
        if p is None:
            return {"error": f"unknown product_id: {item.product_id}; product ids are the gpu/ page names in the wiki"}
        if not p["in_stock"]:
            return {"error": f"out of stock: {p['name']}"}
        lines.append((item.product_id, p["name"], item.qty, p["price_grosze"]))
        total += item.qty * p["price_grosze"]  # price always from the DB, never from the model
    s.draft = {"name": args.name, "email": args.email, "address": args.address, "lines": lines, "total": total}
    _ask_confirmation(s, "confirm_order")
    return {
        "draft": True,
        "items": [{"name": name, "qty": qty, "price": fmt_pln(price)} for _, name, qty, price in lines],
        "total": fmt_pln(total),
        "name": mask_name(args.name),
        "email": mask_email(args.email),
        "address": mask_text(args.address),
        "next": "show this summary and ask the customer to reply 'tak' or 'yes' to confirm",
    }


def confirm_order(ctx: Ctx) -> dict:
    s, db = ctx.session, ctx.db
    if s.draft is None:
        return {"error": "no draft order; call create_order first"}
    if len(s.created_orders) >= MAX_ORDERS_PER_SESSION:
        return {"error": "order limit reached for this conversation; offer handoff_to_human"}
    if not _confirmed(s, "confirm_order"):
        return {"error": "the customer has not confirmed; show the summary and ask them to reply 'tak' or 'yes'"}
    d, oid = s.draft, new_order_id()
    with db:
        db.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)",
            (oid, d["email"], d["name"], d["address"], "pending", None, d["total"],
             datetime.now().isoformat(timespec="seconds")),
        )
        db.executemany(
            "INSERT INTO order_items VALUES (?,?,?,?)", [(oid, pid, qty, price) for pid, _, qty, price in d["lines"]]
        )
    s.draft = None
    s.created_orders.append(oid)
    s.verified_order = oid
    return {**_order_view(db, oid), "created": True}  # after the spread: view's "created" is the date


def cancel_order(ctx: Ctx) -> dict:
    s, db = ctx.session, ctx.db
    if not s.verified_order:
        return {"error": NOT_VERIFIED}
    view = _order_view(db, s.verified_order)
    if view["status"] != "pending":
        return {"error": "only pending orders can be cancelled; offer handoff_to_human"}
    if not _confirmed(s, "cancel_order"):
        _ask_confirmation(s, "cancel_order")
        return {"confirm_required": True, **view, "next": "ask the customer to reply 'tak' or 'yes' to cancel"}
    with db:
        db.execute("UPDATE orders SET status = 'cancelled' WHERE order_id = ? AND status = 'pending'", (s.verified_order,))
    return {"cancelled": True, **_order_view(db, s.verified_order)}


def handoff_to_human(ctx: Ctx, args: HandoffArgs) -> dict:
    LOG_DIR.mkdir(exist_ok=True)
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "session": ctx.session.id, "reason": _clean(args.reason)}
    with open(LOG_DIR / "handoffs.jsonl", "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"handed_off": True, "next": "tell the customer a human agent will contact them; give the contact details from policies/contact"}


TOOLS = {
    "read_wiki": (read_wiki, WikiArgs),
    "verify_order": (verify_order, VerifyArgs),
    "order_status": (order_status, None),
    "create_order": (create_order, CreateOrderArgs),
    "confirm_order": (confirm_order, None),
    "cancel_order": (cancel_order, None),
    "handoff_to_human": (handoff_to_human, HandoffArgs),
}


def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
        },
    }


TOOL_SCHEMAS = [
    _fn("read_wiki", "Read one page of the shop wiki. Start with page 'index' to see all pages.",
        {"page": {"type": "string", "description": "Page name from the index, e.g. 'gpu/rtx-5090'."}}, ["page"]),
    _fn("verify_order", "Verify the customer with order number and email. Required before any order action.",
        {"order_id": {"type": "string", "description": "Format GPU-XXXX-XXXX."},
         "email": {"type": "string", "description": "Email used for the order."}}, ["order_id", "email"]),
    _fn("order_status", "Status of the verified order."),
    _fn("create_order", "Prepare a new order draft and return its summary. Does not place the order.",
        {"items": {"type": "array", "items": {"type": "object", "properties": {
            "product_id": {"type": "string", "description": "Wiki page name without 'gpu/', e.g. 'rtx-5090'."},
            "qty": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["product_id", "qty"]}},
         "name": {"type": "string", "description": "Customer full name."},
         "email": {"type": "string"},
         "address": {"type": "string", "description": "Full shipping address in Poland."}},
        ["items", "name", "email", "address"]),
    _fn("confirm_order", "Place the drafted order. Works only right after the customer replied 'tak' or 'yes' to the summary."),
    _fn("cancel_order", "Cancel the verified order if it is pending. The first call returns a summary to confirm; "
        "call again after the customer replied 'tak' or 'yes'."),
    _fn("handoff_to_human", "Hand the conversation over to a human agent.",
        {"reason": {"type": "string", "description": "Short reason, e.g. 'return request'."}}, ["reason"]),
]


def dispatch(ctx: Ctx, name: str, arguments: str) -> str:
    """Run one tool call from the model. Never raises on bad input: returns a JSON error instead."""
    entry = TOOLS.get(name)
    if entry is None:
        return json.dumps({"error": f"unknown tool: {name[:40]}"})
    fn, model = entry
    try:
        raw = json.loads(arguments or "{}")
        args = model.model_validate(raw) if model else None
    except json.JSONDecodeError:
        return json.dumps({"error": "arguments are not valid JSON"})
    except ValidationError as e:
        details = [f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}" for err in e.errors()]
        return json.dumps({"error": "invalid arguments", "details": details}, ensure_ascii=False)
    try:
        result = fn(ctx, args) if model else fn(ctx)
    except Exception:
        logging.getLogger(__name__).exception("tool %s failed", name)
        result = {"error": "tool failed; offer handoff_to_human"}
    return json.dumps(result, ensure_ascii=False)
