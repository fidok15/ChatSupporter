"""Bot tools: SQLite access, session-bound order actions, wiki reads."""
import json
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

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
        "name": mask_text(o["ship_name"]),
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
    s.verified_order = order_id
    return {"verified": True, **_order_view(ctx.db, order_id)}


def order_status(ctx: Ctx) -> dict:
    if not ctx.session.verified_order:
        return {"error": NOT_VERIFIED}
    return _order_view(ctx.db, ctx.session.verified_order)


TOOLS = {
    "read_wiki": (read_wiki, WikiArgs),
    "verify_order": (verify_order, VerifyArgs),
    "order_status": (order_status, None),
}


def dispatch(ctx: Ctx, name: str, arguments: str) -> str:
    """Run one tool call from the model. Never raises on bad input: returns a JSON error instead."""
    entry = TOOLS.get(name)
    if entry is None:
        return json.dumps({"error": f"unknown tool: {name[:40]}"})
    fn, model = entry
    try:
        raw = json.loads(arguments or "{}")
        result = fn(ctx, model.model_validate(raw)) if model else fn(ctx)
    except json.JSONDecodeError:
        result = {"error": "arguments are not valid JSON"}
    except ValidationError as e:
        details = [f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}" for err in e.errors()]
        result = {"error": "invalid arguments", "details": details}
    return json.dumps(result, ensure_ascii=False)
