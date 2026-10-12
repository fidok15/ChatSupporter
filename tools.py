"""Bot tools: SQLite access, session-bound order actions, wiki reads."""
import re
import secrets
import sqlite3
from pathlib import Path

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
