"""Bot tools: SQLite access, session-bound order actions, wiki reads."""
from pathlib import Path

ROOT = Path(__file__).parent
WIKI_DIR = ROOT / "wiki"


def wiki_pages(wiki_dir: Path = WIKI_DIR) -> dict[str, Path]:
    """Whitelist of readable wiki pages: 'gpu/rtx-5090' -> path."""
    return {p.relative_to(wiki_dir).with_suffix("").as_posix(): p for p in wiki_dir.rglob("*.md")}
