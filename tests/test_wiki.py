import seed
import tools

REQUIRED = {"product_id", "name", "price_pln", "in_stock", "source"}
SECTIONS = ["## Specs", "## Requirements", "## Installation", "## Troubleshooting", "## Compatibility"]
PSU = {
    "gpu/rtx-5090": "1000 W",
    "gpu/rtx-5070": "650 W",
    "gpu/rtx-4090": "850 W",
    "gpu/rtx-4070": "650 W",
    "gpu/rtx-3080": "750 W",
    "gpu/rtx-3060-ti": "600 W",
}


def test_card_pages_have_frontmatter_and_sections():
    cards = sorted((tools.WIKI_DIR / "gpu").glob("*.md"))
    assert len(cards) == 6
    for path in cards:
        text = path.read_text()
        fm = seed.parse_frontmatter(text)
        assert REQUIRED <= fm.keys(), path
        assert fm["product_id"] == path.stem
        assert fm["in_stock"] in ("true", "false")
        assert int(fm["price_pln"]) > 0
        for section in SECTIONS:
            assert section in text, (path, section)


def test_index_lists_every_page():
    index = (tools.WIKI_DIR / "index.md").read_text()
    for key in tools.wiki_pages():
        if key != "index":
            assert key in index, key


def test_psu_facts_match_the_pdfs():
    pages = tools.wiki_pages()
    for key, fact in PSU.items():
        assert fact in pages[key].read_text(), key


def test_parse_frontmatter():
    assert seed.parse_frontmatter("---\na: 1\nb: x: y\n---\nbody") == {"a": "1", "b": "x: y"}
    assert seed.parse_frontmatter("no frontmatter") == {}
