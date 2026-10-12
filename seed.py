"""Build shop.db: products from wiki frontmatter + deterministic sample orders."""


def parse_frontmatter(text: str) -> dict[str, str]:
    """Parse 'key: value' lines between leading '---' fences. No YAML features needed."""
    if not text.startswith("---\n"):
        return {}
    head = text[4:].split("\n---", 1)[0]
    pairs = (line.partition(":") for line in head.splitlines())
    return {k.strip(): v.strip() for k, _, v in pairs if k.strip()}
