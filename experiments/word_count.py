"""Count words per <section> in blog/index.html, stripping HTML/SVG/style."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "blog" / "index.html"


def strip_tags(text: str) -> str:
    # drop <style>...</style>, <svg>...</svg>, <script>...</script> blocks entirely
    for tag in ["style", "svg", "script"]:
        text = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>", " ", text, flags=re.DOTALL | re.IGNORECASE
        )
    # drop <figcaption> too? Actually count those — they're part of prose.
    # remaining tags: strip open/close markup
    text = re.sub(r"<[^>]+>", " ", text)
    # html entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_words(text: str) -> int:
    return len([w for w in text.split() if any(c.isalpha() for c in w)])


def main() -> None:
    html = HTML.read_text()
    # Extract each <section id="...">...</section>
    pattern = re.compile(r'<section id="([^"]+)"[^>]*>(.*?)</section>', re.DOTALL)
    total = 0
    print(f"{'Section':<16} {'Words':>8}   {'Notes'}")
    print("-" * 70)
    for m in pattern.finditer(html):
        sec_id = m.group(1)
        body = m.group(2)
        stripped = strip_tags(body)
        wc = count_words(stripped)
        total += wc
        # note: does it still have the placeholder div?
        placeholder = "will be added" in body.lower() or 'class="placeholder"' in body
        note = "(placeholder)" if placeholder else ""
        print(f"{sec_id:<16} {wc:>8}   {note}")
    print("-" * 70)
    print(f"{'TOTAL':<16} {total:>8}")


if __name__ == "__main__":
    main()
