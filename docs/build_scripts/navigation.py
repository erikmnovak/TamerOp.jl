"""Keep lesson endings and HTML continuations aligned with the reading map."""
from __future__ import annotations

import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import tomllib
from urllib.parse import quote

DOCS = Path(__file__).resolve().parents[1]
REPO = "https://github.com/erikmnovak/TamerOp.jl/blob/main/"


def validate_continuations(config: dict, docs: Path) -> dict:
    """Check the final section's ordered links against the declared route.

    The prose remains authored in the lesson, including portable notebooks.
    Only destinations and titles are shared with the map and site footer.
    """
    records = config["nodes"] + [link for group in config.get("groups", [])
                                  for link in group["links"]]
    by_source = {record["source"]: record for record in records}
    for node in config["nodes"]:
        source = docs / node["source"]
        destinations = [node["next"]]
        if "alternate" in node:
            destinations.append(node["alternate"])
        for target in destinations:
            if target not in by_source or not (docs / target).is_file():
                raise ValueError(f"Unknown continuation in {source.name}: {target}")
            if target == node["source"]:
                raise ValueError(f"Self continuation in {source.name}")
        if source.suffix == ".ipynb":
            notebook = json.loads(source.read_text())
            cell = next(c for c in reversed(notebook["cells"]) if c["cell_type"] == "markdown")
            ending = cell["source"]
            if isinstance(ending, list):
                ending = "".join(ending)
        else:
            ending = re.split(r"^## ", source.read_text(), flags=re.M)[-1]
        links = re.findall(r"\]\(([^)\s]+)\)", ending)
        actual = [(source.parent / link.split("#", 1)[0]).resolve() for link in links]
        expected = [(docs / target).resolve() for target in destinations]
        if actual[:len(expected)] != expected or len(actual) > len(expected):
            raise ValueError(f"Ending of {source.name} must link, in order, to {destinations}; "
                             "update the prose and reading_map.toml together")
    return by_source


def published_pages(manifest: dict, docs: Path, build: Path) -> dict[Path, Path]:
    pages = {path.resolve(): build / path.relative_to(docs / "src").with_suffix(".html")
             for path in (docs / "src").rglob("*.md")}
    pages.update({(docs / source).resolve(): build / Path(page).with_suffix(".html")
                  for source, page in manifest["guides"].items()})
    pages.update({(docs / item["source"]).resolve(): build / Path(item["page"]).with_suffix(".html")
                  for item in manifest["notebooks"]})
    return pages


def continuation_links(node: dict | None, records: dict, pages: dict,
                       page: Path, map_page: Path, docs: Path) -> str:
    def relative(target):
        return quote(Path(os.path.relpath(target, page.parent)).as_posix())

    links = [f'<a class="reading-map-return" href="{relative(map_page)}">Reading map</a>']
    if node is not None:
        target = (docs / node["next"]).resolve()
        href = relative(pages[target]) if target in pages else REPO + quote(target.relative_to(docs.parent).as_posix())
        title = html.escape(records[node["next"]]["title"])
        links.append(f'<a class="docs-footer-nextpage" rel="next" href="{html.escape(href, quote=True)}">Next: {title} »</a>')
    return '<div class="lesson-continuation">' + "".join(links) + '</div>'


class Footer(HTMLParser):
    """Locate Documenter's footer without reserializing mathematics or code."""
    def __init__(self, text: str):
        super().__init__(convert_charrefs=False)
        self.offsets = [0]
        for match in re.finditer("\n", text):
            self.offsets.append(match.end())
        self.active, self.spans = {}, {}
        self.feed(text)

    def absolute_position(self):
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        for name, (opening, start, content, depth) in list(self.active.items()):
            if tag == opening:
                self.active[name] = (opening, start, content, depth + 1)
        classes = dict(attrs).get("class", "").split()
        for name in ("docs-footer", "footer-message"):
            if name in classes:
                if name in self.active or name in self.spans:
                    raise ValueError(f"Duplicate {name} in generated HTML")
                self.active[name] = (tag, self.absolute_position(), self.absolute_position() + len(self.get_starttag_text()), 1)

    def handle_endtag(self, tag):
        for name, (opening, start, content, depth) in list(self.active.items()):
            if tag == opening:
                if depth > 1:
                    self.active[name] = (opening, start, content, depth - 1)
                    continue
                self.spans[name] = (start, content, self.absolute_position(), self.absolute_position() + len(tag) + 3)
                del self.active[name]


def replace_footer(text: str, links: str) -> str:
    parsed = Footer(text)
    if "docs-footer" not in parsed.spans:
        raise ValueError("Missing Documenter footer; review the HTML integration")
    _, start, end, _ = parsed.spans["docs-footer"]
    credit = ""
    if "footer-message" in parsed.spans:
        left, _, _, right = parsed.spans["footer-message"]
        if not start <= left < right <= end:
            raise ValueError("Documenter footer credit is outside the footer")
        credit = text[left:right]
    return text[:start] + links + credit + text[end:]


def finalize_navigation(docs: Path = DOCS):
    manifest = tomllib.loads((docs / "publication.toml").read_text())
    config = tomllib.loads((docs / manifest["reading_map"]["source"]).read_text())
    records = validate_continuations(config, docs)
    build = docs / "build"
    pages = published_pages(manifest, docs, build)
    nodes = {pages[(docs / node["source"]).resolve()]: node for node in config["nodes"]
             if (docs / node["source"]).resolve() in pages}
    map_page = build / Path(manifest["reading_map"]["page"]).with_suffix(".html")
    for page in build.rglob("*.html"):
        links = continuation_links(nodes.get(page), records, pages, page, map_page, docs)
        page.write_text(replace_footer(page.read_text(), links))


if __name__ == "__main__":
    finalize_navigation()
