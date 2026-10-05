"""Add catalog navigation without reserializing Documenter's article or math."""
from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote


@dataclass
class Element:
    tag: str
    attrs: dict
    start: int
    inner: int
    stop: int = 0
    end: int = 0
    parent: "Element | None" = None
    text: list[str] = field(default_factory=list)


class Structure(HTMLParser):
    """Locate exact source spans; generated code, equations and figures stay raw."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr"}

    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.offsets = [0] + [m.end() for m in re.finditer("\n", source)]
        self.elements, self.stack = [], []
        self.feed(source)

    def position(self):
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        start = self.position()
        node = Element(tag, dict(attrs), start, start + len(self.get_starttag_text()),
                       parent=self.stack[-1] if self.stack else None)
        self.elements.append(node)
        if tag in self.VOID:
            node.stop = node.end = node.inner
        else:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            node = self.stack.pop()
            node.stop = node.end = node.inner

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1].tag != tag:
            raise ValueError(f"Unexpected HTML closing tag </{tag}>; review Documenter integration")
        node = self.stack.pop()
        node.stop = self.position()
        node.end = self.source.index(">", node.stop) + 1

    def handle_data(self, value):
        for node in self.stack:
            if node.tag in {"h2", "h3"}:
                node.text.append(value)

    def one(self, *, tag=None, id=None, class_name=None, required=True):
        selected = [n for n in self.elements
                    if (tag is None or n.tag == tag)
                    and (id is None or n.attrs.get("id") == id)
                    and (class_name is None or class_name in n.attrs.get("class", "").split())]
        if len(selected) != 1 and (selected or required):
            raise ValueError(f"Expected one HTML element {tag=}, {id=}, {class_name=}; found {len(selected)}")
        return selected[0] if selected else None


def relative_url(current: str, target: str) -> str:
    path, separator, fragment = target.partition("#")
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or not path:
        raise ValueError(f"Catalog destination must be relative to the site root: {target}")
    link = quote(os.path.relpath(path, str(PurePosixPath(current).parent)).replace(os.sep, "/"))
    return link + ("#" + quote(fragment) if separator else "")


def link(current, target, title, *, css="", description=None):
    active = target == current
    classes = " ".join(filter(None, [css, "is-active" if active else ""]))
    attr = ' aria-current="page"' if active else ""
    detail = f'<span class="site-link-detail">{html.escape(description)}</span>' if description else ""
    return (f'<a href="{html.escape(relative_url(current, target), quote=True)}"'
            f' class="{html.escape(classes, quote=True)}"{attr}>'
            f'{html.escape(title)}{detail}</a>')


def sidebar(catalog: dict, current: str, brand: str, search: str, version: str) -> str:
    anchors = catalog["anchors"]
    active = catalog["pages"].get(current, {}).get("collection")
    pieces = ['<nav class="docs-sidebar site-sidebar" id="site-navigation" aria-label="Documentation">',
              '<div class="site-nav-top">', brand,
              '<button type="button" class="site-nav-close" aria-label="Close navigation">Close</button>',
              '<div class="site-entry-links">']
    for key, title in [("home", "Introduction"), ("install", "Installation"),
                       ("learning", "Learning map"), ("topics", "Topic map")]:
        pieces.append(link(current, anchors[key], title))
    pieces.extend(['</div>', search, '</div>',
                   '<div class="docs-menu site-collections" aria-label="Article collections">'])
    for collection in catalog["collections"]:
        if not collection.get("primary", collection["id"] != "contributing"):
            continue
        chosen = active == collection["id"] or current == collection["page"]
        pieces.append(f'<details class="site-collection" data-collection="{html.escape(collection["id"])}"'
                      + (' open' if chosen else '') + '>')
        pieces.append(f'<summary>{html.escape(collection["title"])}</summary><ul>')
        pieces.append('<li>' + link(current, collection["page"], "Overview") + '</li>')
        items = [item for item in collection["items"] if item["page"] != collection["page"]]
        if not collection["grouped_sidebar"]:
            for item in items:
                pieces.append('<li>' + link(current, item["page"], item["title"]) + '</li>')
            visible_pages = {item["page"] for item in items}
        else:
            # Large collections use their topic index, not a seventy-item tree.
            for topic in collection["topic_groups"]:
                pieces.append('<li>' + link(current, topic["page"], topic["title"]) + '</li>')
            visible_pages = {topic["page"] for topic in collection["topic_groups"]}
        if chosen and current != collection["page"] and current not in visible_pages:
            record = catalog["pages"][current]
            pieces.append('<li class="site-current-article">'
                          + link(current, current, record["title"], description="Current page") + '</li>')
        pieces.append('</ul></details>')
    pieces.extend(['</div><div class="site-nav-bottom">',
                   link(current, anchors["contributing"], "Contributors & contributing"), version,
                   '</div></nav>'])
    return "".join(pieces)


def outline(parsed: Structure, article: Element) -> str:
    headings, seen = [], set()
    for node in parsed.elements:
        if node.tag not in {"h2", "h3"} or not article.inner <= node.start < article.stop:
            continue
        identifier = node.attrs.get("id")
        title = " ".join("".join(node.text).split())
        if not title or not identifier or identifier in seen:
            continue
        seen.add(identifier)
        headings.append((node.tag, identifier, title))
    if not headings:
        return ""
    items = ''.join(f'<li class="site-outline-{tag}"><a href="#{quote(identifier)}">'
                    f'{html.escape(title)}</a></li>' for tag, identifier, title in headings)
    listing = '<ul>' + items + '</ul>'
    # One outline, moved by CSS: open at desktop width and native disclosure below it.
    return ('<aside class="site-outline" aria-label="On this page">'
            '<details class="site-outline-disclosure" open><summary>On this page</summary>'
            + listing + '</details></aside>')


def transform(source: str, current: str, catalog: dict) -> str:
    if 'id="site-navigation"' in source:
        raise ValueError("Site shell already present; rebuild Documenter HTML before finalizing")
    parsed = Structure(source)
    old_sidebar = parsed.one(tag="nav", class_name="docs-sidebar")
    article = parsed.one(tag="article", id="documenter-page")
    brand = parsed.one(class_name="docs-package-name")
    search = parsed.one(id="documenter-search-query")
    version = parsed.one(class_name="docs-version-selector", required=False)
    toggle = parsed.one(id="documenter-sidebar-button")
    body = parsed.one(tag="body")
    head = parsed.one(tag="head")
    if old_sidebar.parent is None or old_sidebar.parent.attrs.get("id") != "documenter":
        raise ValueError("Documenter sidebar is no longer a direct child of #documenter")
    for control in [brand, search, *([version] if version else [])]:
        if not old_sidebar.inner <= control.start < old_sidebar.stop:
            raise ValueError("Documenter sidebar control moved outside its expected container")
    raw = lambda node: source[node.start:node.end] if node else ""
    fresh_sidebar = sidebar(catalog, current, raw(brand), raw(search), raw(version))
    page_outline = outline(parsed, article)
    toggle_html = ('<button type="button" class="docs-sidebar-button docs-navbar-link site-nav-toggle" '
                   'id="documenter-sidebar-button" aria-label="Open navigation" '
                   'aria-controls="site-navigation" aria-expanded="false">'
                   '<span class="fa-solid fa-bars" aria-hidden="true"></span></button>')
    root_url = relative_url(current, "index.html").removesuffix("index.html") or "./"
    metadata = {"root": root_url, "pages": catalog["pages"],
                "topics": {t["id"]: t["title"] for t in catalog["topics"]}}
    payload = json.dumps(metadata, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    tail = ('<script id="site-catalog" type="application/json">' + payload + '</script>'
            '<script defer src="' + relative_url(current, 'assets/site_navigation.js') + '"></script>')
    style = '<link rel="stylesheet" href="' + relative_url(current, 'assets/site_navigation.css') + '"/>'
    if 'site_navigation.css' in source or 'site_navigation.js' in source:
        raise ValueError("Site shell owns its assets; do not also load them through Documenter.HTML assets")
    operations = [(old_sidebar.start, old_sidebar.end, fresh_sidebar),
                  (toggle.start, toggle.end, toggle_html),
                  (article.start, article.start, page_outline),
                  (head.stop, head.stop, style),
                  (body.inner, body.inner, '<a class="site-skip-link" href="#documenter-page">Skip to content</a>'),
                  (body.stop, body.stop, tail)]
    for start, end, replacement in sorted(operations, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def finalize_site(docs: Path) -> None:
    docs = Path(docs)
    catalog = json.loads((docs / '.build/src/catalog.json').read_text())
    build = docs / 'build'
    for asset in ('site_navigation.css', 'site_navigation.js'):
        target = build / 'assets' / asset
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((docs / 'src/assets' / asset).read_bytes())
    # Validate every transformation before changing any generated page.
    changes = [(page, transform(page.read_text(), page.relative_to(build).as_posix(), catalog))
               for page in sorted(build.rglob('*.html'))]
    for page, rendered in changes:
        page.write_text(rendered)
    print(f"Added catalog navigation and article outlines to {len(changes)} pages.")


if __name__ == '__main__':
    finalize_site(Path(__file__).resolve().parents[1])
