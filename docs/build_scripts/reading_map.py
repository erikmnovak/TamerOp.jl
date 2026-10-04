"""Render one linked reading graph and its accessible outline from shared data."""
from __future__ import annotations

from collections import deque
import html
import math
import os
from pathlib import Path
import re
from urllib.parse import quote


def _text(record: dict, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Reading map {key} must be a nonempty string")
    return value


def _number(record: dict, key: str, *, positive: bool = False) -> float:
    value = record.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"Reading map {key} must be a finite {'positive' if positive else 'nonnegative'} number")
    return value


def _identifier(record: dict) -> str:
    value = _text(record, "id")
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", value):
        raise ValueError(f"Invalid reading map id: {value!r}")
    return value


def _validate_path(path: str) -> None:
    """Accept SVG path commands/numbers only, checking each command's arity."""
    token = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
    parts, end = [], 0
    for match in token.finditer(path):
        if path[end:match.start()].strip(" \t\r\n,"):
            raise ValueError("Invalid SVG reading map path")
        parts.append(match.group())
        end = match.end()
    if path[end:].strip(" \t\r\n,") or not parts or parts[0] not in ("M", "m"):
        raise ValueError("Invalid SVG reading map path")
    arities = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
    index = 0
    while index < len(parts):
        command = parts[index].upper()
        if command not in arities:
            raise ValueError("Missing command in SVG reading map path")
        index += 1
        values = []
        while index < len(parts) and parts[index].upper() not in arities:
            value = float(parts[index])
            if not math.isfinite(value):
                raise ValueError("Nonfinite coordinate in SVG reading map path")
            values.append(value)
            index += 1
        arity = arities[command]
        if (arity == 0 and values) or (arity and (not values or len(values) % arity)):
            raise ValueError("Wrong coordinate count in SVG reading map path")
        if command == "A":
            for start in range(0, len(values), 7):
                if min(values[start:start + 2]) < 0 or any(values[start + i] not in (0, 1) for i in (3, 4)):
                    raise ValueError("Invalid arc in SVG reading map path")


def render_reading_map(config: dict, pages: dict[Path, Path], stage: Path,
                       page: Path, docs: Path, repo_url: str) -> str:
    """Return raw HTML; validate the directed acyclic graph and link destinations.

    Sources are relative to ``docs``. Published sources link to their staged
    HTML pages (or downloads); other existing sources link to the repository.
    Arrows suggest reading continuations, without making prerequisite claims.
    """
    docs, stage, page = docs.resolve(), stage.resolve(), page.resolve()
    if not page.is_relative_to(stage):
        raise ValueError("Reading map page must be inside the staging directory")
    staged = {}
    for source, destination in pages.items():
        source, destination = source.resolve(), destination.resolve()
        if not source.is_relative_to(docs) or not destination.is_relative_to(stage):
            raise ValueError("Reading map staged source/destination escapes its directory")
        staged[source] = destination
    if not re.match(r"https?://", repo_url):
        raise ValueError("Reading map repository URL must use HTTP or HTTPS")

    def destination(record: dict) -> tuple[str, str]:
        relative = Path(_text(record, "source"))
        source = (docs / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(docs):
            raise ValueError(f"Reading map source escapes docs: {relative}")
        if not source.is_file() and source not in staged:
            raise ValueError(f"Missing reading map source: {relative}")
        if source in staged:
            target = staged[source]
            if target.suffix == ".md":
                target = target.with_suffix(".html")
            return quote(Path(os.path.relpath(target, page.parent)).as_posix()), "Site page"
        label = "Repository notebook" if source.suffix == ".ipynb" else "Repository guide"
        return repo_url.rstrip("/") + "/" + quote(source.relative_to(docs.parent).as_posix()), label

    width, height = (_number(config, key, positive=True) for key in ("width", "height"))
    node_width, node_height = (_number(config, key, positive=True) for key in ("node_width", "node_height"))
    nodes = config.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Reading map needs at least one node")
    by_id, links, boxes = {}, {}, []
    for node in nodes:
        identity = _identifier(node)
        if identity in by_id:
            raise ValueError(f"Duplicate reading map node: {identity}")
        _text(node, "title")
        _text(node, "question")
        if _text(node, "kind") not in ("Tutorial", "Explanation", "Guide", "Notebook"):
            raise ValueError(f"Unknown reading map kind: {node['kind']}")
        display = node.get("display", "card")
        if display not in ("card", "support", "reference"):
            raise ValueError(f"Unknown reading map display: {display}")
        if "entry" in node and not isinstance(node["entry"], bool):
            raise ValueError("Reading map entry must be boolean")
        if "entry_label" in node:
            _text(node, "entry_label")
            if not node.get("entry") or display != "card":
                raise ValueError("An entry label requires an entry card")
        if display != "reference":
            x, y = (_number(node, key) for key in ("x", "y"))
            w, h = (node_width, node_height) if display == "card" else (
                _number(node, "width", positive=True), _number(node, "height", positive=True))
            if display == "support":
                _text(node, "label")
            if x + w > width or y + h > height:
                raise ValueError(f"Reading map node outside canvas: {identity}")
            for other, ox, oy, ow, oh in boxes:
                if x < ox + ow and ox < x + w and y < oy + oh and oy < y + h:
                    raise ValueError(f"Reading map nodes overlap: {other}, {identity}")
            boxes.append((identity, x, y, w, h))
        by_id[identity] = node
        links[identity] = destination(node)

    edges = config.get("edges", [])
    outgoing, incoming, pairs = {key: [] for key in by_id}, dict.fromkeys(by_id, 0), set()
    for edge in edges:
        source, target = _text(edge, "source"), _text(edge, "target")
        if source not in by_id or target not in by_id:
            raise ValueError(f"Unknown reading map edge endpoint: {source} -> {target}")
        if source == target or (source, target) in pairs:
            raise ValueError(f"Duplicate or self reading map edge: {source} -> {target}")
        if any(by_id[key].get("display", "card") != "card" for key in (source, target)):
            raise ValueError("Draw reading arrows between lesson cards only")
        _validate_path(_text(edge, "path"))
        if edge.get("kind", "continuation") not in ("continuation", "optional"):
            raise ValueError("Unknown reading map edge kind")
        if edge.get("kind") == "optional":
            _text(edge, "label")
            if _number(edge, "label_x") > width or _number(edge, "label_y") > height:
                raise ValueError("Reading map label outside canvas")
        pairs.add((source, target))
        outgoing[source].append(target)
        incoming[target] += 1
    # Setup and references retain real continuations without appearing as
    # compulsory steps. Include those links in cycle checking, not the drawing.
    source_ids = {node["source"]: node["id"] for node in nodes}
    for node in nodes:
        for key in ("next", "alternate"):
            target = source_ids.get(node.get(key))
            if target is None or (node["id"], target) in pairs:
                continue
            if any(by_id[identity].get("display", "card") != "card" for identity in (node["id"], target)):
                pairs.add((node["id"], target))
                outgoing[node["id"]].append(target)
                incoming[target] += 1
    remaining = incoming.copy()
    ready = deque(key for key in by_id if not remaining[key])
    visited = 0
    while ready:
        visited += 1
        for target in outgoing[ready.popleft()]:
            remaining[target] -= 1
            if remaining[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise ValueError("Reading map contains a cycle")

    # Explicit continuations also cover the supporting guides below the graph.
    # The outline presents the same choice, in the same order, as the lesson.
    records = {link["source"]: link for group in config.get("groups", [])
               for link in group.get("links", [])}
    records.update({node["source"]: node for node in nodes})
    for node in nodes:
        if "next" in node:
            for key in ("next", "alternate"):
                if key not in node:
                    continue
                if node[key] not in records:
                    raise ValueError(f"Unknown reading map continuation: {node[key]}")
                target = records[node[key]]
                if "id" in target and (node["id"], target["id"]) not in pairs:
                    raise ValueError(f"Missing reading map continuation arrow: {node['id']} -> {target['id']}")

    escape = html.escape
    optional_labels = {(edge["source"], edge["target"]): edge["label"]
                       for edge in edges if edge.get("kind") == "optional"}
    entry_nodes = [node for node in nodes if node.get("entry_label")]
    support_nodes = [node for node in nodes if node.get("display") == "support"]
    fragments = []
    if entry_nodes:
        fragments.append('<div class="reading-map-mobile-starts" aria-label="Choose a starting point">')
        for node in entry_nodes:
            fragments.append(f'<a href="{escape(links[node["id"]][0])}"><span>{escape(node["entry_label"])}</span><strong>{escape(node["title"])}</strong></a>')
        for node in support_nodes:
            fragments.append(f'<p>{escape(node["label"])}: <a href="{escape(links[node["id"]][0])}">{escape(node["title"])}</a></p>')
        fragments.append('</div>')
    if any(edge.get("kind") == "optional" for edge in edges):
        fragments.append('<div class="reading-map-legend"><span>Solid line: continue</span><span>Dashed line: optional detour</span></div>')
    fragments.extend([
        '<div class="reading-map-scroll" tabindex="0" aria-label="Reading paths. Scroll horizontally to see the whole map, or use the linked outline below.">',
        f'<div class="reading-map-canvas" style="width:{width:g}px;height:{height:g}px">',
        f'<svg class="reading-map-lines" viewBox="0 0 {width:g} {height:g}" width="{width:g}" height="{height:g}" aria-hidden="true" focusable="false" xmlns="http://www.w3.org/2000/svg">',
        '<defs><marker id="reading-map-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>',
    ])
    for edge in edges:
        kind = ' reading-map-optional' if edge.get("kind") == "optional" else ''
        fragments.append(f'<path class="reading-map-edge{kind}" d="{escape(edge["path"])}" fill="none" marker-end="url(#reading-map-arrow)" data-source="{escape(edge["source"])}" data-target="{escape(edge["target"])}"/>')
        if edge.get("kind") == "optional":
            fragments.append(f'<text class="reading-map-edge-label" x="{edge["label_x"]:g}" y="{edge["label_y"]:g}" text-anchor="middle">{escape(edge["label"])}</text>')
    fragments.append('</svg>')
    for node in nodes:
        identity = node["id"]
        url, location = links[identity]
        display = node.get("display", "card")
        if display == "reference":
            continue
        if display == "support":
            fragments.append(f'<aside class="reading-map-support" style="left:{node["x"]:g}px;top:{node["y"]:g}px;width:{node["width"]:g}px;height:{node["height"]:g}px"><span>{escape(node["label"])}</span><a href="{escape(url)}" data-node="{identity}">{escape(node["title"])}</a></aside>')
            continue
        eyebrow = (f'<span class="reading-map-entry-label">{escape(node["entry_label"])}</span>'
                    if node.get("entry_label") else
                    f'<span class="reading-map-kind">{escape(node["kind"])}'
                    + (f' · {location}' if location != "Site page" else '') + '</span>')
        fragments.append(
            f'<a class="reading-map-node" style="left:{node["x"]:g}px;top:{node["y"]:g}px;width:{node_width:g}px;height:{node_height:g}px" href="{escape(url)}" data-node="{identity}" data-entry="{str(node.get("entry", not incoming[identity])).lower()}">'
            + eyebrow +
            f'<strong>{escape(node["title"])}</strong>'
            f'<span class="reading-map-question">{escape(node["question"])}</span></a>')
    fragments.extend(['</div></div>', '<details class="reading-map-outline"><summary>Read the map as a linked outline</summary><ul>'])
    for node in nodes:
        identity = node["id"]
        url, location = links[identity]
        continuation = "End of this route."
        if "next" in node:
            next_page = records[node["next"]]
            end = '' if next_page["title"].endswith(('?', '!', '.')) else '.'
            continuation = f'Continue to: <a href="{escape(destination(next_page)[0])}">{escape(next_page["title"])}</a>{end}'
            if "alternate" in node:
                alternate = records[node["alternate"]]
                end = '' if alternate["title"].endswith(('?', '!', '.')) else '.'
                label = optional_labels.get((identity, alternate.get("id")))
                prompt = f'Optional: {escape(label)}' if label else 'Alternatively'
                continuation += f' {prompt}: <a href="{escape(destination(alternate)[0])}">{escape(alternate["title"])}</a>{end}'
        elif outgoing[identity]:
            continuation = "Continue to: " + ", ".join(
                f'<a href="{escape(links[target][0])}">{escape(by_id[target]["title"])}</a>'
                for target in outgoing[identity]) + "."
        role = node.get("entry_label", "")
        if node.get("display") == "support":
            role = node["label"]
        elif node.get("display") == "reference":
            role = "Supporting reference"
        if role:
            role = f'<strong>{escape(role)}.</strong> '
        fragments.append(
            f'<li><a href="{escape(url)}">{escape(node["title"])}</a> '
            f'<span class="reading-map-location">({location}'
            + (' · starting point' if node.get("entry", not incoming[identity]) else '') + ')</span>'
            + f'<p>{role}{escape(node["question"])} {continuation}</p></li>')
    fragments.append('</ul></details>')

    groups = config.get("groups", [])
    if groups:
        fragments.append('<h2 id="more-questions">More questions to pursue</h2>')
        fragments.append('<div class="reading-map-groups">')
    group_ids = {"more-questions"}
    for group in groups:
        identity = _identifier(group)
        if identity in group_ids:
            raise ValueError(f"Duplicate reading map group: {identity}")
        group_ids.add(identity)
        fragments.append(f'<section class="reading-map-group" aria-labelledby="{identity}"><h3 id="{identity}">{escape(_text(group, "title"))}</h3><p>{escape(_text(group, "description"))}</p><ul>')
        for link in group.get("links", []):
            url, location = destination(link)
            fragments.append(f'<li><a href="{escape(url)}">{escape(_text(link, "title"))}</a> <span class="reading-map-location">({location})</span></li>')
        fragments.append('</ul></section>')
    if groups:
        fragments.append('</div>')
    return "\n".join(fragments)
