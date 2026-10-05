"""Derive published article navigation from editorial records and build routes.

The inventory owns article identity and meaning; publication owns availability.
navigation.toml only chooses collection labels, order and supporting ownership.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import tomllib


TYPE_LABELS = {
    "lesson": "Mathematical lesson",
    "library_guide": "Library guide",
    "recipe": "Task recipe",
    "api_reference": "API reference",
    "implementation_account": "Implementation account",
    "benchmark_report": "Benchmark report",
    "contributor_guide": "Contributor guide",
    "resource": "Supporting resource",
}


def _relative_page(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix not in {".md", ".html"}:
        raise ValueError(f"Invalid site page: {value}")
    return path.as_posix()


def _html_page(value: str) -> str:
    return str(PurePosixPath(_relative_page(value)).with_suffix(".html"))


def _source_page(value: str) -> str:
    return str(PurePosixPath(_relative_page(value)).with_suffix(".md"))


def _href(current: str, target: str, *, html_output=False) -> str:
    target = _html_page(target) if html_output else _source_page(target)
    return Path(os.path.relpath(target, str(PurePosixPath(current).parent))).as_posix()


def _md(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _article_list(articles: list[dict], current: str, *, typed=True) -> str:
    lines = []
    for article in articles:
        label = f" — {article['type_label']}" if typed else ""
        lines.append(f"- [{_md(article['title'])}]({_href(current, article['page'])}){label}. "
                     + article["question"])
    return "\n".join(lines)


def _summary_record(record: dict) -> dict:
    return {key: record[key] for key in
            ("id", "title", "page", "type", "type_label", "question", "topics")}


def _generated_page(page: str, title: str, question: str, *, topics=None,
                    collection=None) -> dict:
    return {"id": "generated:" + page, "title": title, "page": page,
            "source": _source_page(page), "canonical_source": None,
            "type": "resource", "type_label": TYPE_LABELS["resource"],
            "topics": topics or [], "question": question, "collection": collection}


def prepare_catalog(docs: Path, stage: Path) -> dict:
    """Write catalog JSON/TOML, fill collection markers and create topic pages.

    Invoke after canonical Markdown has been staged and links rewritten, but
    before notebook export. Notebook routes are known without executing cells.
    Only existing inventoried sources with an explicit publication route enter
    the catalog. Missing inventory records or conflicting routes fail loudly.
    """
    docs, stage = Path(docs).resolve(), Path(stage).resolve()
    root = docs.parent
    inventory = tomllib.loads((docs / "article_inventory.toml").read_text())
    publication = tomllib.loads((docs / "publication.toml").read_text())
    navigation = tomllib.loads((docs / "navigation.toml").read_text())
    if navigation.get("schema_version") != 1:
        raise ValueError("Unsupported navigation schema")
    records, by_source = {}, {}
    for record in inventory["articles"]:
        if record["id"] in records:
            raise ValueError(f"Duplicate article identity: {record['id']}")
        records[record["id"]] = record
        if record.get("state") != "existing" or not record.get("source"):
            continue
        source = (root / record["source"]).resolve()
        if source in by_source:
            raise ValueError(f"Duplicate canonical article source: {source}")
        by_source[source] = record

    routes: dict[Path, str] = {}
    def add_route(source: Path, destination: str) -> None:
        source = source.resolve()
        page = _html_page(destination)
        if source not in by_source:
            raise ValueError(f"Published source lacks an existing article record: {source}")
        if not source.is_file():
            raise ValueError(f"Missing canonical source: {source}")
        if source in routes and routes[source] != page:
            raise ValueError(f"Canonical source has multiple publication routes: {source}")
        if page in routes.values() and routes.get(source) != page:
            raise ValueError(f"Conflicting publication route: {page}")
        routes[source] = page

    for source in sorted((docs / "src").rglob("*.md")):
        add_route(source, source.relative_to(docs / "src").as_posix())
    for source, page in publication.get("guides", {}).items():
        add_route(docs / source, page)
    for notebook in publication.get("notebooks", []):
        add_route(docs / notebook["source"], notebook["page"])

    pages = {}
    for source, page in routes.items():
        record = by_source[source]
        kind = record["type"]
        if kind not in TYPE_LABELS:
            raise ValueError(f"Unknown article type: {kind}")
        topics = record.get("topics", [])
        if any(topic not in inventory["topics"] for topic in topics):
            raise ValueError(f"Unknown topic on {record['id']}")
        pages[page] = {"id": record["id"], "title": record["title"],
                       "page": page, "source": _source_page(page),
                       "canonical_source": source.relative_to(root).as_posix(),
                       "type": kind, "type_label": TYPE_LABELS[kind],
                       "topics": topics, "question": record["question"],
                       "collection": None}

    anchors = navigation["anchors"]
    for name, page in anchors.items():
        if page not in pages:
            raise ValueError(f"Missing published permanent anchor {name}: {page}")
    anchored = set(anchors.values())
    collections, claimed = [], set()
    for config in navigation["collections"]:
        cid, page = config["id"], _html_page(config["page"])
        if page not in pages:
            raise ValueError(f"Missing collection overview: {page}")
        if any(c["id"] == cid for c in collections):
            raise ValueError(f"Duplicate collection: {cid}")
        order = {article: i for i, article in enumerate(config.get("order", []))}
        members = [item for item in pages.values()
                   if item["type"] != "resource" and item["type"] in config["types"]
                   and item["page"] not in anchored]
        members.sort(key=lambda item: (order.get(item["id"], len(order)),
                                       item["title"].casefold(), item["id"]))
        for item in members:
            if item["page"] in claimed:
                raise ValueError(f"Article belongs to two editorial collections: {item['id']}")
            claimed.add(item["page"])
            item["collection"] = cid
        pages[page]["collection"] = cid
        topic_pages = config.get("topic_pages", False)
        if not isinstance(topic_pages, bool):
            raise ValueError(f"topic_pages must be a boolean for collection: {cid}")
        related_topics = config.get("related_topics", [])
        if not isinstance(related_topics, list) or any(
                topic not in inventory["topics"] for topic in related_topics):
            raise ValueError(f"related_topics must list known topics for collection: {cid}")
        collections.append({"id": cid, "title": config["title"], "page": page,
                            "primary": config.get("primary", True),
                            "topic_pages": topic_pages,
                            "related_topics": related_topics,
                            "items": [_summary_record(item) for item in members],
                            "resources": [], "topic_groups": []})

    lookup = {item["id"]: item for item in pages.values()}
    for config in navigation.get("resources", []):
        resource = lookup.get(config["id"])
        if resource is None:
            continue
        if resource["type"] != "resource":
            raise ValueError(f"Supporting attachment is an article: {config['id']}")
        collection = next((c for c in collections if c["id"] == config["collection"]), None)
        if collection is None:
            raise ValueError(f"Unknown resource collection: {config['collection']}")
        owner = config.get("owner")
        if owner and (owner not in lookup or lookup[owner]["collection"] != collection["id"]):
            raise ValueError(f"Supporting resource owner is not published in its collection: {owner}")
        resource["collection"] = collection["id"]
        resource["owner"] = owner
        collection["resources"].append({**_summary_record(resource), "owner": owner})

    topics = []
    published_articles = [item for item in pages.values()
                          if item["type"] != "resource" and item["page"] not in anchored]
    for tid, title in inventory["topics"].items():
        articles = [item for item in published_articles if tid in item["topics"]]
        if not articles:
            continue
        articles.sort(key=lambda item: (list(TYPE_LABELS).index(item["type"]),
                                        item["title"].casefold()))
        page = f"topics/{tid}.html"
        topics.append({"id": tid, "title": title, "page": page,
                       "articles": [_summary_record(item) for item in articles]})
        if page in pages:
            raise ValueError(f"Generated topic page collides with authored page: {page}")
        pages[page] = _generated_page(page, title,
            f"Which published treatments concern {title.lower()}?", topics=[tid])
        body = f"# {title}\n\nThese treatments share a subject; their labels distinguish the questions they answer.\n\n"
        body += _article_list(articles, page) + "\n\n"
        body += f"[Explore all topics]({_href(page, anchors['topics'])}).\n"
        target = stage / _source_page(page)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)

    threshold = navigation.get("sidebar_threshold", 10)
    if not isinstance(threshold, int) or threshold < 1:
        raise ValueError("sidebar_threshold must be a positive integer")
    for collection in collections:
        collection["grouped_sidebar"] = len(collection["items"]) > threshold
        # Published subject pages can outlive a change in sidebar density.
        if not collection["grouped_sidebar"] and not collection["topic_pages"] and not collection["related_topics"]:
            continue
        # One home cluster per article makes the sidebar a partition. Other
        # topic memberships remain available through the cross-family atlas.
        for tid, title in inventory["topics"].items():
            members = [item for item in collection["items"] if item["topics"] and item["topics"][0] == tid]
            related = not members and tid in collection["related_topics"]
            if related:
                members = [_summary_record(item) for item in published_articles
                           if tid in item["topics"] and item["collection"] != collection["id"]]
            if not members:
                continue
            page = f"collections/{collection['id']}/topics/{tid}.html"
            if page in pages:
                raise ValueError(f"Generated collection topic collides with authored page: {page}")
            pages[page] = _generated_page(page, title,
                (f"Which related treatments concern {title.lower()}?" if related else
                 f"Which {collection['title'].lower()} articles concern {title.lower()}?"),
                topics=[tid], collection=collection["id"])
            group = {"id": tid, "title": title, "page": page, "items": members}
            collection["topic_groups"].append(group)
            target = stage / _source_page(page)
            target.parent.mkdir(parents=True, exist_ok=True)
            introduction = ("Related treatments from other collections." if related else
                            f"{collection['title']} articles in this area.")
            target.write_text(f"# {title}\n\n{introduction}\n\n"
                + _article_list(members, page) + "\n\n"
                + f"[All {collection['title'].lower()}]({_href(page, collection['page'])}) · "
                + f"[Treatments across article types]({_href(page, f'topics/{tid}.html')}).\n")

    by_collection = {item["id"]: item for item in collections}
    for target in stage.rglob("*.md"):
        content = target.read_text()
        current = target.relative_to(stage).as_posix()
        def collection_marker(match):
            cid = match[1]
            if cid not in by_collection:
                raise ValueError(f"Unknown collection marker: {cid}")
            collection = by_collection[cid]
            if collection["grouped_sidebar"]:
                body = "Choose a subject to see its annotated article list:\n\n" + "\n".join(
                    f"- [{_md(group['title'])}]({_href(current, group['page'])}) — "
                    + f"{len(group['items'])} " + ("article." if len(group['items']) == 1 else "articles.")
                    for group in collection["topic_groups"])
            else:
                body = _article_list(collection["items"], current, typed=True)
                if collection["topic_groups"]:
                    links = "\n".join(
                        '<li><a href="'
                        + html.escape(_href(current, group["page"], html_output=True), quote=True)
                        + '">' + html.escape(group["title"]) + '</a></li>'
                        for group in collection["topic_groups"])
                    body += ('\n\n```@raw html\n<details class="collection-subjects">'
                             '<summary>Browse by subject</summary>\n<ul>\n'
                             + links + '\n</ul>\n</details>\n```')
            if collection["resources"]:
                body += "\n\nSupporting resources:\n\n" + _article_list(collection["resources"], current, typed=False)
            return body
        content = re.sub(r"<!--\s*COLLECTION:\s*([a-z_]+)\s*-->", collection_marker, content)
        if "<!-- TOPIC_MAP -->" in content:
            content = content.replace("<!-- TOPIC_MAP -->", _topic_map(topics, current))
        target.write_text(content)

    data = {"schema_version": 1, "anchors": anchors, "collections": collections,
            "topics": topics, "pages": pages}
    (stage / "catalog.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    # Stable flat Documenter order; the rendered shell provides the actual
    # editorial hierarchy without duplicating these pages in Documenter groups.
    ordered = list(dict.fromkeys([anchors["home"], anchors["learning"], anchors["topics"], anchors["install"]]
        + [page for collection in collections for page in
           [collection["page"], *[i["page"] for i in collection["items"]],
            *[i["page"] for i in collection["resources"]]]]
        + list(pages)))
    toml = []
    for page in ordered:
        record = pages[page]
        toml.extend(["[[pages]]", "title = " + json.dumps(record["title"], ensure_ascii=False),
                     "source = " + json.dumps(record["source"], ensure_ascii=False), ""])
    (stage / "catalog_pages.toml").write_text("\n".join(toml))
    return data


def _topic_map(topics: list[dict], current: str) -> str:
    parts = ['```@raw html', '<section class="topic-atlas" aria-label="Topics and their articles">',
             '<nav class="topic-clusters" aria-label="Topic clusters">']
    for topic in topics:
        labels = list(dict.fromkeys(a["type_label"] for a in topic["articles"]))
        href = html.escape(_href(current, topic["page"], html_output=True), quote=True)
        count = len(topic["articles"])
        parts.append('<article class="topic-cluster"><h2><a href="' + href + '">'
                     + html.escape(topic["title"]) + '</a></h2><p>'
                     + str(count) + (' treatment' if count == 1 else ' treatments')
                     + '</p><p class="topic-types">'
                     + html.escape(' · '.join(labels)) + '</p></article>')
    parts.extend(['</nav>', '<details class="topic-outline"><summary>Article outline by topic</summary>'])
    for topic in topics:
        href = html.escape(_href(current, topic["page"], html_output=True), quote=True)
        parts.append('<section><h3><a href="' + href + '">' + html.escape(topic["title"]) + '</a></h3><ul>')
        for article in topic["articles"]:
            href = html.escape(_href(current, article["page"], html_output=True), quote=True)
            parts.append('<li><a href="' + href + '">' + html.escape(article["title"]) + '</a> '
                         + '<span class="article-type">' + html.escape(article["type_label"]) + '</span> — '
                         + html.escape(article["question"]) + '</li>')
        parts.append('</ul></section>')
    parts.extend(['</details></section>', '```'])
    return "\n".join(parts)
