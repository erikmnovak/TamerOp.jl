"""Published-topic and editorial navigation checks without Julia or browsers."""
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest

from site_catalog import prepare_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / "docs"
        self.stage = self.root / "stage"
        self.records = []
        for identity, route in (
                ("home", "index.md"), ("install", "start/install.md"),
                ("reading", "reading_map.md"), ("topics", "topic_map.md"),
                ("contributing", "contributing/index.md"),
                ("mathematics", "collections/mathematics.md"),
                ("using", "collections/using.md")):
            marker = (f"<!-- COLLECTION: {identity} -->" if identity in {"mathematics", "using"}
                      else "<!-- TOPIC_MAP -->" if identity == "topics" else "")
            self.article(identity, f"docs/src/{route}", "resource", content=f"# {identity}\n\n{marker}\n")
        self.guides, self.notebooks = {}, []
        self.navigation = '''schema_version = 1
sidebar_threshold = 2
[anchors]
home = "index.html"
install = "start/install.html"
learning = "reading_map.html"
topics = "topic_map.html"
contributing = "contributing/index.html"
[[collections]]
id = "mathematics"
title = "Mathematics"
page = "collections/mathematics.html"
types = ["lesson"]
order = ["second", "first"]
[[collections]]
id = "using"
title = "Using TamerOp"
page = "collections/using.html"
types = ["library_guide"]
order = []
'''

    def article(self, identity, source, kind, topics=None, state="existing", content="# Article\n"):
        path = self.root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.records.append({"id": identity, "source": source, "type": kind,
            "state": state, "title": identity.replace("_", " ").title(),
            "question": f"What does {identity} show?", "topics": topics or ["algebra"]})

    def stage_catalog(self):
        self.docs.mkdir(exist_ok=True)
        lines = ['schema_version = 1', '[topics]', 'algebra = "Algebra"', 'geometry = "Geometry"']
        for record in self.records:
            lines.append('[[articles]]')
            lines.extend(f"{key} = {json.dumps(value)}" for key, value in record.items())
        (self.docs / "article_inventory.toml").write_text("\n".join(lines))
        lines = ['[guides]'] + [f'{json.dumps(source)} = {json.dumps(page)}'
                                for source, page in self.guides.items()]
        for notebook in self.notebooks:
            lines += ['[[notebooks]]'] + [f'{key} = {json.dumps(value)}' for key, value in notebook.items()]
        (self.docs / "publication.toml").write_text("\n".join(lines))
        (self.docs / "navigation.toml").write_text(self.navigation)
        shutil.copytree(self.docs / "src", self.stage, dirs_exist_ok=True)
        for source, page in self.guides.items():
            target = self.stage / page
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((self.docs / source).read_text())
        return prepare_catalog(self.docs, self.stage)

    def test_only_published_existing_sources_enter_topics_and_collections(self):
        self.article("first", "docs/first.md", "lesson", ["algebra", "geometry"])
        self.article("unpublished", "docs/unpublished.md", "lesson")
        self.article("planned", "docs/planned.md", "lesson", state="planned")
        self.guides["first.md"] = "first.md"
        data = self.stage_catalog()
        ids = {p["id"] for p in data["pages"].values()}
        self.assertIn("first", ids)
        self.assertNotIn("unpublished", ids)
        self.assertNotIn("planned", ids)
        self.assertEqual([t["id"] for t in data["topics"]], ["algebra", "geometry"])
        self.assertTrue(all([a["id"] for a in t["articles"]] == ["first"] for t in data["topics"]))
        self.assertEqual(data["pages"]["first.html"]["canonical_source"], "docs/first.md")

    def test_small_collection_keeps_direct_order_and_large_collection_partitions_by_home_topic(self):
        for identity in ("first", "second"):
            self.article(identity, f"docs/{identity}.md", "lesson")
            self.guides[f"{identity}.md"] = f"{identity}.md"
        for identity, topics in (("a", ["geometry", "algebra"]), ("b", ["algebra"]), ("c", ["geometry"])):
            self.article(identity, f"docs/{identity}.md", "library_guide", topics)
            self.guides[f"{identity}.md"] = f"guides/{identity}.md"
        data = self.stage_catalog()
        math, using = data["collections"]
        self.assertEqual([a["id"] for a in math["items"]], ["second", "first"])
        self.assertEqual(math["topic_groups"], [])
        grouped = [a["id"] for group in using["topic_groups"] for a in group["items"]]
        self.assertCountEqual(grouped, ["a", "b", "c"])
        self.assertEqual(len(grouped), len(set(grouped)))
        group_page = "collections/using/topics/geometry.html"
        self.assertEqual(data["pages"][group_page]["collection"], "using")
        content = (self.stage / group_page.replace(".html", ".md")).read_text()
        self.assertIn("../../../guides/a.md", content)
        self.assertIn("../../../topics/geometry.md", content)
        overview = (self.stage / "collections/using.md").read_text()
        self.assertIn("using/topics/geometry.md", overview)
        self.assertIn("2 articles.", overview)
        self.assertNotIn("guides/a.md", overview)
        self.assertIn("[Second](../second.md)", (self.stage / "collections/mathematics.md").read_text())

    def test_notebook_routes_exist_before_export_and_toml_covers_every_page_once(self):
        self.article("first", "docs/tutorials/first.ipynb", "lesson", content="{}")
        self.notebooks.append({"source": "tutorials/first.ipynb", "page": "tutorials/first.md"})
        data = self.stage_catalog()
        self.assertIn("tutorials/first.html", data["pages"])
        self.assertFalse((self.stage / "tutorials/first.md").exists())
        entries = tomllib.loads((self.stage / "catalog_pages.toml").read_text())["pages"]
        self.assertEqual(len(entries), len(data["pages"]))
        self.assertEqual({e["source"] for e in entries}, {p["source"] for p in data["pages"].values()})

    def test_supporting_resource_is_attached_to_owner_but_excluded_from_topic_articles(self):
        self.article("first", "docs/first.md", "library_guide")
        self.article("data", "docs/data.md", "resource")
        self.guides.update({"first.md": "first.md", "data.md": "data.md"})
        self.navigation += '\n[[resources]]\nid="data"\ncollection="using"\nowner="first"\n'
        data = self.stage_catalog()
        using = data["collections"][1]
        self.assertEqual([item["id"] for item in using["items"]], ["first"])
        self.assertEqual(using["resources"][0]["owner"], "first")
        self.assertEqual(data["pages"]["data.html"]["collection"], "using")
        self.assertEqual([a["id"] for a in data["topics"][0]["articles"]], ["first"])
        self.assertIn("Supporting resources:", (self.stage / "collections/using.md").read_text())

    def test_topic_cards_and_accessible_outline_share_destinations_and_escape_text(self):
        self.article("first", "docs/first.md", "lesson")
        self.records[-1]["title"] = 'Maps < & "spaces"'
        self.guides["first.md"] = "first.md"
        self.stage_catalog()
        content = (self.stage / "topic_map.md").read_text()
        self.assertEqual(content.count('href="topics/algebra.html"'), 2)
        self.assertIn('href="first.html"', content)
        self.assertIn('Maps &lt; &amp; &quot;spaces&quot;', content)
        self.assertIn('<details class="topic-outline">', content)
        self.assertIn("1 treatment</p>", content)
        self.assertNotIn("1 treatments", content)
        self.assertNotIn("<!-- TOPIC_MAP -->", content)

    def test_published_unregistered_or_planned_record_fails(self):
        self.article("planned", "docs/planned.md", "lesson", state="planned")
        self.guides["planned.md"] = "planned.md"
        with self.assertRaisesRegex(ValueError, "lacks an existing article record"):
            self.stage_catalog()

    def test_page_route_escape_and_collision_fail(self):
        self.article("first", "docs/first.md", "lesson")
        self.guides["first.md"] = "../outside.md"
        with self.assertRaisesRegex(ValueError, "Invalid site page"):
            self.stage_catalog()
        self.guides["first.md"] = "index.md"
        with self.assertRaisesRegex(ValueError, "Conflicting publication route"):
            self.stage_catalog()

    def test_duplicate_identity_or_source_fails(self):
        self.article("first", "docs/first.md", "lesson")
        self.records.append(dict(self.records[-1]))
        with self.assertRaisesRegex(ValueError, "Duplicate article identity"):
            self.stage_catalog()
        self.records[-1]["id"] = "different"
        with self.assertRaisesRegex(ValueError, "Duplicate canonical article source"):
            self.stage_catalog()


if __name__ == "__main__":
    unittest.main()
