"""Reading routes, link destinations, and malformed-data checks without Julia."""
import copy
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import tomllib
import unittest

from reading_map import render_reading_map


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.nodes = {}
        self.cards = {}
        self.edges = set()
        self.hrefs = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            self.hrefs.append(attrs["href"])
            if "data-node" in attrs:
                self.nodes[attrs["data-node"]] = attrs["href"]
                if "reading-map-node" in attrs.get("class", "").split():
                    self.cards[attrs["data-node"]] = attrs
        if tag == "path" and "data-source" in attrs:
            self.edges.add((attrs["data-source"], attrs["data-target"]))


class ReadingMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.docs, self.stage = self.root / "docs", self.root / "stage"
        self.docs.mkdir()
        self.stage.mkdir()
        self.pages = {}
        self.config = {"width": 600, "height": 600, "node_width": 160, "node_height": 100,
                       "nodes": [], "edges": []}
        # Two legitimate entry routes converge, then lead to a single final page.
        for identity, x, y in (("ring", 0, 0), ("theory", 300, 0), ("encoding", 150, 200), ("maps", 150, 400)):
            source = self.docs / (identity + ".md")
            source.write_text("# " + identity)
            self.pages[source] = self.stage / "explanations" / source.name
            self.config["nodes"].append({"id": identity, "title": identity.title(), "question": "What can I learn?",
                "source": source.name, "kind": "Explanation", "x": x, "y": y})
        self.config["edges"] = [self.edge("ring", "encoding"), self.edge("theory", "encoding"), self.edge("encoding", "maps")]

    @staticmethod
    def edge(source, target):
        return {"source": source, "target": target, "path": "M 80 100 C 80 150 230 150 230 200"}

    def render(self, config=None):
        return render_reading_map(config or self.config, self.pages, self.stage,
                                  self.stage / "reading_map.md", self.docs,
                                  "https://example.org/repo/blob/main/")

    def test_alternative_routes_converge_and_outline_keeps_all_continuations(self):
        result = self.render()
        parsed = Links(result)
        self.assertEqual(parsed.nodes, {key: f"explanations/{key}.html" for key in ("ring", "theory", "encoding", "maps")})
        self.assertEqual(result.count('data-entry="true"'), 2)
        outline = result.split('<details class="reading-map-outline">')[1]
        self.assertEqual(outline.count('Continue to: <a href="explanations/encoding.html">Encoding</a>'), 2)
        self.assertIn('Continue to: <a href="explanations/maps.html">Maps</a>', outline)
        self.assertEqual(outline.count("End of this route."), 1)
        self.assertNotIn("<script", result)

    def test_disconnected_entry_is_an_honest_separate_route(self):
        self.config["edges"] = [self.edge("ring", "encoding"), self.edge("encoding", "maps")]
        result = self.render()
        self.assertEqual(result.count('data-entry="true"'), 2)
        self.assertEqual(result.count("End of this route."), 2)

    def test_declared_primary_and_alternate_reach_nodes_and_supporting_guides(self):
        self.config["nodes"][0].update(next="encoding.md", alternate="theory.md")
        self.config["edges"].append(self.edge("ring", "theory"))
        self.config["nodes"][1]["entry"] = True
        (self.docs / "algebra.md").write_text("# Algebra")
        self.config["groups"] = [{"id": "algebra", "title": "Algebra", "description": "Go further.",
                                  "links": [{"title": "Algebra guide", "source": "algebra.md"}]}]
        self.config["nodes"][-1]["next"] = "algebra.md"
        outline = self.render().split('<details class="reading-map-outline">')[1]
        self.assertIn('Continue to: <a href="explanations/encoding.html">Encoding</a>. '
                      'Alternatively: <a href="explanations/theory.html">Theory</a>.', outline)
        self.assertIn('Continue to: <a href="https://example.org/repo/blob/main/docs/algebra.md">Algebra guide</a>.', outline)
        self.assertEqual(self.render().count('data-entry="true"'), 2)
        self.config["edges"].pop()
        with self.assertRaisesRegex(ValueError, "Missing reading map continuation arrow"):
            self.render()

    def test_entry_labels_remain_visible_when_an_optional_route_reaches_an_entry(self):
        self.config["nodes"][0].update(entry=True, entry_label="Start with an example")
        self.config["nodes"][1].update(entry=True, entry_label="Start with definitions")
        self.config["edges"].append(dict(self.edge("ring", "theory"), kind="optional",
            label="Explore the full definitions", label_x=250, label_y=150))
        result = self.render()
        parsed = Links(result)
        self.assertEqual({key for key, card in parsed.cards.items() if card["data-entry"] == "true"},
                         {"ring", "theory"})
        mobile = result.split('<div class="reading-map-mobile-starts"')[1].split('</div>')[0]
        outline = result.split('<details class="reading-map-outline">')[1]
        for label in ("Start with an example", "Start with definitions"):
            self.assertIn(f'<span class="reading-map-entry-label">{label}</span>', result)
            self.assertIn(label, mobile)
            self.assertIn(label, outline)

    def add_setup_and_reference(self):
        for identity in ("install", "reference"):
            source = self.docs / (identity + ".md")
            source.write_text("# " + identity)
            self.pages[source] = self.stage / source.name
            self.config["nodes"].append({"id": identity, "title": identity.title(),
                "source": source.name, "question": "How do I use this lesson?", "kind": "Guide",
                "display": "support" if identity == "install" else "reference", "entry": False,
                "next": "ring.md" if identity == "install" else "encoding.md"})
        self.config["nodes"][-2].update(label="To run the notebook", x=0, y=520, width=200, height=60)
        self.config["nodes"][0].update(next="encoding.md", alternate="reference.md")

    def test_setup_and_reference_keep_links_without_becoming_diagram_steps(self):
        self.add_setup_and_reference()
        result = self.render()
        parsed = Links(result)
        self.assertNotIn("install", parsed.cards)
        self.assertNotIn("reference", parsed.cards)
        self.assertEqual(parsed.nodes["install"], "install.html")
        self.assertIn("reference.html", parsed.hrefs)
        self.assertFalse(any({"install", "reference"} & set(edge) for edge in parsed.edges))
        outline = result.split('<details class="reading-map-outline">')[1]
        self.assertIn("To run the notebook", outline)
        self.assertIn("Supporting reference", outline)
        self.assertIn('Continue to: <a href="explanations/ring.html">Ring</a>', outline)
        self.assertIn('Alternatively: <a href="reference.html">Reference</a>', outline)

    def test_hidden_continuations_are_checked_for_cycles_and_unknown_destinations(self):
        self.add_setup_and_reference()
        for update, message in (({"next": "ring.md"}, "cycle"),
                                ({"next": "missing.md"}, "Unknown reading map continuation")):
            with self.subTest(update=update):
                config = copy.deepcopy(self.config)
                config["nodes"][-1].update(update)
                with self.assertRaisesRegex(ValueError, message):
                    self.render(config)
        self.config["edges"].append(self.edge("install", "ring"))
        with self.assertRaisesRegex(ValueError, "lesson cards only"):
            self.render()

    def test_optional_detour_is_labeled_escaped_and_has_valid_label_coordinates(self):
        edge = self.config["edges"][0]
        edge.update(kind="optional", label='Explore <definitions> & "maps"', label_x=250, label_y=150)
        result = self.render()
        self.assertIn('class="reading-map-edge reading-map-optional"', result)
        self.assertIn('Explore &lt;definitions&gt; &amp; &quot;maps&quot;', result)
        self.assertIn("Dashed line: optional detour", result)
        for update in ({"label": " "}, {"label_x": float("nan")}, {"label_y": float("inf")},
                       {"label_x": True}, {"label_y": "150"}, {"label_x": -1}, {"label_y": 601}):
            with self.subTest(update=update):
                config = copy.deepcopy(self.config)
                config["edges"][0].update(update)
                with self.assertRaises(ValueError):
                    self.render(config)

    def test_published_map_offers_two_equal_starts_with_optional_full_definitions(self):
        docs = Path(__file__).resolve().parents[1]
        with (docs / "reading_map.toml").open("rb") as stream:
            config = tomllib.load(stream)
        nodes = {node["id"]: node for node in config["nodes"]}
        starts = {key for key, node in nodes.items() if node.get("entry")}
        self.assertEqual(starts, {"ring", "modules"})
        self.assertEqual(nodes["ring"]["y"], nodes["modules"]["y"])
        self.assertEqual(nodes["ring"]["y"], min(node["y"] for node in nodes.values()
                                               if node.get("display", "card") == "card"))
        self.assertEqual(nodes["install"]["display"], "support")
        self.assertEqual(nodes["ordinary"]["display"], "reference")
        result = render_reading_map(config, {}, self.stage, self.stage / "reading_map.md", docs,
                                    "https://example.org/repo/blob/main/")
        parsed = Links(result)
        self.assertEqual({key for key, card in parsed.cards.items() if card["data-entry"] == "true"}, starts)
        self.assertFalse(any({"install", "ordinary"} & set(edge) for edge in parsed.edges))
        detour = next(edge for edge in config["edges"]
                      if (edge["source"], edge["target"]) == ("bridge", "modules"))
        self.assertEqual(detour["kind"], "optional")
        self.assertEqual(detour["label"], "Explore the full definitions")
        self.assertIn(("bridge", "encoding"), parsed.edges)
        self.assertIn(("modules", "encoding"), parsed.edges)

    def test_repository_and_download_destinations_are_explicit_and_escaped(self):
        source = self.docs / 'a notebook & notes.ipynb'
        source.write_text('{}')
        self.config["nodes"][0].update(source=source.name, kind="Notebook", title='<Read "this"> & learn')
        self.config["groups"] = [{"id": "further", "title": "More & more", "description": "<Look here>",
                                  "links": [{"title": "Notebook", "source": source.name}]}]
        result = self.render()
        self.assertIn('Repository notebook', result)
        self.assertEqual(Links(result).nodes["ring"], 'https://example.org/repo/blob/main/docs/a%20notebook%20%26%20notes.ipynb')
        self.assertIn('&lt;Read &quot;this&quot;&gt; &amp; learn', result)
        self.assertIn('&lt;Look here&gt;', result)
        self.pages[source] = self.stage / "downloads" / source.name
        self.assertEqual(Links(self.render()).nodes["ring"], 'downloads/a%20notebook%20%26%20notes.ipynb')

    def test_missing_and_escaping_sources_and_staged_destinations_fail(self):
        for source, message in (("missing.md", "Missing"), ("../outside.md", "escapes")):
            with self.subTest(source=source):
                config = copy.deepcopy(self.config)
                config["nodes"][0]["source"] = source
                with self.assertRaisesRegex(ValueError, message):
                    self.render(config)
        self.pages[self.docs / "ring.md"] = self.root / "outside.md"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.render()

    def test_cycle_unknown_endpoint_self_edge_and_duplicate_edge_fail(self):
        for edge, message in ((self.edge("maps", "ring"), "cycle"),
                              (self.edge("maps", "absent"), "endpoint"),
                              (self.edge("maps", "maps"), "self"),
                              (self.edge("ring", "encoding"), "Duplicate")):
            with self.subTest(edge=edge):
                config = copy.deepcopy(self.config)
                config["edges"].append(edge)
                with self.assertRaisesRegex(ValueError, message):
                    self.render(config)

    def test_duplicate_nodes_overlap_and_canvas_overflow_fail(self):
        for update, message in (({"id": "ring"}, "Duplicate"), ({"x": 100, "y": 0}, "overlap"),
                                ({"x": 500}, "outside"), ({"x": float("nan")}, "finite")):
            with self.subTest(update=update):
                config = copy.deepcopy(self.config)
                config["nodes"][1].update(update)
                with self.assertRaisesRegex(ValueError, message):
                    self.render(config)

    def test_invalid_svg_cannot_break_out_of_path_attribute(self):
        for path in ('M 0 0"/><script>alert(1)</script>', "M 0", "L 0 0", "M 0 0 C 1 2", "M 0 0 L 1e999 3"):
            with self.subTest(path=path):
                config = copy.deepcopy(self.config)
                config["edges"][0]["path"] = path
                with self.assertRaises(ValueError):
                    self.render(config)


if __name__ == "__main__":
    unittest.main()
