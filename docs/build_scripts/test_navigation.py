"""Lesson routes and HTML footer integration, independent of Julia execution."""
import copy
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from navigation import (
    DOCS,
    continuation_links,
    published_pages,
    replace_footer,
    validate_continuations,
)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.docs = Path(self.tmp.name) / "docs"
        self.docs.mkdir()
        self.build = self.docs / "build"

    def routes(self, source="lesson.md"):
        (self.docs / "next.md").write_text("# Next")
        (self.docs / "other.md").write_text("# Other")
        return {
            "nodes": [{"source": source, "title": "Lesson", "next": "next.md",
                       "alternate": "other.md"}],
            "groups": [{"links": [{"source": "next.md", "title": "Next"},
                                   {"source": "other.md", "title": "Other"}]}],
        }

    def test_repository_endings_match_declared_routes(self):
        config = tomllib.loads((DOCS / "reading_map.toml").read_text())
        records = validate_continuations(config, DOCS)
        self.assertEqual(records["tutorials/ring.ipynb"]["next"], "two_parameters.md")
        self.assertEqual(records["two_parameters.md"]["next"], "finite_encodings.md")
        self.assertEqual(records["practical_tameness.md"]["next"], "math_categories.md")

    def test_markdown_route_drift_is_rejected_without_counting_earlier_sections(self):
        config = self.routes()
        source = self.docs / "lesson.md"
        prefix = "# Lesson\n\n[Earlier discussion](unrelated.md)\n\n## Continue\n\n"
        correct = "Read [Next](next.md#result), or [Other](other.md)."
        source.write_text(prefix + correct)
        validate_continuations(config, self.docs)
        for ending in (
            "Read [Other](other.md), then [Next](next.md).",
            "Read [Next](next.md).",
            correct + " Also [another route](next.md).",
            "There is no continuation link.",
        ):
            with self.subTest(ending=ending):
                source.write_text(prefix + ending)
                with self.assertRaisesRegex(ValueError, "update the prose and reading_map.toml together"):
                    validate_continuations(config, self.docs)
        source.write_text(prefix + correct)
        changed = copy.deepcopy(config)
        changed["nodes"][0]["next"] = "other.md"
        changed["nodes"][0]["alternate"] = "next.md"
        with self.assertRaisesRegex(ValueError, "Ending of lesson.md"):
            validate_continuations(changed, self.docs)

    def test_notebook_uses_final_markdown_cell_with_either_source_representation(self):
        config = self.routes("lesson.ipynb")
        ending = "## Continue\n\nRead [Next](next.md), or [Other](other.md)."
        for source in (ending, ending.splitlines(keepends=True)):
            with self.subTest(source_type=type(source).__name__):
                notebook = {"cells": [
                    {"cell_type": "markdown", "source": "[Earlier](unrelated.md)"},
                    {"cell_type": "markdown", "source": source},
                    {"cell_type": "code", "source": "# No prose links here"},
                ]}
                (self.docs / "lesson.ipynb").write_text(json.dumps(notebook))
                validate_continuations(config, self.docs)
                notebook["cells"][1]["source"] = "Read [Other](other.md)."
                (self.docs / "lesson.ipynb").write_text(json.dumps(notebook))
                with self.assertRaisesRegex(ValueError, "Ending of lesson.ipynb"):
                    validate_continuations(config, self.docs)

    def test_unknown_missing_and_self_continuations_fail(self):
        config = self.routes()
        (self.docs / "lesson.md").write_text("# Lesson\n\n[Next](next.md), [Other](other.md).")
        for target, message in (("absent.md", "Unknown continuation"),
                                ("lesson.md", "Self continuation")):
            with self.subTest(target=target):
                changed = copy.deepcopy(config)
                changed["nodes"][0]["next"] = target
                with self.assertRaisesRegex(ValueError, message):
                    validate_continuations(changed, self.docs)
        (self.docs / "next.md").unlink()
        with self.assertRaisesRegex(ValueError, "Unknown continuation"):
            validate_continuations(config, self.docs)

    def test_nested_page_resolves_site_and_repository_continuations(self):
        page = self.build / "tutorials/deep/lesson.html"
        map_page = self.build / "reading_map.html"
        target = self.docs / "next.md"
        records = {"next.md": {"title": 'Spaces & <maps> "together"'}}
        node = {"next": "next.md"}
        pages = {target.resolve(): self.build / "explanations/next steps.html"}
        links = continuation_links(node, records, pages, page, map_page, self.docs)
        self.assertIn('href="../../reading_map.html"', links)
        self.assertIn('href="../../explanations/next%20steps.html"', links)
        self.assertIn('rel="next"', links)
        self.assertIn('Spaces &amp; &lt;maps&gt; &quot;together&quot;', links)

        source = "tutorials/square & maps.ipynb"
        records[source] = {"title": "Inspect spaces and maps"}
        repository_links = continuation_links({"next": source}, records, pages,
                                              page, map_page, self.docs)
        self.assertIn('href="https://github.com/erikmnovak/TamerOp.jl/blob/main/'
                      'docs/tutorials/square%20%26%20maps.ipynb"', repository_links)
        overview_links = continuation_links(None, records, pages, page, map_page, self.docs)
        self.assertIn("Reading map", overview_links)
        self.assertNotIn('rel="next"', overview_links)

    def test_publication_mapping_covers_authored_staged_and_notebook_pages(self):
        install = self.docs / "src/start/install.md"
        install.parent.mkdir(parents=True)
        install.write_text("# Install")
        manifest = {"guides": {"two_parameters.md": "explanations/two_parameters.md"},
                    "notebooks": [{"source": "tutorials/ring.ipynb", "page": "tutorials/ring.md"}]}
        pages = published_pages(manifest, self.docs, self.build)
        self.assertEqual(pages[install.resolve()], self.build / "start/install.html")
        self.assertEqual(pages[(self.docs / "two_parameters.md").resolve()],
                         self.build / "explanations/two_parameters.html")
        self.assertEqual(pages[(self.docs / "tutorials/ring.ipynb").resolve()],
                         self.build / "tutorials/ring.html")

    def test_footer_replacement_preserves_article_credit_and_remaining_html_exactly(self):
        article = ('<!doctype html>\n<article><p>\\(M \\cong \\pi^* N\\) &amp; maps</p>\n'
                   '<pre><code class="language-julia">x &lt; y\n"$literal"</code></pre></article>\n')
        credit = ('<p class="footer-message">Powered by '
                  '<a href="https://example.org/">Documenter &amp; Julia</a>.</p>')
        footer = ('<nav class="docs-footer"><a class="docs-footer-prevpage" href="old.html">Old</a>'
                  '<a class="docs-footer-nextpage" href="wrong.html">Wrong</a>'
                  '<div class="flexbox-break"></div>' + credit + '</nav>')
        suffix = '\n<script>const settings = {math: "a < b"};</script>\n</body></html>'
        links = '<div class="lesson-continuation"><a href="right.html" rel="next">Next</a></div>'
        result = replace_footer(article + footer + suffix, links)
        self.assertEqual(result, article + '<nav class="docs-footer">' + links + credit + '</nav>' + suffix)
        self.assertNotIn('href="old.html"', result)
        self.assertNotIn('href="wrong.html"', result)
        self.assertEqual(replace_footer(result, links), result)

    def test_missing_duplicate_or_misplaced_footer_markup_fails_loudly(self):
        footer = '<nav class="docs-footer"></nav>'
        for text, message in (
            ('<article>No footer</article>', "Missing Documenter footer"),
            ('<nav class="docs-footer">Never closed', "Missing Documenter footer"),
            (footer + footer, "Duplicate docs-footer"),
            ('<p class="footer-message">Misplaced credit</p>' + footer, "outside the footer"),
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, message):
                    replace_footer(text, "new links")

    def test_nested_navigation_and_credit_containers_are_not_closed_early(self):
        credit = ('<div class="footer-message"><div>Powered by '
                  '<a href="https://example.org/">Documenter</a></div>'
                  '<span>and Julia</span></div>')
        text = ('<article>Lesson</article><nav class="docs-footer">'
                '<nav aria-label="Previous ordering"><a href="wrong.html">Wrong</a></nav>'
                '<p>Still inside the footer</p>' + credit + '</nav><aside>Settings</aside>')
        result = replace_footer(text, '<div>New route</div>')
        self.assertEqual(result, '<article>Lesson</article><nav class="docs-footer">'
                         '<div>New route</div>' + credit + '</nav><aside>Settings</aside>')


if __name__ == "__main__":
    unittest.main()
