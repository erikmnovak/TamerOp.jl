"""Focused conversion/failure-contract checks, independent of Julia compilation."""
import copy
from pathlib import Path
import tempfile
import unittest

import nbformat as nbf

from publish import DOCS, documenter_markdown, export_lesson, rewrite_links, validate_outputs


class PublicationTests(unittest.TestCase):
    def test_math_and_code_remain_distinct(self):
        text = 'Read $[0,5)$ and `$not_math`.\n\n$$H_1(K_t)$$\n\n```julia\n"$literal"\n```'
        result = documenter_markdown(text)
        self.assertIn('``[0,5)``', result)
        self.assertIn('```math\nH_1(K_t)\n```', result)
        self.assertIn('`$not_math`', result)
        self.assertIn('```julia\n"$literal"\n```', result)

    def test_mathematical_evaluation_is_not_a_link(self):
        text = '$$\\mathbb{k}[U](q)$$ and `[x](not-a-file)`'
        self.assertEqual(text, rewrite_links(text, DOCS / 'indicator_presentations.md',
                                            None, {}, DOCS))

    def test_existing_github_anchors_survive_site_migration(self):
        result = documenter_markdown('## How the presentation leads to a finite encoding\n\n## Repeat\n\n## Repeat')
        self.assertIn('(@id how-the-presentation-leads-to-a-finite-encoding)', result)
        self.assertIn('(@id repeat-1)', result)

    def test_staged_and_portable_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            source = DOCS / 'tutorials/ring.ipynb'
            target = (DOCS / 'finite_encodings.md').resolve()
            pages = {target: stage / 'finite_encodings.md'}
            text = '[Next](../finite_encodings.md)'
            self.assertEqual(text, rewrite_links(text, source, stage / 'tutorials/ring.md', pages, stage))
            self.assertIn('https://github.com/', rewrite_links(text, source, None, pages, stage))
            with self.assertRaisesRegex(ValueError, 'Broken source link'):
                rewrite_links('[Missing](absent.md)', source, None, pages, stage)

    def test_benchmark_figures_and_data_stay_local_on_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            source = DOCS / 'benchmarks/qpa.md'
            page = stage / 'benchmarks/qpa.md'
            for name in ['performance_profile.svg', 'timings.csv', 'provenance.json', 'SHA256SUMS']:
                text = '[Result](qpa_v1/' + name + ')'
                self.assertEqual(text, rewrite_links(text, source, page, {}, stage))
                copied = stage / 'benchmarks/qpa_v1' / name
                self.assertEqual(copied.read_bytes(), (DOCS / 'benchmarks/qpa_v1' / name).read_bytes())
                self.assertIn('https://github.com/', rewrite_links(text, source, None, {}, stage))

    def fixture(self):
        # A tiny valid PNG represents a captured figure; Julia integration supplies real ones.
        png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='
        cell = nbf.v4.new_code_cell('display(figure)', execution_count=1,
            metadata={'tamerop': {'figure_alt': 'A checked mathematical figure.'}},
            outputs=[nbf.v4.new_output('display_data', data={'image/png': png, 'text/html': '<b>not the figure</b>'})])
        return nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# Test lesson\n\nA result.'), cell],
            metadata={'language_info': {'name': 'julia'}})

    def test_missing_execution_errors_and_figures_fail(self):
        nb = self.fixture()
        self.assertEqual(validate_outputs(nb), 1)
        for mutation in ('unexecuted', 'missing', 'error', 'undescribed'):
            broken = copy.deepcopy(nb)
            cell = broken.cells[1]
            if mutation == 'unexecuted': cell.execution_count = None
            if mutation == 'missing': cell.outputs = []
            if mutation == 'error': cell.outputs = [nbf.v4.new_output('error', ename='AssertionError', evalue='', traceback=[])]
            if mutation == 'undescribed': cell.metadata = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_outputs(broken)

    def test_conversion_keeps_output_order_and_alt_text(self):
        nb = self.fixture()
        validate_outputs(nb)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            page = stage / 'tutorials/ring.md'
            export_lesson(nb, DOCS / 'tutorials/ring.ipynb', page, {}, stage)
            text = page.read_text()
            self.assertLess(text.index('display(figure)'), text.index('![A checked mathematical figure.]'))
            self.assertIn('```julia', text)
            self.assertNotIn('not the figure', text)
            self.assertIn('docs/tutorials/ring.ipynb', text)
            self.assertEqual(len(list(page.parent.rglob('*.png'))), 1)

    def test_optional_section_preserves_cells_figures_and_reading_order(self):
        nb = self.fixture()
        title = 'Optional: Style & export <figures>'
        nb.cells.insert(1, nbf.v4.new_markdown_cell('## Optional styling',
            metadata={'tamerop': {'optional_section': title}}))
        nb.cells[2].metadata.tamerop.optional_section = title
        nb.cells.append(nbf.v4.new_markdown_cell('## Continue the main lesson'))
        original = copy.deepcopy(nb)
        validate_outputs(nb)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            page = stage / 'tutorials/ring.md'
            export_lesson(nb, DOCS / 'tutorials/ring.ipynb', page, {}, stage)
            text = page.read_text()
            self.assertEqual(text.count('<details class="optional-lesson">'), 1)
            self.assertIn('Style &amp; export &lt;figures&gt;</summary>', text)
            self.assertNotIn('<details open', text)
            self.assertLess(text.index('<details'), text.index('```julia'))
            self.assertLess(text.index('![A checked'), text.index('</details>'))
            self.assertLess(text.index('</details>'), text.index('Continue the main lesson'))
            self.assertEqual(nb.cells[2].source, original.cells[2].source)
            self.assertEqual(len(nb.cells), len(original.cells))
            self.assertNotIn('<details', nb.cells[1].source)
            # A group ending the notebook must still close correctly.
            nb.cells.pop()
            export_lesson(nb, DOCS / 'tutorials/ring.ipynb', page, {}, stage)
            self.assertTrue(page.read_text().rstrip().endswith('</details>\n```'))

    def test_invalid_optional_section_fails(self):
        for title in ('', ' ', True):
            nb = self.fixture()
            nb.cells[1].metadata.tamerop.optional_section = title
            with tempfile.TemporaryDirectory() as tmp, self.subTest(title=title):
                stage = Path(tmp)
                with self.assertRaisesRegex(ValueError, 'optional_section'):
                    export_lesson(nb, DOCS / 'tutorials/ring.ipynb', stage / 'ring.md', {}, stage)

    def test_optional_heading_is_not_repeated_but_keeps_its_anchor(self):
        nb = self.fixture()
        title = 'Optional: Prepare a figure'
        nb.cells.insert(1, nbf.v4.new_markdown_cell('### ' + title + '\n\nKeep this explanation.',
            metadata={'tamerop': {'optional_section': title}}))
        nb.cells[2].metadata.tamerop.optional_section = title
        validate_outputs(nb)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            page = stage / 'ring.md'
            export_lesson(nb, DOCS / 'tutorials/ring.ipynb', page, {}, stage)
            text = page.read_text()
            self.assertEqual(text.count(title), 1)
            self.assertIn('id="optional-prepare-a-figure"', text)
            self.assertIn('Keep this explanation.', text)
            self.assertTrue(nb.cells[1].source.startswith('### ' + title))


if __name__ == '__main__':
    unittest.main()
