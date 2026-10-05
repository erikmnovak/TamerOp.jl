"""Publication and catalog navigation checks with small fixtures, without Julia."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from urllib.parse import quote

import nbformat as nbf

from check_site import Links, check_catalog, check_notebooks
from publish import sha256


class NotebookSiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.docs = Path(temporary.name) / 'docs'
        self.root = self.docs / 'build'
        (self.docs / 'tutorials').mkdir(parents=True)
        (self.root / 'downloads').mkdir(parents=True)
        self.manifest = {'notebooks': [
            {'source': 'tutorials/ring.ipynb', 'page': 'tutorials/ring.md'},
            {'source': 'tutorials/inspect_encoding.ipynb', 'page': 'lessons/square.md'},
        ]}
        self.record = {'notebooks': []}
        self.html = {}
        png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='
        for item in self.manifest['notebooks']:
            source = self.docs / item['source']
            notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(
                'figure', metadata={'tamerop': {'figure_alt': 'The computed result.'}})])
            optional = source.stem == 'inspect_encoding'
            if optional:
                notebook.cells.append(nbf.v4.new_markdown_cell(
                    '## Optional: Live inspection\n\n```julia\nusing WGLMakie\n```',
                    metadata={'tamerop': {'optional_section': 'Optional: Live inspection'}}))
            nbf.write(notebook, source)
            notebook.cells[0].execution_count = 1
            notebook.cells[0].outputs = [nbf.v4.new_output('display_data', data={'image/png': png})]
            download = self.root / 'downloads' / source.name
            nbf.write(notebook, download)
            self.record['notebooks'].append({
                'source': item['source'], 'sha256': sha256(source),
                'download_sha256': sha256(download), 'code_cells': 1, 'figures': 1,
            })
            page = (self.root / Path(item['page']).with_suffix('.html')).resolve()
            parsed = Links()
            parsed.feed(f'<a href="../downloads/{source.name}">Download</a>'
                        '<img src="figure.png" alt="The computed result.">'
                        + ('<details class="optional-lesson"></details>' if optional else ''))
            self.html[page] = parsed
        self.write_record()

    def write_record(self, record=None):
        (self.root / 'downloads/publication.json').write_text(json.dumps(record or self.record))

    def check(self, html=None):
        check_notebooks(self.manifest, self.html if html is None else html, self.root, self.docs)

    def test_every_lesson_uses_its_configured_page_and_keeps_optional_live_prose(self):
        self.check()

    def test_missing_extra_or_duplicate_publication_records_fail(self):
        for mutation in ('missing', 'extra', 'duplicate'):
            changed = copy.deepcopy(self.record)
            if mutation == 'missing':
                changed['notebooks'].pop()
            else:
                item = copy.deepcopy(changed['notebooks'][0])
                if mutation == 'extra':
                    item['source'] = 'tutorials/unknown.ipynb'
                changed['notebooks'].append(item)
            with self.subTest(mutation=mutation):
                self.write_record(changed)
                with self.assertRaisesRegex(ValueError, 'does not match publication.toml'):
                    self.check()

    def test_missing_square_page_figure_optional_section_or_download_link_fails(self):
        page = self.root / 'lessons/square.html'
        for mutation, message in [('page', 'Missing published lesson'),
                                  ('figure', 'Missing published figures'),
                                  ('optional', 'Missing optional sections'),
                                  ('link', 'Missing executed-notebook link')]:
            changed = copy.deepcopy(self.html)
            if mutation == 'page':
                del changed[page]
            elif mutation == 'figure':
                changed[page].images = 0
            elif mutation == 'optional':
                changed[page].optional_sections = 0
            else:
                changed[page].links = []
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                self.check(changed)

    def test_stale_square_source_or_missing_download_fails(self):
        source = self.docs / self.manifest['notebooks'][1]['source']
        original = source.read_bytes()
        source.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'Stale source or modified download'):
            self.check()
        source.write_bytes(original)
        (self.root / 'downloads' / source.name).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing executed download'):
            self.check()

    def test_wrong_executed_cell_record_fails(self):
        self.record['notebooks'][1]['code_cells'] += 1
        self.write_record()
        with self.assertRaisesRegex(ValueError, 'Incorrect executed-cell record'):
            self.check()


class CatalogSiteTests(unittest.TestCase):
    """Small HTML fixtures exercise reader navigation contracts, without Documenter."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.anchors = {'home': 'index.html', 'install': 'start/install.html',
                        'learning': 'reading_map.html', 'topics': 'topic_map.html',
                        'contributing': 'contributing/index.html'}
        self.collections = [{'id': identity, 'page': f'collections/{identity}.html'}
                            for identity in ('mathematics', 'using', 'api', 'implementation', 'benchmarks')]
        self.catalog = {'anchors': self.anchors, 'collections': self.collections + [
            {'id': 'contributing', 'page': self.anchors['contributing']}],
            'pages': {route: {} for route in self.anchors.values()},
            'topics': [{'id': 'encodings', 'page': 'topics/encodings.html',
                        'articles': [{'page': 'lessons/module.html'}]}]}
        self.catalog['pages'].update({item['page']: {'collection': item['id']}
                                      for item in self.collections})
        self.catalog['pages']['lessons/module.html'] = {'collection': 'mathematics'}
        self.catalog['pages']['topics/encodings.html'] = {}
        self.catalog['pages']['contributing/index.html']['collection'] = 'contributing'
        for route in self.catalog['pages']:
            self.write_page(route)

    def write_page(self, route):
        page = self.root / route
        page.parent.mkdir(parents=True, exist_ok=True)

        def link(destination, current=False):
            href = quote(os.path.relpath(self.root / destination, page.parent))
            marker = ' aria-current="page"' if current else ''
            return f'<a href="{href}"{marker}>Destination</a>'

        active = self.catalog['pages'][route].get('collection')
        entries = ''.join(link(self.anchors[key], route == self.anchors[key])
                          for key in ('home', 'install', 'learning', 'topics'))
        branches = []
        for collection in self.collections:
            identity, overview = collection['id'], collection['page']
            shown = link(overview, route == overview)
            if identity == 'mathematics':
                shown += link('lessons/module.html', route == 'lessons/module.html')
            opened = ' open' if active == identity else ''
            branches.append(f'<details data-collection="{identity}"{opened}>'
                            f'<summary>Collection</summary>{shown}</details>')
        outline, article = '', '<h1>Article</h1>'
        if route == 'lessons/module.html':
            article += '<h2 id="spaces">Spaces</h2><h3 id="maps-α">Maps α</h3>'
            outline = ('<aside class="site-outline"><a href="#spaces">Spaces</a>'
                       '<a href="#maps-%CE%B1">Maps</a></aside>')
        elif route == 'topic_map.html':
            article += link('topics/encodings.html')
        elif route == 'topics/encodings.html':
            article += link('lessons/module.html') + link('topic_map.html')
        page.write_text('<html><body><nav id="site-navigation">'
                        f'<div class="site-entry-links">{entries}</div>'
                        + ''.join(branches)
                        + '<div class="site-nav-bottom">'
                        + link(self.anchors['contributing'], route == self.anchors['contributing'])
                        + f'</div></nav>{outline}<article id="documenter-page">{article}</article>'
                        '</body></html>')

    def check(self, catalog=None):
        check_catalog(self.catalog if catalog is None else catalog, self.root)

    def test_published_pages_with_and_without_a_collection_pass_without_modification(self):
        before = {page: page.read_bytes() for page in self.root.rglob('*.html')}
        self.check()
        self.assertEqual(before, {page: page.read_bytes() for page in before})

    def test_missing_catalog_page_or_collection_root_fails(self):
        changed = copy.deepcopy(self.catalog)
        changed['pages']['lessons/missing.html'] = {}
        with self.assertRaisesRegex(ValueError, 'Missing published catalog page'):
            self.check(changed)
        changed = copy.deepcopy(self.catalog)
        changed['collections'].pop(0)
        with self.assertRaisesRegex(ValueError, 'five distinct main collection roots'):
            self.check(changed)

    def test_missing_permanent_or_contributor_link_fails(self):
        page = self.root / 'index.html'
        original = page.read_text()
        for old, new, message in [
            ('href="start/install.html"', 'href="reading_map.html"', 'permanent entry links'),
            ('href="contributing/index.html"', 'href="index.html"', 'contributor footer link'),
        ]:
            with self.subTest(message=message):
                page.write_text(original.replace(old, new))
                with self.assertRaisesRegex(ValueError, message):
                    self.check()
        page.write_text(original)

    def test_sidebar_cannot_omit_a_catalog_collection(self):
        page = self.root / 'index.html'
        page.write_text(page.read_text().replace('data-collection="api"', 'data-collection="unknown"'))
        with self.assertRaisesRegex(ValueError, 'five collection roots'):
            self.check()

    def test_resource_assigned_to_a_main_collection_needs_current_page_context(self):
        route = 'implementation/bibliography.html'
        self.catalog['pages'][route] = {'collection': 'implementation', 'type': 'resource'}
        self.write_page(route)
        with self.assertRaisesRegex(ValueError, 'current-page marker'):
            self.check()

    def test_wrong_open_branch_or_current_page_marker_fails(self):
        page = self.root / 'lessons/module.html'
        original = page.read_text()
        for changed, message in [
            (original.replace('data-collection="using"', 'data-collection="using" open'),
             'only the current collection'),
            (original.replace(' aria-current="page"', ''), 'current-page marker'),
            (original.replace('href="module.html" aria-current="page"',
                              'href="../collections/mathematics.html" aria-current="page"'),
             'collection overview link'),
        ]:
            with self.subTest(message=message):
                page.write_text(changed)
                with self.assertRaisesRegex(ValueError, message):
                    self.check()
        page.write_text(original)

    def test_outline_must_follow_local_h2_h3_anchors_and_stay_outside_the_sidebar(self):
        page = self.root / 'lessons/module.html'
        original = page.read_text()
        for changed, message in [
            (original.replace('href="#spaces"', 'href="#absent"'), 'H2/H3 anchors'),
            (original.replace('href="#spaces"', 'href="../reading_map.html#spaces"'), 'H2/H3 anchors'),
            (original.replace('</div></nav><aside', '</div><aside')
                     .replace('</aside><article', '</aside></nav><article'), 'must be separate'),
        ]:
            with self.subTest(message=message):
                page.write_text(changed)
                with self.assertRaisesRegex(ValueError, message):
                    self.check()
        page.write_text(original)

    def test_topic_overview_and_treatments_must_resolve_to_published_pages(self):
        for route, old, new, message in [
            ('topic_map.html', 'href="topics/encodings.html"', 'href="reading_map.html"',
             'missing published topic treatment'),
            ('topics/encodings.html', 'href="../lessons/module.html"', 'href="../reading_map.html"',
             'missing published topic treatment'),
            ('topics/encodings.html', 'href="../lessons/module.html"', 'href="missing.html"',
             'broken generated topic link'),
        ]:
            page = self.root / route
            original = page.read_text()
            with self.subTest(route=route, message=message):
                page.write_text(original.replace(old, new))
                with self.assertRaisesRegex(ValueError, message):
                    self.check()
            page.write_text(original)

    def test_collection_topic_groups_require_their_published_listing(self):
        changed = copy.deepcopy(self.catalog)
        changed['collections'][0]['topic_groups'] = [
            {'page': 'collections/mathematics/topics/encodings.html',
             'items': [{'page': 'lessons/module.html'}]}]
        with self.assertRaisesRegex(ValueError, 'Generated topic destination is absent'):
            self.check(changed)


if __name__ == '__main__':
    unittest.main()
