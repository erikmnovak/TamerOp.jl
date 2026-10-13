"""Focused conversion/failure-contract checks, independent of Julia compilation."""
import copy
import html
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nbformat as nbf
import publish as publication

from publish import (DOCS, captured_notebook, documenter_markdown, export_lesson,
                     rewrite_links, sha256, validate_capture, validate_outputs)


class PublicationTests(unittest.TestCase):
    def test_reuse_restores_outputs_but_keeps_canonical_prose(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, saved = Path(tmp) / 'source.ipynb', Path(tmp) / 'saved.ipynb'
            nb = self.fixture()
            validate_outputs(nb)
            nbf.write(nb, source)
            nb.cells[0].source = '# Portable download\n\nA rewritten repository link.'
            nbf.write(nb, saved)
            evidence = {'sha256': sha256(source), 'download_sha256': sha256(saved), 'figures': 1}
            restored = captured_notebook(source, saved, evidence)
            self.assertIn('Test lesson', restored.cells[0].source)
            self.assertEqual(restored.cells[1].outputs, nb.cells[1].outputs)
            source.write_text(source.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Changed notebook'):
                captured_notebook(source, saved, evidence)

    def test_reuse_rejects_changed_computation_or_notebook_set(self):
        record = {'package_source_sha256': {'src/TamerOp.jl': 'a'},
                  'docs_manifest_sha256': 'manifest', 'notebooks': [{'source': 'one.ipynb'}]}
        with self.assertRaisesRegex(ValueError, 'Package or documentation environment changed'):
            validate_capture(record, Path('/unused'), {}, {'src/TamerOp.jl': 'b'}, 'manifest')
        with self.assertRaisesRegex(ValueError, 'Notebook publication set changed'):
            validate_capture(record, Path('/unused'), {'notebooks': [{'source': 'two.ipynb'}]},
                             record['package_source_sha256'], 'manifest')

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
            self.assertLess(text.index('display(figure)'), text.index('alt="A checked mathematical figure."'))
            self.assertIn('```julia', text)
            self.assertNotIn('not the figure', text)
            self.assertIn('docs/tutorials/ring.ipynb', text)
            self.assertEqual(len(list(page.parent.rglob('*.png'))), 1)

    def test_optional_visible_captions_keep_literals_for_png_and_svg(self):
        caption = 'Compare [0,2), (1,3] and [0,3]; "quoted" <basis> & coordinates.'
        for mime in ('image/png', 'image/svg+xml'):
            for value in (caption, [caption]):
                with self.subTest(mime=mime, caption_type=type(value).__name__):
                    nb = self.fixture()
                    cell = nb.cells[1]
                    cell.metadata.tamerop.figure_caption = value
                    if mime == 'image/svg+xml':
                        cell.outputs[0].data = {
                            mime: '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"></svg>'}
                    self.assertEqual(validate_outputs(nb), 1)
                    with tempfile.TemporaryDirectory() as tmp:
                        stage = Path(tmp)
                        page = stage / 'tutorials/ring.md'
                        export_lesson(nb, DOCS / 'tutorials/ring.ipynb', page, {}, stage)
                        text = page.read_text()
                        self.assertIn('<figure class="notebook-figure"><a href="', text)
                        self.assertIn('<figcaption>' + html.escape(caption, quote=True)
                                      + '</figcaption></figure>', text)
                        self.assertIn('alt="A checked mathematical figure."', text)
                        self.assertNotIn('<basis>', text)
                        self.assertEqual(text.count('<figcaption>'), 1)

    def test_missing_caption_keeps_original_markup_and_clears_stale_output(self):
        nb = self.fixture()
        nb.cells[1].outputs[0].metadata.figure_caption = 'An earlier caption.'
        validate_outputs(nb)
        self.assertNotIn('figure_caption', nb.cells[1].outputs[0].metadata)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            page = stage / 'tutorials/ring.md'
            export_lesson(nb, DOCS / 'tutorials/ring.ipynb', page, {}, stage)
            text = page.read_text()
            self.assertIn('```@raw html\n<a href="', text)
            self.assertNotIn('<figure', text)
            self.assertNotIn('<figcaption', text)
            self.assertNotIn('An earlier caption.', text)

    def test_invalid_visible_captions_fail(self):
        for caption in (True, 3, {}, [], '', ' ', [''], [3], ['one', 'two']):
            with self.subTest(caption=caption), self.assertRaisesRegex(ValueError, 'figure captions'):
                nb = self.fixture()
                nb.cells[1].metadata.tamerop.figure_caption = caption
                validate_outputs(nb)

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
            self.assertLess(text.index('alt="A checked'), text.index('</details>'))
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


class UnchangedPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.docs = self.root / 'docs'
        self.previous = self.docs / '.build/src/downloads'
        for folder in ('src', 'ext', 'docs/src', 'docs/tutorials', 'docs/build_scripts'):
            (self.root / folder).mkdir(parents=True)
        self.previous.mkdir(parents=True)
        (self.root / 'Project.toml').write_text('name = "Fixture"\n')
        (self.root / 'src/Fixture.jl').write_text('module Fixture\nend\n')
        (self.root / 'src/table.bin').write_bytes(b'checked binary table')
        (self.root / 'ext/FixtureExt.jl').write_text('# Extension fixture\n')
        (self.docs / 'src/index.md').write_text('# Index\n\nA fixture.\n')
        shutil.copy2(DOCS / 'build_scripts/documenter.md.j2', self.docs / 'build_scripts/documenter.md.j2')
        for name in ('Project.toml', 'Manifest.toml', 'requirements.txt'):
            (self.docs / name).write_text('# Pinned fixture environment\n')
        self.config = {'guides': {}, 'notebooks': [
            {'source': f'tutorials/{name}.ipynb', 'page': f'tutorials/{name}.md', 'timeout': 1}
            for name in ('one', 'two')]}
        self.write_config()
        self.runtime = {'julia': 'julia version fixture',
                        'python_packages': {name: '1.0' for name in publication.PYTHON_PACKAGES}}
        self.current = {'git_commit': 'current', 'working_tree_dirty': True}
        self.executed = []
        self.on_execute = lambda: None
        fixture = PublicationTests().fixture()
        records = []
        for lesson in self.config['notebooks']:
            source = self.docs / lesson['source']
            executed = copy.deepcopy(fixture)
            executed.cells[0].source = '# ' + source.stem
            canonical = copy.deepcopy(executed)
            canonical.cells[1].outputs = []
            canonical.cells[1].execution_count = None
            nbf.write(canonical, source)
            nbf.write(executed, self.previous / source.name)
            records.append({'source': lesson['source'], 'sha256': sha256(source),
                            'download_sha256': sha256(self.previous / source.name),
                            'code_cells': 1, 'figures': 1, 'seconds': 1.0})
            exports = self.previous / 'figures' / source.stem
            exports.mkdir(parents=True)
            (exports / 'old.svg').write_text('<svg>old export</svg>')
        environment = self.previous / 'environment'
        environment.mkdir()
        for name in ('Project.toml', 'Manifest.toml', 'requirements.txt', 'publication.toml'):
            shutil.copy2(self.docs / name, environment / name)
        self.addCleanup(patch.stopall)
        patch.multiple(publication, ROOT=self.root, DOCS=self.docs).start()
        self.record = {'notebooks': records, 'git_commit': 'previous', 'working_tree_dirty': False,
                       **copy.deepcopy(self.runtime), 'package_source_sha256': publication.package_source_hashes(),
                       'docs_manifest_sha256': sha256(self.docs / 'Manifest.toml')}
        self.save_record()
        patch.object(publication, 'runtime_versions', side_effect=lambda *args: copy.deepcopy(self.runtime)).start()
        patch.object(publication, 'checkout_state', side_effect=lambda: dict(self.current)).start()
        patch.object(publication, 'prepare_catalog').start()
        patch.object(publication.subprocess, 'run').start()
        outer = self

        class Client:
            def __init__(self, notebook, **kwargs):
                self.nb = notebook
            def create_kernel_manager(self):
                return SimpleNamespace()
            def execute(self):
                outer.executed.append(self.nb.cells[0].source)
                for cell in self.nb.cells:
                    if cell.cell_type == 'code':
                        cell.execution_count = 1
                        cell.outputs = copy.deepcopy(fixture.cells[1].outputs)
                outer.on_execute()

        patch.object(publication, 'NotebookClient', Client).start()

    def write_config(self):
        text = '[guides]\n'
        for lesson in self.config['notebooks']:
            text += '\n[[notebooks]]\n' + ''.join(f'{key} = {json.dumps(value)}\n' for key, value in lesson.items())
        (self.docs / 'publication.toml').write_text(text)

    def save_record(self):
        (self.previous / 'publication.json').write_text(json.dumps(self.record))

    def change_second(self, kind='code'):
        source = self.docs / self.config['notebooks'][1]['source']
        nb = nbf.read(source, as_version=4)
        if kind == 'code': nb.cells[1].source += '\n# Changed computation'
        elif kind == 'prose': nb.cells[0].source += '\n\nChanged prose.'
        elif kind == 'metadata': nb.metadata['purpose'] = 'Changed metadata'
        nbf.write(nb, source)
        if kind == 'bytes': source.write_text(source.read_text() + '\n')

    def build(self, **kwargs):
        publication.publish('julia-fixture', **kwargs)
        return json.loads((self.previous / 'publication.json').read_text())

    def test_mixed_reuse_and_successive_provenance(self):
        self.change_second()
        previous_hash = sha256(self.previous / 'publication.json')
        before = copy.deepcopy(self.record)
        record = self.build(reuse_unchanged=True)
        self.assertEqual(self.executed, ['# two'])
        one, two = record['notebooks']
        self.assertEqual(record['git_commit'], 'current')
        self.assertEqual(one['execution_origin']['git_commit'], 'previous')
        self.assertFalse(one['execution_origin']['working_tree_dirty'])
        self.assertEqual(one['reuse_history'][0]['publication_sha256'], previous_hash)
        self.assertEqual(two['execution_origin']['git_commit'], 'current')
        self.assertEqual(two['reuse_history'], [])
        self.assertTrue((self.previous / 'figures/one/old.svg').is_file())
        self.assertFalse((self.previous / 'figures/two').exists())
        self.assertEqual(self.record, before)
        next_hash = sha256(self.previous / 'publication.json')
        self.current['git_commit'] = 'later'
        self.executed.clear()
        next_record = self.build(reuse_unchanged=True)
        self.assertEqual(self.executed, [])
        self.assertEqual(next_record['git_commit'], 'later')
        for old, new in zip(record['notebooks'], next_record['notebooks']):
            self.assertEqual(new['execution_origin'], old['execution_origin'])
            self.assertEqual(new['reuse_history'][-1]['publication_sha256'], next_hash)
            self.assertEqual(new['reuse_history'][-1]['reused_for_git_commit'], 'later')

    def test_noop_reuse_records_current_build_and_original_executions(self):
        record = self.build(reuse_unchanged=True)
        self.assertEqual(self.executed, [])
        self.assertEqual(record['git_commit'], 'current')
        self.assertEqual([item['execution_origin']['git_commit'] for item in record['notebooks']],
                         ['previous', 'previous'])

    def test_any_notebook_byte_change_executes_fresh(self):
        for kind in ('prose', 'metadata', 'bytes'):
            with self.subTest(kind=kind):
                self.change_second(kind)
                self.executed.clear()
                self.build(reuse_unchanged=True)
                self.assertEqual(len(self.executed), 1)

    def test_strict_mode_still_rejects_changed_notebook(self):
        self.change_second('prose')
        with self.assertRaisesRegex(ValueError, 'Changed notebook'):
            self.build(reuse_executed=True)
        self.assertEqual(self.executed, [])

    def test_strict_mode_keeps_existing_runtime_and_provenance_semantics(self):
        self.runtime['julia'] = 'another runtime'
        self.assertEqual(self.build(reuse_executed=True), self.record)
        self.assertEqual(self.executed, [])

    def test_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'only one'):
            self.build(reuse_executed=True, reuse_unchanged=True)

    def test_corrupt_execution_origin_and_history_fail_closed(self):
        self.record = self.build(reuse_unchanged=True)
        original = copy.deepcopy(self.record['notebooks'][0])
        good = original['reuse_history'][0]
        for history in (None, ['invalid'], [{}], [{**good, 'publication_sha256': 'invalid'}],
                        [{**good, 'publication_git_commit': ''}],
                        [{**good, 'reused_for_working_tree_dirty': 'false'}]):
            with self.subTest(history=history):
                self.record['notebooks'][0] = copy.deepcopy(original)
                self.record['notebooks'][0]['reuse_history'] = history
                self.save_record()
                with self.assertRaisesRegex(ValueError, 'Invalid reuse history'):
                    self.build(reuse_unchanged=True)
        self.record['notebooks'][0] = copy.deepcopy(original)
        self.record['notebooks'][0]['execution_origin']['julia'] = 'wrong execution environment'
        self.save_record()
        with self.assertRaisesRegex(ValueError, 'execution environment differs'):
            self.build(reuse_unchanged=True)
        self.assertEqual(self.executed, [])

    def test_checkout_changes_during_execution_preserve_previous_capture(self):
        self.change_second()
        previous_hash = sha256(self.previous / 'publication.json')
        self.on_execute = lambda: self.current.update(git_commit='different checkout')
        with self.assertRaisesRegex(ValueError, 'Git checkout changed during'):
            self.build(reuse_unchanged=True)
        self.assertEqual(sha256(self.previous / 'publication.json'), previous_hash)

    def test_colliding_capture_names_fail_before_execution(self):
        self.config['notebooks'][1]['source'] = 'other/one.ipynb'
        (self.docs / 'other').mkdir()
        shutil.copy2(self.docs / 'tutorials/two.ipynb', self.docs / 'other/one.ipynb')
        self.write_config()
        shutil.copy2(self.docs / 'publication.toml', self.previous / 'environment/publication.toml')
        self.record['notebooks'][1]['source'] = 'other/one.ipynb'
        self.save_record()
        with self.assertRaisesRegex(ValueError, 'names collide'):
            self.build(reuse_unchanged=True)
        self.assertEqual(self.executed, [])

    def test_changed_lesson_still_requires_a_complete_prior_capture(self):
        self.change_second()
        saved = self.previous / 'two.ipynb'
        original = saved.read_bytes()
        for kind in ('missing', 'hash', 'unexecuted', 'error', 'figure', 'cell_count', 'figure_count'):
            with self.subTest(kind=kind):
                saved.write_bytes(original)
                self.record['notebooks'][1].update(code_cells=1, figures=1)
                if kind == 'missing': saved.unlink()
                elif kind == 'hash': saved.write_bytes(original + b'\n')
                elif kind in ('cell_count', 'figure_count'):
                    self.record['notebooks'][1]['code_cells' if kind == 'cell_count' else 'figures'] = 2
                else:
                    nb = nbf.read(saved, as_version=4)
                    if kind == 'unexecuted': nb.cells[1].execution_count = None
                    elif kind == 'error': nb.cells[1].outputs = [nbf.v4.new_output('error', ename='Failure', evalue='', traceback=[])]
                    else: nb.cells[1].outputs = []
                    nbf.write(nb, saved)
                    self.record['notebooks'][1]['download_sha256'] = sha256(saved)
                self.save_record()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.build(reuse_unchanged=True)
                self.assertEqual(self.executed, [])
                self.record['notebooks'][1]['download_sha256'] = publication.hashlib.sha256(original).hexdigest()

    def test_package_binary_environment_and_runtime_changes_fail_closed(self):
        cases = [(self.root / path) for path in
                 ('Project.toml', 'src/Fixture.jl', 'src/table.bin', 'ext/FixtureExt.jl',
                  'docs/Project.toml', 'docs/Manifest.toml', 'docs/requirements.txt',
                  'docs/.build/src/downloads/environment/Manifest.toml')]
        for path in cases:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                with self.assertRaises(ValueError): self.build(reuse_unchanged=True)
                self.assertEqual(self.executed, [])
                path.write_bytes(original)
        for key in ('julia', 'python_packages'):
            with self.subTest(runtime=key):
                old = copy.deepcopy(self.runtime[key])
                self.runtime[key] = 'changed' if key == 'julia' else {'nbclient': 'changed'}
                with self.assertRaisesRegex(ValueError, 'runtime changed'):
                    self.build(reuse_unchanged=True)
                self.runtime[key] = old

    def test_notebook_set_and_missing_capture_fail_closed(self):
        original = (self.docs / 'publication.toml').read_bytes()
        self.config['notebooks'].pop()
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'publication set changed'):
            self.build(reuse_unchanged=True)
        (self.docs / 'publication.toml').write_bytes(original)
        (self.previous / 'publication.json').unlink()
        with self.assertRaises(FileNotFoundError): self.build(reuse_unchanged=True)
        self.assertEqual(self.executed, [])

    def test_source_changes_during_execution_preserve_the_previous_capture(self):
        self.change_second()
        previous_hash = sha256(self.previous / 'publication.json')
        for kind in ('earlier_notebook', 'new_package_source', 'new_binary', 'environment'):
            with self.subTest(kind=kind):
                path = {'earlier_notebook': self.docs / 'tutorials/one.ipynb',
                        'new_package_source': self.root / 'src/Added.jl',
                        'new_binary': self.root / 'src/added.bin',
                        'environment': self.docs / 'requirements.txt'}[kind]
                original = path.read_bytes() if path.exists() else None
                self.on_execute = lambda p=path, data=original: p.write_bytes((data or b'') + b'\n')
                with self.assertRaisesRegex(ValueError, 'changed during'):
                    self.build(reuse_unchanged=True)
                self.assertEqual(sha256(self.previous / 'publication.json'), previous_hash)
                if original is None: path.unlink()
                else: path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
