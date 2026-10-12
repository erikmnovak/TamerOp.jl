"""Render literal figure captions through Documenter, including interval brackets."""
from html.parser import HTMLParser
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

import nbformat as nbf

from publish import DOCS, export_lesson, validate_outputs


class FigureLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.anchor = None
        self.figures = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a':
            self.anchor = attrs.get('href')
        if tag == 'img':
            self.figures.append((attrs['alt'], attrs['src'], self.anchor))

    def handle_endtag(self, tag):
        if tag == 'a':
            self.anchor = None


class FigureRenderingTests(unittest.TestCase):
    def test_literal_captions_keep_png_svg_and_full_size_links(self):
        julia = os.environ.get('TAMEROP_DOCS_TEST_JULIA') or shutil.which('julia')
        if not julia:
            self.skipTest('Documenter figure regression requires Julia and the instantiated docs environment')
        project = Path(os.environ.get('TAMEROP_DOCS_TEST_PROJECT', DOCS))
        captions = [
            'Finite intervals [0,2) and [0,3), and one essential interval.',
            'The second layer is the tent on [0,2), of height 1.',
            'A closed interval [0,2] and a literal \\alpha; "quoted" <basis> & coordinates.',
            'SVG: [a,b), (a,b] and [a,b], with "labels" & <symbols>.',
        ]
        png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='
        cells = [nbf.v4.new_markdown_cell('# Figure caption fixture')]
        for index, caption in enumerate(captions):
            data = ({'image/png': png} if index < 3 else
                    {'image/svg+xml': '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><rect width="4" height="4"/></svg>'})
            cells.append(nbf.v4.new_code_cell('figure', execution_count=index + 1,
                metadata={'tamerop': {'figure_alt': caption}},
                outputs=[nbf.v4.new_output('display_data', data=data)]))
        notebook = nbf.v4.new_notebook(cells=cells, metadata={'language_info': {'name': 'julia'}})
        self.assertEqual(validate_outputs(notebook), 4)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stage, build = directory / 'src', directory / 'build'
            stage.mkdir()
            source = DOCS / 'tutorials/ring.ipynb'
            export_lesson(notebook, source, stage / 'recipes/fixture.md', {}, stage)
            downloads = stage / 'downloads'
            downloads.mkdir()
            (downloads / source.name).write_text('{}')
            (downloads / 'publication.json').write_text('{}')
            script = directory / 'make.jl'
            script.write_text('''using Documenter
makedocs(sitename="Figure caption fixture", source=ARGS[1], build=ARGS[2],
    modules=Module[], doctest=false, remotes=nothing, checkdocs=:none,
    format=Documenter.HTML(prettyurls=false, edit_link=nothing, footer=nothing),
    pages=["Figure caption fixture" => "recipes/fixture.md"])
''')
            result = subprocess.run([julia, '--startup-file=no', '--threads=1',
                f'--project={project}', str(script), str(stage), str(build)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout)
            page = build / 'recipes/fixture.html'
            parsed = FigureLinks()
            parsed.feed(page.read_text())
            self.assertEqual([alt for alt, _, _ in parsed.figures], captions)
            self.assertEqual([Path(urlsplit(src).path).suffix for _, src, _ in parsed.figures],
                             ['.png', '.png', '.png', '.svg'])
            for _, src, href in parsed.figures:
                self.assertEqual(href, src)
                target = page.parent / unquote(urlsplit(src).path)
                self.assertTrue(target.is_file(), f'Missing rendered figure {src}')


if __name__ == '__main__':
    unittest.main()
