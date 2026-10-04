"""Check local HTML links, images and downloadable executed notebook evidence."""
from html.parser import HTMLParser
import json
from pathlib import Path
import tomllib
from urllib.parse import unquote, urlsplit

import nbformat

from publish import DOCS, sha256, validate_outputs
from navigation import continuation_links, published_pages, validate_continuations


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.images = [], set(), 0
        self.optional_sections = 0
        self.continuations, self.map_links = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'details' and 'optional-lesson' in attrs.get('class', '').split():
            self.optional_sections += 1
            if 'open' in attrs:
                raise ValueError('Optional lesson sections should start collapsed')
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])
            classes = attrs.get('class', '').split()
            if 'docs-footer-prevpage' in classes:
                raise ValueError('Sidebar-order previous link remains in the lesson footer')
            if 'docs-footer-nextpage' in classes:
                self.continuations.append(attrs['href'])
            if 'reading-map-return' in classes:
                self.map_links.append(attrs['href'])
        if tag == 'img':
            self.images += 1
            self.links.append(attrs.get('src', ''))
            if not attrs.get('alt', '').strip():
                raise ValueError('Image has no description')


def check_site():
    root = DOCS / 'build'
    html = {}
    for page in root.rglob('*.html'):
        parsed = Links()
        parsed.feed(page.read_text())
        html[page.resolve()] = parsed
    if not html:
        raise ValueError('No generated HTML pages')
    manifest = tomllib.loads((DOCS / 'publication.toml').read_text())
    config = tomllib.loads((DOCS / manifest['reading_map']['source']).read_text())
    records = validate_continuations(config, DOCS)
    pages = published_pages(manifest, DOCS, root)
    nodes = {pages[(DOCS / node['source']).resolve()]: node for node in config['nodes']
             if (DOCS / node['source']).resolve() in pages}
    map_page = root / Path(manifest['reading_map']['page']).with_suffix('.html')
    for page, parsed in html.items():
        expected = Links()
        expected.feed(continuation_links(nodes.get(page), records, pages, page, map_page, DOCS))
        if (parsed.continuations, parsed.map_links) != (expected.continuations, expected.map_links):
            raise ValueError(f'{page.relative_to(root)}: stale or missing reading continuation')
        for link in parsed.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page
            if target.is_dir():
                target = target / 'index.html'
            if not target.is_file():
                raise ValueError(f'{page.relative_to(root)}: broken link {link}')
            if url.fragment and target in html and unquote(url.fragment) not in html[target].ids:
                raise ValueError(f'{page.relative_to(root)}: missing anchor {link}')
    record = json.loads((root / 'downloads/publication.json').read_text())
    for lesson in record['notebooks']:
        source = DOCS / lesson['source']
        download = root / 'downloads' / source.name
        if sha256(source) != lesson['sha256'] or sha256(download) != lesson['download_sha256']:
            raise ValueError(f'Stale source or modified download: {source.name}')
        notebook = nbformat.read(download, as_version=4)
        figures = validate_outputs(notebook)
        page = root / 'tutorials' / f'{source.stem}.html'
        if figures != lesson['figures'] or html[page.resolve()].images < figures:
            raise ValueError(f'Missing published figures: {source.name}')
        sections = [cell.metadata.get('tamerop', {}).get('optional_section') for cell in notebook.cells]
        groups = sum(section is not None and (i == 0 or section != sections[i-1])
                     for i, section in enumerate(sections))
        if html[page.resolve()].optional_sections != groups:
            raise ValueError(f'Missing optional sections: {source.name}')
    print(f'Publication checks passed: {len(html)} HTML pages; reading continuations, local links, anchors, images and notebook downloads.')


if __name__ == '__main__':
    check_site()
