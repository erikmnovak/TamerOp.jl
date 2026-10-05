"""Check published navigation, local HTML links and executed notebook evidence."""
from html.parser import HTMLParser
import json
from pathlib import Path
import tomllib
from urllib.parse import unquote, urlsplit

import nbformat

from publish import DOCS, sha256, validate_outputs
from navigation import continuation_links, published_pages, validate_continuations
from site_shell import Structure


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


def check_notebooks(manifest, html, root, docs=DOCS):
    """Require an executed download and its configured page for every lesson."""
    expected = {lesson['source']: lesson for lesson in manifest['notebooks']}
    if len(expected) != len(manifest['notebooks']):
        raise ValueError('Duplicate notebook source in publication.toml')
    record = json.loads((root / 'downloads/publication.json').read_text())
    recorded = [lesson['source'] for lesson in record['notebooks']]
    if len(set(recorded)) != len(recorded) or set(recorded) != set(expected):
        raise ValueError('Notebook publication record does not match publication.toml; rebuild every lesson')
    for lesson in record['notebooks']:
        source = docs / lesson['source']
        download = root / 'downloads' / source.name
        if not download.is_file():
            raise ValueError(f'Missing executed download: {source.name}')
        if sha256(source) != lesson['sha256'] or sha256(download) != lesson['download_sha256']:
            raise ValueError(f'Stale source or modified download: {source.name}')
        notebook = nbformat.read(download, as_version=4)
        figures = validate_outputs(notebook)
        if sum(cell.cell_type == 'code' for cell in notebook.cells) != lesson['code_cells']:
            raise ValueError(f'Incorrect executed-cell record: {source.name}')
        page = (root / Path(expected[lesson['source']]['page']).with_suffix('.html')).resolve()
        if page not in html:
            raise ValueError(f'Missing published lesson: {page.relative_to(root)}')
        if figures != lesson['figures'] or html[page].images < figures:
            raise ValueError(f'Missing published figures: {source.name}')
        download_links = [(page.parent / unquote(urlsplit(link).path)).resolve()
                          for link in html[page].links
                          if not urlsplit(link).scheme and not urlsplit(link).netloc]
        if download.resolve() not in download_links:
            raise ValueError(f'Missing executed-notebook link: {source.name}')
        sections = [cell.metadata.get('tamerop', {}).get('optional_section') for cell in notebook.cells]
        groups = sum(section is not None and (i == 0 or section != sections[i-1])
                     for i, section in enumerate(sections))
        if html[page].optional_sections != groups:
            raise ValueError(f'Missing optional sections: {source.name}')


def check_catalog(catalog, root):
    """Check published navigation and local outlines without changing page HTML."""
    root = root.resolve()

    def target(page, href):
        url = urlsplit(href)
        if url.scheme or url.netloc:
            raise ValueError(f'{page.relative_to(root)}: catalog navigation must use local links')
        destination = (page.parent / unquote(url.path)).resolve() if url.path else page
        if not destination.is_relative_to(root):
            raise ValueError(f'{page.relative_to(root)}: catalog link leaves the built site')
        return destination, unquote(url.fragment)

    def published(route):
        destination, fragment = target(root / 'index.html', route)
        if fragment or not destination.is_file() or destination.suffix != '.html':
            raise ValueError(f'Missing published catalog page: {route}')
        return destination

    collections = [item for item in catalog['collections'] if item['id'] != 'contributing']
    identities = [item['id'] for item in collections]
    roots = [item['page'] for item in collections]
    if len(collections) != 6 or len(set(identities)) != 6 or len(set(roots)) != 6:
        raise ValueError('Catalog must define six distinct main collection roots')
    for route in [*catalog['pages'], *roots, *catalog['anchors'].values()]:
        published(route)
    pages = sorted(root.rglob('*.html'))
    extra = {page.relative_to(root).as_posix() for page in pages} - set(catalog['pages'])
    if extra:
        raise ValueError('Built HTML pages are absent from the catalog: ' + ', '.join(sorted(extra)))

    def inside(node, parent):
        return parent.inner <= node.start < parent.stop

    def links(parsed, parent):
        return [node for node in parsed.elements if node.tag == 'a'
                and 'href' in node.attrs and inside(node, parent)]

    structures = {}
    for raw_page in pages:
        page = raw_page.resolve()
        route = page.relative_to(root).as_posix()
        parsed = Structure(page.read_text())
        structures[page] = parsed
        navigation = parsed.one(tag='nav', id='site-navigation')
        entries = parsed.one(class_name='site-entry-links')
        footer = parsed.one(class_name='site-nav-bottom')
        if not inside(entries, navigation) or not inside(footer, navigation):
            raise ValueError(f'{route}: permanent navigation is outside the sidebar')
        actual_entries = [target(page, node.attrs['href']) for node in links(parsed, entries)]
        expected_entries = [(published(catalog['anchors'][key]), '')
                            for key in ('home', 'install', 'learning', 'topics')]
        if actual_entries != expected_entries:
            raise ValueError(f'{route}: missing or reordered permanent entry links')
        contributor = (published(catalog['anchors']['contributing']), '')
        if [target(page, node.attrs['href']) for node in links(parsed, footer)].count(contributor) != 1:
            raise ValueError(f'{route}: missing or duplicated contributor footer link')

        branches = [node for node in parsed.elements if node.tag == 'details'
                    and 'data-collection' in node.attrs and inside(node, navigation)]
        if [node.attrs['data-collection'] for node in branches] != identities:
            raise ValueError(f'{route}: sidebar does not contain the six collection roots')
        requested = catalog['pages'].get(route, {}).get('collection')
        active = [item['id'] for item in collections
                  if requested == item['id'] or route == item['page']]
        opened = [node.attrs['data-collection'] for node in branches if 'open' in node.attrs]
        if opened != active or len(active) > 1:
            raise ValueError(f'{route}: only the current collection may start open')
        current_links = []
        for branch, collection in zip(branches, collections):
            branch_links = links(parsed, branch)
            if [target(page, node.attrs['href']) for node in branch_links].count(
                    (published(collection['page']), '')) != 1:
                raise ValueError(f'{route}: missing or duplicated collection overview link')
            marked = [node for node in branch_links if node.attrs.get('aria-current') == 'page']
            if marked and collection['id'] not in active:
                raise ValueError(f'{route}: current-page marker is outside the current collection')
            current_links.extend(marked)
        if len(current_links) != len(active) or any(
                target(page, node.attrs['href']) != (page, '') for node in current_links):
            raise ValueError(f'{route}: collection current-page marker is missing or incorrect')

        article = parsed.one(tag='article', id='documenter-page')
        headings = [node.attrs['id'] for node in parsed.elements
                    if node.tag in {'h2', 'h3'} and inside(node, article)
                    and node.attrs.get('id') and ''.join(node.text).strip()]
        headings = list(dict.fromkeys(headings))
        outline = parsed.one(class_name='site-outline', required=False)
        if outline is not None and (inside(outline, navigation) or inside(outline, article)):
            raise ValueError(f'{route}: local outline must be separate from collection navigation and article')
        actual_outline = ([target(page, node.attrs['href']) for node in links(parsed, outline)]
                          if outline is not None else [])
        if actual_outline != [(page, heading) for heading in headings]:
            raise ValueError(f'{route}: local outline does not match the article H2/H3 anchors')

    def topic_links(route, required):
        if route not in catalog['pages']:
            raise ValueError(f'Generated topic destination is absent from the catalog: {route}')
        page = published(route)
        parsed = structures.get(page) or Structure(page.read_text())
        article = parsed.one(tag='article', id='documenter-page')
        destinations = set()
        for node in links(parsed, article):
            destination, fragment = target(page, node.attrs['href'])
            if not destination.is_file():
                raise ValueError(f'{route}: broken generated topic link {node.attrs["href"]}')
            if fragment:
                linked = structures.get(destination) or Structure(destination.read_text())
                if not any(item.attrs.get('id') == fragment for item in linked.elements):
                    raise ValueError(f'{route}: missing generated topic anchor {node.attrs["href"]}')
            destinations.add(destination)
        for expected in required:
            if expected not in catalog['pages'] or published(expected) not in destinations:
                raise ValueError(f'{route}: missing published topic treatment {expected}')

    topics = catalog['topics']
    topic_links(catalog['anchors']['topics'], [item['page'] for item in topics])
    for topic in topics:
        topic_links(topic['page'], [item['page'] for item in topic['articles']])
    for collection in collections:
        for group in collection.get('topic_groups', []):
            topic_links(group['page'], [item['page'] for item in group['items']])

    # An orphan group can link to itself and back to the site without giving
    # readers any path into it. Follow actual anchors from the home page.
    visited, pending = set(), [published(catalog['anchors']['home'])]
    while pending:
        page = pending.pop()
        if page in visited:
            continue
        visited.add(page)
        for node in structures[page].elements:
            if node.tag != 'a' or 'href' not in node.attrs:
                continue
            url = urlsplit(node.attrs['href'])
            if url.scheme or url.netloc:
                continue
            destination, _ = target(page, node.attrs['href'])
            if destination.is_dir():
                destination /= 'index.html'
            if destination in structures and destination not in visited:
                pending.append(destination)
    unreachable = set(structures) - visited
    if unreachable:
        routes = sorted(page.relative_to(root).as_posix() for page in unreachable)
        raise ValueError('Catalog pages are unreachable from the home page: ' + ', '.join(routes))


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
    check_notebooks(manifest, html, root)
    check_catalog(json.loads((root / 'catalog.json').read_text()), root)
    print(f'Publication checks passed: {len(html)} HTML pages; catalog coverage and reachability, navigation, local outlines, reading continuations, local links, anchors, images and notebook downloads.')


if __name__ == '__main__':
    check_site()
