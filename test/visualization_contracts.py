#!/usr/bin/env python3
"""Validate and publish the visualization acceptance matrix; select its native owners."""
from __future__ import annotations

import argparse
from datetime import date
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/visualization_support.toml'


def load_and_validate(root=ROOT):
    data = tomllib.loads((root / 'docs/visualization_support.toml').read_text())
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported visualization support schema')
    if snapshot := data.get('verification'):
        date.fromisoformat(snapshot['date'])
        for key in ('environment', 'native', 'live', 'offline'):
            if not isinstance(snapshot[key], str) or not snapshot[key].strip():
                raise ValueError(f'Verification snapshot needs {key}')
    families = data['family']
    ids = [row['id'] for row in families]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate visualization family ID')
    covered = set(data['core_sources'])
    native = set(data['shared_native_tests'])
    browser = set(data['shared_browser_tests'])
    for row in families:
        for key in ('id', 'title', 'scope'):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f'Family needs {key}')
        if not row['native_tests']:
            raise ValueError(f"{row['id']}: a supported family needs native tests")
        for key, allowed in {'live': {'supported', 'unsupported', 'precomputed'},
                             'cairo': {'native', 'shared'}, 'wgl': {'native', 'shared'},
                             'html': {'browser', 'shared', 'unsupported'},
                             'notebooks': {'unverified'}}.items():
            if row[key] not in allowed:
                raise ValueError(f"{row['id']}: invalid {key} status")
        if row['live'] != 'unsupported' and not row['browser_tests']:
            raise ValueError(f"{row['id']}: interactive support needs browser evidence")
        covered.update(row['sources'])
        native.update(row['native_tests'])
        browser.update(row['browser_tests'])
    renderer_actual = {p.relative_to(root).as_posix() for p in (root / 'ext').glob('visualization*.jl')} | {'ext/TamerOpCairoMakieExt.jl', 'ext/TamerOpWGLMakieExt.jl'}
    if renderer_actual != set(data['renderer_sources']):
        raise ValueError('Renderer sources changed; review and update the support matrix')
    actual = {p.relative_to(root).as_posix() for p in (root / 'src/visualization').glob('*.jl')}
    actual.add('src/Visualization.jl')
    if actual != covered:
        raise ValueError(f'Source coverage mismatch: missing={sorted(actual-covered)}, stale={sorted(covered-actual)}')
    registered = set(re.findall(r'"(test_[^"]+\.jl)"', (root / 'test/runner.jl').read_text()))
    actual_native = {p.name for pattern in ('test_visualization*.jl', 'test_visual_session*.jl')
                     for p in (root / 'test').glob(pattern)}
    if actual_native - native:
        raise ValueError(f'Unassigned native owners: {sorted(actual_native-native)}')
    if native - registered:
        raise ValueError(f'Unregistered native owners: {sorted(native-registered)}')
    actual_browser = {p.name for p in (root / 'test/browser').glob('*.spec.mjs')}
    if actual_browser != browser:
        raise ValueError(f'Browser coverage mismatch: missing={sorted(actual_browser-browser)}, stale={sorted(browser-actual_browser)}')
    for name in covered | set(data['renderer_sources']) | {f'test/{p}' for p in native} | {f'test/browser/{p}' for p in browser} | {data['workflow']}:
        if not (root / name).is_file():
            raise ValueError(f'Missing contract source: {name}')
    for filename in actual_browser:
        if re.search(r'test\.(?:describe\.)?(skip|fixme|only)\s*\(', (root / 'test/browser' / filename).read_text()):
            raise ValueError(f'{filename}: required browser acceptance cannot silently skip')
    return data, sorted(native)


def render(data):
    lines = ['# Visualization support and acceptance', '',
        'Use this matrix to choose a display and to review a visualization change. '
        'Support is a declared behavior; verification is evidence for a specified path. '
        'A native figure or serialized HTML is not evidence that a browser or notebook executed it.', '',
        'The maintained source is [visualization_support.toml](visualization_support.toml). '
        'Regenerate this page with `python test/visualization_contracts.py --write`; '
        'CI rejects unassigned source files, unassigned native/browser tests and stale generated tables. '
        'Reviewers must also identify new recipes within existing source files.', '',
        '## Reading the matrix', '',
        '- **N**: family-specific native mathematical/renderer checks; no browser execution implied.',
        '- **B**: maintained Chromium scenario exercises this family through real browser events.',
        '- **S**: supported through shared rendering/export code, covered by shared checks; '
        'not a separate browser acceptance claim for every recipe.',
        '- **U**: integration exists but that named frontend has not been qualified.',
        '- **—**: no such callback/widget is offered. Static selection remains a Julia argument.', '',
        'Cairo means static PNG/SVG/PDF figures. WGL native means native scene construction; '
        'HTML means exported scenes or fixed SVG compositions. Live Chromium means a standalone '
        'Bonito application with Julia running. Saved-view HTML instead selects among explicitly '
        'precomputed states after Julia has exited.', '',
        'The notebook column applies to each of IJulia/Jupyter, Pluto and VS Code: none inherits '
        'standalone-browser acceptance. Static notebook publication and executed Julia cells '
        'also do not certify a live notebook frontend. Firefox and WebKit have no declared '
        'acceptance baseline yet.', '',
        'The maintained standalone-browser environment requires Bonito 5.2 within major '
        'version 5. Browser evidence applies to the resolved harness stack; it does not '
        'qualify older nested-application integrations.', '',
        '## Families and display paths', '',
        '| Family | Cairo | WGL native | Exported HTML | Standalone Chromium | Live notebooks |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in data['family']:
        symbols = {'native':'N','shared':'S','browser':'B','unsupported':'—','unverified':'U'}
        live = 'B (precomputed)' if row['live']=='precomputed' else 'B' if row['live']=='supported' else '—'
        slug = re.sub(r'[^\w\- ]', '', row['title'].lower()).replace(' ', '-')
        lines.append(f"| [{row['title']}](#{slug}) | {symbols[row['cairo']]} | {symbols[row['wgl']]} | {symbols[row['html']]} | {live} | U |")
    lines += ['', 'These entries identify maintained checks, not a claim that a test file proves '
        'its latest execution passed. CI supplies a run result and retained artifacts; '
        'local acceptance records preserve their environment, failures, reruns and scope. '
        'A green shared check does not qualify every possible input, recipe or frontend.', '']
    if snapshot := data.get('verification'):
        lines += ['## Recorded verification', '', f"Acceptance recorded on {snapshot['date']}. {snapshot['environment']}", '',
            f"- Native: {snapshot['native']}", f"- Live browser: {snapshot['live']}",
            f"- Offline exports: {snapshot['offline']}", '',
            'This dated record describes the tested revision and environment; current CI results '
            'remain the evidence for subsequent changes. The matrix validator checks the record\'s '
            'format, not the truth of an execution claim.', '']
    lines += [
        '## Standing A36 and A37 contracts', '',
        '**A36 — mathematical coordinates.** Use asymmetric and nonuniform fixtures wherever '
        'a transpose or midpoint convention could look plausible. Assert actual plotted cell '
        'boundaries and values in both backends, selected axis order, fixed coordinates, masks, '
        'orientation and exact-query behavior. Exact classification precedes floating drawing '
        'conversion. Native slider callbacks and real browser slider transport are separate checks.', '',
        '**A37 — advertised interaction.** Controls need accessible names and keyboard operation '
        'without requiring a canvas click. Tab and Shift+Tab must leave plots and reach later '
        'controls, including after a figure rebuild. Test pointer events when picking is advertised, '
        'read-only hover, invalid-input rollback, shared selection across clients, '
        'disconnection and close/disposal. Verify visible updates and mathematical readouts. '
        'Declare live Julia, static scene navigation and precomputed offline selection distinctly. '
        'At narrow widths, preserve readable scientific labels and keyboard/pointer access to wide '
        'charts; fitting the surrounding page alone does not establish readability.', '',
        '## Required checks for a change', '',
        '1. Update the family entry when adding a recipe, a backend or a new interaction. '
        'Provide an independent mathematical oracle and real backend assertions appropriate to it.',
        '2. For an advertised live control, add a browser scenario with real user input, '
        'readout/drawing assertions and lifecycle coverage. Preserve narrow-layout and keyboard access.',
        '3. Run the matrix validator and affected native/browser owners locally. The '
        '[Visualization CI workflow](../.github/workflows/Visualization.yml) requires both '
        'rendering extensions and runs all registered visualization owners, live Chromium, '
        'then no-server exported-HTML scenarios. Missing offline fixtures fail rather than skip.',
        '4. Review representative figures. Tests of coordinates, transport or DOM layout '
        'do not establish readability or a clear mathematical explanation.', '',
        'Commands and environment setup are in [testing](testing.md) and the '
        '[browser harness](../test/browser/README.md). These are contributor acceptance '
        'requirements; development status and test counts do not belong in mathematical lessons.', '',
        '## Evidence by family', '']
    for row in data['family']:
        links=lambda paths: ', '.join(f'[{Path(p).name}](../{p})' for p in paths)
        lines += [f'### {row["title"]}', '',row['scope'],'',
                  'Native owners: '+links(['test/'+p for p in row['native_tests']])+'.', '']
        if row['browser_tests']:
            lines += ['Browser scenarios: '+links(['test/browser/'+p for p in row['browser_tests']])+'.', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--list-native',action='store_true')
    parser.add_argument('--list-browser',action='store_true')
    parser.add_argument('--run-native',action='store_true')
    parser.add_argument('--julia',default='julia')
    parser.add_argument('--project',default='test/browser')
    args=parser.parse_args()
    try:
        data,owners=load_and_validate()
        page=ROOT/data['generated_page']; expected=render(data)
        if args.write:
            page.write_text(expected)
        elif not page.is_file() or page.read_text()!=expected:
            raise ValueError('Support page is stale; run python test/visualization_contracts.py --write')
        if args.list_browser:
            print('\n'.join(sorted(set(data['shared_browser_tests']) | {p for row in data['family'] for p in row['browser_tests']})))
        elif args.list_native:
            print('\n'.join(owners))
        elif args.run_native:
            cmd=[args.julia,'--startup-file=no','--threads=1','--heap-size-hint=3000M',f'--project={args.project}',str(ROOT/'test/runtests.jl'),
                 *['--file='+p for p in owners], '--require-extension=TamerOpCairoMakieExt',
                 '--require-extension=TamerOpWGLMakieExt']
            return subprocess.call(cmd,cwd=ROOT)
        else:
            print(f"Visualization contracts: {len(data['family'])} families; {len(owners)} native owners; matrix current.")
        return 0
    except (ValueError,KeyError,OSError) as error:
        print(f'Visualization contract error: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
