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
MANIFEST = ROOT / 'test/visualization_support.toml'


def load_and_validate(root=ROOT):
    data = tomllib.loads((root / 'test/visualization_support.toml').read_text())
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



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-native',action='store_true')
    parser.add_argument('--list-browser',action='store_true')
    parser.add_argument('--run-native',action='store_true')
    parser.add_argument('--julia',default='julia')
    parser.add_argument('--project',default='test/browser')
    args=parser.parse_args()
    try:
        data,owners=load_and_validate()
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
            print(f"Visualization contracts: {len(data['family'])} families; {len(owners)} native owners; source and test assignments current.")
        return 0
    except (ValueError,KeyError,OSError) as error:
        print(f'Visualization contract error: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
