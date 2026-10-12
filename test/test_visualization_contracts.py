"""Negative acceptance tests: omissions must fail before expensive graphics jobs."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import visualization_contracts as C


class SupportContractTests(unittest.TestCase):
    def setUp(self):
        self.data, _ = C.load_and_validate()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        paths = set(self.data['core_sources']) | set(self.data['renderer_sources'])
        tests = set(self.data['shared_native_tests'])
        browser = set(self.data['shared_browser_tests'])
        for row in self.data['family']:
            paths.update(row['sources']); tests.update(row['native_tests']); browser.update(row['browser_tests'])
        paths |= {f'test/{p}' for p in tests} | {f'test/browser/{p}' for p in browser}
        paths |= {self.data['workflow'], 'test/runner.jl', 'docs/visualization_support.toml'}
        for name in paths:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((C.ROOT / name).read_text())

    def test_complete_inventory(self):
        data, owners = C.load_and_validate(self.root)
        self.assertEqual(len(data['family']), len(self.data['family']))
        self.assertIn('test_visualization_a98.jl', owners)

    def test_new_builder_requires_assignment(self):
        (self.root / 'src/visualization/builders_new.jl').write_text('')
        with self.assertRaisesRegex(ValueError, 'Source coverage mismatch'):
            C.load_and_validate(self.root)

    def test_new_renderer_requires_review(self):
        (self.root / 'ext/visualization_wgl_new.jl').write_text('')
        with self.assertRaisesRegex(ValueError, 'Renderer sources changed'):
            C.load_and_validate(self.root)

    def test_new_browser_scenario_requires_assignment(self):
        (self.root / 'test/browser/new.spec.mjs').write_text('')
        with self.assertRaisesRegex(ValueError, 'Browser coverage mismatch'):
            C.load_and_validate(self.root)

    def test_live_support_requires_browser_test(self):
        data = copy.deepcopy(self.data)
        row = next(row for row in data['family'] if row['live']=='supported')
        row['browser_tests']=[]
        with patch.object(C.tomllib, 'loads', return_value=data):
            with self.assertRaisesRegex(ValueError, 'interactive support needs browser evidence'):
                C.load_and_validate(self.root)

    def test_skip_is_not_acceptance(self):
        target = self.root / 'test/browser/portable-session.spec.mjs'
        target.write_text('test.skip(!fixture);')
        with self.assertRaisesRegex(ValueError, 'cannot silently skip'):
            C.load_and_validate(self.root)


if __name__=='__main__':
    unittest.main()
