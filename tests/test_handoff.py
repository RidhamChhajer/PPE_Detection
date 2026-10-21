"""Keep private runtime assets out of the source-only handoff."""
import unittest

from tools.prepare_repository import ROOT, source_files, findings, check_links


class HandoffTests(unittest.TestCase):
    def test_allowlist_excludes_private_assets_and_includes_current_setup(self):
        files = source_files()
        names = {path.relative_to(ROOT).as_posix() for path in files}
        self.assertTrue({'DESIGN.md', 'requirements-cpu.txt', 'requirements-auto.txt', 'tests/browser_smoke.cjs',
                         'deployment/requirements-gpu.txt', 'deployment/setup.ps1', 'deployment/start.ps1'} <= names)
        self.assertFalse(any(name.split('/')[0] in
                             ('data', 'models', 'Previous', 'artifacts', '.git', '.aws', '.venv') for name in names))
        self.assertFalse(any(path.suffix in ('.engine', '.onnx', '.pth', '.mp4', '.pem', '.key') for path in files))
        check_links(files)

    def test_secret_findings_expose_locations_only(self):
        candidate = 'sk-' + 'x'*40
        records = findings(candidate, 'example.txt')
        self.assertEqual(records, [dict(path='example.txt', line=1, kind='potential_secret')])
        self.assertNotIn(candidate, repr(records))
