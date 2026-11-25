"""Optional exports preserve statuses, identities, evidence and snapshot semantics."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LEAF = Path(__file__).resolve().parents[1] / 'skills/ops/torch-op-validation'
SCRIPT = LEAF / 'scripts/export_platform.py'
spec = importlib.util.spec_from_file_location('platform_export', SCRIPT)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class PlatformTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = json.loads((LEAF / 'assets/report-example.json').read_text())
        self.source = self.root / 'bundle.json'

    def save(self):
        self.source.write_text(json.dumps(self.bundle))

    def export(self, name='allure'):
        self.save()
        target = self.root / name
        adapter.export_allure(self.source, target)
        return [json.loads(p.read_text()) for p in target.glob('*-result.json')]

    def progress(self):
        return dict(schema_version=1, evidence_class='hardware', planned=10,
                    completed=3, failed=1, running=1, state='running',
                    started_at_seconds=1000, updated_at_seconds=1010)

    def test_all_original_statuses_are_retained_without_false_passes(self):
        row = self.bundle['results'][0]
        self.bundle['run']['evidence_class'] = 'hardware'
        self.bundle['results'] = [{**row, 'case_id': status, 'status': status}
                                  for status in adapter.STATUS_MAP]
        exported = self.export()
        for result in exported:
            params = {p['name']: json.loads(p['value']) for p in result['parameters']}
            self.assertEqual(result['status'], adapter.STATUS_MAP[params['case_id']])
        self.assertEqual(len(exported), len(adapter.STATUS_MAP))

    def test_synthetic_and_unverified_never_appear_as_hardware_passes(self):
        for evidence in ('synthetic', 'unverified'):
            self.bundle['run']['evidence_class'] = evidence
            self.assertEqual({r['status'] for r in self.export(evidence)}, {'skipped'})

    def test_history_changes_with_parameters_not_run_id_and_uuid_is_unique(self):
        first = self.export('one')
        self.bundle['run']['id'] = 'another-run'
        second = self.export('two')
        self.assertEqual({r['historyId'] for r in first}, {r['historyId'] for r in second})
        self.assertFalse({r['uuid'] for r in first} & {r['uuid'] for r in second})
        self.bundle['results'][0]['metadata']['shape'] = [99]
        third = self.export('three')
        self.assertNotEqual({r['historyId'] for r in first}, {r['historyId'] for r in third})

    def test_attachments_are_explicit_and_missing_evidence_downgrades_pass(self):
        self.bundle['run']['evidence_class'] = 'hardware'
        self.bundle['results'] = [self.bundle['results'][0]]
        self.bundle['results'][0]['evidence'] = ['raw.txt']
        (self.root / 'raw.txt').write_text('approved attachment')
        self.save()
        target = self.root / 'attached'
        adapter.export_allure(self.source, target, include_evidence=True)
        result = json.loads(next(target.glob('*-result.json')).read_text())
        self.assertEqual(len(result['attachments']), 2)
        self.assertTrue(any(p.read_bytes() == b'approved attachment' for p in target.glob('*-attachment*')))
        (self.root / 'raw.txt').unlink()
        self.assertEqual(self.export('missing')[0]['status'], 'broken')

    def test_existing_output_and_escaping_evidence_are_rejected(self):
        self.export()
        with self.assertRaises(FileExistsError):
            adapter.export_allure(self.source, self.root / 'allure')
        self.bundle['results'][0]['evidence'] = ['../private.txt']
        self.save()
        with self.assertRaises(ValueError):
            adapter.export_allure(self.source, self.root / 'unsafe', True)
        self.assertFalse((self.root / 'unsafe').exists())

    def test_bundle_metrics_preserve_all_statuses_and_omit_case_labels(self):
        self.save()
        target = self.root / 'results.prom'
        adapter.export_prometheus(target, 'worker"\\a', bundle_path=self.source)
        text = target.read_text()
        self.assertIn('evidence_class="synthetic"', text)
        self.assertIn('stream="worker\\"\\\\a"', text)
        self.assertEqual(len([x for x in text.splitlines() if not x.startswith('#')]), 45)
        self.assertNotIn('case_id=', text)
        self.assertNotIn('updated_at', text)  # Final reports are not heartbeats.
        self.assertEqual(target.stat().st_mode & 0o777, 0o644)
        target.chmod(0o640)
        adapter.export_prometheus(target, 'worker', bundle_path=self.source)
        self.assertEqual(target.stat().st_mode & 0o777, 0o640)

    def test_progress_atomic_update_and_terminal_snapshot(self):
        source = self.root / 'progress.json'
        target = self.root / 'live.prom'
        data = self.progress()
        adapter.write_progress(source, data)
        adapter.export_prometheus(target, 'slot-1', progress_path=source)
        before = target.read_text()
        self.assertIn('torch_op_progress_updated_at_seconds', before)
        self.assertIn('kind="completed",stream="slot-1"} 3', before)
        data.update(completed=10, running=0, state='finished', updated_at_seconds=1020)
        adapter.write_progress(source, data)
        adapter.export_prometheus(target, 'slot-1', progress_path=source)
        self.assertIn('state="finished",stream="slot-1"} 1', target.read_text())
        self.assertEqual(list(self.root.glob('.*.prom.*')), [])

    def test_invalid_progress_never_overwrites_last_good_snapshot(self):
        source = self.root / 'progress.json'
        adapter.write_progress(source, self.progress())
        before = source.read_bytes()
        for changes in ({'completed': 11}, {'failed': 4}, {'running': True},
                        {'updated_at_seconds': 999}, {'updated_at_seconds': float('nan')},
                        {'state': 'finished'}, {'state': 'aborted'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                adapter.write_progress(source, {**self.progress(), **changes})
            self.assertEqual(source.read_bytes(), before)

    def test_cli_exports_example_without_device_dependencies(self):
        self.save()
        for args in (['allure', str(self.source), '--out', str(self.root / 'cli')],
                     ['prometheus', '--bundle', str(self.source), '--stream', 'slot',
                      '--out', str(self.root / 'cli.prom')]):
            done = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True)
            self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
