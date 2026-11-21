"""Check Golden metadata policy gates; no Torch, ATK or device imports."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


LEAF = Path(__file__).resolve().parents[1] / 'skills/ops/torch-op-validation'
SCRIPT = LEAF / 'scripts/validate_golden.py'
spec = importlib.util.spec_from_file_location('golden_metadata_validator', SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
RULES = guard.mapping_rules(guard.read(LEAF / 'assets/golden-dtype-map.json'))


class GoldenPolicyTest(unittest.TestCase):
    def case(self, source='float32', actual='float32', **changes):
        case = {'case_id': 'one', 'phase': 'forward', 'api_type': 'floating',
                'execution_status': 'EXECUTED',
                'inputs': [{'name': 'input', 'source_dtype': source, 'golden_dtype': actual,
                            'verified_before_call': True}],
                'upstream': [], 'evidence': ['executor/actual-call.json']}
        case.update(changes)
        return case

    def evaluate(self, case):
        return guard.validate_records({'schema_version': 1, 'cases': [case]}, RULES)

    def test_original_fp32_to_fp64_is_a_policy_failure_even_if_output_says_fp32(self):
        report = self.evaluate(self.case(actual='float64', outputs=[{'dtype': 'float32'}]))
        self.assertEqual(report['status'], 'FAIL')
        issue = report['cases'][0]['issues'][0]
        self.assertEqual((issue['kind'], issue['location'], issue['expected_dtype'], issue['recorded_dtype']),
                         ('golden_policy_mismatch', 'inputs.input', 'float32', 'float64'))
        self.assertFalse(report['coverage']['numerical_accuracy_assessed'])

    def test_fp32_same_dtype_passes_and_fp64_comparator_statistics_are_irrelevant(self):
        report = self.evaluate(self.case(comparator_statistics_dtype='float64'))
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['coverage']['accepted_executed_metadata_cases'], 1)
        self.assertEqual(report['coverage']['new_operator_calls'], 0)
        self.assertFalse(report['coverage']['hardware_execution_independently_verified'])

    def test_complex_mapping_plan_does_not_claim_complex_testing(self):
        report = self.evaluate(self.case(source='complex32', actual='complex64', execution_status='NOT_RUN'))
        self.assertEqual(report['plan_status'], 'PASS')
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['coverage']['accepted_executed_metadata_cases'], 0)
        self.assertFalse(report['coverage']['complex_numerical_or_ulp_support_verified'])
        same = self.evaluate(self.case(source='complex64', actual='complex64'))
        self.assertEqual(same['status'], 'PASS')

    def test_upstream_wrong_dtype_is_caught(self):
        case = self.case(source='bfloat16', actual='float32', phase='forward_and_backward',
            upstream=[{'name': 'grad_output', 'source_dtype': 'bfloat16', 'golden_dtype': 'bfloat16',
                       'verified_before_call': True}])
        report = self.evaluate(case)
        self.assertEqual(report['status'], 'FAIL')
        self.assertIn('upstream.grad_output', [issue.get('location') for issue in report['cases'][0]['issues']])

    def test_missing_before_call_or_evidence_blocks_an_execution_claim(self):
        for change in ('before', 'evidence'):
            with self.subTest(change=change):
                case = self.case()
                if change == 'before': case['inputs'][0].pop('verified_before_call')
                else: case['evidence'] = []
                report = self.evaluate(case)
                self.assertEqual(report['plan_status'], 'PASS')
                self.assertEqual(report['status'], 'BLOCKED')

    def test_unknown_dtype_or_only_output_records_cannot_pass(self):
        self.assertEqual(self.evaluate(self.case(source='float64', actual='float64'))['status'], 'BLOCKED')
        report = self.evaluate(self.case(inputs=[], outputs=[{'dtype': 'float32'}]))
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['cases'][0]['operands'], [])

    def test_noncomputational_and_integer_operands_require_explicit_preserve(self):
        case = self.case(source='int64', actual='int64')
        self.assertEqual(self.evaluate(case)['status'], 'BLOCKED')
        case['inputs'][0]['policy'] = 'preserve'
        self.assertEqual(self.evaluate(case)['status'], 'PASS')
        case = self.case(source='float16', actual='float16', api_type='non_computational')
        case['inputs'][0]['policy'] = 'preserve'
        self.assertEqual(self.evaluate(case)['status'], 'PASS')
        case['api_type'] = 'floating'
        self.assertEqual(self.evaluate(case)['status'], 'BLOCKED')

    def test_duplicate_cases_and_missing_backward_upstream_are_not_accepted(self):
        with self.assertRaises(ValueError):
            guard.validate_records({'schema_version': 1, 'cases': [self.case(), self.case()]}, RULES)
        self.assertEqual(self.evaluate(self.case(phase='backward'))['status'], 'BLOCKED')

    def test_strict_cli_distinguishes_valid_unrun_plan_from_executed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'records.json'
            source.write_text(json.dumps({'schema_version': 1, 'cases': [self.case(execution_status='NOT_RUN')]}))
            plan, actual = root / 'plan.json', root / 'actual.json'
            one = subprocess.run([sys.executable, str(SCRIPT), str(source), '--out', str(plan), '--strict-plan'], capture_output=True)
            two = subprocess.run([sys.executable, str(SCRIPT), str(source), '--out', str(actual), '--strict'], capture_output=True)
            self.assertEqual(one.returncode, 0, one.stderr)
            self.assertEqual(two.returncode, 1, two.stderr)
            self.assertEqual(json.loads(plan.read_text())['status'], 'BLOCKED')
            before = plan.read_bytes()
            again = subprocess.run([sys.executable, str(SCRIPT), str(source), '--out', str(plan)], capture_output=True)
            self.assertEqual(again.returncode, 2)
            self.assertEqual(plan.read_bytes(), before)

    def test_default_mapping_matches_html_policy_without_loading_torch(self):
        spec = importlib.util.spec_from_file_location('golden_html_policy', LEAF / 'scripts/precision_comparison.py')
        html_policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(html_policy)
        self.assertEqual(RULES, {source: target for source, target, _ in html_policy.GOLDEN_DTYPE_RULES})
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
