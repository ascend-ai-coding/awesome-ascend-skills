#!/usr/bin/env python3
"""Validate Golden dtype plans and before-call records, without running operators."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


DEFAULT_MAPPING = Path(__file__).resolve().parents[1] / 'assets/golden-dtype-map.json'
INTEGER_DTYPES = {'bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64'}
KNOWN_DTYPES = INTEGER_DTYPES | {'float16', 'bfloat16', 'float32', 'float64', 'complex32', 'complex64', 'complex128'}
API_TYPES = {'floating', 'quantized', 'non_computational', 'integer', 'random'}
EXECUTION_STATES = {'EXECUTED', 'NOT_RUN', 'BLOCKED', 'ERROR'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_object, parse_constant=invalid)


def text(value, label):
    if not isinstance(value, str) or not value.strip() or '\0' in value:
        raise ValueError(label + ' must be a nonempty string')
    return value


def dtype(value):
    return value.removeprefix('torch.') if isinstance(value, str) else None


def mapping_rules(data):
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('Mapping schema_version must be integer 1')
    text(data.get('mapping_id'), 'mapping_id')
    supplied = data.get('mapping')
    if not isinstance(supplied, dict) or not supplied:
        raise ValueError('Mapping must contain a nonempty mapping object')
    result = {}
    for source, target in supplied.items():
        source = dtype(text(source, 'mapping source dtype'))
        target = dtype(text(target, 'mapping target dtype'))
        if source in result:
            raise ValueError('Duplicate normalized mapping dtype: ' + source)
        result[source] = target
    return result


def aggregate(statuses):
    return 'FAIL' if 'FAIL' in statuses else 'BLOCKED' if 'BLOCKED' in statuses else 'PASS'


def operand_record(record, group, api_type, rules):
    if not isinstance(record, dict):
        raise ValueError(group + ' operand must be an object')
    name = text(record.get('name'), group + ' operand name')
    source, actual = dtype(record.get('source_dtype')), dtype(record.get('golden_dtype'))
    policy = record.get('policy', 'mapped')
    if policy not in {'mapped', 'preserve'}:
        raise ValueError('Operand policy must be mapped or preserve')
    expected, issues = None, []
    if source is None or actual is None:
        issues.append({'status': 'BLOCKED', 'kind': 'missing_before_call_dtype', 'detail': 'Both source_dtype and golden_dtype are required.'})
    elif policy == 'preserve':
        if source not in KNOWN_DTYPES:
            issues.append({'status': 'BLOCKED', 'kind': 'unknown_dtype', 'detail': 'No reviewed preserve rule for ' + source})
        elif api_type != 'non_computational' and source not in INTEGER_DTYPES:
            issues.append({'status': 'BLOCKED', 'kind': 'invalid_preserve_scope', 'detail': 'Preserve is explicit for non-computational calls or integer/boolean operands; it cannot bypass a floating mapping.'})
        else:
            expected = source
    elif api_type == 'non_computational' or source in INTEGER_DTYPES:
        issues.append({'status': 'BLOCKED', 'kind': 'explicit_preserve_required', 'detail': 'Declare policy=preserve for this operand.'})
    elif source not in rules:
        issues.append({'status': 'BLOCKED', 'kind': 'unmapped_dtype', 'detail': 'The adopted Golden map does not cover ' + source})
    else:
        expected = rules[source]
    if expected is not None and actual != expected:
        issues.append({'status': 'FAIL', 'kind': 'golden_policy_mismatch', 'detail': 'Recorded Golden dtype differs from the adopted operand policy.', 'expected_dtype': expected, 'recorded_dtype': actual})
    plan_status = aggregate([issue['status'] for issue in issues])
    before = record.get('verified_before_call')
    if before is not None and type(before) is not bool:
        raise ValueError('verified_before_call must be boolean')
    return {'name': name, 'group': group, 'source_dtype': source, 'golden_dtype': actual,
            'policy': policy, 'expected_golden_dtype': expected, 'plan_status': plan_status,
            'verified_before_call': before is True, 'issues': issues}


def evaluate_case(case, rules):
    if not isinstance(case, dict):
        raise ValueError('Each case must be an object')
    case_id, phase = text(case.get('case_id'), 'case_id'), text(case.get('phase'), 'phase')
    api_type = case.get('api_type', 'floating')
    if api_type not in API_TYPES:
        raise ValueError('Unknown api_type')
    execution = case.get('execution_status', 'NOT_RUN')
    if execution not in EXECUTION_STATES:
        raise ValueError('execution_status must be EXECUTED, NOT_RUN, BLOCKED or ERROR')
    operands, issues = [], []
    for group in ('inputs', 'upstream'):
        values = case.get(group, [])
        if not isinstance(values, list):
            raise ValueError(group + ' must be an array')
        names = set()
        for value in values:
            row = operand_record(value, group, api_type, rules)
            if row['name'] in names:
                raise ValueError('Duplicate ' + group + ' operand name: ' + row['name'])
            names.add(row['name']); operands.append(row)
            issues.extend(dict(issue, location=group + '.' + row['name']) for issue in row['issues'])
    if not any(row['group'] == 'inputs' for row in operands):
        issues.append({'status': 'BLOCKED', 'kind': 'missing_input_records', 'detail': 'Input dtypes cannot be inferred from output or statistics dtypes.'})
    requires_upstream = case.get('requires_upstream', False)
    if type(requires_upstream) is not bool:
        raise ValueError('requires_upstream must be boolean')
    if ('backward' in phase.lower() or requires_upstream) and not any(row['group'] == 'upstream' for row in operands):
        issues.append({'status': 'BLOCKED', 'kind': 'missing_upstream_records', 'detail': 'Record the frozen upstream gradient and its before-call Golden dtype.'})
    plan_status = aggregate([row['plan_status'] for row in operands] + [issue['status'] for issue in issues])
    if execution != 'EXECUTED':
        issues.append({'status': 'BLOCKED', 'kind': 'execution_not_completed', 'detail': 'A plan or uncompleted call is not executed evidence.'})
    if any(not row['verified_before_call'] for row in operands):
        issues.append({'status': 'BLOCKED', 'kind': 'before_call_not_verified', 'detail': 'All operands need an explicit verified_before_call=true record.'})
    references = case.get('evidence', [])
    if not isinstance(references, list) or any(not isinstance(value, str) or not value.strip() or '\0' in value for value in references):
        raise ValueError('evidence must be an array of nonempty reference strings')
    if not references:
        issues.append({'status': 'BLOCKED', 'kind': 'missing_execution_evidence', 'detail': 'At least one execution-record reference is required.'})
    status = aggregate([plan_status] + [issue['status'] for issue in issues])
    return {'case_id': case_id, 'phase': phase, 'api_type': api_type,
            'source_execution_status': execution, 'plan_status': plan_status, 'status': status,
            'operands': operands, 'issues': issues, 'evidence': references,
            'assessment': 'Golden dtype metadata policy only; not an operator numerical result.'}


def validate_records(data, rules):
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('Input schema_version must be integer 1')
    cases = data.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases must be a nonempty array')
    rows, keys = [], set()
    for case in cases:
        row = evaluate_case(case, rules)
        key = (row['case_id'], row['phase'])
        if key in keys:
            raise ValueError('Duplicate case_id/phase')
        keys.add(key); rows.append(row)
    return {'schema_version': 1, 'assessment': 'Golden dtype metadata policy',
            'status': aggregate([row['status'] for row in rows]),
            'plan_status': aggregate([row['plan_status'] for row in rows]),
            'status_counts': dict(Counter(row['status'] for row in rows)),
            'plan_status_counts': dict(Counter(row['plan_status'] for row in rows)),
            'cases': rows,
            'coverage': {'recorded_cases': len(rows), 'source_execution_claims': sum(row['source_execution_status'] == 'EXECUTED' for row in rows),
                         'accepted_executed_metadata_cases': sum(row['status'] == 'PASS' for row in rows),
                         'new_operator_calls': 0, 'new_device_calls': 0,
                         'hardware_execution_independently_verified': False, 'numerical_accuracy_assessed': False,
                         'complex_numerical_or_ulp_support_verified': False},
            'evidence_scope': 'References are preserved, not opened or authenticated. The executor is responsible for truthful before-call records and actual execution proof. FP64 comparator arithmetic is not interpreted as a Golden execution dtype.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('--mapping', type=Path, default=DEFAULT_MAPPING)
    parser.add_argument('--out', type=Path, required=True, help='new JSON report; parent directory must exist')
    strict = parser.add_mutually_exclusive_group()
    strict.add_argument('--strict', action='store_true', help='nonzero unless executed metadata status is PASS')
    strict.add_argument('--strict-plan', action='store_true', help='preflight: nonzero unless plan_status is PASS; never changes execution status')
    args = parser.parse_args(argv)
    try:
        mapping = read(args.mapping)
        report = validate_records(read(args.records), mapping_rules(mapping))
        report['input_sha256'] = digest(args.records)
        report['mapping'] = {'path': str(args.mapping), 'mapping_id': mapping['mapping_id'], 'sha256': digest(args.mapping)}
        report['selected_gate'] = 'plan_status' if args.strict_plan else 'status'
        with args.out.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError) as error:
        parser.exit(2, 'validate_golden: ' + str(error) + '\n')
    print(json.dumps({'report': str(args.out), 'status': report['status'], 'plan_status': report['plan_status'],
                      'new_operator_calls': 0}))
    return 1 if (args.strict or args.strict_plan) and report[report['selected_gate']] != 'PASS' else 0


if __name__ == '__main__':
    raise SystemExit(main())
