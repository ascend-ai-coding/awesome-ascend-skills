#!/usr/bin/env python3
"""Optional offline Allure export and Node Exporter textfile integration."""

import argparse
from collections import Counter
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import stat
import sys
import tempfile
import uuid

# Import the existing reporting contract, including its duplicate/path checks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_report as report


STATUS_MAP = {
    'PASS': 'passed', 'FAIL': 'failed', 'ERROR': 'broken', 'TIMEOUT': 'broken',
    'BLOCKED': 'broken', 'MEASURED': 'skipped', 'NOT_RUN': 'skipped',
    'UNSUPPORTED': 'skipped', 'NOT_APPLICABLE': 'skipped',
}


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=report._unique_json_keys, parse_constant=invalid)


def load_bundle(path):
    path = Path(path).resolve(strict=True)
    data = read_json(path)
    run, rows = report.validate(data, path.parent)
    return data, run, rows, path.parent


def json_text(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n'


def export_allure(bundle_path, output, include_evidence=False):
    data, run, rows, root = load_bundle(bundle_path)
    output = Path(output)
    prepared = []
    for row in rows:
        identity = {key: row[key] for key in ('operator', *report.KEY_FIELDS)}
        full_name = 'torch-op-validation:' + json.dumps(identity, sort_keys=True, ensure_ascii=False)
        signature = {key: row.get('metadata', {}).get(key) for key in
                     ('dtype', 'input_dtype', 'shape', 'layout', 'attrs', 'case_parameters')}
        parameters = {**identity, **{key: value for key, value in signature.items() if value is not None},
                      'standard_id': run['standard_id'], 'evidence_class': run['evidence_class']}
        stable = hashlib.sha256(json_text(parameters).encode()).hexdigest()
        result_id = str(uuid.uuid4())
        attachments = []
        payloads = []

        def attach(name, contents, media_type, suffix):
            source = str(uuid.uuid4()) + '-attachment' + suffix
            attachments.append({'name': name, 'source': source, 'type': media_type})
            payloads.append((source, contents))

        leaves = [leaf for leaf in data.get('precision_comparisons', [])
                  if leaf['case_id'] == row['case_id'] and leaf['phase'] == row['phase']]
        attach('Original result and precision records', json_text({
            'run': run, 'result': row, 'precision_comparisons': leaves,
            'scope': 'Result-row status only; not an independent 2.1 acceptance decision.',
        }).encode(), 'application/json', '.json')
        missing = []
        paths = list(row.get('evidence', []))
        if row.get('reproduction'):
            paths.append(row['reproduction'])
        for path in sorted(set(paths)):
            target = report._artifact_path(path, root, 'artifact')
            if not target.is_file():
                missing.append(path)
            elif include_evidence:
                attach(path, target.read_bytes(), mimetypes.guess_type(path)[0] or
                       'application/octet-stream', target.suffix)
        mapped = STATUS_MAP[row['status']]
        # Synthetic/unverified rows cannot inflate the hardware pass rate.
        if run['evidence_class'] != 'hardware':
            mapped = 'skipped'
        elif missing and mapped == 'passed':
            mapped = 'broken'
        description = (f"Original status: {row['status']}; evidence: {run['evidence_class']}. "
                       'This is one suite result, not formal operator acceptance. ' + row.get('detail', ''))
        if missing:
            description += '\nMissing referenced files: ' + ', '.join(missing)
        result = {
            'uuid': result_id, 'historyId': stable,
            'testCaseId': hashlib.sha256(full_name.encode()).hexdigest(),
            'fullName': full_name, 'name': f"{row['operator']} / {row['case_id']} / {row['phase']}",
            'status': mapped, 'stage': 'finished', 'description': description,
            'statusDetails': {'message': description},
            'labels': [{'name': 'suite', 'value': row['suite']},
                       {'name': 'framework', 'value': 'torch-op-validation'},
                       {'name': 'evidence_class', 'value': run['evidence_class']}],
            'parameters': [{'name': key, 'value': json.dumps(value, ensure_ascii=False, sort_keys=True)}
                           for key, value in parameters.items()],
            'attachments': attachments,
        }
        prepared.extend(payloads)
        prepared.append((result_id + '-result.json', json_text(result).encode()))
    output.mkdir(parents=True, exist_ok=False)
    for name, contents in prepared:
        (output / name).write_bytes(contents)


def atomic_text(path, text, mode=0o600):
    """Publish a complete snapshot; a failed conversion leaves the old file intact."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Output must not be a symlink')
    permissions = stat.S_IMODE(path.stat().st_mode) if path.exists() else mode
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        os.fchmod(fd, permissions)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_progress(data):
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('Progress schema_version must be 1')
    if data.get('evidence_class') not in report.EVIDENCE_CLASSES:
        raise ValueError('Progress requires evidence_class')
    for field in ('planned', 'completed', 'failed', 'running'):
        if type(data.get(field)) is not int or data[field] < 0:
            raise ValueError(field + ' must be a nonnegative integer')
    if data['failed'] > data['completed'] or data['completed'] + data['running'] > data['planned']:
        raise ValueError('Progress counts are inconsistent')
    for field in ('started_at_seconds', 'updated_at_seconds'):
        report._number(data.get(field), field)
    if data['started_at_seconds'] < 0 or data['updated_at_seconds'] < data['started_at_seconds']:
        raise ValueError('Progress timestamps are inconsistent')
    if data.get('state') not in ('running', 'finished', 'aborted'):
        raise ValueError('Progress state must be running, finished or aborted')
    if data['state'] == 'finished' and data['completed'] != data['planned']:
        raise ValueError('Finished progress requires all slots completed')
    if data['state'] != 'running' and data['running']:
        raise ValueError('A terminal snapshot cannot contain running slots')
    return data


def write_progress(path, data):
    """Executor hook: invoke outside timed regions; counts refer to planned slots."""
    atomic_text(path, json_text(validate_progress(data)))


def label(value):
    return str(value).replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"')


def export_prometheus(output, stream, bundle_path=None, progress_path=None):
    if not stream.strip() or len(stream) > 128 or any(ord(c) < 32 for c in stream):
        raise ValueError('Use a short stable stream label')
    lines = []

    def metric(name, help_text, samples):
        lines.extend([f'# HELP {name} {help_text}', f'# TYPE {name} gauge'])
        for value, dimensions in samples:
            tags = {'stream': stream, 'evidence_class': evidence, **dimensions}
            tags = ','.join(f'{key}="{label(val)}"' for key, val in sorted(tags.items()))
            lines.append(f'{name}{{{tags}}} {value}')

    if progress_path:
        data = validate_progress(read_json(progress_path))
        evidence = data['evidence_class']
        metric('torch_op_progress_slots', 'Planned result slots, not independent cases.',
               [(data[key], {'kind': key}) for key in ('planned', 'completed', 'failed', 'running')])
        for field in ('started_at_seconds', 'updated_at_seconds'):
            metric('torch_op_progress_' + field, 'Executor timestamp in Unix seconds.', [(data[field], {})])
        metric('torch_op_run_state', 'One-hot executor lifecycle; finished does not mean passed.',
               [(int(data['state'] == state), {'state': state}) for state in ('running', 'finished', 'aborted')])
    else:
        _, run, rows, _ = load_bundle(bundle_path)
        evidence = run['evidence_class']
        counts = Counter((row['suite'], row['status']) for row in rows)
        metric('torch_op_result_rows', 'Recorded result rows; not formal acceptance or progress.',
               [(counts[suite, status], {'suite': suite, 'status': status})
                for suite in report.SUITES for status in report.STATUSES])
    output = Path(output)
    if output.suffix != '.prom':
        raise ValueError('Node Exporter output must end in .prom')
    atomic_text(output, '\n'.join(lines) + '\n', mode=0o644)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    allure = sub.add_parser('allure', help='Create a new Allure results directory')
    allure.add_argument('bundle', type=Path)
    allure.add_argument('--out', required=True, type=Path)
    allure.add_argument('--include-evidence', action='store_true', help='Copy approved referenced files')
    prom = sub.add_parser('prometheus', help='Atomically update a Node Exporter .prom file')
    source = prom.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle', type=Path)
    source.add_argument('--progress', type=Path)
    prom.add_argument('--out', required=True, type=Path)
    prom.add_argument('--stream', required=True, help='Stable worker/slot name, not a run ID')
    args = parser.parse_args()
    try:
        if args.command == 'allure':
            export_allure(args.bundle, args.out, args.include_evidence)
        else:
            export_prometheus(args.out, args.stream, args.bundle, args.progress)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'export: {error}\n')


if __name__ == '__main__':
    main()
