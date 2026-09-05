#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Verify: Aggregate scout outputs into file manifest.

Reads S2P0_scout_t.json and S2P0_scout_k.json, verifies all files exist,
and outputs S2P0_file_manifest.json for downstream consumption.

Usage:
    python s2p0_verify.py \
        --op-name AddRmsNorm \
        --op-path /path/to/operator \
        --npu-arch DAV_2201 \
        [--soc-version Ascend910B3] \
        [--output-dir /path/to/output]

Output:
    {output_dir}/S2P0_file_manifest.json
"""
import argparse
import json
import logging
import os
import sys

_logger = logging.getLogger(__name__)

PLATFORM_MAP = {
    'DAV_3510': {'arch_num': '3510', 'chip': 'Ascend950', 'is_950': True},
    'DAV_2201': {'arch_num': '2201', 'chip': 'Ascend910B', 'is_950': False},
    'DAV_2002': {'arch_num': '2002', 'chip': 'Ascend310P', 'is_950': False},
}


def parse_args():
    parser = argparse.ArgumentParser(description="S2P0 Verify: Aggregate scout outputs")
    parser.add_argument("--op-name", required=True, help="Operator name")
    parser.add_argument("--op-path", required=True,
                        help="Operator source directory")
    parser.add_argument("--npu-arch", required=True, choices=list(PLATFORM_MAP.keys()),
                        help="Target NPU architecture")
    parser.add_argument("--soc-version", default=None, help="SOC version")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: {op_path}/tests/whitebox/)")
    return parser.parse_args()


def verify_file_exists(op_path, rel_path):
    """Check if file exists relative to op_path."""
    full_path = os.path.realpath(os.path.join(op_path, rel_path))
    op_root = os.path.realpath(op_path)
    family_root = os.path.realpath(os.path.dirname(op_path))
    allowed_roots = (op_root + os.sep, family_root + os.sep)
    if full_path != op_root and not full_path.startswith(allowed_roots):
        return False
    return os.path.isfile(full_path)


def extract_entry_function(scout_t_data):
    """Extract entry function from first registration in first entry."""
    try:
        if not scout_t_data.get('entries'):
            return None
        first_entry = scout_t_data['entries'][0]
        if not first_entry.get('registrations'):
            return None
        return first_entry['registrations'][0].get('entry_function')
    except (KeyError, IndexError, TypeError) as e:
        _logger.warning("Failed to extract entry_function: %s", e)
        return None


def build_tiling_section(scout_t_data, op_path):
    """Build tiling section from scout_t.json."""
    file_list = []
    missing_files = []

    # P0 files from entries
    for entry in scout_t_data.get('entries', []):
        try:
            rel_path = entry['file']
            if not verify_file_exists(op_path, rel_path):
                missing_files.append(rel_path)
                _logger.warning("Missing P0 tiling file: %s", rel_path)
            file_list.append({
                'path': rel_path,
                'priority': 'P0'
            })
        except KeyError as e:
            _logger.warning("Invalid tiling entry (missing %s): %s", e, entry)
            continue

    # P1 files from p1_files
    for rel_path in scout_t_data.get('p1_files', []):
        if not verify_file_exists(op_path, rel_path):
            missing_files.append(rel_path)
            _logger.warning("Missing P1 tiling file: %s", rel_path)
        file_list.append({
            'path': rel_path,
            'priority': 'P1'
        })

    # Excluded files
    excluded = []
    for ex in scout_t_data.get('excluded', []):
        try:
            excluded.append({
                'path': ex['file'],
                'reason': ex['reason']
            })
        except KeyError as e:
            _logger.warning("Invalid tiling excluded entry (missing %s): %s", e, ex)
            continue

    return {
        'entry_function': extract_entry_function(scout_t_data),
        'file_list': file_list,
        'excluded': excluded
    }, missing_files


def build_kernel_section(scout_k_data, op_path):
    """Build kernel section from scout_k.json."""
    file_list = []
    missing_files = []
    total_key_count = 0

    # P0 files from entries
    for entry in scout_k_data.get('entries', []):
        try:
            rel_path = os.path.join('op_kernel', entry['file'])
            if not verify_file_exists(op_path, rel_path):
                missing_files.append(rel_path)
                _logger.warning("Missing P0 kernel file: %s", rel_path)

            key_count = entry.get('total_key_count', 0)
            total_key_count += key_count

            file_list.append({
                'path': rel_path,
                'priority': 'P0',
                'pattern': entry.get('dispatch_type', 'A'),
                'key_count': key_count
            })
        except KeyError as e:
            _logger.warning("Invalid kernel entry (missing %s): %s", e, entry)
            continue

    # Excluded files
    excluded = []
    for ex in scout_k_data.get('excluded', []):
        try:
            rel_path = os.path.join('op_kernel', ex['file'])
            excluded.append({
                'path': rel_path,
                'reason': ex['reason']
            })
        except KeyError as e:
            _logger.warning("Invalid kernel excluded entry (missing %s): %s", e, ex)
            continue

    return {
        'total_key_count': total_key_count,
        'file_list': file_list,
        'excluded': excluded
    }, missing_files


def write_source_scope(output_dir, tiling_section, kernel_section):
    """Write S2P0_source_scope.md for downstream consumption."""
    scope_path = os.path.join(output_dir, 'S2P0_source_scope.md')

    lines = []
    lines.append('# 源码读取范围')
    lines.append('')
    lines.append('> 严格遵守，禁止自行添加其他文件')
    lines.append('')

    # tiling section
    lines.append('## tiling')
    lines.append('')
    for f in tiling_section['file_list']:
        lines.append(f'- {f["priority"]}: {f["path"]}')
    lines.append('')

    # kernel section
    lines.append('## kernel')
    lines.append('')
    for f in kernel_section['file_list']:
        lines.append(
            f'- {f["priority"]}: {f["path"]} — '
            f'dispatch 块（{f["key_count"]} 条 key, pattern={f["pattern"]}）')
    lines.append(f'- 总计 {kernel_section["total_key_count"]} 条 key')
    lines.append('')

    # excluded section
    lines.append('## 排除（禁止读取）')
    lines.append('')
    all_excluded = tiling_section['excluded'] + kernel_section['excluded']
    if all_excluded:
        for ex in all_excluded:
            lines.append(f'- {ex["path"]} — {ex["reason"]}')
    else:
        lines.append('（无）')
    lines.append('')

    with open(scope_path, 'w') as f:
        f.write('\n'.join(lines))

    _logger.info("Written: %s", scope_path)
    return scope_path


def _load_scout_data(output_dir):
    scout_t_path = os.path.join(output_dir, 'S2P0_scout_t.json')
    scout_k_path = os.path.join(output_dir, 'S2P0_scout_k.json')

    for path, label in [(scout_t_path, 'scout_t.json'),
                         (scout_k_path, 'scout_k.json')]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    with open(scout_t_path, 'r') as f:
        scout_t_data = json.load(f)
    with open(scout_k_path, 'r') as f:
        scout_k_data = json.load(f)

    _logger.info("Read scout_t.json: %d entries, %d p1_files, %d excluded",
                 len(scout_t_data.get('entries', [])),
                 len(scout_t_data.get('p1_files', [])),
                 len(scout_t_data.get('excluded', [])))
    _logger.info("Read scout_k.json: %d entries, %d excluded",
                 len(scout_k_data.get('entries', [])),
                 len(scout_k_data.get('excluded', [])))
    return scout_t_data, scout_k_data


def _write_manifest(output_dir, manifest):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'S2P0_file_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    _logger.info("Written: %s", manifest_path)
    return manifest_path


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    args = parse_args()
    op_path = os.path.abspath(args.op_path)
    npu_arch = args.npu_arch
    output_dir = args.output_dir or os.path.join(op_path, 'tests', 'whitebox')

    try:
        scout_t_data, scout_k_data = _load_scout_data(output_dir)
    except FileNotFoundError as e:
        _logger.error(str(e))
        sys.exit(1)

    tiling_section, tiling_missing = build_tiling_section(scout_t_data, op_path)
    kernel_section, kernel_missing = build_kernel_section(scout_k_data, op_path)
    all_missing = tiling_missing + kernel_missing

    if all_missing:
        _logger.error("Verification failed: %d missing files", len(all_missing))
        for f in all_missing:
            _logger.error("  - %s", f)
        sys.exit(1)

    _logger.info("Verification passed: all files exist")

    manifest = {
        'operator': args.op_name,
        'platform': {'npu_arch': npu_arch, 'soc_version': args.soc_version or ''},
        'verification': {'status': 'pass', 'missing_files': all_missing},
        'tiling': tiling_section,
        'kernel': kernel_section,
    }

    _write_manifest(output_dir, manifest)
    write_source_scope(output_dir, tiling_section, kernel_section)

    _logger.info("  Tiling: %d P0, %d P1, %d excluded",
                 sum(1 for f in tiling_section['file_list'] if f['priority'] == 'P0'),
                 sum(1 for f in tiling_section['file_list'] if f['priority'] == 'P1'),
                 len(tiling_section['excluded']))
    _logger.info("  Kernel: %d P0, %d excluded, %d total keys",
                 len(kernel_section['file_list']),
                 len(kernel_section['excluded']),
                 kernel_section['total_key_count'])
    _logger.info("Done.")


if __name__ == '__main__':
    main()
