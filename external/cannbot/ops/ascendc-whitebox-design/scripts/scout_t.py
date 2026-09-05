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
"""Scout-T: Tiling file reconnaissance script.

Discovers tiling files, detects registration macros, extracts entry function names,
traces #include chains for separated registration/implementation, and identifies
platform-reachable files.

Usage:
    python scout_t.py \
        --op-path /path/to/operator \
        --npu-arch DAV_2201 \
        [--soc-version Ascend910B3] \
        [--chip-model Ascend910B] \
        [--output-dir /path/to/output]

Output:
    {output_dir}/S2P0_scout_t.md
"""
import argparse
import json
import logging
import os
import re
import sys

from _scout_common import (PLATFORM_MAP, ARCH_DIR_MAP, ARCH_FEATURE_MAP,
                           skip_parens, find_def_after)

_logger = logging.getLogger(__name__)

PLATFORM_BRANCH_PATTERNS = [
    re.compile(r'\b(IsRegbaseSocVersion)\b'),
    re.compile(r'\b(IsSocVersion)\b'),
    re.compile(r'\b(ASCEND\w+)\b'),
]

KEY_SETTING_PATTERNS = [
    re.compile(r'\b(SetTilingKey)\b'),
    re.compile(r'\b(GET_TPL_TILING_KEY)\b'),
    re.compile(r'\b(GET_TILING_KEY)\b'),
]

STRONG_KEY_SETTING_PATTERNS = [
    re.compile(r'\bSetTilingKey\s*\('),
    re.compile(r'\bsetTilingKey\s*\('),
    re.compile(r'\bset_tiling_key\s*\('),
    re.compile(r'\bGET_TILING_KEY\b'),
    re.compile(r'\bGET_TPL_TILING_KEY\b'),
    re.compile(r'\bREGISTER_TILING_FOR_TILINGKEY\b'),
]

RE_LOCAL_INCLUDE = re.compile(r'#\s*include\s+"([^"]+)"')


def parse_args():
    parser = argparse.ArgumentParser(description="Scout-T: Tiling file reconnaissance")
    parser.add_argument("--op-name", required=True, help="Operator name (e.g. AddRmsNorm)")
    parser.add_argument("--op-path", required=True, help="Operator source directory (contains op_host/ and op_kernel/)")
    parser.add_argument("--npu-arch", required=True, choices=list(PLATFORM_MAP.keys()),
                        help="Target NPU architecture")
    parser.add_argument("--soc-version", default=None, help="SOC version (for display)")
    parser.add_argument("--chip-model", default=None, help="Chip model (for display)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: {op_path}/tests/whitebox/)")
    return parser.parse_args()


def read_file(filepath):
    try:
        with open(filepath, errors='ignore') as f:
            return f.read()
    except OSError:
        return ''


def has_strong_key_setting(content):
    return any(pattern.search(content) for pattern in STRONG_KEY_SETTING_PATTERNS)


def discover_tiling_files(op_host_dir):
    """Glob op_host/**/*tiling*.cpp"""
    results = []
    if not os.path.isdir(op_host_dir):
        return results
    for root, _, files in os.walk(op_host_dir):
        for fname in files:
            if fname.endswith('.cpp') and 'tiling' in fname.lower():
                results.append(os.path.join(root, fname))
    return sorted(results)


def _detect_file_arch(filepath, op_host_dir):
    rel_path = os.path.relpath(filepath, op_host_dir)
    parts = rel_path.split(os.sep)
    filename_lower = os.path.basename(filepath).lower()

    for part in parts:
        part_lower = part.lower()
        for arch_key, arch_val in ARCH_DIR_MAP.items():
            if arch_key in part_lower:
                return arch_val

    for feat_key, feat_val in ARCH_FEATURE_MAP.items():
        if feat_key in filename_lower:
            return feat_val

    return None


def check_platform(filepath, op_host_dir, npu_arch):
    """Determine if a file is active for the target platform.

    Returns (is_active: bool, reason: str|None).
    """
    rel_path = os.path.relpath(filepath, op_host_dir)
    parts = rel_path.split(os.sep)

    file_arch = _detect_file_arch(filepath, op_host_dir)

    if file_arch is not None:
        if file_arch == npu_arch:
            return True, None
        chip = PLATFORM_MAP.get(file_arch, {}).get('chip', file_arch)
        target_chip = PLATFORM_MAP[npu_arch]['chip']
        return False, f'{chip} 专用，目标 {target_chip} 不可达'

    if PLATFORM_MAP[npu_arch]['is_950'] and len(parts) == 1:
        return None, 'pending_arch_aware'

    return True, None


def extract_impl_optiling(content):
    """Extract all IMPL_OP_OPTILING registrations.

    Returns list of (op_name, func_name, line_number).
    Handles multi-line chains: IMPL_OP_OPTILING(OpName) ... .Tiling(FuncName) ... ;
    """
    registrations = []
    for m in re.finditer(r'IMPL_OP_OPTILING\s*\(\s*(\w+)\s*\)', content):
        op_name = m.group(1)
        start = m.end()
        end = content.find(';', start)
        if end == -1:
            end = len(content)
        block = content[start:end]
        tiling_match = re.search(r'\.Tiling\s*\(\s*([\w:]+)\s*\)', block)
        if tiling_match:
            func_name = tiling_match.group(1)
            line_num = content[:m.start()].count('\n') + 1
            registrations.append((op_name, func_name, line_num))
    return registrations


def extract_template_registrations(content):
    """Extract all REGISTER_OPS_TILING_TEMPLATE registrations.

    Returns list of (op_name, class_name, priority, line_number).
    """
    registrations = []
    for m in re.finditer(
        r'REGISTER_OPS_TILING_TEMPLATE\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*\)',
        content,
    ):
        op_name = m.group(1)
        class_name = m.group(2)
        priority = int(m.group(3))
        line_num = content[:m.start()].count('\n') + 1
        registrations.append((op_name, class_name, priority, line_num))
    return registrations


def extract_rtt_registrations(content):
    """Extract all REGISTER_TILING_TEMPLATE registrations.

    Format: REGISTER_TILING_TEMPLATE("OpName", ClassName, Priority)
    Priority may be a number or a macro name.
    Returns list of (op_name, class_name, priority, line_number).
    """
    registrations = []
    pattern = re.compile(
        r'REGISTER_TILING_TEMPLATE\s*\(\s*"(\w+)"\s*,\s*(\w+)\s*,\s*(\w+)\s*\)',
    )
    for m in pattern.finditer(content):
        op_name = m.group(1)
        class_name = m.group(2)
        priority = m.group(3)
        line_num = content[:m.start()].count('\n') + 1
        registrations.append((op_name, class_name, priority, line_num))
    return registrations


def extract_arch_aware_registrations(content):
    """Extract REGISTER_TILING_TEMPLATE_FIA and REGISTER_TILING_TEMPLATE_WITH_ARCH.

    Returns list of (macro_name, op_name, class_name, arch_list, priority, line_num).
    arch_list is a list of DAV_XXXX strings.
    """
    registrations = []
    pattern = re.compile(
        r'(REGISTER_TILING_TEMPLATE_FIA|REGISTER_TILING_TEMPLATE_WITH_ARCH)'
        r'\s*\(\s*(\w+)\s*,\s*(\w+)\s*,'
        r'\s*std::vector[^{]*\{([^}]*)\}\s*\)\s*,'
        r'\s*(\w+)\s*\)',
        re.DOTALL,
    )
    for m in pattern.finditer(content):
        macro_name = m.group(1)
        op_name = m.group(2)
        class_name = m.group(3)
        arch_raw = m.group(4)
        priority = m.group(5)
        arch_list = re.findall(r'DAV_\d+', arch_raw)
        line_num = content[:m.start()].count('\n') + 1
        registrations.append((macro_name, op_name, class_name, arch_list, priority, line_num))
    return registrations


def find_function_def(filepath, func_name):
    """Check if func_name is defined (has body) in filepath.

    Returns (line_number, is_definition):
      - (line, True)  → definition found (has '{' after params)
      - (line, False) → declaration only (has ';' after params)
      - (None, None)  → not found
    """
    content = read_file(filepath)
    escape_name = re.escape(func_name)
    declaration_line = None

    for m in re.finditer(r'(?<![\w.])' + escape_name + r'\s*\(', content):
        pos = skip_parens(content, m.end())
        if pos is None:
            continue

        if find_def_after(content, pos):
            line = content[:m.start()].count('\n') + 1
            return (line, True)
        if content[pos:pos + 200].find(';') != -1 and declaration_line is None:
            line = content[:m.start()].count('\n') + 1
            declaration_line = line

    return (declaration_line, False) if declaration_line else (None, None)


def find_method_def(filepath, class_name):
    """Find the main tiling method definition for a registered class.

    Searches for ClassName::DoOpTiling or ClassName::DoTiling patterns.
    Returns (line_number, is_definition).
    """
    content = read_file(filepath)
    escape_class = re.escape(class_name)

    for method_name in ['DoOpTiling', 'DoTiling']:
        pattern = escape_class + r'\s*::\s*' + method_name + r'\s*\('
        for m in re.finditer(pattern, content):
            pos = skip_parens(content, m.end())
            if pos is None:
                continue

            if find_def_after(content, pos):
                line = content[:m.start()].count('\n') + 1
                return (line, True)

    return (None, False)


def parse_local_includes(filepath, op_path):
    """Parse #include "..." lines and return resolved local header paths.

    Tries resolving relative to the file's directory, then relative to op_path.
    Only returns paths where the file actually exists.
    """
    includes = []
    file_dir = os.path.dirname(filepath)
    content = read_file(filepath)

    for m in RE_LOCAL_INCLUDE.finditer(content):
        inc_path = m.group(1)

        resolved = os.path.normpath(os.path.join(file_dir, inc_path))
        if os.path.isfile(resolved):
            includes.append(resolved)
            continue

        resolved = os.path.normpath(os.path.join(op_path, inc_path))
        if os.path.isfile(resolved):
            includes.append(resolved)

    return includes


def build_tiling_include_roots(op_path, p0_file):
    op_host_dir = os.path.join(op_path, 'op_host')
    family_root = os.path.dirname(op_path)
    family_name = os.path.basename(family_root)
    roots = [
        os.path.dirname(p0_file),
        op_host_dir,
        os.path.join(op_host_dir, 'arch22'),
        os.path.join(op_host_dir, 'arch35'),
        family_root,
        os.path.join(family_root, f'{family_name}_utils', 'op_host'),
        os.path.join(family_root, f'{family_name}_utils'),
    ]
    return [os.path.abspath(root) for root in roots if os.path.isdir(root)]


def should_skip_header(filepath, npu_arch):
    norm = filepath.replace(os.sep, '/')
    skip_parts = ['/tests/', '/ut/', '/st/', '/register/', '/platform/',
                  '/log/', '/util/', '/exe_graph/']
    if any(part in norm for part in skip_parts):
        return True
    if npu_arch != 'DAV_3510' and '/arch35/' in norm:
        return True
    if npu_arch == 'DAV_3510' and '/arch22/' in norm:
        return True
    return False


def resolve_include(inc_path, current_file, include_roots):
    candidates = [os.path.join(os.path.dirname(current_file), inc_path)]
    candidates.extend(os.path.join(root, inc_path) for root in include_roots)
    for candidate in candidates:
        resolved = os.path.abspath(os.path.normpath(candidate))
        if os.path.isfile(resolved):
            return resolved
    return None


def collect_include_closure(entry_file, include_roots, npu_arch, max_depth=8):
    visited = set()
    ordered = []

    def visit(filepath, depth):
        if depth > max_depth:
            return
        content = read_file(filepath)
        for match in RE_LOCAL_INCLUDE.finditer(content):
            resolved = resolve_include(match.group(1), filepath, include_roots)
            if not resolved or resolved in visited:
                continue
            if not resolved.endswith(('.h', '.hpp', '.hh', '.inc')):
                continue
            if should_skip_header(resolved, npu_arch):
                continue
            visited.add(resolved)
            ordered.append(resolved)
            visit(resolved, depth + 1)

    visit(entry_file, 0)
    return ordered


def extract_set_tiling_key_dependency_symbols(content):
    symbols = set()
    for match in re.finditer(r'\bSetTilingKey\s*\(', content):
        end = skip_parens(content, match.end())
        if end is None:
            continue
        expr = content[match.end():end - 1]
        for func_name in re.findall(r'\b([A-Za-z_]\w*)\s*\(', expr):
            if func_name != 'SetTilingKey':
                symbols.add(func_name)
    return symbols


def defines_symbol(content, symbol):
    escaped = re.escape(symbol)
    return bool(re.search(r'\b' + escaped + r'\s*\(', content))


def select_key_contributing_headers(p0_file, include_headers):
    if has_strong_key_setting(read_file(p0_file)):
        return set(), {}

    selected = []
    reasons = {}
    dependency_symbols = set()
    for header in include_headers:
        content = read_file(header)
        if not has_strong_key_setting(content):
            continue
        selected.append(header)
        reasons[header] = 'P0 has no SetTilingKey; header contains strong key setting'
        dependency_symbols.update(extract_set_tiling_key_dependency_symbols(content))

    for symbol in sorted(dependency_symbols):
        for header in include_headers:
            if header in selected:
                continue
            if defines_symbol(read_file(header), symbol):
                selected.append(header)
                reasons[header] = f'defines SetTilingKey dependency symbol: {symbol}'
                break

    return set(selected), reasons


def _check_cpp_sibling(filepath, func_name, visited):
    if not filepath.endswith('.h'):
        return None
    cpp_file = filepath[:-2] + '.cpp'
    if not os.path.isfile(cpp_file) or cpp_file in visited:
        return None
    visited.add(cpp_file)
    line, is_def = find_function_def(cpp_file, func_name)
    if is_def:
        return (cpp_file, line)
    return None


def trace_implementation(p0_file, func_name, op_path, max_hops=5):
    """Trace #include chain to find where func_name is defined.

    Returns (impl_file, impl_line, hop_count, all_visited_files).
    If not found, returns (None, None, hop_count, all_visited_files).
    """
    initial_includes = parse_local_includes(p0_file, op_path)
    visited = set()
    queue = [(f, 1) for f in initial_includes]
    all_visited = set()
    last_hop = 0

    while queue:
        filepath, hop = queue.pop(0)
        if filepath in visited or hop > max_hops:
            continue
        last_hop = hop
        visited.add(filepath)
        all_visited.add(filepath)

        line, is_def = find_function_def(filepath, func_name)
        if is_def:
            return (filepath, line, hop, all_visited)

        sibling = _check_cpp_sibling(filepath, func_name, visited)
        if sibling:
            all_visited.add(sibling[0])
            return (sibling[0], sibling[1], hop, all_visited)

        for inc in parse_local_includes(filepath, op_path):
            if inc not in visited:
                queue.append((inc, hop + 1))

    return (None, None, last_hop, all_visited)


def detect_platform_branches(content):
    """Detect platform branch patterns in content.

    Returns list of (pattern_name, line_number).
    """
    results = []
    for pattern in PLATFORM_BRANCH_PATTERNS:
        for m in pattern.finditer(content):
            line = content[:m.start()].count('\n') + 1
            results.append((m.group(0), line))
    return results


def detect_key_setting(content):
    """Detect key setting method in content.

    Returns the method name string, or None.
    """
    for pattern in KEY_SETTING_PATTERNS:
        m = pattern.search(content)
        if m:
            return m.group(1)
    return None


def relpath(filepath, op_path):
    """Get path relative to op_path, handling files outside op_path."""
    try:
        return os.path.relpath(filepath, op_path)
    except ValueError:
        return filepath


def _build_impl_entries(impl_regs, filepath, op_path):
    entries = []
    for op_name, func_name, reg_line in impl_regs:
        def_line, is_def = find_function_def(filepath, func_name)
        impl_file = impl_line = None
        hop = 0
        traced_files = set()
        if is_def:
            impl_file = filepath
            impl_line = def_line
        elif def_line is None:
            impl_file, impl_line, hop, traced_files = trace_implementation(
                filepath, func_name, op_path)
        entries.append({
            'op_name': op_name, 'func_name': func_name, 'reg_line': reg_line,
            'def_in_p0': is_def, 'impl_file': impl_file, 'impl_line': impl_line,
            'hop': hop, 'traced_files': traced_files,
        })
    return entries


def _build_class_entries(class_regs, filepath):
    entries = []
    for op_name, class_name, priority, reg_line in class_regs:
        dotiling_line, dotiling_is_def = find_method_def(filepath, class_name)
        entries.append({
            'op_name': op_name, 'class_name': class_name, 'priority': priority,
            'reg_line': reg_line,
            'dotiling_line': dotiling_line if dotiling_is_def else None,
        })
    return entries


def _build_arch_aware_entries(arch_aware_regs, filepath, npu_arch):
    entries = []
    for macro_name, op_name, class_name, arch_list, priority, reg_line in arch_aware_regs:
        is_active = npu_arch in arch_list
        dotiling_line = None
        if is_active:
            dl, dotiling_is_def = find_method_def(filepath, class_name)
            dotiling_line = dl if dotiling_is_def else None
        entries.append({
            'macro_name': macro_name, 'op_name': op_name,
            'class_name': class_name, 'arch_list': arch_list,
            'priority': priority, 'reg_line': reg_line,
            'is_active': is_active, 'dotiling_line': dotiling_line,
        })
    return entries


def _collect_p1_candidates(filepath, op_path, impl_entries):
    p1 = set()
    for entry in impl_entries:
        p1.update(entry.get('traced_files', set()))
        if entry['impl_file'] and not entry['def_in_p0']:
            p1.add(entry['impl_file'])
    return p1


def collect_key_header_p1(analyses, op_path, npu_arch):
    p1_files = set()
    reasons = {}
    for analysis in analyses:
        p0_file = analysis['filepath']
        if has_strong_key_setting(read_file(p0_file)):
            continue
        include_roots = build_tiling_include_roots(op_path, p0_file)
        headers = collect_include_closure(p0_file, include_roots, npu_arch)
        selected, selected_reasons = select_key_contributing_headers(p0_file, headers)
        p1_files.update(selected)
        reasons.update(selected_reasons)
    return p1_files, reasons


def analyze_tiling_file(filepath, op_path, op_host_dir, npu_arch):
    """Analyze a single tiling file for registration info.

    Returns a dict with analysis results, or None if no registration found.
    """
    content = read_file(filepath)
    if not content:
        return None

    impl_regs = extract_impl_optiling(content)
    template_regs = extract_template_registrations(content)
    rtt_regs = extract_rtt_registrations(content)
    arch_aware_regs = extract_arch_aware_registrations(content)

    if not any([impl_regs, template_regs, rtt_regs, arch_aware_regs]):
        return None

    impl_entries = _build_impl_entries(impl_regs, filepath, op_path)
    template_entries = _build_class_entries(template_regs, filepath)
    rtt_entries = _build_class_entries(rtt_regs, filepath)
    arch_aware_entries = _build_arch_aware_entries(arch_aware_regs, filepath, npu_arch)
    p1_candidates = _collect_p1_candidates(filepath, op_path, impl_entries)

    return {
        'filepath': filepath,
        'impl_entries': impl_entries,
        'template_entries': template_entries,
        'rtt_entries': rtt_entries,
        'arch_aware_entries': arch_aware_entries,
        'platform_branches': detect_platform_branches(content),
        'key_setting': detect_key_setting(content),
        'p1_candidates': p1_candidates,
    }


def build_registrations(analysis, op_path):
    """Build unified registrations list from analysis entries."""
    regs = []

    for entry in analysis['impl_entries']:
        impl_rel = relpath(entry['impl_file'], op_path) if entry['impl_file'] else None
        regs.append({
            'macro': 'IMPL_OP_OPTILING',
            'op_name': entry['op_name'],
            'entry_type': 'function',
            'entry_function': entry['func_name'],
            'entry_file': impl_rel,
            'entry_line': entry['impl_line'],
            'entry_in_p0': bool(entry['def_in_p0']),
            'hop': entry['hop'],
        })

    for macro, entries_key in [('REGISTER_OPS_TILING_TEMPLATE', 'template_entries'),
                                ('REGISTER_TILING_TEMPLATE', 'rtt_entries')]:
        for entry in analysis[entries_key]:
            regs.append({
                'macro': macro,
                'op_name': entry['op_name'],
                'entry_type': 'method',
                'entry_function': 'DoOpTiling',
                'class_name': entry['class_name'],
                'priority': str(entry['priority']),
                'entry_file': relpath(analysis['filepath'], op_path) if entry['dotiling_line'] else None,
                'entry_line': entry['dotiling_line'],
                'entry_in_p0': entry['dotiling_line'] is not None,
                'hop': 0,
            })

    for entry in analysis['arch_aware_entries']:
        if not entry['is_active']:
            continue
        regs.append({
            'macro': entry['macro_name'],
            'op_name': entry['op_name'],
            'entry_type': 'method',
            'entry_function': 'DoOpTiling',
            'class_name': entry['class_name'],
            'priority': str(entry['priority']),
            'arch_list': entry['arch_list'],
            'entry_file': relpath(analysis['filepath'], op_path) if entry['dotiling_line'] else None,
            'entry_line': entry['dotiling_line'],
            'entry_in_p0': entry['dotiling_line'] is not None,
            'hop': 0,
        })

    return regs


def write_json(output_dir, scan_result, op_name, npu_arch, soc_version):
    analyses = scan_result['analyses']
    all_p1 = scan_result['all_p1']
    excluded_files = scan_result['excluded_files']
    total_count = scan_result['total_count']
    valid_count = scan_result['valid_count']
    op_path = scan_result['op_path']
    p1_reasons = scan_result.get('p1_reasons', {})
    """Write S2P0_scout_t.json."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'S2P0_scout_t.json')

    entries = []
    for analysis in analyses:
        regs = build_registrations(analysis, op_path)
        branches = [
            {'pattern': name, 'line': line}
            for name, line in analysis['platform_branches']
        ]
        entries.append({
            'file': relpath(analysis['filepath'], op_path),
            'priority': 'P0',
            'registrations': regs,
            'key_setting': analysis['key_setting'],
            'platform_branches': branches,
        })

    data = {
        'operator': op_name,
        'platform': {
            'npu_arch': npu_arch,
            'soc_version': soc_version or '',
        },
        'scan_baseline': {
            'total_files': total_count,
            'valid_files': valid_count,
            'excluded_files': len(excluded_files),
        },
        'entries': entries,
        'p1_files': sorted(relpath(f, op_path) for f in all_p1),
        'p1_file_reasons': {
            relpath(f, op_path): reason for f, reason in sorted(p1_reasons.items())
        },
        'excluded': [
            {'file': relpath(f, op_path), 'reason': reason}
            for f, reason in excluded_files
        ],
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return json_path


def _append_branches(lines, branches):
    if branches:
        branch_str = ', '.join(f'{name}@行{line}' for name, line in branches[:5])
        if len(branches) > 5:
            branch_str += f' ...（共 {len(branches)} 处）'
        lines.append(f'  平台分支: 有（{branch_str}）')
    else:
        lines.append('  平台分支: 无')


def _append_impl_entries(lines, analysis, rel, platform_info):
    for entry in analysis['impl_entries']:
        lines.append(f'  注册宏: IMPL_OP_OPTILING ({entry["op_name"]})')
        lines.append(f'  入口函数: {entry["func_name"]}')
        if entry['def_in_p0']:
            lines.append(f'  定义位置: {rel}:{entry["impl_line"]}（P0 同文件）')
        elif entry['impl_file']:
            impl_rel = relpath(entry['impl_file'], platform_info['op_path'])
            lines.append(f'  定义位置: {impl_rel}:{entry["impl_line"]}（P1, hop {entry["hop"]}）')
        else:
            lines.append('  定义位置: 未找到')
        if analysis['key_setting']:
            lines.append(f'  key 设置: {analysis["key_setting"]}')
        _append_branches(lines, analysis['platform_branches'])
        lines.append('')


def _append_class_entries(lines, analysis, entries_key, macro_name):
    for entry in analysis[entries_key]:
        lines.append(f'  注册宏: {macro_name} ({entry["op_name"]})')
        lines.append(f'  注册类: {entry["class_name"]}（优先级 {entry["priority"]}）')
        if entry['dotiling_line']:
            lines.append(f'  入口方法: {entry["class_name"]}::DoOpTiling@行{entry["dotiling_line"]}')
        else:
            lines.append(f'  入口方法: {entry["class_name"]}::DoOpTiling（未在 P0 中找到定义）')
        _append_branches(lines, analysis['platform_branches'])
        lines.append('')


def _append_arch_aware_entries(lines, analysis):
    for entry in analysis['arch_aware_entries']:
        if not entry['is_active']:
            continue
        arch_str = ', '.join(entry['arch_list'])
        lines.append(f'  注册宏: {entry["macro_name"]} ({entry["op_name"]})')
        lines.append(f'  注册类: {entry["class_name"]}（优先级 {entry["priority"]}）')
        lines.append(f'  arch: {arch_str}')
        if entry['dotiling_line']:
            lines.append(f'  入口方法: {entry["class_name"]}::DoOpTiling@行{entry["dotiling_line"]}')
        else:
            lines.append(f'  入口方法: {entry["class_name"]}::DoOpTiling（未在 P0 中找到定义）')
        _append_branches(lines, analysis['platform_branches'])
        lines.append('')


def _append_p1_section(lines, all_p1, op_path, p1_reasons=None):
    lines.append('## P1 (候选文件)')
    lines.append('')
    if all_p1:
        for p1_path in sorted(all_p1):
            reason = ''
            if p1_reasons and p1_path in p1_reasons:
                reason = f' — {p1_reasons[p1_path]}'
            lines.append(f'  P1: {relpath(p1_path, op_path)}{reason}')
    else:
        lines.append('  （无）')
    lines.append('')


def _append_p2_section(lines, excluded_files, op_path):
    lines.append('## P2 (排除文件)')
    lines.append('')
    if excluded_files:
        for fpath, reason in excluded_files:
            lines.append(f'  P2: {relpath(fpath, op_path)}')
            lines.append(f'    排除原因: {reason}')
    else:
        lines.append('  （无）')
    lines.append('')


def write_report(output_dir, scan_result, platform_info, npu_arch):
    analyses = scan_result['analyses']
    all_p1 = scan_result['all_p1']
    excluded_files = scan_result['excluded_files']
    total_count = scan_result['total_count']
    valid_count = scan_result['valid_count']
    """Write S2P0_scout_t.md report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'S2P0_scout_t.md')
    op_path = platform_info['op_path']

    lines = ['=== TILING SCOUT REPORT ===', '']
    lines.append(f'目标平台: {npu_arch}')
    if platform_info.get('soc_version'):
        lines[-1] += f' ({platform_info["soc_version"]})'
    lines.append('')
    lines.append('## 扫描基准线')
    lines.append('')
    lines.append(f'  全量总计: {total_count} 个文件')
    lines.append(f'  有效（目标平台可达）: {valid_count} 个文件')
    lines.append(f'  排除（目标平台不可达）: {len(excluded_files)} 个文件')
    lines.append('')
    lines.append('## P0 (入口文件)')
    lines.append('')

    for analysis in analyses:
        rel = relpath(analysis['filepath'], op_path)
        lines.append(f'  P0: {rel}')
        lines.append('')
        _append_impl_entries(lines, analysis, rel, platform_info)
        _append_class_entries(lines, analysis, 'template_entries',
                              'REGISTER_OPS_TILING_TEMPLATE')
        _append_class_entries(lines, analysis, 'rtt_entries',
                              'REGISTER_TILING_TEMPLATE')
        _append_arch_aware_entries(lines, analysis)

    _append_p1_section(lines, all_p1, op_path, scan_result.get('p1_reasons', {}))
    _append_p2_section(lines, excluded_files, op_path)

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    return report_path


def _classify_tiling_files(all_files, op_host_dir, npu_arch):
    valid_files = []
    pending_files = []
    excluded_files = []
    for fpath in all_files:
        is_active, reason = check_platform(fpath, op_host_dir, npu_arch)
        if is_active is True:
            valid_files.append(fpath)
        elif is_active is None:
            pending_files.append(fpath)
        else:
            excluded_files.append((fpath, reason))
    return valid_files, pending_files, excluded_files


def _resolve_pending_files(pending_files, npu_arch):
    valid_files = []
    excluded_files = []
    target_chip = PLATFORM_MAP[npu_arch]['chip']
    for fpath in pending_files:
        content = read_file(fpath)
        arch_aware_regs = extract_arch_aware_registrations(content)
        if arch_aware_regs:
            has_match = any(npu_arch in regs[3] for regs in arch_aware_regs)
            if has_match:
                valid_files.append(fpath)
            else:
                arch_str = ', '.join('/'.join(regs[3]) for regs in arch_aware_regs)
                excluded_files.append(
                    (fpath, f'arch-aware 注册均不含目标 {target_chip}（{arch_str}）'))
        else:
            valid_files.append(fpath)
    return valid_files, excluded_files


def _process_valid_files(valid_files, op_path, op_host_dir, npu_arch):
    analyses = []
    p1_siblings = set()
    excluded_files = []
    target_chip = PLATFORM_MAP[npu_arch]['chip']
    for fpath in valid_files:
        analysis = analyze_tiling_file(fpath, op_path, op_host_dir, npu_arch)
        if not analysis:
            p1_siblings.add(fpath)
            continue
        has_other_macro = (analysis['impl_entries'] or analysis['template_entries']
                           or analysis['rtt_entries'])
        arch_entries = analysis['arch_aware_entries']
        if arch_entries and not has_other_macro:
            if all(not e['is_active'] for e in arch_entries):
                arch_str = ', '.join('/'.join(e['arch_list']) for e in arch_entries)
                excluded_files.append(
                    (fpath, f'arch-aware 注册均不含目标 {target_chip}（{arch_str}）'))
                continue
        analyses.append(analysis)
    return analyses, p1_siblings, excluded_files


def _log_summary(info, analyses, all_p1):
    report_path = info['report_path']
    json_path = info['json_path']
    all_files = info['all_files']
    valid_count = info['valid_count']
    excluded_files = info['excluded_files']
    op_path = info['op_path']
    p0_files = sorted(a['filepath'] for a in analyses)
    _logger.info("S2P0_scout_t.md  written to: %s", report_path)
    _logger.info("S2P0_scout_t.json written to: %s", json_path)
    _logger.info("  全量: %d | 有效: %d | 排除: %d",
                 len(all_files), valid_count, len(excluded_files))
    entry_count = sum(
        len(a['impl_entries']) + len(a['template_entries'])
        + len(a['rtt_entries'])
        + len([e for e in a['arch_aware_entries'] if e['is_active']])
        for a in analyses
    )
    _logger.info("  入口条目: %d | P0: %d | P1: %d | P2: %d",
                 entry_count, len(p0_files), len(all_p1), len(excluded_files))
    for label, items in [("P0 (入口文件):", p0_files),
                         ("P1 (候选文件):", sorted(all_p1))]:
        _logger.info("")
        _logger.info(label)
        for f in items:
            _logger.info("  - %s", relpath(f, op_path))
    _logger.info("")
    _logger.info("P2 (排除文件):")
    for f, _ in excluded_files:
        _logger.info("  - %s", relpath(f, op_path))


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    args = parse_args()
    op_path = os.path.abspath(args.op_path)
    op_host_dir = os.path.join(op_path, 'op_host')
    npu_arch = args.npu_arch
    output_dir = args.output_dir or os.path.join(op_path, 'tests', 'whitebox')

    all_files = discover_tiling_files(op_host_dir)
    if not all_files:
        _logger.error("ERROR: no tiling .cpp files found in %s", op_host_dir)
        sys.exit(1)

    valid_files, pending_files, excluded_files = _classify_tiling_files(
        all_files, op_host_dir, npu_arch)

    pending_valid, pending_excluded = _resolve_pending_files(pending_files, npu_arch)
    valid_files.extend(pending_valid)
    excluded_files.extend(pending_excluded)

    analyses, p1_siblings, analysis_excluded = _process_valid_files(
        valid_files, op_path, op_host_dir, npu_arch)
    excluded_files.extend(analysis_excluded)

    valid_count = len(analyses) + len(p1_siblings)
    all_p1 = set(p1_siblings)
    for analysis in analyses:
        all_p1.update(analysis['p1_candidates'])
    key_header_p1, p1_reasons = collect_key_header_p1(analyses, op_path, npu_arch)
    all_p1.update(key_header_p1)

    platform_info = {
        'op_path': op_path,
        'soc_version': args.soc_version,
        'chip_model': args.chip_model,
    }

    scan_result = {
        'analyses': analyses, 'all_p1': all_p1,
        'p1_reasons': p1_reasons,
        'excluded_files': excluded_files,
        'total_count': len(all_files), 'valid_count': valid_count + len(key_header_p1),
        'op_path': op_path,
    }

    report_path = write_report(output_dir, scan_result, platform_info, npu_arch)
    json_path = write_json(output_dir, scan_result,
                           args.op_name, npu_arch, args.soc_version)

    log_info = {
        'report_path': report_path, 'json_path': json_path,
        'all_files': all_files, 'valid_count': scan_result['valid_count'],
        'excluded_files': excluded_files, 'op_path': op_path,
    }
    _log_summary(log_info, analyses, all_p1)


if __name__ == '__main__':
    main()
