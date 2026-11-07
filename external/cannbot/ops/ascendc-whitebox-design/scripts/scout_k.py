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
"""Scout-K: Kernel file reconnaissance script.

Discovers kernel files in op_kernel/, detects dispatch patterns (TILING_KEY_IS,
TILING_KEY_VAR, __global__), extracts key values,
evaluates #if __NPU_ARCH__ guards, and outputs .md + .json reports.

Usage:
    python scout_k.py \
        --op-name AddRmsNorm \
        --op-path /path/to/operator \
        --npu-arch DAV_2201 \
        [--soc-version Ascend910B3] \
        [--output-dir /path/to/output]

Output:
    {output_dir}/S2P0_scout_k.md
    {output_dir}/S2P0_scout_k.json
"""
import argparse
import json
import logging
import os
import re
import sys

from _scout_common import PLATFORM_MAP, ARCH_DIR_MAP, ARCH_FEATURE_MAP, RE_TILING_KEY_IS

_logger = logging.getLogger(__name__)

RE_TILING_KEY_VAR_IF = re.compile(r'#\s*(?:if|elif)\s+.*?TILING_KEY_VAR\s*==\s*(\S+)')
RE_REGISTER_TILING_KEY = re.compile(
    r'REGISTER_TILING_FOR_TILINGKEY\(\s*"([^"]+)"\s*,\s*([^)]+?)\s*\)')
RE_REGISTER_TILING_KEY_MULTILINE = re.compile(
    r'REGISTER_TILING_FOR_TILINGKEY\(\s*"([^"]+)"\s*,\s*([^)]+?)\s*\)',
    re.DOTALL)
RE_STANDALONE_TK = re.compile(r'^\s*TILING_KEY_IS\([^)]+\)\s*;')
RE_GLOBAL_FUNC = re.compile(
    r'__global__\s+(?:__aicore__\s+)?void\s+(\w+)\s*\(')
RE_TEMPLATE_GLOBAL = re.compile(
    r'template\s*<[^>]*>\s*__global__\s+(?:__aicore__\s+)?void\s+(\w+)\s*\(')
RE_IF_GUARD = re.compile(r'#\s*if\b')
RE_ELIF_GUARD = re.compile(r'#\s*elif\b')
RE_ELSE_GUARD = re.compile(r'#\s*else\b')
RE_ENDIF_GUARD = re.compile(r'#\s*endif\b')
RE_NPU_ARCH_DEFINED = re.compile(r'defined\s*\(\s*__NPU_ARCH__\s*\)')
RE_NPU_ARCH_VALUES = re.compile(r'__NPU_ARCH__\s*==\s*(\d+)')
RE_NPU_ARCH_BARE = re.compile(r'__NPU_ARCH__\s*==\s*(\d+)')
RE_INCLUDE = re.compile(r'#\s*include\s+"([^"]+)"')
RE_IF_CONSTEXPR = re.compile(r'if\s+constexpr')
RE_INIT_CALL = re.compile(r'\b(\w+)\s*\.\s*Init\s*\(')
SOURCE_EXTS = ('.h', '.hpp', '.hh', '.inc', '.inl', '.cpp', '.cc', '.cxx')


def parse_args():
    parser = argparse.ArgumentParser(description="Scout-K: Kernel file reconnaissance")
    parser.add_argument("--op-name", required=True, help="Operator name")
    parser.add_argument("--op-path", required=True,
                        help="Operator source directory (contains op_kernel/)")
    parser.add_argument("--npu-arch", required=True, choices=list(PLATFORM_MAP.keys()),
                        help="Target NPU architecture")
    parser.add_argument("--soc-version", default=None, help="SOC version (for display)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: {op_path}/tests/whitebox/)")
    return parser.parse_args()


def relpath(path, base):
    return os.path.relpath(path, base)


def discover_kernel_files(op_kernel_dir):
    files = []
    for root, _, filenames in os.walk(op_kernel_dir):
        for fn in filenames:
            if fn.endswith('.cpp'):
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def _check_feature_exclusion(name_no_ext, target, target_npu_arch):
    for feature in ARCH_FEATURE_MAP:
        if feature == '_apt':
            match = name_no_ext.endswith('_apt')
        else:
            match = feature in name_no_ext.lower()
        if match and not target['is_950']:
            label = '_apt' if feature == '_apt' else feature
            return 'excluded', f"Ascend950 专用（{label}），目标 {target_npu_arch} 不可达"
    return None


def classify_platform(filepath, op_kernel_dir, target_npu_arch):
    target = PLATFORM_MAP[target_npu_arch]
    rel = relpath(filepath, op_kernel_dir)
    parts = rel.replace('\\', '/').split('/')
    basename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(basename)[0]

    for part in parts[:-1]:
        if part not in ARCH_DIR_MAP:
            continue
        mapped = ARCH_DIR_MAP[part]
        if mapped != target_npu_arch:
            return 'excluded', f"Ascend950 专用（{part}/ 目录），目标 {target_npu_arch} 不可达" \
                if mapped == 'DAV_3510' else \
                f"{mapped} 专用（{part}/ 目录），目标 {target_npu_arch} 不可达"

    result = _check_feature_exclusion(name_no_ext, target, target_npu_arch)
    if result:
        return result

    if not target['is_950']:
        return 'valid', None

    is_root = len(parts) == 1
    has_950_feature = any(
        p in ARCH_DIR_MAP and ARCH_DIR_MAP[p] == 'DAV_3510'
        for p in parts[:-1]
    )
    has_950_feature = has_950_feature or any(
        f in name_no_ext.lower() for f in ARCH_FEATURE_MAP
    )
    if is_root and not has_950_feature:
        return 'excluded', f"Ascend910B 实现（非 arch35），目标 Ascend950 不可达"

    return 'valid', None


def evaluate_npu_arch_guard(guard_line, target_arch_num):
    has_npu = RE_NPU_ARCH_DEFINED.search(guard_line)
    values = [m.group(1) for m in RE_NPU_ARCH_VALUES.finditer(guard_line)]

    if not values:
        bare = RE_NPU_ARCH_BARE.search(guard_line)
        if bare:
            values = [bare.group(1)]

    if not values:
        return None

    if has_npu:
        defined_pos = RE_NPU_ARCH_DEFINED.search(guard_line).start()
        prefix = guard_line[:defined_pos]
        is_negated = bool(re.search(r'!\s*\(?\s*$', prefix))

        if is_negated:
            return target_arch_num not in values
        else:
            return target_arch_num in values
    else:
        return target_arch_num in values


def _update_guard_stack(guard_stack, stripped):
    if RE_IF_GUARD.search(stripped) and not RE_ELIF_GUARD.search(stripped):
        guard_stack.append(stripped)
    elif RE_ELIF_GUARD.search(stripped):
        if guard_stack:
            guard_stack[-1] = stripped
    elif RE_ENDIF_GUARD.search(stripped):
        if guard_stack:
            guard_stack.pop()


def _check_branch_close(tk_is_branch_stack, line_idx, brace_depth, result, just_pushed_line):
    if not tk_is_branch_stack or line_idx == just_pushed_line:
        return
    top = tk_is_branch_stack[-1]
    if brace_depth <= top['entry_depth']:
        result['branch_ranges'].append((
            top['key_index'], top['body_start'], line_idx + 1,
        ))
        tk_is_branch_stack.pop()


def _process_tk_is(line, stripped, line_idx, ctx, in_if_condition):
    result = ctx['result']
    guard_stack = ctx['guard_stack']
    tk_is_branch_stack = ctx['tk_is_branch_stack']
    brace_depth = ctx['brace_depth']
    result['total_tk_is_grep_count'] += len(list(RE_TILING_KEY_IS.finditer(line)))
    is_conditional = False
    if re.search(r'\b(if|else\s+if)\b', stripped):
        is_conditional = True
        in_if_condition = True
    elif in_if_condition:
        if stripped.startswith('||') or stripped.startswith('TILING_KEY_IS'):
            is_conditional = True
        else:
            in_if_condition = False

    if RE_STANDALONE_TK.match(line):
        is_conditional = False
        in_if_condition = False

    current_guard = guard_stack[-1] if guard_stack else None
    just_pushed_line = -1

    for m in RE_TILING_KEY_IS.finditer(line):
        key_val = re.sub(r'UL$', '', m.group(1).strip())
        if is_conditional:
            key_index = len(result['tiling_key_is_keys'])
            result['tiling_key_is_keys'].append({
                'value': key_val, 'line': line_idx + 1, 'guard': current_guard,
            })
            tk_is_branch_stack.append({
                'key_index': key_index, 'body_start': line_idx + 1,
                'entry_depth': brace_depth,
            })
            just_pushed_line = line_idx
        else:
            result['tiling_registration_keys'].append({
                'value': key_val, 'line': line_idx + 1,
            })

    return in_if_condition, just_pushed_line


def _process_line_extras(line, stripped, line_idx, guard_stack, result):
    if stripped.startswith('#'):
        tk_var_m = RE_TILING_KEY_VAR_IF.search(stripped)
        if tk_var_m:
            result['tiling_key_var_keys'].append({
                'value': tk_var_m.group(1), 'line': line_idx + 1,
                'guard': guard_stack[-1] if guard_stack else None,
                'expression': stripped,
            })

    reg_m = RE_REGISTER_TILING_KEY.search(line)
    if reg_m:
        result['register_tiling_keys'].append({
            'value': reg_m.group(1).strip(), 'line': line_idx + 1,
            'tiling_class': reg_m.group(2).strip(),
        })

    if 'REGISTER_TILING_DEFAULT' in line:
        result['has_register_tiling_default'] = True

    gf_m = RE_GLOBAL_FUNC.search(line)
    if gf_m:
        result['global_functions'].append({
            'name': gf_m.group(1), 'line': line_idx + 1,
            'is_template': bool(RE_TEMPLATE_GLOBAL.search(line)),
        })

    if RE_IF_CONSTEXPR.search(line):
        result['if_constexpr_count'] += 1


def _collect_multiline_register_keys(lines, result):
    full_content = ''.join(lines)
    existing_lines = {k['line'] for k in result['register_tiling_keys']}
    for m in RE_REGISTER_TILING_KEY_MULTILINE.finditer(full_content):
        line_num = full_content[:m.start()].count('\n') + 1
        if line_num in existing_lines:
            continue
        expr = m.group(1).strip()
        tiling_class = m.group(2).strip().rstrip(';').rstrip(')')
        result['register_tiling_keys'].append({
            'value': expr,
            'line': line_num,
            'tiling_class': tiling_class,
        })


def scan_file(filepath, target_arch_num):
    lines = _read_lines(filepath)
    if lines is None:
        lines = []

    result = {
        'tiling_key_is_keys': [],
        'tiling_key_var_keys': [],
        'register_tiling_keys': [],
        'tiling_registration_keys': [],
        'global_functions': [],
        'if_constexpr_count': 0,
        'has_register_tiling_default': False,
        'total_tk_is_grep_count': 0,
        'branch_ranges': [],
    }

    guard_stack = []
    in_if_condition = False
    brace_depth = 0
    tk_is_branch_stack = []
    just_pushed_line = -1
    ctx = {
        'result': result,
        'guard_stack': guard_stack,
        'tk_is_branch_stack': tk_is_branch_stack,
    }

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        brace_depth += line.count('{') - line.count('}')
        ctx['brace_depth'] = brace_depth

        _check_branch_close(tk_is_branch_stack, line_idx, brace_depth,
                            result, just_pushed_line)
        _update_guard_stack(guard_stack, stripped)

        tk_is_matches = list(RE_TILING_KEY_IS.finditer(line))
        if tk_is_matches:
            in_if_condition, jp = _process_tk_is(
                line, stripped, line_idx, ctx, in_if_condition)
            if jp >= 0:
                just_pushed_line = jp

        _process_line_extras(line, stripped, line_idx, guard_stack, result)

    _collect_multiline_register_keys(lines, result)

    return result


def _read_lines(filepath):
    try:
        with open(filepath, 'r', errors='replace') as f:
            return f.readlines()
    except OSError:
        return None


def _resolve_include(file_dir, op_kernel_dir, inc_path, op_path=None):
    bases = [file_dir, op_kernel_dir]
    if op_path:
        bases.extend([op_path, os.path.dirname(op_path)])
    for base in bases:
        c = os.path.normpath(os.path.join(base, inc_path))
        if os.path.isfile(c) and c.endswith(SOURCE_EXTS):
            return c
    return None


def find_includes(filepath, op_kernel_dir, op_path=None):
    included_h = []
    file_dir = os.path.dirname(filepath)
    file_lines = _read_lines(filepath)
    if file_lines is None:
        return included_h
    for line in file_lines:
        m = RE_INCLUDE.search(line)
        if not m:
            continue
        resolved = _resolve_include(file_dir, op_kernel_dir, m.group(1), op_path)
        if resolved:
            included_h.append(resolved)
    return included_h


def _line_number(content, pos):
    return content[:pos].count('\n') + 1


def _extract_op_template_before_init(content, op_var, init_pos):
    window_start = max(0, init_pos - 4000)
    prefix = content[window_start:init_pos]
    escaped_var = re.escape(op_var)
    template_pat = re.compile(
        r'\b([A-Za-z_]\w*(?:::\w+)*)\s*<[^;{}]{0,2500}?\b'
        + escaped_var + r'\s*(?:\(|;|=)',
        re.DOTALL,
    )
    matches = list(template_pat.finditer(prefix))
    if matches:
        return matches[-1].group(1)

    plain_pat = re.compile(
        r'\b([A-Za-z_]\w*(?:::\w+)*)\s+' + escaped_var + r'\s*(?:\(|;|=)'
    )
    matches = list(plain_pat.finditer(prefix))
    if matches:
        candidate = matches[-1].group(1)
        if candidate not in ('auto', 'const'):
            return candidate
    return None


def _nearest_condition(lines, init_line):
    start = max(0, init_line - 12)
    for idx in range(init_line - 1, start - 1, -1):
        line = lines[idx].strip()
        if 'if constexpr' in line or 'else if constexpr' in line or re.search(r'\bif\s*\(', line):
            cond_lines = []
            for j in range(idx, min(len(lines), idx + 4)):
                cond_lines.append(lines[j].strip())
                if '{' in lines[j]:
                    break
            return ' '.join(cond_lines)
    return ''


def extract_reachable_kernel_templates(filepath, op_kernel_dir):
    content = ''.join(_read_lines(filepath) or [])
    if not content:
        return [], []
    lines = content.splitlines()
    templates = []
    unresolved = []
    seen = set()
    for match in RE_INIT_CALL.finditer(content):
        op_var = match.group(1)
        init_line = _line_number(content, match.start())
        template = _extract_op_template_before_init(content, op_var, match.start())
        item = {
            'op_var': op_var,
            'init_line': init_line,
            'source_file': relpath(filepath, op_kernel_dir),
            'condition': _nearest_condition(lines, init_line),
        }
        if template:
            item['template'] = template
            key = (template, init_line, op_var)
            if key not in seen:
                seen.add(key)
                templates.append(item)
        else:
            unresolved.append(item)
    return templates, unresolved


def collect_include_closure(seed_files, op_kernel_dir, op_path, max_depth=8):
    visited = set()
    files = {}
    edges = []

    def visit(filepath, depth, chain):
        if depth > max_depth:
            return
        for inc_path in find_includes(filepath, op_kernel_dir, op_path):
            edge = {
                'source': relpath(filepath, op_kernel_dir),
                'target': relpath(inc_path, op_kernel_dir),
                'depth': depth + 1,
            }
            edges.append(edge)
            if inc_path in visited:
                continue
            visited.add(inc_path)
            files[inc_path] = {
                'file': relpath(inc_path, op_kernel_dir),
                'include_level': depth + 1,
                'included_by': relpath(filepath, op_kernel_dir),
                'include_chain': [relpath(p, op_kernel_dir) for p in chain + [inc_path]],
            }
            visit(inc_path, depth + 1, chain + [inc_path])

    for seed in seed_files:
        visit(seed, 0, [seed])
    return files, edges


def _template_matches_file(filepath, template_names):
    content = ''.join(_read_lines(filepath) or [])
    matched = []
    for name in template_names:
        escaped = re.escape(name)
        patterns = [
            r'\b(?:class|struct)\s+' + escaped + r'\b',
            r'\b' + escaped + r'\s*<',
            r'\b' + escaped + r'\s*::',
        ]
        if any(re.search(pattern, content) for pattern in patterns):
            matched.append(name)
    return matched


def detect_dispatch_type(scan_result, independent_register_count=0):
    has_tk_is = len(scan_result['tiling_key_is_keys']) > 0
    has_standalone = len(scan_result['tiling_registration_keys']) > 0
    has_tk_var = (len(scan_result['tiling_key_var_keys']) > 0 or
                  independent_register_count > 0)
    has_constexpr = scan_result['if_constexpr_count'] > 0

    if (has_tk_is or has_standalone) and has_tk_var:
        return 'hybrid'
    if has_tk_is:
        return 'A'
    if has_tk_var:
        return 'B'
    if has_constexpr:
        return 'C_constexpr'
    return 'C'


def auto_describe(dispatch_type, scan_result, independent_register_count=0):
    tk_is_count = len(scan_result['tiling_key_is_keys'])
    tk_var_count = (len(scan_result['tiling_key_var_keys']) +
                    independent_register_count)
    reg_count = len(scan_result['tiling_registration_keys'])

    if dispatch_type == 'hybrid':
        parts = []
        if tk_is_count > 0:
            parts.append(f"TILING_KEY_IS if/else 链（{tk_is_count} 个 key）")
        if reg_count > 0:
            parts.append(f"TILING_KEY_IS 注册（{reg_count} 个）")
        parts.append(f"#if TILING_KEY_VAR 编译时分派（{tk_var_count} 个 key）")
        return " + ".join(parts)
    if dispatch_type == 'A':
        return f"TILING_KEY_IS if/else 链，{tk_is_count} 个 key"
    if dispatch_type == 'B':
        if scan_result['register_tiling_keys']:
            return f"REGISTER_TILING_FOR_TILINGKEY 注册，{tk_var_count} 个 key"
        return f"#if TILING_KEY_VAR 编译时分派，{tk_var_count} 个 key"
    if dispatch_type == 'C_constexpr':
        return f"if constexpr 模板分派，{scan_result['if_constexpr_count']} 个分支"
    return "单 kernel 入口，无 dispatch 分支"


def count_keys_in_file(filepath):
    count = 0
    try:
        with open(filepath, 'r', errors='replace') as f:
            for line in f:
                count += len(RE_TILING_KEY_IS.findall(line))
    except OSError:
        pass
    return count


def is_key_active(key_entry, target_arch_num):
    guard = key_entry.get('guard')
    if not guard:
        return True, None
    result = evaluate_npu_arch_guard(guard, target_arch_num)
    if result is None:
        return True, None
    if result:
        return True, None
    return False, guard


def _build_keys(scan, target_arch_num):
    keys = []
    for k in scan['tiling_key_is_keys']:
        active, guard = is_key_active(k, target_arch_num)
        keys.append({
            'value': k['value'], 'line': k['line'], 'active': active,
            'arch_guard': guard, 'source': 'TILING_KEY_IS',
        })

    for k in scan['tiling_key_var_keys']:
        active, guard = is_key_active(k, target_arch_num)
        keys.append({
            'value': k['value'], 'line': k['line'], 'active': active,
            'arch_guard': guard, 'source': 'TILING_KEY_VAR',
        })

    nested_register_lines = set()
    for k in scan['register_tiling_keys']:
        parent_idx = None
        for idx, start, end in scan['branch_ranges']:
            if start <= k['line'] <= end:
                parent_idx = idx
                break
        if parent_idx is not None:
            keys[parent_idx]['tiling_key_var'] = k['value']
            keys[parent_idx]['tiling_class'] = k.get('tiling_class')
            nested_register_lines.add(k['line'])
        else:
            keys.append({
                'value': k['value'], 'line': k['line'], 'active': True,
                'arch_guard': None, 'source': 'REGISTER_TILING_FOR_TILINGKEY',
                'tiling_class': k.get('tiling_class'),
            })

    for k in scan['tiling_registration_keys']:
        keys.append({
            'value': k['value'], 'line': k['line'], 'active': True,
            'arch_guard': None, 'source': 'TILING_KEY_IS_standalone',
        })

    return keys, nested_register_lines


def build_entry(filepath, op_kernel_dir, target_arch_num, source_cpp=None,
                priority='P0', kind='kernel_entry', include_info=None,
                matched_templates=None):
    scan = scan_file(filepath, target_arch_num)
    keys, nested_register_lines = _build_keys(scan, target_arch_num)

    independent_register_count = len(scan['register_tiling_keys']) - len(nested_register_lines)
    dispatch_type = detect_dispatch_type(scan, independent_register_count)
    description = auto_describe(dispatch_type, scan, independent_register_count)
    active_count = sum(1 for k in keys if k['active'])
    tk_is_extracted = sum(
        1 for k in keys if k['source'] in ('TILING_KEY_IS', 'TILING_KEY_IS_standalone'))

    return {
        'file': relpath(filepath, op_kernel_dir),
        'priority': priority,
        'kind': kind,
        'dispatch_type': dispatch_type,
        'entry_mechanism': description,
        'entry_functions': scan['global_functions'],
        'keys': keys,
        'active_key_count': active_count,
        'total_key_count': len(keys),
        'tk_is_extracted_count': tk_is_extracted,
        'tiling_registration_keys': scan['tiling_registration_keys'],
        'has_register_tiling_default': scan['has_register_tiling_default'],
        'total_tk_is_grep_count': scan['total_tk_is_grep_count'],
        'source_cpp': source_cpp,
        'include_level': (include_info or {}).get('include_level'),
        'included_by': (include_info or {}).get('included_by'),
        'include_chain': (include_info or {}).get('include_chain'),
        'matched_templates': matched_templates or [],
    }


def write_json(output_dir, scan_result, op_name, npu_arch, soc_version):
    entries = scan_result['entries']
    excluded = scan_result['excluded']
    total_count = scan_result['total_count']
    valid_count = scan_result['valid_count']
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'S2P0_scout_k.json')

    total_extracted = sum(e['tk_is_extracted_count'] for e in entries)
    total_grep = sum(e['total_tk_is_grep_count'] for e in entries)
    match = (total_extracted == total_grep) if total_grep > 0 else True

    data = {
        'operator': op_name,
        'platform': {
            'npu_arch': npu_arch,
            'soc_version': soc_version or '',
        },
        'scan_baseline': {
            'total_files': total_count,
            'valid_files': valid_count,
            'excluded_files': len(excluded),
        },
        'entries': [
            {
                'file': e['file'],
                'priority': e.get('priority', 'P0'),
                'kind': e.get('kind', 'kernel_entry'),
                'dispatch_type': e['dispatch_type'],
                'entry_mechanism': e['entry_mechanism'],
                'entry_functions': e['entry_functions'],
                'keys': e['keys'],
                'active_key_count': e['active_key_count'],
                'total_key_count': e['total_key_count'],
                'source_cpp': e.get('source_cpp'),
                'include_level': e.get('include_level'),
                'included_by': e.get('included_by'),
                'include_chain': e.get('include_chain'),
                'matched_templates': e.get('matched_templates', []),
            }
            for e in entries
        ],
        'excluded': [
            {
                'file': e['file'], 'reason': e['reason'], 'key_count': e['key_count'],
                'priority': e.get('priority', 'P2'),
                'include_level': e.get('include_level'),
                'included_by': e.get('included_by'),
                'include_chain': e.get('include_chain'),
            }
            for e in excluded
        ],
        'reachable_kernel_templates': scan_result.get('reachable_kernel_templates', []),
        'unresolved_init_calls': scan_result.get('unresolved_init_calls', []),
        'include_edges': scan_result.get('include_edges', []),
        'cross_validation': {
            'extracted_count': total_extracted,
            'grep_count': total_grep,
            'match': match,
        },
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    _logger.info("Written: %s", json_path)
    return json_path


def _append_entry_detail(lines, entry):
    lines.append(f'  优先级: {entry.get("priority", "P0")} ({entry.get("kind", "kernel_entry")})')
    if entry.get('matched_templates'):
        lines.append(f'  命中模板: {", ".join(entry["matched_templates"])}')
    if entry.get('included_by'):
        lines.append(f'  include 来源: {entry["included_by"]}')
    for gf in entry['entry_functions']:
        tmpl = '（模板）' if gf.get('is_template') else ''
        lines.append(f'  入口函数: {gf["name"]}@行{gf["line"]}{tmpl}')
    lines.append(f'  入口机制: {entry["entry_mechanism"]}')
    lines.append(f'  dispatch 类型: {entry["dispatch_type"]}')

    if entry['keys']:
        active = entry['active_key_count']
        total = entry['total_key_count']
        lines.append(f'  key 列表（{total} 个, {active} 个活跃）:')
        for k in entry['keys']:
            status = '活跃' if k['active'] else 'arch_guarded_inactive'
            guard_str = f' [{k["arch_guard"]}]' if k.get('arch_guard') else ''
            lines.append(f'    - {k["value"]} (行{k["line"]}) [{status}]{guard_str}')
            if k.get('tiling_key_var'):
                tc = f' ({k["tiling_class"]})' if k.get('tiling_class') else ''
                lines.append(f'      tiling_key_var: {k["tiling_key_var"]}{tc}')
    elif entry['tiling_registration_keys']:
        lines.append(f'  tiling 注册 key（{len(entry["tiling_registration_keys"])} 个）:')
        for k in entry['tiling_registration_keys']:
            lines.append(f'    - {k["value"]} (行{k["line"]})')

    if entry.get('source_cpp'):
        lines.append(f'  来源: 被 {entry["source_cpp"]} #include')
    lines.append('')


def _append_excluded_section(lines, excluded):
    lines.append('## 排除文件（目标平台不可达）')
    lines.append('')
    if not excluded:
        lines.append('  （无）')
        lines.append('')
        return
    for ex in excluded:
        lines.append(f'  - {ex["file"]}')
        lines.append(f'    优先级: {ex.get("priority", "P2")}')
        lines.append(f'    排除原因: {ex["reason"]}')
        lines.append(f'    key 数量: {ex["key_count"]}')
        if ex.get('included_by'):
            lines.append(f'    include 来源: {ex["included_by"]}')
        lines.append('')


def _append_reachable_templates(lines, templates, unresolved):
    lines.append('## op.Init 可达 kernel 模板')
    lines.append('')
    if templates:
        for item in templates:
            cond = f' 条件: {item["condition"]}' if item.get('condition') else ''
            lines.append(
                f'  - {item["template"]} @ {item["source_file"]}:{item["init_line"]}'
                f' (var={item["op_var"]}){cond}'
            )
    else:
        lines.append('  （未识别，保留 P0 扫描结果）')
    if unresolved:
        lines.append('')
        lines.append('  未解析 Init 调用:')
        for item in unresolved:
            lines.append(f'    - {item["source_file"]}:{item["init_line"]} var={item["op_var"]}')
    lines.append('')


def write_report(output_dir, scan_result, npu_arch, platform_info):
    entries = scan_result['entries']
    excluded = scan_result['excluded']
    total_count = scan_result['total_count']
    valid_count = scan_result['valid_count']
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'S2P0_scout_k.md')

    lines = ['=== KERNEL SCOUT REPORT ===', '']
    lines.append(f'目标平台: {npu_arch} (__NPU_ARCH__={platform_info["arch_num"]})')
    if platform_info.get('soc_version'):
        lines[-1] += f' ({platform_info["soc_version"]})'
    lines.append('')
    lines.append('## 扫描基准线')
    lines.append('')
    lines.append(f'  全量总计: {total_count} 个文件')
    lines.append(f'  有效（目标平台可达）: {valid_count} 个文件')
    lines.append(f'  排除（目标平台不可达）: {len(excluded)} 个文件')
    lines.append('')
    _append_reachable_templates(
        lines,
        scan_result.get('reachable_kernel_templates', []),
        scan_result.get('unresolved_init_calls', []),
    )
    lines.append('## Kernel 入口信息')
    lines.append('')

    for entry in entries:
        lines.append(f'  [{entry["file"]}]')
        _append_entry_detail(lines, entry)

    _append_excluded_section(lines, excluded)

    total_extracted = sum(e['tk_is_extracted_count'] for e in entries)
    total_grep = sum(e['total_tk_is_grep_count'] for e in entries)
    match = (total_extracted == total_grep) if total_grep > 0 else True

    lines.append('## 交叉验证')
    lines.append('')
    lines.append(f'  提取 key 条目数: {total_extracted}')
    lines.append(f'  Grep 计数: {total_grep}')
    lines.append(f'  结果: {"一致 ✓" if match else "不一致 ✗"}')
    lines.append('')

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    _logger.info("Written: %s", report_path)
    return report_path


def _classify_files(all_cpp_files, op_kernel_dir, npu_arch):
    valid_files = []
    excluded_files = []
    for fpath in all_cpp_files:
        status, reason = classify_platform(fpath, op_kernel_dir, npu_arch)
        if status == 'valid':
            valid_files.append(fpath)
        else:
            excluded_files.append({
                'file': relpath(fpath, op_kernel_dir),
                'filepath': fpath,
                'reason': reason,
                'key_count': count_keys_in_file(fpath),
            })
    return valid_files, excluded_files


def _discover_reachable_kernel_files(valid_files, op_path, op_kernel_dir, npu_arch, arch_num):
    entries = []
    p1_files = set()
    p2_files = []
    reachable_templates = []
    unresolved_init_calls = []

    for fpath in valid_files:
        entries.append(build_entry(fpath, op_kernel_dir, arch_num, priority='P0', kind='kernel_entry'))
        templates, unresolved = extract_reachable_kernel_templates(fpath, op_kernel_dir)
        reachable_templates.extend(templates)
        unresolved_init_calls.extend(unresolved)

    template_names = sorted({item['template'] for item in reachable_templates if item.get('template')})
    include_files, include_edges = collect_include_closure(valid_files, op_kernel_dir, op_path)

    for inc_path, inc_info in sorted(include_files.items(), key=lambda item: item[1]['file']):
        h_status, reason = classify_platform(inc_path, op_kernel_dir, npu_arch)
        key_count = count_keys_in_file(inc_path)
        if h_status != 'valid':
            p2_files.append({
                'file': inc_info['file'],
                'filepath': inc_path,
                'reason': reason or 'included but target platform excluded',
                'key_count': key_count,
                'priority': 'P2',
                'include_level': inc_info.get('include_level'),
                'included_by': inc_info.get('included_by'),
                'include_chain': inc_info.get('include_chain'),
            })
            continue

        matched = _template_matches_file(inc_path, template_names) if template_names else []
        if matched:
            p1_files.add(inc_path)
            entries.append(build_entry(
                inc_path,
                op_kernel_dir,
                arch_num,
                source_cpp=inc_info.get('included_by'),
                priority='P1',
                kind='reachable_template_file',
                include_info=inc_info,
                matched_templates=matched,
            ))
        else:
            p2_files.append({
                'file': inc_info['file'],
                'filepath': inc_path,
                'reason': 'included but no reachable kernel template match',
                'key_count': key_count,
                'priority': 'P2',
                'include_level': inc_info.get('include_level'),
                'included_by': inc_info.get('included_by'),
                'include_chain': inc_info.get('include_chain'),
            })

    return entries, p1_files, p2_files, reachable_templates, unresolved_init_calls, include_edges


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    args = parse_args()
    op_path = os.path.abspath(args.op_path)
    op_kernel_dir = os.path.join(op_path, 'op_kernel')
    npu_arch = args.npu_arch
    platform = PLATFORM_MAP[npu_arch]
    output_dir = args.output_dir or os.path.join(op_path, 'tests', 'whitebox')

    if not os.path.isdir(op_kernel_dir):
        _logger.error("op_kernel/ directory not found: %s", op_kernel_dir)
        sys.exit(1)

    all_cpp_files = discover_kernel_files(op_kernel_dir)
    if not all_cpp_files:
        _logger.error("No .cpp files found in %s", op_kernel_dir)
        sys.exit(1)

    _logger.info("Discovered %d .cpp files in op_kernel/", len(all_cpp_files))

    valid_files, excluded_files = _classify_files(all_cpp_files, op_kernel_dir, npu_arch)
    _logger.info("Valid: %d, Excluded: %d", len(valid_files), len(excluded_files))

    entries, discovered_h, p2_files, reachable_templates, unresolved_init_calls, include_edges = \
        _discover_reachable_kernel_files(
            valid_files, op_path, op_kernel_dir, npu_arch, platform['arch_num'])
    excluded_files.extend(p2_files)

    total_count = len(all_cpp_files) + len(discovered_h) + len(p2_files)
    valid_count = len(valid_files) + len(discovered_h)
    _logger.info("Entries: %d (incl. %d discovered .h files)",
                 len(entries), len(discovered_h))

    scan_result = {
        'entries': entries, 'excluded': excluded_files,
        'total_count': total_count, 'valid_count': valid_count,
        'reachable_kernel_templates': reachable_templates,
        'unresolved_init_calls': unresolved_init_calls,
        'include_edges': include_edges,
    }

    write_json(output_dir, scan_result, args.op_name, npu_arch, args.soc_version)

    platform_info = {
        'arch_num': platform['arch_num'],
        'soc_version': args.soc_version,
        'op_path': op_path,
    }
    write_report(output_dir, scan_result, npu_arch, platform_info)
    _logger.info("Done.")


if __name__ == '__main__':
    main()
