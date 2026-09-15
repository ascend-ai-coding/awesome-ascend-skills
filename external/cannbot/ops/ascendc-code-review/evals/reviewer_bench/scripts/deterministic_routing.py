#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""确定性条例路由 — reviewer_bench 适配层

从 manifest 或 review-config 读取 PR 信息，提取变更代码文件，
扫描 references/*.md 的 <适用> 头做声明式匹配，输出 YAML 路由计划。

可两种模式：
  1. 单 PR 路由：--code-files + --diff
  2. 批量路由：--manifest manifest_redline.json → 输出每个 PR 的路由计划 JSON

用法:
    python scripts/deterministic_routing.py --code-files src/foo.cpp --diff changes.diff
    python scripts/deterministic_routing.py --manifest manifest_redline.json --output routing_plans.json
    python scripts/deterministic_routing.py --manifest manifest_redline.json --inject-config review-config-e2e.json
"""
import logging
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

CODE_EXTS = {'.cpp', '.h', '.hpp', '.cc', '.c', '.cu', '.cuh', '.py'}

APPLICABILITY_RE = re.compile(r'<适用>\s*\n(.*?)\n</适用>', re.DOTALL)
ANY_TAG_RE = re.compile(r'<(\S+)>\s*\n(.*?)\n</\1>', re.DOTALL)
FIELD_RE = re.compile(r'^(\S+):\s*(.*)$', re.MULTILINE)
FIELD_VALUE_RE = re.compile(r'^[^:]+:\s*(.+)$', re.MULTILINE)
def _fix_mojibake(text: str) -> str:
    """修复 UTF-8 被误解为 GBK 产生的 mojibake"""
    if '红线' in text or '数值安全' in text:
        return text
    mojibake_markers = ['绾', '鏈', '璇', '閫', '棰', '榛', '浠', '渚', '鍐', '闄']
    if sum(1 for m in mojibake_markers if m in text) >= 3:
        try:
            fixed = text.encode('gbk', errors='replace').decode('utf-8', errors='replace')
            if '红线' in fixed or '安全' in fixed:
                return fixed
        except Exception:
            pass
    return text


def _read_text(path: Path) -> str:
    """自动检测编码读取文件（支持 UTF-8/UTF-16/GBK + mojibake 修复）"""
    raw = path.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        text = raw.decode('utf-16', errors='replace')
    elif raw[:3] == b'\xef\xbb\xbf':
        text = raw[3:].decode('utf-8', errors='replace')
    else:
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            text = raw.decode('gbk', errors='replace')
    return _fix_mojibake(text)


INDEX_TABLE_RE_4COL = re.compile(
    r'\|\s*([\d.]+|RB-[\d.]+|SEC-[\d.]+|API-[\d.]+|PERF-[\d.]+|TOPK-[\d.]+'
    r'|SIMT-[\d.]+|MC2-[\d.]+|GEN-[\d.]+|CMP-[\d.]+|PY-[\d.]+|TIL-[\d.]+|PREC-[\d.]+)'
    r'\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|'
)
INDEX_TABLE_RE_3COL = re.compile(
    r'\|\s*([\d.]+|RB-[\d.]+|SEC-[\d.]+|API-[\d.]+|PERF-[\d.]+|TOPK-[\d.]+'
    r'|SIMT-[\d.]+|MC2-[\d.]+|GEN-[\d.]+|CMP-[\d.]+|PY-[\d.]+|TIL-[\d.]+|PREC-[\d.]+)'
    r'\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|'
)
SIDE_TAG_RE = re.compile(r'\[适用:\s*(All|Kernel|Tiling|Host)\]')

KERNEL_PATTERNS = ['__aicore__', 'AscendC::', 'pipe.InitBuffer', 'pipe.', '__global__', '__simt']
TILING_PATTERNS = ['gert::TilingContext', 'TilingData', 'OP_CHECK_IF', 'GetWorkspaceSize', 'SetBlockDim']

CATEGORY_PRIORITY = {
    '数值安全': 1, '内存安全': 1, '输入验证': 1, '总体原则': 1,
    'API使用': 2, '数据搬运': 2, '类型安全': 2, '资源管理': 2,
    '类与对象': 2, '标准库': 2, 'LOG API 安全': 2,
    '并发安全': 3, '通信同步': 3, '架构决策': 3,
    '性能优化': 4, '精度保护': 4, 'Tiling设计': 4,
    '编译配置': 5, '边界安全': 2, '实现合规': 3, '可追溯性': 3,
    '代码设计': 5, '头文件': 5, '数据类型': 5, '常量': 5,
    '变量': 5, '编码规范': 5, '代码风格': 6, 'LOG API 规范': 6,
    '调试与隐私': 3, '文件操作': 3, '序列化': 3, '异常处理': 3,
    '命名': 6, '格式': 6, '注释': 6, 'API合规': 2,
}

# === 误报排除规则（数据来源：docs/reports/false_positive_analysis.md + docs/custom.md） ===

# 规则 R1：Kernel 侧排除"外部输入校验"类条例
# 依据：__aicore__ 函数不做输入校验，校验职责在 Tiling 层（误报占比 27%）
KERNEL_EXCLUDE_CATEGORIES = {'输入验证'}

# 规则 R2：常量白名单模式（误报占比 9.5%）
# 匹配这些前缀的 constexpr/const/#define 常量，标注"无需校验零值"
CONST_PATTERN_RE = re.compile(
    r'(?:constexpr|const\s+static|static\s+const|#define)\s+(\w+)'
)
CONST_SAFE_PREFIXES = (
    'FP32_', 'FP16_', 'BF16_', 'UB_', 'BLOCK_', 'MAX_', 'MIN_',
    'BYTE_', 'REPEAT_', 'BUFFER_', 'ALIGN_', 'TILE_', 'CUBE_', 'VEC_',
)

# 规则 R3：安全 API/宏白名单（误报占比 7.9%）
# 出现这些宏的位置已有安全保护，标注"无需重复校验"
SAFE_MACRO_RE = re.compile(
    r'OP_CHECK_NULL_WITH_CONTEXT|OP_CHECK_PARAM_VALUE|OP_CHECK_IF|ASCEND_CHECK'
)

# 规则 R4：控制流保护检测（误报占比 19%）
# 检测 if-guard 包裹的除法/数组访问，标注"分支已保护"
IF_GUARD_DIVISION_RE = re.compile(
    r'if\s*\(\s*\w+\s*[!=]=\s*0\s*\).*?\{[^}]*/[^}]*\}',
    re.DOTALL
)


def parse_applicability(text: str) -> dict:
    m = APPLICABILITY_RE.search(text)
    if m:
        block = m.group(1)
        fields = {}
        for fm in FIELD_RE.finditer(block):
            key = fm.group(1).strip()
            val = fm.group(2).strip()
            fields[key] = val
        return fields

    for tm in ANY_TAG_RE.finditer(text):
        block = tm.group(2)
        if not re.search(r':\s*(C\+\+|Python|Build)', block, re.MULTILINE):
            continue
        if not re.search(r':\s*(All|Kernel|Tiling|Host)', block, re.MULTILINE):
            continue
        fields = {}
        for fm in FIELD_RE.finditer(block):
            key = fm.group(1).strip()
            val = fm.group(2).strip()
            fields[key] = val
        value_lines = [fm.group(1).strip() for fm in FIELD_VALUE_RE.finditer(block)]
        if '语言' not in fields and any(v in ('C++', 'Python', 'Build') for v in value_lines):
            for v in value_lines:
                if v in ('C++', 'Python', 'Build', '不限'):
                    fields['语言'] = v
                    break
        if '侧别' not in fields:
            for v in value_lines:
                if any(s in v for s in ('All', 'Kernel', 'Tiling', 'Host')):
                    fields['侧别'] = v
                    break
        if '领域' not in fields:
            bool_vals = [v for v in value_lines if v in ('true', 'false')]
            if bool_vals:
                fields['领域'] = bool_vals[0]
        if '默认启用' not in fields:
            bool_vals = [v for v in value_lines if v in ('true', 'false')]
            if len(bool_vals) >= 2:
                fields['默认启用'] = bool_vals[1]
        if '语言' in fields:
            return fields
    return {}


def parse_index_table(text: str) -> List[dict]:
    entries = []
    current_side = 'All'
    for line in text.splitlines():
        header_match = re.search(r'\[适用:\s*(All|Kernel|Tiling|Host)\]', line)
        if header_match and line.strip().startswith('#'):
            current_side = header_match.group(1).strip()
            continue
        m4 = INDEX_TABLE_RE_4COL.match(line)
        if m4:
            clause_id = m4.group(1).strip()
            title = m4.group(2).strip()
            category = m4.group(3).strip()
            severity = m4.group(4).strip()
            side_match = SIDE_TAG_RE.search(line)
            side_tag = side_match.group(1).strip() if side_match else current_side
            entries.append({
                'clause_id': clause_id,
                'title': title,
                'category': category,
                'severity': severity,
                'side': side_tag,
            })
            continue
        m3 = INDEX_TABLE_RE_3COL.match(line)
        if m3:
            clause_id = m3.group(1).strip()
            title = m3.group(2).strip()
            col3 = m3.group(3).strip()
            if col3 in ('高', '中', '低', '要求', '建议', 'critical', 'high', 'medium', 'low'):
                category = '未分类'
                severity = col3
            else:
                category = col3
                severity = '中'
            side_match = SIDE_TAG_RE.search(line)
            side_tag = side_match.group(1).strip() if side_match else current_side
            entries.append({
                'clause_id': clause_id,
                'title': title,
                'category': category,
                'severity': severity,
                'side': side_tag,
            })
    return entries


def detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in {'.cpp', '.h', '.hpp', '.cc', '.c'}:
        return 'C++'
    if ext in {'.py'}:
        return 'Python'
    if ext in {'.cmake'} or Path(file_path).name in {'CMakeLists.txt', 'Makefile'}:
        return 'Build'
    return 'Unknown'


def detect_side(file_path: str, code_content: str = '') -> str:
    if 'op_kernel/' in file_path or 'op_kernel\\' in file_path:
        return 'Kernel'
    if 'op_host/' in file_path or 'op_host\\' in file_path:
        return 'Tiling'
    if code_content:
        has_kernel = any(p in code_content for p in KERNEL_PATTERNS)
        has_tiling = any(p in code_content for p in TILING_PATTERNS)
        if has_kernel and has_tiling:
            return 'Mixed'
        if has_kernel:
            return 'Kernel'
        if has_tiling:
            return 'Tiling'
    return 'Unknown'


def load_references(ref_dir: Path) -> List[dict]:
    refs = []
    for md in sorted(ref_dir.glob('*.md')):
        if md.name.startswith('_'):
            continue
        text = _read_text(md)
        applicability = parse_applicability(text)
        if not applicability:
            continue
        index_entries = parse_index_table(text)
        triggers = []
        for field_name in ['触发', '触发(直接)', '触发(架构)', '触发(间接)']:
            trigger_field = applicability.get(field_name, '')
            if trigger_field:
                triggers.extend(t.strip() for t in trigger_field.split(',') if t.strip())
        refs.append({
            'file': md.name,
            'path': str(md),
            'applicability': applicability,
            'language': applicability.get('语言', ''),
            'side': applicability.get('侧别', ''),
            'domain': applicability.get('领域', 'false').lower() == 'true',
            'default_enabled': applicability.get('默认启用', 'true').lower() == 'true',
            'triggers': triggers,
            'clauses': index_entries,
        })
    return refs


def match_references(refs: List[dict], language: str, side: str,
                     keywords_hit: Set[str]) -> List[dict]:
    matched = []
    for ref in refs:
        if not ref['default_enabled']:
            continue
        ref_lang = ref['language']
        if ref_lang and ref_lang != '不限' and ref_lang != language:
            continue
        ref_sides = [s.strip() for s in ref['side'].split(',')]
        if side not in ('Mixed', 'Unknown'):
            if 'All' not in ref_sides and side not in ref_sides:
                continue
        elif side == 'Mixed':
            if 'All' not in ref_sides and not any(s in ref_sides for s in ['Kernel', 'Tiling', 'Host']):
                continue
        if ref['domain']:
            if ref['triggers']:
                hit = any(t in keywords_hit for t in ref['triggers'])
                if not hit:
                    continue
        matched.append(ref)
    return matched


def filter_clauses_by_side(ref: dict, side: str) -> List[dict]:
    clauses = ref['clauses']
    if not clauses:
        return []
    if side == 'Mixed':
        return clauses
    filtered = []
    for c in clauses:
        c_side = c.get('side', 'All')
        if c_side == 'All' or c_side == side:
            # 规则 R1：Kernel 侧排除"外部输入校验"类条例
            # 依据：__aicore__ 函数不做输入校验，校验职责在 Tiling 层
            if side == 'Kernel' and c.get('category', '') in KERNEL_EXCLUDE_CATEGORIES:
                continue
            filtered.append(c)
        elif side == 'Tiling' and c_side == 'Host':
            filtered.append(c)
    return filtered


def group_clauses(matched_refs: List[dict], side: str) -> List[dict]:
    by_category: Dict[str, list] = {}
    for ref in matched_refs:
        clauses = filter_clauses_by_side(ref, side)
        for c in clauses:
            cat = c['category']
            if cat not in by_category:
                by_category[cat] = []
            entry = dict(c)
            entry['source_file'] = ref['file']
            by_category[cat].append(entry)

    groups = []
    for cat, entries in sorted(by_category.items(),
                                key=lambda x: CATEGORY_PRIORITY.get(x[0], 5)):
        side_tags = set(e.get('side', 'All') for e in entries)
        if side == 'Mixed':
            tag = '全部'
        elif side_tags == {'Kernel'} or side_tags <= {'Kernel'}:
            tag = '仅Kernel'
        elif side_tags <= {'Tiling', 'Host'}:
            tag = '仅Tiling'
        else:
            tag = '全部'
        source_files = sorted(set(e['source_file'] for e in entries))
        groups.append({
            'category': cat,
            'side_tag': tag,
            'source_files': source_files,
            'clauses': [{'id': e['clause_id'], 'title': e['title']} for e in entries],
        })
    return groups


def extract_code_files_from_diff(diff_path: Path) -> List[str]:
    if not diff_path.exists():
        return []
    text = diff_path.read_text(encoding='utf-8', errors='replace')
    files = []
    for m in re.finditer(r'^diff --git a/(.+?) b/(.+?)$', text, re.MULTILINE):
        f = m.group(2)
        if Path(f).suffix.lower() in CODE_EXTS:
            files.append(f)
    return list(dict.fromkeys(files))


def read_code_files(repo_dir: Path, file_paths: List[str]) -> Dict[str, str]:
    contents = {}
    for fp in file_paths:
        full = repo_dir / fp
        if full.exists() and full.suffix.lower() in CODE_EXTS:
            try:
                contents[fp] = _read_text(full)
            except Exception:
                contents[fp] = ''
    return contents


def grep_keywords(code_contents: Dict[str, str], keywords: List[str]) -> Set[str]:
    hit = set()
    for content in code_contents.values():
        for kw in keywords:
            if kw in content:
                hit.add(kw)
    return hit


def analyze_code_context(code_contents: Dict[str, str], side: str) -> dict:
    """扫描代码特征，生成误报排除上下文（对齐 false_positive_analysis.md 的 R2-R4 规则）。
    
    返回 dict，供 build_routing_prompt_snippet 追加到 prompt 中。
    """
    safe_consts = set()
    safe_macros = set()
    if_guard_divs = []

    for file_path, content in code_contents.items():
        if not content:
            continue

        for m in CONST_PATTERN_RE.finditer(content):
            name = m.group(1)
            if name.startswith(CONST_SAFE_PREFIXES):
                safe_consts.add(name)

        for m in SAFE_MACRO_RE.finditer(content):
            safe_macros.add(m.group(0))

        for m in IF_GUARD_DIVISION_RE.finditer(content):
            snippet = m.group(0).strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + '...'
            if_guard_divs.append(snippet)

    excluded_clauses = []
    if side == 'Kernel' and KERNEL_EXCLUDE_CATEGORIES:
        excluded_clauses.append(
            '输入验证类（4.1 外部输入合法性校验、4.2 内存操作长度校验）— '
            'Kernel 侧不做输入校验，职责在 Tiling 层'
        )

    return {
        'safe_consts': sorted(safe_consts),
        'safe_macros': sorted(safe_macros),
        'if_guard_divs': if_guard_divs[:5],
        'excluded_clauses': excluded_clauses,
    }


def route_single_pr(code_files: List[str], code_contents: Dict[str, str],
                    refs: List[dict]) -> dict:
    all_keywords = set()
    for ref in refs:
        all_keywords.update(ref['triggers'])
    keywords_hit = grep_keywords(code_contents, list(all_keywords))

    languages = set(detect_language(fp) for fp in code_files)
    language = list(languages)[0] if len(languages) == 1 else 'C++'

    sides = set()
    for fp in code_files:
        content = code_contents.get(fp, '')
        sides.add(detect_side(fp, content))
    if len(sides) == 1:
        side = list(sides)[0]
    elif 'Kernel' in sides and 'Tiling' in sides:
        side = 'Mixed'
    else:
        side = 'Unknown'

    matched = match_references(refs, language, side, keywords_hit)
    groups = group_clauses(matched, side)
    total_clauses = sum(len(g['clauses']) for g in groups)

    code_context = analyze_code_context(code_contents, side)

    WAVE_SIZE = 3
    waves = []
    for i, g in enumerate(groups):
        wave_num = i // WAVE_SIZE + 1
        waves.append({
            'wave': wave_num,
            'group': i + 1,
            'category': g['category'],
            'side_tag': g['side_tag'],
            'source_files': g['source_files'],
            'code_range': g['side_tag'],
            'clauses': [{'id': c['id'], 'title': c['title']} for c in g['clauses']],
        })

    return {
        'language': language,
        'side': side,
        'keywords_hit': sorted(keywords_hit),
        'total_clauses': total_clauses,
        'groups': len(groups),
        'waves_count': (len(groups) + WAVE_SIZE - 1) // WAVE_SIZE,
        'code_context': code_context,
        'waves': waves,
    }


def format_yaml(plan: dict) -> str:
    lines = [
        f'language: "{plan["language"]}"',
        f'side: "{plan["side"]}"',
        f'keywords_hit: {plan["keywords_hit"] if plan["keywords_hit"] else "[]"}',
        f'total_clauses: {plan["total_clauses"]}',
        f'groups: {plan["groups"]}',
        f'waves_count: {plan.get("waves_count", 1)}',
        'waves:',
    ]
    for w in plan['waves']:
        lines.append(f'  - wave: {w["wave"]}')
        lines.append(f'    group: {w["group"]}')
        lines.append(f'    category: "{w["category"]}"')
        lines.append(f'    side_tag: "{w["side_tag"]}"')
        lines.append(f'    source_files: {w["source_files"]}')
        lines.append(f'    code_range: "{w["code_range"]}"')
        lines.append(f'    clauses:')
        for c in w['clauses']:
            lines.append(f'      - id: "{c["id"]}"')
            lines.append(f'        title: "{c["title"]}"')
    return '\n'.join(lines)


def build_routing_prompt_snippet(plan: dict) -> str:
    if plan['total_clauses'] == 0:
        return ""
    snippet = f"\n\n【确定性路由结果（已由脚本预计算，跳过 clause-routing 步骤）】"
    snippet += f"\n代码语言: {plan['language']} | 侧别: {plan['side']}"
    snippet += f"\n命中关键词: {', '.join(plan['keywords_hit']) if plan['keywords_hit'] else '无'}"
    snippet += f"\n总条例数: {plan['total_clauses']} | 分组数: {plan['groups']} | 波次数: {plan.get('waves_count', 1)}"

    ctx = plan.get('code_context')
    exclusion_summary = ""
    if ctx and (ctx['safe_consts'] or ctx['safe_macros'] or ctx['excluded_clauses'] or ctx['if_guard_divs']):
        exclusion_summary = "\n    【本波次排除上下文（派发子 Agent 时必须传递）】"
        if ctx['excluded_clauses']:
            exclusion_summary += "\n    已排除条例:"
            for exc in ctx['excluded_clauses']:
                exclusion_summary += f"\n      - {exc}"
        if ctx['safe_consts']:
            exclusion_summary += "\n    常量白名单（不报零值风险）:"
            exclusion_summary += f"\n      {', '.join(ctx['safe_consts'])}"
        if ctx['safe_macros']:
            exclusion_summary += "\n    安全宏（不报空指针风险）:"
            exclusion_summary += f"\n      {', '.join(ctx['safe_macros'])}"
        if ctx['if_guard_divs']:
            exclusion_summary += "\n    分支保护除法（不报除零风险）:"
            for div_snippet in ctx['if_guard_divs']:
                exclusion_summary += f"\n      - {div_snippet}"

    snippet += "\n\n分组规划（按波次排列，每波 ≤3 组，必须逐波完成）:"

    current_wave = 0
    for w in plan['waves']:
        if w['wave'] != current_wave:
            current_wave = w['wave']
            snippet += f"\n  ── 波次 {current_wave} ──"
        clause_strs = [f"{c['id']} {c['title']}" for c in w['clauses']]
        snippet += f"\n  组{w['group']} [{w['side_tag']}] {w['category']}: {'、'.join(clause_strs)}"
    if exclusion_summary:
        snippet += exclusion_summary

    snippet += f"\n\n请直接按以上分组规划执行阶段1逐条检视，跳过 clause-routing 步骤。"
    if exclusion_summary:
        snippet += "\n派发每个检视子 Agent 时，必须将【本波次排除上下文】复制到子 Agent 的 prompt 中。"
        snippet += "\n对白名单常量不报零值风险，对安全宏保护位置不报空指针风险，对分支保护除法不报除零风险。"
    waves_count = plan.get('waves_count', 1)
    if waves_count > 1:
        snippet += f"\n共 {waves_count} 个波次，必须逐波完成后再进入下一波，不要跳过。"
    return snippet


def run_manifest(manifest_path: Path, ref_dir: Path, base_dir: Path) -> List[dict]:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    refs = load_references(ref_dir)
    results = []
    for entry in manifest.get('entries', []):
        repo_dir = base_dir / entry['repo_dir']
        diff_path = base_dir / entry['diff_file']
        code_files = entry.get('code_files', [])
        if not code_files:
            code_files = extract_code_files_from_diff(diff_path)
        code_contents = read_code_files(repo_dir, code_files)
        plan = route_single_pr(code_files, code_contents, refs)
        plan['target_name'] = f"{entry['repo']}-pr-{entry['pr_number']}"
        plan['code_files'] = code_files
        results.append(plan)
        logging.info(f"[ROUTE] {plan['target_name']}: {plan['language']} / {plan['side']} / "
              f"{plan['total_clauses']} clauses / {plan['groups']} groups / "
              f"{plan.get('waves_count', 1)} waves / "
              f"{len(plan['keywords_hit'])} keywords")
    return results


def inject_into_config(config_path: Path, plans: List[dict], output_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding='utf-8'))
    plan_map = {p['target_name']: p for p in plans}
    for target in config.get('targets', []):
        plan = plan_map.get(target['name'])
        if not plan:
            continue
        snippet = build_routing_prompt_snippet(plan)
        if snippet:
            target['prompt'] = target['prompt'] + snippet
    output_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    logging.info(f"[INJECT] 已生成增强配置: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='确定性条例路由 — reviewer_bench 适配层')
    parser.add_argument('--code-files', nargs='+', default=None, help='代码文件路径')
    parser.add_argument('--diff', default=None, help='diff 文件路径')
    parser.add_argument('--repo-dir', default=None, help='完整源码目录（用于读取文件内容）')
    parser.add_argument('--ref-dir', default=None, help='references 目录路径')
    parser.add_argument('--manifest', default=None, help='manifest JSON 文件路径')
    parser.add_argument('--inject-config', default=None, help='注入路由结果到 review-config JSON')
    parser.add_argument('--output', default=None, help='输出文件路径（JSON）')
    parser.add_argument('--format', choices=['yaml', 'json'], default='yaml', help='输出格式')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    default_ref = Path(os.environ.get(
        'ASCENDC_REVIEW_REF_DIR',
        str(Path.home() / '.opencode' / 'skills' / 'ascendc-code-review' / 'references')
    ))
    ref_dir = Path(args.ref_dir) if args.ref_dir else default_ref

    if not ref_dir.exists():
        logging.error(f'错误: references 目录不存在: {ref_dir}')
        logging.info(f'请通过 --ref-dir 指定，或设置 ASCENDC_REVIEW_REF_DIR 环境变量')
        sys.exit(1)

    if args.manifest:
        plans = run_manifest(Path(args.manifest), ref_dir, base_dir)
        if args.inject_config:
            inject_into_config(
                Path(args.inject_config), plans,
                Path(args.output) if args.output else base_dir / 'review-config-routed.json'
            )
        else:
            out_path = Path(args.output) if args.output else base_dir / 'routing_plans.json'
            out_path.write_text(
                json.dumps(plans, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            logging.info(f'\n路由计划已写入: {out_path}')
            logging.info(f'共 {len(plans)} 个 PR')
            for p in plans:
                logging.info(f'  {p["target_name"]}: {p["total_clauses"]} clauses, '
                      f'{p["groups"]} groups, side={p["side"]}')
        return

    if not args.code_files:
        parser.error('--code-files 或 --manifest 必填')

    refs = load_references(ref_dir)
    repo_dir = Path(args.repo_dir) if args.repo_dir else base_dir

    code_files = args.code_files
    if args.diff and not code_files:
        code_files = extract_code_files_from_diff(Path(args.diff))

    code_contents = {}
    for fp in code_files:
        p = Path(fp)
        if not p.is_absolute():
            p = repo_dir / fp
        if p.exists() and p.suffix.lower() in CODE_EXTS:
            try:
                code_contents[fp] = p.read_text(encoding='utf-8', errors='replace')
            except Exception:
                code_contents[fp] = ''

    plan = route_single_pr(code_files, code_contents, refs)

    if args.format == 'yaml':
        print(format_yaml(plan))  # noqa
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))  # noqa

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        logging.info(f'\n路由计划已写入: {out_path}')


if __name__ == '__main__':
    main()
