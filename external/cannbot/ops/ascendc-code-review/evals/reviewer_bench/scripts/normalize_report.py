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
"""报告归一化模块 — 对齐 OpenCodeReview 的确定性工程层。

三阶段处理：
  1. 结构化提取：从报告提取 findings（PR#822 yaml 目录 > Markdown 内嵌 YAML > Markdown 正则），对齐 LlmComment
  2. 定位校正（re-tracking）：解析 diff hunks，标注行号位置状态（in_diff/retracked/line_drift），不覆盖原始行号
  3. 反思过滤（reflection）：保守过滤 + 跨文件去重

输出归一化 JSON，供 run_eval.py 消费。
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CONFIDENCE_THRESHOLD = 30
SEVERITY_MAP = {
    "critical": "critical", "error": "high", "warning": "medium", "info": "low",
    "high": "high", "medium": "medium", "low": "low",
}
CONFIDENCE_TEXT_MAP = {"HIGH": 90, "MED": 60, "MEDIUM": 60, "LOW": 30}

YAML_FINDINGS_RE = re.compile(
    r'##\s*(?:[一二三四五六七八九十\d]+[、\.]?\s*)?结构化发现.*?\n```\w*\n(.*?)```',
    re.DOTALL
)
YAML_FALLBACK_RE = re.compile(
    r'##\s*(?:[一二三四五六七八九十\d]+[、\.]?\s*)?结构化发现\s*\n(.*?)(?:\n##\s|\Z)',
    re.DOTALL
)

MD_FINDING_HEADER_RE = re.compile(r'###\s*\[([^\]]+)\]\s*(.+?)(?:\r?\n|$)')
MD_FILE_RE = re.compile(r'-\s*\*\*文件\*\*:\s*(.+?)(?:\r?\n|$)')
MD_LINE_RE = re.compile(r'-\s*\*\*行号\*\*:\s*(\d+)(?:[-–](\d+))?(?:\r?\n|$)')
MD_SEV_RE = re.compile(r'-\s*\*\*severity\*\*:\s*(critical|high|medium|low|error|warning|info)', re.IGNORECASE)
MD_CONF_RE = re.compile(r'-\s*\*\*置信度\*\*:\s*(?:HIGH|MED|MEDIUM|LOW)\s*\((\d+)%\)', re.IGNORECASE)
MD_CONF_TEXT_RE = re.compile(r'-\s*\*\*置信度\*\*:\s*(HIGH|MED|MEDIUM|LOW)', re.IGNORECASE)
MD_DESC_RE = re.compile(r'-\s*\*\*问题描述\*\*:?\s*\n?\s*(.+?)(?:\r?\n-|\r?\n\*\*|\r?\n```|\Z)', re.DOTALL)

RETRACK_MAX_DISTANCE = 30


def _parse_yaml_findings(text: str) -> List[dict]:
    """解析 YAML 格式的 findings 块（简单解析，不依赖 PyYAML）"""
    m = YAML_FINDINGS_RE.search(text)
    if not m:
        m = YAML_FALLBACK_RE.search(text)
    if not m:
        return []

    yaml_text = m.group(1) if m.lastindex else m.group(0)
    findings = []
    current = {}

    for line in yaml_text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('- id:'):
            if current:
                findings.append(current)
            current = {"id": stripped[5:].strip().strip('"').strip("'")}
        elif stripped and ':' in stripped and current:
            key, _, val = stripped.partition(':')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in ('start_line', 'end_line', 'confidence'):
                try:
                    v = int(val)
                    if key == 'confidence':
                        v = max(0, min(100, v))
                    current[key] = v
                except ValueError:
                    current[key] = 0
            elif key in ('path', 'severity', 'category', 'content', 'suggestion', 'suggestion_code', 'existing_code', 'id'):
                current[key] = val
    if current:
        findings.append(current)

    return findings if findings and any('path' in f for f in findings) else []


def _parse_markdown_findings(text: str) -> List[dict]:
    """从 Markdown 报告正则提取 findings（兜底路径）"""
    findings = []
    blocks = re.split(r'(?=###\s*\[)', text)

    for block in blocks:
        m_header = MD_FINDING_HEADER_RE.match(block)
        if not m_header:
            continue
        clause_id = m_header.group(1).strip()

        m_file = MD_FILE_RE.search(block)
        m_line = MD_LINE_RE.search(block)
        m_sev = MD_SEV_RE.search(block)
        m_conf = MD_CONF_RE.search(block)
        m_conf_text = MD_CONF_TEXT_RE.search(block)
        m_desc = MD_DESC_RE.search(block)

        if not m_file and not m_line:
            continue

        file_path = m_file.group(1).strip() if m_file else ""
        start_line = int(m_line.group(1)) if m_line else 0
        end_line = int(m_line.group(2)) if m_line and m_line.group(2) else start_line
        severity_raw = m_sev.group(1).lower() if m_sev else "medium"
        severity = SEVERITY_MAP.get(severity_raw, "medium")

        if m_conf:
            confidence = max(0, min(100, int(m_conf.group(1))))
        elif m_conf_text:
            confidence = CONFIDENCE_TEXT_MAP.get(m_conf_text.group(1).upper(), 50)
        else:
            confidence = 80

        desc = m_desc.group(1).strip()[:200] if m_desc else m_header.group(2).strip()[:200]

        findings.append({
            "id": clause_id,
            "path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "severity": severity,
            "confidence": confidence,
            "content": desc,
        })

    return findings


def _normalize_finding(d: dict) -> dict:
    """将 PR#822 的 clause yaml schema 归一化为统一格式

    PR#822 字段: clause_id/clause_title/status/confidence(HIGH/MED/LOW)/
                 problem_desc/fix_suggestion/code_snippet{file_path,start_line,end_line,code,line_verified}/
                 evidence{confidence_value}/category/out_of_range
    统一字段: id/path/start_line/end_line/severity/category/confidence/content/suggestion
    """
    if 'clause_id' not in d and 'id' not in d:
        return {}

    clause_id = d.get('clause_id', d.get('id', ''))
    status = d.get('status', '')

    if status == 'PASS':
        return {}

    evidence = d.get('evidence', {})
    if not isinstance(evidence, dict):
        evidence = {}
    conf_raw = evidence.get('confidence_value', d.get('confidence', ''))
    conf_pct = _parse_confidence_pct(conf_raw)
    if conf_pct == 0 and isinstance(conf_raw, str):
        conf_pct = CONFIDENCE_TEXT_MAP.get(conf_raw.upper(), 50)

    snippet = d.get('code_snippet', {})
    if not isinstance(snippet, dict):
        snippet = {}

    severity_raw = d.get('severity', '')
    if not severity_raw:
        conf_text = d.get('confidence', '')
        if conf_text == 'HIGH':
            severity_raw = 'high'
        elif conf_text == 'MED':
            severity_raw = 'medium'
        elif conf_text == 'LOW':
            severity_raw = 'low'
        else:
            severity_raw = 'medium'
    severity = SEVERITY_MAP.get(severity_raw, severity_raw)

    category = d.get('category', 'clause')
    if category == 'clause':
        category = 'bug'

    return {
        'id': clause_id,
        'path': snippet.get('file_path', d.get('path', '')),
        'start_line': snippet.get('start_line', d.get('start_line', 0)) or 0,
        'end_line': snippet.get('end_line', d.get('end_line', 0)) or 0,
        'severity': severity,
        'category': category,
        'confidence': conf_pct,
        'content': d.get('problem_desc', d.get('content', '')),
        'suggestion': d.get('fix_suggestion', d.get('suggestion', '')),
        'suggestion_code': snippet.get('code', d.get('suggestion_code', '')),
        'existing_code': snippet.get('code', d.get('existing_code', '')),
    }


def _parse_confidence_pct(val) -> int:
    """从 confidence_value 字段解析出整数百分比"""
    s = str(val).strip().replace('%', '').replace('+', '')
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


def load_yaml_dir(yaml_dir: Path) -> List[dict]:
    """从 PR#822 的 yaml 输出目录加载 findings

    PR#822 的 skill 通过 collector 将每个 clause 的检视结果落盘为独立 yaml 文件。
    本函数遍历目录下所有 .yaml/.yml 文件，解析为统一 finding 格式。
    """
    if not yaml_dir or not yaml_dir.is_dir():
        return []

    findings = []
    for fpath in sorted(yaml_dir.glob('*.yaml')) + sorted(yaml_dir.glob('*.yml')):
        try:
            raw = fpath.read_bytes()
            if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
                text = raw.decode('utf-16', errors='replace')
            elif raw[:3] == b'\xef\xbb\xbf':
                text = raw[3:].decode('utf-8', errors='replace')
            else:
                text = raw.decode('utf-8', errors='replace')

            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError:
                continue

            if not isinstance(data, dict):
                continue
            if data.get('type') == 'design':
                continue

            finding = _normalize_finding(data)
            if finding:
                findings.append(finding)
        except Exception:
            continue

    return findings


def extract_findings(report_path: Path) -> List[dict]:
    """从 AI 检视报告提取结构化 findings

    优先级：
    1. 同级 yaml 目录（PR#822 collector 落盘的独立 yaml 文件）
    2. Markdown 内嵌 YAML 块（reviewer_bench prompt 要求的结构化发现章节）
    3. Markdown 正则兜底
    """
    yaml_dir = report_path.parent / "yaml"
    if yaml_dir.is_dir():
        findings = load_yaml_dir(yaml_dir)
        if findings:
            return findings

    text = _read_file_auto(report_path)

    findings = _parse_yaml_findings(text)
    if findings:
        return findings

    return _parse_markdown_findings(text)


def _read_file_auto(path: Path) -> str:
    """自动检测编码读取文件（UTF-8/UTF-16/GBK）"""
    raw = path.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode('utf-16', errors='replace')
    elif raw[:3] == b'\xef\xbb\xbf':
        return raw[3:].decode('utf-8', errors='replace')
    else:
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('gbk', errors='replace')


def parse_diff(diff_path: Path) -> Dict[str, List[Tuple[int, int, str]]]:
    """解析 diff 文件，返回 {filepath: [(start_line, end_line, side), ...]}

    side = 'right' (新增行) 或 'left' (删除行)
    """
    if not diff_path or not diff_path.exists():
        return {}

    text = _read_file_auto(diff_path)
    result = {}

    current_file = None
    for line in text.split('\n'):
        m_file = re.match(r'^diff --git a/(.+?) b/(.+?)$', line)
        if m_file:
            current_file = m_file.group(2).strip()
            result[current_file] = []
            continue

        m_hunk = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if m_hunk and current_file:
            old_start = int(m_hunk.group(1))
            old_count = int(m_hunk.group(2) or 1)
            new_start = int(m_hunk.group(3))
            new_count = int(m_hunk.group(4) or 1)

            if old_count > 0:
                result[current_file].append((old_start, old_start + old_count - 1, 'left'))
            if new_count > 0:
                result[current_file].append((new_start, new_start + new_count - 1, 'right'))

    return result


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    return path.replace("\\", "/").split("/")[-1].lower()


def _find_nearest_hunk(line: int, ranges: List[Tuple[int, int, str]],
                       max_distance: int = RETRACK_MAX_DISTANCE) -> Optional[Tuple[int, int, str, int]]:
    """找到距离给定行号最近的 diff hunk，返回 (start, end, side, distance)"""
    if not ranges or line == 0:
        return None

    best = None
    best_dist = float('inf')

    for r_start, r_end, side in ranges:
        if r_start <= line <= r_end:
            return (r_start, r_end, side, 0)

        dist = min(abs(line - r_start), abs(line - r_end))
        if dist < best_dist and dist <= max_distance:
            best_dist = dist
            best = (r_start, r_end, side, dist)

    return best


def _retrack_line(line: int, ranges: List[Tuple[int, int, str]]) -> Tuple[int, str, str, int]:
    """对单行做 re-tracking，返回 (corrected_line, position_status, side, drift_distance)"""
    if line == 0:
        return (0, "no_line", None, 0)

    for r_start, r_end, side in ranges:
        if r_start <= line <= r_end:
            return (line, "in_diff", side, 0)

    nearest = _find_nearest_hunk(line, ranges)
    if nearest:
        r_start, r_end, side, dist = nearest
        corrected = r_start if abs(line - r_start) <= abs(line - r_end) else r_end
        return (corrected, "retracked", side, dist)

    return (line, "line_drift", None, abs(line - ranges[0][0]) if ranges else 0)


def _check_position(line: int, ranges: List[Tuple[int, int, str]]) -> Tuple[str, Optional[str], int]:
    """检查行号位置状态，返回 (position_status, side, drift_distance)

    不修改原始行号，只标注状态。
    """
    if line == 0:
        return ("no_line", None, 0)

    for r_start, r_end, side in ranges:
        if r_start <= line <= r_end:
            return ("in_diff", side, 0)

    nearest = _find_nearest_hunk(line, ranges)
    if nearest:
        r_start, r_end, side, dist = nearest
        return ("retracked", side, dist)

    return ("line_drift", None, abs(line - ranges[0][0]) if ranges else 0)


def positioning(findings: List[dict], diff_ranges: Dict[str, List]) -> List[dict]:
    """定位校正（re-tracking）：标注行号位置状态，保留原始行号用于匹配

    对齐 OCR 的 comment re-tracking module：
    - in_diff: 行号在 diff 范围内
    - retracked: 行号漂移但在容差内，记录最近 hunk 信息作为元数据
    - line_drift: 行号漂移过大
    - file_not_in_diff: 文件不在 diff 中

    保留原始 start_line/end_line 用于匹配（AI 报的行号通常指向实际代码位置，
    修正到 diff hunk 边界反而会偏离）。retracked_line/drift_distance 仅作元数据。
    """
    for f in findings:
        f_path = f.get("path", "")
        f_norm = _normalize_path(f_path)

        matched_ranges = []
        for d_path, ranges in diff_ranges.items():
            if _normalize_path(d_path) == f_norm:
                matched_ranges = ranges
                break

        if not matched_ranges:
            f["position_status"] = "file_not_in_diff"
            f["side"] = None
            f["drift_distance"] = 0
            continue

        start = f.get("start_line", 0)
        end = f.get("end_line", 0) or start

        pos_status, side, drift = _check_position(start, matched_ranges)

        f["position_status"] = pos_status
        f["side"] = side
        f["drift_distance"] = drift

    return findings


def reflection(findings: List[dict], threshold: int = CONFIDENCE_THRESHOLD) -> List[dict]:
    """保守反思过滤 + 跨文件去重

    对齐 OCR 的 review filter 设计：
    1. 阈值 30（非常保守，保召回）
    2. in_diff / retracked 的 findings 不过滤（在变更范围内的发现保留）
    3. critical/high severity 的 findings 不过滤
    4. 跨文件去重：同一条例 ID + 同一 category 的 findings 只保留 confidence 最高的1条
       — 对齐 OCR 的 comment dedup：避免同一问题在多文件重复报
    """
    filtered = []
    for f in findings:
        sev = f.get("severity", "medium")
        conf = f.get("confidence", 0)
        pos = f.get("position_status", "")

        if sev in ("critical", "high"):
            filtered.append(f)
            continue

        if pos in ("in_diff", "retracked"):
            filtered.append(f)
            continue

        if conf >= threshold:
            filtered.append(f)
            continue

        f["filtered_reason"] = f"low_confidence({conf}<{threshold})_not_in_diff"

    deduped = _dedup_cross_file(filtered)
    return deduped


def _dedup_cross_file(findings: List[dict]) -> List[dict]:
    """跨文件去重

    两层策略：
    1. 同条例 ID + content 相似度 ≥ 0.4：去重（同问题跨文件重复报）
    2. 同条例 ID 不同 content：保留（不同代码位置的不同问题）
    """
    import difflib

    by_clause = {}
    no_clause = []
    for f in findings:
        clause_id = f.get("id", "")
        if not clause_id:
            no_clause.append(f)
            continue
        if clause_id not in by_clause:
            by_clause[clause_id] = []
        by_clause[clause_id].append(f)

    result = list(no_clause)
    for clause_id, group in by_clause.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        paths = set(f.get("path", "") for f in group)
        in_diff_count = sum(1 for f in group if f.get("position_status") == "in_diff")

        kept = []
        for f in group:
            f_content = f.get("content", "").lower()
            is_dup = False
            for k_idx, k in enumerate(kept):
                k_content = k.get("content", "").lower()
                ratio = difflib.SequenceMatcher(None, f_content, k_content).ratio()
                if ratio >= 0.4:
                    if f.get("confidence", 0) > k.get("confidence", 0):
                        kept[k_idx] = f
                    is_dup = True
                    break
            if not is_dup:
                kept.append(f)
        result.extend(kept)

    return result


def normalize_report(
    report_path: Path,
    diff_path: Optional[Path] = None,
    confidence_threshold: int = CONFIDENCE_THRESHOLD,
) -> dict:
    """完整归一化流程：提取 → 定位校正 → 反思过滤"""
    findings = extract_findings(report_path)

    diff_ranges = parse_diff(diff_path) if diff_path else {}
    findings = positioning(findings, diff_ranges)

    pre_count = len(findings)
    findings = reflection(findings, confidence_threshold)
    post_count = len(findings)

    retracked_count = sum(1 for f in findings if f.get("position_status") == "retracked")
    in_diff_count = sum(1 for f in findings if f.get("position_status") == "in_diff")

    return {
        "report_file": str(report_path),
        "diff_file": str(diff_path) if diff_path else "",
        "total_extracted": pre_count,
        "after_reflection": post_count,
        "filtered_out": pre_count - post_count,
        "positioning": {
            "in_diff": in_diff_count,
            "retracked": retracked_count,
            "drift_or_other": post_count - in_diff_count - retracked_count,
        },
        "findings": findings,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="归一化 AI 检视报告")
    parser.add_argument("--report", required=True, help="AI 检视报告路径")
    parser.add_argument("--diff", help="diff 文件路径（用于定位校正）")
    parser.add_argument("--output", help="输出 JSON 路径")
    parser.add_argument("--threshold", type=int, default=CONFIDENCE_THRESHOLD, help="confidence 过滤阈值（默认30，保守）")
    args = parser.parse_args()

    result = normalize_report(Path(args.report), Path(args.diff) if args.diff else None, args.threshold)

    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"提取: {result['total_extracted']} → 过滤后: {result['after_reflection']} (过滤 {result['filtered_out']})")  # noqa
    pos = result["positioning"]
    print(f"定位: in_diff={pos['in_diff']} retracked={pos['retracked']} other={pos['drift_or_other']}")  # noqa
    for f in result["findings"]:
        drift = f.get("drift_distance", 0)
        drift_str = f" drift={drift}" if drift > 0 else ""
        orig = f.get("original_start_line", "")
        orig_str = f" orig={orig}" if orig and orig != f.get("start_line", 0) else ""
        print(f"  [{f['id']}] {f.get('path','')}:{f.get('start_line',0)} sev={f.get('severity','')} conf={f.get('confidence',0)} pos={f.get('position_status','')}{drift_str}{orig_str}")  # noqa


if __name__ == "__main__":
    main()
