#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
报告拼接工具 — 读取 yaml 目录，组装精简版 md 报告

由主 Agent 在阶段3（common.report-write.md 薄壳内）调用。读取指定目录下
所有 clause 类 yaml（含 category: clause / style / sunset），组装成
「检视概览 + 检视意见单一大表」两章节报告。

报告形式:
  - 检视概览: 检视文件数（{{FILE_COUNT}} 占位符）/ 代码行数（{{TOTAL_LINES}} 占位符）
              由主 Agent 替换；发现问题总数与高/中/低计数由脚本统计
  - 检视意见: 单一大表（条例编号 | 严重级别 | 问题描述 | 代码位置 | 修改建议），
              按严重级别（高→中→低）→ 文件路径 → 行号排序；SUSPICIOUS 在问题描述
              尾部标注「疑似」，out_of_range 项在代码位置尾部标注
              「PR检视场景，非PR提交的相关代码」
  - 计入问题数: FAIL + SUSPICIOUS + style/sunset 违规 + 范围外；PASS 不渲染不计数
  - 严重级别取 yaml 的 severity 字段（条款声明的 高/中/低）；缺失时按置信度兜底
    （HIGH→高 / MED→中 / LOW→低），仍缺失按「中」

design 类 yaml 不再支持（设计一致性检查已停用）；目录中若残留 type: design
的 yaml，仅告警跳过。

用法:
  python3 workflow.assemble_report.py --dir /tmp/pr1234_xxx \\
      --output /path/to/report.md

退出码: 0=成功, 1=参数错误, 2=目录读取失败
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def _md_cell(val) -> str:
    """转义 markdown 表格单元格中的 | 字符与换行，防止破坏表格结构"""
    if val is None:
        return ""
    return str(val).replace("|", "\\|").replace("\n", " ")


# 判定违规的自信值阈值（自信值 ≥ 本值 → 判定违规），口径见 core/methodology.md
# 「步骤5 — 决策判断」。分级区间以该文件「置信度分级标准」表为准
# （HIGH 80%+ / MED 70-79% / LOW <70%）；本常量只决定「是否违规」，
# 不参与严重级别展示（报告按条款声明的 severity 分档）。改动本常量时须同步
# core/methodology.md。
_VIOLATION_THRESHOLD_PCT = 70


def _parse_confidence_pct(conf_val) -> int:
    """从 confidence_value 字段解析出整数百分比（保留供置信度校验链路使用）"""
    s = str(conf_val).strip().replace("%", "").replace("+", "")
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


# ─── 置信度校验 ──────────────────────────────────────────────────

def parse_score(score_str) -> int:
    """解析证据分值字符串（如 '+40%'、'-15%'）为整数"""
    if score_str is None:
        return 0
    s = str(score_str).strip().replace("%", "").replace("+", "")
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


def verify_confidence(d: dict, fname: str = "") -> bool:
    """校验并修正 clause yaml 的 confidence_value。

    正向证据分值求和 + 负向证据分值求和 = 正确自信值。
    若与 yaml 中写的 confidence_value 不一致 → 原地修正。
    返回 True=已修正, False=无需修正。
    """
    evidence = d.get("evidence")
    if not evidence:
        return False

    # 严格校验：evidence 必须是 dict
    if not isinstance(evidence, dict):
        raise ValueError(
            f"{fname}: evidence must be a mapping "
            f"(positive/negative/confidence_value), "
            f"got {type(evidence).__name__}: {repr(evidence)[:100]}"
        )

    positive = evidence.get("positive", []) or []
    negative = evidence.get("negative", []) or []

    # 严格校验：positive/negative 必须是 list
    if not isinstance(positive, list):
        raise ValueError(f"{fname}: evidence.positive must be a list, got {type(positive).__name__}")
    if not isinstance(negative, list):
        raise ValueError(f"{fname}: evidence.negative must be a list, got {type(negative).__name__}")

    total = 0
    for e in positive:
        if not isinstance(e, dict):
            raise ValueError(f"{fname}: evidence.positive items must be mappings, got {type(e).__name__}")
        total += parse_score(e.get("score", ""))
    for e in negative:
        if not isinstance(e, dict):
            raise ValueError(f"{fname}: evidence.negative items must be mappings, got {type(e).__name__}")
        total += parse_score(e.get("score", ""))

    correct_value = f"{total}%"
    current_value = str(evidence.get("confidence_value", "")).strip()

    if current_value != correct_value:
        evidence["confidence_value"] = correct_value
        return True
    return False


# ─── yaml 加载 ───────────────────────────────────────────────────

def _validate_clause_schema(data: dict, fname: str, schema_errors: List[str]):
    """校验 FAIL/SUSPICIOUS clause 的 code_snippet 结构"""
    status = data.get("status", "")
    if status not in ("FAIL", "SUSPICIOUS"):
        return
    snip = data.get("code_snippet")
    if snip is not None and not isinstance(snip, dict):
        schema_errors.append(
            f"{fname}: code_snippet must be a mapping "
            f"(file_path/start_line/end_line/code), "
            f"got {type(snip).__name__}: {repr(snip)[:100]}"
        )


def _writeback_yaml(data: dict, fpath: str, fname: str):
    """将修改后的 data 写回 yaml 文件"""
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    except Exception as e:
        logger.warning("置信度修正写回失败: %s (%s)", fname, e)


def _fix_confidence_and_writeback(data: dict, fname: str, fpath: str, schema_errors: List[str]) -> int:
    """校验置信度并写回磁盘，返回 confidence_fixed (0 或 1)"""
    try:
        if verify_confidence(data, fname):
            _writeback_yaml(data, fpath, fname)
            return 1
    except ValueError as e:
        schema_errors.append(str(e))
    return 0


def _process_single_yaml(fname: str, fpath: str, schema_errors: List[str]) -> tuple:
    """加载单个 yaml 文件，做 schema 校验 + 置信度修正。

    返回 (data, is_design, confidence_fixed)。
    data 为 None 表示跳过该文件。
    """
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning("跳过无法解析的 yaml: %s (%s)", fname, e)
        return (None, False, 0)
    if not isinstance(data, dict):
        return (None, False, 0)

    if data.get("type") == "design":
        return (data, True, 0)

    _validate_clause_schema(data, fname, schema_errors)
    confidence_fixed = _fix_confidence_and_writeback(data, fname, fpath, schema_errors)

    return (data, False, confidence_fixed)


def _dedup_clauses(clause_list: List[dict]) -> Tuple[List[dict], int]:
    """去重：补派重试或跨组重复提交可能对同一条款重复落盘。

    - PASS：按 clause_id 去重（同条款多组提交只计一次）
    - FAIL/SUSPICIOUS/style：按 (clause_id, file_path, start_line) 三元组去重——
      同条款同位置视为重复（保留状态更严重者：FAIL > SUSPICIOUS > PASS），
      同条款不同位置是不同发现，保留。
    """
    severity = {"FAIL": 3, "SUSPICIOUS": 2, "PASS": 1}
    seen: dict = {}
    for d in clause_list:
        status = d.get("status", "PASS")
        if status == "PASS":
            key = ("pass", d.get("clause_id", ""))
        else:
            snip = d.get("code_snippet") or {}
            key = (d.get("clause_id", ""), snip.get("file_path", ""), snip.get("start_line", ""))
        if key not in seen:
            seen[key] = d
        else:
            old = seen[key]
            if severity.get(status, 0) > severity.get(old.get("status", ""), 0):
                seen[key] = d
    return list(seen.values()), len(clause_list) - len(seen)


def load_yaml_files(yaml_dir: str) -> Tuple[List[dict], List[str]]:
    """加载目录下所有 clause 类 yaml，返回 (clause_list, unreadable)。

    加载时对 clause 类 yaml 做置信度校验：若 evidence 中各证据分值之和与
    confidence_value 不一致，原地修正为正确值。
    design 类 yaml（设计一致性检查已停用）与无法解析的文件按 unreadable 记录，
    在报告头部显式标注，不静默丢弃。
    """
    clause_list: List[dict] = []
    unreadable: List[str] = []
    confidence_fixed = 0
    schema_errors: List[str] = []

    for fname in sorted(os.listdir(yaml_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        fpath = os.path.join(yaml_dir, fname)
        if not os.path.isfile(fpath):
            continue

        data, is_design, cfixed = _process_single_yaml(fname, fpath, schema_errors)
        if data is None:
            # 解析前无法归因到 clause / design（type 字段在内容里），故按文件名记录
            unreadable.append(fname)
            continue
        confidence_fixed += cfixed

        if is_design:
            logger.warning(
                "跳过 design 类 yaml %s（设计一致性检查已停用，不进报告）", fname,
            )
            unreadable.append(fname)
            continue

        clause_list.append(data)

    if unreadable:
        logger.warning("以下 yaml 未组装进报告: %s", "、".join(unreadable))

    # schema 校验失败：列出所有问题后抛异常（由 main 捕获）
    if schema_errors:
        logger.error("%d 个 yaml schema 校验失败:", len(schema_errors))
        for err in schema_errors:
            logger.error("  %s", err)
        raise ValueError(f"{len(schema_errors)} 个 yaml schema 校验失败")

    if confidence_fixed > 0:
        logger.info("置信度校验修正 %d 个 yaml 的 confidence_value", confidence_fixed)

    clause_list, deduped = _dedup_clauses(clause_list)
    if deduped > 0:
        logger.info("自动去重: %d 条重复条款结果已合并（PASS 按 clause_id；FAIL/SUSPICIOUS 按 条款+文件+行号）", deduped)

    return (clause_list, unreadable)


# ─── 统计 ────────────────────────────────────────────────────────

# 严重级别展示顺序（高 → 中 → 低）；未知值排在最后
_SEVERITY_ORDER = {"高": 0, "中": 1, "低": 2}

# 置信度兜底映射（yaml 缺 severity 字段时使用）
_CONFIDENCE_SEVERITY = {"HIGH": "高", "MED": "中", "LOW": "低"}


def resolve_severity(d: dict) -> str:
    """解析发现项的严重级别（高/中/低）。

    优先取 yaml 的 severity 字段（条款声明）；缺失时按置信度兜底
    （HIGH→高 / MED→中 / LOW→低）；仍缺失按「中」。
    """
    sev = str(d.get("severity", "")).strip()
    if sev in _SEVERITY_ORDER:
        return sev
    conf = str(d.get("confidence", "")).strip()
    if conf in _CONFIDENCE_SEVERITY:
        return _CONFIDENCE_SEVERITY[conf]
    return "中"


def collect_findings(clause_list: List[dict]) -> List[dict]:
    """收集所有问题条目（渲染与统计的统一口径）。

    计入: FAIL + SUSPICIOUS（含 category: clause / style / sunset）+ 范围外
    （out_of_range）。PASS 不计入、不渲染。
    """
    findings = []
    for d in clause_list:
        if d.get("status", "") == "PASS":
            continue
        findings.append(d)
    return findings


def compute_stats(findings: List[dict]) -> Tuple[int, int, int, int]:
    """按严重级别统计：返回 (总数, 高, 中, 低)"""
    high = med = low = 0
    for d in findings:
        sev = resolve_severity(d)
        if sev == "高":
            high += 1
        elif sev == "低":
            low += 1
        else:
            med += 1
    return (len(findings), high, med, low)


# ─── 发现条目渲染 ────────────────────────────────────────────────

def finding_sort_key(d: dict) -> tuple:
    """发现项排序键：严重级别（高→中→低）→ 文件路径 → 起始行号，
    使同级别内同一文件的多个检视意见连续排列"""
    snippet = d.get("code_snippet", {}) or {}
    file_path = snippet.get("file_path", "")
    start_line = snippet.get("start_line", 0)
    try:
        start_line = int(start_line)
    except (ValueError, TypeError):
        start_line = 0
    return (_SEVERITY_ORDER.get(resolve_severity(d), 99), file_path, start_line)


def render_code_location(d: dict) -> str:
    """渲染代码位置单元格：{file}:{start}-{end}，附范围外/行号待确认标注"""
    snippet = d.get("code_snippet", {}) or {}
    if not isinstance(snippet, dict):
        return ""
    file_path = snippet.get("file_path", "")
    start = snippet.get("start_line", "")
    end = snippet.get("end_line", "")
    loc = f"{file_path}:{start}-{end}" if file_path else ""
    if d.get("out_of_range"):
        loc += "（PR检视场景，非PR提交的相关代码）"
    if snippet.get("line_verified") is False:
        loc += "（行号待确认）"
    return loc


def render_problem_desc(d: dict) -> str:
    """渲染问题描述单元格：SUSPICIOUS 附疑似标注"""
    desc = str(d.get("problem_desc", "") or "")
    if d.get("status", "") == "SUSPICIOUS":
        desc = desc.rstrip("。") + "。（疑似，建议人工确认）" if desc else "（疑似，建议人工确认）"
    return desc


def render_finding_row(d: dict) -> str:
    """渲染单个发现的大表行：条例编号 | 严重级别 | 问题描述 | 代码位置 | 修改建议"""
    cells = [
        _md_cell(d.get("clause_id", "")),
        _md_cell(resolve_severity(d)),
        _md_cell(render_problem_desc(d)),
        _md_cell(render_code_location(d)),
        _md_cell(d.get("fix_suggestion", "")),
    ]
    return f"| {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} |"


# ─── 报告组装 ───────────────────────────────────────────────────

def _render_header(lines: List[str], findings: List[dict], unreadable=None):
    """渲染报告头部（检视概览）。

    unreadable 为无法解析/已停用类型（design）的 yaml 文件名列表。这类文件
    在完整性校验中仍算「已落盘」（校验基于 ls 文件名），故其承载的条例会
    静默消失——必须在报告中显式标注。
    """
    lines.append("# 代码检视报告")
    lines.append("")
    lines.append("## 检视概览")
    lines.append("")
    lines.append("- 检视文件：{{FILE_COUNT}} 个")
    lines.append("- 代码行数：{{TOTAL_LINES}} 行")
    total, high, med, low = compute_stats(findings)
    lines.append(f"- 发现问题：{total} 个（高 {high} / 中 {med} / 低 {low}）")
    lines.append("")

    if unreadable:
        lines.append(
            f"- ⚠️ 以下 yaml 无法解析、未计入本报告：{'、'.join(unreadable)}"
        )
        lines.append("")


def assemble_report(clause_list: List[dict], unreadable=None) -> str:
    """组装完整报告 md：检视概览 + 检视意见单一大表"""
    lines: List[str] = []

    findings = collect_findings(clause_list)
    findings.sort(key=finding_sort_key)

    # 头部（检视概览）
    _render_header(lines, findings, unreadable)

    # 检视意见（单一大表）
    lines.append("## 检视意见")
    lines.append("")
    if not findings:
        lines.append("本次检视未发现问题。")
        lines.append("")
    else:
        lines.append("| 条例编号 | 严重级别 | 问题描述 | 代码位置 | 修改建议 |")
        lines.append("|---------|---------|---------|---------|---------|")
        for d in findings:
            lines.append(render_finding_row(d))
        lines.append("")

    return "\n".join(lines)


# ─── 主流程 ──────────────────────────────────────────────────────

def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="报告拼接工具 — yaml 目录 → md 报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", required=True, help="yaml 输出目录路径")
    parser.add_argument("--output", required=True, help="报告 md 输出路径")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    yaml_dir = args.dir
    if not os.path.isdir(yaml_dir):
        logger.error("目录不存在: %s", yaml_dir)
        return 2

    try:
        clause_list, unreadable = load_yaml_files(yaml_dir)
    except ValueError:
        return 1

    report_md = assemble_report(clause_list, unreadable=unreadable)

    # 确保输出目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            logger.error("创建输出目录失败: %s", e)
            return 2

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_md)
    except Exception as e:
        logger.error("写报告失败: %s", e)
        return 2

    # 以下为脚本最终产物输出（报告统计摘要）
    findings = collect_findings(clause_list)
    total, high, med, low = compute_stats(findings)
    pass_n = sum(1 for d in clause_list if d.get("status") == "PASS")
    oor_count = sum(1 for d in findings if d.get("out_of_range"))
    style_count = sum(1 for d in findings if d.get("category") == "style")

    logger.info("报告拼接完成")
    logger.info(f"  yaml 目录: {yaml_dir}")
    logger.info(f"  报告路径: {args.output}")
    logger.info(f"  clause yaml: {len(clause_list)} 个（PASS: {pass_n}，问题: {total}，"
                f"其中 style: {style_count}, 范围外: {oor_count}）")
    logger.info(f"  问题统计: 高 {high} / 中 {med} / 低 {low}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
