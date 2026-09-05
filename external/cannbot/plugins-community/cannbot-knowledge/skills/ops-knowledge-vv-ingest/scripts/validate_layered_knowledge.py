#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""校验分层知识（泛化 runbook ⇄ 算子特定 ops）的交叉引用锚点闭合与若干纪律。

用法：
    python3 validate_layered_knowledge.py [--knowledge-root DIR]

DIR 下需有 `runbooks/operator-optimization/vv-fusion-common.md` 与 `ops/**/*.md`（递归子目录）。
`--knowledge-root` 缺省 = `CANNBOT_KNOWLEDGE_ROOT` / `KNOWLEDGE_ROOT` / 当前工作目录。
`--root` 仅作为兼容别名。

退出码：0 = 全过（仅 WARN 不致命）；1 = 有硬错误（HARD）；2 = 找不到目标文件。

检查项：
  [HARD] 悬空锚         —— 每个 `](<rel#锚点>)` / `](rel#锚点)` 的锚点须在目标文件有同名标题
  [HARD] 引用未定义      —— ops 引的 OPT-N/AP-N/CT-N 须在 runbook 有 `## OPT-N` 等
  [HARD] 泛化库标题未瘦身 —— `## OPT/AP/CT` 标题行不得含【标签】或破锚符号 ` ` `<>`
  [HARD] 泛化库泄漏算子特定 —— runbook 出现【算子特定】，或骨架泄漏 golden tiling 变量名
  [HARD] 算子文件泄漏泛化标签 —— ops 出现 优化维度/泛化层级 元数据（旧【】或新 - ** ** 两种载体）
  [WARN] 反链缺失        —— 某泛化 OPT 无任何算子「已知实例」反链
  [WARN] 业务名疑似泄漏    —— 泛化骨架疑似出现 cos/sin/rotate 等业务名（带词边界，可能误报）
  [WARN] 算子结构        —— trick 章节是否前置(§2)、每条 trick 是否含「预期收益」
  [WARN] 可读性(空行)     —— 标签块/各 `**小节**` 与正文之间应空行（防 Markdown 软换行粘连）
  [WARN] 标签块字段       —— 泛化 OPT 缺 优化维度/泛化层级/优先级（新格式另需 摘要/触发/置信度）
  [WARN] CT 字段          —— 泛化 CT-* 缺 症状/根因/规避/预防
  [WARN] golden源        —— ops 未见 `- **golden源**` 结构化字段（供实践 vs golden 比对）
  [WARN] frontmatter     —— ops 缺 YAML frontmatter 必备字段
  [WARN] mermaid 布局     —— ops §N.3 未见 mermaid block-beta UB 内存布局图
  [WARN] UB 占用公式      —— ops §N.3 未见 UB 总占用公式（InitBuffer 字节 / ≤ UB_SIZE 反推）
  [WARN] 独立API章节      —— ops 出现独立「API 速查/汇总/具体实现/清单」章节（应融入伪码）
  [WARN] 本地raw引用      —— ops 正文含本地 `raw/coding-sources` 路径（应改 GitCode blob 永久链接）
  [WARN] 导航 index       —— 缺 ops/index.md 等渐进导航
  [WARN] # 相关托管块      —— ops/runbook 缺文末 `# 相关` okf 托管块（okf:related:start/end）；或仍残留「## 知识来源」节
  [WARN] 反模式专节        —— ops/runbook 有 `## 反模式` 专节时检查格式（无则不报；不再强制预留）
  [WARN] 坏实践全覆盖      —— 某 OPT / §2.x trick 无「坏实践（反例）」字段（无则写「待补充」，避免有的有有的没有）

格式兼容：HARD 检查同时接受旧（`【优化维度: …】` 单行）与新（`- **优化维度**:` 标签块）两种元数据载体。
"""
import argparse
import glob
import logging
import os
import re
import sys

# golden tiling/kernel 具体变量名：出现在「泛化骨架」里即为泄漏（HARD）。非穷举——干净上下文合并子 Agent 是主防线。
GOLDEN_VARS = [
    "oneLoop", "shengMte", "shengUb", "preCoreBatch", "qkcNum", "mulNum",
    "coscdNum", "qPart1Ub", "dstRepSBr", "preCBatch", "cosPart1Ub",
    "q2q1Part1Ub", "halfNum", "preCLTimes", "preCBBTimes", "comBatchBB",
    "cosCast", "sinCast", "speUb", "qcdNum",
    # softmax_v2 等其它算子的 golden 变量（同样不得进泛化骨架）
    "rAligned", "ubFactor", "vlFp32", "binaryAddQuotient", "factorMax",
]
# 业务名（带 ASCII 词边界，可能误报 → WARN）
BIZ_WORDS = ["cos", "sin", "rotate", "rope"]
# frontmatter 必备字段
FM_FIELDS = ["schema_version", "kind", "type", "source_family", "resource", "title", "description", "tags",
             "paradigms", "confidence", "status", "created_at", "updated_at"]

HARD, WARN = [], []


def _emit(message, *args):
    """Write one unprefixed log record to stdout, preserving the CLI contract."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord(__name__, logging.INFO, "", 0, message, args, None)
    handler.handle(record)
    handler.close()


def _read_text(path):
    with open(path, encoding="utf-8") as source_file:
        return source_file.read()


def hard(msg):
    HARD.append(msg)


def warn(msg):
    WARN.append(msg)


def headings(path):
    hs = set()
    with open(path, encoding="utf-8") as source_file:
        for line in source_file:
            m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
            if m:
                hs.add(m.group(1))
    return hs


def extract_links(text):
    """返回所有 markdown 链接的 url（含尖括号 <...> 与裸形式）。"""
    urls = re.findall(r"\]\(<([^>]+)>\)", text)
    urls += re.findall(r"\]\((?!<)([^)\s]+)\)", text)
    return urls


def code_block_lines(path):
    """返回在 ``` 围栏内的行（用于骨架泄漏扫描），排除「已知实例」反链行（含 ops/ 路径）。"""
    out, infence = [], False
    with open(path, encoding="utf-8") as source_file:
        for line in source_file:
            if line.lstrip().startswith("```"):
                infence = not infence
                continue
            if infence and "ops/" not in line:
                out.append(line)
    return out


def _parse_args():
    default_root = os.path.abspath(
        os.environ.get("CANNBOT_KNOWLEDGE_ROOT")
        or os.environ.get("KNOWLEDGE_ROOT")
        or os.getcwd()
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledge-root", "--root", dest="root", default=default_root,
        help="知识根目录（含 runbooks/operator-optimization/vv-fusion-common.md 与 ops/），缺省=仓库根",
    )
    parser.add_argument(
        "--ops", nargs="*", default=None,
        help="只校验这些算子 md（相对 root 或绝对路径）；缺省=递归 ops/ 全部（排除 index.md）",
    )
    return parser.parse_args()


def _resolve_files(root, requested_ops):
    runbook = os.path.join(root, "runbooks", "operator-optimization", "vv-fusion-common.md")
    if requested_ops:
        ops_files = sorted(
            os.path.abspath(path if os.path.isabs(path) else os.path.join(root, path))
            for path in requested_ops
        )
    else:
        ops_files = sorted(glob.glob(os.path.join(root, "ops", "**", "*.md"), recursive=True))
        ops_files = [path for path in ops_files if os.path.basename(path) != "index.md"]
    return runbook, ops_files


def _skip_anchor_check(url):
    return (
        url.startswith("/raw")
        or "coding-sources" in url
        or "gitcode.com" in url
        or "#" not in url
    )


def _check_anchors(root, md_files, headings_by_path):
    for source in md_files:
        base = os.path.dirname(source)
        for url in extract_links(_read_text(source)):
            if _skip_anchor_check(url):
                continue
            path, anchor = url.split("#", 1)
            target = source if path == "" else os.path.normpath(os.path.join(base, path))
            if target in headings_by_path and anchor not in headings_by_path[target]:
                source_rel = os.path.relpath(source, root)
                target_rel = os.path.relpath(target, root)
                hard(f"悬空锚: [{source_rel}] -> {target_rel} #{anchor}")


def _check_definitions_and_backlinks(root, ops_files, runbook_text):
    definitions = set(re.findall(r"^## (OPT-\d+|AP-\d+|CT-\d+)", runbook_text, re.M))
    for operator_file in ops_files:
        references = set(re.findall(r"OPT-\d+|AP-\d+|CT-\d+", _read_text(operator_file)))
        for reference in sorted(references - definitions):
            relpath = os.path.relpath(operator_file, root)
            hard(f"引用未定义: {relpath} 引用 {reference}，但 runbook 无 `## {reference}`")
    for block in re.split(r"\n## ", runbook_text):
        match = re.match(r"(OPT-\d+)", block)
        if match and "ops/" not in block:
            warn(f"反链缺失: runbook {match.group(1)} 无任何算子「已知实例」反链")


def _check_runbook_titles(runbook_text):
    for line in runbook_text.splitlines():
        if not re.match(r"^## (OPT-|AP-|CT-)", line):
            continue
        if "【" in line:
            hard(f"泛化库标题未瘦身（含【标签】）: {line.strip()}")
        if "`" in line or "<" in line or ">" in line:
            hard(f"泛化库标题含破锚符号: {line.strip()}")


def _is_specific_tag_leak(line, negations):
    return (
        "【算子特定" in line
        and not line.lstrip().startswith(">")
        and not any(negation in line for negation in negations)
    )


def _check_runbook_leaks(runbook, runbook_text):
    negations = ("绝不", "不得", "禁", "不标", "不能", "别", "非")
    for line in runbook_text.splitlines():
        if _is_specific_tag_leak(line, negations):
            hard(f"泛化库出现【算子特定】标签: {line.strip()[:80]}")
    for line in code_block_lines(runbook):
        for variable in GOLDEN_VARS:
            if re.search(r"(?<![A-Za-z])" + re.escape(variable) + r"(?![A-Za-z])", line):
                hard(f"泛化骨架泄漏 golden 变量 `{variable}`: {line.strip()[:80]}")
        for word in BIZ_WORDS:
            if re.search(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", line):
                warn(f"泛化骨架疑似业务名 `{word}`（可能误报）: {line.strip()[:80]}")


def _check_operator_metadata_leaks(root, ops_files):
    tags = ("【泛化层级", "【优化维度", "- **泛化层级**", "- **优化维度**")
    for operator_file in ops_files:
        text = _read_text(operator_file)
        for tag in tags:
            if tag in text:
                relpath = os.path.relpath(operator_file, root)
                hard(f"算子文件 {relpath} 出现泛化专有元数据 {tag}")


def _check_operator_structure(root, ops_files):
    for operator_file in ops_files:
        lines = _read_text(operator_file).splitlines()
        section_two = next((line for line in lines if re.match(r"^## 2\. ", line)), "")
        relpath = os.path.relpath(operator_file, root)
        if "trick" not in section_two and "优化" not in section_two:
            actual = section_two.strip()[:40]
            warn(f"{relpath}: §2 似乎不是「算子特定优化 trick」前置章节（实际: {actual}）")
        trick_count = len(re.findall(r"^### 2\.\d", "\n".join(lines), re.M))
        gain_count = sum(1 for line in lines if "**预期收益**" in line or "- **预期收益**" in line)
        if trick_count and gain_count < trick_count:
            warn(f"{relpath}: {trick_count} 条 trick 但只有 {gain_count} 条含「预期收益」字段")


def _check_file_spacing(root, path, section_pattern):
    relpath = os.path.relpath(path, root)
    lines = _read_text(path).splitlines()
    in_fence = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.strip()
        if section_pattern.match(stripped) and index > 0:
            previous = lines[index - 1].strip()
            if previous != "" and not previous.startswith("```"):
                warn(f"可读性(空行): {relpath} 第{index + 1}行「{stripped[:18]}」前缺空行（粘连）")
        is_runbook_heading = re.match(r"^## (OPT-|AP-|CT-)", line)
        has_following_text = index + 1 < len(lines) and lines[index + 1].strip() != ""
        if is_runbook_heading and has_following_text:
            warn(f"可读性(空行): {relpath} 标题「{stripped[:24]}」后缺空行")


def _check_spacing(root, md_files):
    section_labels = (
        "原则", "通用骨架", "示意图", "迁移条件", "坏实践", "已知实例",
        "做法", "症状", "根因", "规避", "预防",
    )
    section_pattern = re.compile(r"^\*\*(" + "|".join(section_labels) + r")")
    for path in md_files:
        _check_file_spacing(root, path, section_pattern)


def _check_new_opt_fields(block, opt_id):
    for field in ("摘要", "触发", "置信度"):
        if f"- **{field}**" not in block:
            warn(f"标签块缺字段(新格式): runbook {opt_id} 缺「{field}」")


def _check_opt_fields(runbook_text):
    for block in re.split(r"\n## ", runbook_text):
        match = re.match(r"(OPT-\d+)", block)
        if not match:
            continue
        opt_id = match.group(1)
        for field in ("优化维度", "泛化层级", "优先级"):
            if f"- **{field}**" not in block and f"【{field}" not in block:
                warn(f"标签块缺字段: runbook {opt_id} 缺「{field}」")
        if "- **优化维度**" in block:
            _check_new_opt_fields(block, opt_id)


def _check_ct_fields(runbook_text):
    for block in re.split(r"\n## ", runbook_text):
        match = re.match(r"(CT-\d+)", block)
        if not match:
            continue
        constraint_id = match.group(1)
        for field in ("症状", "根因", "规避", "预防"):
            if field not in block:
                warn(f"CT 缺字段: runbook {constraint_id} 缺「{field}」")


def _check_golden_sources(root, ops_files):
    for operator_file in ops_files:
        if "**golden源**" not in _read_text(operator_file):
            relpath = os.path.relpath(operator_file, root)
            warn(f"{relpath}: 未见 `- **golden源**` 字段（建议算子 trick/模板挂 golden 出处，供实践 vs golden 比对）")


def _check_frontmatter(root, ops_files):
    for operator_file in ops_files:
        relpath = os.path.relpath(operator_file, root)
        text = _read_text(operator_file)
        if not text.startswith("---"):
            warn(f"{relpath}: 缺 YAML frontmatter（应含 {'/'.join(FM_FIELDS)}）")
            continue
        parts = text.split("---", 2)
        frontmatter = parts[1] if len(parts) >= 3 else ""
        for field in FM_FIELDS:
            if not re.search(rf"^{field}\s*:", frontmatter, re.M):
                warn(f"{relpath}: frontmatter 缺字段 `{field}`")


def _check_ub_documentation(root, ops_files):
    for operator_file in ops_files:
        relpath = os.path.relpath(operator_file, root)
        text = _read_text(operator_file)
        if "```mermaid" not in text or "block-beta" not in text:
            warn(f"{relpath}: 未见 mermaid block-beta UB 内存布局图（§N.3.1）")
    for operator_file in ops_files:
        relpath = os.path.relpath(operator_file, root)
        text = _read_text(operator_file)
        if "UB 总占用" not in text and "InitBuffer" not in text and "UB_SIZE" not in text:
            warn(f"{relpath}: 未见 UB 总占用公式（§N.3.2，InitBuffer 字节 / ≤ UB_SIZE 反推）")


def _api_section_titles(path, pattern):
    titles = []
    with open(path, encoding="utf-8") as source_file:
        for line in source_file:
            if re.match(r"^#{1,3}\s", line) and pattern.search(line):
                titles.append(line.strip()[:40])
    return titles


def _check_api_sections(root, ops_files):
    api_chapter = re.compile(r"API\s*(速查|汇总|具体实现|清单)")
    for operator_file in ops_files:
        relpath = os.path.relpath(operator_file, root)
        for title in _api_section_titles(operator_file, api_chapter):
            warn(f"{relpath}: 出现独立 API 汇总章节「{title}」（API 细节应融入伪码）")


def _check_local_source_links(root, ops_files):
    for operator_file in ops_files:
        if "raw/coding-sources" in _read_text(operator_file):
            relpath = os.path.relpath(operator_file, root)
            warn(f"{relpath}: 正文含本地 raw/coding-sources 路径（应改 GitCode blob 永久链接）")


def _check_related_sections(root, md_files):
    for path in md_files:
        relpath = os.path.relpath(path, root)
        text = _read_text(path)
        has_managed_block = "<!-- okf:related:start -->" in text
        has_related_heading = re.search(r"^#\s+相关\s*$", text, re.M) is not None
        if not has_managed_block or not has_related_heading:
            warn(f"{relpath}: 未见文末 `# 相关` 托管块（okf:related:start/end + `# 相关`）")
        if re.search(r"^##\s+知识来源", text, re.M):
            warn(f"{relpath}: 仍含「## 知识来源」节（应换为 `# 相关`；golden 出处保留在 §1 + frontmatter）")


def _antipattern_body(text):
    if re.search(r"^##\s+反模式(（.*?）)?\s*$", text, re.M) is None:
        return None
    return re.split(r"^##\s+反模式", text, maxsplit=1, flags=re.M)[1].split("\n## ")[0]


def _check_antipattern_sections(root, md_files):
    for path in md_files:
        body = _antipattern_body(_read_text(path))
        if body is None or "待补充" in body:
            continue
        if re.search(r"^###\s+AP-\d+", body, re.M) or body.strip(" \n>-"):
            continue
        relpath = os.path.relpath(path, root)
        warn(f"{relpath}: `## 反模式` 专节非空但无 AP-N 条目（若有反例写 `### AP-N`，无则删该节）")


def _check_operator_bad_practices(root, operator_file):
    operator_section = re.search(r"\n## 2\. .*?(?=\n## )", _read_text(operator_file), re.S)
    if not operator_section:
        return
    relpath = os.path.relpath(operator_file, root)
    for trick_block in re.split(r"\n### 2\.\d", operator_section.group(0))[1:]:
        if "坏实践" not in trick_block:
            heading = trick_block.strip().splitlines()[0][:24] if trick_block.strip() else "?"
            warn(f"坏实践缺失: {relpath} §2 trick「{heading}」无「坏实践」字段（无则写「待补充」）")


def _check_bad_practices(root, ops_files, runbook_text):
    for block in re.split(r"\n## ", runbook_text):
        match = re.match(r"(OPT-\d+)", block)
        if match and "坏实践" not in block:
            warn(f"坏实践缺失: runbook {match.group(1)} 无「坏实践（反例）」字段（无则写「待补充」）")
    for operator_file in ops_files:
        _check_operator_bad_practices(root, operator_file)


def _report(root, runbook, ops_files):
    _emit("== 校验根目录: %s ==", root)
    runbook_rel = os.path.relpath(runbook, root)
    ops_rel = ", ".join(os.path.relpath(path, root) for path in ops_files)
    _emit("   runbook: %s；ops: %s", runbook_rel, ops_rel)
    if HARD:
        _emit("\n[HARD 错误 × %d]", len(HARD))
        for message in HARD:
            _emit("  ✗ %s", message)
    if WARN:
        _emit("\n[WARN 提示 × %d]", len(WARN))
        for message in WARN:
            _emit("  ! %s", message)
    if not HARD and not WARN:
        _emit("\n✅ 全部通过（无 HARD、无 WARN）")
    elif not HARD:
        _emit("\n✅ 无 HARD 错误（%d 条 WARN，请人工确认是否误报）", len(WARN))
    else:
        _emit("\n❌ %d 条 HARD 错误，须修正后重出", len(HARD))
    return 1 if HARD else 0


def main():
    args = _parse_args()
    root = os.path.abspath(args.root)
    runbook, ops_files = _resolve_files(root, args.ops)
    if not os.path.exists(runbook):
        _emit("[FATAL] 找不到 %s", runbook)
        return 2
    if not ops_files:
        _emit("[FATAL] %s/ops/ 下没有算子 *.md（已排除 index.md）", root)
        return 2

    md_files = [runbook] + ops_files
    headings_by_path = {path: headings(path) for path in md_files}
    runbook_text = _read_text(runbook)
    _check_anchors(root, md_files, headings_by_path)
    _check_definitions_and_backlinks(root, ops_files, runbook_text)
    _check_runbook_titles(runbook_text)
    _check_runbook_leaks(runbook, runbook_text)
    _check_operator_metadata_leaks(root, ops_files)
    _check_operator_structure(root, ops_files)
    _check_spacing(root, md_files)
    _check_opt_fields(runbook_text)
    _check_ct_fields(runbook_text)
    _check_golden_sources(root, ops_files)
    _check_frontmatter(root, ops_files)
    _check_ub_documentation(root, ops_files)
    _check_api_sections(root, ops_files)
    _check_local_source_links(root, ops_files)
    if not os.path.exists(os.path.join(root, "ops", "index.md")):
        warn("缺 ops/index.md 渐进导航（OKF）")
    _check_related_sections(root, md_files)
    _check_antipattern_sections(root, md_files)
    _check_bad_practices(root, ops_files, runbook_text)
    return _report(root, runbook, ops_files)


if __name__ == "__main__":
    raise SystemExit(main())
