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
"""端到端评测脚本 — 对齐 aacr-bench judge.py 的 4 阶段匹配 + OCR 归一化。

流程：
  1. normalize_report 提取 AI findings（YAML 优先，Markdown 兜底）+ re-tracking + 保守过滤
  2. 程序化提取 GT issues
  3. 4 阶段匹配：path → side → line(k) → semantic
     - line_k 放宽到 10（对齐 OCR 的定位容差，re-tracking 后仍可能有残差）
     - semantic 匹配用三策略：关键词重叠 + 序列相似度 + 同义词扩展
  4. 分层指标：line_precision/recall + semantic_precision/recall + severity_breakdown

召回率优先策略：
  - line 阶段：放宽 k 到 10，闭区间重叠即算匹配
  - semantic 阶段：任一策略命中即算匹配（OR 而非 AND）
  - reflection：阈值 30，in_diff/critical/high 不过滤
  - re-tracking：标注行号位置状态，不覆盖原始行号

severity 词表统一为 critical/high/medium/low（对齐 OCR LlmComment）。

用法:
    python scripts/run_eval.py --ai-dir reports/e2e_redline --gt-dir benchmark_tasks/top20_redline_and_topk_filtered
    python scripts/run_eval.py --ai-dir reports/redline_verify \
        --gt-dir benchmark_tasks/top20_redline_and_topk_filtered --diff-dir bench_data_redline
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_report import extract_findings, positioning, reflection, parse_diff, CONFIDENCE_THRESHOLD

LINE_K = 10
SEVERITY_LEVELS = ("critical", "high", "medium", "low")

SYNONYM_GROUPS = [
    {"nullptr", "null", "空指针", "none", "nil", "判空", "未判空"},
    {"check", "校验", "检查", "验证", "validate", "verify", "未校验", "校验缺失", "缺少校验"},
    {"overflow", "越界", "溢出", "out of bounds", "out_of_bounds"},
    {"magic number", "魔数", "magic_number", "字面量", "裸数字", "裸数字魔数"},
    {"assert", "断言", "assertion", "ascend_assert"},
    {"divide", "除零", "divide by zero", "div by zero", "除以零", "除数", "k=0", "n=0", "零值", "optkb", "倒数"},
    {"sync", "同步", "synchronize", "barrier", "流水同步", "setflag", "waitflag", "pipebarrier", "crosscore", "syncall", "mte", "mte2", "mte3", "syncmte", "tomte", "fetcheventid", "事件id", "eventid", "流同步"},
    {"buffer", "缓冲区", "ub", "localtensor", "tbuf", "localtensor", "que", "initbuffer", "alloc", "free", "double buffer", "双缓冲"},
    {"init", "初始化", "initialize", "setup", "未初始化"},
    {"return", "返回值", "返回", "retval", "ret", "setblockdim", "未校验返回值"},
    {"context", "上下文", "tilingcontext", "tiling_context"},
    {"shape", "维度", "dim", "dimension", "size", "tensor", "sizetensor"},
    {"type", "类型", "dtype", "datatype", "tbuf", "tque"},
    {"cast", "转换", "convert", "类型转换", "reinterpretcast"},
    {"copy", "搬运", "datacopy", "memcpy", "copyin", "copyout", "datcopypad", "搬运量"},
    {"offset", "偏移", "偏移量", "gm", "globalbuffer", "setglobalbuffer", "uint32", "uint64", "int64", "中间值", "乘加", "溢出"},
    {"scale", "量化", "quant", "dequant", "pertoken", "scale", "dynamcquant", "全零行"},
    {"tile", "tiling", "切分", "tilingdata", "blockdim", "block_num", "blockdim", "核分配", "ub容量", "ub预算"},
    {"permission", "职责", "kernel", "host", "tiling", "分层", "职责边界", "aicore"},
    {"atomic", "atomicmax", "原子操作", "原子", "atomicadd", "原子加", "原子max"},
    {"bounds", "边界", "上界", "下界", "范围", "boundary", "limit", "越界写", "越界风险", "out_of_range"},
    {"index", "索引", "下标", "dstindex", "inputtensor", "数组下标", "数组索引"},
    {"race", "竞态", "race condition", "数据竞争", "并发", "concurrent"},
    {"zero", "零", "0值", "空值", "empty", "全零"},
    {"constexpr", "常量表达式", "编译期常量", "编译期"},
    {"dim3", "dim3", "线程维度", "thread dim", "vf_call"},
    {"scatter", "scatter", "分散", "散布", "scatter"},
    {"deduplicate", "dedup", "去重", "deduplicate", "deduplicateindices"},
    {"loop", "循环", "for", "iteration", "逐元素"},
    {"int", "int", "整型", "整数", "int32", "int64", "uint32", "uint64"},
    {"unsafe", "不安全", "unsafe", "风险", "danger"},
    {"migrate", "迁移", "migrate", "rename", "重命名", "aligned", "alignup"},
    {"align", "对齐", "alignup", "aligned", "ops::aligned"},
]

GT_ISSUE_RE = re.compile(r'####\s*\[(\d+)\]')
GT_FILE_RE = re.compile(r'-\s*\*\*文件\*\*:\s*(.+?)(?:\n|$)')
GT_LINE_RE = re.compile(r'-\s*\*\*行号\*\*:\s*(\d+)(?:\n|$)')
GT_DESC_RE = re.compile(r'-\s*\*\*问题描述\*\*:?\s*\n?\s*>?\s*(.+?)(?:\n|$)', re.DOTALL)

SEVERITY_KEYWORDS = {
    "critical": ["nullptr", "空指针", "crash", "崩溃", "越界", "越界写", "越界风险", "overflow", "underflow", "非法", "数据损坏", "死锁", "atomicmax", "原子", "竞态", "race", "除零", "div by zero", "写零", "k=0", "n=0"],
    "high": ["逻辑错误", "bug", "error", "失败", "failed", "安全漏洞", "index", "索引", "上界", "下界", "边界", "未检查", "缺少校验", "除零", "零值", "倒数", "溢出", "ub容量", "未拒绝"],
    "medium": ["magic number", "魔数", "建议", "规范", "style", "命名", "注释", "缺少", "int", "整型", "类型", "迁移", "align", "对齐", "constexpr", "dim3", "dim3参数"],
    "low": ["格式", "排版", "空格", "换行", "拼写", "命名不一致"],
}


def infer_severity(desc: str) -> str:
    desc_lower = desc.lower()
    for sev, keywords in SEVERITY_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return sev
    return "medium"


def extract_gt_issues(gt_path: Path) -> list:
    """从 GT 报告提取问题清单"""
    text = gt_path.read_text(encoding='utf-8', errors='replace')
    issues = []

    blocks = re.split(r'(?=####\s*\[\d+\])', text)
    for block in blocks:
        m_num = GT_ISSUE_RE.match(block)
        if not m_num:
            continue
        num = m_num.group(1)

        m_file = GT_FILE_RE.search(block)
        m_line = GT_LINE_RE.search(block)
        m_desc = GT_DESC_RE.search(block)

        if not m_desc:
            m_desc = re.search(r'问题描述\*\*:?\s*\n?\s*>?\s*(.+?)(?:\n|$)', block)

        desc = m_desc.group(1).strip() if m_desc else ""
        file_path = m_file.group(1).strip() if m_file else ""
        line = int(m_line.group(1)) if m_line else 0

        if not desc and not file_path:
            continue

        issues.append({
            "id": num,
            "path": file_path,
            "start_line": line,
            "end_line": line,
            "severity": infer_severity(desc),
            "desc": desc[:200],
        })
    return issues


def extract_ai_findings(ai_path: Path, diff_path: Optional[Path] = None) -> list:
    """提取 AI findings：normalize_report（YAML/Markdown）+ 定位校验 + 反思过滤"""
    findings = extract_findings(ai_path)

    if diff_path and diff_path.exists():
        diff_ranges = parse_diff(diff_path)
        findings = positioning(findings, diff_ranges)

    findings = reflection(findings, threshold=CONFIDENCE_THRESHOLD)

    return [{
        "id": f.get("id", ""),
        "path": f.get("path", ""),
        "start_line": f.get("start_line", 0),
        "end_line": f.get("end_line", 0) or f.get("start_line", 0),
        "severity": f.get("severity", "medium"),
        "confidence": f.get("confidence", 0),
        "side": f.get("side"),
        "position_status": f.get("position_status", ""),
        "desc": f.get("content", "")[:200],
    } for f in findings]


def normalize_path(path: str) -> str:
    if not path:
        return ""
    parts = path.replace("\\", "/").split("/")
    return parts[-1].lower() if parts else ""


def _path_match(gt_path: str, ai_path: str) -> bool:
    """阶段1：文件路径匹配（末段路径比较）"""
    n1, n2 = normalize_path(gt_path), normalize_path(ai_path)
    if not n1 or not n2:
        return True
    return n1 == n2


def _side_match(gt_side: Optional[str], ai_side: Optional[str]) -> bool:
    """阶段2：side 匹配（None 视为通配）"""
    if gt_side is None or ai_side is None:
        return True
    return gt_side == ai_side


def _line_match(gt_start: int, gt_end: int, ai_start: int, ai_end: int, k: int = LINE_K) -> bool:
    """阶段3：行号匹配

    对齐 aacr-bench judge.py 的 diff_location_is_same，但修正宽范围问题：
    - AI finding 的行号范围 span ≤ 10：用闭区间重叠或距离 ≤ k
    - AI finding 的 span > 10（函数级 finding）：只用 start_line 做单点匹配
      避免"函数级"finding 吃掉范围内所有 GT 单行的 line 匹配
    """
    if gt_start == 0 or ai_start == 0:
        return True

    ai_span = (ai_end - ai_start) if (ai_end and ai_start) else 0
    if ai_span > 10:
        return abs(gt_start - ai_start) <= k

    has_overlap = not (gt_start > ai_end or gt_end < ai_start)
    if has_overlap:
        return True

    min_dist = min(abs(gt_start - ai_end), abs(gt_end - ai_start))
    return min_dist <= k


def _expand_synonyms(word: str, word_set: set) -> set:
    """扩展同义词：如果 word 在某个同义词组里，或包含同义词组中的词，加入整组"""
    for group in SYNONYM_GROUPS:
        if word in group:
            word_set.update(group)
        else:
            for g in group:
                if g in word or word in g:
                    word_set.update(group)
                    break
    return word_set


def _semantic_match(gt_desc: str, ai_desc: str) -> bool:
    """阶段4：语义匹配（三策略 OR）

    对齐 OCR 的 semantic judgment，使用三策略任一命中即算匹配：
    1. 序列相似度 ≥ 0.35（difflib）
    2. 关键词 Jaccard ≥ 0.25
    3. 同义词扩展后关键词重叠 ≥ 2 个
    """
    if not gt_desc or not ai_desc:
        return False

    gt_lower = gt_desc.lower()
    ai_lower = ai_desc.lower()

    seq_ratio = difflib.SequenceMatcher(None, gt_lower, ai_lower).ratio()
    if seq_ratio >= 0.35:
        return True

    gt_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z_]{2,}', gt_lower))
    ai_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z_]{2,}', ai_lower))
    if not gt_words or not ai_words:
        return False

    jaccard = len(gt_words & ai_words) / len(gt_words | ai_words)
    if jaccard >= 0.25:
        return True

    gt_expanded = set()
    for w in gt_words:
        _expand_synonyms(w, gt_expanded)
    gt_expanded.update(gt_words)

    ai_expanded = set()
    for w in ai_words:
        _expand_synonyms(w, ai_expanded)
    ai_expanded.update(ai_words)

    overlap = len(gt_expanded & ai_expanded)
    raw_overlap = len(gt_words & ai_words)
    if overlap >= 3:
        return True
    if overlap >= 2 and raw_overlap >= 1:
        return True

    return False


def match_issues(gt_issues: list, ai_issues: list, k: int = LINE_K) -> dict:
    """4 阶段匹配：path → side → line(k) → semantic

    支持 1:N 匹配（1条 AI finding 可匹配多条 GT issues）：
    - line_match：1条 AI finding 只贡献1次 line 匹配计数（用于 precision）
    - semantic_match：1条 AI finding 可匹配多条 GT（用于 recall）
    对齐 aacr-bench judge.py 的去重逻辑：line 去重，semantic 不去重。
    """
    matched_pairs = []
    unmatched_gt = []
    used_ai_line = set()

    for gt in gt_issues:
        line_matched = False
        semantic_matched = False
        best_ai_idx = -1

        for i, ai in enumerate(ai_issues):
            if not _path_match(gt["path"], ai["path"]):
                continue
            if not _side_match(gt.get("side"), ai.get("side")):
                continue
            if not _line_match(gt["start_line"], gt["end_line"], ai["start_line"], ai["end_line"], k):
                continue

            if not line_matched and i not in used_ai_line:
                line_matched = True
                used_ai_line.add(i)
                if best_ai_idx < 0:
                    best_ai_idx = i

            if _semantic_match(gt["desc"], ai["desc"]):
                semantic_matched = True
                best_ai_idx = i
                break

        if line_matched or semantic_matched:
            matched_pairs.append((gt, ai_issues[best_ai_idx] if best_ai_idx >= 0 else ai_issues[0], line_matched, semantic_matched))
        else:
            unmatched_gt.append(gt)

    false_positives = [ai_issues[i] for i in range(len(ai_issues)) if i not in used_ai_line]

    semantic_matched_ai = set()
    for _, ai, _, sm in matched_pairs:
        if sm:
            for i, a in enumerate(ai_issues):
                if a is ai:
                    semantic_matched_ai.add(i)
                    break

    return {
        "matched_pairs": matched_pairs,
        "unmatched_gt": unmatched_gt,
        "false_positives": false_positives,
        "line_matched_ai_count": len(used_ai_line),
        "semantic_matched_ai_count": len(semantic_matched_ai),
    }


def compute_metrics(gt_issues: list, ai_issues: list, match_result: dict) -> dict:
    """计算分层指标：line_precision/recall + semantic_precision/recall"""
    gt_total = len(gt_issues)
    ai_total = len(ai_issues)

    line_matched = sum(1 for _, _, lm, _ in match_result["matched_pairs"] if lm)
    semantic_matched = sum(1 for _, _, _, sm in match_result["matched_pairs"] if sm)
    missed = len(match_result["unmatched_gt"])
    fp = len(match_result["false_positives"])

    line_recall = round(line_matched / gt_total, 4) if gt_total > 0 else None
    line_precision = round(match_result["line_matched_ai_count"] / ai_total, 4) if ai_total > 0 else None
    semantic_recall = round(semantic_matched / gt_total, 4) if gt_total > 0 else None
    semantic_precision = round(match_result["semantic_matched_ai_count"] / ai_total, 4) if ai_total > 0 else None

    f1 = None
    if semantic_recall is not None and semantic_precision is not None and (semantic_recall + semantic_precision) > 0:
        f1 = round(2 * semantic_recall * semantic_precision / (semantic_recall + semantic_precision), 4)

    sev_breakdown = {}
    for sev in SEVERITY_LEVELS:
        gt_sev = [g for g in gt_issues if g["severity"] == sev]
        matched_sev = sum(1 for g, _, _, sm in match_result["matched_pairs"] if g["severity"] == sev and sm)
        total_sev = len(gt_sev)
        sev_breakdown[sev] = {
            "total": total_sev,
            "matched": matched_sev,
            "recall": round(matched_sev / total_sev, 4) if total_sev > 0 else None,
        }

    def _fmt_finding(f, prefix=""):
        return f"[{f.get('id', '')}] {f.get('path', '')}:{f.get('start_line', 0)} — {f.get('desc', '')[:80]}"

    details = []
    for g, a, lm, sm in match_result["matched_pairs"]:
        details.append({
            "gt_issue": _fmt_finding(g),
            "severity": g["severity"],
            "line_matched": lm,
            "semantic_matched": sm,
            "matched": sm,
            "ai_issue": _fmt_finding(a),
        })
    for g in match_result["unmatched_gt"]:
        details.append({
            "gt_issue": _fmt_finding(g),
            "severity": g["severity"],
            "line_matched": False,
            "semantic_matched": False,
            "matched": False,
            "ai_issue": "",
        })

    noise_rate = round(fp / ai_total, 4) if ai_total > 0 else 0.0

    return {
        "gt_total": gt_total,
        "ai_total": ai_total,
        "matched": semantic_matched,
        "missed": missed,
        "false_positive": fp,
        "noise_rate": noise_rate,
        "line_recall": line_recall,
        "line_precision": line_precision,
        "semantic_recall": semantic_recall,
        "semantic_precision": semantic_precision,
        "recall": semantic_recall,
        "precision": semantic_precision,
        "f1": f1,
        "severity_breakdown": sev_breakdown,
        "details": details,
        "false_positive_details": [
            {"ai_issue": _fmt_finding(a), "severity": a["severity"]}
            for a in match_result["false_positives"]
        ],
    }


def find_gt_report(gt_dir: Path, repo: str, pr_num: str, commit: str) -> Optional[Path]:
    patterns = [
        f"*{repo}_pr_{pr_num}_{commit}.md" if commit else None,
        f"*{repo}_pr_{int(pr_num):04d}_{commit}.md" if commit else None,
        f"*{repo}_pr_{pr_num}_*.md",
        f"*{repo}_pr_{int(pr_num):04d}_*.md",
    ]
    for p in patterns:
        if not p:
            continue
        matches = list(gt_dir.glob(p))
        if matches:
            return matches[0]
    return None


def load_pr_map(manifest_path: Optional[Path] = None, ai_dir: Optional[Path] = None,
                gt_dir: Optional[Path] = None) -> dict:
    """从 manifest 加载 PR 映射，或从 AI 目录自动扫描

    manifest 格式: {"entries": [{"repo":"ops-math","pr_number":"322","base_commit":"8b80dc338867",...}]}
    自动扫描: 遍历 ai_dir 下的子目录，从目录名解析 repo 和 pr_number
    """
    if manifest_path and manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        pr_map = {}
        for entry in manifest.get('entries', []):
            repo = entry['repo']
            pr_num = entry['pr_number']
            base_commit = entry.get('base_commit', entry.get('commit_hash', ''))
            pr_name = f"{repo}-pr-{pr_num}"
            pr_map[pr_name] = (repo, pr_num, base_commit)
        return pr_map

    if ai_dir and ai_dir.exists():
        pr_map = {}
        for d in sorted(ai_dir.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            parts = name.rsplit('-pr-', 1)
            if len(parts) == 2:
                repo, pr_num = parts[0], parts[1]
                pr_map[name] = (repo, pr_num, '')
        return pr_map

    return {}


def main():
    parser = argparse.ArgumentParser(description="端到端评测（4阶段匹配 + 归一化）")
    parser.add_argument("--ai-dir", default="reports/e2e_redline")
    parser.add_argument("--gt-dir", default="benchmark_tasks/top20_redline_and_topk_filtered")
    parser.add_argument("--diff-dir", default="bench_data_redline", help="diff 文件根目录（用于定位校验）")
    parser.add_argument("--output", default="reports/e2e_redline/eval_results.json")
    parser.add_argument("--line-k", type=int, default=LINE_K, help="行号匹配容差")
    parser.add_argument("--manifest", default=None, help="manifest JSON 路径（提供 PR→commit 映射，不提供则从 ai-dir 自动扫描）")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    ai_dir = base / args.ai_dir
    gt_dir = base / args.gt_dir
    diff_dir = base / args.diff_dir
    out_path = base / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = base / args.manifest if args.manifest else None
    pr_map = load_pr_map(manifest_path, ai_dir, gt_dir)
    if not pr_map:
        print(f"[ERROR] 未找到任何 PR（manifest={manifest_path}, ai_dir={ai_dir}）", file=sys.stderr)
        sys.exit(1)

    all_results = []
    totals = {"gt": 0, "ai": 0, "matched": 0, "missed": 0, "fp": 0, "line_ai": 0, "sem_ai": 0}

    for pr_name, (repo, pr_num, commit) in pr_map.items():
        ai_report = ai_dir / pr_name / "review_report.md"
        if not ai_report.exists():
            print(f"[SKIP] {pr_name}: AI 报告不存在", file=sys.stderr)
            continue

        gt_report = find_gt_report(gt_dir, repo, pr_num, commit)
        if not gt_report:
            print(f"[SKIP] {pr_name}: GT 报告未找到", file=sys.stderr)
            continue

        diff_path = diff_dir / pr_name / "changes.diff"

        print(f"[EVAL] {pr_name}: AI={ai_report.name} GT={gt_report.name}", file=sys.stderr)

        gt_issues = extract_gt_issues(gt_report)
        ai_issues = extract_ai_findings(ai_report, diff_path if diff_path.exists() else None)

        print(f"  GT={len(gt_issues)} AI(normalized)={len(ai_issues)}", file=sys.stderr)

        match_result = match_issues(gt_issues, ai_issues, args.line_k)
        metrics = compute_metrics(gt_issues, ai_issues, match_result)

        metrics["pr_name"] = pr_name
        metrics["gt_report"] = str(gt_report.name)
        metrics["ai_report"] = str(ai_report)
        all_results.append(metrics)

        totals["gt"] += metrics["gt_total"]
        totals["ai"] += metrics["ai_total"]
        totals["matched"] += metrics["matched"]
        totals["missed"] += metrics["missed"]
        totals["fp"] += metrics["false_positive"]
        totals["line_ai"] += match_result["line_matched_ai_count"]
        totals["sem_ai"] += match_result["semantic_matched_ai_count"]

        print(f"  line_recall={metrics['line_recall']} sem_recall={metrics['semantic_recall']} "
              f"sem_precision={metrics['semantic_precision']} f1={metrics['f1']}", file=sys.stderr)

    overall_sem_recall = round(totals["matched"] / totals["gt"], 4) if totals["gt"] > 0 else None
    overall_sem_precision = round(totals["sem_ai"] / totals["ai"], 4) if totals["ai"] > 0 else None
    overall_line_recall = round(totals["line_ai"] / totals["gt"], 4) if totals["gt"] > 0 else None
    overall_line_precision = round(totals["line_ai"] / totals["ai"], 4) if totals["ai"] > 0 else None
    overall_f1 = None
    if overall_sem_recall and overall_sem_precision and (overall_sem_recall + overall_sem_precision) > 0:
        overall_f1 = round(2 * overall_sem_recall * overall_sem_precision / (overall_sem_recall + overall_sem_precision), 4)

    overall_noise_rate = round(totals["fp"] / totals["ai"], 4) if totals["ai"] > 0 else 0.0

    summary = {
        "total_prs": len(all_results),
        "total_gt_issues": totals["gt"],
        "total_ai_issues": totals["ai"],
        "total_matched": totals["matched"],
        "total_missed": totals["missed"],
        "total_false_positive": totals["fp"],
        "overall_noise_rate": overall_noise_rate,
        "overall_line_recall": overall_line_recall,
        "overall_line_precision": overall_line_precision,
        "overall_recall": overall_sem_recall,
        "overall_precision": overall_sem_precision,
        "overall_f1": overall_f1,
        "per_pr": all_results,
    }

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*70}", file=sys.stderr)
    print(f"端到端评测汇总 ({len(all_results)} PRs)", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"GT 问题总数:   {totals['gt']}", file=sys.stderr)
    print(f"AI 发现总数:   {totals['ai']} (归一化+反思过滤后)", file=sys.stderr)
    print(f"Line 匹配:     recall={overall_line_recall} precision={overall_line_precision}", file=sys.stderr)
    print(f"Semantic 匹配: recall={overall_sem_recall} precision={overall_sem_precision} f1={overall_f1}", file=sys.stderr)
    print(f"误报数:        {totals['fp']}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"详细结果: {out_path}", file=sys.stderr)

    for r in all_results:
        print(f"\n--- {r['pr_name']} ---", file=sys.stderr)
        print(f"  line_recall={r['line_recall']} sem_recall={r['semantic_recall']} "
              f"sem_precision={r['semantic_precision']} f1={r['f1']}", file=sys.stderr)
        print(f"  GT={r['gt_total']} AI={r['ai_total']} matched={r['matched']} missed={r['missed']} fp={r['false_positive']}", file=sys.stderr)
        sb = r['severity_breakdown']
        for sev in SEVERITY_LEVELS:
            s = sb[sev]
            if s['total'] > 0:
                print(f"  {sev}: {s['matched']}/{s['total']} (recall={s['recall']})", file=sys.stderr)


if __name__ == "__main__":
    main()
