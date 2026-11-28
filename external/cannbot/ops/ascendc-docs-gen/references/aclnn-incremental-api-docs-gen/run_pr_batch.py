#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Batch runner: parse gitcode_pr.txt, checkout merge_base & head (feat) for each PR
via gitcode_pr_file_get.py, then invoke src/run_all_cli.py to compare the two.

txt 格式 (每行, 空格分隔):
    <pr_url>  <op_path>

    - 第1列: GitCode PR 链接 (含 pull id)
    - 第2列: 算子相对路径 (如 mc2/moe_distribute_dispatch_v2)

参数映射到 run_all_cli.py:
    --original_op_repo    = merge_base checkout 目录
    --modified_op_repo    = head (feat) checkout 目录
    --original_op_path    = 第2列 op_path
    --modified_op_path    = 第2列 op_path
    --output_data_path    = <snapshot_base>/pr_<pull_id>/<op_path_safe>

--md-stem 不再传入: 下游 run_all_cli.py 在省略 --md-stem 时会自动选择该算子
所有可用的 md stem (见 _select_md_stems)。默认 full 模式 (不传 --mode / --validate-only)。

质量指标 (Accuracy): 每个任务完成后从
    <output_data_path>/<op>/<md_stem>/results/detailed_results.json
读取各 doc_part 的 validation 评分 (LLM 5 维加权准确率, 计算见
src/generation_validation/validator/consistency.py), 聚合进批次报告 (JSON+MD)。

以 '#' 开头的行忽略。

CLI:
    python run_pr_batch.py [--dashboard]
        --dashboard    透传给 run_all_cli.py, 启用并行 doc_part 处理的实时面板 (默认关闭)。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TXT_FILE = PROJECT_ROOT / "gitcode_pr.txt"
CACHE_DIR = PROJECT_ROOT / ".temp_git"  # gitcode checkout 缓存
LOG_DIR = PROJECT_ROOT / "data" / "outputs" / "pr_batch_logs"
RUN_ALL_CLI = PROJECT_ROOT / "src" / "run_all_cli.py"
GITCODE_PR_GET = PROJECT_ROOT / "gitcode_pr_file_get.py"
SNAPSHOT_BASE = PROJECT_ROOT / "data" / "outputs" / "snapshot"
TIMING_OUTPUT_DIR = LOG_DIR  # 批次报告与日志统一目录 (data/outputs/pr_batch_logs), 与 WORKFLOW.md 附录一致

# 匹配 gitcode PR URL, 提取 owner/repo/pr_number
PR_URL_RE = re.compile(r"gitcode\.com/([^/]+)/([^/]+)/pull/(\d+)")

# ANSI 转义序列 (含 console.clear() 的 \x1b[2J\x1b[H), 用于 strip 后干净输出到终端/日志
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[A-Za-z]"  # CSI (含 2J, H, 颜色等)
    r"|\x1b\][^\x07]*\x07"  # OSC
    r"|\x1b[@-Z\\-_]"  # 单字符转义
)


def parse_pr_url(pr_url: str) -> tuple[str, str, str]:
    """从 PR URL 提取 (owner, repo, pr_number)。"""
    m = PR_URL_RE.search(pr_url)
    if not m:
        raise ValueError(f"无法解析 PR 链接: {pr_url}")
    return m.group(1), m.group(2), m.group(3)


def parse_tasks(txt_path: Path) -> list[dict]:
    """解析 gitcode_pr.txt, 返回任务列表。跳过 '#' 注释行与空行。

    每行 2 列 (空格分隔): <pr_url>  <op_path>。多余列静默忽略 (兼容遗留 token)。
    """
    tasks = []
    for lineno, raw in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            print(f"[WARN] line {lineno}: 列数不足(需2列), 跳过: {raw}")
            continue
        pr_url, op_path = parts[0], parts[1]
        tasks.append(
            {
                "lineno": lineno,
                "pr_url": pr_url,
                "op_path": op_path,
            }
        )
    return tasks


def checkout_pr(pr_url: str) -> tuple[Path, Path]:
    """调用 gitcode_pr_file_get.py checkout merge_base & head。

    返回 (merge_base_dir, head_dir)。同一 PR 在脚本运行内只 checkout 一次(由调用方缓存)。
    """
    owner, repo, pr_number = parse_pr_url(pr_url)
    output_dir = CACHE_DIR / f"{owner}_{repo}_{pr_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_merge = sorted(output_dir.glob("merge_base_*"))
    existing_head = sorted(output_dir.glob("head_*"))
    if existing_merge and existing_head:
        print("  -> 使用缓存, 跳过 checkout")
        return existing_merge[-1], existing_head[-1]

    # 清理旧 checkout 目录, 避免 PR 更新后 glob 拿到 stale 目录
    for pat in ("merge_base_*", "head_*"):
        for d in output_dir.glob(pat):
            shutil.rmtree(d)

    print("  -> gitcode_pr_file_get: checkout merge_base & head ...")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(GITCODE_PR_GET),
                pr_url,
                "-t",
                "merge_base",
                "head",
                "-o",
                str(output_dir),
            ],
            timeout=7200,
        )
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] checkout timeout: gitcode_pr_file_get 超过 7200s 未完成 ({pr_url}), 已终止")
        sys.exit(1)
    if proc.returncode != 0:
        raise RuntimeError(f"gitcode_pr_file_get 失败 (exit={proc.returncode}) for {pr_url}")

    merge_dirs = sorted(output_dir.glob("merge_base_*"))
    head_dirs = sorted(output_dir.glob("head_*"))
    if not merge_dirs:
        raise RuntimeError(f"未找到 merge_base checkout 目录 for {pr_url}")
    if not head_dirs:
        raise RuntimeError(f"未找到 head checkout 目录 for {pr_url}")
    return merge_dirs[-1], head_dirs[-1]


def build_output_data_path(pull_id: str, op_path: str) -> str:
    """output_data_path = <snapshot_base>/pr_<pull_id>/<op_path_safe>"""
    safe_op = op_path.replace("/", "_")
    return str(SNAPSHOT_BASE / f"pr_{pull_id}" / safe_op)


def load_timing_json(output_data_path: str) -> dict:
    """从 run_all_cli.py 导出的 timing.json 读取耗时数据。

    run_all_cli.py 在每个任务完成后将内部计时写入 <output_data_path>/timing.json,
    包含 total_elapsed / error / stems[] (每个 stem 的各步骤耗时)。
    文件不存在或解析失败时返回空 dict。
    """
    timing_path = Path(output_data_path) / "timing.json"
    if not timing_path.exists():
        return {}
    try:
        return json.loads(timing_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt_acc(v) -> str:
    """格式化 Accuracy 数值: None/非数值显示 '-'。"""
    return f"{v:.2f}" if isinstance(v, (int, float)) else "-"


def load_accuracy_results(output_data_path: str, stems: list[dict] | None = None) -> dict:
    """从 run_all_cli.py 导出的 detailed_results.json 读取质量指标 (Accuracy)。

    run_all_cli.py 对每个 md_stem 导出:
        <output_data_path>/<op>/<md_stem>/results/detailed_results.json
    结构: {doc_part: {.., "validation": [{"accuracy_rate", "passed",
               "consistency_passed", "metric_passed", ...}, ...]}}

    Accuracy = LLM 对生成文档的 5 维加权准确率 (factual_accuracy×0.3 +
    formula_correctness×0.2 + numerical_accuracy×0.2 + description_consistency×0.15 +
    logical_coherence×0.15), 由 consistency.py 计算并随 validation 落盘。

    聚合口径:
        - 按 validation 项 (sub-part 级) 聚合: accuracy 为均值, accuracy_min 为最低分
        - "example" doc_part 的 accuracy_rate 是 0/1 硬编码 (不走 LLM 评分),
          不计入均值, 单独统计 example_passed/example_total
        - evidence_conflict 且无 validation 项的 doc_part 计入 conflict_parts
        - llm_passed 与 accuracy 相互独立: metric_passed / 确定性巡检可单独将
          passed 置 False (见 generation_validation/config.py
          CONSISTENCY_PASS_THRESHOLD 注释)

    stems 传入 timing.json 的 stems[] 时只统计本次运行涉及的 md_stem
    (避免读到历史遗留目录); 为 None (timing.json 缺失) 时统计全部,
    为空列表 (本次运行无完成的 stem) 时统计结果为空。
    无有效数据时返回 {"available": False, ...}。
    """
    base = Path(output_data_path)
    stem_filter = None
    if stems is not None:
        stem_filter = {s.get("stem") for s in stems if s.get("stem")}

    task: dict = {
        "available": False,
        "stems": {},
        "accuracy": None,  # LLM 评分项 accuracy_rate 均值
        "accuracy_min": None,  # LLM 评分项最低分
        "llm_scored": 0,  # 计入均值的 validation 项数
        "llm_passed": 0,  # 其中 passed=True 的项数
        "example_total": 0,
        "example_passed": 0,
        "conflict_parts": 0,  # evidence_conflict 且无评分的 doc_part 数
    }

    all_llm_scores: list[float] = []
    for rf in sorted(base.glob("*/*/results/detailed_results.json")):
        stem = rf.parent.parent.name
        if stem_filter is not None and stem not in stem_filter:
            continue
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        stem_acc: dict = {
            "accuracy": None,
            "accuracy_min": None,
            "llm_scored": 0,
            "llm_passed": 0,
            "example_total": 0,
            "example_passed": 0,
            "conflict_parts": 0,
            "doc_parts": {},
        }
        stem_llm_scores: list[float] = []
        for part, payload in data.items():
            if not isinstance(payload, dict):
                continue
            val = payload.get("validation") or []
            entries = [v for v in val if isinstance(v, dict)]
            scores = [float(v["accuracy_rate"]) for v in entries if isinstance(v.get("accuracy_rate"), (int, float))]
            part_passed = sum(1 for v in entries if v.get("passed") is True)
            conflict = bool(payload.get("evidence_conflict")) and not entries

            stem_acc["doc_parts"][part] = {
                "entries": len(entries),
                "accuracy": round(sum(scores) / len(scores), 2) if scores else None,
                "passed": part_passed,
                "conflict": conflict,
            }
            if part == "example":
                stem_acc["example_total"] += len(entries)
                stem_acc["example_passed"] += part_passed
            else:
                stem_llm_scores.extend(scores)
                stem_acc["llm_scored"] += len(scores)
                stem_acc["llm_passed"] += part_passed
            if conflict:
                stem_acc["conflict_parts"] += 1

        if stem_llm_scores:
            stem_acc["accuracy"] = round(sum(stem_llm_scores) / len(stem_llm_scores), 2)
            stem_acc["accuracy_min"] = min(stem_llm_scores)
        task["stems"][stem] = stem_acc

        all_llm_scores.extend(stem_llm_scores)
        task["llm_scored"] += stem_acc["llm_scored"]
        task["llm_passed"] += stem_acc["llm_passed"]
        task["example_total"] += stem_acc["example_total"]
        task["example_passed"] += stem_acc["example_passed"]
        task["conflict_parts"] += stem_acc["conflict_parts"]

    if all_llm_scores:
        task["accuracy"] = round(sum(all_llm_scores) / len(all_llm_scores), 2)
        task["accuracy_min"] = min(all_llm_scores)
    task["available"] = bool(task["stems"])
    return task


def save_timing_report(
    timings: list[dict],
    pr_checkout_time: dict[str, float],
    output_dir: Path,
) -> tuple[Path, Path]:
    """汇总每个 PR/任务的耗时与质量指标 (Accuracy), 输出 JSON + Markdown 到 output_dir。
    返回 (json_path, md_path)。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 按 PR(pull_id) 聚合
    pr_summary: dict[str, dict] = {}
    for t in timings:
        pid = t["pull_id"]
        s = pr_summary.setdefault(
            pid,
            {
                "pull_id": pid,
                "pr_url": t["pr_url"],
                "op_path": t["op_path"],
                "runs": [],
                "checkout_time": pr_checkout_time.get(t["pr_url"], 0.0),
                "pipeline_time": 0.0,
                "total_time": 0.0,
                "success": 0,
                "failed": 0,
                "accuracy": None,
            },
        )
        s["runs"].append(
            {
                "op_path": t["op_path"],
                "status": t["status"],
                "accuracy": (t.get("accuracy") or {}).get("accuracy"),
                "pipeline_time": t["pipeline_time"],
                "wall_time": t.get("wall_time", 0.0),
                "total_time": t["total_time"],
                "modules": t["modules"],
                "stems": t.get("stems", []),
                "timing_error": t.get("timing_error"),
                "log_file": t["log_file"],
            }
        )
        s["pipeline_time"] += t["pipeline_time"]
        s["total_time"] = s["checkout_time"] + s["pipeline_time"]
        if t["status"] == "success":
            s["success"] += 1
        else:
            s["failed"] += 1

    # 每 PR 的 Accuracy = 各任务 accuracy 的算术平均
    for s in pr_summary.values():
        acc_vals = [r["accuracy"] for r in s["runs"] if r.get("accuracy") is not None]
        s["accuracy"] = round(sum(acc_vals) / len(acc_vals), 2) if acc_vals else None

    # 批次整体 Accuracy (有评分任务的均值)
    acc_all = [(t.get("accuracy") or {}).get("accuracy") for t in timings]
    acc_all = [a for a in acc_all if a is not None]
    accuracy_summary = {
        "tasks_with_accuracy": len(acc_all),
        "overall_accuracy": round(sum(acc_all) / len(acc_all), 2) if acc_all else None,
    }
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_tasks": len(timings),
        "total_prs": len(pr_summary),
        "accuracy_summary": accuracy_summary,
        "pr_checkout_time": pr_checkout_time,
        "pr_summary": list(pr_summary.values()),
        "tasks": timings,
    }
    json_path = output_dir / f"pr_batch_timing_{ts}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md: list[str] = []
    md.append("# PR 批次统计报告")
    md.append("")
    md.append(f"- 生成时间: {report['generated_at']}")
    md.append(f"- 任务总数: {report['total_tasks']}")
    md.append(f"- PR 数: {report['total_prs']}")
    if accuracy_summary["overall_accuracy"] is not None:
        md.append(
            f"- 整体 Accuracy: {accuracy_summary['overall_accuracy']:.2f} "
            f"({accuracy_summary['tasks_with_accuracy']}/{report['total_tasks']} 个任务有评分)"
        )
    md.append("")
    md.append("## 按 PR 汇总")
    md.append("")
    md.append("| PR | op_path | checkout(s) | pipeline(s) | total(s) | 成功/失败 | Accuracy |")
    md.append("|---|---|---|---|---|---|---|")
    for s in pr_summary.values():
        md.append(
            f"| !{s['pull_id']} | {s['op_path']} | {s['checkout_time']:.2f} | "
            f"{s['pipeline_time']:.2f} | {s['total_time']:.2f} | {s['success']}/{s['failed']} | "
            f"{_fmt_acc(s.get('accuracy'))} |"
        )
    md.append("")
    md.append("## 按任务明细")
    md.append("")
    md.append("| Line | PR | op_path | 状态 | Accuracy | checkout(s) | pipeline(s) | wall(s) | total(s) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for t in timings:
        md.append(
            f"| {t['lineno']} | !{t['pull_id']} | {t['op_path']} | {t['status']} | "
            f"{_fmt_acc((t.get('accuracy') or {}).get('accuracy'))} | "
            f"{t['checkout_time']:.2f} | {t['pipeline_time']:.2f} | "
            f"{t.get('wall_time', 0.0):.2f} | {t['total_time']:.2f} |"
        )
    md.append("")
    md.append("## 质量指标 (Accuracy)")
    md.append("")
    md.append("Accuracy = LLM 对生成文档的 5 维加权准确率 (factual_accuracy×0.3 +")
    md.append("formula_correctness×0.2 + numerical_accuracy×0.2 + description_consistency×0.15 +")
    md.append("logical_coherence×0.15, 计算见 src/generation_validation/validator/consistency.py),")
    md.append("读取自各任务的 results/detailed_results.json, 聚合口径:")
    md.append("")
    md.append("- 按 doc_part 内 validation 项 (sub-part 级) 取均值; `min` 为最低分, 暴露短板 part")
    md.append("- `example` 的 accuracy 是 0/1 硬编码 (不走 LLM 评分), 不计入均值, 单列")
    md.append("- `conflict` = evidence_conflict 且无评分的 doc_part 数 (证据冲突提前终止)")
    md.append("- Accuracy 与 passed 相互独立: metric_passed / 确定性巡检可单独将 passed 置 False")
    md.append("- 失败/未生成 detailed_results.json 的任务显示 `-`")
    md.append("")
    md.append("### 按任务汇总")
    md.append("")
    md.append("| Line | PR | op_path | Accuracy | min | LLM评分/通过 | example(通过/总) | conflict |")
    md.append("|---|---|---|---|---|---|---|---|")
    for t in timings:
        acc = t.get("accuracy") or {}
        if not acc.get("available"):
            md.append(f"| {t['lineno']} | !{t['pull_id']} | {t['op_path']} | - | - | - | - | - |")
        else:
            md.append(
                f"| {t['lineno']} | !{t['pull_id']} | {t['op_path']} | "
                f"{_fmt_acc(acc.get('accuracy'))} | {_fmt_acc(acc.get('accuracy_min'))} | "
                f"{acc.get('llm_scored', 0)}/{acc.get('llm_passed', 0)} | "
                f"{acc.get('example_passed', 0)}/{acc.get('example_total', 0)} | "
                f"{acc.get('conflict_parts', 0)} |"
            )
    md.append("")
    md.append("### 按 Stem × doc_part 明细")
    md.append("")
    has_part_rows = any((t.get("accuracy") or {}).get("stems") for t in timings)
    if has_part_rows:
        md.append("| Line | PR | Stem | doc_part | 项数 | Accuracy | passed | conflict |")
        md.append("|---|---|---|---|---|---|---|---|")
        for t in timings:
            acc = t.get("accuracy") or {}
            for stem, s_acc in (acc.get("stems") or {}).items():
                for part, p in (s_acc.get("doc_parts") or {}).items():
                    md.append(
                        f"| {t['lineno']} | !{t['pull_id']} | {stem} | {part} | "
                        f"{p.get('entries', 0)} | {_fmt_acc(p.get('accuracy'))} | "
                        f"{p.get('passed', 0)} | {'是' if p.get('conflict') else '-'} |"
                    )
    else:
        md.append("_(无 doc_part 级质量数据)_")
    md.append("")
    md.append("## 各模块耗时明细 (来自 timing.json, 跨 stem 聚合)")
    md.append("")
    all_modules: list[str] = []
    seen: set[str] = set()
    for t in timings:
        for m in t["modules"]:
            if m not in seen:
                seen.add(m)
                all_modules.append(m)
    if all_modules:
        header = ["Line", "PR"] + [f"{m}(s)" for m in all_modules] + ["pipeline(s)"]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for t in timings:
            row = [str(t["lineno"]), f"!{t['pull_id']}"]
            for m in all_modules:
                v = t["modules"].get(m)
                row.append(f"{v:.2f}" if v is not None else "-")
            row.append(f"{t['pipeline_time']:.2f}")
            md.append("| " + " | ".join(row) + " |")
    else:
        md.append("_(未解析到模块耗时, 可能 timing.json 缺失或无 stems 数据)_")
    md.append("")
    md.append("## Per-Stem 耗时明细")
    md.append("")
    all_step_names: list[str] = []
    seen_steps: set[str] = set()
    for t in timings:
        for s in t.get("stems", []):
            for step_name in s.get("steps", {}):
                if step_name not in seen_steps:
                    seen_steps.add(step_name)
                    all_step_names.append(step_name)
    if all_step_names:
        header = ["Line", "PR", "Stem"] + [f"{m}(s)" for m in all_step_names] + ["stem_total(s)", "Accuracy"]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for t in timings:
            for s in t.get("stems", []):
                row = [str(t["lineno"]), f"!{t['pull_id']}", s.get("stem", "?")]
                steps = s.get("steps", {})
                for m in all_step_names:
                    v = steps.get(m)
                    row.append(f"{v:.2f}" if v is not None else "-")
                stem_total = sum(steps.values()) if steps else 0.0
                row.append(f"{stem_total:.2f}")
                stem_acc = (t.get("accuracy") or {}).get("stems", {}).get(s.get("stem"), {})
                row.append(_fmt_acc(stem_acc.get("accuracy")))
                md.append("| " + " | ".join(row) + " |")
    else:
        md.append("_(无 per-stem 数据)_")
    md.append("")

    md_path = output_dir / f"pr_batch_timing_{ts}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path


def run_pipeline(
    merge_base_dir: Path,
    head_dir: Path,
    op_path: str,
    pull_id: str,
    *,
    dashboard: bool = False,
) -> tuple[int, Path]:
    """调用 src/run_all_cli.py, strip ANSI 后实时输出并写日志。

    dashboard=True 时透传 --dashboard 给 run_all_cli.py (启用并行 doc_part 实时面板)。
    返回 (exit_code, log_file_path)。
    """
    output_data_path = build_output_data_path(pull_id, op_path)
    safe_op = op_path.replace("/", "_")
    log_file = LOG_DIR / f"pr_{pull_id}_{safe_op}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RUN_ALL_CLI),
        "--original_op_repo",
        str(merge_base_dir),
        "--original_op_path",
        op_path,
        "--modified_op_repo",
        str(head_dir),
        "--modified_op_path",
        op_path,
        "--output_data_path",
        output_data_path,
    ]
    if dashboard:
        cmd.append("--dashboard")
    print(f"  output_data_path: {output_data_path}")
    print(f"  log:              {log_file}")
    print("  --- live output ---")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        bufsize=1,
    )
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(_ANSI_RE.sub("", line))
        try:
            proc.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(
                f"  [ERROR] pipeline timeout: run_all_cli 子进程超过 7200s 未结束 "
                f"(PR !{pull_id}, op_path={op_path}), 已强制终止"
            )
            sys.exit(1)
    return proc.returncode, log_file


def main():
    parser = argparse.ArgumentParser(
        description="批次执行 PR 文档生成: 解析 gitcode_pr.txt, checkout 后逐个调用 run_all_cli.py",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="透传给 run_all_cli.py, 启用并行 doc_part 处理的实时面板 (默认关闭)",
    )
    args = parser.parse_args()

    for p, name in [
        (TXT_FILE, "gitcode_pr.txt"),
        (RUN_ALL_CLI, "src/run_all_cli.py"),
        (GITCODE_PR_GET, "gitcode_pr_file_get.py"),
    ]:
        if not p.exists():
            print(f"[ERROR] 找不到 {name}: {p}")
            sys.exit(1)

    tasks = parse_tasks(TXT_FILE)
    print(f"共解析 {len(tasks)} 个任务行")
    if not tasks:
        print("无任务, 退出")
        return

    pr_cache: dict[str, tuple[Path, Path]] = {}  # pr_url -> (merge_base_dir, head_dir)
    pr_checkout_time: dict[str, float] = {}  # pr_url -> 首次 checkout 耗时(秒)
    timings: list[dict] = []  # 每个任务的耗时记录
    total = 0
    success = 0
    failed: list[tuple] = []

    for task in tasks:
        lineno = task["lineno"]
        pr_url = task["pr_url"]
        op_path = task["op_path"]
        _, _, pull_id = parse_pr_url(pr_url)

        print(f"\n{'=' * 70}")
        print(f"[Line {lineno}] PR !{pull_id}  op_path={op_path}")

        # 同一 PR 只 checkout 一次
        if pr_url not in pr_cache:
            ck_start = time.time()
            try:
                pr_cache[pr_url] = checkout_pr(pr_url)
            except Exception as e:
                pr_checkout_time[pr_url] = time.time() - ck_start
                print(f"  [ERROR] checkout 失败: {e}")
                total += 1
                failed.append((lineno, pull_id, op_path, f"checkout failed: {e}"))
                timings.append(
                    {
                        "lineno": lineno,
                        "pr_url": pr_url,
                        "pull_id": pull_id,
                        "op_path": op_path,
                        "status": f"failed: checkout failed: {e}",
                        "accuracy": None,
                        "checkout_time": pr_checkout_time[pr_url],
                        "pipeline_time": 0.0,
                        "wall_time": 0.0,
                        "total_time": pr_checkout_time[pr_url],
                        "modules": {},
                        "stems": [],
                        "timing_error": "checkout failed",
                        "log_file": "",
                    }
                )
                continue
            pr_checkout_time[pr_url] = time.time() - ck_start
        merge_base_dir, head_dir = pr_cache[pr_url]
        print(f"  merge_base: {merge_base_dir}")
        print(f"  head(feat): {head_dir}")

        total += 1
        print(f"\n  --- op_path={op_path} ---")
        ck_time = pr_checkout_time.get(pr_url, 0.0)
        pipe_start = time.time()
        try:
            rc, log_file = run_pipeline(merge_base_dir, head_dir, op_path, pull_id, dashboard=args.dashboard)
        except Exception as e:
            pipeline_secs = time.time() - pipe_start
            output_data_path_str = build_output_data_path(pull_id, op_path)
            timing = load_timing_json(output_data_path_str)
            accuracy = load_accuracy_results(output_data_path_str, timing.get("stems"))
            modules = {}
            for s in timing.get("stems", []):
                for k, v in s.get("steps", {}).items():
                    modules[k] = modules.get(k, 0.0) + v
            failed.append((lineno, pull_id, op_path, str(e)))
            print(f"  -> ERROR: {e}")
            timings.append(
                {
                    "lineno": lineno,
                    "pr_url": pr_url,
                    "pull_id": pull_id,
                    "op_path": op_path,
                    "status": f"failed: {e}",
                    "accuracy": accuracy,
                    "checkout_time": ck_time,
                    "pipeline_time": pipeline_secs,
                    "wall_time": pipeline_secs,
                    "total_time": ck_time + pipeline_secs,
                    "modules": modules,
                    "stems": timing.get("stems", []),
                    "timing_error": str(e),
                    "log_file": "",
                }
            )
            continue
        pipeline_secs = time.time() - pipe_start
        output_data_path_str = build_output_data_path(pull_id, op_path)
        timing = load_timing_json(output_data_path_str)
        accuracy = load_accuracy_results(output_data_path_str, timing.get("stems"))
        modules: dict[str, float] = {}
        for s in timing.get("stems", []):
            for k, v in s.get("steps", {}).items():
                modules[k] = modules.get(k, 0.0) + v
        stems_detail = timing.get("stems", [])
        total_elapsed = timing.get("total_elapsed", 0.0)
        timing_error = timing.get("error")
        if rc != 0:
            failed.append((lineno, pull_id, op_path, f"exit={rc}"))
            print(f"  -> FAILED (exit={rc})")
            status = f"failed: exit={rc}"
        elif not timing or timing_error:
            # rc==0 但 timing.json 缺失或带 error → run_all_cli 内部静默失败, 不能计为 SUCCESS
            reason = "timing.json missing" if not timing else f"timing error: {timing_error}"
            failed.append((lineno, pull_id, op_path, reason))
            print(f"  -> FAILED (silent: {reason})")
            status = f"failed: silent ({reason})"
        else:
            success += 1
            print("  -> SUCCESS")
            status = "success"
        timings.append(
            {
                "lineno": lineno,
                "pr_url": pr_url,
                "pull_id": pull_id,
                "op_path": op_path,
                "status": status,
                "accuracy": accuracy,
                "checkout_time": ck_time,
                "pipeline_time": total_elapsed if total_elapsed else pipeline_secs,
                "wall_time": pipeline_secs,
                "total_time": ck_time + pipeline_secs,
                "modules": modules,
                "stems": stems_detail,
                "timing_error": timing_error,
                "log_file": str(log_file),
            }
        )

    print(f"\n{'=' * 70}")
    print(f"批次结束: 共 {total} 个任务, 成功 {success}, 失败 {len(failed)}")
    for f in failed:
        print(f"  FAILED: line {f[0]} PR !{f[1]} {f[2]} -> {f[3]}")

    json_path, md_path = save_timing_report(timings, pr_checkout_time, TIMING_OUTPUT_DIR)
    print("\n批次统计(耗时+质量)已输出:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
