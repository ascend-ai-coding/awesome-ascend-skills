#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""生成评测配置：扫描检视报告 + 匹配 ground-truth → 组装评测 prompt → 输出 config

与 batch_review 配合使用：复用 batch_review 的并行调度，每个评测 target 对比一份
AI 检视报告与 ground-truth 报告，输出召回率/精确率。

用法:
    # 指定 ground-truth 目录，自动找最新检视报告
    python scripts/gen_eval_config.py --reports-dir benchmark_tasks/top20_transformer_filtered

    # 显式指定检视报告目录
    python scripts/gen_eval_config.py --run-dir reports/run_xxx --reports-dir benchmark_tasks/top20_transformer_filtered
"""
import logging
import argparse
import json
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

TARGET_NAME_RE = re.compile(r'^(?P<repo>[a-z]+-[a-z]+)-pr-(?P<pr>\d+)(?:-(?P<commit7>[0-9a-f]{7}))?$')


def find_latest_run_dir(reports_root: Path) -> Path:
    """自动找 reports/ 下最新的 run_* 目录"""
    run_dirs = sorted(reports_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0] if run_dirs else None


def parse_args():
    parser = argparse.ArgumentParser(description="生成评测配置")
    parser.add_argument("--run-dir", default=None, help="batch_review 的 run 目录（默认自动找最新）")
    parser.add_argument("--manifest", default="manifest.json", help="prepare_bench_data.py 输出的 manifest")
    parser.add_argument("--reports-dir", default="benchmark_tasks/top20_transformer_filtered",
                        help="ground-truth 报告目录（benchmark_tasks/ 下的子路径）")
    parser.add_argument("--template-config", default="src/batch_review/review-config.json")
    parser.add_argument("--output", default="eval-config.json")
    parser.add_argument("--prompt-file", default="prompts/eval_prompt.md",
                        help="评测提示词文件（含 {ai_report} {gt_report} 占位符）")
    return parser.parse_args()


def load_key_pool(template):
    pool = []
    for item in template.get("key_pool", []):
        if isinstance(item, str):
            pool.append({"env": item, "provider_key": ""})
        elif isinstance(item, dict):
            pool.append({"env": item.get("env", ""), "provider_key": item.get("provider_key", "")})
    return pool


def find_gt_report(reports_dir: Path, repo: str, pr_number: str, commit_hash: str) -> Path:
    """通过 repo + pr + commit 在 ground-truth 目录查找报告"""
    pr_num = int(pr_number)
    patterns = [
        f"*{repo}_pr_{pr_num}_{commit_hash}.md",
        f"*{repo}_pr_{pr_num:04d}_{commit_hash}.md",
        f"*{repo}_pr_{pr_num}_*.md",
        f"*{repo}_pr_{pr_num:04d}_*.md",
    ]
    for pattern in patterns:
        matches = list(reports_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent

    reports_root = base_dir / "reports"
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = base_dir / run_dir
    else:
        run_dir = find_latest_run_dir(reports_root)
        if not run_dir:
            logging.error("错误：未找到 reports/run_* 目录，请先运行检视或用 --run-dir 指定")
            sys.exit(1)
        logging.info(f"自动选择检视报告目录: {run_dir}")
    if not run_dir.exists():
        logging.error(f"错误：run 目录不存在: {run_dir}")
        sys.exit(1)

    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = base_dir / reports_dir
    if not reports_dir.exists():
        logging.error(f"错误：ground-truth 目录不存在: {reports_dir}")
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(base_dir / args.template_config, "r", encoding="utf-8") as f:
        template = json.load(f)

    prompt_file = Path(args.prompt_file)
    if not prompt_file.is_absolute():
        prompt_file = base_dir / prompt_file
    if not prompt_file.exists():
        logging.error(f"错误：提示词文件不存在: {prompt_file}")
        sys.exit(1)
    prompt_template = prompt_file.read_text(encoding="utf-8")

    key_pool = load_key_pool(template)
    if not key_pool:
        logging.error("错误：key_pool 为空")
        sys.exit(1)

    manifest_map = {}
    for entry in manifest.get("entries", []):
        manifest_map[(entry["repo"], entry["pr_number"])] = entry

    ai_reports = []
    for target_dir in sorted(run_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        report_file = target_dir / "review_report.md"
        if not report_file.exists():
            continue

        target_name = target_dir.name
        m = TARGET_NAME_RE.match(target_name)
        if not m:
            logging.info(f"跳过 {target_name}：无法解析 target name")
            continue

        repo = m.group("repo")
        pr_number = m.group("pr")
        commit7 = m.group("commit7")

        entry = manifest_map.get((repo, pr_number))
        if not entry:
            logging.info(f"跳过 {target_name}：manifest 中无匹配")
            continue

        commit_hash = entry["commit_hash"]
        if commit7 and not commit_hash.startswith(commit7):
            logging.info(f"跳过 {target_name}：commit 不匹配 ({commit7} vs {commit_hash[:7]})")
            continue

        gt_report = find_gt_report(reports_dir, repo, pr_number, commit_hash)
        if not gt_report:
            logging.info(f"跳过 {target_name}：未找到 ground-truth 报告")
            continue

        ai_reports.append({
            "target_name": target_name,
            "ai_report": str(report_file.absolute()),
            "gt_report": str(gt_report.absolute()),
        })

    if not ai_reports:
        logging.error("错误：未找到可评测的报告对")
        sys.exit(1)

    review = template.get("review", {})
    execution = template.get("execution", {})
    model = review.get("model", "")

    targets = []
    for idx, pair in enumerate(ai_reports):
        prompt = prompt_template.replace("{ai_report}", pair["ai_report"]).replace("{gt_report}", pair["gt_report"])
        key = key_pool[idx % len(key_pool)]
        targets.append({
            "name": f"eval-{pair['target_name']}",
            "prompt": prompt,
            "api_key_env": key["env"],
            "provider_key": key["provider_key"],
            "model": model,
            "extra_files": [pair["ai_report"], pair["gt_report"]],
        })

    out_config = {
        "review": {
            "model": model,
            "timeout_sec": 600,
            "idle_timeout_sec": 300,
        },
        "execution": {
            "max_parallel": execution.get("max_parallel", 3),
            "retry_on_failure": 0,
            "retry_delay_sec": 30,
            "output_dir": "./eval-reports",
        },
        "targets": targets,
    }

    output_path = base_dir / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_config, f, ensure_ascii=False, indent=2)

    logging.info(f"检视报告数：{len(ai_reports)}")
    logging.info(f"生成 target 数：{len(targets)}")
    logging.info(f"key 池大小：{len(key_pool)}")
    for i, key in enumerate(key_pool):
        count = sum(1 for t in targets if t["api_key_env"] == key["env"])
        logging.info(f"  key[{i}] {key['env']} → {count} target")
    logging.info(f"输出路径：{output_path}")


if __name__ == "__main__":
    main()
