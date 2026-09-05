#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""生成 reviewer_bench 评测配置：读 manifest → 组装文件检视 prompt → 输出 config JSON

与 prepare_bench_data.py 配合使用：
  1. prepare_bench_data.py（耗时）：恢复代码、生成 diff → manifest.json
  2. 本脚本（秒级）：读 manifest → 组装 prompt → 输出 batch_review config
"""
import logging
import argparse
import json
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(description="生成 reviewer_bench 评测配置")
    parser.add_argument("--manifest", default="manifest.json", help="prepare_bench_data.py 输出的 manifest")
    parser.add_argument("--template-config", default="src/batch_review/review-config.json")
    parser.add_argument("--output", default="review-config-bench.json")
    parser.add_argument("--prompt-file", default="prompts/review_prompt.md",
                        help="检视提示词文件（含 {repo_dir} {diff_file} {file_count} {code_files} 占位符）")
    return parser.parse_args()


def load_key_pool(template):
    pool = []
    for item in template.get("key_pool", []):
        if isinstance(item, str):
            pool.append({"env": item, "provider_key": "", "base_url": ""})
        elif isinstance(item, dict):
            pool.append({
                "env": item.get("env", ""),
                "provider_key": item.get("provider_key", ""),
                "base_url": item.get("base_url", ""),
            })
    return pool


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(args.template_config, "r", encoding="utf-8") as f:
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
        logging.error("错误：template-config 的 key_pool 为空或解析失败")
        sys.exit(1)

    review = template.get("review", {})
    execution = template.get("execution", {})
    model = review.get("model", "")

    entries = manifest.get("entries", [])

    from collections import Counter
    pr_counts = Counter((e["repo"], e["pr_number"]) for e in entries)

    targets = []
    skipped = 0
    for idx, entry in enumerate(entries):
        code_files = entry.get("code_files", [])
        if not code_files:
            logging.info(f"跳过 {entry['repo']} PR{entry['pr_number']}（无代码文件）")
            skipped += 1
            continue

        if pr_counts[(entry["repo"], entry["pr_number"])] > 1:
            name = f"{entry['repo']}-pr-{entry['pr_number']}-{entry['commit_hash'][:7]}"
        else:
            name = f"{entry['repo']}-pr-{entry['pr_number']}"
        files_text = "\n".join(f"  - {f}" for f in code_files)
        prompt = prompt_template.format(
            repo_dir=entry["repo_dir"],
            diff_file=entry["diff_file"],
            file_count=len(code_files),
            code_files=files_text,
        )
        key = key_pool[(idx - skipped) % len(key_pool)]
        targets.append({
            "name": name,
            "prompt": prompt,
            "api_key_env": key["env"],
            "provider_key": key["provider_key"],
            "base_url": key["base_url"],
            "model": model,
            "path": entry["repo_dir"],
        })

    if not targets:
        logging.error("错误：无可用 target（全部无代码文件或 manifest 为空）")
        sys.exit(1)

    out_config = {
        "engine": template.get("engine", "opencode"),
        "review": {
            "model": model,
            "timeout_sec": review.get("timeout_sec"),
            "idle_timeout_sec": review.get("idle_timeout_sec"),
            "base_url": review.get("base_url", ""),
        },
        "execution": {
            "max_parallel": execution.get("max_parallel"),
            "retry_on_failure": execution.get("retry_on_failure"),
            "retry_delay_sec": execution.get("retry_delay_sec"),
            "output_dir": execution.get("output_dir"),
        },
        "targets": targets,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_config, f, ensure_ascii=False, indent=2)

    logging.info(f"manifest 条目数：{len(entries)}")
    logging.info(f"生成 target 数：{len(targets)}（跳过 {skipped} 无代码文件）")
    logging.info(f"key 池大小：{len(key_pool)}")
    for i, key in enumerate(key_pool):
        count = sum(1 for t in targets if t["api_key_env"] == key["env"])
        logging.info(f"  key[{i}] {key['env']} → {count} target")
    logging.info(f"输出路径：{args.output}")


if __name__ == "__main__":
    main()