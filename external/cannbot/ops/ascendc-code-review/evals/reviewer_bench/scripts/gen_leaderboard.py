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
"""聚合 eval 结果 → leaderboard.json（供前端 index.html 加载）

扫描 eval-reports/run_xxx/ 下的 eval_result.md + batch_result.json，
解析每个 eval target 的 TP/FP/FN/召回率/检出率/耗时，
输出 docs/reports/leaderboard.json。

用法:
    python scripts/gen_leaderboard.py                          # 自动找最新 eval run
    python scripts/gen_leaderboard.py --eval-dir eval-reports/run_xxx
    python scripts/gen_leaderboard.py --review-dir reports/run_xxx  # 同时聚合检视阶段耗时
"""
import logging
import argparse
import json
import os
import re
import sys
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

TARGET_NAME_RE = re.compile(
    r'^eval-(?P<repo>[a-z]+-[a-z]+)-pr-(?P<pr>\d+)(?:-(?P<commit7>[0-9a-f]{7}))?$'
)

RECALL_RE = re.compile(r'召回率.*?([0-9.]+)\s*%', re.IGNORECASE | re.DOTALL)
PRECISION_RE = re.compile(r'精确率.*?([0-9.]+)\s*%', re.IGNORECASE | re.DOTALL)
TP_RE = re.compile(r'\bTP\b[^0-9\n]{0,10}(\d+)', re.IGNORECASE)
FP_RE = re.compile(r'\bFP\b[^0-9\n]{0,10}(\d+)', re.IGNORECASE)
FN_RE = re.compile(r'\bFN\b[^0-9\n]{0,10}(\d+)', re.IGNORECASE)
DETECT_RE = re.compile(r'检出率.*?([0-9.]+)\s*%', re.IGNORECASE | re.DOTALL)
RECALL_FRAC_RE = re.compile(r'(\d+)\s*/\s*(\d+)\s*=\s*[*]*\s*[0-9.]+\s*%', re.IGNORECASE)
PRECISION_FRAC_RE = re.compile(r'(?:精确率|Precision).*?(\d+)\s*/\s*(\d+)\s*=\s*[*]*\s*[0-9.]+', re.IGNORECASE | re.DOTALL)
JSON_BLOCK_RE = re.compile(r'```json\s*(\{[^`]+\})\s*```', re.IGNORECASE | re.DOTALL)


def find_latest_eval_dir(base: Path) -> Optional[Path]:
    run_dirs = sorted(base.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0] if run_dirs else None


def parse_eval_result(md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8", errors="replace")

    jm = JSON_BLOCK_RE.search(text)
    if jm:
        try:
            d = json.loads(jm.group(1))
            tp = int(d.get("tp", 0))
            fp = int(d.get("fp", 0))
            fn = int(d.get("fn", 0))
            recall = float(d.get("recall", 0))
            precision = float(d.get("precision", 0))
            f1 = float(d.get("f1", 0))
            return {
                "recall": recall,
                "precision": precision,
                "detect": precision,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "total_gt": tp + fn,
                "total_ai": tp + fp,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    def grab(pattern):
        m = pattern.search(text)
        return float(m.group(1)) if m else None

    def grab_int(pattern):
        m = pattern.search(text)
        return int(m.group(1)) if m else 0

    def grab_frac(pattern):
        m = pattern.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
        return 0, 0

    recall = grab(RECALL_RE)
    precision = grab(PRECISION_RE)
    detect = grab(DETECT_RE)
    tp = grab_int(TP_RE)
    fp = grab_int(FP_RE)
    fn = grab_int(FN_RE)

    recall_tp, recall_total = grab_frac(RECALL_FRAC_RE)
    prec_tp, prec_total = grab_frac(PRECISION_FRAC_RE)

    if tp == 0 and recall_tp > 0:
        tp = recall_tp
    if fp == 0 and prec_total > 0 and prec_tp > 0:
        fp = prec_total - prec_tp
    if fn == 0 and recall_total > 0 and tp > 0:
        fn = recall_total - tp

    if recall is None and tp is not None and fn is not None and (tp + fn) > 0:
        recall = round(tp / (tp + fn) * 100, 1)
    if precision is None and tp is not None and fp is not None and (tp + fp) > 0:
        precision = round(tp / (tp + fp) * 100, 1)
    if detect is None and precision is not None:
        detect = precision

    total_gt = tp + fn
    if total_gt == 0 and recall_total > 0:
        total_gt = recall_total
    total_ai = tp + fp
    if total_ai == 0 and prec_total > 0:
        total_ai = prec_total

    return {
        "recall": recall or 0.0,
        "precision": precision or 0.0,
        "detect": detect or 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": total_gt,
        "total_ai": total_ai,
    }


def parse_batch_result(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_target_name(name: str) -> Optional[Tuple[str, str]]:
    m = TARGET_NAME_RE.match(name)
    if not m:
        return None
    return m.group("repo"), m.group("pr")


def fmt_status(eval_data: dict) -> str:
    if eval_data["recall"] >= 80 and eval_data["fp"] == 0:
        return "pass"
    if eval_data["recall"] >= 30:
        return "partial"
    return "fail"


def fmt_duration(sec: float) -> str:
    if sec >= 60:
        m = int(sec // 60)
        s = int(round(sec % 60))
        return f"{m}m{s}s" if s > 0 else f"{m}m"
    return f"{int(sec)}s"


def is_new(updated_str: str) -> bool:
    try:
        d = datetime.strptime(updated_str, "%Y-%m-%d").date()
        return (date.today() - d).days <= 2
    except ValueError:
        return False


def build_entry(
    rank: int,
    target_name: str,
    eval_data: dict,
    duration_sec: Optional[float],
    model: str,
    updated_str: str,
    level: str = "3",
) -> dict:
    repo, pr = parse_target_name(target_name) or ("unknown", "0")
    bench_name = f"AscendC-Review-L{level}" if level != "doc" else "AscendC-Review-Doc"

    recall = eval_data["recall"]
    detect = eval_data["detect"]
    items = f'{eval_data["tp"]}/{eval_data["total_gt"]}' if eval_data["total_gt"] > 0 else "—"

    return {
        "rank": rank,
        "model": model or "unknown",
        "bench_name": bench_name,
        "desc": f"PR #{pr} · {repo}",
        "repo": repo,
        "level": level,
        "recall": round(recall, 1),
        "detect": round(detect, 1),
        "duration": int(duration_sec) if duration_sec else 0,
        "items": items,
        "status": fmt_status(eval_data),
        "updated": updated_str,
        "isNew": is_new(updated_str),
        "_tp": eval_data["tp"],
        "_total_gt": eval_data["total_gt"],
    }


def main():
    parser = argparse.ArgumentParser(description="聚合 eval 结果 → leaderboard.json")
    parser.add_argument("--eval-dir", default=None, help="eval-reports/run_xxx 目录（默认自动找最新）")
    parser.add_argument("--review-dir", default=None, help="检视阶段 run 目录（用于补充耗时）")
    parser.add_argument("--manifest", default="manifest.json", help="manifest.json 路径")
    parser.add_argument("--output", default="docs/reports/leaderboard.json", help="输出路径")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent

    eval_dir = Path(args.eval_dir) if args.eval_dir else find_latest_eval_dir(base / "eval-reports")
    if not eval_dir or not eval_dir.exists():
        logging.error(f"错误：eval 目录不存在: {eval_dir}")
        sys.exit(1)
    logging.info(f"eval 目录: {eval_dir}")

    batch_result_path = eval_dir / "batch_result.json"
    if not batch_result_path.exists():
        logging.error(f"错误：batch_result.json 不存在: {batch_result_path}")
        sys.exit(1)

    batch = parse_batch_result(batch_result_path)

    duration_map = {}
    for t in batch.get("targets", []):
        duration_map[t["name"]] = t.get("duration_sec")

    review_durations = {}
    if args.review_dir:
        rdir = Path(args.review_dir)
        if not rdir.is_absolute():
            rdir = base / rdir
        rb_path = rdir / "batch_result.json"
        if rb_path.exists():
            rb = parse_batch_result(rb_path)
            for t in rb.get("targets", []):
                review_durations[t["name"]] = t.get("duration_sec")

    entries = []
    for target_dir in sorted(eval_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        eval_file = target_dir / "eval_result.md"
        if not eval_file.exists():
            eval_file = target_dir / "review_report.md"
        if not eval_file.exists():
            logging.info(f"跳过 {target_dir.name}：无 eval_result.md 或 review_report.md")
            continue

        name = target_dir.name
        eval_data = parse_eval_result(eval_file)

        dur = duration_map.get(name)
        if dur is None and name.startswith("eval-"):
            orig_name = name[5:]
            dur = review_durations.get(orig_name)

        ts_path = target_dir / "task_state.json"
        model = ""
        updated_str = datetime.now().strftime("%Y-%m-%d")
        if ts_path.exists():
            with open(ts_path, "r", encoding="utf-8") as f:
                ts = json.load(f)
            model = ts.get("model", "")
            started = ts.get("started_at", "")
            if started:
                try:
                    updated_str = datetime.fromisoformat(started).strftime("%Y-%m-%d")
                except ValueError:
                    pass

        entries.append(build_entry(
            rank=0, target_name=name, eval_data=eval_data,
            duration_sec=dur, model=model, updated_str=updated_str,
        ))

    entries.sort(key=lambda e: (-e["recall"], -e["detect"], e["duration"]))
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    recalls = [e["recall"] for e in entries if e["recall"] > 0]
    detects = [e["detect"] for e in entries if e["detect"] > 0]
    durations = [e["duration"] for e in entries if e["duration"] > 0]

    total_tp = sum(e["_tp"] for e in entries)
    total_gt = sum(e["_total_gt"] for e in entries)

    kpis = {
        "recall": round(sum(recalls) / len(recalls), 1) if recalls else 0.0,
        "recall_detail": f"命中 {total_tp} / {total_gt}" if total_gt > 0 else "—",
        "detect": round(sum(detects) / len(detects), 1) if detects else 0.0,
        "detect_detail": f"正确检出 {len([e for e in entries if e['status']!='fail'])} / {len(entries)}",
        "avg_duration": round(sum(durations) / len(durations)) if durations else 0,
        "median_duration": round(sorted(durations)[len(durations)//2]) if durations else 0,
        "min_duration": round(min(durations)) if durations else 0,
        "total": len(entries),
        "completed": len([e for e in entries if e["duration"] > 0]),
    }

    for e in entries:
        e.pop("_tp", None)
        e.pop("_total_gt", None)

    output = {
        "generated_at": datetime.now().isoformat(),
        "kpis": kpis,
        "entries": entries,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = base / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    js_path = out_path.with_name("leaderboard_data.js")
    js_content = "window.LEADERBOARD_DATA = " + json.dumps(output, ensure_ascii=False, indent=2) + ";\n"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    html_path = out_path.with_name("index.html")
    if html_path.exists():
        html_text = html_path.read_text(encoding="utf-8")
        marker_start = "    // ========== Inline Data =========="
        marker_js_start = "window.LEADERBOARD_DATA = {"
        marker_js_end = "};\n"
        idx_start = html_text.find(marker_start)
        if idx_start == -1:
            idx_start = html_text.find(marker_js_start)
        if idx_start != -1:
            idx_js_start = html_text.find(marker_js_start, idx_start)
            idx_js_end = html_text.find(marker_js_end, idx_js_start) + len(marker_js_end)
            if idx_js_start != -1 and idx_js_end > idx_js_start:
                html_text = html_text[:idx_js_start] + js_content.rstrip() + "\n" + html_text[idx_js_end:]
                html_path.write_text(html_text, encoding="utf-8")

    logging.info(f"entries: {len(entries)}")
    logging.info(f"KPIs: recall={kpis['recall']}% detect={kpis['detect']}% avg={kpis['avg_duration']}s")
    logging.info(f"输出: {out_path}")
    logging.info(f"输出: {js_path}")
    logging.info(f"内联: {html_path}")


def parse_eval_result_stats(e: dict) -> int:
    try:
        return int(e["items"].split("/")[0]) if "/" in e["items"] else 0
    except (ValueError, IndexError):
        return 0


if __name__ == "__main__":
    main()
