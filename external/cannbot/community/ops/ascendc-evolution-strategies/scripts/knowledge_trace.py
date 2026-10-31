#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# -----------------------------------------------------------------------------------------------------------
"""knowledge_trace.py — 生成"知识使用 trace"：知识仓检索利用 + local-first 本地知识库。

用于进化 agent 链路：在每个 round checkpoint 与会话完成时落盘 knowledge_trace.md，
使知识检索/复用/新策略暂存有可审计记录，而不是靠人工事后翻 transcripts。

用法:
  python3 knowledge_trace.py --evo-dir <EVO_DIR> [--since 2026-10-22T16:30] \
      [--project-root <项目根，含 .cannbot/>] [--projects-dir ~/.claude/projects] \
      [--knowledge-root <cannbot-knowledge>] [--out <EVO_DIR>/knowledge_trace.md]

数据来源:
  1. 知识仓绑定：<project-root>/.cannbot/knowledge.env
  2. 本地知识库：<project-root>/.cannbot/knowledge-local/{staging,promoted,rejected}
  3. 会话证据：Claude Code transcripts（knowledge refs / 卡片读取 / 本地库命中）
  4. EVO_DIR：world_model.json、round_*/parallel_*/implementation_note.txt、evaluation_results.json

退出码: 0 正常（缺失项以 N/A 标注，不 fail）。
"""

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime, timezone


def _parse_since(s):
    if not s:
        return time.time() - 24 * 3600
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def find_project_root(start):
    cur = os.path.abspath(start)
    while cur != "/":
        if os.path.exists(os.path.join(cur, ".cannbot", "knowledge.env")):
            return cur
        cur = os.path.dirname(cur)
    return None


def read_knowledge_env(project_root):
    env = {}
    p = (
        os.path.join(project_root, ".cannbot", "knowledge.env")
        if project_root
        else None
    )
    if p and os.path.exists(p):
        for line in open(p):
            m = re.match(r"\s*(?:export\s+)?(\w+)='?([^'\n]*)'?\s*$", line)
            if m:
                env[m.group(1)] = m.group(2)
    return env


def list_local_kb(project_root):
    out = {"staging": [], "promoted": [], "rejected": [], "manifest": []}
    if not project_root:
        return out
    base = os.path.join(project_root, ".cannbot", "knowledge-local")
    for kind in ("staging", "promoted", "rejected"):
        d = os.path.join(base, kind)
        if os.path.isdir(d):
            for f in sorted(glob.glob(os.path.join(d, "*.md"))):
                out[kind].append((os.path.basename(f), os.path.getmtime(f)))
    mf = os.path.join(base, "manifest.jsonl")
    if os.path.exists(mf):
        for line in open(mf):
            try:
                out["manifest"].append(json.loads(line))
            except Exception:
                pass
    return out


def scan_transcripts(projects_dir, since_ts, projects_filter=None):
    stats = {
        "files": 0,
        "refs": 0,
        "cards": set(),
        "local_kb": set(),
        "pids": set(),
        "xids": set(),
        "per_file": [],
    }
    if not projects_dir or not os.path.isdir(projects_dir):
        return stats
    files = (
        glob.glob(os.path.join(projects_dir, "*", "*.jsonl"))
        + glob.glob(os.path.join(projects_dir, "*", "subagents", "*.jsonl"))
        + glob.glob(os.path.join(projects_dir, "*", "*", "subagents", "*.jsonl"))
    )
    if projects_filter:
        files = [f for f in files if projects_filter in f]
    for f in files:
        try:
            if os.path.getmtime(f) < since_ts:
                continue
            t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        stats["files"] += 1
        n = len(
            re.findall(
                r"load_playbook|knowledge-local|evolution_strategies|cards/p\d+", t
            )
        )
        stats["refs"] += n
        stats["cards"].update(re.findall(r"cards/(p\d+[a-z0-9_]*)\.md", t))
        stats["local_kb"].update(
            re.findall(r"knowledge-local/(?:staging|promoted|rejected)/[a-z0-9_.-]+", t)
        )
        stats["pids"].update(re.findall(r"\bP\d{2,3}\b", t))
        stats["xids"].update(re.findall(r"\bX\d{2,3}\b", t))
        if n:
            stats["per_file"].append((os.path.basename(f), n))
    stats["per_file"].sort(key=lambda x: -x[1])
    return stats


def scan_evo_dir(evo_dir):
    ids = set()
    evals = []
    if not evo_dir or not os.path.isdir(evo_dir):
        return ids, evals
    for name in ("world_model.json", "world_model_final.json"):
        p = os.path.join(evo_dir, name)
        if os.path.exists(p):
            try:
                ids.update(
                    re.findall(
                        r"\b[PX]\d{2,3}\b",
                        open(p, encoding="utf-8", errors="ignore").read(),
                    )
                )
            except Exception:
                pass
    for p in glob.glob(
        os.path.join(evo_dir, "round_*", "parallel_*", "implementation_note.txt")
    ):
        try:
            ids.update(
                re.findall(
                    r"\b[PX]\d{2,3}\b",
                    open(p, encoding="utf-8", errors="ignore").read(),
                )
            )
        except Exception:
            pass
    for p in sorted(
        glob.glob(
            os.path.join(evo_dir, "round_*", "parallel_*", "evaluation_results.json")
        )
    ):
        try:
            r = json.load(open(p))
            evals.append(
                (
                    p.replace(evo_dir.rstrip("/") + "/", ""),
                    r.get("speedup") or r.get("mean_speedup"),
                    r.get("precision_passed"),
                    r.get("device_id"),
                )
            )
        except Exception:
            pass
    return ids, evals


def count_knowledge_writes(knowledge_root, since_ts):
    """知识仓在窗口内的写入数（local-first 纪律应为 0，artifacts/缓存除外）。"""
    if not knowledge_root or not os.path.isdir(knowledge_root):
        return None
    n = 0
    for f in glob.glob(os.path.join(knowledge_root, "**", "*"), recursive=True):
        if os.path.isfile(f) and "/artifacts/" not in f and "__pycache__" not in f:
            try:
                if os.path.getmtime(f) >= since_ts:
                    n += 1
            except OSError:
                pass
    return n


def fmt_ts(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser(
        description="生成知识使用 trace（知识仓检索 + local-first 本地库）"
    )
    ap.add_argument("--evo-dir", default=None, help="进化输出目录（EVO_DIR）")
    ap.add_argument(
        "--project-root",
        default=None,
        help="项目根（含 .cannbot/），默认从 EVO_DIR 向上查找",
    )
    ap.add_argument(
        "--since", default=None, help="统计窗口起点（ISO 时间或 epoch 秒），默认近 24h"
    )
    ap.add_argument(
        "--projects-dir",
        default=os.path.expanduser("~/.claude/projects"),
        help="Claude Code transcripts 根目录",
    )
    ap.add_argument(
        "--projects-filter",
        default=None,
        help="仅扫描路径含该子串的 transcript（如 ascendc-ops-perf-evolution）",
    )
    ap.add_argument(
        "--knowledge-root",
        default=None,
        help="cannbot-knowledge 路径，用于检查窗口内写入",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="输出 markdown 路径，默认 <EVO_DIR>/knowledge_trace.md",
    )
    args = ap.parse_args()

    since_ts = _parse_since(args.since)
    if args.project_root:
        project_root = args.project_root
    elif args.evo_dir:
        project_root = find_project_root(args.evo_dir)
    else:
        project_root = find_project_root(os.getcwd())
    env = read_knowledge_env(project_root)
    knowledge_root = args.knowledge_root or env.get("CANNBOT_KNOWLEDGE_ROOT")
    local_kb = list_local_kb(project_root)
    tstats = scan_transcripts(args.projects_dir, since_ts, args.projects_filter)
    evo_ids, evals = scan_evo_dir(args.evo_dir)
    repo_writes = count_knowledge_writes(knowledge_root, since_ts)

    lines = []
    title = (
        os.path.basename(args.evo_dir.rstrip("/"))
        if args.evo_dir
        else "knowledge-trace"
    )
    lines.append(f"# 知识使用 Trace — {title}")
    lines.append("")
    lines.append(f"- 生成时间：{fmt_ts(time.time())}")
    lines.append(f"- 统计窗口：{fmt_ts(since_ts)} 起")
    lines.append(f"- EVO_DIR：{args.evo_dir or 'N/A'}")
    lines.append(f"- 项目根：{project_root or 'N/A'}")
    lines.append("")

    lines.append("## 1. 知识仓绑定")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(
        f"| CANNBOT_KNOWLEDGE_ROOT | {env.get('CANNBOT_KNOWLEDGE_ROOT', knowledge_root or 'N/A')} |"
    )
    lines.append(f"| 模式 | {env.get('CANNBOT_KNOWLEDGE_MODE', 'N/A')} |")
    lines.append(f"| commit | {env.get('CANNBOT_KNOWLEDGE_COMMIT', 'N/A')} |")
    writes_disp = repo_writes if repo_writes is not None else "N/A"
    lines.append(f"| 窗口内知识仓写入数（local-first 应为 0） | {writes_disp} |")
    lines.append("")

    lines.append("## 2. 本地知识库（local-first）")
    lines.append("")
    lines.append("| 分区 | 文件（mtime） |")
    lines.append("|---|---|")
    for kind in ("staging", "promoted", "rejected"):
        items = local_kb[kind]
        if items:
            files_disp = ", ".join(f"{n} ({fmt_ts(m)})" for n, m in items)
        else:
            files_disp = "（空）"
        lines.append(f"| {kind} | {files_disp} |")
    new_items = [
        n
        for kind in ("staging", "promoted")
        for n, m in local_kb[kind]
        if m >= since_ts
    ]
    lines.append("")
    lines.append(
        f"- 窗口内新增/变更的本地条目：{', '.join(new_items) if new_items else '无'}"
    )
    if local_kb["manifest"]:
        last = json.dumps(local_kb["manifest"][-1], ensure_ascii=False)[:160]
        lines.append(f"- manifest 记录数：{len(local_kb['manifest'])}（最近：{last}）")
    lines.append("")

    lines.append("## 3. 会话检索证据（transcripts）")
    lines.append("")
    lines.append(
        f"- 扫描 transcript 文件数：{tstats['files']}；知识引用总数：{tstats['refs']}"
    )
    if tstats["per_file"]:
        lines.append("")
        lines.append("| transcript | 知识引用数 |")
        lines.append("|---|---|")
        for f, n in tstats["per_file"][:8]:
            lines.append(f"| {f} | {n} |")
    lines.append("")
    lines.append(
        f"- 实际读取卡片：{', '.join(sorted(tstats['cards'])) if tstats['cards'] else '无'}"
    )
    lines.append(
        f"- 本地知识库命中：{', '.join(sorted(tstats['local_kb'])) if tstats['local_kb'] else '无'}"
    )
    lines.append(
        f"- 引用策略 ID：P×{len(tstats['pids'])}，X×{len(tstats['xids'])}"
        + (
            f"（P 样例：{', '.join(sorted(tstats['pids'])[:10])}）"
            if tstats["pids"]
            else ""
        )
    )
    lines.append("")

    lines.append("## 4. EVO_DIR 策略落地")
    lines.append("")
    ids_disp = ", ".join(sorted(evo_ids))[:400] if evo_ids else "无"
    lines.append(f"- world_model/实现说明中的策略引用：{ids_disp}")
    if evals:
        lines.append("")
        lines.append("| 变体 | speedup | 精度 | device |")
        lines.append("|---|---|---|---|")
        for path, sp, prec, dev in evals:
            lines.append(f"| {path} | {sp} | {prec} | {dev} |")
    lines.append("")

    lines.append("## 5. 判定")
    lines.append("")
    retrieval_ok = tstats["refs"] > 0 and (tstats["cards"] or tstats["local_kb"])
    local_ok = (repo_writes == 0) if repo_writes is not None else None
    lines.append(
        f"- 知识仓检索利用：{'PASS' if retrieval_ok else 'NO-EVIDENCE'}"
        + (
            "（有引用但未见卡片/本地库命中）"
            if tstats["refs"] and not (tstats["cards"] or tstats["local_kb"])
            else ""
        )
    )
    if local_ok is True:
        local_disp = "PASS（知识仓零写入）"
    elif local_ok is False:
        local_disp = "FAIL（窗口内知识仓有写入）"
    else:
        local_disp = "N/A"
    lines.append(f"- local-first 纪律：{local_disp}")
    if new_items:
        staging_disp = "有（" + ", ".join(new_items) + "）"
    else:
        staging_disp = "无"
    lines.append(f"- 窗口内新自进化策略入 staging：{staging_disp}")
    lines.append("")

    out = args.out or (
        os.path.join(args.evo_dir, "knowledge_trace.md")
        if args.evo_dir
        else "knowledge_trace.md"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"knowledge trace written: {out}")
    ret_disp = "PASS" if retrieval_ok else "NO-EVIDENCE"
    print(
        f"  retrieval={ret_disp} refs={tstats['refs']} cards={len(tstats['cards'])}"
        f" local_kb_hits={len(tstats['local_kb'])} repo_writes={repo_writes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
