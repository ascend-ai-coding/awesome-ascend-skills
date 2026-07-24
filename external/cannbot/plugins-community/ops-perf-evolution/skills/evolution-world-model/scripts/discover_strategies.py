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
# ----------------------------------------------------------------------------------------------------------
"""
discover_strategies.py — v3.2 Phase D3: 从 lineage.jsonl 自动提炼创新策略组合

扫描 evo 输出的 lineage.jsonl，找出满足以下条件的节点：
  (1) precision_passed = true
  (2) speedup ≥ threshold × baseline_speedup (默认 threshold=1.10)
  (3) strategy_combination 不在 known combos 中（"创新组合"标识）

对每个候选自动生成 discovered/disc_X.md 草稿模板，LLM 在下一轮 refine 时
确认补充 frontmatter（complexity、conflicts_with 等）并撰写正文。

用法：
    # 扫描单文件
    python3 discover_strategies.py --lineage output/X/artifacts/lineage.jsonl

    # 扫所有 evo 输出
    python3 discover_strategies.py --scan-all

    # 调阈值（speedup ≥ 1.20 × baseline）
    python3 discover_strategies.py --lineage ... --threshold 1.20

    # 不写文件，只列候选
    python3 discover_strategies.py --lineage ... --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果（JSON/表格/报告）走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)

DEFAULT_THRESHOLD = 1.10
DEFAULT_DISCOVERED_DIR = Path("plugins-community/ops-perf-evolution/skills/evolution-strategies/references/discovered")


def iter_lineage(lineage_path: Path) -> Iterable[dict]:
    """Yield each entry of a lineage.jsonl file."""
    with open(lineage_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def find_innovation_candidates(
    entries: Iterable[dict],
    baseline_speedup: float = 1.0,
    threshold: float = DEFAULT_THRESHOLD,
    known_combos: set[frozenset] | None = None,
) -> list[dict]:
    """从 lineage 条目中筛选创新候选。

    创新条件：
    - precision_passed = true（兼容 null：用 speedup > baseline 兜底）
    - speedup ≥ baseline_speedup × threshold
    - strategy_combination (frozenset) 不在 known_combos
    """
    known_combos = known_combos or set()
    candidates: list[dict] = []
    for e in entries:
        # v3.2 C8-T2: precision_passed=null 时用 speedup > baseline 兜底
        precision_ok = e.get("precision_passed")
        speedup = e.get("speedup") or 0
        if precision_ok is None:
            precision_ok = speedup > baseline_speedup
        if not precision_ok:
            continue
        if speedup < baseline_speedup * threshold:
            continue
        strategies = e.get("strategies") or []
        if len(strategies) < 2:
            # 单策略不算"组合创新"
            continue
        combo = frozenset(strategies)
        if combo in known_combos:
            continue
        candidates.append(e)
        known_combos.add(combo)
    return candidates


def load_known_combos(discovered_dir: Path) -> set[frozenset]:
    """从 discovered/disc_*.md 已有文件读取已注册的组合，避免重复生成。"""
    known: set[frozenset] = set()
    if not discovered_dir.exists():
        return known
    for f in discovered_dir.glob("disc_*.md"):
        text = f.read_text(encoding="utf-8")
        # 简单解析 frontmatter strategies: [P1, P5]
        import re
        m = re.search(r"^strategies:\s*\[(.+)\]", text, re.MULTILINE)
        if m:
            ids = [s.strip().strip("\"'") for s in m.group(1).split(",")]
            known.add(frozenset(ids))
    return known


def next_disc_id(discovered_dir: Path) -> str:
    """生成下一个 disc_X 的编号（X1, X2, ...）。"""
    if not discovered_dir.exists():
        return "X1"
    nums: list[int] = []
    for f in discovered_dir.glob("disc_X*.md"):
        # 从文件名提取 X 后面的数字
        import re
        m = re.match(r"disc_X(\d+)", f.stem)
        if m:
            nums.append(int(m.group(1)))
    return f"X{max(nums) + 1}" if nums else "X1"


def _render_frontmatter(disc_id: str, candidate: dict, strategies_str: str,
                        bottleneck_str: str) -> str:
    """渲染 disc_X.md 的 frontmatter 部分。"""
    speedup = candidate.get("speedup", 0)
    node_id = candidate.get("node_id", "unknown")
    round_n = candidate.get("round", "?")
    parallel = candidate.get("parallel", "?")
    parent_id = candidate.get("parent_id", "root")
    strategies = candidate.get("strategies") or []
    return f"""---
id: {disc_id}
description: 进化发现的策略组合 — {strategies_str}
strategies: [{strategies_str}]
bottlenecks: [{bottleneck_str}]
op_families: [datapath]      # TODO: LLM 在下一轮 refine 时根据上下文修正
complexity: L2               # 默认 L2 discovered，TODO: 评估实际复杂度
conflicts_with: []           # TODO: 评估互斥关系
synergizes_with: {strategies}  # 内部组合天然 synergize
requires: []                 # TODO: 评估前置依赖
has_preconditions: false     # TODO: 后续补 disc_{disc_id}.yaml
has_playbook: false          # TODO: 后续补 playbook
quantified_gain:
  - shape: "TBD"
    baseline_us: null
    optimized_us: null
    speedup: {speedup}
    source: "round_{round_n}_parallel_{parallel}_node_{node_id}"
discovered_at: round={round_n}, node={node_id}, parent={parent_id}
---
"""


def _render_body(disc_id: str, candidate: dict, strategies_str: str,
                 bottleneck_str: str) -> str:
    """渲染 disc_X.md 的正文部分。"""
    speedup = candidate.get("speedup", 0)
    node_id = candidate.get("node_id", "unknown")
    round_n = candidate.get("round", "?")
    parallel = candidate.get("parallel", "?")
    parent_id = candidate.get("parent_id", "root")
    diagnosis = candidate.get("diagnosis") or {}
    diagnosis_text = diagnosis.get("diagnosis_text", "")
    return f"""# {disc_id}: {strategies_str} (Discovered Combination)

> 自动生成的 discovered_strategies 草稿（v3.2 Phase D3 提炼）
> **TODO**: LLM 在下一轮 refine 时确认 frontmatter + 撰写正文

## 自动检测信号

- **来源**：round={round_n}, parallel={parallel}, node_id={node_id}
- **父节点**：{parent_id}
- **策略组合**：[{strategies_str}]
- **实测加速比**：{speedup}× (高于 baseline × threshold)
- **诊断标签**：{bottleneck_str if bottleneck_str else "(无)"}

## 诊断 narrative（来自 lineage）

{diagnosis_text or "（lineage 中未记录 diagnosis_text，请 LLM 补充）"}

## 待补充章节（LLM 任务清单）

- [ ] **核心思想**：为什么这个组合有效？（对照单策略 Card 的 "## 核心思想"）
- [ ] **代码骨架**：组合应用的关键代码片段（取该 node 的 modified_files/ 摘录）
- [ ] **关键修改点**：每个策略在本组合中的作用
- [ ] **适用场景**：除了当前算子，还可能适用于哪些族
- [ ] **常见陷阱**：组合时易出错的点

## 验证状态

- [ ] LLM 已确认 frontmatter (complexity / op_families / conflicts_with)
- [ ] 已补写 Preconditions YAML (disc_{disc_id}.yaml)
- [ ] 已补写 Playbook MD
- [ ] 已在 INDEX.json 注册 (运行 generate_index_json.py 自动)
"""


def render_template(disc_id: str, candidate: dict) -> str:
    """渲染 disc_X.md 草稿。LLM 需在下一轮 refine 时补全 placeholder。"""
    strategies = candidate.get("strategies") or []
    bottleneck_labels = (candidate.get("diagnosis") or {}).get("bottleneck_labels", []) or []

    strategies_str = ", ".join(strategies)
    bottleneck_str = ", ".join(bottleneck_labels) if bottleneck_labels else ""

    return (_render_frontmatter(disc_id, candidate, strategies_str, bottleneck_str)
            + "\n"
            + _render_body(disc_id, candidate, strategies_str, bottleneck_str))


def _collect_lineage_files(args) -> list[Path]:
    """收集待扫描的 lineage.jsonl 文件列表。"""
    if args.lineage:
        return [args.lineage]
    project_root = Path.cwd()
    return list(project_root.glob("output/*_evo_*/artifacts/lineage.jsonl"))


def _scan_candidates(lineage_files: list[Path], args,
                     known_combos: set[frozenset]) -> list[dict]:
    """逐个扫描 lineage 文件，汇总创新候选。"""
    all_candidates: list[dict] = []
    for f in lineage_files:
        if not f.exists():
            LOGGER.warning("WARN: %s not found, skip", f)
            continue
        entries = list(iter_lineage(f))
        candidates = find_innovation_candidates(
            entries,
            baseline_speedup=args.baseline_speedup,
            threshold=args.threshold,
            known_combos=known_combos,
        )
        if candidates:
            LOGGER.info("%s: %d candidate(s)", f, len(candidates))
            all_candidates.extend(candidates)
    return all_candidates


def _write_drafts(candidates: list[dict], args) -> list[Path]:
    """为每个候选写 disc_X.md 草稿。"""
    args.discovered_dir.mkdir(exist_ok=True, parents=True)
    written: list[Path] = []
    for c in candidates:
        disc_id = next_disc_id(args.discovered_dir)
        out_path = args.discovered_dir / f"disc_{disc_id}.md"
        out_path.write_text(render_template(disc_id, c), encoding="utf-8")
        written.append(out_path)
        if args.summary:
            LOGGER.info("  ✓ %s ← %s (speedup=%s)",
                        out_path.name, c.get('strategies'), c.get('speedup'))
    return written


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="自动提炼 discovered_strategies")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--lineage", type=Path,
                     help="单个 lineage.jsonl 路径")
    src.add_argument("--scan-all", action="store_true",
                     help="扫所有 output/*_evo_*/artifacts/lineage.jsonl")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"speedup 必须 ≥ baseline × threshold (默认 {DEFAULT_THRESHOLD})")
    parser.add_argument("--baseline-speedup", type=float, default=1.0,
                        help="baseline 加速比基准（默认 1.0）")
    parser.add_argument("--discovered-dir", type=Path,
                        default=DEFAULT_DISCOVERED_DIR,
                        help="discovered/ 输出目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列候选，不写文件")
    parser.add_argument("--summary", action="store_true",
                        help="人类友好摘要")
    args = parser.parse_args()

    lineage_files = _collect_lineage_files(args)
    if not lineage_files:
        LOGGER.warning("No lineage.jsonl found under output/*_evo_*/artifacts/")
        return 0

    # 加载已知组合避免重复
    known_combos = load_known_combos(args.discovered_dir)
    LOGGER.info("已知 discovered combos: %d", len(known_combos))

    all_candidates = _scan_candidates(lineage_files, args, known_combos)

    if not all_candidates:
        LOGGER.info("✓ No new innovation candidates found "
                    "(threshold=%s, baseline=%s)", args.threshold, args.baseline_speedup)
        return 0

    LOGGER.info("Total new candidates: %d", len(all_candidates))

    if args.dry_run:
        for c in all_candidates:
            DATA_LOGGER.info("  - %s | speedup=%s | node=%s",
                             c.get('strategies'), c.get('speedup'), c.get('node_id'))
        return 0

    written = _write_drafts(all_candidates, args)

    LOGGER.info("Wrote %d disc_X.md draft(s) to %s", len(written), args.discovered_dir)
    LOGGER.info("Run `python3 plugins-community/ops-perf-evolution/skills/"
                "evolution-strategies/scripts/generate_index_json.py` "
                "to refresh INDEX.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
