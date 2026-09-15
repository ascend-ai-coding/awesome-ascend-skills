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
prepare_strategy_resources.py — v3.2 Phase C7

为 partial-prompt 的 [STRATEGY RESOURCES] 段准备完整内容。
封装三步：filter-candidates → load_playbook → get-read-keys → 拼装。

接收一组策略 ID + 算子上下文，输出一段 markdown，可直接填入 partial-prompt
的 {strategy_resources_block} 变量。

用法（主 agent 启动 partial 前调用）：
    python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/prepare_strategy_resources.py \\
        --strategy-ids "P1,P5,P10" \\
        --kernel-dir output/RmsNorm_ops-evo_<ts>/shared/original/ \\
        --evo-dir output/RmsNorm_ops-evo_<ts>/ \\
        --wm-path output/RmsNorm_ops-evo_<ts>/world_model.json \\
        --node-id n1 \\
        > /tmp/srblock_n1.md

  # 然后把 /tmp/srblock_n1.md 的内容填入 partial-prompt 的
  # {strategy_resources_block} 变量

退出码：
  0  success（正常注入 or 全部过滤后已自动改写节点 + 生成 open_exploration srblock）
  1  脚本调用失败（filter 或 load_playbook 任一失败）

v3.2 C7-fix: 当全部候选被 Preconditions 过滤时（passed=[]），脚本主动改写
world_model.json 中对应节点的 strategy_combination=[] + mode=open_exploration，
并生成正常的开放探索 srblock。主 agent 拿到 exit 0 + srblock，无需做任何
特殊处理，partial 收到的 prompt 中不再出现已被过滤的策略 ID —— 从源头消除空洞。
"""

from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果（JSON/表格/报告）走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
SKILL_SCRIPTS = PROJECT_ROOT / "plugins-community/ops-perf-evolution/skills/evolution-strategies/scripts"
WM_OPS = SCRIPT_DIR / "wm_ops.py"
STATE_OPS = SCRIPT_DIR / "state_ops.py"


def _resolve_script(rel_path: str) -> Path:
    """Search project_root + cwd for the script."""
    for base in (PROJECT_ROOT, Path.cwd()):
        p = base / rel_path
        if p.exists():
            return p
    return Path(rel_path)


def step_1_filter(args, candidate_ids: list[str]) -> dict:
    """调用 wm_ops filter-candidates，返回 passed/failed/filtered_by_keys。"""
    cmd = [
        sys.executable, str(WM_OPS), "filter-candidates",
        "--candidate-ids", ",".join(candidate_ids),
        "--kernel-dir", args.kernel_dir,
    ]
    if args.baseline_eval:
        cmd.extend(["--baseline-eval", args.baseline_eval])
    if args.wm_path:
        cmd.extend(["--wm-path", args.wm_path])
    if args.node_id:
        cmd.extend(["--node-id", args.node_id])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"_error": "filter-candidates timed out", "passed": candidate_ids,
                "failed": [], "filtered_by_keys": []}

    if proc.returncode != 0:
        return {"_error": f"filter-candidates exit {proc.returncode}: {proc.stderr[:200]}",
                "passed": candidate_ids, "failed": [], "filtered_by_keys": []}

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"_error": "filter-candidates output not JSON",
                "passed": candidate_ids, "failed": [], "filtered_by_keys": []}


def step_2_load_playbook(passed_ids: list[str]) -> str:
    """调用 load_playbook 拿到 Playbook 全文。返回 markdown 字符串。"""
    if not passed_ids:
        return ""

    load_playbook_py = SKILL_SCRIPTS / "load_playbook.py"
    if not load_playbook_py.exists():
        return f"⚠️  load_playbook.py not found at {load_playbook_py}"

    cmd = [
        sys.executable, str(load_playbook_py),
        "--strategy-ids", ",".join(passed_ids),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "⚠️  load_playbook.py timed out"

    if proc.returncode != 0:
        return f"⚠️  load_playbook.py exit {proc.returncode}: {proc.stderr[:200]}"
    return proc.stdout.strip() or "（无对应 Playbook）"


def step_3_get_excluded(evo_dir: str) -> str:
    """调用 state_ops get-read-keys 拿 Excluded 段 markdown。"""
    if not evo_dir:
        return ""

    cmd = [
        sys.executable, str(STATE_OPS), "get-read-keys",
        "--evo-dir", evo_dir,
        "--format", "markdown",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def step_override_node_in_wm(
    wm_path: str, node_id: str,
    original_strategies: list[str],
    filter_reasons: str,
) -> bool:
    """当所有候选策略被过滤时，主动改写 world_model.json 中的节点。

    将 strategy_combination 清空、mode 切换为 open_exploration，
    并写入 _mode_overridden 标记供事后审计。

    返回 True 表示改写成功，False 表示跳过（wm_path 或 node_id 缺失、节点不存在）。
    """
    if not wm_path or not node_id:
        return False

    wm_file = Path(wm_path)
    if not wm_file.exists():
        LOGGER.warning("WARN: world_model.json not found at %s, skip node override",
                       wm_path)
        return False

    try:
        with open(wm_file, "r", encoding="utf-8") as f:
            wm = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        LOGGER.warning("WARN: cannot read %s: %s, skip node override", wm_path, e)
        return False

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    node = nodes.get(node_id)
    if node is None:
        LOGGER.warning("WARN: node %s not found in %s, skip node override",
                       node_id, wm_path)
        return False

    # 记录原始状态供事后审计
    node["_mode_overridden"] = True
    node["_original_strategies"] = list(original_strategies)
    node["_override_reason"] = filter_reasons

    # 硬改写：清空策略组合 + 切换为开放探索
    node["strategy_combination"] = []
    node["mode"] = "open_exploration"

    # 更新描述
    original_desc = node.get("description", "")
    override_note = (
        f"[MODE OVERRIDE: 全部策略 ({', '.join(original_strategies)}) "
        f"被 Preconditions 过滤 → 切换 open_exploration]"
    )
    if original_desc:
        node["description"] = override_note + " " + original_desc
    else:
        node["description"] = override_note

    try:
        with open(wm_file, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
    except OSError as e:
        LOGGER.warning("WARN: cannot write %s: %s, skip node override", wm_path, e)
        return False

    LOGGER.info("  [override] node %s: strategy_combination %s → [], "
                "mode → open_exploration", node_id, original_strategies)
    return True


def _format_filter_reasons(failed: list[dict]) -> str:
    """把 failed 列表格式化为一行可读字符串。"""
    parts = []
    for f in failed:
        fid = f.get("id", "?")
        checks = f.get("checks", [])
        reasons = "; ".join(
            c.get("fail_msg", c.get("id", "?")) for c in checks
        )
        parts.append(f"{fid}: {reasons}")
    return " | ".join(parts)


@dataclass
class RenderContext:
    """render_block 的输入封装。"""
    passed: list[str]
    failed: list[dict]
    filtered_by_keys: list[str]
    playbook_md: str
    excluded_md: str


def _render_guided_block(ctx: RenderContext, parts: list[str]):
    """正常策略引导分支。"""
    parts.append("## Recommended Strategies（已通过 Preconditions 硬过滤）")
    parts.append("")
    parts.append(f"采纳策略 (Primary)：`{', '.join(ctx.passed)}`")
    parts.append("")
    parts.append("这些策略已通过适用性检查（BUFFER_NUM、shape、dtype 等条件），"
                 "你必须按下方 Playbook 严格执行，不要再质疑适用性。")

    if ctx.filtered_by_keys:
        parts.append("")
        parts.append(f"被过滤的策略检查项（{len(ctx.filtered_by_keys)} 项，"
                     "记录到 node.filtered_by 供事后分析）：")
        for k in ctx.filtered_by_keys:
            parts.append(f"- `{k}`")

    # Playbook SOP 全文
    if ctx.playbook_md and ctx.playbook_md.strip():
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append("## Playbooks（每个 has_playbook=true 的策略附完整 SOP）")
        parts.append("")
        parts.append(ctx.playbook_md)


def _render_exploration_block(ctx: RenderContext, parts: list[str]):
    """所有策略被过滤：开放探索分支。"""
    original_ids = [f.get("id", "?") for f in ctx.failed]
    parts.append("## Open Exploration Guide（策略已由 Preconditions 自动切换）")
    parts.append("")
    parts.append(f"原始策略 `{', '.join(original_ids)}` 经 Preconditions 硬门控检查，"
                 "**全部被判定为不适用于当前算子**。")
    parts.append("节点已由 `prepare_strategy_resources.py` 自动改写："
                 "`strategy_combination=[]`, `mode=open_exploration`。")
    parts.append("")
    parts.append("### 你的任务")
    parts.append("不要读取任何策略卡片文件（`cards/P*_*.md`）。那些策略已确认不适用，"
                 "读了也不会用——只会浪费 token。")
    parts.append("请从第一性原理出发，结合本 prompt 中已有的 Profiling Context "
                 "（pipeline ratios、bottleneck type、硬件参数），设计并实现定向优化。")
    parts.append("")
    parts.append("### 过滤详情（供参考，理解为什么被过滤）")
    for f in ctx.failed:
        fid = f.get("id", "?")
        checks = f.get("checks", [])
        parts.append(f"- **{fid}**:")
        for c in checks:
            parts.append(f"  - `{c.get('id', '?')}` → {c.get('fail_msg', '检查失败')}")

    if ctx.filtered_by_keys:
        parts.append("")
        parts.append(f"`filtered_by` 键（{len(ctx.filtered_by_keys)} 项）：")
        for k in ctx.filtered_by_keys:
            parts.append(f"- `{k}`")

    parts.append("")
    parts.append("### 指导原则")
    parts.append("1. 分析 [Profiling Context] 中的瓶颈标签和 pipeline ratios")
    parts.append("2. 从硬件特性出发：AICore 架构、UB/L1 容量、DMA 带宽、Cube/Vector 单元利用率")
    parts.append("3. 在 `implementation_note.txt` 首行记录："
                 "`MODE OVERRIDE: strategies filtered → open_exploration`")
    parts.append("4. 提出**新颖的、针对当前瓶颈的**优化方向，不限策略库中的卡片")


def render_block(ctx: RenderContext) -> str:
    """拼装最终 markdown。

    两个分支：
      - passed 非空：正常策略引导 srblock（策略列表 + Playbook + Excluded）
      - passed 为空：开放探索 srblock（说明过滤原因 + 第一性原理指导 + Profiling Context）
    """
    parts = []

    if ctx.passed:
        _render_guided_block(ctx, parts)
    else:
        _render_exploration_block(ctx, parts)

    # ── Excluded 段（两种分支共用）──
    if ctx.excluded_md and "暂无已读" not in ctx.excluded_md:
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append(ctx.excluded_md)
        parts.append("")
        parts.append("**说明**：上述 source_key 你不需要再 Read，主 agent 已通过 prompt 注入相关内容。")

    return "\n".join(parts) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare {strategy_resources_block} for partial-prompt injection"
    )
    parser.add_argument("--strategy-ids", required=True,
                        help="逗号分隔的候选策略 ID（如 'P1,P5,P10'）")
    parser.add_argument("--kernel-dir", required=True,
                        help="算子源码目录（含 op_kernel/ 和 op_host/）")
    parser.add_argument("--evo-dir", default=None,
                        help="evo 输出根目录（用于读 state.read_keys）")
    parser.add_argument("--baseline-eval", default=None,
                        help="baseline_evaluation.json 路径")
    parser.add_argument("--wm-path", default=None,
                        help="world_model.json (配合 --node-id 写 filtered_by + 节点改写)")
    parser.add_argument("--node-id", default=None,
                        help="节点 ID（写 filtered_by + 节点改写）")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出文件（默认 stdout）")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args()

    candidate_ids = [s.strip() for s in args.strategy_ids.split(",") if s.strip()]
    if not candidate_ids:
        LOGGER.error("ERROR: --strategy-ids cannot be empty")
        return 1

    # Step 1: filter
    filter_result = step_1_filter(args, candidate_ids)
    if "_error" in filter_result:
        LOGGER.warning("WARN: filter step degraded: %s", filter_result['_error'])
    passed = filter_result.get("passed", candidate_ids)
    failed = filter_result.get("failed", [])
    filtered_by_keys = filter_result.get("filtered_by_keys", [])

    # ── v3.2 C7-fix: passed 为空时，主动改写 wm.json 节点 ──
    if not passed and candidate_ids:
        filter_reasons = _format_filter_reasons(failed)
        step_override_node_in_wm(
            args.wm_path, args.node_id,
            original_strategies=candidate_ids,
            filter_reasons=filter_reasons,
        )

    # Step 2: load playbooks（passed 为空时跳过）
    playbook_md = step_2_load_playbook(passed)

    # Step 3: excluded
    excluded_md = step_3_get_excluded(args.evo_dir) if args.evo_dir else ""

    # Render
    block = render_block(RenderContext(
        passed=passed,
        failed=failed,
        filtered_by_keys=filtered_by_keys,
        playbook_md=playbook_md,
        excluded_md=excluded_md,
    ))

    # Output
    if args.output:
        args.output.write_text(block, encoding="utf-8")
        LOGGER.info("Wrote %s (%d chars)", args.output, len(block))
    else:
        DATA_LOGGER.info("%s", block)

    # 始终返回 0 — 脚本已自行解决了 all-filtered 场景
    # （主 agent 不再需要判断 exit code 2 做兜底处理）
    return 0


if __name__ == "__main__":
    sys.exit(main())
