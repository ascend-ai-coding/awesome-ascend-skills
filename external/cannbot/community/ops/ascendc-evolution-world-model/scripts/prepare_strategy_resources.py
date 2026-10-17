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
    python3 community/ops/ascendc-evolution-world-model/scripts/prepare_strategy_resources.py \\
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
import os
import re
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
SKILLS_ROOT = SCRIPT_DIR.parent.parent  # community/ops
PROJECT_ROOT = SCRIPT_DIR.parents[3]
SKILL_SCRIPTS = SKILLS_ROOT / "ascendc-evolution-strategies/scripts"
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
        sys.executable,
        str(WM_OPS),
        "filter-candidates",
        "--candidate-ids",
        ",".join(candidate_ids),
        "--kernel-dir",
        args.kernel_dir,
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
        return {
            "_error": "filter-candidates timed out",
            "passed": candidate_ids,
            "failed": [],
            "filtered_by_keys": [],
        }

    if proc.returncode != 0:
        return {
            "_error": f"filter-candidates exit {proc.returncode}: {proc.stderr[:200]}",
            "passed": candidate_ids,
            "failed": [],
            "filtered_by_keys": [],
        }

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "_error": "filter-candidates output not JSON",
            "passed": candidate_ids,
            "failed": [],
            "filtered_by_keys": [],
        }


def step_2_load_playbook(passed_ids: list[str]) -> str:
    """调用 load_playbook 拿到 Playbook 全文。返回 markdown 字符串。"""
    if not passed_ids:
        return ""

    load_playbook_py = SKILL_SCRIPTS / "load_playbook.py"
    if not load_playbook_py.exists():
        return f"⚠️  load_playbook.py not found at {load_playbook_py}"

    cmd = [
        sys.executable,
        str(load_playbook_py),
        "--strategy-ids",
        ",".join(passed_ids),
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
        sys.executable,
        str(STATE_OPS),
        "get-read-keys",
        "--evo-dir",
        evo_dir,
        "--format",
        "markdown",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _find_fallback_strategies(
    excluded_ids: list[str],
    count: int = 2,
) -> list[str]:
    """P2: 从 INDEX.json 选"永不触发 Preconditions"的替代策略。

    全部候选被 Preconditions 过滤时，优先注入这些替代策略（保证再次过滤也会通过），
    而不是直接降级 open_exploration。候选 = 未废弃 + 无 preconditions 文件的策略，
    按是否覆盖 attention/flash_attention 排序（常见进化目标）。
    """
    index_path = SKILLS_ROOT / "ascendc-evolution-strategies/references/INDEX.json"
    if not index_path.exists():
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    excluded = set(excluded_ids)
    cards = idx.get("cards", [])
    fallback = [
        c
        for c in cards
        if (
            c.get("id")
            and c["id"] not in excluded
            and not c.get("deprecated")
            and not c.get("has_preconditions")
        )
    ]
    # 排序：attention/flash_attention 相关优先，其次按 id
    fallback.sort(
        key=lambda c: (
            not ({"attention", "flash_attention"} & set(c.get("op_families") or [])),
            c.get("id", ""),
        )
    )
    return [c["id"] for c in fallback[:count]]


def step_override_node_in_wm(
    wm_path: str,
    node_id: str,
    original_strategies: list[str],
    filter_reasons: str,
    fallback_strategies: list[str] = None,
) -> bool:
    """当所有候选策略被过滤时，主动改写 world_model.json 中的节点。

    - 有 fallback_strategies：注入替代策略（保持 strategy_guided 模式），带
      _replaced_from 审计标记，避免退化为无策略的 open_exploration。
    - 无 fallback_strategies：strategy_combination 清空、mode 切换为
      open_exploration，写入 _mode_overridden 标记供事后审计。

    返回 True 表示改写成功，False 表示跳过（wm_path 或 node_id 缺失、节点不存在）。
    """
    if not wm_path or not node_id:
        return False

    wm_file = Path(wm_path)
    if not wm_file.exists():
        LOGGER.warning(
            "WARN: world_model.json not found at %s, skip node override", wm_path
        )
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
        LOGGER.warning(
            "WARN: node %s not found in %s, skip node override", node_id, wm_path
        )
        return False

    # 记录原始状态供事后审计
    node["_original_strategies"] = list(original_strategies)
    node["_override_reason"] = filter_reasons
    original_desc = node.get("description", "")

    if fallback_strategies:
        # P2: 注入替代策略，保持 strategy_guided 方向，避免退化为 open_exploration
        node["strategy_combination"] = list(fallback_strategies)
        node["_replaced_from"] = list(original_strategies)
        replace_note = (
            f"[P2: 策略替换] 原策略 ({', '.join(original_strategies)}) "
            f"被 Preconditions 过滤 → 注入替代 {', '.join(fallback_strategies)}]"
        )
        node["description"] = replace_note + (
            " " + original_desc if original_desc else ""
        )
    else:
        # 硬改写：清空策略组合 + 切换为开放探索
        node["_mode_overridden"] = True
        node["strategy_combination"] = []
        node["mode"] = "open_exploration"
        override_note = (
            f"[MODE OVERRIDE: 全部策略 ({', '.join(original_strategies)}) "
            f"被 Preconditions 过滤且无替代 → 切换 open_exploration]"
        )
        node["description"] = override_note + (
            " " + original_desc if original_desc else ""
        )

    try:
        with open(wm_file, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
    except OSError as e:
        LOGGER.warning("WARN: cannot write %s: %s, skip node override", wm_path, e)
        return False

    LOGGER.info(
        "  [override] node %s: strategy_combination %s → [], mode → open_exploration",
        node_id,
        original_strategies,
    )
    return True


def _format_filter_reasons(failed: list[dict]) -> str:
    """把 failed 列表格式化为一行可读字符串。"""
    parts = []
    for f in failed:
        fid = f.get("id", "?")
        checks = f.get("checks", [])
        reasons = "; ".join(c.get("fail_msg", c.get("id", "?")) for c in checks)
        parts.append(f"{fid}: {reasons}")
    return " | ".join(parts)


def _load_strategy_index() -> dict:
    """加载策略库 INDEX.json，失败返回空 dict。"""
    idx_path = (
        PROJECT_ROOT
        / "community/ops/ascendc-evolution-strategies/references/INDEX.json"
    )
    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _category_from_external_ref(ref: str) -> str:
    """从 ascendc-tiling-design 路径推断类别，如 matmul/reduction/...。"""
    parts = ref.split("/")
    marker = "ascendc-tiling-design"
    if marker in parts:
        idx = parts.index(marker)
        if idx + 2 < len(parts):
            return parts[idx + 2]
    return "general"


def _build_external_references_md(passed_ids: list[str]) -> str:
    """把 passed 策略对应卡片中的 references 追加为 [External References] markdown。

    references 指向 ascendc-tiling-design 文档，按类别分组列出。
    """
    idx = _load_strategy_index()
    refs_by_cat: dict[str, set[str]] = {}
    for pid in passed_ids:
        for card in idx.get("cards", []):
            if card.get("id") != pid:
                continue
            for ref in card.get("references") or []:
                cat = _category_from_external_ref(ref)
                refs_by_cat.setdefault(cat, set()).add(ref)
    if not refs_by_cat:
        return ""

    lines = [
        "",
        "---",
        "",
        "## External References（策略卡片关联的 ascendc-tiling-design 文档）",
        "",
        "以下文档为当前策略组合提供了系统化的 Tiling 设计背景知识，子 agent 可在此基础上更准确地实现优化。",
    ]
    for cat in sorted(refs_by_cat):
        lines.append(f"\n### {cat}")
        for ref in sorted(refs_by_cat[cat]):
            lines.append(f"- `{ref}`")
    return "\n".join(lines) + "\n"


def _resolve_knowledge_root() -> Path | None:
    """按三级来源解析 cannbot-knowledge 仓根（与 load_playbook.py 对齐）：

    1. CANNBOT_KNOWLEDGE_ROOT 环境变量
    2. 自 cwd 向上最近的 .cannbot/knowledge.env（cannbot-knowledge install.sh 写入项目）
    3. skills 仓真实路径的兄弟检出 ../cannbot-knowledge（开发布局便利回退）
    全部落空返回 None（调用方告警跳过，不静默裸奔）。
    """
    env_root = os.environ.get("CANNBOT_KNOWLEDGE_ROOT", "").strip()
    if env_root and Path(env_root).expanduser().is_dir():
        return Path(env_root).expanduser()
    d = Path.cwd()
    while True:
        cand = d / ".cannbot" / "knowledge.env"
        if cand.is_file():
            try:
                for line in cand.read_text(encoding="utf-8").splitlines():
                    m = re.match(
                        r"\s*(?:export\s+)?CANNBOT_KNOWLEDGE_ROOT\s*=\s*(.*?)\s*$", line
                    )
                    if m:
                        val = m.group(1)
                        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                            val = val[1:-1]
                        if val and Path(val).expanduser().is_dir():
                            return Path(val).expanduser()
            except OSError:
                pass
            break
        if d.parent == d:
            break
        d = d.parent
    # <repo>/community/ops/ascendc-evolution-world-model/scripts → 仓根父目录下的兄弟检出
    repo_root = SCRIPT_DIR.parents[3]
    cand = repo_root.parent / "cannbot-knowledge"
    if cand.is_dir():
        return cand
    return None


def _build_discovered_md(passed_ids: list[str]) -> str:
    """把 passed 中的 X 系列（进化发现策略）全文注入资源块。

    X 系列没有 Preconditions / Playbook 文件。优先读取 OKF 增强版
    （ascendc-evolution-knowledge/references/okf/strategies/discovered/，经 INDEX.json
    cards 数组中 kind=discovered_strategy 的卡反查 doc_path），未迁移到 OKF
    的回落兼容层（INDEX.json discovered 数组，doc_repo=cannbot-knowledge，
    doc_path 相对知识仓根）。
    """
    idx = _load_strategy_index()
    # OKF 增强版（优先）：cards 数组中 kind=discovered_strategy 的卡
    okf_disc = {
        c.get("id"): c.get("doc_path")
        for c in idx.get("cards", [])
        if c.get("kind") == "discovered_strategy"
    }
    # 兼容层：discovered 数组（doc_repo=cannbot-knowledge，doc_path 相对知识仓根）
    compat_disc = {e.get("id"): e.get("doc_path") for e in idx.get("discovered", [])}
    knowledge_root: Path | None = None
    knowledge_root_tried = False
    sections = []
    for pid in passed_ids:
        doc_path = okf_disc.get(pid) or compat_disc.get(pid)
        if not doc_path:
            continue
        # OKF doc_path 相对 community/ops（旧数据带 "skills/" 前缀则剥除）；
        # 兼容层 doc_path 相对 cannbot-knowledge 仓根（三级来源解析）
        if pid in okf_disc:
            rel = doc_path.removeprefix("skills/")
            fpath = SKILLS_ROOT / rel
        else:
            if not knowledge_root_tried:
                knowledge_root = _resolve_knowledge_root()
                knowledge_root_tried = True
            if knowledge_root is None:
                LOGGER.warning(
                    "WARN: 无法定位 cannbot-knowledge 仓，跳过 X 卡注入（%s）。"
                    "请 export CANNBOT_KNOWLEDGE_ROOT=<知识仓根> 或在项目根运行其 install.sh。",
                    pid,
                )
                continue
            fpath = knowledge_root / doc_path
        try:
            content = fpath.read_text(encoding="utf-8").strip()
        except OSError:
            LOGGER.warning("WARN: discovered doc not found: %s", fpath)
            continue
        sections.append(
            f"<!-- BEGIN {pid}: {doc_path} -->\n{content}\n<!-- END {pid} -->"
        )
    if not sections:
        return ""

    lines = [
        "",
        "---",
        "",
        "## Discovered Strategies（X 系列进化发现策略全文）",
        "",
        "以下策略为既往进化中发现的方案，含核心思路、适用场景与实现要点，按此执行。",
        "",
        "\n\n".join(sections),
    ]
    return "\n".join(lines) + "\n"


@dataclass
class RenderContext:
    """render_block 的输入封装。"""

    passed: list[str]
    failed: list[dict]
    filtered_by_keys: list[str]
    playbook_md: str
    excluded_md: str
    discovered_md: str
    failure_anti_md: str


def _build_failure_patterns_anti_md(passed_ids: list[str]) -> str:
    """策略组合命中已知失败模式时，注入 anti 提示。

    从 INDEX.json cards 中 kind=failure_pattern 的卡，查找 failed_combination
    完全被 passed_ids 覆盖的组合，输出警告 + Resolution 指引，防止复用
    既往已证实会失败的策略组合（OKF runbooks/failure-patterns/）。
    """
    if not passed_ids:
        return ""
    idx = _load_strategy_index()
    passed_set = set(passed_ids)
    alerts = []
    for c in idx.get("cards", []):
        if c.get("kind") != "failure_pattern":
            continue
        fc = c.get("failed_combination") or []
        if not fc or not set(fc).issubset(passed_set):
            continue
        alerts.append(c)
    if not alerts:
        return ""

    lines = [
        "",
        "---",
        "",
        "## ⚠️ Failure Pattern Alerts（策略组合命中已知失败模式）",
        "",
        "以下组合在既往进化中被证实会失败，请避开或按 Resolution 处理：",
        "",
    ]
    for a in alerts:
        lines.append(
            f"- **{a.get('id', '?')}** [{a.get('status', '')}]: "
            f"组合 `{' + '.join(a.get('failed_combination', []))}` → "
            f"{a.get('phenomenon', '?')}（severity: {a.get('severity', '?')}）"
        )
        lines.append(
            f"  - 详见 `{a.get('doc_path', '')}`，先读其中的 Resolution 再决定是否规避。"
        )
    return "\n".join(lines) + "\n"


def _render_guided_block(ctx: RenderContext, parts: list[str]):
    """正常策略引导分支。"""
    parts.append("## Recommended Strategies（已通过 Preconditions 硬过滤）")
    parts.append("")
    parts.append(f"采纳策略 (Primary)：`{', '.join(ctx.passed)}`")
    parts.append("")
    parts.append(
        "这些策略已通过适用性检查（BUFFER_NUM、shape、dtype 等条件），"
        "你必须按下方 Playbook 严格执行，不要再质疑适用性。"
    )

    if ctx.filtered_by_keys:
        parts.append("")
        parts.append(
            f"被过滤的策略检查项（{len(ctx.filtered_by_keys)} 项，"
            "记录到 node.filtered_by 供事后分析）："
        )
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

    # 追加 ascendc-tiling-design 外部参考（来自 INDEX.json card references）
    external_md = _build_external_references_md(ctx.passed)
    if external_md:
        parts.append(external_md)

    # 追加 X 系列进化发现策略全文（来自 INDEX.json discovered 数组）
    if ctx.discovered_md:
        parts.append(ctx.discovered_md)


def _render_exploration_block(ctx: RenderContext, parts: list[str]):
    """所有策略被过滤：开放探索分支。"""
    original_ids = [f.get("id", "?") for f in ctx.failed]
    parts.append("## Open Exploration Guide（策略已由 Preconditions 自动切换）")
    parts.append("")
    parts.append(
        f"原始策略 `{', '.join(original_ids)}` 经 Preconditions 硬门控检查，"
        "**全部被判定为不适用于当前算子**。"
    )
    parts.append(
        "节点已由 `prepare_strategy_resources.py` 自动改写："
        "`strategy_combination=[]`, `mode=open_exploration`。"
    )
    parts.append("")
    parts.append("### 你的任务")
    parts.append(
        "不要读取任何策略卡原文（cannbot-knowledge 仓 evolution_strategies 卡片）。那些策略已确认不适用，"
        "读了也不会用——只会浪费 token。"
    )
    parts.append(
        "请从第一性原理出发，结合本 prompt 中已有的 Profiling Context "
        "（pipeline ratios、bottleneck type、硬件参数），设计并实现定向优化。"
    )
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
    parts.append(
        "2. 从硬件特性出发：AICore 架构、UB/L1 容量、DMA 带宽、Cube/Vector 单元利用率"
    )
    parts.append(
        "3. 在 `implementation_note.txt` 首行记录："
        "`MODE OVERRIDE: strategies filtered → open_exploration`"
    )
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

    # ── 失败模式 anti 提示（策略组合命中已知失败组合时）──
    if ctx.failure_anti_md:
        parts.append(ctx.failure_anti_md)

    # ── Excluded 段（两种分支共用）──
    if ctx.excluded_md and "暂无已读" not in ctx.excluded_md:
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append(ctx.excluded_md)
        parts.append("")
        parts.append(
            "**说明**：上述 source_key 你不需要再 Read，主 agent 已通过 prompt 注入相关内容。"
        )

    return "\n".join(parts) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare {strategy_resources_block} for partial-prompt injection"
    )
    parser.add_argument(
        "--strategy-ids", required=True, help="逗号分隔的候选策略 ID（如 'P1,P5,P10'）"
    )
    parser.add_argument(
        "--kernel-dir", required=True, help="算子源码目录（含 op_kernel/ 和 op_host/）"
    )
    parser.add_argument(
        "--evo-dir", default=None, help="evo 输出根目录（用于读 state.read_keys）"
    )
    parser.add_argument(
        "--baseline-eval", default=None, help="baseline_evaluation.json 路径"
    )
    parser.add_argument(
        "--wm-path",
        default=None,
        help="world_model.json (配合 --node-id 写 filtered_by + 节点改写)",
    )
    parser.add_argument(
        "--node-id", default=None, help="节点 ID（写 filtered_by + 节点改写）"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="输出文件（默认 stdout）"
    )
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
        LOGGER.warning("WARN: filter step degraded: %s", filter_result["_error"])
    passed = filter_result.get("passed", candidate_ids)
    failed = filter_result.get("failed", [])
    filtered_by_keys = filter_result.get("filtered_by_keys", [])

    # ── v3.2 C7-fix: passed 为空时，主动改写 wm.json 节点 ──
    if not passed and candidate_ids:
        filter_reasons = _format_filter_reasons(failed)
        # P2: 先尝试注入替代策略（无 precondition、不会再被过滤），避免直接退化为
        # open_exploration；无替代时才降级 open。
        fallback = _find_fallback_strategies(
            candidate_ids + [f.get("id", "") for f in failed]
        )
        if fallback:
            passed = fallback
            LOGGER.info(
                "[P2] 原策略全被 Preconditions 过滤，注入替代策略 %s (原: %s)",
                fallback,
                candidate_ids,
            )
        step_override_node_in_wm(
            args.wm_path,
            args.node_id,
            original_strategies=candidate_ids,
            filter_reasons=filter_reasons,
            fallback_strategies=passed if fallback else None,
        )

    # Step 2: load playbooks（passed 为空时跳过）
    playbook_md = step_2_load_playbook(passed)

    # Step 3: excluded
    excluded_md = step_3_get_excluded(args.evo_dir) if args.evo_dir else ""

    # Render
    block = render_block(
        RenderContext(
            passed=passed,
            failed=failed,
            filtered_by_keys=filtered_by_keys,
            playbook_md=playbook_md,
            excluded_md=excluded_md,
            discovered_md=_build_discovered_md(passed),
            failure_anti_md=_build_failure_patterns_anti_md(passed),
        )
    )

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
