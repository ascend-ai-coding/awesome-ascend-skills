#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

# -----------------------------------------------------------------------------------------------------------
# CANN Open Software License Agreement:
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

"""liveness.py — 逻辑存活节点（L）/ P_baseline 计算器。

用途：
  1. P 的理论下界（原 L 值）交叉校验——L 已从设计流程移除，本脚本以秒级计算
     替代原 LLM 逐 op 手写 trace 的叙事成本（防呆价值保留、成本归零）。
  2. P_baseline（不考虑 VF 融合的基线 P）预计算——传入 pinned（独占分配、
     全程驻留计入的 tensor）即可得到 P_baseline。

输入（JSON，stdin 或文件）：
  {
    "steps": [
      {"id": "S1", "op": "CopyIn", "inputs": [], "outputs": ["x"]},
      {"id": "S2", "op": "Mul", "inputs": ["x", "y"], "outputs": ["t1"]},
      ...
    ],
    "pinned":   ["preReduceResult"],      // 可选：独占分配 tensor，全程计 1（→ P_baseline 模式）
    "excluded": ["var_sum"],              // 可选：跨阶段驻留、归属其他阶段的 tensor，不计入
    "exclusive_branches": [["B1S1", "B1S2"], ["B2S1"]]  // 可选：Tuple Reduce 互斥分支（按 step id），
                                          //   各分支与公共步骤组成变体，L 取各变体峰值 max（不叠加）
  }

  steps 按调度顺序排列（顺序即调度决策；换顺序会得到不同峰值）。

输出：
  - 默认：逐 step 三态 trace（执行前/中/后: 持有=[...]）+ 峰值结论
  - --json：机器可读 {"L": int, "peak_step": str, "pinned": [...], "trace": [...]}

公平基线（与范式文档一致）：
  1. 二元及以上操作不假设 dst 与 src 复用——每个 op 的 src 和 dst 各占独立节点
  2. 单步内不假设 in-place
  3. 跨步"用完释放"允许——一个 tensor 的所有消费者执行完后从存活列表移除
  4. 无消费者的 tensor（最终输出/悬空产出）持有到流程结束

用法：
  python3 liveness.py dag.json
  python3 liveness.py dag.json --pinned preReduceResult   # 命令行补 pinned
  python3 liveness.py --demo                              # 内置已知答案自检
"""
import argparse
import json
import sys
from os.path import dirname, join


def load_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simulate(steps, pinned=(), excluded=()):
    """对给定顺序的 steps 做 liveness 仿真。

    返回 (L, peak_step, trace_lines)。pinned 全程计 1 且不参与释放推导；
    excluded 不计入持有（归属其他阶段）。
    """
    pinned = set(pinned)
    excluded = set(excluded)

    # 消费者计数：tensor -> 消费它的 step 数（按出现次数计）
    consumers = {}
    for st in steps:
        for t in st["inputs"]:
            consumers[t] = consumers.get(t, 0) + 1

    live = set()
    peak = 0
    peak_step = None
    trace = []
    for st in steps:
        before = sorted(live)
        # 输入必须已存活（DAG 顺序校验）。excluded 例外：跨阶段驻留量可被读取
        # （归属其他阶段、由该阶段保证可用）但不计入持有——否则 excluded 字段
        # 一旦被后续步骤读取就直接报错，"持有但不计入"的跨阶段建模无法表达。
        for t in st["inputs"]:
            if t not in live and t not in excluded:
                raise ValueError(
                    f"step {st['id']}: 输入 '{t}' 未存活——steps 顺序违反 DAG 依赖"
                )
        # pinned（独占分配）只计一次：经由 +len(pinned) 常驻计入，须从 during
        # 中排除——否则中流产出且持有的 pinned tensor 会同时出现在 during 与
        # pinned 计数里，被双计（P_baseline 系统性高估）。
        during = sorted((live | set(st["outputs"])) - excluded - pinned)
        held_during = len(during) + len(pinned)
        # 消费：本 step 读过的 tensor 计数减一
        for t in st["inputs"]:
            consumers[t] -= 1
            if consumers[t] == 0:
                live.discard(t)
        # 产出加入存活（除非 excluded）
        for t in st["outputs"]:
            if t not in excluded:
                live.add(t)
        after = sorted(live)
        trace.append({
            "id": st["id"], "op": st.get("op", ""),
            "before": before, "during": during, "after": after,
            "pinned": sorted(pinned),
        })
        if held_during > peak:
            peak = held_during
            peak_step = st["id"]
    return peak, peak_step, trace


def run_variant(spec, step_ids):
    """按 step id 子集（保持原相对顺序）跑一次仿真。"""
    steps = [s for s in spec["steps"] if s["id"] in step_ids]
    return simulate(steps, spec.get("pinned", []), spec.get("excluded", []))


def analyze(spec):
    branches = spec.get("exclusive_branches")
    if branches:
        # 互斥分支：公共步骤 + 每个分支组成一个变体，峰值取 max（不叠加）
        step_ids = {s["id"] for s in spec["steps"]}
        for bi, br in enumerate(branches):
            for sid in br:
                if sid not in step_ids:
                    raise ValueError(
                        f"exclusive_branches 分支{bi} 的 step id '{sid}' 不存在于 steps——"
                        f"拼错的 id 会静默漏跑步骤、P 被低估且无报错"
                    )
        branch_ids = {sid
                      for br in branches
                      for sid in br}
        common = [s["id"] for s in spec["steps"] if s["id"] not in branch_ids]
        results = []
        for br in branches:
            variant = set(common) | set(br)
            peak, peak_step, trace = run_variant(spec, variant)
            results.append((peak, peak_step, trace))
        best = max(results, key=lambda r: r[0])
        return best[0], best[1], best[2], len(results)
    peak, peak_step, trace = simulate(
        spec["steps"], spec.get("pinned", []), spec.get("excluded", []))
    return peak, peak_step, trace, 1


def print_report(spec, peak, peak_step, trace, variants):
    pinned = spec.get("pinned", [])
    mode = f"P_baseline（含 pinned {pinned}）" if pinned else "L（理论下界）"
    print(f"=== {mode} 仿真结果（{variants} 个变体取 max） ===")
    for row in trace:
        pin = f" +{row['pinned']}(pinned)" if row["pinned"] else ""
        print(f"// {row['id']}: {row['op']}")
        print(f"//   执行前: 持有={row['before']}{pin}")
        print(f"//   执行中: 持有={row['during']}{pin}"
              + ("    ← 峰值" if row["id"] == peak_step else ""))
        print(f"//   执行后: 持有={row['after']}{pin}")
    print(f"\n结论：{mode} = {peak}，峰值步骤 = {peak_step}")
    if pinned:
        print("（pinned 为独占分配 tensor，全程计 1——此值即 P_baseline）")
    else:
        print("（无 pinned 时为纯 tensor liveness——即 P 的理论下界，供交叉校验）")


DEMO_PRE = {
    "steps": [
        {"id": "S6", "op": "CopyIn mean_all", "inputs": [], "outputs": ["mean_all"]},
        {"id": "S7", "op": "CopyIn mean_broadcast", "inputs": [], "outputs": ["mean_broadcast"]},
        {"id": "S8", "op": "Sub", "inputs": ["mean_all", "mean_broadcast"], "outputs": ["mean_sub"]},
        {"id": "S9", "op": "Square", "inputs": ["mean_sub"], "outputs": ["mean_var"]},
        {"id": "S1", "op": "CopyIn invert_std_all", "inputs": [], "outputs": ["invert_std_all"]},
        {"id": "S2", "op": "Cast", "inputs": ["invert_std_all"], "outputs": ["invert_std_all_fp32"]},
        {"id": "S3", "op": "Div", "inputs": ["invert_std_all_fp32"], "outputs": ["std_all"]},
        {"id": "S4", "op": "Square", "inputs": ["std_all"], "outputs": ["var_all_square_base"]},
        {"id": "S5", "op": "Sub", "inputs": ["var_all_square_base"], "outputs": ["var_all_square_eps"]},
        {"id": "S10", "op": "Add", "inputs": ["var_all_square_eps", "mean_var"], "outputs": ["mean_var_sum"]},
        {"id": "S11", "op": "CopyIn count_all", "inputs": [], "outputs": ["count_all"]},
        {"id": "S12", "op": "Mul", "inputs": ["mean_var_sum", "count_all"], "outputs": ["mean_var_count"]},
    ],
}


def demo():
    ok = True
    # 用例 1：最优调度（先 mean 链再 invert_std 链）→ L = 3（文档已知答案）
    peak, _, _, _ = analyze(DEMO_PRE)
    if peak != 3:
        print(f"FAIL: demo L 期望 3，实际 {peak}")
        ok = False
    else:
        print("PASS: 最优调度 L = 3（与 reduction-binary-base-dag-buffers §2 已知答案一致）")
    # 用例 2：pinned preReduceResult → P_baseline = 4（文档已知答案）
    spec = dict(DEMO_PRE)
    spec["pinned"] = ["mean_var_count"]
    peak, _, _, _ = analyze(spec)
    if peak != 4:
        print(f"FAIL: demo P_baseline 期望 4，实际 {peak}")
        ok = False
    else:
        print("PASS: pinned 独占分配后 P_baseline = 4（与文档已知答案一致）")
    # 用例 3：朴素顺序（S1..S12 原序）→ 峰值应更高（4），证明调度顺序影响峰值
    naive = {"steps": [
        {"id": "S1", "op": "CopyIn", "inputs": [], "outputs": ["invert_std_all"]},
        {"id": "S2", "op": "Cast", "inputs": ["invert_std_all"], "outputs": ["invert_std_all_fp32"]},
        {"id": "S3", "op": "Div", "inputs": ["invert_std_all_fp32"], "outputs": ["std_all"]},
        {"id": "S4", "op": "Square", "inputs": ["std_all"], "outputs": ["var_all_square_base"]},
        {"id": "S5", "op": "Sub", "inputs": ["var_all_square_base"], "outputs": ["var_all_square_eps"]},
        {"id": "S6", "op": "CopyIn", "inputs": [], "outputs": ["mean_all"]},
        {"id": "S7", "op": "CopyIn", "inputs": [], "outputs": ["mean_broadcast"]},
        {"id": "S8", "op": "Sub", "inputs": ["mean_all", "mean_broadcast"], "outputs": ["mean_sub"]},
        {"id": "S9", "op": "Square", "inputs": ["mean_sub"], "outputs": ["mean_var"]},
        {"id": "S10", "op": "Add", "inputs": ["var_all_square_eps", "mean_var"], "outputs": ["mean_var_sum"]},
        {"id": "S11", "op": "CopyIn", "inputs": [], "outputs": ["count_all"]},
        {"id": "S12", "op": "Mul", "inputs": ["mean_var_sum", "count_all"], "outputs": ["mean_var_count"]},
    ]}
    peak, _, _, _ = analyze(naive)
    if peak != 4:
        print(f"FAIL: 朴素顺序期望 4，实际 {peak}")
        ok = False
    else:
        print("PASS: 朴素顺序 L = 4 > 最优调度 3（验证调度顺序敏感性）")
    # 用例 4：pinned 中流产出（随仓 example-dag.json + --pinned mu：mu 在 S2 产出、
    # S4 消费，峰值时刻仍在存活）——pinned 只计一次，P_baseline = 池峰值 2 + mu = 3
    # （防 pinned 双计回归）
    with open(join(dirname(__file__), "example-dag.json"), encoding="utf-8") as f:
        midflow = json.load(f)
    midflow["pinned"] = ["mu"]
    peak4, ps4, _, _ = analyze(midflow)
    if peak4 != 3 or ps4 != "S4":
        print(f"FAIL: pinned 中流产出场景 P_baseline 期望 3（池峰值 2 + 独占 mu），实际 {peak4}（峰值步骤 {ps4}）")
        ok = False
    else:
        print("PASS: pinned 中流产出场景 P_baseline = 3（pinned 只计一次，无双计）")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="逻辑存活节点 L / P_baseline 计算器")
    ap.add_argument("dag", nargs="?", help="计算图 JSON 文件路径")
    ap.add_argument("--pinned", nargs="*", default=[], help="独占分配 tensor（→ P_baseline 模式）")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--demo", action="store_true", help="运行内置已知答案自检")
    args = ap.parse_args()

    if args.demo:
        return demo()
    if not args.dag:
        ap.error("需要 dag 文件路径（或 --demo）")

    spec = load_spec(args.dag)
    if args.pinned:
        spec["pinned"] = list(set(spec.get("pinned", []) + args.pinned))

    try:
        peak, peak_step, trace, variants = analyze(spec)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            "value": peak, "peak_step": peak_step,
            "mode": "P_baseline" if spec.get("pinned") else "L",
            "pinned": spec.get("pinned", []),
            "trace": trace,
        }, ensure_ascii=False, indent=2))
    else:
        print_report(spec, peak, peak_step, trace, variants)
    return 0


if __name__ == "__main__":
    sys.exit(main())
