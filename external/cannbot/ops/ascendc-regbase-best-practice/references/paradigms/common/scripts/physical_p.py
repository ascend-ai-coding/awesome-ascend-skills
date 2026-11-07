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

"""physical_p.py — 物理存活节点（P）峰值计算器（融合后物理 trace 预计算）。

用途：VF 段编排 + 融合方案确定后，给定 buffer 分配方案，程序化预计算
P 峰值与三态 trace；LLM 只做复核，不再逐行手写 trace。
内置下界自检（P < 输入数+输出数 → 须解释成因：VF 消中间量 / I/O 错峰等；该 bound 假设输入输出同时驻 UB，非硬性下界）。

输入（JSON，stdin 或文件）：
  {
    "inputs":  ["mean_all", "mean_broadcast"],   // GM 输入 tensor 名（下界自检用）
    "outputs": ["out0"],                          // 输出 tensor 名（下界自检用）
    "pinned":  ["B3"],                            // 可选：独占分配 buffer，全程计 1
    "steps": [
      {"id": "S1", "op": "CopyInBrc", "reads": [], "writes": ["B0"]},
      {"id": "S2", "op": "asc_vf_call", "reads": ["B0","B1"], "writes": ["B2"]},
      ...
    ],
    "exclusive_branches": [["B1S1"], ["B2S1"]]   // 可选：Tuple Reduce 互斥分支（按 step id）
  }

  steps 按调度顺序排列；reads/writes 为物理 buffer 名。同一 buffer 可多次
  复用（前序内容消费完后重写）；重写仍持有消费者的 buffer 会报 WAR 违规。

输出：
  - 默认：三态 trace（执行前/中/后: 持有=[B0,B1,...]）+ P 结论 + 下界自检
  - --json：机器可读 {"P": int, "peak_step": str, "lower_bound_check": {...}, "trace": [...]}

持有法则（与范式文档一致）：
  - 执行中 = 执行前持有 + 本次写入的 buffer（本次读取的 buffer 必然已在持有中）
  - 执行后 = 执行中 − 本次读完后不再有未来消费者的 buffer
  - P = 全流程"执行中"峰值的最大值（+ pinned 常驻数）
  - 校验：读取未持有的 buffer → 顺序违规；重写仍有未来消费者的 buffer → WAR 违规

用法：
  python3 physical_p.py flow.json
  python3 physical_p.py --demo    # 内置已知答案自检
"""
import argparse
import json
import sys
from os.path import dirname, join


def simulate(steps, pinned=()):
    """两遍扫描：第一遍建立 (buffer, 内容序号)→消费者总数；第二遍持有法则仿真。"""
    pinned = set(pinned)
    warnings = []

    # 第一遍：内容序号（buffer 第 k 次写入创建内容 k）与各内容的消费者总数。
    # step 内先读旧内容、后写新内容。
    seq = {}
    consumers = {}
    for st in steps:
        for b in st.get("reads", []):
            if b in seq:
                consumers[(b, seq[b])] = consumers.get((b, seq[b]), 0) + 1
        for b in st.get("writes", []):
            seq[b] = seq.get(b, 0) + 1
            consumers[(b, seq[b])] = 0

    # 第二遍：持有法则仿真
    write_cnt = {}   # buffer -> 已写入次数（与第一遍 seq 同步递增）
    pinned_seen = set()  # 已完成首次写入的 pinned buffer（未初始化的 pinned 槽不可读）
    held = {}        # buffer -> 当前内容序号
    peak = 0
    peak_step = None
    trace = []
    for st in steps:
        reads = st.get("reads", [])
        writes = st.get("writes", [])
        before = sorted(held.keys())
        for b in reads:
            # pinned 放行仅限"已初始化"的独占槽：独占分配 ≠ 内容已初始化——
            # 首次写入前读取仍是顺序违规（Use 先于 CopyIn 的错误流不得静默通过）
            if b not in held and (b not in pinned or b not in pinned_seen):
                raise ValueError(
                    f"step {st['id']}: 读取未持有的 buffer '{b}'——顺序违规"
                    + ("（pinned 槽首次写入前不可读）" if b in pinned else "")
                )
        for b in writes:
            if b in held:
                # 同 step 先读后写（cacheBuf 累加 acc += f(x)，步内约定先读后写）合法：
                # 按本 step 对该 buffer 的读取出现次数扣减——reads=["B0","B0"] 的
                # x*x 就地写形态两次自身读都要扣除，只扣一次会误报 WAR。
                future = consumers[(b, held[b])] - reads.count(b)
                if future > 0:
                    warnings.append(
                        f"step {st['id']}: 重写 '{b}' 时其当前内容仍有 "
                        f"{future} 个未来消费者——WAR 违规，须先释放再复用"
                    )
                elif consumers[(b, held[b])] == 0 and reads.count(b) == 0:
                    # 变体内悬空内容（本变体从未有消费者、本步也未读）被覆写——
                    # 保守口径要求悬空占槽到流程结束（与 liveness/allocate 一致）；
                    # 该映射把悬空值让给了后续写入，可能掩盖漏写的消费步骤
                    warnings.append(
                        f"step {st['id']}: 覆写悬空内容 '{b}'（本变体无任何消费者）——"
                        f"保守口径悬空占槽到流程结束；请检查是否漏写消费步骤，"
                        f"或确认采用非保守释放口径并同步三脚本"
                    )
        # pinned（独占分配）只计一次：经由 +len(pinned) 常驻计入，须从 during
        # 中排除——否则中流写入且持有的 pinned buffer 会同时出现在 during 与
        # pinned 计数里，被双计（P 系统性高估）。
        during = sorted((set(held.keys()) | set(writes)) - pinned)
        held_during = len(during) + len(pinned)
        for b in reads:
            if b in held:
                key = (b, held[b])
                if consumers[key] > 0:
                    consumers[key] -= 1
        # 释放：仅当本 step 读过 且 读完后无未来消费者，且本 step 未重写该 buffer。
        # 悬空写入（mean/rstd 这类只作输出的统计量，消费者计数恒 0）持有到被读或
        # 流程结束——否则 P 被低估、下界自检虚假通过（与 liveness.py 公平基线第 4 条
        # 「无消费者的 tensor 持有到流程结束」口径一致）。
        read_set = set(reads)
        for b in list(held.keys()):
            if b in read_set and b not in writes and consumers[(b, held[b])] == 0:
                del held[b]
        for b in writes:
            if b in pinned:
                pinned_seen.add(b)
            write_cnt[b] = write_cnt.get(b, 0) + 1
            held[b] = write_cnt[b]
        after = sorted(held.keys())
        trace.append({
            "id": st["id"], "op": st.get("op", ""),
            "before": before, "during": during, "after": after,
            "pinned": sorted(pinned),
        })
        if held_during > peak:
            peak = held_during
            peak_step = st["id"]
    return peak, peak_step, trace, warnings


def analyze(spec):
    branches = spec.get("exclusive_branches")
    if branches:
        # Tuple Reduce 互斥分支：公共步骤 + 每个分支组成变体，P 取各变体峰值 max（不叠加）
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
        results = []
        for br in branches:
            ids = set(br)
            # 保序筛选（与 liveness.py run_variant 同口径）：变体 = 原输入 steps 的相对顺序下
            # 保留公共步骤与本分支步骤。禁止 common+branch 拼接——那会把公共链提到分支之前，
            # 使 z 提前覆写 B0、分支后读到的已非原值，计数序列不再对应任何合法执行
            # （实测合法 3 槽输入被误报 P=2，且恰好躲过下界自检）。
            variant = [s for s in spec["steps"] if s["id"] not in branch_ids or s["id"] in ids]
            results.append(simulate(variant, spec.get("pinned", [])))
        best = max(results, key=lambda r: r[0])
        return best + (len(results),)
    peak, ps, trace, warns = simulate(spec["steps"], spec.get("pinned", []))
    return peak, ps, trace, warns, 1


def lower_bound_check(spec, p):
    n_in = len(spec.get("inputs", []))
    n_out = len(spec.get("outputs", []))
    bound = n_in + n_out
    triggered = p < bound
    return {
        "P": p, "inputs": n_in, "outputs": n_out, "bound": bound,
        "triggered": triggered,
        "message": (
            f"⚠ P={p} < 输入{n_in}+输出{n_out}={bound}：须解释成因（输入数+输出数假设 GM 输入输出同时驻 UB，"
            f"并非硬性同时存活下界）——合法成因含 ① VF 寄存器链消掉中间量 ② I/O 错峰调度"
            f"（串行 CopyIn→计算→CopyOut 流水下输入输出非同时驻留）等，经解释方可进入 tiling"
            if triggered else
            f"通过（P={p} ≥ 输入{n_in}+输出{n_out}={bound}）"
        ),
    }


def print_report(spec, peak, peak_step, trace, warnings, variants, lb):
    print(f"=== 物理存活节点 P 仿真（{variants} 个变体取 max） ===")
    for row in trace:
        pin = f" +{row['pinned']}(pinned)" if row["pinned"] else ""
        print(f"// {row['id']}: {row['op']}")
        print(f"//   执行前: 持有={row['before']}{pin}")
        print(f"//   执行中: 持有={row['during']}{pin}"
              + ("    ← P 峰值" if row["id"] == peak_step else ""))
        print(f"//   执行后: 持有={row['after']}{pin}")
    for w in warnings:
        print(f"⚠ {w}")
    print(f"\nP = {peak}，峰值步骤 = {peak_step}（含 pinned {spec.get('pinned', [])}）")
    print(f"下界自检：{lb['message']}")
    if lb["triggered"]:
        print("  → 未解释清楚前不得进入 tiling（UB 预算不等式的 P 值不可信）")


DEMO_PRE_FUSED = {
    "inputs": ["mean_all", "mean_broadcast", "invert_std_all", "count_all"],
    "outputs": ["mean_var_count"],
    "pinned": ["B3"],
    "steps": [
        {"id": "S6", "op": "CopyIn mean_all → B0", "reads": [], "writes": ["B0"]},
        {"id": "S7", "op": "CopyIn mean_broadcast → B1", "reads": [], "writes": ["B1"]},
        {"id": "S8-S9", "op": "VF: Sub+Square → mean_var", "reads": ["B0", "B1"], "writes": ["B2"]},
        {"id": "S1", "op": "CopyIn invert_std_all → B0", "reads": [], "writes": ["B0"]},
        {"id": "S2-S5", "op": "VF: Cast+Div+Square+Sub", "reads": ["B0"], "writes": ["B1"]},
        {"id": "S10", "op": "Add", "reads": ["B2", "B1"], "writes": ["B0"]},
        {"id": "S11", "op": "CopyIn count_all → B1", "reads": [], "writes": ["B1"]},
        {"id": "S12", "op": "Mul → preReduceResult", "reads": ["B0", "B1"], "writes": ["B3"]},
    ],
}


def demo():
    ok = True
    peak, ps, trace, warns, _ = analyze(DEMO_PRE_FUSED)
    if peak != 4:
        print(f"FAIL: demo P 期望 4（复用池 3 + 独占 B3），实际 {peak}")
        ok = False
    else:
        print("PASS: P = 4（与 reduction-binary-base-dag-buffers §2 物理 trace 已知答案一致）")
    if warns:
        print(f"FAIL: demo 不应产生 WAR 警告，实际 {warns}")
        ok = False
    else:
        print("PASS: 无 WAR 违规（B0/B1 复用时前序内容均已消费完）")
    lb = lower_bound_check(DEMO_PRE_FUSED, peak)
    if lb["triggered"]:
        print(f"PASS: 下界自检按预期触发（{lb['message']}）——文档示例解释为输入错峰 + VF 链消中间值")
    else:
        print("FAIL: 下界自检应触发（4 < 4+1）")
        ok = False
    # 用例 4：pinned 中流写入（随仓 example-flow.json：B1 在 S2 写入、S4 消费，
    # 峰值时刻仍在持有）——pinned 只计一次，P = 复用池 2 + B1 = 3
    # （防 pinned 双计回归；该场景即仓库内 example-flow.json 的真值）
    with open(join(dirname(__file__), "example-flow.json"), encoding="utf-8") as f:
        midflow = json.load(f)
    peak2, ps2, _, _, _ = analyze(midflow)
    if peak2 != 3 or ps2 != "S4":
        print(f"FAIL: pinned 中流写入场景 P 期望 3（复用池 2 + 独占 B1），实际 {peak2}（峰值步骤 {ps2}）")
        ok = False
    else:
        print("PASS: pinned 中流写入场景 P = 3（pinned 只计一次，无双计）")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="物理存活节点 P 峰值计算器")
    ap.add_argument("flow", nargs="?", help="融合后物理流 JSON 文件路径")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--demo", action="store_true", help="运行内置已知答案自检")
    args = ap.parse_args()

    if args.demo:
        return demo()
    if not args.flow:
        ap.error("需要 flow 文件路径（或 --demo）")

    with open(args.flow, "r", encoding="utf-8") as f:
        spec = json.load(f)

    try:
        peak, peak_step, trace, warnings, variants = analyze(spec)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    lb = lower_bound_check(spec, peak)
    if args.json:
        print(json.dumps({
            "P": peak, "peak_step": peak_step,
            "warnings": warnings, "lower_bound_check": lb, "trace": trace,
        }, ensure_ascii=False, indent=2))
    else:
        print_report(spec, peak, peak_step, trace, warnings, variants, lb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
