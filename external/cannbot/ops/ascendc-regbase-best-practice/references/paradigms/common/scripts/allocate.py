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

"""allocate.py — UB buffer 槽位分配器（贪心配槽）。

用途：
  VF 段编排 + 融合方案定稿后，给定调度顺序（steps 顺序即调度决策）、独占分配
  tensor（pinned，语义决策）与互斥分支声明（exclusive_branches），程序化生成
  复用槽位映射（B0..Bn），并输出 B 名替换后的物理流 JSON（直接作为
  physical_p.py 的输入）。分配只做"执行"，不做"决策"：调度顺序、pinned
  集合、分支结构必须由设计侧（LLM）给出，脚本对给定输入产出确定性映射。

  贪心策略：按 steps 顺序登场，写入 tensor 时选编号最小且当前内容已死的槽
  复用；无可用槽则新开。直线流下该策略即最少槽位（区间图按登场顺序贪心
  最优）；互斥分支（Tuple Reduce）各分支共用物理槽，P 峰值取各变体 max 不
  叠加。槽可用性按**变体视角**判定——回溯该变体可见的最后一次写入（公共 +
  本分支），异分支写入视为不存在；公共步骤的写须在**所有**变体中安全，分支
  写只须在本分支变体中安全。选槽即 WAR 检查——仍有未来消费者的内容不会
  被判死，正确性由构造保证。本步内不假设 in-place（同一步骤读过的槽不承担
  该步的写入）。

输入（JSON，stdin 或文件；tensor 级物理流，单赋值 SSA）：
  {
    "inputs":  ["x", "gamma"],            // GM 输入 tensor 名（透传，供下游下界自检）
    "outputs": ["y"],                     // 输出 tensor 名（透传，供下游下界自检）
    "pinned":  ["mu"],                    // 独占分配 tensor（全程占 1 槽，不复用）
    "steps": [
      {"id": "S1", "op": "CopyIn x", "reads": [], "writes": ["x"]},
      {"id": "S2", "op": "ReduceMean(axis=-1)", "reads": ["x"], "writes": ["mu"]},
      ...
    ],
    "exclusive_branches": [["T1S1", "T1S2"], ["T2S1"]]   // 可选：互斥分支（按 step id）
  }

  约束：
    1. 每个 tensor 只允许被写入一次（单赋值）；GM 输入也须先 CopyIn（reads
       必须引用此前已写入的 tensor）。
    2. 分支内 tensor 只允许在同一分支内消费——公共步骤或异分支读取分支内
       tensor 属于模型违例（该依赖应把消费步骤声明进同一分支）。
    3. 跨阶段驻留量（归属其他阶段分析的 tensor，如驻 cacheBuf 的归约结果）
       不进本阶段输入，按范式文档口径归属其分配阶段。
    4. 无消费者的产出（悬空 tensor）其槽**不复用、占槽到流程结束**——三脚本
       统一口径（liveness/physical_p「持有到流程结束」、allocate blocked()
       恒阻断），防悬空漏 CopyOut 导致 P 低估（同一份流 allocate 槽数 <
       liveness L 的不变量破坏）；真实输出应补 CopyOut 消费步骤。
    5. 同一步骤多产出（如 VF 链多消费者物化 writes 两个 tensor）各占独立
       槽位——前序产出仍有未来消费者时本步骤后续写入不得复用其槽。

输出：
  - 默认：槽位映射表（B0: t1 → t2 → ...，标注 pinned / 分支）+ 槽位数
  - --out FILE：B 名替换后的物理流 JSON（physical_p.py 输入；pinned 同步译为 B 名）
  - --json：机器可读（mapping / slots / flow / 由 physical_p 语义复算的 P 峰值）
  - --demo：内置已知答案自检（锚定 reduction-binary-base-dag-buffers §2 与
    scripts/example-flow.json 两组文档真值，另含互斥分支共享、坏调度防呆、
    跨分支变体视角、同步骤多产出四类防回归用例）

用法：
  python3 allocate.py flow.json
  python3 allocate.py flow.json --out flow.allocated.json
  python3 allocate.py --demo
"""
import argparse
import json
import sys
from physical_p import analyze as pp_analyze  # P 峰值语义复用：分配器出映射，physical_p 当裁判


def load_spec(path):
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def allocate(spec):
    """对 tensor 级物理流做贪心配槽。

    返回 (slot_of, slot_order, slot_count, pinned_tensors, branch_of)。
    slot_of: tensor -> 槽名；slot_order: 槽名 -> 按登场顺序的 tensor 列表。
    语义决策（调度顺序 = steps 顺序、pinned、exclusive_branches）全部来自输入。
    """
    steps = spec["steps"]
    pinned_tensors = list(dict.fromkeys(spec.get("pinned", [])))
    branches = spec.get("exclusive_branches") or []

    branch_of = {}  # step id -> 分支编号
    step_ids = {st["id"] for st in steps}
    for bi, br in enumerate(branches):
        for sid in br:
            if sid not in step_ids:
                raise ValueError(
                    f"exclusive_branches 分支{bi} 的 step id '{sid}' 不存在于 steps——"
                    f"拼错的 id 会静默漏跑步骤、P 被低估且无报错"
                )
            if sid in branch_of:
                raise ValueError(f"step '{sid}' 同时属于多个互斥分支")
            branch_of[sid] = bi

    # ---- 结构校验 ----
    written_by = {}  # tensor -> 写入它的 step 下标
    for idx, st in enumerate(steps):
        for t in st.get("writes", []):
            if t in written_by:
                raise ValueError(
                    f"step {st['id']}: tensor '{t}' 被写入多次——输入须为单赋值（SSA）形态，"
                    f"循环累积请改写为逐轮新 tensor"
                )
            written_by[t] = idx
    for t in pinned_tensors:
        if t not in written_by:
            raise ValueError(f"pinned tensor '{t}' 从未被写入——独占槽必须有产出的内容")

    def variant_of(st):
        return branch_of.get(st["id"], "common")

    # 公共/异分支读取分支内 tensor → 模型违例
    for st in steps:
        v_reader = variant_of(st)
        for t in st.get("reads", []):
            w = written_by.get(t)
            if w is None:
                continue  # 读未写入留给顺序校验报错
            v_writer = variant_of(steps[w])
            if v_writer != "common" and v_writer != v_reader:
                raise ValueError(
                    f"step {st['id']}（{'公共步骤' if v_reader == 'common' else f'分支{v_reader}'}）"
                    f"读取分支{v_writer}内 tensor '{t}'——分支内产出只允许同分支消费，"
                    f"请把该消费步骤声明进分支 {v_writer}"
                )

    # ---- 消费者表：tensor -> [(step 下标, 变体)] ----
    consumers = {}
    for idx, st in enumerate(steps):
        v = variant_of(st)
        for t in st.get("reads", []):
            consumers.setdefault(t, []).append((idx, v))

    def blocked(content, cur_idx, cur_reads, allowed_variants):
        """槽内内容 content 在变体视角下是否阻断复用。

        阻断条件：① 本步骤正在读它（不假设同步骤 in-place）；② 在允许变体
        集内存在下标更大的未来消费者；③ 悬空产出（**变体视角**：在允许变体集内无任何消费者，含仅异分支消费的分支内悬空）恒阻断——
        与 liveness/physical_p「无消费者的 tensor 持有到流程结束」口径一致
        （保守口径防 P 低估；真实输出应补 CopyOut 步骤，见报告警告）。
        allowed_variants = {common, cv}：公共
        消费者在任何变体都执行；分支消费者只在自身分支变体执行——互斥分支
        的内容在异分支语境下不阻断（任何执行中二者不同时存活）。
        """
        if content in cur_reads:
            return True
        # 悬空判断取变体视角：在允许变体集内无任何消费者才算悬空——
        # "仅异分支消费"的值在本变体同样是悬空（分支内悬空也恒阻断），
        # 与 liveness（分支投影后计消费者、悬空持有）口径一致
        if not any(vj in allowed_variants for _, vj in consumers.get(content, ())):
            return True
        return any(
            j > cur_idx and vj in allowed_variants
            for j, vj in consumers.get(content, ())
        )

    # ---- 贪心配槽（单趟线性扫描；槽可用性按变体视角判定）----
    slot_of = {}         # tensor -> 槽名
    slot_history = {}    # 槽名 -> [(step 下标, tensor, 写入变体)] 按线性序追加
    slot_pinned = set()  # 独占槽，永不参与复用
    slot_order = {}      # 槽名 -> 登场 tensor 序列（报表用）
    next_slot = 0

    def content_in(slot, cv, cur_idx):
        """变体 cv 的执行中、当前分配点之前 slot 的最后内容；无则 None。

        只回溯该变体可见的写入（公共写入 + 分支 cv 写入）——异分支写入在
        变体 cv 的执行中不发生，槽内容保持其之前的公共/本分支内容。分支 0
        覆写过的槽，在分支 1 视角下仍是覆写前的内容。j <= cur_idx：同一步骤
        内更早的写入（多产出步骤，如 VF 链多消费者物化）是当前内容，对后续
        写入的选槽可见——否则同步骤第二产出会复用第一产出的槽并覆写它。
        """
        for j, t, vj in reversed(slot_history[slot]):
            if (vj == "common" or vj == cv) and j <= cur_idx:
                return t
        return None

    for idx, st in enumerate(steps):
        reads = st.get("reads", [])
        writes = st.get("writes", [])
        # 顺序校验：读先于写
        for t in reads:
            if t not in slot_of:
                raise ValueError(f"step {st['id']}: 读取未写入的 tensor '{t}'——顺序违规（GM 输入须先 CopyIn）")
        v = variant_of(st)
        # 复用安全性须在写入实际发生的每个执行变体中成立：
        # 公共写在所有变体执行；分支写仅在本分支变体执行
        if branches:
            check_variants = list(range(len(branches))) if v == "common" else [v]
        else:
            check_variants = ["common"]
        for t in writes:
            if t in pinned_tensors:
                slot = f"B{next_slot}"
                next_slot += 1
                slot_pinned.add(slot)
            else:
                slot = None
                for k in range(next_slot):
                    cand = f"B{k}"
                    if cand in slot_pinned:
                        continue
                    usable = True
                    for cv in check_variants:
                        content = content_in(cand, cv, idx)
                        if content is not None and blocked(content, idx, reads, {"common", cv}):
                            usable = False
                            break
                    if usable:
                        slot = cand
                        break
                if slot is None:
                    slot = f"B{next_slot}"
                    next_slot += 1
            slot_of[t] = slot
            slot_history.setdefault(slot, []).append((idx, t, v))
            slot_order.setdefault(slot, []).append(t)

    return slot_of, slot_order, next_slot, pinned_tensors, branch_of


def build_flow(spec, slot_of, pinned_tensors):
    """生成 B 名替换后的物理流 JSON（physical_p.py 输入）。"""
    steps = []
    for st in spec["steps"]:
        reads = [slot_of[t] for t in st.get("reads", [])]
        writes = [slot_of[t] for t in st.get("writes", [])]
        op = st.get("op", "")
        if writes:
            op = f"{op} → {','.join(writes)}"
        steps.append({"id": st["id"], "op": op, "reads": reads, "writes": writes})
    flow = {
        "inputs": spec.get("inputs", []),
        "outputs": spec.get("outputs", []),
        "pinned": [slot_of[t] for t in pinned_tensors],
        "steps": steps,
    }
    if spec.get("exclusive_branches"):
        flow["exclusive_branches"] = spec["exclusive_branches"]
    return flow


def print_report(spec, slot_of, slot_order, slot_count, pinned_tensors, branch_of, peak, peak_step, variants):
    print(f"=== 贪心配槽结果（{variants} 个变体取 max） ===")
    # tensor -> 写入步骤（分支标注用）
    writer_step = {}
    for st in spec["steps"]:
        for t in st.get("writes", []):
            writer_step[t] = st["id"]
    for k in range(slot_count):
        slot = f"B{k}"
        entries = []
        for t in slot_order.get(slot, ()):
            tag = ""
            v = branch_of.get(writer_step.get(t, ""))
            if v is not None:
                tag = f"[分支{v}]"
            entries.append(f"{t}{tag}")
        pin = "（pinned 独占）" if slot in {slot_of[t] for t in pinned_tensors} else ""
        print(f"  {slot}: {' → '.join(entries) if entries else '（空）'}{pin}")
    print(f"\n槽位数 = {slot_count}，其中 pinned 独占 {len(pinned_tensors)} 个")
    if peak is not None and slot_count > peak:
        print(f"⚠ 槽位数 {slot_count} > 变体峰值 P {peak}（分支值跨公共链存活时贪心映射须对全部分支安全，"
              f"新开槽多于任一分支实际需要）——UB 容量预算必须按实际槽位数 {slot_count} 而非 P {peak} 计算"
              f"（perBufBytes = UB / {slot_count}）；如需收敛槽位可优化调度顺序使分支结果更早消费")
    print(f"P 峰值 = {peak}（峰值步骤 {peak_step}，physical_p 语义复算，含 pinned）")
    print("→ 请用 --out 落盘物理流并交 physical_p.py 复核（P / WAR / 下界自检）")


DEMO_PRE = {
    # reduction-binary-base-dag-buffers §2 SyncBatchNormGatherStatsWithCounts
    # preReducePhase（融合后物理流，tensor 级）：文档已知答案
    # B0: mean_all → invert_std_all → mean_var_sum；B1: mean_broadcast →
    # var_all_square_ε → count_all；B2: mean_var；B3(pinned): mean_var_count；P_pre = 4
    "inputs": ["mean_all", "mean_broadcast", "invert_std_all", "count_all"],
    "outputs": ["mean_var_count"],
    "pinned": ["mean_var_count"],
    "steps": [
        {"id": "S6", "op": "CopyIn mean_all", "reads": [], "writes": ["mean_all"]},
        {"id": "S7", "op": "CopyIn mean_broadcast", "reads": [], "writes": ["mean_broadcast"]},
        {"id": "S8-S9", "op": "VF: Sub+Square", "reads": ["mean_all", "mean_broadcast"], "writes": ["mean_var"]},
        {"id": "S1", "op": "CopyIn invert_std_all", "reads": [], "writes": ["invert_std_all"]},
        {"id": "S2-S5", "op": "VF: Cast+Div+Square+Sub", "reads": ["invert_std_all"], "writes": ["var_all_square_ε"]},
        {"id": "S10", "op": "Add", "reads": ["mean_var", "var_all_square_ε"], "writes": ["mean_var_sum"]},
        {"id": "S11", "op": "CopyIn count_all", "reads": [], "writes": ["count_all"]},
        {"id": "S12", "op": "Mul", "reads": ["mean_var_sum", "count_all"], "writes": ["mean_var_count"]},
    ],
}

DEMO_MEAN_GAMMA = {
    # scripts/README §从伪码到 JSON：y = γ · mean(x)（mu 落独占 cacheBuf）
    # 期望与 example-flow.json 一致：B0: x→gamma；B1(pinned): mu；B2: y；P = 3
    "inputs": ["x", "gamma"],
    "outputs": ["y"],
    "pinned": ["mu"],
    "steps": [
        {"id": "S1", "op": "CopyIn x", "reads": [], "writes": ["x"]},
        {"id": "S2", "op": "ReduceMean(axis=-1)", "reads": ["x"], "writes": ["mu"]},
        {"id": "S3", "op": "CopyIn gamma", "reads": [], "writes": ["gamma"]},
        {"id": "S4", "op": "VF: Mul(brc)", "reads": ["gamma", "mu"], "writes": ["y"]},
    ],
}

DEMO_BRANCHES = {
    # Tuple Reduce 互斥分支：两分支共用物理槽（t1/t2 不同时存活），P 取 max 不叠加
    "inputs": ["x", "w"],
    "outputs": ["o1", "o2"],
    "pinned": [],
    "steps": [
        {"id": "C1", "op": "CopyIn x", "reads": [], "writes": ["x"]},
        {"id": "C2", "op": "CopyIn w", "reads": [], "writes": ["w"]},
        {"id": "T1a", "op": "VF: f1(x,w)", "reads": ["x", "w"], "writes": ["t1"]},
        {"id": "T1b", "op": "CopyOut o1", "reads": ["t1"], "writes": []},
        {"id": "T2a", "op": "VF: f2(x,w)", "reads": ["x", "w"], "writes": ["t2"]},
        {"id": "T2b", "op": "CopyOut o2", "reads": ["t2"], "writes": []},
    ],
    "exclusive_branches": [["T1a", "T1b"], ["T2a", "T2b"]],
}


def demo():
    ok = True
    # 用例 1：SyncBatchNorm preReducePhase —— 复刻文档 §2 复用映射与 P_pre = 4
    slot_of, order, _, pinned, _ = allocate(DEMO_PRE)
    expect = {
        "B0": ["mean_all", "invert_std_all", "mean_var_sum"],
        "B1": ["mean_broadcast", "var_all_square_ε", "count_all"],
        "B2": ["mean_var"],
        "B3": ["mean_var_count"],
    }
    if order != expect:
        print(f"FAIL: SyncBatchNorm 映射期望 {expect}，实际 {order}")
        ok = False
    else:
        print("PASS: SyncBatchNorm pre 映射 = 文档 reduction-binary-base-dag-buffers §2 已知答案（B0/B1/B2 池 + B3 独占）")
    flow = build_flow(DEMO_PRE, slot_of, pinned)
    peak, ps, _, warns, _ = pp_analyze(flow)
    if peak != 4 or warns:
        print(f"FAIL: SyncBatchNorm P 期望 4（无 WAR），实际 {peak}，warns={warns}")
        ok = False
    else:
        print("PASS: SyncBatchNorm P_pre = 4，无 WAR 违规（与 §2 P 结论表一致）")

    # 用例 2：y = γ·mean(x) —— 生成流与随仓 example-flow.json 同构，P = 3
    slot_of, order, _, pinned, _ = allocate(DEMO_MEAN_GAMMA)
    flow = build_flow(DEMO_MEAN_GAMMA, slot_of, pinned)
    expect = {"B0": ["x", "gamma"], "B1": ["mu"], "B2": ["y"]}
    if order != expect or flow_pinned(flow) != ["B1"]:
        print(f"FAIL: γ·mean(x) 映射期望 {expect}（pinned→B1），实际 {order}，pinned={flow_pinned(flow)}")
        ok = False
    else:
        print("PASS: γ·mean(x) 映射 = example-flow.json 真值（B0: x→gamma 复用 + B1 独占 + B2）")
    peak, ps, _, warns, _ = pp_analyze(flow)
    if peak != 3 or ps != "S4" or warns:
        print(f"FAIL: γ·mean(x) P 期望 3（峰值 S4，无 WAR），实际 {peak}@{ps}，warns={warns}")
        ok = False
    else:
        print("PASS: γ·mean(x) P = 3（峰值 S4），与 example-flow.json 复算一致")

    # 用例 3：互斥分支共享物理槽，P 取 max 不叠加
    slot_of, order, _, pinned, _ = allocate(DEMO_BRANCHES)
    if slot_of["t1"] != slot_of["t2"]:
        print(f"FAIL: 互斥分支应共享槽（t1/t2 同槽），实际 t1→{slot_of['t1']} t2→{slot_of['t2']}")
        ok = False
    else:
        print("PASS: 互斥分支 t1/t2 共享同一物理槽（不同时存活）")
    flow = build_flow(DEMO_BRANCHES, slot_of, pinned)
    peak, ps, _, warns, _ = pp_analyze(flow)
    if peak != 3 or warns:
        print(f"FAIL: 互斥分支 P 期望 3（max 不叠加，无 WAR），实际 {peak}，warns={warns}")
        ok = False
    else:
        print("PASS: 互斥分支 P = 3（两变体各峰值 3 取 max，非 4 叠加）")

    # 用例 4：WAR 防呆——调度顺序违规（覆写仍有未来消费者的内容）应被分配器避开：
    # 把 S3（CopyIn gamma）提到 S2（ReduceMean 读 x）之前，B0 仍承载 x，
    # gamma 只能新开槽——映射自动修正，不会产生 WAR
    spec_war = {
        "inputs": ["x", "gamma"], "outputs": ["y"], "pinned": ["mu"],
        "steps": [
            {"id": "S1", "op": "CopyIn x", "reads": [], "writes": ["x"]},
            {"id": "S3", "op": "CopyIn gamma", "reads": [], "writes": ["gamma"]},
            {"id": "S2", "op": "ReduceMean(axis=-1)", "reads": ["x"], "writes": ["mu"]},
            {"id": "S4", "op": "VF: Mul(brc)", "reads": ["gamma", "mu"], "writes": ["y"]},
        ],
    }
    slot_of, order, _, pinned, _ = allocate(spec_war)
    flow = build_flow(spec_war, slot_of, pinned)
    _, _, _, warns, _ = pp_analyze(flow)
    if slot_of["gamma"] == "B0" or warns:
        print(f"FAIL: 坏调度下应避开 B0（x 仍活）且无 WAR，实际 gamma→{slot_of['gamma']}，warns={warns}")
        ok = False
    else:
        print("PASS: 坏调度自动避开仍活内容的槽（构造性 WAR 防呆），physical_p 复核无违规")

    # 用例 5：跨分支公共槽防呆（变体视角，含分支内悬空恒阻断）——两层断言：
    # ① 变体 0：c1 的消费者（T2b）在分支 1，变体 0 视角下 c1 为分支内悬空
    #    → 恒阻断 → t1 不得复用 B0（新开 B1）；
    # ② 变体 1：c1 有未来消费者（T2b）→ t2 亦避开 B0（B1 在变体 1 视角
    #    为异分支槽/空槽，可用）。B 名流时序自洽，映射错误对 physical_p
    #    不可见（tensor 身份已丢失），只能靠映射断言捕获
    spec_cross = {
        "inputs": ["c1"], "outputs": ["o1", "o2"], "pinned": [],
        "steps": [
            {"id": "C1", "op": "CopyIn c1", "reads": [], "writes": ["c1"]},
            {"id": "T1a", "op": "VF: f1()", "reads": [], "writes": ["t1"]},
            {"id": "T1b", "op": "CopyOut o1", "reads": ["t1"], "writes": []},
            {"id": "T2a", "op": "VF: f2()", "reads": [], "writes": ["t2"]},
            {"id": "T2b", "op": "CopyOut o2(c1,t2)", "reads": ["c1", "t2"], "writes": []},
        ],
        "exclusive_branches": [["T1a", "T1b"], ["T2a", "T2b"]],
    }
    slot_of, order, _, pinned, _ = allocate(spec_cross)
    flow = build_flow(spec_cross, slot_of, pinned)
    _, _, _, warns, _ = pp_analyze(flow)
    if slot_of["t1"] == "B0" or slot_of["t2"] == "B0" or warns:
        print(f"FAIL: 跨分支场景期望 t1/t2 均避开 B0（变体0悬空阻断 + 变体1未来消费者阻断），"
              f"实际 t1→{slot_of['t1']} t2→{slot_of['t2']}，warns={warns}")
        ok = False
    else:
        print("PASS: 跨分支公共槽防呆——t1 避开 B0（变体0悬空恒阻断）且 t2 避开 B0（变体1未来消费者），无 WAR")

    # 用例 6：同步骤多产出防呆——VF 链 writes [out, mid]（多消费者物化），
    # out 有未来消费者（S4 CopyOut）：mid 不得复用 out 刚分到的槽
    spec_multi = {
        "inputs": ["x"], "outputs": ["out", "mid"], "pinned": [],
        "steps": [
            {"id": "S1", "op": "CopyIn x", "reads": [], "writes": ["x"]},
            {"id": "S2", "op": "VF: chain (multi-output)", "reads": ["x"], "writes": ["out", "mid"]},
            {"id": "S3", "op": "consume mid", "reads": ["mid"], "writes": []},
            {"id": "S4", "op": "CopyOut out", "reads": ["out"], "writes": []},
        ],
    }
    slot_of, order, _, pinned, _ = allocate(spec_multi)
    flow = build_flow(spec_multi, slot_of, pinned)
    peak, _, _, warns, _ = pp_analyze(flow)
    if slot_of["out"] == slot_of["mid"] or warns:
        print(f"FAIL: 同步骤多产出期望 out/mid 异槽且无 WAR，实际 out→{slot_of['out']} "
              f"mid→{slot_of['mid']}，warns={warns}")
        ok = False
    else:
        print(f"PASS: 同步骤多产出各占独立槽（out→{slot_of['out']}，mid→{slot_of['mid']}），无 WAR")
    return 0 if ok else 1


def flow_pinned(flow):
    return flow.get("pinned", [])


def main():
    ap = argparse.ArgumentParser(description="UB buffer 槽位分配器（贪心配槽）")
    ap.add_argument("spec", nargs="?", help="tensor 级物理流 JSON 文件路径（- 读 stdin）")
    ap.add_argument("--out", help="B 名物理流输出文件（physical_p.py 输入）")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--demo", action="store_true", help="运行内置已知答案自检")
    args = ap.parse_args()

    if args.demo:
        return demo()
    if not args.spec:
        ap.error("需要 spec 文件路径（或 --demo）")

    spec = load_spec(args.spec)
    try:
        slot_of, slot_order, slot_count, pinned_tensors, branch_of = allocate(spec)
        flow = build_flow(spec, slot_of, pinned_tensors)
        peak, peak_step, _, warns, variants = pp_analyze(flow)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if warns:
        for w in warns:
            print(f"⚠ {w}", file=sys.stderr)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(flow, f, ensure_ascii=False, indent=2)
            f.write("\n")

    if args.json:
        print(json.dumps({
            "slots": slot_count,
            "P": peak, "peak_step": peak_step,
            "warnings": warns,
            "mapping": {k: slot_order[k] for k in sorted(slot_order, key=lambda s: int(s[1:]))},
            "flow": flow,
        }, ensure_ascii=False, indent=2))
    else:
        print_report(spec, slot_of, slot_order, slot_count, pinned_tensors,
                     branch_of, peak, peak_step, variants)
        if args.out:
            print(f"物理流已写出：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
