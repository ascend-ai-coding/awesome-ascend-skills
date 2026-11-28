---
type: "PyPTO Performance Optimization Card"
title: "消费者按整块就绪时，改为子块增量累加并跨发布保留部分和"
description: "消费者只在完整归约块就绪后启动、生产者却按子块完成时，把生产者改为子块成组发射，消费者按新增子块在片上累加部分和，最后一块再写出。"
status: "stable"
tags: ["pypto-pro", "pipeline", "cross-engine", "partial-sum"]
item_id: "pipe-08"
bound_hint: "PIPELINE"
applicability: "生产者的完成粒度是子块，消费者的一个输出需要归约多个子块；当前就绪计数按整块推进，单块场景下消费者在整个生产阶段空闲；子块边界可与消费者归约轴切分对齐，部分和能常驻片上累加器"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖片上累加器常驻、pl.matmul/pl.matmul_acc、pl.set_validshape、pl.store，以及 pl.system.set_cross_core/pl.system.wait_cross_core 与 pl.system.sync_all 的发布握手；子块顺序与消费者归约轴切分须在同一 tiling key 内可证明"
---

# 技术卡片 pipe-08：消费者按整块就绪时，改为子块增量累加并跨发布保留部分和

- **适用 bound**：PIPELINE（跨引擎/跨阶段等待）
- **一句话**：把“攒满整块才启动消费者”改为“生产者按归约子块成组发射，消费者按新增子块在片上累加部分和，最后一块再写出”。

## 何时用（诊断特征）

- 生产者每个发布单元完成一个子块；消费者的一个输出块需要对多个子块做结合性归约（如 K 轴累加）。
- 就绪计数按整块推进：`ready = completed // items_per_block`。单块（一块即全部 item）时，`ready` 直到最后一个发布单元才从 0 变 1。
- 生产者尾段与消费者串行，且消费者在总耗时中占可见比例。
- 子块顺序可与消费者归约轴切分对齐；部分和可常驻片上累加器，容量满足每核活跃输出块数。
- 多组场景中消费者已能按组流水时，收益主要出现在单组或组内整块边界。

## 何时不适用

- 消费者必须等全部子块到齐才能开始（需要完整归一化、全量归约等非结合操作），部分和无法增量更新。
- 每核需同时保留多个输出块的部分和，片上容量不足；应先改输出块归属。
- 子块边界与归约轴切分不能对齐（如归约长度不是子块宽度的整数倍），或发布单元无法按子块成组。
- 消费者已按整块发布且组间充分流水，细化就绪粒度不再缩短关键路径。

## 原理

消费者的就绪条件是整块级，生产者的实际完成粒度是子块级。记每块含 `S` 个子块、每个子块有 `P` 个并行项，则 `items_per_block = P*S`。把生产者按消费者归约轴排序（同一子块的所有并行项连续完成），每次发布后都有一组新的完整子块；消费者把这些子块累加进常驻部分和，跨发布不重算、不写回 GM，最后一个子块完成后再写出。于是消费者第 `s` 个发布周期可与生产者第 `s+1` 个周期重叠，原本暴露的生产者尾段被吸收。

发布单元是生产者一次跨核通知覆盖的 item 集合；就绪计数是消费者据此推算已完成 item 数的标量。

这不等于把整块改小：消费者块形状、累加器容量和输出布局都不变，只改生产者发射顺序与消费者就绪粒度。

## 怎么改（before / after）

以下为**结构示意**，只展示发射顺序、就绪计数与累加器生命周期；`pl.range`/`pl.min`/`pl.matmul`/`pl.matmul_acc`/`pl.set_validshape`/`pl.store` 取自已验证 kernel，`produce`/`publish`/`acc_for` 等为占位符。尺寸、子块宽度与容量须按当前算子重算，片段不可直接编译。

**before（整块就绪，单块时整段串行）：**

```结构示意
# producer：按原顺序平铺（如并行项优先），发布单元内混合多个子块
for slot in pl.range(slot_count):
    for item in slot_items(slot):
        produce(item)                       # item -> (parallel_idx, sub_idx)
    publish()

# consumer：整块就绪才启动
for slot in pl.range(slot_count):
    wait_publish()
    done = pl.min((slot + 1) * slot_cap, total_items)
    ready = done // items_per_block         # 单块时最后才由 0 变 1
    for blk in pl.range(prev_ready, ready):
        compute_full_block(blk)             # 从第一个子块重算
```

**after（子块有序 + 增量部分和）：**

```结构示意
# producer：按归约轴成组（item 编号 = sub_idx * parallel_count + parallel_idx）
for slot in pl.range(slot_count):
    for item in slot_items(slot):
        parallel_idx, sub_idx = split(item)
        produce(parallel_idx, sub_idx)
    publish()

# consumer：按新增子块增量累加，部分和常驻片上
for slot in pl.range(slot_count):
    wait_publish()
    done_before = pl.min(slot * slot_cap, total_items)
    done = pl.min((slot + 1) * slot_cap, total_items)
    sub_lo = done_before // parallel_count
    sub_hi = done // parallel_count
    for out_blk in pl.range(core_id, block_count, core_count):
        partial = acc_for(out_blk)
        for sub in pl.range(sub_lo, sub_hi):
            acc_partial(partial, sub)       # 首次使用 matmul，后续 matmul_acc
        if sub_hi == sub_per_block:
            finish_and_store(partial, out_blk)  # set_validshape + store
```

**关键前提：**

- 部分和必须跨发布保留在同一个片上累加器；累加器首次使用时用 `pl.matmul` 初始化，后续用 `pl.matmul_acc`。
- 消费者侧每次发布后需有一次核组内屏障，确保各核在同一发布周期上计算 `done`，否则各核的 `sub_lo/sub_hi` 会错相；混核 kernel 用消费者所在核组的 `pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)`（或对应的 AIV_ONLY），若改用 `MIX` 则 Cube/Vector 两侧都必须调用。
- 最后一个子块的写出条件、某次发布没有新增完整子块（空区间）的情况都要覆盖；缺少最终写出会静默丢结果。
- 生产者重排会改变 item 的全局编号；按 item 直接寻址的状态/输出必须改为显式 `(parallel_idx, sub_idx)`。

## 性能与验证指标

- 指标：消费者首次启动时刻、消费者空闲区间、生产者尾段暴露时间、`Task Duration`；同时记录片上累加器占用和每核活跃输出块数。
- 正确性：覆盖首个子块、最后一个子块写出、空发布区间、发布单元跨子块组、不同核数、每核多个输出块；部分和跨发布保留时，输出须与整块一次计算在容差内一致。
- 测量：在目标形态固定输入与采样协议，对比启用子块增量输出前后的消费者启动时刻与端到端耗时；收益随核数与组数变化，须 A/B 复测。

## 技术限制与风险

- 每核活跃输出块数 × 部分和容量不得超过片上累加器；单核多块时先改块归属。
- 部分和在发布之间不得被其他阶段复用同一累加器地址。
- 发布单元可跨子块组，`done // parallel_count` 只统计完整组；若生产者不是严格按序完成，不能直接用全局 `done`，需改为按组完成计数。
- 该改动不改变数值顺序（仍按子块顺序累加）；若原实现累加顺序不同，须重新过精度门。
- 与相邻卡片：`pipe-10` 只扫描槽位/发布粒度，不改变整块就绪条件；`pipe-02`/`pipe-03` 重排或交错既有阶段；`cube-03` 跨核切 K 并以原子加合并部分和。本卡改就绪粒度并在消费者核内跨发布累加，不引入原子写。

## 参考资料

- 相邻卡片：`pipe-10`（槽位/发布粒度扫描）、`pipe-02`/`pipe-03`（阶段重排与交错）、`cube-03`（跨核 K 切分部分和）。
- 验证案例：`ops/pro/kimi_k3/mega_recurrent_kda_replayssm`（`DESIGN.md` §5.2.1）的单组 head 区间增量输出，使用 `pl.matmul_acc`、常驻 L0C 部分和与 `sync_all(AIC_ONLY)`；收益随 batch/核数变化，未在本卡泛化。
