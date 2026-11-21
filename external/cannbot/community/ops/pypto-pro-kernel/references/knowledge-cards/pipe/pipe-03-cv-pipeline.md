---
type: "PyPTO Performance Optimization Card"
title: "C/V 交错流水"
description: "把 Cube 和 Vector 中彼此独立的工作错开执行，让下一轮前部工作和上一轮尾部工作重叠。"
status: "stable"
tags: ["pypto-pro", "pipe", "mixed", "preload", "synchronization"]
item_id: "pipe-03"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/crosscore-sync.md#preload-scheduling-and-slot-lifetime"
      scope: "C/V 交错需要独立工作、跨阶段 slot 生命周期和完整的填充/稳态/排空。"
      boundary: "手工事件序列不等同于自动 stage transform；必须按选定层次验证前向/反向事件、物理槽位与所有边界迭代。"
bound_hint: "PIPELINE"
applicability: "同一个 kernel 同时有 Cube 和 Vector 工作，流水图显示两者整段串行，并且相邻轮次之间存在可证明的独立工作。"
target_api_gate: "需要当前版本支持 Cube/Vector 分区、TileGroup 的 next()、独立 buffer 和跨核事件；具体 API、地址、mutex 和事件语义必须在目标版本复核。"
---

# 技术卡片 pipe-03：C/V 交错流水

## 何时用（诊断特征）

- 循环体按 `C1 → V1 → C2 → V2` 排列，流水图显示每一段都等上一段结束。
- 下一轮的 Cube 前部工作只需要已经准备好的输入，却被上一轮 Vector 尾部挡住。
- 代码虽然声明了多个 buffer，但地址、mutex 或每轮取槽方式让它们实际串行。

## 何时不适用

- 相邻工作有真实的读后写、写后读或写后写依赖。
- 没有独立的物理 buffer，或增加 buffer 后超出片上容量。
- 事件只能放在阶段末尾，无法准确表示数据真正完成的时间。

## 原理

把循环拆成三部分：上一轮的尾部、当前轮的主体、下一轮可以提前做的前部。先发射已经满足输入条件的 Cube 工作，再发射不依赖它的 Vector 工作；下一轮使用另一组物理槽位，等真正需要的数据就绪后再等待事件。

稳态循环之外还要单独处理首轮和末轮：首轮填充流水，末轮排空流水。只改中间循环会漏算最后一轮或提前复用仍在使用的槽位。

## 怎么改（before / after）

下面是能力门控伪码，只描述排布，不保证可直接编译。

```能力门控伪码
# before：每轮严格串行
for work in works:
    C1(work)
    V1(work)
    C2(work)
    V2(work)

# after：先填充，再交错，最后排空
preload(C1(first))
for work in works:
    if has_previous:
        C2(previous)          # 先做上一轮已就绪的 Cube 工作
    V1(work)                  # 与上一轮尾部重叠
    preload(C1(next_work))    # 只预取已证明独立的部分
    if needs_wait(work):
        wait(real_producer_event(work))
    V2(work)
drain(last_work)
```

TileGroup 要用 `next()` 轮转槽位；每个槽位使用独立地址和 mutex。事件要紧跟真正的 move/写入，消费者只等待对应的生产事件。

## 性能与验证指标

重点看 Cube/Vector 重叠比例、事件等待时间、MTE 空泡和 kernel 中位数。正确性要覆盖首轮、稳态、末轮、尾块和多轮并发。

历史资料有过从 350us 降到 207us 的汇总，但没有逐项消融数据，且未在当前版本复现，不能据此承诺收益。

## 技术限制与风险

- `next()` 只负责轮转句柄，不会自动修复地址重叠或 mutex 冲突。
- 事件提前发出会造成脏读，发得太晚则流水仍然串行。
- 额外 buffer 会增加片上存储；容量不足时可能反而变慢。
- 不同设备和工具链对跨核事件、TileGroup 和 C/V mutex 的规则可能不同，使用前必须补证。
