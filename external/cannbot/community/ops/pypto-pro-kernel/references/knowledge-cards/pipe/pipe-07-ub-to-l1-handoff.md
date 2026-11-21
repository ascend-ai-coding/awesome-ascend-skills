---
type: "PyPTO Performance Optimization Card"
title: "UB 直接交给 L1，少一次 GM 往返"
description: "Vector 产出的 tile 如果马上给 Cube 使用，可以直接写入 L1，省掉中间的 GM 写回和再次读取。"
status: "stable"
tags: ["pypto-pro", "pipe", "cross-engine", "memory"]
item_id: "pipe-07"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#transformed-operands-format-residency-and-repeated-work"
      scope: "Vector 产生 Cube 操作数后，通过片上驻留和交接减少中间 GM 写回、读回。"
      boundary: "目标章节还讨论生产者复用、所有权和物化方案的权衡；本卡不与整个章节等价。若 GM 副本还有其他消费者，必要写回必须保留。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:ub-l1"
      scope: "核对 Vector 产物经 UB 交给 L1/Cube 时的物理表示、布局、有效范围和消费同步。"
      boundary: "该关联不证明当前布局和工具链已经验证；本卡删除 GM 往返，与目标内 ub-l1-extra-layout-move 候选删除局部 ND→NZ 转换是不同改法。"
bound_hint: "VEC、MTE2、MTE3"
applicability: "Vector 和 Cube 在同一次 kernel 中前后相接，Vector 的结果只需被后面的 Cube 消费，且 GM 读写位于关键路径。"
target_api_gate: "需要当前版本支持 UB 到 L1 的布局转换、pl.insert、valid shape 和跨核事件；具体 tile、地址和事件语义必须在目标设备上复核。"
---

# 技术卡片 pipe-07：UB 直接交给 L1，少一次 GM 往返

## 何时用（诊断特征）

- Vector 算出一个 tile，紧接着由 Cube 当作 matmul 输入。
- 现在的路径是 `UB → GM → L1`，GM 的读回落在事件等待之后，拖慢了关键路径。
- 中间转换和写入 L1 的成本小于省掉的 GM 往返。

如果 GM 副本还有其他消费者，保留 GM 写回，只删除 Cube 侧那次不必要的 GM 读取。

## 何时不适用

- 生产者和消费者不在同一次 kernel，或跨核之间没有可用的共享 L1。
- GM 读写不在关键路径，新增 staging 只会增加工作。
- L1 容量、地址轮转或同步条件无法满足。

## 原理

Vector 侧先把 ND tile 转成 Cube 需要的 NZ 形式，再写入 L1 槽位；Cube 等待生产事件后，直接从 L1 搬到 L0A/L0B。这样把一段 `GM 写 + GM 读` 换成片上搬运。

同一个 L1 槽位跨轮复用时，要证明上一轮已经读完；证明不了就使用两个独立槽位。事件应在真正的数据写完后立即发出，并在 Cube 第一次读 L1 前等待。

## 怎么改（before / after）

下面是能力门控伪码，只说明数据流，不能直接编译。`pl.insert`、布局转换、事件和 tile 声明要按目标版本补证。

```能力门控伪码
# before
vec_result = compute()
pl.store(gm, vec_result)
set_event(E)
wait_event(E)
cube_input = pl.load(l1, gm)

# after
vec_result = compute()
nz = convert_nd_to_nz(vec_result)
pl.insert(l1_slot, nz, offset)
set_event_after_l1_write(E)
wait_event_before_l1_read(E)
cube_input = pl.move(l0, l1_slot)
```

需要同时检查：

1. 源 tile、转换 tile 和 L1 槽位的 valid shape 与布局声明一致；
2. 两个 Vector subblock 的写入范围不重叠；
3. L1 槽位、mutex 和事件在相邻轮次中不会发生 WAW/WAR 冲突。

## 性能与验证指标

重点看 GM 读写时间、MTE3/MTE2 空泡、事件等待和端到端 kernel 时间。对比原路径和直通路径时，固定设备、shape、dtype 和采样方法，覆盖完整 tile 和尾块。

历史资料显示小 tile、串行链上的收益更容易出现；这些数据未在当前算子复现，不能直接当作收益承诺。

正确性至少要覆盖多轮并发、尾块和不同 subblock 分配，并核对中间 tile 的布局和值。

## 技术限制与风险

- `pl.insert` 不会自动把 ND 数据变成 NZ；缺少转换会产生稳定的布局错误。
- L1 目的 tile 通常不能像 UB tile 一样随意切片，半片写入需要用合法的 insert 形式。
- 增加槽位会占用 L1；地址相同或 mutex 相同会把流水重新串行化。
- 950PR、其他版本和不同布局的支持情况待验证。
