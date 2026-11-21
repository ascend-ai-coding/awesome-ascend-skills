---
type: "PyPTO Performance Optimization Card"
title: "跨迭代预取下一项，并用编译期变体选择"
description: "迭代边界前的输入/状态装载落在关键路径时，把下一迭代的装载移到当前迭代最后一个消费者之后；预取只在部分网格有利时，用编译期变体选择预取/直载，而不是在热循环里保留运行时分支。"
status: "stable"
tags: ["pypto-pro", "pipeline", "prefetch", "tiling-key"]
item_id: "pipe-09"
bound_hint: "PIPELINE、MTE2"
applicability: "每个迭代开头有一批固定输入/状态装载，且这些缓冲在上一迭代最后一个消费者之后已可覆盖；下一迭代索引可由当前迭代单调推出；预取收益只在部分核数/形状成立"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖轮转/同址复用缓冲、pl.load/pl.cast、auto_mutex，以及 TilingKeyField 等编译期变体；预取目标必须是上一迭代已释放的缓冲"
---

# 技术卡片 pipe-09：跨迭代预取下一项，并用编译期变体选择

- **适用 bound**：PIPELINE、MTE2
- **一句话**：把下一迭代的输入/状态装载提前到当前迭代尾部，隐藏迭代边界的 MTE2 延迟；只在部分网格有利时，用编译期变体切换预取/直载，而不是在热循环里判断。

## 何时用（诊断特征）

- 每次迭代开头固定装载一批输入/状态，装载量与迭代计算量可比。
- 目标缓冲在当前迭代最后一个消费者之后已空闲，下一迭代索引可由当前迭代单调推出。
- trace 在迭代边界出现 MTE2 空泡，或 MTE2 时间与总时间可比。
- 预取收益随核数/形状变化：部分配置稳定受益，部分配置中性或回退。

## 何时不适用

- 下一迭代索引依赖运行时数据，或下一迭代可能被跳过。
- 预取目标缓冲仍被当前迭代尾部使用（如状态在最后一次更新时仍在读）。
- 预取显著提高峰值片上占用，挤压并行阶段的轮转槽。
- 单次迭代很短，预取的 setup/同步成本大于可隐藏的延迟。
- 所有配置都稳定受益或都无收益：不需要编译期变体。

## 原理

迭代边界的装载原本夹在“上一迭代完成 → 下一迭代第一个消费者”之间。把装载移到上一迭代最后一个消费者之后，MTE2 与当前迭代尾部的计算/写出重叠，下一迭代开始前数据已就绪。关键是逐个缓冲确认最后一个消费者：状态在最后一次更新之后、输入在最后一次读取之后、复用的输出缓冲在最后一次写出之后。

开关方式同样重要。把 `if prefetch:` 留在热循环里，会让编译器为整个迭代体保留两条路径的寄存器、分支和调度约束；即使某条路径最终不执行，也可能拖慢该路径。把开关提到编译期（tiling key / 编译期常量），parser 只生成一条路径；不需要预取的配置保留独立的、无分支的直载函数。

## 怎么改（before / after）

以下为**结构示意**，只展示装载位置与变体选择；`pl.*` 形态取自已验证 kernel，`stage`/`compute`/`emit`/`next_item` 为占位符，缓冲生命周期、stride 与峰值占用须按当前算子核验，片段不可直接编译。

**before（装载在下一迭代开头）：**

```结构示意
for it in iterations(worker):
    stage(it)                 # 迭代边界：MTE2 落在关键路径
    compute(it)
    emit(it)
```

**after（上一迭代尾部预取 + 编译期变体）：**

```结构示意
if PREFETCH == 1:             # 编译期常量/TilingKey 字段，只保留一条路径
    loop_prefetch(worker)
else:
    loop_direct(worker)       # 无预取分支的直载路径

def loop_prefetch(worker):
    stage(first_item(worker)) # prologue：只对有效 worker 发射
    for it in iterations(worker):
        compute(it)           # 消费已装载的输入/状态
        nxt = next_item(it)
        if nxt < total:
            stage(nxt)        # 目标缓冲已被 compute 释放
        emit(it)              # 与 stage(nxt) 的 MTE2 重叠
```

**关键前提：**

- `stage(nxt)` 必须排在该缓冲最后一个消费者之后；提前会覆盖仍在读的数据。
- `next_item` 的 stride 必须由当前迭代单调推出；跨 slot/分组时要证明 stride 不变。
- prologue 只对有效 worker 发射；空 worker 不能越界读。
- 预取与当前迭代尾部共享片上空间时，必须错开地址或延后预取；否则峰值占用会回退。

## 性能与验证指标

- 指标：MTE2 忙碌/空泡、迭代边界空泡、`Task Duration`、峰值片上占用；同时确认预取没有挤压轮转槽。
- 正确性：覆盖首迭代、末迭代、空 worker、跨 slot 边界、不同 stride 与峰值占用临界配置；预取/直载两版输出须一致。
- 测量：在目标形态固定输入与采样协议，对比预取/直载两版；运行时条件版本在部分形态可能负收益，须 A/B 复测。

## 技术限制与风险

- 预取与当前迭代尾部共享片上空间；尾部仍占用同一缓冲时必须错开地址或延后。
- 运行时 `if` 在热循环内可能改变寄存器分配与指令调度；不要用“功能等价”判断其成本。
- 编译期变体会增加 tiling key 组合与编译缓存条目；须确认 key 位宽与缓存策略可接受。
- 预取的 `next_item` 索引必须与状态/输出寻址一致；迭代顺序改变时，状态偏移要显式按新索引计算。
- 与相邻卡片：`cube-05` 预取 K 循环内的操作数；本卡预取跨迭代的输入/状态。`pipe-02`/`pipe-03` 重排或交错迭代内的阶段；本卡移动迭代边界上的装载。`mem-06` 通过常驻消除 GM 往返；本卡只在数据仍需从 GM 装载时隐藏延迟，两者可叠加。`pipe-10` 调发布粒度；本卡移动装载位置，二者可独立使用。

## 参考资料

- 相关能力：轮转/常驻缓冲、TilingKey 编译期变体与分派。
- 验证案例：`ops/pro/kimi_k3/mega_recurrent_kda_replayssm`（`DESIGN.md` §5.2.2、`op_kernel/mega_recurrent_kda_replayssm.py` 的 `KEY_PREFETCH` / `_m3_loop_prefetch`）。该算子中预取只在部分网格有利（8 核 N=6 多组走 inline），实测 8 核大 batch 无回退；具体阈值未在本卡泛化。
