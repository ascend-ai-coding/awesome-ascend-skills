---
type: "PyPTO Performance Optimization Card"
title: "循环/递推状态片上常驻（消除 GM 往返）"
description: "循环或递推的状态与跨迭代中间量按迭代写回 GM 再读入、落在关键路径上时，改为在 UB 或 L1 单 buffer 常驻，消除 GM 往返。"
status: "stable"
tags: ["pypto-pro", "memory", "resident"]
item_id: "mem-06"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-row-reduce.md#choose-where-dependent-phases-share-state"
      scope: "同 owner 的循环/递推状态按完整序列常驻 UB，避免逐迭代 GM 写回再读取。"
      boundary: "状态跨 VF 调用须显式 Tile 保存；gated outer-state 只是所声明范围内的正确性锚点，不继承当前性能结论。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/memory-allocation.md#aliasing-and-mutual-exclusion"
      scope: "常驻状态与轮转输入的物理地址、生命周期及同步所有权分离。"
      boundary: "跨核生产消费不能单方面取消 GM 依赖；容量、reset 边界与所有消费者结束时间仍须核验。"
bound_hint: "MTE2、MTE3"
applicability: "循环/递推的状态或跨迭代中间量按每次迭代写回 GM 再读入；片上容量可容纳常驻 live set 与其邻居的轮转 buffer"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 make_tile/TileGroup 的常驻地址、auto_mutex 生命周期与跨核同步合同"
---

# 技术卡片 mem-06：循环/递推状态片上常驻（消除 GM 往返）

- **适用 bound**：MTE2、MTE3
- **一句话**：把循环携带的状态与跨迭代中间量固定在 UB 或 L1 单 buffer，去掉每次迭代的 GM 写回与读入。

## 何时用（诊断特征）

- 循环或递推的 state（如线性注意力的 `[D,D]` 状态）每次迭代经 GM 写回、下一次迭代再读入。
- 生成物或 trace 显示关键路径上存在该 GM 写回与读入的 MTE3/MTE2 段，且它们夹在计算之间串行。
- 片上容量可容纳常驻 live set，且不影响并行阶段的 UB/L1 预算。

## 何时不适用

- 取数方与生产方位于不同核且合同要求经 GM 交换（跨核中间量走 GM 属既有偏离）时，不能仅凭本卡
  取消 GM；须先判定角色是否可迁移到同核。
- 常驻 live set 超出 UB/L1 预算，或挤压并行阶段所需的轮转 buffer。
- 状态需要跨核归约或原子更新。

## 原理

每次迭代对状态做“GM 写回 → 下一次 GM 读入”会在关键路径上串入两段 MTE 搬运，并叠加 L2/主存延迟；
状态在片上单 buffer 常驻后，生产与消费直接经片上通路衔接，串行搬运被消除。常驻 buffer 的生命周期覆盖整个
循环，与轮转输入 buffer 分离，复用依据见 当前 allocation owner 的角色与生命周期规则。

## 怎么改（before / after）

以下为结构示意，展示数据流改动；所用 `pl.*` 形态（`pl.make_tile` 固定地址、`pl.load`/`pl.store` 二维
整块搬运、`pl.range`）是历史结构示意，未作为当前目标的完整实现重新验证。片段是教学级最小示意：例题设定循环携带
状态 `S[D, D]`（如线性注意力递推状态），`update_state_vf` 代表“读旧状态 + 本迭代输入、写新状态”
的就地更新；常驻 buffer 地址、容量与循环内轮转 tile group 的地址区间规划须按当前 DESIGN 核验后嵌入。

**before（每迭代 GM 写回 + 读入）：**

```python
with pl.section_vector():
    # 工作 buffer：每轮先读入上一迭代状态，算完再写回
    s_ub = pl.make_tile(
        pl.TileType(shape=[D, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=D * D * 4)
    for it in pl.range(0, n_iter):
        pl.load(s_ub, state_gm, [0, 0])          # MTE2：每轮 GM 读入（关键路径）
        update_state_vf(s_ub, in_ub[it])         # S = compute(S, in[it])
        pl.store(state_gm, s_ub, [0, 0])         # MTE3：每轮 GM 写回（关键路径）
```

**after（状态片上单 buffer 常驻，GM 只在首末各走一次）：**

```python
with pl.section_vector():
    # 常驻 buffer：循环外一次性分配固定地址，整个循环生命周期内不轮转、不与轮转 buffer 重叠
    s_ub = pl.make_tile(
        pl.TileType(shape=[D, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=D * D * 4)
    pl.load(s_ub, state_gm, [0, 0])              # 初始化读入：仅一次
    for it in pl.range(0, n_iter):
        update_state_vf(s_ub, in_ub[it])         # 片上直接衔接，无 GM 往返
    pl.store(state_gm, s_ub, [0, 0])             # 末迭代写回：仅一次
```

**关键陷阱**：常驻 buffer 必须与循环内输入/输出的轮转 tile group 使用不重叠的地址区间（角色划分见
当前 allocation owner）；把常驻 buffer 地址并进轮转组会被下一拍覆盖，表现为运行值被重置或读到陈旧值。

## 性能与验证指标

比较状态相关 MTE2/MTE3 字节、关键路径长度与同条件 `Task Duration`；正确性按完整套件回归，
覆盖首迭代初始化、末迭代写回与跨核可见性。

## 技术限制与风险

- 常驻 buffer 与轮转 buffer 的地址/生命周期须分离；复用同一地址会引入轮转别名错误。
- 跨核生产/消费的状态不能单方面取消 GM；须先确认角色同核或补齐核内同步。
- 常驻占用提高 UB/L1 峰值；须重算峰值并确认未越界。

## 参考资料

- [当前 allocation/lifetimes owner](../../../../pypto-pro-kernel/references/api/memory-allocation.md)：持久/轮转/scratch 角色、slot 与完整容量；[row state residency](../../../../pypto-pro-kernel/references/patterns/vf-row-reduce.md#choose-where-dependent-phases-share-state) 提供当前数据流候选。
