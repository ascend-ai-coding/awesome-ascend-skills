---
type: "PyPTO Performance Optimization Card"
title: "小 tile 固定开销与粒度修正"
description: "tile 数巨大、每 tile 计算量小时，每块固定标量开销（尾判定、动态 set_validshape、搬运参数构造、循环簿记）摊不开；增大 tile 行块高把块数按行块高比例降下来即得全部收益。"
status: "stable"
tags: ["pypto-pro", "scalar", "tiling"]
item_id: "scalar-03"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-row-reduce.md#batch-independent-rows-even-when-each-row-fits"
      scope: "通过增大每块有效工作量减少逐 tile 固定标量与 descriptor 开销，并检查最后一个 worker 的工作。"
      boundary: "本卡改变静态 Tile 粒度，尾判定保持独立；不据历史每块耗时或 TR 值生成通用运行时阈值。"
bound_hint: "scalar"
applicability: "总计算量小或 shape 小，块数远大于每块工作量（如行块高取 1~2）；scalar 占比高、发射率低，热点在逐块固定标量开销而非数据搬运或计算本身"
target_api_gate: "仅限已核验 pl.make_tile_group/TileType、pl.range、pl.min、pl.set_validshape 的 Ascend 950PR 或 950DT 工具链"
---

# 技术卡片 scalar-03：小 tile 固定开销与粒度修正

- **适用 bound**：Scalar / 小 case
- **一句话**：kernel 每处理一个 tile 都要付一套固定的小开销（尾判定、设
  validshape、构造搬运地址、更新循环变量）。块数 = ceil(m / TR)：行块高
  `TR` 越大，块数越少，这套开销重复的次数就越少——**增大 TR 就是本卡的
  全部收益**。

## 何时用（诊断特征）

- shape 小或总计算量小，循环块数与每块计算量之比悬殊（极端：行块高 1，
  块数 = 行数）。
- 小 case 的启动头 / Init / 取指类固定开销占比高（此类成本随框架托管、
  DSL 层不可直接操作），用户侧可操作对应物即本卡的粒度修正。
- profiler 显示 scalar 占比高、指令发射率低，且热点逐块均匀分布在整个
  工作循环，而非集中在少数几块。
- 源码侧静态账本可数：每块固定标量操作次数 × 块数，明显大于每块的
  数据搬运与向量计算所需。

## 何时不适用

- 行数本身少于目标行块高时，主循环为空、退化为单尾块，改粒度无意义。
- 行块高增大受 Vec memory / tile 容量上限约束（tile 字节数随 TR 线性增长）。
- 每块内存在逐行变化的逻辑（逐行不同偏移、条件或跨行依赖）时，必须先
  确认合并后语义不变，否则不可合并。
- 搬运或计算已是主瓶颈时，减块数只减固定开销，不改变数据总量，应先修
  主瓶颈。

## 原理

- 细粒度 tile 的每块固定标量开销清单：尾判定（减法 + `pl.min` + 比较）、
  两次动态 `set_validshape`、`load`/`store` 的偏移构造、循环变量更新。
  固定开销总量与块数成正比，与每块实际行数无关。
- 行块高从 `TR_b` 增大到 `TR_a` 后，块数从 `ceil(m/TR_b)` 降为
  `ceil(m/TR_a)`——**这是本卡唯一的性能杠杆**（实测 TR 1→8 贡献全部
  收益，~140ns/块的固定开销由 1 行分摊改为 8 行分摊）。
- 尾块处理与粒度正交：换粒度后尾块写法原样沿用（`pl.min` 判定留在循环
  内），无需其它改动。
- 搬运总量与计算总量不变，改变的只是固定开销的重复次数。

## 怎么改（before / after）

以下为嵌入片段（`pypto_pro` 0.2.0、FP32、tile 宽 `NN=128`、Vec memory、
单 block 启动、`@pl.jit(auto_mutex=True)` 上下文中核验；计算为
`y = x + x`，可替换为任意逐块 tile 计算）。**before/after 唯一差异是
行块高 `TR`（1 → 8），尾块写法原样沿用，其余代码一字不改**：

```python
TR = 1   # before：行块高 1（块数 = m）    → after 改为 8（块数 = ceil(m/8)）

ing = pl.make_tile_group(
    type=pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
    addrs=0x0, mutex_ids=[0])
og = pl.make_tile_group(
    type=pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
    addrs=0x8000, mutex_ids=[1])
with pl.section_vector():
    nt = (m + TR - 1) // TR           # 块数：TR=1 时 = m；TR=8 时 = ceil(m/8)
    xi = ing.next()
    yi = og.next()
    for t in pl.range(0, nt, 1):
        vr = pl.min(TR, m - t * TR)   # 每块尾判定（写法两档相同）
        pl.set_validshape(xi, [vr, NN])   # 每块动态 validshape
        pl.load(xi, x, [t * TR, 0])
        pl.set_validshape(yi, [vr, NN])
        pl.add(yi, xi, xi)
        pl.store(y, yi, [t * TR, 0])
```

实测（Ascend950PR、m=4096/16384）：以上"只改粒度"即获得与完整改写相同的
全部收益（+82%，5.5x）——固定开销的重复次数从 m 次降为 m/8 次。

## 性能与验证指标

- 正确性：改写前后输出应逐元素一致，并与 golden 一致；整除、带尾块、
  仅尾块、单行 shape 均须覆盖。
- 性能观察项：块数（静态可数）、scalar 泳道占比、kernel 总时长。
- 实测参考（Ascend950PR、`y=x+x`、m=4096~65536，见验证报告）：
  TR 1→8 端到端 **+75~82%（4.0~5.6x）**；每块固定开销 **~140ns**（TR=1
  实测）；带宽饱和后收益封顶（TR=8 下 ~47-50ns/行）。收益取决于固定开销
  在总时间中的占比，未实测前不作承诺。

## 技术限制与风险

- 行块高是 `TileType` shape 的编译期常量，改粒度即改 tile group 定义；
  两种粒度是两个不同 kernel，不可运行时切换（需要多粒度时走 tiling_key
  编译期特化，见 [tiling_key 特化示例](../../../examples/matmul/tilingkey_specialize/)）。
- tile handle 在循环外 `next()` 一次、循环内按偏移复用，依赖
  `auto_mutex` 串行化；跨循环重叠搬运需另行验证。
- 两个 tile group 的地址区间必须不重叠（示例 `0x0` / `0x8000`）。
- `make_tile_group` 仅支持 inline `TileType` + 标量 `addrs` + 单元素
  `mutex_ids`（变量 type + 列表 addrs 触发解析器崩溃）。

## 参考资料

- 语法：`pypto_pro.language` 的 `make_tile_group` / `TileType`（shape 与
  valid_shape）、`range`（循环迭代）、`min` / `max`（标量最值）、
  `set_validshape`（部分块有效形状）。
