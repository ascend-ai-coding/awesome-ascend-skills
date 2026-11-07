# VEC Bound 诊断页

> 通用诊断适用于 arch22 / arch35，具体指标与卡片按目标能力选用；RegBase / Reg / MicroAPI / VF / RVEC 相关内容仅适用于 arch35（A5），示例阈值不作为跨代默认值。


VEC bound 是 elementwise、activation、reduction 类算子最常见的瓶颈。Vector 计算单元成为耗时主导，MTE2/MTE3 搬运单元相对空闲。

> 本页诊断对象是已经确认属于 AIV 的向量计算单元。融合算子中的 Cube MMAD/内嵌 MatMul 即使外层 profiling 同时出现 Vector 活动，也必须保留 MatMul 族路由；不能把 VEC 占比或低 MAC 利用率直接解释为应将 MM 替换成 AIV。

> 本页只做**诊断**（判定 + 特征）；具体优化手段（怎么改）见技术卡片库 `techniques/`。采集与指标口径统一见 `metrics.md`。

---

## 判定条件

- Vector 单元利用率高，`aiv_vec_ratio` 是耗时主导
- Vector 指令占据主要执行时间
- MTE2 和 Cube 单元相对空闲

**瓶颈严重程度分级**：

| VEC 占比 | 等级 | 优化方向 |
|----------|------|---------|
| 50–65% | 轻度 | DoubleBuffer + UB 融合有较大收益 |
| 65–80% | 中度 | 减少 Cast 或融合指令 |
| >80% | 深度 | DAV_3510 先评估寄存器驻留 VF（见下注）；A2/A3 按目标高层 API 与指令证据分析，通用方向包括减计算量与中间搬运 |

> **深度 VEC bound：先评估寄存器驻留 VF 改写（正常分析路径）**。vec 占比高不等于微架构收益耗尽：向量链即使已在 UB 驻留，每个向量 API 仍各带一次 UB 读 + 一次 UB 写。将链改写为寄存器驻留 VF 形式——`AscendC::Reg` 全链接力，入口 load 一次、链内寄存器传递、出口 store 一次，消除链内 UB staging 与中转——可消除这些往返。该实现路线限 DAV_3510 且只适用于已确认的 AIV 向量计算链；Cube MMAD/内嵌 MatMul 必须回到 MatMul 族评估 Cube 流水，不能由本页结论触发 AIV 替换。不能从 VEC 占比推导 A2/A3 支持 Reg。数值顺序与舍入仍须按目标精度门禁验证，不能预先保证位相同。

---

## 仿真图分析要点

- 识别 Vector 指令密集区
- 检查 Vector 指令间的数据依赖性，寻找可并行执行的机会

**VEC bound trace 特征**：

```
Time: 0 ---------------------------------------- 100ms

SCALAR     |##............................| 5%
SCALARLDST |##............................| 4%
MTE2       |.##....##....##....##.........|15%
VECTOR     |..############################|65%  <- 主导
MTE3       |..........................####.|10%
```

VECTOR 行持续活跃，MTE2 有明显空闲等待，说明搬运速度快于计算速度。泳道读法见 `metrics.md` §3。

---

## VEC 内部指令分析

从 trace 提取 VECTOR 泳道事件，按 `name` 分类统计：

| 指令类型 | 典型指令 | 含义 |
|---------|---------|------|
| 算术类 | `vec_add`, `vec_mul`, `vec_sub` | 基本计算，低延迟 |
| 超越函数类 | `vec_exp`, `vec_log`, `vec_rec`, `vec_rsqrt` | 高延迟指令，无硬件加速 |
| 类型转换类 | `vcvt_f2f`, `vcvt_f2s`, `vcvt_s2f` | Cast 开销 |
| 归约类 | `vec_reduce_sum`, `vec_reduce_max` | reduction 开销 |

**判断规则**：
- Cast 占比 >20% → 类型转换密集场景，优化 Cast 有高收益
- 超越函数占比 >30% → 深度 VEC bound，优化空间有限（转算法层减计算量）

> ascend950 regbase 范式使用 **RVEC** 单元：RVECEX（执行）、RVECLD（加载）、RVECST（存储）、RVECSU（设置）。与 msprof 泳道的对应见 `metrics.md` §4。

---

## DoubleBuffer 检查

在 trace 中观察 MTE2 和 VECTOR 行的时间重叠：

| 模式 | 特征 | 含义 |
|------|------|------|
| **DB 生效** | MTE2 和 VECTOR 交替出现 | 搬入与计算重叠，流水健康 |
| **DB 未生效** | MTE2 全在前，VECTOR 全在后 | 串行执行，需开启或修复 DoubleBuffer |

---

## 优化手段（详见技术卡片库 `techniques/`）

诊断出细分特征后，到 `techniques/` 取对应卡片实施：

| 诊断特征 | 首选卡片 |
| --- | --- |
| 相邻可融合 Vector 操作、中间落 GM | [vec-01-ub-fusion](techniques/vec-01-ub-fusion.md)、[vec-03-fused-instr](techniques/vec-03-fused-instr.md) |
| Cast 占比 >20% | [vec-02-reduce-cast](techniques/vec-02-reduce-cast.md)、[vec-05-eliminate-ub-staging](techniques/vec-05-eliminate-ub-staging.md) |
| 向量链耗时主导（含已 UB 驻留的链；每个向量 API 各带一次 UB 读写，常见于已做过 vec-01 的链） | [vec-05-eliminate-ub-staging](techniques/vec-05-eliminate-ub-staging.md)（改法 A 推广到全链：入口 `Reg::LoadAlign` 一次、链内 RegTensor 接力、出口 `Reg::StoreAlign` 一次；触发不要求可见 staging 往返） |
| 标量 round-trip（GetValue/SetValue、V_S/S_V 同步） | [vec-04-scalar-roundtrip-vectorize](techniques/vec-04-scalar-roundtrip-vectorize.md) |
| 反复 load 不变量 / 行长 ≪ VL lane 利用率低 | [vec-06-broadcast-once-vl-merge](techniques/vec-06-broadcast-once-vl-merge.md) |
| 归约累加器串行链、VECTOR 周期性空泡 | [vec-07-low-latency-reduce](techniques/vec-07-low-latency-reduce.md)、[vec-08-multi-accumulator-unroll](techniques/vec-08-multi-accumulator-unroll.md) |
| 超越函数深度 bound（`aiv_vec_ratio` >90%） | [vec-11-reduce-compute-lut](techniques/vec-11-reduce-compute-lut.md) |
| VF 反复启停 / setup 重复 | [vec-09-loop-sink-into-vf](techniques/vec-09-loop-sink-into-vf.md) |
| 小定长循环 | [vec-10-small-loop-unroll](techniques/vec-10-small-loop-unroll.md) |
| trace 现 preg spill/reload（尤 int64） | [vec-12-preg-split](techniques/vec-12-preg-split.md) |
| VF 尾块另设 tailTwoVL/tailOneVL 循环、计算体重复 | [vec-13-vf-tail-unified-mask](techniques/vec-13-vf-tail-unified-mask.md) |
| 一行拆段分别 load、段长非 32B 对齐、VF load/store 非对齐 | [vec-14-aligned-split-copy](techniques/vec-14-aligned-split-copy.md) |

卡片总览见 `techniques/INDEX.md`。

---

## Tiling 修正建议

- 调整 UB 布局以支持更高效的 RegBase 访问模式
- 调整 tile 粒度匹配 Vector 融合窗口
- 增大 tile 尺寸减少循环次数，降低流水启停开销；tile 上限须按目标 SoC 的实际可用 UB 容量及其他 buffer 占用计算，不使用型号对应的固定 UB 数值。
- 使能 DoubleBuffer 时，实际可用 UB 需除以 2
