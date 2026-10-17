# TileLang Ascend 高性能编程规范

<适用>
语言: Python, TileLang DSL
侧别: All
领域: true
触发: T.copy, T.Persistent, T.Pipelined, T.annotate_buffer_versions, T.SimdVF, T.SimtVF, T.gemm, T.reduce_, alloc_shared, benchmark
默认启用: true
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 3
</检视负载>

## 目的

检查 TileLang 算子的性能、资源使用和数值精度问题。

## 快速索引

### 性能优化规范（PERF-*）

| 规范编号 | 规范名称 | 严重级别 |
|---------|---------|---------|
| PERF-1 | 避免热点循环逐元素 GM 操作 | 高 |
| PERF-2 | 禁止写死可查询的硬件参数 | 高 |
| PERF-3 | 多阶段流水和 Buffer Version 配套 | 高 |
| PERF-4 | 单次搬运量和搬运次数合理 | 中 |
| PERF-5 | 避免 GM 重复读取 | 中 |
| PERF-6 | 尾块处理正确 | 高 |

### 精度规范（PREC-*）

| 规范编号 | 规范名称 | 严重级别 |
|---------|---------|---------|
| PREC-1 | 流水线数据依赖正确 | 高 |
| PREC-2 | 除零保护 | 高 |
| PREC-3 | 低精度输入使用足够精度的中间累加 | 高 |
| PREC-4 | 特殊值和数值稳定性符合接口合约 | 高 |

### Tiling 设计规范（TIL-*）

| 规范编号 | 规范名称 | 严重级别 |
|---------|---------|---------|
| TIL-1 | 多核负载均衡 | 中 |
| TIL-2 | 片上缓存容量不溢出 | 高 |
| TIL-3 | Buffer 规划合理 | 中 |

## 适用场景

普通 Python tiling/dispatch 和 TileLang Kernel。

---

## 检视前置要求

检视版本相关 API 时，先定位实际导入的 TileLang，并检查当前 Ascend example/test 和所选 Ascend 后端 lowering。
容量、对齐和硬件资源结论必须绑定目标 SoC、实际 TileLang/PTO lowering 或编译资源报告。

---

## 性能优化规范

### PERF-1: 避免热点循环逐元素 GM 操作

**严重级别**：高

### 问题描述

热点循环中逐元素访问 GM 或发起大量小搬运会降低带宽利用率。连续区域优先使用 GM↔UB `T.copy` 后批量计算；gather/scatter、短记录和复杂分支需要结合实际访问模式分析，不能机械要求全部改成 DMA。

### 错误示例

```python
for i in T.serial(valid):
    out[offset + i] = inp[offset + i]
```

### 正确示例

```python
T.copy(inp[offset : offset + valid], copy_ub[:valid])
T.copy(copy_ub[:valid], out[offset : offset + valid])
```

### 检视方法

定位最内层热循环，统计 GM load/store 和 `T.copy` 次数；只有能证明访问连续、可安全合并且位于热点时才报性能问题。

---

### PERF-2: 禁止写死可查询的硬件参数

**严重级别**：高

### 问题描述

Vector、Cube 和混合核分别使用已确认的可用 AIV、AIC 和配对资源，构建模板时显式传入核数。TK 的 `src/cann_ops_tilelang/config.py` 提供 `get_num_vector_cores` 与 `get_num_ai_cores`；复用时核对其实现与设备匹配。直接写死核数会损害跨设备适配。

### 错误示例

```python
num_cores = 20
```

### 正确示例

```python
# available_aiv_cores 由目标硬件查询传入
num_cores = min(available_aiv_cores, num_tasks)
```

小任务核数通常不超过独立 task 数。若为特定 L2 domain 调度上调核数，必须记录原因。

---

### PERF-3: 多阶段流水和 Buffer Version 配套

**严重级别**：高

### 问题描述

`num_stages > 1` 时，生产/消费 buffer 需要获得足够版本，首次和末次迭代的数据依赖必须成立。当前 Ascend AutoSchedule 可以自动选择版本数；显式 `T.annotate_buffer_versions` 是对调度器选择的覆盖，不是启用流水的固定仪式。常驻只读数据通常保持单版本。

### 正确示例

```python
T.annotate_buffer_versions({tile_ub: num_stages})
for w in T.Pipelined(num_outer_iters, num_stages=num_stages):
    ...
```

### 检视方法

检查 loop body 中每个 buffer 的写入、读取、store、版本数和同步；不存在 annotation 时查看 lowering/编译结果中的实际版本选择，不能仅根据源码 API 名称判断 overlap 已生效，也不能仅因缺少 annotation 判错。版本增加还必须计入片上容量。

---

### PERF-4: 单次搬运量和搬运次数合理

**严重级别**：中

### 问题描述

小搬运会增加指令开销；过大 tile 又可能造成 UB 溢出、并行度降低或尾块浪费。应结合连续维、dtype、32B DMA 和 task 数选择 tile。

### 检视方法

计算每次 `T.copy` 的字节数和循环次数，确认相邻连续 copy 是否可合并，同时检查合并后 UB 容量和 tail。
没有热点 profiling 或可证明的数量级退化时只给优化候选，不判 FAIL。

---

### PERF-5: 避免 GM 重复读取

**严重级别**：中

### 问题描述

循环不变的广播参数、scale 或索引表若在每个 task 中重复从 GM 搬入，会增加确定的 GM traffic；但常驻也会增加 UB/L1 占用并可能降低 tile 或 stage。

### 检视方法

1. 以 GM 地址和 task loop 为单位统计同一数据的读取次数。
2. 区分跨 task 不变数据、每 tile 不同数据和因 cache hint 故意重读的数据。
3. 计算常驻后的容量、版本数和可用并行度，再判断是否值得复用。

### 判定方法

静态上可证明同一数据被无效重复搬运且不存在容量收益交换时可以报告；否则列出复用候选及需要验证的代表 case。

---

### PERF-6: 尾块处理正确

**严重级别**：高

### 问题描述

尾块同时影响 GM 有效范围、UB padding、SIMD mask、task 数和输出写回。仅在计算中使用 mask，不能修复已经发生的越界 GM 搬运；仅缩短 GM copy，也不能保证完整寄存器访问不会越过 UB。

### 检视方法

检查 0、1、对齐值前后、tile 前后和最大公开 shape。逐步证明 `offset`、`valid`、src/dst slice、UB footprint 与写回范围一致；接口明确只支持某些余数类时核对 wrapper/dispatch 是否真的限制了输入。

### 判定方法

存在合法输入会漏算、重复、越界或读取未初始化 padding 时判 FAIL。接口不承诺通用尾块且入口已经拒绝不支持 shape 时不要求新增 fallback。

---

## 精度规范

### PREC-1: 流水线数据依赖正确

**严重级别**：高

### 问题描述

DMA、SimdVF/SimtVF/Cube 计算和 GM store 之间若缺少正确 producer/consumer 依赖，可能读取旧版本、覆盖仍在使用的数据，或在首轮使用未初始化版本。

### 检视方法

按 load→compute→store 建立每个 buffer 的读写图，检查循环携带依赖、首轮和末轮、跨执行域消费以及手工多缓冲索引。显式 barrier 只能针对可说明的 hazard，不能用粗粒度同步掩盖地址或版本错误。

### 证据要求

源码可静态证明冲突时可直接报告；涉及 AutoSchedule 自动同步时必须结合当前 PTO lowering/生成代码或可复现设备错误。

### PREC-2: 除零保护

**严重级别**：高

shape、group size、reduction count、norm denominator 和 scale 作为除数前必须由编译期约束、wrapper 校验或 Kernel guard 保证非零。按 `tilelang-red-line.md` 的除数来源表追踪，不能把 `T.ceildiv` 本身当成保护。

### PREC-3: 低精度输入使用足够精度的中间累加

**严重级别**：高

### 问题描述

fp16/bf16/FP8/FP4 输入在长归约、方差、softmax 状态和 GEMM 中直接低精度累加，可能产生随归约长度增长的误差、溢出或下溢。

### 检视方法

从输入 load 开始追踪 `T.cast`/SIMD `vcvt`、fragment/UB dtype、reducer dtype、L0C dtype和最终 store。fp16/bf16 reduction、norm、softmax 统计量和 GEMM 默认 fp32 累加，输出仅在边界处转回目标 dtype。

### 排除规则

算法明确要求低精度累加，或当前实现具有覆盖最大归约长度和极值输入的误差证据时可以接受。不能通过放宽 tolerance 或降低 reference 精度形成排除依据。

---

### PREC-4: 特殊值和数值稳定性符合接口合约

**严重级别**：高

### 问题描述

NaN、Inf、正负零、全零行、极大/极小值和抵消输入可能改变比较、归约、除法及类型转换结果。稳定 softmax、norm、scale 等算法必须保持其数学变换和接口约定。

### 检视方法

先从公开 reference 和测试确定特殊值语义，再核对 max/sum、`exp`/`log`、`rsqrt`、clamp、rounding 和饱和顺序。检查全零 denominator/amax 的下限保护，以及尾块填充值是否会参与有效归约。

### 判定方法

实现与公开 reference 的特殊值语义明确不一致，或存在可构造输入破坏输出契约时判 FAIL；接口未定义该语义时标记待确认并建议补充测试。

---

## Tiling 设计规范

### TIL-1: 多核负载均衡

### 问题描述

task domain、wave size 和 core id 映射错误会造成遗漏或重复；即使映射正确，极端长尾、每核重复固定开销和 task 少于 core 也可能限制性能。

### 检视方法

展开 `T.Persistent(domain, num_cores, core_id)` 的多轴 task 空间，证明每个逻辑 task 恰好被处理一次。检查 task 少于 core、非整除、极小/极大 shape、不同轴不均匀以及人工 flatten/unflatten。性能层面的核数选择按代表 case 验证，正确性映射错误直接报告。

### TIL-2: 片上缓存容量不溢出

### 问题描述

源码中的单份 shape 不是实际资源占用；buffer versions、对齐 padding、常驻数据、临时 fragment 和 unroll 后寄存器共同决定容量与 spill。

### 检视方法

逐 buffer 计算 `元素数 × dtype.bytes × 实际 versions`，加 padding、resident 和安全余量；分别核对 UB、L1、L0A/L0B/L0C。寄存器压力通过当前编译结果或生成代码判断。

### 证据要求

容量上限必须来自目标 SoC、实际编译资源报告或当前后端精确配置；无法确认目标容量时标记待确认并说明所需编译/资源证据，不套用跨 SoC 固定容量判 FAIL。

### TIL-3: Buffer 规划合理

### 问题描述

重复存储同一数据、无收益多版本化、生命周期过长或远大于访问 footprint 的 padding 会挤占片上资源；过度复用同一 buffer 又可能制造不透明别名和流水依赖。

### 检视方法

对每个 buffer 标注用途、scope、shape、dtype、版本数、首次写、最后读和是否跨 task 常驻。合并或复用建议必须同时检查数据依赖、可读性、容量和性能，不以“buffer 数越少越好”机械判断。

---

## 检视检查清单

### 性能优化

- [ ] 热点 GM 访问已合理批量化
- [ ] 核数来自正确的硬件查询接口
- [ ] Pipeline 和 Buffer Version 配套
- [ ] 搬运量、次数与数据复用合理
- [ ] 尾块正确且无异常退化

### 精度

- [ ] 流水数据依赖和初始化正确
- [ ] 所有动态除数有非零保证
- [ ] 低精度输入使用足够精度的中间状态
- [ ] 特殊值和稳定算法语义与公开接口一致

### Tiling 设计

- [ ] task 映射完整、无重复且负载均衡
- [ ] UB/L1/L0/寄存器容量不溢出
