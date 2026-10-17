# SIMT 算子性能优化指南

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

## 适用范围与原则

本文面向 Ascend 950（DAV_3510）上的 Ascend C SIMT 算子，适合条件分支、离散索引、随机访存、细粒度原子操作等不规则计算。

Ascend 950 SIMT 优化应优先遵循以下顺序：

1. 先选择正确实现路径：规则连续流量优先 SIMD/MTE，不规则计算再用 SIMT。
2. 再优化 GM/DCache/UB 数据流和线程映射。
3. 然后平衡线程并行度、寄存器压力、同步和指令吞吐。
4. 所有结论都通过代表 case 的精度与性能 A/B 验证。

## 规则速查

| 规则 | 优先级 | 核心目标 |
|---|---|---|
| Rule 001 | 高 | 用编译期分发消除 VF 内运行时分支 |
| Rule 002 | 高 | 让同一 warp 内线程尽量走相同路径 |
| Rule 003 | 高 | 优化线程映射和 GM 连续性，尽量使用合适的向量类型访存 |
| Rule 004 | 高 | 连续流量走 SIMD/MTE，不规则流量走 SIMT |
| Rule 005 | 高 | 按复用特征选择 DCache 策略，一次性读取旁路 DCache |
| Rule 006 | 高 | 用 UB 常驻减少重复 GM 访问，并规避 bank conflict |
| Rule 007 | 高 | 用 fast-division 和特化降低地址计算成本 |
| Rule 008 | 高 | 在线程数、铺核和寄存器预算之间取平衡 |
| Rule 009 | 高 | 控制寄存器压力并避免 spill |
| Rule 010 | 高 | 无 spill 前提下尽量展开静态循环 |
| Rule 011 | 中 | 最小化同步次数和同步范围 |
| Rule 012 | 中 | 仅在可摊薄固定开销时做搬算重叠 |
| Rule 013 | 中 | 选择原生低成本指令和合适精度 |
| Rule 014 | 高 | 用有效带宽和 profiling 数据闭环验证 |

## 优化规则库

### Rule 001：SIMT VF 内部避免可消除的 if-else 分支

- **分类**：分支优化
- **优先级**：高
- **适用场景**：`__simt_vf__` 内存在由 mode、dtype、shape 属性或功能开关控制的分支，且条件在 VF 调用前已经确定
- **核心方法**：在 VF 外分发，通过模板参数和 `if constexpr` 生成无运行时分支的特化版本

同一 warp 内线程进入不同分支时，不同路径通常需要分开执行。若条件对整个 VF 调用一致，应将判断提升到外层，避免每个线程重复判断。

**不推荐**：

```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024)
inline void OpFooSimt(__gm__ T* x, __gm__ T* y, int32_t mode, uint32_t totalLength)
{
    uint32_t idx = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
    uint32_t step = Simt::GetThreadNum() * Simt::GetBlockNum();
    for (uint32_t i = idx; i < totalLength; i += step) {
        T value = x[i];
        if (mode == 0) {
            y[i] = value + static_cast<T>(1);
        } else {
            y[i] = value * static_cast<T>(2);
        }
    }
}
```

**推荐**：

```cpp
template <typename T, int32_t Mode>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024)
inline void OpFooSimt(__gm__ T* x, __gm__ T* y, uint32_t totalLength)
{
    uint32_t idx = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
    uint32_t step = Simt::GetThreadNum() * Simt::GetBlockNum();
    for (uint32_t i = idx; i < totalLength; i += step) {
        T value = x[i];
        if constexpr (Mode == 0) {
            y[i] = value + static_cast<T>(1);
        } else {
            y[i] = value * static_cast<T>(2);
        }
    }
}
```

应控制模板组合数量。推荐在 kernel 内对少量关键 mode 做 `switch`，再调用特化 VF；不要无界组合 dtype、rank、mode 和线程数，导致二进制膨胀。

### Rule 002：按 warp 对齐控制流，分离尾块和罕见路径

- **分类**：分支优化
- **优先级**：高
- **适用场景**：分支条件由线程号、数据边界或离散索引决定，无法在 VF 外完全消除

优化顺序如下：

1. 调整线程到数据的映射，让同一 warp 内线程尽量命中相同分支。
2. 将尾块、非法索引或罕见路径单独处理，避免主循环持续携带分支。
3. 对仅含少量廉价指令的分支，可比较分支与谓词化/掩码写法；不能默认“无分支”一定更快。
4. 若改成同时计算两个分支，会增加指令数或寄存器压力，则保留分支并 A/B 实测。

warp 宽度、分支重汇合和谓词化结果应以 DAV_3510 编译器输出及实测为准。

### Rule 003：优化线程映射与 GM 访问连续性

- **分类**：访存优化
- **优先级**：高
- **适用场景**：相邻线程存在跨行、跨维或随机 GM 访问，实际搬运字节明显大于有效字节

优先让相邻线程访问相邻地址：

- 将 `threadIdx` 映射到连续的最内维，而不是大步长维度。
- 每个 AI Core 处理连续的输出区间，并按实际搬运接口和 profiling 结果确定对齐粒度。
- 输入和输出无法同时连续时，优先优化流量更大、访问次数更多或写放大更严重的一侧。
- 不规则写无法合并时，可评估按目标地址分桶/分块；必须同时计算重复扫描 index 带来的读放大。
- 数据连续时，应主动检查能否用短向量类型替代逐元素标量访存，并尽量选择目标编译器、数据类型和访存接口支持的合适向量宽度。候选类型不设封闭白名单，例如 `float2`、`float4`、`half2`，以及目标环境实际提供的 `half4` 或其他向量类型；从较宽类型开始评估，再按对齐、尾部和实测结果回退到较窄类型。
- 短向量只扩大单线程一次访问宽度，不能替代相邻线程地址连续这一要求。还要区分“元素位宽”和“向量 lane 数”：例如 SIMT 短向量 `int4` 通常表示 4 个 `int32`，不是 4-bit 整数类型。

当最内维在输入和输出中都保持连续时，可让一个线程处理 `C` 个连续元素：

```cpp
for (uint32_t tile = threadIdx.x; tile < tileCount; tile += blockDim.x) {
    uint32_t srcBase = CalcSrcTileBase(tile);  // 只对外层维度做索引分解
    uint32_t dstBase = tile * C;
    for (uint32_t c = 0; c < C; ++c) {
        output[dstBase + c] = input[srcBase + c];
    }
}
```

标量主体可按实际数据类型替换为合适的向量类型。下面仅用 `float4` 演示“向量主体 + 标量尾部”的通用写法；实际实现可以换成 `float2`、`half2`、目标环境支持的 `half4` 或其他向量类型，并同步调整 `VecT`、lane 数和对齐要求。这里的 `float4` 是由 4 个 `float` 组成的短向量，不是 FP4 数据类型：

```cpp
using VecT = float4;                  // 仅作示例，也可选择其他受支持向量类型
constexpr uint32_t kLanes = 4;       // VecT 包含的标量元素数
__gm__ VecT* inputVec = reinterpret_cast<__gm__ VecT*>(input);
__gm__ VecT* outputVec = reinterpret_cast<__gm__ VecT*>(output);
uint32_t vecCount = count / kLanes;  // count 表示标量元素数

for (uint32_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
    outputVec[i] = inputVec[i];
}

// 不足一个 VecT 的尾部保持标量处理。
for (uint32_t i = vecCount * kLanes + threadIdx.x; i < count; i += blockDim.x) {
    output[i] = input[i];
}
```

如需同时控制缓存路径，可在目标 CANN 版本为所选 `VecT` 提供重载时使用 `asc_ldcg(VecT*)` / `asc_stcg(VecT*, VecT)`；缓存策略仍按 Rule 005 选择。向量类型选择按以下顺序进行：

1. 从目标 CANN 版本的类型定义和访存接口重载中建立候选集合；项目自定义的向量类型也必须确认编译器能生成目标宽度的 GM load/store，不能只凭类型名判断。
2. 优先尝试一次承载更多有效元素的类型，再结合自然对齐逐级回退。例如 `float4` 需要 16 字节对齐，而换成其他类型时应使用该类型自己的对齐要求，不能沿用固定的 16 字节条件。
3. 主体按 `VecT` 的 lane 数处理；对齐前缀和不足一个向量的尾部单独处理，禁止未对齐强转、越界读取或越界写入。
4. 将向量宽度作为调优参数，通过代表 case A/B 比较指令数、有效带宽、地址计算、寄存器压力和尾部开销。原则是尽量使用向量类型，但不强求所有 case 使用相同类型或相同宽度。

Ascend 950PR/950DT、CANN 9.1.0 官方资料已确认多种短向量 GM 访存能力及 `float2`、`float4`、`half2` 等类型；该列表只是当前版本的已确认示例，不是本文维护的完整类型清单。其他类型（包括目标环境中的 `half4`）以对应版本头文件、API 重载和编译结果为准。

该方法同时提高单线程有效访存量，并少做一层坐标分解。但若整个操作已退化为规则连续 copy，应按 Rule 004 切换到 SIMD/MTE，而不是继续堆叠 SIMT 访存。

### Rule 004：连续流量优先 SIMD/MTE，SIMT 用于不规则部分

- **分类**：实现路径选型
- **优先级**：高
- **适用场景**：算子同时包含连续 copy/cast/init 和离散 gather/scatter/条件计算

Ascend 950 上，规则连续流量通常更适合：

- `DataCopy` / `DataCopyPad`
- NDDMA
- Vector API / Reg API
- 带 TPipe 的双缓冲搬运

典型混合实现：

1. clone、fill-zero、cast、连续 slice 搬运走 SIMD/MTE。
2. 离散 index、随机 GM 访问、warp 内协作或细粒度原子走 SIMT。
3. 只有小 shape 且 MTE/队列启动开销无法摊薄时，才考虑全程直接 GM 的 SIMT 快路径。

选型必须比较端到端成本，不能只比较计算循环。SIMD 路径需计入 GM→UB→GM、同步和尾块处理；SIMT 路径需计入标量事务放大、DCache 和地址计算成本。

### Rule 005：按复用特征选择 DCache 策略

- **分类**：缓存优化
- **优先级**：高
- **适用场景**：一次性流式输入与重复访问的热点数据共同竞争 SIMT DCache

按数据复用特征分三类：

| 数据特征 | 推荐策略 |
|---|---|
| 每个元素只读取一次 | 旁路 DCache 读取，避免挤出热点数据 |
| 会被同一核重复读取 | 使用 DCache，并尽量提高热点数据命中率 |
| 写出后不再被当前核读取 | 在版本支持时使用显式 non-cacheable store |

Ascend 950 的 CANN 9.1 官方 SIMT 样例使用 cache hint 明确表达策略：

```cpp
T value = asc_ldcg(input + idx);       // 不在 DCache 分配，适合一次性读取
T hot = asc_ldca(table + tableIdx);    // 使用 DCache，适合重复访问
asc_stcg(output + idx, value + hot);   // 不在 DCache 分配，适合一次性写出
```

若当前代码使用裸 GM 指针，也可对一次性读取的数据使用 `volatile` 指向类型，使读取旁路 SIMT DCache：

```cpp
__gm__ volatile T* inputOnce = reinterpret_cast<__gm__ volatile T*>(input);
T value = inputOnce[idx];
```

注意限定符位置：需要的是“指向 volatile 数据的指针”`volatile T*`，不是仅让指针变量本身不可优化的 `T* volatile`。

约束：

- `volatile` 还会阻止编译器合并、缓存或删除相应读取，可能增加指令数；不要用于会重复访问的热点数据。
- 旁路 DCache 不代表旁路所有缓存层级，也不替代原子操作、内存屏障或线程同步。
- 官方资料明确支持 `volatile` GM 读取用于获取其他核写入后的最新值；本文不推导 `volatile` store 具有完全相同的 cache 语义。
- CANN 9.0 与 9.1 对 `asc_ldcg/asc_ldca/asc_stcg` 在纯 SIMT、混合 SIMD+SIMT 模式下的支持范围不同；集成前必须检查目标版本 API 约束。
- 用生成指令、`DCache Read GM` 和 kernel 耗时验证旁路是否实际生效及是否有收益。

### Rule 006：用 UB 常驻消除重复 GM 访问，并规避 bank conflict

- **分类**：片上复用
- **优先级**：高
- **适用场景**：同一 index、查找表或小数据块会被多个线程/多轮循环重复读取

若热点数据能放入 UB，可先全载或分块载入，再由 SIMT 线程从 UB 读取，以减少重复 GM 访问。

启用前必须同时满足：

1. UB 容量足够，并为队列、临时张量和尾块保留空间。
2. 重用节省的 GM 流量大于 GM→UB 搬运和同步成本。
3. UB 索引模式不会造成严重 bank conflict。
4. 跨线程生产/消费关系具有正确的同步。

不要固定使用末维 `+1` 等单一 padding 规则。应根据 Ascend UB bank/sub-bank 映射分析 stride，再选择 padding、奇数 stride 或分块布局，并用 `ResourceConflictRatio.csv` 中的 bank conflict 指标验证。

### Rule 007：用 fast-division 和维度特化降低地址计算成本

- **分类**：指令优化
- **优先级**：高
- **适用场景**：逐元素循环需要把线性下标拆成多维坐标，热点路径含多次整数除法和取模

运行期除数在 VF 调用前已知时，先计算 magic/shift：

```cpp
uint32_t magic;
uint32_t shift;
GetUintDivMagicAndShift(magic, shift, divisor);
```

VF 内复用商值，并用乘减得到余数：

```cpp
uint32_t quotient = Simt::UintDiv(index, magic, shift);
uint32_t remainder = index - quotient * divisor;
```

进一步优化：

- 除数是编译期 2 的幂时使用移位和掩码，或让编译器完成强度削减。
- 2D/3D/4D 采用静态展开版本，避免 ND 运行时循环。
- 末维无需转置或无需拆分时，直接删除对应除法。
- magic/shift、shape 和 stride 移出逐元素循环。
- 先复用已有商值，不要为商和余数重复调用 `UintDiv`。

### Rule 008：平衡线程数、铺核和寄存器预算

- **分类**：执行配置
- **优先级**：高

线程数的目标不是取最大值，而是同时做到：

1. 有足够线程隐藏 GM 和指令依赖延迟。
2. block 数足以覆盖可用 AIV 核，避免并行宽度不足时只启动少数核。
3. 每线程寄存器预算足够，不发生 spill。

DAV_3510 上可用寄存器预算：

| 每 block 线程数 | 每线程可用寄存器数 |
|---|---|
| 1025～2048 | 16 |
| 513～1024 | 32 |
| 257～512 | 64 |
| 1～256 | 127 |

建议把线程数视为候选集而非固定答案：

| VF 特征 | 起始候选 |
|---|---|
| 简单、32 位索引、GM latency bound | 2048 / 1024 |
| 中等地址计算或计算密度 | 1024 / 512 |
| 64 位索引、多级 fast-div、复杂控制流 | 512 / 256 |
| 可并行任务数很少 | 降低每 block 线程数，以增加可启动 block 数 |

`LAUNCH_BOUND` 和实际启动线程数必须匹配。二者要求编译期常量时，在 kernel 内对有限候选值做 `switch` 分发到不同 VF 实例。

### Rule 009：控制寄存器压力并避免 spill

- **分类**：资源优化
- **优先级**：高

常见高压来源：

- 无必要的 64 位索引和 64 位中间结果
- 大型局部数组或动态索引数组
- 多维通用循环同时保留大量 shape、stride、magic 和坐标
- 过度内联、过深展开和过长变量生命周期
- 同时计算多个分支结果

优化方法：

- 能证明所有线性地址和乘法中间值不溢出时，使用 32 位索引；否则保留 64 位。
- 按 rank、dtype 或索引位宽做有限特化，删除无效状态。
- 缩短临时变量生命周期，避免所有坐标同时存活。
- 将线程数和 `LAUNCH_BOUND` 按索引位宽模板化。
- 结合生成代码、编译资源报告、仿真和上板耗时判断 spill；“编译成功”不能证明没有 spill。

### Rule 010：无 spill 前提下尽量展开静态循环

- **分类**：指令与 ILP 优化
- **优先级**：高
- **适用场景**：循环次数为编译期常量、循环体较短，且不同迭代间存在可并行指令

静态展开可以消除循环计数和跳转，并让多个迭代的独立指令交错调度，提高 ILP：

```cpp
template <int32_t N>
__simt_vf__ __aicore__ inline float ReduceSmall(const float* values)
{
    float sum = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < N; ++i) {
        sum += values[i];
    }
    return sum;
}
```

可使用 `#pragma unroll N` 控制展开因子；编译器无法得到静态边界时，可通过模板参数、AOT 常量化或少量手工展开提供静态信息。

展开门控：

- stack/register 不发生 spill 是硬门槛，再逐步增加展开因子。
- 展开会增加同时存活的变量和代码体积；过度展开可能降低可驻留线程数或增加 ICache 压力。
- 循环体包含昂贵分支、同步、原子或动态地址链时，完全展开不一定有收益。
- 用 `1/2/4/8/完全展开` 等候选做 A/B，选择“无 spill 且性能最好”的因子，而不是默认完全展开。

### Rule 011：最小化同步次数和同步范围

- **分类**：同步优化
- **优先级**：中

- 数据只在同一 warp 内生产和消费时，使用满足语义的 warp 级机制。
- 只有跨 warp 共享数据时才使用 block 级同步。
- 仅要求写入可见时，区分 memory fence 与 barrier；二者不能互相替代。
- 将同步移出热点内循环，或增加每次同步之间的有效工作量。
- 删除同步前必须证明所有读写依赖、执行顺序和可见性仍正确。

应按目标 CANN 版本支持的 SIMT 同步原语实现并验证，不能用未确认的同步语义替代现有依赖。

### Rule 012：仅在可摊薄固定开销时做搬算重叠

- **分类**：流水优化
- **优先级**：中

对于“GM→UB→计算→GM”的混合 SIMD/SIMT 路径，可用 MTE、TPipe 和双缓冲重叠搬运与计算。启用条件：

- 数据能稳定切成多个 tile。
- 单 tile 计算或搬运时间足以覆盖事件、队列和切换开销。
- UB 能容纳 ping/pong buffer 和临时数据。
- 尾块不会破坏流水正确性。

小 shape、单 tile 或直接 GM 的 SIMT 路径通常不应引入 TPipe/双缓冲，其固定开销可能超过收益。

### Rule 013：选择原生低成本指令和合适精度

- **分类**：指令优化
- **优先级**：中

- 小整数次幂优先显式乘法，不调用通用 `pow`。
- 有对应原生能力时，评估 FMA、reciprocal、rsqrt 等专用指令。
- 单精度计算使用 `1.0f` 等单精度常量，避免隐式 FP64 计算和转换。
- 只在误差预算允许时使用低精度或近似指令，并对正常值、极值、NaN/Inf 和边界值做精度验证。
- 使用前必须检查目标 CANN 版本的 Ascend SIMT API、数据类型支持和实际生成指令。

指令减少与寄存器压力需一起评估。公共子表达式复用可能减少计算，也可能延长变量生命周期。

### Rule 014：用有效带宽和 profiling 数据闭环验证

- **分类**：性能验证
- **优先级**：高

对访存型算子计算逻辑有效带宽：

```text
effective_bandwidth = (logical_read_bytes + logical_write_bytes) / kernel_time
```

逻辑字节数只统计算法必须读取/写出的数据。再结合 profiling 中的实际 GM 流量、DCache 访问、流水占比和资源冲突判断：

- 实际流量远大于逻辑流量：优先检查离散访问、未合并事务、重复加载和写放大。
- DCache Read GM 高且热点命中率低：按 Rule 005 区分一次性数据与热点数据。
- `SCALAR` / 地址计算占比高：检查 div/mod、动态 rank、分支和循环控制。
- bank conflict 高：检查 UB stride、padding 和线程映射。
- 增加线程数无收益或性能回退：检查并行宽度、寄存器预算和 spill。

每条优化必须使用真实或有代表性的 shape、dtype、32/64 位索引及连续/离散访问 case 做 A/B，并同时验证精度。不要用单个有利 case 推导全局准入条件。

## 推荐调优流程

1. **选路径**：先判断连续流量是否应走 SIMD/MTE，确认 SIMT 只承担不规则部分。
2. **看数据流**：计算逻辑字节数，检查 GM 访问连续性、DCache 复用和 UB 常驻机会。
3. **降地址成本**：特化 rank，使用 fast-division，外提 shape/stride/magic。
4. **配执行资源**：联合搜索 block 数、线程数、索引位宽和 `LAUNCH_BOUND`。
5. **打磨指令**：在无 spill 前提下展开静态循环，减少同步，选择低成本指令。
6. **验证泛化**：精度通过后，对代表 case 做 A/B，记录收益和退化边界。

## 资料来源

- Ascend 950 SIMT DCache 官方样例：`$ASC_DEVKIT_DIR/examples/03_simt_api/03_best_practices/00_memory_optimizations/cache_hint/README.md`。
- SIMT 访存函数：`$ASC_DEVKIT_DIR/docs/zh/api/SIMT-API/访存函数/访存函数简介.md`。
- SIMT 短向量类型、大小与自然对齐：`$ASC_DEVKIT_DIR/docs/zh/api/SIMT-API/SIMT编程简介/扩展语法/内置数据类型-139.md`。
- 短向量主体与标量尾部官方实践：`$ASC_DEVKIT_DIR/docs/zh/guide/算子实践参考/SIMT算子性能优化/内存访问/短向量类型提升效率.md`。
- `volatile` 与跨核 GM 可见性案例：`$ASC_DEVKIT_DIR/docs/zh/guide/算子实践参考/优秀实践/SIMT实现InsertHashTable算子案例.md`。
- 循环展开和寄存器约束：`$ASC_DEVKIT_DIR/docs/zh/guide/算子实践参考/SIMD算子性能优化/矢量计算/VF性能优化/指令双发优化.md`；SIMT 中同样需以生成代码和实测确认收益。
