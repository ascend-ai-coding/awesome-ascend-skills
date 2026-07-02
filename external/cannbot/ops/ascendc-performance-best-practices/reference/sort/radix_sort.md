# 选择/基数排序类算子性能优化实践

本文档聚焦 Ascend C 选择 / 排序类算子（TopK、KthValue、Sort、ArgSort、Histogram 等）在大规模数据场景下的性能优化实现：算法层重构、流水线并行、指令级技巧、精度保障与跨核同步。

> **当前覆盖范围**：本文档基于 **A2 大规模数据** 实战验证；A3、A5 等架构的硬件细节差异需另行验证后再套用。覆盖算子族包括选择类（TopK / KthValue / Median / Quantile）、排序类（Sort / ArgSort / ArgMax / Min）、分桶统计类（Histogram / BinCount / Unique）、过滤输出类（Where / NonZero / MaskedSelect）。

> **共性特征**：数据量远大于 UB（需多遍扫描）、输出稀疏（合格元素 << 全部元素）、跨核同步开销显著。

> **瓶颈定位**：profiling、指标解读、msprof 协议等不在本文档范围内，参见独立 profiling skill。本文档假设读者已经识别出瓶颈类型，从这里选择对应实现。

---

## 1. 算法层面优化（收益最大）

### 1.1 向量化二分搜索替代多级直方图

**适用场景**：在大数据中定位"第 K 大"阈值（TopK、KthValue、Median、Quantile）。

**传统方案**：多级直方图（radix sort 思想），每级统计 bin counts，逐级缩小范围。低位级别元素匹配率低（<2%），向量化效率极差，退化为标量扫描。

**优化方案**：在值域上做二分搜索。每步 1 次全量向量化扫描 + 1 次 Compare + 1 次 ReduceSum + 1 次全局同步。

```
lo = 值域下界, hi = 值域上界
for step = 0 to bit_width - 1:
    mid = (lo + hi) / 2
    globalCount = Σ CountGE(mid)    // 全核向量化
    if globalCount ≥ K: lo = mid
    else: hi = mid
threshold = lo
```

**收益**：消除低位标量扫描瓶颈（实测 -9.4ms，-46%）。

**代价**：bit_width 次全量数据扫描（DMA 带宽）。适用于 DMA 带宽有余量（MTE2 占比偏低）的场景。

**推广**：
- **Quantile**：二分搜索找 count = N × q 的阈值
- **Median**：K = N/2 的 TopK
- **KthValue**：K 为参数的 TopK
- **Histogram 均衡化**：二分搜索找各百分位

### 1.2 分桶粒度 × 级数权衡

**原理**：对于 B-bit 值域，分成 L 级，每级 b_i bits（Σb_i = B）。

| 方案 | 每级 bins | 向量比较/subchunk | 级数 | 数据扫描次数 |
|------|----------|------------------|------|------------|
| 8+8 | 256 | 256 × EQ | 2 | 2 |
| 4+4+4+4 | 16 | 16 × EQ | 4 | 4（后两级可标量） |
| 3+3+5+5 | 8/32 | 8 × GE + 8 × EQ | 2 vector + 2 scalar | 2 vector + 1 scalar |
| **16 × 1-bit** | **2** | **1 × GE** | **16** | **16**（全向量化） |

**选择原则**：
- 标量主导：增加二分步数，减少标量路径（→ 向量化二分搜索）
- 向量主导：减少每级 bins 数（3+3 而非 4+4）
- DMA 主导：减少扫描次数（8+8 两级 vs 16 步二分）

### 1.3 保序域变换一次性完成

**适用场景**：浮点数据需要整数可比较格式（BF16/FP16/FP32 → sortable uint）。

**优化**：全量数据一次性转换并写回 GM，后续 N 次扫描直接加载，跳过重复转换。

```
Pass 0:  for each chunk: DMA(GM→UB) → Transform(UB) → DMA(UB→GM)
后续 N 次 CountGE: for each chunk: DMA(GM→UB) → Compare(UB)            // 无 Transform
结束前: for each chunk: DMA(GM→UB) → InverseTransform(UB) → DMA(UB→GM) // 恢复原始
```

**收益**：N 次 Transform → 1 次。当 N ≥ 3 时有净收益。

**注意**：
- 原地修改输入 tensor → kernel 结束前必须用逆变换恢复
- 逆变换不一定 = 正变换（如 Adds 饱和导致 VtS ≠ SortToRaw）
- 额外 2 遍 DMA 读写（~0.7ms × 2）

### 1.4 预计算与复用

| 技巧 | 说明 |
|------|------|
| **per-core bin counts 缓存** | P1L1 的 localCoarseBins_ 在 P1L2/P2 中复用，避免重复统计 |
| **32×32 table 一次填充** | P2L1 标量扫描同时填充 32×32 lookup table，P2L2 零成本查表 |
| **rawThresh 预计算** | equal 元素的 value 固定（= rawThreshold），预计算一次供所有 core |

---

## 2. 流水线并行优化

### 2.1 TQue Double Buffer（MTE2 || Vector）

**适用条件**：MTE2 占比偏高且 Vector 有足够计算量。

**ARCH 关键限制（A2）**：
- **TBuf 模型**：`PipeBarrier<PIPE_ALL>` 全停，无法实现 MTE2/Vector overlap
- **TQue 模型**：`AllocTensor/EnQue/DeQue/FreeTensor` 自动同步，支持真正的流水线并行

```cpp
TQue<QuePosition::VECIN, 2> dataQue_;  // BUFFER_NUM=2

// CopyIn:  AllocTensor → DataCopy → EnQue    (MTE2 管道)
// Compute: DeQue → VectorOps → FreeTensor    (Vector 管道)
```

**标准模式**：

```
Prolog:   CopyIn(0), CopyIn(1)
Steady:   for i = 0..N-1:
              data = DeQue()
              Compute(data)
              FreeTensor(data)
              if (i+2 < N): CopyIn(i+2)     // 与当前 Compute 并行
```

**关键注意**：
- `AllocTensor` 在 queue 满时**阻塞**等待 `FreeTensor`
- 不能预先 Alloc 所有 chunk（死锁！）
- UB 预算：需要 2× 数据 buffer
- TBuf 手动 double buffer（两个 TBuf + PipeBarrier 切换）在 A2 上**不工作**

### 2.2 DMA 后置发射（单 buffer overlap）

**适用场景**：Vector 处理完当前 chunk 的数据后，数据 buffer 空闲，可立即发起下一 chunk 的 DMA。

```
GenGEMask(dataBuf_, ...)         // Vector: 读 dataBuf_, 写 temp1Buf_
PipeBarrier<PIPE_V>()            // 等 Vector 完成 → dataBuf_ 空闲
DataCopy(dataBuf_, xGm_[next])   // MTE2: 写 dataBuf_ (与下面的 SumMask 并行)
SumMask()                        // Vector+Scalar: 读 temp1Buf_ (不读 dataBuf_!)
PipeBarrier<PIPE_ALL>()          // 等 MTE2 完成
```

**前提**：后续处理（SumMask）不读 dataBuf_，只读 temp1Buf_。

**收益**：比 TQue 小（仅掩盖 SumMask 阶段的 DMA），但不需要额外 buffer。

### 2.3 Barrier 并行

在 FullBarrier（GM 轮询）期间，其他管道空闲。可在 barrier 前发起 DMA prefetch：

```
DmaWrite(localCount)
DataCopy(dataBuf_, xGm_[nextStep_chunk0])  // prefetch for next binary search step
FullBarrier()                              // ~200μs GM 轮询，DMA 同时完成
```

---

## 3. 指令级优化

### 3.1 向量化直方图技巧

| 技巧 | 说明 | 收益 |
|------|------|------|
| **累计 GE 替代 EQ** | `CompareScalar(GE, b)` 得到 `count ≥ b`，差分推导 bin count。bin[0] = total - cum[1]，省 1 次比较 | -1/N 比较 |
| **2-bin batch ReduceSum** | 两个 bin 共享一个 PipeBarrier：sel+reduce → alt+reduce → barrier → read both | -50% barrier |
| **mask → half 计数** | `And(mask, 0x3C00)` 将 0xFFFF→half(1.0)，省去 Select+Duplicate | -2 指令/subchunk |
| **合理 subchunk 大小** | half ReduceSum 精确到 2048；float ReduceSum 精确到 2^24 | 精度保障 |

### 3.2 标量快速跳过

**适用场景**：稀疏条件过滤（合格率 < 5%）。

**16 元素 OR 跳过**：

```cpp
uint32_t w0..w7 = mask32.GetValue(hj..hj+7);  // 8 个 uint32 = 16 个 uint16 mask
if ((w0|w1|w2|w3|w4|w5|w6|w7) == 0) continue; // 全不合格 → 跳 16 个元素
```

**XOR 快速匹配**：

```cpp
uint32_t x = (word ^ magic) & highBitMask;     // 高 bits 不匹配 → x 非零
if ((x0 & x1 & x2 & x3) == mask) continue;     // 4×uint32 (8 元素) 全不匹配
```

**适用于**：TopK 输出（~0.5% 合格率）、NonZero、MaskedSelect。

### 3.3 标量 GM 操作流水线

**非阻塞 GM 读**：`xGm_.GetValue(idx)` 发出后 scalar 立即继续，延迟被后续指令隐藏。

**阻塞 GM 写**：`valGm_.SetValue(pos, val)` 等待写完成才继续。

**流水线效果**：

```
element i:   [GetValue_i]─────────[SetVal_i][SetIdx_i]
element i+1:              [GetValue_{i+1}]──────────[SetVal_{i+1}][SetIdx_{i+1}]
```

**推论**：
- 不要用同步 scalar 计算替代非阻塞 GM 读（破坏流水线 → 变慢）
- 减少 value 输出不一定更快（流水线空洞 → 更慢）
- 增加无关 GM 读可以填充流水线空隙（免费延迟隐藏）

---

## 4. 精度保障

### 4.1 浮点→整数保序变换

| 数据类型 | 变换规则 | 向量实现 |
|---------|---------|---------|
| **BF16** | 正：XOR 0x8000，负：NOT | 7 ops: ShiftRight+Not+Adds+And+Not+And+Or |
| **FP16** | 正：XOR 0x8000，负：NOT | 同 BF16（format 相同 trick） |
| **FP32** | 正：XOR 0x80000000，负：NOT(all 32 bits) | 需要 int32 向量操作 |

**逆变换**（SortToRaw）：
- 高半区（bit_sign=1 → 原正数）：清除符号位
- 低半区（bit_sign=0 → 原负数）：NOT

**注意**：正变换 ≠ 逆变换（因为 Adds 饱和），必须分开实现。

### 4.2 A2 Adds(int16) 饱和问题

`Adds(data, scalar)` 在溢出时**饱和到 ±32767/−32768**，不回绕。

**影响**：`Adds(sortable, -threshold)` 当 sortable 和 threshold 分属 0x8000 两侧时，结果符号位错误。

**修正模板**：

```
if threshold == 0x8000:
    mask = ShiftRight(data, 15)         // 特判：直接用符号位
elif threshold >= 0x8000:
    mask = Adds+ShiftRight+Not          // 同半区正确
    And(mask, ShiftRight(data, 15))     // 低半区 → 不合格
else:
    mask = Adds+ShiftRight+Not
    Or(mask, ShiftRight(data, 15))      // 高半区 → 合格
```

### 4.3 half ReduceSum 精度限制

half 尾数 10 bits → 精确整数范围 [0, 2048]。

**规则**：`HIST_SUBCHUNK ≤ 2048`，保证每次 ReduceSum 结果 ≤ 2048。

**替代**：float32 ReduceSum 精确到 2^24（16M），但 UB 占用翻倍。

### 4.4 DMA padding 陷阱

DMA 要求 32-byte 对齐。最后一个 chunk 不足对齐长度时，padding 区域含残留数据。

**修正**：Host tiling `ALIGN = HIST_SUBCHUNK`，确保 `elemsPerCore` 是 subchunk 的整数倍。

---

## 5. 跨核同步（A2 无硬件 barrier）

```cpp
// FullBarrier: 所有 core → core 0
每个 core: 写 cookie 到 GM flags[step][bid]
Core 0:    for each core: while (GM[step][c] != cookie) { DMA read + poll }

// BroadcastBarrier: core 0 → 所有 core
Core 0:    写 cookie 到 GM flags[step][C]
其他 core: while (GM[step][C] != cookie) { DMA read + poll }
```

**优化**：
- 减少 barrier 次数（二分搜索 16 步 vs 多级直方图 4-6 步）
- 每步 barrier 只传 1 个 count（vs 256-bin 直方图）
- Workspace layout 复用（sHist_ 每步覆盖写）

---

## 6. 瓶颈类型 → 实现选择速查

> 假设瓶颈类型已通过 profiling skill 确定，下表给出每类瓶颈对应的实现策略与本文档锚点。

```
瓶颈类型
    │
    ├─ Scalar 主导
    │   ├─ 标量直方图/扫描 → 向量化二分搜索 (§1.1)
    │   ├─ 标量输出 → 无法优化（硬件限制），保持 GM SetValue
    │   └─ 标量 barrier → 减少同步次数 (§5)
    │
    ├─ Vector 主导
    │   ├─ CompareScalar 过多 → 减少 bins / level (§1.2)
    │   ├─ 重复 Transform → 一次性转换 + 复用 (§1.3)
    │   └─ 指令级 → cumGE、2-bin batch、mask trick (§3.1)
    │
    ├─ DMA/MTE2 主导（且管道无重叠）
    │   ├─ TQue Double Buffer (§2.1, 首选)
    │   ├─ 减少扫描次数 (§1.2)
    │   └─ DMA 后置发射 (§2.2)
    │
    └─ 各管道均衡 + 已显著重叠
        → 已接近硬件极限，考虑算法层重构或接受当前性能
```

### 6.1 A2 硬件特性速查

| 特性 | 行为 | 影响 |
|------|------|------|
| Adds(int16) | 饱和（非回绕） | 跨 0x8000 需 cross-half 修正 |
| ReduceSum | 仅支持 half / float（无整型） | 必须 mask → half → ReduceSum |
| half 精度 | 精确整数 ≤ 2048 | SUBCHUNK ≤ 2048 |
| `xGm_.GetValue` | 非阻塞流水线读 | 不可替换为同步计算 |
| `SetValue` | 阻塞写 | 是标量输出的真实瓶颈 |
| scalar → UB → DMA | cache 不一致 | batch DMA 写出不可行 |
| TBuf + PipeBarrier | 不支持 MTE2/Vec overlap | 必须用 TQue |
| `TQue<VECIN, 2>` | 支持真正并行 | double buffer 唯一正确方式 |
| UB 容量 | 192KB per AIV | TQue×2 + 3×TBuf = 172KB 可行 |
| Cross-core sync | 无硬件 barrier | GM 轮询实现 |

---

## 7. 实战案例

### 7.1 RadixTopk（选择类）

| 版本 | Kernel | 核心优化 | 优化类型 |
|------|--------|---------|---------|
| 版本一 | 23.4ms | 4+4+4+4 直方图 | baseline |
| 版本二 | 20.5ms | 3+3+5+5 分桶 | 算法重构 (§1.2) |
| 版本三 | 20.8ms | 累积 GE + 精度修复 | 指令级 (§3.1) + 精度 (§4) |
| 版本四 | 15.0ms | 16 步向量化二分搜索 + 一次性 VtS | 算法重构 (§1.1, §1.3) |
| 版本五 | 13.2ms | TQue Double Buffer | 流水线并行 (§2.1) |

**总提升**：23.4ms → 13.2ms（**-44%**，1.79x vs torch.topk）

### 7.2 方法论推广到其他算子

| 算子 | 适用优化 | 关键考量 |
|------|---------|---------|
| **Sort/ArgSort** | 保序变换 (§1.3) + 二分搜索找分位点 + 分桶重排 | 需要有序输出，不能只找阈值 |
| **Unique** | 排序后去重 → 二分搜索找值域分段 + 向量化 mask | 输出大小未知 → 两遍扫描 |
| **NonZero** | 向量 mask (§3.2) + 标量紧凑输出 | 与 TopK Pass 3 输出模式相同 |
| **Histogram** | 向量化 bin counting (§3.1) + 跨核归约 | bins 数决定向量比较次数 |
| **ArgMax/ArgMin** | 向量 ReduceMax 找值 + 二分搜索定位位置 | 单值输出，更简单 |
| **Quantile** | 二分搜索 (§1.1) 找 count=N×q 的阈值 | 与 TopK 几乎相同 |
| **MaskedSelect** | 向量 mask + 标量 GM 输出 (§3.2, §3.3) | 与 TopK Pass 3 相同 |

---

## 8. 反例与反模式

| 反模式 | 问题 | 建议替代 |
|------|------|---------|
| 多级直方图低位标量扫描 | 低位元素匹配率 <2%，向量化退化为标量，scalar 主导 | 向量化二分搜索 (§1.1) |
| 重复执行浮点→sortable 变换 | N 次扫描每次都做 Transform，重复指令成本 | 一次性 Transform 写回 GM，后续直接读 (§1.3) |
| TBuf + PipeBarrier 手动 double buffer | `PipeBarrier<PIPE_ALL>` 全停，无法 MTE2/Vector overlap | 用 `TQue<VECIN, 2>` (§2.1) |
| 预先 Alloc 所有 chunk 的 Tensor | `AllocTensor` 在 queue 满时阻塞 → 流水线死锁 | Steady 阶段按需 Alloc/Free (§2.1) |
| 同步 scalar 计算替代非阻塞 GetValue | 破坏 GM 读流水线，scalar 反而变慢 | 保留 `xGm_.GetValue` 非阻塞读 (§3.3) |
| 减少 SetValue 输出量来 "省时" | 标量流水线出现空洞，反而更慢 | 接受输出成本，或用无关 GetValue 填洞 (§3.3) |
| `Adds(int16)` 跨 0x8000 直接比较 | 饱和而非回绕，符号位结果错误 | 分高/低半区做 cross-half 修正 (§4.2) |
| half ReduceSum subchunk > 2048 | half 尾数 10 bits，累加结果丢精度 | `SUBCHUNK ≤ 2048`，或换 float32 ReduceSum (§4.3) |
| 最后 chunk 不补齐就 DMA | padding 区域残留数据混入计算 | Host tiling 对齐到 `HIST_SUBCHUNK` (§4.4) |
| 每步 barrier 传 256-bin 直方图 | 跨核同步数据量大，barrier 开销显著 | 二分搜索每步只传 1 个 count (§5) |
| 默认套用通用 sort API | 大数据稀疏选择场景下吞吐极差 | 走专门的 radix / 二分搜索路径 |