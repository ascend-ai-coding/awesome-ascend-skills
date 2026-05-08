# Index Operator Tiling Strategy

适用于基于 AscendC Gather 指令的索引类算子，如 `torch.gather`、`torch.index_select`、`torch.scatter`、`torch.scatter_add`、`torch.index_add`、`torch.index_put` 等。这类算子的共同特征是：输入包含一个数据张量和一个索引张量，沿某一维度按索引收集或分散元素。

## 算子特性

- **计算模式**: 按行索引收集/分散，行间独立无依赖
- **访存模式**: 数据行顺序加载，索引驱动的随机访存（Gather 路径下由硬件处理）
- **计算强度**: 低（主要开销在 DMA 搬运和索引转换）
- **优化重点**: 根据 input 行大小选择最优执行路径，最大化多行批量处理
- **核心接口**: `AscendC::Gather`（硬件索引收集）、`AscendC::Muls`（索引→字节偏移）、`AscendC::ReinterpretCast`、`SetAtomicAdd`（原子累加）
- **BF16 处理**: 搬运类算子不做数值计算，bf16 与 fp16 均为 2 字节，可直接走 `half` 路径搬运，无需升精度到 fp32

## 两种索引模式与 3D 维度统一化

通过 `transpose(dim, -1)` 将目标维度移到最后一维，再 reshape 合成 3D：

```
batch = dims[0] * ... * dims[dim-1]     // dim 之前的维度之积
rows  = dims[dim+1] * ... * dims[ndim-1] // dim 之后的维度之积
```

### 模式 A: 单行更新（index_select 风格）— 对应 `index_op_host/kernel` 模板

index 为 1D，所有行共享同一组索引。适用于 `torch.index_select(input, dim, index)`。

> **完整实现代码**: `ascendc-operator-code-gen/templates/index_op_host.cpp` 和 `index_op_kernel.cpp`

#### 维度参数

**无需 transpose**，直接按 3D 布局计算偏移:

| 参数 | 含义 |
|------|------|
| outerSize | dim 之前的维度之积 |
| srcDimSize (N) | input 在 dim 维度的大小 |
| innerSize | dim 之后的维度之积 |
| indexSize (K) | index 长度（1D） |

```
input:  [outerSize, srcDimSize, innerSize]
index:  [K]                                   ← 1D，共享
output: [outerSize, indexSize, innerSize]

语义: output[o, k, i] = input[o, index[k], i]
偏移: inputOffset  = outerIdx * srcDimSize * innerSize + indexValue * innerSize
      outputOffset = outerIdx * indexSize  * innerSize + indexIdx   * innerSize
```

#### 数据布局

```
┌─────────────────────────────────────────────────────────────┐
│                         全局内存 (GM)                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Input:  [outerSize, srcDimSize, innerSize]            │  │
│  │  Index:  [indexSize]                                   │  │
│  │  Output: [outerSize, indexSize, innerSize]             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌───────────┐        ┌───────────┐        ┌───────────┐
    │  Core 0   │        │  Core 1   │  ...   │  Core N   │
    │ outer 0   │        │ outer 1   │        │ outer M   │
    │ index 0-K │        │ index 0-K │        │ index 0-K │
    └───────────┘        └───────────┘        └───────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐        ┌───────────┐        ┌───────────┐
    │   UB 0    │        │   UB 1    │        │  UB N     │
    │ indexBuf  │        │ indexBuf  │        │ indexBuf  │
    │ outQueue  │        │ outQueue  │        │ outQueue  │
    └───────────┘        └───────────┘        └───────────┘
```

#### Block 级 Tiling（核间切分）

**动态核分配策略**: 每个 outer 按 index 批次分配核数，最大化 indexBuf 利用率。

- 每核最多处理 `MAX_INDEX_BATCH=512` 个 index，减少索引加载次数
- `idealCoresPerOuter = ceil(indexSize / MAX_INDEX_BATCH)`
- `totalTasks = outerSize * idealCoresPerOuter`
- 若 `totalTasks ≤ 40`: 各核拿理想分配
- 若 `totalTasks > 40`: 压缩 `coresPerOuter`，核循环处理多任务

**任务分配示例**:

| outerSize | indexSize | idealCores | totalTasks | coresPerOuter | blockDim | 每核处理index数 |
|-----------|-----------|-----------|------------|---------------|----------|----------------|
| 10 | 100 | 1 | 10 | 1 | 10 | 100 |
| 10 | 1000 | 2 | 20 | 2 | 20 | 500 |
| 10 | 5000 | 10 | 100 | 10 | 40 | 500 |
| 100 | 500 | 1 | 100 | 1 | 40 | 500 |
| 1000 | 100 | 1 | 1000 | 1 | 40 | 100 |
| 1000 | 5000 | 10 | 10000 | 1 | 40 | 5000 |

**Kernel 端核循环**: taskId 交错分配（核 0 处理任务 0,40,80...；核 1 处理 1,41,81...），从 taskId 解码 `outerIdx = taskId / coresPerOuter` 和 `indexRange = [subIdx * indexPerCore, ...)`。

#### UB 级 Tiling（核内切分）

```
innerSize ≤ 4096 ?
  │
  Yes ──→ 小尾轴: 多行批量处理 (batchRows)，indexBuf 分批加载
  │
  No  ──→ 大尾轴: 分块处理，沿 innerSize 切分为 tileSize 块
```

#### UB 分配表

**小尾轴场景**（innerSize ≤ 4096）:

| Buffer | 大小计算 | 用途 |
|--------|---------|------|
| indexBuf | `ceil(indexBatchSize * 4 / 32) * 32` | 共享索引缓冲 |
| outQueue | `alignedTileSize * sizeof(T) * batchRows * BUFFER_NUM` | 双缓冲输出 |

**大尾轴场景**（innerSize > 4096）:

| Buffer | 大小计算 | 用途 |
|--------|---------|------|
| indexBuf | `ceil(indexBatchSize * 4 / 32) * 32` | 共享索引缓冲 |
| outQueue | `alignedTileSize * sizeof(T) * BUFFER_NUM` | 双缓冲输出（单行分块） |

**UB 约束**: `indexBuf + outQueue ≤ UB_CAPACITY`

- **输出形状**: `input.shape[:dim] + [K] + input.shape[dim+1:]`

### 模式 B: 单个更新（gather 风格）— 对应 `index_op_per_elem_host/kernel` 模板

index 与 output 同形（同 ndim），每行有独立索引。适用于 `torch.gather(input, dim, index)`。

| 参数 | 含义 |
|------|------|
| batch | dim 之前的维度之积 |
| rows | dim 之后的维度之积 |
| N (srcDimSize) | input 最后一维大小 |
| K (indexSize) | index 最后一维大小 |

```
input:  [..., N]  →  transpose+reshape → [batch, rows, N]
index:  [..., K]  →  transpose+reshape → [batch, rows, K]   ← 每行独立
output: [..., K]  →  transpose+reshape → [batch, rows, K]
```

- **Kernel 端**: 索引逐行加载，`maxRowsPerBatch = UB / (inputUB + indexUB + outputUB)`
- **输出形状**: `index.sizes()`

## Tiling 参数定义

### 参数结构体

```cpp
struct IndexTilingData {
    int64_t inputRows;        // 数据总行数 batch * rows（模式 B/C）
    int64_t srcDimSize;       // input 最后一维大小 N
    int64_t indexSize;        // index 最后一维大小 K
    int64_t totalRows;        // 实际参与计算的行数
    int64_t rowOffset;        // 行偏移（分批 launch 时使用）
};

// 模式 A (index_select) 专用扩展
struct IndexSelectTilingData {
    int64_t outerSize;        // dim 之前的维度之积
    int64_t srcDimSize;       // dim 维度大小
    int64_t innerSize;        // dim 之后的维度之积
    int64_t indexSize;        // index 长度（1D）

    int64_t totalLength;      // 总数据量 = outerSize * indexSize * innerSize

    int64_t coresPerOuter;    // 每个 outer 分配的核数
    int64_t totalTasks;       // 总任务数 = outerSize * coresPerOuter
    int64_t usedCoreNum;      // 实际使用的核数（≤ MAX_CORES）

    int64_t indexBatchSize;   // 索引批次大小
    int64_t indexBatchCount;  // 索引批次数
    int64_t tileSize;         // UB 单次处理的 innerSize 块大小
    int64_t alignedTileSize;  // 对齐后的 tile 长度
    int64_t tileCount;        // innerSize 分块数量
    int64_t batchRows;        // 小尾轴场景批量行数
    bool isSmallInner;        // 是否小尾轴场景
};
```

### 关键常量定义

```cpp
constexpr int64_t UB_CAPACITY = 192 * 1024;         // UB 大小限制（设备相关）
constexpr int64_t MAX_GATHER_SRC_BYTES = 32 * 1024; // Gather 指令源数据最大字节数
constexpr int64_t MAX_CORES = 40;                    // AI Core 数量（设备相关）
constexpr int64_t MAX_INDEX_BATCH = 512;             // indexBuf 最大容量（每核）
constexpr int64_t SMALL_INNER_THRESHOLD = 4096;      // 小尾轴阈值
constexpr int64_t ALIGN_NUM = 8;                     // float32: 8 elements = 32 bytes
constexpr int64_t BUFFER_NUM = 2;                    // 双缓冲
```

---

## Host 端实现

### 1. 三种分核模式

根据算子读写语义选择不同的核间分配策略：

| 模式 | 适用算子 | 核心约束 | 策略 |
|------|---------|---------|------|
| **A: 动态核分配** | index_select, gather | 只读 input，核间无冲突 | 按 index 批次动态分配 |
| **B: 每核一行** | scatter, scatter_add | 写回 self，多核不能写同一行 | 每核独占一行 |
| **C: AtomicAdd** | index_put(accumulate), index_add | 多核可能写同一位置 | 原子累加 |

### 2. 模式 A: 动态核分配

每个 outer 维度按 index 批次分配核数，总任务数超过 40 核时压缩每 outer 的核数。

```cpp
constexpr int64_t MAX_INDEX_BATCH = 512;  // indexBuf 最大容量

int64_t idealCoresPerOuter = (indexSize + MAX_INDEX_BATCH - 1) / MAX_INDEX_BATCH;
if (idealCoresPerOuter < 1) idealCoresPerOuter = 1;

int64_t totalTasks = outerSize * idealCoresPerOuter;
int64_t coresPerOuter;
uint32_t blockDim;

if (totalTasks <= MAX_CORES) {
    coresPerOuter = idealCoresPerOuter;
    blockDim = static_cast<uint32_t>(totalTasks);
} else {
    coresPerOuter = MAX_CORES / outerSize;
    if (coresPerOuter < 1) coresPerOuter = 1;
    blockDim = MAX_CORES;
}
```

**任务分配示例**:

| outerSize | indexSize | idealCores | totalTasks | coresPerOuter | blockDim |
|-----------|-----------|-----------|------------|---------------|----------|
| 10 | 100 | 1 | 10 | 1 | 10 |
| 10 | 5000 | 10 | 100 | 10 | 40 |
| 100 | 500 | 1 | 100 | 1 | 40 |
| 1000 | 5000 | 10 | 10000 | 1 | 40 |

### 3. 模式 B: 每核一行 + 分批 launch

scatter 是**写回操作**，多核写入 self 同一行会产生数据竞争。每核独占一行，outerSize > MAX_CORES 时 Host 端分批 launch：

```cpp
int64_t processed = 0;
while (processed < outerSize) {
    int64_t batch = std::min(outerSize - processed, MAX_CORES);
    uint32_t blockDim = static_cast<uint32_t>(batch);
    if (blockDim < 1) blockDim = 1;

    auto outputBatch = outputFlat.narrow(0, processed, batch);
    auto indexBatch  = indexFlat.narrow(0, processed, batch);
    auto srcBatch    = srcFlat.narrow(0, processed, batch);

    EXEC_KERNEL_CMD(scatter, blockDim, outputBatch, indexBatch, srcBatch,
                    batch, innerSize, indexSize, reduceMode, dtypeSize);
    processed += batch;
}
```

### 4. 模式 C: AtomicAdd（index_put / index_add）

accumulate 模式下多核可能写同一 output 位置。Host 端按行切分分配核，kernel 端用 `SetAtomicAdd` 保证原子累加。

```cpp
uint32_t blockDim = static_cast<uint32_t>(std::min(outerSize, MAX_CORES));
if (blockDim < 1) blockDim = 1;

EXEC_KERNEL_CMD(index_put, blockDim, selfFlat, indexFlat, valueFlat,
                batch, innerSize, indexSize, /*accumulate=*/1, dtypeSize);
```

### 5. 维度统一化

根据算子类型选择不同策略:

| 算子 | 策略 | 代码模板 |
|------|------|---------|
| index_select | 无需 transpose，3D 布局直接计算偏移 | `index_op_host.cpp` |
| gather/scatter | `transpose(dim, -1)` + reshape 为 3D | `index_op_per_elem_host.cpp` |

### 6. Index 类型转换

Host 端统一转 int32，Kernel 端按需处理:

```cpp
auto indexInt32 = indexContiguous.to(at::kInt);  // int64 → int32
```

- index_select: Kernel 端直接按 3D 偏移访问，无需 Muls
- gather: Kernel 端通过 Muls 转为字节偏移再调用 Gather 指令

### 7. 输出 reshape

- index_select: 输出形状 = `input.shape[:dim] + [K] + input.shape[dim+1:]`，直接设置
- gather: 输出形状 = `index.sizes()`，需 transpose 还原

**避免 Host 端 CPU 同步**: 不要在 Host 端做 index 范围校验（`index.to(CPU)`），会导致 NPU→CPU 同步，Host 耗时从 ~20us 飙升到 ~9000us。

---

## Kernel 端实现

> **完整实现代码**:
> - index_select: `ascendc-operator-code-gen/templates/index_op_kernel.cpp`
> - gather/scatter: `ascendc-operator-code-gen/templates/index_op_per_elem_kernel.cpp`

### 1. index_select 执行路径

index_select 无需 Gather 指令（每个 index 选择一整行 innerSize 连续元素），直接 DataCopyPad 搬运:

```
innerSize ≤ 4096 ?
  │
  Yes ──→ 小尾轴: indexBuf 分批加载 + outQueue(batchRows) 多行批量搬运
  │
  No  ──→ 大尾轴: indexBuf 分批加载 + outQueue(tileSize) 沿 innerSize 分块搬运
```

### 2. gather/scatter 执行路径（GatherPath/UBPath/ElementPath）

根据每行 UB 需求和 Gather 指令约束，自动选择最优路径：

```
                    ┌─────────────────────────────┐
                    │ 计算 perRowUB                │
                    │ inputRowBytes + indexRowBytes │
                    │ + outputRowBytes              │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ perRowUB ≤ UB_CAPACITY?       │
                    └──┬─────────────────┬─┘
                      Yes                  No
                       │                    │
          ┌────────────▼─────┐    ┌─────────▼──────────┐
          │ srcDimSize*sizeof(T)    │ ElementPath (path=2) │
          │ ≤ MAX_GATHER_SRC_BYTES? │ 逐元素 GM 直接访问    │
          │ 且 ≥ 32 字节?           │                       │
          └──┬──────────────┬─┘    └──────────────────────┘
            Yes              No
             │                │
  ┌──────────▼─────┐  ┌──────▼──────────┐
  │ GatherPath     │  │ UBPath (path=1)  │
  │ (path=0)       │  │ 标量 UB 访问     │
  │ 硬件 Gather    │  │                  │
  │ 多行批量处理   │  └─────────────────┘
  └────────────────┘
```

### 3. gather/scatter UB 级 Tiling 与 Gather 指令约束

> 完整实现代码: `index_op_per_elem_kernel.cpp`

**按 K 切分策略**:

| 维度 | 小 K 模式 | 大 K 模式 |
|------|----------|----------|
| output 写出次数 | 1 次 | ceil(K/tileK) 次 |
| input 搬入次数 | ceil(N/tileN) 次 | 每轮 1 次或按需 |
| index 搬入次数 | 1 次（常驻 UB） | ceil(K/tileK) 次 |

**Gather 指令约束**:

| 约束 | 值 | 说明 |
|------|---|------|
| 源数据最大字节数 | 32KB | srcDimSize * sizeof(T) ≤ 32768 |
| 源数据最小字节数 | 32B | srcDimSize * sizeof(T) ≥ 32 |
| 索引类型 | uint32_t | **字节偏移**，需 Muls(sizeof(T)) 转换 |
| UB 多行批量 | `maxRowsPerBatch = UB / perRowUB` | input/index/output 各乘以 maxRows |

**perRowUB 约束表**:

| 项目 | float32 | float16/bfloat16 |
|------|---------|---------|
| inputBuf | N * 4 (对齐32B) | N * 2 (对齐32B) |
| indexBuf | K * 4 (对齐32B) | K * 4 (对齐32B) |
| outputBuf | K * 4 (对齐32B) | K * 2 (对齐32B) |
| **perRowUB** | (N + 2K) * 4 | N * 2 + K * 6 |

### 4. 分核模式 A: 动态核分配 Kernel 端

当 totalTasks > MAX_CORES 时核会循环处理任务：

```cpp
int64_t blockIdx = GetBlockIdx();
int64_t totalTasks = outerSize_ * coresPerOuter_;
int64_t usedCores = min(totalTasks, MAX_CORES);
if (blockIdx >= usedCores) return;

int64_t indexPerCore = (indexSize_ + coresPerOuter_ - 1) / coresPerOuter_;

// 核 0 处理任务 0, 40, 80...; 核 1 处理 1, 41, 81...
for (int64_t taskId = blockIdx; taskId < totalTasks; taskId += usedCores) {
    int64_t outerIdx = taskId / coresPerOuter_;
    int64_t subIdx   = taskId % coresPerOuter_;
    int64_t indexStart = subIdx * indexPerCore;
    int64_t indexEnd   = min(indexStart + indexPerCore, indexSize_);
    if (indexStart >= indexSize_) continue;
    ProcessIndexRange(outerIdx, indexStart, indexEnd);
}
```

### 5. 分核模式 B: 每核一行 Kernel 端（scatter）

每核独占一行，核内串行完成"读 self 行 → 按 index 写入 src 值 → 写回 self 行"。

```cpp
int64_t blockIdx = GetBlockIdx();
if (blockIdx >= outerSize_) return;
int64_t outerIdx = blockIdx;

// 1. 读入 self 的一整行到 UB
DataCopyPad(selfLocal, selfGm[outerIdx * innerSize_], ...);
// 2. 分批加载 index + src，逐元素 scatter 写入 selfLocal
for (batch ...) { /* scatter into selfLocal */ }
// 3. 写回 self 行
DataCopyPad(selfGm[outerIdx * innerSize_], selfLocal, ...);
```

> scatter 每核独占一行，核内串行执行，不存在核间竞争。scatter_add 同理（累加也在核内完成）。只有当多核必须写同一行时才需要 AtomicAdd。

### 6. 分核模式 C: AtomicAdd Kernel 端

当多核可能写入 output 的**同一位置**时，必须使用硬件原子累加。

**反例**（先读后写，丢失累加结果）:

```cpp
// ❌ 多核同时读→加→写，会丢失部分累加
DataCopyPad(outLocal, outputGm[outOffset], ...);  // 读
Add(outLocal, outLocal, srcLocal, len);             // 加
DataCopyPad(outputGm[outOffset], outLocal, ...);    // 写
```

**正例**（AtomicAdd，单次搬入一次搬出）:

```cpp
// ✅ 原子累加：src 加载一次到 UB，一次原子写出到 GM
SetAtomicAdd<T>();
DataCopyPad(outputGm[outOffset], srcLocal, copyParams);  // 原子累加到 GM
SetAtomicNone();  // 必须立即关闭！
```

#### API 说明

| API | 功能 |
|-----|------|
| `SetAtomicAdd<T>()` | 开启原子加法，后续 DataCopyPad 将数据原子加到目标地址 |
| `SetAtomicNone()` | **必须立即关闭**，否则污染后续内存操作 |

#### 完整示例

```cpp
__aicore__ inline void AtomicAddRow(int64_t gmOffset, LocalTensor<T> src, int64_t len) {
    SetAtomicAdd<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(len * sizeof(T));
    DataCopyPad(outputGm[gmOffset], src, copyParams);  // 原子累加
    SetAtomicNone();  // 必须立即关闭
}
```

**关键注意事项**:

1. **必须立即关闭**: `SetAtomicAdd` 后紧跟 DataCopyPad，然后立即 `SetAtomicNone`
2. **性能权衡**: 原子操作比普通写慢，仅在存在数据竞争风险时使用
3. **适用场景**: index_put(accumulate)、index_add、scatter_add（多核同行时）、embedding_bag

**实测效果**（index_add 算子）:

| 指标 | 先读后写 | AtomicAdd |
|------|---------|-----------|
| 精度通过率 | 45% | 97%+ |
| MARE | 高（数据竞争） | < 1e-6 |

### 7. 同步策略

初期开发使用 `PipeBarrier<PIPE_ALL>()`，正确性验证后替换为轻量同步：

| 阶段 | 推荐同步 | 说明 |
|------|---------|------|
| DataCopyPad GM→UB | `event_t MTE2_S` | 仅同步 MTE2→S |
| Muls 向量计算 | `event_t V_S` | 仅同步 V→S |
| Gather 向量计算 | `event_t V_S` | 仅同步 V→S |
| DataCopyPad UB→GM | `event_t MTE3_S` | 仅同步 S→MTE3 |

```cpp
event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
SetFlag<HardEvent::MTE2_S>(evt);
WaitFlag<HardEvent::MTE2_S>(evt);
```

---

## 适用算子列表

| 算子 | 分核模式 | 维度参数 | Host 端 | Kernel 端 |
|------|---------|---------|---------|----------|
| `torch.index_select(input, dim, index)` | A: 动态核分配 | outerSize/srcDimSize/innerSize | 3D 布局，1D index | indexBuf 共享，innerSize 批量拷贝 |
| `torch.gather(input, dim, index)` | A: 动态核分配 | batch/rows/N | transpose + 3D reshape | Gather 收集，逐行 index |
| `torch.scatter(input, dim, index, src)` | B: 每核一行 | batch/rows/N | 双输入 + transpose | 读→scatter→写回 |
| `torch.scatter_add(input, dim, index, src)` | B: 每核一行 | batch/rows/N | 累加语义 | 核内 GetValue+Add+SetValue |
| `torch.index_add(input, dim, index, src, alpha)` | C: AtomicAdd | batch/rows/N | 累加语义 | SetAtomicAdd 原子累加 |
| `torch.index_put(input, indices, values, accumulate=True)` | C: AtomicAdd | batch/rows/N | 累加语义 | SetAtomicAdd 原子累加 |

## 全流程总结

**三级决策**: 先按读写语义选分核模式(A/B/C)，再按 N 选执行路径(GatherPath/UBPath/ElementPath)，最后按 K 选搬入搬出策略(小K多次搬入一次搬出/大K分批搬入搬出)。

| 分核模式 | 执行路径 | K 模式 | UB 分配 | AscendC 接口 |
|---------|---------|--------|---------|-------------|
| A: 动态核分配 (index_select) | 小尾轴 | — | indexBuf(K) + outQueue(innerSize × batchRows) | indexBuf 共享，多行批量 CopyInRow + CopyOutBatch |
| A: 动态核分配 (index_select) | 大尾轴 | — | indexBuf(K) + outQueue(tileSize) | 沿 innerSize 分块处理 |
| A: 动态核分配 (gather) | GatherPath | 小 K | inputBuf(N) + indexBuf(K) + outputBuf(K) | DataCopyPad + Muls + Gather |
| A: 动态核分配 (gather) | GatherPath | 大 K | inputBuf(N) + indexBuf(tileK) + outputBuf(tileK) | 沿 K 分批 Gather |
| A: 动态核分配 (gather) | UBPath | — | ubInputBuf(N) + ubIndexBuf(K) + ubOutputBuf(K) | GetValue/SetValue |
| A: 动态核分配 (gather) | ElementPath | — | indexBuf + outputBuf | GM.GetValue |
| B: 每核一行 | — | — | selfBuf(innerSize) + indexBuf + srcBuf | 读→scatter→写回 |
| C: AtomicAdd | — | 小 K | srcBuf + indexBuf | 单次搬入, SetAtomicAdd 一次搬出 |
