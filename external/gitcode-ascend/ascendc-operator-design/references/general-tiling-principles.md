# General Tiling Principles

适用于所有算子类型的通用 tiling 原则和策略。

## Tiling 基本概念

### 为什么需要 Tiling？

1. **内存限制**: UB (Unified Buffer) 大小有限（如 910B 示例为 192KB，实际编码时通过接口获取）
2. **并行性**: 将大任务分解为可并行的小任务
3. **数据局部性**: 提高缓存命中率
4. **延迟隐藏**: 使用 double buffer 隐藏内存访问延迟

### Tiling 的核心目标

- **最大化 UB 利用率**: 尽可能使用大的 tile size
- **最小化 GM 访问**: 减少全局内存访问次数
- **最大化并行度**: 充分利用多个 block/core
- **保持对齐**: 满足硬件对齐要求

## 两级 Tiling 架构

AscendC 算子采用两级 Tiling 策略：

```
┌─────────────────────────────────────────────────────────────┐
│                    全局内存 (GM)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              totalLength 元素数据                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Core 0  │     │  Core 1  │ ... │ Core 39  │   ← Block级Tiling (核间切分)
    │ formerLen│     │ formerLen│     │ tailLen  │
    └──────────┘     └──────────┘     └──────────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   UB 0   │     │   UB 1   │     │  UB 39   │   ← UB级Tiling (核内切分)
    │ tileLen  │     │ tileLen  │     │ tileLen  │
    │ tileLen  │     │ tileLen  │     │ tileLen  │
    │   ...    │     │   ...    │     │   ...    │
    └──────────┘     └──────────┘     └──────────┘
```

## Block级 Tiling（核间切分）

将数据分配到多个 AI Core 上并行处理。

### 策略要点

1. **Cache Line 对齐**: 512 字节
2. **负载均衡**: 整核/尾核策略

### Cache Line 对齐（512字节）

Block级 Tiling 时，每个核处理的数据块必须 512 字节对齐，以优化内存访问效率。

```cpp
constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;

// 计算对齐后的元素数
int64_t alignElements = CACHE_LINE_BYTE_LENGTH / dtypeSize;

// 向上对齐
int64_t alignedLength = ((length + alignElements - 1) / alignElements) * alignElements;

// 向下对齐
int64_t alignedLength = (length / alignElements) * alignElements;
```

### 整核/尾核负载均衡

为了实现核间负载均衡，将核心分为整核和尾核：

- **整核 (former)**: 处理略多的数据
- **尾核 (tail)**: 处理略少的数据
- **差异**: 最多相差一个 Cache Line 大小

```cpp
constexpr int64_t CORE_NUM = 40;  // 核心数量（实际编码时通过接口获取）

// 计算对齐后的总长度（以 Cache Line 为单位）
int64_t totalLengthAligned = ((totalLength + CACHE_LINE_BYTE_LENGTH - 1) / CACHE_LINE_BYTE_LENGTH) * CACHE_LINE_BYTE_LENGTH;

// 整核数量 = 余数
int64_t formerNum = (totalLengthAligned / CACHE_LINE_BYTE_LENGTH) % CORE_NUM;
if (formerNum == 0) {
    formerNum = CORE_NUM;  // 刚好平均分配时，所有核都是整核
}

// 尾核数量
int64_t tailNum = CORE_NUM - formerNum;

// 整核数据长度（向上对齐）
int64_t formerLength = ((totalLengthAligned / CORE_NUM + CACHE_LINE_BYTE_LENGTH - 1) / CACHE_LINE_BYTE_LENGTH) * CACHE_LINE_BYTE_LENGTH;

// 尾核数据长度（向下对齐）
int64_t tailLength = (totalLengthAligned / CORE_NUM / CACHE_LINE_BYTE_LENGTH) * CACHE_LINE_BYTE_LENGTH;
```

### 数据分布示意

```
GM:  [============================================================]
       │←───── formerLength ─────→│←── tailLength ──→│
       │←──── Core 0 ────→│←─── Core 1 ───→│ ... │←─ Core 39 ─→│
       └────── 整核 ──────┘                └──── 尾核 ────┘
```

## UB级 Tiling（核内切分）

在每个 AI Core 内部，将数据分块处理。

### 策略要点

1. 根据 UB 分配表确定 buffer 系数
2. tileLength 由 Host 端根据 UB 大小计算
3. **UB 对齐**: 32 字节
4. **UB 容量**: 不超过 UB_SIZE_LIMIT（实际编码时通过接口获取）

### 精度处理说明

**重要**: NPU 计算单元不支持 float16/bfloat16 数据类型的直接计算，必须升精度到 float32 后再进行计算。

| 输入数据类型 | 处理方式 | 计算精度 | UB 影响 |
|------------|---------|---------|--------|
| float16 | **升精度到 float32** | float32 | 需要额外 float32 buffer |
| bfloat16 | **升精度到 float32** | float32 | 需要额外 float32 buffer |
| float32 | 直接计算 | float32 | 无额外开销 |

对于 float16/bfloat16 输入的算子，需要分配额外的 float32 计算缓冲区：

```cpp
// float16/bfloat16 输入时的处理流程
LocalTensor<half> xLocalFp16 = inQueueX.AllocTensor<half>();
LocalTensor<float> xLocalFp32 = tempBuffer.Get<float>();

// 升精度到 float32
Cast(xLocalFp32, xLocalFp16, RoundMode::CAST_NONE, tileLength);
// 在 float32 上计算
Compute(outputLocalFp32, xLocalFp32, tileLength);
// 降精度回 float16
Cast(outputLocalFp16, outputLocalFp32, RoundMode::CAST_RINT, tileLength);
```

### UB 对齐（32字节）

UB 内部数据访问要求 32 字节对齐。

```cpp
constexpr int64_t UB_ALIGN_BYTES = 32;

// 计算对齐后的元素数
int64_t ubAlignElements = UB_ALIGN_BYTES / dtypeSize;

// 对齐 tileLength
int64_t tileLengthAligned = ((tileLength + ubAlignElements - 1) / ubAlignElements) * ubAlignElements;
```

### 常见数据类型的对齐元素数

| 数据类型 | 大小 | Cache Line 对齐元素数 | UB 对齐元素数 |
|----------|------|----------------------|--------------|
| bfloat16 | 2 bytes | 256 | 16 |
| float16 | 2 bytes | 256 | 16 |
| float32 | 4 bytes | 128 | 8 |

### UB 分配表模板

#### 单输入单输出

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * dtypeSize | 输入数据缓冲 | BUFFER_NUM | tileLength * dtypeSize * 2 |
| outQueueY  | tileLength * dtypeSize | 输出数据缓冲 | BUFFER_NUM | tileLength * dtypeSize * 2 |
| **总计**   | - | - | - | **tileLength * dtypeSize * 4** |

**约束**: `tileLength * dtypeSize * 4 <= UB_SIZE_LIMIT`

#### 双输入单输出

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * dtypeSize | 输入1数据缓冲 | BUFFER_NUM | tileLength * dtypeSize * 2 |
| inQueueY   | tileLength * dtypeSize | 输入2数据缓冲 | BUFFER_NUM | tileLength * dtypeSize * 2 |
| outQueueZ  | tileLength * dtypeSize | 输出数据缓冲 | BUFFER_NUM | tileLength * dtypeSize * 2 |
| **总计**   | - | - | - | **tileLength * dtypeSize * 6** |

**约束**: `tileLength * dtypeSize * 6 <= UB_SIZE_LIMIT`

#### 带临时 Buffer

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | inputTileLength * dtypeSize | 输入数据缓冲 | BUFFER_NUM | inputTileLength * dtypeSize * 2 |
| tempBuffer | tempBufferSize * dtypeSize | 临时计算缓冲 | 1 | tempBufferSize * dtypeSize |
| outQueueY  | outputTileLength * dtypeSize | 输出数据缓冲 | BUFFER_NUM | outputTileLength * dtypeSize * 2 |
| **总计**   | - | - | - | **((inputTileLength + outputTileLength) * 2 + tempBufferSize) * dtypeSize** |

**约束**: `((inputTileLength + outputTileLength) * 2 + tempBufferSize) * dtypeSize <= UB_SIZE_LIMIT`

### tileLength 计算

根据 UB 分配表中的总系数，计算 tileLength：

```cpp
// 根据算子类型和数据类型确定 buffer 系数
// float32 单输入: 系数 = 4
// float32 双输入: 系数 = 6
// float16/bfloat16 单输入: 系数 = 8 (需要额外 float32 buffer)
// float16/bfloat16 双输入: 系数 = 12 (需要额外 float32 buffer)

int64_t bufferCoefficient = /* 根据UB分配表确定 */;
int64_t maxTileElements = UB_SIZE_LIMIT / bufferCoefficient;  // UB_SIZE_LIMIT 实际编码时通过接口获取

// 32 字节对齐
int64_t alignElements = 32 / dtypeSize;
int64_t tileLength = (maxTileElements / alignElements) * alignElements;
```

## 通用 Tiling 参数结构体

### 基础结构体

```cpp
struct BaseTilingData {
    int64_t totalLength;        // 总数据长度

    int64_t formerNum;          // 整核数量
    int64_t formerLength;       // 整核数据长度
    int64_t tailNum;            // 尾核数量
    int64_t tailLength;         // 尾核数据长度

    int64_t tileLength;         // UB单次处理长度
};
```

### 扩展结构体（用于复杂算子）

```cpp
struct AdvancedTilingData {
    // 基础参数
    int64_t totalLength;
    int64_t formerNum;
    int64_t formerLength;
    int64_t tailNum;
    int64_t tailLength;
    int64_t tileLength;

    // 扩展参数
    int64_t inputTileLength;    // 输入 tile 长度
    int64_t outputTileLength;   // 输出 tile 长度
    int64_t tempBufferSize;     // 临时 buffer 大小

    // 多维参数
    int64_t tileDimX;           // X 维度 tile 大小
    int64_t tileDimY;           // Y 维度 tile 大小
    int64_t tileDimZ;           // Z 维度 tile 大小
};
```

## Double Buffer 优化

### 原理

```
时间线：
Tile 0: [Load] [Compute] [Store]
Tile 1:        [Load] [Compute] [Store]
Tile 2:               [Load] [Compute] [Store]

使用 Double Buffer：
Buffer A: [Load Tile 0] -------- [Load Tile 2] --------
Buffer B: -------- [Load Tile 1] -------- [Load Tile 3]
Compute:  -------- [Compute T0] [Compute T1] [Compute T2]
```

### 配置

```cpp
constexpr int64_t BUFFER_NUM = 2;  // Double buffer

// 初始化 queue
pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * dtypeSize);
pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * dtypeSize);
```

### 使用

```cpp
// CopyIn: 使用 AllocTensor 自动轮换 buffer
LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
DataCopy(xLocal, xGlobal[tileIdx * tileLength], currentTileLength);
inQueueX.EnQue(xLocal);

// Compute: 从 queue 中取出数据
LocalTensor<T> xLocal = inQueueX.DeQue<T>();
LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
Compute(yLocal, xLocal, currentTileLength);
inQueueX.FreeTensor(xLocal);
outQueueY.EnQue<T>(yLocal);

// CopyOut: 写回并释放 buffer
LocalTensor<T> yLocal = outQueueY.DeQue<T>();
DataCopy(yGlobal[tileIdx * tileLength], yLocal, currentTileLength);
outQueueY.FreeTensor(yLocal);
```

## 常见优化技巧

### 1. 向量化访问

```cpp
// 使用 DataCopy 而不是逐元素访问
// 好
DataCopy(dstLocal, srcGlobal, tileLength);

// 避免
for (int64_t i = 0; i < tileLength; ++i) {
    dstLocal.SetValue(i, srcGlobal.GetValue(i));
}
```

### 2. 数据复用

```cpp
// 将可复用的数据加载到 UB 并多次使用
LocalTensor<T> weightLocal = weightBuffer.Get<T>();
DataCopy(weightLocal, weightGlobal, weightSize);

for (int64_t i = 0; i < iterations; ++i) {
    // 复用 weightLocal
    Compute(outputLocal, inputLocal, weightLocal);
}
```

### 3. 边界处理

```cpp
// 计算当前核处理的 tile 数量和尾块长度
int64_t tileNum = (blockLength + tileLength - 1) / tileLength;
int64_t tailTileLength = blockLength - (tileNum - 1) * tileLength;

// 处理整块
for (int64_t i = 0; i < tileNum - 1; ++i) {
    ProcessTile(i, tileLength);
}

// 处理尾块
ProcessTile(tileNum - 1, tailTileLength);
```

### 4. 避免 Bank Conflict

```cpp
// 确保 buffer 大小不会导致 bank conflict
// 通常通过合理的对齐和 padding 解决
int64_t bufferSize = AlignTo32(tileLength * dtypeSize, dtypeSize);
```

### 5. 减少同步

```cpp
// 尽量减少 block 间同步
// 使用单 block 或独立的 tile 处理
```

## Tiling 策略选择决策树

```
开始
  |
  v
数据量 < 10K? --Yes--> 单 Block, 一次处理
  |
  No
  v
逐元素操作? --Yes--> Elementwise Tiling (see elementwise-tiling.md)
  |
  No
  v
归约操作? --Yes--> Reduction Tiling (see reduction-tiling.md)
  |
  No
  v
其他复杂操作? --Yes--> 自定义 Tiling (参考通用原则)
  |
  No
  v
使用基础 Tiling 策略
```

## 调试技巧

### 1. 打印 Tiling 参数

```cpp
// Host 端打印
printf("Tiling Params:\n");
printf("  totalLength: %ld\n", params.totalLength);
printf("  formerNum: %ld, formerLength: %ld\n", params.formerNum, params.formerLength);
printf("  tailNum: %ld, tailLength: %ld\n", params.tailNum, params.tailLength);
printf("  tileLength: %ld\n", params.tileLength);
printf("  UB usage: %ld bytes\n", params.tileLength * dtypeSize * bufferCoefficient);
```

### 2. 验证对齐

```cpp
// 检查 tileLength 是否对齐
if (tileLength * dtypeSize % 32 != 0) {
    printf("Warning: tileLength not aligned to 32 bytes!\n");
}

// 检查 blockLength 是否对齐
if (blockLength * dtypeSize % 512 != 0) {
    printf("Warning: blockLength not aligned to 512 bytes!\n");
}
```

### 3. 检查 UB 大小

```cpp
// 确保 UB 使用不超过限制（UB_SIZE_LIMIT 实际编码时通过接口获取）
int64_t totalUB = /* 计算总 UB 使用 */;
if (totalUB > UB_SIZE_LIMIT) {
    printf("Error: UB usage (%ld) exceeds limit (%ld)!\n", totalUB, UB_SIZE_LIMIT);
}
```

## 参考资源

- AscendC 编程指南
- CANN 开发者文档
- 性能优化最佳实践

## 相关 Tiling 策略文档

- `elementwise-tiling.md` - 逐元素操作（包含完整的两级 Tiling 实现）
- `reduction-tiling.md` - 归约操作
