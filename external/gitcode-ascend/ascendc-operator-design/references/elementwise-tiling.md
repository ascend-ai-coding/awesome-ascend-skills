# Elementwise Operator Tiling Strategy

适用于逐元素操作（如 add, mul, relu, sigmoid 等）的 tiling 策略。

## 算子特性

- **计算模式**: 每个元素独立计算，无数据依赖
- **访存模式**: 规则的顺序访问，易于向量化
- **计算强度**: 低（通常是 memory-bound）
- **优化重点**: 最大化内存带宽利用率

## 两级 Tiling 策略概述

AscendC 算子采用两级 Tiling 策略来充分利用硬件并行能力：

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

## Tiling 参数定义

### 参数结构体

```cpp
struct ElementwiseTilingData {
    int64_t totalLength;        // 总数据长度
    int64_t usedCoreNum;        // 实际使用的核数

    int64_t formerNum;          // 整核数量
    int64_t formerLength;       // 整核数据长度
    int64_t tailNum;            // 尾核数量
    int64_t tailLength;         // 尾核数据长度

    int64_t tileLength;         // UB单次可处理的数据
};
```

### 关键常量定义

```cpp
constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;  // Cache Line 大小
constexpr int64_t CORE_NUM = 40;                  // 核心数量（实际编码时通过接口获取）
constexpr int64_t UB_SIZE_LIMIT = 192 * 1024;     // UB 大小限制（实际编码时通过接口获取）
constexpr int64_t ALIGN_NUM = CACHE_LINE_BYTE_LENGTH;  // 对齐单位
```

## Block级Tiling（核间切分）

Block级Tiling将数据分配到多个核心上，实现核间并行。

### 策略要点

1. **Cache Line对齐**: 每个核处理的数据块大小必须是 512 字节对齐
2. **核间负载均衡**: 分为整核和尾核，每个核处理的数据最多只差一个 Cache Line 大小

### 参数计算公式

```cpp
// 1. 计算整核上的数据量
int64_t totalLengthCore = (totalLength + CORE_NUM - 1) / CORE_NUM;
int64_t totalLengthCoreAlign = (totalLengthCore + CACHE_LINE_BYTE_LENGTH - 1) / CACHE_LINE_BYTE_LENGTH * CACHE_LINE_BYTE_LENGTH;

// 2. 实际使用的核数
int64_t usedCoreNum = (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;

// 3. 计算整核数量
int64_t formerNum = usedCoreNum - 1;

// 4.计算整核数据长度
int64_t formerLength = totalLengthCoreAlign;

// 5. 计算尾核数量
int64_t tailNum = 1;

// 6. 计算尾核数据长度
int64_t tailLength = totalLength - (usedCoreNum - 1) * formerLength;
```

### 核间数据分布示例

假设 `totalLength = 1,000,000`，`CORE_NUM = 40`，`dtypeSize = 2` (float16)：

| 核类型 | 数量 | 每核数据量 | 说明 |
|--------|------|-----------|------|
| 整核 | 39 | 12,544 元素 | 处理略少的数据 |
| 尾核 | 1 | 21,568 元素 | 处理略多的数据 |

### 核间切分 验证

formerNum * formerLength + tailNum * tailLength == totalLength

### 核间切分验证模板

```
- **核间切分是否满足约束**: [是/否]
```

## UB级Tiling（核内UB切分）

UB级Tiling在每个核内部将数据分块处理，适应 UB 缓冲区大小。

### 策略要点

1. **根据 UB 分配表确定 buffer 需求**
2. **tileLength 由 Host 端根据 UB 大小和数据类型计算**
3. **每个核内部循环处理多个 UB 块**
4. **32 字节对齐**（UB 内部对齐要求）

### 精度处理说明

**重要**: NPU 计算单元不支持 float16/bfloat16 数据类型的直接计算，必须升精度到 float32 后再进行计算。

| 输入数据类型 | 处理方式 | 计算精度 | UB 影响 |
|------------|---------|---------|--------|
| float16 | **升精度到 float32** | float16 | 无额外开销 |
| bfloat16 | **升精度到 float32** | float32 | 需要额外 float32 buffer |
| float32 | 直接计算 | float32 | 无额外开销 |

对于 float16/bfloat16 输入的算子，需要在 UB 中分配额外的 float32 计算缓冲区：

```cpp
// float16/bfloat16 输入时的处理流程
// 1. 加载 float16/bfloat16 数据到 UB
// 2. Cast 到 float32 进行计算
// 3. Cast 结果回 float16/bfloat16
// 4. 写出到 GM

LocalTensor<half> xLocalFp16 = inQueueX.AllocTensor<half>();
LocalTensor<float> xLocalFp32 = tempBuffer.Get<float>();

// 升精度到 float32
Cast(xLocalFp32, xLocalFp16, RoundMode::CAST_NONE, tileLength);
// 在 float32 上计算
Compute(outputLocalFp32, xLocalFp32, tileLength);
// 降精度回 float16
Cast(outputLocalFp16, outputLocalFp32, RoundMode::CAST_RINT, tileLength);
```

### UB 分配表

根据算子的输入数量和数据类型，UB 分配有所不同：

#### 单输入算子（如 relu, sigmoid）

**float32 输入:**

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * 4 | 输入数据缓冲 (fp32) | BUFFER_NUM | tileLength * 8 |
| outQueueY  | tileLength * 4 | 输出数据缓冲 (fp32) | BUFFER_NUM | tileLength * 8 |
| tempBuffer | tileLength * 4 | 计算缓冲 | 1 | tileLength * 4 |
| **总计**   | - | - | - | **tileLength * 20** |

**float16/bfloat16 输入（需要升精度到 float32）:**

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * 2 | 输入数据缓冲 (bf16/fp16) | BUFFER_NUM | tileLength * 4 |
| outQueueY  | tileLength * 2 | 输出数据缓冲 (bf16/fp16) | BUFFER_NUM | tileLength * 4 |
| tempBuffer | tileLength * 4 | float32 计算缓冲 | 2 | tileLength * 8 |
| **总计**   | - | - | - | **tileLength * 16** |

#### 双输入算子（如 add, mul）

**float32 输入:**

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * 4 | 输入1数据缓冲 (fp32) | BUFFER_NUM | tileLength * 8|
| inQueueY   | tileLength * 4 | 输入2数据缓冲 (fp32) | BUFFER_NUM | tileLength * 8 |
| outQueueZ  | tileLength * 4 | 输出数据缓冲 (fp32) | BUFFER_NUM | tileLength * 8 |
| tempBuffer | tileLength * 4 | 计算缓冲 | 2 | tileLength * 8 |
| **总计**   | - | - | - | **tileLength * 32** |

**float16/bfloat16 输入（需要升精度到 float32）:**

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * 2 | 输入1数据缓冲 (bf16/fp16) | BUFFER_NUM | tileLength * 4 |
| inQueueY   | tileLength * 2 | 输入2数据缓冲 (bf16/fp16) | BUFFER_NUM | tileLength * 4 |
| outQueueZ  | tileLength * 2 | 输出数据缓冲 (bf16/fp16) | BUFFER_NUM | tileLength * 4 |
| tempBufferX | tileLength * 4 | float32 计算缓冲1 | 1 | tileLength * 4 |
| tempBufferY | tileLength * 4 | float32 计算缓冲2 | 1 | tileLength * 4 |
| tempBufferZ | tileLength * 4 | float32 输出缓冲 | 1 | tileLength * 4 |
| **总计**   | - | - | - | **tileLength * 24** |

### tileLength 计算

根据 UB 分配表中的总系数，计算 tileLength：

```cpp
// 根据 UB 分配表确定 buffer 系数
// float32 单输入: 系数 = 20
// float32 双输入: 系数 = 32
// float16/bfloat16 单输入: 系数 = 16 (需要额外 float32 buffer)
// float16/bfloat16 双输入: 系数 = 24 (需要额外 float32 buffer)

int64_t bufferCoefficient = 24;  // 示例：float16 双输入算子
int64_t maxTileElements = UB_SIZE_LIMIT / bufferCoefficient;

// 32 字节对齐
int64_t alignElements = 32 / sizeof(DTYPE_X);
int64_t tileLength = (maxTileElements / alignElements) * alignElements;
```

### UB 约束验证

| 算子类型 | 数据类型 | 约束条件 | 最大 tileLength（示例） |
|----------|----------|----------|----------------|
| 单输入 | float32 | tileLength * 20 <= UB_SIZE_LIMIT | 9830 |
| 单输入 | float16 | tileLength * 16 <= UB_SIZE_LIMIT | 12288 |
| 单输入 | bfloat16 | tileLength * 16 <= UB_SIZE_LIMIT | 12288 |
| 双输入 | float32 | tileLength * 32 <= UB_SIZE_LIMIT | 6144 |
| 双输入 | float16 | tileLength * 24 <= UB_SIZE_LIMIT | 8192 |
| 双输入 | bfloat16 | tileLength * 24 <= UB_SIZE_LIMIT | 8192 |

### UB约束验证模板

```
- **UB使用**: [X] bytes ([Y]% of UB_SIZE_LIMIT)
- **UB限制**: [Z] bytes（UB_SIZE_LIMIT 实际编码时通过接口获取，示例值 192KB）
- **是否满足约束**: [是/否]
- **对齐要求**: 32字节对齐
```

## Kernel侧实现

### 执行流程（核内循环）

```cpp
__aicore__ inline void Process() {
    int64_t coreLength = AscendC::GetBlockIdx() == tiling->usedCoreNum - 1 ? this->tailLength : this->formerLength;
    int64_t tileNum = (coreLength + this->tileLength - 1) / this->tileLength;
    int64_t tailTileLength = coreLength - (tileNum - 1) * this->tileLength;

    for (int64_t i = 0; i < tileNum - 1; ++i) {
        // 处理整块 tileLength
        CopyIn(i, this->tileLength);
        Compute(i, this->tileLength);
        CopyOut(i, this->tileLength);
    }
    // 处理尾块 tailTileLength
    CopyIn(tileNum - 1, tailTileLength);
    Compute(tileNum - 1, tailTileLength);
    CopyOut(tileNum - 1, tailTileLength);
}
```

## Host 端完整示例

```cpp
void ComputeTilingData(const std::vector<int64_t>& shape,
                       int64_t dtypeSize,
                       bool isBfloat16,
                       int64_t inputCount,  // 1 或 2
                       ElementwiseTilingData& params) {
    // 计算总长度
    params.totalLength = 1;
    for (auto dim : shape) {
        params.totalLength *= dim;
    }

    // Block级 Tiling 参数计算
    int64_t totalLengthCore = (totalLength + CORE_NUM - 1) / CORE_NUM;
    int64_t totalLengthCoreAlign = (totalLengthCore + CACHE_LINE_BYTE_LENGTH - 1) / CACHE_LINE_BYTE_LENGTH * CACHE_LINE_BYTE_LENGTH;

    int64_t usedCoreNum = (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;

    int64_t formerNum = usedCoreNum - 1;

    int64_t formerLength = totalLengthCoreAlign;

    int64_t tailNum = 1;

    int64_t tailLength = totalLength - (usedCoreNum - 1) * formerLength;

    // UB级 Tiling 参数计算
    // 根据 UB 分配表确定 buffer 系数
    int64_t bufferCoefficient;
    if (inputCount == 1) {
        bufferCoefficient = isBfloat16 ? 16 : 20;  // 单输入
    } else {
        bufferCoefficient = isBfloat16 ? 24 : 32;  // 双输入
    }

    int64_t maxTileElements = UB_SIZE_LIMIT / bufferCoefficient;  // UB_SIZE_LIMIT 实际编码时通过接口获取

    // 32 字节对齐
    int64_t alignElements = 32 / dtypeSize;
    params.tileLength = (maxTileElements / alignElements) * alignElements;
}
```

## 优化建议

### 1. Double Buffer

使用 double buffer 隐藏内存延迟，实现计算与数据传输的流水线。

```cpp
constexpr int64_t BUFFER_NUM = 2;
pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * dtypeSize);
pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * dtypeSize);
pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * dtypeSize);
```

### 2. 向量化加载

使用 DataCopy 一次性加载整个 tile，而非逐元素访问。

```cpp
// 推荐
LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
DataCopy(xLocal, xGlobal, tileLength);

// 避免
for (int64_t i = 0; i < tileLength; ++i) {
    xLocal.SetValue(i, xGlobal.GetValue(i));
}
```

### 3. 边界处理

最后一个 tile 可能不满，需要使用实际长度。

```cpp
int64_t currentTileLength = (tileIdx == tileNum - 1) ? tailTileLength : tileLength;
```

## 参考实现

- `csrc/ops/add/add.cpp` - 双输入 elementwise 算子
- `csrc/ops/relu/relu.cpp` - 单输入 elementwise 算子
