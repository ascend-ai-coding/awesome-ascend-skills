# Reduction Operator Tiling Strategy

适用于归约操作（如 sum, max, min, mean 等）的 tiling 策略。

## 算子特性

- **计算模式**: 沿某个轴或全局归约，有数据依赖
- **访存模式**: 需要多次读取或跨轴访问
- **计算强度**: 低到中等（通常是 memory-bound）
- **优化重点**: 减少中间结果存储，优化归约轴访问

## 两级 Tiling 策略概述

```
┌─────────────────────────────────────────────────────────────┐
│                    全局内存 (GM)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              totalLength 非归约维度元素总数            │   │
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
struct ReductionTilingData {
    int64_t totalLength;        // 总数据长度（非归约维度元素总数）
    int64_t reduceLength;       // 归约轴长度
    int64_t reduceLengthAlign;  // 对齐后的归约长度

    int64_t formerNum;           // 整核数量
    int64_t formerLength;        // 整核数据长度
    int64_t tailNum;             // 尾核数量
    int64_t tailLength;          // 尾核数据长度

    int64_t tileLength;          // UB单次处理长度
    int64_t isReduceLastDim;    // 是否在最后一维归约
};
```

### 关键常量定义

```cpp
constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;  // Cache Line 大小
constexpr int64_t CORE_NUM = 40;                  // 核心数量（实际编码时通过接口获取）
constexpr int64_t UB_SIZE_LIMIT = 192 * 1024;     // UB 大小限制（实际编码时通过接口获取）
```

## Block级Tiling（核间切分）

### 策略要点

1. **Cache Line对齐**: 每个核处理的数据块 512 字节对齐
2. **负载均衡**: 整核/尾核策略

### 参数计算

```cpp
// 整核数量
int64_t formerNum = totalLength % CORE_NUM;
if (formerNum == 0) {
    formerNum = CORE_NUM;
}

// 尾核数量
int64_t tailNum = CORE_NUM - formerNum;

// 归约轴对齐后的长度
int64_t reduceLengthAlign = ((reduceLength + CACHE_LINE_BYTE_LENGTH - 1) / CACHE_LINE_BYTE_LENGTH) * CACHE_LINE_BYTE_LENGTH;

// 整核数据长度
int64_t formerLength = totalLength / CORE_NUM + 1;

// 尾核数据长度
int64_t tailLength = totalLength / CORE_NUM;
```

## UB级Tiling（核内切分）

### 策略要点

1. 根据 UB 分配表确定 buffer 需求
2. tileLength 由 Host 端根据 UB 大小计算
3. 32 字节对齐（UB 内部对齐要求）

### 精度处理说明

**重要**: NPU 计算单元不支持 float16/bfloat16 数据类型的直接计算，必须升精度到 float32 后再进行计算。

| 输入数据类型 | 处理方式 | 计算精度 | UB 影响 |
|------------|---------|---------|--------|
| float16 | **升精度到 float32** | float32 | 需要额外 float32 buffer |
| bfloat16 | **升精度到 float32** | float32 | 需要额外 float32 buffer |
| float32 | 直接计算 | float32 | 无额外开销 |

对于 bfloat16/float16 输入的归约算子，通常需要升精度到 FP32 进行计算以避免精度损失：

```cpp
// 升精度示例
LocalTensor<float> xLocalFp32 = tempBuffer.Get<float>();
Cast(xLocalFp32, xLocalFp16, RoundMode::CAST_NONE, tileLength);
// 在 FP32 上进行归约计算
ReduceSum(outputLocal, xLocalFp32, reduceLength);
```

### UB 分配表

#### 单轴归约（沿最后一维）

**float32 输入:**

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * reduceLengthAlign * dtypeSize | 输入数据缓冲 | 2 | tileLength * reduceLengthAlign * dtypeSize |
| outQueueY  | tileLength * reduceLengthAlign * dtypeSize | 归约结果 | 2 | tileLength * reduceLengthAlign * dtypeSize |
| tempBuffer | reduceLengthAlign * dtypeSize + 32 (对齐) | 临时结果 | 1 | reduceLengthAlign * dtypeSize + 32 |
| **总计**   | - | - | - | **tileLength * reduceLengthAlign * dtypeSize * 4 + reduceLengthAlign * dtypeSize + 32** |

**float16/bfloat16 输入（需要升精度到 float32）:**

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * reduceLengthAlign * 2 | 输入数据缓冲 (fp16/bf16) | 2 | tileLength * reduceLengthAlign * 2 |
| outQueueY  | tileLength * reduceLengthAlign * 2 | 归约结果 | 2 | tileLength * reduceLengthAlign * 2 |
| tempBuffer1 | tileLength * reduceLengthAlign * 4 | float32 计算缓冲 | 1 | tileLength * reduceLengthAlign * 4 |
| tempBuffer2 | reduceLengthAlign * 4 + 32 (对齐) | 临时结果 | 1 | reduceLengthAlign * 4 + 32 |
| **总计**   | - | - | - | **tileLength * reduceLengthAlign * 2 * 4 + tileLength * reduceLengthAlign * 4 + reduceLengthAlign * 4 + 32** |

**注意**: 临时归约缓冲区虽然逻辑上只需要存储1个值，但受硬件限制需要开辟32B空间。

#### 多轴归约（如全局归约）

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * dtypeSize | 输入数据缓冲 | 1 | tileLength * dtypeSize |
| outQueueY  | 32 (对齐) | 最终归约结果 | 1 | 32 |
| tempBuffer | BLOCK_DIM * dtypeSize | block内中间结果 | 1 | BLOCK_DIM * dtypeSize |
| **总计**   | - | - | - | **tileLength * dtypeSize + BLOCK_DIM * dtypeSize + 32** |

#### 跨轴归约（非最后一维）

| Buffer 名称 | 大小（字节） | 用途 | 数量 | 总大小 |
|------------|------------|------|------|--------|
| inQueueX   | tileLength * dtypeSize | 输入数据缓冲 | 1 | tileLength * dtypeSize |
| outQueueY  | outputLength * dtypeSize | 输出数据缓冲 | 1 | outputLength * dtypeSize |
| **总计**   | - | - | - | **(tileLength + outputLength) * dtypeSize** |

### tileLength 计算

根据 UB 分配表中的总系数，计算 tileLength：

```cpp
// 单轴归约示例
int64_t tmpBuffer = isBfloat16 ? 4 : 0;
int64_t maxTileElements = (UB_SIZE_LIMIT - 32 - reduceLengthAlign * dtypeSize) / (reduceLengthAlign * (dtypeSize * 4 + tmpBuffer));
```

### UB 约束验证

| 归约类型 | 数据类型 | 约束条件 | 说明 |
|----------|----------|----------|------|
| 单轴归约 | float16 | tileLength * reduceLengthAlign * 2 * 4 + tileLength * reduceLengthAlign * 4 + reduceLengthAlign * 4 + 32 <= UB_SIZE_LIMIT | 视 reduceLengthAlign 而定 |
| 单轴归约 | bfloat16 | tileLength * reduceLengthAlign * 2 * 4 + tileLength * reduceLengthAlign * 4 + reduceLengthAlign * 4 + 32 <= UB_SIZE_LIMIT | 视 reduceLengthAlign 而定 |
| 单轴归约 | float32 | tileLength * reduceLengthAlign * dtypeSize * 4 + reduceLengthAlign * dtypeSize + 32 <= UB_SIZE_LIMIT | 视 reduceLengthAlign 而定 |

## 优化建议

### 1. 分层归约

```cpp
// 第一层：每个 tile 内归约
LocalTensor<float> reduceLocal = tempBuffer.Get<float>();
ReduceSum(reduceLocal, xLocal, reduceLength);

// 第二层：跨 tile 归约（如果需要）
// 使用 atomic add 或 sync
```

### 2. 避免跨轴访问

```cpp
// 好的做法：归约轴是最后一维
// 输入: [M, N], 归约 axis=1
// 内存访问是连续的

// 避免的做法：归约轴不是最后一维
// 输入: [M, N], 归约 axis=0
// 需要转置或 stride 访问
```

### 3. 使用硬件归约指令

```cpp
// 使用 AscendC 内置归约 API
ReduceSum(outputLocal, inputLocal, reduceLength);
ReduceMax(outputLocal, inputLocal, reduceLength);
ReduceMin(outputLocal, inputLocal, reduceLength);
```

### 4. 初始化处理

```cpp
// 第一个 tile 需要初始化输出
if (tileIdx == 0) {
    // 初始化输出 buffer
    Duplicate(outputLocal, initValue, outputLength);
}

// 后续 tile 累加
if (tileIdx > 0) {
    // 读取之前的输出并累加
    DataCopy(outputLocal, outputGlobal, outputLength);
    Add(outputLocal, outputLocal, tempBuffer, outputLength);
}
```

## 代码模板

### Host 端 Tiling 计算

```cpp
void ComputeReductionTiling(const std::vector<int64_t>& inputShape,
                            int64_t reduceAxis,
                            int64_t dtypeSize,
                            bool isBfloat16,
                            ReductionTilingData& params) {
    // 计算总长度和归约长度
    params.totalLength = 1;
    for (auto dim : (reduceAxis - 1)) {
        params.totalLength *= dim;
    }

    params.reduceLength = inputShape[reduceAxis];
    params.isReduceLastDim = (reduceAxis == static_cast<int64_t>(inputShape.size()) - 1);

    // Block级 Tiling 参数计算
    params.formerNum = totalLength % CORE_NUM;
    if (params.formerNum == 0) {
        params.formerNum = CORE_NUM;
    }
    params.tailNum = CORE_NUM - params.formerNum;
    params.formerLength = totalLength / CORE_NUM + 1;
    params.tailLength = totalLength / CORE_NUM;

    // UB级 Tiling 参数计算
    // 根据 UB 分配表确定 buffer 系数
    int64_t tmpBuffer = isBfloat16 ? 4 : 0;
    int64_t maxTileElements = (UB_SIZE_LIMIT - 32 - reduceLengthAlign * dtypeSize) / (reduceLengthAlign * (dtypeSize * 4 + tmpBuffer));  // UB_SIZE_LIMIT 实际编码时通过接口获取

    params.tileLength = maxTileElements;
    params.reduceLengthAlign = ((params.reduceLength + 512 - 1) / 512) * 512;
}
```

### Kernel 端实现（沿最后一维归约）

```cpp
template <typename T>
__aicore__ inline void ReduceSumKernel<T>::Init(GM_ADDR x, GM_ADDR y, ReductionTilingData* tiling) {
    if (AscendC::GetBlockIdx() < tiling->formerNum) {
        // 整核
        this->blockLength = tiling->formerLength;
        xGm.SetGlobalBuffer(
            (__gm__ T*)x + tiling->formerLength * AscendC::GetBlockIdx(),
            tiling->formerLength
        );
    } else {
        // 尾核
        this->blockLength = tiling->tailLength;
        int64_t tailIdx = AscendC::GetBlockIdx() - tiling->formerNum;
        int64_t offset = tiling->formerLength * tiling->formerNum + tiling->tailLength * tailIdx;
        xGm.SetGlobalBuffer((__gm__ T*)x + offset, tiling->tailLength);
    }
    yGm.SetGlobalBuffer((__gm__ T*)y, 1);  // 归约结果
    this->tileLength = tiling->tileLength;
    this->reduceLength = tiling->reduceLength;
}

template <typename T>
__aicore__ inline void ReduceSumKernel<T>::Process() {
    int64_t tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
    int64_t tailTileLength = this->blockLength - (tileNum - 1) * this->tileLength;

    for (int64_t i = 0; i < tileNum - 1; ++i) {
        CopyIn(i, this->tileLength);
        Compute(i, this->tileLength);
        CopyOut(i, this->tileLength);
    }
    // 处理尾块
    CopyIn(tileNum - 1, tailTileLength);
    Compute(tileNum - 1, tailTileLength);
    CopyOut(tileNum - 1, tailTileLength);
}
```

## 参考实现

- `csrc/ops/sum/sum.cpp` - 沿轴求和归约
- `csrc/ops/max/max.cpp` - 沿轴最大值归约
