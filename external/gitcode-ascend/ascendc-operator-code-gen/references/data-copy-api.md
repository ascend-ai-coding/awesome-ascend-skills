# AscendC 数据搬运 API 参考

## ⚠️ 生产规则

**GM ↔ UB 搬运必须使用 DataCopyPad**，不要使用 DataCopy。

| API | 适用场景 | 生产代码 |
|-----|---------|---------|
| **DataCopyPad** | GM ↔ UB（所有情况） | ✅ 必须使用 |
| DataCopy | UB ↔ UB 内部拷贝 | ✅ 允许 |
| DataCopy GM↔UB | 仅当 count*sizeof(T) 严格 32B 对齐 | ⚠️ 仅调试/原型 |
| GlobalTensor::SetValue/GetValue | 逐元素 GM 访问 | ❌ 禁止（极低效） |

## DataCopyPad（推荐）

### GM → UB

```cpp
AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
AscendC::DataCopyPadExtParams<T> padParams{isPad, leftPad, rightPad, padValue};
AscendC::DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
```

### UB → GM

```cpp
AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
AscendC::DataCopyPad(dstGlobal, srcLocal, copyParams);
```

### DataCopyExtParams 参数

| 参数 | 含义 | 单位 | 范围 |
|------|------|------|------|
| blockCount | 数据块个数（通常=行数） | - | [1, 4095] |
| blockLen | 每块长度 | **字节** | - |
| srcStride | 源相邻块间隔 | **GM=字节, UB=32B块** | - |
| dstStride | 目的相邻块间隔 | **GM=字节, UB=32B块** | - |

**⚠️ Stride 单位不同**: GM 侧用字节，UB 侧用 32B DataBlock。这是最常见的搬运 bug 来源。

### DataCopyPadExtParams 参数（仅 GM→UB）

| 参数 | 含义 |
|------|------|
| isPad | 是否填充自定义值 |
| leftPadding | 左侧填充元素个数（≤32字节） |
| rightPadding | 右侧填充元素个数（≤32字节） |
| padValue | 填充值 |

### rLength vs rLengthAlign 用法

| 参数 | 使用 rLength（有效长度） | 使用 rLengthAlign（对齐长度） |
|------|------------------------|----------------------------|
| blockLen (CopyIn) | `rLength * sizeof(T)` | - |
| blockLen (CopyOut) | `rLength * sizeof(T)` | - |
| srcStride (CopyIn, GM侧) | - | `rLengthAlign * sizeof(T)` |
| dstStride (CopyIn, UB侧) | - | `rLengthAlign * sizeof(T) / 32` |
| srcStride (CopyOut, UB侧) | - | `(rLengthAlign - rLength) * sizeof(T) / 32` |
| dstStride (CopyOut, GM侧) | - | `rLengthAlign * sizeof(T)` |
| 计算 API count | `rLength` | - |
| UB 行偏移 | - | `rowIdx * rLengthAlign` |
| InitBuffer 大小 | - | `rLengthAlign * sizeof(T)` |

**关键**: CopyOut 的 `srcStride` 是块间**间隔**（padding 部分），不是完整行长度。

### CopyIn/CopyOut 一致性

CopyIn 用 DataCopyPad 时，CopyOut 也必须用 DataCopyPad。混用会导致行错位。

## 基础数据搬运 (DataCopy)

仅用于 UB ↔ UB 内部拷贝。

```cpp
// UB -> UB
AscendC::DataCopy(dstLocal, srcLocal, count);
```

**参数**：
- `count`: 元素个数，`count * sizeof(T)` 需32字节对齐

### 非连续搬运 (DataCopyParams)

```cpp
AscendC::DataCopyParams params;
params.blockCount = 1;   // 连续数据块个数 [1, 4095]
params.blockLen = 8;     // 每块长度，单位DataBlock(32B) [1, 65535]
params.srcGap = 0;       // 源相邻块间隔，单位DataBlock(32B)
params.dstGap = 0;       // 目的相邻块间隔，单位DataBlock(32B)

AscendC::DataCopy(dstLocal, srcGlobal, params);
```

**示意图**：
```
blockCount=2, blockLen=8, srcGap=0, dstGap=1
源: [====8块====][====8块====]
目的: [====8块====][gap][====8块====]
```

## 切片数据搬运 (SliceInfo)

```cpp
AscendC::SliceInfo srcSliceInfo[] = {{16, 70, 7, 3, 87}, {0, 2, 1, 1, 3}};
AscendC::SliceInfo dstSliceInfo[] = {{0, 47, 0, 3, 48}, {0, 1, 0, 1, 2}};
uint32_t dimValue = 2;

AscendC::DataCopy(dstLocal, srcGlobal, dstSliceInfo, srcSliceInfo, dimValue);
```

**SliceInfo 结构**：
| 参数 | 含义 |
|------|------|
| startIndex | 切片起始位置 |
| endIndex | 切片终止位置 |
| stride | 相邻切片间隔（元素个数） |
| burstLen | 每片数据长度，单位DataBlock(32B)，dimValue>1时必须为1 |
| shapeValue | 当前维度原始长度 |

## 非对齐搬运 (DataCopyPad)

```cpp
// GM -> UB，支持非32字节对齐
AscendC::DataCopyExtParams copyParams{1, 20 * sizeof(half), 0, 0, 0};
AscendC::DataCopyPadExtParams<half> padParams{true, 0, 2, 0};  // isPad, leftPad, rightPad, padValue
AscendC::DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);

// UB -> GM
AscendC::DataCopyPad(dstGlobal, srcLocal, copyParams);
```

**DataCopyExtParams**：
| 参数 | 含义 | 单位 |
|------|------|------|
| blockCount | 连续数据块个数 | - |
| blockLen | 每块长度 | **字节** |
| srcStride | 源相邻块间隔 | GM:字节, UB:DataBlock |
| dstStride | 目的相邻块间隔 | GM:字节, UB:DataBlock |

**DataCopyPadExtParams**：
| 参数 | 含义 |
|------|------|
| isPad | 是否填充自定义值 |
| leftPadding | 左侧填充元素个数（≤32字节） |
| rightPadding | 右侧填充元素个数（≤32字节） |
| padValue | 填充值 |

## UB内部拷贝 (Copy)

```cpp
// VECIN/VECCALC/VECOUT 之间的拷贝
AscendC::Copy(dstLocal, srcLocal, mask, repeatTime, {dstStride, srcStride, dstRepStride, srcRepStride});
```

**CopyRepeatParams**：
| 参数 | 含义 |
|------|------|
| dstStride/srcStride | 同一迭代内DataBlock步长 |
| dstRepeatSize/srcRepeatSize | 相邻迭代间步长 |

```cpp
// 示例：连续拷贝512个int16_t
uint64_t mask = 128;
AscendC::Copy(dstLocal, srcLocal, mask, 4, {1, 1, 8, 8});
```

## 增强数据搬运 (DataCopyEnhancedParams)

```cpp
AscendC::DataCopyParams intriParams;
AscendC::DataCopyEnhancedParams enhancedParams;
enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;  // 或 BLOCK_MODE_VECTOR
AscendC::DataCopy(dstLocal, srcLocal, intriParams, enhancedParams);
```

**blockMode 模式**：
| 模式 | 传输单位 | 适用通路 |
|------|----------|----------|
| BLOCK_MODE_MATRIX | 16×16 cube | CO1 -> CO2 |
| BLOCK_MODE_VECTOR | 1×16 cube | CO1 -> CO2 |
| BLOCK_MODE_NORMAL | 32B | 通用通路 |

**量化模式 (deqScale)**：
| 模式 | 说明 |
|------|------|
| DEQ | int32 -> half，使用deqValue |
| DEQ8 | int32 -> int8/uint8 |
| DEQ16 | int32 -> half/int16 |
| VDEQ/VDEQ8/VDEQ16 | 使用deqTensorAddr参数向量 |

## 数据通路速查

| 通路 | 源 | 目的 | 说明 |
|------|----|----|------|
| GM -> UB | GlobalTensor | LocalTensor(VECIN) | CopyIn阶段 |
| UB -> GM | LocalTensor(VECOUT) | GlobalTensor | CopyOut阶段 |
| UB -> UB | LocalTensor | LocalTensor | Compute阶段 |
| UB -> L1 | LocalTensor | LocalTensor(A1/B1/TSCM) | 大数据缓存 |
| CO1 -> CO2 | LocalTensor(CO1) | LocalTensor(CO2) | 矩阵计算结果 |

## 地址对齐要求

| 位置 | 对齐要求 |
|------|----------|
| UB (VECIN/VECOUT) | 32字节 |
| L1 Buffer | 32字节 |
| GM | 按数据类型大小对齐 |
| C2 | 64字节 |
| C2PIPE2GM | 128字节 |

## 常用代码模式

### CopyIn（多行批量搬入）

```cpp
__aicore__ inline void CopyIn(uint32_t startRow, uint32_t rows) {
    LocalTensor<half> srcLocal = inQueue.AllocTensor<half>();
    // blockCount=行数, blockLen=每行有效字节, srcStride=GM行间距(字节), dstStride=UB行间距(32B块)
    AscendC::DataCopyExtParams copyParams{
        static_cast<uint16_t>(rows),                          // blockCount
        static_cast<uint32_t>(cols * sizeof(half)),           // blockLen (有效数据)
        static_cast<uint32_t>(totalCols * sizeof(half)),      // srcStride (GM, 字节)
        static_cast<uint16_t>(alignedCols * sizeof(half) / 32) // dstStride (UB, 32B块)
    };
    AscendC::DataCopyPadExtParams<half> padParams{true, 0,
        static_cast<uint8_t>(alignedCols - cols), 0};
    AscendC::DataCopyPad(srcLocal, srcGlobal[startRow * totalCols], copyParams, padParams);
    inQueue.EnQue(srcLocal);
}
```

### CopyOut（多行批量搬出）

```cpp
__aicore__ inline void CopyOut(uint32_t startRow, uint32_t rows) {
    LocalTensor<half> dstLocal = outQueue.DeQue<half>();
    AscendC::DataCopyExtParams copyParams{
        static_cast<uint16_t>(rows),
        static_cast<uint32_t>(cols * sizeof(half)),            // 只搬出有效数据
        static_cast<uint16_t>((alignedCols - cols) * sizeof(half) / 32), // srcStride: padding 间隔
        static_cast<uint32_t>(totalCols * sizeof(half))        // dstStride (GM, 字节)
    };
    AscendC::DataCopyPad(dstGlobal[startRow * totalCols], dstLocal, copyParams);
    outQueue.FreeTensor(dstLocal);
}
```

### Elementwise 连续搬运

```cpp
__aicore__ inline void CopyIn() {
    LocalTensor<half> srcLocal = inQueue.AllocTensor<half>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tileLength * sizeof(half)), 0, 0, 0};
    AscendC::DataCopyPad(srcLocal, srcGlobal[offset], copyParams);
    inQueue.EnQue(srcLocal);
}
```

## 地址对齐要求

| 位置 | 对齐要求 |
|------|----------|
| UB (VECIN/VECOUT) | 32字节 |
| L1 Buffer | 32字节 |
| GM | 按数据类型大小对齐 |

## 32 字节对齐计算

```cpp
// 按 32B 对齐的元素数
uint32_t alignedCols = ((cols * sizeof(T) + 31) / 32) * (32 / sizeof(T));

// 等价公式
uint32_t elemsPerBlock = 32 / sizeof(T);  // half:16, float:8
uint32_t alignedCols = ((cols + elemsPerBlock - 1) / elemsPerBlock) * elemsPerBlock;
```
