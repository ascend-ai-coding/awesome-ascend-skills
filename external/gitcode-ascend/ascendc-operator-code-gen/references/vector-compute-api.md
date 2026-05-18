# AscendC 矢量计算 API 参考

## API 调用模式

矢量计算 API 提供三种调用方式：

### 1. 整个 Tensor 参与计算（运算符重载）
```cpp
dstLocal = src0Local + src1Local;  // Add
dstLocal = src0Local < src1Local;  // Compare
```

### 2. Tensor 前 n 个数据计算
```cpp
AscendC::Add(dstLocal, src0Local, src1Local, count);
```

### 3. Tensor 高维切分计算
```cpp
// 连续模式
AscendC::Add(dstLocal, src0Local, src1Local, mask, repeatTime, repeatParams);
// 逐bit模式
AscendC::Add(dstLocal, src0Local, src1Local, maskArray, repeatTime, repeatParams);
```

## mask 参数

控制每次迭代参与计算的元素：

| 模式 | 说明 | 取值范围 |
|------|------|----------|
| 连续模式 | 前面连续多少个元素 | 16位:[1,128], 32位:[1,64], 64位:[1,32] |
| 逐bit模式 | 按位控制 | 16位:mask[2], 32位:mask[1] |

```cpp
// 连续模式
uint64_t mask = 128;  // 处理前128个元素

// 逐bit模式
uint64_t mask[2] = {UINT64_MAX, UINT64_MAX};  // 处理全部128个元素
```

## repeatParams 参数

### BinaryRepeatParams（双源操作数）
```cpp
AscendC::BinaryRepeatParams {dstBlkStride, src0BlkStride, src1BlkStride,
                              dstRepStride, src0RepStride, src1RepStride};
```

### UnaryRepeatParams（单源操作数）
```cpp
AscendC::UnaryRepeatParams {dstBlkStride, srcBlkStride, dstRepStride, srcRepStride};
```

**常用配置**：连续数据处理
- half: `{1, 1, 1, 8, 8, 8}`
- float: `{1, 1, 8, 8}` (UnaryRepeatParams)

## 基础算术 API

### 二元运算
```cpp
// Add, Sub, Mul, Div, Max, Min
AscendC::Add(dstLocal, src0Local, src1Local, count);
AscendC::Mul(dstLocal, src0Local, src1Local, mask, repeatTime, {1, 1, 1, 8, 8, 8});

// 运算符重载
dstLocal = src0Local + src1Local;
```

**支持类型**：half, int16_t, int32_t, float

### 一元运算
```cpp
// Abs, Exp, Ln, Sqrt, Rsqrt, Reciprocal, Relu, LeakyRelu, Tanh
AscendC::Exp(dstLocal, srcLocal, count);
```

### 标量运算（优先使用）
```cpp
// Adds, Muls, Maxs, Mins — 直接对每个元素做标量操作
AscendC::Adds(dstLocal, srcLocal, scalarValue, count);
AscendC::Muls(dstLocal, srcLocal, scalarValue, count);
```

**标量优化**: 行归约后需要对每行减去/除以一个标量值时，**优先用 Adds/Muls**:

| 需求 | 推荐 | 不推荐 |
|------|------|--------|
| `x - scalar` | `Adds(dst, src, -scalar, len)` | Duplicate(tmp, scalar) + Sub |
| `x / scalar` | `Muls(dst, src, 1.0f/scalar, len)` | Duplicate(tmp, scalar) + Div |
| `x * scalar` | `Muls(dst, src, scalar, len)` | Duplicate(tmp, scalar) + Mul |

### 多行广播（BinaryRepeatParams）

对多行数据做同一个向量的运算（如每行减去 max 向量）:

```cpp
// src1RepStride=0 使 src1 在每次 repeat 时不前进，实现广播
uint64_t mask = alignedCols / (32 / sizeof(float));  // 每 repeat 处理的元素数
uint32_t repeatTime = rowCount;
AscendC::Sub(dst, src0, src1, mask, repeatTime,
             {1, 1, 1,                                    // blkStride
              alignedCols / 8, alignedCols / 8, 0});      // repStride: src1=0 广播
```

**⚠️ repeatTime 限制**: repeatTime 参数类型为 `uint8_t`，最大值 255。超过需分批处理:
```cpp
while (remaining > 0) {
    uint8_t batch = static_cast<uint8_t>(std::min(remaining, (int64_t)255));
    AscendC::Sub(dst[offset], src0[offset], src1, mask, batch, params);
    offset += batch * alignedCols;
    remaining -= batch;
}
```

## 类型转换 API (Cast)

```cpp
AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_RINT, count);
```

### RoundMode 舍入模式
| 模式 | 说明 |
|------|------|
| CAST_NONE | 不舍入（无精度损失时） |
| CAST_RINT | 四舍六入五成双 |
| CAST_FLOOR | 向负无穷舍入 |
| CAST_CEIL | 向正无穷舍入 |
| CAST_ROUND | 四舍五入 |
| CAST_TRUNC | 向零舍入 |
| CAST_ODD | 最近邻奇数舍入 |

### 常用转换组合
| 源类型 | 目的类型 | 推荐 RoundMode | 场景 |
|--------|----------|----------------|------|
| half → float | CAST_NONE | 升精度（无损） |
| bfloat16 → float | CAST_NONE | 升精度（无损） |
| float → half | CAST_ROUND | 降精度（通用） |
| float → bfloat16 | CAST_ROUND | 降精度（通用） |
| float → int32_t | CAST_RINT / CAST_ROUND | 量化 |
| int32_t → float | CAST_NONE | 反量化（无损） |
| int8_t → half | CAST_NONE | 量化输入 |
| half → int8_t | CAST_RINT | 量化输出 |

### 混合精度模式（FP16/BF16 升精度计算）

归约/归一化类算子（Softmax、LayerNorm 等）需要 FP32 中间精度保证数值稳定性:

```cpp
// Init 阶段: 额外分配 FP32 工作缓冲区
pipe.InitBuffer(calcBuf, alignedCols * sizeof(float));  // FP32 计算空间

// Compute 阶段: 逐行 Cast → FP32 计算 → Cast 回
LocalTensor<half> inLocal = inQueue.DeQue<half>();
LocalTensor<float> workLocal = calcBuf.Get<float>();
AscendC::Cast(workLocal, inLocal[rowIdx * alignedCols], RoundMode::CAST_NONE, rLength);
// ... FP32 归约/计算 ...
AscendC::Cast(outLocal[rowIdx * alignedCols], workLocal, RoundMode::CAST_ROUND, rLength);
```

### 高维切分示例
```cpp
// half -> int32_t
uint64_t mask = 64;  // 以int32_t为准
AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_CEIL, mask, 8, {1, 1, 8, 4});
```

## 归约计算 API

### Level 2 归约（单行/任意长度）

```cpp
// ReduceSum / ReduceMax / ReduceMin
// tmpBuffer 类型必须与 T 相同，不能是 uint8_t
AscendC::ReduceSum(dstLocal, srcLocal, sharedTmpBuffer, count);
AscendC::ReduceMax(dstLocal, srcLocal, sharedTmpBuffer, count);
```

- `count` = **有效元素数**（rLength），不是对齐后的长度
- `tmpBuffer` 类型必须是 `LocalTensor<T>`（与源相同类型）
- `dst` **不能**与 `tmpBuffer` 指向同一块内存

### tmpBuffer 大小计算
```cpp
int elementsPerBlock = 32 / sizeof(T);      // half:16, float:8
int elementsPerRepeat = 256 / sizeof(T);    // half:128, float:64
int firstMaxRepeat = (count + elementsPerRepeat - 1) / elementsPerRepeat;
int tmpBufferSize = ((firstMaxRepeat + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
// 分配: pipe.InitBuffer(tmpBuf, tmpBufferSize * sizeof(T));
```

### Pattern 归约（批量多行 2D）

对齐数据的批量行归约，性能更优:

```cpp
// Pattern::Reduce::AR — 归约最后一维（每行归约为一个标量）
// srcShape = {rows, alignedCols}，alignedCols 必须 32B 对齐
AscendC::ReduceMax(dstLocal, srcLocal, sharedTmpBuffer,
                   srcShape, AscendC::Pattern::Reduce::AR, srcInnerPad);
```

| 参数 | 说明 |
|------|------|
| srcShape | `{rows, alignedCols}`，alignedCols 必须 32 字节对齐 |
| Pattern::Reduce::AR | 归约最后一维（每行→标量） |
| Pattern::Reduce::RA | 归约第一维（每列→标量） |
| srcInnerPad | A2/A3 平台必须为 `true` |

**tmpBuffer 大小**: 使用 `GetReduceMaxMaxMinTmpSize` / `GetReduceSumTmpSize` 计算:
```cpp
uint32_t tmpSize = AscendC::GetReduceMaxMaxMinTmpSize<T>(srcShape);
pipe.InitBuffer(tmpBuf, tmpSize);
```

**⚠️ Pattern 归约要求 alignedCols 是 32B 对齐的**，非对齐数据使用 Level 2 逐行归约。

### WholeReduceSum / BlockReduceSum

硬件指令，性能更优但限制更多:
```cpp
AscendC::WholeReduceSum(dstLocal, srcLocal, count);
```

## 比较与选择 API

### Compare
```cpp
// 运算符重载
dstLocal = src0Local < src1Local;

// 函数调用
AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LT, count);
```

**CMPMODE**: LT(<), GT(>), GE(>=), LE(<=), EQ(==), NE(!=)

**输出**：uint8_t 类型，按 bit 位存储结果

**⚠️ 256 字节对齐约束**: Compare API 要求参与比较的数据区域是 **256 字节的整数倍**。非对齐时需要 padding:
```cpp
// 不足 256B 的部分填充 ±inf / FLT_MAX，确保 padding 区不影响结果
uint32_t alignedCount = ((count * sizeof(T) + 255) / 256) * (256 / sizeof(T));
AscendC::Duplicate(src[count], paddingValue, alignedCount - count);  // padding
AscendC::Compare(dst, src0, src1, CMPMODE::LT, alignedCount);
```

### Select
```cpp
// 模式0：两个tensor选取（selMask有位数限制）
AscendC::Select(dstLocal, maskLocal, src0Local, src1Local,
                AscendC::SELMODE::VSEL_CMPMASK_SPR, count);

// 模式1：tensor和scalar选取
AscendC::Select(dstLocal, maskLocal, src0Local, scalarValue,
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);

// 模式2：两个tensor选取（selMask连续消耗）
AscendC::Select(dstLocal, maskLocal, src0Local, src1Local,
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
```

**selMask 规则**：bit 位为1选 src0，为0选 src1

## 数据填充 API (Duplicate)

```cpp
AscendC::Duplicate(dstLocal, scalarValue, count);

// 高维切分
AscendC::Duplicate(dstLocal, scalarValue, mask, repeatTime, dstBlkStride, dstRepStride);
```

## 复合计算 API

```cpp
// FusedMulAdd: dst = src0 * src1 + src2
AscendC::FusedMulAdd(dstLocal, src0Local, src1Local, src2Local, count);

// FusedMulAddRelu: dst = Relu(src0 * src1 + src2)
AscendC::FusedMulAddRelu(dstLocal, src0Local, src1Local, src2Local, count);

// Axpy: dst = a * x + y
AscendC::Axpy(dstLocal, aLocal, xLocal, yLocal, count);
```

## 常用代码模式

### 元素级运算
```cpp
__aicore__ inline void Compute()
{
    LocalTensor<half> src0 = inQueueX.DeQue<half>();
    LocalTensor<half> src1 = inQueueY.DeQue<half>();
    LocalTensor<half> dst = outQueueZ.AllocTensor<half>();

    AscendC::Add(dst, src0, src1, tileLength);

    outQueueZ.EnQue(dst);
    inQueueX.FreeTensor(src0);
    inQueueY.FreeTensor(src1);
}
```

### 升精度计算（FP16 -> FP32）
```cpp
__aicore__ inline void Compute()
{
    LocalTensor<half> src0 = inQueueX.DeQue<half>();
    LocalTensor<half> src1 = inQueueY.DeQue<half>();
    LocalTensor<half> dst = outQueueZ.AllocTensor<half>();

    // 复用内存进行类型转换
    LocalTensor<float> src0Fp32 = src0.ReinterpretCast<float>();
    LocalTensor<float> src1Fp32 = src1.ReinterpretCast<float>();
    LocalTensor<float> dstFp32 = dst.ReinterpretCast<float>();

    AscendC::Cast(src0Fp32, src0, AscendC::RoundMode::CAST_NONE, tileLength);
    AscendC::Cast(src1Fp32, src1, AscendC::RoundMode::CAST_NONE, tileLength);
    AscendC::Add(dstFp32, src0Fp32, src1Fp32, tileLength);
    AscendC::Cast(dst, dstFp32, AscendC::RoundMode::CAST_NONE, tileLength);

    outQueueZ.EnQue(dst);
    inQueueX.FreeTensor(src0);
    inQueueY.FreeTensor(src1);
}
```

### 条件选择
```cpp
__aicore__ inline void Compute()
{
    LocalTensor<float> src0 = inQueueX.DeQue<float>();
    LocalTensor<float> src1 = inQueueY.DeQue<float>();
    LocalTensor<uint8_t> cmpResult = tmpQueue.AllocTensor<uint8_t>();
    LocalTensor<float> dst = outQueueZ.AllocTensor<float>();

    // 比较
    AscendC::Compare(cmpResult, src0, src1, AscendC::CMPMODE::LT, tileLength);
    // 选择
    AscendC::Select(dst, cmpResult, src0, src1,
                    AscendC::SELMODE::VSEL_CMPMASK_SPR, tileLength);

    outQueueZ.EnQue(dst);
    inQueueX.FreeTensor(src0);
    inQueueY.FreeTensor(src1);
    tmpQueue.FreeTensor(cmpResult);
}
```

## 通用约束

- **地址对齐**：LocalTensor 起始地址需 32 字节对齐
- **数据类型一致**：源操作数和目的操作数类型需一致（Cast 除外）
- **TPosition**：支持 VECIN/VECCALC/VECOUT
- **repeatTime ≤ 255**: 使用高维切分模式时，repeatTime 为 `uint8_t`，传入 >255 会静默截断为 0 导致错误结果。需在 host 端限制或 kernel 端分批
- **dst ≠ tmpBuffer**: ReduceMax/ReduceSum 的 dst 不能与 tmpBuffer 是同一块内存
- **禁止 std:: 数学函数**: Kernel 中禁止 `std::min/max/abs/sqrt/exp` 等，使用 AscendC 向量 API 或三元运算符替代
