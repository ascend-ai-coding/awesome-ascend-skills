# L5：Register-based 重写（MicroAPI）详细指南

## 触发条件（满足任一）

1. **性能关键路径**：RMSNorm/RoPE 等高频算子，需要极致性能
2. **量化 Cast 链路复杂**：FP32→FP8/HiFloat8/INT8 多步量化
3. **需要溢出模式控制**：RMSNorm/Softmax 等数值范围可控的算子
4. **950 新增数据类型**：FP8_E4M3FN/FP8_E5M2/HiFloat8 的 Cast 需 MicroAPI

## L5 的本质

950 的 Vector Core 从 MemBase（220x）变为 RegBase（351x）。数据不再必须在 UB 内存中流转，可以直接在寄存器中操作。L5 是对核心计算路径的**全部重写**，从 Memory-based 编程模式切换到 Register-based MicroAPI 编程模式。

## 改动 1：MicroAPI 命名空间引入

```cpp
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::UpdateMask;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::CastTrait;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskMergeMode;
```

## 改动 2：寄存器向量长度

```cpp
// 由平台决定，不再硬编码
constexpr static uint32_t VL_FP32 =
    static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);
```

## 改动 3：CastTrait 配置（替代传统 RoundMode）

950 中类型转换不再使用 `RoundMode` 枚举，改用 `CastTrait` 结构体：

```cpp
// half/bf16 → fp32
constexpr static AscendC::MicroAPI::CastTrait CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN
};

// fp32 → fp16
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};

// fp32 → fp8（950 新增量化类型，饱和模式）
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FLOAT8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};

// fp32 → hifloat8（非饱和模式）
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_HIFLOAT8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND
};

// fp32 → int16
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_INT16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};

// int16 → fp16
constexpr AscendC::MicroAPI::CastTrait CAST_INT16_TO_FP16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};

// fp16 → int8
constexpr AscendC::MicroAPI::CastTrait CAST_FP16_TO_INT8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};
```

### CastTrait 四要素

| 字段 | 含义 | 常用值 |
|------|------|--------|
| RegLayout | 寄存器布局 | `ZERO`（默认） |
| SatMode | 饱和模式 | `SAT`（量化时用）、`NO_SAT`（反量化时用）、`UNKNOWN` |
| MaskMergeMode | 掩码合并模式 | `ZEROING`（默认） |
| RoundMode | 舍入模式 | `CAST_RINT`、`CAST_ROUND`、`UNKNOWN` |

### 量化 vs 反量化的 SatMode 规则

- **量化（高精度→低精度）**：用 `SatMode::SAT`，防止溢出
- **反量化（低精度→高精度）**：用 `SatMode::NO_SAT`，无溢出风险

## 改动 4：Register-based 核心计算

### 完整示例：RMSNorm + Mul

```cpp
__aicore__ inline void RmsNormPostWithMul(...) {
    __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
    __local_mem__ T_KV* x = (__local_mem__ T_KV*)xTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
        MaskReg pMask = UpdateMask<float>(count);
        MaskReg pFull = CreateMask<float, MaskPattern::ALL>();

        for (uint16_t i = 0; i < loopTimes; ++i) {
            LoadTensorForDtypeT<T_KV>(x, reg0, pMask, i * stride);
            LoadTensorForDtypeT<T_KV>(gamma, reg1, pMask, 0);
            MicroAPI::Mul(reg2, reg0, reg0, pMask);       // x * x
            ReduceSum(reg2, reg2, pMask);                   // ReduceSum(x*x)
            MicroAPI::Muls(reg3, reg2, reciprocal, pFull);  // * 1/N
            MicroAPI::Adds(reg4, reg3, epsilon, pFull);     // + eps
            MicroAPI::Sqrt(reg5, reg4, pFull);              // sqrt(...)
            MicroAPI::Div(reg6, reg0, reg5, pMask);         // x / sqrt(...)
            MicroAPI::Mul(reg7, reg1, reg6, pMask);         // gamma * norm
            StoreTensorForDtypeTOut<float>(dst, reg7, pMask, i * stride);
        }
    }
}
```

### 关键要素

1. **`__VEC_SCOPE__`**：必须包裹所有 MicroAPI 计算代码
2. **`RegTensor<T>`**：寄存器张量声明，替代 `LocalTensor<T>`
3. **`MaskReg`**：显式掩码，替代隐式 count 参数
   - `UpdateMask<float>(count)`：按实际元素数创建掩码
   - `CreateMask<float, MaskPattern::ALL>()`：全 1 掩码
4. **`LoadTensorForDtypeT<T_KV>`**：带自动类型转换的加载（half/bf16→fp32）
5. **`StoreTensorForDtypeTOut<T>`**：带自动类型转换的存储（fp32→half/bf16/fp8）
6. **`MicroAPI::` 前缀**：所有向量计算 API 都在 MicroAPI 命名空间下

## 改动 5：溢出模式控制

950 可通过 SPR 寄存器控制溢出检测，对 RMSNorm/Softmax 等数值范围可控的算子，关闭溢出检测可提升性能：

```cpp
#define GLOBAL_OVERFLOW_MODE_CTRL 60

// 保存原始溢出模式
#if (__NPU_ARCH__ == 3510)
    int64_t globalOriOverflowMode = AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
#endif

// 设置为 0（关闭溢出检测，提升性能）
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
#endif

// ... 核心计算 ...

// 恢复原始溢出模式
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(globalOriOverflowMode);
#endif
```

### 使用原则

- **必须保存/恢复**：溢出模式是全局状态，修改后必须恢复
- **仅限数值范围可控的算子**：RMSNorm（输出始终有界）、Softmax（输出 [0,1]）
- **必须用 `__NPU_ARCH__ == 3510` 保护**：220x 平台不支持此 SPR

## 改动 6：量化 Pack + 非对齐 Store（仅 INT8 量化算子需要）

> **注意**：本节内容仅适用于输出包含 INT8 类型的量化算子。如果算子只涉及 FP16/BF16/FP8/HiFloat8，Cast 都是一步完成，无需三步转换。

### FP32 → INT8 三步量化

**910b 和 950 都不支持 FP32 直接转 INT8**，硬件没有这个 Cast 指令，必须经过中间类型。但两个平台的中间类型和同步方式不同：

**950 Register-based 路径**（FP32 → INT16 → FP16 → INT8）：

```cpp
MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmpInt16, src, preg);
MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmpHalf, tmpInt16, preg);
MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantInt8, tmpHalf, preg);
```

**910b Memory-based 路径**（FP32 → INT32 → FP16 → INT8）：

```cpp
Cast(src.ReinterpretCast<int32_t>(), src, RoundMode::CAST_RINT, size);     // FP32→INT32
PipeBarrier<PIPE_V>();
SetDeqScale((half)1.0f);                                                  // 设置反量化缩放因子
PipeBarrier<PIPE_V>();
Cast(src.ReinterpretCast<half>(), src.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);  // INT32→FP16
PipeBarrier<PIPE_V>();
Cast(dst, src.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);      // FP16→INT8
```

**关键差异**：
- 950 用 INT16 作中间类型（更省寄存器），910b 用 INT32（4字节对齐）
- 950 不需要 `SetDeqScale` 和 `PipeBarrier`，910b 每步之间需要同步
- 950 量化后需要 Pack 压缩才能 Store，910b 不需要

### 两级 Pack 压缩

量化后需要 Pack 将数据压缩到更窄的位宽：

```cpp
Pack((RegTensor<uint16_t>&)tmpInt16, (RegTensor<uint32_t>&)quantInt8);
Pack((RegTensor<uint8_t>&)quantInt8, (RegTensor<uint16_t>&)tmpInt16);
```

### 非对齐 Store

当输出数据长度不是对齐倍数时，使用 `DataCopyUnAlign`：

```cpp
DataCopyUnAlign(output, quantInt8, uValue, postUpdateStride);
```

## Register-based vs Memory-based 完整对照

| 操作 | Memory-based（220x） | Register-based（351x） |
|------|---------------------|----------------------|
| 数据声明 | `LocalTensor<float> xLocal` | `RegTensor<float> reg0` |
| GM→UB 加载 | `DataCopy(xLocal, xGm, count)` | `DataCopy<LoadDist::DIST_UNPACK_B16>(reg0, addr)` |
| 类型转换 | `Cast<float, half>(dst, src, RoundMode::CAST_RINT, count)` | `Cast<float, half, CAST_B16_TO_B32>(dst, src, maskReg)` |
| 向量计算 | `Mul(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Mul(regDst, regSrc1, regSrc2, maskReg)` |
| 标量计算 | `Muls(dstLocal, srcLocal, scalar, count)` | `MicroAPI::Muls(regDst, regReg, scalar, maskFull)` |
| 归约 | `ReduceSumCustom(dst, src, work, count)` | `ReduceSum(regDst, regSrc, maskReg)` |
| UB→GM 存储 | `DataCopy(yGm, yLocal, count)` | `DataCopy<StoreDist::DIST_NORM>(addr, reg, maskReg)` |
| 非对齐存储 | `DataCopyPad(yGm, yLocal, padParams)` | `DataCopyUnAlign(addr, reg, unalignReg, stride)` |
| 掩码 | 隐式（通过 count 参数） | 显式 `MaskReg` + `CreateMask/UpdateMask` |
| 内存屏障 | `SetFlag<HardEvent::MTE2_V>` / `WaitFlag` | `LocalMemBar<MemType::UB>` |
| 计算域 | 默认 | `__VEC_SCOPE__ { ... }` |
| 量化 Pack | 不需要 | `Pack((RegTensor<uint16_t>&)dst, (RegTensor<uint32_t>&)src)` |

## L5 迁移检查清单

- [ ] `RegTensor` / `MaskReg` / `CastTrait` 是否正确使用
- [ ] `LoadDist` / `StoreDist` 是否正确配置
- [ ] `LocalMemBar<MemType::UB>` 内存屏障是否正确使用
- [ ] `__VEC_SCOPE__` 计算域是否包裹核心计算
- [ ] 溢出模式控制（GetCtrlSpr/SetCtrlSpr）是否需要，是否正确保存/恢复
- [ ] 量化 Pack（两级 Pack 压缩）是否正确实现
- [ ] `DataCopyUnAlign` 非对齐存储是否正确使用
- [ ] FP8/HiFloat8 量化 CastTrait 的 SatMode 是否正确（量化用 SAT，反量化用 NO_SAT）
- [ ] FP32→INT8 是否经过三步量化（FP32→INT16→FP16→INT8）
- [ ] `__NPU_ARCH__ == 3510` 条件编译保护是否正确

## 架构差异全景（L5 迁移决策依据）

| 维度 | 220x（MemBase） | 351x（RegBase） | 影响 |
|------|----------------|----------------|------|
| Vector Core 架构 | MemBase（数据在 UB 内存中流转） | RegBase（数据可直接在寄存器中操作） | L5 |
| 编程模式 | 仅 SIMD | SIMD + SIMT + SIMD/SIMT 混合 | L4 |
| UB 容量 | 128KB（910b）/ 192KB（910_93） | 256KB | L3 |
| SIMT DCache | 无 | 复用 UB 空间，需预留 40KB | L3 |
| L1→GM 通路 | ✓ 支持 | ✗ 删除 | L3+ |
| GM→L0A/L0B 通路 | ✓ 支持 | ✗ 删除 | L3+ |
| UB→L1 通路 | ✗ 不支持 | ✓ 新增 | L3+ |
| L0C→UB 通路 | ✗ 不支持 | ✓ 新增 | L3+ |
| Subnormal 支持 | ✓ 硬件支持 | ✗ 默认不支持（需软仿） | L3 |
| 新数据类型 | - | fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t | L3/L5 |
| 溢出模式控制 | - | GetCtrlSpr/SetCtrlSpr（SPR 寄存器） | L5 |

## 代表算子

**KvRmsnormRopeCache**：唯一真正使用 Register-based 模式的算子，包含：
- MicroAPI 重写核心计算（RMSNorm + RoPE）
- 溢出模式控制（关闭溢出检测提升性能）
- FP8/HiFloat8 量化 CastTrait
- 量化 Pack + DataCopyUnAlign
- 26 种数据类型组合（910b 仅 8 种）

---

## 补充 1：通用 Load/Store 函数模板

### LoadTensorForDtypeT\<T\>：带自动类型转换的加载

支持 `half`/`bfloat16_t`/`float` 三种输入类型自动推导，half/bf16→fp32 自动转换：

```cpp
template <typename T>
__aicore__ inline void LoadTensorForDtypeT(
    __local_mem__ T* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + offset));
        Cast<float, half, CAST_B16_TO_B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + offset));
        Cast<float, bfloat16_t, CAST_B16_TO_B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float*)(input) + offset));
    }
}
```

**设计要点**：
- 使用 `if constexpr` 编译期分支，零运行时开销
- half/bf16 加载后立即 Cast 为 fp32，后续计算全部在 fp32 寄存器中进行
- float 类型直接 `DataCopy`，无需 Cast

### StoreTensorForDtypeTOut\<T\>：带自动类型转换的存储

支持 `float`/`half`/`bfloat16_t` 输出，fp32→half/bf16 自动转换：

```cpp
template <typename T>
__aicore__ inline void StoreTensorForDtypeTOut(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        RegTensor<T> xFp16;
        Cast<T, float, CAST_FP32_TO_FP16>(xFp16, src, preg);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}
```

**设计要点**：
- float 输出直接 `DataCopy`，使用 `StoreDist::DIST_NORM`
- half/bf16 输出先 Cast 再 `DataCopy`，使用 `StoreDist::DIST_PACK_B32`
- 注意 Load 用 `DIST_UNPACK_B16`，Store 用 `DIST_PACK_B32`，方向相反

---

## 补充 2：非对齐 Store 模板

### StoreUnAlignOneTensor\<T\>：支持量化类型的非对齐存储

支持 `int8_t`/`hifloat8_t`/`fp8_e5m2_t`/`fp8_e4m3fn_t` 等量化类型，内部自动选择量化路径：

```cpp
template <typename T>
__aicore__ inline void StoreUnAlignOneTensor(
    __local_mem__ T*& output, MicroAPI::RegTensor<float>& src,
    MicroAPI::UnalignReg& uValue, MicroAPI::MaskReg& preg,
    uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, int8_t>::value) {
        // FP32 → INT16 → FP16 → INT8 三步量化
        AscendC::MicroAPI::RegTensor<half> tmpHalf;
        AscendC::MicroAPI::RegTensor<int16_t> tmpInt16;
        AscendC::MicroAPI::RegTensor<int8_t> quantInt8;
        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmpInt16, src, preg);
        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmpHalf, tmpInt16, preg);
        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantInt8, tmpHalf, preg);
        Pack((MicroAPI::RegTensor<uint16_t>&)tmpInt16, (MicroAPI::RegTensor<uint32_t>&)quantInt8);
        Pack((MicroAPI::RegTensor<uint8_t>&)quantInt8, (MicroAPI::RegTensor<uint16_t>&)tmpInt16);
        DataCopyUnAlign(output, quantInt8, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, hifloat8_t>::value) {
        // FP32 → HiFloat8 直接量化 + Pack + DataCopyUnAlign
        AscendC::MicroAPI::RegTensor<hifloat8_t> quantHifloat8;
        AscendC::MicroAPI::Cast<hifloat8_t, float, CAST_FP32_TO_HIFLOAT8>(quantHifloat8, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t>&)quantHifloat8, (MicroAPI::RegTensor<uint32_t>&)quantHifloat8);
        Pack((MicroAPI::RegTensor<uint8_t>&)quantHifloat8, (MicroAPI::RegTensor<uint16_t>&)quantHifloat8);
        DataCopyUnAlign(output, quantHifloat8, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, fp8_e5m2_t>::value) {
        // FP32 → FP8_E5M2 直接量化 + Pack + DataCopyUnAlign
        AscendC::MicroAPI::RegTensor<fp8_e5m2_t> quantFloat8E5m2;
        AscendC::MicroAPI::Cast<fp8_e5m2_t, float, CAST_FP32_TO_FLOAT8>(quantFloat8E5m2, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t>&)quantFloat8E5m2, (MicroAPI::RegTensor<uint32_t>&)quantFloat8E5m2);
        Pack((MicroAPI::RegTensor<uint8_t>&)quantFloat8E5m2, (MicroAPI::RegTensor<uint16_t>&)quantFloat8E5m2);
        DataCopyUnAlign(output, quantFloat8E5m2, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
        // FP32 → FP8_E4M3FN 直接量化 + Pack + DataCopyUnAlign
        AscendC::MicroAPI::RegTensor<fp8_e4m3fn_t> quantFloat8E4m3;
        AscendC::MicroAPI::Cast<fp8_e4m3fn_t, float, CAST_FP32_TO_FLOAT8>(quantFloat8E4m3, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t>&)quantFloat8E4m3, (MicroAPI::RegTensor<uint32_t>&)quantFloat8E4m3);
        Pack((MicroAPI::RegTensor<uint8_t>&)quantFloat8E4m3, (MicroAPI::RegTensor<uint16_t>&)quantFloat8E4m3);
        DataCopyUnAlign(output, quantFloat8E4m3, uValue, postUpdateStride);
    }
}
```

### 量化路径对比

| 输出类型 | 量化路径 | Cast 步数 | 需要 Pack | SatMode |
|---------|---------|----------|----------|---------|
| int8_t | FP32→INT16→FP16→INT8 | 3 步 | ✓ 两级 Pack | SAT |
| hifloat8_t | FP32→HiFloat8 | 1 步 | ✓ 两级 Pack | NO_SAT |
| fp8_e5m2_t | FP32→FP8_E5M2 | 1 步 | ✓ 两级 Pack | SAT |
| fp8_e4m3fn_t | FP32→FP8_E4M3FN | 1 步 | ✓ 两级 Pack | SAT |

**关键差异**：
- INT8 是唯一需要三步量化的类型（硬件不支持 FP32→INT8 直接转换）
- FP8/HiFloat8 可一步量化，但仍需 Pack 压缩后才能 `DataCopyUnAlign`
- 所有量化类型输出均使用 `DataCopyUnAlign`（非对齐存储），因为量化后数据长度通常不是对齐倍数

---

## 补充 3：Subnormal 处理

### 351x 默认不支持 Subnormal

SubNormal 浮点数指的是指数位全为 0、尾数不为 0 的浮点数，用于表示比最小正常数更小的值，避免"下溢为 0"。**351x 版本默认不支持 Subnormal，Subnormal 浮点数在计算中被视为 0**。

### 涉及 API

以下基础 API 受 Subnormal 影响：

| API | Config 结构体 | algo 参数说明 |
|-----|-------------|-------------|
| Exp | ExpConfig | ExpAlgo::INTRINSIC / PRECISION_1ULP_FTZ_TRUE / PRECISION_1ULP_FTZ_FALSE |
| Ln | LnConfig | LnAlgo::INTRINSIC / PRECISION_1ULP_FTZ_TRUE / PRECISION_1ULP_FTZ_FALSE |
| Reciprocal | ReciprocalConfig | ReciprocalAlgo::INTRINSIC / PRECISION_1ULP_FTZ_TRUE / PRECISION_1ULP_FTZ_FALSE |
| Sqrt | SqrtConfig | SqrtAlgo::INTRINSIC / FAST_INVERSE / PRECISION_1ULP_FTZ_TRUE / PRECISION_0ULP_FTZ_FALSE / PRECISION_1ULP_FTZ_FALSE |
| Rsqrt | RsqrtConfig | RsqrtAlgo::INTRINSIC / FAST_INVERSE / PRECISION_1ULP_FTZ_TRUE / PRECISION_0ULP_FTZ_FALSE / PRECISION_1ULP_FTZ_FALSE |
| Div | DivConfig | DivAlgo::INTRINSIC / DIFF_COMPENSATION / PRECISION_1ULP_FTZ_TRUE / PRECISION_0ULP_FTZ_TRUE / PRECISION_0ULP_FTZ_FALSE / PRECISION_1ULP_FTZ_FALSE |

### algo 参数含义

| algo 值 | 含义 | Subnormal 处理 |
|---------|------|---------------|
| `INTRINSIC` | 使用单指令计算（**默认值**） | Subnormal 被近似为 0 |
| `PRECISION_1ULP_FTZ_TRUE` | 使用单指令计算 | Subnormal 被近似为 0 |
| `PRECISION_1ULP_FTZ_FALSE` | 支持Subnormal数据计算 | 软件模拟，精度扩展，避免下溢为 0 |

### 220x vs 351x 代码对比

**220x 版本**（默认支持 Subnormal，无需 Config）：

```cpp
AscendC::Ln(dstLocal, srcLocal, count);
AscendC::Exp(dstLocal, srcLocal, count);
AscendC::Sqrt(dstLocal, srcLocal, count);
AscendC::Div(dstLocal, src1Local, src2Local, count);
```

**351x 版本 — 支持 Subnormal**（需 Config 配置）：

```cpp
constexpr AscendC::LnConfig LN_CONFIG = {
    AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE
};
AscendC::Ln<T, LN_CONFIG>(dstLocal, srcLocal, count);

constexpr AscendC::ExpConfig EXP_CONFIG = {
    AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE
};
AscendC::Exp<T, EXP_CONFIG>(dstLocal, srcLocal, count);

constexpr AscendC::SqrtConfig SQRT_CONFIG = {
    AscendC::SqrtAlgo::PRECISION_1ULP_FTZ_FALSE
};
AscendC::Sqrt<T, SQRT_CONFIG>(dstLocal, srcLocal, count);

constexpr AscendC::DivConfig DIV_CONFIG = {
    AscendC::DivAlgo::PRECISION_1ULP_FTZ_FALSE
};
AscendC::Div<T, DIV_CONFIG>(dstLocal, src1Local, src2Local, count);
```

**351x 版本 — 高性能模式**（Subnormal 视为 0，无需额外配置）：

```cpp
// 默认值即为 INTRINSIC，Subnormal → 0
AscendC::Ln(dstLocal, srcLocal, count);
AscendC::Exp(dstLocal, srcLocal, count);
AscendC::Sqrt(dstLocal, srcLocal, count);
AscendC::Div(dstLocal, src1Local, src2Local, count);
```

### Register-based 模式下的 Subnormal 处理

在 MicroAPI（Register-based）模式下，Exp/Ln/Sqrt/Div 的模板参数 `mode` 支持 `MaskMergeMode` 或对应的 `XxxSpecificMode*` 指针。`XxxSpecificMode` 结构体包含 `mrgMode` 和 `algo` 两个字段：

```cpp
// 351x 架构下的 SpecificMode 定义（来自 kernel_reg_compute_utils.h）
struct ExpSpecificMode {
    MaskMergeMode mrgMode = MaskMergeMode::ZEROING;
    ExpAlgo algo = ExpAlgo::INTRINSIC;
};

struct LnSpecificMode {
    MaskMergeMode mrgMode = MaskMergeMode::ZEROING;
    LnAlgo algo = LnAlgo::INTRINSIC;
};

struct SqrtSpecificMode {
    MaskMergeMode mrgMode = MaskMergeMode::ZEROING;
    bool precisionMode = false;
    SqrtAlgo algo = SqrtAlgo::INTRINSIC;
};

struct DivSpecificMode {
    MaskMergeMode mrgMode = MaskMergeMode::ZEROING;
    bool precisionMode = false;
    DivAlgo algo = DivAlgo::INTRINSIC;
};
```

**Register-based 模式下支持 Subnormal 的调用方式**：

```cpp
__VEC_SCOPE__
{
    // 定义支持 Subnormal 的 SpecificMode
    constexpr MicroAPI::LnSpecificMode LN_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    constexpr MicroAPI::ExpSpecificMode EXP_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    constexpr MicroAPI::SqrtSpecificMode SQRT_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        false,
        AscendC::SqrtAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    constexpr MicroAPI::DivSpecificMode DIV_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        false,
        AscendC::DivAlgo::PRECISION_1ULP_FTZ_FALSE
    };

    RegTensor<float> regDst, regSrc;
    MaskReg pMask = CreateMask<float, MaskPattern::ALL>();

    // 使用 SpecificMode 指针作为 mode 参数
    MicroAPI::Ln<float, &LN_SUBNORMAL_MODE>(regDst, regSrc, pMask);
    MicroAPI::Exp<float, &EXP_SUBNORMAL_MODE>(regDst, regSrc, pMask);
    MicroAPI::Sqrt<float, &SQRT_SUBNORMAL_MODE>(regDst, regSrc, pMask);
    MicroAPI::Div<float, &DIV_SUBNORMAL_MODE>(regDst, regSrc, pMask);
}
```

**注意**：`MicroAPI` 命名空间实际上是 `AscendC::Reg` 的别名（`namespace MicroAPI = Reg;`），因此 `MicroAPI::LnSpecificMode` 等价于 `AscendC::Reg::LnSpecificMode`。

### Subnormal 处理决策流程

```
算子是否使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal？
├─ 否 → 无需处理 Subnormal
└─ 是 → 输入数据是否可能包含 Subnormal 浮点数？
    ├─ 否 → 使用默认 INTRINSIC 模式（高性能，Subnormal→0）
    └─ 是 → 使用 PRECISION_1ULP_FTZ_FALSE（软件模拟，精度扩展）
```

---

## 补充 4：架构差异全景表（220x vs 351x 完整对照）

| 维度 | 220x（MemBase，910b/910_93） | 351x（RegBase，950） | 影响层级 |
|------|---------------------------|---------------------|---------|
| Vector Core 架构 | MemBase（数据在 UB 内存中流转） | RegBase（数据可直接在寄存器中操作） | L5 |
| 编程模式 | 仅 SIMD | SIMD + SIMT + SIMD/SIMT 混合 | L4 |
| UB 容量 | 128KB（910b）/ 192KB（910_93） | 256KB | L3 |
| UB 结构 | 16 bank group × 3 bank × 4KB | 8 bank group × 2 bank × 16KB | L3 |
| SIMT DCache | 无 | 复用 UB 空间，需预留 40KB | L3 |
| L1→GM 通路 | ✓ 支持 | ✗ 删除 | L3+ |
| GM→L0A/L0B 通路 | ✓ 支持 | ✗ 删除 | L3+ |
| UB→L1 通路 | ✗ 不支持 | ✓ 新增 | L3+ |
| L0C→UB 通路 | ✗ 不支持 | ✓ 新增 | L3+ |
| Subnormal 支持 | ✓ 硬件支持 | ✗ 默认不支持（需软仿） | L3 |
| int4b_t Cube 计算 | ✓ 支持 | ✗ 不支持 | L3 |
| 4:2 稀疏 | ✓ 支持 | ✗ 不支持 | L3 |
| L0A 分形格式 | ZZ | NZ | L3+ |
| SetLoadDataBoundary | ✓ 支持 | ✗ 删除 | L3+ |
| 新数据类型 | - | fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t | L3/L5 |
| 溢出模式控制 | - | GetCtrlSpr/SetCtrlSpr（SPR 寄存器） | L5 |

---

## 补充 5：相关参考文档路径

| 文档类别 | 路径 | 说明 |
|---------|------|------|
| Register-based Vector / MicroAPI | `references/reg-base-vector/` | Reg 矢量计算编程指南 |
| Memory-based Vector 操作 | `references/memory-base-vector/` | 掩码操作、归约计算、高维切分、连续计算 API |
| 迁移相关官方文档 | `references/migration/` | 220x→351x 架构变更、基础/高阶 API 迁移指导、算子编译迁移、兼容性说明 |
| SIMD 编程参考 | `references/simd/` | SIMD BuiltIn 关键字、语言扩展层 C API |
| SIMT 编程参考 | `references/simt/` | SIMT 线程架构、同步机制、AI Core SIMT 编程 |
| API 概述 | `references/api-overview/` | 编程接口概述、高阶/基础 API 概述 |
