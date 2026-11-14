# Cast 规则

> b16↔fp32、int32↔fp32、int8/uint8↔fp32 的 CastTrait 配置和 DIST 配对。

## 1 CastTrait 结构定义

```cpp
struct CastTrait {
    Reg::RegLayout      layout;       // 寄存器布局：ZERO / ONE / TWO / THREE / UNKNOWN
    Reg::SatMode        sat;          // 饱和模式：UNKNOWN / NO_SAT / SAT
    Reg::MaskMergeMode  maskMerge;    // Mask 合并模式：ZEROING / MERGING
    RoundMode           round;        // ⚠ 顶层命名空间 AscendC::RoundMode，不在 Reg:: 下
};
```

**⚠ 关键约束**：`RoundMode` 不在 `AscendC::Reg::` 下，是顶层 `AscendC::RoundMode`。写成 `Reg::RoundMode::...` 会编译报错。

## 2 支持的类型转换矩阵

### 2.1 浮点转整数

| src dtype | dst dtype | 支持的 round mode | 支持的 sat mode |
|-----------|-----------|------------------|-----------------|
| half | int4x2_t | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |
| half | int8_t | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |
| half | uint8_t | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |
| half | int16_t | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |
| half | int32_t | RINT, ROUND, FLOOR, CEIL, TRUNC | UNKNOWN |
| float | int16_t | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |
| float | int32_t | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |
| float | int64_t | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |

### 2.2 浮点转浮点

| src dtype | dst dtype | 支持的 round mode | 支持的 sat mode |
|-----------|-----------|------------------|-----------------|
| half | float | UNKNOWN | UNKNOWN |
| bfloat16_t | float | UNKNOWN | UNKNOWN |
| float | half | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |
| float | bfloat16_t | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |
| half | bfloat16_t | RINT, ROUND, FLOOR, CEIL, TRUNC | UNKNOWN |
| bfloat16_t | half | RINT, ROUND, FLOOR, CEIL, TRUNC | NO_SAT, SAT |

### 2.3 整数转浮点

| src dtype | dst dtype | 支持的 round mode | 支持的 sat mode |
|-----------|-----------|------------------|-----------------|
| int4x2_t | half | UNKNOWN | - |
| int4x2_t | bfloat16_t | UNKNOWN | - |
| uint8_t | half | UNKNOWN | - |
| int8_t | half | UNKNOWN | - |
| int16_t | float | RINT, ROUND, FLOOR, CEIL, TRUNC | - |
| int64_t | float | RINT, ROUND, FLOOR, CEIL, TRUNC | - |
| int16_t | half | UNKNOWN | - |
| int32_t | float | UNKNOWN | - |

### 2.4 整数转整数

| src dtype | dst dtype | 支持的 sat mode |
|-----------|-----------|-----------------|
| uint32_t | uint8_t | NO_SAT, SAT |
| int32_t | uint8_t | NO_SAT, SAT |
| uint16_t | uint8_t | NO_SAT, SAT |
| int16_t | uint8_t | NO_SAT, SAT |
| uint32_t | uint16_t | NO_SAT, SAT |
| uint32_t | int16_t | NO_SAT, SAT |
| int32_t | uint16_t | NO_SAT, SAT |
| int32_t | int16_t | NO_SAT, SAT |
| int16_t | uint16_t | UNKNOWN |
| uint8_t | uint16_t | UNKNOWN |
| int8_t | int16_t | UNKNOWN |
| uint16_t | uint32_t | UNKNOWN |
| int16_t | uint32_t | UNKNOWN |
| int16_t | int32_t | UNKNOWN |
| int32_t | int64_t | UNKNOWN |
| uint8_t | uint32_t | NO_SAT, SAT |
| int8_t | int32_t | NO_SAT, SAT |
| int16_t | int4x2_t | NO_SAT, SAT |
| int4x2_t | int16_t | NO_SAT, SAT |
| int64_t | int32_t | NO_SAT, SAT |

## 3 RoundMode 舍入规则详解

### 3.1 CAST_RINT（就近舍入，距离相等时向偶数进位）

- 输入 3.3 → 输出 3
- 输入 5.9 → 输出 6
- 输入 5.5 → 输出 6（6 是偶数，5 是奇数，向 6 舍入）
- 输入 4.5 → 输出 4（4 是偶数）
- 输入 -2.4 → 输出 -2
- 输入 -3.6 → 输出 -4

### 3.2 CAST_ROUND（就近舍入，距离相等时向远离 0 方向进位）

- 输入 3.3 → 输出 3
- 输入 5.9 → 输出 6
- 输入 5.5 → 输出 6（6 相比 5，距离 0 更远）
- 输入 -2.4 → 输出 -2
- 输入 -3.6 → 输出 -4
- 输入 -6.5 → 输出 -7（-7 相比 -6，距离 0 更远）

### 3.3 CAST_FLOOR（向负无穷方向舍入）

- 输入 3.2 → 输出 3
- 输入 7.9 → 输出 7
- 输入 -4.6 → 输出 -5
- 输入 -3.1 → 输出 -4

### 3.4 CAST_CEIL（向正无穷方向舍入）

- 输入 3.2 → 输出 4
- 输入 7.9 → 输出 8
- 输入 -4.6 → 输出 -4
- 输入 -3.1 → 输出 -3

### 3.5 CAST_TRUNC（向 0 方向舍入）

- 输入 3.2 → 输出 3
- 输入 7.9 → 输出 7
- 输入 -4.6 → 输出 -4
- 输入 -3.1 → 输出 -3

## 4 fp16/bf16 <-> fp32 转换模式

### 4.1 扩位 vs 缩位：必须用不同 CastTrait

| | 扩位 b16→fp32 | 缩位 fp32→b16 |
|---|---|---|
| `sat` | `UNKNOWN` | `NO_SAT`（不可 `UNKNOWN`） |
| `round` | `CAST_NONE` | `CAST_RINT`（不可 `CAST_NONE`） |

**推荐模板（直接拷贝）**：

```cpp
// 扩位 b16 → fp32
static constexpr AscendC::Reg::CastTrait kCastTraitB16ToF32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE};

// 缩位 fp32 → b16（fp16 / bf16 统一）
static constexpr AscendC::Reg::CastTrait kCastTraitF32ToB16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
```

**约束**：
- 扩位与缩位不可共用同一份 trait
- bf16 缩位比 fp16 更严格，统一用 `CAST_RINT` 最安全

### 4.2 b16 加载/写出配对

| 方向 | 模式 | 说明 |
|------|------|------|
| UB → 寄存器 b16→fp32 | `LoadAlign<T, DIST_UNPACK_B16>` | 每次加载 VL/2 字节（64 half=128B），解包到 fp32 lane 低 16 位 |
| 寄存器 → UB fp32→b16 | `StoreAlign<T, DIST_PACK_B32>` | 从 fp32 lane 低 16 位提取 b16，打包成 dense |

**遗漏 `DIST_PACK_B32` 的典型症状**：output[0] 正确，output[1..] 全错。默认 DIST_NORM 把寄存器当 dense 128-lane b16 读，把奇数 lane 的 0 混入 UB。

### 4.3 完整循环范式

```cpp
// __simd_vf__ 函数
template <typename D_T>
__simd_vf__ inline void CastLoopVfImpl(
    __ubuf__ D_T* srcUb, __ubuf__ D_T* dstUb,
    uint32_t totalElems, uint16_t repeat)
{
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::RegTensor<D_T>   b16Reg;
    AscendC::Reg::MaskReg          mask;

    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = i * VL_F32;
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        // 1) 加载
        if constexpr (NeedCast<D_T>()) {
            AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                b16Reg, srcUb + off);
            // kCastTraitB16ToF32：b16→fp32 扩位 CastTrait，需在命名空间级别定义
            AscendC::Reg::Cast<float, D_T, kCastTraitB16ToF32>(f32Reg, b16Reg, mask);
        } else {
            AscendC::Reg::LoadAlign(f32Reg, reinterpret_cast<__ubuf__ float*>(srcUb) + off);
        }

        // 2) 计算（fp32）
        // ...

        // 3) 写出
        if constexpr (NeedCast<D_T>()) {
            // kCastTraitF32ToB16：fp32→b16 缩位 CastTrait，需在命名空间级别定义
            AscendC::Reg::Cast<D_T, float, kCastTraitF32ToB16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                dstUb + off, b16Reg, mask);
        } else {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(dstUb) + off,
                                     f32Reg, mask);
        }
    }
}

// 调用侧
asc_vf_call<CastLoopVfImpl<D_T>>(srcUb, dstUb, totalElems, repeat);
```

`NeedCast<D_T>()` = `!AscendC::IsSameType<D_T, float>::value`。

## 5 int8/uint8 <-> fp32 转换模式（两步转换）

int8/uint8 与 fp32 之间**不能直接转换**，必须经过 half 中间类型，采用两步转换链。

### 5.1 int8/uint8 → fp32（加载 + 扩位）

**转换链**：`int8/uint8 → half → float`

**CastTrait 配置**：

```cpp
// 第一步：B8 → B16（扩位）
constexpr static MicroAPI::CastTrait castTraitB82B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,  // 扩位不需要指定舍入模式
};

// 第二步：B16 → F32（扩位）
constexpr static MicroAPI::CastTrait cutsomCastTrait0 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,  // 扩位时 TRUNC 或 UNKNOWN 均可
};
```

**加载模式**：`DIST_UNPACK4_B8`

```cpp
// 1) 加载 B8 数据（解包）
MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
    vregInput1, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));

// 2) 第一步转换：B8 → B16
MicroAPI::Cast<half, T, castTraitB82B16>(regTensor5, vregInput1, mask);

// 3) 第二步转换：B16 → F32
MicroAPI::Cast<float, half, cutsomCastTrait0>(regTensor3, regTensor5, mask);
```

### 5.2 fp32 → int8/uint8（溢出环绕 + 缩位 + 写出）

**转换链**：`float → int32（溢出处理）→ float → half → int8/uint8`

**⚠ 关键约束**：计算结果无法保证不超出 int8/uint8 范围（[-128, 127] 或 [0, 255]），必须进行溢出环绕处理（wraparound）而非饱和（saturation）。

**CastTrait 配置**：

```cpp
// 溢出处理阶段
constexpr static MicroAPI::CastTrait castTraitFLOAT2INT32 = {
    MicroAPI::RegLayout::UNKNOWN,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};

constexpr static MicroAPI::CastTrait castTraitINT322FLOAT = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};

// 缩位阶段
constexpr static MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,      // 缩位必须 NO_SAT
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,           // 缩位必须指定舍入模式
};

constexpr static MicroAPI::CastTrait castTraitB162B8 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,      // 缩位必须 NO_SAT
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,          // 向 0 舍入
};
```

**完整转换流程**：

```cpp
// ========== 阶段 1：溢出环绕处理 ==========
// 1) F32 → INT32（截断）
MicroAPI::Cast<int32_t, float, castTraitFLOAT2INT32>(regTensor1, regTensor3, mask);

// 2) 位运算取低 8 位（& 0xFF）
MicroAPI::Duplicate(regTensor2, 255);
MicroAPI::And<int32_t, MicroAPI::MaskMergeMode::ZEROING>(
    regTensor1, regTensor1, regTensor2, mask);

// 3) INT32 → F32
MicroAPI::Cast<float, int32_t, castTraitINT322FLOAT>(regTensor3, regTensor1, mask);

// 4) 偏移到 uint8 范围 [0, 255]
MicroAPI::Adds<float>(regTensor3, regTensor3, (float)128, mask);

// 5) 计算 mod 256（实现环绕）
MicroAPI::Muls<float>(regTensor4, regTensor3, (float)(0.00390625), mask);  // /256
MicroAPI::Truncate<float, RoundMode::CAST_TRUNC, MicroAPI::MaskMergeMode::ZEROING>(
    regTensor4, regTensor4, mask);
MicroAPI::Muls<float>(regTensor4, regTensor4, (float)(256), mask);
MicroAPI::Sub<float>(regTensor3, regTensor3, regTensor4, mask);

// 6) 偏移回 int8 范围 [-128, 127]
MicroAPI::Adds<float>(regTensor3, regTensor3, (float)(-128), mask);

// ========== 阶段 2：F32 → B16 → B8 转换 ==========
// 7) F32 → B16
MicroAPI::Cast<half, float, castTraitB322B16>(regTensor5, regTensor3, mask);

// 8) B16 → B8
MicroAPI::Cast<T, half, castTraitB162B8>(vregOutput, regTensor5, mask);

// ========== 阶段 3：写出 ==========
// 9) 写出 B8 数据（打包）
MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK4_B32>(
    (__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
```

**溢出环绕原理**：
- int8 范围 [-128, 127]，uint8 范围 [0, 255]
- 通过位运算 `& 0xFF` 取低 8 位实现环绕
- 偏移 +128 转到 uint8 空间，mod 256 后偏移 -128 回到 int8 空间
- 示例：257 → &0xFF → 1 → +128 → 129 → mod256 → 129 → -128 → 1

## 6 饱和模式 vs 非饱和模式

### 6.1 浮点数转整数

**非饱和模式（NO_SAT）**：
- 输入数据超过输出类型最值时，结果被截断为目标格式的数据宽度
- 例如：输入 half 值为 257，输出 uint8_t 值为 1
- 输入 +/-inf，返回输出类型的对应最值
- 输入 nan，返回 0

**饱和模式（SAT）**：
- 输入数据超过输出类型最值时，返回输出类型的对应最值
- 例如：输入 half 值为 257，输出 uint8 值为 255
- 输入 half 值为 -inf，输出 uint8_t 值为 0
- 输入 nan，返回 0

### 6.2 浮点数转浮点数

**非饱和模式（NO_SAT）**：
- 输入数据为 nan 时，输出为 nan
- 输入 +/-inf 时，输出为 +/-inf

**饱和模式（SAT）**：
- 输入为 nan 时，输出为 0
- 输入数据超过输出类型最值时，返回输出类型的对应最值

**约束**：
- 当输出类型 float32 时，只支持不饱和模式
- 当输出类型为 fp8_e4m3fn_t 时，由于没有 inf 表示格式，输出为 nan
- 当输出类型为 fp8_e5m2_t/fp8_e4m3fn_t 时，输入 nan，默认输出为 0

### 6.3 整数转整数

**非饱和模式（NO_SAT）**：
- 输入数据会截断为目标数据格式
- 例如：输入 int32_t 值为 256，输出 uint8_t 值为 0

**饱和模式（SAT）**：
- 输入数据超出目标数据范围，会饱和为目标数据最值

**约束**：
- 对于窄数据类型（如 int16_t）转宽数据类型（如 uint32_t），CastTrait 的 SatMode 字段填 `UNKNOWN`（见 §2.4 表），硬件实际行为为饱和模式（由 `SetCtrlSpr(ISASI)` 寄存器控制），输入负数会被饱和成 0

## 7 编译错误对照表

| 错误信息 | 根因 | 修复 |
|---------|------|------|
| `no member named 'RoundMode' in namespace 'AscendC::Reg'` | 写成 `Reg::RoundMode::...` | 改成顶层 `AscendC::RoundMode::...` |
| `vcvt (f322f16) can only be: RS_DISABLE, RS_ENABLE` | `SatMode::UNKNOWN` 用在缩位 | 缩位 trait 改 `NO_SAT` |
| `roundMode can't be CAST_NONE when float to bfloat16_t` | `CAST_NONE` 用在 fp32→bf16 | 改 `CAST_RINT` |
| `vcvt (f322bf16) can only be: ROUND_R/A/F/C/Z` | 同上（fp16 路径也会触发） | 改 `CAST_RINT` |
| b16 加载报 lane 不匹配 | 漏了 `DIST_UNPACK_B16` | b16 加载加 `LoadDist::DIST_UNPACK_B16` |
| b16 输出 output[0]对、其余全错 | 漏了 `DIST_PACK_B32` | b16 写出加 `StoreDist::DIST_PACK_B32` |
| b8 加载报 lane 不匹配 | 漏了 `DIST_UNPACK4_B8` | b8 加载加 `LoadDist::DIST_UNPACK4_B8` |
| b8 输出 output[0]对、其余全错 | 漏了 `DIST_PACK4_B32` | b8 写出加 `StoreDist::DIST_PACK4_B32` |

## 8 最佳实践总结

### 8.1 选择转换路径

| 源类型 | 目标类型 | 转换路径 | LoadDist | StoreDist |
|--------|---------|---------|----------|-----------|
| fp16/bf16 | fp32 | 直接转换 | `DIST_UNPACK_B16` | - |
| fp32 | fp16/bf16 | 直接转换 | - | `DIST_PACK_B32` |
| int8/uint8 | fp32 | B8→B16→F32 | `DIST_UNPACK4_B8` | - |
| fp32 | int8/uint8 | F32→B16→B8 | - | `DIST_PACK4_B32` |

### 8.2 CastTrait 选择原则

1. **扩位（小→大）**：`sat=UNKNOWN`, `round=CAST_NONE` 或 `CAST_TRUNC`
2. **缩位（大→小）**：`sat=NO_SAT`, `round=CAST_RINT`（最安全）
3. **整数转浮点**：`sat=UNKNOWN`, `round=UNKNOWN` 或指定舍入模式
4. **浮点转整数**：`sat=NO_SAT` 或 `SAT`, `round=CAST_TRUNC`（截断）或 `CAST_RINT`（舍入）
