# VF 编程规则

> regbase VF 编程的 API 速查、循环范式、硬约束。

## 1 命名空间与核心类型

| 类型 | 说明 |
|------|------|
| `AscendC::Reg::RegTensor<T>` | 向量寄存器，VL=256B。b16→128 元素，b32→64 元素 |
| `AscendC::Reg::MaskReg` | 掩码寄存器 |
| `__ubuf__ T*` | UB 地址指针，由 `localTensor.GetPhyAddr()` 获取 |

## 2 Load / Store

**API 选型**：新代码统一走 `LoadAlign` / `StoreAlign`（地址要求 32B 对齐）。

```cpp
// Load: UB → 寄存器（无 mask；srcAddr 必须 32B 对齐）
AscendC::Reg::LoadAlign(vReg, ubPtr + offset);

// Load b16→f32: DIST_UNPACK_B16 只加载 VL/2 字节 (64 half = 128B)
AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(tReg, tPtr + offset);

// Store: 寄存器 → UB（带 mask；dstAddr 必须 32B 对齐）
AscendC::Reg::StoreAlign(ubPtr + offset, vReg, maskReg);

// Store fp32→b16: DIST_PACK_B32 与 DIST_UNPACK_B16 对称
AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(ubPtr + offset, vReg, maskReg);
```

**约束**：
- 地址非 32B 对齐时用 `LoadUnAlign` / `StoreUnAlign` + `UnalignRegFor[Load|Store]` 状态对象配对
- b16 加载必须 `DIST_UNPACK_B16`，写出必须 `DIST_PACK_B32`
- int8/uint8 加载 `DIST_UNPACK4_B8`，写出 `DIST_PACK4_B32`（详见 [cast-rules.md](cast-rules.md) §5）

## 3 UpdateMask

```cpp
template <typename T>
__simd_callee__ inline MaskReg UpdateMask(uint32_t& scalarValue);
```

**关键**：`scalarValue` 是引用传递，内部自动递减 VL_T（float:64，half:128）。**调用后不可再手动递减**。

```cpp
uint32_t remaining = static_cast<uint32_t>(totalElems);
for (uint16_t i = 0; i < repeatTime; ++i) {
    maskReg = AscendC::Reg::UpdateMask<float>(remaining);
    // remaining 已被自动递减，不可再写 remaining -= 64  // ❌ 会变成每轮减 2×VL
}
```

## 4 MaskMergeMode

| 模式 | 说明 |
|------|------|
| `ZEROING` | mask 外元素置零（常用） |
| `MERGING` | mask 外元素保持原值 |

## 5 计算操作速查

```cpp
// Binary: dst = src0 op src1
AscendC::Reg::Add/Sub/Mul/Div/Max/Min(dstReg, src0Reg, src1Reg, maskReg);
// 带 MaskMergeMode:
AscendC::Reg::Add<T, AscendC::Reg::MaskMergeMode::ZEROING>(dst, src0, src1, mask);

// Scalar: dst = src op scalar（无 Subs，用 Adds 负值代替）
AscendC::Reg::Adds/Muls(dstReg, srcReg, scalarVal, maskReg);

// Unary
AscendC::Reg::Sqrt/Abs/Exp/Ln/Neg/Relu(dstReg, srcReg, maskReg);

// Duplicate
AscendC::Reg::Duplicate(dstReg, scalarValue);

// Cast
AscendC::Reg::Cast<DstT, SrcT, castTrait>(dstReg, srcReg, maskReg);
```

## 6 LocalMemBar 使用规则

`LocalMemBar<VEC_STORE, VEC_LOAD>()` 保证之前所有 VEC_STORE 写入对后续 VEC_LOAD 可见。仅作用于 Vector 单元内部。

| 场景 | 是否需要 |
|------|---------|
| 每次迭代读写**不同地址**（`off = i * VL`） | **不需要** |
| 同一迭代内 load-modify-store 同一地址 | **不需要** |
| 跨不同 buffer（src→dst elewise） | **不需要** |
| 连续迭代读写**同一地址**（in-place reduce） | **需要** |
| 内层循环写，外层下次循环读同一地址 | **需要** |

**常见错误**：在标准 elewise 循环末尾加 `LocalMemBar` 是多余的——每次迭代目标地址都不同。

## 7 VF 循环范式

```cpp
constexpr uint32_t kRepF32 = 256 / sizeof(float);  // = 64
uint16_t repeatTime = static_cast<uint16_t>(
    (static_cast<uint32_t>(totalElems) + kRepF32 - 1) / kRepF32);
for (uint16_t i = 0; i < repeatTime; ++i) {
    int32_t offset = static_cast<int32_t>(i) * kRepF32;
    // ...
}
```

**约束**：`offset` 用 `static_cast<int32_t>(i) * kRepF32`，避免 `uint16_t` × `int32_t` 隐式转换。

## 8 `__simd_vf__` 函数 + `asc_vf_call` 调用范式

```cpp
// __simd_vf__ 函数：C 函数格式，不能是类成员函数
template <typename T>
__simd_vf__ inline void XxxVfImpl(
    __ubuf__ T* src, __ubuf__ T* dst,
    uint32_t totalElems, uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<T> aReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(aReg, src + off);
        // ... 计算
        AscendC::Reg::StoreAlign(dst + off, aReg, mask);
    }
}

// 调用侧：asc_vf_call 显式传递所有模板参数
asc_vf_call<XxxVfImpl<T>>(src, dst, totalElems, repeatTime);
```

## 9 VF 7 条硬约束

1. **for 循环必须从 0 开始**
2. **VF 内最多 4 层 for**
3. **循环控制变量必须是 `uint16_t`**
4. **VF 内禁止数组**
5. **VF 内禁止访问类的"对象级"成员变量**（在 `asc_vf_call` 调用侧提取为临时变量/参数传入）
6. **VF 内禁止运行时 `if/else`**（用 `if constexpr` 代替）
7. **使用 `asc_vf_call` 启动 SIMD VF 函数**

## 10 `__simd_vf__` 函数约束

- C 函数格式，**不能是类成员函数**
- **不能引用外部模板**（如 `Reducer::identity` 需作为参数传入）
- 支持模板参数（如 `typename T` / `bool isTailR`）
- 支持内部 `if constexpr`
- `asc_vf_call` 显式传递所有模板参数
