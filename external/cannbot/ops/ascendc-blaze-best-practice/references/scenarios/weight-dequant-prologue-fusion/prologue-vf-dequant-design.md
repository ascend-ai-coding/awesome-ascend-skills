# VF 反量化链路专题

本文是本场景 Prologue 层的 VF 反量化链路专题。Step 3 用它设计 Prologue delta 中的 VF Cast 合同；DESIGN 冻结后，Step 3 用它编译对应 PLAN action；Step 4 只有在 PLAN 将本文绑定到当前 action 时才读取。本文不选择 Blaze 组装方案，也不提供固定 VF recipe。

## 1. 输入与输出合同

输入：`matmul_base_analysis`、Investigation 中已闭合的 VF/Cast API 事实、反量化公式合同和当前 dtype 约束。输出：

```text
vf_dequant_contract
cast_chain_contract
vf_validation_additions
```

每个结论必须引用当前候选组装方案评估、witness 和 evidence IDs。历史 `source_observed`/`device_verified` 只有在 Investigation、Blaze 组装方案、构建和测试范围一致时才能复用；否则标记 `unverified`。

## 2. 反量化公式合同

DESIGN 必须冻结以下反量化语义：

```text
B_dequant[i,j] = (B[i,j] + offset[j]) * scale[j]
```

| 对象           | 必填合同                                                                       |
| -------------- | ------------------------------------------------------------------------------ |
| B 输入         | 低比特 dtype（int8/fp8）、逻辑/物理 shape、layout、perchannel 量化模式         |
| Cast 中间值    | 每步 Cast 的输入/输出 dtype、精度语义、执行域；按 BType 分支选择中间寄存器类型 |
| scale          | dtype（bf16/fp16）、shape`(N,)`、perchannel 加载方式、广播映射               |
| offset         | dtype、shape`(N,)`、可选语义、缺省为 0 的条件                                |
| B_dequant 输出 | dtype（bf16/fp16）、写入 L1 的 layout 和生命周期                               |

offset 存在时先 Add 后 Mul；offset 缺省时跳过 Add 分支。公式顺序、dtype 转换链和舍入行为必须与 CPU Golden 完全一致。

## 3. Cast 链合同

VF 反量化在 AIV 侧执行，使用 `__simd_vf__` 执行域。DESIGN 必须按当前 B 输入 dtype 和目标 B_dequant dtype 冻结 Cast 链。下表汇总当前已验证的 cast 链路（CANN 9.1.0 / dav-3510）：

| B dtype → B_dequant dtype | Cast 步骤                   | 中间寄存器类型         | 验证状态         |
| -------------------------- | --------------------------- | ---------------------- | ---------------- |
| int8 → bf16               | 两步：int8→half→bf16      | RegTensor&lt;half&gt;  | ✅ 编译+精度通过 |
| int8 → fp16               | 一步：int8→half            | —                     | ✅ 编译+精度通过 |
| fp8_e4m3fn → bf16         | 两步：fp8_e4m3→float→bf16 | RegTensor&lt;float&gt; | ✅ 编译+精度通过 |
| fp8_e4m3fn → fp16         | 两步：fp8_e4m3→float→half | RegTensor&lt;float&gt; | ✅ 编译+精度通过 |
| fp8_e5m2 → bf16           | 两步：fp8_e5m2→float→bf16 | RegTensor&lt;float&gt; | ✅ 编译+精度通过 |
| fp8_e5m2 → fp16           | 两步：fp8_e5m2→float→half | RegTensor&lt;float&gt; | 编译通过，同链路 |

### 3.1 Cast 链路实践参考

以下为各 dtype 的实际实现方式，供 Investigation 和 DESIGN 参考。**这些不是固定 recipe**——具体 API 参数（CastTrait/LoadDist/StoreDist）应从当前 CANN 源码确认。

#### int8（8bit→16bit）

- LoadDist：`DIST_UNPACK_B8`（每字节展开为 16-bit slot `[byte, 0]`）
- Cast 链：int8→half 直接（目标 fp16）；int8→half→bf16 两步（目标 bf16，硬件不支持 int8→bf16 直转）
- 注意点：当目标为 bf16 时，int8 需经 half 中转

#### fp8_e4m3fn / fp8_e5m2（8bit→32bit→16bit，拆合模式）

`DIST_UNPACK_B8` 将每个 fp8 字节展开为 16-bit slot `[byte, 0]`。由于 fp8→float 是 8bit→32bit 转换，需要用两种 RegLayout 分别提取偶数和奇数 lane 的 fp8 值到两个 float 寄存器，再经 float→bf16 的两种 RegLayout 分拆到两个 bf16 寄存器，最后用 `Or` 合并为一个密集 bf16 寄存器。

**CastTrait 配置：**

| CastTrait                    | RegLayout | SatMode     | MaskMergeMode | RoundMode      | 用途                           |
| ---------------------------- | --------- | ----------- | ------------- | -------------- | ------------------------------ |
| `castTraitFp8ToFP32_ZERO`  | `ZERO`  | `UNKNOWN` | `ZEROING`   | `UNKNOWN`    | fp8→float, 提取偶数 lane      |
| `castTraitFp8ToFP32_TWO`   | `TWO`   | `UNKNOWN` | `ZEROING`   | `UNKNOWN`    | fp8→float, 提取奇数 lane      |
| `castTraitFP32ToBF16_ZERO` | `ZERO`  | `NO_SAT`  | `ZEROING`   | `CAST_RINT` | float→bf16, even 16-bit slots |
| `castTraitFP32ToBF16_ONE`  | `ONE`   | `NO_SAT`  | `ZEROING`   | `CAST_RINT` | float→bf16, odd 16-bit slots  |

**Mask 类型规则：** mask 类型必须与 Cast 的**源操作数类型**一致。fp8→float Cast 用 `CreateMask<uint8_t>`；float→bf16 Cast 用 `CreateMask<float>`；Or/Add/Mul/StoreAlign 用 `CreateMask<DequantBType>`。

```cpp
MaskReg maskAll = CreateMask<uint8_t, MaskPattern::ALL>();      // fp8→float Cast 用
MaskReg maskFp32 = CreateMask<float, MaskPattern::ALL>();       // float→bf16 Cast 用
MaskReg maskBf16 = CreateMask<DequantBType, MaskPattern::ALL>(); // Or/Add/Mul/StoreAlign 用

**`Or` 的 `bfloat16_t` 限制：** `Or` 的 `static_assert` 不包含 `bfloat16_t`，需 `reinterpret_cast` 为 `RegTensor<uint16_t>&` 调用。

**VF 函数完整调用序列：**

```cpp
// 寄存器声明
RegTensor<BType> regBIn;
RegTensor<float> regFp32A, regFp32B;      // 偶数/奇数 lane 的 float 中间值
RegTensor<DequantBType> regBf16A, regBf16B; // even/odd 16-bit slots 的 bf16
RegTensor<DequantBType> regBOut;           // 合并后的密集 bf16
RegTensor<DequantBType> regScale, regOffset;
MaskReg maskAll = CreateMask<uint8_t, MaskPattern::ALL>();   // fp8→float Cast 用
MaskReg maskFp32 = CreateMask<float, MaskPattern::ALL>();    // float→bf16 Cast 用
MaskReg maskBf16 = CreateMask<DequantBType, MaskPattern::ALL>(); // Or/Add/Mul/Store 用

// CastTrait 声明（static constexpr，作为 NTTP 传入 Cast）
static constexpr CastTrait castTraitFp8ToFP32_ZERO = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr CastTrait castTraitFp8ToFP32_TWO = {
    RegLayout::TWO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr CastTrait castTraitFP32ToBF16_ZERO = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr CastTrait castTraitFP32ToBF16_ONE = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// 每个 K-N tile 的 VF 计算循环体：
// 1. 加载 scale/offset（按 transB 选 LoadDist）
LoadAlign<DequantBType, LOAD_TRAIT_SCALE>(regScale, scalePhyAddr + vfNOffset);
if constexpr (HasOffset) {
    LoadAlign<DequantBType, LOAD_TRAIT_SCALE>(regOffset, offsetPhyAddr + vfNOffset);
}

// 2. 加载 fp8 权重（DIST_UNPACK_B8: 每字节 → [byte, 0] 16-bit slot）
LoadAlign<BType, LoadDist::DIST_UNPACK_B8>(regBIn, bInPhyAddr + ...);

// 3. fp8→float 拆分（两路：偶数 lane + 奇数 lane）
Cast<float, BType, castTraitFp8ToFP32_ZERO>(regFp32A, regBIn, maskAll);  // 偶数 → 64 float
Cast<float, BType, castTraitFp8ToFP32_TWO>(regFp32B, regBIn, maskAll);   // 奇数 → 64 float

// 4. float→bf16 分拆（两路：even slots + odd slots）
Cast<DequantBType, float, castTraitFP32ToBF16_ZERO>(regBf16A, regFp32A, maskFp32);
Cast<DequantBType, float, castTraitFP32ToBF16_ONE>(regBf16B, regFp32B, maskFp32);

// 5. Or 合并为密集 bf16（需 reinterpret_cast 为 uint16_t）
Or<uint16_t, MaskMergeMode::ZEROING>(
    (RegTensor<uint16_t>&)regBOut,
    (RegTensor<uint16_t>&)regBf16A,
    (RegTensor<uint16_t>&)regBf16B, maskBf16);

// 6. 反量化：Add(offset) → Mul(scale)
if constexpr (HasOffset) {
    Add(regBOut, regBOut, regOffset, maskBf16);
}
Mul(regBOut, regBOut, regScale, maskBf16);

// 7. 写回 UB（NZ/ZN strided，outerAxisSize 含 +1 bank conflict padding）
StoreAlign<DequantBType, DataCopyMode::DATA_BLOCK_COPY>(
    bOutPhyAddr + outerAxisSize * innerOffset + BLOCK_CUBE * outerIdx,
    regBOut, outerAxisSize, maskBf16);
```

> **数据流图：**
>
> ```
> UB fp8 bytes
>   → LoadAlign<DIST_UNPACK_B8> → regBIn: [b0, 0, b1, 0, b2, 0, ...] (128 fp8 in 16-bit slots)
>   → Cast<float, ZERO>  → regFp32A: [b0→f32, _, b2→f32, _, ...]  (偶数 lane, 64 float)
>   → Cast<float, TWO>   → regFp32B: [b1→f32, _, b3→f32, _, ...]  (奇数 lane, 64 float)
>   → Cast<bf16, ZERO>   → regBf16A: [b0→bf16, 0, b2→bf16, 0, ...]  (even 16-bit slots)
>   → Cast<bf16, ONE>    → regBf16B: [0, b1→bf16, 0, b3→bf16, ...]  (odd 16-bit slots)
>   → Or<uint16_t>       → regBOut:  [b0→bf16, b1→bf16, b2→bf16, ...]  (dense 128 bf16)
>   → [Add offset] → [Mul scale]
>   → StoreAlign<DATA_BLOCK_COPY> → UB (NZ/ZN strided)
> ```

### 3.2 扩展新 dtype 的合同

扩展新 dtype 时，DESIGN 必须为每步 Cast 记录：

```text
cast_step_id
input_dtype
output_dtype
casttrait_configuration （从 asc-devkit API 文档确认，编译+精度验证）
loaddist                 （从 asc-devkit LoadAlign 文档确认，编译+精度验证）
storedist                （从 asc-devkit StoreAlign 文档确认，编译+精度验证）
source_ref
observed_limitation
```

Investigation 必须确认所需 Cast dtype 组合在当前 CANN 版本中受支持（查阅 asc-devkit `Cast.md` 数据类型组合表）。CastTrait 不可用或签名不闭合时，标记 blocking 并返回 Step 2 补充调查。

## 4. CastTrait 声明形式约束

CastTrait 作为 NTTP（非类型模板参数）传入 VF 函数时，必须使用 `static constexpr` 声明，不得使用 `constexpr`。理由：`constexpr` 变量作为 NTTP 在某些编译上下文中无法推导，导致编译失败。

DESIGN 必须记录当前 Investigation 确认的 CastTrait 声明形式和 source_ref。PLAN 的 VF 实现动作必须验证复制后的 CastTrait 声明与源码一致。

## 5. VF 调用结构合同

VF 反量化使用三层结构：VF 函数定义（`__simd_vf__` 执行域）→ 参数打包 struct → 调用点（`asc_vf_call`）。DESIGN 必须冻结：

| 层              | 必填合同                                                              |
| --------------- | --------------------------------------------------------------------- |
| VF 函数         | 执行域修饰符、模板参数（BType/DequantBType）、参数列表、Cast 调用顺序 |
| 参数打包 struct | 字段名/类型、UB 地址指针（`__ubuf__`）、迭代计数、offset 存在性分支 |
| 调用点          | `asc_vf_call` 的参数传递、VF 函数绑定、执行域约束                   |

VF 函数内的 Cast 分支通过 `if constexpr` 按 BType 在编译期选择，不产生运行时分支开销。不同 BType 使用不同的中间寄存器类型（int8 走 half，fp8 走 float）。hasOffset 为 false 时编译期排除 Add 路径。

Cast 指令的饱和模式通过 CastTrait 的 SatMode 字段和 SetCtrlSpr 指令控制。`SetCtrlSpr` 必须在 `__simd_vf__` 外调用。具体配置策略由 Investigation 从 asc-devkit `Cast.md` 确认。

### 5.1 scale/offset LoadDist 参考

scale/offset 始终为 `DequantBType`（bf16/fp16），其 LoadDist 仅取决于 transB 和量化模式：

| transB | 量化模式 | LoadDist | 依据 |
|---|---|---|---|
| true (Nk) | perchannel | `DIST_BRC_B16` | 广播单个 b16 到所有 slot |
| true (Nk) | pertensor | `Duplicate`（标量广播） | 单值广播 |
| false (Kn) | perchannel | `DIST_NORM` | 正常加载 |
| false (Kn) | pertensor | `Duplicate` | 单值广播 |

切换 B weight dtype 不影响 scale/offset 的 LoadDist。

## 6. 验证门禁

- Cast 链每步的 dtype 转换顺序与 CPU Golden 完全一致；
- CastTrait、LoadDist、StoreDist 的配置经编译+精度验证确认正确；
- Golden 必须使用与 VF 指令行为一致的计算精度（如 PyTorch bf16 运算）；
- hasOffset 和 hasBias 的组合均需覆盖；
- 各 dtype 边界值（int8: -128/127, fp8 最值）的反量化结果在 Golden 容差内；
- CastTrait 声明形式（`static constexpr`）与 Investigation 确认的源码一致；
- 官方源码区与 Blaze Asset 原文件保持零改动。

具体 dtype、shape 和重复次数来自 DESIGN/PLAN，不由本文固定。
