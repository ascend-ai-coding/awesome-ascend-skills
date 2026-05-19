---
title: RegBase DataCopy 分层
purpose: 区分 RegBase 中 GM/UB 搬运与 UB/寄存器搬运，避免把 kernel-side DataCopy 写进 VF body。
read_when:
  - 需要设计 RegBase 的 CopyIn、Compute、CopyOut 数据流。
  - 不确定 `DataCopyPad`、`Reg::LoadAlign`、`Reg::StoreAlign` 应该放在哪一层。
  - 审查发现 GM/UB、UB/Register 两类搬运语义混在一起。
not_for:
  - 纯 MemBase / LocalTensor 算子实现。
  - 只做 Host tiling 参数推导。
keywords:
  - datacopy
  - DataCopyPad
  - LoadAlign
  - StoreAlign
  - RegTensor
next_reads:
  - regbase_api_reference.md
  - compute_api_membase_vs_regbase.md
  - ../dev-experience/regbase_programming_notes.md
depth: foundation
topic_type: api
---

# RegBase DataCopy 分层

RegBase 算子的 data movement 分两层。名字都像“copy”，但作用域、对象类型、执行管线和 tail 处理完全不同。

| 层级 | 数据方向 | 典型 API | 对象 | 放置位置 | 关键语义 |
|---|---|---|---|---|---|
| GM/UB staging | GM -> UB、UB -> GM | `DataCopy`、`DataCopyPad` | `GlobalTensor<T>`、`LocalTensor<T>` | `CopyIn` / `CopyOut` | MTE 搬运，解决全局内存到 UB tile 的准备和写回 |
| UB/Register VF 搬运 | UB address -> `RegTensor`、`RegTensor` -> UB address | `AscendC::Reg::LoadAlign`、`AscendC::Reg::StoreAlign`、`Load`、`Store` | `__ubuf__ T*`、`RegTensor<T>`、`MaskReg` | `__simd_vf__` / `__VEC_SCOPE__` 内 | 寄存器级 load/store，配合 `MaskReg` 控制 active lanes 和 tail |

设计文档里应同时写清这两层。只写“DataCopy 到寄存器”或“从 UB 复制到 RegTensor”都不够精确。

## 标准 RegBase 数据流

RegBase 推荐结构：

1. `CopyIn`：使用 `DataCopy` / `DataCopyPad` 把 GM tile 搬到 UB。
2. `Compute`：取得 UB address、count 和 loop 信息，通过 `asc_vf_call<...>` 进入 `__simd_vf__`。
3. VF body：使用 `AscendC::Reg::LoadAlign` / `Load` 把 UB address load 到 `RegTensor`。
4. VF body：使用 `AscendC::Reg::*` 在 `RegTensor` 上计算。
5. VF body：使用 `AscendC::Reg::StoreAlign` / `Store` 把 `RegTensor` store 回 UB address。
6. `CopyOut`：使用 `DataCopy` / `DataCopyPad` 把 UB tile 写回 GM。

## GM/UB 搬运：`DataCopy` / `DataCopyPad`

`DataCopy` / `DataCopyPad` 是 kernel 侧搬运 API，用于 `GlobalTensor<T>` 与 `LocalTensor<T>` 之间的数据交换。它们不应该出现在 `__simd_vf__` 函数内，也不直接生成 `RegTensor`。

常见 copy-in / copy-out 形态：

```cpp
// CopyIn: GM -> UB
AscendC::DataCopyPad(localUb, xGm[progress],
    {(uint16_t)1, (uint16_t)(count * sizeof(float)), (uint16_t)0, (uint16_t)0},
    {false, 0, 0, 0});

// CopyOut: UB -> GM
AscendC::DataCopyPad(yGm[progress], localUb,
    {(uint16_t)1, (uint16_t)(count * sizeof(float)), (uint16_t)0, (uint16_t)0});
```

使用规则：

- 对齐和 tail 不确定时，GM -> UB 优先考虑 `DataCopyPad`，并写清 padding 是否影响数学语义。
- `DataCopyParams.blockLen` 描述当前 GM/UB 搬运长度；不要把它误写成 VF lanes 数。
- `DataCopyPad` 的 padding 只属于 UB staging，不等于 VF mask。进入 VF 后仍要用 `MaskReg` 处理 tail。
- `SetValue` / `GetValue` 只作为调试辅助，不是生产搬运路径。

## UB/Register 搬运：`Reg::Load*` / `Reg::Store*`

进入 `__simd_vf__` 后，VF body 接收的是 `__ubuf__ T*` 地址。此时使用 `AscendC::Reg::*` load/store，把 UB 数据映射到寄存器对象。

代表签名来自 CANN 9.0.0 `kernel_reg_compute_datacopy_intf.h`：

```cpp
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T* srcAddr);

template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg, MaskReg& mask);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void Load(U& dstReg, __ubuf__ T* srcAddr);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void Store(__ubuf__ T* dstAddr, U& srcReg, uint32_t count);
```

这里的 `__simd_callee__` 是 SDK RegBase primitive 的声明形态。新算子的 VF 入口仍应写成 `__simd_vf__`，并由 `asc_vf_call` 调用。

典型 VF body：

```cpp
__simd_vf__ inline void MyRegBaseVf(__ubuf__ float* in, __ubuf__ float* out,
    uint32_t count, uint16_t loopNum, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask;

    for (uint16_t i = 0; i < loopNum; ++i) {
        uint32_t valid = count - static_cast<uint32_t>(i) * oneRepeatSize;
        if (valid > oneRepeatSize) {
            valid = oneRepeatSize;
        }

        mask = AscendC::Reg::UpdateMask<float>(valid);
        AscendC::Reg::LoadAlign(xReg, in + static_cast<uint32_t>(i) * oneRepeatSize);
        // ... AscendC::Reg::* compute ...
        AscendC::Reg::StoreAlign(out + static_cast<uint32_t>(i) * oneRepeatSize, yReg, mask);
    }
}
```

使用规则：

- `LoadAlign` / `StoreAlign` 的对象是 UB address 与 `RegTensor`，不是 GM address。
- `MaskReg` 控制 VF active lanes；tail mask 必须来自当前 repeat 的有效元素数。
- `LoadDist` / `StoreDist` 描述寄存器 load/store 分布策略，不是 queue 同步或 MTE 搬运参数。
- 如果使用 `LoadUnAlign*` / `StoreUnAlign*`，必须说明为什么 aligned 形态不可用，以及 unaligned 预处理对象的生命周期。

## 审查清单

- `CopyIn` / `CopyOut` 是否只负责 GM/UB staging。
- VF body 是否只接收 `__ubuf__` 地址，不直接访问 GM。
- 是否用 `Reg::Load*` 把 UB address 转为 `RegTensor`，再用 `Reg::Store*` 写回 UB。
- `DataCopyPad` padding 与 `MaskReg` tail 是否分别说明，不能互相替代。
- 设计文档是否把 GM/UB 搬运长度、UB offset、VF repeat size、tail lanes 分开记录。

## 相关文档

- [[regbase_api_reference]]
- [[compute_api_membase_vs_regbase]]
- [[regbase_api_whitelist]]
- [[../dev-experience/regbase_programming_notes]]
- [[../pitfalls/regbase_vs_membase_confusions]]
