# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 6534
- **PR作者**: ranczhw
- **代码文件**: 5 个文件
- **代码侧别**: Kernel侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 13 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 13 | 10 | 100% |

---

## 发现问题


### 文件: mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp（Kernel侧）

#### [3] 人工检视意见

- **提出人**: liuboxi
- **作者**: ranczhw
- **文件**: mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp
- **行号**: 1031
- **评论时间**: 2026-06-08
- **Commit**: d9c273b6874a
- **问题描述**:

  > 该函数无调用点，明确是否为冗余代码，考虑删除。

- **代码片段**（行1031）:
```cpp
1021 |         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
1022 |         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
1023 |         AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0);
1024 |         AscendC::Duplicate(tmp, 0, num);
1025 |         AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
1026 |         AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
1027 |         AscendC::DataCopy(tokenPerExpert, tmp, num);
1028 |     }
1029 | 
1030 |     CATLASS_DEVICE
1031 |     void UpdateAicFlags(const Params &params)
1032 |     {
1033 |         float flagBase = 1.0f * params.expertPerRank;
1034 |         __gm__ float *aicFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;
1035 |         float flag = 0.0f;
1036 |         float lastflag = -1.0f;
1037 |         AscendC::LocalTensor<float> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<float>(0);
1038 |         __gm__ float *flagPtr = workspaceInfo.ptrSoftFlagBase;
1039 |         AscendC::GlobalTensor<float> flagGM;
1040 |         flagGM.SetGlobalBuffer(flagPtr);
```

---

#### [4] 人工检视意见

- **提出人**: liuboxi
- **作者**: ranczhw
- **文件**: mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp
- **行号**: 1085
- **评论时间**: 2026-06-08
- **Commit**: d9c273b6874a
- **问题描述**:

  > set未调用，wait有调用，明确是否已正确配对使用，避免出现同步问题。

- **代码片段**（行1085）:
```cpp
1075 |     {
1076 |         int32_t offset = n_tasks - task_id;
1077 |         if (offset <= 0 || (offset & (offset - 1)) != 0) {
1078 |             return false;
1079 |         } else {
1080 |             return true;
1081 |         }
1082 |     }
1083 | 
1084 |     CATLASS_DEVICE
1085 |     void CombineSetFlag()
1086 |     {
1087 |         if constexpr (std::is_same_v<ElementB, int8_t>) {
1088 |             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
1089 |             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
1090 |             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
1091 |             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
1092 |             AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
1093 |             AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
1094 |             AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
```

---

#### [6] 人工检视意见

- **提出人**: liuboxi
- **作者**: ranczhw
- **文件**: mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp
- **行号**: 1786
- **评论时间**: 2026-06-08
- **Commit**: d9c273b6874a
- **问题描述**:

  > 这里硬编码16，为魔鬼数字，无法看出与其他地方的关联，规格扩大等情况修改也容易遗漏。

- **代码片段**（行1786）:
```cpp
1776 |     Arch::Resource<ArchTag> resource;
1777 | 
1778 |     uint32_t coreIdx;
1779 |     uint32_t coreNum;
1780 |     uint32_t nSyncSwiglu;
1781 | 
1782 |     Params params;
1783 |     WorkspaceInfo workspaceInfo;
1784 |     PeermemInfo peermemInfo;
1785 | 
1786 |     uint32_t dequantSum[16] = {0};
1787 | 
1788 |     // ========== Common tensors (all types) ==========
1789 |     AscendC::GlobalTensor<ElementC> gmC;
1790 |     AscendC::GlobalTensor<ElementC> gmC2;
1791 |     AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;
1792 |     AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;
1793 |     AscendC::GlobalTensor<bool> gmXActiveMask;
1794 |     AscendC::GlobalTensor<int32_t> tokenPerExpert;
1795 |     AscendC::GlobalTensor<int32_t> cumsumMM;
```

---

#### [7] 人工检视意见

- **提出人**: liuboxi
- **作者**: ranczhw
- **文件**: mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp
- **行号**: 1718
- **评论时间**: 2026-06-08
- **Commit**: d9c273b6874a
- **问题描述**:

  > 存在较多开发过程中的调试宏，W4A8_DEBUG、SOFT_SYNC、ENABLE_TIMER确定必要性。

- **代码片段**（行1718）:
```cpp
1708 |                 ptrC2 = params.ptrWorkspace + workspaceOffset;
1709 |                 workspaceOffset += params.maxOutputSize * n2 * sizeof(ElementC);
1710 |             }
1711 | 
1712 |             if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
1713 |                 ptrA1Int4 = params.ptrWorkspace + workspaceOffset;
1714 |                 workspaceOffset += params.maxOutputSize * params.problemShape.k();
1715 |                 ptrA2Int4 = params.ptrWorkspace + workspaceOffset;
1716 |                 workspaceOffset += params.maxOutputSize * k2;
1717 |                 ptrCGMM1 = params.ptrWorkspace + workspaceOffset;
1718 | #ifdef W4A8_DEBUG
1719 |                 workspaceOffset += params.maxOutputSize * params.problemShape.n() * sizeof(float);
1720 | #endif
1721 |                 ptrCGMM2 = params.ptrWorkspace + workspaceOffset;
1722 | #ifdef W4A8_DEBUG
1723 |                 workspaceOffset += params.maxOutputSize * n2 * sizeof(float);
1724 | #endif
1725 |             } else if constexpr (std::is_same_v<ElementA, int8_t>) {
1726 |                 ptrA = params.ptrWorkspace + workspaceOffset;
1727 |                 ptrPermutedToken = ptrA;
```

---


### 文件: mc2/mega_moe/op_kernel/arch22/template_linear_algebra_v2/gemm/tile/cast_fp8_to_fp16.hpp（Kernel侧）

#### [8] 人工检视意见

- **提出人**: liuboxi
- **作者**: ranczhw
- **文件**: mc2/mega_moe/op_kernel/arch22/template_linear_algebra_v2/gemm/tile/cast_fp8_to_fp16.hpp
- **行号**: 41
- **评论时间**: 2026-06-07
- **Commit**: d9c273b6874a
- **问题描述**:

  > 无效代码需要删除，保持代码简洁。

- **代码片段**（行41）:
```cpp
  31 |     using ElementDst = typename DstType_::Element;
  32 |     using LayoutSrc = typename SrcType_::Layout;
  33 |     using LayoutDst = typename DstType_::Layout;
  34 |     using LayoutRowMajor = Catlass::layout::RowMajor;
  35 | 
  36 |     static_assert(std::is_same_v<LayoutSrc, layout::RowMajor> || std::is_same_v<LayoutSrc, layout::ColumnMajor>,
  37 |         "Unsupported layout, only can be Row/Column Major.");
  38 |     static_assert(std::is_same_v<LayoutDst, LayoutSrc>, "layout src and dst must be the same.");
  39 | 
  40 |     static const uint32_t Alignment = 256;
  41 |     // static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(int8_t);
  42 | 
  43 |     struct Params {
  44 |         half scalar;
  45 |         half zeroPoint;
  46 | 
  47 |         CATLASS_HOST_DEVICE
  48 |         Params() = default;
  49 | 
  50 |         CATLASS_HOST_DEVICE
```
---

## 被检视代码

> 本报告基于 PR 6534 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `mc2/mega_moe/op_kernel/arch22/mega_moe.h`
- `mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp`
- `mc2/mega_moe/op_kernel/arch22/mega_moe_tiling_a2a3.h`
- `mc2/mega_moe/op_kernel/arch22/template_linear_algebra_v2/gemm/tile/cast_fp8_to_fp16.hpp`
- `mc2/mega_moe/op_kernel/arch22/utils/hccl_shmem.hpp`
