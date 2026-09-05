# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 4099
- **PR作者**: t00620168
- **代码文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_infershape.cpp, moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **代码侧别**: Tiling侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 16 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| FAIL（发现问题） | 16 | 100% |

---

## 发现问题

### 文件: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_infershape.cpp（Tiling侧）

#### [1] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_infershape.cpp
- **行号**: 745
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > MXQUANT_FP4_E2M1 模式下的输入 dtype 校验被注释掉了，意味着任何类型的输入都不会被拦截。其他量化模式（static、dynamic、mxfp8 等）都有严格的 dtype 检查，这里应该也加上。如果是暂时跳过，建议加 TODO 注释说明原因和预计补上的时间。
  > 无效注释代码应删除

- **代码片段**（行745）:
```cpp
 735 |         || QuantMode::HIF8_CAST == quantMode || QuantMode::HIF8_PERTOKEN == quantMode
 736 |         || QuantMode::HIF8_PERTENSOR == quantMode) {
 737 |         if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16) {
 738 |             OP_LOGE(
 739 |                 context,
 740 |                 "When quant_mode=%ld, xDtype should be DT_FLOAT16 or DT_BF16. Current got unexpected dtype id of %d.",
 741 |                 quantMode, xDtype);
 742 |             return ge::GRAPH_FAILED;
 743 |         }
 744 |     } else if (QuantMode::MXQUANT_FP4_E2M1 == quantMode) {
 745 |         // if (xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16) {
 746 |         //     OP_LOGE(context, 
 747 |         //     "When quant_mode=%ld, xDtype should be DT_FLOAT, DT_FLOAT16 or DT_BF16. Current got unexpected dtype id of %d.",
 748 |         //     quantMode, xDtype);
 749 |         //     return ge::GRAPH_FAILED;
 750 |         // }
 751 |     }
 752 | 
 753 |     if (QuantMode::STATIC_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode) {
 754 |         expandedXDtype = ge::DT_INT8;
```

---

#### [3] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_infershape.cpp
- **行号**: 256
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 错误信息说 "should be in [%d, %d], %d, %d, %d or %d" 但实际传入了 NON_QUANT, NON_QUANT 两次（第一个参数和第二个参数都是 NON_QUANT），同时 quant_mode 值 4 和 5 不在有效范围内但也没在错误信息中体现。建议改成列出所有有效值，或者直接用 validQuantModes 集合来生成错误信息，避免手动维护两份列表不一致。

- **代码片段**（行256）:
```cpp
 246 |     if (nullptr == attrs) {
 247 |         OP_LOGE(context, "The RuntimeAttrs for quant_mode is none.");
 248 |         return ge::GRAPH_FAILED;
 249 |     }
 250 |     const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_QUANT_MODE);
 251 |     if (nullptr == quantModePtr) {
 252 |         OP_LOGE(context, "The quant_mode should not be null.");
 253 |         return ge::GRAPH_FAILED;
 254 |     }
 255 |     quantMode = *quantModePtr;
 256 |     if (validQuantModes.count(quantMode) == 0) {
 257 |         OP_LOGE(context, "The quant_mode should be in [%d, %d], %d, %d, %d or %d. But it is %d.", QuantMode::NON_QUANT,
 258 |                 QuantMode::NON_QUANT, QuantMode::MXQUANT_FP8_E4M3FN, QuantMode::HIF8_CAST, QuantMode::HIF8_PERTENSOR,
 259 |                 QuantMode::HIF8_PERTOKEN, QuantMode::MXQUANT_FP4_E2M1, quantMode);
 260 |         return ge::GRAPH_FAILED;
 261 |     }
 262 |     OP_LOGD(context, "End to do GetAndCheckQuantMode.");
 263 |     return ge::GRAPH_SUCCESS;
 264 | }
 265 | 
```
---

#### [7] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_infershape.cpp
- **行号**: 836
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > InferShapeRange4MoeInitRoutingV3 中 MXQUANT_FP4_E2M1 分支的 expanded_scale shape range 设置代码被注释掉了，导致该模式下 expanded_scale 的 shape range 未被正确设置。这可能导致动态 shape 场景下推导异常。

- **代码片段**（行836）:
```cpp
 826 |         count->GetMin()->SetDim(0, 0);
 827 |         count->GetMax()->SetDim(0, -1);
 828 |     }
 829 | 
 830 |     if (expanded_scale->GetMin() != nullptr && expanded_scale->GetMax() != nullptr) {
 831 |         const auto *attrsPtr = context->GetAttrs();
 832 |         OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
 833 |         const int64_t *quantModePtr = attrsPtr->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_QUANT_MODE);
 834 |         OP_CHECK_NULL_WITH_CONTEXT(context, quantModePtr);
 835 |         int64_t quantMode = *quantModePtr;
 836 |         if (quantMode == QuantMode::MXQUANT_FP4_E2M1) {
 837 |             // expanded_scale->GetMin()->SetDimNum(DIM_THREE);
 838 |             // expanded_scale->GetMax()->SetDimNum(DIM_THREE);
 839 |             // for (size_t i = 0; i < DIM_THREE; i++) {
 840 |             //     expanded_scale->GetMin()->SetDim(i, 0);
 841 |             //     expanded_scale->GetMax()->SetDim(i, -1);
 842 |             // }
 843 |         } else if (quantMode == QuantMode::MXQUANT_FP8_E5M2 || quantMode == QuantMode::MXQUANT_FP8_E4M3FN) {
 844 |             expanded_scale->GetMin()->SetDimNum(DIM_TWO);
 845 |             expanded_scale->GetMax()->SetDimNum(DIM_TWO);
```

---

### 文件: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp（Tiling侧）

#### [8] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 674
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 这里用了单等号 = 而不是双等号 ==，是赋值而不是比较。这会导致两个严重问题：1) 条件判断永远为 true（因为 QUANT_MODE_MXFP4_E2M1 = 9 非零），所有 quantMode 都会走进这个分支；2) quantMode_ 的值被覆盖为 9，后续所有依赖 quantMode_ 的逻辑都会出错。必须改成 quantMode_ == QUANT_MODE_MXFP4_E2M1。

- **代码片段**（行674）:
```cpp
 664 |                                                                            DataType::DT_BF16, DataType::DT_INT8};
 665 |     static const std::unordered_set<DataType> MX_OR_HIF8_QUANT_SUPPORTED_DTYPES = {ge::DataType::DT_FLOAT16,
 666 |                                                                                    ge::DataType::DT_BF16};
 667 |     unordered_set<DataType> supportedDtypes;
 668 |     if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN ||
 669 |         quantMode_ == QUANT_MODE_HIF8_CAST || quantMode_ == QUANT_MODE_HIF8_PERTENSOR ||
 670 |         quantMode_ == QUANT_MODE_HIF8_PERTOKEN) {
 671 |         supportedDtypes = MX_OR_HIF8_QUANT_SUPPORTED_DTYPES;
 672 |     } else if (quantMode_ == QUANT_MODE_UNQUANT) {
 673 |         supportedDtypes = UNQUANT_SUPPORTED_DTYPES;
 674 |     } else if (quantMode_ = QUANT_MODE_MXFP4_E2M1) {
 675 |         supportedDtypes = MXFP4QUANT_SUPPORTED_DTYPES;
 676 |     } else if (quantMode_ == QUANT_MODE_STATIC) {
 677 |         supportedDtypes = STATIC_QUANT_SUPPORTED_DTYPES;
 678 |     } else {
 679 |         //! 出于历史调用的兼容性，这里不拦截quant_mode=1（动态量化）下输入x为int8类型，仅资料说明此时算子输出expandedX、expandedScale无意义
 680 |         supportedDtypes = DYNAMIC_QUANT_SUPPORTED_DTYPES;
 681 |     }
 682 |     OP_CHECK_IF(supportedDtypes.count(xDtype_) == 0,
 683 |                 OP_LOGE(context_, "Unsupported dtype of input x: %d under quant_mode: %ld.", xDtype_, quantMode_),
```

---

#### [9] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 420
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 这个条件里 QUANT_MODE_MXFP4_E2M1 前面少了 quantMode_ ==，现在 QUANT_MODE_MXFP4_E2M1 是一个裸的常量表达式，值为 9（非零），所以这个 if 分支对所有 quantMode 都会为 true。应该改为 quantMode_ == QUANT_MODE_MXFP4_E2M1。

- **代码片段**（行420）:
```cpp
 410 |         Tiling4GatherOutCompute();
 411 |     }
 412 |     return ge::GRAPH_SUCCESS;
 413 | }
 414 | 
 415 | uint64_t MoeInitRoutingV3Arch35TilingClass::GetTilingKey() const
 416 | {
 417 |     OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetTilingKey()");
 418 | 
 419 |     int64_t quantModeFactor = quantMode_ + 1;
 420 |     if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN || QUANT_MODE_MXFP4_E2M1) {
 421 |         // 对于MXFP8量化，两种模式在TilingKey体现的QuantMode都为3。
 422 |         // 其余非量化为0，静态量化为1，动态量化为2，即都是quantMode_+1
 423 |         // 可以用与最低的UNQUANT的数值的差值来作为quantModeFactor，这里值就为3
 424 |         quantModeFactor = QUANT_MODE_MXFP8_E5M2 - QUANT_MODE_UNQUANT;
 425 |         // quantMode >= 9时，都可以走这个TilingKey的计算
 426 |         if (quantMode_ >= QUANT_MODE_MXFP4_E2M1) {
 427 |             return static_cast<uint64_t>(QUANT_MODE_MXFP4_E2M1 * TILINGKEY_BASE + sortMode_ * SORT_CORE_TILINGKEY_BASE +
 428 |                                  QUANT_MODE_TILINGKEY_BASE + rowIdxType_ * DROP_MODE_TILINGKEY_BASE);
 429 |         }
```

---

#### [10] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 987
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 错误信息字符串里有两个连续的逗号 "quant_mode %ld,, current"，应该去掉一个。

- **代码片段**（行987）:
```cpp
 977 |                             "The dim1 of output expanded_scale should be %ld under "
 978 |                             "quant_mode %ld,, current is %ld.",
 979 |                             expectedDim1, quantMode_, dim1),
 980 |                     return ge::GRAPH_FAILED);
 981 |     }
 982 |     if (expectedDim2 != -1) {
 983 |         int64_t dim2 = expandedScaleShape_.GetDim(2);
 984 |         OP_CHECK_IF(dim2 != expectedDim2,
 985 |                     OP_LOGE(context_,
 986 |                             "The dim2 of output expanded_scale should be %ld under "
 987 |                             "quant_mode %ld,, current is %ld.",
 988 |                             expectedDim2, quantMode_, dim2),
 989 |                     return ge::GRAPH_FAILED);
 990 |     }
 991 |     return ge::GRAPH_SUCCESS;
 992 | }
 993 | 
 994 | ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckOutputs()
 995 | {
 996 |     OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckOutputs()");
```

---

#### [11] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 674
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 这里用了 `=` 赋值而不是 `==` 比较。`quantMode_ = QUANT_MODE_MXFP4_E2M1` 会把 quantMode_ 的值改成 9，然后表达式永远为 true（非零），导致所有后续的 else if 分支都不会走到。这意味着不管原来的 quantMode_ 是什么，执行完这行后 quantMode_ 就变成了 MXFP4，supportedDtypes 也会被错误地设置为 MXFP4QUANT_SUPPORTED_DTYPES。这是个严重 bug，必须改成 `==`。

- **代码片段**（行673）:
```cpp
 663 |     static const unordered_set<DataType> DYNAMIC_QUANT_SUPPORTED_DTYPES = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
 664 |                                                                            DataType::DT_BF16, DataType::DT_INT8};
 665 |     static const std::unordered_set<DataType> MX_OR_HIF8_QUANT_SUPPORTED_DTYPES = {ge::DataType::DT_FLOAT16,
 666 |                                                                                    ge::DataType::DT_BF16};
 667 |     unordered_set<DataType> supportedDtypes;
 668 |     if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN ||
 669 |         quantMode_ == QUANT_MODE_HIF8_CAST || quantMode_ == QUANT_MODE_HIF8_PERTENSOR ||
 670 |         quantMode_ == QUANT_MODE_HIF8_PERTOKEN) {
 671 |         supportedDtypes = MX_OR_HIF8_QUANT_SUPPORTED_DTYPES;
 672 |     } else if (quantMode_ == QUANT_MODE_UNQUANT) {
 673 |         supportedDtypes = UNQUANT_SUPPORTED_DTYPES;
 674 |     } else if (quantMode_ = QUANT_MODE_MXFP4_E2M1) {
 675 |         supportedDtypes = MXFP4QUANT_SUPPORTED_DTYPES;
 676 |     } else if (quantMode_ == QUANT_MODE_STATIC) {
 677 |         supportedDtypes = STATIC_QUANT_SUPPORTED_DTYPES;
 678 |     } else {
 679 |         //! 出于历史调用的兼容性，这里不拦截quant_mode=1（动态量化）下输入x为int8类型，仅资料说明此时算子输出expandedX、expandedScale无意义
 680 |         supportedDtypes = DYNAMIC_QUANT_SUPPORTED_DTYPES;
 681 |     }
 682 |     OP_CHECK_IF(supportedDtypes.count(xDtype_) == 0,
```

---

#### [12] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 420
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 条件里 `QUANT_MODE_MXFP4_E2M1` 前面少写了 `quantMode_ ==`。现在这个表达式是 `quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN || QUANT_MODE_MXFP4_E2M1`，最后一个 `QUANT_MODE_MXFP4_E2M1` 的值是 9，非零永远为 true，所以这个 if 条件对任何 quantMode_ 值都会进入。结果就是所有量化模式的 TilingKey 都会按 MXFP8 的逻辑去算，这会导致错误的 kernel 调度。需要改成 `quantMode_ == QUANT_MODE_MXFP4_E2M1`。

- **代码片段**（行420）:
```cpp
 410 |         Tiling4GatherOutCompute();
 411 |     }
 412 |     return ge::GRAPH_SUCCESS;
 413 | }
 414 | 
 415 | uint64_t MoeInitRoutingV3Arch35TilingClass::GetTilingKey() const
 416 | {
 417 |     OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetTilingKey()");
 418 | 
 419 |     int64_t quantModeFactor = quantMode_ + 1;
 420 |     if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN || QUANT_MODE_MXFP4_E2M1) {
 421 |         // 对于MXFP8量化，两种模式在TilingKey体现的QuantMode都为3。
 422 |         // 其余非量化为0，静态量化为1，动态量化为2，即都是quantMode_+1
 423 |         // 可以用与最低的UNQUANT的数值的差值来作为quantModeFactor，这里值就为3
 424 |         quantModeFactor = QUANT_MODE_MXFP8_E5M2 - QUANT_MODE_UNQUANT;
 425 |         // quantMode >= 9时，都可以走这个TilingKey的计算
 426 |         if (quantMode_ >= QUANT_MODE_MXFP4_E2M1) {
 427 |             return static_cast<uint64_t>(QUANT_MODE_MXFP4_E2M1 * TILINGKEY_BASE + sortMode_ * SORT_CORE_TILINGKEY_BASE +
 428 |                                  QUANT_MODE_TILINGKEY_BASE + rowIdxType_ * DROP_MODE_TILINGKEY_BASE);
 429 |         }
```

---

#### [13] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 985
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 错误信息里有两个逗号 `quant_mode %ld,, current is %ld`，多了一个逗号，应该是 `quant_mode %ld, current is %ld`。

- **代码片段**（行985）:
```cpp
 975 |         OP_CHECK_IF(dim1 != expectedDim1,
 976 |                     OP_LOGE(context_,
 977 |                             "The dim1 of output expanded_scale should be %ld under "
 978 |                             "quant_mode %ld,, current is %ld.",
 979 |                             expectedDim1, quantMode_, dim1),
 980 |                     return ge::GRAPH_FAILED);
 981 |     }
 982 |     if (expectedDim2 != -1) {
 983 |         int64_t dim2 = expandedScaleShape_.GetDim(2);
 984 |         OP_CHECK_IF(dim2 != expectedDim2,
 985 |                     OP_LOGE(context_,
 986 |                             "The dim2 of output expanded_scale should be %ld under "
 987 |                             "quant_mode %ld,, current is %ld.",
 988 |                             expectedDim2, quantMode_, dim2),
 989 |                     return ge::GRAPH_FAILED);
 990 |     }
 991 |     return ge::GRAPH_SUCCESS;
 992 | }
 993 | 
 994 | ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckOutputs()
```

---

#### [14] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 987
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 错误信息字符串里有两个连续的逗号 `quant_mode %ld,, current is %ld`，多了一个逗号。同样的问题在上面 dim1 的校验错误信息（第 978 行）里也有，建议一并修复。

- **代码片段**（行987）:
```cpp
 977 |                             "The dim1 of output expanded_scale should be %ld under "
 978 |                             "quant_mode %ld,, current is %ld.",
 979 |                             expectedDim1, quantMode_, dim1),
 980 |                     return ge::GRAPH_FAILED);
 981 |     }
 982 |     if (expectedDim2 != -1) {
 983 |         int64_t dim2 = expandedScaleShape_.GetDim(2);
 984 |         OP_CHECK_IF(dim2 != expectedDim2,
 985 |                     OP_LOGE(context_,
 986 |                             "The dim2 of output expanded_scale should be %ld under "
 987 |                             "quant_mode %ld,, current is %ld.",
 988 |                             expectedDim2, quantMode_, dim2),
 989 |                     return ge::GRAPH_FAILED);
 990 |     }
 991 |     return ge::GRAPH_SUCCESS;
 992 | }
 993 | 
 994 | ge::graphStatus MoeInitRoutingV3Arch35TilingClass::CheckOutputs()
 995 | {
 996 |     OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::CheckOutputs()");
```

---

#### [15] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 674
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 严重bug：`else if (quantMode_ = QUANT_MODE_MXFP4_E2M1)` 使用了赋值运算符 `=` 而非比较运算符 `==`。这会导致 quantMode_ 被无条件修改为 QUANT_MODE_MXFP4_E2M1（值为9），且条件始终为 true，后续所有 else if 分支永远不会执行。应改为 `quantMode_ == QUANT_MODE_MXFP4_E2M1`。

- **代码片段**（行674）:
```cpp
 664 |                                                                            DataType::DT_BF16, DataType::DT_INT8};
 665 |     static const std::unordered_set<DataType> MX_OR_HIF8_QUANT_SUPPORTED_DTYPES = {ge::DataType::DT_FLOAT16,
 666 |                                                                                    ge::DataType::DT_BF16};
 667 |     unordered_set<DataType> supportedDtypes;
 668 |     if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN ||
 669 |         quantMode_ == QUANT_MODE_HIF8_CAST || quantMode_ == QUANT_MODE_HIF8_PERTENSOR ||
 670 |         quantMode_ == QUANT_MODE_HIF8_PERTOKEN) {
 671 |         supportedDtypes = MX_OR_HIF8_QUANT_SUPPORTED_DTYPES;
 672 |     } else if (quantMode_ == QUANT_MODE_UNQUANT) {
 673 |         supportedDtypes = UNQUANT_SUPPORTED_DTYPES;
 674 |     } else if (quantMode_ = QUANT_MODE_MXFP4_E2M1) {
 675 |         supportedDtypes = MXFP4QUANT_SUPPORTED_DTYPES;
 676 |     } else if (quantMode_ == QUANT_MODE_STATIC) {
 677 |         supportedDtypes = STATIC_QUANT_SUPPORTED_DTYPES;
 678 |     } else {
 679 |         //! 出于历史调用的兼容性，这里不拦截quant_mode=1（动态量化）下输入x为int8类型，仅资料说明此时算子输出expandedX、expandedScale无意义
 680 |         supportedDtypes = DYNAMIC_QUANT_SUPPORTED_DTYPES;
 681 |     }
 682 |     OP_CHECK_IF(supportedDtypes.count(xDtype_) == 0,
 683 |                 OP_LOGE(context_, "Unsupported dtype of input x: %d under quant_mode: %ld.", xDtype_, quantMode_),
```

---

#### [16] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 420
- **评论时间**: 2026-04-14
- **Commit**: b68e9745d63a
- **问题描述**:

  > 严重bug：条件判断 `quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN || QUANT_MODE_MXFP4_E2M1` 中第三个条件缺少与 quantMode_ 的比较，QUANT_MODE_MXFP4_E2M1 作为非零常量（值为9）恒为 true，导致该 if 分支对所有 quantMode 值都会命中。应改为 `quantMode_ == QUANT_MODE_MXFP4_E2M1`。

- **代码片段**（行420）:
```cpp
 410 |         Tiling4GatherOutCompute();
 411 |     }
 412 |     return ge::GRAPH_SUCCESS;
 413 | }
 414 | 
 415 | uint64_t MoeInitRoutingV3Arch35TilingClass::GetTilingKey() const
 416 | {
 417 |     OP_LOGD(context_, "Entered MoeInitRoutingV3Arch35TilingClass::GetTilingKey()");
 418 | 
 419 |     int64_t quantModeFactor = quantMode_ + 1;
 420 |     if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN || QUANT_MODE_MXFP4_E2M1) {
 421 |         // 对于MXFP8量化，两种模式在TilingKey体现的QuantMode都为3。
 422 |         // 其余非量化为0，静态量化为1，动态量化为2，即都是quantMode_+1
 423 |         // 可以用与最低的UNQUANT的数值的差值来作为quantModeFactor，这里值就为3
 424 |         quantModeFactor = QUANT_MODE_MXFP8_E5M2 - QUANT_MODE_UNQUANT;
 425 |         // quantMode >= 9时，都可以走这个TilingKey的计算
 426 |         if (quantMode_ >= QUANT_MODE_MXFP4_E2M1) {
 427 |             return static_cast<uint64_t>(QUANT_MODE_MXFP4_E2M1 * TILINGKEY_BASE + sortMode_ * SORT_CORE_TILINGKEY_BASE +
 428 |                                  QUANT_MODE_TILINGKEY_BASE + rowIdxType_ * DROP_MODE_TILINGKEY_BASE);
 429 |         }
```

---

## 被检视代码

> 本报告基于 PR 4099 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `moe/moe_init_routing_v3/op_host/moe_init_routing_v3_infershape.cpp`
- `moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp`
