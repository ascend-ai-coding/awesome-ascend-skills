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
| 16 | 12 | 100% |

---

## 发现问题

### 文件: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp（Tiling侧）

#### [7] 人工检视意见

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

#### [8] 人工检视意见

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


#### [9] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: t00620168
- **文件**: moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp
- **行号**: 673
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

#### [10] 人工检视意见

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


#### [11] 人工检视意见

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

#### [12] 人工检视意见

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
