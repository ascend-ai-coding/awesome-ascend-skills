# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 1193
- **PR作者**: xuxiongjie
- **代码文件**: 4 个文件
- **代码侧别**: Tiling侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 16 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 16 | 7 | 100% |

---

## 发现问题


### 文件: gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp（Tiling侧）

#### [2] 人工检视意见

- **提出人**: dai-zhentao
- **作者**: xuxiongjie
- **文件**: gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp
- **行号**: 1944
- **评论时间**: 2026-02-12
- **Commit**: 5469207233ec
- **问题描述**:

  > (*params.scaleOptional)[0]判空校验

- **代码片段**（行1944）:
```cpp
1934 |     return false;
1935 | }
1936 | 
1937 | static void SetTransposedTensorListContiguous(gmm::GroupedMatmulParams &params, aclOpExecutor *executorPtr)
1938 | {
1939 |   bool isPerTileQuantMode = IsPerTileQuantMode(params);
1940 |   DataType weightDtype = (*params.weight)[0]->GetDataType();
1941 |   if (params.transposeX) {
1942 |     std::vector<aclTensor*> xTensorList;
1943 |     gmm::CreateContiguousTensorList(params.x, xTensorList, executorPtr);
1944 |     params.x = executorPtr->AllocTensorList(xTensorList.data(), xTensorList.size());
1945 |     if (params.perTokenScaleOptional != nullptr &&
1946 |         ((*params.perTokenScaleOptional)[0]->GetDataType() == DataType::DT_FLOAT8_E8M0 || isPerTileQuantMode)) {
1947 |       std::vector<aclTensor*> perTokenScaleTensorList;
1948 |       if (isPerTileQuantMode) {
1949 |         gmm::CreateContiguousTensorList(params.perTokenScaleOptional, perTokenScaleTensorList, executorPtr);
1950 |       } else {
1951 |         gmm::CreateContiguousTensorListForPertoken(params.perTokenScaleOptional, perTokenScaleTensorList, executorPtr);}
1952 |       params.perTokenScaleOptional = executorPtr->AllocTensorList(perTokenScaleTensorList.data(), perTokenScaleTensorList.size());
1953 |     }
```

---

#### [3] 人工检视意见

- **提出人**: dai-zhentao
- **作者**: xuxiongjie
- **文件**: gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp
- **行号**: 1953
- **评论时间**: 2026-02-12
- **Commit**: 5469207233ec
- **问题描述**:

  > (*params.antiquantScaleOptional)[0]空指针校验

- **代码片段**（行1953）:
```cpp
1943 |     gmm::CreateContiguousTensorList(params.x, xTensorList, executorPtr);
1944 |     params.x = executorPtr->AllocTensorList(xTensorList.data(), xTensorList.size());
1945 |     if (params.perTokenScaleOptional != nullptr &&
1946 |         ((*params.perTokenScaleOptional)[0]->GetDataType() == DataType::DT_FLOAT8_E8M0 || isPerTileQuantMode)) {
1947 |       std::vector<aclTensor*> perTokenScaleTensorList;
1948 |       if (isPerTileQuantMode) {
1949 |         gmm::CreateContiguousTensorList(params.perTokenScaleOptional, perTokenScaleTensorList, executorPtr);
1950 |       } else {
1951 |         gmm::CreateContiguousTensorListForPertoken(params.perTokenScaleOptional, perTokenScaleTensorList, executorPtr);}
1952 |       params.perTokenScaleOptional = executorPtr->AllocTensorList(perTokenScaleTensorList.data(), perTokenScaleTensorList.size());
1953 |     }
1954 |   }
1955 |   if (params.transposeWeight) {
1956 |     std::vector<aclTensor *> weightTensorList;
1957 |     auto nZShape = (*params.weight)[0]->GetStorageShape();
1958 |     gmm::CreateContiguousTensorList(params.weight, weightTensorList, executorPtr);
1959 |     params.weight = executorPtr->AllocTensorList(weightTensorList.data(), weightTensorList.size());
1960 |     if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
1961 |         ((IsQuant(params.xDtype, weightDtype) &&
1962 |           (*params.weight)[0]->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) ||
```

---


### 文件: gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp（Tiling侧）

#### [4] 人工检视意见

- **提出人**: chaoying-zhang
- **作者**: xuxiongjie
- **文件**: gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp
- **行号**: 801
- **评论时间**: 2026-02-12
- **Commit**: 5469207233ec
- **问题描述**:

  > 不要使用GetInputTensor获取Shape和Dtype，使用GetDynamicInputShape获取，且如果该调用在GetNumOfInputs之后可以不用重复判空

- **代码片段**（行801）:
```cpp
 791 |                                 "Data type [%s] is not supported for bias, when xDtype is [%s].",
 792 |                                 ge::TypeUtils::DataTypeToSerialString(biasDtype_).c_str(),
 793 |                                 ge::TypeUtils::DataTypeToSerialString(xDType_).c_str()),
 794 |                         return false);
 795 |         }
 796 |     }
 797 |     return true;
 798 | }
 799 | 
 800 | bool GroupedWeightQuantBatchMatmulTiling::CheckGroupTypeAndSplitItem(const gert::TilingContext *context) const
 801 | {
 802 |     if (IsA16W4ND()) {
 803 |         OP_CHECK_IF(
 804 |             (groupType_ != GroupType::NO_SPLIT) && (groupType_ != GroupType::SPLIT_M),
 805 |             OP_LOGE(context->GetNodeName(), "when x-weight is bf16/fp16-int32/int4, grouptype only supports -1 or 0."),
 806 |             return false);
 807 |         if (groupType_ == GroupType::NO_SPLIT) {
 808 |             OP_CHECK_IF((splitItem_ != 0 && splitItem_ != 1),
 809 |                         OP_LOGE(context->GetNodeName(), "When grouptype is -1. splititem can only be 0 or 1."),
 810 |                         return false);
```

---

#### [5] 人工检视意见

- **提出人**: chaoying-zhang
- **作者**: xuxiongjie
- **文件**: gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp
- **行号**: 812
- **评论时间**: 2026-02-12
- **Commit**: 5469207233ec
- **问题描述**:

  > 不要使用GetInputTensor获取Shape和Dtype，使用GetDynamicInputShape获取，如果前面没有判断过PerTokenScale的shape非空，需要判空

- **代码片段**（行812）:
```cpp
 802 |     if (IsA16W4ND()) {
 803 |         OP_CHECK_IF(
 804 |             (groupType_ != GroupType::NO_SPLIT) && (groupType_ != GroupType::SPLIT_M),
 805 |             OP_LOGE(context->GetNodeName(), "when x-weight is bf16/fp16-int32/int4, grouptype only supports -1 or 0."),
 806 |             return false);
 807 |         if (groupType_ == GroupType::NO_SPLIT) {
 808 |             OP_CHECK_IF((splitItem_ != 0 && splitItem_ != 1),
 809 |                         OP_LOGE(context->GetNodeName(), "When grouptype is -1. splititem can only be 0 or 1."),
 810 |                         return false);
 811 |         } else {
 812 |             OP_CHECK_IF((splitItem_ != 2 && splitItem_ != 3),
 813 |                         OP_LOGE(context->GetNodeName(), "When grouptype is 0. splititem can only be 2 or 3."),
 814 |                         return false);
 815 |         }
 816 |     }
 817 |     return true;
 818 | }
 819 | 
 820 | bool GroupedWeightQuantBatchMatmulTiling::CheckTransposeStatus(const gert::TilingContext *context) const
 821 | {
```

---

#### [6] 人工检视意见

- **提出人**: chaoying-zhang
- **作者**: xuxiongjie
- **文件**: gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp
- **行号**: 823
- **评论时间**: 2026-02-12
- **Commit**: 5469207233ec
- **问题描述**:

  > 这里的校验现在在SetShapeListMultiXMultiWeightMultiY之前，此时groupNum_、nSize_、kSize_都还没被正确赋值，会被错误拦截。建议合并进CheckEveryTensor的CheckTensorDim和CheckTensorShape(context, ANTIQUANT_SCALE_IDX, i, "antiquantScale")的逻辑里，保证整体性

- **代码片段**（行823）:
```cpp
 813 |                         OP_LOGE(context->GetNodeName(), "When grouptype is 0. splititem can only be 2 or 3."),
 814 |                         return false);
 815 |         }
 816 |     }
 817 |     return true;
 818 | }
 819 | 
 820 | bool GroupedWeightQuantBatchMatmulTiling::CheckTransposeStatus(const gert::TilingContext *context) const
 821 | {
 822 |     OP_CHECK_IF(transA_, OP_LOGE(context->GetNodeName(), "Transposed A is not supported. "), return false);
 823 |     return true;
 824 | }
 825 | 
 826 | bool GroupedWeightQuantBatchMatmulTiling::CheckEmptyTensor(const gert::TilingContext *context)
 827 | {
 828 |     // all M or N be zero, get true
 829 |     bool zeroM = true;
 830 |     bool zeroN = true;
 831 |     // exist one K be zero, get true
 832 |     bool zeroK = false;
```

---

#### [7] 人工检视意见

- **提出人**: chaoying-zhang
- **作者**: xuxiongjie
- **文件**: gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp
- **行号**: 822
- **评论时间**: 2026-02-12
- **Commit**: 5469207233ec
- **问题描述**:

  > 这里的成员变量也会有还没被正确赋值的情况，需要整改。这个函数可以暂时单写，不用合并进CheckEveryTensor的逻辑里

- **代码片段**（行822）:
```cpp
 812 |             OP_CHECK_IF((splitItem_ != 2 && splitItem_ != 3),
 813 |                         OP_LOGE(context->GetNodeName(), "When grouptype is 0. splititem can only be 2 or 3."),
 814 |                         return false);
 815 |         }
 816 |     }
 817 |     return true;
 818 | }
 819 | 
 820 | bool GroupedWeightQuantBatchMatmulTiling::CheckTransposeStatus(const gert::TilingContext *context) const
 821 | {
 822 |     OP_CHECK_IF(transA_, OP_LOGE(context->GetNodeName(), "Transposed A is not supported. "), return false);
 823 |     return true;
 824 | }
 825 | 
 826 | bool GroupedWeightQuantBatchMatmulTiling::CheckEmptyTensor(const gert::TilingContext *context)
 827 | {
 828 |     // all M or N be zero, get true
 829 |     bool zeroM = true;
 830 |     bool zeroN = true;
 831 |     // exist one K be zero, get true
```

---

## 被检视代码

> 本报告基于 PR 1193 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `gmm/grouped_matmul/op_host/grouped_matmul_infershape_weight_quant_checker.cpp`
- `gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp`
- `gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul_weight_quant_950_checker.cpp`
- `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp`
