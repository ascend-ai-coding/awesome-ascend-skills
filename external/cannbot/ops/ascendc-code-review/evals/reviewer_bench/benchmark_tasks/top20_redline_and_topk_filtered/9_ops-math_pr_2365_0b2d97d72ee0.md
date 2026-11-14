# 代码检视报告

## 检视概览
- **仓库**: ops-math
- **PR编号**: 2365
- **PR作者**: wang-shilong32
- **代码文件**: 5 个文件
- **代码侧别**: Tiling侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 12 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 12 | 9 | 100% |

---

## 发现问题

### 文件: random/tile_with_axis/op_host/arch35/tile_with_axis_tiling.cpp（Tiling侧）

#### [1] 人工检视意见

- **提出人**: zhangzijie
- **作者**: wang-shilong32
- **文件**: random/tile_with_axis/op_host/arch35/tile_with_axis_tiling.cpp
- **行号**: 77
- **评论时间**: 2026-04-23
- **Commit**: 0b2d97d72ee0
- **问题描述**:

  > GetAttrPointer 返回值未检查就直接解引用，如果属性不存在返回 nullptr，会导致程序崩溃。应该在解引用前检查返回值是否为 nullptr。

- **代码片段**（行77）:
```cpp
  67 |     uint64_t ubSize = 0;
  68 |     int64_t coreNum = 0;
  69 |     OP_CHECK_IF(
  70 |         GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
  71 |         OP_LOGE(context, "GetPlatformInfo error"),
  72 |         return ge::GRAPH_FAILED);
  73 | 
  74 |     // 2. 获取属性
  75 |     auto attrs = context->GetAttrs();
  76 |     OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  77 |     int32_t axis = *(attrs->GetAttrPointer<int32_t>(0));
  78 |     int32_t tiles = *(attrs->GetAttrPointer<int32_t>(1));
  79 | 
  80 |     // 3. 获取输入 shape
  81 |     auto xShapePtr = context->GetInputShape(0);
  82 |     OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
  83 |     auto xShape = xShapePtr->GetStorageShape();
  84 |     int32_t rank = static_cast<int32_t>(xShape.GetDimNum());
  85 | 
  86 |     // 负轴处理
```

---

#### [2] 人工检视意见

- **提出人**: zhangzijie
- **作者**: wang-shilong32
- **文件**: random/tile_with_axis/op_host/arch35/tile_with_axis_tiling.cpp
- **行号**: 78
- **评论时间**: 2026-04-23
- **Commit**: 0b2d97d72ee0
- **问题描述**:

  > GetAttrPointer 返回值未检查就直接解引用，如果 tiles 属性缺失返回 nullptr，会导致程序崩溃。应该在解引用前检查返回值是否为 nullptr。

- **代码片段**（行78）:
```cpp
  68 |     int64_t coreNum = 0;
  69 |     OP_CHECK_IF(
  70 |         GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
  71 |         OP_LOGE(context, "GetPlatformInfo error"),
  72 |         return ge::GRAPH_FAILED);
  73 | 
  74 |     // 2. 获取属性
  75 |     auto attrs = context->GetAttrs();
  76 |     OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  77 |     int32_t axis = *(attrs->GetAttrPointer<int32_t>(0));
  78 |     int32_t tiles = *(attrs->GetAttrPointer<int32_t>(1));
  79 | 
  80 |     // 3. 获取输入 shape
  81 |     auto xShapePtr = context->GetInputShape(0);
  82 |     OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
  83 |     auto xShape = xShapePtr->GetStorageShape();
  84 |     int32_t rank = static_cast<int32_t>(xShape.GetDimNum());
  85 | 
  86 |     // 负轴处理
  87 |     if (axis < 0) {
```

---

#### [3] 人工检视意见

- **提出人**: zhangzijie
- **作者**: wang-shilong32
- **文件**: random/tile_with_axis/op_host/arch35/tile_with_axis_tiling.cpp
- **行号**: 87
- **评论时间**: 2026-04-23
- **Commit**: 0b2d97d72ee0
- **问题描述**:

  > 负轴处理逻辑不完整，当 axis 的绝对值大于 rank 时，转换后的值仍然无效（负数或超出范围）。应该添加边界检查确保 axis 在有效范围内（0 <= axis < rank）。

- **代码片段**（行87）:
```cpp
  77 |     int32_t axis = *(attrs->GetAttrPointer<int32_t>(0));
  78 |     int32_t tiles = *(attrs->GetAttrPointer<int32_t>(1));
  79 | 
  80 |     // 3. 获取输入 shape
  81 |     auto xShapePtr = context->GetInputShape(0);
  82 |     OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
  83 |     auto xShape = xShapePtr->GetStorageShape();
  84 |     int32_t rank = static_cast<int32_t>(xShape.GetDimNum());
  85 | 
  86 |     // 负轴处理
  87 |     if (axis < 0) {
  88 |         axis += rank;
  89 |     }
  90 | 
  91 |     // 4. 计算索引映射参数
  92 |     int64_t outerSize = 1;
  93 |     for (int32_t i = 0; i < axis; i++) {
  94 |         outerSize *= xShape.GetDim(i);
  95 |     }
  96 |     int64_t inputAxisSize = xShape.GetDim(axis);
```

---

#### [4] 人工检视意见

- **提出人**: zhangzijie
- **作者**: wang-shilong32
- **文件**: random/tile_with_axis/op_host/arch35/tile_with_axis_tiling.cpp
- **行号**: 78
- **评论时间**: 2026-04-23
- **Commit**: 0b2d97d72ee0
- **问题描述**:

  > tiles 是必选属性且规范要求 >= 1，但代码中没有验证这个约束。tiles <= 0 会导致 totalOutputElements 计算错误，应该在获取 tiles 后添加有效性检查。

- **代码片段**（行78）:
```cpp
  68 |     int64_t coreNum = 0;
  69 |     OP_CHECK_IF(
  70 |         GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
  71 |         OP_LOGE(context, "GetPlatformInfo error"),
  72 |         return ge::GRAPH_FAILED);
  73 | 
  74 |     // 2. 获取属性
  75 |     auto attrs = context->GetAttrs();
  76 |     OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  77 |     int32_t axis = *(attrs->GetAttrPointer<int32_t>(0));
  78 |     int32_t tiles = *(attrs->GetAttrPointer<int32_t>(1));
  79 | 
  80 |     // 3. 获取输入 shape
  81 |     auto xShapePtr = context->GetInputShape(0);
  82 |     OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
  83 |     auto xShape = xShapePtr->GetStorageShape();
  84 |     int32_t rank = static_cast<int32_t>(xShape.GetDimNum());
  85 | 
  86 |     // 负轴处理
  87 |     if (axis < 0) {
```

---

#### [5] 人工检视意见

- **提出人**: zhangzijie
- **作者**: wang-shilong32
- **文件**: random/tile_with_axis/op_host/arch35/tile_with_axis_tiling.cpp
- **行号**: 141
- **评论时间**: 2026-04-23
- **Commit**: 0b2d97d72ee0
- **问题描述**:

  > 核数设置未验证上限。needCoreNum 可能超过平台实际 AIV 核数，虽然在 SIMT 场景不涉及 SyncAll，但仍建议添加核数上限检查，确保不超过平台核数。

- **代码片段**（行141）:
```cpp
 131 |     tilingData->needCoreNum = static_cast<int32_t>(needCoreNum);
 132 |     tilingData->totalOutputElements = totalOutputElements;
 133 |     tilingData->perCoreElements = perCoreElements;
 134 |     tilingData->lastCoreElements = lastCoreElements;
 135 |     tilingData->outerSize = outerSize;
 136 |     tilingData->inputAxisSize = inputAxisSize;
 137 |     tilingData->innerSize = innerSize;
 138 |     tilingData->tiles = tiles;
 139 | 
 140 |     // 8. 设置核数和 local memory
 141 |     context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
 142 |     context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));
 143 | 
 144 |     // 9. 获取数据类型并设置 tiling key
 145 |     auto inputDesc = context->GetInputDesc(0);
 146 |     OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
 147 |     ge::DataType dataType = inputDesc->GetDataType();
 148 | 
 149 |     uint64_t tilingKey = 0;
 150 |     if (dataType == ge::DT_FLOAT) {
```

---


### 文件: random/tile_with_axis/op_host/tile_with_axis_infershape.cpp（Tiling侧）

#### [6] 人工检视意见

- **提出人**: zhangzijie
- **作者**: wang-shilong32
- **文件**: random/tile_with_axis/op_host/tile_with_axis_infershape.cpp
- **行号**: 35
- **评论时间**: 2026-04-23
- **Commit**: 0b2d97d72ee0
- **问题描述**:

  > GetAttrPointer 返回值未检查就直接解引用，如果属性不存在返回 nullptr，会导致程序崩溃。应该在解引用前检查返回值是否为 nullptr。

- **代码片段**（行35）:
```cpp
  25 | {
  26 |     OP_LOGD(context->GetNodeName(), "Begin to do InferShapeTileWithAxis");
  27 | 
  28 |     // 获取输入 shape
  29 |     const gert::Shape* xShape = context->GetInputShape(IDX_0);
  30 |     OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
  31 | 
  32 |     // 获取属性: axis 和 tiles
  33 |     auto attrs = context->GetAttrs();
  34 |     OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  35 |     int32_t axis = *(attrs->GetAttrPointer<int32_t>(0));   // axis, 默认值 1
  36 |     int32_t tiles = *(attrs->GetAttrPointer<int32_t>(1));  // tiles, 必选
  37 | 
  38 |     // 负轴处理
  39 |     int32_t rank = static_cast<int32_t>(xShape->GetDimNum());
  40 |     if (axis < 0) {
  41 |         axis += rank;
  42 |     }
  43 | 
  44 |     // 设置输出 shape
```

---

#### [7] 人工检视意见

- **提出人**: zhangzijie
- **作者**: wang-shilong32
- **文件**: random/tile_with_axis/op_host/tile_with_axis_infershape.cpp
- **行号**: 50
- **评论时间**: 2026-04-23
- **Commit**: 0b2d97d72ee0
- **问题描述**:

  > 维度乘法可能整数溢出。当 xShape 在 axis 维度很大且 tiles 也很大时，乘积可能超出 int64_t 范围导致溢出。应该添加溢出检查或在计算前验证边界。

- **代码片段**（行50）:
```cpp
  40 |     if (axis < 0) {
  41 |         axis += rank;
  42 |     }
  43 | 
  44 |     // 设置输出 shape
  45 |     gert::Shape* yShape = context->GetOutputShape(IDX_0);
  46 |     OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
  47 |     yShape->SetDimNum(rank);
  48 |     for (int32_t i = 0; i < rank; i++) {
  49 |         if (i == axis) {
  50 |             yShape->SetDim(i, xShape->GetDim(i) * tiles);
  51 |         } else {
  52 |             yShape->SetDim(i, xShape->GetDim(i));
  53 |         }
  54 |     }
  55 | 
  56 |     OP_LOGD(context->GetNodeName(), "End to do InferShapeTileWithAxis");
  57 |     return GRAPH_SUCCESS;
  58 | }
  59 | 
```

---

### 文件: random/tile_with_axis/op_kernel/arch35/tile_with_axis_simt.h（Kernel侧）

#### [8] 人工检视意见

- **提出人**: zhangzijie
- **作者**: wang-shilong32
- **文件**: random/tile_with_axis/op_kernel/arch35/tile_with_axis_simt.h
- **行号**: 42
- **评论时间**: 2026-04-23
- **Commit**: 0b2d97d72ee0
- **问题描述**:

  > 类型转换缺少边界检查。将 innerSize、inputAxisSize 直接转换为 uint64_t，如果输入参数为负数（虽然语义上不应该），转换后会变成很大的正数导致计算错误。应该在转换前检查参数有效性。

- **代码片段**（行42）:
```cpp
  32 | inline void OpTileWithAxisSimtKernel(
  33 |     int64_t currentCoreElements,
  34 |     int64_t perCoreElements,
  35 |     int64_t outerSize,
  36 |     int64_t inputAxisSize,
  37 |     int64_t innerSize,
  38 |     int32_t tiles,
  39 |     __gm__ T* input_gm,
  40 |     __gm__ T* output_gm)
  41 | {
  42 |     const uint64_t uInnerSize = static_cast<uint64_t>(innerSize);
  43 |     const uint64_t uInputAxisSize = static_cast<uint64_t>(inputAxisSize);
  44 |     const uint64_t uOutputAxisDim = uInputAxisSize * static_cast<uint64_t>(tiles);
  45 | 
  46 |     for (uint64_t idx = static_cast<uint64_t>(
  47 |              AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
  48 |              AscendC::Simt::GetThreadIdx());
  49 |          idx < static_cast<uint64_t>(currentCoreElements);
  50 |          idx += static_cast<uint64_t>(AscendC::Simt::GetThreadNum() *
  51 |                                        AscendC::Simt::GetBlockNum())) {
```

---

#### [9] 人工检视意见

- **提出人**: zhangzijie
- **作者**: wang-shilong32
- **文件**: random/tile_with_axis/op_kernel/arch35/tile_with_axis_simt.h
- **行号**: 53
- **评论时间**: 2026-04-23
- **Commit**: 0b2d97d72ee0
- **问题描述**:

  > globalOutIdx 计算可能溢出。GetBlockIdx() * perCoreElements 的乘积可能非常大，虽然使用了 uint64_t，但在极端情况下仍可能溢出。应该添加溢出检查或限制输入范围。

- **代码片段**（行53）:
```cpp
  43 |     const uint64_t uInputAxisSize = static_cast<uint64_t>(inputAxisSize);
  44 |     const uint64_t uOutputAxisDim = uInputAxisSize * static_cast<uint64_t>(tiles);
  45 | 
  46 |     for (uint64_t idx = static_cast<uint64_t>(
  47 |              AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
  48 |              AscendC::Simt::GetThreadIdx());
  49 |          idx < static_cast<uint64_t>(currentCoreElements);
  50 |          idx += static_cast<uint64_t>(AscendC::Simt::GetThreadNum() *
  51 |                                        AscendC::Simt::GetBlockNum())) {
  52 |         // 计算全局输出展平索引
  53 |         uint64_t globalOutIdx = static_cast<uint64_t>(
  54 |             AscendC::Simt::GetBlockIdx()) *
  55 |             static_cast<uint64_t>(perCoreElements) + idx;
  56 | 
  57 |         // 从输出展平索引分解多维坐标
  58 |         uint64_t innerIdx = globalOutIdx % uInnerSize;
  59 |         uint64_t remainder = globalOutIdx / uInnerSize;
  60 |         uint64_t outAxisIdx = remainder % uOutputAxisDim;
  61 |         uint64_t inAxisIdx = outAxisIdx % uInputAxisSize;
  62 |         uint64_t outerIdx = remainder / uOutputAxisDim;
```

---


## 被检视代码

> 本报告基于 PR 2365 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `random/tile_with_axis/op_host/arch35/tile_with_axis_tiling.cpp`
- `random/tile_with_axis/op_host/tile_with_axis_def.cpp`
- `random/tile_with_axis/op_host/tile_with_axis_infershape.cpp`
- `random/tile_with_axis/op_kernel/arch35/tile_with_axis_simt.h`
- `random/tile_with_axis/op_kernel/arch35/tile_with_axis_tiling_data.h`
