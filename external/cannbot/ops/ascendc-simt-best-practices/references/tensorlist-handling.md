# TensorList 动态输入处理

## 概述

`ListTensorDesc` 是昇腾 C 算子开发中用于描述动态输入/动态输出列表的核心数据结构，主要用于核函数直调场景下手动管理多个 Tensor 的元信息与数据指针。

## ListTensorDesc 内存布局

```
ListTensorDesc:
├── ptrOffset (uint64_t)           // dataPtr 的偏移量
├── tensorDesc[0] (TensorDesc)      // 第 0 个 tensor 的描述
│   ├── dim (uint32_t)
│   ├── index (uint32_t)
│   └── shape[SHAPE_DIM] (uint64_t[])  // SHAPE_DIM 固定为 8
├── tensorDesc[1] (TensorDesc)
│   └── ...
└── dataPtr[TENSOR_DESC_NUM] (uintptr_t[])  // 数据指针数组
```

## Host 侧获取 tensorlist 信息

```cpp
int32_t INPUT_IDX = 0;
auto computeNodeInfoPtr = tilingContext_->GetComputeNodeInfo();
auto idxInstanceInfoPtr = computeNodeInfoPtr.GetInputInstanceInfo(INPUT_IDX);
uint64_t tensorNum = idxInstanceInfoPtr->GetInstanceNum();

for (uint64_t i = 0; i < tensorNum; i++) {
    auto idxTensorShapePtr = tilingContext_->GetDynamicInputShape(INPUT_IDX, i);
    auto idxTensorShape = idxTensorShapePtr->GetStorageShape();
    auto idxTensorDtypePtr = tilingContext_->GetDynamicInputDesc(INPUT_IDX, i);
    auto idxDtype = idxTensorDtypePtr->GetDataType();
}
```

### Host 侧 API

| API | 说明 |
|-----|------|
| `GetInputInstanceInfo(ir_index)` | 获取输入实例化对象 |
| `GetInstanceNum()` | 获取动态输入中实际 tensor 个数 |
| `GetDynamicInputShape(dynamicInputIndex, tensorIndex)` | 获取指定 tensor 的 shape |
| `GetDynamicInputDataType(dynamicInputIndex, tensorIndex)` | 获取指定 tensor 的 dtype |

## Device 侧解析（方式一：SIMT VF 内部解析）

```cpp
// 获取第 idx 个 tensor 的 GM 地址
__simt_callee__ inline __gm__ T* SimtGetTensorAddr(GM_ADDR tensorListPtr, int64_t idx) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

// 获取第 idx 个 tensor 的第 dim 维大小
__simt_callee__ inline uint64_t SimtGetTensorShape(GM_ADDR tensorListPtr, int64_t idx, uint32_t dim) {
    uint32_t dimSize = *(reinterpret_cast<__gm__ uint64_t*>(tensorListPtr) + 1) & 0xffffffff;
    uint32_t descStructSize = 1 + dimSize;
    __gm__ uint64_t* shapeAddrStart = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr) + 1 + descStructSize * idx;
    __gm__ uint64_t* shapePtr = shapeAddrStart + 1 + dim;
    return *shapePtr;
}
```

## Device 侧解析（方式二：SIMT VF 外部解析）

```cpp
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;
ListTensorDesc xList(reinterpret_cast<__gm__ void*>(x));

// 获取各 tensor 地址
__gm__ T* input0Addr = xList.GetDataPtr<T>(0);
__gm__ T* input1Addr = xList.GetDataPtr<T>(1);

// 获取各 tensor shape
TensorDesc<T> desc;
xList.GetDesc(desc, 0);
uint64_t input0dim0 = desc.GetShape(0);
```

## 选择建议

| tensor 个数 | 推荐方式 | 原因 |
|------------|---------|------|
| 有限且已知 | host 侧获取 shape → tilingdata 传递 | 简单直接 |
| 不限或大量 | kernel 内直接解析 | 避免 tilingdata 传递大量 shape |

> **注意**：tensor 的地址信息必须在 kernel 侧进行解析，禁止在 host 侧解析后通过 tilingdata 传递。