# OP_KERNEL UT 详细指南

## 概述

OP_KERNEL UT 测试 Kernel 计算逻辑，分两种类型：
- **AscendC Kernel**：AI Core 上的 Kernel，使用 CPU 模拟执行（tikicpulib）
- **AICPU Kernel**：AICPU 上的 Kernel

> **注意**：`op_kernel/`（AscendC Kernel）和 `op_kernel_aicpu/`（AICPU Kernel）是互斥的，一个算子只会存在其中一种。

---

## 目录结构

```
<repo>/<category>/<op>/tests/ut/
├── op_kernel/                  # AscendC Kernel UT
│   ├── CMakeLists.txt
│   ├── test_<op>.cpp
│   └── <op>_data/              # 测试数据
└── op_kernel_aicpu/            # AICPU Kernel UT
    ├── CMakeLists.txt
    └── test_<op>.cpp
```

---

## AscendC Kernel UT

### 核心组件

```cpp
#include "tikicpulib.h"

// 1. 声明kernel入口函数
extern "C" __global__ __aicore__ void <op>(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

// 2. 内存分配
uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(<Op>TilingData));

// 3. 设置Tiling数据
<Op>TilingData* tilingData = reinterpret_cast<<Op>TilingData*>(tiling);
tilingData->usedCoreNum = 1;
tilingData->totalLength = 1024;

// 4. 执行Kernel
ICPU_SET_TILING_KEY(tilingKey);
ICPU_RUN_KF(<op>, numBlocks, x, y, workspace, tilingData);

// 5. 释放内存
AscendC::GmFree((void*)x);
AscendC::GmFree((void*)y);
AscendC::GmFree((void*)workspace);
AscendC::GmFree((void*)tiling);
```

### 两种Tiling参数获取模式

| 模式 | 适用场景 | 说明 |
|------|---------|------|
| 手动设置 | Tiling参数简单、固定 | 手动构造TilingData结构体 |
| 自动获取 | Tiling参数复杂（推荐） | 使用ExecuteTiling从Tiling实现获取 |

### 手动设置模式示例

```cpp
TEST_F(add_lora_test, test_add_lora_0) {
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16781184);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AddLoraTilingData));

    AddLoraTilingData* tilingData = reinterpret_cast<AddLoraTilingData*>(tiling);
    tilingData->usedCoreNum = 20;
    tilingData->batch = 1;

    ICPU_SET_TILING_KEY(100001);
    ICPU_RUN_KF(add_lora, 20, x, y, workspace, tilingData);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
```

### 自动获取模式示例

```cpp
#include "tiling_case_executor.h"

TEST_F(IsInfTest, test_case_float16_1) {
    // 1. 构造Tiling上下文并获取参数
    optiling::IsInfCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("IsInf",
        {{{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{128, 64}, {128, 64}}, ge::DT_BOOL, ge::FORMAT_ND}},
        &compileInfo);

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);

    // 2. 使用TilingInfo中的参数
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    // 3. 执行
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(is_inf, tilingInfo.blockNum, x, y, workspace, tiling);
}
```

### CMakeLists.txt

```cmake
if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(<op>_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/<op>_tiling.cpp
    )
    AddOpTestCase(<op> "ascend910b" "-DDTYPE_X=float" "${<op>_tiling_files}")
endif()
```

---

## AICPU Kernel UT

### 核心组件

```cpp
#include "gtest/gtest.h"
#define private public
#define protected public
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

// NodeDefBuilder
#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "OpName", "OpName")               \
        .Input({"input1", data_types[0], shapes[0], datas[0]})       \
        .Output({"output", data_types[1], shapes[1], datas[1]});

// 执行
RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

// 验证
bool compare = CompareResult(output, expected, count);
```

### AICPU 测试示例

```cpp
TEST_F(TEST_ADD_UT, FLOAT_TENSOR_ADD_TENSOR_SUCC) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};

    float input1[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float input2[6] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    float output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float expected[6] = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
    EXPECT_EQ(CompareResult(output, expected, 6), true);
}
```

---

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| `cannot find -lpem_davinci` | 算子不支持op_kernel，删除op_kernel UT目录 |
| 内存访问错误 | 使用32字节对齐：`CeilAlign(size, 32)` |
| KernelMode需要设置 | Vector API需要：`AscendC::SetKernelMode(KernelMode::AIV_MODE)` |

---

## 编译命令

```bash
# 编译 op_kernel UT
bash build.sh -u --opkernel --ops='<op_name>' --soc='<soc_version>'

# 编译并生成覆盖率
bash build.sh -u --opkernel --ops='<op_name>' --soc='<soc_version>' --cov
```