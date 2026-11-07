# MoE Dispatch/Combine 编译框架特需说明

本文记录 mc2 算子直调工程在编译框架上与普通 AscendC 算子的差异，并沉淀 sample 中可复用的工程 blueprint：host 多卡调用方式、工程编译方式和算子验证链路。

参考工程：`moe_dispatch_direct_invoke_sample/`

## 工程结构差异

与通用单文件 `.asc` 工程不同，mc2 算子直调工程为 `host + kernel` 双层结构：

```
moe_dispatch_direct_invoke_sample/
├── CMakeLists.txt          # host 可执行 + kernel 库双目标
├── build.sh                # 封装 cmake + make
├── moe_dispatch.cpp        # kernel launch 入口（非 .asc）
├── include/
│   ├── tiling_data.h       # HcclA3/A5OpResParam + MoeDispatchTilingInfo
│   ├── moe_dispatch_base_compat.h  # 平台 compat 层（GetBaseWindAddrByRankId 等）
│   └── utils.h             # AlignUp / CeilDiv / SyncFunc（SetFlag+WaitFlag 封装）等工具函数
├── kernel/
│   ├── moe_dispatch.h      # kernel 主流程
│   └── mte_dispatch_comm.h # window 地址计算 + 状态写入/等待
└── test/
    └── test_moe_dispatch.cpp  # host 测试：HCCL 初始化 + 通信资源 + launch
```

这份结构的核心价值在于明确工程分层职责：

- host 侧文件负责设备初始化、多卡通信域建立、stream 管理、通信资源创建和 kernel launch
- kernel 主流程文件负责算子计算本体
- include 中的 tiling / compat 头文件负责 HCCL 约定结构和平台差异适配
- test / script 负责结果校验和端到端验证

这几层职责共同构成 mc2 直调工程的基本组织方式。

## CMakeLists.txt 关键差异

### 双目标

```cmake
# kernel 库（编译为 .so，供 host 调用）
ascendc_library(ascendc_kernels SHARED moe_dispatch.cpp)
ascendc_include_directories(ascendc_kernels PRIVATE ${INCLUDE_DIRS})

# host 可执行（链接 HCCL + tiling + platform）
add_executable(test_moe_dispatch test/test_moe_dispatch.cpp)
```

### mc2 特殊链接依赖

```cmake
target_link_libraries(test_moe_dispatch PRIVATE
    ascendc_kernels
    tiling_api      # tiling API，提供 GetTilingKey 等
    platform        # 平台能力查询
    pthread
    hccl            # HCCL 核心，提供 HcclAllocComResourceByTiling
    hccl_fwk        # HCCL 框架层
    hcomm           # 通信底层
)
```

普通 AscendC 算子不需要 `hccl`、`hccl_fwk`、`hcomm`，这三个是 mc2 特有依赖。

### CANN 路径

```cmake
# 支持两个安装路径
set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
```

### host 侧编译选项

```cmake
target_compile_options(test_moe_dispatch PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(test_moe_dispatch PRIVATE SOC_VERSION="${SOC_VERSION}")
```

`-D_GLIBCXX_USE_CXX11_ABI=0` 是 HCCL 对 ABI 兼容性的要求，**不要移除**。

## 平台宏处理

`moe_dispatch_base_compat.h` 通过 `__NPU_ARCH__` 在编译期选择 dav-3510 / dav-2201 实现（对应样例结构名 `HcclA5OpResParam` / `HcclA3OpResParam`，A3/A5 为工程惯称）：

```c++
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    using HcclOpParam = HcclA5OpResParam;
    // dav-3510（Ascend 950PR/950DT）地址实现
#else
    using HcclOpParam = HcclA3OpResParam;
    // dav-2201（Atlas A2/A3 系列）地址实现（默认）
#endif
```

**`SOC_VERSION` 控制 `__NPU_ARCH__`**，通过 `-DSOC_VERSION=` 传入编译器。目标 dav-2201（Atlas A2/A3 系列，工程惯称 A3）芯片不需要特殊设置；dav-3510（Ascend 950PR/950DT，工程惯称 A5）需要确认 `SOC_VERSION` 正确。

## Host 侧通信资源创建（mc2 特需）

直调工程 host 侧需显式调用 `HcclAllocComResourceByTiling` 创建通信资源并传给 kernel：

```c++
// host test/test_moe_dispatch.cpp 的核心流程
HcclComm comm = ...;                // HCCL 通信域
void *stream = ...;                 // 当前 stream
void *mc2Context = nullptr;
// 创建 MTE 通信资源，返回 mc2Context 指针（内含 window 地址等）
// 注意：签名是 HcclAllocComResourceByTiling(comm, stream, mc2Tiling, &mc2Context)，
//       mc2Tiling 传完整 tiling（host 侧指向含 Mc2InitTiling 的 TilingData），
//       不能只传 &tilingData.mc2CcTiling，也不可省略 stream（与 asc-devkit hccl_mc2.h 一致）。
HcclAllocComResourceByTiling(comm, stream, tilingData, &mc2Context);

// 将 mc2Context 的地址传给 kernel（作为 GM_ADDR）
// kernel 中通过 InitHcclContextByAddr(mc2Context, ...) 访问
```

**与 registry invoke 的区别**：registry 模式下 mc2Context 由框架自动注入；直调模式下必须手动创建和传递，`InitHcclContextByAddr` 内不再调用任何框架接口。

从工程职责看，这一节对应 host 侧多卡调用路径：

- 设备和通信域初始化，拿到 `HcclComm`
- 基于 tiling 创建通信资源，拿到 `mc2Context`
- 将 `mc2Context` 和 tiling 一起传给 kernel launch
- 由 host 测试或运行入口负责同步、回收和结果校验

多卡调用路径本身是 sample 最重要的工程价值之一，不属于隐含在 test 文件中的附带信息。

## 头文件包含顺序

kernel 文件的典型包含顺序：

```c++
// 优先使用正式安装路径的头，fallback 到本地 compat
#if __has_include("common/op_kernel/moe_distribute_base.h")
#include "common/op_kernel/moe_distribute_base.h"
#else
#include "moe_dispatch_base_compat.h"
#endif
#include "utils.h"
#include "basic_api/kernel_basic_intf.h"
```

这个 `__has_include` 保护允许工程同时在 mc2 内部环境（有 `moe_distribute_base.h`）和外部样例工程（只有 compat 层）中编译。

## build.sh 简化流程

```bash
# build.sh 核心
cmake -DSOC_VERSION=${SOC_VERSION} -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## 验证链路

验证链路包括：

- host 测试入口负责准备输入、初始化 HCCL、launch kernel、回收输出
- 校验脚本负责将输出结果与预期或基线比较
- README 或运行说明负责给出多卡启动方式、环境变量和执行命令

这部分属于算子工程交付说明的组成部分。

## 新增或修改算子时的检查清单

- `tiling_data.h` 中 `HcclA3OpResParam` / `HcclA5OpResParam` 结构体字段顺序不能随意调整（与 HCCL ABI 绑定）
- 新增 tiling 字段加在 `MoeDispatchTilingInfo` 中，不要修改 `Mc2InitTiling` / `Mc2CcTiling`
- kernel 函数签名修改后，`moe_dispatch.cpp` 和 `test_moe_dispatch.cpp` 需同步更新
- 链接库版本随 CANN 升级变化时，只需修改 `CMakeLists.txt` 的 `target_link_libraries`

## 下一跳

- 工程文件落点与改动定位：`change-routing.md`
- 通信资源和地址获取接口：`../api-rules/mte-address-access.md`
- `samples` 路径入口：`index.md`
