# SHMEM 代码规范

> **仓内路径**：下文 `examples/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

本文档定义 SHMEM 通信算子与通算融合算子的代码规范，用于指导 `shmem-ops-code-gen` 生成符合项目标准的代码。

## 0. 官方样例对齐原则

生成代码的风格优先对齐官方 CANN samples 和当前仓库的 SHMEM examples。走读新算子时，先参考 CANN samples 中相近算子的目录组织、CMake 写法、脚本入口和数据文件布局，再参考 `examples/` 中 SHMEM 初始化、Host/Device 分层、工具函数和通信 API 用法。

代表性参考：CANN samples 中的 moe dispatch/combine 样例、`examples/utils`、`examples/dispatch_gmm_combine`。

生成代码必须体现以下边界：

| 位置 | 约束 |
| --- | --- |
| License | C++/Python/Shell 文件带 CANN Open Software License 头；临时草稿可省略，交付不可省略 |
| 通用逻辑 | 日志、ACL/RT/SHMEM 检查、文件读写、目录创建、rank 解析等放入 `utils.h` 或 example-local utility，不在 `main.cpp` 散写 |
| `main.cpp` 职责 | 参数解析、ACL/SHMEM lifecycle、资源分配、数据读写、kernel launch、同步；禁止 route/payload/tiling/packing/layout/golden/checker |
| `main.cpp` PE 模型 | 单进程单 PE，一次进程只绑一个 device 和一个 `my_pe`；多 PE 由 `scripts/run.sh`/MPI/外部 launcher 启动多个独立进程 |
| Device kernel | kernel、launch wrapper、声明分别放 `*_kernels.cpp`/`*_kernels.h`，不塞进 Host 主流程 |
| 脚本入口 | `SCRIPT_DIR`/`PROJECT_ROOT` 定位工程，校验参数，清理旧输出，生成输入/golden，按 PE 启动进程并 `wait`，最后调 checker |
| CMake | 优先 target-scoped（`target_include_directories`/`target_compile_options`/`target_link_libraries`）；兼容官方 example 时可全局，但限制影响范围 |

### 0.1 SHMEM 开源代码风格硬约束

生成 C/C++、公共头文件、示例代码和文档代码片段时，必须同时遵守 SHMEM 开源代码风格指南中的约束：

| 约束 | 要求 |
| --- | --- |
| 公开 API 稳定 | 新增/修改公共函数、类型、宏、枚举值前必须考虑兼容性；ABI 结构体字段尾部追加，不重排已有字段 |
| 内部实现不外露 | 公共头文件、示例代码和用户文档中不得出现 `aclshmemi_*` 作为用户可调用接口或可依赖类型 |
| 示例只展示推荐用法 | 只使用 `aclshmem_*` 和 `aclshmemx_*` 公共 API，不包含内部头文件，不调用内部函数，不依赖未文档化环境变量 |
| 异步语义写清楚 | 提交成功、入队成功、远端可见、通信完成是不同概念；接口注释和示例必须区分，并展示完成路径 |
| 最小必要抽象 | 新增封装必须服务于稳定 ABI、跨模块复用或清晰边界，不为局部代码制造额外层级 |

**C/C++ 格式要求**：

| 规则 | 说明 |
| --- | --- |
| 缩进 | 4 个空格，单行不超过 120 列 |
| 指针 | 左对齐 `int *ptr`，不写 `int* ptr` 或 `int * ptr` |
| 参数换行 | 参数多时按语义分组换行，不强行挤一行 |
| 未使用参数 | `UNUSED_PARAM(x)` 或等价机制显式标记 |
| 命名风格 | 文件名/局部变量/SHMEM API helper 用 `snake_case`；example-local Host helper 可 PascalCase（与官方 sample 对齐），同模块内一致；布尔变量用 `is_`/`has_`/`can_` 前缀 |
| 宏命名 | 公共宏 `CANN_SHMEM_<模块>_<语义>`；内部宏 `SHMEMI_<模块>_<语义>`；枚举值 `ACLSHMEM_*` 或 `ACLSHMEMX_*` |

**公共头文件和 API 注释要求**：

| 规则 | 说明 |
| --- | --- |
| 文件后缀 | 公共声明用 `.h`，实现用 `.cpp`；Device 模板/内联可用 `.hpp`，但不作为 ABI 承诺 |
| 头文件自包含 | 不依赖隐式 include 顺序；新增头文件用项目统一 include guard 或 `#pragma once` |
| Host API 导出 | 用项目统一导出宏，返回 `int` 或 `aclshmem_error_code_t`，注释明确线程安全、阻塞性、异步完成条件 |
| Device API 标记 | 用项目统一 device 标记，注释明确调用域（device-only/block/warp/vector/host+device），不引入非必要 runtime 分支 |
| Doxygen 注释 | 公共 API 至少包含 `@brief`、参数 `[in]/[out]/[in,out]`、返回值、执行域、阻塞语义、完成语义、线程安全 |
| 注释质量 | 说明单位、范围、生命周期和行为影响；避免 "the count" 这类重复字面含义 |

**示例代码必须完整展示**：初始化、参数检查、错误处理、通信完成、资源释放和失败退出路径。示例代码不得把内部调试入口、内部头文件或实验能力包装成用户推荐用法。

## 1. 错误处理规范

### 1.1 ACL 错误检查

**必须使用宏进行错误检查**，不要裸写 `if (status != ACL_ERROR_NONE)`。

推荐使用 `utils.h` 中定义的宏：

```cpp
// 简单检查，打印错误位置
ACL_CHECK(aclrtCreateStream(&stream));

// 带返回值的检查
ACL_CHECK_WITH_RET(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST),
                   ERROR_LOG("Failed to allocate memory"),
                   return false);
```

**反模式**：
```cpp
// ❌ 不要这样写
if (aclrtCreateStream(&stream) != ACL_ERROR_NONE) {
    std::cerr << "Failed to create stream" << std::endl;
    return false;
}
```

### 1.2 SHMEM 错误检查

SHMEM API 返回值必须检查：

```cpp
// 定义检查宏
#define CHECK_SHMEM(expr) CheckShmem((expr), #expr, __FILE__, __LINE__)

bool CheckShmem(int status, const char *expr, const char *file, int line)
{
    if (status == ACLSHMEM_SUCCESS) {
        return true;
    }
    std::cerr << file << ":" << line << " SHMEM call failed: "
              << expr << ", ret=" << status << std::endl;
    return false;
}

// 使用
if (!CHECK_SHMEM(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes))) {
    return 1;
}
```

### 1.3 早返回宏与资源归属

带 `return` 的检查宏只能用在不持有局部资源的代码段，或有统一 cleanup 的函数中。一旦已分配资源（`aclrtMalloc`、`aclrtMallocHost`、`aclshmem_malloc`、stream、event、文件句柄），后续 API 失败不能直接从宏中返回。

推荐模式如下，其中 `CHECK_ACL_BOOL`/`CHECK_SHMEM_BOOL` 表示只返回 `bool`、不直接 `return` 的检查 helper：

```cpp
bool LaunchRmaBlocks(...)
{
    void *send_device = nullptr;
    void *route_device = nullptr;
    bool ok = true;

    ok = ok && CHECK_ACL_BOOL(aclrtMalloc(&send_device, send_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ok = ok && CHECK_ACL_BOOL(aclrtMalloc(&route_device, route_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!ok) {
        goto cleanup;
    }

    ok = ok && CHECK_ACL_BOOL(aclrtMemcpy(send_device, send_bytes, host.data(), send_bytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE));
    if (!ok) {
        goto cleanup;
    }

cleanup:
    if (route_device != nullptr) {
        ok = CHECK_ACL_BOOL(aclrtFree(route_device)) && ok;
    }
    if (send_device != nullptr) {
        ok = CHECK_ACL_BOOL(aclrtFree(send_device)) && ok;
    }
    return ok;
}
```

反模式：

```cpp
void *send_device = nullptr;
void *route_device = nullptr;
CHECK_ACL(aclrtMalloc(&send_device, send_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
CHECK_ACL(aclrtMalloc(&route_device, route_bytes, ACL_MEM_MALLOC_HUGE_FIRST)); // 失败会泄露 send_device
```

### 1.4 同步和 cleanup 返回值

所有有返回值的 ACL/RT/SHMEM 调用都应检查，包括 `aclshmem_barrier_all`、`aclshmem_finalize`、`aclrtDestroyStream`、`aclrtResetDevice`、`aclFinalize`。cleanup 阶段不要忽略错误，推荐累计 `cleanup_ok`，最终与主流程 `ok` 合并返回。

```cpp
bool cleanup_ok = true;
if (stream != nullptr) {
    cleanup_ok = CHECK_ACL_BOOL(aclrtDestroyStream(stream)) && cleanup_ok;
}
if (shmem_inited) {
    cleanup_ok = CHECK_SHMEM_BOOL(aclshmem_finalize()) && cleanup_ok;
}
if (device_set) {
    cleanup_ok = CHECK_ACL_BOOL(aclrtResetDevice(device_id)) && cleanup_ok;
}
if (acl_inited) {
    cleanup_ok = CHECK_ACL_BOOL(aclFinalize()) && cleanup_ok;
}
return ok && cleanup_ok;
```

### 1.5 空指针检查

**使用统一的空指针检查模式**：

```cpp
// ✅ 推荐：提前返回模式
if (ptr == nullptr) {
    ERROR_LOG("Pointer is null");
    return ACLSHMEM_INVALID_VALUE;
}
// 继续正常逻辑

// ✅ 推荐：使用宏简化重复检查
#define CHECK_NULL_PTR(ptr, msg) \
    do { \
        if ((ptr) == nullptr) { \
            ERROR_LOG(msg); \
            return ACLSHMEM_INVALID_VALUE; \
        } \
    } while (0)

CHECK_NULL_PTR(attributes, "attributes is null");
CHECK_NULL_PTR(default_uid, "default_uid is null");
CHECK_NULL_PTR(ip_port, "ip_port is null");
```

**反模式**：
```cpp
// ❌ 不要写大量重复的 if 判断
if (attributes != nullptr) {
    if (default_uid != nullptr) {
        if (ip_port != nullptr) {
            // 嵌套过深
        }
    }
}

// ❌ 不要在每个函数中重复写相同的检查逻辑
if (ptr == nullptr) {
    std::cerr << "ptr is null" << std::endl;
    return -1;
}
```

## 2. 函数设计规范

### 2.1 避免一行函数

**不要为简单的算术运算创建单独的函数**，除非该运算在多处复用且有明确的语义抽象。

**反模式**（来自 dispatch_kernel.cpp）：
```cpp
// ❌ 这些一行函数没有必要
FORCE_INLINE_AICORE int64_t CountIndex(int32_t src_pe, int32_t dst_pe,
                                       int32_t local_expert, int32_t n_pes,
                                       int32_t experts_per_pe)
{
    return (static_cast<int64_t>(src_pe) * n_pes + dst_pe) * experts_per_pe + local_expert;
}

FORCE_INLINE_AICORE int64_t OutputRowIndex(int32_t local_expert, int32_t slot,
                                           int32_t hidden, int32_t max_tokens_per_expert)
{
    return (static_cast<int64_t>(local_expert) * max_tokens_per_expert + slot) * hidden;
}

FORCE_INLINE_AICORE int64_t MetaIndex(int32_t local_expert, int32_t slot,
                                      int32_t max_tokens_per_expert)
{
    return (static_cast<int64_t>(local_expert) * max_tokens_per_expert + slot) * 3;
}
```

**推荐**：
```cpp
// ✅ 直接在使用处计算，添加注释说明语义
// count_matrix layout: [src_pe][dst_pe][local_expert]
const int64_t count_idx = (static_cast<int64_t>(src_pe) * n_pes + dst_pe) * experts_per_pe + local_expert;

// output layout: [local_expert][slot][hidden]
const int64_t output_offset = (static_cast<int64_t>(local_expert) * max_tokens_per_expert + slot) * hidden;

// metadata layout: [local_expert][slot][3] where 3 = [src_pe, src_token, expert]
const int64_t meta_offset = (static_cast<int64_t>(local_expert) * max_tokens_per_expert + slot) * 3;
```

**例外情况**：如果索引计算逻辑复杂、在多处使用、或需要保持与设计文档中的符号一致性，可以封装为函数。

### 2.2 GM 加载辅助函数

对于 Device 侧的 GM 加载，可以使用简单的辅助函数：

```cpp
// ✅ 这种辅助函数是合理的，因为它封装了类型转换和解引用
template <typename T>
FORCE_INLINE_AICORE T gm_load(__gm__ T *addr)
{
    return *addr;
}
```

### 2.3 参数验证函数

复杂的参数验证逻辑应该封装为独立函数：

```cpp
// ✅ 推荐：将参数设置和验证封装为函数
int32_t SetAttr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size,
                const char *ip_port, aclshmemx_uniqueid_t *default_uid,
                aclshmemx_init_attr_t *attributes)
{
    CHECK_NULL_PTR(attributes, "attributes is null");
    CHECK_NULL_PTR(default_uid, "default_uid is null");
    CHECK_NULL_PTR(ip_port, "ip_port is null");

    size_t ip_len = std::min(std::strlen(ip_port),
                             static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN) - 1);
    std::copy_n(ip_port, ip_len, attributes->ip_port);
    attributes->ip_port[ip_len] = '\0';

    if (attributes->ip_port[0] == '\0') {
        return ACLSHMEM_INVALID_VALUE;
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE,
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(default_uid);

    return ACLSHMEM_SUCCESS;
}
```

## 3. 资源管理规范

### 3.1 资源释放顺序

资源必须按照与分配相反的顺序释放：

```cpp
// ✅ 正确的释放顺序
bool RunCase(const Options &opts)
{
    aclrtStream stream = nullptr;
    void *input_dev = nullptr;
    void *output_sym = nullptr;
    bool ok = true;

    // 分配资源
    ok = ok && CHECK_ACL_BOOL(aclrtCreateStream(&stream));
    ok = ok && CHECK_ACL_BOOL(aclrtMalloc(&input_dev, size, ACL_MEM_MALLOC_HUGE_FIRST));
    output_sym = aclshmem_malloc(output_size);

    if (!ok || output_sym == nullptr) {
        // 清理已分配的资源
        goto cleanup;
    }

    // 使用资源...

cleanup:
    // 按相反顺序释放
    if (output_sym != nullptr) {
        aclshmem_free(output_sym);
    }
    if (input_dev != nullptr) {
        ok = CHECK_ACL_BOOL(aclrtFree(input_dev)) && ok;
    }
    if (stream != nullptr) {
        ok = CHECK_ACL_BOOL(aclrtDestroyStream(stream)) && ok;
    }

    return ok;
}
```

### 3.2 SHMEM 生命周期

SHMEM 初始化和清理必须遵循固定顺序：

```cpp
// ✅ 正确的生命周期管理
int main(int argc, char **argv)
{
    // 1. ACL 初始化
    bool ok = true;
    ok = ok && CHECK_ACL_BOOL(aclInit(nullptr));
    ok = ok && CHECK_ACL_BOOL(aclrtSetDevice(device_id));

    // 2. SHMEM 初始化
    ok = ok && CHECK_SHMEM_BOOL(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    // 3. 使用 SHMEM
    ok = RunCase(opts) && ok;

    // 4. 清理（逆序）
    bool cleanup_ok = true;
    cleanup_ok = CHECK_SHMEM_BOOL(aclshmem_finalize()) && cleanup_ok;
    cleanup_ok = CHECK_ACL_BOOL(aclrtResetDevice(device_id)) && cleanup_ok;
    cleanup_ok = CHECK_ACL_BOOL(aclFinalize()) && cleanup_ok;

    return (!ok || !cleanup_ok) ? 1 : 0;
}
```

### 3.3 资源归属表

每个 Host run path 应能一眼看出资源归属。生成代码时，建议在实现计划或代码结构中保持以下资源分层：

| 资源 | 分配 API | 释放 API | 释放位置 |
| --- | --- | --- | --- |
| ACL runtime | `aclInit`、`aclrtSetDevice` | `aclrtResetDevice`、`aclFinalize` | 进程退出前 |
| stream | `aclrtCreateStream` | `aclrtDestroyStream` | 所有 kernel 和 stream sync 后 |
| 普通 device memory | `aclrtMalloc` | `aclrtFree` | 持有函数的 cleanup |
| Host pinned memory | `aclrtMallocHost` | `aclrtFreeHost` | 持有函数的 cleanup |
| symmetric memory | `aclshmem_malloc` | `aclshmem_free` | 所有 PE 使用完且 finalize 前 |
| SHMEM lifecycle | `aclshmemx_init_attr` | `aclshmem_finalize` | 所有 symmetric memory 释放后 |

局部临时 device buffer 不要依赖“后续流程总会成功”来释放；只要函数内有多次分配，就必须有统一 cleanup 或 RAII wrapper。

### 3.4 SHMEM barrier 与 symmetric memory

| 约束 | 说明 |
| --- | --- |
| 释放前同步 | 释放 symmetric memory 前，相关 kernel/RMA/barrier/stream sync 必须已完成 |
| 顺序 | `aclshmem_free` 必须早于 `aclshmem_finalize` |
| finalize 后禁止 | 不得再访问 symmetric pointer，不得再调用依赖 SHMEM runtime 的 barrier/query API |

## 4. 命名规范

### 4.1 变量命名

- 使用描述性名称，避免单字母变量（循环索引除外）
- 单位必须在变量名中体现：

```cpp
// ✅ 推荐
size_t input_elems = tokens_per_pe * hidden;
size_t input_bytes = input_elems * sizeof(int32_t);
int64_t offset_bytes = row * hidden * sizeof(int32_t);
int64_t offset_elems = row * hidden;

// ❌ 避免
size_t input_size;  // 不清楚是字节还是元素
int64_t offset;     // 不清楚单位
```

### 4.2 常量命名

- 全局常量使用 `k` 前缀 + PascalCase
- 宏常量使用全大写 + 下划线
- 同一生成模块内不要混用 `DEFAULT_LOCAL_MEM_SIZE`、`BLOCK_BYTES` 和 `kFlagBlockBytes` 这类常量风格。生成代码默认使用 `kPascalCase`；全大写只留给宏、编译开关和必须对齐第三方头文件的符号。

```cpp
// ✅ 推荐
constexpr uint64_t kLocalMemSize = 1024ULL * 1024ULL * 1024ULL;
constexpr int32_t kMaxGlobalExperts = 1024;
constexpr int32_t kDefaultBlockDim = 8;

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#define CHECK_ACL(expr) CheckAcl((expr), #expr, __FILE__, __LINE__)
```

### 4.3 函数命名

- 公共 SHMEM API 函数使用 `snake_case`，并使用 `aclshmem_*` 或 `aclshmemx_*` 前缀；禁止新增驼峰式公共 API 或无前缀公共符号。
- 内部 SHMEM 实现函数使用 `snake_case`，建议使用 `aclshmemi_*` 前缀，且不得出现在公共头文件、示例代码或用户文档中。
- 生成算子 example-local 的 Host helper 可沿用官方 sample 中的 PascalCase 风格，但不得伪装成公共 API；同一模块内必须保持一致。
- Device kernel 入口可沿用 AscendC / sample 风格，Device 内部 helper 可使用 PascalCase 或 camelCase；若是新增公共 Device API，则按公共 API 命名和注释要求处理。

```cpp
// Host 函数
bool ParseOptions(int argc, char **argv, Options *opts);
bool WriteInt32File(const std::string &path, const std::vector<int32_t> &data);

// Device kernel
extern "C" __global__ __aicore__ void ShmemDispatchI32(...);

// Device 实现函数
ACLSHMEM_DEVICE void DispatchI32Impl(...);
```

## 5. 代码组织规范

### 5.1 文件结构

标准的 SHMEM 算子项目应包含：

```
op_name/
├── CMakeLists.txt
├── README.md
├── docs/
│   └── design.md
├── src/
│   ├── main.cpp          # Host 入口、参数解析、初始化、launch、结果写出
│   ├── op_name_kernel.cpp
│   ├── op_name_kernel.h  # kernel 声明和 Host launch wrapper
│   ├── op_host_plan.cpp  # 可选：复杂 Host tiling/layout/packing 计划
│   ├── op_host_plan.h
│   ├── op_io.cpp         # 可选：文件读写、case 目录、rank 输出
│   ├── op_io.h
│   ├── op_runtime.cpp    # 可选：资源封装、错误检查、启动辅助
│   └── op_runtime.h
├── scripts/
│   ├── gen_data.py
│   ├── check_result.py
│   ├── run.sh
│   └── run_case_matrix.py
├── baseline/              # 有 baseline 时 MUST 存在
│   ├── CMakeLists.txt
│   ├── src/
│   │   └── op_name_baseline.cpp
│   └── scripts/
│       └── run_baseline.sh
└── docs/
    └── results.md        # 测试结果记录
```

### 5.2 Host 主流程边界

官方样例允许 `main.cpp` 承担 Host orchestration，但生成的交付代码应采用更严格的边界。若官方 sample 使用 fork-style demo launcher，只参考其模块边界和数据流，不继承到 `main.cpp`。

**逻辑归属表**：

| 逻辑类型 | 测试数据/oracle | 算子运行时 Host 计划 | 算子执行语义 |
| --- | --- | --- | --- |
| route table / payload 编码解码 / 数据预处理 | Python | 独立 C++ Host 模块 | Device kernel |
| dtype cast / quant / half-bfloat16 转换 | Python | 独立 C++ Host 模块 | Device kernel |
| layout / offset / shape 派生 / tiling 参数 / 文件读写 / 资源管理 | — | 可拆到 C++ helper | — |
| 多 PE 启动 | `scripts/run.sh` / MPI / 外部 launcher | — | — |

**拆分信号**：Host 文件超过约 400 行，且大部分不是 lifecycle、资源管理和 launch 编排。

推荐拆分：

```text
src/
├── main.cpp                  # 参数、lifecycle、资源、launch、读写
├── op_host_plan.h/.cpp       # Host 侧 tiling、route、payload、packing 计划
├── op_layout.h/.cpp          # layout、offset、shape 派生
├── op_io.h/.cpp              # 文件读写、case 目录、rank 输出
├── op_runtime.h/.cpp         # 资源封装、错误检查、启动辅助
└── op_kernels.h/.cpp         # Host launch wrapper + Device kernel
└── utils.h                   # 日志、错误检查、通用 helper
```

`main.cpp` 中保留清晰的阶段调用即可，例如 `LoadInputs -> AllocateBuffers -> LaunchDispatch -> SaveOutputs -> Cleanup`。

复杂 Host 模块约束：

| 约束 | 说明 |
| --- | --- |
| I/O | 输入输出使用明确 struct，不直接依赖全局变量或隐式环境 |
| lifecycle | 不拥有 ACL/SHMEM lifecycle；资源创建和释放由 `main.cpp` 或 runtime wrapper 统一管理 |
| 禁止事项 | 不生成 golden、不做精度比较、不启动多 PE 子进程 |
| 可验证 | 能通过小输入独立验证，或在 `gen_data.py`/`check_result.py` 中有等价逻辑可交叉检查 |

### 5.3 头文件包含顺序

```cpp
// 1. 标准库
#include <iostream>
#include <vector>
#include <cstdint>

// 2. 第三方库
#include <acl/acl.h>

// 3. SHMEM 库
#include "shmem.h"

// 4. 项目内部头文件
#include "op_name_kernel.h"
#include "utils.h"
```

### 5.4 匿名命名空间

内部辅助函数和常量应放在匿名命名空间中：

```cpp
namespace {

constexpr int32_t kMaxExperts = 1024;

bool CheckAcl(aclError status, const char *expr, const char *file, int line)
{
    // ...
}

} // namespace

// 公开接口
void shmem_dispatch_i32(...)
{
    // ...
}
```

## 6. Device 代码规范

### 6.1 FFTS 配置

需要 Device barrier 的 kernel 必须在入口设置 FFTS：

```cpp
ACLSHMEM_DEVICE void KernelImpl(uint64_t ffts_addr, ...)
{
    util_set_ffts_config(ffts_addr);

    const int32_t block_idx = static_cast<int32_t>(GetBlockIdx());
    const int32_t my_pe = aclshmem_my_pe();
    const int32_t n_pes = aclshmem_n_pes();

    // kernel 逻辑...
}
```

### 6.2 同步规范

> **Legacy 例外（范围严格限定）**
>
> | 场景 | `aclshmemx_barrier_all_vec` | 规范 |
> | --- | --- | --- |
> | **custom-ops 新算子**（`shmem-ops-code-gen` 交付） | **禁止新增** | **MUST** 用 `aclshmem_barrier_all()` / `aclshmem_barrier(team)`；见 [internal-api-boundary.md §2](internal-api-boundary.md) |
> | **最小 diff 对齐 upstream `examples/`** | 可 **原样保留** 已有调用 | 仅未改动的 legacy 行；**禁止**在新 phase 同步路径中主动选用 |
> | **代码走读** | custom-ops 新写代码出现 → **FAIL** | legacy 未改动行 → 记为 known legacy，不扩写 |

- NBI 操作后必须有明确的完成路径
- **单引擎** MTE put/get 后用 `aclshmemx_mte_quiet()`；SDMA 用 `aclshmemx_sdma_quiet()`；多引擎混用或收尾阶段再用 `aclshmem_quiet()`
- **新代码** phase 同步 **MUST** 使用 `aclshmem_barrier_all()` 或 `aclshmem_barrier(team)`；**NEVER** 在新路径中选用 `aclshmemx_barrier_all_vec()`（legacy 保留见上表）
- 机内跨 PE 搬运必须使用 `aclshmem_*` 或公开 `aclshmemx_*` 数据面接口；禁止用 `DataCopy` 直接读写远端 PE 地址

```cpp
// ✅ 正确的同步模式
// Phase 1: 写入数据
for (int peer = 0; peer < n_pes; ++peer) {
    aclshmem_int32_p(dst, value, peer);
}
aclshmemx_mte_quiet();  // MTE 标量写后 drain（多引擎混用时 aclshmem_quiet）

aclshmem_barrier_all();  // 等待所有 PE 完成 Phase 1

// Phase 2: 读取数据
// 现在可以安全读取其他 PE 写入的数据
```

### 6.3 跨 PE 传输接口边界

`DataCopy` 只能表达本 PE 内本地 GM/UB/local buffer 搬运。跨 PE 数据传输必须通过 SHMEM 数据面接口：

| 接口类型 | 示例 |
| --- | --- |
| 标准接口 | `aclshmem_<type>_put/get/p/g`、`putmem/getmem`、`iput/iget`、`put_signal` |
| 公开 engine 接口 | `aclshmemx_mte_put_nbi/get_nbi`、`aclshmemx_sdma_put_nbi/get_nbi`、`aclshmemx_roce_put_nbi/get_nbi` |

禁止模式：

```cpp
// ❌ 不允许：把远端 symmetric 地址当本地 GM 直接 DataCopy。
auto *remote = reinterpret_cast<__gm__ int32_t *>(aclshmem_ptr(local_sym, peer));
DataCopy(remote, local, elem_count);
```

推荐模式：

```cpp
// ✅ 使用 SHMEM 数据面接口表达跨 PE put。
aclshmem_int32_put(remote_sym, local, elem_count, peer);
aclshmemx_mte_quiet();
```

如果需要把本地 GM 数据先整理到 UB，可以先对本 PE 本地地址使用 `DataCopy`，再通过 SHMEM put/get/engine API 完成跨 PE 搬运。

### 6.4 UB buffer、event 和 magic value

Device 侧 MTE/SDMA/RDMA 接口的 UB buffer、event id、block bytes 等参数必须集中命名，不在调用点散落 magic value。

| 规则 | 说明 |
| --- | --- |
| 命名常量 | UB placeholder、event id、block bytes 必须封装为 `constexpr` 命名常量 |
| 空 UB pointer | 若 API 允许空 UB pointer，必须封装为命名 helper 并用一行注释说明 API 前提 |
| event 复用 | 同一 event id 必须成对 `SetFlag`/`WaitFlag`，不跨未完成 NBI 操作复用 |
| ping-pong | 优先使用两个命名 event id，避免读写阶段互相覆盖 |

```cpp
// MTE put only needs this placeholder UB pointer for the API contract.
constexpr uint32_t kMtePutUbBytes = 512;
constexpr int32_t kRmaBlockBytes = 512;
constexpr auto kRmaEventId = EVENT_ID0;
__ubuf__ uint8_t *ub_placeholder = reinterpret_cast<__ubuf__ uint8_t *>(0);
```

同一个 event id 的复用必须成对出现 `SetFlag`/`WaitFlag`，并且不要跨未完成的 NBI 操作复用。ping-pong 搬运优先使用两个命名 event id，避免读写阶段互相覆盖。

### 6.5 类型转换

GM 地址转换应显式且一致：

```cpp
// ✅ 推荐：在 kernel 入口统一转换
__gm__ int32_t *input_tokens = reinterpret_cast<__gm__ int32_t *>(input_addr);
__gm__ int32_t *expert_idx = reinterpret_cast<__gm__ int32_t *>(expert_addr);

// 后续直接使用类型化指针
int32_t expert = gm_load(expert_idx + token);
```

## 7. 注释规范

### 7.1 何时写注释

- **必须注释**：非显而易见的算法逻辑、特殊的边界条件、性能优化的原因
- **不要注释**：显而易见的代码、重复代码内容的描述

```cpp
// ✅ 好的注释
// Compute prefix sum over count_matrix to get compact output offset
// This avoids atomics and ensures deterministic placement
int32_t prefix = 0;
for (int32_t src_pe = 0; src_pe < my_pe; ++src_pe) {
    prefix += gm_load(count_matrix + count_idx);
}

// ❌ 不必要的注释
// Loop through all tokens
for (int32_t token = 0; token < tokens_per_pe; ++token) {
    // Get expert index
    int32_t expert = gm_load(expert_idx + token);
}
```

### 7.2 文件头注释

每个文件应包含版权声明：

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except
 * in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
```

## 8. 测试和调试规范

### 8.1 输入生成

测试输入应包含 rank 特征，便于定位问题：

```python
# ✅ 推荐：gen_data.py 生成包含 PE 信息的输入
def make_rank_pattern(pe_id: int, tokens_per_pe: int, hidden: int) -> np.ndarray:
    input_data = np.zeros((tokens_per_pe, hidden), dtype=np.int32)
    for token in range(tokens_per_pe):
        for h in range(hidden):
            # 格式：PE_ID * 1000000 + token * 1000 + hidden_idx
            input_data[token, h] = (pe_id + 1) * 1000000 + token * 1000 + h
    return input_data
```

### 8.2 输出文件命名

输出文件必须包含 PE ID 和 case ID，避免覆盖：

```cpp
// ✅ 推荐
std::string output_file = JoinPath(opts.output_dir,
    "output_pe" + std::to_string(opts.pe_id) + ".bin");
std::string meta_file = JoinPath(opts.output_dir,
    "meta_pe" + std::to_string(opts.pe_id) + ".bin");
```

### 8.3 Sentinel 值

未使用的输出应初始化为 sentinel 值，便于检测错误：

```cpp
// ✅ 推荐：使用 -1 或 0xFF 作为 sentinel
CHECK_ACL(aclrtMemset(output_sym, output_size, 0xFF, output_size));
CHECK_ACL(aclrtMemset(meta_sym, meta_size, 0xFF, meta_size));
```

## 9. 性能相关规范

### 9.0 实现与设计一致性

进入性能采集前先对照 `design.md` 走读实际编译路径：

| 检查项 | 要求 |
| --- | --- |
| CMake target | 必须包含设计声明的 kernel、Host helper、transport 源文件和模板 |
| launch 参数 | `block_dim`、phase 顺序、core partition、tile/chunk/tail 必须与设计一致 |
| `block_dim=1` | 仅临时调试；除非设计证明单核是目标实现，否则不能作为 correctness 交付 |
| 并发补齐 | 从无到有属于 correctness 补齐，不算性能优化轮次 |
| transport API | 必须出现在实际编译路径中，不能只存在于未调用文件或过期文档 |

### 9.1 避免不必要的分支

```cpp
// ❌ 避免：在热路径中重复检查常量条件
for (int token = 0; token < tokens_per_pe; ++token) {
    if (global_experts <= MAX_EXPERTS) {  // 这个条件在循环外已知
        // ...
    }
}

// ✅ 推荐：将条件提到循环外
if (global_experts <= MAX_EXPERTS) {
    for (int token = 0; token < tokens_per_pe; ++token) {
        // ...
    }
}
```

### 9.2 局部变量优化

对于频繁访问的数组，使用局部数组缓存：

```cpp
// ✅ 推荐：使用栈上数组缓存计数
int32_t local_counts[MAX_GLOBAL_EXPERTS];
for (int32_t i = 0; i < global_experts; ++i) {
    local_counts[i] = 0;
}

// 累加到局部数组
for (int32_t token = 0; token < tokens_per_pe; ++token) {
    int32_t expert = gm_load(expert_idx + token);
    if (expert >= 0 && expert < global_experts) {
        local_counts[expert] += 1;
    }
}

// 最后一次性写入远端
for (int32_t peer = 0; peer < n_pes; ++peer) {
    for (int32_t e = 0; e < global_experts; ++e) {
        aclshmem_int32_p(dst + e, local_counts[e], peer);
    }
}
```

## 10. CMake 与脚本规范

### 10.1 CMake

生成 CMake 时优先使用 target-scoped 风格，便于和官方 sample 的现代写法对齐，也避免全局 include/link 污染其他 target。

```cmake
add_executable(my_op src/main.cpp src/my_op_kernels.cpp)
target_include_directories(my_op PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${SHMEM_HOME_PATH}/shmem/include"
)
target_compile_options(my_op PRIVATE -O2)
target_link_directories(my_op PRIVATE
    "${SHMEM_HOME_PATH}/shmem/lib"
    "${ASCEND_HOME_PATH}/lib64"
)
target_link_libraries(my_op PRIVATE shmem ascendcl runtime stdc++ pthread)
```

CMake 规范要求：

| 规则 | 要求 |
| --- | --- |
| 依赖发现 | 通过 `ASCEND_HOME_PATH`/`SHMEM_HOME_PATH` 或 design 指定路径；不硬编码用户机器路径 |
| 缺失处理 | 缺少 `shmem.h`/`libshmem.so`/CANN include/lib 时 `message(FATAL_ERROR ...)` 给出修复提示 |
| 编译选项 | `NPU_ARCH`/CCE/FFTS 开关放在 target 上，避免影响 Host-only 工具 |
| 运行时链接 | 独立工程设置 `BUILD_RPATH` 或 run script `LD_LIBRARY_PATH`，保证二进制找到 SHMEM 和 CANN runtime |

### 10.2 Shell 脚本

运行脚本对齐官方样例的结构：

```bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
```

脚本规则：

| 规则 | 要求 |
| --- | --- |
| 参数 | 校验必需参数，未知参数直接失败 |
| 输出目录 | 按 case/rank 隔离，运行前清理旧结果 |
| 多 PE 进程 | 后台启动后必须 `wait`，检查 checker 返回值 |
| 导出变量 | `SHMEM_UID_SESSION_ID`、`LD_LIBRARY_PATH`、rank 输入/输出路径、`PYTHON_CMD`，来源可追踪 |
| Python 依赖 | 默认 `PYTHON_CMD=${PYTHON_CMD:-python3}`，运行前检查可用性；缺少时输出清晰错误，提示用户提供 `PYTHON_CMD` 或激活 conda/venv |
| 多 PE 入口 | 必须在脚本层完成，禁止 `main.cpp` 中 fork/spawn/system |

多 PE 启动必须在脚本层完成，典型结构如下：

```bash
for ((rank = 0; rank < PE_SIZE; rank++)); do
    "${EXEC_BIN}" --n-pes "${PE_SIZE}" --rank "${rank}" --device "$((FIRST_NPU + rank))" &
done
wait
```

不要生成“一个 `main.cpp` 进程 fork 多个 PE 子进程”的入口，即使用户伪代码采用这种写法，也要在生成阶段改写为脚本启动多个独立进程。

### 10.3 Python 数据与 checker

| 规则 | 要求 |
| --- | --- |
| 参数 | 使用 `argparse`，固定随机种子 |
| 目录结构 | `input/golden/output/chip_{rank}` |
| 职责分离 | `gen_data.py` 负责输入和 golden；`check_result.py` 负责读取算子输出并给出 mismatch 明细 |
| 禁止混入 | checker 逻辑不得混入 C++ Host 主流程 |
| 依赖 | 必须在 README 或 test contract 中列出；测试命令使用同一 `PYTHON_CMD` |

## 11. 常见反模式总结

| 反模式 | 问题 | 正确做法 |
|--------|------|----------|
| 裸写错误检查 | 代码重复，难以维护 | 使用统一的错误检查宏 |
| 过多一行函数 | 增加调用开销，降低可读性 | 直接内联计算，添加注释 |
| 嵌套的空指针检查 | 代码嵌套过深 | 使用提前返回或检查宏 |
| 不释放资源 | 内存泄漏 | 按相反顺序释放所有资源 |
| 早返回宏跨过 cleanup | 局部 device/host 资源泄漏 | 统一 cleanup 或 RAII |
| 忽略 barrier/cleanup 返回值 | 失败被静默吞掉 | 累计 `cleanup_ok` 并返回 |
| NBI 后无同步 | 数据竞争 | 使用 quiet/barrier 确保完成 |
| 跨 PE 搬运直接用 `DataCopy` | 绕过 SHMEM runtime 的拓扑、engine 和同步语义 | 使用 `aclshmem_*` 或公开 `aclshmemx_*` 数据面接口 |
| 未接入 target 的高性能 kernel | 文档声称优化但实际未运行 | 检查 CMake target 和 launch 路径 |
| 固定 `block_dim=1` 交付 | 并发设计未落地 | 按 design 落地多 core/多 lane，单核仅作临时调试 |
| 把并发补齐记为性能优化 | 混淆 correctness 和 optimization | 并发补齐先完成，再开始优化轮次 |
| 单位不明确的变量名 | 容易出错 | 变量名包含单位（bytes/elems） |
| 常量命名混用 | 风格不稳定，审查困难 | 生成代码统一 `kPascalCase`，宏才用全大写 |
| `main.cpp` 承担 route/payload/tiling/packing/checker | 主流程过重，且破坏职责分离 | 复杂 Host 逻辑拆到独立 `.cpp/.h`；golden/checker 放到 Python；算子语义放到 Device kernel |
| `main.cpp` fork/spawn 多 PE 子进程 | 破坏单进程单 PE 约束，资源和日志边界不清 | `scripts/run.sh` 或外部 launcher 启动多个独立进程 |
| 全局 CMake include/link 污染 | 影响其他 target，难定位依赖 | 使用 target-scoped CMake |
| 循环内重复检查常量 | 性能损失 | 将条件提到循环外 |
| 输出文件被覆盖 | 多 PE 测试失败 | 文件名包含 PE ID |

## 12. 代码审查清单

生成代码后，应检查以下项目：

- [ ] C/C++ 缩进 4 空格，单行不超过 120 列，指针写作 `int *ptr`
- [ ] 公共头文件、示例代码和用户文档中没有用户可见的 `aclshmemi_*`
- [ ] 示例代码只使用 `aclshmem_*` / `aclshmemx_*` 公共 API，不包含内部头文件，不调用内部函数
- [ ] 异步示例展示了 `quiet`、`barrier`、`sync` 或 stream 同步等完成路径，并说明远端可见条件
- [ ] 公共 API 注释包含参数方向、返回值、执行域、阻塞语义、完成语义和线程安全说明
- [ ] 头文件自包含，include guard 或 `#pragma once` 风格与项目一致
- [ ] 有意未使用的参数使用 `UNUSED_PARAM(x)` 或等价机制
- [ ] 所有 ACL/SHMEM 调用都有错误检查
- [ ] `aclshmem_barrier_all`、`aclrtDestroyStream`、`aclrtResetDevice`、`aclFinalize` 等同步和 cleanup 调用没有被忽略
- [ ] 资源按相反顺序释放
- [ ] 持有局部资源的函数没有用早返回宏绕过 cleanup
- [ ] 空指针检查使用统一模式
- [ ] 没有不必要的一行函数
- [ ] 变量名包含单位信息
- [ ] 常量命名在同一模块内保持一致，宏和 constexpr 不混用风格
- [ ] NBI 操作后有明确的同步
- [ ] 跨 PE 数据传输使用 `aclshmem_*` 或公开 `aclshmemx_*` 接口，没有直接 `DataCopy` 远端地址
- [ ] CMake target、实际 kernel/source、transport API 和 launch block_dim 与 `design.md` 一致
- [ ] correctness 交付不是固定 `block_dim=1` 的单核临时路径，除非 design 明确要求单核
- [ ] Device kernel 设置了 FFTS
- [ ] UB placeholder、event id、block bytes 等 Device magic value 已命名并解释
- [ ] `main.cpp` 只做 Host 编排，不包含复杂 route/payload/tiling/packing 逻辑；Host 复杂逻辑已拆到独立 `.cpp/.h`
- [ ] golden/checker 未留在 Host C++ 侧
- [ ] `main.cpp` 单进程单 PE，不包含 `fork`、`spawn`、`system` 或多 PE 子进程管理逻辑
- [ ] CMake 使用 target-scoped include/link/compile options，依赖缺失时 fail fast
- [ ] run script 使用 `SCRIPT_DIR`/`PROJECT_ROOT`，多 PE 输出按 rank 隔离
- [ ] 输出文件名包含 PE ID
- [ ] 测试输入包含 rank 特征
- [ ] 未使用的输出初始化为 sentinel 值
- [ ] 注释解释了"为什么"而不是"是什么"
- [ ] 文件包含版权声明
