# SHMEM 构建、运行与测试

> **仓内路径**：下文 `examples/`、`install/`、`scripts/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。
> **custom-ops 命令唯一参考**：[custom-ops-entrypoints.md](custom-ops-entrypoints.md)（Skill 内禁止引用独立 shell 脚本文件；命令以该文档代码段为准）。

本文说明 SHMEM 算子两类构建方式：独立工程依赖已编译 SHMEM 库，以及算子作为 `shmem/examples` 内置 example 编译。`shmem-ops-code-gen` 只负责生成代码和测试入口；实际构建、运行和失败定位由 `shmem-ops-compile-debug` 执行。

## 1. 通用环境

构建和运行前，必须确认以下环境已就绪：

**CANN 环境**：`ASCEND_HOME_PATH` 已设置，`bisheng` 编译器可用。

- 已设置：直接使用。
- **未设置**：Agent **MUST** 询问用户选择「默认系统安装」或「自定义 `set_env.sh` 路径」，并在自定义选项中给出路径样例；**禁止**未确认就 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
- 完整流程见 [cann-env-resolution.md](cann-env-resolution.md)。

确认后记录 `CANN_SET_ENV`（`set_env.sh` 绝对路径），后续 build/run/`docker exec` 均使用该变量。

**Docker 容器（用户指定时 — MUST）**：若 `phase0_intake.docker_container` 非空，本节所有命令 **MUST** 包在 `docker exec <container> bash -lc '...'` 内执行，禁止宿主机裸跑。详见 [docker-exec-contract.md](../../shmem-ops-dev/references/docker-exec-contract.md)。

**SHMEM 核心库**：`install/shmem/lib/libshmem.so` 和 bootstrap 插件存在。in-tree example 不存在时先执行 `bash scripts/build.sh -examples`（**必须带 `-examples`**，否则无 `build/bin/<op_name>` target），再 `source install/set_env.sh`。

**Python 环境**：`PYTHON_CMD` 或 `python3` 可用，且能导入测试脚本需要的依赖（`numpy`，按脚本再检查 `torch` 等）。缺失时询问用户提供可执行文件或 conda/venv 激活命令。

**构建产物位置**：`build/lib`、`build/bin`（编译产物），`install/shmem/include` 和 `install/shmem/lib`（安装产物）。

常用构建参数：

| 参数 | 用途 |
| --- | --- |
| `-examples` | 编译 `examples/` 下的样例 |
| `-debug` | Debug 构建，定义 `DEBUG_MODE` |
| `-enable_rdma` | 开启 RDMA 相关 target 和编译宏 |
| `-enable_simt` | 开启 SIMT example |
| `-enable_ascendc_dump` | 开启 AscendC Dump/printf 调测宏 |
| `-mssanitizer` | example 编译 sanitizer 支持 |
| `-soc_type Ascend910B` | **910B 主机 MUST 显式指定**（省略可能导致 arch/默认不匹配） |
| `-soc_type Ascend950` | Ascend950 类 SOC 使用 |
| `-uttests` | 编译单元测试 |
| `-python_example` | 编译部分 Python/Torch 扩展示例 |

注意：`scripts/build.sh` 每次会清理并重建 `build/` 和 `install/`。调试单 target 时优先使用已有 `build/` 做增量构建（`cmake --build build --target <target> -j`），**不要**反复全量 `build.sh`。

### 1.1 运行时环境（MUST）

运行 example 前完整环境链：

```bash
source ${CANN_SET_ENV}    # Phase 0 用户确认的 set_env.sh
cd /path/to/shmem
bash scripts/build.sh -examples    # 首次或 install/set_env.sh 缺失时
source install/set_env.sh
cd examples/<op_name> && bash scripts/run.sh <pe_size>
```

`install/set_env.sh` 注入的路径（**不可省略**）：

- `$SHMEM_HOME_PATH/shmem/lib`（`libshmem.so` + `aclshmem_bootstrap_*.so`）
- `/usr/local/Ascend/driver/lib64/driver/`
- 可选：`$SHMEM_HOME_PATH/shmem/torch_binding/kernels`

算子 `scripts/run.sh` 在激活 SHMEM install 环境（`install/set_env.sh`）**之后** MUST 内联 [env-setup.snippet.md](env-setup.snippet.md) 中的：

- `setup_shmem_runtime_env ${PROJECT_ROOT}` — 完整 CANN + install + `build/lib`
- `setup_shmem_dynamic_endpoints` — 用户未 export 时随机化 `IPPORT` / `SHMEM_UID_SESSION_ID`
- `warn_shmem_stale_processes` — 检测长期运行的 `torch_test_*.py`

**禁止**在模板中写死 `IPPORT=tcp://127.0.0.1:27010` 和 `SHMEM_UID_SESSION_ID=127.0.0.1:8899`（多轮测试、Torch 测试并行时极易冲突）。

**反模式**：只设 `LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:...` 而不 `source install/set_env.sh` — 缺 bootstrap 插件和 driver 库，典型症状为 `aclError:100000` 或 golden 全 FAIL。

**假 FAIL 典型链**（必须先查日志，勿改 kernel）：

```text
address in use for bind listen on 127.0.0.1:27010
→ AccStoreServer startup failed
→ Memory Heap Not Initialized
→ aclshmem_malloc 无效 / D2H 全 0
→ result_compare FAILED
```

**Agent 要求**：Phase 4 必须在目标环境（用户指定的 Docker 容器或本机 NPU）实际执行 `scripts/run.sh`（见 [custom-ops-entrypoints.md](custom-ops-entrypoints.md) §2），不得仅输出命令。

## 2. 模式 A：独立工程依赖已编译 SHMEM 库

### 2.1 适用场景

使用该模式时，算子工程位于 shmem 仓库外，不修改 `shmem/examples`。工程只依赖：

- `${SHMEM_HOME_PATH}/shmem/include`
- `${SHMEM_HOME_PATH}/shmem/lib/libshmem.so`
- `${SHMEM_HOME_PATH}/shmem/lib/libshmem_utils.so`
- bootstrap 插件：`aclshmem_bootstrap_config_store.so`、`aclshmem_bootstrap_uid.so`，MPI 场景还需要 `aclshmem_bootstrap_mpi.so`

bootstrap 插件应与 `libshmem.so` 位于同一目录；不要只拷贝 `libshmem.so`。

本地仓库中的 `examples/init/CMakeLists.txt` 虽位于 examples 目录，但它展示了“链接 `install/shmem/lib` 中已编译 SHMEM 库”的写法，可作为独立工程 CMake 参考。

### 2.2 构建前置

```bash
export CANN_ENV=/path/to/user/specified/set_env.sh
source "${CANN_ENV}"
cd /path/to/shmem
bash scripts/build.sh
source install/set_env.sh
```

确认：

```bash
test -f "$SHMEM_HOME_PATH/shmem/include/shmem.h"
test -f "$SHMEM_HOME_PATH/shmem/lib/libshmem.so"
test -f "$SHMEM_HOME_PATH/shmem/lib/aclshmem_bootstrap_config_store.so"
```

### 2.3 独立工程 CMake 要点

独立工程需要自己提供 CANN、AscendC、SHMEM include 和 link 配置。最小骨架：

```cmake
cmake_minimum_required(VERSION 3.18)
project(shmem_op LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER bisheng)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "ASCEND_HOME_PATH is not set; ask the user for CANN set_env.sh and source it before build")
endif()
if(NOT DEFINED ENV{SHMEM_HOME_PATH})
  message(FATAL_ERROR "SHMEM_HOME_PATH is not set; source shmem/install/set_env.sh first")
endif()

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})

add_executable(my_shmem_op
  src/main.cpp
  src/my_shmem_op_kernel.cpp
  src/op_host_plan.cpp        # 可选：复杂 Host 计划逻辑
  src/op_io.cpp               # 可选：文件读写 helper
  src/op_runtime.cpp          # 可选：资源和错误处理 helper
)
# SHMEM_REPO: path to SHMEM source checkout (passed via -DSHMEM_REPO=...)
# Device .h under install/ include implementation .hpp under src/device/
target_include_directories(my_shmem_op PRIVATE
  ${ASCEND_HOME_PATH}/compiler/tikcpp
  ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/include/experiment/runtime
  ${ASCEND_HOME_PATH}/include/experiment/msprof
  ${SHMEM_HOME_PATH}/shmem/include
  ${SHMEM_REPO}/src/device
  ${SHMEM_REPO}/src/device_simt
  ${SHMEM_REPO}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(my_shmem_op PRIVATE -O2 -std=c++17 -xc++)
target_link_options(my_shmem_op PRIVATE --cce-fatobj-link)
target_link_directories(my_shmem_op PRIVATE
  ${ASCEND_HOME_PATH}/lib64
  ${SHMEM_HOME_PATH}/shmem/lib
)
target_link_libraries(my_shmem_op PRIVATE shmem ascendcl runtime stdc++ pthread)

> ⚠️ **禁止全局命令**：`link_libraries()`、`include_directories()`、`link_directories()` 为全局命令，违反 code-gen P0 规则（见 code-style.md §10.1）。**MUST** 全部使用 target-scoped 形式（`target_link_libraries` / `target_include_directories` / `target_link_directories`）。
```

若算子采用简单官方 example 根目录布局，也可以把 `main.cpp` 和 kernel 文件放在根目录；但生成算子若包含复杂 Host 计划逻辑，优先使用 `src/` 目录并在 target 中显式列出 helper 源文件，头文件也放在 `src/` 中。

如果包含 Device kernel，必须补齐与目标 SOC 匹配的 CCE/AscendC 编译选项。可从本地 `shmem/CMakeLists.txt` 的 `CMAKE_CCE_COMPILE_OPTIONS` 和相近 example 复制，不要凭空写 arch。

### 2.4 独立工程构建与运行命令

**MUST** 使用 [custom-ops-entrypoints.md](custom-ops-entrypoints.md) 中的命令代码段（编译 §1、运行 §2、matrix §3、Torch §4）。性能采集见 [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md)。

Agent 在 skill/交付文档/聊天摘要中 **NEVER** 把相对路径 `cd custom-ops/<op> && cmake -S .` 或省略 `${SHMEM_REPO}` 的裸路径作为首选入口（**MUST** 用 [custom-ops-entrypoints.md](custom-ops-entrypoints.md) §1–§3 的全路径命令）。

<details>
<summary>底层 cmake（仅 Agent 本地调试，非交付首选）</summary>

```bash
cd custom-ops/<op_name>
cmake -S . -B build && cmake --build build -j
cd custom-ops/<op_name> && bash scripts/run.sh ...
```
</details>

运行前确认：

```bash
ldd build/bin/my_shmem_op | grep -E "libshmem.so|not found"
```

若部署到自定义目录，把 `libshmem.so`、`libshmem_utils.so` 和 bootstrap 插件一起放入同一 `lib/`，并让 `LD_LIBRARY_PATH` 优先指向该目录。

## 3. 模式 B：算子作为 shmem/examples 内置 example

### 3.1 适用场景

使用该模式时，算子目录位于：

```text
shmem/examples/<op_name>/
```

适合复用 `examples/utils`、`examples/templates/include`、catcoc 模板、现有 example 分核/CMake 函数，以及需要随 SHMEM 源码一起编译的算子。

### 3.2 目录结构

通信 collective 常见结构：

```text
examples/<op_name>/
  CMakeLists.txt
  README.md
  docs/
    design.md
  src/
    main.cpp
    <op_name>_kernel.cpp
    <op_name>_kernel.h
    op_host_plan.cpp
    op_host_plan.h
  scripts/
    run.sh
    gen_data.py
    check_result.py
```

独立工程或 in-tree 示例使用对应的 CMake 规则构建。

```text
examples/<op_name>/
  CMakeLists.txt
  README.md
  src/
    main.cpp
    op_host_plan.cpp
    op_host_plan.h
  scripts/
    run.sh
    test_shapes.csv
```

### 3.3 CMake 接入

在 `examples/<op_name>/CMakeLists.txt` 中优先复用仓库函数：

```cmake
aclshmem_add_collective_example(<op_name>)
```

该函数假定存在官方约定的 `main.cpp` 和 `<op_name>_kernel.cpp`，会生成：

- `build/bin/<op_name>`
- `build/lib/lib<op_name>_kernel.so`

独立工程使用独立 CMakeLists.txt 构建；可参考 `examples/allgather/CMakeLists.txt`（仓内原生 example，用 `aclshmem_add_collective_example(<op>)` 生成 `build/lib/lib<op>_kernel.so` 共享库）自定义 shared library 与依赖关系；tiling library 等官方 helper 未覆盖的能力在 example-local `CMakeLists.txt` 中显式 `add_library` 声明。**禁止**引用 `custom-ops/<op>/CMakeLists.txt` 作参考——该路径由本 skill 生成、非 upstream 自带。

如果生成算子采用 `src/` 目录和多个 Host helper `.cpp/.h` 的模块化布局，不要强行套用只识别根目录文件的 helper 函数；应在 example-local `CMakeLists.txt` 中显式列出 `src/main.cpp`、kernel 源文件和 Host helper 源文件，并用 `target_include_directories(... PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)` 补齐头文件搜索路径。若必须复用官方 helper，则提供薄包装文件或调整目录使其符合 helper 的实际假设。

然后把新目录加入 `examples/CMakeLists.txt` 对应列表：

- 普通 example：加入主 `foreach(EXAMPLE ...)`。
- RDMA example：加入 `if(ACLSHMEM_RDMA_SUPPORT)` 内列表，并使用 `bash scripts/build.sh -examples -enable_rdma`。
- SIMT example：加入 `if(ACLSHMEM_SIMT_SUPPORT)` 内列表。

### 3.4 编译命令

首次完整配置并编译 examples：

```bash
export CANN_ENV=/path/to/user/specified/set_env.sh
source "${CANN_ENV}"
cd /path/to/shmem
bash scripts/build.sh -examples
source install/set_env.sh
```

常用变体：

```bash
bash scripts/build.sh -examples -debug
bash scripts/build.sh -examples -enable_ascendc_dump
bash scripts/build.sh -examples -enable_rdma
bash scripts/build.sh -examples -soc_type Ascend950
```

首次配置完成后，单 target 调试可用：

```bash
cmake --build build --target <op_name> -j
cmake --build build --target <op_name>_kernel -j
```

注意：再次执行 `scripts/build.sh` 会清空并重建 `build/`，增量调试时不要反复用它。

### 3.5 运行命令

优先使用 example 自带脚本：

```bash
cd /path/to/shmem/examples/<op_name>
bash scripts/run.sh 0,1
```

或按通信 demo 风格：

```bash
cd /path/to/shmem/examples/<op_name>
bash run.sh -pes 2 -fnpu 0 -gnpus 2 -ipport tcp://127.0.0.1:8766
```

run 脚本应做：

- **MUST** 调用 `setup_shmem_runtime_env ${PROJECT_ROOT}`（见 [env-setup.snippet.md](env-setup.snippet.md)），或等价地：`source install/set_env.sh` 后再追加 `build/lib` 和 `${ASCEND_HOME_PATH}/lib64`
- **MUST** 调用 `setup_shmem_dynamic_endpoints`（用户未 export 时自动分配 `IPPORT` / `SHMEM_UID_SESSION_ID`）
- 可选：`warn_shmem_stale_processes` 提示仍在运行的 `torch_test_*.py`
- 设置 `SHMEM_UID_SESSION_ID`（与 `ip_port` 参数分离；二者均勿写死为全局默认值）
- 检查 `build/bin/<op_name>` 存在，否则提示 `bash scripts/build.sh -examples`
- 每个 PE 启动一个进程，参数包含 `n_pes`、rank、`ip_port`、device id、shape/dtype
- 输出目录按 PE/case 隔离
- 等待所有 PE 退出，并聚合返回码
- 运行 checker，返回非 0 表示失败

有官方参考 example 时，优先复用其 `golden.py`/`result_compare.py` 和全尺寸测试数据，勿缩小 cache 做 smoke 导致 Memcpy 尺寸不匹配。

## 4. 正确性测试入口

测试脚本至少提供：

| 入口 | 目的 |
| --- | --- |
| smoke | 2 PE、小 shape、单 dtype，验证 init/malloc/launch/finalize |
| correctness | 按 `design.md` case matrix 验证 golden |
| stress | tail、chunk 边界、不均衡 PE、repeats |
| perf | correctness 通过后才可运行，带 warmup/repeats |

结果记录必须包含 command、workdir、case、oracle、tolerance、pass/fail 和日志路径。

## 5. 常见产物位置

| 模式 | 可执行文件 | kernel/shared library | 安装库 |
| --- | --- | --- | --- |
| 独立工程 | `<op>/build/bin/<target>` | `<op>/build/lib/*.so` | `$SHMEM_HOME_PATH/shmem/lib` |
| in-tree example | `shmem/build/bin/<op_name>` | `shmem/build/lib/lib<op_name>_kernel.so` | `shmem/install/shmem/lib` |
