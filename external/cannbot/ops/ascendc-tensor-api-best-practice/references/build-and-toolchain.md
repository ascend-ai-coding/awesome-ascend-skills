# Tensor API 构建与工具链

> 文档与样例事实源：asc-devkit。版本变化后必须复核本文结论。

## 0. 文档与样例事实源

手册或官方样例缺失时运行：

    cd <skill-root>
    python3 scripts/ensure_docs.py

脚本自动 clone 或 fast-forward `~/asc-devkit`；该工作区被非 master 分支占用或与 master 历史分叉时，自动维护 `~/asc-devkit-master` 浅镜像提供手册与样例，不修改主工作区。dirty、origin 不匹配时阻断。`--check-only` 不联网。输出 JSON 的 `selected_root` 为事实源根目录（显式传 `--asc-devkit-root` 时不启用 fallback）。

## 1. 目标与入口

- 目标芯片：Ascend 950PR / 950DT。
- 目标架构：dav-3510。
- ASC 编译参数：--npu-arch=dav-3510 -xasc。
- 用户未特别指定时，开发算子或样例默认采用 asc-devkit
  `examples/01_simd_cpp_api/07_tensor_api/` 官方样例工程结构；kernel 与 host `main()`
  位于同一 `<op_name>.asc` 联合体文件，不默认拆成 .cpp。
- CMake 使用 `LANGUAGES ASC CXX` 构建单个 `demo`；配置期硬检查 run mode 只允许 `npu`/`sim`、
  目标架构必须是 `dav-3510`，不满足直接 FATAL。
- Tensor API 唯一公共入口：#include "tensor_api/tensor.h"。
- 禁止直接 include impl/ 或内部子目录头文件。

## 1.1 默认工程结构

以官方样例目录为模板：

    <op_name>/
      CMakeLists.txt
      <op_name>.asc
      data_utils.h
      README.md
      README_en.md
      scripts/
        gen_data.py
        verify_result.py

CMake 合同：

    cmake_minimum_required(VERSION 3.16)
    set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu, sim")
    set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
    find_package(ASC REQUIRED)
    project(<op_name> LANGUAGES ASC CXX)
    add_executable(demo <op_name>.asc)
    target_compile_options(demo PRIVATE
      $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    )

官方样例还提供两条硬约束：非 `npu/sim` 的 run mode 直接 FATAL；目标架构不是
`dav-3510` 直接 FATAL。依赖 experimental API 的样例必须增加
`CANN_ASC_USE_EXPERIMENTAL=ON` 门禁，并给 ASC 编译加
`-DCANN_ASC_USE_EXPERIMENTAL`。

运行流程：

    mkdir -p build && cd build
    cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..
    make -j
    python3 ../scripts/gen_data.py
    ./demo
    python3 ../scripts/verify_result.py output/output.bin output/golden.bin

切换 `sim` 模式使用 `-DCMAKE_ASC_RUN_MODE=sim`；切换前清理 CMake 缓存。

## 2. CANN 版本矩阵

| CANN | 结论 |
|---|---|
| 9.1.0 | 系统 tensor_api 头过老，缺 copy/mmad 等 trait；可用自带头 |
| 9.2.0-beta.2 | 全链路实证可用 |
| 9.3.0 weekly | 头较新，但 runtime 静态库含新符号，勿用于对外分发产物 |

## 3. 链接与 CMake

1. 9.1.0/9.2.0 的 libascendc_runtime 常见为静态库，链接集需包含 ascendcl、runtime、ascendc_runtime 和框架依赖。
2. Msprof 运行时符号可能依赖宿主进程解析，纯 demo 进程需证明或提供 stub。
3. CMake ≥3.31 时保留 CMAKE_LINK_DEPENDS_USE_LINKER=FALSE，避免 bisheng 链接器收到不支持的参数。
4. find_package(ASC) 使用架构目录 cmake 模块；外层桥接文件可能展开失败。
5. CMake 配置阶段硬检查目标架构，不匹配立即 FATAL。

## 4. 分层验证

| 环境 | 路径 |
|---|---|
| 本地无 NPU / torch_npu | sim 全量内核编译 + DEMO_ONLY npu 编译 |
| 远程 950 | 完整 npu + torch_npu 链路 |
| Kernel-only | 仅证明 kernel 编译/运行，不做端到端结论 |

## 5. include 排查

同名头冲突时查看真实 target 编译命令和预处理路径，再做最小调整。不要盲目添加 -I。必须确认当前 target 的语言、平台和内置头路径。
