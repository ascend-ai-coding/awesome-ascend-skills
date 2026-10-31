# Tensor API 构建与工具链

> 文档与样例事实源：asc-devkit。版本变化后必须复核本文结论。

## 0. 文档与样例事实源

手册或官方样例缺失时运行：

    python3 ops/ascendc-tensor-api-best-practice/scripts/ensure_docs.py

脚本自动 clone 或 fast-forward `~/asc-devkit`；该工作区被非 master 分支占用或与 master 历史分叉时，自动维护 `~/asc-devkit-master` 浅镜像提供手册与样例，不修改主工作区。dirty、origin 不匹配时阻断。`--check-only` 不联网。输出 JSON 的 `selected_root` 为事实源根目录（显式传 `--asc-devkit-root` 时不启用 fallback）。

## 1. 目标与入口

- 目标芯片：Ascend 950PR / 950DT。
- 目标架构：dav-3510。
- ASC 编译参数：--npu-arch=dav-3510 -xasc。
- direct launch kernel 使用 .cpp，不使用 .asc。
- Tensor API 唯一公共入口：#include "tensor_api/tensor.h"。
- 禁止直接 include impl/ 或内部子目录头文件。

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
