# torch_npu 中文文档参考（Reference）

本文档为 **@torch_npu_doc** 提供文档索引与要点，便于在用户请求时引用工作区内 `torch_npu` 项目的中文文档内容。所有路径均相对于项目根或 `torch_npu` 仓库根。

---

## 一、文档索引（按模块）

### 1. 根目录中文文档

| 路径 | 说明 |
|------|------|
| `torch_npu/README.zh.md` | 项目总览、PyTorch/Python 版本配套表、昇腾辅助软件表、环境部署（二进制/源码）、安装后验证、卸载、硬件配套、分支维护策略、参考文档链接 |
| `torch_npu/CONTRIBUTING.zh.md` | 贡献指南：CLA、Fork、测试用例、代码风格、门禁、Fork-Pull、报告问题、提出 PR |

### 2. docs/zh 概览与入门

| 路径 | 说明 |
|------|------|
| `torch_npu/docs/zh/overview/product_overview.md` | Ascend Extension for PyTorch 是什么、总体架构、关键功能特性 |
| `torch_npu/docs/zh/quick_start/quick_start.md` | 快速入门：环境准备、模型迁移训练（GPU→NPU 示例）、混合精度说明 |
| `torch_npu/docs/zh/release_notes/release_notes.md` | 版本说明、PyTorch/CANN/Extension 配套表、组件配套、维护策略 |
| `torch_npu/docs/zh/security_statement/security_statement.md` | 安全声明 |

### 3. docs/zh 框架特性指南（framework_feature_guide_pytorch）

路径前缀：`torch_npu/docs/zh/framework_feature_guide_pytorch/`

| 路径 | 说明 |
|------|------|
| `overview.md` | 框架特性概述 |
| `memory_resource_optimization.md` | 内存资源优化总览 |
| `virtual_memory.md` | 虚拟内存 |
| `memory_snapshot.md` | 内存快照 |
| `custom_memory_allocator.md` | 自定义内存分配器（NPUPluggableAllocator） |
| `multistream_memory_reuse.md` | 多流内存复用 |
| `memory_sharing_ipc.md` | 内存共享（IPC） |
| `communication_performance_optimization.md` | 通信性能优化 |
| `torch_npu_run.md` | torch_npu_run 使用 |
| `ranktable_link_setup.md` | ranktable 建链 |
| `computing_performance_optimization.md` | 计算性能优化 |
| `automatic_core_binding.md` | 自动绑核 |
| `stream_taskqueue_parallel_delivery.md` | Stream 级 TaskQueue 并行下发 |
| `assisted_error_locating.md` | 辅助报错定位（特征值检测、WatchDog） |
| `feature_value_detection.md` | 特征值检测 |
| `watchdog.md` | WatchDog |
| `parameter_setting.md` | 参数配置 |
| `setting_HCCL_communicator_parameter.md` | 通过 pg_options 配置 HCCL 通信域参数 |
| `pytorch_graph_mode.md` | PyTorch 图模式 |
| `pytorch_compilation_mode.md` | PyTorch 编译模式（torch.compile） |
| `custom_operator_adaptation.md` | 自定义算子适配开发总览 |
| `opplugin_operator_adaptation.md` | 基于 OpPlugin 算子适配 |
| `adaptation_overview_opplugin.md` | OpPlugin 适配概述 |
| `adaptation_flow_opplugin.md` | OpPlugin 算子适配流程 |
| `adaptation_preparation_opplugin.md` | OpPlugin 适配前准备 |
| `adaptation_development_opplugin.md` | OpPlugin 适配开发 |
| `adaptation_compile_opplugin.md` | OpPlugin 编译验证 |
| `adaptation_description_opplugin.md` | OpPlugin 适配说明（详细） |
| `sample_call_opplugin.md` | OpPlugin 调用样例 |
| `reference.md` | 常见参考 |
| `c_extensions_operator_adaptation.md` | 基于 C++ extensions 算子适配 |
| `adaptation_description_extension.md` | C++ extensions 适配说明 |
| `single_operator_adaptation.md` | 单算子 API 调用适配 |
| `adaptation_description_single.md` | 单算子适配开发（C++ 结构、注册、前反向、Meta 设备） |
| `sample_call_single.md` | 单算子调用样例 |
| `kernel_launch_operator_adaptation.md` | kernel 直调算子适配 |
| `adaptation_description_kernel.md` | kernel 直调适配开发 |
| `sample_call_kernel.md` | kernel 直调调用样例 |

### 4. docs/zh 故障处理（troubleshooting）

路径前缀：`torch_npu/docs/zh/troubleshooting/`

| 路径 | 说明 |
|------|------|
| `menu_troubleshooting.md` | 故障处理菜单（入口） |
| `troubleshooting_process.md` | 故障处理流程 |
| `usage_instruction.md` | 使用说明 |
| `error_codes_introduction.md` | Error Code 介绍 |
| `ERR-001.md` ~ `ERR-012.md`, `ERR-100.md`, `ERR-200.md`, `ERR-300.md`, `ERR-999.md` | 各错误码说明 |
| `error_information_analysis_guide.md` | 报错信息分析指导 |
| `error_information_introduction.md` | 报错信息分析说明 |
| `error_information_classification.md` | 报错信息分类 |
| `command_output.md` | 回显信息 |
| `plog_log.md` | plog 日志信息 |
| `error_information_analysis.md` | 报错信息分析 |
| `locating_coredump_faults.md` | coredump 问题定位 |
| `troubleshooting_cases.md` | 故障案例集 |
| `communication_operato_transfers_Non-contiguous_tensors.md` | 通信算子传入非连续 tensor |
| `failed_verify_op_parameters.md` | 调用算子参数校验失败 |
| `port_number_distributed_task_in_use.md` | 分布式任务端口号被占用 |
| `variables_used_gradient_computation_modified_by_in-place_op.md` | 用于梯度计算的变量被 inplace 操作修改 |
| `unsupported_op_called.md` | 调用不支持的算子 |
| `HCCL_timeout.md` | HCCL 超时 |
| `operator_called_error.md` | 算子调用报错 |
| `initialization_error.md` | 初始化报错 |
| `communication_domain_link_establishment_timeout.md` | 通信域建链超时 |

### 5. docs/zh 原生 API（native_apis）

按 PyTorch 版本分目录：`pytorch_2-6-0`、`pytorch_2-7-1`、`pytorch_2-8-0`、`pytorch_2-9-0`。每目录下含 `torch.md`、`torch-nn.md`、`torch-distributed.md`、`torch-utils*.md`、`overview.md`、`torch-Tensor.md`、`torch-Storage.md` 等大量 API 说明，用于查询 PyTorch 原生接口在 NPU 上的适配情况。

---

## 二、关键文档内容摘要

### README.zh.md（项目根）

- **版本配套**：PyTorch 与 Python 版本配套表、昇腾辅助软件（CANN 与 Extension 分支/版本对应）。
- **环境部署**：先装 CANN → 再装 PyTorch（x86 用 cpu 版）→ 依赖 pyyaml、setuptools → `pip3 install torch-npu==x.x.x`；源码安装见克隆分支、Docker 构建、编译步骤。
- **验证**：`source set_env.sh` 后执行 `x = torch.randn(2,2).npu(); z = x.mm(y)` 等快速验证。
- **卸载**：`pip3 uninstall torch_npu`。
- **参考文档**：软件安装指南、网络模型迁移和训练、算子适配、PyTorch 原生接口清单、自定义 API 参考（昇腾社区链接）。

### product_overview.md

- Ascend Extension for PyTorch 是使昇腾 NPU 支持 PyTorch 的适配框架。
- 总体架构：torch_npu 插件 + PyTorch 原生/第三方库适配。
- 关键特性：适配昇腾、动态图/自动微分/Profiling/优化器、自定义算子、分布式（单机/多机、集合通信）、ONNX 推理与离线转换。

### quick_start.md

- 环境：驱动与固件、CANN 安装（参见 CANN 安装指南），PyTorch 与 torch_npu 安装参见安装指南。
- 模型迁移：以 CNN+MNIST 为例，将 `device = torch.device('cuda:0')` 改为 `'npu:0'`，数据与模型 `.to(device)`，Atlas 训练系列需开启混合精度；A2/A3 系列可选。

### adaptation_description_single.md（单算子 C++ 适配）

- 基于 C++ extensions，通过 torch_npu 调用单算子 API。
- 目录结构：`csrc/`（add_custom.cpp、registration.cpp、pytorch_npu_helper.hpp）、`custom_ops/`（Python 接口）、`setup.py`、`test/`。
- 步骤：在 registration.cpp 用 TORCH_LIBRARY 注册 schema；在 add_custom.cpp 实现 NPU 前向/反向并绑定（EXEC_NPU_CMD、AutogradPrivateUse1）；可选为 Meta 设备注册实现以便入图；编译安装后 eager/torch.compile 测试。

### adaptation_description_extension.md

- C++ extensions 提供将自定义算子映射到昇腾的能力；单算子 API 调用见 single_operator_adaptation，kernel 直调见 kernel_launch_operator_adaptation。

### custom_memory_allocator.md

- 支持从 so 加载自定义 NPU 内存分配器。
- 接口：`torch_npu.npu.NPUPluggableAllocator(path_to_so_file, alloc_fn_name, free_fn_name)`，`torch_npu.npu.memory.change_current_allocator(new_alloc)`。
- 适用：特殊内存需求、提高内存利用率与训练性能。

### assisted_error_locating.md

- 辅助报错定位包含：特征值检测（feature_value_detection）、WatchDog（watchdog）。

### troubleshooting（故障处理）

- 入口为 menu_troubleshooting.md，包含：故障处理流程、Error Code 介绍与各 ERR-xxx、报错信息分析指导（回显、plog、coredump）、故障案例集（非连续 tensor、参数校验失败、端口占用、inplace 修改梯度变量、不支持算子、HCCL 超时、算子报错、初始化报错、建链超时等）。

### release_notes.md

- 产品版本信息（如 7.3.0）、PyTorch/CANN/Extension 配套表、Python 版本、组件配套（openMind、DrivingSDK）、分支维护策略。

### CONTRIBUTING.zh.md

- 贡献前需签署 CLA；Fork 仓库、阅读 README；开发指导含测试用例、代码风格、门禁、Fork-Pull、报告问题、提出 PR。

---

## 三、使用说明（给 Agent）

当用户使用 **@torch_npu_doc** 时：

1. **优先读取本 reference.md**，根据用户问题定位到上述索引中的对应文档路径。
2. **按需打开工作区内对应文件**（如 `torch_npu/README.zh.md`、`torch_npu/docs/zh/quick_start/quick_start.md`、`torch_npu/docs/zh/framework_feature_guide_pytorch/xxx.md`、`torch_npu/docs/zh/troubleshooting/xxx.md`），提取与问题相关的段落或步骤。
3. **回答时引用文档内容**：安装/验证/卸载用 README.zh.md；迁移与快速体验用 quick_start.md；算子适配用 adaptation_description_* 与 menu_framework_feature；故障与错误码用 troubleshooting 目录；版本与配套用 release_notes 与 README.zh.md。
