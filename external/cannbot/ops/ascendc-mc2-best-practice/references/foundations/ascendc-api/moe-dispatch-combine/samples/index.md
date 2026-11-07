# MoE Dispatch/Combine Samples

这个专题模块承接算子生成中的 `samples` 路径，负责沉淀 sample 可复用的工程 blueprint：host 侧多卡调用方式、工程编译方式、算子验证链路，以及这些职责对应的文件落点。

## 适用场景

- 还没补齐芯片、EP、expert、quant 和关键输出中间量规格
- 还没决定工程采用什么目录分层、编译入口和文件落点
- 还没说明 host 侧如何初始化多卡环境、创建通信资源并在多卡上调用 kernel
- 还没说明工程如何编译、如何做结果校验、如何组织验证链路
- 还在梳理 dispatch / combine 的阶段语义、接口参数和中间量闭环
- 需要对照默认样例工程，确定哪些文件要参考、哪些文件要改

## 不适用场景

- `DataCopyPad`、`SetValue/GetValue`、状态发布/等待和 cache 可见性细节问题：`../api-rules/index.md`
- window 物理布局、状态槽切分、workspace 组织和分核边界问题：`../tiling-scheme/index.md`

dispatch/combine 的整体设计背景、阶段协作关系和方案动机说明见 `../reading/design-overview.md`。

## 进入条件

本模块的前置条件不包含单核实现或分核设计完成；其进入条件是当前内容仍处于规格补齐、工程对齐、接口语义对齐或文件落点判断阶段。

## 在三部分中的位置

生成路径默认拆成三部分：

1. `samples`：给出样例工程可复用的工程 blueprint，包括 host 多卡调用方式、构建方式、验证链路、接口语义和改动落点
2. `api-rules`：补齐 MoE 场景下的 API 约束、共享访问和状态协议规则
3. `tiling-scheme`：设计 window 布局、分阶段分核、共享 workspace 和跨核统计方案

本模块只覆盖第 1 部分；API 细节归入 `../api-rules/index.md`，分核方案设计归入 `../tiling-scheme/index.md`。

## 工程起点

- `moe_dispatch_direct_invoke_sample/` 主要作为工程 blueprint 参考，尤其用于说明 host 多卡调用方式、编译入口和验证链路
- 不要回退到通用单文件 `.asc` 模板
- 工程实现既可在现有项目中增量修改，也可按目标需求独立组织

## 这个 blueprint 要说明什么

`samples` 路径记录的是工程 blueprint，而不是代码阅读提示。该路径覆盖的工程事实包括：

1. host 侧如何初始化设备、多卡通信域、stream 和通信资源，并把 `mc2Context` 传给 kernel
2. kernel 入口、tiling 数据、通信上下文、主流程代码分别落在哪些文件
3. 工程如何通过 `CMakeLists.txt`、`build.sh` 和依赖库完成编译
4. 验证链路如何组织，包括 host 测试、结果校验脚本和 README 中的运行说明

这四项共同构成 `samples` 路径的职责边界。

## 工程形态

- 允许并推荐 `host cpp + kernel h + include + test + scripts` 多文件结构
- 不按通用 `add.asc` 样例审查工程
- 可以参考样例工程现有命名和分层，例如 `moe_dispatch.cpp`、`kernel/moe_dispatch.h`、`kernel/mte_dispatch_comm.h`、`include/tiling_data.h`、`test/test_moe_dispatch.cpp`、`scripts/verify_dispatch.py`，并按当前实现需求抽取合适的组织方式

## 编译与交付口径

- host 侧 launch 路径、多卡通信初始化、`CMakeLists.txt` / `build.sh` 以及 ACL/HCCL 依赖属于本路径的编译说明范围
- 验证链路包括 host 测试入口、校验脚本和运行时依赖
- 不要为了套通用工作流，强行改造成单文件 `.asc` 编译路径
- 交付件至少检查：构建文件、kernel 主流程、通信/tiling 头文件、host 测试或 launch、结果校验脚本、README/说明文档

## 本模块只回答四件事

1. 先看 `spec-template.md`，补齐芯片、EP、expert、quant、输出中间量（`expandIdx`、`epRecvCounts`、`epSendCounts`、`expertTokensCountPerRank`）
2. host 侧多卡通信域初始化、通信资源创建、`mc2Context` 传递和 kernel launch 路径；工程结构与编译细节见 `build-framework.md`
3. 目录分层、构建方式、验证链路和关键文件落点；接口语义和中间量闭环分别见 `dispatch-dataflow.md` 或 `combine-dataflow.md`
4. sample 内 helper 分层、职责和文件落点见 `sample-helper-map.md`；API 规则见 `../api-rules/index.md`；分核方案见 `../tiling-scheme/index.md`

## 详细页面

- 编译框架特需说明（mc2 特有依赖、通信资源创建、平台宏）：`build-framework.md`
- 规格模板：`spec-template.md`
- dispatch 流程说明：`dispatch-dataflow.md`
- combine 流程说明：`combine-dataflow.md`
- sample helper 分层与查找：`sample-helper-map.md`
- 改动定位：`change-routing.md`

## Review 口径

- 文件扩展名、命名形式等规范问题可以按 MoE 特殊性处理
- 配置参数、算法逻辑、通信资源、输出布局、结果校验一致性必须严格验证
- 对代码做修改后，注释和文档也应同步更新，确保反映当前实现

## 不要做的事

- 不要完全脱离已有参考工程和编译链路，直接按通用 `.asc` 模板生造 dispatch/combine 工程
- 不将 `samples` 路径退化为纯代码阅读入口；host 调用方式、编译方式和验证链路均属于该路径的显式知识范围
- 不要一开始同时重写 host、上下文辅助层、comm、kernel 全链路

## 下一跳

- 进入 API 细节：`../api-rules/index.md`
- 进入分核方案：`../tiling-scheme/index.md`
- 生成受阻时补背景：`../reading/design-overview.md`
- 阅读已有实现：`../reading/`