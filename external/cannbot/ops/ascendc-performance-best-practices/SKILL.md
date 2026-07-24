---
name: external-cannbot-ops-ascendc-performance-best-practices
trigger: query
description: Ascend C 算子性能优化最佳实践库。按算子族组织优化经验与参考代码总结，供性能优化实施阶段查询。触发：查询某类算子的性能优化参考实现、实施某项优化时需加载对应优化经验时。
original-name: ascendc-performance-best-practices
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# Ascend C 算子性能优化最佳实践

## 算子分类体系

按 **算子族（operator family）** 组织优化知识，同一族内所有变体共享同一份文档（如 `matmul`、`matmul_mxfp4`、`batch_matmul`、`matmul_a16w16` 等均归入 `matmul` 族；`matmul_all_reduce`、`allgather_matmul`、`matmul_reducescatter`、`alltoall_matmul` 等均归入 `mc2` 族）。

| 类别 | 典型算子 | 适用架构 | 优化设计指南 |
|------|---------|---------|------------|
| MatMul 矩阵乘类 | MatMul, BatchMatMul, MatMul_MXFP4, MatMul_A16W16 | DAV_3510 | ✅ [性能优化指南](reference/matmul/guide.md) |
| MC² 通算融合类 | matmul_all_reduce, allgather_matmul, matmul_reducescatter, alltoall_matmul | DAV_3510 | ✅ [性能优化指南](reference/mc2/guide.md) |
| RadixSort 基数排序类 | TopK, KthValue, Sort, ArgSort, ArgMax/Min | DAV_2201 / DAV_3510 | ✅ [性能优化指南](reference/sort/radix_sort.md) |
| Scalar 编码与诊断 | 任意 ScalarBound 算子 | DAV_2201 / DAV_3510 | ✅ [性能优化指南](reference/scalar/guide.md) |
| Reduction 归约类 | ReduceSum, Softmax, LayerNorm, RMSNorm, ArgMax | DAV_3510 | ✅ [实现索引](reference/reduce/guide.md) · [模板代码与使用指南](reference/reduce/templates/usage_guide.md) · [公共基类](reference/reduce/templates/dav310/softmax_v2_base.template) · [State Resident](reference/reduce/templates/state_resident_design.md) |
| Elementwise 逐元素类 | Sin, Cos, Abs, Exp | DAV_3510 | ✅ [双缓冲设计](reference/elementwise/double_buffer_design.md) · [向量化效率优化](reference/elementwise/vector_efficiency_design.md) |
| Broadcast 广播类 | Add, Mul, Sub 等含广播轴算子 | DAV_3510 | ✅ [性能优化指南](reference/broadcast/broadcast_design.md)（选型型，见结构 B） |
| Conversion 数据转换类 | Transpose, Concat, Split | DAV_3510 | ✅ [Transpose融合设计](reference/conversion/transpose_fusion_design.md) |
| Convolution 卷积类 | Conv2D, DepthwiseConv | — | 📋 规划中 |
| NN 神经网络类 | FlashAttention, GroupNorm | — | 📋 规划中 |
| Random 随机类 | RandomUniform, Dropout | — | 📋 规划中 |
| SIMT 线程级算子 | 条件分支、离散索引等不规则操作 | DAV_3510 | ✅ [性能优化指南](reference/simt/optimization-guide.md) |

> 未收录的算子族返回「该算子族优化知识暂未收录」。各族详细的优化类型、叠加关系、选型决策见该族 `reference/<family>/` 目录。
> 
> **融合算子的优化**：对于融合算子（如 AddRelu、MulSub 等组合算子），可将其拆解为上述基础算子族，分别获取各基础算子的性能优化实践后，结合融合场景进行适配。例如 AddRelu 可拆解为 Broadcast（Add）+ Elementwise（Relu），分别参考对应族的优化指南。

## 公共优化能力（跨算子族通用）

| 优化类型 | 适用场景 | 文档 |
|---------|---------|------|
| **尾块处理（Tail Block）** | 数据量不能被 tile 大小整除的场景 | ✅ [尾块处理指南](reference/common/tail_block_design.md) |
| **数据搬运（DataCopy）** | 非对齐、小批量多次搬运等的场景 | ✅ [数据搬运](reference/common/datacopy_optimization_design.md) |
| **UB/TBuf常驻复用与Bank冲突规避** |  大量tile/loop都重复从GM搬运，会造成大量冗余MTE2开销的场景 | ✅ [UB/TBuf常驻复用与Bank冲突规避](reference/common/ub_resident_design.md |

## 查询方式

| 输入 | 必需 | 说明 |
|------|------|------|
| 算子名 | 是 | 如 `matmul` / `matmul_mxfp4` / `batch_matmul` |
| 优化类型 | 否 | 如 `pingpong` / `swat` / `streamk` / `fullload` / `scale_coalescing` / `mte2_preload`；不提供则加载全部 |

查询流程：**算子名 → 映射到算子族 → 定位 `reference/<family>/` → 按优化类型筛选文档**。算子族映射规则：精确匹配族名直接命中；以族名为前缀或核心词（如 `matmul_mxfp4`、`batch_matmul`）归入该族；其他形态由调用方按功能显式指定。

### 模板代码查找链路

当算子族提供可复用的模板代码时（如 Reduction 族的 5 个 Softmax 模板），按以下链路逐层查找：

```
SKILL.md（本文件）
  → 算子分类表中定位算子族，点击 guide.md 链接
    → reference/<family>/guide.md
      → 模板表中选择方案对应的模板，点击 .md 链接
        → reference/<family>/templates/<模板名>.md
          → 说明文档中链接同名 .h / .cpp / .template 代码文件
            → reference/<family>/templates/<模板名>.{h,cpp,template}（以此为代码骨架实施）
```

> **模板代码文件类型**：`.h`、`.cpp`、`.template` 都是模板代码。`.template` 是参考模板源码（如 `reference/reduce/templates/dav310/*.template`），使用前需按该目录 `usage_guide.md` 将 `.template` 转成同名 `.h`/`.cpp`（去掉 `.template` 后缀），转换后才可读、可编译。

> ⚠️ 查到模板代码（`.h`/`.cpp`/`.template`）后，**必须直接拷贝模板文件到项目中使用**，而非"参考模板从头重写"。具体操作：
> 1. 若为 `.template`，先按 `usage_guide.md` 转成 `.h`/`.cpp`；再将文件拷贝到项目的 `op_kernel/` 目录（保留 `dav310/` 等子目录结构）
> 2. 仅做**最小适配**使其编译通过（如命名空间别名、TilingData 字段补充等）
> 3. 在 kernel.cpp 中 `#include` 模板头文件，按分支条件 dispatch 到对应模板类
> 4. 模板内的 MicroAPI VF 计算（RegTensor、MaskReg、LoadAsFp32、StoreFromFp32 等）**必须原样保留**
>
> **常见编译修复**（遇到 MicroAPI 报错时按此顺序尝试，禁止直接降级为高层 API）：
> - **命名空间别名**：模板可能用 `AscendC::MicroAPI::`，而 CANN 安装中实际为 `AscendC::Reg::`。加一行 `namespace MicroAPI = Reg;` 即可解决
> - **API 签名适配**：对照 CANN 实际头文件（`$ASCEND_HOME_PATH/x86_64-linux/asc/include/basic_api/reg_compute/`）调整参数数量/类型
> - **头文件 include 路径**：确保 `kernel_reg.h` 或 `kernel_reg_compute_*.h` 被 `kernel_operator.h` 正确引入
>
> 禁止查到模板后仍从零编写 kernel。

## 性能优化文档结构（两类，均为性能优化，按算子族性质选其一）

> 两类都是"让算子更快"的最佳实践，区别只在优化的形态：A 是对单一实现调参，B 是在多种实现间选对路径。

### A. 单实现调优型（对单一实现做某项性能优化，如 matmul pingpong/swat）

每份 `<优化类型>_design.md` 的章节组织：

**必选章节：**

1. 优化目标 —— 效果与量化收益（kernel μs / MTE2 段 / CUBE busy 等）
2. 架构概览 —— 存储层级、数据流、buffer 布局、事件同步模型
3. 关键参数 —— 新增 / 调整字段与 Host 侧计算
4. 核心计算循环 —— 改造前后对照（含事件同步）
5. 优化的关键修改点 —— 表格形式

**可选章节：**

6. 注意事项 / 约束 —— 前置条件、L1/L0 预算、边界与兼容性
7. 实施常见问题与解决方案 —— 高频踩坑与根因
8. 实测性能、选型决策、与其他优化的叠加关系、自检清单

### B. 多实现选型型（族内有多种实现路径，选对即性能最优，如 `broadcast`）

这类族的性能优化**本质就是按场景选对实现**——`broadcast` 的 OneDim 前置快路径 + tile 内 NDDMA / DataCopyPad / UB 三类实现，选错直接慢甚至出错（nan）；选对才快。这同样是性能最佳实践，只是优化点在"选型"而非"调参"。结构：

1. `<family>_design.md` 总览 —— 速查表、**选型决策树**（决定性能的核心）、tiling/切分逻辑、约束
2. 每种实现一份分册（如 `nddma_design.md`）—— 适用条件、Kernel 写法、注意
3. `code/` 子目录 —— 可直接参考/裁剪的 AscendC 范式源码（`.h/.cpp`）
4. 进阶/边界单列（如 `advanced_tiling.md`）

> 选 B 时"收益"体现为**选对实现避免劣化**，可不强制量化 μs，但必须给出**选型依据**（什么场景用哪种实现、为什么快/为什么错）。

## 扩展新算子族

1. 创建 `reference/<family>/` 目录（以族名命名，非单个变体）
2. 判断族性质：优化型 → 按 A 写 `<优化类型>_design.md`；实现指导型 → 按 B 写总览+分册(+code/)
3. 更新本 SKILL.md 分类表格

## 依赖

无外部第三方依赖。知识主要以 Markdown 文档内置；实现指导型算子族（如 `broadcast`）另含
`reference/<family>/code/` 下的 AscendC 范式源码（`.h/.cpp`），随 skill 一并版本化、不依赖工作区外部目录。
