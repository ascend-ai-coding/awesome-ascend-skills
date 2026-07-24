# Triton Agent 知识库：GPU Triton → Ascend910_95 NPU 迁移与优化

## 知识库定位

本知识库面向 **Triton Agent**，聚焦 **Ascend910_95**（架构代号 `dav-c310`，Reg-based 架构），提供 GPU Triton 算子迁移到 NPU 的全链路知识。

核心特征：

- **问题导向**：每篇文档以"触发条件"开头，Agent 可直接按场景匹配
- **代码驱动**：所有优化建议均附 GPU→NPU 代码对比，可直接套用
- **910_95 专属**：所有参数、约束、优化策略均以 910_95 为基准，不混用 910B/A2 等其他架构参数

### 关键硬件约束（全文通用）

| 参数 | 值 | 说明 |
|------|-----|------|
| 架构代号 | `dav-c310` | Ascend910_95 系列统一架构 |
| 架构分类 | **Reg-based** | 核间同步使用 SetFlag/WaitFlag，非 FFTS |
| UB 可用容量 | **248 KB** | 256KB - 8KB（编译器预留） |
| L0C 容量 | **256 KB** | Cube 输出缓冲区 |
| L1 容量 | 512 KB | Cube 输入缓冲区 |
| UB 对齐 | **32B** | Vector 算子尾轴须 32B 整除 |
| L0C 对齐 | **512B** | Cube 输出须 512B 整除 |
| AI Core 组成 | 1 Cube + 2 Vector | VectorCoreCount = 2 * AiCoreCount |

---

## 快速导航：按使用场景

| 场景 | 推荐文档 | 说明 |
|------|---------|------|
| **首次迁移 GPU 算子** | [01-migration-overview](01-migration-overview.md) → [02-api-differences](02-api-differences.md) | 先理解架构差异，再查 API 替代 |
| **确认硬件规格/内存容量** | [00-hardware-quick-ref](00-hardware-quick-ref.md) | UB/L1/L0C 容量、对齐、数据通路 |
| **设定 Tiling / Grid 分核** | [03-tiling-and-grid](03-tiling-and-grid.md) | UB 预算公式、32B 对齐、Grid 收缩 |
| **优化内存访问模式** | [04-memory-access-patterns](04-memory-access-patterns.md) | 取余→Mask、Load 重排、指针+=优化 |
| **Cube+Vector 融合算子** | [05-cv-pipeline-optimization](05-cv-pipeline-optimization.md) | CV 流水线、fixpipe 通路、L0C→UB 直通 |
| **性能远低于预期（10x+）** | [06-scalar-degradation-avoidance](06-scalar-degradation-avoidance.md) | 标量降级排查，i64/SIMT 模式降级条件 |
| **配置编译参数** | [07-compile-params](07-compile-params.md) | multibuffer/enable_mixed_cv/sync_solver 等 |
| **数据类型转换/精度保护** | [08-data-type-precision](08-data-type-precision.md) | 类型支持矩阵、累加精度、溢出保护 |
| **迁移 tl.make_block_ptr** | [09-block-pointer-migration](09-block-pointer-migration.md) | stride/order 差异、转置语义、边界检查 |
| **配置 Autotune** | [10-autotune-on-npu](10-autotune-on-npu.md) | 移除 num_stages/num_warps、2 的幂约束 |
| **tl.dot + Bias 融合** | [11-fixpipe-and-bias-fusion](11-fixpipe-and-bias-fusion.md) | FixPipe 随路操作、bias 融合到 Cube 流水线 |
| **多次 tl.store 合并** | [12-store-merge](12-store-merge.md) | 连续 store 合并为单次 MTE3 搬运 |
| **tl.where 单位置优化** | [13-where-optimization](13-where-optimization.md) | where→get_element+insert_slice 替换 |
| **使用 NPU 扩展 API** | [14-compile-hint-and-extension](14-compile-hint-and-extension.md) | compile_hint、Buffer 模型、Ascend 扩展 API |
| **Cube-Vector 同步/Barrier** | [15-sync-and-barrier](15-sync-and-barrier.md) | 三层同步：Pipe Sync / Block Sync / UnitFlag |
| **编译/运行报错排查** | [16-debugging-common-errors](16-debugging-common-errors.md) | UB 溢出、coreDim 超限、死锁、精度异常 |
| **迁移 Flash Attention** | [17-flash-attention-migration](17-flash-attention-migration.md) | 12 个 Diff 点，含前向+反向 |
| **迁移 Fused Matmul** | [18-fused-matmul-migration](18-fused-matmul-migration.md) | Autotune/编译参数/指针算术全链路 |
| **迁移 Fused SwiGLU** | [19-fused-swiglu-migration](19-fused-swiglu-migration.md) | libdevice→tl.fdiv/tl.exp、SPLIT_K 处理 |
| **迁移 RoPE** | [20-rope-migration](20-rope-migration.md) | Block Pointer 2D→3D、Grid 3D→1D |
| **迁移 Softcap** | [21-softcap-migration](21-softcap-migration.md) | 核内 Tiling 子块、multibuffer、VF 融合 |
| **深度性能调优** | [22-advanced-optimization](22-advanced-optimization.md) | 高级优化技巧合集 |

---

## 文档列表

### 第一部分：基础与迁移（00-02）

| 文件 | 主题 | 核心内容 |
|------|------|---------|
| [00-hardware-quick-ref.md](00-hardware-quick-ref.md) | Ascend910_95 硬件速查 | AI Core 规格、内存层次（UB/L1/L0C/L0A/L0B/DCache）、对齐约束、数据通路、TightlyCoupledBuffer |
| [01-migration-overview.md](01-migration-overview.md) | GPU→NPU 迁移全景 | 架构差异（SM vs AI Core）、内存层次对比、并行调度差异、数据搬运流水线、数据类型支持、迁移 7 步流程 |
| [02-api-differences.md](02-api-differences.md) | Triton API 差异速查 | 不支持 API 及替代方案、行为差异 API、NPU 特有扩展 API（al.sort/al.flip/al.fixpipe 等） |

### 第二部分：性能优化（03-14）

| 文件 | 主题 | 核心内容 |
|------|------|---------|
| [03-tiling-and-grid.md](03-tiling-and-grid.md) | Tiling/Grid 分核策略 | UB 容量计算与预算公式、32B 对齐处理、Grid 收缩（≤AI Core 总数）、多行处理提升 SIMD 利用率 |
| [04-memory-access-patterns.md](04-memory-access-patterns.md) | 内存访问模式优化 | 取余→Mask 替代、Load 指令重排序、避免循环内 += 更新指针、入参静态化（tl.constexpr） |
| [05-cv-pipeline-optimization.md](05-cv-pipeline-optimization.md) | CV 流水线优化 | Cube/Vector 分离架构、PIPE 枚举、fixpipe 通路（L0C→UB 直通）、CV 串行→流水线并行、scope(core_mode) |
| [06-scalar-degradation-avoidance.md](06-scalar-degradation-avoidance.md) | 标量降级规避 | 910_95 Reg-based 降级条件（vs 910B）、归约/算术/SIMT 模式降级速查、规避策略 |
| [07-compile-params.md](07-compile-params.md) | NPU 编译参数速查 | 核心参数（enable_flatten/multibuffer/enable_mixed_cv/sync_solver）、辅助参数、compile_mode、精度参数 |
| [08-data-type-precision.md](08-data-type-precision.md) | 数据类型与精度保护 | 完全/部分/不支持类型矩阵、累加精度保护（dot→fp32）、类型转换溢出、FP8 支持 |
| [09-block-pointer-migration.md](09-block-pointer-migration.md) | Block Pointer 迁移 | stride/order 差异（不允许 stride 交换转置）、boundary_check/padding_option、load_if/store_if |
| [10-autotune-on-npu.md](10-autotune-on-npu.md) | NPU Autotune 配置 | 移除 num_stages/num_warps、Config 仅含 Tiling、候选值须为 2 的幂、NPU 编译参数传入方式 |
| [11-fixpipe-and-bias-fusion.md](11-fixpipe-and-bias-fusion.md) | FixPipe 与 Bias 融合 | bias 累加融合到 Cube 流水线、tl.trans+tl.dot 的 fixpipe 随路转置、L0C→UB 直通优化 |
| [12-store-merge.md](12-store-merge.md) | Store 合并优化 | 多次 tl.store→单次连续 MTE3 搬运、合并前提条件、tl.where 整合数据 |
| [13-where-optimization.md](13-where-optimization.md) | Where 条件优化 | 单位置 False 的 tl.where→get_element+insert_slice、离散访存规避 |
| [14-compile-hint-and-extension.md](14-compile-hint-and-extension.md) | compile_hint 与 Ascend 扩展 API | compile_hint 打断 VFusion、insert_slice/extract_slice/get_element、sync_block_set/wait、Buffer 模型 |

### 第三部分：同步、调试与实例（15-22）

| 文件 | 主题 | 核心内容 |
|------|------|---------|
| [15-sync-and-barrier.md](15-sync-and-barrier.md) | 同步与 Barrier | 三层同步架构（Pipe Sync/Block Sync/UnitFlag）、910_95 Reg-based 同步机制、循环中 tl.dot 同步 |
| [16-debugging-common-errors.md](16-debugging-common-errors.md) | 常见编译/运行错误速查 | UB 溢出、coreDim 超限、数据类型不支持、对齐错误、死锁、精度异常、Scalar 退化 |
| [17-flash-attention-migration.md](17-flash-attention-migration.md) | Flash Attention 迁移实例 | 12 个 Diff 点：Autotune/CV 流水线/Softmax 分块/O_L2 缓存/反向 QKV 融合 |
| [18-fused-matmul-migration.md](18-fused-matmul-migration.md) | Fused Matmul 迁移实例 | Autotune/编译参数/指针算术/GROUP_SIZE_M/Softmax 温度缩放 |
| [19-fused-swiglu-migration.md](19-fused-swiglu-migration.md) | Fused SwiGLU 迁移实例 | libdevice→tl.fdiv/tl.exp、指针递增→offset 重算、tl.sum 轴变化、SPLIT_K 处理 |
| [20-rope-migration.md](20-rope-migration.md) | RoPE 迁移实例 | Block Pointer 2D→3D、Grid 3D→1D、stride/order 适配、libdevice.pow 路径 |
| [21-softcap-migration.md](21-softcap-migration.md) | Softcap 迁移实例 | 核内 Tiling 子块→BLOCK_NUM 多块循环、multibuffer API、VF 融合 |
| [22-advanced-optimization.md](22-advanced-optimization.md) | 高级优化技巧合集 | Profiling 方法、编译期优化、性能优化检查清单 |

---

## 与其他知识库的关系

本项目包含三个互补的文档知识库，各有侧重：

```
docs_for_triton_agent/          ← 本知识库
  定位：Agent 迁移实操手册
  特点：问题导向 + 代码驱动 + 910_95 专属
  受众：Triton Agent（自动迁移/优化）

docs_triton_ascend/             ← Triton-Ascend 开发者文档
  定位：Triton-Ascend 完整技术文档
  特点：体系化教学 + API 参考 + 编译流程
  受众：人类开发者 + Agent
  回答：如何使用 Triton-Ascend

docs_ascendnpu_ir/              ← AscendNPU-IR 编译器文档
  定位：IR 层精确定义与变换规则
  特点：MLIR 方言语义 + 编译 Pass + 操作规范
  受众：编译器开发者
  回答：编译器如何实现
```

### 交叉引用关系

| 本知识库文档 | docs_triton_ascend 对应 | docs_ascendnpu_ir 对应 |
|-------------|------------------------|----------------------|
| 00-hardware-quick-ref | 01-Programming-Model/03-memory-model | 00-Architecture/02-memory-hierarchy |
| 02-api-differences | 02-Core-API/* + 09-Reference/01-api-support-matrix | 01-HIVM-Dialect/02-Vector-Operations/* |
| 05-cv-pipeline-optimization | 03-Ascend-Extensions/02-pipe-and-core + 05-Performance-Optimization/05-cv-fusion | 00-Architecture/03-pipeline-execution-model |
| 06-scalar-degradation-avoidance | - | 01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering |
| 11-fixpipe-and-bias-fusion | 03-Ascend-Extensions/03-fixpipe | 01-HIVM-Dialect/01-DMA-Operations/06-fixpipe |
| 14-compile-hint-and-extension | 03-Ascend-Extensions/* | 01-HIVM-Dialect/04-Synchronization/* |
| 15-sync-and-barrier | 03-Ascend-Extensions/04-sync-operations | 01-HIVM-Dialect/04-Synchronization/* |

### 使用建议

- **迁移算子**：优先查阅本知识库（docs_for_triton_agent），快速定位 Diff 和代码模式
- **理解 API 语义**：查阅 docs_triton_ascend，获取完整的 API 参考、参数说明和示例
- **排查编译器行为**：查阅 docs_ascendnpu_ir，理解 IR 降级路径和操作语义
- **深度调优**：三库交叉引用，从本库的优化策略追溯到 IR 层的实现原理

---

## 文档模板

每篇文档遵循统一结构，便于 Agent 按步骤匹配和检索：

```
# [主题标题]

## 触发条件
  Agent 何时需要参考本文档？列出具体场景和代码模式。

## 核心知识
  结构化呈现关键信息（表格/公式/架构图），Agent 可直接提取。

## 代码模式
  GPU→NPU 代码对比，标注关键改动点，Agent 可直接套用。

## 910_95 特别注意
  与 910B/A2 的差异、910_95 专属约束和优化机会。

## 相关文档
  交叉引用链接（本库 + docs_triton_ascend + docs_ascendnpu_ir）。
```

### 模板设计原则

| 原则 | 说明 |
|------|------|
| 触发条件先行 | Agent 按场景匹配文档，触发条件必须精确、可枚举 |
| 表格优先 | 规格参数、差异对比、降级条件等用表格呈现，便于 Agent 解析 |
| 代码即答案 | 每个优化点提供 GPU→NPU 代码对比，减少 Agent 推理步骤 |
| 910_95 隔离 | 明确标注 910_95 与其他架构的差异，避免 Agent 误用 910B 参数 |
| 源码可溯 | 关键结论标注源码文件和行号，Agent 可验证或深入查阅 |
