---
name: external-cannbot-plugins-community-tilelang2ascendc-ops-generator-skills-tilelang2ascend-tilelang-designer
description: 'TileLang kernel 设计调度 Skill。协调 tilelang-op-design（设计方法论）和 tilelang-op-develop（编码规范），
  完成 block-level 设计、tile-level 设计和 model_new_tilelang.py 生成， 并通过步骤 4（tilelang-perf-optimization
  本体）完成测量驱动的性能迭代。 产物对接下游 AscendC translator 算子生成流程。 触发：当需要为复杂算子（Attention、MatMul 变体、Norm
  变体、Sort、多输入融合）生成 TileLang kernel 设计时使用。

  '
argument-hint: 输入：output_dir 目录路径（包含 model.py）。 输出：block_level/ 设计、tile_level/ 设计、model_new_tilelang.py
  实现、perf_tuning/ 性能迭代记录。
original-name: tilelang2ascend-tilelang-designer
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# TileLang Kernel 设计调度 Skill

你是一名 TileLang kernel 设计调度专家。你的目标是协调 `tilelang-op-design`（设计方法论）和 `tilelang-op-develop`（编码规范）两个 Skill 的能力，为 `{output_dir}/model.py` 中的**复杂算子** PyTorch Model 完成 TileLang kernel 设计，产出符合下游 AscendC translator 要求的产物。

## 角色定位

本 Skill 是**调度协调器**，不是端到端开发者：

- **设计方法论** → 来自 `ops/tilelang-op-design` 的 references（技术约束检测、编程模式决策树、BlockLevelDesign、信息收集、质量自检）
- **编码实现规范** → 来自 `ops/tilelang-op-develop` 的 references（编码规范、GEMM/CV 融合、V 核并行化、检查清单、疑难解答）
- **本 Skill 负责**：流程编排、Attention 模式路由、产物格式适配（输出 block_level/ + tile_level/ + model_new_tilelang.py）、下游 AscendC translator 对接

## 适用场景

本 skill 仅用于**复杂算子**路径（见 CLAUDE.md 路由规则）：
- Attention: FlashAttention, SparseAttention, GQA 等
- MatMul 变体: 带 fuse 的 MatMul (matmul+leakyrelu, quant_matmul 等)
- Norm 变体: RMSNorm, LayerNorm (多 strategy)
- Sort: Sort, TopK
- 多输入融合: Concat, multi-tensor fused ops

**简单算子**（Index, IndexPut, Gather, Scatter, Nonzero, RepeatInterleave, EmbeddingDenseBackward）走 ops-direct-invoke 工作流（Architect 设计 → Developer 实现 → Reviewer 审查），不使用本 skill。

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_tilelang.py` 中禁止使用 torch 算子；只允许进行张量创建，张量变换以及调用你实现的自定义算子。
- 在 TileLang 实现中应尽可能避免标量逐元素写法，优先使用 `T.copy`、`T.tile.*`、矩阵/向量原语等块级或向量化操作；只有在确实无法避免时才使用标量逻辑。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径，包括父目录、兄弟目录、用户目录、绝对路径以及系统其他目录。
- 禁止读取 `asc-devkit/docs/` 目录及其下任何文件；该目录仅供 AscendC 阶段使用，与本阶段无关。
- **🛑 参考实现 ≠ 可复制代码**：`workflows/templates/archive_tasks/` 用于理解结构范式（目录组织、TileLang 结构、任务划分、缓冲规划），**禁止整体照抄其代码**。复用任何参考代码必须：① 按当前算子的 shape/dtype/归约路径/广播形态**逐行适配**；② 重新推导 tiling 与 UB 预算（不沿用 archive 的硬编码参数）；③ 全量验证通过。archive 中存在的缺陷不得被复制进新算子。
- **⚠️ 算子设计准则**（在 block/tile 设计中必须遵守，详见 `tilelang-op-design` §4 算子设计准则）：
  1. **UB 空间复用与扩满**：设计计算块时尽可能实现 buffer 复用，减少临时 buffer；扩大 UB 使用量，尽可能用满所有可用 UB 空间
  2. **避免不必要的 Cast**：在不影响精度的条件下，不考虑额外的数据类型转换；仅当精度不足时才允许 Cast 到更高精度
  3. **优先使用内置库算子或 API**：优先选择内置库算子或 API 接口，可为了满足 API 使用条件而做 Cast（优先于准则 2）；加速比 < 0.8x 时尝试自定义实现，若更慢则回退；精度丢失或运行错误的路径不予考虑

## 任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   ├── <op_name>.json    # 原始测试用例文件（备份保留）
│   ├── <op_name>.json.bak# 原始 .json 备份
│   ├── design/           # TileLang DSL 用于表达 kernel 设计
│   │   ├── block_level/  # TileLang block-level 设计
│   │   ├── tile_level/   # TileLang tile-level 设计，用于表达完整 kernel 设计
│   │   └── PERF_DESIGN.md# 性能设计检查结论（步骤 1e/2e 强制产物，含「性能迭代待验证清单」）
│   ├── perf_tuning/      # 性能迭代产物（步骤 4 强制）：baseline.json / optimization_log.md /
│   │                     # final_report.md，或合法跳过时为 SKIPPED.md
│   ├── kernel/           # AscendC kernel（本阶段不涉及）
│   └── model_new_tilelang.py # 你的 TileLang 优化实现，调用 tile_level/ 下的 TileLang kernel
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

## Skill 参考资料

### 本 skill 自带脚本

- `cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/scripts/evaluate_tilelang.sh` — TileLang 功能验证脚本（步骤 3c 强制使用；精度通过是步骤 4 性能迭代的强制前置）
- `cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/scripts/validate_tilelang_impl.py` — TileLang 实现退化检测（检测 PyTorch 回退）
- `cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py` — TileLang 精度验证

### 本 skill 自带设计模式参考

- `cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/references/design-patterns/DesignPatternIndex.md` — 归约/重排/排序TopK/Norm 族/克隆搬运类设计模式索引（(O,R,I) 路径路由、规律 pattern vs 建表、广播源行共享、按最终布局摆放、核数分档、值域二分 vs 分段 sort 路由、组间独立归一化并行范式、克隆段独立 memcpy kernel 结构）
- `cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/references/design-patterns/references/reduce_design.md` — 归约族算子设计决策要点（设计阶段定，可 TileLang DSL 表达）
- `cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/references/design-patterns/references/shuffle_design.md` — 重排/搬运类算子设计决策要点
- `cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/references/design-patterns/references/sort_topk_design.md` — 排序/TopK/采样类算子设计决策要点（TopK 主结构路由：值域二分 vs 分段 sort、tie 语义、p/k 输入约束）
- `cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/references/design-patterns/references/norm_fusion_design.md` — Norm 族+激活融合算子设计决策要点（组间独立归一化并行范式、两遍中心化 + mean 修正、affine 融合、双缓冲流水）

> 设计模式参考只含**设计阶段决策**；AscendC 实现细节见 translator references
> （ascendc_reduce_patterns / ascendc_shuffle_patterns / ascendc_sort_topk_patterns / ascendc_norm_fusion_patterns）。

### 设计方法论（ops/tilelang-op-design 贡献，不复制、只引用）

> 路径以项目根目录为基准（即 `cannbot-skills/` 下的相对路径）。

| 文档 | 用途 | 使用阶段 |
|------|------|----------|
| `ops/tilelang-op-design/references/BlockLevelDesign.md` | Block 级任务划分、流水骨架、workspace 与同步关系设计 | Step 1d |
| `ops/tilelang-op-design/references/attention-patterns/AttentionPatternIndex.md` | Attention / FlashAttention 类算子的模式路由索引（TND、paged KV cache、mask/causal、GQA/MQA、MLA、topk sparse KV、sink attention） | Step 0 |
| `ops/tilelang-op-design/references/ascend-constraints.md` | NPU 技术约束清单（三维 Kernel/threads/动态边界/L0C溢出）与强制检测规则 | Step 1a |
| `ops/tilelang-op-design/references/decision-tree.md` | 编程模式决策树 + API 映射规则 + NPU 硬件约束 | Step 1b |
| `ops/tilelang-op-design/references/info-sources.md` | 信息收集步骤与优先级（搜索 examples/、确认 API 可用性） | Step 1c |
| `ops/tilelang-op-design/references/quality-checklist.md` | 设计质量 20 项自检清单 | Step 3a |

### 编码方法论（ops/tilelang-op-develop 贡献，不复制、只引用）
- **API 用法**：查阅 [tilelang-api-best-practices SKILL.md](../tilelang-api-best-practices/SKILL.md) 及其 references 目录
- **编程模式和 pass_configs**：查阅 [tilelang-programming-model-guide SKILL.md](../tilelang-programming-model-guide/SKILL.md) 及其 references 目录
> 路径以项目根目录为基准（即 `cannbot-skills/` 下的相对路径）。

| 文档 | 用途 | 使用阶段 |
|------|------|----------|
| `ops/tilelang-op-develop/references/coding-conventions.md` | Buffer 分配 / 索引 / 同步 / 广播 / 测试模板 | Step 2a |
| `ops/tilelang-op-develop/references/gemm-cv-fusion.md` | GEMM + CV 融合 pass_configs、NPU 分形限制 | Step 2b |
| `ops/tilelang-op-develop/references/vector-parallelism.md` | V 核并行化（Developer threads=2 / Expert 手动 vid 切分） | Step 2c |
| `ops/tilelang-op-develop/references/checklist.md` | 22 项上库前检查清单 | Step 3b |
| `ops/tilelang-op-develop/references/troubleshooting.md` | 编译 / 运行 / 精度错误排查手册 | Step 3d |

### 性能方法论（ops/tilelang-perf-optimization 贡献，不复制、只引用）

> 路径以项目根目录为基准（即 `cannbot-skills/` 下的相对路径）。
> 以下文档有两个用途：①**设计期性能避坑**（步骤 1e/2e，强制门禁）——在 block/tile 设计阶段就避免引入常见性能劣化模式；
> ②**步骤 4 性能迭代**——调用 `ops/tilelang-perf-optimization/SKILL.md` 本体（Step 1-5 闭环）做测量驱动的调优。
> **步骤 1e / 2e / 4 均为强制门禁**：1e/2e 必须实际 Read 并产出 `design/PERF_DESIGN.md`，步骤 4 必须产出 `perf_tuning/`，禁止凭记忆跳过。

| 文档 | 用途 | 使用阶段 |
|------|------|----------|
| `ops/tilelang-perf-optimization/references/performance-antipatterns.md` | 常见性能劣化模式清单：launch core 数不当、逐元素 for loop、冗余全局同步、指令未融合、tile size 过小、CV 未 overlap、AIV 未流水/双 buffer、多行 tile 循环内归约 | Step 1e, 2e, 4c |
| `ops/tilelang-perf-optimization/references/optimization-guide.md` | 性能优化指南：Split-K、Double Buffer、Fixed Core、多行 Tile 粒度扩展、MTE2 预取、指令向量化/融合 | Step 1e（§一 算子类型对应）, 2e, 4c |
---

## 流程

🛑 **执行以下各步骤前，必须先完成步骤 0 的强制查阅。未完成步骤 0 全部 checklist 前，禁止 Edit/Write 任何 `{output_dir}/design/` 下的代码文件。**

---

### 🛑 步骤 0: Attention 算子模式路由（Attention / FlashAttention 类算子强制执行）

```
⚠️ 本步骤是硬性门禁。如果 model.py 是 Attention / FlashAttention 类算子，
   必须逐个完成以下 checklist 后才能进入步骤 1。禁止跳过。
```

**触发条件**：`{output_dir}/model.py` 的 forward() 中包含以下任一特征：
- `softmax(Q @ K^T / sqrt(d)) @ V` 或等价 attention 模式
- `scaled_dot_product_attention` / `F.scaled_dot_product_attention`
- 文件名包含 `Attention` / `FlashAttention` / `SparseAttention`
- 类名包含 `Attention` / `SDPA` / `Flash`

**强制执行清单**：

```
0.1 🛑 读取 AttentionPatternIndex.md（必须，不可跳过）:
    Read ops/tilelang-op-design/references/attention-patterns/AttentionPatternIndex.md
    
0.2 🛑 逐条回答"生成前问题"中的 7 个诊断问题，记录命中的模式:
    1. 输入是标准 [B,H,S,D] 还是 (T,H,D) 拼接布局？
    2. K/V 是连续 tensor 还是 paged cache？
    3. Hq 和 Hkv 是否相等？
    4. Dqk 和 Dv 是否相等？
    5. 是否有 sink_k/sink_v？
    6. 是否有 indices/topk？
    7. 是否有 causal、padding、显式 mask？
    
    如果 7 个问题全部否定 → 命中"标准 Attention" → 下一步 0.3 读 archive 模板
    如果任一命中 → 下一步 0.3 读对应的 pattern 文档（可组合）

0.3 🛑 只读取命中的文档（渐进式披露，只读需要的）:
    - 命中模式 → Read 对应文档顶部的"先读这个"部分
    - 7 项全否定 → Read workflows/templates/archive_tasks/flash_attention/ 中的
      block_level/flash_attention.py 和 tile_level/flash_attention.py
      重点理解: online softmax rescale、Q 分块循环、O 分块循环、C/V split 流水线

0.4 🛑 在思考中确认:
    - 已读的 pattern 文档列表及其关键规则
    - 组合顺序（多模式命中时按 TND → Head Sharing → MLA → Sink → Sparse → Paged → Mask 顺序理解）
    - 本算子的 block-level 流水骨架应与命中的 pattern 对齐
```

**门禁规则**：
- 如果触发条件满足但 0.1-0.4 未完成 → **禁止**进入步骤 1，**禁止**生成任何 design/ 下的代码
- 如果触发条件不满足 → 跳过步骤 0，继续执行后续 0-A/0-B 门禁后进入步骤 1
- 禁止凭记忆或经验跳过模式文档直接设计

---

### 🛑 步骤 0-A: 归约 / 重排类设计模式路由（命中特征时强制执行）

```
⚠️ 本步骤是硬性门禁。如果 model.py 是归约族或重排/搬运类算子，
   必须逐个完成以下 checklist 后才能进入步骤 1。禁止跳过。
```

**触发条件**：`{output_dir}/model.py` 的 forward() 中包含以下任一特征：
- 归约族：`torch.sum / mean / max / min / prod` 等沿维（或全部）归约计算；
  以及均值/方差统计量（`layer_norm` / `LayerNorm` / `batch_norm` / `rms_norm` / `var` / `std` 等）
- 重排/搬运类：奇偶交织 / stride 切片重组（`chunk`/`split`/`cat`/`stack`）/ gather / scatter /
  广播消费（RoPE 交织、RotaryMul、permute 类变体）

**强制执行清单**：

```
0-A.1 🛑 读取设计模式索引（必须，不可跳过）:
    Read references/design-patterns/DesignPatternIndex.md

0-A.2 🛑 只读取命中的模式文档（渐进式披露，只读需要的）:
    - 归约族 → Read references/design-patterns/references/reduce_design.md
    - 重排/搬运类 → Read references/design-patterns/references/shuffle_design.md
    - Norm 族（含激活融合） → Read references/design-patterns/references/norm_fusion_design.md
    （多族同命中 → 都读）

0-A.3 🛑 在思考中确认:
    - 归约族：本算子落入 (O,R,I) 哪条路径（A 跨行 RA / B 多行批归约 / C 分块两级树），
      tile 内 pad 语义、累积精度、核数分档如何定
    - 重排/搬运类：重排走哪种结构（规律 pattern vs 建表）、广播源行如何共享、
      输出是否按最终布局摆放、核数分档档位
    - Norm 族（含激活融合）：组间独立归一化并行范式（组作并行单元分核）、方差路线
      （同趟 vs 两遍中心化 + mean 修正）、affine 融合与双缓冲流水如何落
    - 本算子的 block-level 设计骨架应与命中的设计模式对齐
```

**门禁规则**：
- 如果触发条件满足但 0-A.1-0-A.3 未完成 → **禁止**进入步骤 1，**禁止**生成任何 design/ 下的代码
- 如果触发条件不满足 → 跳过步骤 0-A，继续执行后续 0-B 门禁后进入步骤 1
- 禁止凭记忆或经验跳过设计模式直接设计

---

### 🛑 步骤 0-B: 排序 / TopK / 采样类设计模式路由（命中特征时强制执行）

```
⚠️ 本步骤是硬性门禁。如果 model.py 是排序 / TopK / 采样类算子
   （找第 k 大、排序列选、概率阈值过滤），必须逐个完成以下 checklist
   后才能进入步骤 1。禁止跳过。
```

**触发条件**：`{output_dir}/model.py` 的 forward() 中包含以下任一特征：
- `torch.sort` / `.sort(dim=..., descending=...)` / `topk` / `torch.topk`
- `kthvalue` / 找第 k 大 / 按第 k 大值做 mask 过滤
- top-k + top-p 组合过滤 / 采样（`npu_top_k_top_p` / `top_k_top_p_sample` 类）
- 按阈值过滤 + 概率累加（cumsum）组合

**强制执行清单**：

```
0-B.1 🛑 读取设计模式索引（必须，不可跳过）:
    Read references/design-patterns/DesignPatternIndex.md

0-B.2 🛑 读取排序/TopK 设计要点:
    Read references/design-patterns/references/sort_topk_design.md

0-B.3 🛑 在思考中确认:
    - 输出形态（一票否决）：算子输出是有序值/有序索引（argsort 下标、gather 按序
      消费，如 MoE gating）还是掩码/阈值过滤？前者值域二分不适用，必须排序路径
    - TopK 主结构路由：本算子走「值域二分」还是「排序路径（整行/分段 sort）」？
      按行/组实际键长判（单行交叉点 ≈8192；分组每组 ≤1024 一律排序路径）
    - 已完成复杂度+硬件效率估算并写入设计文档：二分 iter×N×3op vs 排序单遍+归并
      固定开销；k≤2 的 reduce-max 替代同样须估算，禁止默认更快
    - tie 语义：参考实现是「值阈值」（保留所有 >= kth）还是「位置阈值」（恰好 k 个）？
      值域二分天然对齐值阈值；分段 sort 需两步（求 kth → 重收集 >= kth）才能对齐
    - 输入约束：p ∈ [0,1]、k ≤ min(N,1024) 是否已在设计/评测中定好
    - 本算子的 block-level 设计骨架应与命中的设计模式对齐
```

**门禁规则**：
- 如果触发条件满足但 0-B.1-0-B.3 未完成 → **禁止**进入步骤 1，**禁止**生成任何 design/ 下的代码
- 如果触发条件不满足 → 跳过步骤 0-B，直接进入步骤 1
- 禁止凭记忆或经验跳过设计模式直接设计

---

### 步骤 1: Block 层级设计

生成 `{output_dir}/design/block_level/` 下的 block-level 设计，并同步生成 `{output_dir}/model_new_tilelang.py` 骨架。在这一步只确定 block 级任务划分、流水骨架、workspace 与同步关系，具体计算细节先标记为 `TODO(tile-level)`。

🛑 **在生成 block_level/ 代码前，必须先完成以下技术分析。**

#### 1a. 技术约束检测（强制）

Read `ops/tilelang-op-design/references/ascend-constraints.md`，对当前算子逐项检测 6 项：

| 检测项 | 触发条件 |
|--------|----------|
| 三维 Kernel | 算子是否需要 >1 维 block 并行 |
| threads 参数 | 是否需要 threads > 2 |
| 动态循环边界 | 循环次数是否依赖 tensor 值 |
| GPU 专用 API | 用户参考实现是否含 CUDA API |
| GEMM 非整除 | M/N 是否不被 block size 整除 |
| L0C 溢出 | block_M × block_N × 4 > 128KB |

检测到违反 → 输出警告 + Ascend 替代方案后再继续。

#### 1b. 编程模式决策（强制）

Read `ops/tilelang-op-design/references/decision-tree.md`，按决策树判定：
- 计算类型：纯 Vector / 纯 Cube / 混合（CV 融合）
- 编程模式：Developer（推荐，threads=2 消 workspace/vid）/ Expert / 混合
- API 选型：T.gemm_v0 vs T.mma、alloc_L1 vs alloc_shared

#### 1c. 信息收集（强制）

Read `ops/tilelang-op-design/references/info-sources.md`：
- 搜索 examples/ 同类实现
- 确认 API 可用性

#### 1d. 生成 block_level/ + model_new_tilelang.py 骨架

参考 `ops/tilelang-op-design/references/BlockLevelDesign.md`：
- 任务划分：确定 block 间分工、block 内遍历顺序
- 流水设计：C/V 分工与交错执行方案
- 模版设计：必要时定义多个 `T.prim_func` 按 shape 选择
- block 级设计必须与步骤 0 命中的 pattern 文档中的地址公式、循环结构、数据流对齐
- `model_new_tilelang.py` 负责输入布局整理 + kernel 调用

#### 1e. 设计期性能初检（强制）

```
⚠️ 本步骤是硬性门禁。block 级设计确定后、进入 tile 级实现前，必须完成以下检查
   并将结论写入 {output_dir}/design/PERF_DESIGN.md。未完成本步骤前，禁止生成
   tile_level/ 下的任何代码。禁止凭记忆或经验跳过本步骤。
```

在 block 级设计确定后、进入 tile 级实现前，对照性能反模式清单检查设计骨架是否存在明显性能隐患，避免将性能问题固化到 tile 级代码中。

Read `ops/tilelang-perf-optimization/references/performance-antipatterns.md`，重点关注以下与 block 级设计直接相关的项目：

| 检查项 | 触发条件 | 设计期修正方向 |
|--------|----------|---------------|
| launch core 数 | `block_num` >> 24 物理核 → 按任务数 launch | 改用 Fixed Core 模式，按物理核数 launch，核内 `T.serial` 分配多任务 |
| launch core 数 | `block_num` < 24 仍按 24 launch | 按 `min(block_num, 24)` launch |
| CV 未 overlap | CV 融合型但主循环串行（Cube 写 workspace → Vector 读） | 改用 `T.Pipelined` + AUTO_CV_COMBINE/AUTO_CV_SYNC |
| tile size 过小 | L0C/UB 实际使用量远小于容量 | 在不超限前提下成倍扩大 block_M/block_N |
| AIV 未流水 | 纯 AIV 算子 GM→UB→Vector→GM 串行 | 改用 `T.Pipelined` 或手动双 buffer |

Read `ops/tilelang-perf-optimization/references/optimization-guide.md` §一（算子类型对应），确认本算子的优化范围与优先级顺序。

**Cube/CV 融合型补充**：判断本算子是否需要预留 pingpong/swat/streamk 等高级策略的扩展空间（如在 block 级设计中预留 Split-K 切分维度）。

**CV 融合型补充**：确认设计的流水结构不会落入 Cube/Vector 串行乒乓（各闲一半）的 lockstep 调度。

**🛑 PERF_DESIGN.md 必须包含以下内容（缺项视为未完成 1e）**：

1. 算子类型判定（纯 Cube / 纯 Vector / CV 融合）及 optimization-guide §一 对应的优化范围与优先级
2. 上表中与本算子相关的检查项逐项结论（命中 / 未命中 / 不适用 + 一句理由）
3. Cube / CV 融合型：选用的高级策略，或在 block 级设计中预留的扩展空间（如 Split-K 切分维度、pingpong/swat 适配点）
4. CV 融合型：锁步风险结论（流水结构如何避免 C/V 串行乒乓）
5. `## 性能迭代待验证清单` 章节（见下方格式；1e 阶段无待验证项时写"暂无"，章节本身不可缺）

**性能迭代待验证清单（标准化格式，供步骤 4 消费）**：所有"设计期保留、待实测验证"的项（如保留的疑似冗余同步、暂不启用的高级策略、可扩大的 tile 配置）必须结构化为下表，禁止散落在正文描述中：

```
| # | 待验证项 | 假设 | 验证方法 | 命中时的修改点 |
|---|---------|------|---------|---------------|
| V1 | K-loop 内 barrier_all 冗余 | AUTO_SYNC 下可移除 | msprof 同步开销占比 | 移除手写 barrier，复验精度+性能 |
```

---

### 步骤 2: Tile 层级设计

在第一步基础上继续生成 `{output_dir}/design/tile_level/`。直接以 block-level 设计为骨架，在 tile-level 中补全各处 `TODO(tile-level)`，完成用于表达设计意图的 TileLang 设计与实现。

#### 2a. 编码规范检查

Read `ops/tilelang-op-develop/references/coding-conventions.md`：
- Buffer 分配 shape 一致性
- 数据搬运索引正确性（读/写/workspace 必须一致）
- 同步策略匹配编程模式（Developer 自动/AUTO_SYNC，Expert 手动/barrier_all）

#### 2b. GEMM/CV 融合专项（仅含 GEMM 或 CV 融合时）

Read `ops/tilelang-op-develop/references/gemm-cv-fusion.md`：
- gemm_v0 / T.mma 首次调用 init=True（语义上首次累加需初始化）
- 4 pass_configs 完整（AUTO_SYNC + AUTO_CV_COMBINE + AUTO_CV_SYNC + MEMORY_PLANNING）
- Developer 默认 threads=2 + 片上直连（消 workspace/vid）

#### 2c. V 核并行化

Read `ops/tilelang-op-develop/references/vector-parallelism.md`：
- Developer：threads=2 自动并行，无 vid 偏移
- Expert/回退：手动 vid 切分 + 索引一致性

#### 2d. 生成 tile_level/ 代码

- **API 用法**：查阅 [tilelang-api-best-practices SKILL.md](../tilelang-api-best-practices/SKILL.md) 及其 references 目录
- **编程模式和 pass_configs**：查阅 [tilelang-programming-model-guide SKILL.md](../tilelang-programming-model-guide/SKILL.md) 及其 references 目录


#### 2e. Tile 级性能反模式终检（强制）

```
⚠️ 本步骤是硬性门禁。tile 级代码完成后，必须完成以下终检，并把结论追加到
   {output_dir}/design/PERF_DESIGN.md。未完成本步骤前，禁止进入步骤 3。
```

在 tile 级代码完成后，对照性能反模式清单做最终扫描，确认 tile 级实现未引入新的性能劣化。

Read `ops/tilelang-perf-optimization/references/performance-antipatterns.md`，重点扫描以下 tile 级容易引入的问题：

| 检查项 | 识别特征 | 修正方向 |
|--------|----------|---------|
| 逐元素/逐行 for loop | Vector 侧 `for row in range(block_M): T.tile.sub(...)` | broadcast 到同 shape 后一次性向量化 |
| 冗余全局同步 | 循环内频繁 `T.barrier_all()` / `T.sync_all()` | 改用 `T.set_flag`/`T.wait_flag` 精确依赖，或交给 AUTO_SYNC |
| 指令未融合 | `mul + add` / `max(x,0)` / `sqrt + div` 连续出现 | 替换为 `mul_add_dst` / `relu` / `rsqrt` 等复合指令 |
| 多行 tile 循环内归约 | 多行 tile 循环内每块调用 `reduce_sum` | 循环内累积到完整 buffer，循环外一次性归约 |
| AIV 未流水/双 buffer | 纯 AIV 搬入→计算→搬出完全串行 | 改用 `T.Pipelined` 或手动双 buffer |

Read `ops/tilelang-perf-optimization/references/optimization-guide.md`：
- Cube 型：确认 Split-K 切分 + Double Buffer + Fixed Core 是否已纳入设计
- Vector 型：确认 Double Buffer + 指令向量化/融合 + Fixed Core 是否已考虑
- CV 融合型：确认核内优化后再检查核间流水（`T.Pipelined` num_stages、cross_interval）

**🛑 终检结论必须追加到 PERF_DESIGN.md**，包含：上表逐项扫描结论（命中 / 未命中 / 不适用 + 一句理由）、命中的反模式及对应修正、optimization-guide 三类确认项的勾选结果。**2e 阶段新产生的"保留待实测"项（如与权威示例一致而暂留的 barrier、记录为后续优化点的策略）必须追加到 `## 性能迭代待验证清单` 表中。**

---

### 步骤 3: 自检

🛑 **强制项**：确认 `{output_dir}/design/PERF_DESIGN.md` 存在，且包含 1e（block 级初检）与 2e（tile 级终检）两轮完整结论及 `## 性能迭代待验证清单` 章节（无待验证项时写"暂无"）；缺失或缺项则返回对应步骤补齐，禁止直接交付。其余自检项为可选——如用户明确要求，或为了排查 DSL 语法 / 编译问题，可执行以下自检。

#### 3a. 设计质量自检

Read `ops/tilelang-op-design/references/quality-checklist.md`，按 20 项逐项检查设计产物。
必须项（1,2,3,7,8,9,13,19,20）全部通过，推荐项 ≥ 4/9。

#### 3b. 编码规范自检

Read `ops/tilelang-op-develop/references/checklist.md`，按 22 项逐项检查代码产物。

#### 3c. TileLang 功能验证（强制）

调用 `cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/scripts/evaluate_tilelang.sh {output_dir}` 执行功能验证。**本步骤默认必须执行、且精度必须通过**——它是步骤 4 性能迭代的强制前置条件（`tilelang-perf-optimization` 的核心约束：精度未通过，禁止性能优化；3c 不过则步骤 4 不会触发）。

**唯一的跳过条件**：通过对照实验确认属 **TileLang 编译器/框架底层不支持**（如特定 dtype 的 cube MMA 不支持、框架 bug），必须持有证据（例如同结构 fp16 通过 / fp32 失败）。满足时允许跳过，但必须在最终说明中记录跳过原因与对照实验证据；条件允许时，应先用框架支持的 dtype（如 fp16/bf16）做代理验证确认设计逻辑正确——代理验证不通过则说明设计本身有问题，不得以"框架问题"为由放行。

**禁止事项**：
- 禁止以"评测可选""不作为 gate"为由不执行 evaluate_tilelang.sh
- 禁止无对照实验证据就以"框架问题"跳过；验证失败且不属于编译器底层不支持时，必须按 3d 排查并回到步骤 1/2 迭代修复
- 禁止为了通过 TileLang 验证而扭曲设计（如改用降精度路径规避）

TileLang 精度结果不作为交付的 correctness gate（交付精度以 Phase 4 AscendC 验证为准）。TileLang 性能数据的使用口径：**作为步骤 4（Phase 3 内部性能迭代）的测量输入**，但不进入 Phase 5 的对比报告、不作为交付的性能结论。

#### 3d. 疑难解答

如遇编译/运行/精度错误，Read `ops/tilelang-op-develop/references/troubleshooting.md` 按错误类型排查：

| 错误类型 | 排查方向 |
|---------|---------|
| 编译错误 | buffer 大小、API 参数、对齐 |
| 运行错误 | 索引越界、同步缺失 |
| 精度错误 | Golden 实现、输出形状 |

---

### 步骤 4: TileLang 性能迭代（强制门禁，p_retry ≤ 3）

```
⚠️ 本步骤是硬性门禁。未完成本步骤（含合法跳过）前，禁止交付 Phase 3 产物、
   禁止进入下游 AscendC 转译（Phase 4）。
```

本步骤调用 `ops/tilelang-perf-optimization` skill **本体**（Step 1-5 迭代闭环），对 TileLang 实现做测量驱动的性能迭代。skill 文档中的项目布局按以下映射理解：`examples/{op}/` → `{output_dir}/`，测试入口 → 本项目的 TileLang 测试脚本（`evaluate_tilelang.sh` 的 python 入口或等效脚本）。达标线：**geomean ≥ 0.6x**（TileLang kernel 时间 vs PyTorch reference kernel 时间，msprof 采集）。

#### 4a. 前置判定（能否测量）

- **3c 功能验证通过（精度达标）→ 进入 4b**。精度未通过时不允许进入 4b/4c（tilelang-perf-optimization 核心约束：精度未通过禁止性能优化）
- TileLang 编译器/框架底层不支持导致无法执行：必须满足 3c 的对照实验证据标准（如同结构 fp16 通过 / fp32 失败）且代理验证通过 → 写 `{output_dir}/perf_tuning/SKIPPED.md`（跳过原因 + 对照实验证据 + 代理验证结果）→ 本步骤完成，允许交付
- **无对照实验证据的"不可执行"不允许跳过**，按 3d 排查并返回步骤 1/2 修复

#### 4b. 基线采集

用 `ops-profiling` 的 `msprof_profile_run.sh` 标准模式（`--` 包任意可执行文件）分别采集 reference 与 TileLang 实现的 kernel 时间：

```bash
bash msprof_profile_run.sh --warm-up=3 --output=./msprof_ref -- python <reference 测试入口>
bash msprof_profile_run.sh --warm-up=3 --output=./msprof_tl  -- python <tilelang 测试入口>
```

- 采集纪律：`npu-smi info` 确认目标卡空闲、≥20 次取中位数、记录 NPU ID / shape / dtype
- 计算 `geomean_speedup = geomean(ref_kernel_us / tl_kernel_us)`，写入 `{output_dir}/perf_tuning/baseline.json`
- **geomean ≥ 0.6 → 基线即达标**：记录到 `optimization_log.md`，跳到 4d 产出 final_report.md 后交付

#### 4c. 迭代循环（p_retry = 0..2）

调用 `ops/tilelang-perf-optimization` skill 的 Step 3-5：

1. **Step 3 识别优化点**：在 `{output_dir}/perf_tuning/optimization_log.md` 输出 Part A 优化点清单 + `[ORDER-PLAN]`。**清单必须以 PERF_DESIGN.md「性能迭代待验证清单」为强制种子**（逐条判定适用/不适用 + 原因），再补充反模式扫描新发现项
2. **Step 4 逐项实施**：每个优化点走 6 子步骤（`[ORDER-CHECK]` → Read 文档 → Edit `design/tile_level/` + `model_new_tilelang.py` → msprof 复测 → `[RESULT-#N]` → 失败重读修复）；每次 Edit 只改一个优化点；**精度复验不过的改动禁止计入性能收益**（回退或修复后再计）
3. **Step 5 效果验证**：全 case 复测，geomean ≥ 0.6 → break
4. 连续 3 个优化点无提升 → 提前终止本轮循环

#### 4d. 终止与上报

- 达标 / p_retry 耗尽 / 连续无提升 → **必须先做上限分析**（算术强度拆解 + Amdahl 上限 + roofline 判定），确认剩余空间不足以支撑继续迭代；禁止只报"连续无提升"
- 产出 `{output_dir}/perf_tuning/final_report.md`：基线 vs 最终 geomean、已实施优化点列表及各自收益、上限分析结论、未达标原因（若未达标）

#### 🛑 交付门禁

`{output_dir}/perf_tuning/` 下必须存在以下产物组合之一，缺失视为步骤 4 未完成，禁止交付：

| 情形 | 必需产物 |
|------|---------|
| 合法跳过（4a） | `SKIPPED.md` |
| 基线即达标（4b） | `baseline.json` + `optimization_log.md` + `final_report.md` |
| 迭代完成/预算耗尽（4c→4d） | `baseline.json` + `optimization_log.md`（含 `[ORDER-PLAN]` 与逐项 `[RESULT-#N]`）+ `final_report.md`（含上限分析） |

- **🛑 零迭代交付封堵**：基线 geomean < 0.6x 且 `optimization_log.md` 中无任何 `[RESULT-#N]`
  迭代记录（p_retry=0）时，仅有 `baseline.json` + `final_report.md` **不算完成**——必须回到
  4c 执行迭代，或按 4a 标准补齐 `SKIPPED.md`（结构性阻塞的对照实验证据 + Phase 4 缓解计划，
  如"TileLang 无跨核同步原语 → Phase 4 双 kernel 拆分"）。实测违规案例：以 p_retry=0 +
  仅上限分析交付 0.31x 基线，阻塞证据与缓解计划均缺失，下游被迫在 AscendC 阶段补课。

性能迭代完成后，优化定稿的 `design/tile_level/` 与 `model_new_tilelang.py` 即为 Phase 4 转译输入；若迭代中修改了设计结构（如任务划分、tile 配置），同步更新 PERF_DESIGN.md 对应结论。
