---
name: external-cannbot-ops-lab-tilelang-to-ascendc-skills-tilelang2ascend-tilelang-designer
description: 'TileLang kernel 设计与实现专家 Skill。为 PyTorch Model 设计并实现自定义 TileLang kernel：
  完成 block-level 设计、tile-level 设计，并生成 model_new_tilelang.py 调用自定义 TileLang kernel。
  当需要为复杂算子设计 TileLang kernel 时，使用此 skill。

  '
argument-hint: 输入：output_dir 目录路径（包含 model.py）。 输出：block_level/ 设计、tile_level/ 设计、model_new_tilelang.py
  实现。
original-name: tilelang2ascend-tilelang-designer
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# TileLang Kernel 设计 Skill

你是一名 TileLang kernel 设计与实现专家。你的目标是为 `{output_dir}/model.py` 中的**复杂算子** PyTorch Model 设计并实现自定义 TileLang kernel：完成 block-level 设计、tile-level 设计，并生成 `{output_dir}/model_new_tilelang.py` 调用自定义 TileLang kernel。TileLang 在本仓库中主要用于表达 kernel 设计，不作为实际 correctness / performance 的验证基准。

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
- 禁止读取 `.claude/skills/tilelang2ascend-tilelang-designer/references/TileLang-AscendC-API-Mapping.md`；该文档是 TileLang 到 AscendC 的转译映射，仅供 AscendC 阶段使用，与本阶段无关。

## 任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   ├── <op_name>.json    # 原始测试用例文件（备份保留）
│   ├── <op_name>.json.bak# 原始 .json 备份
│   ├── design/           # TileLang DSL 用于表达 kernel 设计
│   │   ├── block_level/  # TileLang block-level 设计
│   │   └── tile_level/   # TileLang tile-level 设计，用于表达完整 kernel 设计
│   ├── kernel/           # AscendC kernel（本阶段不涉及）
│   └── model_new_tilelang.py # 你的 TileLang 优化实现，调用 tile_level/ 下的 TileLang kernel
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

## Skill 参考资料
本 skill 提供以下参考资料（位于 `.claude/skills/tilelang2ascend-tilelang-designer/references/` 目录）：
- `.claude/skills/tilelang2ascend-tilelang-designer/references/BlockLevelDesign.md` — Block 层级设计指南
- `.claude/skills/tilelang2ascend-tilelang-designer/references/TileLangAscendProgrammingGuide.md` — TileLang Ascend 编程指南
- `.claude/skills/tilelang2ascend-tilelang-designer/references/TileLangDebug.md` — TileLang 调试指南（仅在需要排查 DSL 表达问题时参考）
- `.claude/skills/tilelang2ascend-tilelang-designer/references/attention-patterns/AttentionPatternIndex.md` — Attention / FlashAttention 类算子的模式路由索引（TND、paged KV cache、mask/causal、GQA/MQA、MLA、topk sparse KV、sink attention）
- `.claude/skills/tilelang2ascend-tilelang-designer/scripts/evaluate_tilelang.sh` — TileLang 评测脚本（当前仅供可选调试，不作为流程 gate）

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录可以作为参考实现。

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
    Read .claude/skills/tilelang2ascend-tilelang-designer/references/attention-patterns/AttentionPatternIndex.md
    
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
- 如果触发条件不满足 → 跳过步骤 0，直接进入步骤 1
- 禁止凭记忆或经验跳过模式文档直接设计

---

1. `Block 层级设计`
   生成 `{output_dir}/design/block_level/` 下的 block-level 设计，并同步生成 `{output_dir}/model_new_tilelang.py`。在这一步只确定 block 级任务划分、流水骨架、workspace 与同步关系，具体计算细节先标记为 `TODO(tile-level)`。
   参考文档：`.claude/skills/tilelang2ascend-tilelang-designer/references/BlockLevelDesign.md`。block 级设计必须与步骤 0 命中的 pattern 文档中的地址公式、循环结构、数据流对齐。
2. `Tile 层级设计`
   在第一步基础上继续生成 `{output_dir}/design/tile_level/`。直接以 block-level 设计为骨架，在 tile-level 中补全各处 `TODO(tile-level)`，完成用于表达设计意图的 TileLang 设计与实现。
   参考文档：`.claude/skills/tilelang2ascend-tilelang-designer/references/TileLangAscendProgrammingGuide.md`
3. `TileLang 自检（可选）`
   如用户明确要求，或为了排查 DSL 语法 / 编译问题，可调用 `.claude/skills/tilelang2ascend-tilelang-designer/scripts/evaluate_tilelang.sh {output_dir}` 做辅助检查；但 TileLang 结果当前不作为 correctness gate，也不作为性能测试输入。若遇到框架语义缺陷、尾块处理异常或其他 TileLang 自身 bug，应保留设计表达并在最终说明中明确记录，不要为了通过 TileLang 验证而扭曲设计。
   参考文档：`.claude/skills/tilelang2ascend-tilelang-designer/references/TileLangDebug.md`
