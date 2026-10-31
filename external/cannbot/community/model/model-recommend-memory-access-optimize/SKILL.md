---
name: external-cannbot-community-model-model-recommend-memory-access-optimize
description: 访存优化（工作流条目 9）。对 profiling 确认的 memory-bound 热点（embedding 查表/Gather/大 GEMM
  权重读取/密集 elementwise 访存链）实施访存优化：图结构侧等价改写、kernel 侧双缓冲预取、精度模式联动。触发：访存优化、memory-bound、带宽利用率、embedding
  查表热点、权重搬运、GM 往返、布局优化、预取。
original-name: model-recommend-memory-access-optimize
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 访存优化（布局与搬运重叠）

## Prerequisites

- [performance-analysis](../model-recommend-performance-analysis/SKILL.md) profiling 数据**确认存在 memory-bound 热点**（查表/Gather/大 GEMM/密集访存链，带宽利用率对照本机实测 copy 带宽锚定）
- 候选热点有**安全的依赖窗口**：前序算子不能同为 memory-bound（否则争抢带宽）
- 带宽锚定数据可用（`data/env-baseline/` 实测 copy 带宽）

## 目标

对 memory-bound 热点实施访存优化，把搬运与计算重叠、减少 GM 往返。

**重要性**：推荐模型典型 memory-bound（embedding 查表每请求读取大表、大 GEMM 权重全量读取但算力吃不满）；本机带宽有限，访存优化是稳定收益来源。

## 输入 / 输出契约

**输入**：profiling 数据（memory-bound 热点清单 + 依赖窗口）。

**输出**：带开关保护的优化实现 + 性能报告（目标算子耗时、关键路径、带宽利用率前后对比）。

## GE 图模式下的可用手段（分层，按成本升序）

> 与 torch 工作流的差异：无 `npu_prefetch` 类直插 API，图由 GE 编译，不可在 python 层逐算子插预取。手段按层分流：

1. **图结构侧（首选，零开发）**：onnx 等价改写减少 GM 往返——中间张量物化消除（Cast 链/Reshape-Transpose 链压缩）、查表形态优化（等价索引变体）、布局友好形态；经 [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) 手段实施
2. **kernel 侧**：热点为自定义算子（[kernel-generation](../model-recommend-kernel-generation/SKILL.md) / [custom-op-development](../model-recommend-custom-op-development/SKILL.md) 产物）时，在 AscendC kernel 内做双缓冲/预取（`DataCopy` 与计算重叠），随算子开发交付
3. **精度模式联动**：`FORCE_16` / `ENABLE_HF32` 减少搬运字节数（须过 [precision-comparison](../model-recommend-precision-comparison/SKILL.md) 门禁，逐项归因）
4. **框架侧需求记录**：GE 不支持图级显式 stream prefetch；如需 host 侧预取属框架层改造——**记录需求不擅自改框架**，移交用户决策

## 工作流步骤

### Step 1: 方案设计

1. **确定优化位置**：profiling 识别 memory-bound 热点 + 依赖窗口分析（前序算子带宽占用）
2. **手段选型**：按上述分层表选最小成本可行项；kernel 侧项须先确认该算子已是/将是自定义算子
3. 设计方案（修改文件/开关传递/预期收益区间）输出并等待用户确认

### Step 2: 实施开发

- 新增优化一律带 enable 开关（默认关闭）；onnx 改写保留原始版本
- kernel 侧改动遵循 [kernel-generation](../model-recommend-kernel-generation/SKILL.md) 流程

### Step 3: 验证测试（双门禁执行点，实际执行，记录真实结果）

- **单点验证原则**：先验证单个热点，成功后再扩展
- **精度门禁先行**：优化前后串行 verify 全 pass——不过 → 回退，不进性能测量
- **性能门禁**：目标算子耗时、关键路径、带宽利用率（有效字节数 ÷ 实测耗时 vs 锚定带宽）前后对比；**端到端主指标不劣化超过噪声带**（访存优化不得以其他段劣化为代价；若目标算子改善但端到端回退，判定失败并回退或重设计）
- **落账**：通过项写 perf_ledger；失败项回退并记录原因
- **测试报告**：baseline（开关关）与优化后（开关开）数据

## 昇腾 NPU 专属要点

- **带宽基准以本机实测锚定**（`data/env-baseline/`），不引用纸面值；bound 判定与收益评估同源
- **先保守后激进**：kernel 双缓冲从保守 UB 切分起步，按 profiling 逐步调大
- embedding 查表热点注意表访问局部性：请求 batch 内索引分布影响实际带宽利用，分析时记录索引分布特征
- 图模式联动：改写后的图形态变化会影响 GE 融合（dump_graph 复核），收益归因须排除融合变化干扰（单变量原则）
- 所有优化带开关保护，默认关闭；`THREAD_NUM>1` 下不验证精度（框架约束）

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/memory-access-optimize/`：访存优化方案与性能报告（实现代码按工程归属：onnx 改写在模型工作区，kernel 在 ops/）
- **数据类交付件** → `data/{model_name}/memory-access-optimize/`：baseline vs 优化后对比数据（目标算子/关键路径/带宽利用率）
- **中间产物** → `tmp/{model_name}/memory-access-optimize/`：profiling 中间数据、候选位置分析草稿、试验性改写
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 目标算子耗时与关键路径改善有前后数据
- **双门禁通过**：串行 verify 全 pass；端到端主指标不劣化超过噪声带；已写 perf_ledger（回退项含原因）
- 带宽利用率有量化对比（有效带宽 vs 锚定）
- 开关可关闭且默认关闭；框架侧需求已记录移交（如有）

## 依据

【实测】profiling 热点数据、本机带宽锚定。
【框架】onnx 改写手段与 GE 约束（无图级 prefetch API）。
【技能库经验】预取原则（依赖窗口、单点验证、先保守后激进、开关保护）。
