---
name: external-cannbot-ops-lab-tilelang-to-ascendc-skills-tilelang2ascend-precision-tuning
description: '用于DumpTensor进行AscendC算子精度的调试。

  Use when:

  - AscendC kernel / 算子精度失败，结果不对，数值错误，部分位置错误，或 NaN/Inf

  - 需要用 DumpTensor / dumptensor / dump tensor 看中间结果、GM/UB/Workspace 数据

  - 需要分段定位 `Cube输入/中间/输出`、`Vector输入/中间/输出` 哪一段出问题'
original-name: tilelang2ascend-precision-tuning
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# AscendC DumpTensor 精度调试

系统性的 AscendC kernel 精度问题定位方法，按算子类型选择对应调试流程。

## 核心规则

- 对 AscendC **精度问题**，`DumpTensor` 是默认首选工具。不要因为“看起来不复杂”或“怀疑是别的问题”就跳过 dump；在能插桩的位置，先 dump 再分析。
- `DumpTensor` **不会单独写日志系统**，而是直接把内容打印到当前任务的终端输出。需要持久化或便于 grep 时，主动把任务输出重定向到 debug 日志文件再分析。
- 发现某个中间量异常时，禁止只盯着这一个点空想根因。必须沿数据流**向前回溯一跳或多跳**验证它的直接输入；例如 `S = Q @ K^T` 异常时，先回查同一块 `Q` 和 `K` 是否已经错误，再判断 MMAD、layout、fixpipe 或同步问题。
- dump 过多会淹没有效信息。默认先做**最小抽样**：首核、首轮、首 slot、前 8/16/32 个元素；只有在首样本正常但问题只出现在尾块/后续轮次时，再追加尾块、末 slot 或指定 block 的 dump。
- 数据异常除了公式、offset、shape、搬运参数，也要把**数据依赖同步缺失/时机错误**作为一等候选根因。尤其检查 GM→UB（MTE2）后 UB 上计算、UB 计算后 UB→GM（MTE3）、以及跨核 Workspace 生产/消费之间是否有必要同步。
- 只在精度验证失败时使用，定位完成后移除 DumpTensor 调试代码。
- 禁止在生产代码中保留 DumpTensor（有性能开销）。
- 本 skill 不涉及 TileLang 调试，仅用于 AscendC kernel。
- 编译错误、运行错误、卡死问题不使用本 skill。
- 每次进行调试迭代后必须将关键发现写入 `{task_dir}/ascendc_debug_trace.md`（`{task_dir}` 为当前 `simple_task/` 或 `full_task/`），格式如下：

```markdown
## 迭代 N

**操作**: 插入 DumpTensor 位置、参数、验证用例。

**结果**: NPU vs CPU golden 数值对比、错误特征。

**分析**: 从结果推导的结论。

**下一步**: 后续调试或修复方向。

**状态**: 调试中 / 已定位 / 已修复验证通过
```

## 算子类型判定

| 类型 | 特征 | 示例 |
|------|------|------|
| 类型 A - 简单 Vector | 单核、无 Cube、无跨核同步、线性数据流 | Add、Mul、Softmax、LayerNorm |
| 类型 B - 复杂 Cube+Vector | 双核分离、Mmad/Matmul、跨核同步、Workspace 中间数据 | FlashAttention、Matmul fusion |

## 通用执行顺序

每次调试遵循这个顺序，禁止跳步：

1. 先缩小 case：优先单 batch、小 head、最小可复现 seq/block。
2. 先插最少量 dump：首核/首轮/首 slot，`dumpSize` 先用 8/16/32。
3. 先验输入，再验当前怀疑点，再验下一跳。
4. 一旦某个中间量错误，必须回溯其直接输入，不要只围绕该中间量猜测。
5. 若生产端 dump 正确、消费端 dump 错误，优先检查同步、slot 管理、DataCopy 参数和读写地址。
6. 屏幕输出过多时，把任务 stdout/stderr 重定向到文件，用 `grep` 按 `desc`过滤阅读。

### 类型 A - 简单 Vector 算子

**追踪点**：CopyIn 后 → Compute 步骤 → CopyOut 前

**编号约定**：
- 400-499：Vector输入（CopyIn 后）
- 500-599：Vector中间（Compute 后）
- 600-699：Vector输出（CopyOut 前）

详细步骤请读取 `references/debug-workflow-simple.md`

### 类型 B - 复杂 Cube+Vector 算子

**核心原则**：追踪 GM/UB 上的 ND 数据，避免 L1/L0 的 NZ 格式。

**编号约定**：
- 100-199：Cube输入结果
- 200-299：Cube中间结果
- 300-399：Cube输出结果
- 400-499：Vector输入结果
- 500-599：Vector中间结果
- 600-699：Vector输出结果

详细步骤请读取 `references/debug-workflow-complex.md`

## 参考资料

按需读取，不要一次性全读：
- `references/api-reference.md` - DumpTensor API 参数说明（必读）
- `references/error-patterns.md` - 常见错误模式与根因对照表
- `references/debug-workflow-simple.md` - 类型 A 详细调试步骤 (适用于Vector简单算子)
- `references/debug-workflow-complex.md` - 类型 B 详细调试步骤（适用于CV融合算子）
