---
name: external-cannbot-community-ops-tilelang2ascend-trace-recorder
description: '执行 trace 记录员 + 知识演进 Skill。在算子任务完成后，回顾整个执行过程， 生成结构化的 trace 记录（{output_dir}/trace.md），并从
  trace.md 提取 "走偏点"与成功模式，按 knowledge-ingest 的通用原则 沉淀为知识卡写入共享知识库 runbooks/，供后续算子生成任务
  经 knowledge-query 检索复用。当算子任务完成后需要记录执行过程 并进行知识演进时，使用此 skill。

  '
argument-hint: 输入：output_dir 目录路径、各阶段执行结果信息。 输出：{output_dir}/trace.md 结构化执行记录 + 共享知识库
  runbooks/ 更新 （新卡片 / 必要导航与日志 / 检索索引验证）。
original-name: tilelang2ascend-trace-recorder
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-28'
synced-commit: 7a09cdfac4b2869dca6b6fd4531a0a5e539d6096
license: UNKNOWN
---

# Trace 记录 Skill

你是一名执行 trace 记录员。你的目标是在算子任务完成后，回顾整个执行过程，生成结构化的 trace 记录。

## 关键限制
- 只允许在 `{output_dir}/` 目录下创建 `trace.md` 文件
- 不要修改 `{output_dir}/` 中的任何其他文件
- 只记录事实，不做改进建议（那是 meta-agent 的工作）

## 信息来源

回顾本次会话中的以下信息：
- 各阶段的执行结果（成功/失败）
- 评测脚本的输出（`@scripts/evaluate_tilelang.sh`、`@scripts/evaluate_ascendc.sh` 的返回）
- ops-profiling（msprof 模式）的性能测试结果
- Agent 的迭代过程（尝试了什么、失败了几次、最终如何解决）
- 遇到的错误信息

补充约定：
- TileLang 功能验证（evaluate_tilelang.sh）为强制步骤：默认必须执行且精度必须通过（精度通过是 Phase 3 Step 4 性能迭代的强制前置）；交付的 correctness gate 仍以 AscendC 验证为准
- 未执行 TileLang 验证属流程违规，必须在"走偏点"中记录；仅当对照实验证明编译器/框架底层不支持时才可合法跳过，应如实记录为"跳过"并写明原因
  （含框架侧问题的对照实验证据，如 fp16 通过 / fp32 失败；若做了 fp16/bf16 代理验证，记录其结论）

## 输出格式

将以下内容写入 `{output_dir}/trace.md`：

```markdown
# Trace: {算子名称}

- 时间: {当前日期时间}
- 算子: {output_dir 对应的算子名}
- 设计路径: {design.md (简单算子) / TileLang (复杂算子)}
- 最终结果: SKIP / PASS / FAIL (tilelang) | PASS / FAIL (ascendc)

## 阶段零: Case 精简

- 结果: 通过 / 失败 / 跳过
- 原始 case 数: {n}
- 精简后 case 数: {n}
- 备注: {如有异常情况}

## 阶段〇.五: 设计文档 (仅简单算子路径)

- 结果: 通过 / 失败 / 跳过
- 说明: 简单算子路径生成 design/design.md；复杂算子路径填写"跳过(走 TileLang 路径)"
- 备注: {如设计文档修正次数}

## 阶段一: TileLang (仅复杂算子路径)

- 结果: 通过 / 失败 / 跳过
- evaluate_tilelang.sh 执行次数: {n}
- 关键错误信息: {评测脚本返回的错误，原文引用}
- 性能设计检查（1e/2e 强制项，依据 design/PERF_DESIGN.md 记录）:
  - 算子类型判定: {纯 Cube / 纯 Vector / CV 融合}
  - 1e block 级初检: {命中的反模式与设计期修正；无命中则写"全部未命中/不适用"}
  - 2e tile 级终检: {命中的反模式与设计期修正；无命中则写"全部未命中/不适用"}
  - 高级策略路由结论: {Cube 型写 cube_advanced_strategies 选用/预留策略；其余写"不适用"}
  - PERF_DESIGN.md 缺失或缺项 → 在"走偏点"中记录为流程违规
- 性能迭代（步骤 4 强制项，依据 perf_tuning/ 记录）:
  - 结果: 达标 / 预算耗尽未达标 / 合法跳过（SKIPPED.md，附原因与对照实验证据）
  - 基线 geomean: {baseline.json 数值} → 最终 geomean: {final_report.md 数值}
  - p_retry 轮数与已实施优化点: {逐项名称 + 各自收益，依据 optimization_log.md 的 [RESULT-#N]}
  - 未达标时: {final_report.md 的上限分析结论（roofline/Amdahl）摘要}
  - perf_tuning/ 缺失必需产物 → 在"走偏点"中记录为流程违规
- Agent 行为记录:
  - 第 1 轮: {agent 做了什么，结果如何}
  - 第 2 轮: {修改了什么，结果如何}
  - ...
- 走偏点: {agent 做了哪些无效/错误/冗余的尝试，以及可能的原因}

## 阶段二: AscendC

- 结果: 通过 / 失败
- 设计路径: design.md (简单算子) / TileLang 转译 (复杂算子)
- 产物: kernel/op_host/<op>.cpp, kernel/op_kernel/<op>.cpp, kernel/ops.h, kernel/register.cpp
- 编译: evaluate_ascendc.sh (cmake + make + whl)
- evaluate_ascendc.sh 执行次数: {n}
- 关键错误信息: {评测脚本返回的错误，原文引用}
- Agent 行为记录:
  - 第 1 轮: {agent 做了什么，结果如何}
  - 第 2 轮: {修改了什么，结果如何}
  - ...
- 走偏点: {agent 做了哪些无效/错误/冗余的尝试，以及可能的原因}


## 阶段三: 性能分析

- 结果: 完成
- ops-profiling（msprof 模式）执行详情:
  - 测试配置: device=npu, warmup=5, repeat=10, seed=0
  - 测试的实现: reference / tilelang / ascendc
  - 总体统计:
    - reference: mean=0.086ms, median=0.070ms, min=0.050ms, max=0.362ms, std=0.046ms
    - tilelang: mean=0.327ms, median=0.202ms, min=0.147ms, max=3.284ms, std=0.503ms
    - ascendc: mean=0.186ms, median=0.090ms, min=0.054ms, max=2.443ms, std=0.366ms
  - 性能结论:
    - 三个实现均执行成功（Status=OK）
    - AscendC 整体快于 TileLang，按 mean 统计约快 1.76x（0.327 / 0.186）
    - AscendC 仍慢于 reference，按 mean 统计约为 reference 的 0.46x；reference 约快 2.16x
    - TileLang 慢于 reference，按 mean 统计约为 reference 的 0.26x；reference 约快 3.80x
  - 典型大 shape case 观察:
    - case[47] shape=(1, 8, 16384, 64), float16, half: reference=0.169ms, tilelang=1.065ms, ascendc=0.626ms
    - case[48] shape=(1, 8, 32768, 64), float16, half: reference=0.261ms, tilelang=1.915ms, ascendc=1.155ms
    - case[49] shape=(1, 8, 32768, 64), bfloat16, interleave: reference=0.292ms, tilelang=3.271ms, ascendc=2.434ms

## 汇总表报告

- 说明: 延迟单位为 ms，按 ops-profiling（msprof 模式）的 mean 统计；加速比 =  PyTorch 参考延迟/生成 AscendC 代码延迟 。加速比>0.6性能0.6x pytorch填是，否则填否。性能0.8x pytorch同理。

| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | PyTorch 参考延迟 | 生成AscendC代码延迟 | 加速比 | 最终状态 | 精度正确 | 性能0.6x pytorch | 性能0.8x pytorch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | RotaryMul | vector | ✅ | ✅ | 0.086 | 0.186 | 0.46 | 成功 | 是 | 否 | 否 |

## 评测输出摘要

{粘贴最后一次 evaluate 脚本的关键输出片段，包括 PASS/FAIL 状态和错误详情}
```

### 记录原则

1. **精确引用**: 错误信息、评测输出使用原文，不要改写或总结
2. **行为序列**: 每轮迭代记录 agent 的实际操作（改了什么文件、改了什么逻辑），而非笼统的"修复了 bug"
3. **走偏分析**: 重点记录 agent 做了哪些最终被证明无效的尝试，这是 meta-agent 优化 harness 的核心输入
4. **省略成功**: 如果某阶段一次通过且无异常，简要记录即可，不需要展开
5. **如实跳过**: TileLang 未验证不是异常；如果流程按约定跳过，应明确记录“跳过”及原因，不要误记为失败

---

## 知识演进（Phase 7 后半：trace.md → 共享知识库）

记录完 trace.md 后，继续完成**知识演进**：把走偏点与成功模式提炼为可复用经验，
经 `infra/knowledge-ingest` 沉淀进共享知识库，形成
"生成 → trace → 演进 → 更高效生成"的闭环。

> **标准 skill 约束（强制）**：
> - **沉淀**：遵循 `infra/knowledge-ingest` 的通用原则与当前知识仓规范，先执行其
>   安装入口准备知识库。本演进属开发轨迹类知识，落点按当前仓规范确定。
>   **禁止**再调用 `scripts/evolve_traces.py` 或任何自维护脚本直接产卡。
> - **检索/去重**：一律用 `knowledge-query` skill（`knowledge_query.py`），
>   **禁止**自维护 INDEX.md 或 grep 卡片目录做去重。
> - **知识库根**：使用安装入口返回的完整 `root` 作为 `CANNBOT_KNOWLEDGE_ROOT`。
>   写入前用 `infra/knowledge-query` 的 `knowledge_query.py discover --json` 确认 root；
>   知识库缺失时由安装入口克隆并准备，准备失败时停止并上报，不猜路径。

### 演进闭环

```
任务完成 → trace.md
    │ 1. 阅读 trace 原文（走偏点 / 行为轮次 / 关键错误）
    │ 2. 检索去重（knowledge-query: search 查同根因卡片）
    │ 3. 建/更新卡（knowledge-ingest 通用原则与当前仓目录规范）
    │ 4. 交叉链接（按当前仓规范）
    │ 5. 按 knowledge-ingest 验证卡片、导航、日志与检索索引
    ▼
共享知识库 runbooks/ ← 后续生成任务经 knowledge-query 检索消费（生成前检索 + 各阶段按需）
```

### 信息来源（按顺序）

1. 本次会话产出的 trace.md：
   - `走偏点` 段 — 每个走偏点 = 一条**负面教训**（尝试了什么、为何无效、根因、应怎么做）
   - `Agent 行为记录` 的"第 N 轮" — 成功路径的**正面模式** + 每轮成本
   - `关键错误信息` — 错误签名（如 `RegisterAscendBinary mix ret 107000`、
     `vector core exception (507035)`）→ 根因映射
   - `evaluate_*.sh 执行次数` — 迭代成本量化
2. 知识库现有卡片（经 knowledge-query 检索）— 去重与增量更新依据

### 流程

1. **阅读原文**：完整阅读 trace.md 的走偏点与对应轮次原文，确认根因链与修复方向，
   禁止只看摘要就成文。
2. **检索去重**（knowledge-query，替代原 INDEX.md 对照）：
   ```bash
   python3 <query-skill-dir>/scripts/knowledge_query.py search \
     --query "<走偏点/错误签名关键词>" --domain ops --technology ascendc \
     --type runbooks --platform <目标平台> --match text -k 5
   ```
   `<query-skill-dir>` 为 `infra/knowledge-query` 的实际安装目录；读取候选卡片全文后再判断。

   - 同根因不同表述 → 更新现有卡片（按当前仓规范追加来源与证据，**增量补充，
     绝不整体覆盖**），不新建
   - 不同根因 → 新建卡片
   - 现有卡片已覆盖且无新信息 → 跳过（在演进报告中记"跳过"及理由）
   - 平台/环境升级导致失效（如 API 恢复可用）→ 按当前仓生命周期规范标记失效和替代关系，保留证据
3. **撰写 / 更新卡片**：遵循 `infra/knowledge-ingest` 的通用原则，按其读取的当前知识仓
   Profile、字段和目录规范确定落点与 Frontmatter。正文保留以下开发轨迹证据：

```markdown
# <一句话标题：做什么（禁止 X）>

## 触发条件
- <生成/修复时命中这些情况 → 先读本节>

## 症状
- <该问题长什么样（错误信息、输出特征）>

## 根因
- <根因链>

## 正确做法
- <修复方向 / 应怎么做>

## 反例（历史真实失败）
- <具体任务 + 轮次 + 实际结果>

## 证据
- <trace.md 引用 + 关键数据>
```

   - **优先沉淀通用知识**：算子特异的一次性细节（单算子 shape/参数组合）不单独成卡，
     合并进同根因通用条目；只有可复用于后续算子生成的经验才写入
4. **交叉链接**：按当前知识仓规范维护与相关卡片的相对链接，确认目标真实存在。
5. **收尾验证**：按 `knowledge-ingest` 的检查与交付流程验证卡片、必要导航和日志，
   运行知识仓质量检查，重建并验证检索索引，复查目标查询。
6. **演进报告**：向调用方汇报新增/更新/跳过卡片数、验证结果及迭代成本。

### 知识演进质量原则

- 每轮 trace 只新增少量高价值条目，拒绝"为了演进而演进"的批量灌水
- 命中通用坑时优先更新现有卡片而非新建（保持知识库收敛）
