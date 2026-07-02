---
name: external-cannbot-ops-pypto-op-perf-tune
description: PyPTO 算子性能分析和自动调优技能。用于对生成及新开发的算子进行性能分析及自动调优，包括算子用例执行及精度校验、性能数据采集及分析、分步骤性能调优和生成性能分析报告。当用户需要分析
  PyPTO 算子性能、进行性能调优、生成性能报告时使用此技能。触发词：算子性能调优、性能分析、自动调优、性能优化、泳道图分析。
original-name: pypto-op-perf-tune
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# PyPTO 算子性能分析和自动调优

## 概述

此技能提供 PyPTO 算子性能调优的完整工作流程，包括精度校验、性能数据采集、性能分析和迭代调优。

**⚠️ 环境要求**：本技能全程需要 NPU 环境。调优启动前必须通过环境检查（编排器 S1_SETUP 子阶段 S1a），包含 NPU 可用性、PTO-ISA 兼容性、PyPTO 及 torch_npu 校验。

## ⛔ 启动方式（必读）

**性能调优必须通过编排器（tune-orchestrator）驱动整个流程，确保严格按照步骤顺序执行。**

**报告模板关系说明**：本技能包含三套模板：（1）阶段交接摘要模板（7节，步骤 4.2 定义，编排器控制）；（2）最终状态看板（步骤 5.1 内联）；（3）perf-analyzer 性能报告模板（`perf-analyzer/templates/performance_report_template.md`）。最终调优报告以阶段交接摘要为骨架，整合 perf-analyzer 报告的性能数据和最终状态看板。

### 启动步骤

```
1. 加载编排器子技能（全程常驻，不卸载）：
   → 加载 tune-orchestrator/SKILL.md

2. 编排器接管流程控制后，按编排器的状态机逐步推进。
   编排器会在需要时指示你加载领域子技能：
   - S3_ANALYZE 阶段     → 加载 perf-analyzer/SKILL.md
   - PHASE_FRONTEND 阶段 → 加载 tune-frontend/SKILL.md
   - PHASE_SWIMLANE 阶段 → 加载 tune-swimlane/SKILL.md
   - PHASE_INCORE 阶段   → 加载 tune-incore/SKILL.md
```

### 编排器与领域子技能的关系

```
┌─────────────────────────────────────────────────┐
│    tune-orchestrator（常驻，全程不卸载）           │
│    状态机 + 门控检查 + Todo 强制 + 迭代轮次控制    │
│         │                    │                │
│    ┌────▼────┐   Task   ┌────▼─────┐          │
│    │tune-    │ subagent │tune-      │          │
│    │frontend │ ──►      │swimlane   │ ──►      │
│    │(按需加载) │  隔离    │(按需加载)  │          │
│    └─────────┘          └───────────┘          │
│                              │                  │
│                         ┌────▼────┐            │
│                         │tune-    │            │
│                         │incore   │            │
│                         │(按需加载) │            │
│                         └─────────┘            │
└─────────────────────────────────────────────────┘
```

### 状态映射：编排器状态 ↔ 主技能步骤

**第一级：主流程状态 ↔ 主技能步骤**

| 编排器状态 | 对应主技能步骤 | 具体内容 | 编排器完成条件 |
|-----------|--------------|---------|--------------|
| INIT | 步骤 0：确定调优目标 | 询问用户目标、计算目标值、设置终止条件 | 用户确认目标性能值 |
| S1_SETUP | 步骤 1：环境检查 + 精度校验 | 加载 pypto-environment-setup 检查环境（S1a），环境通过后执行精度校验（S1b） | 环境检查清单全部 ✅ + 精度通过 |
| S2_COLLECT | 步骤 2：性能数据采集 | 启用 debug_options、运行用例、确认数据文件（2.1-2.3） | swimlane.json 存在 |
| S3_ANALYZE | 步骤 3：性能数据分析 | 加载 perf-analyzer、生成报告、建立基准（3.1-3.3） | 性能报告文件存在 |
| S4_TUNE | 步骤 4：分步骤调优 | 见下方第二级映射 | 3个PHASE全部完成 |
| S5_REPORT | 步骤 5：生成报告 | 输出最终看板、填充报告、保存（5.1） | 报告已保存 + debug_options已还原 |

**第二级：S4_TUNE 子阶段 ↔ 步骤 4 各子步骤**

| 编排器 PHASE | 对应步骤 4 子步骤 | 加载的子技能 | 优化内容 | 退出条件 |
|-------------|----------------|------------|---------|---------|
| PHASE_FRONTEND | 步骤 4.0 第1步 + 步骤 4.1 开箱部分 + 编排器 ITER 迭代循环 | tune-frontend | 代码写法、TileShape、BLOCK_SIZE、基础 runtime_options | 达标 或 连续5轮无提升 |
| PHASE_SWIMLANE | 步骤 4.0 第2步 + 步骤 4.1 决策树 + 编排器 ITER 迭代循环 | tune-swimlane | 核使用率分析、负载均衡、合图、Stitch、调度策略、TileShape 深度调优 | 达标 或 连续8轮无提升 |
| PHASE_INCORE | 步骤 4.0 第3步 + 步骤 4.1 决策树 + 编排器 ITER 迭代循环 | tune-incore | 指令级优化、核内流水、特殊Shape | 达标 或 达到理论上限 |

**第三级：迭代轮次子状态 ↔ 编排器 ITER 迭代循环中的各环节**

| 编排器迭代子状态 | 具体操作 | 对应主技能 |
|----------------|---------|---------|
| ITER_START | 根据子技能指南或决策树（步骤 4.1）选择一个优化点 | 步骤 4.1 |
| ITER_MODIFY | 每次只修改一个参数 | 步骤 4.0 |
| ITER_VERIFY | 运行测试用例，按步骤 1.3 的检查流程验证 | 步骤 1.3 |
| ITER_MEASURE | 采集性能数据，对比基准性能，计算提升百分比 | 步骤 2-3 |
| ITER_RECORD | 更新 Todo 性能记录表 | 编排器 Todo |
| ITER_JUDGE | 判断是否达标 / 连续无提升次数是否达阈值 | 编排器退出条件 |
| ITER_ROLLBACK | 回退代码修改，按编排器的异常处理记录 | 编排器异常处理 |

**阶段交接 ↔ 编排器 PHASE_SUMMARY**

| 编排器强制动作 | 对应编排器章节 | 说明 |
|-------------|-------------|------|
| PHASE 退出时生成摘要 | orchestrator「阶段交接摘要模板」 | 按 7 节模板生成完整摘要 |
| ⛔ 通过 Task subagent 进行隔离 | orchestrator「阶段交接三步流程」 | 强制门控！隔离为独立会话 |
| 交接并进入下一阶段 | orchestrator「阶段交接三步流程」 | 按流程图校验→确认→启动新 subagent |
| 校验摘要完整性 | orchestrator「摘要校验清单」 | 逐项确认检查项 |

### 禁止事项

- ❌ 禁止不加载编排器直接开始调优
- ❌ 禁止在编排器未指示时自行加载领域子技能
- ❌ 禁止跳过编排器的自检流程
- ❌ 禁止跳过环境检查（S1_SETUP S1a）直接进入精度校验
- ❌ 禁止卸载编排器后继续调优

---

## 核心原则

**⚠️⚠️⚠️ 非常重要：所有的调优验证必须上板执行，拒绝理论猜测，凭空捏造！！！**

### 1. 性能调优前提（最高优先级）
**⚠️ 非常重要：性能调优必须建立在精度正确的基础上！**

**⛔ 禁止：没有精度验证通过的记录，绝对禁止进入任何调优步骤！**

**核心要求**：
1. ✅ **精度必须通过**：首先确保算子精度校验通过，才能进行性能调优
2. ✅ **每次验证精度**：每次调优修改后，必须重新验证精度（不要怕麻烦！）
3. ❌ **精度失败不修复**：
   - 首轮失败：不进行修复，可以换卡尝试，多次失败让用户确认
   - 调优修改导致失败：可以进行简单分析后，如果不能解决，则回退修改，记录失败原因，尝试其他优化方案
   - ⚠️ 精度问题是算子实现问题，不是调优能解决的，可以尝试，但不强制解决

### 2. 主动学习原则

**⚠️ 重要：拒绝盲目调优，主动查询，主动学习**

1. 拒绝盲目无脑试错式调优
不瞎猜、不瞎改、不凭感觉乱配置。
2. 以文档 / 资料库为依据
遇到问题查官方文档、权威资料、经典案例。
3. 遇到不清晰的接口，不确定使用方法时，主动查询 API 接口文档

**资料库**
1. [高性能编程实践](../../../models) -- 介绍了很多高性能的编程案例，可以参考其中的高性能写法进行优化
2. [API 接口文档](../../../docs/zh/api) -- 介绍了整个 pypto 仓库的所有接口及调优参数使用说明

### 3. 阶段摘要与上下文隔离原则

**⛔ 阶段切换时的摘要生成和上下文隔离由编排器（tune-orchestrator）统一控制，详见 orchestrator 的「阶段交接摘要模板」和「PHASE_SUMMARY」状态。**

**核心要求**：
1. ✅ 每个子技能阶段结束时，编排器自动触发 PHASE_SUMMARY 状态
2. ✅ 生成阶段交接摘要 → 通过 Task subagent 进行隔离 → 启动下一阶段
3. ❌ 禁止：不做摘要直接进入下一阶段
4. ❌ 禁止：生成摘要但不启动 Task subagent 隔离

---

## 步骤 0：确定性能调优目标

**⚠️ 重要：必须明确性能目标，否则无法判断何时停止调优！**

### 0.1 询问用户性能目标

**必须询问用户**：
```
请明确性能调优目标：
- 需要提升几倍性能？（例如：提升 5 倍）
- 或者需要达到多少执行时间？（例如：≤5000 us）
```

**如果用户未说明**，请主动询问，不要猜测！

### 0.2 计算具体目标值

根据用户输入计算具体目标：
```python
# 示例：用户要求提升 5 倍
原始执行时间 = 27469.66 us
目标执行时间 = 原始执行时间 / 5 = 5493.93 us

# 示例：用户要求执行时间≤5000 us
目标执行时间 = 5000 us
```

### 0.3 设置调优终止条件

**自动终止条件**：
1. ✅ 达到性能目标（执行时间 ≤ 目标值）
2. ✅ 核心利用率 > 80% 且 气泡率 < 10%
3. ✅ 达到调优时间限制（默认 12 小时）

**手动终止条件**：
1. 用户明确要求停止（编排器 ITER_JUDGE 中检查）

---

## 步骤 1：环境检查 + 精度校验

### 1.0 环境检查（调优前置门控）

> ⚠️ **此步骤由编排器 S1_SETUP 的 S1a 子阶段驱动，开始调优前强制执行，不可跳过。**

**操作流程**：

1. 加载 `pypto-environment-setup` 技能
2. 逐项检查以下清单：

```
调优环境检查清单（⛔ 全部 ✅ 才能进入精度校验）：
□ NPU 环境可用：npu-smi info 正常，torch_npu.npu.device_count() > 0
  - 使用 torch_npu.npu.device_count() 确认卡数（npu-smi info 可能截断）
□ TILE_FWK_DEVICE_ID 已设置为空闲 chip id
□ PTO-ISA 兼容性通过：
  - 检测命令：grep -rq "DivAlgorithm" "${PTO_TILE_LIB_CODE_PATH}/include/pto/"
  - 失败则：git clone https://gitcode.com/cann/pto-isa.git 并 export PTO_TILE_LIB_CODE_PATH
□ PyPTO 已编译安装：python3 -c "import pypto" 无报错
□ torch_npu 可用：python3 -c "import torch_npu; assert torch.npu.is_available()"
□ 算子文件存在且语法正确：python3 -c "import py_compile; py_compile.compile('<op_file>', doraise=True)"
```

**不通过时的处理**：
- 按 pypto-environment-setup 技能中的修复流程执行
- 修复后重新检查清单，全部 ✅ 才放行到精度校验
- 向用户报告环境状态（含 PTO-ISA 来源、可用卡数等关键信息）

---

### 1.1 编译策略

**⚠️ 重要：首次进行精度校验，需要进行编译。**
**⚠️ 重要：如果只修改了算子测试或 impl 代码，直接运行即可，不需要编译。**

| 修改类型 | 是否需要编译 | 原因 |
|---------|------------|------|
| 首次执行 | ✅ 需要编译 | 第一次执行需要更新 whl 包 |
| 算子测试或 impl 代码（*.py） | ❌ 不需要 | Python 代码即时生效 |
| framework 代码 | ✅ 需要编译 | C++ 代码需要重新编译 |
| python/pypto目录 | ✅ 需要编译 | 核心框架代码 |

**编译命令**（仅在需要时执行）：
```bash
# 执行编译
python3 build_ci.py -f python3 --disable_auto_execute
# 设置环境变量
export PYTHONPATH=./pypto/build_out/:$PYTHONPATH
export LD_LIBRARY_PATH=./pypto/build_out/pypto/lib/:$LD_LIBRARY_PATH
```

### 1.2 执行算子用例

```bash
python3 custom/operator_name/operator.py --run-mode npu
```

### 1.3 精度校验（⛔ 强制检查点)

**⛔ 禁止：必须完成本步骤并通过后，才能进入步骤2！**

**执行精度校验**：
```bash
python3 custom/operator_name/operator.py --run-mode npu
```
**⛔ 强制检查流程（每次调优修改后必须执行）**：
1. ✅ 必须运行测试用例
2. ✅ 命令必须正常退出（无 timeout、无 segfault）
3. ✅ 输出中必须包含 "passed" 或 "success"，不包含 "failed"/"error"/"timeout"
4. ✅ 必须记录精度验证结果（包含验证时间和命令）
5. ❌ 禁止：假设精度通过、跳过验证、使用之前的验证结果

**⛔ 强制记录验证结果（必须填写）**：
```markdown
### 精度验证记录
- 验证时间: YYYY-MM-DD HH:MM:SS
- 验证命令: python3 xxx.py --run-mode npu
- 验证结果: ✅ 通过 / ❌ 失败
- 关键输出: [粘贴 "test passed" 或报错信息]
```

**✅ 通过：继续执行步骤 2（性能数据采集）**

然后启用性能数据采集（修改 debug_options）

**❌ 失败：用户确认处理**


**失败处理流程**：

1. **首次失败**：
   - ⚠️ **不进行修复**，需要用户自行确认
   - 可以尝试换卡运行（更换 TILE_FWK_DEVICE_ID）
   - 检查环境配置是否正确

2. **换卡尝试**：
   ```bash
    # 查看可用 NPU 卡
   npu-smi info

   # 尝试其他卡
   export TILE_FWK_DEVICE_ID=1  # 或其他可用卡号
   python3 custom/operator_name/operator.py --run-mode npu
   ```

3. **多次失败或超时**：
   - 如果尝试多次（建议 3 次）仍然失败
   - 或运行超时（建议 5 分钟）
   - **停止调优**，让用户确认是否继续

**⚠️ 重要提示**：
- 精度问题是算子实现的问题，不是性能调优能解决的
- 性能调优建立在精度正确的基础上
- 如果精度无法通过，应该先修复算子实现

---

## 步骤 2：性能数据采集

### 2.1 启用性能数据采集

在算子实现文件中，修改 `@pypto.frontend.jit` 装饰器，添加 `debug_options` 参数：

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1}  # 启用性能数据采集
)
def kernel_function(...):
    # 算子实现
    pass
```

**`debug_options` 参数说明**：

| 参数 | 作用 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `runtime_debug_mode` | 启用运行时调试模式，生成泳道图、气泡分析等性能数据文件 | 不设置（不生成性能数据） | `1`（启用） |

**影响范围**：
- 会增加编译时间
- 会增加输出文件大小（泳道图 JSON 可能有数十 MB）
- 不影响算子计算精度

**使用示例**：
```python
# 调优前：添加 debug_options
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1}
)

# 调优结束后：移除 debug_options（还原为原始状态）
@pypto.frontend.jit(
    # debug_options 已移除
)
```

**⚠️ 重要提示**：性能调优任务结束时（S5_REPORT 阶段），必须将 `debug_options` 移除或置为空字典，避免影响正常使用性能。

**⚠️ 基准性能差异说明**：调优全程的性能数据在 `debug_options={"runtime_debug_mode": 1}` 下采集，用于横向对比优化效果。调优结束后移除 debug_options，还原为生产配置即可。

### 2.2 重新运行（不需要编译）

如果只修改了算子 impl 代码，直接运行即可：
```bash
python3 custom/operator_name/operator.py --run-mode npu
```

### 2.3 性能数据文件位置

**⚠️ 关键：output 目录的位置取决于执行算子命令时的工作目录（即 `python3 xxx.py --run-mode npu` 时所在的目录）。**

输出目录格式为 `<执行命令时的工作目录>/output/output_<时间戳>/`，典型场景：

| 执行场景 | 执行命令 | output 目录位置 |
|---------|---------|----------------|
| 在算子目录下执行 | `cd custom/op && python3 op.py --run-mode npu` | `custom/op/output/output_*/` |
| 在项目根目录执行 | `python3 custom/op/op.py --run-mode npu` | `./output/output_*/` |

**查找最新输出目录**（需在正确的工作目录下执行）：
```bash
ls -lt output/ | head -n 2
```

输出目录下包含以下文件：
- `merged_swimlane.json` - 泳道图数据文件
- `machine_runtime_operator_trace.json` - 性能追踪文件
- `bubble_analysis.log` - 气泡分析报告

---

## 步骤 3：性能数据分析

**⚠️ 重要：这个过程中的优化建议用于后续分步骤性能分析及调优时使用，不要在这里立即开始优化！**

### 3.1 分析性能

使用 `perf-analyzer` 子技能，分析性能数据，生成性能报告和优化建议。
```bash
# 加载性能分析技能
Read perf-analyzer/SKILL.md
```

### 3.2 查看性能报告

性能报告保存在 `output/output_时间戳/performance_analysis_report.md`。

### 3.3 建立性能基准

**必须记录基准性能**：
```markdown
## 基准性能（未优化）
- 执行时间: XXX us
- 核心利用率: XX%
- 气泡率: XX%
- 负载均衡度: XX%
```

---

## 步骤 4：分步骤性能分析及调优

**⚠️ 重要：必须按顺序加载子技能获取详细调优指南！**

### 4.0 调优流程总览

**固定执行顺序**：

```
第1步：开箱性能调优
├─ 加载 tune-frontend 子技能
├─ 根据性能基准优化代码写法、TileShape、BLOCK_SIZE
├─ ⚠️ 不需要查看性能报告的详细分析，只需对比性能基准
├─ 建立性能基准
└─ 📋 编排器触发 PHASE_SUMMARY（摘要 + Task subagent 隔离）

第2步：深度性能调优
├─ 基于上一阶段交接摘要启动
├─ 加载 tune-swimlane 子技能
├─ 查看性能报告，分析泳道图
├─ 优化调度策略：Stitch 调优、合图调优、L1Reuse优化
├─ 基于性能报告指导优化方向
└─ 📋 编排器触发 PHASE_SUMMARY（摘要 + Task subagent 隔离）

第3步：核内性能调优
├─ 基于上一阶段交接摘要启动
├─ 加载 tune-incore 子技能
├─ 查看性能报告，分析核内瓶颈
├─ 指令级优化、核内流水优化
└─ 特殊 Shape 处理
```

**⚠️ 关键：每个阶段结束后，编排器自动触发 PHASE_SUMMARY 生成阶段交接摘要并通过 Task subagent 隔离（详见 tune-orchestrator/SKILL.md），避免上下文膨胀导致后续调优质量退化！**

**⚠️ 重要说明**：
- **开箱性能调优**：不需要查看性能报告**的详细分析**，但需要对比基准执行时间
- **深度性能调优**：需要查看性能报告，分析泳道图和性能瓶颈
- **核内性能调优**：需要查看性能报告，分析核内指令和流水线

进入和退出条件由编排器的「退出条件详细定义」章节统一管理。

### 4.1 性能问题诊断

**⚠️ 重要：开箱性能调优不需要查看性能报告！**

**开箱性能调优**：直接根据性能基准（执行时间）进行优化，不需要分析详细性能报告

**深度/核内性能调优**：根据性能分析报告，使用决策树选择优化方向。编号对应 [shared/optimization_catalog.md](shared/optimization_catalog.md)「二、按症状索引」。

```
症状A：气泡率 > 10%（对应目录症状A）
  ├─ ⭐⭐⭐ F-3 循环次数优化 → 增大tile size或切块
  ├─ ⭐⭐⭐ F-8 内层unroll → unroll_list=[64,16,4]
  ├─ ⭐⭐   S-9 Stitch调优 → stitch_function_max_num: 128
  ├─ ⭐⭐   S-4/S-5 合图优化 → sg_set_scope 或 nbuffer
  └─ ⭐     S-10 调度策略 → device_sched_mode调整

症状B：核心利用率 < 50%（对应目录症状B）
  ├─ ⭐⭐⭐ F-1 任务粒度检查 → 增大Matmul M/N轴
  ├─ ⭐⭐⭐ F-9 Cube TileShape → 使用推荐配置
  ├─ ⭐⭐⭐ S-1 核使用率分析 → analyze_core_usage.py
  ├─ ⭐⭐⭐ S-2 核填充 → 减小L0/L1增加任务数
  ├─ ⭐⭐   F-4 Reshape全局优化 → reshape(inplace=True)外提+合轴
  ├─ ⭐     S-10 调度策略 → device_sched_mode
  └─ ⭐     S-6/S-7 Cube合图 → L1Reuse或CubeNBuffer（核满后启用）

症状C：负载不均衡（AicoreTime差异 > 20%）（对应目录症状C）
  ├─ ⭐⭐⭐ S-3 负载均衡分析 → 按total(us)排序→调整瓶颈
  ├─ ⭐⭐   S-11 TileShape深度调优 → 减小瓶颈子图L0/L1
  └─ ⭐     S-4 手动合图 → sg_set_scope合并子图

症状D：单task耗时过长（对应目录症状D）
  ├─ ⭐⭐⭐ I-1 小Shape矩阵乘 → Vector预处理reshape
  ├─ ⭐⭐   I-2 L2 Cache策略 → NONE_CACHEABLE（融合算子批量设置权重）
  ├─ ⭐⭐   I-3 冗余计算消依赖 → 复制数据使分支独立
  ├─ ⭐⭐   I-4 尾轴长度优化 → concat/transpose增大尾轴
  ├─ ⭐⭐   I-9 valid_shape尾块零填充避免 → valid_shape标记有效数据范围
  ├─ ⭐⭐   I-6 操作数连续性检查 → reshape/transpose修复非连续输入
  ├─ ⭐⭐   I-7 Gather/Scatter搬运方向优化 → HBM→L1用cube_tile, L1→HBM用assemble
  ├─ ⭐⭐   I-8 submit_before_loop计算与搬运重叠 → submit_before_loop=True
  └─ ⭐     I-5 TileOperation实现检查 → 与Ascend C对比（兜底）
```

### 4.2 阶段摘要数据模板

**⛔ 阶段切换时的摘要生成和上下文隔离由编排器（tune-orchestrator）的 PHASE_SUMMARY 状态统一控制，模板和流程详见 tune-orchestrator/SKILL.md 的「阶段交接摘要模板」章节。**

---

## 步骤 5：生成性能调优报告

### 5.1 报告模板

**⚠️ 重要：调优结束后，必须生成调优报告！**

**S5_REPORT TODO 管理**（编排器 S5_REPORT 状态强制执行）：
1. 更新 Todo 状态机进度：将所有状态（S1-S5）标记为 ✅
2. 输出最终状态看板（见下方「最终调优状态」模板）
3. Todo 中记录最终性能结果（基准→最终执行时间、累计提升百分比）和迭代统计（总轮次、成功率、最佳优化）
4. 确认 debug_options 已还原的记录写入 Todo

**操作步骤**：

1. **输出最终状态看板**：
```markdown
## 📊 最终调优状态

### 目标达成
- 目标: 提升 1 倍 (XX → XX us)
- 实际: 提升XX% (XX → XX us)
- 达成率: XX%
- 状态: ✅达标 / ❌未达标

### 优化总结
- 总轮次: X轮
- 成功率: X%
- 耗时: XX分钟
- 最佳优化: [优化项] (+XX%)

### 性能趋势
基准 → 最终: XX us → XX us
```

2. **生成最终调优报告（填充下方整合模板）**：

   最终调优报告整合三个数据来源：
   - **perf-analyzer 性能数据**：从步骤 3 生成的 `performance_analysis_report.md` 中提取核心利用率、气泡率、负载均衡度等指标
   - **阶段交接摘要**：从三个 PHASE_SUMMARY 中提取已采纳/已失败的优化记录
   - **最终状态看板**：步骤 1 输出的目标达成状态

   ````markdown

# [算子名称] 性能调优报告

   ## 1. 调优概述

   - 算子名称: [名称]
   - 算子文件: [impl文件路径]
   - 调优目标: 提升 X 倍 / ≤X us
   - 实际达成: 提升 XX% (XX us → XX us)
   - 调优时长: XX 分钟
   - 调优总轮次: X 轮

   ## 2. 性能对比

   | 指标 | 原始性能 | 最终性能 | 提升 |
   |------|---------|----------|------|
   | 执行时间 (us) | XX | XX | XX% |
   | 核心利用率 | XX% | XX% | +XX% |
   | 气泡率 | XX% | XX% | -XX% |
   | 负载均衡度 | XX% | XX% | +XX% |

   **性能倍数**: X.XX 倍

   ### 性能评级（perf-analyzer）

   | 指标 | 最终值 | 评级 |
   |------|-------|------|
   | 核心利用率 | XX% | ⭐⭐⭐XX |
   | 气泡率 | XX% | ⭐⭐⭐XX |
   | 负载均衡度 | XX% | ⭐⭐⭐XX |
   | 综合评级 | - | ⭐⭐⭐XX |

   ## 3. 目标达成

   - 目标: [原始目标描述]
   - 实际: [实际结果]
   - 达成率: XX%
   - 状态: ✅已达标 / ❌未达标

   ## 4. 已采纳优化（代码中已生效）

   | # | 阶段 | 优化项 | 修改内容 | 性能收益 | 代码位置 |
   |---|------|--------|---------|---------|---------|
   | 1 | 开箱 | [优化名称] | [具体修改] | +X% | [文件:行号] |
   | 2 | 深度 | [优化名称] | [具体修改] | +X% | [文件:行号] |

   ## 5. 已失败优化（避免重试）

   | # | 阶段 | 尝试项 | 失败原因 | 备注 |
   |---|------|--------|---------|------|
   | 1 | 开箱 | [优化名称] | [精度失败/OOM/性能回退] | [补充说明] |

   ## 6. 最佳配置（最终代码关键参数）

   ```python
   @pypto.frontend.jit(
       runtime_options={...},
       pass_options={...}
   )
   BLOCK_SIZE = [最终值]
   TileShape配置 = [最终值]
   ```

   ## 7. 调优记录

   | 轮次 | 阶段 | 编号 | 优化内容 | 执行时间(us) | 变化 | 最终状态 |
   |------|------|------|---------|-------------|------|---------|
   | 基准 | - | - | - | XX.XX | - | ✅ |
   | X-X | 开箱 | F-X | [优化内容] | XX.XX | -X% | ✅ |
   | X-X | 深度 | S-X | [优化内容] | XX.XX | +X% | ❌回退 |
   | ... | ... | ... | ... | ... | ... | ... |

   ## 8. 问题记录

   - [问题1描述]
   - [问题2描述]

   ## 9. 性能数据文件

   - 泳道图: {output_dir}/merged_swimlane.json
   - 气泡分析: {output_dir}/bubble_analysis.log
   - 性能追踪: {output_dir}/machine_runtime_operator_trace.json

   ---
   *报告生成时间: {timestamp}*

   ````

3. 保存调优报告到算子目录下（文件名：`{op_name}_tuning_report.md`）

4. **还原 debug_options**：将 `@pypto.frontend.jit` 中的 `debug_options` 移除或置为空，避免影响正常使用性能

## 常见错误

### 错误 1：跳过子技能直接调优

**错误表现**：
- 完成步骤3（性能数据分析）后，直接开始尝试优化
- 没有加载对应的子技能获取详细指南
- 凭经验或猜测进行调优

**正确做法**：
1. 根据性能分析报告判断调优方向（参考步骤 4.1 决策流程）
2. 读取对应子技能的 SKILL.md 文件
3. 按照子技能中的详细指南执行调优
4. 严格执行"修改一处 → 测试 → 验证"的迭代流程

### 错误 2：一次性修改多处优化点

**错误表现**：
- 同时修改多个配置参数
- 全部修改完才测试性能

**正确做法**：
- 每次只修改一个优化点
- 修改后立即测试
- 验证精度和性能
- 记录结果后再尝试下一个优化点

### 错误 3：不验证精度

**错误表现**：
- 修改代码后直接测试性能，不验证精度
- 认为小改动不会影响精度

**正确做法**：
- **每次修改都要验证精度，不要怕麻烦！**
- 即使是小改动，也要验证精度
- 精度失败，不要立即回退，要进行简单分析尝试后，如果还是不能解决，则回退

---

## 参考资料

### 子技能
- [tune-orchestrator](tune-orchestrator/SKILL.md) - ⛔ 流程编排控制器（最先加载，全程常驻）
- [perf-analyzer](perf-analyzer/SKILL.md) - 性能分析
- [tune-frontend](tune-frontend/SKILL.md) - 开箱性能调优
- [tune-swimlane](tune-swimlane/SKILL.md) - 深度性能调优
- [tune-incore](tune-incore/SKILL.md) - 核内性能调优

### 案例和文档
- [性能调优文档](../../../docs/zh/tutorials/debug/performance.md)
- [Matmul 高性能编程](../../../docs/zh/tutorials/debug/matmul_performance_guide.md)
- [性能优化案例](../../../docs/zh/tutorials/debug/performance_case_quantindexerprolog.md)
- [性能调优报告模板](./perf-analyzer/templates/performance_report_template.md)
- [高性能编程实践](../../../models/)
- [API 接口文档](../../../docs/zh/api/)
