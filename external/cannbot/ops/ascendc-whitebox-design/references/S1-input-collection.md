# Step 1：输入收集

## 全局约束（强制）

**顺序**：1.0 → 1.1 → 1.2 → 1.3 → 1.4，严格逐步执行，禁止合并、跳过或重排。

**禁止抢跑**：每步仅允许其自身描述的操作（如 1.0 仅允许平台检测，1.1 才能用 question，1.2 才能搜索路径），前一步未完成不得启动下一步。

**question 工具规范（Step 1.1 专用）**：
- 单次调用，所有未收集参数作为 `questions` 数组的独立元素逐项呈现（独立 header、独立 question、独立 options）
- 禁止创建"全部使用默认值"或"自定义参数"等聚合选项
- 已从用户消息或前置步骤获取的值，以文字回显，不再放入 questions 数组

---

## 1.0 平台检测

执行 `npu-smi info` 获取当前 NPU 设备信息，从输出中识别芯片名称（Name 列）。**禁止猜测和推导**——必须以硬件检测结果为准。

检测方式（按优先级降级）：
1. 执行 `npu-smi info`，从输出中识别芯片名称
2. 若 npu-smi 不可用 → 读取环境变量 `$ASCEND_SOC_VERSION`
3. 两者均不可用 → 报告「无法检测 NPU 平台，请手动指定」，等待用户输入

> **注意**：禁止使用 `npu-smi info -t board -i 0`（硬编码卡号 0）。

将检测到的芯片名称，结合 `npu-arch` skill 的 `references/npu-hardware-params.md §0 产品映射表` 确定 `npu_arch`（如 DAV_3510/DAV_2201）和 `soc_version`（如 ASCEND950/ASCEND910B）。§0 表中芯片型号列为模糊匹配（如 `910B3` 匹配 `Ascend910B1~B4`）。

产出：`npu_arch`、`soc_version`、`chip_model`。

---

## 1.1 收集用户输入

在 Step 1.0 完成后，使用 question 工具收集以下 3 项输入（已从用户消息或前置步骤获取的值以文字回显，仅将未收集项作为 questions 数组中的独立条目）。

| # | 项 | 提示语 | 必填 | 默认值/可选值 |
|---|-----|-------|------|-------------|
| 1 | 算子名称 | "请输入算子名称" | 是 | 自由文本 |
| 2 | Step 4 闸门 | "是否跳过 Step 4 闸门？" | 是 | 默认不跳过；可选跳过（一次跑完全流程） |
| 3 | 算子路径 | "请选择算子路径获取方式：自动查找 / 手动输入路径" | 是 | 默认「自动查找」 |

---

## 1.2 定位算子路径

> ⚠️ **本步骤由主 Agent 直接执行。禁止派发子 agent。禁止使用 Task 工具。只能使用 Bash 工具执行 find 命令。**

在 Step 1.1 完成后执行。由用户对「算子路径」的选择驱动：
- 选择「自动查找」→ 执行以下搜索逻辑，定位算子路径。
- 选择「手动输入」→ 仅验证用户提供的路径（确认 op_kernel/ 和 op_host/ 目录存在），验证通过则跳过搜索，直接进入 Step 1.3。

**「自动查找」分支（两阶段）**：

> **前提**：用户输入的 `op_name` 始终视为正确的规范名（camel_to_snake(OP_ADD(op_type)) 的结果）。查找失败时不质疑用户输入，而是切换查找策略。

**Phase 1（目录名匹配）**：

> **查找方式（强制）**：限定**当前工作目录（$PWD）**，禁止跨目录搜索。使用 bash `find $PWD -maxdepth 4 -type d -regex '.*/ops-[^/]+/[^/]+/{op_name}'` 按目录名精确匹配（禁止子串匹配）。禁止派发子 agent。

主 Agent 验证每个候选目录同时存在 op_kernel/ 和 op_host/，过滤不合格项。唯一结果→使用；多项→让用户选择。**未找到→进入 Phase 2**。

**Phase 2（OP_ADD 反查）**：

> 当 Phase 1 按目录名未找到时，说明算子目录名与规范 op_name 不一致（如目录 `lamb_next_m_v`，规范名 `lamb_next_mv`）。通过 `_def.cpp` 中的 `OP_ADD(op_type)` 反查。

```bash
python3 {skill_base}/scripts/normalize_op_name.py find --op-name {op_name} --search-root $PWD
```

- 输出 `op_path` 非空 → 使用该路径（唯一结果直接使用；多候选→让用户选择）
- 输出 `op_path` 为空 → 报告"未找到"，不得扩大搜索范围

**「手动输入」分支**：

验证用户提供的路径下存在 `op_kernel/` 和 `op_host/` 目录。验证失败→报错并要求用户修正路径。

---

## 1.3 查询技术参数

在 Step 1.2 完成后执行。加载 `npu-arch` skill，Read `references/npu-hardware-params.md`，按以下规则获取平台参数：

1. **VectorCore 核数**：从 §子型号变化参数 读取匹配 chip_model 的 VectorCore 数（§中 CubeCore:VectorCore = 1:2，取 VectorCore 列）。若存在多个子型号（如 Ascend950PR 有 PCIE/Server），通过 Step 1.0 `npu-smi info` 输出中的 HBM-Usage 容量区分（112 GB=PCIE，128 GB=Server）。
2. **UB 大小**：从 §2 各架构参数 读取对应 NpuArch 的 `ub_size` 值。

以 `platform` 对象传递给子 agent，字段为 `npu_arch` / `soc_version` / `chip_model` / `core_count`（VectorCore 数） / `ub_size`（单位 KB）。

---

## 1.4 确认摘要

在 Step 1.0-1.3 全部完成后执行。**必须使用 question 工具**展示确认摘要并等待用户确认，禁止仅以文字输出方式等待用户回复。

**question 工具规范（Step 1.4 专用）**：
- 必须使用 question 工具的单次调用，将确认摘要作为 `question` 字段内容
- `options` 必须包含「确认，开始分析」和「需要修改」两个选项
- 禁止仅以文字输出（非 question 工具）展示摘要后等待用户回复

确认摘要内容（填入实际值后作为 question 字段）：

```
── 硬件参数（自动检测）──
目标平台 {chip_model}（{npu_arch}），将使用 {core_count} 核、{ub_size}KB UB。
── 用户输入 ──
- 算子名称：{op_name}
- 算子路径：{op_path}
- Step 4 闸门：{gate_status}

结果输出到 {算子源码路径}/tests/whitebox/。
如需修改核数、UB 大小，或有额外特殊条件需添加，请告知。
确认后开始分析。
```

占位符填入规则：
- `{chip_model}`/`{npu_arch}`/`{core_count}`/`{ub_size}` — 由 Step 1.0 检测 + 1.3 映射
- `{op_name}`/`{op_path}`/`{gate_status}` — 由 Step 1.1 用户输入
- `{算子源码路径}` — 即 Step 1.2 确定的算子路径

question 工具选项定义：

| 选项 label | 选项 description |
|------------|-----------------|
| 确认，开始分析 | 所有参数正确，进入 Step 2 |
| 需要修改 | 核数/UB — 主 Agent 更新 `platform` 对象后回到 1.4 重新展示；其它参数 — 回到 1.1 重新收集 |
