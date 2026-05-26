---
name: external-gitcode-ascend-atb-csv-testcase-generator
description: 'ATB CSV 测试用例生成技能。当用户需要为 ATB 算子创建 CSV 格式的泛化测试用例时调用此技能。 覆盖：正例设计、反例设计、性能测试用例、CSV
  格式规范。

  '
keywords:
- atb
- csv
- test
- testcase
- 昇腾
- 泛化测试
- 正例
- 反例
- 性能测试
metadata:
  author: ascend-transformer-boost-team + Claude Code + deepseek-v4-pro
  version: 1.4.0
  created: '2026-04-19'
  updated: '2026-04-29'
  skill-type: testcase
  gates:
  - id: gate-csv
    description: CSV 测试用例确认
    trigger: CSV 用例设计完成后，必须等待用户确认再进入实际迁移阶段
hooks:
  PreToolUse:
  - matcher: Write|Edit|Bash
    hooks:
    - type: command
      command: ([ -z "$ATB_REPO_PATH" ] || [ ! -d "$ATB_REPO_PATH" ]) && echo '[PATH
        CHECK] ATB_REPO_PATH 未设置或无效，请先向用户获取 ATB 路径' >&2 || true
  PostToolUse:
  - matcher: Write
    hooks:
    - type: command
      command: echo '[HIL GATE 2] CSV 用例已生成/修改。若 CSV 用例设计完成，请向用户展示确认检查表并等待"确认通过"。未收到用户确认前，严禁进入实际迁移阶段。'
  Stop:
  - hooks:
    - type: command
      command: echo '[CHECK] 确认 CSV 用例已获用户确认（Gate 2）后，方可结束此技能。'
original-name: atb-csv-testcase-generator
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ATB CSV Testcase Generator

## 功能概述

此技能用于为 ATB 算子生成 CSV 格式的泛化测试用例，支持正例、反例和性能测试用例设计。

## 调用时机

在以下情况下调用此技能：
- 用户需要为 ATB 算子创建 CSV 测试用例
- 用户需要生成正例、反例或性能测试用例

## 任务分块

| 类型 | 说明 | 数量建议 |
|------|------|----------|
| 正例 | 验证正常功能，覆盖典型场景 | 5-15 个 |
| 反例 | 覆盖所有参数校验分支 | 10-20 个 |
| 性能测试 | 大小 shape 性能验证 | 5-10 个 |

## 任务前提知识

1. **参考已有 CSV 文件**：`tests/apitest/opstest/csv/` 目录下有丰富的参考用例
2. **性能测试标记**：`TestType` 列填写 `Performance` 即为性能测试
3. **预期错误标记**：`ExpectedError` 列填写错误类型，如 `NO_ERROR`、`I:ERROR_INVALID_TENSOR_DIM` 等

<details>
<summary>正例设计原则（点击展开）</summary>

### 数据类型覆盖
- float16 (最常用)
- bf16 (高精度场景)

### Shape 典型性
包含主流模型的常见 shape：
- **Llama**: 512, 7168 (hidden_size=3584)
- **GPT**: 256, 12288 (hidden_size=6144)
- **Mini batch**: 1-16 tokens
- **Large batch**: 1024-2048 tokens

### 边界条件
- UB 上限边界（如 hidden_size=31424）
- 非对齐 hidden_size（如 130, 65）
- 最小 shape（如 1, 64）

### 正例模板
```csv
CaseNum|CaseName|OpName|OpParam|InNum|InDType|InFormat|InShape|OutNum|OutDType|OutFormat|OutShape|DataGenType|DataGenRange|InTensorFile|OutTensorFile|TestType|TestLevel|FromModel|SocVersion|ExpectedError
1      |XxxBase |XxxOperation|{"param":0}|1|float16|nd|128,4096|2|int8;float|nd;nd|128,2048;128|customize|-100,100;|||||||Ascend910B;Ascend950|NO_ERROR
```
</details>

<details>
<summary>反例设计原则（点击展开）</summary>

### 数据类型错误
```csv
# 只支持 float16/bf16，不支持 int64
SwigluQuantErrorDtypeInt64|SwigluQuantOperation|{"quantType":0}|1|int64|nd|128,4096|...|I:ERROR_INVALID_TENSOR_INI_MATCH
```

### 维度错误
```csv
# dimNum 必须为 2
SwigluQuantErrorDimNum|SwigluQuantOperation|{"quantType":0}|1|float16|nd|128,4096,1|...|I:ERROR_INVALID_TENSOR_DIM_NUM
```

### 维度值错误
```csv
# hidden_size 必须能被 2 整除
SwigluQuantErrorHiddenSizeOdd|SwigluQuantOperation|{"quantType":0}|1|float16|nd|128,4097|...|I:ERROR_INVALID_TENSOR_DIM
```

### 参数错误
```csv
# quantType 只能为 0
SwigluQuantErrorParam|SwigluQuantOperation|{"quantType":1}|1|float16|nd|128,4096|...|C:ERROR_INVALID_PARAM
```

### Shape 边界超限
```csv
# UB 上限检查
SwigluQuantErrorHiddenSizeTooLarge|SwigluQuantOperation|{"quantType":0}|1|float16|nd|5,15714|...|I:ERROR_INVALID_TENSOR_DIM
```

### 反例覆盖检查表

| 校验类型 | ExpectedError | 覆盖场景 |
|----------|--------------|----------|
| 数据类型 | `I:ERROR_INVALID_TENSOR_INI_MATCH` | int64, int32, float32, bool |
| 数据格式 | `I:ERROR_INVALID_TENSOR_INI_MATCH` | fractal_z, ncdhw 等 |
| 维度数量 | `I:ERROR_INVALID_TENSOR_DIM_NUM` | 1D, 3D, 4D |
| 维度值 | `I:ERROR_INVALID_TENSOR_DIM` | 0, 奇数, 超限 |
| 参数校验 | `C:ERROR_INVALID_PARAM` | 非法参数值 |
| 输出 Shape | `S:ERROR_INVALID_TENSOR_DIM` | shape 不匹配 |
| UB 边界 | `I:ERROR_INVALID_TENSOR_DIM` | hidden_size 过大 |
</details>

<details>
<summary>性能测试设计原则（点击展开）</summary>

### 大小 Shape 覆盖
```csv
# 大 shape
SwigluQuantPerformanceLarge|SwigluQuantOperation|{"quantType":0}|1|float16|nd|2048,16384|...|Performance|Ascend910B;Ascend950|NO_ERROR

# 中 shape
SwigluQuantPerformanceMedium|SwigluQuantOperation|{"quantType":0}|1|float16|nd|512,8192|...|Performance|Ascend910B;Ascend950|NO_ERROR

# 小 shape
SwigluQuantPerformanceSmall|SwigluQuantOperation|{"quantType":0}|1|float16|nd|64,256|...|Performance|Ascend910B;Ascend950|NO_ERROR
```

### 不同数据类型
```csv
# bf16 性能
SwigluQuantPerformanceBf16|SwigluQuantOperation|{"quantType":0}|1|bf16|nd|1024,8192|...|Performance|Ascend910B;Ascend950|NO_ERROR
```

### 边界值测试
```csv
# hidden_size 边界
SwigluQuantPerformanceHidden4096|SwigluQuantOperation|{"quantType":0}|1|float16|nd|256,8192|...|Performance|Ascend910B;Ascend950|NO_ERROR
```
</details>

<details>
<summary>CSV 格式规范（点击展开）</summary>

### 列说明
| 列号 | 名称 | 说明 |
|------|------|------|
| 1 | CaseNum | 用例编号 |
| 2 | CaseName | 用例名称 |
| 3 | OpName | 算子名称（如 `SwigluQuantOperation`） |
| 4 | OpParam | JSON 格式参数字符串 |
| 5 | InNum | 输入张量数量 |
| 6 | InDType | 输入数据类型（分号分隔） |
| 7 | InFormat | 输入数据格式（分号分隔） |
| 8 | InShape | 输入形状（分号分隔） |
| 9 | OutNum | 输出张量数量 |
| 10 | OutDType | 输出数据类型 |
| 11 | OutFormat | 输出数据格式 |
| 12 | OutShape | 输出形状 |
| 13 | DataGenType | 数据生成类型（通常为 `customize`） |
| 14 | DataGenRange | 数据生成范围 |
| 15 | InTensorFile | 输入张量文件（空） |
| 16 | OutTensorFile | 输出张量文件（空） |
| 17 | TestType | 测试类型（空或 `Performance`） |
| 18 | TestLevel | 测试级别（空） |
| 19 | FromModel | 来源模型（空） |
| 20 | SocVersion | 目标 SoC（如 `Ascend910B`、`Ascend950`。多个设备用 `;` 分隔：`Ascend910B;Ascend950`） |
| 21 | ExpectedError | 期望错误（`NO_ERROR` 或错误类型） |

### 数据类型映射

在 CSV 中使用 data_generation.py 支持的数据类型名称：

| 实际类型 | CSV 中使用 |
|----------|-----------|
| float32 | `float` |
| float16 | `float16` 或 `half` |
| bfloat16 | `bf16` 或 `bfloat16` |
| int8 | `int8` 或 `char` |
| int32 | `int32` 或 `int` |
| int64 | `int64` 或 `long` |

### 常用 ExpectedError 类型

| 错误类型 | 说明 | 触发场景 |
|----------|------|----------|
| `NO_ERROR` | 无错误，正例预期 | 正常执行 |
| `I:ERROR_INVALID_TENSOR_INI_MATCH` | 张量初始化不匹配 | dtype/format 错误 |
| `I:ERROR_INVALID_TENSOR_DIM_NUM` | 维度数量错误 | dimNum 不符合预期 |
| `I:ERROR_INVALID_TENSOR_DIM` | 维度值错误 | 维度值超限/无效 |
| `C:ERROR_INVALID_PARAM` | 参数错误 | 非法参数值 |
| `S:ERROR_INVALID_TENSOR_DIM` | 输出 Shape 错误 | 输出 tensor shape 不匹配 |
</details>

### 多设备测试设计原则

**设计 CSV 测试用例时，必须同时覆盖 Ascend910B 和 Ascend950 设备。**

#### 合并 vs 分离规则

| 情况 | SocVersion | 说明 |
|------|-----------|------|
| 910B 和 950 **完全一致**（场景/约束/dtype/format 均相同） | `Ascend910B;Ascend950` | 合并为一条用例 |
| 910B 和 950 **存在任何差异** | 分两条，各写各自 SocVersion | **必须分离** |

#### 必须分离的场景（不限于以下）

| 差异类型 | 示例 | 处理方式 |
|---------|------|---------|
| **dtype 支持不同** | 910B 支持 bf16，950 不支持 | 分两条：910B bf16 / 950 用其他 dtype |
| **format 支持不同** | 910B 支持 ND，950 仅支持 ND_TRANSPOSE | 分两条，各自指定合法 format |
| **参数约束不同** | 910B quantType=0,1；950 quantType=0 | 分两条，各自合法/非法参数 |
| **维度限制不同** | 910B hidden_size≤31424；950 hidden_size≤32768 | 正向/边界分别设计 |
| **ExpectedError 不同** | 同一非法输入，910B 报 ERROR_A，950 报 ERROR_B | 分两条，各自 ExpectedError |
| **性能基准不同** | 910B 和 950 性能预期不同 | 分两条，各自 Performance |

#### 设计流程

1. **先分析接口文档**：对比 910B 和 950 的 ACLNN 接口规格，标记所有差异点
2. **先写共同用例**：无差异的场景用 `Ascend910B;Ascend950`
3. **再写差异用例**：每个差异点，为 910B 和 950 各写一条独立用例
4. **差异用例命名**：`{OpName}{场景}910B` / `{OpName}{场景}950` 便于区分

> **注意**：实际运行测试时仅执行 Ascend910B 用例（当前测试环境限制）。Ascend950 用例由用户在 950 设备上手动验证。

---

## ⚠️ 路径约束（必须执行）

执行此技能前，**必须从用户处获取以下路径**：
- `<ATB_REPO_PATH>`: ATB 仓库路径（如 `{your working path}ascend-transformer-boost`）

---

## 编写步骤

### Step 1: 获取接口规格
从用户或官方文档获取算子的：
- 输入输出定义
- 参数约束
- 数据类型支持

### Step 2: 阅读源码获取校验逻辑
```bash
# 查看参数校验
cat <ATB_REPO_PATH>/src/ops/ops_infer/<op_name>/<op_name>_operation.cpp

# 查找 ERROR_INVALID_* 分支
grep -n "ERROR_INVALID" <ATB_REPO_PATH>/src/ops/ops_infer/<op_name>/<op_name>_operation.cpp
```

### Step 3: 设计正例
覆盖：
- 典型数据类型的基例
- 主流模型的常见 shape
- 边界条件

### Step 4: 设计反例
覆盖所有校验分支：
- 数据类型校验
- 维度数量校验
- 维度值校验
- 参数校验
- 输出 tensor 校验

### Step 5: 设计性能测试
覆盖：
- 大/中/小 shape
- 不同数据类型
- 边界值

### Step 6: 创建 CSV 文件（双文件输出）

**⚠️ 必须产出两个独立 CSV 文件，不可合并为一个**：

```bash
# 创建功能用 CSV（正例 + 反例）
touch <ATB_REPO_PATH>/tests/apitest/opstest/csv/<op_name>_func.csv

# 创建性能用 CSV（性能用例）
touch <ATB_REPO_PATH>/tests/apitest/opstest/csv/<op_name>_perf.csv

# 可选：创建合并全量文件
touch <ATB_REPO_PATH>/tests/apitest/opstest/csv/<op_name>_test.csv
```

#### 拆分规则

| 文件 | 内容 | TestType 列 | 用途 |
|------|------|------------|------|
| `{op}_func.csv` | 所有 TestType 非 Performance 的用例（正例+反例） | 空或 Function | Phase 5a 前台快速验证 |
| `{op}_perf.csv` | 所有 TestType=Performance 的用例 | Performance | Phase 5a 后台 subagent 运行 |
| `{op}_test.csv` | 全部用例（func + perf 合并） | 混合 | 全量参考 / Phase 5b 全量回归 |

#### 拆分理由
- 正例/反例用 `-t 1`（默认），秒级完成
- 性能用例默认 400 次迭代，耗时长且可能 OOM
- 分文件允许 Phase 5a 先快速验证功能，再后台跑性能
- `_func.csv` 和 `_perf.csv` 各有独立完整 header（21 列）

### Step 7: 自动引入 Reviewer Agent 评审 ⭐

**⚠️ 强制步骤 — Gate 2 之前必须执行**

CSV 文件创建后，**必须**调用 `superpowers:requesting-code-review` agent 进行评审，不可跳过。

**评审内容**:

| 评审维度 | 检查内容 |
|---------|---------|
| 完整度 | 正例/反例/性能是否覆盖所有必要场景 |
| 正确性 | OpParam JSON 是否与设计文档参数映射一致 |
| Shape 一致性 | InShape/OutShape 是否与 ACLNN 接口约束一致 |
| 设备分离 | 910B/950 差异场景是否正确分离 SocVersion |
| 格式规范 | 21 列、`|` 分隔符、数据类型名称 |

**评审输入**:
- CSV 文件路径
- 设计文档路径（来自 Phase 1 的 `*_replacement_design.md`）
- ACLNN 接口参数约束（来自详设第 2 章）

**评审通过标准**:
- 无遗漏的正例场景（dtype/shape/边界）
- 反例覆盖所有 `ERROR_INVALID_*` 分支
- 所有 Shape 经 ACLNN 参数约束预检通过
- 无格式错误

**评审问题修复后**，方可进入 Gate 2 用户确认。

### Step 8: 运行测试验证
```bash
source <CANN_PATH>/set_env.sh
source <ATB_REPO_PATH>/output/atb/set_env.sh --cxx_abi=1
cd <ATB_REPO_PATH>/tests/framework/python/CsvOpsTestTool
python3 atb_csv_ops_test.py -i <CSV_FILE_PATH>
```

<details>
<summary>苏格拉底式提问模板（点击展开）</summary>

在开始编写前，确认以下问题：

1. **算子接口**：
   - 输入/输出张量的数量和形状？
   - 支持的数据类型？
   - 参数的合法取值范围？

2. **已有参考**：
   - 是否有同类型算子的 CSV 可参考？
   - 是否已有该算子的测试用例？

3. **特殊约束**：
   - 是否有 UB 限制？
   - 是否有内存限制？
   - 是否有特定 shape 要求？
</details>

## 输出模板

```
## CSV 测试用例设计

### 算子信息
- **算子名称**: XxxOperation
- **接口规格**: [简述]

### 正例列表 (N 个)
| CaseName | InShape | OutShape | 说明 |
|----------|---------|----------|------|
| XxxBase | 128,4096 | 128,2048 | 基例 |
| ... | ... | ... | ... |

### 反例列表 (M 个)
| CaseName | ErrorType | 覆盖分支 |
|----------|-----------|----------|
| XxxErrorDtype | ERROR_INVALID_TENSOR_INI_MATCH | dtype 校验 |
| ... | ... | ... |

### 性能测试列表 (P 个)
| CaseName | InShape | 说明 |
|----------|---------|------|
| XxxPerfLarge | 2048,16384 | 大 shape |
| ... | ... | ... |

### 统计
- 正例: N 个
- 反例: M 个
- 性能测试: P 个
- 总计: N+M+P 个

### 保存路径
- **功能 CSV**: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_func.csv`（正例+反例）
- **性能 CSV**: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_perf.csv`（性能用例）
- **合并 CSV**（可选）: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_test.csv`
- **备份/预览**: `{WORKING_DIR}/`
```

## Human in the Loop: 用户确认检查点 ⭐

**⚠️ 强制检查点 - 不可跳过**

CSV 用例设计完成后，必须等待用户确认后方可进入实际迁移阶段。

### 确认前准备
1. **保存 CSV**: 将生成的 CSV 用例保存至 ATB 仓库目录
   - **功能 CSV**: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_func.csv`（正例+反例）
   - **性能 CSV**: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_perf.csv`（性能用例）
   - **合并 CSV**（可选）: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_test.csv`

2. **准备确认清单**: 供用户逐项确认

### 用户确认检查表

```markdown
## CSV 用例确认 - {OpName}

### 待确认文件
- **功能 CSV**: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_func.csv`（正例 X 个 + 反例 Y 个）
- **性能 CSV**: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_perf.csv`（性能 Z 个）

### 确认检查表
| 检查项 | 说明 | 用户确认 |
|--------|------|----------|
| 正例覆盖典型场景 | 包含 float16/bf16 基础用例，主流模型常见 shape | [ ] |
| 反例覆盖所有校验分支 | 覆盖所有 ERROR_INVALID_* 类型 | [ ] |
| 性能测试覆盖大小 shape | 包含大/中/小 shape 测试 | [ ] |
| CSV 已拆分为双文件 | `_func.csv` 和 `_perf.csv` 各自独立 | [ ] |
| CSV 格式符合规范 | 列数=21，分隔符=|，无多余空格 | [ ] |
| 数据类型名称正确 | 使用 data_generation.py 支持的名称 | [ ] |
| ExpectedError 类型正确 | 与参数校验逻辑匹配 | [ ] |
| 可进入实际迁移阶段 | CSV 用例已通过审查 | [ ] |

### 确认方式
- **通过**: 回复 "确认通过" 或逐项打勾确认
- **修改**: 指出具体问题，返回修改后重新确认

### 后续步骤
确认通过后，将进入 Phase 3: 实际迁移阶段
- 下一步调用: `atb-aclnn-operator-migration`

**⚠️ 未收到用户确认前，禁止进入实际迁移阶段**
```

### 向用户呈现方式

```markdown
我已生成 {OpName} 算子的 CSV 测试用例，已拆分为两个独立文件，请确认以下内容：

📄 **功能 CSV**: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_func.csv`
   - 正例: N 个
   - 反例: M 个
📄 **性能 CSV**: `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_perf.csv`
   - 性能测试: P 个

Phase 5a 测试时将分两步：
1. 前台跑 _func.csv（秒级完成）
2. 后台 subagent 逐个跑 _perf.csv（避免 OOM）

请检查：
1. 正例是否覆盖了所有典型场景和数据类型？
2. 反例是否覆盖了所有参数校验分支？
3. 性能测试的 shape 选择是否合理？
4. CSV 拆分是否正确（正例/反例在 _func，性能在 _perf）？

回复 "确认通过" 进入实际迁移阶段，或指出需要修改的地方。
```

<details>
<summary>保存路径规范（点击展开）</summary>

| 文件类型 | 路径 | 说明 |
|----------|------|------|
| 功能 CSV | `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_func.csv` | 正例+反例，Phase 5a 前台跑 |
| 性能 CSV | `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_perf.csv` | 性能用例，Phase 5a 后台 subagent 跑 |
| 合并 CSV（可选） | `{ATB_REPO_PATH}/tests/apitest/opstest/csv/{op_name}_test.csv` | 全量参考

**变量说明**:
- `ATB_REPO_PATH`: ATB 仓库路径
- `WORKING_DIR`: working_files 目录
- `op_name`: 算子名称小写（如 `{xxx}quant`）
</details>

## 执行结果

> ⚠️ **执行后填写**：技能执行完成后，参照下方格式填写实际执行结果。

### 检查点检查表

| 步骤 | 检查点描述 | 状态 |
|------|-----------|------|
| - | 正例设计完成 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| - | 反例设计完成 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| - | 性能测试设计完成 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| - | CSV 文件已创建（双文件） | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| - | **CSV 已拆分为 _func.csv 和 _perf.csv** | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| - | **用户已确认 CSV 用例** ⭐ | ⬜ 待确认 |

**VERDICT: ✅ SUCCESS / ⚠️ PARTIAL / ❌ FAILED / ⏭️ SKIPPED**

### 问题列表（若有）

| 等级 | 检查点 | 问题描述 | 建议 |
|------|--------|---------|------|
| 🔴 CRITICAL | - | - | - |
| 🟡 WARNING | - | - | - |

### 执行摘要

- **执行时间**：
- **算子名称**：
- **正例数量**：
- **反例数量**：
- **性能测试数量**：
- **通过率**：
