---
name: external-gitcode-ascend-atb-aclnn-operator-replacement-designer
description: '自动生成 ATB 到 ACLNN 算子替换的详细设计文档。接收用户提供的 ATB 和 ACLNN 接口文档链接， 输出包含参数映射、开发自测、风险评估的
  7 章结构化设计文档。 TRIGGER when: 用户需要将 ATB 算子替换为 ACLNN 算子并撰写设计文档。

  '
keywords:
- atb
- aclnn
- operator
- replacement
- design
- 算子
- 设计文档
- 昇腾
metadata:
  author: ascend-transformer-boost-team + Claude Code + deepseek-v4-pro
  version: 1.4.0
  created: '2026-04-17'
  updated: '2026-04-29'
  skill-type: design
  gates:
  - id: gate-design
    description: 设计文档确认
    trigger: 设计文档生成完成后，必须等待用户确认再进入 CSV 用例设计阶段
hooks:
  PreToolUse:
  - matcher: Write|Edit|Bash
    hooks:
    - type: command
      command: ([ -z "$CANN_PATH" ] || [ ! -f "$CANN_PATH/set_env.sh" ]) && echo '[PATH
        CHECK] CANN_PATH 未设置或无效，请先向用户获取 CANN 路径' >&2; ([ -z "$ATB_REPO_PATH" ] ||
        [ ! -d "$ATB_REPO_PATH" ]) && echo '[PATH CHECK] ATB_REPO_PATH 未设置或无效，请先向用户获取
        ATB 路径' >&2 || true
  PostToolUse:
  - matcher: Write
    hooks:
    - type: command
      command: echo '[HIL GATE 1] 设计文档已生成/修改。若设计文档已完成，请向用户展示确认检查表并等待"确认通过"。未收到用户确认前，严禁进入
        CSV 用例设计阶段。'
  Stop:
  - hooks:
    - type: command
      command: echo '[CHECK] 确认设计文档已获用户确认（Gate 1）后，方可结束此技能。'
original-name: atb-aclnn-operator-replacement-designer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ATB到ACLNN算子替换详设文档生成器

## 功能概述

该工具能够根据用户提供的ATB和ACLNN接口文档链接，自动生成完整的算子替换详细设计文档。支持多种算子类型，提供标准化的文档结构和内容。

## 调用时机

在以下情况下调用此技能：
- 用户需要将 ATB 算子替换为 ACLNN 算子并撰写设计文档
- 用户已有 ATB 算子接口和 ACLNN 接口文档链接
- 变更涉及 LayerNorm、Softmax、RmsNorm 等标准算子

## 规模检测与分级

**调用此技能时，必须首先执行规模检测，确定使用完整模板还是轻量模板。**

### 检测步骤

```bash
# 在 ATB 仓库中搜索目标算子的 ACLNN Runner 是否已存在
find <ATB_REPO_PATH>/src/ops/ops_infer/<op_type>/ -name '*_aclnn_runner.cpp' -type f 2>/dev/null
find <ATB_REPO_PATH>/src/ops/ops_infer/<op_type>/ -name '*_aclnn_runner.h' -type f 2>/dev/null
```

### 分级标准

| 场景 | 判断条件 | 模板等级 | 说明 |
|------|---------|---------|------|
| **已有 Runner 接入** | `*_aclnn_runner.cpp` 已存在且已注册 `REG_RUNNER_TYPE` | **轻量模板** (4 节) | 仅需参数映射 + 约束 + 风险 |
| **全新迁移** | `*_aclnn_runner.cpp` 不存在 | **完整模板** (7 章) | 需要完整设计文档 |

### 向用户报告检测结果的标���话术

```markdown
## 规模检测结果

算子: {OpType}
ATB 仓库: <ATB_REPO_PATH>

检测: {op_type}_aclnn_runner.{h,cpp} [✅ 已存在 / 不存在]

→ 使用 [轻量模板 / 完整模板] 生成设计文档。
→ 预计生成 [4 节 / 7 章] 文档。
```

### 轻量模板（已有 Runner 模式）

当 ACLNN Runner 已存在时（如仅需修改 Operation 的 CreateRunner 以挂载已有 runner），生成以下精简文档：

```markdown
# {OpType} OPS→ACLNN 替换设计文档（轻量版）

## 文档信息
| 项目 | 内容 |
|------|------|
| 算子名称 | {OpType} |
| 迁移类型 | 挂载已有 ACLNN Runner（{op}_aclnn_runner.cpp 已存在） |
| 变更范围 | 仅修改 {op}_operation.cpp 的 CreateRunner，添加 910B/950 设备检测 |
| 目标设备 | 910B / 950 |

## 需求简述
- 将 {OpType} 算子在 910B/950 上从 OPS 切换至 ACLNN
- 非 910B/950 设备保持 OPS 路径
- ACLNN Runner 已存在且通过 REG_RUNNER_TYPE 注册，仅需接入分发

## 参数映射关系
| ATB参数 | ACLNN参数 | 映射类型 | 说明 |
|---------|-----------|----------|------|
| [从已有 runner 代码提取] | [从已有 runner 代码提取] | [直接/计算] | |

## 规格约束
- 数据类型: [从已有 runner 和接口文档提取]
- 输入约束: [提取]
- 参数约束: [提取]
- 设备相关: 950 上仅支持 {受支持的变体}

## 风险评估
| 风险项 | 风险等级 | 应对策略 |
|--------|---------|---------|
| 设备检测逻辑错误 | 低 | 参考已有 runner (Gather/Split/Repeat) 的分发模式 |
| 变体路由错误 | 中 | 在 CreateRunner 中加严格的 layerType/quantType 过滤 |
| 非目标设备回归 | 低 | 非 910B/950 设备保持 OPS 路径不变 |
```

轻量模板生成规则：
1. 仅保留 4 节：文档信息 + 需求简述 + 参数映射 + 约束 + 风险
2. 跳过完整版的需求详情、规格说明（DFX）、开发自测、实现方案、结论章节
3. 文档仍保存至 `{WORKING_DIR}/{op_type}_replacement_design.md`，标注为轻量版

### 轻量版确认检查表

轻量版使用精简的 3 项确认表：

```markdown
## 设计文档确认 - {OpType}（轻量版）

| 检查项 | 说明 | 用户确认 |
|--------|------|----------|
| 参数映射关系正确 | ATB→ACLNN 映射准确 | [ ] |
| 规格约束清晰 | 数据类型、参数约束完整 | [ ] |
| 风险评估完整 | 风险项和应对策略充分 | [ ] |
```

## ⚠️ 路径约束（必须执行）

执行此技能前，**必须从用户处获取以下路径**：
- `<CANN_PATH>`: CANN 安装路径（如 `/usr/local/Ascend/ascend-toolkit/latest`）
- `<ATB_REPO_PATH>`: ATB 仓库路径（如 `{your working path}ascend-transformer-boost`）

**若用户未提供或路径无效，立即停止并报错。**

脚本校验示例：
```bash
if [ -z "$CANN_PATH" ] || [ ! -f "$CANN_PATH/set_env.sh" ]; then
    echo "ERROR: CANN_PATH 未设置或无效，请用户提供。"
    exit 1
fi
if [ -z "$ATB_REPO_PATH" ] || [ ! -d "$ATB_REPO_PATH" ]; then
    echo "ERROR: ATB_REPO_PATH 未设置或无效，请用户提供。"
    exit 1
fi
```

---

## 前置条件

- ATB 算子接口文档链接（用户需提供）
- ACLNN 算子接口文档链接（用户需提供）
- （可选）算子类型名称（如 LayerNorm）

## WebFetch 失败阻断规则 ⛔

**⚠️ 强制规则 — 不可通过代码反推绕过**

WebFetch 是获取接口文档的途径。以下规则覆盖所有场景：

| 场景 | 行为 |
|------|------|
| WebFetch 两个 URL 均成功 | 继续分析 |
| WebFetch 任一 URL 失败 | **立即停止，打断用户** |
| WebFetch 返回 404/403/超时 | **立即停止，打断用户** |
| ACLNN Runner 已存在于代码仓 | **仍需文档成功** — 代码不能替代官方文档 |

**打断用户的标准话术**：
```markdown
## ⛔ WebFetch 失败 — 需要接口文档

获取以下接口文档失败：
- [URL] → [失败原因]

**请提供以下任一形式**：
1. 可直接访问的文档 URL
2. ACLNN 开源仓中对应算子的接口头文件路径/内容
3. CANN 安装目录下的接口文档（如 `$ASCEND_TOOLKIT_HOME/opp/*/op_proto/` 下的算子定义）
4. 手动提供的完整参数列表（类型、含义、约束）

在没有完整接口文档前，禁止进入设计阶段。
```

**禁止行为**：
- ❌ 从 `*_aclnn_runner.cpp` 代码反推接口参数
- ❌ 从 `infer_op_params.h` 结构体推断 ACLNN 约束
- ❌ 以「代码仓已有实现」为由跳过文档获取

## 工作流程总览

```
规模检测（是否存在 *_aclnn_runner.cpp）→ 接口文档分析 → 参数映射建立 → 按模板等级生成文档
```

> **检查点系统**：每个阶段完成后，请对照 [执行结果](#执行结果) 章节的检查点表确认状态。

## 输入要求

用户需要提供以下信息：
1. ATB算子接口文档链接
2. ACLNN算子接口文档链接
3. 可选：算子类型（如LayerNorm、Softmax、RmsNorm等）

## 核心功能

### 自动分析接口文档
- 解析ATB算子的参数结构
- 解析ACLNN算子的接口定义
- 提取关键参数和约束条件

### 生成标准化文档结构
- 需求详情
- 算子IR与参数配置
- 规格说明（DFX、规格约束、功能冲突）
- 开发自测（功能测试、反例测试、性能测试）
- 实现方案
- 风险评估与应对策略
- 结论

### 智能参数映射
- 自动建立ATB与ACLNN参数之间的映射关系
- 识别参数差异和转换规则
- 处理特殊参数的适配逻辑

<details>
<summary>详细工作流程（点击展开）</summary>

### 阶段1: 接口文档分析

#### ATB算子分析
- 从ATB接口文档中提取算子参数结构
- 识别核心参数（如epsilon、axis、quantType等）
- 确定参数约束条件

#### ACLNN算子分析
- 从ACLNN接口文档中提取接口定义
- 识别输入输出参数
- 确定接口调用方式

### 阶段2: 参数映射建立

| 映射类型 | 处理方式 |
|---------|----------|
| 直接映射 | 参数名和含义完全一致，直接映射 |
| 计算映射 | 根据公式或规则进行转换（如beginNormAxis → normalizedShape） |
| 废弃参数 | 识别不再需要的参数并说明 |
| 新增参数 | 识别新增参数并说明默认值或计算方式 |

### 阶段3: 文档内容生成

#### 需求详情
```markdown
## 需求详情

### 项目背景
为了兼容性和性能提升，需要将ATB中的{算子类型}算子替换为ACLNN算子。

### 需求目标
- 将ATB中的{算子类型}算子替换为对应的ACLNN算子
- 保持原有接口的兼容性，确保上层应用无需修改代码即可迁移
- 提升算子执行性能，充分利用昇腾硬件加速能力
- 确保替换后的算子功能正确性和稳定性

### 适用范围
本设计适用于ATB仓中的算子：
- {算子类型}算子
```

#### 算子IR与参数配置
```markdown
## 算子IR与参数配置

### ATB算子参数结构
```cpp
struct {OpType}Param {
    // 从文档中提取的参数结构
};
```

### ACLNN算子接口
```cpp
// 从文档中提取的ACLNN接口
```

### 参数映射关系
| ATB参数 | ACLNN参数 | 映射关系 |
|---------|-----------|----------|
| param1 | param1 | 直接映射 |
| param2 | param2 | 计算映射 |
| param3 | - | 废弃参数 |
```
</details>

## 参考实现

### softmax_aclnn_runner 参考

softmax_aclnn_runner是一个已实现的ACLNN runner示例，可作为参考。详细实现请参考 `references/softmax_aclnn_runner_reference.md` 文件，其中包含完整的头文件和源文件实现。

### 实现模式

所有ACLNN runner实现应遵循以下模式：

1. **继承自AclnnRunner基类**
2. **实现核心方法**：
   - `BuildAclnnVariantPack`：构建ACLNN变体包，处理输入输出张量
   - `SetAclNNWorkspaceExecutor`：设置ACLNN工作空间和执行器
   - `LaunchAclnnKernel`：启动ACLNN内核执行计算
   - `LoadMethod`：加载ACLNN函数指针
3. **注册Runner类型**：使用 `REG_RUNNER_TYPE` 宏注册Runner

### 代码结构

```
src/ops/ops_infer/{op_type}/{op_type}_aclnn_runner.h/cpp
```

其中 `{op_type}` 为算子类型，如 `softmax`、`layernorm` 等。

## 最佳实践

1. **接口文档分析**：仔细阅读ATB和ACLNN接口文档，确保准确理解参数含义和约束条件
2. **参数映射**：建立清晰的参数映射关系，特别注意计算映射和特殊参数的处理
3. **测试用例设计**：设计全面的测试用例，覆盖正常、异常和性能场景
4. **风险评估**：充分考虑可能的风险并制定应对策略
5. **代码结构**：遵循现有ACLNN runner的实现模式，保持代码一致性
6. **working_files 集成**：设计文档必须保存至 `working_files` 目录，便于用户确认和后续追溯

## Human in the Loop: 用户确认检查点 ⭐

**⚠️ 强制检查点 - 不可跳过**

设计文档生成完成后，必须等待用户确认后方可进入下一阶段。

### 确认前准备
1. **保存文档**: 将生成的设计文档保存至 `working_files` 目录
   - **文件路径**: `{WORKING_DIR}/{op_type}_replacement_design.md`
   - **变量说明**:
     - `WORKING_DIR`: 由用户指定的 working_files 目录
     - `op_type`: 算子类型小写（如 `swigluquant`, `layernorm`）

2. **准备确认清单**: 供用户逐项确认

### 用户确认检查表

```markdown
## 设计文档确认 - {OpType}

### 待确认文档
- **设计文档路径**: `{WORKING_DIR}/{op_type}_replacement_design.md`
- **ATB 接口链接**: [用户提供的链接]
- **ACLNN 接口链接**: [用户提供的链接]

### 确认检查表
| 检查项 | 说明 | 用户确认 |
|--------|------|----------|
| 参数映射关系正确 | 第 2.3 节参数映射表准确反映了 ATB→ACLNN 的映射关系 | [ ] |
| 风险评估合理 | 第 6 章识别的风险项和应对策略完整 | [ ] |
| 实现方案可行 | 第 5 章实现方案符合 ATB ACLNN Runner 标准模式 | [ ] |
| 测试用例设计可行 | 第 4 章测试用例覆盖全面，可进入 CSV 设计阶段 | [ ] |
| 规格约束清晰 | 第 3 章规格约束明确，便于编写反例测试 | [ ] |

### 确认方式
- **通过**: 回复 "确认通过" 或逐项打勾确认
- **修改**: 指出具体问题，返回修改后重新确认

### 后续步骤
确认通过后，将进入 Phase 2: CSV 用例设计阶段
- 下一步调用: `atb-csv-testcase-generator`

**⚠️ 未收到用户确认前，禁止进入 CSV 用例设计阶段**
```

### 向用户呈现方式

```markdown
我已生成 {OpType} 算子的替换设计文档，请确认以下内容：

📄 **文档位置**: `{WORKING_DIR}/{op_type}_replacement_design.md`

请检查：
1. 第 2.3 节"参数映射关系"是否正确？
2. 第 6 章"风险评估"是否遗漏重要风险？
3. 第 5 章"实现方案"是否符合预期？
4. 第 4 章"开发自测"是否覆盖全面？

回复 "确认通过" 进入 CSV 用例设计阶段，或指出需要修改的地方。
```

## 工具集成

该skill可以与以下工具配合使用：
- `WebFetch`：自动获取和解析接口文档内容
- `Read`：读取现有代码文件以了解实现模式
- `Write`：生成设计文档并保存至 working_files
- `Edit`：根据需要调整文档内容

通过使用该skill，用户可以快速生成高质量的ATB到ACLNN算子替换设计文档，提高开发效率和文档质量。

## 执行结果

> ⚠️ **执行后填写**：技能执行完成后，参照下方格式填写实际执行结果。

### 检查点检查表

| 步骤 | 检查点描述 | 状态 |
|------|-----------|------|
| - | ATB 接口 URL 已提供 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| - | ACLNN 接口 URL 已提供 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 1 | ATB 算子参数已提取 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 1 | ACLNN 接口参数已提取 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 2 | 参数映射表已创建 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 2 | 至少一个计算映射已定义 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 3 | 设计文档 7 个章节完整 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 3 | 风险评估表存在 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 4 | AclnnRunner 子类模式已记录 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 4 | 实现代码结构已定义 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| HIL | 设计文档已保存至 working_files | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| HIL | **用户已确认设计文档** ⭐ | ⬜ 待确认 |

**VERDICT: ✅ SUCCESS / ⚠️ PARTIAL / ❌ FAILED / ⏭️ SKIPPED**

### 问题列表（若有）

| 等级 | 检查点 | 问题描述 | 建议 |
|------|--------|---------|------|
| 🔴 CRITICAL | - | - | - |
| 🟡 WARNING | - | - | - |

### 执行摘要

- **执行时间**：
- **算子类型**：
- **ATB 接口链接**：
- **ACLNN 接口链接**：
- **通过率**：
