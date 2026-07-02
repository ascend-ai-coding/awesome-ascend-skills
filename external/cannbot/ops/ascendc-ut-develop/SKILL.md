---
name: external-cannbot-ops-ascendc-ut-develop
description: Ascend C 算子 UT 开发与覆盖率增强技能。通过分析 op_host / op_api / op_kernel 的测试空白、生成或补充
  UT 用例并定位未覆盖代码来提升覆盖率并支持生成覆盖率报告。当用户提及 UT、单元测试、覆盖率、补测、未覆盖代码或需要新增/完善 UT 时使用，不适用于 ST
  测试。
original-name: ascendc-ut-develop
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# AscendC 算子 UT 开发

## 入口参数

| 参数名 | 含义 | 取值约束 | 初值推断 |
|-------|------|---------|---------|
| op_name | 算子名 | 采用下划线命名法 | 驼峰命名自动转换，无法推断时询问 |
| repo_type | 仓库类型 | 枚举值["ops-math", "ops-nn", "ops-transformer", "ops-cv", "custom"] | 根据工作目录推断，无法推断时询问 |
| soc_type | 芯片架构 | 枚举值列表["ascend310p", "ascend910b", "ascend910_93", "ascend950"] | 用户没有提及则默认全选 |
| test_model | 测试模块 | 枚举值列表["opapi", "ophost", "opkernel"] | 用户没有提及则默认全选 |
| interactive_mode | 使用模式 | 枚举值["auto", "interactive"]，auto 为自动模式不向用户问询，interactive 为交互模式允许向用户问询 | 默认 "auto" |

**格式约定**：整个 skill 里的所有文档中，入口参数使用 `${参数名}` 标记，需替换为真实值后再理解。

**调用方式**：必须以 `repo_type=custom` 模式调用本技能（跳过问卷，走 0.1.2 自定义工程快速通道）。

---

## ⚠️ 强制前置步骤（**不可跳过**）

> - **这是强制步骤，必须在执行主流程前完成！**
> - **在此步骤完成前，禁止阅读任何 references/ 目录下的文档和当前工作目录下的文件！**
> - **子步骤必须逐一执行，不允许同时执行或跳步！**

### 0.1 发送问卷确认入口参数

根据"初值推断"规则，尝试获取所有入口参数的值（仅根据已有上下文简单推断，**不阅读具体代码和子目录**）。

#### 0.1.0 仓库类型自动检测

**在执行问卷前，必须先确定项目路径，然后检测仓名**：

1. **优先从用户消息中提取项目路径**（如用户明确指定了路径）
2. **用户未指定时，使用 `pwd` 获取当前工作目录**
3. 基于最终确定的项目路径进行仓名匹配

检查路径的**目录组件**（以 `/` 分隔的路径段）是否**完全等于**以下标准仓名之一：`ops-math`、`ops-nn`、`ops-transformer`、`ops-cv`

> **⚠️ 必须是精确的目录名匹配，禁止子串匹配。** 例如路径 `/home/user/ops-math-utils` 中的 `ops-math-utils` 不等于 `ops-math`，应判定为 custom。

**判断逻辑：**

| 检测结果 | repo_type | interactive_mode | 后续行为 |
|---------|-----------|-----------------|---------|
| 路径中存在完全等于标准仓名的目录段 | 匹配到的仓名 | 按默认值 "auto" | 进入 **0.1.1 标准确认流程** |
| 路径中不存在完全等于标准仓名的目录段 | `"custom"` | `"auto"` | 进入 **0.1.2 自定义工程快速通道** |

#### 0.1.1 标准确认流程（路径匹配到标准仓名）

**立即**使用 `question` 工具，向用户确认入口参数是否正确，问卷内容使用 `assets/question.json`，在推断出的选项的 `label` 后加上 "【推荐】"，**除此之外不得有任何修改**

收到用户答复后，根据变量的"含义"理解用户真实需求。

#### 0.1.2 自定义工程快速通道（路径未匹配到标准仓名）

**跳过所有问卷**，从用户消息和上下文中推断所有入口参数，**禁止使用 `question` 工具**：

| 参数 | 取值 |
|------|------|
| repo_type | `"custom"` |
| interactive_mode | `"auto"` |
| soc_type | 从用户消息或上下文推断。**无法推断时**从算子的 `_def.cpp` 文件中提取：在 `${op_path}/op_host/` 下查找 `*_def.cpp`，解析所有 `AddConfig("xxx", ...)` 调用中的芯片名（如 `"ascend910b"`、`"ascend950"`），作为 soc_type 取值；若有多个则全部使用 |
| test_model | `["opapi", "ophost"]`（custom 工程不支持 opkernel） |
| op_name | 从用户消息或上下文推断（如用户提及的算子名、目录名等） |

设定完成后，**直接进入 Step 0.2**。

### 0.2 创建 TODO.md

**立即**使用 `todowrite` 工具创建 Todos，内容使用 `assets/todo.json`，**不允许进行任何修改。**

### 0.3 创建 tmp 目录

创建 `/tmp/cannbot_${op_name}/` 目录，用于存放所有中间文件：

```bash
mkdir -p /tmp/cannbot_${op_name}
```

**重要规则**：
- 将入口参数存储在该目录下的 `params.json` 中
- 所有中间文件都应放在该目录下
- 子 Agent 的临时文件也要存放在该目录下

---

## 主流程

### 流程执行规则

用户所有可能的需求都已经被抽象到主流程中，主流程的步骤在不同的需求下会重写为不同实现，你必须**严格遵守**以下规则：

1. **严格顺序执行**：必须按照 Step 1 → Step 2 → Step 3 → ... 的顺序执行，不得跳过或乱序
2. **禁止提前阅读**：**在执行到某个步骤前，绝对禁止阅读该步骤的文档！**
3. **即时更新进度**：每完成或跳过一个子步骤，**立即**使用 `todowrite` 工具更新 TODO.md，将其标记为 `[x]`
4. **交互模式强制约束**：当 `${interactive_mode} == "interactive"` 时，**必须**在每个询问点使用 `question` 工具与用户确认，**禁止**绕过用户自行决策。Step 4.4a 的询问步骤是交互模式的核心特征，跳过即违背用户意图。

### 主流程链接

[Step 1](./references/workflow/step1.md)
[Step 2](./references/workflow/step2.md)
[Step 3](./references/workflow/step3.md)
[Step 4](./references/workflow/step4.md)
[Step 5](./references/workflow/step5.md)