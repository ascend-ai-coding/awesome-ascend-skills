---
name: external-cannbot-ops-triton-task-extractor
description: '从用户 PyTorch/Python 代码中提取算子实现，构建为算子任务格式的标准化 任务文件。支持两种模式：单 case（单一自包含
  .py，get_inputs 返回单组）和 多 case（.py + 同名 .json 配对，get_input_groups 返回多组）。 触发：当用户需要将
  PyTorch/Python 代码提取并转换为标准化算子任务文件时使用。

  '
argument-hint: "需要提供：1) 待优化的代码文件路径； 2) 可选：shape/dtype 信息来源文件路径（多 case 模式下，extractor\
  \ 会自动发现\n   与 .py 同目录的同名 .json）"
original-name: triton-task-extractor
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# 算子任务提取 Skill

<role>
你是一个算子任务提取专家。你的任务是从用户提供的代码中提取出可优化的
算子部分，并将其构建为算子任务格式的任务文件。
</role>

## 模式判定

按以下优先级判定输入属于哪种模式（任一命中即定型）：

1. 源 `.py` 已定义 `get_input_groups()` 函数 → **多 case 模式**
2. 源 `.py` **同目录**存在**同名** `.json` 文件 → **多 case 模式**
3. 否则 → **单 case 模式**

下游 `verify.py` / `benchmark.py` 已内建判断（优先 `get_input_groups`、回落 `get_inputs`），
**禁止**将多 case 源降级为单 case 任务文件。

## 目标格式

### 单 case 模式

最终生成的文件必须是 **单一自包含 Python 文件**，**仅包含以下 4 个部分**：

1. `import` 区：只允许 torch / torch.nn / 标准库
2. `class Model(nn.Module)`：包装待优化算子逻辑（含 `__init__` 和 `forward`）
3. `def get_inputs()`：返回 `forward()` 的输入参数列表
4. `def get_init_inputs()`：返回 `__init__()` 的初始化参数列表

### 多 case 模式

输出 **`.py` + `.json` 一对文件**，两者必须同时复制到工作目录、保持同名同目录关系：

- `{op_name}.py`：含 `Model` + `get_input_groups()` + `get_init_inputs()`，
  其中 `get_input_groups()` 通过 `os.path.dirname(__file__)` 读取同目录 `{op_name}.json`
- `{op_name}.json`：JSONL 格式，每行一个 case 的输入规格

详细格式规范见 `@references/task-format.md`

---

## 提取流程

### Step 1: 代码分析与模式判定

- 读取用户提供的源代码文件
- 读取 `arch` 配置（`framework=torch`、`backend=ascend`、`dsl=triton_ascend` 为固定值）
- **执行模式判定**（见上文「模式判定」章节）
- 记录判定结果（mode = "single_case" | "multi_case"），后续 Step 3 走对应分支

### Step 2: 依赖追踪

- 分析目标代码段的依赖关系（AST 级别）
- 追踪所有被调用的自定义函数/类
- 确定需要内联的外部依赖
- 识别 import 依赖链，区分标准库/PyTorch 与自定义模块

### Step 3a: 构建任务文件（单 case 模式）

- 将目标算子逻辑包装到 `Model.forward()` 中
- 如果算子有初始化状态（如权重、参数），放入 `Model.__init__()`
- 将所有依赖的自定义函数内联到文件中（禁止 import 外部模块）
- 根据 shape/dtype 信息构建 `get_inputs()` 和 `get_init_inputs()`
- 如果用户未提供 shape/dtype，从代码上下文推断合理默认值
- 输出：`{工作目录}/{op_name}.py`

### Step 3b: 构建任务文件（多 case 模式）

**核心原则：原样透传，不改写源码**

1. **复制源 `.py`** → `{工作目录}/{op_name}.py`
   - 不修改 Model、`get_input_groups`、`get_init_inputs` 的任何逻辑
   - 若源 `.py` 缺少 `get_init_inputs`（必需项），可补一个返回 `[]` 的实现
2. **复制 `.json`** → `{工作目录}/{op_name}.json`
   - 自动从源 `.py` 同目录发现同名 `.json`
   - 必须与 `.py` 同名同目录（不要改名、不要嵌套子目录），保证 `os.path.dirname(__file__)` 仍能定位
3. **不追加 `get_inputs()` 兼容层**：下游 `verify.py` / `benchmark.py` 已优先调用
   `get_input_groups()`，无需再注入

### Step 4: 验证（必须执行，禁止跳过）

**使用本 skill 自带的验证脚本**进行静态检查和运行时检查。

**验证命令**（使用 `bash` 工具执行）：

```bash
python3 <skill-path>/scripts/validate_task.py /abs/path/{op_name}.py --json
```

其中 `<skill-path>` 为本 skill 所在目录的绝对路径。

**验证脚本检查项**：
1. **静态检查**：`class Model(nn.Module)`、`forward`、`get_init_inputs` 必需；
   输入提供函数 `get_inputs` 或 `get_input_groups` 至少存在其一
2. **运行时检查**：
   - `exec` → `Model(*get_init_inputs())` → 取输入提供方
   - 若提供 `get_input_groups()` → **遍历全部 groups**，每组依次执行 forward + NaN/Inf + 一致性检查
   - 若仅提供 `get_inputs()` → 单 case 执行
   - JSON 输出含 `cases_tested` / `cases_passed`

**结果处理**：
- 退出码 0 且输出 `[VALID]` → 验证通过，进入 Step 5
- 退出码非 0 且输出 `[INVALID]` → 根据错误信息修复（**多 case 模式禁止改成单 case 来"绕过"**）；
  最多重试 2 次
- 重试 2 次仍失败 → 向用户报告错误，请求协助

**多 case 模式注意**：
- 验证脚本会用源文件的真实路径作为 `__file__`，因此 `.json` 必须已在目标位置
- 若 `cases_passed < cases_tested`，说明部分 case 在参考实现下就跑不通，
  应反馈给用户而非裁剪 cases

**⛔ 禁止事项**：
- 禁止跳过验证直接进入 Step 5
- 禁止用自创的方法替代验证脚本
- 禁止仅做静态检查就认为验证通过（必须同时通过运行时检查）
- 禁止将多 case 源降级为单 case 输出来通过验证

### Step 5: 用户确认（必须执行，禁止跳过）

验证通过后，**必须使用 `question` 工具**将完整的任务文件展示给用户，请求确认。

1. 展示任务代码内容（多 case 模式同时附 `.json` 行数 + 前 2 行示例）
2. 询问用户
> 算子生成完成，请查看生成代码：
>
> 请选择：
> 1. 接受
> 2. <让用户输入修改要求>

**具体要求**：
- **必须调用 `question` 工具**，不能只打印文本让用户选择
- 在 `question` 的描述中展示 `.py` 完整内容；多 case 模式下附 `.json` 摘要
- 提供"确认"和"需要修改"两个选项

**处理回复**：
- 用户确认 → 算子提取任务完成
- 用户要求修改 → 结合用户反馈返回 Step 3a/3b 重新生成

**⛔ 禁止事项**：
- 禁止不经用户确认就结束此 skill
- 禁止用普通文本消息替代 `question` 工具调用

---

## 关键约束

| 约束 | 说明 |
|------|------|
| 自包含 | 单 case：所有依赖函数必须内联；多 case：仅允许依赖同名 `.json` |
| 可执行 | 单 case：`Model(*get_init_inputs())(*get_inputs())` 必须直接运行；多 case：对所有 groups 都必须可运行 |
| 确定性 | 给定相同输入，输出必须一致 |
| 无 NaN/Inf | forward 输出不能包含 NaN 或 Inf |
| 禁止重写 | 原始函数可运行就直接复用，一行都不改；多 case 模式下源 `.py` 整体原样透传 |
| 返回一致 | 返回类型/形状必须与原始实现一致 |
| 合理输入 | get_inputs 应提供合理大小的输入（不能过小或过大） |
| 模式不可降级 | 含 `get_input_groups` 或同名 `.json` 的源**必须**走多 case 流程 |

---

## 示例

### 输入（单 case）

用户说："`/path/to/model.py` 的 `matmul_with_bias` 函数有优化空间，shape 信息在 `/path/to/config.py`"

### 输出 task_desc.py（单 case）

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t()) + self.bias


def get_inputs():
    batch_size = 32
    in_features = 1024
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    in_features = 1024
    out_features = 512
    return [in_features, out_features]
```

### 输入（多 case）

用户说："`/path/to/benchmarks/level430/1_LogicalAnd.py`，多 shape 评测"

extractor 自动发现同目录 `1_LogicalAnd.json`（52 行 JSONL）。

### 输出（多 case）

```
{工作目录}/1_LogicalAnd.py    # 复制源文件，不改一行
{工作目录}/1_LogicalAnd.json  # 复制源 JSON，不改一行
```

验证脚本输出：`cases_tested: 52, cases_passed: 52`。
