---
name: external-gitcode-ascend-ascendc-operator-doc-gen
description: 为AscendC算子生成PyTorch风格的接口文档（README.md）。触发场景：编译调试通过后需要生成接口文档，或用户提到"生成算子文档"、"创建README"、"文档化算子"、"帮我写文档"（算子上下文）、"算子文档"时使用。
original-name: ascendc-operator-doc-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# AscendC 算子接口文档生成

从算子源代码提取接口信息，生成 **PyTorch 官方文档风格** 的中文 API 接口文档。

**前置条件**：编译测试通过（Phase 3 完成），以下文件均已就绪：
- `csrc/register.cpp` — 包含 `m.def` 注册的 Python 调用 schema
- `csrc/ops.h` — 包含 C++ 函数声明
- `csrc/ops/<op_name>/design.md` — 包含算法描述、参数说明、dtype 支持、约束条件
- `csrc/ops/<op_name>/op_host/<op_name>.cpp` — 包含 TORCH_CHECK 约束和实际参数处理逻辑
- `tests/test_<op_name>.py` — 包含可运行的调用示例

## 工作流程

```
信息提取 → 文档结构组装 → 文件生成 → 聊天界面展示
```

---

## 阶段 1: 信息提取

从以下源文件中提取接口文档所需的全部信息。**MUST** 逐一读取这些文件，不可跳过。

### 1.1 从 `csrc/register.cpp` 提取 Python 调用签名

找到 `m.def("<op_name>(...)` 行，提取完整的 schema 字符串。

**提取内容**：
- 函数名
- 参数列表（含类型和默认值）
- 返回类型

**示例**：
```
m.def("acosh(Tensor self) -> Tensor");
```
→ 签名: `torch.ops.npu.acosh(self) → Tensor`

**Schema 类型到 Python 类型映射**：

| Schema 类型 | Python 文档类型 | 示例 |
|-------------|----------------|------|
| `Tensor` | *Tensor* | 必选张量参数 |
| `Tensor?` | *Tensor, optional* | 可选张量参数 |
| `Tensor(a!)` | *Tensor* | 原地修改张量 |
| `int` | *int* | 整数参数 |
| `int?` | *int, optional* | 可选整数参数 |
| `int[]` | *list[int]* | 整数列表 |
| `int[3]` | *list[int]* | 固定长度整数列表 |
| `float` | *float* | 浮点参数 |
| `bool` | *bool* | 布尔参数 |
| `str?` | *str, optional* | 可选字符串参数 |

### 1.2 从 `csrc/ops.h` 提取 C++ 函数声明

找到 `namespace ascend_kernel` 中对应函数的完整 C++ 签名。

**提取内容**：
- 返回类型（`at::Tensor`、`void` 等）
- 参数类型和参数名
- 参数是否为 const 引用 / optional

### 1.3 从 `design.md` 提取算法和设计信息

**提取内容**：

| 提取项 | 设计文档章节 | 文档用途 |
|--------|------------|---------|
| 算法描述 / 数学公式 | 计算逻辑设计 | 功能描述段落 |
| 参数语义说明 | 算子接口定义 | 参数说明段落 |
| 支持的数据类型 | 算子接口定义 | 支持的数据类型段落 |
| 输入输出 shape 约束 | Tiling策略 / 算子接口定义 | Shape 段落 |
| 有效输入范围 / 约束 | 注意事项 / 接口定义 | 约束条件段落 |

### 1.4 从 `op_host/<op_name>.cpp` 提取运行时约束

搜索所有 `TORCH_CHECK(...)` 语句，提取：

| TORCH_CHECK 内容 | 文档用途 |
|-----------------|---------|
| 维度检查（`dim() == N`） | Shape 约束 |
| dtype 检查（`scalar_type() == kHalf`） | 支持的数据类型 |
| 值域检查（数值范围限制） | 约束条件 |
| 参数互斥/依赖关系 | 约束条件 / 参数说明 |

### 1.5 从 `tests/test_<op_name>.py` 提取使用示例

找到最简洁且完整的调用示例（优先选 `run_simple_test` 或 `test_basic` 中的代码），提取：

- 输入 tensor 构造方式
- 算子调用语句（`torch.ops.npu.<op_name>(...)`）
- 输出处理方式

---

## 阶段 2: 文档结构组装

按以下 **固定结构** 组装文档内容。格式严格参考 PyTorch 官方文档（如 `torch.nn.RMSNorm`、`torch.abs`），**文档正文使用中文**。

### 文档模板

```markdown
# torch.ops.npu.<op_name>

```
torch.ops.npu.<op_name>(<param1>, <param2>, ..., <paramN>) → <ReturnType>
```

<功能描述：1-3句中文说明算子做什么。如果有数学公式，用 LaTeX 行内公式展示。>

<如有数学公式，用独立公式块展示：>

$$
<公式>
$$

## 参数说明

- **<param1>** (*<type>*) – <中文描述>
- **<param2>** (*<type>*) – <中文描述>
- **<param3>** (*<type>, optional*) – <中文描述>。默认值: `<默认值>`

## 支持的数据类型

`torch.float16`, `torch.bfloat16`, `torch.float32`

## Shape

- **输入**: <shape描述，使用数学符号如 (N, *), (S, N, D) 等>
- **输出**: <shape描述>

<如有额外 shape 规则，以列表形式补充>

## 约束条件

- <约束条件1>
- <约束条件2>

## 使用示例

```python
>>> import torch
>>> import torch_npu
>>> import ascend_kernel
>>> <构造输入>
>>> <调用算子>
>>> <展示输出>
```

## 返回值

*<返回类型>* – <中文返回值描述>
```

### 各段落详细规范

#### 标题签名

- 格式：`torch.ops.npu.<op_name>(<参数列表>) → <返回类型>`
- 参数列表从 `register.cpp` 的 schema 提取，仅保留参数名（不带类型）
- 有默认值的参数写为 `<name>=<default>`
- 返回类型：`Tensor`、`tuple[Tensor, ...]`、`None`（对应 C++ `void`）

#### 功能描述

- 使用中文撰写
- 第一句话概括算子功能
- 如有数学公式，用 LaTeX 展示
- 如有多种工作模式（由参数控制），分别说明

#### 参数说明

- 每个参数一行，格式：`- **<name>** (*<type>*) – <中文描述>`
- 可选参数标注 `optional`：`- **<name>** (*<type>, optional*) – <中文描述>。默认值: \`<value>\``
- 参数描述应包含：语义说明、shape 要求（如适用）、有效值范围（如适用）
- Tensor 参数说明 shape 格式，如 "形状: `(S, N, D)`"
- 布尔/枚举参数说明各取值的含义

#### 支持的数据类型

- 列出算子支持的所有 PyTorch dtype
- 格式：`` `torch.float16`, `torch.bfloat16`, `torch.float32` ``
- 从 op_host 的 TORCH_CHECK 和 design.md 交叉验证

#### Shape

- 使用中文标签（**输入**、**输出**）
- 描述输入输出 tensor 的 shape 语义
- 使用大写字母表示各维度含义，如 `(N, C, H, W)` — N: 批大小, C: 通道数, ...
- 如果 shape 随参数变化，分情况说明

#### 约束条件

- 使用中文撰写
- 列出所有 TORCH_CHECK 中检查的约束条件
- 列出 design.md 中提到的有效输入范围
- 如果存在参数之间的互斥/依赖关系，明确说明

#### 使用示例

- 提供可直接在 NPU 上运行的完整代码片段
- 包含 `import` 语句
- 使用 `>>>` 前缀（doctest 风格）
- 输入数据使用小尺寸（便于展示）
- 展示至少 1 个典型用例
- 如果有多种使用模式，各展示 1 个

#### 返回值

- 使用中文描述
- 格式：`*<type>* – <中文描述>`
- 说明返回 tensor 的 shape 和 dtype（如与输入一致则注明"与输入 dtype 一致"）
- 如返回多个值（tuple），逐一说明

---

## 阶段 3: 文件生成

将组装好的文档写入：

```
ascend-kernel/csrc/ops/<op_name>/README.md
```

**文件写入规则**：
- 如果 README.md 已存在，**覆盖**旧内容
- 使用 UTF-8 编码
- 数学公式使用标准 LaTeX 语法（`$...$` 行内，`$$...$$` 块级）

---

## 阶段 4: 在交互界面展示文档（MANDATORY）

文件生成后，**MUST** 将完整的 README.md 内容直接输出到聊天界面中。

**展示格式**：

```
### 接口文档已生成

文件路径: `ascend-kernel/csrc/ops/<op_name>/README.md`

<完整 README.md 内容>
```

**要求**：
1. 展示完整文档内容，不要截断
2. 展示文件路径供用户查看
3. 如果某个段落因信息不足无法填写，用 `[TODO: ...]` 标注并提醒用户补充

---

## 完整示例

以下是一个假设的 `acosh` 算子的接口文档示例，展示最终生成效果：

> **说明**：此示例仅用于展示文档格式，实际生成时所有信息均从源代码提取。

````markdown
# torch.ops.npu.acosh

```
torch.ops.npu.acosh(self) → Tensor
```

逐元素计算输入张量的反双曲余弦值。

$$
\text{out}_i = \cosh^{-1}(\text{input}_i) = \ln(\text{input}_i + \sqrt{\text{input}_i^2 - 1})
$$

## 参数说明

- **self** (*Tensor*) – 输入张量，元素值必须 $\geq 1$。支持任意形状。

## 支持的数据类型

`torch.float16`, `torch.float32`

## Shape

- **输入**: $(*)$，支持任意形状
- **输出**: $(*)$，与输入形状相同

## 约束条件

- 仅支持 `float16` 和 `float32` 数据类型
- 输入张量的所有元素必须 $\geq 1$，否则结果为 `NaN`

## 使用示例

```python
>>> import torch
>>> import torch_npu
>>> import ascend_kernel
>>> x = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float32, device="npu:0")
>>> output = torch.ops.npu.acosh(x)
>>> output
tensor([0.0000, 1.3170, 1.7627, 2.9932], device='npu:0')
```

## 返回值

*Tensor* – 反双曲余弦计算结果，形状与输入相同，dtype 与输入一致。
````

---

## 检查清单

文档生成后按以下清单逐项验证：

- [ ] **签名一致性**: `torch.ops.npu.<op_name>(...)` 的参数列表与 `register.cpp` 的 `m.def` 完全一致
- [ ] **参数完整性**: 每个参数都有类型标注和中文语义描述
- [ ] **默认值正确**: 有默认值的参数在签名和参数说明中都标注了默认值
- [ ] **dtype 准确**: 支持的数据类型与 op_host TORCH_CHECK 和 design.md 一致
- [ ] **Shape 清晰**: 输入输出 shape 描述使用了有语义的维度符号
- [ ] **约束完整**: 所有 TORCH_CHECK 的检查条件都已体现
- [ ] **示例可运行**: 使用示例中的代码是从 test 文件中提炼的可执行代码
- [ ] **返回值明确**: 返回类型、shape、dtype 都已说明
- [ ] **中文表述**: 功能描述、参数说明、约束条件、返回值均使用中文
- [ ] **文件已生成**: README.md 已写入 `csrc/ops/<op_name>/README.md`
- [ ] **已在聊天界面展示完整文档内容**

## 反模式清单

- **NEVER** 编造参数或 dtype 信息，所有信息必须从源代码提取
- **NEVER** 跳过 TORCH_CHECK 约束的提取
- **NEVER** 使用与 register.cpp schema 不一致的参数名或类型
- **NEVER** 省略使用示例段落
- **NEVER** 仅输出文件路径而不在聊天界面展示完整文档内容
- **NEVER** 修改算子源代码，本 skill 是只读文档生成
- **NEVER** 使用英文撰写文档正文（标题签名、代码、数学公式除外）

## 可读取文件范围

| 文件 | 读取内容 |
|------|---------|
| `csrc/register.cpp` | Python 调用 schema（`m.def`） |
| `csrc/ops.h` | C++ 函数声明 |
| `csrc/ops/<op_name>/design.md` | 算法描述、参数说明、dtype、约束 |
| `csrc/ops/<op_name>/op_host/<op_name>.cpp` | TORCH_CHECK 约束、参数处理逻辑 |
| `tests/test_<op_name>.py` | 使用示例 |
