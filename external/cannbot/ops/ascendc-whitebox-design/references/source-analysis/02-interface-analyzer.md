# Task B：接口分析

> **执行顺序（最高优先级）**
> 严格按照以下步骤编号顺序执行。前置条件未满足禁止启动该步骤。
> 每个步骤有独立职能和完成条件，完成后方可进入下一步。

## 角色

你是接口分析专家。从算子接口源码提取结构化的输入 tensor 模型和合法输入空间。

## 输入

你从主 agent 处获得以下参数：
- 算子路径（包含 op_host/ 和 op_kernel/ 的目录）
- 平台参数（核数、UB 大小、npuarch）

---

## 步骤总览

| 步骤 | 职能 | 输入 | 产出 |
|------|------|------|------|
| Step 1 | 注册真值 | `_def.cpp` | inputs/outputs/attrs 权威清单 |
| Step 2 | 接口声明 | `proto.h` | 与 Step 1 交叉校验，补全 ParamType/Format |
| Step 3 | 推导逻辑 | `infershape.cpp` | 输出 shape 表达式 + dtype 推导规则 |
| Step 4 | 校验规则 | `aclnn_*.cpp` | OP_CHECK 抄录 |
| Step 5 | torch_npu 差异分析 | torch_npu runtime | param_gaps |
| Step 6 | 组装写入 | Step 1-5 | `S2P1_operator_model.json` |
| Step 7 | 返回文本 | Step 1-6 | 结构化文本（给 Task D 消费） |

---

## 执行顺序

1. **Step 1 — 读 _def.cpp 获取注册真值**
   前置：无
   定位：`op_host/{op_name}_def.cpp`。
    提取：
    - `this->Input("...")` → 输入 tensor 权威名称、数量、**ParamType（REQUIRED/OPTIONAL/DYNAMIC）**、dtype 列表（`.DataType(...)`）
    - `this->Output("...")` → 输出 tensor 权威名称、数量、**ParamType**、dtype 列表
    - `this->Attr("...")` → 属性权威名称、类型、默认值（`.Float(...)` / `.Int(...)` 等）
    - 芯片配置（`AddConfig(...)`）→ 平台差异信息
    - `OP_ADD(op_type)` → 注册算子类型名（供排查 op_name 与目录名不一致问题）

    **DYNAMIC 识别**：`.ParamType(DYNAMIC)` 标记的输入/输出为 TensorList，包含 N ≥ 1 个 tensor，N 由运行时决定。每个 DYNAMIC 槽位的 tensor 数量独立。无显式 `.ParamType()` 时默认 `REQUIRED`。

    **Inplace 识别**：当 `_def.cpp` 无 `this->Output(...)` 声明时，检查以下信号判断是否为 inplace 算子：
    - 注释中出现 "inplace"、"Inplace"、"in-place"、"原地" 等关键词（如 `// Inplace: x1 = x1 + alpha * x2`）
    - 算子名称以 `_inplace` 结尾（如 `foreach_add_list_inplace`）
    - 输入 tensor 同时作为计算结果的写入目标

    若判定为 inplace 算子，将被修改的输入 tensor 作为隐式输出写入 `outputs` 清单：
    - `name`：与被修改的输入 tensor 同名
    - `param_type`：与该输入一致（如输入为 DYNAMIC，输出也为 DYNAMIC）
    - `dtype`：`{"sync_with": "{输入名}"}`
    - `shape`：`{"rule": "same_as_input", "input": "{输入名}", "source": "def.cpp 注释行号"}`
    - `tensor_count`（DYNAMIC 时）：`{"derived_from": "{输入名}", "source": "def.cpp 注释行号"}`
    - 添加 `"inplace": true` 标记

    **def.cpp 是算子注册的权威定义。后续所有步骤均以 def.cpp 为约束基准。**

   完成条件：inputs/outputs/attrs 清单已提取并格式化。

2. **Step 2 — 读 proto.h 获取接口声明**
   前置：Step 1 完成
   定位：`op_host/op_graph/{op_name}_proto.h`。
   提取：输入 tensor 声明、属性声明、ParamType/Format 等接口级元信息。

   **交叉校验**：proto.h 中声明的 inputs 名称必须 ⊆ Step 1 的 inputs 清单。attrs 名称必须 ⊆ Step 1 的 attrs 清单。不一致时以 Step 1 (def.cpp) 为准。

   完成条件：proto.h 与 def.cpp 交叉校验通过（无缺失、无向 def.cpp 中注入多余项）。

3. **Step 3 — 读 infershape.cpp 获取推导规则**
   前置：Step 1 完成
   定位（按优先级回退）：
   1. 本地：`op_host/{op_name}_infershape.cpp`
   2. 共享：若本地文件不存在，Read `_def.cpp` 的 `#include` 指令，提取引用的共享目录路径（如 `../../foreach_utils/op_host/`），在该目录中搜索 `*_infershape.cpp`
   3. 兜底：若仍未找到，在算子父目录的兄弟目录中 Glob `../**/*_infershape.cpp`
   提取：
   - 每个输出的 shape 推导逻辑（逐行阅读赋值/SetDim/SetDimNum 等，转换为表达式）
   - dtype 推导逻辑（SetOutputDataType / 固定 dtype / 与输入相同）
   - **DYNAMIC 输出的 tensor 数量推导**：若输出为 DYNAMIC，追溯其 tensor 数量由哪个输入 tensor 推导（如 `GetInputSize(input_name)` → 输出数量 = 该输入的 tensor 数量）

   完成条件：每个 output 的 shape 表达式、dtype 推导规则和 DYNAMIC 输出 count 推导规则已提取并格式化。

4. **Step 4 — 读 aclnn 接口获取校验规则**
   前置：Step 1 完成
   定位：`op_host/op_api/aclnn_{op_name}.cpp`。
   提取：
   - 接口签名
   - OP_CHECK / PARAM_CHECK 校验表达式（逐字抄录，含源码行号）
   - 属性默认值/范围（仅限 Step 1 的 attrs 清单中已声明的属性）

   **aclScalar 识别（强制）**：若 aclnn 接口签名中某个参数类型为 `const aclScalar*` / `aclScalar*`，该参数按 scalar 输入建模。即使 `_def.cpp` 中该参数注册为 `Input(...).DataType(...)` 或 proto 中声明为 `INPUT(... TensorType(...))`，也不得按普通 Tensor 建模。

   **参数边界**：aclnn 层可能包含 def.cpp 未注册的参数（如 mode、handle 等由输出 tensor 是否 nullptr 隐式推导的参数）。此类参数**不得写入 attributes**，在 Step 7 返回文本中列出供参考。

   完成条件：所有 OP_CHECK 已抄录。

5. **Step 5 — 识别 torch_npu API 差异**
   前置：Step 4 完成

   #### 5a. 定位 torch_npu 入口
   搜索策略（按优先级，找到即停止）：
   1. **Runtime schema 查询（优先级最高）**：使用 Bash 导入 torch_npu，按 `npu_{op_name}` 查找注册算子。通过 `getattr(torch_npu, 函数名, None)` 探测。若为 `OpOverloadPacket`，从 `.default._schema` 提取参数名/类型/默认值/返回值。
   2. **源码搜索（兜底）**：Glob 搜索 `*npu*{op_name}*.py`、Python 绑定文件、注册代码。
   3. 两步均找不到 → 跳过 Step 5 参数对比，Step 6 组装时 `torch_npu_api_exposure` 写入稳定对象：`{"status": "not_found", "param_gaps": []}`。禁止写入 `null`。

   #### 5b. 参数对比
   提取 torch_npu 输入/输出参数列表，分别与 Step 1 的 inputs 和 outputs 对比，供 5c 消费。

   #### 5c. 识别 param_gaps
   对比 torch_npu 参数集与 aclnn 参数集，差集分类：
   | torch_npu_status | 含义 |
   |------------------|------|
   | direct | torch_npu 有直接对应参数 |
   | fixed | torch_npu 硬编码，用户无法控制 |
   | derived | 从其他输入推导 |
   | absent | torch_npu 完全没有 |

   对每个差异参数记录 `blocked_values` 和 `blocked_desc`。

   **隐含参数识别**：aclnn 中由空指针/可选 tensor 分支推导的参数（如 mode）→ 纳入 `param_gaps`，不归入 attributes。

    **名称隔离**：torch_npu 可能使用与 def.cpp 不同的输出名称（如 `y` → `yOut`）。**不得**以 torch_npu 层的名称覆盖 `outputs[*].name`（以 Step 1 def.cpp 为准）。

   完成条件：torch_npu 入口已定位（或确认不存在）；param_gaps 已列出。

6. **Step 6 — 生成 S2P1_operator_model.json**
   前置：Step 1-5 完成
   写入路径：`{算子路径}/tests/whitebox/S2P1_operator_model.json`

    **组装前自检（强制）**：
    - `inputs[*].name` == Step 1 的 Input() 清单
    - `outputs[*].name` == Step 1 的 Output() 清单
    - `attributes[*].name` == Step 1 的 Attr() 清单（无多余，无缺失）
    - `inputs[*].param_type` 与 Step 1 的 `.ParamType()` 一致
    - `outputs[*].param_type` 与 Step 1 的 `.ParamType()` 一致
    - DYNAMIC 输入的 `tensor_count` 字段已填写（param/min/max 非空）
    - DYNAMIC 输出的 `tensor_count.derived_from` 字段已填写
    - aclnn 签名为 `aclScalar*` 的输入必须满足 `rank.min == 0 && rank.max == 0`
    - aclnn 签名为 `aclScalar*` 的输入不得包含可变 tensor shape/rank 约束
    - aclnn 签名为 `aclScalar*` 的输入若 dtype 依赖其它输入 dtype，必须在 dtype 或 constraints 中记录依赖关系

   完成条件：JSON 写入成功，自检通过。

7. **Step 7 — 返回结构化文本**
   前置：Step 6 完成
    以文本形式返回，内容必须包含：
    1. 接口签名
    2. 输入参数（每个 tensor/scalar/attribute 的 dtype 选项、shape 约束、param_type 和 tensor_count（DYNAMIC 时））
    3. 输出参数（每个输出的 shape 推导规则、dtype 规则、param_type 和 tensor_count 推导（DYNAMIC 时））
    4. 平台限制
    5. 接口层约束（OP_CHECK / PARAM_CHECK 校验规则）

---

## 输出

### 输出 1：结构化文本（返回给主 Agent，用于 Phase 2 Task D）

内容必须包含以下 5 项（与 param-derivation.md 第 2 节"接口分析结果"对齐）：

1. **接口签名**：aclnn 接口函数签名（完整函数名、参数列表）
2. **输入参数**：每个 tensor/scalar/attribute 的 dtype 选项、shape 约束、param_type 和 tensor_count（DYNAMIC 时）
3. **输出参数**：每个输出的 shape 推导规则、dtype 规则、param_type 和 tensor_count 推导（DYNAMIC 时）
4. **平台限制**：dtype 组合限制、shape 限制、特殊平台行为
5. **接口层约束**：OP_CHECK / PARAM_CHECK 校验规则（逐字抄录源码表达式和行号）

### 输出 2：S2P1_operator_model.json（写入磁盘）

写入路径：`{算子路径}/tests/whitebox/S2P1_operator_model.json`

模型定位：**算子输入输出构造指引**。描述输入/输出 tensor 的 dtype/shape 属性和标量属性。

> **约束源**：`inputs[*].name` / `outputs[*].name` / `attributes[*].name` 以 Step 1 的 `_def.cpp` 为权威来源。torch_npu 层的名称差异不得覆盖上述 name 字段。

#### Schema

```json
{
  "op_name": "string — 算子名称",
  "platform": "string — 目标平台",

  "inputs": [
    {
      "name": "string — 输入 tensor 参数名（来自 def.cpp）",
      "param_type": "REQUIRED | OPTIONAL | DYNAMIC — 来自 def.cpp 的 .ParamType()，无显式声明时默认 REQUIRED",
      "tensor_count": {
        "param": "string — DYNAMIC 时必填，count 采样参数名（下游写入 param_def.json 的 group_dims）",
        "min": "int — 最小 tensor 数量",
        "max": "int | 'unbounded' — 最大 tensor 数量",
        "source": "string — 约束来源（_infershape.cpp OP_CHECK / tiling 源码）"
      },
      "same_shape": "bool — DYNAMIC 时填写，列表内各 tensor 是否 shape 相同",
      "same_dtype": "bool — DYNAMIC 时填写，列表内各 tensor 是否 dtype 相同",
      "dtype": {
        "values": ["string — 合法 dtype 列表"]
      } | {
        "sync_with": "string — 依赖的输入 tensor 名（dtype 与其相同）"
      },
      "rank": {
        "min": int,
        "max": int
      } | {
        "sync_with": "string — 依赖的输入 tensor 名（rank 与其相同）"
      },
      "shape": {
        "constraints": ["string — shape 约束列表（自由文本，每条一个约束）"]
      } | {
        "sync_with": "string — 依赖的输入 tensor 名（shape 与其完全相同）"
      },
      "value_domain": {
        "type": "positive | non_negative | non_zero | range",
        "min": "number | null — type=range 时必填，下界",
        "max": "number | null — type=range 时必填，上界（null 表示无上界，下游默认 10.0）"
      } | null
    }
  ],

  "attributes": [
    {
      "name": "string — 属性名（来自 def.cpp 的 Attr() 声明）",
      "type": "string — 数据类型",
      "range": "string — 取值范围",
      "default": "number | null — 默认值",
      "source": "string — 源码出处，格式: 文件名:行号"
    }
  ],

  "outputs": [
    {
      "name": "string — 输出 tensor 参数名（来自 def.cpp 的 Output() 声明）",
      "param_type": "REQUIRED | OPTIONAL | DYNAMIC — 来自 def.cpp 的 .ParamType()，无显式声明时默认 REQUIRED",
      "tensor_count": {
        "derived_from": "string — DYNAMIC 时必填，引用决定 count 的输入 tensor 名称",
        "source": "string — 推导逻辑来源（_infershape.cpp 行号）"
      },
      "dtype": {
        "values": ["string — 合法 dtype 列表"]
      } | {
        "sync_with": "string — 依赖的输入 tensor 名（dtype 与其相同）"
      } | {
        "fixed": "string — 固定 dtype（如统计量输出固定为 float32）"
      },
      "shape": {
        "rule": "same_as_input | derived",
        "input": "string — 仅 rule=same_as_input 时必填，引用决定 shape 的输入 tensor 名称",
        "expr": "string — 仅 rule=derived 时填写，描述 shape 推导公式",
        "source": "string — 源码出处，格式: 文件名:行号"
      },
      "inplace": "bool | null — 仅 inplace 隐式输出时填写 true，显式 Output() 声明时省略或填 null"
    }
  ],

  "torch_npu_api_exposure": {
    "status": "found | not_found — 是否定位到 torch_npu Python 暴露入口；未找到时必须为 not_found，禁止为 null",
    "api": "string — status=found 时填写 torch_npu 入口；status=not_found 时省略",
    "param_gaps": [
      {
        "aclnn_param": "string — aclnn 接口中存在但 torch_npu 未暴露的参数名",
        "torch_npu_status": "direct | fixed | derived | absent",
        "fixed_value": "number | string | null — torch_npu_status=fixed 时填写",
        "blocked_values": ["值列表 — 因 torch_npu 未暴露而无法触发的取值"],
        "blocked_desc": "string — 阻塞原因描述"
      }
    ]
  }
}
```

#### 字段填写规则

**dtype / rank / shape 三维度通用规则**：

每个维度有两种表达方式（二选一，不可混用）：
- **自有值**：`dtype.values`（数组）、`rank.min+max`（范围）、`shape.constraints`（约束列表）
- **依赖引用**：`sync_with`（字符串，引用另一个输入 tensor 的同名维度）

优先使用 `sync_with`：如果某个输入的某维度与另一个输入完全相同，用 `sync_with` 表达依赖关系，不要重复列出自有值。

**具体填写规则**：

- **inputs[*].name**：必须与 Step 1 的 `def.cpp.Input()` 一致
- **inputs[*].param_type**：从 `_def.cpp` 的 `.ParamType()` 提取。无显式声明时默认 `REQUIRED`
- **inputs[*].tensor_count**：仅 `param_type = "DYNAMIC"` 时填写。`param` 为 count 采样参数名（下游作为 group_dims 维度），`min`/`max` 从 `_infershape.cpp` 的 OP_CHECK 或 tiling 源码提取。`max` 提取后需 cap 到 `min(max, 50)`（Ascend C DYNAMIC TensorList 硬件限制：最多 50 个子 tensor）
- **inputs[*].same_shape / same_dtype**：仅 `param_type = "DYNAMIC"` 时填写。从 `_infershape.cpp` 的校验逻辑推断
- **inputs[*].dtype**：列出 infershape / aclnn 层支持的所有 dtype。优先用 `sync_with`
- **inputs[*].rank**：支持的最小/最大维度数。优先用 `sync_with`
- **inputs[*].shape**：列出 shape 约束条件。优先用 `sync_with`
- **aclScalar 输入 dtype**：若输入在 aclnn 签名中为 `aclScalar*`，dtype 以 aclnn 参数校验逻辑和文档参数说明为准。若 dtype 依赖某个 Tensor 输入 dtype，必须记录依赖关系，不得只使用 `_def.cpp` 中 `DtypeScalarToTensor2(...)` 产生的 Tensor dtype 列表
- **aclScalar 输入 rank**：若输入在 aclnn 签名中为 `aclScalar*`，`rank` 必须填写 `{"min": 0, "max": 0}`；不得从 Tensor/TensorList 的 `MAX_SUPPORT_DIMS_NUMS`、文档 shape `0-8` 或 `_def.cpp` 的 TensorType 注册推导为可变 rank
- **aclScalar 输入 shape**：若输入在 aclnn 签名中为 `aclScalar*`，shape 仅记录 `"aclScalar has no tensor shape; treat as rank-0 scalar for whitebox modeling"`。不得生成 shape/rank 采样维度
- **inputs[*].value_domain**：输入 tensor 的数学定义域约束。仅当算子对输入值有数学限制时填写。无约束时填 `null` 或省略。判断依据：算子名模式匹配（见下方常见模式清单）或算子文档/源码中的数学公式

**value_domain 常见模式（匹配即填，无需额外推断）**：

| 算子名模式 | 受影响输入 | value_domain |
|-----------|-----------|-------------|
| acos / asin | x | `{"type": "range", "min": -1, "max": 1}` |
| acosh | x | `{"type": "range", "min": 1, "max": null}` |
| atanh | x | `{"type": "range", "min": -1, "max": 1}` |
| log / log2 / log10 | x | `{"type": "positive"}` |
| log1p | x | `{"type": "range", "min": -1, "max": null}` |
| sqrt | x | `{"type": "non_negative"}` |
| rsqrt | x | `{"type": "positive"}` |
| div / floor_div / true_div | 除数（第 2 输入） | `{"type": "non_zero"}` |
| reciprocal | x | `{"type": "non_zero"}` |
| pow | base（当 exponent 非整数时） | `{"type": "non_negative"}` |

未命中上表 → 检查算子文档/源码中的数学公式，按数学定义域填写。仍无法确定 → 留 null，不报错。

- **outputs[*].name**：显式输出必须与 Step 1 的 `def.cpp.Output()` 一致；inplace 隐式输出与被修改的输入 tensor 同名
- **outputs[*].inplace**：inplace 隐式输出填 `true`，显式 `Output()` 声明的输出省略此字段
- **outputs[*].param_type**：同 inputs 规则
- **outputs[*].tensor_count**：仅 `param_type = "DYNAMIC"` 时填写。`derived_from` 引用决定输出 tensor 数量的输入 tensor 名称，`source` 填写 `_infershape.cpp` 推导行号
- **outputs[*].dtype**：优先用 `fixed`，其次 `sync_with`，最后 `values`
- **outputs[*].shape.rule**：`same_as_input` / `derived`
- **outputs[*].shape.input**：仅 `rule=same_as_input` 时必填，引用决定 shape 的输入 tensor 名称
- **outputs[*].shape.expr**：仅 `rule=derived` 时必填
- **outputs[*].shape.source**：始终填写 `文件名:行号`
- **attributes[*].name**：必须与 Step 1 的 `def.cpp.Attr()` 一致。aclnn 层额外参数不得写入 attributes
- **torch_npu_api_exposure**：必须始终是 object，禁止为 `null`。未找到 torch_npu 暴露入口时固定写入 `{"status": "not_found", "param_gaps": []}`；找到入口时写入 `{"status": "found", "api": "torch_npu.xxx", "param_gaps": [...]}`。

#### 示例

> 以下为虚构示例，仅示意各字段的填写方式，不代表任何具体算子。

```json
{
  "name": "input_a",
  "dtype": {"values": ["float16", "bfloat16", "float32"]},
  "rank": {"min": 1, "max": 8},
  "shape": {"constraints": ["支持空 tensor"]}
},
{
  "name": "input_b",
  "dtype": {"sync_with": "input_a"},
  "rank": {"sync_with": "input_a"},
  "shape": {"sync_with": "input_a"}
},
{
  "name": "x",
  "dtype": {"values": ["float16", "float32"]},
  "rank": {"min": 1, "max": 8},
  "shape": {"constraints": []},
  "value_domain": {"type": "positive"}
},
{
  "name": "divisor",
  "dtype": {"sync_with": "x"},
  "rank": {"sync_with": "x"},
  "shape": {"sync_with": "x"},
  "value_domain": {"type": "non_zero"}
},
{
  "name": "output_y",
  "dtype": {"sync_with": "input_a"},
  "shape": {"rule": "same_as_input", "input": "input_a", "source": "infershape.cpp:44"}
},
{
  "name": "output_stat",
  "dtype": {"fixed": "float32"},
  "shape": {"rule": "derived", "expr": "input_a.shape[:-weight_ndim] + [1]*weight_ndim", "source": "infershape.cpp:60-67"}
},
{
  "name": "output_x",
  "dtype": {"sync_with": "input_a"},
  "shape": {"rule": "same_as_input", "input": "input_a", "source": "infershape.cpp:45"}
},
{
  "name": "x_list",
  "param_type": "DYNAMIC",
  "tensor_count": {"param": "x_list_count", "min": 1, "max": 10, "source": "infershape.cpp:30"},
  "same_shape": true,
  "same_dtype": true,
  "dtype": {"values": ["float16", "float32"]},
  "rank": {"min": 1, "max": 8},
  "shape": {"constraints": ["列表内各 tensor shape 相同"]}
},
{
  "name": "y_list",
  "param_type": "DYNAMIC",
  "tensor_count": {"derived_from": "x_list", "source": "infershape.cpp:50"},
  "dtype": {"sync_with": "x_list"},
  "shape": {"rule": "same_as_input:x_list", "source": "infershape.cpp:52"}
}
```

---

## 关键规则

1. **以 def.cpp 为约束源**：inputs/outputs/attributes 的名称和数量必须以 def.cpp 为权威
2. **以源码为准**：所有 dtype、shape、约束必须从源码提取，不猜测
3. **逐字抄录约束**：OP_CHECK / PARAM_CHECK 表达式在输出 1（文本）中逐字抄录
4. **优先用 sync_with**：dtype/rank/shape 与其他输入相同时，用 `sync_with` 表达依赖
5. **DYNAMIC 独立性**：每个 DYNAMIC 槽位的 tensor count 独立采样。DYNAMIC 输入之间可能存在 shape/dtype/rank 等关联约束（通过 `sync_with` 或 `shape.constraints` 表达），但 tensor count 各自独立。DYNAMIC 输出的 count 从某个输入 tensor 推导（`derived_from`），不独立采样
6. **DYNAMIC 必填字段**：`param_type = "DYNAMIC"` 的输入必须填写 `tensor_count`（含 `param`/`min`/`max`）、`same_shape`、`same_dtype`；DYNAMIC 输出必须填写 `tensor_count.derived_from`
7. **value_domain 按表匹配**：优先查常见模式清单，未命中时从源码/文档推断，仍无法确定则留 null。禁止猜测不存在的定义域约束
8. **torch_npu_api_exposure 稳定对象**：无论是否找到 torch_npu 暴露入口，`torch_npu_api_exposure` 都必须是 object；未找到时写 `status=not_found` 和空 `param_gaps`，禁止写 `null`

---

## 严格禁止

1. 禁止编造 dtype 或 shape 约束 — 必须从源码提取
2. 禁止在同一个维度上同时使用 `values`/`min+max`/`constraints` 和 `sync_with`
3. 禁止省略任何输入 tensor（def.cpp 中声明的输入必须全部列出）
4. 禁止将 def.cpp 中未声明的参数写入 `attributes` 节 — aclnn 层额外参数在 Step 7 返回文本中列出，不写入 JSON
5. 禁止以 torch_npu 层的名称覆盖 `outputs[*].name`（以 def.cpp 为准）
6. 禁止 `sync_with` 循环引用 — 若 A 的某维度 sync_with B，则 B 的同一维度不得 sync_with A
7. 禁止省略 DYNAMIC 输入的 `tensor_count` 字段 — `param_type = "DYNAMIC"` 时必须填写 param/min/max
8. 禁止省略 DYNAMIC 输出的 `tensor_count.derived_from` 字段 — 必须引用决定 count 的输入 tensor
9. 禁止将 inplace 算子的 `outputs` 留空 — 当 `_def.cpp` 无 `Output()` 声明时，必须检查注释和算子名称判断是否为 inplace 模式，若是则将被修改的输入 tensor 作为隐式输出写入 `outputs`
10. 禁止将 Tensor/TensorList 的 rank/shape 约束套用到 aclScalar 输入
11. 禁止把 aclnn 签名为 `aclScalar*` 的参数建模为 rank 0-8 的普通 Tensor
