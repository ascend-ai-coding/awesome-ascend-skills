# Step 3：参数推导

> **前置条件**：Task D Step 2（路径分组）已完成，groups 已确定

为每个 group 推导路由维度和非路由维度的取值范围，产出能完整映射到算子输入空间的参数组合规格。**一次读 tiling 源码，完成以下全部步骤**。

所有 group 分析完成后，一次性写入 `S2P2_analysis_data.json`（仅含范围，不含具体值），然后运行 `assemble_dim_spec.py` 生成 `S2P2_dim_spec.json`，再运行 `pick_dims.py` 自动生成 `S2P2_param_def_groups.json`。`build_param_def.py` 转换在 Step 4.1 执行。

## 输入

| 数据 | 来源 |
|------|------|
| group 级约束（定义 tiling mode 的条件） | Step 2 分组分析（内存） |
| 每个 group 的路径及其 conditions | S2P1_path_list.json |
| input_variables / internal_variables / caller_options | S2P1_path_list.json |
| source_constraints | S2P1_path_list.json |
| 内部变量计算链 | tiling 源码（需读取） |

## 输出

| 产出 | 写入位置 | 产出步骤 |
|------|---------|---------|
| 路由维度范围 | `per_dtype[{dtype}].dims.{param}` | 2.4 |
| 复合维度范围 | `per_dtype[{dtype}].compound_dims.{compound_name}` | 2.4 |
| 非路由维度范围 | `group_dims.{dim}` | 步骤 3 |
| constraint_note | `constraint_note` | 5.2 |
| tiling key（per_dtype） | `per_dtype[{dtype}].key` | 5.1 |
| tiling key（顶层） | 顶层 `tiling_keys`，去重排序 | 步骤 6 |

## 命名规则

维度名沿用 `S2P1_path_list.json` 中的 tiling 源码变量名（含义见 `S2P1_tiling_glossary.md`），禁止使用抽象分解名（如 `hidden_size`、`batch_size`）。禁止将 `internal_variables` 中的纯计算中间变量作为 param 名。

## 执行层级说明

同一 group 内的约束分为两层：

- **Group 级约束**：所有 path 共享的条件（定义 tiling mode），继承自 Step 2 分组分析
- **Path 级约束**：每条 path 独有的条件（区分 group 内的不同分支）

Group 级约束对所有 path 相同，只需分析一次；Path 级约束因 path 而异，需逐 path 分析。

---

## 步骤 1：约束分层与维度分类

对每个 group，先分离约束层级，再分类维度。

### 1a. 约束分层

对每个 group：

1. **Group 级约束**：继承自 Step 2 的分组分析（定义 tiling mode 的条件）。这些条件在 Step 2 分组时已确定，Step 3 直接使用。

2. **Path 级约束**：对每条 path，其 path 级约束 = 该 path 的全部 conditions − group 级约束。这些条件区分 group 内的不同分支。

### 1b. 维度归属判定

**执行层级**：Group 级，对每个 group 执行一次。

基于约束分层结果，对每个维度按以下 3 类判定：

**写入位置的区别**：`per_dtype` 按 path × dtype 分桶，每个 entry 独立取值；`group_dims` 所有 path × dtype 共享同一组取值。路由维度因 path 而异，必须写入 `per_dtype`。

**判定依据**：是否出现在约束中（含 `expr` 表达式中引用的变量）、是否影响 tensor 空间。

1. **路由维度**：出现在 path 级约束中，或虽出现在 group 级约束但使用 `ref` 字段或影响分支选择 → 写入 `per_dtype[{dtype}].dims.{param}`，具体取值策略（耦合/退化/常规）由步骤 2.1 逐 path 判定

2. **Group 级维度**：写入 `group_dims.{dim}`，包含两种子情况：
   - **有约束但值固定**：仅出现在 group 级约束中，且使用 `value` 字段（值固定为常量），不影响 path 分支选择 → 记录 `{"value": 常量}`
   - **无约束**：不出现在任何约束中，但属于 `input_variables` 且影响 tensor 空间 → 步骤 3 记录范围

3. **不纳入**：不出现在任何约束中，且不影响 tensor 空间（如 epsilon 等标量属性）→ 由下游 case-mapper 注入默认值

**float 属性特殊值**：若为路由维度，选取 `0.0`（零值边界）、负值（若源码允许）、极小正数（精度边界）。

---

## 步骤 2：路由维度分析与范围记录

基于步骤 1 的分类结果，为路由维度推导取值范围。输出写入 `per_dtype[{dtype}].dims.{param}` 和 `per_dtype[{dtype}].compound_dims.{compound_name}`。

2.1-2.4 均按 Path 级执行，对每条路径独立分析。

### 2.1 提取参数 + 取值策略判定（Path 级）

**执行层级**：Path 级，对每条路径独立执行。

对每条 path：

a. 遍历该 path 的 path 级 conditions（步骤 1a 产出），提取引用的变量 → 确定该 path 的 params。

b. 对每个 path 级路由维度，逐条检查其 condition，判定取值策略：
    - **耦合**：多个维度之间存在数学关联，分为三类：
      - **确定性耦合**：一个维度可由另一个唯一推导（`B = f(A)`）。
        - **优先级 1（显式变量间关系）**：若该 path 的 conditions 中存在 `{"var": "A", "op": "...", "ref": "B"}` 格式（`ref` 为真变量），且 A、B 均已保留为路由维度，则 A 和 B 构成耦合组，数学关系由 `op` 决定（`==` 为恒等，`<=` 为序关系，`!=` 为互斥约束等）。
        - **优先级 2（隐式数学关系）**：若一组路由维度的任意合法取值组合均满足同一数学关系（不随 group/dtype 变化，如比例、和/积、互质等），且该关系不是由 conditions 显式声明的，则标记为耦合组。
        - 两种优先级可能重叠（如 conditions 声明 `A == B`，同时 tiling 源码隐含 `B == A * scale`）。优先级 1 的显式关系优先，不再重复识别为优先级 2。
      - **约束耦合**（优先级 3）：多个维度通过乘积、加减等数学约束纠缠在一起，一个维度的取值范围依赖于另一个维度的值（如 `A * B * dtypeSize >= N`）。`expr` 类型的 condition 引用多个真变量时，这些变量构成约束耦合组。与确定性耦合的区别：确定性耦合中 derived 由 base 唯一确定；约束耦合中 derived 的取值范围受 base 约束，但不是唯一值。
      - **互斥耦合**（优先级 4）：多个维度不能同时取某些特定值组合。当 `source_constraints` 中存在跨维度互斥约束（如 `!(A && B)`、`A != X || B != Y`），且涉及的多个维度均出现在该 path 的 conditions 中时，标记为互斥耦合组。与确定性/约束耦合的区别：互斥耦合不定义维度间的推导关系，而是定义合法组合的枚举集合。
      - `ref` 目标为伪变量时按退化或常规处理。
    - **退化**：取值恒为单一常量（condition 使用 `value` 字段，或 `ref` 目标为伪变量）→ 记录 `{"value": fixed_value}`。在 `constraint_note` 中标注退化原因。
   - **常规**：其余情况 → 步骤 2.4 记录范围。

### 2.2 回溯内部变量（Path 级）

**执行层级**：Path 级，对每条路径独立执行。

若该 path 的 conditions 中引用了 `internal_variables`（步骤 2.1a 提取的 params 中包含 internal_variable），读 tiling 源码回溯到 `input_variable`：
- 若计算链不经过任何 input_variable，跳过该条件（无需回溯）
- 识别内部变量 → 找到 tiling 源码中的赋值/计算位置
- 追溯计算链（如 `{internal_var} = {formula}({input_var}, {core_count})` → `{input_var} ≤ {N}`）
- 代入当前目标平台的常量计算等价边界值

### 2.3 确认取值范围（Path 级）

**执行层级**：Path 级，对每条路径独立执行。

综合以下约束来源与步骤 2.2 回溯得到的等价边界值，确认下界和上界。范围外的值不纳入取值列表。

| 约束类型 | 来源 | 默认/兜底 | 说明 |
|----------|------|----------|------|
| 下界 | `source_constraints` | 1 | — |
| 上界 | `source_constraints` | 实操上限（如 ≤ 65536），避免海量用例 | — |
| 跨维度互斥 | `source_constraints` | — | 检查是否存在两个或多个维度不能同时取某些值的约束，若存在则标记为互斥耦合组 |
| 路由阈值 | group 触发/退出的边界值 | — | 不同 group 间以该值为分界线 |
| 对齐约束 | tiling 源码 | 无对齐要求 | — |

### 2.4 记录取值范围（Path 级）

**执行层级**：Path 级，对每条路径独立执行。每条路径根据自身 path 级 conditions（步骤 1a 产出）确定具体边界值，互不干扰。

退化维度的值已在步骤 2.1 确定（`{"value": fixed_value}`），跳过 2.2-2.4。

对每条路径的每个 path 级路由维度，按取值策略记录范围：

**常规维度**：
- **数值范围型**（候选值 ≥5）：记录 `{"lo": 下界, "hi": 上界, "count": 5}`。lo/hi 由步骤 2.3 确定。
- **dtype 依赖下界型**：当 lo 由 `dim * dtypeSize >= N` 约束推导而来时，记录 `{"lo": "_auto", "hi": 上界, "count": 5}`，并在顶层 `_dtype_adjust` 中声明 `{"dim": {"type": "min_bytes", "bytes": N}}`。脚本会按各 dtype 的 dtypeSize 自动计算 lo。
- **离散类别型**（候选值 <5，如枚举型路由参数）：记录 `{"values": [全部有效值]}`。

**耦合维度**（步骤 2.1 标记的耦合组，记录 compound_dims 对象）：

**确定性耦合格式**：
- **base**：从耦合组中选一个维度作为基准维（通常为下界/上界约束最明确的那个），记录 `{"name": "基准维名", "lo": N, "hi": M, "count": 5}`
- **derived**：其余维度与基准维的确定性数学关系，使用 Python 表达式字符串（如 `"lenDesH_ * scaleH"`）
- **params**：表达式中引用的参数范围，使用 `{"参数名": {"lo": N, "hi": M}}`

**约束耦合格式**：
- **base**：从耦合组中选一个维度作为基准维（通常为约束中便于求解的变量），记录 `{"name": "基准维名", "lo": N, "hi": M, "count": 5}`
- **derived**：其余维度在给定 base 值后的取值范围，使用 min/max 表达式对象：
  ```json
  {"derived_dim": {"min": "expr_str", "max": "expr_str_or_null", "lo": N, "hi": M}}
  ```
  - `min`：derived 的下界表达式（对应 `>=` 约束求解），Python 表达式字符串，null 表示无下界约束
  - `max`：derived 的上界表达式（对应 `<=` 约束求解），null 表示无上界约束
  - `lo`/`hi`：derived 维度的原始范围，与 min/max 表达式求交集
- **params**：表达式中引用的参数范围（同确定性耦合）
- 表达式可引用 base 变量名、params 中的参数、`_dtype_size`（当前 dtype 的字节数，由 pick_dims.py 自动注入 eval 上下文）

**互斥耦合格式**：
- 显式枚举所有合法组合，记录为 `{"values": [{"dimA": v1, "dimB": v2}, {"dimA": v3, "dimB": v4}, ...]}`
- 每个 dict 代表一组合法取值组合，脚本从中随机选取
- 适用场景：维度取值空间小（候选值 <5）且存在互斥约束，无法通过独立采样保证合法性

**示例**（通用推导过程）：
- group 级约束：`{format_var} == {format_value}`（所有 path 共享）
- path 级约束：`{dim} ≤ {threshold}`、`{internal_var} == {value}`、`{platform_var} ≠ {platform}`
- 回溯 `{internal_var} == {value}`：L 行号 `{internal_var} = {formula}` → 等价于 `{input_var} ≤ {boundary}`
- 回溯 `{platform_var} ≠ {platform}`：代入目标平台常量 → 恒满足 / 恒不满足 / 条件简化
- spec 记录：`{"lo": 下界, "hi": 上界, "count": 5}`

**约束耦合示例**：
- condition：`{"expr": "A * B * dtypeSizeY_", "op": ">=", "value": 128}`
- 识别：A 和 B 通过乘积约束纠缠，dtypeSizeY_ 是 dtype 相关常量
- 选 base：A（约束中便于求解的变量）
- 求解 derived：B >= 128 / (A * _dtype_size) → `min = "math.ceil(128 / (A * _dtype_size))"`
- spec 记录：
  ```json
  "compound_dims": {
    "A_B_product": {
      "base": {"name": "A", "lo": 1, "hi": 256, "count": 5},
      "derived": {"B": {"min": "math.ceil(128 / (A * _dtype_size))", "max": null, "lo": 1, "hi": 256}},
      "params": {}
    }
  }
  ```
- pick_dims.py 处理：对每个 base 值 A=8（float16），计算 min=ceil(128/(8*2))=8，在 [max(1,8), min(256,256)]=[8,256] 内采样 B

---

## 步骤 3：非路由维度范围记录

这些维度不影响代码分支，但影响 tensor 的实际大小。为每个 group 提供 10 个取值作为 POOL，供工作流 Step 5（case mapper）的 `compress_group_pool()` 随机抽样生成测试用例。

对步骤 1b 归属为"group 级维度"的维度，推导其在各 group 中的取值范围。输出写入 `group_dims.{dim}`。与步骤 2 独立，可并行。

注：若某 group 无影响 tensor shape 的非路由维度，该 group 不含 `group_dims` 字段。

### 3.1 确定取值区间

按 Task D Step 2（路径分组）的 group 划分逻辑确定各维度在各 group 中的取值区间（区间信息来自 constraint_note 或 tiling 源码中的隐式边界）。区间确定规则：

a. 取值区间互斥的 group → 各自独立确定区间，区间不重叠。

b. 取值区间相同或不受限的 group → 每个 group 也独立确定区间。不受限区间（如 lo 明确但 hi 未定）的上限按以下优先级依次查找，取首个适用的值：
   1) `source_constraints` 中该维度的显式上限；
   2) 强制兜底值 500（仅当 tiling 源码或硬件限制表明需要更小时可降低，如 coreNum_=64 可作为 lenN_ 的上界）。
   规则 1 不适用时，必须使用规则 2，禁止自行选取更小的值。

**乘积约束**：同一 group 内所有 group_dims 维度的 hi 值之积不得超过 10000。若超过，按比例缩减各维度的 hi 值（优先缩减取值空间较大的维度）。

### 3.2 记录范围

对步骤 3.1 确定的非路由维度区间，记录 `{"lo": 下界, "hi": 上界, "count": 10}`。
脚本将在 [lo, hi] 内随机选取 10 个不重复整数。

---

## 步骤 4：跨层级元素数校验（基于范围）

**执行层级**：Path 级，对每条路径的每种 dtype 独立执行。

**前置**：步骤 2（路由维度）和步骤 3（非路由维度）均已完成。

本步骤基于 spec 中的 lo/hi 范围（而非具体取值）校验元素数约束。

### 4.1 计算极值范围

- **max_product** = 各维度 hi 值之积
- **min_product** = 各维度 lo 值之积
- 退化维度（`{"value": v}`）按该值计算
- **`_auto` 维度**：按 `_dtype_adjust` 声明的 bytes 值和最小 dtypeSize（通常为 float32=4）计算具体值参与极值计算，即 `computed_lo = ceil(bytes / min_dtype_size)`

### 4.2 校验与调整

- **max_product > upper_bound** → 缩减 per_dtype 维度的 hi 值（优先缩减常规维度，保持耦合关系不变）
- **min_product < lower_bound** → 增大 per_dtype 维度的 lo 值
- 仅调整 per_dtype 维度，不修改 group_dims
- **compound_dims.derived 维度**：不调整其 lo/hi 值。约束耦合的实际下界由 pick_dims.py 运行时计算，步骤 4 仅调整常规 dims 中的维度。
- 最多 3 轮，残余风险记入 constraint_note

---

## 步骤 5：计算 tiling key + constraint_note

为每条路径计算 tiling key 值（用于工作流 Step 5（case mapper）生成 case 时标识路径），并为每个 group 编写约束说明文本（供下游校验和人工审查）。

### 5.1 计算 tiling_keys

从 tiling 源码中确认每个 group 对应的 tiling key 常量值（通常为 `tilingKey_ = TILING_KEY_XXX` 直接赋值，无需计算）。

1. 对每个 group 内各 dtype 的路径，确认其 tiling key 常量值
2. **顶层 `tiling_keys`**：在步骤 6.1 写入 spec 文件。此处与 Step 2 key 覆盖校验结果交叉核对
3. **per_dtype**：每个 dtype 条目的 key 值写入 `per_dtype[{dtype}].key`

多条路径映射到相同 key 值 → 去重。同一 key 可能出现在多个 group。

### 5.2 编写 constraint_note

为每个 group 编写约束说明，汇总该 group 的维度取值约束。

**编写规则**：

- 只能引用当前 group 的 param 维度名（如 `{dim}`、`{dtype}`）和具体数值
- 禁止引用内部/中间变量名（如 tiling 源码中的 `{internal_var}`）
- 若约束来源于内部变量条件，必须将内部变量→param 的等价过程写入 S2P2_traceability.md 的「内部变量 → params 等价推导」表
- 退化维度（步骤 2.1 退化策略）须在 constraint_note 中标注其恒定取值和原因

---

## 步骤 6：一次性写入 analysis_data + 运行脚本

所有 group 完成步骤 1-5 后，一次性执行：

### 6.1 组装 S2P2_analysis_data.json

使用 Write 工具创建 S2P2_analysis_data.json，包含紧凑格式的范围定义。

**顶层字段**：
- `tiling_keys`：Step 2 分组时已确定的所有 tiling key（去重排序）
- `_dtype_adjust`：声明哪些维度需要根据 dtype size 自动计算
- `groups`：所有 group 的紧凑范围定义

**`_dtype_adjust` 格式**：

```json
"_dtype_adjust": {
  "<dim_name>": {"type": "min_bytes", "bytes": <int>}
}
```

声明哪些维度需要根据 dtype size 计算值。`type=min_bytes` 表示 `value = ceil(bytes / dtypeSize)`。

**每个 group 包含**：
- `id`：group 标识（如 "G1"）
- `mode`：触发模式一句话描述
- `constraint_note`：约束说明
- `group_dims`：Group 级维度范围对象（条件必填，无则省略该字段）
- `per_dtype`：按 dtype 分组的取值范围配置（紧凑格式）

**per_dtype 紧凑格式**：
- `_template`：所有 dtype 共用的 entries，脚本自动展开为每个 dtype 的独立 entries
- `<dtype>`：dtype 特定 entries，覆盖 `_template` 展开结果（可选，用于不同 dtype 走不同 tiling 路径的场景）

**`_auto` 标记**：在 dims 中使用 `"_auto"` 标记需要自动计算的字段：

```json
{"lo": "_auto", "hi": 1024, "count": 5}
```

脚本会从 `_dtype_adjust` 规则计算 `ceil(bytes / dtypeSize)` 并替换 `"_auto"`。`lo` 和 `hi` 均支持 `"_auto"`。

**per_dtype.{dtype}.entry 内字段**：
- `path`：S2P1 path ID
- `key`：tiling key 值
- `dims`：常规/离散维度范围对象
- `compound_dims`：耦合维度范围对象（条件必填，无则省略）

**格式示例**：

```json
{
  "tiling_keys": [10000, 20000, 30000],
  "_dtype_adjust": {
    "{dim_with_byte_constraint}": {"type": "min_bytes", "bytes": 128}
  },
  "groups": [
    {
      "id": "G1",
      "mode": "{mode_name} — ...",
      "constraint_note": "...",
      "group_dims": {
        "{group_dim}": {"lo": 1, "hi": 1024, "count": 10}
      },
      "per_dtype": {
        "_template": [
          {
            "path": "P1", "key": 10000,
            "dims": {
              "{dim_with_byte_constraint}": {"lo": "_auto", "hi": 1024, "count": 5},
              "{discrete_attr}": {"values": [0, 1]},
              "{fixed_dim}": {"value": 1}
            },
            "compound_dims": {
              "{compound_name}": {
                "base": {"name": "{base_dim}", "lo": 1, "hi": 32, "count": 5},
                "derived": {"{derived_dim}": "{base_dim} * {scale_param}"},
                "params": {"{scale_param}": {"lo": 1, "hi": 5}}
              },
              "{constraint_compound_name}": {
                "base": {"name": "{base_dim}", "lo": 1, "hi": 256, "count": 5},
                "derived": {
                  "{constrained_dim}": {
                    "min": "math.ceil({threshold} / ({base_dim} * _dtype_size))",
                    "max": null,
                    "lo": 1, "hi": 256
                  }
                },
                "params": {}
              }
            }
          }
        ]
      }
    }
  ]
}
```

**维度规格类型**：
- `{"lo": N, "hi": M, "count": K}` → 范围型，脚本随机选 K 个不重复整数
- `{"lo": "_auto", "hi": M, "count": K}` → 范围型，lo 从 `_dtype_adjust` 自动计算
- `{"values": [...]}` → 离散型，脚本直接使用
- `{"value": V}` → 固定型，脚本生成 `[V]`

**禁止字段**：以下字段已废弃，禁止出现在 S2P2_analysis_data.json 的任何层级中：
`t`、`coverage`、`thresholds`、`anchor_dim`、`per_value`、`alignment`、`constraints`(JSON 格式)、`low_configs`、`desc_rules`。

验证：`python3 -c "import json; json.load(open('S2P2_analysis_data.json'))"`

### 6.1.1 运行 assemble_dim_spec.py

```bash
python3 {skill_base}/scripts/assemble_dim_spec.py \
  --input S2P2_analysis_data.json \
  --operator-model S2P1_operator_model.json \
  --output S2P2_dim_spec.json
```

脚本职责：
- 读取 S2P2_analysis_data.json（紧凑格式）
- 读取 S2P1_operator_model.json（提取 platform/dtypes/dtype_tensors）
- 识别 `"_auto"` 标记，按 `_dtype_adjust` 规则计算维度值
- 展开 `_template` → 按 dtype 复制 per_dtype entries（支持 dtype 特定覆盖）
- 组装顶层结构，写入 S2P2_dim_spec.json

验证：`python3 -c "import json; json.load(open('S2P2_dim_spec.json'))"`

### 6.2 运行 pick_dims.py

```bash
python3 {skill_base}/scripts/pick_dims.py \
  --input S2P2_dim_spec.json \
  --output S2P2_param_def_groups.json
```

### 6.3 格式化

```bash
python3 {skill_base}/scripts/format_json.py S2P2_param_def_groups.json
```

`build_param_def.py` 转换由 Step 4.1 执行。

**完成标志**：S2P2_param_def_groups.json 已生成且 python3 可正常 `json.load()` 读取。
