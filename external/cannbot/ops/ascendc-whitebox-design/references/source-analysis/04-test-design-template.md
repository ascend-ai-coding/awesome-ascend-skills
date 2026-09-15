# S2P3_test_design.md 模板

## 用途

源码分析完成后，Phase 3b 生成 `S2P3_test_design.md`。该文件用于固化 Step 2 的路径、group、参数维度与 case 设计，并作为 Step 3 后验证结论写回和 Step 4 用户确认的依据。

Phase 3b 分为两段：

1. 脚本生成区：由 `{skill_base}/scripts/generate_test_design.py` 从标准产物确定性生成。
2. LLM 分析区：由主 Agent 基于脚本生成区和 Step 2 产物补充分析，不得修改脚本生成区事实。

最终输出文件路径保持不变：`{output_dir}/S2P3_test_design.md`。

## 输入文件

脚本默认读取 `{output_dir}` 下的以下文件：

| 文件 | 用途 |
|------|------|
| `S2P1_path_list.json` | 路径、conditions、kernel、reachability、group |
| `S2P1_operator_model.json` | inputs / outputs / attributes / API 暴露差异 |
| `S2P1_low_configs.json` | 常见网络配置数量统计 |
| `S2P2_param_def.json` | group 顺序、per_dtype 参数维度、tiling_keys |
| `S2P2_cases.json` | case 总数、按 group/key/dtype 分布 |
| `S2P2_traceability.md` | LLM 分析区的推导链依据 |

## 生成命令

```bash
python3 {skill_base}/scripts/generate_test_design.py \
  --op-name {op_name} \
  --op-path {op_path} \
  --output-dir {output_dir}
```

默认行为：

- 文件不存在：生成脚本区、LLM 分析区占位、verifier 区。
- 文件存在：只替换脚本区，保留 LLM 分析区和 verifier 区。
- `--force`：重写整个文件。
- `--overwrite-llm-section`：重置 LLM 分析区。

## 分区标记

`S2P3_test_design.md` 必须包含以下 marker，用于脚本重跑和 Step 3 写回定位：

```markdown
<!-- BEGIN SCRIPT GENERATED SECTION -->
...
<!-- END SCRIPT GENERATED SECTION -->

<!-- BEGIN LLM ANALYSIS SECTION -->
...
<!-- END LLM ANALYSIS SECTION -->

<!-- BEGIN VERIFIER SECTION -->
...
<!-- END VERIFIER SECTION -->
```

## 章节结构（强制）

```markdown
# {OpName} 白盒测试设计

<!-- BEGIN SCRIPT GENERATED SECTION -->
## 1. 生成与输入摘要
## 2. 接口与参数模型
## 3. 路径与 Group 覆盖
### 3.1 代码路径全景
### 3.2 测试关注点（groups）
## 4. Case 枚举与一致性校验
## 5. 自动发现的未确认项
<!-- END SCRIPT GENERATED SECTION -->

<!-- BEGIN LLM ANALYSIS SECTION -->
## 6. 测试设计分析
### 6.1 事实摘要与设计结论
### 6.2 关键派生变量
### 6.3 执行模式分析
## 7. 风险与补充建议
<!-- END LLM ANALYSIS SECTION -->

<!-- BEGIN VERIFIER SECTION -->
## 8. Step 3 验证结论（原 §9 验证结论）
（Step 3 完成后由 verifier 填写）
<!-- END VERIFIER SECTION -->
```

## 脚本生成区要求

### 1. 生成与输入摘要

由脚本生成，包含：

- 算子名、平台、输出目录。
- 输入文件解析状态。
- path 数、group 数、case 数、low config 数。
- reachability 分布。

### 2. 接口与参数模型

由脚本从 `S2P1_operator_model.json` 生成：

- inputs / outputs / attributes 表。
- dtype / rank / shape / value_domain 摘要。
- 若存在 API 暴露差异（如 `torch_npu_api_exposure.param_gaps`），必须展示。

### 3. 路径与 Group 覆盖

本章是 Phase 3 的事实核心，必须由脚本确定性生成。

#### 3.1 代码路径全景

脚本必须逐条列出 `S2P1_path_list.json` 中每个 path：

| 字段 | 来源 |
|------|------|
| path id | `paths[*].id` |
| tiling key | `paths[*].tiling_key` |
| reachability | `paths[*].reachability` |
| group | `paths[*].group` |
| conditions | `paths[*].conditions` |
| kernel | `paths[*].key_instructions` |
| source | `paths[*].source` |

禁止折叠或只写概要。path 数必须等于 `len(S2P1_path_list.json.paths)`。

#### 3.2 测试关注点（groups）

脚本必须按 `S2P2_param_def.json.groups` 顺序列出每个 group：

- `id`
- `mode`
- `constraint_note`
- `per_dtype`
- `path`
- `key`
- entry 中除 `path` / `key` 外的所有维度字段

维度字段必须通用遍历，禁止 hardcode `numCol`、`numRow`、`M/N/K` 等具体变量名。禁止引入 `S2P2_param_def.json` 中不存在的维度值。

### 4. Case 枚举与一致性校验

脚本生成：

- `S2P2_cases.json` case 总数。
- 按 `_group` / `key` 分布。
- 若 `S2P2_param_def.json.dtype_tensors[*].param` 存在，按对应 dtype 参数统计。
- 自动一致性校验表。

脚本必须执行以下硬校验，失败则不得进入 LLM 分析或 Step 3：

1. 必需输入文件存在且 JSON 可解析。
2. path ID 唯一且非空。
3. group ID 和顺序可读取。
4. reachable path 必须有 group。
5. `param_def.groups[*].per_dtype[*].path` 必须存在于 path list。
6. `S2P2_cases.json[*].path` 必须存在于 path list。

### 5. 自动发现的未确认项

脚本根据结构化字段自动生成：

- `reachability == disputed` 的 path。
- `reachability == api_warn` / `api_dead` / `dead` 的 path。
- `S2P1_operator_model.json` 中的 `param_gaps`。
- `S2P1_path_list.json.completeness_checklist.unresolved_items`。

无未确认项时写“无”。

## LLM 分析区要求

LLM 只能填写 `<!-- BEGIN LLM ANALYSIS SECTION -->` 与 `<!-- END LLM ANALYSIS SECTION -->` 之间的内容，禁止修改脚本生成区和 verifier 区。

LLM 必须遵守：

- 只基于脚本生成区、`S2P2_traceability.md`、`S2P1_path_list.json.source_constraints`、`S2P1_operator_model.json` 分析。
- 不得新增 `S2P2_param_def.json` 中不存在的维度值。
- 不得改写 path/group/reachability/key 等脚本区事实。
- 信息不足时明确写“未从结构化产物中得到稳定结论”，不得猜测。

### 6. 测试设计分析

包含三个小节：

1. `6.1 事实摘要与设计结论`：概述可覆盖路径、group 设计、主要不可达或 API 警告情况。
2. `6.2 关键派生变量`：优先基于 `S2P2_traceability.md` 的推导链；无稳定信息时引用 `source_constraints`。
3. `6.3 执行模式分析`：分析分核、UB、对齐、尾块、特殊路径等；无稳定信息时明确说明。

### 7. 风险与补充建议

汇总：

- API 暴露限制。
- dead/orphan 路径是否需要专项验证。
- disputed/api_warn 的处理建议。
- 是否需要额外 case。
- 是否可进入 Step 3。

## Verifier 区要求

`## 8. Step 3 验证结论（原 §9 验证结论）` 保留给 Step 3 后写回。Step 3 / Step 4 流程引用“验证结论节”时，统一指该章节。

## 通用维度指引

### data_range

每个 group 应包含 `data_range` 维度（除非算子有特殊限制），控制输入 tensor 的数据值域：

```json
"data_range": ["normal", "zero", "extreme", "negative", "tiny_pos", "all_ones", "near_zero", "with_inf", "with_nan"]
```

| 标签 | 含义 | 测什么 |
|------|------|--------|
| normal | torch.randn 正态随机 | 一般场景 |
| zero | 全零 | 零值传播、除零保护 |
| extreme | 接近 dtype 最大值 | 溢出、饱和 |
| negative | 全负数 | sigmoid/silu 负值分支 |
| tiny_pos | 极小正数（~1e-6） | 精度损失、scale 除零 |
| all_ones | 全 1 | 恒等验证 |
| near_zero | 接近零的正负混合 | 符号翻转、舍入 |
| with_inf | 正常数据中混入 inf | inf 传播处理 |
| with_nan | 正常数据中混入 nan | nan 传播处理 |

定义域约束：当 `S2P1_operator_model.json` 中某输入的 `value_domain` 非 null 时，Step 5c 的 `expand_high()` 会自动过滤不兼容的 data_range 标签。

### ndim

如果算子支持多种 rank（如 ndim 2~8），可在参数定义中加入 ndim 维度。脚本只展示 `S2P2_param_def.json` 中已有维度，不自行推导 ndim。
