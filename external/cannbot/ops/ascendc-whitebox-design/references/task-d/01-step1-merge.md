# Step 1：可达性标注

> **前置条件**：已 Read 完整的 S2P1_path_list.json 和 S2P1_operator_model.json

对 S2P1_path_list.json 中每条路径，结合 S2P1_operator_model.json 的接口约束信息，判定其可达性。

## 输出

为每条路径标注以下字段（内存中标注，不写文件——所有落盘由 04-step4-output.md 统一执行）：

| 字段 | 必填 | 说明 |
|------|------|------|
| `reachability` | 是 | 5 选 1（见下表） |
| `dead_reason` | 条件 | reachability ∈ {dead, api_dead, api_warn} 时必填 |

**reachability 枚举**：

| 值 | 含义 | 判定来源 | 下游处理 |
|----|------|---------|---------|
| `reachable` | 存在至少一组合法输入能同时满足该路径的所有 conditions | 步骤 5（默认） | 纳入 Step 2 分组 |
| `dead` | 路径不可能被触发 | 步骤 3 规则 2-4 | 不纳入 Step 2 |
| `api_dead` | torch_npu API 不支持，用户无法通过 Python 触发 | 步骤 3 规则 1 | 不纳入 Step 2 |
| `api_warn` | torch_npu API 参数受限，部分取值不可达 | 步骤 3 规则 1 | 不纳入 Step 2 |
| `disputed` | 接口层声明不支持但 kernel 有完整实现 | 步骤 4 | 不纳入 Step 2，交由用户确认 |

## 执行流程

对每条路径按以下顺序执行（满足即短路，跳过后续步骤）：

### 1. 前置过滤

路径已被上游标记为 `reachability: "dead"` 时，直接保留 dead 分类，跳过重新判定。

### 2. 数据准备

从已 Read 的 `S2P1_operator_model.json` 中提取 `torch_npu_api_exposure`；无此节或为 null 时跳过步骤 3 的规则 1。

### 3. Dead 判定

按序号顺次判定，满足任一即标记为对应类别（dead / api_dead / api_warn），路径保留在列表中：

1. **torch_npu API 不可达**（标记为 `api_dead` 或 `api_warn`）：读取 `torch_npu_api_exposure.param_gaps`，对每个 gap 匹配 `aclnn_param` 到 path conditions 变量名。按 `torch_npu_status` 判定：`absent` 且取值∈blocked_values → `api_dead`；`fixed` 且取值≠fixed_value → `api_dead`；`derived` → `api_warn`；映射失败 → `api_warn`。`dead_reason` 格式：`"torch_npu_api_unsupported: {param}={value} - {desc}"`。

2. **kernel 中无实现**（`key_instructions` 标记为 `NO_KERNEL_DISPATCH`）
3. **条件组合被源码约束完全排除**
4. **tiling 无法产出对应 key**（如 dtype 与 tiling 分支硬编码矛盾）

### 4. Disputed 判定

接口层声明不支持但 kernel 有完整实现 → 标记为 `disputed`。

### 5. 默认 Reachable

以上均不满足 → 标记为 `reachable`。
