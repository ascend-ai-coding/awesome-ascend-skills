# Step 2：路径分组

> **前置条件**：Step 1 已完成；仅对 reachable 路径分组

对所有 reachable 路径，按 tiling 源码的分支判断逻辑组织为 group。每个 group 对应一种 tiling 模式，作为 Step 3 参数推导的基本单元。

## 输出

执行以下操作，结果在内存中标注。`group` 和 `groups` 由 Step 4.2 写入 path_list.json，`mode` 由 Step 3 步骤 6 写入 S2P2_analysis_data.json。

1. 为每条 reachable 路径回填 `group` 字段
2. 构建顶层 `groups` 列表
3. 为每个 group 撰写 `mode` 描述（一句话概括触发条件）

| 产出 | 位置 | 说明 |
|------|------|------|
| `group` | 每条 reachable 路径 | 标识所属 group 的 id |
| `groups` | 顶层 | 所有 group id 的有序列表 |
| `mode` | 每个 group | 一句话概括触发条件（如 `"MODE_LARGE — {route_var} > {threshold}"`），将在 Step 3 写入 S2P2_analysis_data.json |

group 命名规则：读 tiling 源码中的模式常量或分支标识命名（如 `MODE_NCHW` → "nchw"）。降级路径在上游路径 id 对应 group 名后加 `_degrade` 后缀。

## 分组规则

路径 ID 采用 `T{t}K{k}` 命名（见 `04-path-config-schema.md` §路径 ID 命名规则），分组可直接从 ID 提取，无需重新分析 tiling 源码。

### 基本规则

**T 前缀归组**：共享相同 `T{t}` 前缀的 reachable 路径自动归入同一 group。group id 使用语义名称（读 tiling 源码中的模式常量或分支标识命名，如 `MODE_NCHW` → "nchw"），不使用 T 序号作为 group id。

### 降级路径

id 含 `d{num}` 后缀的路径（如 `T5K1d1`）为降级路径，独立为单独 group，group 命名在父 group 名后加 `_degrade` 后缀。

判定特征：路径 id 含 `d{num}` 后缀，或 conditions 含 `boundary_check`，或 tiling 源码中 DoTiling 函数内有条件跳转到其他 key 的逻辑。

## 校验

1. **路径全覆盖**：遍历所有 reachable 路径，确认每条已分配 `group`
2. **tiling key 互斥**：不同 group 的 tiling key 集合互不相交（每个 key 恰好属于一个 group）。同一 group 内可包含多个 key（如同一 DoTiling 函数对应多条路径）。例外：降级 group 的 key 可与正常 group 的 key 重叠
3. **tiling key 覆盖**：汇总所有 group 覆盖的 tiling key，对比 total_key_count，验证所有 key 均被覆盖
4. **group 数量合理性**：group 数量应等于 tiling 源码中 tiling 策略选择函数的分支数（不含 default）加上降级路径数

发现遗漏 → 标记并补上对应路径后再输出。
