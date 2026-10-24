# 策略元数据 strategy_meta Schema

策略卡正文已迁至 cannbot-knowledge 仓（`knowledge/contrib/ascendc/optimizations/evolution_strategies/`，
frontmatter 遵循该仓 OKF schema）。卡片的**机器元数据**（程序化筛选用）吸收在
`references/preconditions/{ID}.yaml` 的 `strategy_meta:` 块中（位于 `checks:` 段之前），
是 `query_strategies.py` / `generate_index_json.py` 的权威数据源。

## 完整示例

```yaml
strategy_id: P1
description: 双缓冲适用性检查 — ...

strategy_meta:
  id: "P1"
  slug: "P1_double_buffering"
  title: "P1: 双缓冲机制 (Double Buffering)"
  bottlenecks: [mte2_stall, no_overlap]
  op_families: [elementwise, normalization, attention]
  conflicts_with: [P19]
  synergizes_with: [P5, P14]
  requires: []
  complexity: "L1"
  deprecated: false
  has_preconditions: true
  has_playbook: true
  doc_repo: cannbot-knowledge
  doc_path: "knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p1_double_buffer.md"
  playbook_path: "knowledge/contrib/ascendc/runbooks/performance/p1_double_buffering.md"
  playbook_slug: "P1_double_buffering"
  quantified_gain: [{"shape": "M=1024,K=4096,N=2048,bf16", "baseline_us": 66.0, "optimized_us": 63.3, "speedup": 1.04, "source": "round_3_parallel_2"}]

checks:
  - id: ...
```

## 字段规范

### `id` (必填)
- 类型：字符串
- 约束：必须与 yaml 文件名一致（`P1.yaml` → `id: P1`）
- 取值：`P1`-`P147`（性能，含缺号）, `D1`-`D5`（数据类型）, `A1`-`A8`（精度）, `R1`-`R8`（A5 架构，全部 deprecated）, `X1+`（进化发现，正文在 cannbot-knowledge 仓 `knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/`，OKF frontmatter）

### `slug` / `title` (必填)
- `slug`：原策略卡文件名 stem（`{ID}_{name}`），INDEX.json `source_key` 的组成部分
- `title`：策略卡 H1 标题

### `bottlenecks` (必填，可为空列表 `[]`)
- 类型：字符串列表（行内 flow 格式）
- 含义：该策略能**缓解**的瓶颈类型
- 取值枚举（26 种，与 `community/ops/ascendc-evolution-world-model/scripts/profiling_evidence.py` BOTTLENECK_STRATEGY_MAP 对齐）：

| 瓶颈值 | 含义 |
|---|---|
| `mte2_stall` | MTE2（Scalar→UB 搬运）停等 |
| `mte3_stall` | MTE3（UB→GM 搬出）停等 |
| `tiling_imbalance` | 分核不均衡 |
| `scalar_loading` | Scalar 加载瓶颈 |
| `scalar_compute` | Scalar 计算瓶颈 |
| `compute_bound` | 计算密集型 |
| `near_optimal` | 接近最优（无明显瓶颈）|
| `no_overlap` | 无流水重叠 |
| `partial_overlap` | 部分流水重叠 |
| `undersize_transfer` | 搬运粒度过小 |
| `icache_miss` | 指令缓存未命中 |
| `bus_contention` | 总线竞争 |
| `l2_cache_thrash` | L2 Cache 抖动 |
| `ub_memory_pressure` | UB 内存压力 |
| `memory_bound` | GM 访存带宽打满 |
| `mte1_stall` | MTE1（GM→L1 搬运）停等 |
| `mte2_bound` | MTE2 搬入带宽打满（区别于 `mte2_stall` 的停等）|
| `parallelism_underutil` | 核间并行度不足（未用满 AI Core）|
| `tiling_mismatch` | tile 尺寸与硬件约束/shape 不匹配 |
| `sync_overhead` | 同步开销过大（barrier/event 过密）|
| `tail_block` | 尾块/尾包处理拖慢整体 |
| `ub_overflow` | UB 溢出风险 |
| `ub_pressure` | UB 容量紧张 |
| `vec_underutilization` | Vector 核利用率不足 |
| `cube_underutilization` | Cube 核利用率不足 |
| `portability` | 跨芯片可移植性问题 |

**填写依据**：对着 `strategy-index.md` "按瓶颈查表"，找到该策略出现在哪些瓶颈行的 Primary/Secondary 列。

### `op_families` (必填，可为空列表 `[]`)
- 类型：字符串列表（行内 flow 格式）
- 含义：该策略**适用**的算子族
- 取值枚举（24 种）：

| 算子族 | 含义 |
|---|---|
| `elementwise` | 逐元素算子（foreach、Add、Mul 等）|
| `normalization` | LayerNorm / RMSNorm |
| `reduction` | 归约算子（Sum、Max、Mean 等）|
| `softmax` | Softmax |
| `attention` | Softmax / Attention（通用）|
| `flash_attention` | Flash Attention 系列（特化）|
| `cv_fusion` | Cube+Vector 融合算子 |
| `matmul` | 矩阵乘 |
| `moe` | MoE FFN |
| `quantization` | 量化/反量化 |
| `pooling_gather` | Pooling / Gather |
| `optimizer` | Optimizer 算子（Adam、SGD 等）|
| `index_scatter` | Index / Scatter |
| `broadcast_mask` | Broadcast / Mask |
| `special` | 特殊/复杂算子 |
| `datapath` | 通用数据搬运通路（所有算子族都适用，如 D1-D5, P1-P12, A1-A8）|
| `ssm` / `linear_attention` / `recurrent` | 线性递归/SSM 系 |
| `transpose` | 转置 / Permute 类 |
| `topk` | TopK 类 |
| `sort` | 排序类 |
| `conversion` | 数据类型/格式转换类 |
| `broadcast` | 广播类 |

**填写依据**：对着 `strategy-index.md` "按算子类型快速查表"。若 L0 通用策略，填 `[datapath]`。

### `complexity` (必填，单值)
- 类型：字符串
- 取值：`L0` / `L1` / `L2`

| 等级 | 含义 | 示例 |
|---|---|---|
| `L0` | **参数级**调整 | `BUFFER_NUM=2`（P1 的参数层面）、tile_size 调整 |
| `L1` | **局部重构** | 重写循环结构（P1 完整实现）、修改分核逻辑（P4）|
| `L2` | **架构重构** | CV 流水预发射（P14）、L1 7-buffer（P18）、整体计算范式改变 |

**填写依据**：读卡片 `## 代码骨架` 段落：
- 只有常量修改 → L0
- 有新增/删除循环、buffer 结构 → L1
- 改变整体计算范式、多核协作 → L2

### `conflicts_with` (必填，可为空列表 `[]`)
- 类型：字符串列表（其他策略 ID）
- 含义：与当前策略**互斥**的策略，不能同时使用
- **注意**：`conflicts_with` 是对称关系，若 P1 冲突 P19，则 P19 也应冲突 P1。

### `synergizes_with` (必填，可为空列表 `[]`)
- 类型：字符串列表（其他策略 ID）
- 含义：推荐**搭配**使用的策略（典型互补：P1+P5、P14+P18、P8+P20）

### `requires` (可选，默认 `[]`)
- 类型：字符串列表（其他策略 ID）
- 含义：当前策略**依赖**的前置策略（强依赖，缺则本策略不成立），与 `synergizes_with`（协同推荐）不同
- 典型示例：P1 `requires: [P5, P7]`、P14 `requires: [P18]`

### `deprecated` / `deprecated_reason` (必填 / 条件必填)
- `deprecated: true` 的策略（当前 R1-R8 + P6/P13/P42/P62/P82）仅作方向留档，
  `query_strategies.py` 仍会列出但 INDEX.json 带 `deprecated` 标记，落地前必须确认 active 状态。

### `has_preconditions` (必填，bool)
- 含义：本 yaml 是否含 `checks:` 可执行检查段
- 用途：wm_ops select 阶段决定是否需要调用 `check_preconditions.py` 做硬过滤
- **可由脚本自动校准**：`backfill_resource_flags.py`

### `has_playbook` (必填，bool)
- 含义：是否存在对应 Playbook（cannbot-knowledge 仓 `runbooks/performance/`，本块 `playbook_path` 字段）
- 用途：partial-prompt 注入时决定是否加载 Playbook 全文
- **可由脚本自动校准**：`backfill_resource_flags.py`

### `doc_repo` / `doc_path` / `playbook_path` / `playbook_slug` (必填 / 条件必填)
- `doc_repo`：恒为 `cannbot-knowledge`，标记 doc_path 的归属仓
- `doc_path`：策略卡在 cannbot-knowledge 仓的相对路径（`knowledge/` 前缀）
- `playbook_path` / `playbook_slug`：有 Playbook 时存在；`playbook_slug` 为原 playbook 文件名 stem
- 路径解析（`load_playbook.py`）：环境变量 `CANNBOT_KNOWLEDGE_ROOT` > 自 cwd 向上最近的 `.cannbot/knowledge.env`（cannbot-knowledge install.sh 写入）> 兄弟检出 `../cannbot-knowledge`；全部落空时 fail-loud（exit 3）

### `quantified_gain` (可选，单行 JSON flow)
- 类型：单行 JSON 数组（**不是** YAML 块列表——轻量 parser 按行解析，JSON 保证单行可读）
- 含义：实测加速比，每条对应一个验证过的 shape 组合
- 字段：`shape` / `baseline_us` / `optimized_us` / `speedup` / `source`
- 用途：select 阶段加权、cross-session 种子复用查询
- **填写时机**：跑 evo loop 时由 wm_ops refine 自动追加到本字段（不再是卡片 frontmatter）

## 排版规范

- `strategy_meta:` 块必须位于 `checks:` 段**之前**（`check_preconditions.py` 的轻量 parser 会跳过它）
- 列表字段一律行内 flow 格式 `[a, b]`；字符串含特殊字符时用双引号
- `quantified_gain` 必须单行（JSON），不得折行
- 无 checks 的策略（如部分通用卡）yaml 只含 `strategy_id` / `description` / `strategy_meta`

## 校验

运行 `query_strategies.py --validate-all` 校验所有 strategy_meta：
- ✅ 必填字段完整
- ✅ 枚举值合法
- ✅ `id` 与文件名一致
- ✅ `conflicts_with`/`synergizes_with` 中的 ID 都存在

## 使用时的 Token 预算

- 单个 strategy_meta ≈ 15-20 行
- 主 agent 通过 `query_strategies.py` / `load_playbook.py` 程序化加载策略内容，
  子 agent 看到的 Playbook content 不含 OKF frontmatter（自动剥离）
