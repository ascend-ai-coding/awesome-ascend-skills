# 策略卡片 Frontmatter Schema

每个策略卡片文件（`{ID}_*.md`）头部必须包含 YAML frontmatter，用于程序化筛选（`query_strategies.py`）。

## 完整示例

```yaml
---
id: P1
bottlenecks:
  - mte2_stall
  - no_overlap
op_families:
  - elementwise
  - normalization
  - attention
complexity: L1
conflicts_with: [P19]
synergizes_with: [P5, P14]
requires: []                       # v3.2 新增
has_preconditions: true            # v3.2 新增
has_playbook: true                 # v3.2 新增
quantified_gain:                   # v3.2 新增（可选）
  - shape: "M=1024,K=4096,N=2048,bf16"
    baseline_us: 66.0
    optimized_us: 63.3
    speedup: 1.04
    source: "round_3_parallel_2"
---

# P1: 双缓冲机制 (Double Buffering)

## 核心思想
...
```

## 字段规范

### `id` (必填)
- 类型：字符串
- 约束：必须与文件名前缀一致（如 `P1_*.md` → `id: P1`）
- 取值：`P1`-`P12`, `P14`-`P88`（性能）, `D1`-`D5`（数据类型）, `A1`-`A8`（精度）, `X1+`（进化发现）

### `bottlenecks` (必填，可为空列表 `[]`)
- 类型：字符串列表
- 含义：该策略能**缓解**的瓶颈类型
- 取值枚举（14 种，与 `plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/profiling_evidence.py` BOTTLENECK_STRATEGY_MAP 对齐）：

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

**填写依据**：对着 `strategy_index.md` L250-269 "按瓶颈查表"，找到该策略出现在哪些瓶颈行的 Primary/Secondary 列。

### `op_families` (必填，可为空列表 `[]`)
- 类型：字符串列表
- 含义：该策略**适用**的算子族
- 取值枚举（15 种）：

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

**填写依据**：对着 `strategy_index.md` L233-248 "按算子类型快速查表"，找到该策略出现在哪些算子行。若 L0 通用策略，填 `[datapath]`。

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

**填写依据**：
1. 读完整策略文件的 `## Trade-off` / `When Not to Use` 段落
2. 同类互斥策略（如 P1 标准双缓冲 vs P19 自定义 Ping-Pong 双缓冲，同一 buffer 不能同时用两种管理方式）
3. 若无明确冲突，填 `[]`

**注意**：`conflicts_with` 是对称关系，若 P1 冲突 P19，则 P19 也应冲突 P1。

### `synergizes_with` (必填，可为空列表 `[]`)
- 类型：字符串列表（其他策略 ID）
- 含义：推荐**搭配**使用的策略

**填写依据**：
1. 读完整策略文件的 `## Combine with` / `## Related` 段落
2. 典型互补关系：
   - P1 双缓冲 + P5 流水同步（几乎必配）
   - P14 CV 预发射 + P18 L1 7-buffer（FA 专用组合）
   - P8 UB 分区 + P20 三缓冲（内存管理强化）
3. 若无明确协同，填 `[]`

---

## v3.2 新增字段

### `requires` (可选，默认 `[]`)
- 类型：字符串列表（其他策略 ID）
- 含义：当前策略**依赖**的前置策略，必须先应用前置才能用本策略
- 与 `synergizes_with` 的区别：`requires` 是强依赖（缺则本策略不成立），`synergizes_with` 是协同（推荐组合）

**典型示例**：
- P1 `requires: [P5, P7]` — 双缓冲需要流水同步 + 32B 对齐做基础
- P14 `requires: [P18]` — CV 预发射依赖 L1 buffer 设计

### `has_preconditions` (必填，bool)
- 类型：布尔值
- 含义：是否存在对应的 Preconditions YAML 文件（`references/preconditions/{id}.yaml`）
- 用途：wm_ops select 阶段决定是否需要调用 `check_preconditions.py` 做硬过滤
- **可由脚本自动填充**：扫描 preconditions/ 目录即可推断

### `has_playbook` (必填，bool)
- 类型：布尔值
- 含义：是否存在对应的 Playbook 文件（`references/playbooks/{id}_*.md`）
- 用途：partial-prompt 注入时决定是否加载 Playbook 全文
- **可由脚本自动填充**：扫描 playbooks/ 目录即可推断

### `quantified_gain` (可选，列表)
- 类型：对象列表
- 含义：cannbot 风格的实测加速比，每条对应一个验证过的 shape 组合
- 字段：
  - `shape`: 算子 shape 描述（如 `"M=1024,K=4096,N=2048,bf16"`）
  - `baseline_us`: baseline 实测耗时（微秒）
  - `optimized_us`: 应用策略后耗时
  - `speedup`: 加速比（`baseline_us / optimized_us`）
  - `source`: 数据来源标识（如 `"round_3_parallel_2"`、`"manual_benchmark"`）
- 用途：
  - select 阶段加权（有验证过的高加速比应优先选）
  - cross-session 种子复用查询（Phase E）
- **填写时机**：跑 evo loop 时由 wm_ops refine 自动追加（Phase D），手工补充也可

## 排版规范

- `---` 标记必须位于文件第一行和 frontmatter 结束行
- frontmatter 后紧跟空行，再是 `# ID: Name` 标题
- 列表字段使用 `- item` 或 `[a, b]` 均可
- 字段顺序按 schema 列出顺序（id → bottlenecks → op_families → complexity → conflicts_with → synergizes_with → requires → has_preconditions → has_playbook → quantified_gain）

## 校验

运行 `query_strategies.py --validate-all` 校验所有卡片的 frontmatter：
- ✅ 必填字段完整
- ✅ 枚举值合法
- ✅ YAML 语法正确
- ✅ `id` 与文件名前缀一致
- ✅ `conflicts_with`/`synergizes_with` 中的 ID 都存在

## 使用时的 Token 预算

- 单个策略卡片 frontmatter ≈ 10-15 行
- 主 agent 注入 3 个策略卡片给子 agent 时，frontmatter 占 ~40 行（不到子 agent 上下文的 1%）
- `extract_strategy_reference.py` **会自动剥离 frontmatter**，子 agent 看到的 content 不含元数据
