---
name: external-cannbot-ops-pypto-op-perf-tune-tune-swimlane
description: PyPTO 算子深度性能调优技能。通过泳道图分析及调优性能，包括 Stitch 调优、TileShape 深度调优、合图调优、调度策略调优等。当用户需要进行深度性能调优、泳道图分析、Stitch
  优化、合图优化时使用此技能。触发词：深度性能调优、泳道图分析、Stitch 调优、合图调优、调度优化。
original-name: tune-swimlane
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# PyPTO 算子深度性能调优

## 概述

深度性能调优通过泳道图分析及调优性能，采用 man-in-loop 的方式，通过获取并分析当前算子性能数据，针对性调整各性能配置参数，经过迭代调优逐步逼近最佳性能。

> 资料获取统一使用 skill `pypto-docs-search`：按需搜索算子参考实现等文件/目录/内容。

## ⛔ 前置条件（强制门控）

1. **完成开箱性能调优**：先进行代码级优化
2. **精度校验通过**：确保算子计算正确
3. **已采集性能数据**：生成泳道图和气泡分析报告

**⛔ ⛔ ⛔ 独立采集数据（强制）：每次进入此阶段时，必须重新运行测试（带 debug_options）采集最新泳道图数据。禁止复用 FRONTEND 阶段或其他轮次的旧数据！修改代码后性能特征已变，旧数据无法反映当前状态，以此决策会导致错误结论。⛔ ⛔ ⛔**

## 泳道图分析

### 泳道图文件位置

泳道图数据文件位于 `output/output_*/` 目录：
- `merged_swimlane.json` - 泳道图数据文件
- `bubble_analysis.log` - 气泡分析报告

### 查看泳道图

1. 通过 PyPTO Toolkit 插件查看
2. 或在 https://ui.perfetto.dev/ 上传泳道图文件
3. 查看泳道图文件及日志信息

### 泳道图关键信息

- 任务的执行顺序和耗时信息
- 各核心的工作时间和等待时间
- 气泡（线程等待调度的时间）
- 任务依赖关系

## 调优方向

⛔ **进入门控**：进入 ITER 循环前，回复中必须已包含以下产出物（缺失则回退补充）：
- S-1：analyze_core_usage.py 的输出结果（每个 leafHash 的核使用率 + FULL/NOT FULL 判定）
- S-2：每个 NOT FULL 子图的 TileShape 调整记录（调整前后核使用率对比）
- S-3：按 total(us) 降序排列的负载均衡分析（瓶颈子图识别 + 差距量化）
- **调优点清单：对照 [shared/optimization_catalog.md](../shared/optimization_catalog.md) 的 S-1~S-21 全表逐项标记**，每项标记为 ✅已尝试 / ❌已失败 / ❌不适用(须注明原因) / ⏳待尝试。**禁止仅列出"自己觉得有用"的项，必须覆盖全表。** 清单中存在 ⏳待尝试 项时禁止退出本阶段。

⛔ 每次 ITER_START 选择优化点时，必须在回复中写出"选择 [S-X]，依据：[分析结果中的具体数据]"。无法写出依据 → 前置分析未完成 → 回退补充。

### 1. Stitch 调优

Stitch 配置决定了多少个 root function 被同时下发调度。

#### 1.1 配置方法

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)

```

**参考资料**
- [stitch_function_max_num 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-frontend-jit.md)

#### 1.2 参数影响

| 参数值 | 优点 | 缺点 |
|--------|------|------|
| 过小（如 1） | - | 每个任务需同步，调度开销大 |
| 适中（如 128） | 泳道图紧凑，调度开销低 | - |
| 过大（如 512） | 泳道图更紧凑 | 调度耗时增加，workspace 增加 |

#### 1.3 调优建议

在内存资源允许的前提下，逐步增大 Stitch 配置，结合泳道图和端到端总耗时数据调整参数。

### 2. TileShape 深度调优

> **⛔ 执行 TileShape 深度调优时，必须加载 [TileShape 深度调优](references/tileshape-deep-tuning.md) 获取完整指南。**

Matmul TileShape 调优决策树（分支细节与示例见该参考 §1）：M、N 足够大（可同时分满两轴）→ Compute Bound，取 mL1=nL1=128~256，最大化算数强度；M 或 N 一维较小 → Memory Bound，小维不切分（mL1=M / nL1=N），kAL1/kBL1 独立配置让大矩阵驻留 L1、最小化重复载入；M、N 均小而 K 极大 → K 轴分核（`enable_split_k=True`）增加并行度。

### 3. 核使用率分析与负载均衡（合图前置条件）

> **⛔ ⛔ 合图调优前，必须加载 [核使用率分析与负载均衡](references/core-usage-load-balancing.md) 完成核使用率分析。未完成本步骤禁止进入第 4 节合图。**

### 4. 合图调优

> **⛔ ⛔ ⛔ 配置任何合图参数前，必须加载 [合图调优](references/merge-optimization.md) 获取完整指南。合图配置错误是性能退化的最常见原因。**

> **🔥 A5 平台专属 [S-14] Mix合图**：若 `pypto.platform.npuarch == 'DAV_3510'`，CV 间搬运可用 Mix合图走 CV 通路消除（CV 最优配比 1:2）。自动合图（`pypto.experimental.auto_mix_partition(1)`）和手动合图（`sg_set_scope`）是同一功能的两种开关方式，功能完全一致，异常处理（编译超时/退化）方式也完全一致。优先尝试自动合图，按 merge-optimization.md §4.1 关键路径（Step 0→1→2→3→4→5→6）逐步执行，当前开关方式充分调优后未达到预期目标性能或需要最优性能时再切换另一种开关方式。限制条件必须逐项满足。**⚠️ Mix合图首次开启与配套 TileShape（L0 调小 + L1 调大 + loop tile 调小）须作为原子优化点提交**。

> **⛔ 编译超时处理摘要**（完整步骤见 [merge-optimization.md §4.1 Step 2](references/merge-optimization.md)）：⛔ 禁止移除 Mix合图配置，按以下优先级逐个尝试：
> 1. 移除 `debug_options` + 加 `host_options={"compile_monitor_enable": 0}`（最常见原因）
> 2. 回退 `unroll_list` 降档（如 `[8,4,2,1]→[4,2,1]→[2,1]`）
> 3. 回退 `nbuffer` 到 `{-1:1}` 或 `{"DEFAULT": 1}`（nbuffer>1 导致子图膨胀）
> 4. 调小核内 TileShape / loop tile / vec tile（保住 unroll）
>
> 若以上仍超时：自动合图→切换手动合图（`sg_set_scope` 只包裹核心 CV 交替段，缩小 scope）；手动合图→放出部分段（CV 全合→不全合）。详见 merge-optimization.md §4.1 Step 2d。


### 5. 调度策略调优

当上下游子图之间依赖较为简单，或下游子图输入 Tensor 的 L2 命中率较为重要时，推荐使用 L2 亲和调度。

```python
@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})

```

**调优建议**：
- 尝试不同的调度策略，值域范围是[0, 3]

**注意事项**：综合考虑 L2 复用与负载均衡的影响，不同场景的最佳配置策略不同。

**参考资料**
- [device_sched_mode 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-frontend-jit.md)


### 6. Matmul 访存布局优化（L2 命中率优化）

> **在大 shape Matmul 场景下（M、N、K 全部较大），即使 TileShape 配置了推荐值，固定的分核布局可能导致 L2 命中率偏低、MTE2 带宽利用率不足，此时应加载 [Matmul 访存布局优化](references/matmul-l2-layout.md) 获取 L2 命中率优化方法。**

### 7. ooo_sched_mode（A5 平台，CV 交替算子，非通用）

> ⚠️ **非通用优化点**：仅对 A5 平台上具有连续 Cube↔Vector 交替结构的算子（如 attention 类的 Q@K^T→softmax→P@V 模式）有效。纯 Cube 或纯 Vector 算子无收益。

**原理**：`ooo_sched_mode`（out-of-order 调度模式）优化 CV 交替段的指令调度，减少核间同步开销。取值范围为 `{"", "GAPMIN", "HLF"}`，配置在 `pass_options`。

**取值说明**：

| 取值 | 调度策略 |
|---|---|
| `""`（默认） | 基于拓扑序遍历和局部搜索的调度（GapMin 调度 + local-search） |
| `"GAPMIN"` | 仅执行 GapMin 调度，跳过 local-search |
| `"HLF"` | Highest Level First 调度（按任务到汇点最长路径降序排列后做 EFT 插入调度） |

**配置方法**（配置在 `pass_options`）：

```python
@pypto.frontend.jit(
    pass_options={
        "ooo_sched_mode": "HLF",
    },
)
```

**调优建议**：
- 建议在 S-14 Mix合图调优之后尝试本项（ooo_sched_mode 优化 CV 交替段的指令调度，Mix合图建立 CV 通路后效果更显著；S-14 失败回退后也可独立尝试）
- 通常与 [S-16] vf_options 配合使用
- **推荐尝试顺序**：先试 `"GAPMIN"`（仅 GapMin 调度，跳过 local-search，适合 CV 交替结构），再试 `"HLF"`（Highest Level First），最后试默认 `""`。attention 类 CV 交替结构推荐 `"GAPMIN"` 优先
- 三个取值需逐个尝试验证

**参考资料**

- [ooo_sched_mode 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_pass_options.md)


### 8. VF 融合（A5 平台）

> **适用范围**：A5 平台（`pypto.platform.npuarch == 'DAV_3510'`）所有含 Vector 计算的算子通用。VF 融合在 A5 上默认开启，本节是让其效果更好的增强手段。完整指南见 [merge-optimization.md §5](references/merge-optimization.md#5-vf-融合编排原则与增强旋钮)。

VF 融合体系包含五个层面：

| 层面 | 机制 | 适用 | 编排器编号 |
|---|---|---|---|
| 默认功能 | VF 融合自动开启 | A5 通用 | — |
| 代码编排 | 三条编排原则（相同 dst shape / 区间约束 / Reduce 最后 expand 最前） | A5 通用，独立于 mix合图 | [S-17] |
| 编译选项 | `vf_options` | A5 含 Vector 计算的算子通用 | [S-16] |
| 调优模式旋钮 | `sg_set_tunevf_mode` | A5 含 Vector 计算的算子通用 | [S-18] |
| 增强旋钮 | `sg_set_ooo_scope` | ⛔ 仅 mix 子图 | 配合 [S-14] |

**`vf_options` 配置方法**（配置在 `codegen_options`）：

```python
@pypto.frontend.jit(
    codegen_options={
        "vf_options": "-mllvm -cce-vf-enable-vloopv2-recognizer=true -mllvm -enable-pto-colop-fusion=true"
    }
)
```

**`sg_set_tunevf_mode` 配置方法**（控制 VF 调优 Pass 行为模式，默认 `0`，取值 `{0,1,2}`）：

```python
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_tunevf_mode=2)  # vf 融合优先
```

| 取值 | 模式 | 行为 |
|---|---|---|
| `0` | 均衡模式（默认） | 在 OoO Pass 输出的 op 序列基础上自动调整 op 顺序，平衡流水与 VF 融合 |
| `1` | 指令流水优先 | 不改变 OoO 排好的 op 执行序 |
| `2` | vf 融合优先 | 不考虑性能建模收益评估，尽量调整 op 顺序保证更大范围 VF 融合 |

**调优建议**：
- 三条编排原则（[S-17]）让融合候选更多，`vf_options`（[S-16]）让融合生成代码更优，两者互补可叠加收益
- `sg_set_tunevf_mode`（[S-18]）控制 Pass 调整 op 序列的激进程度；编排原则就位后再用 `=2` 收益更明显，流水退化则回退 `=1`
- `sg_set_ooo_scope` 须在 [S-14] Mix合图段内使用，独立使用无效

**参考资料**

- [sg_set_tunevf_mode 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_pass_options.md)

### 9. ready_on_host_tensors（减少 host 下发开销）

**适用场景**：算子有小 tensor 输入通过 AICPU gather 下发（如 paged KV cache 类算子的 block_table、actual_seq 等索引类 tensor），且 wall time 远大于 AICore E2E Time。

**原理**：部分 tensor（如 block_table、kv_act_seqs）数据量小但需要 AICPU 读取后下发给核。默认流程中 AICPU 需等待这些 tensor 从 device 拷回 host 才能读取，标记为 host ready 后跳过该等待。

**配置方法**（配置在 `runtime_options`）：

```python
@pypto.frontend.jit(
    runtime_options={
        "ready_on_host_tensors": ["block_table", "kv_act_seqs"],
    }
)
```

**诊断方法**：
1. 对比 AICore E2E Time 与 wall time，若 wall time >> AICore E2E Time，说明 host 下发开销主导
2. 检查算子输入中是否有小 tensor（索引表、序列长度等）通过 AICPU gather 下发
3. 将这些 tensor 名称加入 `ready_on_host_tensors` 列表

**调优建议**：
- 该优化减少 wall time 中的 host 下发开销，不影响 AICore E2E Time
- 当 wall time 接近 AICore E2E Time 时，该优化无收益
- tensor 名称须与函数参数名一致


### 10. max_workspace_kb + host_options 完整性检查（对应优化点 S-20）

> **⛔ 强制前置**：所有算子在 S2_COLLECT 阶段首次运行时，必须检查 NPU 编译输出中的 workspace 推荐提示。

**原理**：NPU 编译器在编译输出中会给出 `max_workspace_kb` 推荐值（如 "Recommended: set max_workspace_kb near 1607648KB"）。未设置时编译器使用默认 workspace，可能无法激活 memory-driven mode，影响调度优化。

**检查方法**：
1. 在 S2_COLLECT 阶段首次运行算子时，检查 stdout 中是否包含 `Recommended: set max_workspace_kb` 提示
2. 若有提示，提取推荐值并设置到 `runtime_options`
3. 同时检查是否设置了 `host_options={"compile_monitor_enable": 0}` 减少编译监控开销

**配置方法**：

```python
@pypto.frontend.jit(
    runtime_options={
        "max_workspace_kb": 1607648,  # 从 NPU 输出推荐值获取
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    },
    host_options={"compile_monitor_enable": 0},
)
```

**⚠️ NPU 输出示例**：
```
Recommended: set max_workspace_kb near 531052KB and above 10380KB to activate memory-driven mode.
```
→ 从 "531052KB" 提取值 531052，设置为 `"max_workspace_kb": 531052`

**调优建议**：
- 该优化影响编译器的调度策略，应作为基础配置在调优早期设置
- workspace 过大可能占用过多显存，按推荐值设置即可
- `host_options` 不影响算子计算性能，仅减少编译开销


### 11. Mix合图多 scope 策略（对应优化点 S-21）

> **适用场景**：A5 平台 + 算子 loop 体内最后一个 Cube 之后的 V 段涉及跨迭代依赖（需从 Mix scope 放出），或有多段无数据依赖的 CV 段可分别独立合图。

**原理**：Mix scope 内的 CV 交替数据应通过 CV 通路直接传递（不走 DDR）。但如果 Mix scope 内包含写跨迭代依赖 tensor 的 V 段（该 tensor 在 loop 外声明、loop 内被写入，其值从上一轮迭代传递到下一轮），会打断 CV 通路的数据流。这类 V 段应从 Mix scope 中放出，用独立 `sg_set_scope` 做普通合图，消除子图间调度开销。**多 scope 策略**包含两种模式：

#### scope 划分条件

**Step 1: 识别跨迭代依赖**

扫描 loop 体内的操作，找满足以下条件的 tensor：
- 在 loop 外声明
- 在 loop 内被写入（作为任意操作的输出）
- 在 loop 内被读取（作为后续操作的输入）

这类 tensor 的值从上一轮迭代传递到下一轮，构成跨迭代依赖。

> 在 loop 内声明并使用的 tensor 不构成跨迭代依赖（生命周期仅限单轮）。在 loop 外声明但 loop 内只读不写的 tensor 不构成跨迭代依赖（如权重）。

**Step 2: 判断最后一个 Cube 之后的 V 段是否涉及跨迭代依赖**

在 loop 体内从后往前扫描，找到最后一个 Cube 操作（`pypto.matmul`）。该 matmul 之后的连续 V 操作段为"最后一段 V"：
- 该 V 段中存在写入跨迭代依赖 tensor 的操作 → 需要切 scope
- 该 V 段中不存在此类操作 → 不切 scope

**Step 3: 按情况处理**

- **不涉及跨迭代依赖** → 不切 scope，整个 loop 体留在 Mix scope（切出只增加 GM 搬运代价，无收益）
- **涉及跨迭代依赖** → 切 scope，Mix scope 包含到最后一个 Cube（含），V 段用独立普通合图 scope 包裹。切后与非切实测对比，保留性能更优的：
  - 编译超时 → 缩小 scope / 调小 TileShape 后重试
  - 两者都可行 → 保留性能更优的（V 段较短时非切可能更优，GM 代价小于调度收益；V 段较长时切通常更优）

**注意事项**：
- scope 划分前先确认代码无冗余计算（如单 tile 路径重复 matmul），冗余计算是算法级问题，应先消除再做 scope 优化
- 条件分支内如有 Cube 操作，scope 关闭位置应放在分支内部，让 Cube 留在 Mix scope，不要包进普通合图

**策略1：Mix scope 放出的 V 段用独立 sg_set_scope 做普通合图**

放出的 V 段应立即用独立 `sg_set_scope` 做普通合图，消除子图间调度开销。

```
Mix scope (ID 例如 20001): V0→C1→V1→C2 (CV 链，走 CV 通路)
  sg_set_scope=20001
  ... V0 gather + dequant + C1 matmul + V1 softmax + C2 matmul ...
  sg_set_scope=-1

普通合图 scope (ID 例如 1): V2 (online softmax update，纯 Vector)
  sg_set_scope=1
  ... V2: max/sum/exp/oi_update operations ...
  sg_set_scope=-1
```

```python
# ===== 策略1：Mix scope + 放出V段普通合图 =====
# Mix合图 scope: V0→C1→V1→C2
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=20001)

# ... V0: paged gather + dequant ...
# ... C1: QK^T matmul ...
# ... V1: per-block softmax ...
# ... C2: PV matmul ...

if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=-1)

# 普通合图 scope: V2 online softmax update
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=1)

# ... V2: max/sum/exp/oi_update operations ...

if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=-1)
```

**策略2：多段无数据依赖的 CV 段分为多段独立 Mix scope**

当算子有多段 CV 交替段且段间无数据依赖时，将每段分别用独立 Mix scope 包裹，提升编译器调度灵活性。

```
Mix scope 1 (ID 例如 10001): V10→C11→V11→C12 (第1段 CV 链)
  sg_set_scope=10001
  ... V10→C11→V11→C12 ...
  sg_set_scope=-1

Mix scope 2 (ID 例如 10002): V20→C21→V21→C22 (第2段 CV 链，与第1段无数据依赖)
  sg_set_scope=10002
  ... V20→C21→V21→C22 ...
  sg_set_scope=-1

普通合图 scope (ID 例如 3): 两段生成数据的后续计算
  sg_set_scope=3
  ... 合并/归约操作 ...
  sg_set_scope=-1
```

```python
# ===== 策略2：多段独立 Mix scope =====
# 第1段 CV 交替段
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=10001)
# ... V10→C11→V11→C12 ...
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=-1)

# 第2段 CV 交替段（与第1段无数据依赖）
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=10002)
# ... V20→C21→V21→C22 ...
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=-1)

# 两段生成数据的后续计算
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=3)
# ... 合并/归约操作 ...
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=-1)
```

**scope ID 命名规则**：
- scope ID 无功能差异，仅作唯一标志，每个 scope 使用不重复的正整数即可；示例中 Mix scope 用大 ID、普通合图用小 ID 仅为便于阅读（merge-optimization.md §4.1 已声明 ID 无特殊语义）
- 每个 scope 使用不同的正整数 ID

**⚠️ 适用条件**：
- 最后一个 Cube 之后的 V 段涉及跨迭代依赖（V 段中写入在 loop 外声明、loop 内读写的 tensor）
- V 段是纯 Vector 操作（无 Cube 夹杂）

> 完整的 Step 0b 规则3 scope 布局方法详见 [merge-optimization.md §4.1 Step 0b](references/merge-optimization.md)。


## 调优检查清单

**⛔ 必须按以下清单逐项执行。每项标记为 ✅已尝试 或 ❌已失败（附原因），禁止跳过。完整优化点信息参考 [shared/optimization_catalog.md](../shared/optimization_catalog.md)。**

**优化优先级**：
1. ⭐⭐⭐ **P0 - 核使用率分析 + 核填充** → 详见 [S-1][S-2]
2. ⭐⭐⭐ **P1 - 负载均衡** → 详见 [S-3]
3. ⭐⭐ **P2 - TileShape 深度调优 + 访存布局优化** → 详见 [S-11][S-12][S-13]
4. ⭐⭐⭐ **P2 - 普通合图（sg_set_scope，只包裹 V 段）** → 详见 [S-4]
5. ⭐⭐⭐ **P2+ - A5 Mix合图（包裹 CV 段，仅 A5 平台 `npuarch=='DAV_3510'`）** → 详见 [S-14]
6. ⭐⭐ **P2 - ooo_sched_mode + VF融合 + ready_on_host_tensors** → 详见 [S-15][S-16][S-17][S-18][S-19]
7. ⭐⭐ **P3 - 自动合图** → 详见 [S-5][S-6][S-7][S-8]
8. ⭐⭐ **P4 - Stitch + 调度策略** → 详见 [S-9][S-10]

**🔥 P0 - 核使用率分析 [S-1]**（合图前置条件）：
- [ ] [S-1] 是否运行 analyze_core_usage.py 统计每个 leafHash 核使用率
- [ ] [S-2] 每个 NOT FULL 子图是否已尝试 TileShape 调整（减小 L0/L1 增加任务数）
- [ ] 是否尝试完所有轴的 TileShape 调整后才判定"核无法再增"

**🔥 P1 - 负载均衡 [S-3]**（核填充后强制）：
- [ ] [S-3] 是否按 total(us) 降序排列所有子图，识别瓶颈子图
- [ ] 瓶颈差距是否已量化（>20% 必须优化）
- [ ] [S-11] 是否针对瓶颈子图尝试了 Cube TileShape 深度调优（每次只调一个子图）
- [ ] [S-12] 是否对 Vector 计算尝试了 Vector TileShape 深度调优（调整 TileShape 对齐上下游）
- [ ] [S-13] 大 shape Matmul 是否检查了 MTE2 带宽利用率并尝试了分核布局优化

**🔥 P2 - 手动合图 [S-4]**（最重要但最易跳过）：
- [ ] [S-4] 是否运行 analyze_aiv_dep_chains.py 分析 AIV 依赖链
- [ ] 是否检查了可合并的连续 Vector 操作（有直接数据依赖、同循环层级、无 Cube 夹杂）
- [ ] 是否对每个可合并链段尝试了 sg_set_scope
- [ ] 如果跳过此项，是否说明了具体原因（而非"觉得不适用"）

**🔥 P2+ - A5 Mix合图 [S-14]**（A5 平台 + CV 交替算子，仅 `npuarch=='DAV_3510'`）：

> **⛔ 调优Mix合图时必须严格按 [merge-optimization.md §4.1 关键路径](references/merge-optimization.md) 的 Step 0→1→2→3→4→5→6 顺序执行。⛔ 自动合图和手动合图是两条独立调试线，各自必须完整走 Step 0→6。下方检查清单仅用于最终核查，不可替代关键路径的顺序执行。特别是：Step 3c（8项配套参数逐个调优，全部标记✅或❌后才允许退出）和 Step 4（调整scope框架，⛔ 3c调完后禁止跳过到Step 5）是最容易遗漏的步骤。**
- [ ] [S-14] 是否确认平台为 A5（`pypto.platform.npuarch == 'DAV_3510'`）
- [ ] 是否确认算子存在 CV 交替结构（纯 Cube/Vector 算子不适用 Mix合图，跳过本节）
- [ ] ⛔ 进入 Mix合图前必须先完成 Step 0 数据流分析（0a: CV 数据流表 → 0b: scope 布局方案 → 0c: 配套参数推荐值 → 0d: 分析结论），Step 1 的原子优化点必须基于 Step 0 结论
- [ ] 选择一种 Mix合图开关方式（优先尝试 `pypto.experimental.auto_mix_partition(1)`，或 `sg_set_scope`），配合 Step 0c 推荐的配套参数 + nbuffer 默认 1:1，作为**原子优化点**一次性提交实测性能（详见 merge-optimization.md §4.1 关键路径 Step 1）
- [ ] 达标 → 完成；⛔ 编译超时 → 按关键路径 Step 2（2a→2d 顺序）处理，**禁止移除 Mix合图**；⛔ 退化 → **禁止跳过分析直接回退**，按关键路径 Step 3 阶段 A 先分析退化原因（3a: Q1 CV通路 / Q2 spill / Q3 退化因素 → 3b: 在当前 scope 内调参 → 3c: ⛔ 8项配套参数全部逐个调优并标记✅/❌），3c 调完后 → **⛔ 无论"仍退化"还是"有提升但未达标"都必须进入 Step 4**（调整 scope 框架：全合↔不全合），Step 4 仍无收益 → Step 5 切换另一种开关方式（自动↔手动），**⛔ 切换后等同于Mix合图重新开始：nbuffer重置为1，必须重走Step 3→4完整流程，禁止切换后直接判断退化就回退**
- [ ] 是否检查 CV 间数据传递为 1:N 或 N:1（不支持 M:N 多对多，否则走 DDR）；消费者约束：一个 matmul 结果只喂一条 vector 链，同一 L0C_COPY_UB 的 vector 消费者须在同一 AIV 核
- [ ] 是否检查 CV 间 shape 变化单调（无交叉大小，如 [64,128]→[128,64] 禁止）
- [ ] ⛔ 是否检查算子衔接形态：matmul 直接接 vector（中间无其他 Cube）；最终 TileGraph 是否匹配直连识别模式（小块→大块 ASSEMBLE 汇聚 / 大块→小块 VIEW 拆分）；框架自动插入或合法手写的 assemble/view 均可
- [ ] ⛔ 是否检查 shape/tile 硬数值约束：衔接 tensor 须 2D；L0C→UB vec tile 两维 16 对齐；cube tile 与 vec tile 衔接轴相等或整数倍；UB→L1 内轴切分 32B 对齐；assemble 场景输出 ≤ UB×0.35
- [ ] 是否配置 `vec_nbuffer_setting={-1:1}` + `cube_nbuffer_setting={-1:1}`（后者无效试 `cube_l1_reuse_setting={-1:1}`）
- [ ] 是否重新调整unroll大小，如最内层loop尝试unroll更多子图或全unroll减少调度开销
- [ ] Mix合图生效后是否同步调整核内 TileShape：**L0 调小**（增加核内并行度）+ **L1 调大**（减少重复搬运）+ **loop tile 调小**（增加 task 数弥补 Mix 串行化导致的并行度损失），三者作为一组同时调整。⚠️ 注意区分核内 TileShape（`set_cube_tile_shapes`/`set_vec_tile_shapes`）与 loop tile（如 s2_tile），方向不同需独立权衡
- [ ] ⛔ 是否检查 UB 使用：单个 tensor 的ND+NZ的总大小须 < 248KB
- [ ] ⛔ 是否验证 CV 通路生效：解析 program.json，追踪 CV 间数据流向——识别 CV 间应传递的数据（通过 semantic_label 定位），检查每个数据走 CV 通路 opcode（✅）还是 DDR 中转（COPY_OUT 后紧接 COPY_IN 无 CV_SYNC 邻居 ❌）。⛔ 不能只看 opcode 是否存在，须确认所有 CV 间数据都走 CV 通路。有 DDR 中转 → 排查 UB 超限 / shape 非单调 / tensor 生命周期过长
- [ ] 检查泳道图中 spill（WorkspaceGm）数量是否 ≤ 20（经验阈值），超过则调小 TileShape或调小 nbuffer 减少 spill
- [ ] ⛔ **nbuffer 调优流程（必做，不可跳过）**：①初始值设为1（`vec_nbuffer:{"DEFAULT":1}` + `cube_nbuffer:{-1:1}` + `cube_l1_reuse:{-1:1}`）→ ②在nbuffer=1基线上完成其他参数调优（TileShape/unroll/ooo_sched_mode等）→ ③从1开始按2的幂次递增逐值实测vec_nbuffer（1→2→4→8→16→32…，1/2/4/8是常用范围不是上限，8有收益继续试16/32，劣化则回退不再增大）→ ④在vec_nbuffer最优值基础上同理逐值实测cube_nbuffer和cube_l1_reuse。详见 catalog S-14 nbuffer调优流程
- [ ] 首次配置（原子优化点）后若性能劣化被回退，后续每轮 ITER_MODIFY 是否以"Mix合图 + 配套 TileShape"为基底叠加新参数试错（而非在无 Mix合图基础上试其他参数），直到组合生效或确认所有配套手段均无收益才彻底回退 Mix合图
- [ ] ⛔ 是否以 AICore E2E Time（核上 compute）而非 wall time / device time 判断优化效果——Mix合图可能降低 wall time 但增加核上 compute，须以 AICore E2E Time 下降为准
- [ ] 若 Mix scope 放出了 V 段（涉及跨迭代依赖的段），是否对放出的 V 段用独立 sg_set_scope 做了普通合图（参见 [S-21] §11）
- [ ] ⛔ CV 全合和 CV 不全合都是合法调优路径。选定一个 scope 范围后，必须先在该范围内充分调参（配合调小 TileShape、对比 nbuffer、排查 DDR 回退、调整 unroll），所有参数都调完仍退化才切换 scope 范围（如 CV 全合→CV 不全合），在新范围内重新充分调参
- [ ] ⛔ 多次尝试仍无收益时，是否按 [merge-optimization.md §4.7](references/merge-optimization.md)「Mix合图失败诊断子流程」执行诊断——从 DDR 回退追溯到代码具体行，区分不可修复约束 vs 可修复代码结构，对可修复断点执行代码结构改造（合并 gather / view 复用 / 消除中间 assemble），修复后重试 Mix合图

**P2 - VF 融合编排原则 [S-17]**（A5 平台，含 Vector 计算的算子通用，独立于 mix合图）：
- [ ] [S-17] 连续 vec 算子序列是否遵循三条编排原则（相同 dst shape / 区间约束同区间或无 overlap / Reduce 最后 expand 最前），详见 merge-optimization.md §5.2

**P2 - VF 调优 Pass 行为模式 [S-18]**（A5 平台，含 Vector 计算的算子通用，独立于 mix合图）：
- [ ] [S-18] 是否在编排原则就位后尝试 `sg_set_tunevf_mode`（默认 `0` 均衡；融合收益不足试 `2` 融合优先；流水退化回退 `1` 流水优先），详见 merge-optimization.md §5.4

**P2+ - vf 融合增强旋钮 `sg_set_ooo_scope`**（A5 平台，⛔ 仅对 mix 子图生效，须在 mix合图段内使用）：
- [ ] 是否用 `sg_set_ooo_scope` 包裹需融合的连续 vec 算子段，避免被搬运/同步指令打断融合调度（开关旋钮，详见 merge-optimization.md §5.3）

**P3 - 自动合图 [S-5~S-8]**：
- [ ] [S-5] 是否对短耗时（<10us）AIV 任务尝试了 vec_nbuffer_setting
- [ ] [S-6] 是否对核满的 AIC 子图尝试了 cube_l1_reuse_setting
- [ ] [S-7] 是否对核满的 AIC 子图尝试了 cube_nbuffer_setting
- [ ] [S-8] 如已配置 S-6/S-7，是否检查了协同使用是否过大

**P2 - ooo_sched_mode [S-15] + VF融合 [S-16~S-18]**：
- [ ] [S-15] A5 平台 + CV 交替结构，是否尝试了 ooo_sched_mode（先试 GAPMIN，再试 HLF）
- [ ] [S-16] A5 平台 + 含 Vector 计算的算子，是否尝试了 vf_options（codegen_options）
- [ ] [S-17] 连续 vec 算子序列是否遵循三条编排原则（相同 dst shape / 区间约束 / Reduce 最后 expand 最前）
- [ ] [S-18] 是否在编排原则就位后尝试了 sg_set_tunevf_mode（默认 0；融合不足试 2；流水退化回退 1）

**P2 - ready_on_host_tensors [S-19]**：
- [ ] [S-19] 若 wall time >> AICore E2E Time，是否检查了小 tensor 输入并配置了 ready_on_host_tensors

**P4 - Stitch [S-9] + 调度策略 [S-10]**：
- [ ] [S-9] 是否尝试了 stitch_function_max_num 调整
- [ ] [S-10] 是否尝试了 device_sched_mode 调整（1/2/3）

**🔥 P0 - max_workspace_kb + host_options [S-20]**（强制前置，所有算子）：
- [ ] [S-20] 是否检查了 NPU 编译输出中的 "Recommended: set max_workspace_kb" 提示
- [ ] 是否按推荐值设置了 max_workspace_kb
- [ ] 是否设置了 host_options={"compile_monitor_enable": 0}

**🔥 P2+ - Mix合图多 scope 策略 [S-21]**（A5 平台 + 最后一个 Cube 后的 V 段涉及跨迭代依赖，或多段无数据依赖的 CV 段）：
- [ ] [S-21] 是否按 §11 的 scope 划分条件诊断：识别跨迭代依赖 tensor → 判断最后一段 V 是否涉及 → 决定是否切 scope
- [ ] 是否对放出的 V 段用独立 sg_set_scope 做了普通合图
- [ ] 切 scope 后是否与非切实测对比，保留性能更优的
- [ ] scope ID 是否互不重复（Mix 与普通 scope 使用不同正整数即可，大小无特殊语义）


## 常见问题

> 遇到以下问题时，加载 [常见问题](references/faq.md) 获取详细解答：
> - Q1: 泳道图文件在哪里
> - Q2: 如何查看性能统计
> - Q3: 气泡是什么
> - Q4: 控制开销占比过高怎么办
> - Q5: 如何选择合适的 Tilesize


## 参考资料

- [性能调优文档](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/performance.md)
- [Matmul 高性能编程](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/matmul_performance_guide.md)
- 算子参考实现案例（attention 类等）：用 `pypto-docs-search` 搜索算子参考实现
- [性能优化案例](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/performance_case_quantindexerprolog.md)
