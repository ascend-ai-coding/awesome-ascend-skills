---
name: external-cannbot-ops-pypto-op-perf-tune-tune-swimlane
description: PyPTO 算子深度性能调优技能。通过泳道图分析及调优性能，包括 Stitch 调优、TileShape 深度调优、合图调优、调度策略调优等。当用户需要进行深度性能调优、泳道图分析、Stitch
  优化、合图优化时使用此技能。触发词：深度性能调优、泳道图分析、Stitch 调优、合图调优、调度优化。
original-name: tune-swimlane
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-06-14'
synced-commit: f2b0db65abdced502b1cd1956e2e48e6e0197add
license: UNKNOWN
---

# PyPTO 算子深度性能调优

## 概述

深度性能调优通过泳道图分析及调优性能，采用 man-in-loop 的方式，通过获取并分析当前算子性能数据，针对性调整各性能配置参数，经过迭代调优逐步逼近最佳性能。

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

### 1. Stitch 调优

Stitch 配置决定了多少个 root function 被同时下发调度。

#### 1.1 配置方法

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)

```

**参考资料**
- [stitch_function_max_num 参数设置说明](../../../../docs/zh/api/config/pypto-frontend-jit.md)

#### 1.2 参数影响

| 参数值 | 优点 | 缺点 |
|--------|------|------|
| 过小（如 1） | - | 每个任务需同步，调度开销大 |
| 适中（如 128） | 泳道图紧凑，调度开销低 | - |
| 过大（如 512） | 泳道图更紧凑 | 调度耗时增加，workspace 增加 |

#### 1.3 调优建议

在内存资源允许的前提下，逐步增大 Stitch 配置，结合泳道图和端到端总耗时数据调整参数。

### 2. TileShape 深度调优

#### 2.1 Matmul TileShape 深度调优

Matmul 的 TileShape 深度调优需要根据 M、N、K 的实际大小和硬件规格综合选择策略。核心目标是：保证分满核的前提下，最大化算数强度 / L2 命中率，最小化重复载入。

**调优决策树**：

```

M、N 是否足够大（可同时分满 M、N 轴）？
├─ 是（训练场景，如 M=N=6144）
│   ├─ 目标 → Compute Bound（算数强度最大化）
│   ├─ TileShape → mL1=nL1=128~256，如 [128,128],[64,256],[256,256]
│   └─ 额外优化 → 分核布局调优提升 L2 命中率（→ §6）
│
├─ M 或 N 中有一维较小（推理/Decode 场景）
│   ├─ 目标 → Memory Bound（减少重复载入）
│   ├─ 策略 → 小维不切分（mL1=M 或 nL1=N），kAL1/kBL1 独立配置
│   └─ 示例 → [96,96],[64,1536,256],[128,128]（A 矩阵驻留 L1）
│
└─ M、N 均小，K 极大
    ├─ 目标 → 增加并行度
    ├─ 策略 → K 轴分核（enable_split_k=True）
    └─ 示例 → [128,128],[64,256],[128,128], enable_split_k=True

```

##### 2.1.1 减少重复载入

当 M、N 中有某个维度相对较小时，Matmul 切分前的算数强度较小，一般只能达到 Memory Bound。此时优化核心是：保证分满核（核使用率 ≥ 80%）的前提下，最小化 MTE2 重复载入量。

**方式一：增大 L1 减少切分次数**

```python
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 256])

```

每一个 list 中前面的代表 L0 的切块大小，后面的代表 L1 的切块大小。L1 设大可以减少切分次数，从而减少重复载入。

**方式二：K 轴独立配置，A 矩阵驻留 L1（推荐）**

```python
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kAL1, kBL1], [nL0, nL1])

```

- `kAL1 = K`：A 矩阵的 K 轴整体搬入 L1 并驻留，消除 A 的重复搬运
- `kBL1 = 256`：B 矩阵的 K 轴按 256 分批载入

示例（M=96, K=1536, N=3072, FP16）：

```python
pypto.set_cube_tile_shapes([96, 96], [64, 1536, 256], [128, 128])

```

**理论依据**：MTE2 总载入量 = M·K·N/nL1·aByte + K·N·M/mL1·bByte。当 A 矩阵驻留 L1 时，载入量降为 M·K·aByte + K·N·bByte，消除 A 矩阵的重复搬运。

##### 2.1.2 K 轴分核

当 M、N 较小而 K 轴极大时，仅在 M、N 轴做分核可能无法用满核。此时将 K 轴切分到多核并行计算部分和再累加。提供两种方式：

**方式一：自动切 K（`enable_split_k=True`）**

框架自动将 K 轴切分到多核并行，使用简单，但部分和累加顺序不定导致**非确定性计算**，逐次运行结果可能存在微小差异。

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 128],
    enable_split_k=True)

```

**约束**：
- 仅支持 2 维矩阵（3 维/4 维不支持）
- 仅支持 out_dtype 为 DT_FP32 或 DT_INT32
- 不支持叠加 Bias、FixPipe 反量化

**方式二：手动切 K（前端 loop 实现）**

在前端手动编写 K 轴循环，显式控制分块加载和部分和累加，累加顺序确定，**不存在确定性计算问题**。

```python
c = pypto.Tensor([M, N], pypto.DT_FP32)
c.fill(0.0)
for kIdx in pypto.loop(K // kL1, name="LOOP_K", idx_name="kIdx"):
    a_block = pypto.view(a, [M, kL1], [0, kIdx * kL1])
    b_block = pypto.view(b, [kL1, N], [kIdx * kL1, 0])
    pypto.set_cube_tile_shapes([mL1, mL1], [64, kL1], [nL1, nL1])
    c_partial = pypto.matmul(a_block, b_block, out_dtype=pypto.DT_FP32)
    c = c + c_partial

```

**选择建议**：精度敏感场景优先使用方式二（手动切 K）；追求简便且接受微小精度波动的场景使用方式一（`enable_split_k=True`）。

##### 2.1.3 TileShape 选择策略速查

| Shape 特征 | 典型场景 | TileShape 策略 | 收益目标 |
|-----------|---------|---------------|---------|
| M,N,K 均大（>4096） | 大 Batch 训练 | [128,128],[64,256],[256,256] + 分核布局优化（见 §6） | Compute Bound + L2 调优 |
| M=N=6144 | 大 Shape Matmul | mL1=128,nL1=256,mDim=6,nDim=4 | 2.1ms→1.6ms（+31%） |
| M<256，N,K 大 | Decode/推理 | mL1=M, kAL1=K, kBL1=256, nL1≥128 | 消除 A 矩阵重复载入 |
| M=96,K=1536,N=3072 | 典型推理 | [96,96],[64,1536,256],[128,128] | A 矩阵驻留 L1 |
| M,N 小，K>4096 | 特殊 Shape | enable_split_k=True | K 轴多核并行 |
| M 极大，N 极小 | 小 Shape Matmul | 优先走 I-1（Vector 预处理），再按上述策略 | 见核内调优 |

#### 2.2 Vector TileShape 深度调优

**原则 1**：下游 Vector Operation 的 TileShape 尽可能使用上游 Operation 的输出 TileShape

```python
# 上下游 TileShape 对齐时，可以合并在一个子图
# Transpose TileShape: (64, 128)
# Add TileShape 应优先选择: (128, 64)

```

**原则 2**：根据泳道图上的子图大小和并行度调整

- 并行核数较少（<一半 Vector 核）：减小 TileShape
- 子图耗时短、调度开销占比较高：增大 TileShape

**原则 3**：调整相邻 Cube 和 Vector Operation 的 TileShape，使依赖更简单


### 3. 核使用率分析与负载均衡（合图前置条件）

**⛔ 重要：合图调优前，必须先完成核使用率分析。核未用满时优先用满核，再考虑合图。**

合图（L1reuse / CubeNBuffer / VecNBuffer）的前提是核已用满。如果核没用满就合图，会将本就不多的任务进一步合并到更少的核上，反而降低并行度、导致性能退化。

#### 3.0 核使用率分析流程

```

采集泳道图数据（debug_options={"runtime_debug_mode": 1}）
    ↓
统计每个 leafHash 分布在多少个 core 上
    ↓
对每个 AIC/AIV leafHash，判断核是否用满
     ├─ 核未满 → 优先通过 TileShape 调整用满核（跳到 3.2）
    └─ 核已满 → 进入合图阶段（跳到第 4 节）

```

#### 3.1 统计核使用率

从 `merged_swimlane.json` 的 `traceEvents` 中解析每个 leafHash 占用的 core 数量，并与芯片理论核数对比：

> **路径说明**：以下脚本命令在技能目录 `pypto/.agents/skills/pypto-op-perf-tune/tune-swimlane/` 下执行

```bash
python3 scripts/analyze_core_usage.py <output_dir> [--device-id N]

```

**参数说明**：
- `output_dir`：泳道图数据目录（含 `merged_swimlane.json`）
- `--device-id`：NPU 设备号，用于通过 `torch.npu.get_device_properties()` 查询理论核数（默认 0）

**输出示例**：

```

Theoretical cores: AIC=24, AIV=48

psgId | type | tasks |    used/total (usage%) |  avg(us) | total(us) |    status | suggestion
---------------------------------------------------------------------------------------------
   11 |  AIC |     8 |             8/24 (33%) |     43.4 |     347.5 |  NOT FULL | FILL CORES FIRST (reduce TileShape)
    3 |  AIC |    16 |            16/24 (67%) |      7.4 |     118.1 |  NOT FULL | FILL CORES FIRST (reduce TileShape)
    8 |  AIC |     8 |             8/24 (33%) |      8.1 |      64.9 |  NOT FULL | FILL CORES FIRST (reduce TileShape)
    6 |  AIV |     4 |              4/48 (8%) |      6.4 |      25.5 |  NOT FULL | FILL CORES FIRST (reduce TileShape)

Summary:	 
    NOT FULL (fill cores first): 4 leafHash(es)	 
    FULL (can merge): 0 leafHash(es)	 

Next step: For NOT FULL leafHashes, use leafhash_to_code.py to map to frontend code,
           then adjust set_cube_tile_shapes() to increase task count.

```

**关键指标说明**：
- **理论核数**：通过 `torch.npu.get_device_properties(device_id)` 获取 `cube_core_num`（AIC）和 `vector_core_num`（AIV），这是芯片硬件层面的核数，不随任务数变化。底层对应 `platform.h` 中 `SoC::GetAICCoreNum()` / `SoC::GetAIVCoreNum()`。
- **实际使用核数**：从 `merged_swimlane.json` 的 `traceEvents` 中统计每个 leafHash 实际被分配到的 core 数量。框架可能因任务数不足而未用满所有核。
- **cores 列**：`实际使用核数/理论核数 (使用率%)`
- **FULL**：实际使用核数 ≥ 理论核数，可进入合图阶段
- **NOT FULL**：实际使用核数 < 理论核数，优先通过 TileShape 调整增加任务数用满核

#### 3.2 核未满时的优化策略

**核心思路**：核未满说明任务数不够，需要通过减小 TileShape 来增加任务数，让更多核参与计算。

**操作步骤**：

1. 运行 `leafhash_to_code.py` 将 leafHash 映射到前端代码行：

```bash
python3 scripts/leafhash_to_code.py <output_dir>

```

2. 定位到对应的 `pypto.set_cube_tile_shapes()` 调用

3. 减小 nL0/nL1（N 轴切块大小）以增加 N 轴任务数：

```python
# 原来：N=4096, nL0=256, nL1=256 → 4096/256=16 个 N 轴任务，但 M=1 → 总共 16 个任务
pypto.set_cube_tile_shapes([16, 16], [128, 256], [256, 256])

# 减小 nL0/nL1：4096/128=32 个 N 轴任务 → 可分配到更多核
pypto.set_cube_tile_shapes([16, 16], [128, 256], [128, 128])

```

4. **约束**：`nL0 <= nL1 && nL1 % nL0 == 0`，否则编译报错

5. **负载均衡注意**：
   - 任务数增加 ≠ 性能提升。任务过小（<10us）时调度开销占比增大，反而退化
   - 需要实测验证，找到任务数和单任务耗时的平衡点
   - 建议逐步减小（如 256→128→64），每步实测

**⛔ 禁止以结构限制为由跳过核填充**：即使某轴（如 M 轴）因语义固定（如 decode 阶段 M=1），其他轴的 TileShape 仍可调整来增加任务数和核心利用率。必须尝试完所有轴的 TileShape 调整（减小 L0/L1 增加任务数）后，才能判定"核无法再增"并进入合图阶段。禁止仅凭某轴固定就跳过核填充。

#### 3.3 负载均衡优化（核填充后的强制步骤）

**⛔ 完成核填充后、进入合图前，必须执行负载均衡分析。禁止跳过。**

核填充只保证单个子图的核心利用率，但不保证多个子图之间的负载均衡。如果某个子图耗时远大于其他子图，即使其他子图核心利用率 100%，整体执行时间仍受限于瓶颈子图。

**负载均衡分析流程**：

```

Step 1: 按总耗时排序所有子图
    → 使用 analyze_core_usage.py 输出，按 total(us) 降序排列
    → 识别瓶颈子图（total 最大的子图）

Step 2: 量化瓶颈差距
    → 理想情况：所有子图 total 趋于一致
    → 计算：(最大total - 次大total) / 最大total × 100%
    → 差距 > 20% → 存在严重负载不均，必须优化

Step 3: 针对瓶颈子图优化 TileShape
    → 瓶颈子图的核心利用率是否还有提升空间？
      - 是 → 继续减小 TileShape L0/L1 增加任务数
      - 否 → 尝试调整非瓶颈子图的 TileShape（增大 L0/L1 减少任务数），为瓶颈子图腾出 L1/UB 资源
    → 每次只调整一个子图的 TileShape

Step 4: 重新采集数据验证
    → 调整后必须重新采集泳道图数据（debug_options={"runtime_debug_mode": 1} → 运行 → debug_options={"runtime_debug_mode": 0}）
    → 对比瓶颈子图 total 是否下降
    → 如果瓶颈转移（新的子图变成最大 total），回到 Step 1

Step 5: 负载均衡达标后进入合图阶段
    → 条件：瓶颈子图 total 与次大 total 差距 < 20%，或连续 3 轮调整无法改善

```

**负载均衡优化原则**：

1. **优化瓶颈，不优化平均**：目标不是让所有子图核使用率一致，而是让最大 total 的子图执行时间最短
2. **单子图单次调整**：每次只改一个子图的 TileShape，重新采集数据对比，禁止同时改多个子图
3. **资源竞争意识**：减小非瓶颈子图的 TileShape 可能增加其任务数，但会与瓶颈子图竞争核心资源；增大非瓶颈子图的 TileShape 可以为瓶颈子图腾出资源
4. **量化验证**：每次调整后必须重新采集泳道图数据，用实际数据验证，禁止凭推测判定

**负载均衡分析输出格式**：

```markdown
### 负载均衡分析
| 排名 | psgId | 操作 | tasks | cores | total(us) | avg(us) | 优化方向 |
|------|-------|------|-------|-------|-----------|---------|---------|
| 1 (瓶颈) | 4 | data copy | 32 | 16/48 | 905.0 | 28.3 | 增加核数 |
| 2 | 3 | QKV proj | 6 | 6/24 | 665.5 | 110.9 | 增加核数 |
| 3 | 25 | output proj | 4 | 4/24 | 313.2 | 78.3 | 增加核数 |

瓶颈占比: 905.0 / 总计 2210.0 = 41.0%
瓶颈与次大差距: (905.0 - 665.5) / 905.0 = 26.4% → 存在严重负载不均

```

#### 3.4 核已满后的合图阶段

核用满后，才能进入合图调优（第 4 节）。合图的目的是减少调度开销和数据重复搬运，而非增加并行度。


### 4. 合图调优

合图是指将计算图中多个逻辑上独立的 Task 合并为一个逻辑子图。

**⛔ ⛔ ⛔ ⛔ 合图前强制三步分析（禁止跳过任何一步）：⛔ ⛔ ⛔ ⛔**

1. **第 1 步：核使用率分析**（analyze_core_usage.py）— 判定每个 leafHash FULL/NOT FULL。核未满必须优先通过 TileShape 填核，禁止跳过直接合图。
2. **第 2 步：泳道图分析**（analyze_swimlane.py --outer-loops N）— 获取每个 hashOrder 的 subGraphCount、t/iter，输出 Merge Tuning Guide。**必须基于当前最新性能数据的泳道图运行此脚本**，禁止使用旧数据或直接猜测 hashOrder。
3. **第 3 步：根据 Merge Tuning Guide 的 hashOrder 和粒度建议，设置特定参数**。禁止不用分析结果盲试全局参数（如 `{-1: N}`）。

**⚠️⚠️⚠️ 关键原则：**
1. **⛔ 合图前必须先完成核使用率分析（第 3 节）**：核未满时优先用满核，核满后再合图。盲目合图会导致核未满时性能退化。
2. pass_options key 有两种格式（**同一 dict 内禁止混用**）：
   - **整数键格式**：`{-1: N}`（全局通配所有子图），`vec_nbuffer_setting` 中必须加 `-2: 1` 作为 merge enable 标志
   - **字符串键格式**：`{"DEFAULT": M, "func{magic}_{order}": N}`（精细控制特定 hashOrder）
   - **场景选择**：需要所有子图均合图时用整数键 `-1`；仅需合图某几个 hashOrder 时用字符串键
   - hashOrder 来源：`merged_swimlane.json` 中每个事件的 `args.hashOrder-hint` 字段，对应三种合图类型：
     * `l1ReuseInfo hashOrder` → `cube_l1_reuse_setting` 的 key
     * `cubeMergeInfo hashOrder` → `cube_nbuffer_setting` 的 key
     * `vecMergeInfo hashOrder` → `vec_nbuffer_setting` 的 key
3. Value (N) 是合并粒度，每 N 个同构子图合并为一个。设为 1 表示不合并。
4. 合并粒度应由 **subGraphCount**（hashOrder-hint 中的同构子图数量）和核心数决定，常用值为 1/2/4/8/16。
5. **⛔ outer-loops 必须手动计算**：先阅读 kernel 代码确定 loop 嵌套结构，手动计算 outer-loops，再传入 `--outer-loops` 参数。脚本默认 `outer-loops=1`（auto 值不可靠），必须使用正确的计算值。

**合图调优标准流程**：

```bash
# Step 0: 核使用率分析（⛔ 强制前置，详见第 3 节）
# 对每个 AIC/AIV leafHash，统计其分布在多少个 core 上
# 判定规则：
#   - cores < total_cores 且可通过 TileShape 增加 → 先用满核，再回来
#   - cores == total_cores 或 TileShape 已充分尝试仍无法增核 → 进入 Step 1

# Step 1: 根据 loop 次数计算外层循环次数 outer-loops（必填），再用 analyze_swimlane.py 分析泳道图数据
python3 scripts/analyze_swimlane.py \
    output/output_<最新目录> --outer-loops xxx

# Step 2: 从输出确定：
#   - hashOrder 列 → 即为合图的 key（set_pass_options 的 key）
#   - core 列 → AIC 对应 cube 配置，AIV 对应 vec 配置
#   - t/iter 列 → 单个 root function 中子图数量，合图粒度参考值

# Step 3: 根据分析结果设置配置

```

#### 4.0 确定合图粒度

`subGraphCount` 是 `hashOrder-hint` 中提供的同构子图总数（跨所有外层循环迭代）。`t/iter = subGraphCount / outer_loops`，表示单个 root function 中该 hashOrder 的子图数量，直接指导合图粒度。

**获取方式**：从 `merged_swimlane.json` 中每个事件的 `args.hashOrder-hint` 字段解析，格式为：

```

l1ReuseInfo hashOrder: func15_1, subGraphCount: 90
cubeMergeInfo hashOrder: func15_1, subGraphCount: 90
vecMergeInfo hashOrder: func5_4, subGraphCount: 40

```

**自动分析**：运行 `analyze_swimlane.py` 后，Merge Tuning Guide 部分会自动输出每个 hashOrder 的 subGraphCount、t/iter 和建议粒度。

#### 确定 outer_loops（⛔ 必须手动计算，脚本默认 outer-loops=1）

`outer_loops` 是外层循环的总迭代次数，`t/iter = subGraphCount / outer_loops`。

**⛔ 禁止依赖默认值 1**：脚本默认 outer-loops=1 仅为保证脚本不中断运行。默认值会导致 t/iter 错误（偏大），合图粒度建议也全部错误。
**必须人工阅读 kernel 代码计算 outer_loops 后传入 `--outer-loops`。**

**手动计算方法**：分析实现代码的 loop 嵌套和 tile 切块：

```python
# 示例：flash_attention_score_grad
# B=2, N=8, S=256, S_TILE=128, s_loop=S//S_TILE=2
# loop 嵌套: b(2) × n(8) × s1(s_loop=2) × s2(s_loop=2)
# 外层循环（s2 之外）: b × n × s1 = 2 × 8 × 2 = 32
for b_idx in pypto.loop(b, ...):
    for n_idx in pypto.loop(N, ...):
        for s1_idx in pypto.loop(s_loop, ...):
            for s2_idx in pypto.loop(s_loop, ...):   # 最内层
                ...

```

分析方法：
1. 找到 kernel 函数中所有 `pypto.loop()` 调用，确定嵌套层级
2. 最内层循环（通常是带 `unroll_list` 的那个）不参与 outer_loops 计算
3. `outer_loops = 各外层循环次数的乘积`

可用 `--outer-loops` 参数手动指定精确值：

```bash
python3 scripts/analyze_swimlane.py output/output_xxx --outer-loops 32

```

#### t/iter 对合图粒度的指导

| t/iter | 含义 | 核状态 | 合图粒度建议 |
|--------|------|--------|-------------|
| 1 | 单个 root function 中只有 1 个子图 | — | 粒度 1（不合并），或跨 root function 尝试 2/4 |
| 2 | 单个 root function 中有 2 个子图 | 未满 | cube 类粒度 1（不合并）；vec_nbuffer 可尝试 `{-2: 1, -1: 2}` 或 `{"DEFAULT": 1, "func5_4": 4}` |
| 2 | 同上 | 已满 | 优先试 2，再试 4 |
| 4+ | 单个 root function 中有多个子图 | 未满 | 优先试 vec_nbuffer，cube 类通常退化 |
| 4+ | 同上 | 已满 | 可试 2/4/8，逐步增大 |

**⚠️ 核未满时的合图策略**：
- `vec_nbuffer_setting`：可尝试，用整数键 `{-2: 1, -1: N}` 或字符串键 `{"DEFAULT": 1, "func5_4": N}`，从 N=4 开始逐步试 8/16
- `cube_l1_reuse_setting` / `cube_nbuffer_setting`：通常退化，不建议设置；核未满时合图会进一步减少并行度

**⚠️ 粒度过大的风险**：
- avg<10us 的短耗时子图，合图粒度不宜超过 8（实测 N=16 时退化）
- 合图粒度可以大于 t/iter（跨 root function 合并），但不宜过大以免 L1/UB 内存溢出
- 每次调整后必须实测验证端到端耗时，禁止凭推测判定

#### 4.0.1 输出解读示例

```
#   leafHash                 min(us)   max(us)   avg(us)  total(us) core hashOrder    subGCnt t/iter root_name                                  compute_ops
1   17445...                   13.74     42.30     28.83    2537.46  AIC func15_1          90   11.2 ...        L1_TO_L0A+...
2   14789...                   42.14     46.12     44.67     938.10  AIC func15_2          22    2.8 ...        L1_TO_L0A+...
3   10766...                   28.76     31.54     29.91     598.16  AIC func5_0           20    2.5 ...        L1_TO_L0A+...
4   10531...                   25.16     28.64     26.91     565.18  AIC func15_0          22    2.8 ...        L1_TO_L0A+...
5   85886...                    1.12      3.72      1.69     108.20  AIV func11_0           1    0.1 ...        MULS+BAR.V+...
6   16650...                    3.80      4.26      4.09      12.28  AIV func5_4           40    5.0 ...        (pure copy)

outer_loops=8 (user-specified)

================================================================================
Merge Tuning Guide (hashOrder = merge key)
================================================================================

[AIC] cube_l1_reuse_setting:
  hashOrder=func15_1: subGraphCount=90, t/iter=11, avg=28.49us
    -> integer key: {-1: 2/4/8/16} (global)
    -> func key:    {"DEFAULT": 2/4, "func15_1": 2/4/8/16} (specific)
  hashOrder=func15_2: subGraphCount=22, t/iter=3, avg=44.08us
    -> integer key: {-1: 2/4/8} (global)
    -> func key:    {"DEFAULT": 2/4, "func15_2": 2/4/8} (specific)

[AIC] cube_nbuffer_setting:
  hashOrder=func15_1: subGraphCount=90, t/iter=11, avg=28.49us
    -> integer key: {-1: 2/4/8/16} (global)
    -> func key:    {"DEFAULT": 2/4, "func15_1": 2/4/8/16} (specific)
  hashOrder=func5_0: subGraphCount=20, t/iter=2, avg=29.91us
    -> integer key: {-1: 2/4/8} (global)
    -> func key:    {"DEFAULT": 2/4, "func5_0": 2/4/8} (specific)

[AIV] vec_nbuffer_setting:
  hashOrder=func5_4: subGraphCount=40, t/iter=5, avg=3.78us
    -> integer key: {-2: 1, -1: 2/4/8/16} (global)
    -> func key:    {"DEFAULT": 1, "func5_4": 2/4/8/16} (specific)

**解读**：
- hashOrder=func15_1 的 AIC 子图 subGCnt=90、t/iter=11（单个 root function 有 11 个子图），所有子图均合图时用整数键 `{-1: 4}`，仅合此 hashOrder 时用 `{"DEFAULT": 1, "func15_1": 4}`
- hashOrder=func5_0 的 AIC 子图 subGCnt=20、t/iter=2、avg=29.91us，所有子图均合图时用 `{-1: 2}`，仅合此 hashOrder 时用 `{"DEFAULT": 1, "func5_0": 2}`
- hashOrder=func5_4 的 AIV 子图 subGCnt=40、t/iter=5，所有子图均合图时用 `{-2: 1, -1: 4}`，仅合此 hashOrder 时用 `{"DEFAULT": 1, "func5_4": 4}`

#### 4.1 Vector 合图

**⛔ 重要原则**：`vec_nbuffer_setting` 中**必须**添加 `-2: 1` 配置，以规避部分合图不生效的问题。无论后续如何调优粒度，此配置不可省略。

##### 4.1.1 自动合图方案（vec_nbuffer_setting）

```python
# 整数键格式（全局通配所有子图）：
@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-2: 1, -1: 8}
    }
)

# 字符串键格式（精细控制特定 hashOrder）：
@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {"DEFAULT": 1, "func5_4": 8}
    }
)

```

**适用场景**：自动切图的vector task之间有直接依赖关系，且每一个task耗时很短（<10us）

**参数说明**
- 整数键格式：`-1:N` 代表所有 vector 子图按 N 的粒度合图；`-2:1` 必须添加作为 merge enable 标志
- 字符串键格式：`"DEFAULT":1` 是必需的 merge enable 标志；`"func5_4":N` 仅对 hashOrder=func5_4 的子图生效
- **禁止混用**：同一个 dict 内不能同时包含整数键和字符串键

**调优方法**：
1. 运行 [analyze_swimlane.py](scripts/analyze_swimlane.py)，查看 `[AIV] vec_nbuffer_setting` 部分的输出
2. 根据 `hashOrder` 确定合图 key，根据 `t/iter` 确定粒度参考值
3. t/iter=1 的组先设为 1，t/iter≥2 的组设为对应值或更小
4. 需所有子图均合图时用整数键 `{-2: 1, -1: N}`，需精细控制特定 hashOrder 时用字符串键

**参考资料**
- [vec_nbuffer_setting 参数设置说明](../../../../docs/zh/api/config/pypto-set_pass_options.md)

##### 4.1.2 手动合图方案（sg_set_scope）

通过 `sg_set_scope` 将有数据依赖的连续 Vector 操作强制合并到同一子图，减少子图间调度开销和数据搬运。

```python
pypto.set_pass_options(sg_set_scope=1)
# ... 连续的 Vector 操作（有直接数据依赖、同循环层级、无 Cube 夹杂）...
pypto.set_pass_options(sg_set_scope=-1)

```

**约束**：
- 仅对有直接上下游数据依赖的 Vector 操作生效
- 不要包裹 Cube（matmul）操作，也不要包裹其后继含 Cube 的节点
- 跨 `pypto.loop` 边界不能合并
- 每个 scope 使用不同的正整数 ID

###### 4.1.2.1 依赖链分析工具

使用 `analyze_aiv_dep_chains.py` 从 `dyn_topo.txt` 中提取 AIV 任务之间的依赖链路，自动识别 cube 边界并给出 sg_set_scope 合并建议。

**用法**：

```bash
python3 scripts/analyze_aiv_dep_chains.py <output_dir>
python3 scripts/analyze_aiv_dep_chains.py <output_dir> --json result.json

```

**输入文件**（`output_dir` 中）：
- `dyn_topo.txt` — 任务动态拓扑（含 successors 依赖，必需）
- `program.json` — 程序编译数据（可选，用于标注操作类型）

**输出分两部分**：

**Part 1: 原始依赖链**（完整链路，不截断）

```

链路A（16次）
3907163356593077760
  │
  ▼
2360323566658746396
  │
  ▼
2768731787098226973
  3907163356593077760: op=10001, psg=1, [vec] CAST+CAST
  2360323566658746396: op=10002, psg=2, [vec] MUL+CAST+CAST+ROWSUM_SINGLE
  2768731787098226973: op=10001, psg=1, [vec] MULS+SUB+EXP+DIV+SUB+MUL+CAST+CAST

```

**Part 2: sg_set_scope 优化建议**（在 cube 边界截断）

脚本自动检测每个 AIV 节点的后继是否包含 cube（matmul）任务。截断规则：
- 遇到后继含 cube 的 AIV 节点时，**保留该节点但不继续展开后继**
- 截断后 ≥2 节点且 psgId 有变化的链段，建议用 `sg_set_scope` 合并

```

sg_set_scope 优化建议

  建议 1: 截断后 3 个节点, 16 次, psgId 变化: 1 → 2 → 1
    3907163356593077760
      │
      ▼
    2360323566658746396
      │
      ▼
    2768731787098226973
    3907163356593077760: psg=1, [vec] CAST+CAST
    2360323566658746396: psg=2, [vec] MUL+CAST+CAST+ROWSUM_SINGLE
    2768731787098226973: psg=1, [vec] MULS+SUB+EXP+DIV+SUB+MUL+CAST+CAST [✂ cube边界]
    → 建议: 用 sg_set_scope 包裹 psgId 1 → 2 → 1 的 vector 操作
    ✂ 截断点 (后继含 cube): ['2768731787098226973']

```

**输出字段说明**：
- **leafHash**：叶子函数哈希，通过 `program.json` 的 `hash` 字段可映射到具体函数
- **opmagic**：操作类型标识
- **psgId**：当前所属子图 ID
- **[vec]/[cube]**：操作核心类型（基于 opcode 自动判断，含 `A_MUL_B/A_MULACC_B` 为 cube，否则为 vec）
- **opcode 序列**：过滤掉框架指令后的实际计算指令
- **✂ cube边界**：该 AIV 节点的后继包含 cube 任务

**⚠️ 重要：脚本建议是候选，必须经过 4.1.2.2 映射验证后才能实施。**

###### 4.1.2.2 从建议到实施的验证流程

脚本的优化建议是基于 `dyn_topo.txt` 的自动分析，不能直接用于修改前端代码。必须通过 `program.json` 的 `file`/`line` 字段将 leafHash 映射到前端代码，验证可合并性，并确认代码连续性。

详细的映射方法和自动映射工具参见 [leafHash → 前端代码映射方法](references/leafhash-to-code-mapping.md)。

**自动映射工具**：

```bash
# 查看指定 leafHash 的代码位置
python3 scripts/leafhash_to_code.py <output_dir> --leafhash <hash>

# 查看所有 leafHash
python3 scripts/leafhash_to_code.py <output_dir>

```

**验证检查清单**（对建议中的每个链段逐项检查）：

| 检查项 | 验证方法 | 通过标准 |
|:---|:---|:---|
| 数据依赖 | dyn_topo 中存在 VEC→VEC successors 边 | 有直接数据依赖 |
| 同循环层级 | dyn_topo 的 rootIndex 比对 | 所有节点 rootIndex 相同 |
| 纯 vector 操作 | program.json ops 中无 A_MUL_B/A_MULACC_B | 无 cube 指令 |
| 无 cube 后置依赖 | dyn_topo successors 中无 coreType=1 | 后继不含 matmul |
| 代码行连续性 | file/line 映射，确认中间无夹杂 | scope 内只有被合并的操作 |
| ✂ cube边界节点排除 | 脚本标记的截断点 | 有 cube 后继的节点不参与合并 |

**只有全部通过的链段才是可合并的。**

**代码连续性检查与调整**：

通过 `leafhash_to_code.py` 确认每个 leafHash 对应的前端代码行后，检查待合并的代码行之间是否夹带不相关操作。如果两个 leaf 对应的代码行之间有其他操作（如无关变量的 view），直接包裹 sg_set_scope 会把这些操作也卷入合并。

此时需要调整前端代码顺序，将不相关的操作移到 sg_set_scope 包裹范围之外，使待合并的操作紧密相邻。PyPTO 是声明式构图，只要数据依赖关系不变，代码顺序可以调整。

**调整原则**：
- 只移动与合并段无数据依赖的操作
- 移动后的代码不能跨越 `pypto.loop` 边界
- scope 必须覆盖所有参与合并的 leaf 的全部代码行，不能只包裹部分操作
- 调整后必须重新运行精度验证

**完整工作流程**：
1. 运行测试用例采集泳道数据（需 `debug_options={"runtime_debug_mode": 1}`）
2. 运行 `analyze_aiv_dep_chains.py` 分析依赖链，获取 Part 2 优化建议
3. 运行 `leafhash_to_code.py` 将 leafHash 映射到前端代码行
4. 用验证检查清单过滤，排除不可合并的段
5. 检查代码连续性：合并段对应的代码行之间是否有不相关操作
6. 如有夹杂，调整代码顺序使合并段紧密相邻
7. 在连续的代码段位置插入 `sg_set_scope`
8. 验证精度和性能

**参考资料**
- [leafHash → 前端代码映射方法](references/leafhash-to-code-mapping.md)
- [sg_set_scope 参数设置说明](../../../../docs/zh/api/config/pypto-set_pass_options.md)


#### 4.2 Cube 合图

##### 4.2.1 L1Reuse 策略（默认开启，用于合并具有 L1 重复搬运的子图）

**适用场景**：matmul 的 M 或 N 轴进行了切分，存在重复搬运

```python
# 整数键格式（全局通配所有子图）：
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {-1: 2}}
)

# 字符串键格式（精细控制特定 hashOrder）：
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {"DEFAULT": 2, "func15_1": 8}}
)

```

**调优方法**：
1. 运行 [analyze_swimlane.py](scripts/analyze_swimlane.py)，查看 `[AIC] cube_l1_reuse_setting` 部分的输出
2. 根据 `hashOrder` 确定合图 key，优先对 total 耗时大且有重复搬运的子图调优
3. `t/iter` 越大（单个 root function 中子图越多），L1 复用收益越高，可设更大粒度
4. ⚠️ **核未满时通常退化**：如果 `analyze_core_usage.py` 显示核未满，cube_l1_reuse 通常导致性能退化，不建议设置

**参考资料**
- [cube_l1_reuse_setting 参数设置说明](../../../../docs/zh/api/config/pypto-set_pass_options.md)

##### 4.2.2 CubeNBuffer 策略（用于合并同构的子图）

**适用场景**：
- 同构子图数量很多，且每一个task的执行耗时很短（10us以下）
- K 轴很长且没有切 K

```python
# 整数键格式（全局通配所有子图）：
@pypto.frontend.jit(
    pass_options={"cube_nbuffer_setting": {-1: 2}}
)

# 字符串键格式（精细控制特定 hashOrder）：
@pypto.frontend.jit(
    pass_options={"cube_nbuffer_setting": {"DEFAULT": 2, "func15_1": 4}}
)

```

**调优方法**：
1. 运行 [analyze_swimlane.py](scripts/analyze_swimlane.py)，查看 `[AIC] cube_nbuffer_setting` 部分的输出
2. 根据 `hashOrder` 确定合图 key，根据 `t/iter` 和 avg 耗时确定粒度
3. avg<10us 且 t/iter≥2 的组优先设置，如 `cube_nbuffer_setting: {-1: 2}` 或 `cube_nbuffer_setting: {"DEFAULT": 1, "func5_0": 2}`
4. ⚠️ **核未满时通常退化**：如果 `analyze_core_usage.py` 显示核未满，cube_nbuffer 通常导致性能退化，不建议设置

**参考资料**
- [cube_nbuffer_setting 参数设置说明](../../../../docs/zh/api/config/pypto-set_pass_options.md)

##### 4.2.3 L1Reuse 与 CubeNBuffer 的协同使用

**⚠️ 重要：两者作用维度不同，需协同配置，不宜同时过大。**

| 参数 | 合并维度 | 核心目的 |
|------|---------|---------|
| cube_l1_reuse_setting | 消除 GM 数据重复搬运 | 多个子图复用同一份 L1 数据，减少搬运开销 |
| cube_nbuffer_setting | 合并结构相同的 AIC 子图 | 减少调度开销，提升核心利用率 |

**协同原则**：
1. **两者不宜同时设置过大**：cube_l1_reuse 通过消除重复搬运带来收益，合并力度越大 L1 复用越好；cube_nbuffer 通过合并同构子图减少调度开销。但两者同时过大会导致单个子图过大，占用过多 L1/UB 内存，反而引发性能退化。
2. **优先调 cube_l1_reuse_setting**：先确定 L1 数据复用的合并力度（消除重复搬运是更直接的收益），再调整 cube_nbuffer_setting。
3. **观察 Task Count 变化**：合图后 Total Task Count 应适度下降。如果 Task Count 不降反升（例如 1664→6400），说明合图配置过度，应回退。
4. **用 analyze_swimlane.py 确定参数**：hashOrder 即合图 key，t/iter 指导粒度。

**反面案例**（flash_attention_score_grad 实测）：

```python
# baseline: 728us, Task=1664
"cube_l1_reuse_setting": {-1: 8},
"cube_nbuffer_setting": {-1: 8}
# → 711us ✅ (Task=1664, 利用率 58.8%→61.4%)

# 过度合图: 734us ❌ 性能退化
"cube_l1_reuse_setting": {-1: 8},
"cube_nbuffer_setting": {-1: 16}

```

##### 4.2.4 自动合图模式（空字典 `{}`）的风险

**⚠️ 风险提示：自动模式可能过度合图导致性能严重退化，不建议直接使用。**

当设置为空字典 `{}` 时，Pass 会根据硬件核心数自动计算合并粒度。但自动模式不了解算子的实际数据流特征，可能将不应合并的子图强行合并，导致：
- Task 数暴增（如 1664→6400）
- 子图过大，L1/UB 内存争用
- 核心利用率大幅下降（如 58%→48%）

**反面案例**（flash_attention_score_grad 实测）：

```python
# 自动模式: 1038us ❌ 性能退化 42%
"cube_l1_reuse_setting": {},
"cube_nbuffer_setting": {}
# Task Count: 1664 → 6400, 利用率: 58.8% → 48.7%

```

**建议**：始终使用 [analyze_swimlane.py](scripts/analyze_swimlane.py) 分析泳道图获取 hashOrder 和 t/iter 后手动精确配置，避免使用空字典 `{}` 自动模式。


### 5. 调度策略调优

当上下游子图之间依赖较为简单，或下游子图输入 Tensor 的 L2 命中率较为重要时，推荐使用 L2 亲和调度。

```python
@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})

```

**调优建议**：
- 尝试不同的调度策略，值域范围是[0, 3]

**注意事项**：综合考虑 L2 复用与负载均衡的影响，不同场景的最佳配置策略不同。

**参考资料**
- [device_sched_mode 参数设置说明](../../../../docs/zh/api/config/pypto-frontend-jit.md)


### 6. Matmul 访存布局优化（L2 命中率优化）

**问题**：大 shape Matmul 场景下（M、N、K 全部较大），即使 TileShape 配置了推荐值，固定的分核布局可能导致 L2 命中率偏低、MTE2 带宽利用率不足。

**原理**：L2 命中率由单轮次分核数 mDim、nDim 和 mL1、nL1 共同决定：

```

l2_hit_ratio = 1 - (1/(nDim·nL1) + 1/(mDim·mL1)) / (1/nL1 + 1/mL1)

```

**最优条件**：`nDim·nL1 = mDim·mL1`，即 M、N 轴分核到 L1 的数据量相等，L2 复用最大化。

例如 mL1=128、nL1=256、24 核平台时，推荐：
- mDim=6、nDim=4 → 6×128 = 4×256 = 1024 ✅
- mDim=8、nDim=3 → 8×128 ≈ 3×256 = 768 ✅

**优化方法**：在 M、N 轴外层再套一层 loop，手动控制每轮 M 和 N 的计算范围，从而控制分核数：

```python
def create_mm_with_l2_opt(M, K, N, mL1, nL1, mDim, nDim):
    @pypto.frontend.jit(...)
    def matmul_kernel(a, b):
        pypto.set_cube_tile_shapes([mL1, mL1], [64, 256], [nL1, nL1])

        m_view = mL1 * mDim
        n_view = nL1 * nDim
        m_loop = (M + m_view - 1) // m_view
        n_loop = (N + n_view - 1) // n_view
        out = pypto.Tensor([M, N], pypto.DT_FP16)

        for m_idx in pypto.loop(m_loop, name="LOOP_m", idx_name="m_idx"):
            for n_idx in pypto.loop(n_loop, name="LOOP_n", idx_name="n_idx"):
                a_block = a[m_idx*m_view : m_idx*m_view+m_view, :]
                b_block = b[:, n_idx*n_view : n_idx*n_view+n_view]
                out_block = pypto.matmul(a_block, b_block,
                    out_dtype=pypto.DT_FP16)
                out[m_idx*m_view:m_idx*m_view+m_view,
                    n_idx*n_view:n_idx*n_view+n_view] = out_block
        return out
    return matmul_kernel

```

**调试方法**：
1. 从泳道图或气泡分析中检查 MTE2 带宽利用率（是否明显低于 HBM 带宽）
2. 根据 mL1/nL1 和芯片总核数，按最优条件计算 mDim/nDim
3. 按上述模式改写，重新采集性能数据对比

**收益**：M=N=K=6144、TileShape=[128,128],[64,256],[256,256] 场景：
- 默认分核：2.1ms，等效算力约 220 TFLOPS
- L2 优化（mDim=6,nDim=4）：1.6ms，等效算力约 290 TFLOPS（+31%）

## 调优检查清单

**⛔ 必须按以下清单逐项执行。每项标记为 ✅已尝试 或 ❌已失败（附原因），禁止跳过。完整优化点信息参考 [shared/optimization_catalog.md](../shared/optimization_catalog.md)。**

**优化优先级**：
1. ⭐⭐⭐ **P0 - 核使用率分析 + 核填充** → 详见 [S-1][S-2]
2. ⭐⭐⭐ **P1 - 负载均衡 + TileShape 深度调优 + 访存布局优化** → 详见 [S-3][S-11][S-12][S-13]
3. ⭐⭐⭐ **P2 - 手动合图（sg_set_scope）** → 详见 [S-4]
4. ⭐⭐ **P3 - 自动合图** → 详见 [S-5][S-6][S-7][S-8]
5. ⭐ **P4 - Stitch + 调度策略** → 详见 [S-9][S-10]

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

**P3 - 自动合图 [S-5~S-8]**：
- [ ] [S-5] 是否对短耗时（<10us）AIV 任务尝试了 vec_nbuffer_setting
- [ ] [S-6] 是否对核满的 AIC 子图尝试了 cube_l1_reuse_setting
- [ ] [S-7] 是否对核满的 AIC 子图尝试了 cube_nbuffer_setting
- [ ] [S-8] 如已配置 S-6/S-7，是否检查了协同使用是否过大

**P4 - Stitch [S-9] + 调度策略 [S-10]**：
- [ ] [S-9] 是否尝试了 stitch_function_max_num 调整
- [ ] [S-10] 是否尝试了 device_sched_mode 调整（1/2/3）


## 常见问题

### Q1: 泳道图文件在哪里？

A: 泳道图文件在 `output/output_*/` 目录下，其中 `*` 是时间戳。

### Q2: 如何查看性能统计？

A: 使用 PyPTO Toolkit 打开 `merged_swimlane.json` 文件，然后点击 "查看性能报告" 按钮。

### Q3: 气泡是什么？

A: 气泡是指线程等待调度的时间，表示线程空闲的时间段。气泡率越低，说明调度效率越高。

### Q4: 控制开销占比过高怎么办？

A: 对于小数据量，控制开销占比高是正常现象。可以通过增加数据规模来降低控制开销占比。

### Q5: 如何选择合适的 Tilesize？

A:

* 对于 Cube 计算：推荐使用 [128, 128], [64, 256], [256, 256] 或 [256, 256], [64, 256], [128, 128]
* 对于 Vector 计算：推荐使用 [32, 512] 或 [64, 512]
* 需要根据具体场景（输入 shape、dtype、format 等）以及硬件平台进行综合考虑


## 参考资料

- [性能调优文档](../../../../docs/zh/tutorials/debug/performance.md)
- [Matmul 高性能编程](../../../../docs/zh/tutorials/debug/matmul_performance_guide.md)
- [GLM Attention 案例](../../../../models/glm_v4_5/glm_attention.py)
- [性能优化案例](../../../../docs/zh/tutorials/debug/performance_case_quantindexerprolog.md)
