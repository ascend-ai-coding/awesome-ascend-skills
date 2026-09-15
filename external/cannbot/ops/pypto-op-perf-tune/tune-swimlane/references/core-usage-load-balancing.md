### 核使用率分析与负载均衡（合图前置条件）

**⛔ 重要：合图调优前，必须先完成核使用率分析。核未用满时优先用满核，再考虑合图。**

合图（L1reuse / CubeNBuffer / VecNBuffer）的前提是核已用满。如果核没用满就合图，会将本就不多的任务进一步合并到更少的核上，反而降低并行度、导致性能退化。

#### 1.0 核使用率分析流程

```

采集泳道图数据（debug_options={"runtime_debug_mode": 1}）
    ↓
统计每个 leafHash 分布在多少个 core 上
    ↓
对每个 AIC/AIV leafHash，判断核是否用满
     ├─ 核未满 → 优先通过 TileShape 调整用满核（跳到 3.2）
    └─ 核已满 → 进入合图阶段（跳到第 4 节）

```

#### 2. 统计核使用率

从 `merged_swimlane.json` 的 `traceEvents` 中解析每个 leafHash 占用的 core 数量，并与芯片理论核数对比：

> **路径说明**：以下脚本命令在技能目录 `skills/pypto-op-perf-tune/tune-swimlane/` 下执行

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

#### 3. 核未满时的优化策略

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

#### 4. 负载均衡优化（核填充后的强制步骤）

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

#### 5. 核已满后的合图阶段

核用满后，才能进入合图调优（第 4 节）。合图的目的是减少调度开销和数据重复搬运，而非增加并行度。
