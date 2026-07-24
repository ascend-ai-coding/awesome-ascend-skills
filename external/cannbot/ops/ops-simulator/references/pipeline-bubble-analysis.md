# 流水线空泡分析（Simulator Trace）

读取 simulator 产出的 trace.json（Chrome Trace Format），按以下标准自主判定流水空泡类型和根因。

**定位**：trace 分析回答"主 pipeline（VECTOR/CUBE）为什么 idle"（WHY）。

**使用时机**：
- 无 NPU 硬件时，直接通过仿真定位性能瓶颈
- 有 NPU 时，下沉到 trace 查看具体等待原因

---

## 脚本辅助预分析

如需批量处理多核 trace，可运行 `scripts/trace_bubble_analyzer.py` 获取预分类报告：

```bash
# 分析单个 trace
python3 {skill_path}/scripts/trace_bubble_analyzer.py report/trace_core0.json

# 分析 simulator 输出目录（自动发现所有核）
python3 {skill_path}/scripts/trace_bubble_analyzer.py ./cannsim_Ascend950_*/report/

# 输出 JSON 供进一步分析
python3 {skill_path}/scripts/trace_bubble_analyzer.py ./cannsim_Ascend950_*/report/ --json -o bubble_report.json
```
---

## Pipeline 识别

读取 trace 时，从 `traceEvents` 中的 metadata 事件（`ph="M"`, `name="process_name"`）自动发现 PID 与 pipeline 的映射关系。

**需要注意的格式差异**：

| 情况 | 说明 |
|------|------|
| cannsim VECTOR 细分 | cannsim 将 VECTOR 拆分为 RVECSU/RVECEX/RVECLD/RVECST/RVECLP，分析时统一归并为 VECTOR |
| FIXPIPE 别名 | cannsim 中字段名为 `FIXP`，与 `FIXPIPE` 等价 |
| 指令缓存未命中 | msprof 中为 `CACHEMISS`，cannsim 中为 `ICACHELOAD` |

---

## 空泡分类体系

空泡 = 主 pipeline（veccore: VECTOR / cubecore: CUBE）的 duration 事件之间的 idle 时间（`ph="X"` 事件，gap = next.ts - current.end_ts）。

对每个空泡查看**同一时间段内其他 pipeline 的状态**，按以下标准分类。

### 一级分类

| 类别 | 含义 | 可优化 | 典型根因 |
|------|------|--------|---------|
| **NORMAL** | 正常发射间距 | No | 指令依赖或硬件发射限制 |
| **STRUCTURAL** | 流水线结构性开销 | No | drain/barrier/icache miss/首尾 tile 填充排空 |
| **DATA_STALL** | 数据搬运等待 | ★★ | MTE2/MTE3 未完成，VEC/CUBE 在等待数据 |
| **SCALAR_OVERHEAD** | 标量计算/加载阻塞 | ★ | SCALARLDST 参数加载或地址计算阻塞主 pipeline |
| **RESOURCE_CONTENTION** | 资源竞争 | △ | MTE2+MTE3 总线竞争、UB bank conflict |
| **CROSS_CORE** | 跨核不均衡 | ★ | 各核工作量不均，部分核提前完成等待 |
| **CUBE_VECTOR** | Cube-Vector 协同 | ★ | MatMul 类算子中 Cube 和 Vector 流水不匹配 |

### 二级分类判定规则

**四档过滤（按 gap 大小）**：

| 档位 | gap 范围 | 初步方向 | 说明 |
|------|----------|---------|------|
| **Tier A** | < 1 ps | **NORMAL** | 正常发射间距，无需深入分析 |
| **Tier B** | 1 ~ 30 ps | **STRUCTURAL** | 检查是否 drain/barrier |
| **Tier C** | 30 ~ 500 ps | **SCALAR / DATA** | 检查 scalar 开销或数据搬运等待 |
| **Tier D** | > 500 ps | **SYNC / COLD_START** | 检查显式同步、冷启动、尾排空 |

**判定方法**：
1. 先按 gap 大小确定档位（上表）
2. 再检查该空泡窗口内其他 pipeline 的活跃状态，匹配下表子类型
3. VecCore 与 CubeCore 使用不同归因优先级（CubeCore 额外检查 FIXPIPE）

**"同期覆盖率"**：某 pipeline 在空泡窗口内的活跃时长 / 窗口总时长。覆盖率只在 Tier C 和 Tier D 使用：
- Tier C（30~500ps）：覆盖率 **> 30%** 即判定为"busy"
- Tier D（>500ps）：覆盖率 **> 50%** 即判定为"busy"
- Tier A/B 不检查覆盖率（A 直接 NORMAL，B 直接查 drain/barrier）

#### NORMAL

| 子类型 | 含义 | 判定标准 | 优化方向 |
|--------|------|---------|---------|
| `N_ISSUE_GAP` | 正常发射间距 | gap < 1.0 ps；或 1-30ps 但前指令非 drain 类 | 无需优化 |

#### STRUCTURAL

| 子类型 | 含义 | 判定标准 | 优化方向 |
|--------|------|---------|---------|
| `S_DRAIN` | 流水线排空 | **Tier B（1~30ps）**：前指令为 VNCHWCONV 等 drain 类；**Tier C（30~500ps）**：所有 pipeline 均 idle；**Tier D（>500ps）**：不属于其他任何子类型时兜底 | 无需优化 |
| `S_BARRIER` | PipeBarrier 同步 | 1-30ps 且前指令为 PipeBarrier / BAR | 无需优化 |
| `S_COLD_START` | 首 tile 流水线填充 | gap ≥ 500ps 且为首 tile（iteration_index == 0） | 减少核数、缩小 TilingData |
| `S_TAIL_DRAIN` | 尾 tile 流水线排空 | gap ≥ 500ps 且为尾 tile（iteration_index == 末位） | 减少核数、缩小 TilingData |
| `S_ICACHE_MISS` | 指令缓存未命中 | 同期 CACHEMISS 活跃（任意 Tier） | 减少代码量、减少条件分支 |
| `S_FLOWCTRL` | 流控指令开销 | 同期 FLOWCTRL 活跃（Tier C，30~500ps） | 无需优化 |

#### DATA_STALL

| 子类型 | 含义 | 判定标准 | 优化方向 |
|--------|------|---------|---------|
| `D_MTE2_WAIT` | 显式等待 MTE2 数据加载 | gap ≥ 500ps 且存在 WAIT_FLAG 声明等待 MTE2 | 使能双缓冲，重叠 MTE2 与计算 |
| `D_MTE2_IMPLICIT` | 隐式等待 MTE2 | **Tier C（30~500ps）**：MTE2 覆盖率 > 30%；**Tier D（>500ps）**：无 WAIT_FLAG 但 MTE2 覆盖率 > 50% | 使能双缓冲，重叠 MTE2 与计算 |
| `D_MTE3_WAIT` | 显式等待 MTE3 写回 | gap ≥ 500ps 且存在 WAIT_FLAG 声明等待 MTE3 | 使能双缓冲，重叠 MTE3 与计算 |
| `D_MTE3_IMPLICIT` | 隐式等待 MTE3 | **Tier C（30~500ps）**：MTE3 覆盖率 > 30%；**Tier D（>500ps）**：无 WAIT_FLAG 但 MTE3 覆盖率 > 50% | 使能双缓冲，重叠 MTE3 与计算 |
| `D_MTE2_UNDERSIZE` | MTE2 搬运粒度过小 | 单次搬运量 < 16KB 或 MTE2 事件数/tile > 20 | 增大 tile、向量化搬运、512B 对齐 |
| `D_MTE3_UNDERSIZE` | MTE3 搬运粒度过小 | 同上，针对 MTE3 通道 | 增大 tile、向量化搬运 |
| `D_NO_OVERLAP` | 双缓冲未生效 | overlap_pct < 5%；或 Tier D（>500ps）无 WAIT_FLAG 且不属于其他任何子类型 | 检查 bufNum=2、EnQue/DeQue 配对 |
| `D_PARTIAL_OVERLAP` | 双缓冲部分生效 | overlap_pct 5%-30% | 调整 tile 大小或 UB 分区 |

#### SCALAR_OVERHEAD

| 子类型 | 含义 | 判定标准 | 优化方向 |
|--------|------|---------|---------|
| `SC_LDST_BLOCK` | 标量参数加载阻塞 | SCALARLDST 覆盖率 > 30%（Tier C，30~500ps）；或 WAIT_FLAG 等待 SCALARLDST | 参数预取、缩小 TilingData |
| `SC_COMPUTE_BLOCK` | 标量地址计算阻塞 | SCALAR 覆盖率 > 30%（Tier C，30~500ps）；或 WAIT_FLAG 等待 SCALAR | 简化地址计算、预计算偏移、缩小 TilingData |
| `SC_TILING_COMPLEX` | 标量 tiling 逻辑复杂 | scalar 事件密集且含多层循环/条件分支 | 简化控制流、移出不变量到 Host |

#### RESOURCE_CONTENTION

| 子类型 | 含义 | 判定标准 | 优化方向 |
|--------|------|---------|---------|
| `R_UB_PRESSURE` | UB bank conflict | bank conflict 占比 > 5%（需结合 ResourceConflictRatio.csv） | 调整 UB 地址、添加 padding |
| `R_ICACHE_THRASH` | 指令缓存抖动 | 代码段过大或频繁跳转导致 icache 未命中率高 | 减少代码量、减少条件分支 |
| `R_BUS_CONTENTION` | MTE2+MTE3 总线竞争 | 两通道同时高覆盖率且 gap 增大 | 错开搬运时序 |

#### CROSS_CORE

| 子类型 | 含义 | 判定标准 | 优化方向 |
|--------|------|---------|---------|
| `X_TILING_IMBALANCE` | 核间 tiling 切分不均衡 | imbalance_ratio = max_time / min_time > 1.3 | 调整 tiling、均匀分配数据 |
| `X_TAIL_CORE` | 尾核等待 | 某核尾 tile 空泡显著高于其他核 | 均匀分配尾块、调整 Block Dim |
| `X_SYNC_BARRIER` | 跨核同步屏障等待 | 多核在相近时间点同时出现大范围 idle | 减少同步点、异步迭代 |

#### CUBE_VECTOR

| 子类型 | 含义 | 判定标准 | 优化方向 |
|--------|------|---------|---------|
| `CV_CUBE_WAIT` | Vector 等待 Cube 完成 | AIV 空闲时 AIC 仍在计算（需同时分析 AIC/AIV trace） | 调整 AIC:AIV 比例 |
| `CV_VECTOR_WAIT` | Cube 等待 Vector 完成 | AIC 空闲时 AIC 仍在计算 | 调整 AIC:AIV 比例 |
| `CV_HANDOFF` | FIXPIPE 格式转换阻塞 | FIXPIPE 覆盖率 > 30%（Tier C，30~500ps）或 > 50%（Tier D，>500ps）；或 WAIT_FLAG 等待 FIXPIPE | 增加 workspace 份数 |

> **注**：> - CROSS_CORE 和 CUBE_VECTOR 中的部分子类型需多核 trace 对比或额外 CSV 数据，脚本无法自动判定，需自主计算
> - RESOURCE_CONTENTION 类（UB bank conflict、icache thrash、bus contention）需结合额外性能计数器或 ResourceConflictRatio.csv，脚本无法直接判定
> - DMA 搬运效率（undersize）需统计单次搬运量和事件密度，脚本未实现，可直接从 trace 事件计算

---

## 关键阈值

判定时使用以下阈值：

### 空泡大小快速过滤

参见上文"四档过滤"表格。快速记忆：

| 档位 | gap | 一句话 |
|------|-----|--------|
| Tier A | < 1 ps | 太小，忽略 |
| Tier B | 1~30 ps | 看前面是不是 drain/barrier |
| Tier C | 30~500 ps | 看谁 busy（scalar/data） |
| Tier D | > 500 ps | 看同步/冷启动/尾排空 |

> 以上仅为快速过滤，最终分类仍需查归因树表格匹配。

### 跨核不均衡

| 指标 | 均衡 | 轻度不均衡 | 严重不均衡 |
|------|------|-----------|-----------|
| `imbalance_ratio = max_time / min_time` | <= 1.3 | 1.3-2.0 | > 2.0 |

> 从各 core 的 trace 统计 VECTOR/CUBE 总活跃时间后计算。

### 双缓冲生效判定

| 重叠比例 | 状态 | 对应子类型 |
|---------|------|-----------|
| `overlap_pct < 5%` | 未生效 | `D_NO_OVERLAP` |
| `overlap_pct 5-30%` | 部分生效 | `D_PARTIAL_OVERLAP` |
| `overlap_pct > 30%` | 效果良好 | — |

> `overlap_pct = intersect(MTE2_duration, VECTOR_duration) / union(MTE2_duration, VECTOR_duration)`，按单个 tile 计算。

### DMA 搬运效率

| 指标 | 低效 | 正常 |
|------|------|------|
| 单次搬运量 | < 16KB | >= 16KB |
| MTE2 事件数/tile | > 20 | <= 20 |

> 低效 → `D_MTE2_UNDERSIZE` / `D_MTE3_UNDERSIZE`。

### 周期性模式

| 模式 | 判定标准 | 含义 |
|------|---------|------|
| **periodic** | 各 tile 的空泡在时间轴上位置对齐（差异 < 5% tile 周期） | 系统性问题，优化可全局消除 |
| **sporadic** | 空泡位置随机分布 | 偶发，非主要瓶颈 |
| **cold_start_dominant** | 首 tile 空泡占总空泡 > 30% | 首 tile 填充开销大 |
| **tail_dominant** | 尾 tile 空泡占总空泡 > 30% | 尾块处理逻辑有问题 |

---

## 优化可行性约束

提出优化方案前，自主验证以下硬件约束：

### 双缓冲可行性

| 约束 | 检查方式 |
|------|---------|
| UB 容量 | `2 * tile_size * sizeof(dtype) * pipe_count <= UB_total` |
| 对齐 | 所有 buffer 起始地址为 32B 整数倍 |

### 增大 Tile 可行性

| 约束 | 检查方式 |
|------|---------|
| UB 容量 | `tile_size * sizeof(dtype) * pipe_count <= UB_per_buffer`（考虑双缓冲后） |
| 负载均衡 | 增大后尾块不能过大（`tail_elements / tile_elements < 0.3`） |

---

## 分析报告输出格式

完成 trace 分析后，输出以下关键结论（控制在 5-10 行内，结论先行）：

### 1. 总体评估

| 项 | 输出 |
|----|------|
| 是否有空泡 | 是 / 否（空泡数 > 0 即判定为"是"） |
| 空泡密度 | 空泡总时长 / core 总运行时长（百分比） |
| 主瓶颈类别 | NORMAL / STRUCTURAL / DATA_STALL / SCALAR_OVERHEAD / CUBE_VECTOR 中占比最高者 |
| 可优化比例 | 可优化子类型空泡时长 / 总空泡时长（百分比） |

### 2. Top 3 关键空泡

按空泡时长（gap_us）从大到小排序，取前 3 个：

```
#1 [子类型] gap=XXX us | 根因：XXX | 优化：XXX（如可优化）
#2 [子类型] gap=XXX us | 根因：XXX | 优化：XXX
#3 [子类型] gap=XXX us | 根因：XXX | 优化：XXX
```

**判定"关键"的标准**：
- 优先选 gap > 500ps（Tier D）的空泡
- 其次选可优化（optimizable=True）的空泡
- 若前两者不足 3 个，再补 STRUCTURAL 类中最大的

### 3. 行动建议

一句话总结：
- **可优化为主** → "建议优化 XXX（如使能双缓冲/简化标量计算）"
- **STRUCTURAL 为主** → "空泡以结构性开销为主，优化空间有限，建议检查核数/TilingData"
- **无显著空泡** → "空泡密度低，性能瓶颈不在 pipeline idle，建议排查其他因素"
