# 性能采集（msprof + L2 cache flush）参考

本文档承载 MC2 skill 的"性能子能力"。MC2 算子性能采集的三个核心要点：

1. **必须刷 L2 cache**：每轮主 kernel 前调 `heavy_add_kernel`，否则前一轮的 B 矩阵驻留 L2 让带宽指标虚高；
2. **用 `msprof --ai-core=on --aic-mode=task-based` 采集**：以 task（kernel launch）为单位记录，**不用 `--warm-up`**；
3. **多卡数据后处理规则**：每张卡取**最后 5 次** main kernel 的 Task Duration 求平均，4 张卡的平均值取**最大值**作为整体性能。

> 父级性能采集流程见 `ops-profiling/SKILL.md`。本文件只补充 MC2 场景的差异化要点。

---

## 1. AIV+URMA 冷缓存采样为什么逐轮刷 L2

Blaze-SHMEM 示例的 perf 模式默认运行 10 轮，主 kernel 每轮读取同一份 B 矩阵。如果不在每轮之间清空 L2：

- 下一轮的 B 直接从 L2 命中，**MTE2 带宽指标虚高**；
- 通信开销（UDMA）相对计算的占比也会失真。

参考工程的解法：每轮主 kernel 前调用 `heavy_add_kernel` 扫一遍大于目标芯片 L2 容量的 bf16 buffer（参考工程示例值 256 MB），把 B 挤出 L2。

> **为什么不用 `--warm-up`？** 本示例使用 task-based 采集，程序连续跑 10 轮，按 §4.4 统计最后 5 轮。`heavy_add_kernel` 控制 L2 缓存条件，不能替代预热；后段样本仍不稳定时应增加运行轮次。`msprof op` 的 `--warm-up` 参数对 MC2 不生效，不能依赖它完成预热。

---

## 2. heavy_kernels.h 剖析

参考工程在 `include/kernel/heavy_kernels.h` 提供两个"重负载"kernel：

### 2.1 heavy_exp_kernel（boost 预热）

```cpp
__global__ __aicore__ __vector__ void heavy_exp_kernel(
    GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t blockLength)
```

- 输入：bf16 buffer（参考工程示例值 10000×10000 = 200 MB）；
- 操作：`y = exp(x)`，全部 AIV 核并行（block 数按目标芯片实际 AIV 核数确定，参考工程示例值 56）；
- 用途：**去 host bound**——上板首次运行受 host 启动开销影响，先跑 100 次 exp 把 host 端的异步任务队列填满，让 NPU 进入稳定状态。

参考工程 `src/all_to_all_matmul.cpp` 注释掉了 boost 阶段（`heavy_exp_kernel` 调用被注释）；新算子按需启用。

### 2.2 heavy_add_kernel（L2 cache flush）

```cpp
__global__ __aicore__ __vector__ void heavy_add_kernel(
    GM_ADDR x, int64_t totalLength, int64_t blockLength)
```

- 输入：bf16 flush buffer，大小按目标芯片实际 L2 容量动态确定、须大于 L2 容量（参考工程示例值 128×1024×1024 = 256 MB）；
- 操作：`x += 1`，全部 AIV 核并行（block 数按目标芯片实际 AIV 核数确定，参考工程示例值 56）；
- 用途：**刷 L2 cache**——flush buffer 大于目标芯片 L2 容量即可，扫描一遍把 B 矩阵挤出 L2（L2 容量等芯片规格以 npu-arch skill 为唯一知识源，本文档不复述数值）。

参考工程 `src/all_to_all_matmul.cpp` 的 perf 主循环（`mode == "perf"` 分支）：

```cpp
constexpr int PERF_LOOP_COUNT = 10;
for (int i = 0; i < PERF_LOOP_COUNT; ++i) {
    // 1. 刷 L2
    heavy_add_kernel<<<HEAVY_BLOCK_NUM, nullptr, stream>>>(
        cacheFlush, CACHE_FLUSH_ELEM_COUNT, cacheFlushBlockLen);
    // 2. 跑主 kernel
    AllToAllQuantMatmulKernelE4M3E4M3<<<tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
        shmemSpace, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
    ACL_CHECK(aclrtSynchronizeStream(stream));
}
```

每轮的 `heavy_add_kernel` + 主 kernel 都会被 msprof 单独采集，产生独立的 task 记录。分析时只取主 kernel 的记录。

### 2.3 HEAVY_BLOCK_NUM 的选择

```cpp
constexpr int64_t HEAVY_BLOCK_NUM = 56;  // 示例值，按目标芯片实际 AIV 核数设置
```

HEAVY_BLOCK_NUM 取目标芯片的**实际 AIV 核数**（AIV 核数属芯片规格，以 npu-arch skill 为唯一知识源，本文档不复述典型值；运行时经 `npu-smi info` 或平台规格接口查询），让 heavy_add_kernel 占满所有 AIV 核，最大化 L2 flush 效果。

---

## 3. perf mode 调用约定

参考工程的 host 程序通过第 5 个命令行参数区分 precision/perf 模式：

```bash
# precision 模式（默认）：单次 kernel + 精度比对
./build/all_to_all_matmul 2048 8192 3584 4

# perf 模式：10 轮 (cache_flush + main kernel)，供 msprof 采集
./build/all_to_all_matmul 2048 8192 3584 4 perf
```

`run.sh` 透传这个参数（第 5 个位置参数）：

```bash
bash run.sh 2048 8192 3584 4 perf
```

### precision vs perf 流程对比

| 步骤 | precision 模式 | perf 模式 |
|------|---------------|-----------|
| 数据准备 | gen_data.py 生成 input_*.bin | 同 precision |
| kernel 调用 | 单次 `AllToAllQuantMatmulKernelE4M3E4M3` | 10 轮（cache_flush + main） |
| 输出 | npu_out.bin + shmem_*.bin | 无（perf 模式不写文件） |
| verify_result.py | 跑（比对精度） | 不跑（跳过） |
| 用途 | 开发期精度验证 | msprof task 采集 |

---

## 4. msprof 采集命令与数据提取

### 4.1 标准采集命令

```bash
PROJ="$(pwd)"  # 算子工程根目录

msprof --ai-core=on --aic-mode=task-based \
    --output="${PROJ}/docs/perf/round_001" \
    --application="${PROJ}/build/all_to_all_matmul 2048 8192 3584 4 perf"
```

### 4.2 关键参数

| 参数 | 说明 | MC2 场景 |
|------|------|---------|
| `--ai-core=on` | 开启 AI Core 数据采集 | 必设 |
| `--aic-mode=task-based` | 以 task（kernel launch）为单位采集 | 必设——每个 kernel 一条记录 |
| `--output=<dir>` | 输出目录 | 直接归档到 `docs/perf/round_NNN/` |
| `--application="<cmd>"` | 被采集程序（含参数，整体用引号包起来） | 第 5 参必传 `perf` |
| **不用** `--warm-up` | 本示例由程序多轮运行，统计后 5 轮 | L2 flush 单独控制缓存条件 |

### 4.3 输出目录结构

msprof 在 `--output` 指定目录下生成 `PROF_{timestamp}_{pid}/` 子目录：

```
{output}/
└── PROF_{timestamp}_{pid}/
    └── mindstudio_profiler_output/
        ├── op_summary_{ts}_{pid}.csv   # 每个 kernel 的 task 记录（最关键）
        ├── op_stat_{ts}_{pid}.csv      # 按 Op Name 聚合的统计
        └── ...
```

**`op_summary_*.csv` 的关键字段**（实际 header 以本地 CANN 版本为准）：

| 字段 | 含义 |
|------|------|
| `Op Name` | kernel 名（如 `AllToAllQuantMatmulKernelE4M3E4M3`、`heavy_add_kernel`） |
| `Task Duration (us)` | 单次 task 的耗时（微秒） |
| `aiv_time (us)` / `aic_time (us)` | AIV / AIC 流水耗时 |
| `L2 Hit Rate` | L2 cache 命中率 |

### 4.4 多卡数据后处理规则

**核心规则**（多卡并行，整体性能由最慢的卡决定）：

```
1. 每张卡（rank）的 op_summary_*.csv 中：
   - 过滤 Op Name == 主 kernel（如 AllToAllQuantMatmulKernelE4M3E4M3）的记录
   - 共 10 条（对应 PERF_LOOP_COUNT=10）
   - 取最后 5 条的 Task Duration (us) 求平均 → 该卡性能 P_i

2. 整体性能 = max(P_0, P_1, P_2, P_3)
```

**为什么取最后 5 条？** 前 5 轮可能受冷启动 / device 频率未稳定影响，取后 5 轮均值降低前期干扰；若仍不稳定，按 §7 增加运行轮次。

**为什么 4 卡取最大？** MC2 是多卡并行 + 跨卡同步（`aclshmem_barrier_all`），整体性能由最慢的卡（木桶效应）决定。取平均会高估性能。其他卡数按实际 Rank 数取最大值。该口径是各卡均值的最大值，用于单个融合 kernel；多 kernel 或 task 外通信参与时，还需测量完整调用耗时。

比较基线与候选时，保持输入、Rank 数、运行轮次、缓存条件和统计口径一致。

### 4.5 多卡后处理

使用 `ops-profiling` skill 的 `perf_summary.py` 进行多卡数据后处理：

```bash
# 使用 ops-profiling skill 的统一解析工具（生成统计摘要并归档，用法见 ops-profiling/SKILL.md）
python3 ${SKILL_PATH}/scripts/perf_summary.py "$ROUND_DIR" ops/{op_name}
```

> `perf_summary.py` 对 CSV 指标生成 min/avg/max 统计摘要（不做判定）。MC2 场景需特别关注的后处理规则（§4.4）：每卡取最后 5 次 main kernel 的 Task Duration 求平均，4 卡取最大值作为整体性能——该规则需基于其输出的 CSV/摘要手动计算。

也可用本技能的 [extract_perf.py](scripts/extract_perf.py) 直接计算 §4.4 的 last-N / max-rank 指标。它要求每卡独立的 `PROF_*` 目录、各目录一个 `op_summary_*.csv`，并按完整 `Op Name` 过滤主 kernel；不替代公共统计摘要与归档工具。

```bash
python3 /path/to/ascendc-mc2-best-practice/references/shared/scripts/extract_perf.py \
    "$ROUND_DIR" "AllToAllQuantMatmulKernelE4M3E4M3" 5
```

---

## 5. MC2 性能指标判定

### 5.1 性能验收

按计算与通信阶段耗时确认 [Bound 类型](../../../ascendc-perf-optimize/references/comm-compute/bound_diagnosis.md#bound-类型)，指标定义与基线口径统一见 [性能指标](../../../ascendc-perf-optimize/references/comm-compute/index.md#性能指标)，采样与多卡统计沿用 §4.4。

1. **先看加速比**：相对同功能小算子组合的加速比默认要求 **≥1.1**。
2. **未达标再看掩盖率**：默认以 **70%** 为诊断标准；低于该值或缺少同口径阶段数据时，按 §5.2 分析流水与 Tiling 切分。
3. **掩盖率达标但加速比未达标**：检查通算耗时是否高度不均衡、基线是否可比，并按 [膨胀分析](../../../ascendc-perf-optimize/references/comm-compute/expansion_analysis.md) 排查切分带来的额外成本；掩盖率达标不能替代加速比达标。

### 5.2 未达标时的流水分析

用时间线核对通信提交、数据可用、矩阵执行与缓冲释放；AICPU/CCU 通信需观察对应引擎，不能用 AIV 时间代替。

- AIV 的 `aiv_time(us)` ≈ AIC 的 `aic_time(us)` 不能证明重叠；仅凭 AIV/AIC 汇总耗时，不能确定时间线上的通信重叠比例。
- 按 [流水配平](../../../ascendc-perf-optimize/references/comm-compute/pipeline_balancing.md) 结合 Bound 类型分析切分；启动、排空和缓冲复用等待使用 [公式化 Tiling](pipeline_tuning.md#21-公式化-tiling估时与候选) 与 [逐块递推](pipeline_tuning.md#22-主尾块有限缓冲与本地计算) 核对。
- Cube/Vector 流水占比仅作辅助定位，不设统一达标阈值；`vec_ratio` 高不能直接说明通信 Put 过密。

---

## 6. 优化方向速查

| 瓶颈信号 | 优化方向 |
|----------|---------|
| 整体 Task Duration >> 理论耗时 | 按 §5.2 核对模型与时间线；确认等待是瓶颈后，再检查 `CrossCoreWaitFlag` 的 idx 一致性与无谓 wait |
| 时间线显示小消息提交开销较大 | 尝试增大合法 `headMSize`、减少 Put 次数，复测带宽与通算重叠 |
| AIC cube ratio 低 | 区分等待通信、任务数不足与矩阵计算受限，再重新求解 Blaze Tiling（包括 baseK、scaleKL1） |
| L2 命中率过高（>80%） | L2 flush 没起作用，检查 `heavy_add_kernel` 调用与 `CACHE_FLUSH_ELEM_COUNT` |
| 单核不均衡（核间） | 调整 Blaze Tiling 的尾块分裂策略（mBaseTailSplitCnt / nBaseTailSplitCnt） |
| 单卡不均衡（卡间） | 按 §4.4 核对各 Rank 耗时，检查负载分配、同步等待及设备健康与可用核数 |
| AIC 内 rank 循环开销大 | 增大 `headMSize`，减少 tile 数；或合并多次 mmadOp_ 调用 |

详细优化方法见 `ops-profiling/references/optimization_quickref.md`。

---

## 7. 排错速查

| 现象 | 可能原因 | 排查方向 |
|------|---------|---------|
| `op_summary_*.csv` 无主 kernel 记录 | 跑的是 precision 模式 | 第 5 参传 `perf` |
| `op_summary_*.csv` 只有 heavy_add_kernel | 主 kernel 在 precision 路径 | 同上 |
| PROF_xxx 子目录数量 < rankNum | 部分 rank fork 失败 / 提前退出 | 查 host stderr，确认所有子进程都跑完 |
| 主 kernel 记录 < 5 条 | `PERF_LOOP_COUNT < 5` | 调大 `src/*.cpp` 中的 PERF_LOOP_COUNT |
| L2 命中率 >80% | L2 flush 失效 | 检查 `heavy_add_kernel` 的 totalLength 是否大于目标芯片 L2 容量 |
| perf 模式跑得极慢 | Host 下发间隙或设备等待 | 用时间线定位；保留本示例每轮 `aclrtSynchronizeStream`，不删除计时与缓冲复用所需的完成等待 |
| 某张卡的 avg 明显高于其他卡 | 负载、同步等待或设备状态差异 | 按 §4.4 核对各 Rank 耗时，再按 §6 排查 |
| Task Duration 在后 5 轮仍波动大 | device 频率未稳定 / 数据依赖 | 增大 PERF_LOOP_COUNT 到 20，取最后 5 |
| 对照路径（torch/分步实现）trace 出现无关 kernel、计时失真 | 同步方式污染：用空转 kernel（如 `torch.exp` 循环）或全局 sync 做计时同步，会 flush 全局流水线 | 对照实验用 `torch.npu.current_stream().synchronize()` 仅同步当前流；验证 trace 中无对照组无关 kernel |

---

## 8. 后续阅读

| 想了解 | 读 |
|--------|---|
| msprof CSV 字段详解 | `ops-profiling/references/csv_fields_reference.md` |
| 单卡算子的性能优化方法 | `ops-profiling/references/optimization_quickref.md` |
| 通算融合整体架构 | `mc2_architecture.md` |
| `tileCnt` 调优（通算并行旋钮） | `pipeline_tuning.md` |
| 参考工程改造食谱 | `codebase_map.md` |
