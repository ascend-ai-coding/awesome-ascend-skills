---
name: external-cannbot-ops-ops-profiling
description: NPU 性能采集与分析，用于采集算子性能数据、定位性能瓶颈并给出优化建议。当用户在算子开发过程中提到"上板性能"、"算子性能测试"、"硬件性能验证"、"NPU性能采集"、"NPU
  profiling"等场景时触发。
original-name: ops-profiling
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# 上板性能采集与调优

在真实 NPU 上采集算子性能数据，系统化解读 8 个 CSV 指标文件，判定性能是否达标，定位瓶颈类型，并给出可操作的优化建议。

---

## 适用场景

| 场景 | 说明 |
|------|------|
| 算子开发完成后的性能验收 | 确认算子达到预期性能水平 |
| 性能问题定位 | 通过 CSV 指标精确定位瓶颈 |
| 优化效果验证 | 对比优化前后的 CSV 数据 |
| Agent team 测试阶段 | tester/developer 调用，自动化性能分析 |

---

## 工作流

```
Step 1: 构建算子
    ↓
Step 2: msprof op 采集数据
    ↓
Step 3: 归档数据 + 生成统计摘要
    ↓
Step 4: 读摘要 → 对照性能标准判定（需要时读原始 CSV）
    ↓
Step 5: 如不达标，查阅瓶颈优化速查表
    ↓
Step 6: 修改代码 → 回到 Step 2 重新采集（数据自动归档为新一轮）
```

### Step 1: 构建算子

**直调算子**:
```bash
cd operators/{operator_name} && mkdir -p build && cd build && cmake .. && make -j
```

**aclnn 算子**:
```bash
bash build.sh --pkg --soc=ascend910b --ops={operator_name} --vendor_name=custom -j16
./build_out/*.run --install-path=$CANN
bash build.sh --run_example {operator_name} eager cust --vendor_name=custom
```

### Step 2: msprof op 采集

```bash
# 基本用法
msprof op ./demo

# 推荐用法（含预热 + 指定输出）
msprof op --warm-up=10 --output=./msprof_output ./demo

# 多次运行取均值
msprof op --warm-up=10 --launch-count=5 --output=./msprof_output ./demo
```

**关键参数**:

| 参数 | 说明 | 何时使用 |
|------|------|---------|
| `--warm-up=N` | 预热 N 次后再采集 | **始终建议**，避免 DVFS（动态调频）影响首次运行 |
| `--launch-count=N` | 运行 N 次取均值 | 需要统计稳定性时 |
| `--output=<dir>` | 指定输出目录 | 避免结果散落 |
| 无需 `--soc-version` | 上板自动检测硬件 | — |

**输出**: 在指定目录或当前目录下生成 `OPPROF_{timestamp}_XXX/` 文件夹。

### Step 3: 归档数据 + 生成统计摘要

```bash
# 找到最新 OPPROF 目录
OPPROF_DIR=$(ls -td <output_dir>/OPPROF_* | head -1)

# 归档 CSV + 生成摘要（自动创建 docs/perf/round_NNN/）
python3 {skill_path}/scripts/perf_summary.py $OPPROF_DIR operators/{operator_name}
```

脚本会自动：
1. 在 `operators/{operator_name}/docs/perf/round_NNN/` 创建归档目录（轮次自动递增）
2. 复制所有 8 个 CSV 原始文件到归档目录
3. 生成 `summary.txt` 统计摘要（所有指标的 min/avg/max，**不做判定**）

**Agent 分析流程**：
1. **先读 `summary.txt`** — 获取全局概览（约 30 行紧凑文本）
2. **结合 `references/csv_fields_reference.md`** — 理解各指标含义和阈值
3. **发现异常时读原始 CSV** — 如核间不均衡，Read `PipeUtilization.csv` 查看逐核数据
4. **结合 `references/optimization_quickref.md`** — 定位瓶颈类型和优化方法

> **重要**：`summary.txt` 只是统计聚合，不包含分析判定。所有瓶颈判定、优化建议由 Agent 结合 Step 4 和 Step 5 的标准自主完成。原始 CSV 文件完整保留在同目录下，随时可以 Read。

### Step 4: 性能标准判定

对照下表判定算子性能是否达标。**性能达标 = 最长流水的实际耗时接近理论耗时**。

#### 4.1 总体判定流程

```
读取 OpBasicInfo.csv → 获取 Task Duration 和 Block Dim
    ↓
读取 PipeUtilization.csv → 找到各流水占比最高的单元
    ↓
计算理论耗时（搬运量/带宽 或 计算量/算力）
    ↓
比较实际耗时 vs 理论耗时
    ├── 差距 <20% → 性能达标（已接近硬件极限）
    ├── 差距 20-50% → 有优化空间，查阅瓶颈优化表
    └── 差距 >50% → 严重瓶颈，必须优化
```

#### 4.2 各指标达标标准

| 指标 | 达标条件 | 警告条件 | 严重问题 |
|------|---------|---------|---------|
| **核间负载均衡** | 各核 `ai*_time(us)` 差异 <10% | 差异 10-30% | 差异 >30% |
| **Block Dim** | 等于可用核数（910B: 20~40 核） | 远小于可用核数 | Block Dim = 1 |
| **VEC ratio** | 与算子类型匹配（见 4.3） | VEC ratio >80% | VEC ratio >90% 且无优化空间 |
| **MTE2 ratio** | <30%（计算型算子） | 30-50% | >50%（搬运成为瓶颈） |
| **fixpipe_ratio** | <5% | 5-15% | >15%（地址未对齐） |
| **icache_miss_rate** | <5% | 5-15% | >15%（代码量过大） |
| **bank conflict 总占比** | `aiv_vec_total_cflt_ratio` <5% | 5-15% | >15% |
| **L2 Cache 总命中率** | >80% | 50-80% | <50% |
| **头开销** | <总耗时的 10% | 10-30% | >30% |
| **DoubleBuffer 效果** | MTE2/VEC 重叠 >30% | 重叠 10-30% | 重叠 <5% |
| **带宽利用率** | `bw_usage_rate` >60% | 30-60% | <30% |

#### 4.3 不同算子类型的预期 ratio 分布

| 算子类型 | 主导流水 | 预期 ratio | 异常信号 |
|---------|---------|-----------|---------|
| **Elementwise**（Add/Mul/Relu） | VEC | vec_ratio 50-80% | MTE2 ratio > VEC ratio |
| **Reduction**（ReduceSum/Max） | VEC | vec_ratio 40-70% | scalar_ratio >20% |
| **Activation**（Softmax/Gelu） | VEC | vec_ratio 60-85% | 大量 cast 指令 |
| **MatMul** | CUBE | cube_ratio 40-70% | vec_ratio > cube_ratio |
| **纯搬运**（Transpose/Concat） | MTE2/MTE3 | mte2+mte3 合计 >50% | VEC ratio >30% |

#### 4.4 理论耗时计算

**搬运理论耗时**:
```
理论耗时(us) = 搬运数据量(Byte) / GM 峰值带宽
```

Atlas A2: GM 峰值带宽约 1.8 TB/s = 1.8e12 Byte/s

示例：float 类型 4096x4096 矩阵搬运理论耗时 = 4 * 4096 * 4096 / 1.8e12 = 37.28 us

**计算理论耗时**:
```
理论耗时(us) = 计算元素数 / 理论算力
```

Atlas A2: float32 Vector 峰值算力约 11.06 TOPS

示例：32K float 单指令计算理论耗时 = 32000 / 11.06e12 = 0.003 us

**判定逻辑**:
- 实际 MTE2 耗时 ≈ 搬运理论耗时 → MTE2 已达上限，优化方向是流水编排
- 实际 MTE2 耗时 >> 搬运理论耗时 → MTE2 未达上限，检查对齐和搬运粒度
- 实际 VEC 耗时 ≈ 理论计算耗时 → VEC 已达上限，优化空间有限
- Task Duration ≈ 最长流水耗时 → 流水编排良好，其他流水已被掩盖

### Step 5: 瓶颈定位与优化

确认瓶颈类型后，查阅 `references/optimization_quickref.md` 获取具体优化方法。

**快速查找**:

| 瓶颈类型 | 判定条件 | 首选优化 |
|---------|---------|---------|
| **VEC Bound** | `aiv_vec_ratio` 最高 | UB 融合、减少 Cast、融合指令 |
| **MTE2 Bound** | `ai*_mte2_ratio` 最高 | 增大搬运粒度 ≥16KB、512B 对齐、L2 CacheMode |
| **CUBE Bound** | `aic_cube_ratio` 最高 | L0C 累加、L1 数据复用 |
| **SCALAR Bound** | `ai*_scalar_ratio` >30% | 缩小 TilingData、减少核数 |
| **核间不均衡** | 各核耗时差异 >10% | 调整 Tiling 切分策略 |
| **Bank Conflict** | `vec_bank_cflt_ratio` >5% | 调整 UB 地址、添加 padding |
| **头开销大** | 头开销占比 >30% | 减少核数、缩小 TilingData、TPipe 外置 |
| **DoubleBuffer 未生效** | MTE2/VEC 无重叠 | 检查 InitBuffer 是否设置 bufNum=2 |
| **流水线气泡** | 多单元均 30-50%，无主导 | 增加 workspace 份数、异步迭代 |

### Step 6: 验证优化效果

每次优化后，重新运行 Step 2 + Step 3。数据自动归档为 `round_NNN+1`。

**对比方法**：
```bash
# 对比两轮摘要
diff operators/{operator_name}/docs/perf/round_001/summary.txt operators/{operator_name}/docs/perf/round_002/summary.txt

# 或直接读两个 summary.txt 进行对比分析
```

**对比要点**:
1. Task Duration 是否下降
2. 瓶颈单元的 ratio 是否改善
3. 核间均衡是否改善（aiv_time min/max 差距）
4. 是否引入新的瓶颈

---

## 数据目录结构

### msprof 输出（临时）

```
OPPROF_{timestamp}_XXX/         # msprof op 直接输出（临时目录）
├── dump/                       # 原始性能数据（无需关注）
├── OpBasicInfo.csv             # 算子基本信息（名称、核数、总耗时、频率）
├── PipeUtilization.csv         # 各流水线单元耗时和占比（最重要）
├── ArithmeticUtilization.csv   # Cube/Vector 指令 cycle 占比和计算量
├── Memory.csv                  # 内存读写带宽和数据搬运量
├── MemoryL0.csv                # L0A/L0B/L0C 读写带宽
├── MemoryUB.csv                # UB 读写带宽（Vector/Scalar）
├── L2Cache.csv                 # L2 Cache 命中率
├── ResourceConflictRatio.csv   # Bank conflict 和资源冲突占比
└── visualize_data.bin          # MindStudio Insight 可视化文件
```

### 归档目录（持久）

```
operators/{算子名}/docs/perf/
├── round_001/                  # 第一轮采集（基线）
│   ├── OpBasicInfo.csv         # 完整原始 CSV（从 OPPROF 复制）
│   ├── PipeUtilization.csv
│   ├── Memory.csv
│   ├── ResourceConflictRatio.csv
│   ├── L2Cache.csv
│   ├── ArithmeticUtilization.csv
│   ├── MemoryUB.csv
│   ├── MemoryL0.csv
│   └── summary.txt            # 统计摘要（min/avg/max，不含判定）
├── round_002/                  # 第二轮（优化后对比）
│   ├── *.csv
│   └── summary.txt
└── ...                         # 自动递增
```

各 CSV 文件的完整字段说明 → `references/csv_fields_reference.md`

---

## 上板 vs 仿真选择

| 维度 | 上板 (msprof op) | 仿真 (msprof op simulator) |
|------|-----------------|---------------------------|
| 需要 NPU | 是 | 否 |
| 时序精度 | 真实硬件时序 | 周期级模型估算 |
| 输出 | 8 个 CSV 文件 | CSV + trace.json |
| 指令级流水图 | 需加参数或单独仿真 | 默认输出 |
| **资源冲突数据** | **有**（ResourceConflictRatio.csv）| 无 |
| **L2 Cache** | **真实命中率** | 估算 |
| DVFS 影响 | 有（需 warm-up） | 无 |
| 适合阶段 | 性能验收、生产调优 | 早期开发、指令级调试 |

**建议**: 开发阶段用仿真快速迭代，验收阶段用上板确认真实性能。

---

## 注意事项

1. **必须 warm-up**: 首次运行受 DVFS 影响，耗时偏高。始终使用 `--warm-up=10`
2. **频率检查**: 读取 `OpBasicInfo.csv` 的 `Current Freq` 和 `Rated Freq`，若 Current < Rated，说明芯片未满频运行
3. **MTE2/MTE3 带宽共享**: 同时读写 GM 时，总带宽被共享，理论耗时应按 `(MTE2搬运量 + MTE3搬运量) / GM带宽` 计算
4. **小数据量场景**: 数据量很小时头开销占比会很高，这不一定是算子问题，而是数据量不足
5. **多核同地址访问**: 多核同时读同一 512B 地址范围会被串行化，导致 MTE2 耗时异常

---

## 参考资源

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| `references/csv_fields_reference.md` | 8 个 CSV 文件的完整字段定义和阈值 | Step 4 分析时，需要理解具体字段含义 |
| `references/optimization_quickref.md` | 各瓶颈类型的具体优化方法和案例 | Step 5 定位瓶颈后，查找优化方法 |
| `scripts/perf_summary.py` | 统计摘要生成 + CSV 归档 | Step 3 自动归档和生成摘要 |
