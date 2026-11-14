> 平台：Ascend950。本文件及其同目录资源仅用于此分支；相对脚本路径以本文件所在目录为基准。

# TileLang 算子上板性能采集与调优

执行脚本前，将本 `instructions.md` 所在的 `Ascend950/` 真实目录设为 `PROFILING_SKILL_DIR`；`references/`、`scripts/` 等相对资源路径也以该目录为基准。性能产物写入调用方指定的输出目录。

在真实 NPU 上采集 TileLang kernel 性能数据，系统化解读 8 个 CSV 指标文件，判定性能是否达标，定位瓶颈类型，并给出可操作的优化建议。

## 后端

上游传入 `{tilelang_target}` 时原样使用；独立调用且用户未指定时，先询问 `PTO` 或 `AscendC`。映射：`PTO → pto`，`AscendC → ascend`。预运行、msprof 和复测必须显式设置同一 `TILELANG_DEFAULT_TARGET`，禁止默认、回退或混用后端。

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

多 launch 算子、混合 AIC/AIV 计时或用户要求 pytest 最终验收时，先读 [整链计时与采集](references/kernel-chain-measurement.md)。下面一个 case 一次 kernel 的命令用于单核入口；多核链路按实际匹配 launch 数量与名称集合调整。

```
Step 1: 准备并校验 TileLang profiling 入口
    ↓
Step 2: msprof op 采集指标
    ├── 成功 → 继续
    └── 仅 DBI/插桩/导出故障且直调正常 → 普通 msprof fallback
    ↓
Step 3: 归档数据 + 生成统计摘要
    ↓
Step 4: 读摘要 → 对照性能标准判定（需要时读原始 CSV）
    ↓
Step 5: 如不达标，查阅瓶颈优化速查表
    ↓
Step 6: 修改代码 → 回到 Step 2 重新采集（数据自动归档为新一轮）
```

### Step 1: 准备并校验 TileLang profiling 入口

使用上游工作流从当前 baseline 或优化方案的最新 TileLang 算子源码生成独立 `profiling_file`。该文件按已锁定的 case 顺序内嵌全部 `CASES`（Standard 来自 `cases.csv`，Flash 来自用户确认的参数），只接受 `--device` 以及可选的 `--warm-up`、`--launch-count`，并只调用文件内的本地目标 kernel。默认 `warm-up=0`、`launch-count=1`，供 `msprof op` 在同一进程中为每个 case 调用一次；普通 `msprof` fallback 由入口为每个 case 内部执行指定次数。禁止加入 host 计时作为 kernel 性能结果。

采集前先触发 TileLang JIT 并确认目标 kernel 在 NPU 上实际运行：

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<tilelang_target> \
  python <profiling_file> --device <physical_device_id>
```

每轮 baseline、每个优化方案以及源码修改后的方案，都必须从对应目录的最新源码重新生成 `profiling_file`；禁止复用旧内嵌实现测量新 kernel。若入口直接导入生产 wrapper，也须重新确认导入路径、源码哈希及调用链。入口旁记录算子源码 SHA256、case 清单 SHA256、默认/显式 launch 参数和实际解析出的 kernel 名。

### Step 2: msprof op 采集

```bash
CASE_COUNT=<selected_case_count>

# 一个进程采集全部 case
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<tilelang_target> \
  msprof op --warm-up=10 --launch-count="$CASE_COUNT" --output=<case_output> \
  --kernel-name=<expected_kernel> \
  python <profiling_file> --device <physical_device_id>
```

**关键参数**:

| 参数 | 说明 | 何时使用 |
|------|------|---------|
| `--warm-up=N` | 工具内预热 | 使用 10，避免 DVFS 影响首次运行 |
| `--launch-count=N` | 最多采集 N 次匹配的 kernel launch，不会替应用重复执行 | 单核时为 `CASES` 数量；整链按实际匹配调用总数 |
| `--output=<dir>` | 指定输出目录 | 避免结果散落 |
| `--kernel-name=<name>` | 只采集已校验的目标 TileLang kernel | 单核入口必须使用；整链按上述参考核对过滤能力、核集合和完整次数 |
| 无需 `--soc-version` | 上板自动检测硬件 | — |

执行上板采集前优先只运行一次 `npu-smi info`，依据健康状态、显存和进程信息选择候选物理设备，不逐设备启动探测 kernel。若未安装 `npu-smi`，改用一次 `asys info -r=status -d=0` 检查设备状态；若两个命令都不可用，则跳过状态预检查并自行选择物理设备，不得仅因此阻塞采集。msprof 命令必须移除继承的 `ASCEND_RT_VISIBLE_DEVICES`，由 host adapter 直接选择物理设备；禁止用 `ASCEND_RT_VISIBLE_DEVICES=<非零物理 ID>` 驱动 `msprof op`。正确性 pytest 仍可按其既有规则使用可见设备列表。

不得因为系统中存在任意 `msprof` 进程就全局等待；其他物理设备上的采集不是本次任务的阻塞条件。若日志出现 `GetUserDevIdByDeviceId`、`ErrCode=107001` 或 `Failed to convert the driver device ID`，判定为设备 ID 映射错误，使用新输出目录并在移除可见设备映射、显式选择物理设备后重试，不得归因为并发占用。

**输出**: 在指定目录或当前目录下生成 `OPPROF_{timestamp}_XXX/`；每个 case 对应 `<resolved_kernel>/<launch_index>/`，`launch_index` 按顺序映射到 `CASES`。目录数、索引连续性或普通指标 8 个 CSV 的完整性不满足时停止。

### Step 2.5: 普通 msprof fallback

`msprof op` 是首选。只有同时满足以下条件时才允许 fallback：

1. 同一份最新入口直调编译和运行正常；
2. `msprof op` 失败点明确位于 DBI 初始化、插桩或结果导出，而不是 kernel 编译、设备执行、精度、目标名或 case 映射；
3. 使用全新的输出目录，保持设备、输入、case 顺序和正式采集次数不变。

普通 `msprof` 没有采集前的精确 kernel-name 过滤，因此必须让入口在一个进程中为每个 case 先 warm-up、再正式 launch：

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<tilelang_target> \
  msprof --output=<new_output> \
  --application="python <profiling_file> --device <physical_device_id> --warm-up 10 --launch-count 2" \
  --task-time=on --aic-mode=task-based --aic-metrics=PipeUtilization
```

导出后在 `mindstudio_profiler_output/task_time*.csv`（或当前版本等价的 task/op summary）中按**完整且实际解析出的目标 kernel 名**后过滤。要求匹配行数恰好等于：

```text
case_count * (warm_up + launch_count)
```

匹配行按入口的 case 顺序分组，每组前 `warm_up` 行丢弃，后 `launch_count` 行作为该 case 正式结果。行数、顺序、kernel 名或源码/case SHA 任一不一致时停止，禁止猜测映射。fallback 结果必须标明来源为普通 `msprof`；它仍是设备侧 task/kernel 时间，不是 pytest 总耗时或 host 计时。

普通 `msprof` 若能导出 task-based pipe 指标，可用于同口径趋势与流水判定；不能导出某项 `msprof op` 专属 CSV 时如实标为缺失，不能伪造或把不等价字段混算。基线和候选的最终比较优先使用同一种采集方式；若一方不得不 fallback，另一方也用普通 `msprof` 重采。

错误分类按失败位置而非只看末尾错误码。例如设备初始化阶段的 `507033` 应先检查设备健康与映射，不能归为 kernel 失败；`msprof op` 的 `507035` 只有在直调正常且日志证明位于 DBI/插桩阶段时，才能触发上述 fallback。相同故障重试时始终使用新输出目录。

### Step 3: 归档数据 + 生成统计摘要

```bash
# 找到最新 OPPROF 目录
OPPROF_DIR=$(ls -td <output_dir>/OPPROF_* | head -1)

# 按 launch_index 与 CASES 的映射逐 case 归档
mkdir -p <variant_output_dir>/<case_id>
python3 "$PROFILING_SKILL_DIR/scripts/perf_summary.py" \
  $OPPROF_DIR/<resolved_kernel>/<launch_index> <variant_output_dir>/<case_id> \
  --kernel-name <expected_kernel> --round-name <case_or_round_name>
```

`msprof op` 结果由脚本自动：
1. 在已存在的 `<variant_output_dir>/<case_id>/` 下创建 `docs/perf/<case_or_round_name>/` 归档目录；省略 `--round-name` 时使用自动递增的 `round_NNN`
2. 复制当前 launch 的 8 个 CSV，并将时间戳后缀文件名恢复为标准名称
3. 校验 `<expected_kernel>` 在 `OpBasicInfo.csv` 中唯一匹配；零匹配或多匹配立即失败，避免分析到其他 kernel
4. 生成 `summary.txt` 统计摘要（仅聚合目标 kernel；AIC/AIV 混合核分别统计，**不做判定**）

普通 `msprof` fallback 不使用这个要求 8 个专属 CSV 的脚本。按 Step 2.5 校验并归档原始 task/op summary、目标核过滤结果、case 映射、预热/正式次数和源码/case SHA，逐 case 生成 `summary.txt`；摘要注明采集方式、设备侧耗时及缺失指标，供后续分析使用。

**Agent 分析流程**：
1. **先读 `summary.txt`** — 获取全局概览（约 30 行紧凑文本）
2. **结合 `references/csv_fields_reference.md`** — 理解各指标含义和阈值
3. **发现异常时读原始 CSV** — 如核间不均衡，Read `PipeUtilization.csv` 查看逐核数据
4. **按 4.5 章节核算实际搬运数据量与运算数据量** — 从目标 TileLang 源码枚举 GM 读写张量、Cube/Vector 计算量（区分两者），用于算力与带宽核算
5. **结合 `references/optimization_quickref.md` 和目标 TileLang 源码** — 将 profiling 瓶颈映射为 TileLang Tiling、数据布局、流水和计算实现的优化方向
6. **用中文输出分析文件** — 交付必须使用中文

> **重要**：`summary.txt` 只是统计聚合，不包含分析判定。所有瓶颈判定、优化建议由 Agent 结合 Step 4 和 Step 5 的标准自主完成。原始 CSV 文件完整保留在同目录下，随时可以 Read。

### Step 4: 性能标准判定

用户给定逐 case 耗时、带宽或 pytest 验收口径时，以该目标判定是否达标。下表与理论耗时用于定位优化空间，不能替代用户验收；表内经验阈值不是单项充分诊断条件。

#### 4.1 总体判定流程

```
读取 OpBasicInfo.csv → 获取 Task Duration 和 Block Dim
    ↓
读取 PipeUtilization.csv → 找到各流水占比最高的单元
    ↓
按 4.5 核算实际搬运数据量 + 实际运算数据量（区分 Cube/Vector）
    ↓
计算理论耗时（搬运量/带宽 或 计算量/算力）
    ↓
比较实际耗时 vs 理论耗时
    ├── 差距 <20% → 接近所用模型的估算下界，另核对用户目标
    ├── 差距 20-50% → 评估优化空间，查阅瓶颈优化表
    └── 差距 >50% → 优先排查模型、计时口径和瓶颈
```

#### 4.2 各指标达标标准

| 指标 | 达标条件 | 警告条件 | 严重问题 |
|------|---------|---------|---------|
| **核间负载均衡** | 各核 `ai*_time(us)` 差异 <10% | 差异 10-30% | 差异 >30% |
| **Block Dim** | 等于可用核数（910B: 20~40 核） | 远小于可用核数 | Block Dim = 1 |
| **VEC ratio** | 与算子类型匹配（见 4.3） | VEC ratio >80% | VEC ratio >90% 且无优化空间 |
| **MTE2 ratio** | <30%（计算型算子） | 30-50% | >50%（搬运成为瓶颈） |
| **fixpipe_ratio** | <5% | 5-15% | >15%（检查输出转换、原子写回及对齐，不能直接断言未对齐） |
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

> **前置**：计算理论耗时前，必须先按 4.5 章节核算该算子**实际搬运数据量**与**实际运算数据量**，不能只套峰值公式。运算数据量必须区分 Cube 与 Vector。

**搬运理论耗时**:
**计算前复用统一入口传入的 `full_soc` 探测证据；证据缺失或失效时，先按名称加载 `ascendc-env-check` Skill（仅使用其探测脚本），将该 Skill 真实目录设为 `ASCENDC_ENV_CHECK_DIR`，运行 `python3 "$ASCENDC_ENV_CHECK_DIR/scripts/get_npu_arch.py" --json` 重新探测。不得从当前 Skill 目录拼接跨 Skill 相对路径，也不得从 `npu-arch` 静态映射反推本机型号。仅在脚本退出码为 0、`full_soc` 为 Ascend950PR / Ascend950DT 系列且 `npu_arch=3510` 时消费新证据，再根据完整型号选择下列对应数据。**
```
理论耗时(us) = 实际搬运数据量(Byte) / GM 峰值带宽
```

Ascend950PR: GM 峰值带宽约 1.6 TB/s
Ascend950DT: GM 峰值带宽约 4 TB/s

**计算理论耗时**:
```
理论耗时(us) = 实际计算量(FLOP) / 对应单元理论算力
```

Ascend950PR:
cube_fp16_bf16: 432 TFLOPS; cube_tf32_hf32: 216 TFLOPS; cube_fp32: 26 TFLOPS; cube_int8_fp8: 864 TOPS;
vector_fp16_bf16_add: 27 TFLOPS; vector_fp16_bf16_fma_axpy: 54 TFLOPS; vector_fp32_add: 13.5 TFLOPS; vector_fp32_fma: 13.5 TFLOPS; vector_in32: 13.5 TFLOPS

Ascend950DT:
cube_fp16_bf16: 432 TFLOPS; cube_tf32_hf32: 216 TFLOPS; cube_fp32: 26 TFLOPS; cube_int8_fp8: 864 TOPS;
vector_fp16_bf16_add: 27 TFLOPS; vector_fp16_bf16_fma_axpy: 54 TFLOPS; vector_fp32_add: 13.5 TFLOPS; vector_fp32_fma: 13.5 TFLOPS; vector_in32: 13.5 TFLOPS

#### 4.5 搬运量与计算量核算（算力/带宽分析前提）

> 上板数据只有真实耗时，**搬运量、计算量必须从算子源码自行核算**，据此才能算出有效带宽、实际算力和理论耗时。运算数据量必须区分 Cube 与 Vector。

##### 4.5.1 实际搬运数据量核算（GM↔UB）

从算子源码枚举所有 GM 张量的读写，逐个核算字节数：

| 方向 | 张量 | 字节计算 |
|------|------|---------|
| 读 | 每个输入 | 元素总数 × 单位元素字节数 |
| 写 | 每个输出 | 元素总数 × 单位元素字节数 |

- 元素总数 = 张量全部维度之积（`prod(shape)`）；单位元素字节数由 dtype 决定（如 fp32=4、bf16=2）。
- 累加得「读总量 / 写总量 / 总搬运量」。
- **多输入/多份数据**：若某张量在 GM 中按多份存在（如按维度切分的多片、多份中间结果），搬运量须计入全部份，不能只算一份。
- 与 msprof `Memory.csv` 的 `read_main_memory_datas` / `GM→UB` 等字段交叉验证，确认核算口径。

##### 4.5.2 实际运算数据量核算（区分 Cube / Vector）

从源码判定算子类型与计算量归属：

- **Cube 计算量**：`T.gemm` 路径 → 每 tile `2 × M × N × K` FLOP；无 gemm 则为 0。
- **Vector 计算量**：`T.SimdVF` / `T.Parallel` 内逐元素运算，按「每元素运算次数 × 元素数」核算逻辑 FLOP。

判定依据（与 `ArithmeticUtilization.csv` 交叉验证）：
- `aic_cube_fops > 0` / `aic_mac_ratio > 0` → 有 Cube 计算
- `aic_cube_fops = 0` → 纯 Vector 算子，Cube 计算量 = 0

##### 4.5.3 算力与带宽核算

```
有效带宽(GB/s)   = 总搬运字节 / 实测稳态耗时
带宽利用率       = 有效带宽 / GM 峰值带宽
实际算力(FLOP/s) = 实际计算量 / 实测稳态耗时
```

上式按源码逻辑字节计算的比例仅称“有效带宽比”，不是物理 HBM 利用率；后者需要 profiler 的实际 HBM 流量与精确设备 SKU 峰值。共享缓存、原子读改写和多次消费会使两种字节口径不同，不能混用。

- 只读 / 只写按对应方向核算；读写同时发生时注意 MTE2/MTE3 带宽共享。
- 计算密集型算子用实际算力 vs 理论算力判断是否接近计算上限。
- 小数据量算子优先看搬运理论耗时与头开销占比，不要单看带宽利用率。

**判定逻辑**:
- 实际 MTE2 耗时 ≈ 搬运理论耗时 → MTE2 已达上限，优化方向是流水编排
- 实际 MTE2 耗时 >> 搬运理论耗时 → MTE2 未达上限，检查对齐和搬运粒度
- 实际 VEC 耗时 ≈ 理论计算耗时 → VEC 已达上限，优化空间有限
- Task Duration ≈ 最长流水耗时 → 流水编排良好，其他流水已被掩盖

### Step 5: 瓶颈定位与优化

确认瓶颈类型后，查阅 `references/optimization_quickref.md`，结合目标 TileLang 源码制定具体优化方法。

**快速查找**:

| 瓶颈类型 | 判定条件 | 首选优化 |
|---------|---------|---------|
| **VEC Bound** | `aiv_vec_ratio` 最高 | UB 融合、减少 Cast、融合指令 |
| **MTE2 Bound** | `ai*_mte2_ratio` 最高 | 增大连续搬运粒度、检查对齐、Tile 复用和 L2 命中 |
| **CUBE Bound** | `aic_cube_ratio` 最高 | 调整 Tile shape，提升 L0/L1 数据复用 |
| **SCALAR Bound** | `ai*_scalar_ratio` >30% | 减少动态分支、标量循环和运行时索引计算 |
| **核间不均衡** | 各核耗时差异 >10% | 调整 Tiling 切分策略 |
| **Bank Conflict** | `vec_bank_cflt_ratio` >5% | 调整 TileLang 布局、stride 或 padding |
| **头开销大** | 头开销占比 >30% | 减少核数和动态分支，简化 kernel 启动路径 |
| **DoubleBuffer 未生效** | MTE2/VEC 无重叠 | 调整 pipeline stage，检查搬运与计算重叠 |
| **流水线气泡** | 多单元均 30-50%，无主导 | 调整 pipeline stage、Tile 切分和异步迭代 |

### Step 6: 验证优化效果

每次优化后，重新运行 Step 2 + Step 3。省略 `--round-name` 时数据自动归档为 `round_NNN+1`；显式命名时必须为新目录，脚本拒绝覆盖旧归档。

**对比方法**：
```bash
# 对比两轮摘要
diff <variant_output_dir>/docs/perf/round_001/summary.txt <variant_output_dir>/docs/perf/round_002/summary.txt

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
OPPROF_{timestamp}_XXX/<resolved_kernel>/
├── 0/                          # CASES[0]
│   ├── OpBasicInfo_<timestamp>.csv
│   ├── PipeUtilization_<timestamp>.csv
│   └── ...                     # 共 8 个指标 CSV
├── 1/                          # CASES[1]
└── ...
```

### 归档目录（持久）

```
<variant_output_dir>/docs/perf/
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
| `references/optimization_quickref.md` | profiling 瓶颈到 TileLang Tiling、buffer、流水、布局和计算实现的优化映射 | Step 5 定位瓶颈后，制定具体 TileLang 修改方案 |
| `scripts/perf_summary.py` | 统计摘要生成 + CSV 归档 | Step 3 自动归档和生成摘要 |

> 同一核心、同一时间分母下的流水 ratio 之和可作重叠线索，不能仅凭约 100% 断言完全串行，也不能把 130% 当作硬门槛。结合生成代码的事件/版本轮转和可用 timeline 验证；AIC/AIV 的比例不直接相加。理想重叠估算只是候选收益上界，仍受依赖、共享带宽和同步约束。
