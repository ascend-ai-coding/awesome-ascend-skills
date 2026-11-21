# 性能基线、瓶颈定位与优化建议总览

本参考在性能优化与性能验收节点按 implementation_route=tensor_api_custom 条件加载。给出「目标在哪 → 瓶颈在哪 → 怎么优化」的方法链，是 [vector-performance-patterns.md](vector-performance-patterns.md)、[tiling-strategy.md](tiling-strategy.md) 的入口调度层。

## 1. 理论参数（roofline 基线）

先算理论上限，才知道优化空间。

- **搬运类流水（MTE1/MTE2/MTE3）理论耗时 = 搬运数据量(Byte) / 理论带宽**
- **计算类流水（Cube/Vector/Scalar）理论耗时 = 计算数据量(Element) / 理论算力**

两点修正（官方强调，易被忽略）：
1. **共享带宽**：MTE2/MTE3 同时 GM 读/写时，耗时 = (MTE2 量 + MTE3 量) / GM 带宽，不能各自按满带宽算。
2. **小块低效**：每次搬运数据块越小，有效带宽利用率越低，实测达不到理论带宽。→ 优先一次搬大块。

**DAV_3510（Ascend 950 系列）整芯片峰值参数速查**（出处：昇腾 950 NPU 架构白皮书；分档值对应不同产品 SKU 的核数，以实际设备为准）：

| 项 | 950PR | 950DT |
|---|---|---|
| 片上内存（GM）带宽 | 1.6（低配 1.4）TB/s · 128GB | 4 TB/s · 144GB |
| L2 Cache | 128（低配 112）MB | 128 MB |
| AI 子系统（1 Cube + 2 Vector） | 32/28 Cube + 64/56 Vector | 36/32/28 Cube + 72/64/56 Vector |
| Cube BF16/FP16 | 432/378 TFLOPS | 486/432/378 |
| Cube TF32 | 216/189 | 243/216/189 |
| Cube INT8 | 865/756 TOPS | 973/865/756 |
| Cube FP8/MXFP8/HiF8 | 865/756 | 973/865/756 |
| Cube MXFP4 | 1730/1513 | 1946/1730/1513 |
| Vector FP32 | 27/23 | 30/27/23 |
| Vector FP16/BF16 | 54/47 | 60/54/47 |
| Vector INT8 | 54/47 TOPS | 60/54/47 |

单 AI Core 存储（白皮书表 4-2）：L1 512KB、L0A 64KB、L0B 64KB、L0C 256KB、UB 512KB。

两点换算注意：
1. **单核算力 ≈ 整芯片峰值 / 核数**（单 Cube BF16 = 432T/32 ≈ 13.5 TFLOPS；单 Vector FP32 = 27T/64 ≈ 0.42 TFLOPS）。
2. **单核可达带宽不能简单用整芯片带宽 / 核数**：HBM 共享 + 访存模式决定，roofline 的「理论带宽」优先用目标平台空核/单核 kernel 实测，整芯片值只作上限锚点。
3. **UB 口径**：白皮书 512KB 为 per AI Core（1 Cube + 2 Vector 子系统）；本 Skill `api-dataflow-layout.md` 用 256KB 为单 Vector core 可用基线。两者差异以实际编译选项与芯片手册复核，不凭白皮书直接改容量门禁。

> roofline 只取方法；上述为整芯片锚点，**per-core 数值一律以目标平台实测为准**。

## 2. 采集手段

| 手段 | 产出 | 用途 |
|---|---|---|
| `msopprof` 上板 Profiling | `op_summary_*.csv`、`PipeUtilization.csv` | 各流水耗时占比、核数、头开销 |
| msOpProf 仿真流水图 | Chrome tracing 可视化 | 看各流水是否断流 |

## 3. 瓶颈定位四方法

拿到数据后按顺序判：

1. **流水占比判 bound 类型**：对比 `aiv_mte2_time / mte3_time / vec_time / scalar_time` 与理论耗时。
   - 某流水实测 ≈ 理论值 → 该流水 bound，已到上限。
   - 搬运流水（MTE）耗时 ≈ 总时长 → **搬运 bound**：下一步做流水优化 + Tiling，把其他流水藏进搬运流水里。
   - 计算流水（vec/cube）占比高 → **计算 bound**：走 [vector-performance-patterns.md](vector-performance-patterns.md)。
2. **核数/Tiling**：`Block Dim` 小于平台可用核数 → 有核空闲、算力浪费 → Tiling 没切满，改 [tiling-strategy.md](tiling-strategy.md)。
3. **仿真流水图看断流**：各流水有规律性断流 → 存在数据依赖/同步等待 → Double Buffer + Tiling + 内存优化。
4. **头开销**：头开销 = 核启动 + 核取址 TLB MISS + 同地址访问冲突 + 变量资源初始化，与计算无关。空核函数（Kernel）的 `TaskDuration` 即启动开销。推理类微秒级算子尤其要看这一项 → 调整核数、核函数型、DCI 编译选项。

## 4. 优化建议总览（按分类路由）

> 本表按官方 optimization_summary 分类归纳；其优化项名以传统 AscendC 表达，括号内为 Tensor API 路线视角的转写。

| 分类 | 优化手段 | 去向 |
|---|---|---|
| Tiling | 核间负载均衡、切满核数 | [tiling-strategy.md](tiling-strategy.md) |
| 头尾开销 | 合适核数/Kernel 型、限制 TilingData 大小、避免管道/缓冲资源在对象内创建（官方 TPipe 项，本路线禁用 TPipe）、删 Workspace 冗余、DCI 编译选项 | 本文 §3-4 |
| 流水编排 | Double Buffer、异步 copy/mmad 免同步（官方 Iterate/IterateAll 项） | [vector-performance-patterns.md](vector-performance-patterns.md) §6、[sync-and-lifecycle.md](sync-and-lifecycle.md) |
| 内存访问 | 一次搬大块、GM 512B 对齐、高效搬运 API、避免同地址访问、合理 L2 CacheMode、共享临时 Buffer、输入/输出缓冲复用（官方 VECIN/VECOUT 项）、缩减 ShapeInfo 维度、避免 UB bank 冲突、L2 Cache 切分 | [api-dataflow-layout.md](api-dataflow-layout.md) |
| 矢量计算 | UB 融合连续 vector 计算、Counter 模式、全局掩码复用、低延迟指令优化归约 | [vector-performance-patterns.md](vector-performance-patterns.md) |
| 矩阵计算 | hifloat8、BT Buffer bias、FP Buffer 随路量化、L0C Buffer 累加、小矩阵长驻 L1、AtomicAdd | [performance-patterns.md](performance-patterns.md)、[precision-model.md](precision-model.md) |

## 5. 达标线与对标（性能验收判据）

性能验收先定「对标谁、算不算达标」，写进 DESIGN §10 基线栏。

**对标对象（选其一）**：

1. **官方参考实现**：同 shape 同 dtype 的官方优化算子耗时（如 torch_npu matmul / 官方高性能样例）——最直接。
2. **roofline 理论值**：单核算力/带宽 × 数据量 = 理论耗时，实测 / 理论 = 利用率。
3. **流水利用率**：Cube 的 mac_ratio、Vector 的 IPC（距理论 2.0）、搬运流水占比。

**达标参考（以目标平台实测校准）**：

| 指标 | 达标参考 | 说明 |
|---|---|---|
| 搬运 bound 带宽利用率 | ≥ 80~90% | mte2/mte3 实测耗时 / 理论搬运耗时 |
| Cube mac_ratio | ≥ 85% | 官方 Matmul 递进序列证到 85.4%，此时 MTE2 bound 96.3%（[performance-patterns.md](performance-patterns.md) §2） |
| Vector IPC | 接近依赖链上限 | GELU 8 步指令链证到 1.25（理论 2.0，受链长限制） |
| 头开销 | 微秒级 kernel ≤ 数十 µs | 空核 TaskDuration 即启动下限 |

**判据**：

1. 实测已达「当前 bound 类型」的理论利用率 → **已收敛**，不再抠计算/搬运，转而评估换算法或精度路线。
2. 实测远离达标线但存在明显单一 bound → 按 §4 分类继续优化，勿跳步。
3. 跨轮波动跨越达标线 → 记录多轮中位数与离散度，不以单次计时定论（[review-checklist.md](review-checklist.md) 性能第 3 条）。

**纪律**：达标线是「收敛信号」而非「必须达标」——硬件极限 case（如 [precision-model.md](precision-model.md) §1 的 bf16 大 K 超标）即使未达标也记录证据、标注「硬件极限」，不得标「未优化」。

## 6. 定位纪律

1. 先采集后动手，不看数据就优化是猜。
2. 先判 bound 类型（搬运/计算/头开销），再选手段，方向错则收益被淹没。
3. 每轮优化后重采数据，对比理论值确认收敛；负收益也要记录。
4. 数值一律以目标平台（DAV_3510）实测为准，不套用官方示例量级。

## 事实源

- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/performance_analysis/{collect_performance_data,analyze_performance_data}.md`
- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/simd_operator_optimization/optimization_summary.md`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时在 skill 根目录运行 `scripts/ensure_docs.py` 自动获取。
