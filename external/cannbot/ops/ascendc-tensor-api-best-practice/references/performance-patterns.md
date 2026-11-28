---
type: reference
status: stable
platforms:
- '950'
sources:
- resource: project execution records
  role: primary
- resource: asc-devkit master official samples
  role: supporting
updated_at: '2026-10-29'
---

# 性能瓶颈地图与优化模式（A5 Cube GEMM 家族）

本参考在性能优化和性能验收节点按 implementation_route=tensor_api_custom 条件加载。

> 实证载体：GroupedMatmul 20 case + 官方 matmul_high_performance 9 段递进路径。
>
> 本节优化手段的机理是架构层面（访存计算比、L1 多块缓存、L2 分块、编译期 Tiling、流水重叠），跨 API 通用。官方 matmul_high_performance 用传统 AscendC 高阶 API（CFG_MDL/TCubeTiling/MatmulConfig/IterateAll 等）表达，其中 Tiling 参数名（depthA1/stepKa/dbL0A 等）仅用于描述机理；Tensor API 侧落地走 `make_mmad`/`copy` 的 trait 参数（`unit_flag_mode` 等），映射以 `mmad_tensor_api` / `matmul_mxfp4_tensor_api_high_performance` 样例为准。

## 1. 瓶颈地图（按 case 形态分类）

| case 形态 | 瓶颈 | 机理 |
|---|---|---|
| 大 K（K≥512） | **L1→L0B 流量** | B 每 tile 全 K 深度重复下 L0；L0B 64KB 无法常驻 >256 K 深度的 B 列片 |
| 转置/tw 类 | 转置搬运 + C 写出 | 数据搬运 dominate，算力占比低 |
| 小 shape | launch/tiling 固定开销 | total tiles < 核数 → 核空闲；kernel 时间被固定开销淹没 |
| 大 E（group 数） | tile 调度顺序 | 调度顺序影响 B 的 L1 命中率（见 §3 优化模式 1） |

**定位顺序**：先算 total tiles vs 核数（判小 case 是否调度受限）→ 再看 L0B 重下载数（判大 K 是否流量受限）→ 最后看搬运/写出占比。

## 2. Cube 递进优化序列（matmul 9 段路径）

官方 matmul_high_performance 用 9 个递进 case 展示从基础到极限的完整路径（24 核 4x6 切分，shape 大数据量）。按序叠加、逐步实测，勿跳步。各段实证数据见下表（Task Duration / mac_ratio / 关键降幅）：

| 段 | 手段 | Task Duration | mac_ratio | 关键降幅 |
|---|---|---|---|---|
| 0 | 单核 base[64,64,64] | 基线 | — | — |
| 1 | base 块调优 [128,256,64] | — | — | MTE2 **-66.42%** |
| 2 | 多核切分 2x12 | 12541.22µs | 27.6% | — |
| 3 | 多核切分 4x6 | 12283.84µs | 31.2% | — |
| 4 | MDL 大包搬运 | 5039.86µs | 69.4% | **-58.98%**，MTE2 -52.97% |
| 5 | L1Cache 参数 | 4156.76µs | 85.4% | -17.52%，达 MTE2 bound |
| 6 | L2Cache 分块 | 4088.36µs | 86% | -68.4µs |
| 7 | 常量 Tiling | 4053.44µs | 86.3% | scalar **-44.76%** |
| 8 | UnitFlag | — | — | 计/搬流水重叠 |

### 2.1 base 块选择：访存计算比最小化

- `baseM/baseN/baseK` = 一次 mmad 指令在 L0 上的 M/N/K 轴长度（元素数）。
- **原则**：Cube 计算量相同时，访存数据量最小——base 块越大，单位计算的访存越少（访存计算比越低）。
- **判据**：大 shape 用大 base（受 L0C 256KB 与 mmad m/n/k 范围约束）；小 shape 用大 base 反而浪费（尾块 padding 占比高）。base 块选择在 [tiling-strategy.md](tiling-strategy.md) 的「主块长度反推」之上再平衡访存计算比。
- 实证：base [64,64,64] → [128,256,64]，MTE2 从 582.5ms 降到 195.6ms（-66.42%）。

### 2.2 多核切分

沿 M/N 维把输出切到多核，核数 ≤ 平台可用 AI Core。实证步进 2x12 → 4x6。多核负载与尾核分布见 [tiling-strategy.md](tiling-strategy.md) §4。

### 2.3 MDL 大包搬运（L1 多块缓存）

- **机理**：MTE2 从「每次只搬一个 base 块」改为「L1 缓存多个 base 块、一次搬多块」，显著减少 GM→L1 搬运次数，并配 MTE1 double buffer 让搬入与计算重叠。
- **约束（官方）**：`dbL0A = dbL0B = 2`（MTE1 双缓冲）；`depthA1 / (stepM × stepKa) = 2`；`depthB1 / (stepN × stepKb) = 2`。参数含义：dbL0A/B = MTE1 是否双缓冲；depthA1/B1 = L1 全载 base 块份数；stepM/stepKa/stepN/stepKb = 缓存在 M/K/N 方向的 base 倍数。
- 实证：Case 3→4，Task Duration -58.98%，mac_ratio 31.2%→69.4%，MTE2 -52.97%。

### 2.4 L1Cache 参数优化（扩满 L1）

- **占用量算**：`A 占用 = depthA1 × baseM × baseK × dtype 字节`；`B 占用 = depthB1 × baseN × baseK × dtype 字节`；两者相加 ≤ L1 512KB。
- **做法**：MDL 自动算的 Tiling 未必吃满 L1，手动调大 depthA1/stepKa，把 L1 塞满以最大化搬运力度。
- 实证：depthA1 4→16、stepKa→8，L1 占满（256KB + 256KB = 512KB），Case 4→5 -17.52%，mac_ratio 69.4%→85.4%，随后 mte2_ratio 达 96.3% → 转 L2Cache 缓解。

### 2.5 L2Cache 分块（复用矩阵驻留）

- **L2Cache 128MB**（白皮书基线；样例 README 写 192MB，以设备实测为准），纯读带宽约 GM 的 3~4 倍。
- **机理**：总数据量 > L2Cache 时，把复用方（B）驻留 L2，切分另一方（A 的 M 轴）分块计算，Step2 直接读 L2 命中，避免重复从 GM 拉 B。
- 实证：Case 5→6，-68.4µs（MTE2 -4.05%），mac_ratio 86%。
- 判据：MTE2 bound 且总数据量 > L2Cache 时才上；数据量 < L2Cache 时无收益。

### 2.6 常量 Tiling（编译期 Tiling）

- **机理**：Matmul 初始化与迭代间有大量 Scalar 计算（阻塞 MTE2 流水、抬高头开销）。把 Tiling 参数提前到编译期（constexpr/模板参数），运行时免 Scalar 动态计算。
- 实证：Case 6→7，scalar_time 1753.5→968.6µs（**-44.76%**）。
- Tensor API 侧：`custom_op_tensor_api` 的 constexpr 模板参数即常量 Tiling 形态；对比 `matmul_bias_fusion` 的 `copy_tiling_from_gm`（运行时 Tiling，动态 shape 用）。

### 2.7 UnitFlag（MMAD-FIXPIPE 流水重叠）

- **机理**：并行化计算与搬运（FIXPIPE 写出）流水，消除串行依赖。
- Tensor API 侧：`unit_flag_mode::enable`（`mmad_params` 第 4 参 + `l0c_to_gm/l0c_to_ub` copy 的 `unit_flag` 字段）；默认 `disable`。
- 判据：计算与 Fixpipe 写出存在串行依赖（Fixpipe 等 mmad 完成）时启用，收益在写回占比高的 case。

## 3. 有效优化模式（GroupedMatmul 实证）

### 3.1 连续段 tile 分配（grid-stride → 连续分段）

- **做法**：按核序号连续分配 tile 段（nt 外层），核内沿 N 维复用 B 列片
- **效果**：case 9 提升 **10.4×**（267µs → 25.6µs）
- **机理**：grid-stride 让同一核的 tile 在矩阵上跳跃，B 列片每次都 miss；连续分配让相邻 tile 共享 B 的 L1 驻留

### 3.2 B 多槽 L1 常驻复用

- B 列片按槽位缓存在 L1，跨 tile 复用；**槽位数必须过容量门禁**（见 §4）
- 跨 expert 场景槽位必须区分 expert 维度（否则数值错配，见 precision-model.md §4）

## 4. L1/L0 容量预算门禁（构建期强制）

```
L1 总预算 = 512KB（A5）
B 槽位占用 = 槽数 × 列片尺寸 × dtype
门禁：占用 ≤ 预算 × 安全利用率（实证 459KB / 87.6% 可行）
```

- 超限的失败模式是**全挂 + 超时**（非报错），极难定位——必须构建期 CMake 检查硬失败
- 曾踩坑：8 槽 576KB > 512KB → 有 bias case 全挂 + 305ms 超时；降到 6 槽后正常

## 5. 性能验证纪律

1. **PERF 口径**：warmup 3 + repeat 多次取统计值（aclrtEvent / demo PERF 模式）；单次计时不可作证据
2. **实验账本**：每个优化实验记录「变化 → 结果」；**负结果也记**（例：bf16 base_k=256 无效——mmad k=256 被拆 2×128，误差/性能均不变——但无回归，保留决策写清依据）
3. **数值一致性门禁**：优化后输出必须 bit 级一致（性能优化不得改数值）
4. **优化无回归验证**：动过调度/buffer 策略后全量 case 回归，不只测目标 case（跨 expert 错配类 bug 只在大 E case 显形）

## 事实源

- `$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_high_performance/`
- `$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/07_tensor_api/{mmad_tensor_api, matmul_mxfp4_tensor_api_high_performance}/`
- GroupedMatmul 20 case 实证（本 Skill 项目内部）
- 代表性外部知识卡：`~/cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_224_manual_mmad_real_k_and_transpose_gating.md`、`~/cannbot-knowledge/knowledge/ops/ascendc/concepts/roofline_model.md`。采用前按 [knowledge-integration.md](knowledge-integration.md) 读完整卡并记录 status、来源和平台。

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时在 skill 根目录运行 `scripts/ensure_docs.py` 自动获取。
