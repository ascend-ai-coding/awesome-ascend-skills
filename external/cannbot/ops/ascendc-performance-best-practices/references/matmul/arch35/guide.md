# MatMul 类算子性能优化策略索引

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

按策略类型查找对应的适用场景与详细设计文档。

适用于 `matmul` 族所有变体：`matmul`、`matmul_mxfp4`、`matmul_a16w16`、`batch_matmul`、`matmul_all_reduce`、`grouped_matmul` 等。

> **建模先行**：选型前先按 [performance_modeling.md](performance_modeling.md) 完成各流水理论耗时评估与 Bound 判定（含真/假 MTE2 Bound、VF Bound 区分），再按下表选择策略。基本块（baseM/baseN/baseK）与切分层级（kL1/scaleKL1）的选型约束见 [basic_block_design.md](basic_block_design.md)，各策略文档中的基本块参数以其为准。

| 策略 | 适用场景 | 核心手段 | 详细文档 |
|------|---------|---------|---------|
| **pingpong** | 所有 matmul（基线，其它优化的默认前置） | L1/L0 双缓冲 + MTE2/MTE1/Fixpipe 事件同步流水化；`L1_BUFFER_NUM` 取满足 L1 预算的最大整数（选型见 [basic_block_design.md](basic_block_design.md)） | [pingpong_design.md](pingpong_design.md) |
| **swat** | MN 够并行（`totalBlockCnt > aicNum` 数倍）但存在尾碎片负载不均 | 机制 A Serpentine 行窗口（B 侧复用 + 流式带宽）/ 机制 B 尾块合并 / 机制 C 末轮二维再切分 | [swat_design.md](swat_design.md) |
| **streamk** | MN 欠并行（`totalBlockCnt < aicNum`）+ 长 K（≥ 2048），或 baseM/N 已到 CUBE 粒度下限的 MN 不整除情形 | SK / DP+SK 两子模式：切 K 到空闲核，AIV 从 workspace 归约（baseK 上限公式见 [basic_block_design.md](basic_block_design.md) §2.2） | [streamk_design.md](streamk_design.md) |
| **fullload**（A-Full-Load / B-Full-Load） | 有"小侧矩阵"（一侧 ≤ L1/2）+ 对侧循环次数 `T ≥ 2` + 真 MTE2 bound | 小侧矩阵 + 随路 Scale 一次性驻留 L1，消除对侧循环中的重复 GM→L1 搬运 | [fullload_design.md](fullload_design.md) |
| **scale_coalescing** | 假 MTE2 bound（MTE2 busy 高但带宽利用率 < 70%）+ 存在 < 20 KB 的 scale / bias / LUT 小块 | `scaleKL1 = SCALE_L1_BUFFER_NUM × kL1`，把 K 向 `baseK` 切碎的小块合并成一次大 MTE2 | [scale_coalescing_design.md](scale_coalescing_design.md) |
| **mte2_preload** | pingpong 已开 + 各流水 busy ≤ 70%（准无 bound）+ 流水图可见 MTE2_PING/PONG 间 gap + `kL1TileNum ≥ 2` | Kernel 主循环改造为「段 1 首轮 PING / 段 2 预取 PONG / 段 3 消费」三段结构；零 TilingData 数值改动 | [mte2_preload_design.md](mte2_preload_design.md) |
| **unitflag** | MMAD 与 FIXPIPE 串行执行（FIXPIPE Bound 或 L0C 被完整输出块占满无法开 Double Buffer） | MMAD/Fixpipe 按 512B 粒度硬件级同步：尾轮 `FINAL_ACCUMULATION(3)`、非尾轮 `NO_FINAL_ACCUMULATION(2)`，删除整段 `M_FIX` 事件 | [unitflag_design.md](unitflag_design.md) |
| **k_tail_stepwise_copyout** | UnitFlag 已开后仍存在下一 Tile 复用 L0C 的读写冲突（MMAD 空洞累积）；完整输出块放不进半个 L0C；MXFP4 内轴对齐 256B / MXFP8 对齐 128B | K 尾轮循环顺序 `KL1 -> N Split -> KL0`：N 子块全 K 累加完成立即搬出，用第二子块计算遮盖搬出尾部 | [k_tail_stepwise_copyout_design.md](k_tail_stepwise_copyout_design.md) |
| **l1_bank_conflict** | MTE1 Bound / MMAD 断流：L1 双 Bank（各 256KB）下 Ping/Pong 顺序排布引发 MTE2 写与 MTE1 读同 Bank 冲突 | Half-L1 Ping / Half-L1 Pong 布局：`l1Offset = (TOTAL_L1_SIZE >> 1) * l1BufId`，读写分居不同 Bank | [l1_bank_conflict_design.md](l1_bank_conflict_design.md) |
| **weightnz** | MTE2 Bound + 权重固定的推理场景（训练场景禁用） | 权重离线预转 FRACTAL_NZ（FP16/BF16 基本块 16×16，FP32 16×8），GM→L1 退化为普通 DataCopy，消除 ND2NZ 随路转换 | [weightnz_design.md](weightnz_design.md) |
| **grouped_matmul** | `grouped_matmul` 变体（MoE 多专家）分组场景独有：组间核分配不连续、M 轴 group_list 快慢核、小 m 大 weight 的 L2 污染、3-Buffer 容量 | 组间连续核分配 / M 轴均衡 / 双页表 bypass L2 / 3-Buffer 容量约束 | [grouped_matmul_design.md](grouped_matmul_design.md) |

## Blaze 路线（实现范式选择，先于上表策略判定）

上表策略均是对现有 AscendC 实现的调参增强。当目标架构为 Ascend950/DAV_3510 时，matmul 族（含 quantized/MX/融合变体）选路必须**先完成 Blaze 路线判定**（`blaze_native` / `blaze_custom` / `blaze_migration` / `unsupported`），判定为 `unsupported` 时记录 `unsupported_points` 证据后继续 AscendC 策略选型；Blaze 路线与 AscendC 策略并列作为候选方案，不因判定通过而强制选用。ops-tensor 只读且 commit 进证据身份。详见 [blaze_design.md](blaze_design.md)。

## 性能建模入口

- [performance_modeling.md](performance_modeling.md)：统一建模方法论——各流水理论耗时公式（MMAD/MTE2/MTE1/FIXPIPE/VF）、Bound 判定条件化简、Bound 类型 → 策略映射表、多分组（GroupedMatmul）组间时间加权建模。所有策略选型以该文档的 Bound 判定为前提。
- [basic_block_design.md](basic_block_design.md)：基本块与切分层级基础——三级切分体系（m/k/n → kL1 → baseM/K/N → m0/k0/n0）、各 Tensor 搬运排布变换、基本块硬约束（分型粒度 / L0A·L0B·L0C 容量 / MX 对齐补零）、Bound 驱动的调整方向、L1 联动预算、尾块处理、Host 侧确定流程与回退链、典型配置参考。
