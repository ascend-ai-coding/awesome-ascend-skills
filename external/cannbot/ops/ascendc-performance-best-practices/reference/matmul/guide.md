# MatMul 类算子性能优化策略索引

按策略类型查找对应的适用场景与详细设计文档。

适用于 `matmul` 族所有变体：`matmul`、`matmul_mxfp4`、`matmul_a16w16`、`batch_matmul`、`matmul_all_reduce` 等。

| 策略 | 适用场景 | 核心手段 | 详细文档 |
|------|---------|---------|---------|
| **pingpong** | 所有 matmul（基线，其它优化的默认前置） | L1/L0 双缓冲 + MTE2/MTE1/Fixpipe 事件同步流水化；`L1_BUFFER_NUM` 取满足 L1 预算的最大整数 | [pingpong_design.md](pingpong_design.md) |
| **swat** | MN 够并行（`totalBlockCnt > aicNum` 数倍）但存在尾碎片负载不均 | 机制 A Serpentine 行窗口（B 侧复用 + 流式带宽）/ 机制 B 尾块合并 / 机制 C 末轮二维再切分 | [swat_design.md](swat_design.md) |
| **streamk** | MN 欠并行（`totalBlockCnt < aicNum`）+ 长 K（≥ 2048），或 baseM/N 已到 CUBE 粒度下限的 MN 不整除情形 | SK / DP+SK 两子模式：切 K 到空闲核，AIV 从 workspace 归约 | [streamk_design.md](streamk_design.md) |
| **fullload**（A-Full-Load / B-Full-Load） | 有"小侧矩阵"（一侧 ≤ L1/2）+ 对侧循环次数 `T ≥ 2` + 真 MTE2 bound | 小侧矩阵 + 随路 Scale 一次性驻留 L1，消除对侧循环中的重复 GM→L1 搬运 | [fullload_design.md](fullload_design.md) |
| **scale_coalescing** | 假 MTE2 bound（MTE2 busy 高但带宽利用率 < 70%）+ 存在 < 20 KB 的 scale / bias / LUT 小块 | `scaleKL1 = SCALE_L1_BUFFER_NUM × kL1`，把 K 向 `baseK` 切碎的小块合并成一次大 MTE2 | [scale_coalescing_design.md](scale_coalescing_design.md) |
| **mte2_preload** | pingpong 已开 + 各流水 busy ≤ 70%（准无 bound）+ 流水图可见 MTE2_PING/PONG 间 gap + `kL1TileNum ≥ 2` | Kernel 主循环改造为「段 1 首轮 PING / 段 2 预取 PONG / 段 3 消费」三段结构；零 TilingData 数值改动 | [mte2_preload_design.md](mte2_preload_design.md) |
