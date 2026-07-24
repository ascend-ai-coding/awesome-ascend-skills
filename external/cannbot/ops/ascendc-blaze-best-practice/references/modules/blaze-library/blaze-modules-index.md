# Blaze 库模块索引

> **定位**：Blaze 库各层模块的查阅索引。不重复 ops-tensor 仓已有的文档内容，只提供索引和查阅路径。

---

## §1 概述

ops-tensor 仓已克隆到本地（由 plugin init.sh 自动完成），文档位于 `ops-tensor/docs/API/`。

Blaze 库模块按 Kernel / Block / Tile / Epilogue / Policy 五层组织。各层的详细 API 文档、模板签名、参数说明和约束条件，请直接查阅 ops-tensor 仓对应文档。

---

## §2 Kernel 层索引

| 文档 | ops-tensor 路径 | 说明 |
|------|----------------|------|
| Kernel 公共框架 | `docs/API/gemm/kernel/kernel.md` | 统一模板参数、Params 结构、核心方法 |
| KernelMatmulBasic | `docs/API/gemm/kernel/kernel_matmul_basic.md` | 基础矩阵乘 Kernel，仅 AIC |
| KernelQbmmCube | `docs/API/gemm/kernel/kernel_qbmm_cube.md` | Fixpipe 量化 Batch Matmul，仅 AIC |
| KernelQbmmMix | `docs/API/gemm/kernel/kernel_qbmm_mix.md` | MIX 量化 Batch Matmul（AIC+AIV） |
| KernelQbmmMixWithoutBatch | `docs/API/gemm/kernel/kernel_qbmm_mix_without_batch.md` | MIX 量化单 Batch Matmul（常用） |
| KernelQbmmMx | `docs/API/gemm/kernel/kernel_qbmm_mx.md` | MX 量化 Batch Matmul |
| KernelQbmmMxWithoutBatch | `docs/API/gemm/kernel/kernel_qbmm_mx_without_batch.md` | MX 量化单 Batch Matmul（常用） |
| KernelMatmulStreamK | `docs/API/gemm/kernel/kernel_matmul_streamk.md` | StreamK 矩阵乘 |

---

## §3 Block 层索引

### BlockMmad

| 文档 | ops-tensor 路径 | 说明 |
|------|----------------|------|
| BlockMmad 公共框架 | `docs/API/gemm/block/block_mmad.md` | 统一模板参数（9 参数）、Params、Init/operator() |
| BlockMmadMatmulBasic | `docs/API/gemm/block/block_mmad_matmul_basic.md` | 基础矩阵乘 Block，L1/L0 双缓冲 |
| BlockMmadA8W8FixpipeQuant | `docs/API/gemm/block/block_mmad_a8w8_fixpipe_quant.md` | A8W8 Fixpipe 反量化 Block（仅 AIC） |
| BlockMmadA8W8Mix | `docs/API/gemm/block/block_mmad_a8w8_mix.md` | A8W8 MIX Block（L0C→UB，AIV 反量化） |
| BlockMmadQbmmMx | `docs/API/gemm/block/block_mmad_qbmm_mx.md` | MX 量化 Block，Scale 反量化 |
| BlockMmadMatmulStreamK | `docs/API/gemm/block/block_mmad_matmul_streamk.md` | StreamK Block，workspace 输出 |

### BlockScheduler

| 文档 | ops-tensor 路径 | 说明 |
|------|----------------|------|
| Scheduler 公共框架 | `docs/API/gemm/block/block_scheduler.md` | 统一模板参数、Z 型扫描、尾块处理 |
| BlockSchedulerMatmulBasic | `docs/API/gemm/block/block_scheduler_matmul_basic.md` | Basic 调度器，尾块切分、SplitK |
| BlockSchedulerMatmulStreamK | `docs/API/gemm/block/block_scheduler_matmul_streamk.md` | StreamK 调度器，DP+SK 混合 |
| BlockSchedulerQbmmMx | `docs/API/gemm/block/block_scheduler_qbmm_mx.md` | QBMM 调度器，Batch 维度切分 |

---

## §4 Tile 层索引

| 文档 | ops-tensor 路径 | 说明 |
|------|----------------|------|
| TileMmadMX | `docs/API/gemm/tile/tile_mmad_mx.md` | MX Mmad Trait 定义 |
| PadMxKL1 | `docs/API/gemm/tile/pad_mx_kl1.md` | MX K 轴 Padding，L1 数据对齐补零 |

---

## §5 Epilogue 层索引

| 文档 | ops-tensor 路径 | 说明 |
|------|----------------|------|
| BlockEpilogue 公共框架 | `docs/API/epilogue/block/block_epilogue.md` | 统一类型别名、Params、核心方法 |
| BlockEpilogueEmpty | `docs/API/epilogue/block/block_epilogue_empty.md` | 空后处理，用于不支持后处理的 Kernel |
| BlockEpilogueDequant | `docs/API/epilogue/block/block_epilogue_dequant.md` | MIX 反量化 Epilogue（AIV，支持 per-token/per-channel/per-tensor） |
| BlockEpilogueStreamK | `docs/API/epilogue/block/block_epilogue_matmul_streamk.md` | StreamK 后处理，workspace 汇聚 |

---

## §6 Policy 层

Policy 定义在源码中：`include/blaze/gemm/policy/dispatch_policy.h`

| Policy | ScheduleType | 适用场景 |
|--------|-------------|---------|
| `MatmulMultiBlockBasic` | `KernelMmadMultiBlockBasic` | 普通 matmul |
| `MatmulMultiBlockWithStreamK` | `KernelMultiBlockStreamK` | StreamK 大 K |
| `MatmulWithScaleFixpipeQuant` | `KernelMmadWithScaleFixpipeQuant` | A8W8 量化（Fixpipe 反量化，仅 AIC） |
| `MatmulWithScaleMix` | `KernelMmadWithScaleMix` / `KernelMmadWithScaleMixWithoutBatch` | A8W8 量化（MIX，AIC+AIV） |
| `MatmulWithScaleMx` | `KernelMmadWithScaleMx` 或 `KernelMmadWithScaleMxWithoutBatch` | MX 量化 |

---

## §7 场景推荐矩阵

| 开发场景 | 推荐路径 | 推荐 Kernel | 推荐 BlockMmad | 推荐 Scheduler | 推荐 Tiling |
|---------|---------|------------|---------------|---------------|------------|
| 普通 MatMul 单算子 fp16/bf16/fp32 | **blaze 库** | `GemmUniversal` | `Blaze::Gemm::Block::BlockMmad` (Basic) | `BlockSchedulerMatmulBasic` | Basic MatMul tiling |
| A8W8 量化 matmul（per-tensor x1Scale） | **blaze 库** | `GemmUniversal` | `Blaze::Gemm::Block::BlockMmad` (FixpipeQuant) | `BlockSchedulerQuantBatchMatmulV3` | `MatmulTilingSwat` |
| A8W8 量化 matmul（全部量化模式） | **blaze 库** | `QbmmMixWithoutBatch` | `Blaze::Gemm::Block::BlockMmad` (Mix) | `BlockSchedulerQuantBatchMatmulV3` | `MatmulTilingSwat` |
| MX 量化 matmul fp8/fp4 | **blaze 库** | `GemmUniversal` | `Blaze::Gemm::Block::BlockMmad` (ScaleMx) | `BlockSchedulerQuantBatchMatmulV3` | `QuantMatmulTilingSwat` |
| matmul + epilogue 融合 | blaze_custom | `MatmulKernelFused` | `Block::BlockMmad` (NO_FULL_LOAD) | `MatmulBlockScheduler` | `MatmulTilingSwat` |
| Grouped matmul | blaze_custom | `GroupMatmulKernel` | `Block::BlockMmad` (NO_FULL_LOAD) | `GroupMatmulBlockSchedulerSplitM` | `MatmulTilingSwat` |

A8W8 量化 matmul 两条路径的选择规则详见 `references/scenarios/a8w8-quant-matmul-development.md` §2。普通 MatMul 单算子优先查阅 `KernelMatmulBasic`、`BlockMmadMatmulBasic`、`BlockSchedulerMatmulBasic`、`BlockEpilogueEmpty`。blaze_custom 模块的详细使用说明见 `references/modules/blaze-custom/` 目录。
