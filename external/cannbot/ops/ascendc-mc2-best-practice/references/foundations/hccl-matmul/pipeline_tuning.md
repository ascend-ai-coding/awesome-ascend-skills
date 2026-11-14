# hccl-matmul 通算并行调优（tileCnt，910B / A2）

两阶段策略（`tileCnt=1` 串行基线 → 扫描找最优）与 `headMSize = M / tileCnt` 概念复用 [`../../shared/pipeline_tuning.md`](../../shared/pipeline_tuning.md)，**本文不重复**。下文只写本路线差异。

> 蓝本：[`all_gather_matmul/`](all_gather_matmul)。A2 上 `algConfig` 为预留字段，仅 FullMesh。

## 1. 本路线的 tileCnt 含义

`tileCnt` = M 轴切分块数，作为 `AllGather<true>(..., repeat=tileCnt)` / `AllReduce(..., repeat=tileCnt)` 的 repeat；逐 tile `Commit`/`Wait` 与 Matmul 流水。蓝本代码以 `tileTiling.M` / `tailTiling.M` 体现单 tile M 长度，无 `headMSize` 字面量。

## 2. A2 特有约束

| 约束 | 说明 | 违反后果 |
|------|------|----------|
| 一通信域 Prepare ≤ 63 | A2 上限（不含 A3 的 `InterHcclGroupSync`） | Prepare 失败 |
| 扫描候选 | `{1,2,4,8,16}`（`MAX_TILE_CNT=16`） | 超过公式化 tiling 默认上限 |
| NZ 外轴 16 对齐 | 单 tile M 按 dtype block 对齐 | MMAD 报错 |
| UB 192KB / L0C 128KB | 单 tile 须 fit | OOM / tiling 失败 |
| 无 doublering | A2 仅 FullMesh | 误传也不生效 |

**无需 950 的 L2 flush**：本路线通信由 AICPU 执行，不占 AI Core L2 residency。采集流程仍复用 [`../../shared/profiling_mc2.md`](../../shared/profiling_mc2.md) 的 msprof task-based 与多卡 last-N 取 max；跳过 `heavy_add_kernel`。

## 3. 公式化 tiling 系数（MC2-repo 辅助，非官方 API）

蓝本权威副本：`all_gather_matmul/x_all_gather_matmul/mc2_common/op_host/op_tiling/`。系数以头文件为准，暂不需真机校准。

| 来源 | 常量 |
|------|------|
| `hccl_formulaic_tiling.h` | `MAX_TILE_CNT=16`、`commGrowRatio=1.15`、`ALLGATHERMM_COMMTIME_FACTOR=2`、`ALLREDUCE_COMMTIME_FACTOR=2.0517`、`REDUCESCATTER_COMMTIME_FACTOR=2.09136` |
| `hccl_performance.h` | `HCCL_MIN_TILE_LEN=64KB`、`FULL_MESH_TIME_FACTOR=2.0`、`FITTING_RANK=8`、`LOCAL_REDUCE_FACTOR=0.4` |
| `matmul_performance.h` | `COMPUTES_PER_CYCLE=4096`、`MARK_CORE_NUM_SOC910B=20`、`CYCLE_PER_MICRO_SEC=1.8*1024`、`MAX_CUBE_UTIL=0.95`、`AVERAGE_CUBE_UTIL=0.75` |
| `mc2_tiling_utils.h`（仅定义、蓝本无其它引用） | `AICPU_NUM_BLOCKS_A2=6`、`ALL_GATHER_HCCL_MEM_LIMIT=256MB`、`ALL_GATHER_HCCL_NUM_LIMIT=16` |

A2 子类 `XAllGatherPlusMMA2A3` 硬编码 `SocVersion::SOC910_B`。A3 的额外 `*0.6` 因子不适用。

## 4. 调优决策（本路线）

```
tileCnt=1 串行基线跑通（精度 + R1–R8）?
  ├─ 否 → 修精度/审查
  └─ 是 → 扫描 tileCnt ∈ {1,2,4,8,16}
            ├─ cube_ratio<40% → 增 tileCnt
            ├─ cube_ratio>70% → 减 tileCnt 或关 L2CACHE
            └─ 选最优 → 写入 DESIGN.md
```

910B 核数：910B2 为 24 Cube / 48 Vector；910B3/B4 为 20/40。`SetBlockDim` 据实核数。
