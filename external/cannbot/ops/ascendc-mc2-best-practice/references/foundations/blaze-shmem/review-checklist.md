# blaze-shmem 路线代码审查验收条件

> Reviewer 在 Step 4 逐项检查。违反任意红线项 = FAIL。

## 红线项

| # | 检查项 | 验收条件 |
|---|--------|---------|
| R1 | 架构白名单 | CMakeLists.txt 中 `npu-arch` 落在 skill 声明的支持架构集合内（见 [`capability-declaration.md`](../../capability-declaration.md)） |
| R2 | 无 HCCL 高阶 API | 代码中不含 `Hccl::`（完整禁止清单见 [`comm_shmem.md`](comm_shmem.md) §5） |
| R3 | 无 asc-devkit matmul | 代码中不含 `AscendC::Matmul` |
| R4 | 通信走 SHMEM | 头文件含 `shmem.h`，device 侧用 `aclshmemx_udma_*`/`aclshmem_barrier_all`（`aclshmemx_barrier_all_vec` 已废弃，见 [`comm_shmem.md`](comm_shmem.md) §4.4） |
| R5 | Matmul 走 Blaze | 头文件含 `blaze/gemm/block/block_mmad*.h` |
| R6 | L2 flush 证据 | 代码中含 L2 flush kernel 调用或等效实现（见 [`profiling_mc2.md`](../../shared/profiling_mc2.md)） |

## 流程门禁

| # | 检查项 | 验收条件 |
|---|--------|---------|
| R7 | 流程门禁完整 | `docs/` 下 DESIGN/PLAN/WALKTHROUGH/REVIEW.md 齐全；环境检查通过 |

## 常见 FAIL 原因

| 现象 | 根因 | 修复方向 |
|------|------|---------|
| 代码中含 `Hccl::AllReduce` | 开发者把 HCCL 当成"熟悉的 API"用了 | 要求 Developer 改写为 `aclshmemx_udma_put_nbi` + 自实现 Reduce 逻辑 |
| 代码中含 `AscendC::Matmul` | 开发者误用 asc-devkit 接口 | 替换为 `Blaze::Gemm::Block::BlockMmad` |
| SHMEM 空间不足崩溃 | DESIGN.md 的空间预算没算对 | 重算 `SHMEM_SPACE_SIZE` |
| 精度对不上但无报错 | 多半是 ProcessSingleBatch 中 rank==rankId 分支错（未切换到本卡 GM）/ `remoteRankCnt` 没从 0 起算 | 对照 `qbmm_mx_kernel.h` 注释核对 |
