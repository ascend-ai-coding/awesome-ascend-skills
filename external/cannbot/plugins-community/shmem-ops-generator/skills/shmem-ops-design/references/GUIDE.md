# References Guide

本目录包含 `shmem-ops-design` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [op-dsl.md](op-dsl.md) | Canonical DSL 字段定义和归一化规则 | Step 1 输入规范化时 |
| [capacity.md](capacity.md) | SHMEM 已有能力、功能域和边界 | Step 2 能力盘点时 |
| [core-allocation.md](core-allocation.md) | 分核、phase、chunk、通算 overlap 策略 | Step 3 执行方案设计时 |
| [design-template.md](design-template.md) | `design.md` 输出模板 | Step 4 生成契约输出时 |
| [implementation-boundary.md](implementation-boundary.md) | Device/Host/Python 实现边界、统一 kernel 路径 | Step 3 执行方案设计时 |
| [hardware-architecture.md](hardware-architecture.md) | AI Core、存储层次、传输引擎、同步机制 | Step 2-3 硬件约束参考 |
| [../../shmem-ops-dev/references/shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO` | Read `${SHMEM_REPO}/examples/`、`docs/` 对照能力时 |
| [../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md](../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md) | 仓内文档与 examples 索引 | capacity 中示例路径不确定时 |

通算融合交接（`meta.op_kind=fused_compute_comm` 时 design 完成后由 code-gen 消费）：

| 文件 | 用途 |
| --- | --- |
| [../../shmem-ops-code-gen/templates/fused-compute/GUIDE.md](../../shmem-ops-code-gen/templates/fused-compute/GUIDE.md) | AIC/AIV 通算融合代码模板 |
| [../../shmem-ops-dev/references/fused-compute-phase-gates.md](../../shmem-ops-dev/references/fused-compute-phase-gates.md) | 通算融合开发规范 |
| [../../shmem-ops-performance-optim/references/compute-optimization.md](../../shmem-ops-performance-optim/references/compute-optimization.md) | Phase 6.5 compute 瓶颈调优（TileShape、Swizzle 等） |
