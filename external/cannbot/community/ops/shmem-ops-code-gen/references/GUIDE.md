# References Guide

本目录包含 `shmem-ops-code-gen` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [internal-api-boundary.md](internal-api-boundary.md) | **禁止** `aclshmemi_*`、deprecated barrier、quiet 误用 | 代码生成与走读前 |
| [api.md](api.md) | SHMEM API 分类和选择参考 | 选择 lifecycle/memory/transport/sync API 时 |
| [code-patterns.md](code-patterns.md) | Host/Device 代码组织模式 | 步骤 4 代码生成时 |
| [atomic-add-pattern.md](atomic-add-pattern.md) | `SetAtomicAdd<T>()` 累加：安全顺序、边界、风险 | 实现 reduce/累加时 |
| [code-style.md](code-style.md) | C/C++ 代码规范和审查清单 | 代码生成后审查时 |
| [readme-spec.md](readme-spec.md) | 算子 README.md 格式要求 | 生成 README 时 |
| [../../shmem-ops-dev/references/shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO` | Read 仓内 examples/API 前 |
| [../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md](../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md) | 仓内 `docs/`、`include/` 索引 | API 路径不确定时 |

模板文件见 [../templates/communication/GUIDE.md](../templates/communication/GUIDE.md)（纯通信，分步读取）和 [../templates/fused-compute/GUIDE.md](../templates/fused-compute/GUIDE.md)（`op_kind=fused_compute_comm` 通算融合）。
