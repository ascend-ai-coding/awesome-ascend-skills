# References Guide

本目录包含 `shmem-ops-testcase-gen` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [correctness.md](correctness.md) | case matrix 生成规则、golden 构造、invariant 验证 | 步骤 1-4 全过程 |
| [precision-standard.md](precision-standard.md) | 精度标准：OpTypes 分类、rtol/atol、双统计判定 | 步骤 4 生成 check_result.py 时 |
| [test-structure-template.md](test-structure-template.md) | gen_data.py / check_result.py / runshell 脚本文件模板 | 步骤 3-5 脚本生成时 |
| [testcase-scale-standard.md](testcase-scale-standard.md) | 规模分档 XS/S/M/L、最小 case 数要求 | 步骤 2 生成 case matrix 时 |
| [../../shmem-ops-dev/references/shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO` / `SKILLS_ROOT` | 读仓内 examples/docs 对照 golden 前 |
| [../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md](../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md) | 仓内 `docs/`、`examples/` 索引 | API/示例路径不确定时 |
