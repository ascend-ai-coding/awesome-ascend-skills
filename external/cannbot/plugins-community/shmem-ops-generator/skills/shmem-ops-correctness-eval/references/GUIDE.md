# References Guide

本目录为 `shmem-ops-correctness-eval` 提供正确性验证参考资料索引。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [correctness.md](../../shmem-ops-testcase-gen/references/correctness.md) | golden 构造、invariant 映射、失败分类 | 执行 case matrix 时 |
| [precision-standard.md](../../shmem-ops-testcase-gen/references/precision-standard.md) | rtol/atol、双统计判定 | 判断精度时 |
| [../../shmem-ops-compile-debug/references/debug.md](../../shmem-ops-compile-debug/references/debug.md) | 运行失败、init、全 0 输出 | 失败定位时 |
| [../../shmem-ops-dev/references/shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO` | 对照仓内 examples 验证 golden 时 |

运行与失败定位由 `shmem-ops-compile-debug` 执行；本 skill 负责结果判定与报告。
