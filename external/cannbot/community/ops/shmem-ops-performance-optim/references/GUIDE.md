# 参考资料索引

本目录包含 `shmem-ops-performance-optim` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [optimization-patterns.md](optimization-patterns.md) | 通信、流水线、同步、内存、分核优化手段 | OptimStep 1 定位瓶颈后选择对应优化 |
| [compute-optimization.md](compute-optimization.md) | 通算融合中 Matmul/Compute 调优（TileShape、Swizzle、DispatchPolicy、SplitK、存储层次） | 瓶颈在 compute 部分时 |

跨 skill 参考：

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [../shmem-ops-performance-eval/references/perf-chat-output-spec.md](../../shmem-ops-performance-eval/references/perf-chat-output-spec.md) | 每轮聊天 Δ% 自动输出 | OptimStep 5 完成后 |
| [../../shmem-ops-dev/references/agent-execution-contract.md](../../shmem-ops-dev/references/agent-execution-contract.md) | 统一实现、禁止中断、Phase 6.5 门控 | OptimStep 3 机制改动前 |
| [../../shmem-ops-dev/references/shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO` | 对照仓内 examples 优化前 |
