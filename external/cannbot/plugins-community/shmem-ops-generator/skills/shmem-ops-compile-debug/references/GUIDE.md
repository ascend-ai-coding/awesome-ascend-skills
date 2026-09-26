# References Guide

本目录包含 `shmem-ops-compile-debug` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [custom-ops-entrypoints.md](custom-ops-entrypoints.md) | custom-ops 编译 / 运行 / matrix / Torch 命令代码段 | 步骤 3 及交付文档 |
| [env-setup.snippet.md](env-setup.snippet.md) | `scripts/run.sh` 运行时环境（内联函数） | 生成/调试 `scripts/run.sh` 时 |
| [debug.md](debug.md) | 失败定位和调试手段 | 步骤 5 失败分类时 |
| [log-debug.md](log-debug.md) | SHMEM 日志环境变量与日志格式 | 初始化/运行时失败时 |
| [dump-debug.md](dump-debug.md) | AscendC DumpTensor/printf 调测 API | Device 数据/同步异常时 |
| [shmem-repo-docs-index.md](shmem-repo-docs-index.md) | 仓内 `docs/`（**只读**）、`include/`、examples 索引 | API/行为不确定、init 失败排障时 |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) | **读仓内文件前** | 定位 `SHMEM_REPO` / `SKILLS_ROOT` |
| [cann-env-resolution.md](cann-env-resolution.md) | CANN 路径确认与 source 流程 | Phase 0 / 编译前环境未就绪时 |
