# References Guide

本目录包含 `shmem-ops-torch-bind` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [custom-ops-torch-layout.md](custom-ops-torch-layout.md) | `custom-ops/torch_binding/` 共享外层布局与产物命名 | 独立工程 Torch 接入时 **始终** |
| [../../shmem-ops-dev/references/shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO` / `SKILLS_ROOT` | Read `${SHMEM_REPO}/examples/torch_binding/` 前 |
| [../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md](../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md) | 仓内 `docs/`、`examples/` 索引 | API/示例路径不确定时 |
| [../../shmem-ops-compile-debug/references/env-setup.snippet.md](../../shmem-ops-compile-debug/references/env-setup.snippet.md) | 动态 `IPPORT` / `SHMEM_UID_SESSION_ID` | 生成 `torch_test_*.py` 多 PE 环境时 |
| [../../shmem-ops-compile-debug/references/debug.md](../../shmem-ops-compile-debug/references/debug.md) | init 失败、端口占用、全 0 输出 | Torch 8PE 测试失败时 |

仓内参考实现（先定位 `SHMEM_REPO` 再 Read）：

| 仓内路径 | 用途 |
| --- | --- |
| `${SHMEM_REPO}/examples/torch_binding/` | CustomClass、CMake、注册宏 |
| `${SHMEM_REPO}/examples/python_extension/torch_test/` | 多 PE Python 测试脚本 |
