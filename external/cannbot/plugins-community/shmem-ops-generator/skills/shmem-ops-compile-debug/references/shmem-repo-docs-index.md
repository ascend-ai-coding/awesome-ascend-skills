# SHMEM 仓库文档与 API 索引（算子开发参考）

生成、调试、优化 custom-ops 算子时，**SHMEM API/官方文档** 以本表 **SHMEM 仓原生** 路径为准（`docs/`、`install/shmem/include/`、`examples/`）。
**custom-ops/** 为 skill 生成交付物，**不在**本表；其规范见 [custom-ops-entrypoints.md](custom-ops-entrypoints.md)、[env-setup.snippet.md](env-setup.snippet.md)。

> **`docs/` 只读**：仓内 `docs/` 为官方文档，Agent **不得修改**；仅 Read 引用。技能侧补充写在 skill 树内（如本目录 `log-debug.md`）。
>
> **路径规范**：Skill 不一定在 SHMEM 仓内。读下表任何文档前 **MUST** 先按 [shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) 定位 `SHMEM_REPO`，再 `Read \`${SHMEM_REPO}/<下表路径>\``。**禁止**用 skill 文件相对路径（如 `../../../../docs/...`）链接仓内文档。

表中「仓内路径」均相对于 `SHMEM_REPO`（如 `/path/to/shmem`）。Read 时使用完整路径：`${SHMEM_REPO}/docs/debug/log_debug.md`。

---

## 1. 调试文档（`docs/debug/`）

| 仓内路径 | 用途 | 可参考时机 |
| --- | --- | --- |
| `docs/debug/log_debug.md` | SHMEM 日志环境变量、格式、bootstrap/init 阶段日志解读 | **`aclshmemx_init_attr` 失败**、输出全 0、端口/bootstrap 问题 |
| `docs/debug/Troubleshooting_FAQs.md` | 对称内存分配、local_mem_size、使用限制 | malloc 异常、精度错、init 参数不一致 |
| `docs/debug/dump_debug.md` | Device DumpTensor/printf | Kernel 侧数据/同步异常 |
| `docs/debug/profiling.md` | 性能 profiling | Phase 6/6.5 |
| `docs/debug/tools_debug.md` | 调试工具 | 需配合 CANN 工具链时 |

技能内摘要：[log-debug.md](log-debug.md)、[dump-debug.md](dump-debug.md)、[debug.md](debug.md)。

---

## 2. API 与原理（`docs/api/`、`docs/`）

| 仓内路径 | 用途 |
| --- | --- |
| `docs/api/env_vars_intro.md` | `SHMEM_UID_SESSION_ID`、`SHMEM_INSTANCE_PORT_RANGE` 等 |
| `docs/api/stream_api_usage.md` | Stream + init_attr 用法 |
| `docs/api/atomic_api_sync_async_comparison.md` | 原子/sync/async API 选型 |
| `docs/principles.md` | `aclshmemx_init_attr` 初始化流程与 state |
| `docs/glossary.md` | 术语表 |
| `docs/compilation_build_guide.md` | 编译构建 |
| `docs/deployment/shmem_so_library_dependencies.md` | bootstrap 插件部署 |
| `docs/example/example.md` | 各 bootstrap 模式示例 |
| `docs/quickstart.md` | 快速上手 |

---

## 3. 头文件与接口说明（`install/shmem/include/`）

构建并 `source `${SHMEM_REPO}/install/set_env.sh` 后，安装目录头文件可作为 **API 参考真源**：

| 仓内路径 | 用途 |
| --- | --- |
| `install/shmem/include/shmem.h` | Host/Device 入口、生命周期 |
| `install/shmem/include/host/` | Host 侧 RMA、bootstrap、team |
| `install/shmem/include/device/` | Device 侧 gm2gm、sync、MTE/RDMA/SDMA |
| `install/shmem/include/utils/` | prof、工具头 |

代码生成时：API 选型**建议**与上述头文件签名一致；不确定时在 `${SHMEM_REPO}` 下 `Grep install/shmem/include` 或 Read 对应 `.hpp`。

源码侧镜像：`${SHMEM_REPO}/include/`、`${SHMEM_REPO}/src/`（未 install 时）。

---

## 4. 示例代码（`examples/`）

| 仓内路径 | 用途 |
| --- | --- |
| `examples/allgather/allgather_kernel.cpp` | sender/receiver 分核、per-chunk signal |
| `examples/multi_instance/` | 端口与 instance_id |
| `examples/*/CMakeLists.txt`、`scripts/run.sh` | 构建与多 PE 启动 |
| `examples/torch_binding/` | Torch CustomClass 参考（见 torch-bind skill） |
| `examples/python_extension/torch_test/` | 多 PE Python 测试参考 |

custom-ops 默认 **独立工程**；仅在 design 明确要求 in-tree 时对照 `${SHMEM_REPO}/examples/` 修改根 CMake。

---

## 5. `aclshmemx_init_attr` 失败 — 参考排障

1. 定位 `SHMEM_REPO`（见 [shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md)）
2. Read `${SHMEM_REPO}/docs/debug/log_debug.md` 对照 bootstrap/init 日志阶段
3. 执行下方日志采集

### 5.1 开启日志并重跑

```bash
export SHMEM_LOG_LEVEL=DEBUG
export SHMEM_LOG_TO_STDOUT=1
# 或落盘：export SHMEM_LOG_PATH=${HOME}/shmem/log
```

### 5.2 读最新日志

```bash
LOG_DIR="${SHMEM_LOG_PATH:-$HOME/shmem/log}"
ls -lt "${LOG_DIR}"/aclshmem_*.log 2>/dev/null | head -5
grep -iE 'fail|error|address in use|bootstrap|AccStore|Memory Heap' "${LOG_DIR}"/aclshmem_*.log | tail -30
```

### 5.3 常见根因 → 动作

| 日志/现象 | 动作 |
| --- | --- |
| `address in use` | `pkill -f 'build/bin/<op>|torch_test_'`；`scripts/run.sh` 动态 `IPPORT`/`SHMEM_UID_SESSION_ID` |
| `local size diffs` | 各 PE `local_mem_size` 一致（见 `docs/debug/Troubleshooting_FAQs.md`） |
| `Memory Heap Not Initialized` | init 失败；优先查 bootstrap 与环境 |
| 无日志 | 确认 `source `${SHMEM_REPO}/install/set_env.sh`；bootstrap 插件与 `libshmem.so` 同目录 |

完整表见 [debug.md §4](debug.md) 与 [agent-execution-contract.md](../../shmem-ops-dev/references/agent-execution-contract.md) §5。
