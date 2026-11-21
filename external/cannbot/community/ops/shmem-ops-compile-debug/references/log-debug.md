# SHMEM 日志调试

> **仓内参考文档**：先定位 `SHMEM_REPO`（[shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md)），再 Read `${SHMEM_REPO}/docs/debug/log_debug.md`
> 本文件为技能内摘要；排障时可对照仓内文档与最新日志。

完整 SHMEM 文档索引见 [shmem-repo-docs-index.md](shmem-repo-docs-index.md)。

## 环境变量

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `SHMEM_LOG_LEVEL` | 日志等级：DEBUG / INFO / WARN / ERROR / FATAL | ERROR |
| `SHMEM_LOG_TO_STDOUT` | 是否输出到控制台（0=关闭，1=开启） | 0（落盘到文件） |
| `SHMEM_LOG_PATH` | 日志保存路径 | `${HOME}/shmem/log` |

调试时建议设置为 DEBUG 或 INFO：

```bash
export SHMEM_LOG_LEVEL=DEBUG
export SHMEM_LOG_TO_STDOUT=1
```

如果开启 `SHMEM_LOG_TO_STDOUT=1`，日志会在控制台打印，不再落盘到文件。

## 日志格式

每条 SHMEM 日志包含：时间、日志级别、进程号、日志模块、日志文件、行号、日志信息。

## 关键日志阶段

### 初始化阶段

初始化时会报出 bootstrap 使用的 flag（如 `ACLSHMEMX_INIT_WITH_DEFAULT`），然后开始 bootstrap 初始化，并检测环境变量设置情况（如 `SHMEM_UID_SESSION_ID` 是否已设置）。

### 网络检测

SHMEM 有一个 root 0 节点。在单机环境该节点可使用回环地址，但在集群环境使用回环地址是错误的。初始化过程会报出当前的 root 0（remote address）是否是回环地址。如果 root 0 是回环则默认单机环境。当前 PE 的 IP 信息会在 `netifaddr` 日志中报出。

### bootstrap 和初始化完成

- bootstrap 成功后有专门日志打印，同时标明 PE 号
- 初始化成功后有专门日志打印，同时标明 PE 号
- 去初始化成功后有专门日志打印，同时标明 PE 号

### socket 信息

bootstrap 过程涉及多个 socket 的创建和使用，这些 socket 信息会在日志中体现。更详细的 socket 信息可通过开启 DEBUG 级别获取。

## 定位能力边界

SHMEM 日志当前主要提供 Host 侧的定位能力。Device 侧如算子发生报错时可能无法根据 SHMEM 日志定位，需配合 CANN 或 AscendC DumpTensor/printf 工具进行定位（见 [dump-debug.md](dump-debug.md)）。

## `aclshmemx_init_attr` 失败时（参考）

1. 可设置 `SHMEM_LOG_LEVEL=DEBUG`、`SHMEM_LOG_TO_STDOUT=1`（或读 `${SHMEM_LOG_PATH:-$HOME/shmem/log}/aclshmem_*.log`）
2. 在日志中定位 bootstrap/init 失败点（见上文「关键日志阶段」）
3. 对照 [debug.md §4](debug.md) 与 `${SHMEM_REPO}/docs/debug/Troubleshooting_FAQs.md`
4. 环境恢复后再跑正确性；init 未成功时优先排查环境，再考虑 kernel

```bash
grep -iE 'fail|error|address in use|bootstrap|AccStore' "${SHMEM_LOG_PATH:-$HOME/shmem/log}"/aclshmem_*.log | tail -20
```
