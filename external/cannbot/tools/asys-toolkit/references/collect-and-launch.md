# 故障信息收集与业务复跑

覆盖 `asys collect`（不复跑收集）、`asys launch`（复跑后收集）、`asys collect -r=stacktrace`（实时堆栈导出）。

---

## asys collect — 故障信息收集

不复跑业务，仅收集故障信息，例如软硬件信息、日志等。

### 命令格式

```bash
asys collect --task_dir=path1 --tar="True" --output=path2
```

### 参数

| 参数 | 是否必选 | 说明 |
|------|:---:|------|
| `--task_dir` | 可选 | 指定收集算子编译文件（`.o`、`.json` 等）和 dump 文件（GE dump 图、TF Adapter dump 图、exception dump 等）的目录。未指定或从该目录下没有收集到时，asys 会从环境上自动收集 |
| `--tar` | 可选 | 是否将结果输出目录压缩为 `*.tar.gz`。值为 `T` / `True` 时压缩且不保留原目录；`F` / `False` 时不压缩。默认不压缩，值不区分大小写 |
| `--output` | 可选 | 结果输出目录的路径前缀，最终输出目录为 `{output}/asys_output_timestamp`。不带该参数时输出到命令行执行目录 |

> 自动收集受环境变量影响，执行收集时环境变量值需与业务运行时保持一致，否则收集到的信息可能不准确。涉及 `ASCEND_PROCESS_LOG_PATH`、`NPU_COLLECT_PATH`、`DUMP_GRAPH_PATH`、`ASCEND_WORK_PATH`、`ASCEND_CACHE_PATH`、`ASCEND_CUSTOM_OPP_PATH`。

### 使用示例

```bash
asys collect --task_dir=$HOME/cache_path --tar="True" --output=$HOME/dfx_info
```

---

## asys launch — 业务复跑 + 故障信息收集

复跑业务后再收集故障信息。

### 命令格式

```bash
asys launch --task="sh ../app_run.sh" --tar="True" --output=path
```

### 参数

| 参数 | 是否必选 | 说明 |
|------|:---:|------|
| `--task` | **必选** | 复跑业务的执行命令，需填写完整命令，如 `"sh ../app_run.sh"`（`sh` 为执行任务的命令，`../app_run.sh` 为任务脚本） |
| `--tar` | 可选 | 同 `collect`：`T` / `True` 压缩为 `*.tar.gz` 且不保留原目录；`F` / `False` 不压缩。默认不压缩，不区分大小写 |
| `--output` | 可选 | 结果输出目录的路径前缀，最终为 `{output}/asys_output_timestamp`。不带该参数时输出到命令行执行目录 |

**不支持原有执行脚本内部直接后台执行的方式。** 例如通过 `sh cmd.sh` 拉起用例，而 `cmd.sh` 里执行 `python3 test.py &`，这种任务无法感知结束点，暂不支持。

### 注意事项

1. 业务复跑**默认开启**算子编译文件、GE dump 图和 TF Adapter dump 图收集功能。
2. **环境变量双向覆盖**（见下）。
3. `asys launch` 会拉起子进程执行业务命令。若用户主动终止 launch 命令，业务子进程可能未退出，需自行终止。

### 环境变量覆盖关系（易踩坑）

`asys launch` 执行时会**自动开启**下列环境变量临时存放收集到的信息，命令结束时自动关闭：

```
NPU_COLLECT_PATH
ASCEND_PROCESS_LOG_PATH
ASCEND_WORK_PATH
```

两个方向的覆盖都要注意：

1. 执行 `asys launch` 前手动设置了这些环境变量，则用户设置的这部分**会被覆盖、不生效**。
2. 复跑的用户任务脚本中涉及这些环境变量，则可能导致 **asys 设置的环境变量被覆盖、不生效**，进而无法收集对应的信息。

### 使用示例

```bash
asys launch --task="sh ../run.sh" --tar="True" --output=$HOME/dfx_info
```

### 可选配置 asys.ini

修改 asys 工具目录下 `ascend_system_advisor/asys/conf/asys.ini` 的 `[launch]` 段，可打开或关闭算子编译文件和 dump 图收集等功能。不修改则采用默认配置：

```ini
[launch]
graph = TRUE
ops = TRUE
dump_ge_graph = 2
dump_graph_level = 3
log_level = INFO
log_event_enable = TRUE
log_print_to_stdout = FALSE
```

| 配置项 | 含义 | 对应环境变量 |
|--------|------|-------------|
| `graph` | 是否收集 Graph 图信息。取值 `TRUE` 收集 / `FALSE` 不收集。设为 `FALSE` 时 `dump_ge_graph`、`dump_graph_level` 不生效 | — |
| `ops` | 是否收集算子编译信息。取值 `TRUE` / `FALSE` | — |
| `dump_ge_graph` | 控制 dump 图的内容多少。取值 `2` 为不含权重等数据的基础版 dump | `DUMP_GE_GRAPH` |
| `dump_graph_level` | 控制 dump 图的数量。取值 `3` 为 dump 最后阶段的生成图 | `DUMP_GRAPH_LEVEL` |
| `log_level` | 应用类日志的全局日志级别及各模块日志级别 | `ASCEND_GLOBAL_LOG_LEVEL` |
| `log_event_enable` | 应用类日志是否开启 Event 日志。取值 `TRUE` / `FALSE` | `ASCEND_GLOBAL_EVENT_ENABLE` |
| `log_print_to_stdout` | 是否开启在终端屏幕打印日志。取值 `TRUE` / `FALSE` | `ASCEND_SLOG_PRINT_TO_STDOUT` |

asys 启动时环境变量默认使用该文件里的配置值，但如果复跑任务脚本中将这些环境变量配置为其它值，则**以复跑任务脚本中设置的值为准**，可能造成收集到的维测信息不满足定位需求。

---

## 产物结构

`collect` 与 `launch` 均在 `{output}/asys_output_timestamp` 下生成：

```
├── asys_output_timestamp
   ├── software_info.txt            // 安装包版本、环境变量、依赖软件、系统信息
   ├── hardware_info.txt            // host 和 device 侧硬件信息
   ├── status_info.txt              // device 信息，含芯片型号、CPU 和 AI Core 利用率等
   ├── health_result.txt            // device 健康信息，包括故障码和故障信息
   └── dfx
       ├── bbox                     // Device 侧的黑匣子信息
       ├── data-dump                // 发生 AI Core Error 时生成的 dump 文件
       ├── graph                    // dump 图信息，包含 GE 与 TF Adapter 的 dump 图
       ├── ops                      // 算子编译信息，含 *.o 和 *.json、自定义算子配置信息等
       ├── stackcore                // 报错触发 coredump 时的 core 文件信息
       ├── atrace                   // trace 落盘信息，含 trace 二进制文件解析的明文文件
       └── log
           ├── device
           │     ├── dev-os-{id}
           │           ├── firmware      // 固件生成的日志
           │           ├── slogd         // 日志相关进程的维测日志
           │           ├── application   // 业务进程产生的非 EVENT 级别应用日志
           │           └── system        // 常驻进程生成的日志
           └── host
                ├── message         // message/syslog 日志
                ├── install         // 包历史安装情况的日志
                ├── cann            // 应用类日志
                └── driver          // Host 侧驱动日志
```

`launch` 的 `log/host/` 下另有两项：

- `screen.txt`：打印到终端屏幕的日志。内容为空则可能应用任务中设置了重定向。
- `user_cmd`：用户执行任务的命令。

`launch` 的 `dfx/ops` 还额外含**算子编译过程信息**。

**host 硬件信息**包括内核版本信息、CPU 型号、内存和硬盘使用情况等；**device 信息**包括设备个数、aicpu 个数等。

> `collect` 的 `data-dump` 为 L0 发生 AI Core Error 时生成的 L0 exception dump 文件；**L0 exception dump 不收集 `graph` 目录信息**。

> `health_result.txt` 存放**所有**故障码和故障信息；`asys health` 在终端屏幕仅显示前 5 组故障。

### 可收集信息清单

| 分类 | 描述 | 权限/前提 |
|------|------|---------|
| 软件信息 | 软件包版本、环境变量、软件依赖、系统信息 | — |
| Host 侧日志 | CANN 软件栈日志、message 日志 | — |
| Device 侧固件日志 | `device-*` 日志 | **需 root** |
| Device 侧系统日志 | message 日志、device-os 日志 | **需 root** |
| 黑匣子、stackcore 文件、coretrace 文件 | — | **需 root** |
| 任务打印日志 | — | — |
| run 包安装日志 | — | run 包安装用户与应用程序执行用户一致 |
| dump 信息 | GE dump 图、TF Adapter dump 图、发生 AI Core Error 时生成的 dump 文件 | — |
| 算子编译 `*.o`、`*.json` 文件 | — | — |
| 算子编译过程信息文件 | 编译成功失败、编译结果是复用的缓存/在线编译/二进制等 | **仅业务复跑时收集**，且需设 `NPU_COLLECT_PATH`；设置后系统在该目录下新建 `/extra-info/ops/`，写入 `op_compile_stats.log`。不设置则系统不生成该 log 文件，asys 也不会收集 |
| 自定义算子配置信息（`*.json`） | 设 `ASCEND_OPP_PATH` 时按 `${ASCEND_OPP_PATH}/vendors/config.ini` 的 `load_priority` 字段收集 `${ASCEND_OPP_PATH}/vendors` 下的 `config/*.json`；设 `ASCEND_CUSTOM_OPP_PATH` 时收集该目录下的 `config/*.json`。否则不收集 | 对应环境变量已设置 |
| 用户用例执行的命令信息 | — | — |
| 调试版本的二进制信息 | `${ASCEND_OPP_PATH}/debug_kernel` 目录下的信息 | 需提前正确配置 `ASCEND_OPP_PATH`，未配置或配置不正确则默认不收集 |

### 自定义依赖软件版本采集

`software_info.txt` 中收集的第三方依赖软件版本信息可自定义。在 asys 工具目录下的 `ascend_system_advisor/conf/dependent_package.csv` 中增加或删除配置项，每行对应一个配置项，逗号分割依赖项名字和查询指令，**逗号后无空格**：

```
make,make --version
cmake,cmake --version
unzip,unzip -v
zlib1g,dpkg -l zlib1g| grep zlib1g| grep ii
zlib1g-dev,dpkg -l zlib1g-dev| grep zlib1g-dev| grep ii
libsqlite3-dev,dpkg -l libsqlite3-dev| grep libsqlite3-dev| grep ii
openssl,dpkg -l openssl| grep openssl| grep ii
libssl-dev,dpkg -l libssl-dev| grep libssl-dev| grep ii
libffi-dev,dpkg -l libffi-dev| grep libffi-dev| grep ii
```

---

## asys collect -r=stacktrace — 实时堆栈导出

适用于训练、推理业务**进程卡住**场景。在业务未卡死时执行，可能出现信号发送失败、bin 文件生成超时、bin 文件解析失败等异常，无法正常导出堆栈信息。

**不支持对同一个卡住进程并行导出堆栈信息**，否则可能执行命令失败。

### 命令格式

```bash
asys collect -r=stacktrace --remote=pid --all --quiet --timeout=num --output=path
```

### 参数

| 参数 | 是否必选 | 说明 |
|------|:---:|------|
| `-r` | **必选** | 设置为 `stacktrace`。不设置本参数则表示执行故障信息收集功能，此时**不能与 `--remote`、`--all`、`--quiet` 混用** |
| `--remote` | `-r=stacktrace` 时必选 | 卡住进程的进程 ID，取值要求 **≥ 2**。传入的进程 ID 不存在时 asys 报错退出 |
| `--all` | `-r=stacktrace` 时必选 | 导出卡住进程中所有线程的堆栈信息 |
| `--quiet` | 可选 | 关闭与用户交互的开关。不设置则默认开启交互，需用户确认当前服务器上是否打开 trace 处理的信号集 |
| `--timeout` | 可选 | 导出实时堆栈的超时时间，取值范围 `[1, 60]`，单位秒。默认 **10 秒** |
| `--output` | 可选 | 结果输出目录的路径前缀，最终为 `{output}/asys_output_timestamp` |

### 信号 35 的风险

导出实时堆栈信息时需要**向指定进程发送信号 35**。如果关闭 trace 处理的信号集，则会**终止卡住进程**，无法导出堆栈信息。

「打开 trace 处理的信号集」指：将 `ASCEND_COREDUMP_SIGNAL` 设置为非 `none` 的其他值，或未设置该环境变量。加 `--quiet` 跳过交互确认前必须先确认此项。

### 使用示例

```bash
asys collect -r=stacktrace --remote=892839 --all --quiet --timeout=10 --output=./
```

输出示例：

```
2026-06-26 03:12:35,573 [ASYS] [INFO]: asys start.
2026-06-26 03:12:35,615 [ASYS] [WARNING]: This command sends signal 35 to the process:892839. If the process is executed to disable signal receiving through the environment variable ASCEND_COREDUMP_SIGNAL=none, the process:892839 will be killed.
2026-06-26 03:12:35,615 [ASYS] [INFO]: bin file generate path is /root/ascend/atrace, get from default path.
2026-06-26 03:12:36,236 [ASYS] [INFO]: bin file generated, awaiting stack trace completion.
2026-06-26 03:12:36,988 [ASYS] [INFO]: start parse bin file
2026-06-26 03:12:36,997 [ASYS] [INFO]: stackcore file path: /root/ascend/atrace/trace_892633_892633_20260626030851615735/stackcore_event_892839_20260626031235677855/stackcore_tracer_35_892839_aoe_20260626031235677902.txt
2026-06-26 03:12:36,998 [ASYS] [INFO]: Stacktrace output directory: /root/asys_output_20260626031235573
2026-06-26 03:12:36,998 [ASYS] [INFO]: collect task execute finish.
2026-06-26 03:12:36,998 [ASYS] [INFO]: asys finish.
```

关键字段为 `stackcore file path`。拿到该文件后可继续用 `asys analyze -r=stackcore` 做符号解析，见 [analyze-files.md](analyze-files.md)。

导出超时报错的定位方法见 [faq.md](faq.md)。

### 注意事项

导出实时堆栈信息时 asys 会检索 trace 日志所在目录，若 trace 日志文件过多会导致执行时间长。建议**先清理 trace 日志**（默认路径 `$HOME/ascend/atrace/`）再执行。
