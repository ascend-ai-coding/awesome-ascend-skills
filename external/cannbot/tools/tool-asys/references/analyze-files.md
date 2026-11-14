# 文件解析

`asys analyze` 的六种解析模式：`trace`、`coredump`、`stackcore`、`coretrace`、`ub`、`aicore_error`。

## 模式速查

| `-r` 取值 | 输入 | 前置依赖 | 产品支持 |
|-----------|------|---------|---------|
| `trace` | trace 日志 `*.bin` | asys 环境版本需与产生 trace 的环境**版本一致** | 全系列 |
| `coredump` | core 文件 + 可执行文件 | **gdb** | 全系列 |
| `stackcore` | `stackcore_*.txt` | **readelf**、**addr2line** | 全系列 |
| `coretrace` | `coretrace.*` | **addr2line** | **仅 Ascend 950PR / 950DT** |
| `ub` | UB 维测信息 `*.bin` | — | 全系列 |
| `aicore_error` | 日志 / dump 目录 | — | 全系列 |

> `--output` 通用行为：最终目录为 `{output}/asys_output_timestamp`；不带该参数时输出到命令行执行目录；取值为空、无效字符串、目录无写权限或创建目录失败时，asys 退出执行并报错。

---

## -r=trace — trace 文件解析

将 trace 日志文件（`*.bin`）解析为 `.txt` 格式。trace 文件的获取方式见《日志参考》的「查看 trace 日志」。

### 命令格式

```bash
asys analyze -r=trace --file=filename --output=path
```

### 参数

| 参数 | 是否必选 | 说明 |
|------|:---:|------|
| `-r` | **必选** | 设置为 `trace`。**使用 asys 的环境版本要与产生 trace 日志的环境版本保持一致** |
| `--file` | trace 模式必选 | 用于解析**单个**文件，设置为包含路径的文件名 |
| `--output` | 可选 | 结果输出目录的路径前缀 |

### 使用示例

```bash
asys analyze -r=trace --file=schedule_tracer_demo.bin --output=$HOME/dfx_info
```

### 输出

解析后的 txt 内容示例：

```
2024-05-09 19:08:12.408.800 demo0: tid0[0], count0[0], tag0[struct0 tag], streamId0[0], deviceIdArray0[0, 1], hostIdArray0[1, 2, 3, 4]
2024-05-09 19:08:12.408.804 demo1: tag1[struct1 tag], streamId1[0], deviceIdArray1[0, 1], hostIdArray1[1, 2, 3, 4]
```

每行是一条带毫秒/微秒级时间戳的记录，后跟事件名和方括号包裹的字段。不同事件的字段集合并不固定（`demo1` 就没有 tid/count），数组类字段以逗号分隔多个 ID。

---

## -r=coredump — coredump 文件解析

在执行任务过程中进程中断退出、软件退出时报 `Segmentation fault` 等错误时，可用该功能解析 coredump 生成的 core 文件，获取 stackcore 格式的堆栈文件（`*.txt`）。

### 前置条件

coredump 解析功能依赖 gdb，需提前安装：

```bash
apt-get install gdb    # 或
yum install gdb
```

### 命令格式

```bash
asys analyze -r=coredump --exe_file=filename --core_file=filename --reg=reglevel --symbol=value --output=path
```

### 参数

| 参数 | 必选性 | 说明 |
|------|:---:|------|
| `-r` | **必选** | 设置为 `coredump` |
| `--exe_file` | coredump 模式必选 | 程序 coredump 时的可执行文件，设置为包含路径的文件名。**需保证与 `core_file` 相匹配**，否则解析结果错误 |
| `--core_file` | coredump 模式必选 | 程序 coredump 时生成的 core 文件，设置为包含路径的文件名。**需保证与 `exe_file` 相匹配** |
| `--reg` | 可选，默认 `0` | 添加寄存器数据的模式，只支持 `0` / `1` / `2` |
| `--symbol` | 可选，默认 `0` | 解析模式，只支持 `0` / `1` |
| `--output` | 可选 | 结果输出目录的路径前缀 |

**`--reg` 取值**：

| 取值 | 含义 |
|:---:|------|
| `0` | 不添加寄存器数据 |
| `1` | 每个线程加一条寄存器数据 |
| `2` | 线程的所有栈都添加寄存器数据（占用 Host 资源较多，比较耗时） |

**`--symbol` 取值**：

| 取值 | 含义 |
|:---:|------|
| `0` | 将所有带地址行解析成 stackcore 格式的文件，其他行显示 `Ignore` 表示跳过不解析 |
| `1` | 只解析 `in ?? ()` 行，其他行保留 gdb 堆栈的原数据 |

> 地址不存在或栈溢出可能导致 asys 无法解析。

### 使用示例与输出

**symbol=0（解析所有地址行）**：

```bash
asys analyze -r=coredump --exe_file=atrace_test --core_file=core_atrace_test_001 --symbol=0 --output=$HOME/dfx_info
```

```
[process]
crash reason: SIGABRT
crash pid: 37246
crash tid: 37246

[stack]
Thread 1 (37246)
#00 0x00007fbad83792bf 0x00007fbad830b000 /usr/local/python3.7.5/lib/libpython3.7m.so.1.0
#01                                       Ignore
#02 0x00007fbad83d8c22 0x00007fbad830b000 /usr/local/python3.7.5/lib/libpython3.7m.so.1.0
......

[maps]
    Start Addr           End Addr       Size     Offset objfile
0x562677ed1000     0x562677ed2000     0x1000        0x0 /usr/local/python3.7.5/bin/python3.7
......
```

**symbol=1（只解析 `in ?? ()` 行）**：

```bash
asys analyze -r=coredump --exe_file=atrace_test --core_file=core_atrace_test_002 --symbol=1 --output=$HOME/dfx_info
```

```
[process]
crash reason: SIGABRT
crash pid: 37246
crash tid: 37246

[stack]
Thread 1 (37246)
#00 0x00007fbad83792bf in lookdict_unicode (value_addr=0x7ffea1e917e8, hash=<optimized out>, key=<optimized out>, mp=0x7fba98907fa0) at Objects/dictobject.c:811
#01 lookdict_unicode (mp=0x7fba98907fa0, key=<optimized out>, hash=<optimized out>, value_addr=0x7ffea1e917e8) at Objects/dictobject.c:783
......
```

三段结构：

| 段 | 内容 |
|----|------|
| `[process]` | 崩溃原因（如 SIGABRT）、崩溃进程号与线程号 |
| `[stack]` | 按线程列出帧序号与地址 |
| `[maps]` | 内存映射表，列为 Start Addr、End Addr、Size、Offset、objfile |

解析结果后续可用 `-r=stackcore` 功能进一步解析。

---

## -r=stackcore — stackcore 文件解析

解析 stackcore 格式的文件（`*.txt`）。

**三种获取来源**：

1. Host 侧 stackcore 文件（`stackcore_tracer_*.txt`），见《日志参考》的「查看 trace 日志」章节
2. Device 侧 stackcore 文件，见《msnpureport 工具》的「导出 Device 侧系统类日志和其他维测信息 > 单次导出」章节
3. 使用 asys 的 coredump 文件解析功能获得

### 前置条件

- 依赖 **readelf** 进行文件信息的获取、依赖 **addr2line** 进行堆栈函数名和行号的解析。两者都是 Linux 系统自带工具，需确保已安装且执行该脚本的用户有权限执行。
- **需在获取 stackcore 文件的环境中执行解析命令**，否则可能导致解析结果不准确。

### 命令格式

```bash
# 解析单个文件
asys analyze -r=stackcore --file=filename --symbol_path=path1,path2 --output=path3

# 递归解析目录
asys analyze -r=stackcore --path=directory --symbol_path=path1,path2 --output=path3
```

### 参数

| 参数 | 是否必选 | 说明 |
|------|:---:|------|
| `-r` | **必选** | 设置为 `stackcore` |
| `--file` | 与 `--path` **二选一** | 用于解析单个文件，设置为包含路径的文件名 |
| `--path` | 与 `--file` **二选一，两者不能同时存在** | 指定目录，解析该目录**及其子目录**下的多个文件 |
| `--symbol_path` | 可选 | 解析所需的动态库目录，可传多个目录用逗号隔开。**只扫描当前目录下的动态库，先找路径 1 再找路径 2，不扫描子目录**。不指定时从 stackcore 文件中获取动态库路径 |
| `--output` | 可选 | 结果输出目录的路径前缀 |

> 为防止误解析，建议将相关动态库放在一个路径下。不指定 `--symbol_path` 时为防止找不到动态库文件，建议仅在发生 coredump 错误的环境上使用。

### 使用示例

```bash
asys analyze -r=stackcore --file=stackcore_tracer_test.txt --symbol_path=$HOME/test1,$HOME/test2 --output=$HOME/dfx_info
```

### 输出

线程信息以 `Thread num (线程id，线程名)` 开头，线程名获取失败则显示 `unknown`：

```
[process]
crash reason:6
crash pid:37246
crash tid:37246
crash stack base:0x00007ffea1e96000
crash stack top:0x00007ffea1e91770

[stack]
Thread 1 (37246, python3.7)
#00 0x00007fbad83792bf lookdict_unicode in dictobject.c:811 from libpython3.7m.so.1.0
#01                    lookdict_unicode in dictobject.c:783 from libpython3.7m.so.1.0
#02 0x00007fbad83d8c22 PyDict_GetItem in dictobject.c:1328 from libpython3.7m.so.1.0
......

[maps]
e0000380000-e0000381000 rw-p 00000000 00:00 0
562677ed1000-562677ed2000 r--p 00000000 fd:00 13113992                   /usr/local/python3.7.5/bin/python3.7
......
```

---

## -r=coretrace — coretrace 文件解析

解析 coretrace 格式的文件（`coretrace.*`）。该文件通过《msnpureport 工具》的「导出 Device 侧系统类日志和其他维测信息 > 单次导出」章节导出。

> **产品支持：仅 Ascend 950PR / Ascend 950DT 支持。** Atlas A3 训练/推理系列、Atlas A2 训练/推理系列、Atlas 200I/500 A2 推理产品、Atlas 推理系列、Atlas 训练系列均**不支持**。

### 前置条件

- 依赖 **addr2line** 进行堆栈函数名和行号的解析，需确保已安装且执行该脚本的用户有权限执行。
- **需在获取 coretrace 文件的环境中执行解析命令**，否则可能导致解析结果不准确。

### 命令格式

```bash
asys analyze -r=coretrace --file=filename --symbol_path=path1,path2 --output=path3
# 或
asys analyze -r=coretrace --path=directory --symbol_path=path1,path2 --output=path3
```

### 参数

参数语义与 `-r=stackcore` 一致：`--file` 与 `--path` 二选一且不能同时存在，`--symbol_path` 逗号分隔按序查找且不扫描子目录，`--output` 为路径前缀。

### 使用示例

```bash
asys analyze -r=coretrace --file=coretrace.log-daemon.12335.11.1749181305 --symbol_path=$HOME/test1,$HOME/test2 --output=$HOME/dfx_info
```

### 输出

首行标明信号与进程，线程信息以 `PID num (线程id，线程名)` 开头，解析内容按 `{地址} {函数名} {二进制名}` 格式输出：

```
Signal 11 pid 12335

PID 12335 TGID 12335 comm log-daemon
0xdfffcac78a68    0x00000000000b4a68: clock_nanosleep at ??:?    /usr/lib64/libc.so.6
0xaaaad1e22dec    0x0000000000012de8: ToolSleep at log_system_api.c:704    /var/log-daemon
0xaaaad1e1cb58    0x000000000000cb54: main at log_daemon.c:217    /var/log-daemon
......

PID 12356 TGID 12335 comm adx_get_file_th
0xdfffcaca550c    0x00000000000e150c: ioctl at ??:?    /usr/lib64/libc.so.6
......
```

若函数名获取失败，则显示解析过程中自动计算出来的十六进制值。此时需检查 `--symbol_path` 配置目录是否正确，以及该目录下的动态库文件是否正确。

---

## -r=ub — UB 文件解析

解析 UB（Unified Bus）维测信息文件。UB 文件通过《msnpureport 工具》的「导出 Device 侧系统类日志和其他维测信息 > 单次导出」章节导出。

### 命令格式

```bash
asys analyze -r=ub --path=directory --output=path
```

### 参数

| 参数 | 是否必选 | 说明 |
|------|:---:|------|
| `-r` | **必选** | 设置为 `ub`，用于解析二进制格式的 UB 维测信息采集文件 |
| `--path` | ub 模式必选 | 指定目录，解析该目录下的二进制文件 |
| `--output` | 可选 | 结果输出目录的路径前缀 |

asys 会读取 `--path` 路径下的以下二进制文件，并解析为**同名的 txt 文件**：

| 文件 | 内容 |
|------|------|
| `ubnl_dfx_config_item.bin` | UB 网络层配置表项 |
| `ubnl_dfx_statistic.bin` | UB 网络层统计信息 |
| `ubnl_dfx_ssu_schedule.bin` | UB 网络层 SSU（System Scheduling Unit）调度队列统计和队列丢包统计 |
| `ubmem_daw.bin` | UB memory 各配置表项 |
| `ubtpl_acl_src.bin` | UB Transport 层关键表项和配置 |
| `sl_to_vl.bin` | UB QOS（Quality of Service）配置与表项 |

### 使用示例与输出

```bash
asys analyze -r=ub --path=/home/test/msnpureport/device-0/ub
```

```
2026-02-12 14:23:10,020 [ASYS] [INFO]: asys start.
2026-02-12 14:23:10,021 [ASYS] [INFO]: asys output directory: /home/test/asys_output_20260212142310021
2026-02-12 14:23:10,032 [ASYS] [INFO]: Conversion successful! /home/test/msnpureport/device-0/ub/ubnl_dfx_statistic.bin has been converted to text file /home/test/asys_output_20260212142310021/ubnl_dfx_statistic.txt
......
2026-02-12 14:23:10,049 [ASYS] [INFO]: analyze task execute finish.
2026-02-12 14:23:10,049 [ASYS] [INFO]: asys finish.
```

根据终端界面提示的路径获取解析后的 txt 文件。

---

## -r=aicore_error — AI Core Error 故障信息解析

执行业务时，若日志文件或屏幕打印信息中包含 AI Core Error 报错，可用该功能快速定位原因。

**触发关键字**：

```
there is an aivec error exception
there is an aicore error exception
```

### 命令格式

```bash
# aic_err_info_timestamp 为存放 AI Core Error 问题信息的目录，请根据实际情况替换
asys analyze -r=aicore_error -d=deviceId --path=${HOME}/aic_err_info_timestamp
```

### 参数

| 参数 | 必选性 | 说明 |
|------|:---:|------|
| `-r` | **必选** | 设置为 `aicore_error` |
| `-d` | 可选 | 指定待操作的 deviceId。不设置时默认 device 0 的配置 |
| `--path` | 可选 | 存放日志和 dump 文件等故障信息的目录。**不配置时 asys 会自动收集** |
| `--output` | 可选 | 结果输出目录。不带该参数时输出到命令行执行目录 |

### 自动收集的环境变量依赖

不配置 `--path` 时自动收集受环境变量影响，**执行 asys 命令时环境变量值需与业务运行时保持一致**，否则收集到的信息可能不准确：

```
ASCEND_PROCESS_LOG_PATH
NPU_COLLECT_PATH
DUMP_GRAPH_PATH
ASCEND_WORK_PATH
ASCEND_CACHE_PATH
ASCEND_CUSTOM_OPP_PATH
```

如果这些环境变量都不存在，会从执行 asys 命令的当前目录下收集故障信息。

### 使用示例

```bash
asys analyze -r=aicore_error --path=${HOME}/aic_err_info_timestamp
```

### 输出

执行命令后根据终端界面提示的 `info.txt` 文件所在路径，通过其中的提示信息进行问题分析和定位。

> 若收集的信息中存在**多个** AI Core Error 问题，本工具按日志时间解析**第一次出现的**那个。

### 注意事项

1. 为保证解析数据的准确性，在复现 AI Core Error 问题前**建议先清理日志**。
2. 避免循环拷贝：`--output` 目录**不能为 `--path` 目录或其子目录**。

---

## 符号解析常见问题

`stackcore` / `coretrace` 解析结果里出现 `?` 字符，可能有三类原因：

| 原因 | 修复 |
|------|------|
| **编译选项**：动态库文件编译时没有使用 `-g` 选项，未在文件中保留调试信息 | 重新编译时加 `-g` |
| **未添加链接参数**：未使用 `-rdynamic` 通知链接器将所有符号添加到动态符号表 | 链接时加 `-rdynamic` |
| **未找到动态库**：未找到相匹配的动态库 | 检查 `--symbol_path` 及目录下的动态库文件 |

**部分动态库解析出的行号会有少许偏差**，原因有两类：

- **编译选项**：不同的编译选项，特别是与调试信息相关的选项，可能会造成影响。
- **优化级别**：较高的优化级别可能导致代码重组和优化，使行号与原始源代码的对应关系发生偏差。
