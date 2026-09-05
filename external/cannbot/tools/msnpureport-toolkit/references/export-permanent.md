# 连续导出 Device 侧系统类日志和其他维测信息

## 功能说明

msnpureport 提供**常驻进程模式**持续导出 Device 侧系统日志。

在模型推理/训练任务运行时，执行单次导出命令可能出现日志导出不全的情况，比如：

- Device 出现异常导致无法导出 Device 侧日志
- 任务执行时间过长导致 Device 侧日志被清理（老化）掉

此时可以**在任务执行之前**，在 Host 侧运行该命令连续导出 Device 侧的日志和文件，确保能够获取到 Device 异常前的所有日志和文件，便于定位问题。

## 命令格式

```sh
msnpureport report --permanent [options]
```

## 注意事项

- 该命令**仅支持 root 用户**执行。且须确保执行该命令的当前路径对普通用户无访问权限，否则普通用户也能操作导出的日志文件，存在恶意删除日志文件或泄露系统信息等安全风险。
- 不建议在同路径同时刻并发使用，以免造成落盘时间戳目录名冲突；**一个 Device 不支持并发使用**。
- 容器场景不支持使用该命令（特权容器场景除外，无此限制）。
- 在异常场景下，可能会出现连续导出失败的情况。
- 终止导出（或模型推理/训练完成后），需使用 `Ctrl+C` 或 `kill -15 <pid>` 结束进程。pid 为连续导出进程 ID，可通过 `ps -elf | grep msnpureport` 查询。

## 参数说明

### -d \<deviceId> 或 --device \<deviceId>

可选，指定 Device ID（逻辑 ID），默认导出所有 Device 日志。

SMP 工作模式下，指定一个 OS 内的任意 Device ID 时会导出该 OS 下的所有 Device 日志；**一个 OS 内不支持多个导出进程并发执行**。例如 device 0 和 device 1 共用一个 OS 时，不能同时执行 `msnpureport report --permanent -d 0` 和 `msnpureport report --permanent -d 1`。

### 文件数量与大小控制参数

均为可选参数。数量类参数在日志文件数超过设定值时，**新日志覆盖最早的日志**；大小类参数在文件超过设定值时，日志生成到新文件中。

| 参数 | 控制对象 | 取值范围 | 默认值 |
|------|---------|---------|-------|
| `--sys_log_num` | Device 侧系统进程产生的调试日志/运行日志的文件数量（分别控制调试日志目录与运行日志目录下各自的最大文件数） | [1, 1000] | 10 |
| `--sys_log_size` | Device 侧系统进程产生的调试日志和运行日志的单个文件大小（单位 M） | [1, 100] | 10 |
| `--event_log_num` | Device 侧应用进程产生的 EVENT 日志的文件数量 | [1, 1000] | 10 |
| `--event_log_size` | Device 侧应用进程产生的 EVENT 日志的单个文件大小（单位 M） | [1, 100] | 10 |
| `--sec_log_num` | Device 侧应用进程产生的安全日志的文件数量 | [1, 1000] | 10 |
| `--sec_log_size` | Device 侧应用进程产生的安全日志的单个文件大小（单位 M） | [1, 100] | 10 |
| `--fw_log_num` | 非 Control CPU 上的系统类日志的文件数量 | [1, 1000] | 10 |
| `--fw_log_size` | 非 Control CPU 上的系统类日志的单个文件大小（单位 M） | [1, 100] | 10 |
| `--stackcore_num` | stackcore 目录下的文件数量 | [1, 100] | 50 |
| `--bbox_num` | 黑匣子日志文件夹数量 | [1, 10000] | 1000 |
| `--fault_event_num` | Proc 文件和 Device 侧 OS 系统日志的文件夹（`fault_event_{故障码}_*`）数量，Device 故障触发时生成 | [1, 100] | 10 |
| `--slogd_log_num` | 日志工具自身产生的维测日志的文件数量 | [1, 100] | 2 |
| `--app_dir_num` | Device 侧未来得及回传到 Host 的剩余应用类日志文件数量 | [1, 100] | 8 |

**`--stackcore_num` 具体指以下两个目录下各自的最大文件数量**：

- `当前目录/*/stackcore/{dev-os-id}/stackcore.slogd.xxx`
- `当前目录/*/stackcore/{dev-os-id}/udf/stackcore.slogd.xxx`

不同产品型号产生的文件名可能不一样，部分型号下文件名为 `coretrace.slogd.xx`。

### -o \<path> 或 --output \<path>

可选，指定导出文件路径。要求配置为**相对路径**，且该路径必须**已存在**。不建议配置到单次导出任务已存在的时间戳目录下。

配置示例：`--output mypath`

## 使用示例

1. 以 root 用户登录 Host 侧服务器。
2. 创建一个目录用来存放导出的文件（如 `/var/log/npu/report/`），确保当前用户有读、写、执行权限。
3. 进入以上目录，执行导出命令：

    ```sh
    msnpureport report --permanent
    ```

4. 任务完成后终止导出：`Ctrl+C` 或 `kill -15 <pid>`。

## 输出说明

导出到当前目录（如 `/var/log/npu/report`）下以时间戳命名的文件夹中。连续导出的目录结构与单次导出略有差异，见 [output-layout.md](output-layout.md)。
