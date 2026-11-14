# 单次导出 Device 侧系统类日志和其他维测信息

## 功能说明

导出 Device 侧系统类日志，主要包括：

- **Control CPU 上的系统类日志**：包括内核态日志和系统进程运行产生的用户态日志，主要反映 AI 处理器的整体运行情况。
- **非 Control CPU（例如低功耗）上的系统类日志**：主要反映低功耗、Task Scheduler、ISP 等组件的运行情况。

另外还支持导出黑匣子日志、Device 进程 coredump 时的调用栈信息、Host 侧驱动内核日志等维测信息。

## 命令格式

方式一（推荐）：

```sh
msnpureport report [options]
```

方式二：

```sh
msnpureport [options]
```

## 注意事项

- 该命令仅支持 root 用户执行。且须确保执行该命令的当前路径对普通用户无访问权限，否则普通用户也能操作导出的日志文件，存在恶意删除日志文件或泄露系统信息等安全风险。
- 支持多进程运行，**建议不超过 4 个**，否则可能因资源不足导致执行失败；不建议在同路径同时刻并发使用，以免造成落盘时间戳目录名冲突。
- 容器场景下仅支持导出 Host 侧驱动内核日志，其他日志不支持导出（特权容器场景除外）。
- 容器场景下命令需添加 `--docker` 参数，并配置环境变量：

    ```sh
    export PATH=/usr/local/Ascend/driver/tools:$PATH
    ```

## 参数说明

### -d \<deviceId> 或 --device \<deviceId>

可选，指定 Device ID（逻辑 ID），默认导出所有 Device 日志。部分产品存在 SMP 工作模式（多个 Device 共用一个 OS 组成一组），此时指定组内任意 Device ID，会导出该 OS 下所有 Device 的日志。

**该参数仅支持通过方式一传入**，不支持通过方式二传入。

### 导出范围参数对比

不指定参数、`-a`、`-f` 三者为递进关系，差异集中在黑匣子相关信息：

| 导出内容 | 不指定参数 | `-a`/`--all` | `-f`/`--force` |
|---------|:---:|:---:|:---:|
| Control CPU 上的系统类日志 | ✓ | ✓ | ✓ |
| 非 Control CPU 上的系统类日志 | ✓ | ✓ | ✓ |
| Device 侧进程 coredump 时的调用栈信息 | ✓ | ✓ | ✓ |
| 黑匣子日志 | ✓ | ✓ | ✓ |
| 黑匣子设备事件信息 | — | ✓ | ✓ |
| 黑匣子存储空间中的历史维测信息 | — | — | ✓ |
| 事件调度模块的维测信息 | ✓ | ✓ | ✓ |
| Host 侧驱动内核日志 | ✓ | ✓ | ✓ |
| Device 侧 OS 系统日志 | ✓ | ✓ | ✓ |

> 说明：执行 `-a` 或 `-f` 导出黑匣子相关信息时，如果存在卡住或导出慢的情况，建议执行 `npu-smi set -t reset -i id -c chip_id` 复位芯片后重新导出，详细请参考《npu-smi 命令参考》。

### -t \<type_id> 或 --type \<type_id>

可选，指定导出的日志类型：

| 取值 | 导出内容 |
|------|---------|
| 0 | Device 侧系统类日志（Control CPU + 非 Control CPU）；其他维测信息：coredump 调用栈、黑匣子日志、黑匣子设备事件信息、事件调度模块维测信息、Host 侧驱动内核日志、Device 侧 OS 系统日志 |
| 1 | Device 侧系统类日志（Control CPU + 非 Control CPU）；事件调度模块维测信息、Device 侧 OS 系统日志和其他模块日志（如 DVPP、TEE 日志、驱动 proc 日志等，各产品略有差异） |
| 2 | 黑匣子日志、黑匣子设备事件信息、黑匣子存储空间中的历史维测信息。**不支持多进程并发执行** |
| 3 | Device 侧进程 coredump 时的调用栈信息 |
| 4 | vmcore 文件。**不支持多进程并发执行** |
| 5 | 其他模块日志，如 DVPP、TEE 日志、驱动 proc 日志等，不同产品型号支持的类型略有差异 |
| 6 | Unified Bus 统一总线的维测信息 |
| 7 | AO（Always Online）区的日志信息 |
| 8 | AO 计数 |
| 9 | 串口录音 |

**`-t 2` 补充**：对于 Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品，还支持导出当前硬件寄存器信息，导出后建议执行 `npu-smi set -t reset -i id -c chip_id` 复位芯片。

**`-t 4` 补充**：对于 Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品、Ascend 950PR/Ascend 950DT，当 Device OS 心跳丢失时会同时生成 vmcore 文件和黑匣子日志，其他产品不会生成 vmcore 文件。

**`-t 6`/`-t 7`/`-t 8`/`-t 9` 支持情况**：仅 Ascend 950PR/Ascend 950DT 支持；Atlas 200I/500 A2 推理产品、Atlas 推理系列产品、Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品均不支持。

## 使用示例

1. 以 root 用户登录 Host 侧服务器。
2. 创建一个目录用来存放导出的文件（如 `/var/log/npu/report`），确保当前用户有读、写、执行权限。
3. 进入以上目录，执行导出命令。

    ```sh
    msnpureport report
    ```

## 输出说明

Device 侧的相关日志和文件被导出到 Host 侧，存储到当前目录下以时间戳命名的文件夹中。目录结构见 [output-layout.md](output-layout.md)。
