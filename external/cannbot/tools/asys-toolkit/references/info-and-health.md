# 软硬件、Device 状态信息展示与健康检查

覆盖 `asys info` 与 `asys health`。

---

## asys info — 软硬件与 Device 状态

收集安装包版本信息、Device 温度、功率等。

### 命令格式

```bash
asys info -r="status" -d=deviceId
```

### 参数

| 参数 | 必选性 | 说明 |
|------|:---:|------|
| `-r` | **必选** | 指定需展示的信息类型，取值 `status` / `software` / `hardware` |
| `-d` | 可选 | 指定需要展示信息的 deviceId。不设置时默认展示 device 0 的信息，**仅 `-r=status` 时有效** |

### `-r` 三种取值

| 取值 | 输出内容 |
|------|---------|
| `status` | device 的信息，包含芯片型号、温度、健康状态、CPU 和 AI Core 信息等 |
| `software` | Host 的软件信息，包含系统和内核版本、CANN 包版本等 |
| `hardware` | Host 和 Device 的硬件信息 —— Host 的 CPU 型号与核数、内存容量和硬盘容量；Device 的 NPU 个数与型号，AI CPU / AI Core / AI Vector 个数等 |

### 使用示例

```bash
# 查看 device 0 的运行状态
asys info -r="status" -d=0

# 查看 CANN 包版本与系统内核版本
asys info -r="software"

# 查看 Host 与 Device 硬件规格
asys info -r="hardware"
```

### `-r=status` 输出示例

各产品型号的输出信息有所不同，请以实际输出信息为准。

```
 +----------------------------------+------------------------+
 | Device ID: 0                     | INFORMATION            |
 +==================================+========================+
 | Chip Name                        | Ascend xxxxxxxxxx      |
 | Power (W)                        | 1021                   |
 | Temperature (C)                  | 55                     |
 | health                           | Healthy                |
 +--- CPU Information --------------+------------------------+
 | AI CPU Count                     | 6                      |
 | AI CPU Usage (%)                 | 0                      |
 | Control CPU Count                | 1                      |
 | Control CPU Usage (%)            | 1                      |
 | Control CPU Frequency (MHZ)      | 2000                   |
 +--- AI Core Information ----------+------------------------+
 | AI Core Count                    | 20                     |
 | AI Core Usage (%)                | 0                      |
 | AI Core Frequency (MHZ)          | 800                    |
 | AI Core Voltage (MV)             | 900                    |
 +--- Memory Information -----------+------------------------+
 | HBM Total (MB)                   | 65536                  |
 | HBM Used (MB)                    | 3382.52                |
 | HBM Bandwidth Usage (%)          | 0                      |
 | HBM Frequency (MHZ)              | 1600                   |
 +----------------------------------+------------------------+
```

四个区段的字段：

| 区段 | 字段 |
|------|------|
| 顶部概要 | Chip Name、Power (W)、Temperature (C)、health |
| CPU Information | AI CPU Count、AI CPU Usage (%)、Control CPU Count、Control CPU Usage (%)、Control CPU Frequency (MHZ) |
| AI Core Information | AI Core Count、AI Core Usage (%)、AI Core Frequency (MHZ)、AI Core Voltage (MV) |
| Memory Information | HBM Total (MB)、HBM Used (MB)、HBM Bandwidth Usage (%)、HBM Frequency (MHZ) |

> `AI Core Voltage (MV)` 可用于压力检测前后的电压对比，见 [diagnose-and-config.md](diagnose-and-config.md)。

---

## asys health — 健康检查

检查所有 Device 或指定 Device 的健康状态。若不健康，会展示报错信息。

### 命令格式

```bash
asys health -d=deviceId
```

### 参数

| 参数 | 是否必选 | 说明 |
|------|:---:|------|
| `-d` | 可选 | 指定需要显示健康状态的 deviceId。不指定时显示所有 device 的健康状态；指定时若 device 有异常，则在终端屏幕显示故障码和故障信息，**仅显示前 5 组故障** |

在故障信息收集和业务复跑+故障信息收集时，会将**所有**故障码和故障信息写入 `health_result.txt` 文件。

### 使用示例和输出

不指定 device，所有 device 都正常（以双卡为例）：

```
asys health
 +------------------------+------------------------------+
 | Group of 2 Device      | Overall Health: Healthy      |
 +========================+==============================+
 | Device ID: 0           | Healthy                      |
 +------------------------+------------------------------+
 | Device ID: 1           | Healthy                      |
 +------------------------+------------------------------+
```

指定 device，device 正常：

```
asys health -d=0
 +-------------------+------------------------------+
 | Device ID: 0      | Overall Health: Healthy      |
 |                   | ErrorCode Num: 0             |
 +===================+==============================+
```

指定 device，device 异常：

```
asys health -d=0
 +-------------------+------------------------------+
 | Device ID: 0      | Overall Health: Warning      |
 |                   | ErrorCode Num: 1             |
 +===================+==============================+
 | 0xa419321c        | lp pmbus error               |
 +-------------------+------------------------------+
```

### 输出字段

| 字段 | 含义 |
|------|------|
| `Group of N Device` | 分组内 device 数量，仅在不指定 `-d` 时出现 |
| `Overall Health` | 整体健康状态 |
| `Device ID` | 设备编号 |
| `ErrorCode Num` | 故障码数量，指定 `-d` 时显示 |
| 故障行 | 左列为故障码，右列为故障描述 |

### 状态与故障级别映射

| 故障级别 | asys health 状态 |
|---------|-----------------|
| 提示 | `Healthy` |
| 次要 | `Warning` |
| 重要 | `Alarm` |
| 紧急 | `Critical` |
| 未知 | `Unknown` |

**判读要点**：

- **没有问题、正常状态也会显示 `Healthy`**，与「提示」级别共用该状态值。看到 `Healthy` 不代表完全没有告警项。
- 终端屏幕仅显示前 5 组故障，完整清单看 `health_result.txt`。
- 故障码的详细描述需查阅对应版本的《黑匣子异常错误码信息列表》和《健康管理故障定义》。

### 设备可用性判断

供环境检查类流程复用的简化判据：

| 状态 | 可用性 |
|------|--------|
| `Healthy` | 可用 |
| `Warning` | 可用 |
| 其他（`Alarm` / `Critical` / `Unknown`） | 不可用 |
