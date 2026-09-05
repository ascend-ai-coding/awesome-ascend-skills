# AI Core 相关维测配置设置

本文覆盖四类 AI Core 故障定位配置：TaskSchedule 自动复位加速器、AI Core singlecommit、屏蔽 AI Core/Vector Core、icache bit 翻转校验范围。

## 通用注意事项

以下注意事项适用于本文全部命令：

- **仅支持 root 用户**执行。
- 容器场景下需添加 `--docker` 参数，并配置环境变量：

    ```sh
    export PATH=/usr/local/Ascend/driver/tools:$PATH
    ```

- **昇腾 AI 应用进程运行过程中不建议执行这些命令**，否则可能会导致应用进程运行异常，建议应用进程退出后执行。
- `-d <deviceId>`/`--device <deviceId>` 为可选参数，指定 Device ID（逻辑 ID），**默认为 0**。SMP 工作模式下，指定一个 OS 内的任意 Device ID 时，会导出该 OS 下的所有 Device 日志。
- 配置生效后可用 `msnpureport config --get` 查询确认，见 [config-query.md](config-query.md)。

## 恢复方式速查

| 配置 | 定位用取值 | 恢复方式 |
|------|-----------|---------|
| `--accelerator_recover` | `0`（不自动复位） | **必须重启运行环境**才能恢复 AI Core 业务；重启后默认自动复位 |
| `--singlecommit` | `1`（串行） | 设为 `0`；重启运行环境后默认关闭 |
| `--aic_switch` | `0`（屏蔽） | 设为 `1` 且 `--coreid 0xFFFF`；重启运行环境后默认不屏蔽 |
| `--aiv_switch` | `0`（屏蔽） | 设为 `1` 且 `--coreid 0xFFFF`；重启运行环境后默认不屏蔽 |
| `--icachecheck` | 按需设置 | 设置前先查询原值，定位后改回 |

---

## 设置 TaskSchedule 是否自动复位加速器

### 命令功能

设置 TaskSchedule 是否自动复位加速器，以便导出更详细、更精准的寄存器信息定位问题，但**配置后会影响执行性能**。

### 命令格式

```sh
msnpureport config --set [--device <deviceId>] --accelerator_recover <recoverflag>
```

### 参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `--accelerator_recover <recoverflag>` | 必选 | `0`：TaskSchedule 不自动复位加速器。设置为该值、完成问题定位后，**必须重启运行环境恢复 AI Core 业务**，重启环境后默认自动复位加速器。<br>`1`：TaskSchedule 自动复位加速器，**默认该选项** |

### 配合导出寄存器信息的定位流程

针对 Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品，AI Core 问题定位场景下：

1. 执行本命令设置不自动复位加速器（`--accelerator_recover 0`）。
2. **再次复现问题**。
3. 使用 `--type 2` 参数导出，可获得更详细、更精准的寄存器信息，见 [export-one-time.md](export-one-time.md)。
4. 定位完成后重启运行环境恢复业务。

### 使用示例

```sh
msnpureport config --set --device 1 --accelerator_recover 0
```

---

## 设置 AI Core 的 singlecommit

### 命令功能

设置 AI Core 上任务串联或并行执行。

### 命令格式

```sh
msnpureport config --set [--device <deviceId>] --singlecommit <singlecommitflag>
```

### 参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `--singlecommit <singlecommitflag>` | 必选 | `0`：关闭 AI Core singlecommit 模式，此时 AI Core 内部多指令并行，**默认该选项**。<br>`1`：开启 AI Core singlecommit 模式，此时 AI Core 内部多指令串行。设置为该值、完成问题定位后需及时关闭；若重启运行环境，默认关闭 |

### 使用示例

```sh
msnpureport config --set --device 1 --singlecommit 0
```

---

## 屏蔽指定 AI Core 上的任务执行

### 命令功能

屏蔽指定 AI Core 上的任务执行，以便排查哪个 AI Core 故障。**建议单算子调用场景下使用该命令排查问题。**

对于 AI Core 分离架构的产品来说，此处的 AI Core 特指 **AIC 核**。分离架构是将 AI Core 拆成矩阵计算（AI Cube，AIC）和向量计算（AI Vector，AIV）两个独立的核，从而实现矩阵计算与向量计算的解耦。典型的分离架构产品包括 Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品、Ascend 950PR/Ascend 950DT。

### 命令格式

```sh
msnpureport config --set [--device <deviceId>] --aic_switch <switchflag> --coreid <coreid>
```

### 参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `--aic_switch <switchflag>` | 必选 | `0`：屏蔽 AI Core。设置为该值后，若重启运行环境，默认不屏蔽。<br>`1`：不屏蔽 AI Core，**默认该选项** |
| `--coreid <coreid>` | 必选 | 指定 core id，多个 core id 之间以英文逗号分隔，**最多可同时指定 4 个** AI Core。只要有一个 core id 无效，该命令即返回报错。<br>恢复初始值：`--aic_switch 1` + `--coreid 0xFFFF` |

### 核数约束

对于涉及**核间同步**的算子，算子必须一次拿到所需要的核数才执行。使用该命令屏蔽指定核后，若屏蔽后的实际核数小于算子需要的核数，会导致**算子执行失败**。

- 屏蔽后的实际核数：用 `msnpureport config --get` 查询 `Aic Coremask`。
- 算子需要的核数：查看 Host 侧应用程序调试日志（默认路径 `$HOME/ascend/log/debug/plog/plog-{pid}_*.log`），搜索关键字 `core_num`。

### 使用示例

```sh
msnpureport config --set --device 0 --aic_switch 0 --coreid 3,4
```

---

## 屏蔽指定 Vector Core 上的任务执行

### 命令功能

屏蔽指定 Vector Core 上的任务执行，以便排查哪个 Vector Core 故障。**建议单算子调用场景下使用该命令排查问题。**

### 命令格式

```sh
msnpureport config --set [--device <deviceId>] --aiv_switch <switchflag> --coreid <coreid>
```

### 参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `--aiv_switch <switchflag>` | 必选 | `0`：屏蔽 Vector Core。设置为该值后，若重启运行环境，默认不屏蔽。<br>`1`：不屏蔽 Vector Core，**默认该选项** |
| `--coreid <coreid>` | 必选 | 指定 core id，多个 core id 之间以英文逗号分隔，**最多可同时指定 4 个** Vector Core。只要有一个 core id 无效，该命令即返回报错。<br>恢复初始值：`--aiv_switch 1` + `--coreid 0xFFFF` |

### 核数约束

同 AI Core 屏蔽，剩余核数不足会导致涉及核间同步的算子执行失败。实际剩余核数查询 `Aiv Coremask` 字段。

### 使用示例

```sh
msnpureport config --set --device 0 --aiv_switch 0 --coreid 5,6
```

### 产品支持

Atlas 200I/500 A2 推理产品、Atlas 训练系列产品**不支持**该命令，详见 [product-support.md](product-support.md)。

---

## 设置 icache bit 翻转校验范围

### 命令功能

设置 icache bit 翻转校验范围，以便定位算子问题。

### 命令格式

```sh
msnpureport config --set [--device <deviceId>] --icachecheck <value>
```

### 参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `--icachecheck <value>` | 必选 | 取值范围：[0, 131072]，单位 KB。例如设置为 128，则校验从出错 PC 往前 128K 到出错 PC 往后 128K 的 icache 与 GM（Global Memory）是否一致 |

设置前可先用 `msnpureport config --get` 获取当前环境上的 icache 翻转校验范围（即 `Icache check Range` 字段值）。

### 使用示例

```sh
msnpureport config --set --device 0 --icachecheck 128
```

### 产品支持

Atlas 训练系列产品**不支持**该命令。
