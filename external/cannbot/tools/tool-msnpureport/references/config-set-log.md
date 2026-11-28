# 设置 Device 侧系统类日志级别

## 命令功能

设置 Device 侧系统类日志的日志级别。

## 命令格式

方式一（推荐）：

```sh
msnpureport config --set --log [options]
```

方式二：

```sh
msnpureport [options]
```

## 注意事项

- 该命令**仅支持 root 用户**执行。
- 容器场景下，命令执行时需要添加 `--docker` 参数，且需要配置 msnpureport 环境变量：

    ```sh
    export PATH=/usr/local/Ascend/driver/tools:$PATH
    ```

## 参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `-d <deviceId>` 或 `--device <deviceId>` | 可选 | 指定 Device ID（逻辑 ID），默认为 0。SMP 工作模式下，指定一个 OS 内的任意 Device ID 时，会导出该 OS 下的所有 Device 日志 |
| `-g <level>` 或 `--global <level>` | 可选 | 设置全局级的日志级别。取值：`debug`、`info`、`warning`、`error`、`null`（不输出日志） |
| `-m <module:level>` 或 `--module <module:level>` | 可选 | 设置模块级的日志级别。`module` 为模块名称（例如 SLOG 等），`level` 取值：`debug`、`info`、`warning`、`error`、`null` |
| `-e <level>` 或 `--event <level>` | 可选 | 设置是否开启 Event 日志。`enable`：开启（**默认开启**）；`disable`：不开启 |

> 可用模块名可通过 `msnpureport config --get` 输出中的 `Module [xxx]` 字段确认。

## 使用示例

设置全局日志级别为 info：

```sh
msnpureport config --set --log -g info
```

设置 CCE 模块的日志级别为 debug：

```sh
msnpureport config --set --log -m CCE:debug
```

设置是否开启 Device 侧 event 日志：

```sh
msnpureport config --set --log -e enable
```

以上三种方式均可用 `-d`/`--device` 指定对应逻辑 ID 的设备，例如设置逻辑 ID 为 1 的设备全局日志级别为 info：

```sh
msnpureport config --set --log -g info -d 1
```

## 输出说明

无。设置结果可通过 `msnpureport config --get` 查询确认。

## SMP 形态特殊说明

部分产品可能存在 SMP 工作模式，即多个 Device 共用一个 OS 组成一组，OS 上部署 Device 共用的进程模块，各个 Device 小核上部署各自独有的模块，如 TS、LP、IMP 和 IMU 模块。

此时通过 `-g -d` 设置日志级别，表示修改 **OS 上共用进程模块**及 `-d` 指定 **Device 小核上所有模块**的日志级别。

## 物理 ID 与逻辑 ID 换算

- 使用 `npu-smi info` 命令查看到的设备上的 NPU Device ID 即为**物理 ID**。假设物理 ID 的取值范围是 0~7，Device0~Device3 四个为一组，Device4~Device7 四个为一组。物理 ID 的分组情况请以设备实际情况为准。
- 查询到的物理 ID 会按照数字大小从小到大自上而下排列，对应的**逻辑 ID** 则从 0 开始按顺序为 0~n。

**换算示例**：查询到的物理 ID 为 `0, 1, 4, 5`，那么对应逻辑 ID 则为 `0, 1, 2, 3`。根据分组情况可判断物理 ID `0, 1` 为一组、`4, 5` 为一组，那么可以推断出逻辑 ID `0, 1` 为一组、`2, 3` 为一组。
