# 查询 Device 维测配置信息

## 命令功能

查询 Device 配置信息，包括 Device 日志级别、TaskSchedule 是否自动复位加速器、AI Core 上任务串联或并行执行模式等。

## 命令格式

方式一（推荐，查询全量配置）：

```sh
msnpureport config --get [--device <deviceId>]
```

方式二（仅支持查询日志级别）：

```sh
msnpureport -r [--device <deviceId>]
```

## 注意事项

容器场景下，命令执行时需要添加 `--docker` 参数，且需要配置 msnpureport 环境变量：

```sh
export PATH=/usr/local/Ascend/driver/tools:$PATH
```

## 参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `-d <deviceId>` 或 `--device <deviceId>` | 可选 | 指定 Device ID（逻辑 ID），**默认为 0**。SMP 工作模式下，指定一个 OS 内的任意 Device ID 时，会导出该 OS 下的所有 Device 日志 |
| `-r` 或 `--request` | 必选 | 查询 Device 侧 slog 系统类日志的级别，包括全局级、模块级和是否开启 Event 日志。不指定 Device ID 时默认查询 Device 0。**该参数仅方式二支持** |

> 注意：与导出命令不同，查询命令不指定 `-d` 时默认只查 Device 0，而非所有 Device。

## 使用示例

```sh
msnpureport config --get
```

## 输出说明

```sh
+--------------------------------+--------------------------------+
| Device ID: 0                   | Current Configuration          |
+================================+================================+
| Icache check Range             | 1000                           |
| Accelerator Recover            | Enable                         |
| Aic Coremask                   | 0x1ff1cff                      |
| Aiv Coremask                   | 0x3ffff03f0ffff                |
| Aic Singlecommit               | Enable                         |
+-------- Log Level -------------+--------------------------------+
| Global Level                   | ERROR                          |
| Event Level                    | ENABLE                         |
| Module [IMP]                   | ERROR                          |
| Module [IMU]                   | ERROR                          |
| Module [LP]                    | ERROR                          |
| Module [TSDUMP]                | ERROR                          |
| Module [TS]                    | ERROR                          |
...
```

## 字段解读

| 字段 | 含义 | 相关设置命令 |
|------|------|------------|
| `Icache check Range` | icache bit 翻转校验范围（单位 KB） | `--icachecheck`，见 [config-set-aicore.md](config-set-aicore.md) |
| `Accelerator Recover` | TaskSchedule 是否自动复位加速器 | `--accelerator_recover` |
| `Aic Coremask` | AI Core 屏蔽状态，**bitmap 结构** | `--aic_switch` + `--coreid` |
| `Aiv Coremask` | Vector Core 屏蔽状态，**bitmap 结构** | `--aiv_switch` + `--coreid` |
| `Aic Singlecommit` | AI Core 内部多指令串行/并行 | `--singlecommit` |
| `Global Level` | Device 侧系统类日志全局级别 | `config --set --log -g`，见 [config-set-log.md](config-set-log.md) |
| `Event Level` | Event 日志是否开启 | `config --set --log -e` |
| `Module [xxx]` | 各模块日志级别 | `config --set --log -m` |

**Coremask 用途**：屏蔽核后想确认实际剩余核数，用本命令查询 `Aic Coremask`/`Aiv Coremask`。算子所需核数可从 Host 侧应用调试日志中查看，默认路径 `$HOME/ascend/log/debug/plog/plog-{pid}_*.log`，搜索关键字 `core_num`。
