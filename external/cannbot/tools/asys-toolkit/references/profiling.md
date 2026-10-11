# 性能数据采集

`asys profiling` 采集关键性能数据，辅助分析性能问题。底层下发 msprof 命令。

> **注意事项**：对于 Atlas 200I/500 A2 推理产品、Atlas 推理系列产品、Atlas 训练系列产品，**不支持**使用该功能。

---

## 命令格式

```bash
asys profiling -r=aicore -p=time -d=deviceId --output=./ --aic_metrics=PipeUtilization
```

## 参数

| 参数 | 必选性 | 说明 |
|------|:---:|------|
| `-r` | **必选** | 采集类型，类型为字符串枚举。**支持输入多个枚举类型，以英文逗号分隔** |
| `-p` | **必选** | 采集间隔，单位秒。最小值 `1`，最大值 `30*24*3600` |
| `-d` | 可选 | 指定待操作的 deviceId，**仅支持输入单个 deviceId**，默认值 `0` |
| `--output` | 可选 | 结果输出目录的路径前缀，最终输出目录为 `{output}/asys_profiling_result_timestamp`。不带该参数时输出到命令行执行目录 |
| `--aic_metrics` | 可选 | AI Core PMU（performance monitor unit，性能监测单元）类型，**当采集类型包含 `aicore` 时该参数生效** |

### `-r` 采集类型

| 取值 | 采集内容 |
|------|---------|
| `dvpp` | dvpp 的性能数据，例如执行时间、利用率等 |
| `aicore` | AI Core 的性能数据，例如 cube 及 vector 类型指令耗时和占比、计算单元和搬运单元耗时占比等 |
| `os` | 系统内存数据、AI CPU 利用率、Ctrl CPU 利用率等 |
| `memory` | 内存读取速率和带宽数据，包括片上内存、三级缓存等 |
| `link` | 带宽数据，例如集合通信带宽、PCIe 带宽等 |
| `power` | 低功耗数据 |

### `--aic_metrics` PMU 类型

| 取值 | 含义 |
|------|------|
| `PipeUtilization` | **默认值**。计算单元和搬运单元耗时占比 |
| `ArithmeticUtilization` | cube 及 vector 类型指令耗时和占比 |
| `Memory` | 内存读写带宽速率 |
| `MemoryL0` | L0 读写带宽速率 |
| `MemoryUB` | UB 读写带宽速率 |
| `ResourceConflictRatio` | 资源冲突占比 |
| `L2Cache` | L2Cache 命中率 |
| `MemoryAccess` | 算子在 AI Core 上的访存带宽数据量 |

---

## 使用示例

```bash
# 采集 AI Core 的性能数据
asys profiling -r=aicore -p=10 -d=0 --output=./ --aic_metrics=PipeUtilization

# 多类型同时采集（英文逗号分隔）
asys profiling -r=aicore,memory,link -p=10 -d=0 --output=$HOME/prof
```

---

## 输出说明

命令执行成功后会提示如下信息，并在 `{output}/asys_profiling_result_timestamp` 目录下生成采集结果文件：

```
2025-11-27 20:15:45,141 [ASYS] [INFO]: asys start.
2025-11-27 20:15:45,141 [ASYS] [INFO]: Start run: msprof --output=./ --sys-period=10 --sys-devices=0 --ai-core=on --aic-mode=sample-based --aic-metrics=PipeUtilization, please wait about 10 seconds.
2025-11-27 20:16:04,335 [ASYS] [INFO]: Succeeded in running aicore profiling, [INFO] Start profiling....
[INFO] Start export data in PROF_000001_20251127201545157_03062849EPFNHDPB.
......
[INFO] Query all data in PROF_000001_20251127201545157_03062849EPFNHDPB done.
[INFO] Profiling finished.
[INFO] Process profiling data complete. Data is saved in /xxx/ascend_system_advisor/asys/asys_profiling_result_20251127201545110/PROF_000001_20251127201545157_03062849EPFNHDPB
2025-11-27 20:16:04,336 [ASYS] [INFO]: profiling task execute finish.
2025-11-27 20:16:04,336 [ASYS] [INFO]: asys finish.
```

日志会**回显实际下发的 msprof 命令**，并提示大约需等待与 `-p` 相同的秒数。最终落盘路径见 `Data is saved in` 一行。

结果文件的详细解释参见《性能调优工具》中的性能数据文件参考。

---

## 注意事项

- 不带 `--output` 时，结果落在命令行执行目录。
- `--output` 取值为空、无效字符串、指定路径目录无写权限、或创建目录失败时，asys 退出执行并报错。
- `-d` **仅支持输入单个 deviceId**。

## 与算子性能采集的分工

`asys profiling` 是系统级周期采样，按 `-p` 指定的间隔采集整机/整卡指标，适合看系统层面的资源占用趋势。单算子的耗时拆解、算子级 PMU 分析走 msprof 主路径，见 `ops-profiling` skill。
