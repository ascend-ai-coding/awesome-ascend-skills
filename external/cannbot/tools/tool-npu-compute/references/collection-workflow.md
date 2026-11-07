# npu-compute 采集与报告流程

## 确认请求类型

用户只询问使用方式时，生成命令但不执行。用户明确要求运行或采集时，才启动 npu-compute 和目标程序。

生成命令前确认目标程序及其工作目录、指标范围、目标程序参数以及是否指定报告输出位置。指标范围可以是默认 basic、显式 Set、一个或多个 Section，或 Set 与 Section 的组合。缺少的信息会改变程序行为时，不猜测参数。

## 采集前检查

1. 使用 Set 时运行 `npu-compute --list-sets`；使用明确 Section 时运行 `npu-compute --list-sections`，确认安装版本支持所需名称。
2. 确认目标程序能从指定工作目录访问。
3. 脚本无法直接执行时，使用其解释器。
4. 指定 `.npu-rep` 文件时，确认父目录已存在且目标文件尚不存在。
5. 指定报告目录时，确认目录已存在。

## 确认集合和执行采集

需要查看集合内容时单独执行，不附加目标程序：

```bash
npu-compute --list-sets
```

确认指标范围后，从目标程序正常运行所需的工作目录执行采集：

```bash
npu-compute ./add_custom
npu-compute --set basic ./add_custom
npu-compute --set full ./add_custom
npu-compute --set basic --section ResourceConflictRatio ./add_custom
npu-compute --set basic -- ./add_custom --help
```

未指定 `--set` 和 `--section` 时默认使用 `basic`。同时指定 Set 和 Section 时，按用户表达顺序生成选项，按最终命令行出现顺序展开，并按 Section 首次出现的位置去重。`--` 后第一项是目标程序，其余内容均作为目标程序参数。

Set 只是批量选择 Section 的方式，不改变 `.npu-rep` 报告生成、确认和解包流程。

记录实际执行命令、npu-compute 退出状态、`report` 路径以及目标程序和 npu-compute 输出的错误。

采集成功以退出状态为 `0`、终端发布 `report` 路径且报告文件存在为准。目标程序自身输出成功不能单独证明采集成功。

## 解包报告

默认解包到当前目录下的唯一子目录：

```bash
npu-compute --import ./report.npu-rep
```

也可以指定已有父目录：

```bash
npu-compute --import ./report.npu-rep --export ./unpacked-results
```

解包成功后记录退出状态和终端输出的 `unpacked` 路径，并检查该目录实际存在。报告中的子目录会在解包结果中恢复相同层级。

检查完整解包结果：

```bash
python3 scripts/inspect_collection.py <unpacked 路径> --format markdown
```

检查结果分别列出根级文件、各子结果集合和未识别项目。未识别项目保留相对于输入路径的位置，不将其归入已知结果。

## 识别结果

| 文件 | 处理方式 |
|---|---|
| `HardwareInfo.jsonl` | 检查主机和 Device 信息。 |
| `<Section>.csv` | 检查表头、数据行、字段含义和可用性。 |
| `PipeTrace.json` | 检查 Pipeline 事件、轨道和时间范围。 |
| `summary.jsonl` | 识别 PMU Section 汇总记录和 `OpInfoSummary`。 |

存在 `collection-p...` 子目录时，分别记录根级结果和每个子结果集合，不把不同结果集合中的同名文件合并为一份数据。

## 失败处理

- Set 或 Section 不支持时，报告安装版本返回的合法名称。
- 目标程序不可执行时，区分路径不存在、无执行权限和脚本缺少解释器。
- 目标程序返回非零状态时，报告目标程序及其状态，不声称报告发布成功。
- 缺少 `HardwareInfo.jsonl`、报告打包失败或解包失败时，保留终端错误供定位。
- 检测到嵌套采集时，移除目标程序或脚本中的内层 npu-compute 调用后重新采集。
