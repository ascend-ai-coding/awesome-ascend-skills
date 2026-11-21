# npu-compute CLI 使用说明

## 命令格式

```text
npu-compute [options] [--] [program] [program-arguments]
```

采集命令由工具选项、目标程序和目标程序参数组成。工具选项写在目标程序之前；识别到目标程序后，其余参数均传递给目标程序。`--` 用于明确结束工具选项解析，下一项是目标程序。

## 选项

| 选项 | 含义 |
|---|---|
| `-h`, `--help` | 输出帮助信息。出现参数错误时，先输出错误，再输出帮助。 |
| `--list-sets` | 输出当前安装版本支持的 Set 及其有序 Section 成员。 |
| `--set <name>` | 选择预定义 Set。名称区分大小写，可以重复指定，也可以和 `--section` 组合。 |
| `--list-sections` | 输出当前安装版本支持的 Section 名称。 |
| `--section <name>` | 选择一个 Section。名称区分大小写，可重复指定；重复项按首次出现去重。 |
| `--replay-mode <mode>` | 预留采集模式选项，当前仅支持 `kernel`；不指定时使用 `kernel`。 |
| `-i`, `--import <report>` | 解包 npu-compute 生成的 `.npu-rep` 报告。 |
| `-o`, `--export <path>` | 指定采集报告文件、报告父目录或解包父目录。 |
| `--` | 结束工具选项解析。 |

`--help` 可以和其他合法参数同时出现，此时只输出帮助。`--list-sets` 和 `--list-sections` 均为独立查询操作，不启动目标程序；与 `--help` 同时使用时只输出帮助。`--list-sets` 不接受值，也不能重复指定。

## Set 和 Section

当前 Set 为：

- `basic`：`Pipeline`、`PipeUtilization`、`Memory`、`MemoryL0`、`MemoryUB`、`L2Cache`、`ArithmeticUtilization`。
- `full`：按 `basic` 的顺序包含全部成员，并在末尾增加 `ResourceConflictRatio`。

Set 名称区分大小写。可运行以下命令核对实际安装版本提供的集合：

```bash
npu-compute --list-sets
```

当前 Section 为：

- `PipeUtilization`
- `Memory`
- `MemoryL0`
- `MemoryUB`
- `L2Cache`
- `Pipeline`
- `ArithmeticUtilization`
- `ResourceConflictRatio`

指定目标程序但未指定 `--set` 和 `--section` 时，默认使用 `basic`。明确指定任意 `--section` 后，不再自动补充默认 `basic`。

`--set` 和 `--section` 可以重复、交错使用。工具按命令行出现顺序展开 Set，并按 Section 首次出现的位置去重。例如，`--section Memory --set basic` 会先保留 `Memory`，再追加 basic 中尚未出现的 Section。

可运行以下命令核对实际安装版本支持的 Section：

```bash
npu-compute --list-sections
```

## 目标程序和参数

目标程序可以使用：

- 相对于当前工作目录的路径，例如 `./build/add_custom`；
- 可由 `PATH` 查找到的可执行命令，例如 `python3`。

这里的可执行命令可以是二进制程序，也可以是具有执行权限和有效解释器声明的脚本。脚本没有执行权限时，显式指定解释器：

```bash
npu-compute --section Memory bash run.sh
```

目标程序后的所有内容都作为目标程序参数，包括与工具选项同名的参数：

```bash
npu-compute --set basic -- ./add_custom --help --section app-value
```

`--` 显式结束 npu-compute 工具选项解析。`--` 后第一项是目标程序，其余内容全部原样传给目标程序；目标程序参数可能与工具选项同名时优先使用该形式。

## 采集

默认 basic：

```bash
npu-compute ./add_custom
```

显式 basic：

```bash
npu-compute --set basic ./add_custom
```

完整指标：

```bash
npu-compute --set full ./add_custom
```

basic 追加额外 Section：

```bash
npu-compute --set basic --section ResourceConflictRatio ./add_custom
npu-compute --section Memory --set basic ./add_custom
```

单 Section：

```bash
npu-compute --section Memory ./add_custom
```

多个 Section：

```bash
npu-compute \
  --section PipeUtilization \
  --section Memory \
  --section L2Cache \
  ./add_custom
```

未指定 `--export` 时，报告以唯一名称写入当前目录。采集成功后终端输出：

```text
npu-compute: report= <报告路径>
```

### 指定报告输出

指定一个尚不存在的 `.npu-rep` 文件；该文件的父目录必须存在：

```bash
npu-compute --section Memory --export ./memory.npu-rep ./add_custom
```

指定已有目录时，工具在该目录中生成唯一命名的 `.npu-rep`：

```bash
npu-compute --section Memory --export ./reports ./add_custom
```

工具不会覆盖已有报告文件。

## 解包报告

在当前目录创建唯一解包目录：

```bash
npu-compute --import ./memory.npu-rep
```

在指定的已有父目录中创建唯一解包目录：

```bash
npu-compute --import ./memory.npu-rep --export ./unpacked-results
```

解包成功后终端输出实际目录：

```text
npu-compute: unpacked= <目录>
```

读取该路径中的文件，不假定数据直接写在 `--export` 指定的父目录中。

## 使用约束

- Set 名称区分大小写。
- Section 名称区分大小写。
- `--list-sets` 和 `--list-sections` 用于查询，不启动目标程序，也不生成采集报告。
- 采集需要目标程序；未指定 Set 或 Section 时默认使用 `basic`。
- `--set` 和 `--section` 不能与 `--import` 组合。
- `--import` 用于解包，不启动目标程序。
- 目标程序或其脚本再次启动 npu-compute 采集时，外层采集会失败。
- 目标程序返回非零状态或被信号终止时，不发布成功报告。
