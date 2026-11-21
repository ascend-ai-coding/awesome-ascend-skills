# 分析 AI Core Error 问题

## 功能说明

分析 AI Core Error 问题的故障信息，辅助定位 AI Core Error 问题。

执行业务时，若日志文件或屏幕打印信息中包含如下 AI Core Error 报错，需要先获取 AI Core Error 问题相关的故障信息，再配合使用 msaicerr 分析故障信息：

```bash
# 报错示例
there is an xx aicore error

# 或报错示例
there is an xx aivec error
```

## 注意事项

在收集到的故障信息中，请提前检查：

- `dfx/data-dump` 目录下是否存在 dump 文件
- 是否存在异常算子编译信息（算子编译 `*.o` 和 `*.json` 文件）
- `dfx/log/host/cann` 目录下是否存在日志文件

若不存在，则无法使用 msaicerr 工具提取 AI Core Error 信息。

## 命令格式

```bash
python3 msaicerr.py -p path1 -out path2 -dev 0
```

## 参数说明

### -p 或 --report_path

**必选参数**，分析 AI Core Error 问题时用于指定 AI Core Error 故障信息所在的目录。

不能进入 `-p` 参数指定的目录或子目录下执行 msaicerr 工具，否则会出现工具解析卡住或失败的情况。

### -out 或 --output_path

可选参数，指定解析结果文件的存放路径。如果不指定，则解析结果默认存放在执行命令的当前路径下。

- `-out` 参数指定的目录**不能为 `-p` 参数指定的目录或子目录**，否则会出现工具解析卡住或失败的情况。
- 若 `-out` 参数指定值为空或无效字符串、或指定目录无写权限、或创建目录失败，则 msaicerr 工具退出并报错。

### -dev 或 --device_id

可选参数，指定运行内置算子样例的 Device ID，不设置该参数时默认 Device ID 为 0。

在分析 AI Core Error 问题时，msaicerr 工具会运行一个内置算子样例，用于检查软硬件环境是否正常。

## 使用示例和输出说明

```bash
python3 msaicerr.py -p $HOME/aic_err_info -out $HOME/result
```

执行命令后，用户根据终端界面提示的 info.txt 文件所在的路径，通过 info.txt 文件中的提示信息进行问题分析和定位。

若故障信息中存在多个 AI Core Error 问题，则 msaicerr 工具按日志时间解析**第一次出现**的 AI Core Error 问题。

在执行 msaicerr.py 工具后，在执行 msaicerr.py 工具的同级目录下，会生成 `debug_info.txt` 或 `info_{时间戳}/debug_info.txt` 文件，用于记录工具执行过程中的日志信息。

## 排查顺序建议

1. 确认算子不在[不支持清单](functions-and-restrictions.md)中。
2. 检查故障信息目录三要素（dump 文件、算子编译产物、cann 日志）是否齐全。
3. 在 `{install_path}/tools/msaicerr` 下执行分析，`-p`/`-out` 指向互不包含的独立路径。
4. 读终端提示的 `info.txt` 定位问题；工具本身报错或卡住则看 `debug_info.txt`。
5. 需要进一步看算子输入输出数据时，用 [dump-parsing.md](dump-parsing.md) 解析 dump 文件。
