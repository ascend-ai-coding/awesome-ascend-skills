# 解析 Dump 文件

## 功能说明

将 Dump 文件解析成 .bin 或 .npy 文件，文件中记录算子的输入、输出、workspace 等信息。

## 命令格式

```bash
python3 msaicerr.py -d path1 -out path2
```

## 参数说明

### -d 或 --data

**必选参数**，解析 Dump 文件时用于指定 Dump 文件路径，**包含文件名**。

### -out 或 --output_path

可选参数，指定解析结果文件的存放路径。如果不指定，则解析结果默认跟 Dump 文件存放在同一路径下。

## 使用示例和输出说明

```bash
python3 msaicerr.py -d /demo/extra-info/data-dump/0/exception_info.2.1.20250611171538370
```

输出示例：

```bash
[INFO] The dump file directory will be used to as the output directory of the parsed results.
[INFO] Parse dump file finished, result path is: /demo/dfx/data-dump/0
```

根据提示获取解析结果文件。

在执行 msaicerr.py 工具后，在执行 msaicerr.py 工具的同级目录下，会生成 `debug_info.txt` 文件，用于记录工具执行过程中的日志信息。

## 数据类型不识别的处理

若 `debug_info.txt` 中提示 `Can not read with dtype xxx`，则表示存在工具不能识别的数据类型，需由用户自行安装第三方库文件。

例如提示 `Can not read with dtype bfloat16`，则需安装 `bfloat16ext` 库。

## 后续步骤

解析得到的 `*.bin` 文件可用 [dump-dtype-conversion.md](dump-dtype-conversion.md) 按指定数据类型转换为 `.npy`，便于用 numpy 直接查看数值。
