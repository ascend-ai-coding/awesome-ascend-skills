# 转换 Dump 文件数据类型

## 功能说明

转换 `*.bin` 格式的 Dump 文件中的数据类型，输出 `.npy` 文件。

`*.bin` 格式的 Dump 文件可从[解析 Dump 文件](dump-parsing.md)中获取。

## 命令格式

```bash
python3 msaicerr.py -d path1 -out path2 -dtype int8
```

## 参数说明

### -d 或 --data

**必选参数**，指定 `*.bin` 格式的 Dump 文件路径，**包含文件名**。

### -out 或 --output_path

可选参数，指定 `.npy` 结果文件的存放路径。如果不指定，则结果文件默认跟 Dump 文件存放在同一路径下。

### -dtype 或 --dest_dtype

**必选参数**，指定待转换 Dump 文件中的数据类型。

如果指定的数据类型与源 Dump 文件中的数据类型不一致，则执行解析命令时会给出 Warning 提示，并**按用户指定的数据类型解析**。

取值范围：

```
float32, float16, float64, int8, int16, int32, int64,
uint8, uint16, uint32, uint64, bool, bfloat16
```

## 使用示例和输出说明

```bash
python3 msaicerr.py -d /demo/extra-info/data-dump/0/exception_info.2.1.20250611171538370.input.0.bin -dtype int8
```

输出示例：

```bash
[INFO] The dump file directory will be used to as the output directory of the parsed results.
[INFO] Success convert bin to npy: /demo/extra-info/data-dump/0/exception_info.2.1.20250611171538370.input.0.bin -> /demo/extra-info/data-dump/0/exception_info.2.1.20250611171538370.input.0.int8.npy
```

根据提示获取转换结果文件。

在执行 msaicerr.py 工具后，在执行 msaicerr.py 工具的同级目录下，会生成 `debug_info.txt` 文件，用于记录工具执行过程中的日志信息。

## 使用要点

**dtype 要和算子原型对齐**。工具不会拒绝不匹配的 dtype，只给 Warning 并按指定类型强行解析——此时数值是错的但文件能生成。确认算子该输入/输出的真实数据类型后再转，避免拿错误数值做精度判断。

**输出文件名带 dtype 标识**，如 `.input.0.int8.npy`，同一 bin 用不同 dtype 转多次不会互相覆盖，可用于反推真实类型。

**bfloat16 需要额外库**。转换 bfloat16 时若报不能识别，安装 `bfloat16ext` 库后重试。
