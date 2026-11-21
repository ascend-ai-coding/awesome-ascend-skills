# 检查环境

## 功能说明

运行内置算子样例检查软硬件环境。

## 命令格式

```bash
python3 msaicerr.py -e -dev 0
```

## 参数说明

### -e 或 --env

**必选参数**，表示检查环境。

### -dev 或 --device_id

可选参数，指定运行内置算子样例的 Device ID，不设置该参数默认 Device ID 为 0。msaicerr 工具会运行一个内置算子样例，用于检查软硬件环境是否正常。

## 使用示例和输出说明

```bash
python3 msaicerr.py -e
```

输出示例：

```bash
[INFO] Total device count: 1
[INFO] Valid device_id 0
[INFO] Get soc_version: xxxxxxx
[INFO] Start to test env with golden op.
[INFO] The build-in sample operator runs successfully, The environment is normal.
```

在执行 msaicerr.py 工具后，在执行 msaicerr.py 工具的同级目录下，会生成 `debug_info.txt` 文件，用于记录工具执行过程中的日志信息。

## 使用时机

**AI Core Error 定位的第一步**。先用 `-e` 确认环境本身正常，再分析故障信息——如果内置算子样例都跑不过，说明问题在环境（驱动、固件、CANN 版本匹配等）而不在业务算子，继续分析业务日志意义不大。

**`-p` 分析已内含该检查**。执行 `-p` 分析 AI Core Error 时工具会自动跑一次内置算子样例，无需单独再执行 `-e`。单独使用 `-e` 的场景是：还没采集到故障信息，或想先验证某张卡是否可用。

**逐卡验证**。怀疑特定卡异常时，用 `-dev` 指定该卡的 Device ID 分别验证，对比输出可以区分是单卡问题还是整机环境问题。
