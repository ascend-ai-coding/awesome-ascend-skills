# NPU 命令速查

## 推荐原则

> **优先使用结构化子命令（`-t` + key:value 格式），避免直接解析 `npu-smi info` 主表格输出。**
>
> 原因：`npu-smi info` 主表格的格式随驱动版本变化（列宽、换行、字段名等），脚本解析极易失效。`-t` 子命令返回稳定的 key:value 格式，更适合自动化处理。

## 查询可用命令

npu-smi 支持的命令随驱动版本变化，**请以 `--help` 输出为准**。

```bash
# 查看 npu-smi 顶层命令
npu-smi --help

# 查看 info 子命令和可用 -t type
npu-smi info --help
```

## 稳定命令

以下命令在不同版本中相对稳定：

### 查看设备映射

```bash
# 获取 NPU ID 与芯片的映射关系（固定格式，脚本友好）
npu-smi info -m
```

输出示例：

```
NPU ID    Chip ID    Chip Logic ID    Chip Name
  5         0           0               Ascend 910B3
```

### 结构化查询（key:value 格式）

```bash
# 第一步：通过 info -m 确认设备 ID
npu-smi info -m

# 第二步：将 <device_id> 替换为实际 ID 进行查询
npu-smi info -t <type> -i <device_id>
```

常用 type（请以 `npu-smi info --help` 输出为准）：

| type | 说明 |
|------|------|
| `health` | 健康状态 |
| `temp` | 温度 |
| `power` | 功耗 |
| `memory` | 内存容量/时钟 |
| `usages` | 使用率（AICore、HBM 等） |
| `common` | 批量获取（温度、功耗、AICore、HBM 等） |

> ⚠️ 部分 type 在特定设备上可能不支持（返回 "This device does not support querying ..."）。

### 输出格式

`-t` 子命令返回 key:value 格式：

```
NPU ID                         : 5
Chip Count                     : 1
Health                         : OK
Chip ID                        : 0
```

### 主表格输出（人类阅读用）

```bash
npu-smi info
```

返回表格格式：

```
+------+---------------+--------+--------+--------+
| ID   | Name          | Health | Power  | Temp   |
+------+---------------+--------+--------+--------+
| 5    | Ascend910B3   | OK     | 125W   | 39C    |
+------+---------------+--------+--------+--------+
```

> ⚠️ **警告**：此表格格式在不同驱动版本下可能变化，**禁止在脚本中通过 awk/regex 解析此格式**。

## 脚本自动化

在脚本中自动获取 NPU 信息时，推荐编写 **Python 脚本** 调用结构化子命令，避免硬编码命令列表：

```python
# 获取完整设备信息（JSON 格式）
python3 scripts/_npu_info.py --json

# 列出所有设备 ID
python3 scripts/_npu_info.py --list

# 查看设备健康状态
python3 scripts/_npu_info.py --health
```

脚本内部通过 `npu-smi info --help` **动态发现** 当前环境支持的子命令，优先使用 `npu-smi info -t common` 批量获取数据，比逐个调用 `-t` 子命令更高效。

## 设备 ID 映射

- 设备 ID 从 `npu-smi info -m` 第一列获取，**不一定是 0**
- 芯片 ID（chip_id）可能与设备 ID 不同
- 多卡环境注意配置 `ASCEND_DEVICE_ID` 环境变量
