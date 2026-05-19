# NPU 命令速查

## 常用 npu-smi 命令

### 设备查询

```bash
# 查看设备列表（简洁）
npu-smi list

# 查看所有设备详细信息
npu-smi info

# 查看指定设备信息
npu-smi info -i 0

# 查看设备健康状态
npu-smi health
```

### 资源监控

```bash
# 实时监控（每秒刷新）
npu-smi top

# 查看内存使用
npu-smi info -t memory
npu-smi info -t memory -i 0

# 查看设备温度
npu-smi info -t temp
npu-smi info -t temp -i 0

# 查看设备功耗
npu-smi info -t power
npu-smi info -t power -i 0
```

### 进程管理

```bash
# 查看占用 NPU 的进程
npu-smi info -t usages

# 查看指定设备的进程
npu-smi info -t usages -i 0

# 强制释放进程
npu-smi release -i 0 -p <pid>
```

### 设备管理

```bash
# 锁定/解锁设备
npu-smi lock -i 0
npu-smi unlock -i 0

# 查看设备性能模式
npu-smi perf

# 设置性能模式
npu-smi perf -i 0 -m <mode>
```

## 输出格式

### 表格输出（默认）
```
+------+---------------+--------+--------+--------+
| ID   | Name          | Health | Power  | Temp   |
+------+---------------+--------+--------+--------+
| 0    | Ascend910     | OK     | 125W   | 58C    |
+------+---------------+--------+--------+--------+
```

## 设备 ID 映射

- 设备 ID 从 0 开始（`npu-smi list` 显示）
- 芯片 ID（chip_id）可能与设备 ID 不同
- 多卡环境注意配置 `ASCEND_DEVICE_ID` 环境变量