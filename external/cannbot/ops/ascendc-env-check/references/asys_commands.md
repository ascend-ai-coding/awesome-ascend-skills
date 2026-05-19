# asys 命令速查

> 当 `npu-smi` 不可用或未检测到设备时，使用 `asys` 作为回退工具。
> 路径：`$ASCEND_HOME_PATH/tools/ascend_system_advisor/asys/asys`

## 设备健康检查

```bash
# 检查所有设备健康状态
asys health

# 检查指定设备（-d 后接 Device ID）
asys health -d 0
```

> `-d` 参数指定的是 **Device ID**（从 0 开始），不是 Card ID。

**设备可用性判断**：

| 状态 | 可用性 |
|------|--------|
| Healthy | 可用 |
| Warning | 可用 |
| 其他 | 不可用 |

## 设备状态查询

```bash
# 查看设备状态（芯片型号、温度、功耗、HBM、AI Core 等）
asys info -r status

# 查看指定设备（-d 后接 Device ID）
asys info -r status -d 0
```

## 硬件信息

```bash
# 查看主机和设备硬件信息（CPU、NPU 数量、PCIe 等）
asys info -r hardware
```

## 软件信息

```bash
# 查看主机和设备软件版本（驱动、固件、runtime 等）
asys info -r software
```

## 诊断

```bash
# 硬件诊断（支持 910B/910_93/950）
asys diagnose
```
