# asys 命令速查

> **芯片型号识别的首选工具**（npu-smi 的 Chip Name 作为 short-soc-version 不可信，issue #587）；
> 同时作为 npu-smi 不可用时的设备查询回退工具。
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

## 硬件信息（芯片识别核心来源）

```bash
# 查看主机和设备硬件信息（CPU、NPU 数量、PCIe 等）
asys info -r hardware
```

关键输出字段：

| 字段 | 用途 |
|------|------|
| Chip Info | 芯片完整型号（full-soc-version，如 `Ascend 950PR_9579 V100`） |
| Arch Info | NpuArch（如 `2201`；部分 asys 版本无此字段，回退 ini） |
| NPU Count | 设备数 |

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
