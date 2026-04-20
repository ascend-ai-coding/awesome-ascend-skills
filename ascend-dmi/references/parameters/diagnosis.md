# 故障诊断参数说明

> 来源: MindCluster 7.3.0 开发文档 - 昇腾社区

---

## 主要参数

| 参数 | 说明 | 是否必填 | 默认值 |
|-----|------|---------|-------|
| `-dg, --dg, --diagnosis` | 启用故障诊断 | 是 | - |
| `--se, --scene` | 诊断场景 | 否（一键式时必填） | - |
| `-i, --items` | 诊断检查项 | 否 | 除 aicore/prbs/edp/tdp/aicpu/nic 外 |
| `-s, --stress` | 压力测试标志 | 否（压测时必填） | - |
| `-d, --device` | 指定设备 ID | 否 | 所有设备 |
| `-r, --result` | 结果保存路径 | 否 | 默认路径 |
| `-q, --quiet` | 跳过确认提示 | 否 | 提示确认 |
| `-fmt, --format` | 输出格式 | 否 | normal |

---

## 诊断场景 (--se) 说明

| 场景 | 说明 | 参考耗时 | 包含诊断项 |
|-----|------|---------|-----------|
| `healthCheck` | 健康检查 | ≤2min | driver/cann/device/network/signalQuality/hbm |
| `performanceCheck` | 性能规格 | 14min~3h | bandwidth/aiflops/nic |
| `stressTest` | 压力测试 | 1.5h~9.5h | aicore/hbm/bandwidth/edp/tdp/aicpu |

**组合使用**：逗号分隔，如 `healthCheck,performanceCheck,stressTest`

---

## 诊断检查项 (-i) 说明

### 健康检查项

| 检查项 | 说明 |
|-------|------|
| `driver` | 驱动健康诊断 |
| `cann` | CANN 与驱动兼容性诊断 |
| `device` | 芯片诊断 |
| `network` | 网络健康诊断 |
| `hbm` / `chipMemory` | 片上内存诊断 |
| `signalQuality` | 眼图诊断 |

### 性能规格检查项

| 检查项 | 说明 |
|-------|------|
| `bandwidth` | 带宽诊断 |
| `aiflops` | 算力诊断 |
| `nic` | NIC 诊断 |
| `prbs` | PRBS 码流诊断 |

### 压力测试检查项

| 检查项 | 说明 | 需配合 `-s` |
|-------|------|------------|
| `aicore` | AICORE 压测/诊断 | 是 |
| `hbm` | 片上内存压测 | 是 |
| `random` | 片上内存高危地址压测 | 是 |
| `bandwidth` | P2P 压测 | 是 |
| `edp` | EDP 功耗压测 | 是 |
| `tdp` | TDP 功耗压测 | 是 |
| `aicpu` | AICPU 压测 | 是 |
| `dsa` | DSA 压测 | 是 |

**注意**：aicpu 不支持与其他诊断项一起使用。

---

## 常用参数组合

### 一键式组合诊断

```bash
# 健康检查
ascend-dmi --dg --se healthCheck -q

# 性能规格测试
ascend-dmi --dg --se performanceCheck -q

# 压力测试
ascend-dmi --dg --se stressTest -q

# 全部场景
ascend-dmi --dg --se healthCheck,performanceCheck,stressTest -q
```

### 指定诊断项

```bash
# 驱动和芯片诊断
ascend-dmi --dg -i driver,device -q

# 片上内存诊断
ascend-dmi --dg -i hbm -d 0 -q
```

### 压力测试

```bash
# AICORE 压测
ascend-dmi --dg -s -i aicore -d 0 -q

# 片上内存压测
ascend-dmi --dg -s -i hbm -d 0 -q

# P2P 压测
ascend-dmi --dg -s -i bandwidth -d 0 -q
```

---

## 结果保存路径 (-r)

### 默认路径

- **root 用户**：`/var/log/ascend_check`
- **非 root 用户**：`$HOME/var/log/ascend_check`

### 自定义路径

```bash
ascend-dmi --dg --se healthCheck -r /test -q
```

- 在指定路径创建 `ascend_check` 文件夹
- 建议权限设置为 700

---

## 诊断结果状态

| 状态 | 含义 |
|-----|------|
| `PASS` | 通过 |
| `FAIL` | 失败 |
| `SKIP` | 不支持 |
| `HEALTH` | 健康 |
| `GENERAL_WARN` | 一般警告 |
| `IMPORTANT_WARN` | 重要警告 |
| `EMERGENCY_WARN` | 紧急警告 |
