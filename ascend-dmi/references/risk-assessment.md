# Ascend DMI 风险评估矩阵

## 功能风险总览

### 无风险（直接执行）

| 功能 | 命令 |
|-----|------|
| 帮助信息 | `ascend-dmi -h` |
| 版本信息 | `ascend-dmi -v` |
| 设备状态查询 | `ascend-dmi -i` |
| 兼容性检查 | `ascend-dmi -c` |
| 眼图测试 | `ascend-dmi --sq` |

### 中等风险（告知影响 → 用户确认 → 执行）

| 功能 | 命令 | 影响 |
|-----|------|------|
| 带宽测试 | `ascend-dmi --bw` | 独占设备，中断训练/推理 |
| P2P 带宽测试 | `ascend-dmi --bw -t p2p` | 独占设备，中断训练/推理 |
| 算力测试 | `ascend-dmi -f` | 满负载运行，影响其他业务 |
| 功耗测试 | `ascend-dmi -p` | 满负载运行，影响其他业务 |
| 码流测试 | PRBS 码流诊断 | 独占 RoCE 网口 |

**确认模板**：
```
该操作会影响 NPU 训练或推理业务：
- 测试将独占 NPU 设备
- 正在运行的训练/推理任务将被中断
- 建议在业务低峰期执行

是否确认执行？
```

### 高风险（二次确认 + 检查设备占用）

| 功能 | 命令 | 影响 |
|-----|------|------|
| NPU 恢复 | `ascend-dmi -r` | 复位 NPU，所有业务中断，可能掉卡 |
| 压力测试 | `ascend-dmi --dg -s` | 长时间高负载，可能硬件过热 |
| AICORE 压测 | `ascend-dmi --dg -s -i aicore` | 极端负载，可能触发保护机制 |
| 内存高危地址压测 | `ascend-dmi --dg -s -i random` | 测试内存故障地址 |

**确认模板**：
```
危险操作警告：
1. [具体风险说明]
2. 可能导致 NPU 暂时不可用
3. 可能需要重启恢复

请确认：已停止 NPU 相关业务，了解操作风险并确认执行。
```

---

## 诊断场景风险明细

### healthCheck

| 诊断项 | 影响业务 |
|-------|---------|
| driver / cann / device / network / signalQuality | 否 |
| hbm / chipMemory | **是**（占用内存） |

整体：会影响业务，需确认。

### performanceCheck

所有项（bandwidth / aiflops / nic / prbs）都影响业务。

### stressTest

所有项（aicore / hbm / random / bandwidth / edp / tdp / aicpu / dsa）都影响业务且风险高，需二次确认。

---

## 执行前检查

```bash
# 检查设备是否被占用
fuser -v /dev/davinci*

# 检查设备状态
npu-smi info

# 检查温度（>90°C 时不建议做算力/功耗测试）
npu-smi info -t health
```

---

## 诊断场景耗时参考

| 场景 | 耗时 |
|-----|------|
| healthCheck | ≤2 分钟 |
| performanceCheck | 14 分钟 ~ 3 小时 |
| stressTest | 1.5 ~ 9.5 小时 |
