# Ascend DMI 命令速查表

> 来源: MindCluster 7.3.0 开发文档 - 昇腾社区

---

## 信息查询类

| 命令 | 功能 | 示例 |
|-----|------|-----|
| `ascend-dmi -h` | 查看帮助信息 | `ascend-dmi -h` |
| `ascend-dmi -v` | 查看版本信息 | `ascend-dmi -v` |
| `ascend-dmi -i` | 设备实时状态查询（简要） | `ascend-dmi -i` |
| `ascend-dmi -i -b` | 设备实时状态查询（基本信息） | `ascend-dmi -i -b` |
| `ascend-dmi -i --dt` | 设备实时状态查询（详细信息） | `ascend-dmi -i --dt` |
| `ascend-dmi -c` | 软硬件版本兼容性测试 | `ascend-dmi -c` |

---

## 性能测试类

### 带宽测试 (--bw)

| 命令 | 功能 | 示例 |
|-----|------|-----|
| `ascend-dmi --bw` | 默认带宽测试（h2d/d2h/d2d 步长模式） | `ascend-dmi --bw -q` |
| `ascend-dmi --bw -t h2d` | Host to Device 带宽测试 | `ascend-dmi --bw -t h2d -d 0` |
| `ascend-dmi --bw -t d2h` | Device to Host 带宽测试 | `ascend-dmi --bw -t d2h -d 0` |
| `ascend-dmi --bw -t d2d` | Device to Device 带宽测试 | `ascend-dmi --bw -t d2d -d 0` |
| `ascend-dmi --bw -t p2p` | P2P 带宽测试（全量矩阵） | `ascend-dmi --bw -t p2p -q` |
| `ascend-dmi --bw -t p2p --ds X --dd Y` | P2P 指定源和目标设备 | `ascend-dmi --bw -t p2p --ds 0 --dd 1 -s 128M --et 100` |

### 算力测试 (-f)

| 命令 | 功能 | 示例 |
|-----|------|-----|
| `ascend-dmi -f` | 默认算力测试（FP16） | `ascend-dmi -f -q` |
| `ascend-dmi -f -t fp16` | FP16 算力测试 | `ascend-dmi -f -t fp16 -d 0 --et 60` |
| `ascend-dmi -f -t fp32` | FP32 算力测试（A2/A3系列） | `ascend-dmi -f -t fp32 -d 0 -q` |
| `ascend-dmi -f -t hf32` | HF32 算力测试（A2/A3系列） | `ascend-dmi -f -t hf32 -d 0 -q` |
| `ascend-dmi -f -t bf16` | BF16 算力测试（A2/A3系列） | `ascend-dmi -f -t bf16 -d 0 -q` |
| `ascend-dmi -f -t int8` | INT8 算力测试 | `ascend-dmi -f -t int8 -d 0 -q` |
| `ascend-dmi -f --all` | 整机算力测试 | `ascend-dmi -f --all -q` |

### 功耗测试 (-p)

| 命令 | 功能 | 示例 |
|-----|------|-----|
| `ascend-dmi -p` | 默认功耗测试（TDP模式） | `ascend-dmi -p -q` |
| `ascend-dmi -p -pt edp` | EDP 功耗压力测试 | `ascend-dmi -p -pt edp -q` |
| `ascend-dmi -p -pt tdp` | TDP 功耗压力测试 | `ascend-dmi -p -pt tdp --dur 300` |
| `ascend-dmi -p --dur N` | 指定测试时长（秒） | `ascend-dmi -p --dur 60 --it 5` |

### 眼图测试 (--sq)

| 命令 | 功能 | 示例 |
|-----|------|-----|
| `ascend-dmi --sq` | 眼图测试 | `ascend-dmi --sq -t pcie -d 0` |
| `ascend-dmi --sq -t pcie` | PCIe 眼图测试 | `ascend-dmi --sq -t pcie` |
| `ascend-dmi --sq -t hbm` | HBM 眼图测试 | `ascend-dmi --sq -t hbm` |
| `ascend-dmi --sq -t roce` | RoCE 眼图测试 | `ascend-dmi --sq -t roce` |
| `ascend-dmi --sq -t all` | 全部类型眼图测试 | `ascend-dmi --sq -t all` |

---

## 故障诊断类 (--dg)

### 一键式组合诊断

| 命令 | 功能 | 示例 |
|-----|------|-----|
| `ascend-dmi --dg --se healthCheck` | 健康检查场景 | `ascend-dmi --dg --se healthCheck -q` |
| `ascend-dmi --dg --se performanceCheck` | 性能规格测试场景 | `ascend-dmi --dg --se performanceCheck -q` |
| `ascend-dmi --dg --se stressTest` | 压力测试场景 | `ascend-dmi --dg --se stressTest -q` |
| `ascend-dmi --dg --se healthCheck,performanceCheck,stressTest` | 全部场景 | `ascend-dmi --dg --se healthCheck,performanceCheck,stressTest -q` |

### 指定诊断项诊断

| 命令 | 功能 | 示例 |
|-----|------|-----|
| `ascend-dmi --dg -i driver` | 驱动健康诊断 | `ascend-dmi --dg -i driver` |
| `ascend-dmi --dg -i cann` | CANN与驱动兼容性诊断 | `ascend-dmi --dg -i cann` |
| `ascend-dmi --dg -i device` | 芯片诊断 | `ascend-dmi --dg -i device` |
| `ascend-dmi --dg -i network` | 网络健康诊断 | `ascend-dmi --dg -i network` |
| `ascend-dmi --dg -i bandwidth` | 带宽诊断 | `ascend-dmi --dg -i bandwidth` |
| `ascend-dmi --dg -i aiflops` | 算力诊断 | `ascend-dmi --dg -i aiflops` |
| `ascend-dmi --dg -i hbm` | 片上内存诊断 | `ascend-dmi --dg -i hbm` |
| `ascend-dmi --dg -i signalQuality` | 眼图诊断 | `ascend-dmi --dg -i signalQuality` |

### 压力测试

| 命令 | 功能 | 示例 |
|-----|------|-----|
| `ascend-dmi --dg -s -i aicore` | AICORE 压测 | `ascend-dmi --dg -s -i aicore -d 0 -q` |
| `ascend-dmi --dg -s -i hbm` | 片上内存压测 | `ascend-dmi --dg -s -i hbm -d 0 -q` |
| `ascend-dmi --dg -s -i random` | 片上内存高危地址压测 | `ascend-dmi --dg -s -i random -d 0 -q` |
| `ascend-dmi --dg -s -i bandwidth` | P2P 压测 | `ascend-dmi --dg -s -i bandwidth -d 0 -q` |
| `ascend-dmi --dg -s -i edp` | EDP 功耗压测 | `ascend-dmi --dg -s -i edp -d 0 -q` |
| `ascend-dmi --dg -s -i tdp` | TDP 功耗压测 | `ascend-dmi --dg -s -i tdp -d 0 -q` |
| `ascend-dmi --dg -s -i aicpu` | AICPU 压测 | `ascend-dmi --dg -s -i aicpu -d 0 -q` |

---

## NPU 恢复类

| 命令 | 功能 | 示例 |
|-----|------|-----|
| `ascend-dmi -r` | NPU 环境恢复 | `ascend-dmi -r -d 0,1,2 -q` |

---

## 公共参数说明

| 参数 | 说明 | 适用场景 |
|-----|------|---------|
| `-d, --device` | 指定设备ID（可多个，逗号分隔） | 性能测试、诊断、恢复 |
| `-q, --quiet` | 跳过确认提示 | 性能测试、诊断、恢复 |
| `-fmt json` | JSON 格式输出 | 所有功能 |
| `-h, --help` | 查看帮助 | 所有功能 |

### 性能测试专用参数

| 参数 | 说明 | 默认值 | 范围 |
|-----|------|-------|------|
| `-t, --type` | 测试类型 | - | 见各测试说明 |
| `-s, --size` | 传输数据大小 | 步长模式 | 1B~512M（A3系列可到4G） |
| `--et, --execute-times` | 迭代次数 | 步长5/定长40 | 1~1000 |

### 诊断专用参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--se, --scene` | 诊断场景 | - |
| `-i, --items` | 诊断检查项 | 除aicore/prbs/edp/tdp/aicpu/nic外 |
| `-r, --result` | 结果保存路径 | /var/log/ascend_check 或 $HOME/var/log/ascend_check |
| `-s, --stress` | 压力测试标志 | - |
