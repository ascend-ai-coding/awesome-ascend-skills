# 性能基线建立指南

本文档提供详细的性能基线建立方法，用于对比 SuperKernel 优化前后的性能差异。

---

## 为什么需要性能基线

性能基线是评估优化效果的参照标准：
- **量化提升**：精确计算性能提升比例
- **问题定位**：性能下降时快速定位问题
- **方案对比**：对比不同 Scope 策略的效果
- **回归测试**：确保后续修改不影响性能

---

## 性能指标

### 核心指标

| 指标 | 说明 | 单位 | 重要性 |
|------|------|------|--------|
| **Decode 单步耗时** | 生成一个 token 的时间 | ms | 🔴 高 |
| **吞吐量** | 每秒生成的 token 数 | tokens/s | 🔴 高 |
| **首 token 延迟** | Prefill 阶段耗时 | ms | 🟡 中 |
| **内存占用** | 推理时的显存使用 | GB | 🟡 中 |
| **编译时间** | torch.compile 耗时 | s | 🟢 低 |

### 辅助指标

| 指标 | 说明 | 用途 |
|------|------|------|
| **CPU 使用率** | CPU 占用百分比 | 识别 CPU 瓶颈 |
| **NPU 利用率** | NPU 计算资源使用率 | 识别 NPU 瓶颈 |
| **通信开销** | 多卡场景的通信时间 | 优化并行策略 |

---

## 基线建立流程

### 步骤 1：准备测试环境

#### 1.1 确认硬件配置

```bash
# 查看 NPU 信息
npu-smi info

# 确认硬件型号
# 应该是 Atlas A3 系列
```

#### 1.2 确认软件版本

```bash
# CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# PyTorch 版本
python -c "import torch; print(torch.__version__)"

# torch_npu 版本
python -c "import torch_npu; print(torch_npu.__version__)"
```

#### 1.3 设置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PYTHONPATH=/path/to/cann-recipes-infer:$PYTHONPATH
```

---

### 步骤 2：配置基线测试

#### 2.1 禁用 SuperKernel

修改 `config.yaml`：

```yaml
exe_mode: "ge_graph"              # 保持图模式
model_config:
  enable_superkernel: False       # 禁用 SuperKernel
  enable_multi_streams: False     # 禁用其他优化
  enable_cache_compile: False
```

#### 2.2 配置测试参数

```yaml
# 推理配置
batch_size: 1                     # 固定 batch size
max_new_tokens: 100               # 固定生成长度
temperature: 0.0                  # 固定采样参数（确保可复现）
top_p: 1.0
top_k: 1

# 数据集配置
dataset: "custom"                 # 使用固定的测试数据
prompt: "请介绍一下人工智能的发展历史。"  # 固定 prompt
```

#### 2.3 准备测试脚本

创建 `scripts/benchmark_baseline.sh`：

```bash
#!/bin/bash

# 设置环境
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 进入模型目录
cd cann-recipes-infer/models/${MODEL_NAME}

# 运行基线测试
echo "=== Running Baseline Test ==="
python -u cann-recipes-infer/models/runner_*.py \
    --config config.yaml \
    --output baseline_output.json \
    2>&1 | tee baseline.log

# 提取性能数据
python ../../scripts/extract_performance.py baseline.log > baseline_performance.json

echo "=== Baseline Test Complete ==="
cat baseline_performance.json
```

---

### 步骤 3：运行基线测试

#### 3.1 预热运行

```bash
# 第一次运行（预热，不计入统计）
bash scripts/benchmark_baseline.sh

# 等待编译完成
```

#### 3.2 正式测试

```bash
# 运行 3-5 次，取平均值
for i in {1..5}; do
    echo "=== Run $i ==="
    bash scripts/benchmark_baseline.sh
    sleep 5
done

# 计算平均值
python scripts/calculate_average.py baseline_performance_*.json > baseline_avg.json
```

---

### 步骤 4：记录基线数据

#### 4.1 性能数据格式

创建 `baseline_performance.json`：

```json
{
  "test_info": {
    "model_name": "deepseek-r1",
    "hardware": "Atlas A3",
    "cann_version": "8.3.RC1",
    "pytorch_version": "2.6.0",
    "torch_npu_version": "2.6.0",
    "test_date": "2026-03-30",
    "config": {
      "exe_mode": "ge_graph",
      "enable_superkernel": false,
      "batch_size": 1,
      "max_new_tokens": 100
    }
  },
  "performance": {
    "decode_latency_ms": 15.2,
    "throughput_tokens_per_sec": 65.8,
    "prefill_latency_ms": 120.5,
    "memory_usage_gb": 24.3,
    "compile_time_sec": 45.2
  },
  "statistics": {
    "runs": 5,
    "decode_latency_std": 0.3,
    "throughput_std": 1.2
  }
}
```

#### 4.2 日志文件

保存完整的运行日志：
- `baseline.log` - 完整输出
- `baseline_performance.json` - 性能数据
- `baseline_output.json` - 模型输出

---

### 步骤 5：验证基线稳定性

#### 5.1 检查波动

```python
import json
import numpy as np

# 读取多次运行的数据
data = []
for i in range(1, 6):
    with open(f'baseline_performance_{i}.json') as f:
        data.append(json.load(f)['performance']['decode_latency_ms'])

# 计算统计量
mean = np.mean(data)
std = np.std(data)
cv = std / mean * 100  # 变异系数

print(f"Mean: {mean:.2f} ms")
print(f"Std: {std:.2f} ms")
print(f"CV: {cv:.2f}%")

# 判断稳定性
if cv < 5:
    print("✓ 基线稳定")
else:
    print("✗ 基线波动较大，建议重新测试")
```

#### 5.2 稳定性标准

| 变异系数 (CV) | 稳定性 | 建议 |
|--------------|--------|------|
| < 5% | 优秀 | 可以使用 |
| 5% - 10% | 良好 | 可以使用，注意波动 |
| 10% - 20% | 一般 | 建议增加测试次数 |
| > 20% | 较差 | 需要排查环境问题 |

---

## 优化后性能测试

### 步骤 1：启用 SuperKernel

修改 `config.yaml`：

```yaml
exe_mode: "ge_graph"
model_config:
  enable_superkernel: True        # 启用 SuperKernel
  # 其他配置保持与基线一致
```

### 步骤 2：运行优化测试

```bash
# 使用相同的测试脚本
bash scripts/benchmark_optimized.sh

# 运行多次取平均
for i in {1..5}; do
    echo "=== Run $i ==="
    bash scripts/benchmark_optimized.sh
    sleep 5
done

python scripts/calculate_average.py optimized_performance_*.json > optimized_avg.json
```

### 步骤 3：性能对比

创建 `scripts/compare_performance.py`：

```python
import json

# 读取基线和优化后的数据
with open('baseline_avg.json') as f:
    baseline = json.load(f)

with open('optimized_avg.json') as f:
    optimized = json.load(f)

# 计算性能提升
def calculate_improvement(baseline_val, optimized_val, lower_is_better=True):
    if lower_is_better:
        improvement = (baseline_val - optimized_val) / baseline_val * 100
    else:
        improvement = (optimized_val - baseline_val) / baseline_val * 100
    return improvement

# 对比各项指标
metrics = {
    'decode_latency_ms': True,   # lower is better
    'throughput_tokens_per_sec': False,  # higher is better
    'memory_usage_gb': True,
}

print("=== Performance Comparison ===\n")
for metric, lower_is_better in metrics.items():
    baseline_val = baseline['performance'][metric]
    optimized_val = optimized['performance'][metric]
    improvement = calculate_improvement(baseline_val, optimized_val, lower_is_better)

    print(f"{metric}:")
    print(f"  Baseline:  {baseline_val:.2f}")
    print(f"  Optimized: {optimized_val:.2f}")
    print(f"  Improvement: {improvement:+.2f}%")
    print()

# 生成对比报告
report = {
    'baseline': baseline,
    'optimized': optimized,
    'improvements': {}
}

for metric, lower_is_better in metrics.items():
    baseline_val = baseline['performance'][metric]
    optimized_val = optimized['performance'][metric]
    improvement = calculate_improvement(baseline_val, optimized_val, lower_is_better)
    report['improvements'][metric] = {
        'baseline': baseline_val,
        'optimized': optimized_val,
        'improvement_percent': improvement
    }

with open('performance_comparison.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Report saved to performance_comparison.json")
```

---

## 性能分析工具

### 工具 1：NPU Profiler

```python
import torch_npu

# 启用 profiler
torch_npu.npu.profile(use_npu=True)

with torch_npu.npu.profile():
    output = model(input_ids)

# 查看性能报告
# 结果保存在 prof/ 目录
```

### 工具 2：CANN Profiler

```bash
# 在 config.yaml 中启用
enable_profiler: True

# 运行模型
bash infer.sh

# 查看 profiling 结果
ls prof/
```

### 工具 3：自定义计时

```python
import time
import torch_npu

def benchmark_decode(model, input_ids, num_steps=100):
    # 预热
    for _ in range(10):
        _ = model.decode(input_ids)

    # 正式测试
    torch_npu.npu.synchronize()
    start = time.time()

    for _ in range(num_steps):
        _ = model.decode(input_ids)

    torch_npu.npu.synchronize()
    end = time.time()

    avg_latency = (end - start) / num_steps * 1000  # ms
    throughput = num_steps / (end - start)  # steps/s

    return {
        'avg_latency_ms': avg_latency,
        'throughput_steps_per_sec': throughput
    }
```

---

## 性能报告模板

### 完整报告结构

```markdown
# SuperKernel 性能测试报告

## 测试环境

- 模型：DeepSeek-R1
- 硬件：Atlas A3 (8 卡)
- CANN 版本：8.3.RC1
- PyTorch 版本：2.6.0
- 测试日期：2026-03-30

## 测试配置

- Batch Size: 1
- Max New Tokens: 100
- Scope 策略：仅 Attention 模块

## 性能对比

| 指标 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| Decode 单步耗时 | 15.2 ms | 12.8 ms | +15.8% |
| 吞吐量 | 65.8 tokens/s | 78.1 tokens/s | +18.7% |
| 内存占用 | 24.3 GB | 24.5 GB | -0.8% |

## 详细数据

### 基线性能
- Decode 单步耗时：15.2 ± 0.3 ms
- 吞吐量：65.8 ± 1.2 tokens/s
- Prefill 延迟：120.5 ms
- 内存占用：24.3 GB

### 优化后性能
- Decode 单步耗时：12.8 ± 0.2 ms
- 吞吐量：78.1 ± 1.0 tokens/s
- Prefill 延迟：120.3 ms
- 内存占用：24.5 GB

## 结论

SuperKernel 优化在 Decode 阶段带来了 **15.8%** 的性能提升，吞吐量提升 **18.7%**。内存占用基本持平。优化效果符合预期。

## 下一步

- 尝试扩大 Scope 范围到全 Decoder 层
- 结合多流并行进一步优化
- 在更长序列上测试性能
```

---

## 常见问题

### Q1: 基线性能波动大怎么办？

**可能原因**：
1. 系统负载不稳定
2. NPU 频率波动
3. 测试数据不一致

**解决方法**：
1. 关闭其他进程，确保系统空闲
2. 固定 NPU 频率
3. 使用固定的测试数据和参数
4. 增加测试次数，取中位数而非平均值

### Q2: 如何确保测试的公平性？

**关键点**：
1. 使用相同的测试数据
2. 使用相同的配置参数（除了 enable_superkernel）
3. 在相同的环境下测试
4. 预热后再测试
5. 多次运行取平均值

### Q3: 性能提升不明显怎么办？

**排查步骤**：
1. 确认 SuperKernel 是否真正启用（查看日志）
2. 检查 Scope 范围是否合适
3. 使用 profiler 分析瓶颈
4. 尝试不同的 Scope 策略
5. 结合其他优化技术

---

## 最佳实践

1. **固定测试条件**：确保基线和优化测试使用完全相同的条件
2. **多次测试**：至少运行 3-5 次，取平均值
3. **预热运行**：第一次运行不计入统计
4. **记录详细信息**：保存完整的配置、日志和性能数据
5. **版本控制**：使用 git 记录每次测试的代码版本
6. **自动化**：编写脚本自动化测试流程
7. **可视化**：使用图表展示性能对比

---

## 参考资源

- Scope 分析指南：`scope-analysis-guide.md`
- 调试指南：`debugging-guide.md`
- 性能分析工具：`../../scripts/performance_analysis.py`
- 官方文档：[CANN 性能分析工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/devtools/auxiliarydevtool/atlasprofiling_16_0001.html)
