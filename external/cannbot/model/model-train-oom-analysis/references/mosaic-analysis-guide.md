# Mosaic 内存分析工具使用指南

[Mosaic](https://github.com/facebookresearch/mosaic) 是 Meta 开源的 PyTorch 内存快照后处理分析工具。

## 安装

```bash
# Mosaic 未发布到 PyPI；建议锁定到已验证的 commit/tag（如 ...mosaic.git@<commit>），
# 避免拉到上游默认分支不可复现的最新代码。
pip install git+https://github.com/facebookresearch/mosaic.git
```

## CLI 命令速查

### 1. 峰值内存栈追踪 — `mosaic_get_memory_usage_peak`

分析峰值内存时刻的所有分配及其调用栈，**定位具体哪些操作导致了内存峰值**：

```bash
mosaic_get_memory_usage_peak --snapshot <snapshot.pickle>
```

输出内容：

- **Total Peak Dynamic Memory Usage**：运行时动态分配的峰值内存
- **Total Static Memory Usage**：快照开始前已存在的基线内存
- **Total Overall Peak Memory Usage**：Dynamic + Static
- 每个分配的完整调用栈及大小

**解读要点**：

- 观察 Top 几个分配的栈追踪，识别是来自 optimizer（`adam.py`）、backward（`autograd`）还是 forward 的激活
- 对比 baseline 和问题版本的调用次数差异（如 optimizer 追踪了更多 tensor）

### 2. 分类内存 Profile — `mosaic_get_memory_profile`

按类别（activation、gradient、optimizer、parameter）生成内存使用可视化 HTML：

```bash
# 标准分类 profile
mosaic_get_memory_profile --snapshot <snapshot.pickle> \
    --out-path profile.html \
    --profile categories

# 保持分配顺序（便于看内存随时间变化）
mosaic_get_memory_profile --snapshot <snapshot.pickle> \
    --out-path profile_ordered.html \
    --profile categories \
    --preserve-allocation-order

# 自定义正则匹配（追踪特定模式）
mosaic_get_memory_profile --snapshot <snapshot.pickle> \
    --out-path profile_custom.html \
    --profile custom \
    --custom-profile '{"hccl": "hccl", "moe_expert": "expert"}'
```

**分类说明**：

| 类别 | 含义 |
|------|------|
| Activation | 前向传播保存用于反向的中间张量 |
| Gradient | 反向传播计算的梯度 |
| Optimizer State | Adam/SGD 的动量和方差缓冲 |
| Parameter | 模型权重 |

### 3. 常用对比分析流程

```bash
# 1. 生成两个场景的 snapshot（如开启 AC 前后）
# 2. 分别生成分类 profile
mosaic_get_memory_profile --snapshot baseline.pickle \
    --out-path baseline_profile.html --profile categories
mosaic_get_memory_profile --snapshot with_ac.pickle \
    --out-path ac_profile.html --profile categories

# 3. 分别查看峰值
mosaic_get_memory_usage_peak --snapshot baseline.pickle
mosaic_get_memory_usage_peak --snapshot with_ac.pickle
```

## Python API

当需要编程集成或在脚本中自动化分析时，使用 Mosaic Python API：

```python
from mosaic.libmosaic.analyzer.memory_abstract import MemoryAbstract

# 加载并分析
memory_abstract = MemoryAbstract(memory_snapshot_file="snapshot.pickle")
memory_abstract.load_memory_snapshot()

# 分析峰值内存
memory_abstract.memory_snapshot.analyze_memory_snapshot(opt="memory_peak")

# 获取结果
dynamic_peak = memory_abstract.memory_snapshot.dynamic_memory_peak
static_memory = memory_abstract.memory_snapshot.static_memory
overall_peak = dynamic_peak + static_memory

print(f"动态峰值: {dynamic_peak / 1024**3:.3f} GiB")
print(f"静态内存: {static_memory / 1024**3:.3f} GiB")
print(f"总峰值:   {overall_peak / 1024**3:.3f} GiB")
```

## 适用场景

| 场景 | 推荐工具 |
|------|---------|
| 快速定位峰值内存来源 | `mosaic_get_memory_usage_peak` |
| 可视化各类别内存占比 | `mosaic_get_memory_profile --profile categories` |
| 追踪特定操作(如 HCCL)的内存 | `mosaic_get_memory_profile --profile custom` |
| 对比优化前后效果 | 两次 profile + 对比 HTML |
| CI/CD 内存回归检测 | Python API + 阈值断言 |

> [!NOTE]
> Mosaic 目前对 CUDA/GPU snapshot 支持最完善。对于 NPU snapshot，基本的 segments/blocks 分析可正常工作，但部分可视化功能可能需要适配。如遇到兼容问题，可使用本 skill 自带的 `analyze_snapshot.py` 脚本作为替代。
