# 精度测试前置数值验证指南

## 概述

**本指南规定的是精度测试脚本编写前必须执行的前置验证步骤**。在编写完整精度测试脚本之前，**MUST** 先用小规模数据对比 NPU 输出和 CPU 参考实现的逐元素差异，确保两者语义一致。

## 前置验证流程

### Step 1：构造小规模输入

```python
import os
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '<vendors_path>'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

import torch
import torch_npu
torch.ops.load_library("<path_to>/libcustom_ops.so")

# 构造小规模输入（B=1~2, V=16~32）
B, V = 2, 16
logits = torch.randn(B, V, dtype=torch.float16, device='npu:0')
```

### Step 2：调用 NPU 算子，打印全部输出

```python
# 调用 NPU 算子
result = torch.ops.npu.<op_name>(logits, ...)

# 打印全部输出的逐元素值
for i, out in enumerate(result):
    print(f"Output {i}:")
    print(f"  shape: {out.shape}, dtype: {out.dtype}")
    print(f"  values: {out.cpu().float()}")
    print(f"  min: {out.cpu().float().min().item()}, max: {out.cpu().float().max().item()}")
    print(f"  sum: {out.cpu().float().sum().item()}")
    print(f"  has -inf: {torch.isinf(out.cpu().float()).any().item()}")
    print(f"  has nan: {torch.isnan(out.cpu().float()).any().item()}")
    print()
```

### Step 3：手写最简 CPU 参考实现，打印逐元素值

```python
def cpu_reference(logits_cpu, ...):
    # 最简实现，先不优化，确保正确性
    ...
    return result

cpu_result = cpu_reference(logits.cpu().float(), ...)
for i, out in enumerate(cpu_result):
    print(f"CPU Output {i}:")
    print(f"  shape: {out.shape}, dtype: {out.dtype}")
    print(f"  values: {out}")
    print(f"  min: {out.min().item()}, max: {out.max().item()}")
    print(f"  sum: {out.sum().item()}")
    print()
```

### Step 4：逐元素对比，确认语义一致

**MUST 逐项检查以下内容**：

| 检查项 | 检查方法 | 通过标准 |
|--------|---------|---------|
| 输出 shape 一致 | `npu.shape == cpu.shape` | 完全一致 |
| 输出 dtype 一致 | `npu.dtype == cpu.dtype` | 完全一致 |
| 有效位置数值量级一致 | 逐元素对比有效位置 | 差异 < 1e-3 |
| 无效位置填充值语义一致 | 检查 NPU 和 CPU 的无效位置值 | 语义一致（都是 0.0 或都是 -inf 等） |
| 是否归一化 | 检查有效位置值之和 | NPU 和 CPU 行为一致 |
| 是否升精度 | 检查中间计算精度 | NPU 和 CPU 行为一致 |
| 索引输出 | 检查索引值范围和含义 | 从 0 开始 / 从 1 开始一致 |

### Step 5：如发现不一致，调整 CPU 参考实现

**常见语义差异及处理方式**：

| 差异类型 | 典型表现 | 处理方式 |
|---------|---------|---------|
| 无效位置填充值不同 | NPU 用 `0.0`，CPU 用 `-inf` | CPU 参考实现改为与 NPU 一致的填充值 |
| 归一化行为不同 | NPU 有效位置值之和为 1.0，CPU 不为 1.0 | CPU 参考实现添加归一化步骤 |
| 升精度行为不同 | NPU 内部升到 FP32 计算，CPU 直接 FP16 | CPU 参考实现也升到 FP32 计算 |
| 索引起始值不同 | NPU 索引从 0 开始，CPU 从 1 开始 | CPU 参考实现改为从 0 开始 |
| 输出顺序不同 | NPU 降序排列，CPU 升序排列 | CPU 参考实现改为降序 |
| 条件分支不同 | NPU 对 topP≤0 保留 1 个 token，CPU 保留 0 个 | CPU 参考实现对齐 NPU 行为 |

## 前置验证检查清单

- [ ] 小规模输入已构造（B=1~2, V=16~32）
- [ ] NPU 输出已打印（逐元素值 + shape + dtype + min/max/sum）
- [ ] CPU 参考输出已打印（逐元素值 + shape + dtype + min/max/sum）
- [ ] 输出 shape 一致
- [ ] 输出 dtype 一致
- [ ] 有效位置数值差异 < 1e-3
- [ ] 无效位置填充值语义一致
- [ ] 归一化行为一致
- [ ] 升精度行为一致
- [ ] 索引输出语义一致
- [ ] 如有不一致，CPU 参考实现已调整并对齐

## 反模式

- ❌ 不做小规模数值对比直接写完整精度测试脚本
- ❌ 凭直觉假设 NPU 输出语义
- ❌ 忽略算子文档中的数学公式
- ❌ 不打印 NPU 输出的逐元素值就假设输出格式
- ❌ 只比较 shape 和 dtype，不比较数值
- ❌ 发现不一致后不调整 CPU 参考实现，而是调整精度阈值

## 精度指标计算的有效点过滤

对于输出中包含无效位置（填充值为 0.0 或 -inf）的算子，**MUST** 在计算 MERE/MARE 前过滤无效点：

```python
def compute_metrics_valid_only(npu_out, cpu_ref, invalid_value=0.0):
    npu_f = npu_out.cpu().float()
    ref_f = cpu_ref.float()

    # 过滤无效点：两者都接近 0 的位置视为无效
    valid = (ref_f.abs() > 1e-10) & (npu_f.abs() > 1e-10)

    if valid.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0, 1.0, 0

    npu_valid = npu_f[valid]
    ref_valid = ref_f[valid]
    num_valid = valid.sum().item()

    abs_err = (npu_valid - ref_valid).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rel_err = abs_err / (ref_valid.abs() + 1e-7)
    mare = rel_err.max().item()
    mere = rel_err.mean().item()

    cos = 1.0
    if num_valid > 1:
        cos = torch.nn.functional.cosine_similarity(
            npu_valid.unsqueeze(0), ref_valid.unsqueeze(0)
        ).item()

    return max_abs, mean_abs, mare, mere, cos, num_valid
```

**关键**：无效点的判定标准取决于算子语义，**MUST** 在前置验证中确认无效点的填充值，然后在精度测试中使用一致的过滤逻辑。
