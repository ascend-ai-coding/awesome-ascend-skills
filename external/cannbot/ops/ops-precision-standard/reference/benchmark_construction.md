# 标杆构造方法

## 1. 标杆选择原则

**采用单标杆比对：与更高精度的实现（CPU 或昇腾小算子拼接）直接比较。**

标杆应为比被测算子更高精度的参考实现，逐级降级备选。

## 2. 标杆类型与使用场景

| 优先级 | 标杆类型 | 适用场景 | 实现方法 |
|-------|---------|---------|---------|
| **1** | 高精度 CPU 实现 | 标准算子 | 直接对标 |
| **2** | 昇腾小算子拼接组合实现 | 融合算子、量化算子 | 组合实现 |
| **3** | 自行构造的 CPU 实现 | 非标准数据类型 | 自行开发 |

## 3. 标杆实现方法

### 3.1 第一优先级：高精度 CPU 实现

**适用场景：** 标准算子，业界已有成熟实现

**实现示例：**
```python
# PyTorch CPU（高精度）作为 Golden
import torch
golden = torch.matmul(input_a, input_b).cpu().numpy()

# NPU 实现
npu_output = aclnn_matmul(input_a_npu, input_b_npu)
```

### 3.2 第二优先级：昇腾小算子拼接组合

**适用场景：**
- 融合算子（如 FlashAttention）
- 量化算子（多算子串联）
- 特殊融合结构

**实现示例：**
```python
# FlashAttention 用小算子组合实现
def flash_attention_reference(Q, K, V):
    scores = np.matmul(Q, K.transpose())
    attention = softmax(scores / np.sqrt(d_k))
    output = np.matmul(attention, V)
    return output
```

### 3.3 第三优先级：自行构造 CPU 实现

**适用场景：** 非标准数据类型（如 HiFLOAT8）

**注意事项：**
- 确保 CPU 实现正确性
- 使用高精度数值计算
- 提供充分测试用例
- 文档化实现细节

## 4. 单标杆比对

**定义：** 与单一精度标杆直接比较，标杆应为更高精度实现。

**实现要点：**
```python
# Golden: 高精度 CPU 实现或昇腾小算子拼接
golden = cpu_implementation_high_precision(input)

# NPU 实现
npu_output = npu_implementation(input)

# 直接比对误差指标与阈值
mere = mean_relative_error(npu_output, golden)
mare = max_relative_error(npu_output, golden)
```
