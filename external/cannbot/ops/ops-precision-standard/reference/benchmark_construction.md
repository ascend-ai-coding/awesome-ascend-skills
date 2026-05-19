# 标杆构造方法

## 1. 标杆选择原则

**竞品对标优先，逐级降级备选**

## 2. 标杆类型与使用场景

| 优先级 | 标杆类型 | 适用场景 | 实现方法 |
|-------|---------|---------|---------|
| **1** | 业界CPU/三方芯片同等功能算子 | 标准算子 | 直接对标 |
| **2** | 小算子拼接组合实现 | 融合算子、量化算子 | 组合实现 |
| **3** | 自行构造的CPU实现 | 非标准数据类型 | 自行开发 |

## 3. 标杆实现方法

### 3.1 第一优先级：业界CPU/三方芯片

**适用场景：** 标准算子，业界已有成熟实现

**实现示例：**
```python
# PyTorch CPU作为Golden
import torch
golden = torch.matmul(input_a, input_b).cpu().numpy()

# NVIDIA GPU作为三方标杆
benchmark = torch.matmul(input_a.cuda(), input_b.cuda()).cpu().numpy()

# NPU实现
npu_output = aclnn_matmul(input_a_npu, input_b_npu)
```

### 3.2 第二优先级：小算子拼接组合

**适用场景：**
- 融合算子（如FlashAttention）
- 量化算子（多算子串联）
- 特殊融合结构

**实现示例：**
```python
# FlashAttention用小算子组合实现
def flash_attention_reference(Q, K, V):
    scores = np.matmul(Q, K.transpose())
    attention = softmax(scores / np.sqrt(d_k))
    output = np.matmul(attention, V)
    return output
```

### 3.3 第三优先级：自行构造CPU实现

**适用场景：** 非标准数据类型（如HiFLOAT8）

**注意事项：**
- 确保CPU实现正确性
- 使用高精度数值计算
- 提供充分测试用例
- 文档化实现细节

## 4. 比对方法选择

### 4.1 单标杆比对

**定义：** 与单一精度标杆直接比较

**适用算子：**
- 非计算类
- 整数计算类
- 量化类（整型输出）
- 生态算子

**要求：** 标杆应为更高精度实现

### 4.2 双标杆比对

**定义：** 以CPU为Golden，同时评估NPU和三方芯片

**适用算子：**
- 浮点计算类（商用标准）
- 量化类（浮点输出）

**实现要点：**
```python
# Golden: 高精度CPU实现
golden = cpu_implementation_high_precision(input)

# Benchmark: 三方芯片
benchmark = gpu_implementation(input)

# NPU实现
npu_output = npu_implementation(input)

# 计算Ratio
mare_ratio = mare(npu, golden) / mare(benchmark, golden)
```