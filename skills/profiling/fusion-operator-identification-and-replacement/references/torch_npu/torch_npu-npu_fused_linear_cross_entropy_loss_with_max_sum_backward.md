# torch\_npu.npu\_fused\_linear\_cross\_entropy\_loss\_with\_max\_sum\_backward

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：词汇表并行场景下交叉熵损失计算的梯度算子。用于计算叶子节点`input`和`weight`的梯度。需要获得npu\_fused\_linear\_online\_max\_sum和npu\_fused\_cross\_entropy\_loss\_with\_max\_sum的相关输出作为本接口输入。支持高性能模式（传入Softmax）和省显存模式（传入`logits_max`和`sum_exp_logits`）。

- 计算公式：

&emsp;&emsp;高性能模式（`softmax`非None）：

$$
\text{softmax} \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = \mathbf{1} - \text{target\_mask}.view(-1) \in \mathbb{R}^{BT}
$$

$$
\text{softmax}[\text{arange\_1d}, \text{masked\_target}] \leftarrow \text{softmax}[\text{arange\_1d}, \text{masked\_target}] - \text{softmax\_update}
$$

$$
\text{softmax} \leftarrow \text{softmax} \odot \text{grad}.unsqueeze(-1) \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{softmax} \cdot \text{weight}^T \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{softmax}^T \cdot \text{input} \in \mathbb{R}^{V \times H}
$$

&emsp;&emsp;省显存模式（`softmax`为None）：

$$
\text{vocab\_parallel\_logits} = \text{input} \cdot \text{weight}^T \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{logits\_sub} = \text{vocab\_parallel\_logits} - \text{logits\_max}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} = \exp(\text{logits\_sub}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} \gets \frac{\text{exp\_logits}}{\text{sum\_exp\_logits}.unsqueeze(-1)} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_logits} = \text{exp\_logits} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_2d} = \text{grad\_logits}.view(-1, \text{partition\_vocab\_size}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \quad \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = 1 - \text{target\_mask}.view(-1) \quad \in \mathbb{R}^{BT}
$$

$$
\text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] \gets \text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] - \text{softmax\_update}
$$

$$
\text{grad\_logits} \gets \text{grad\_logits} \odot \text{grad}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{grad\_logits} \cdot \text{weight} \quad \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{grad\_logits}^T \cdot \text{input} \quad \in \mathbb{R}^{V \times H}
$$

## 函数原型

```python
torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(grad, input, weight, target_mask, masked_target, label_smoothing=0.0, logits_max=None, sum_exp_logits=None, softmax=None) -> (Tensor, Tensor)
```

## 参数说明

- **grad**（`Tensor`）：必选参数，当前节点的梯度，1维Tensor。数据类型支持`torch.float32`，数据格式支持ND。支持空Tensor。
- **input**（`Tensor`）：必选参数，矩阵乘的输入矩阵，2维Tensor。数据类型支持`torch.float16`、`torch.bfloat16`，数据格式支持ND。第0维长度需与`grad`一致。支持空Tensor。
- **weight**（`Tensor`）：必选参数，矩阵乘的权重矩阵，2维Tensor。数据类型需与`input`一致，数据格式支持ND。第0维长度不支持小于128，第1维长度需与`input`的第1维一致。
- **target\_mask**（`Tensor`）：必选参数，目标词ID是否在范围内的位掩码，1维Tensor。数据类型支持`torch.uint8`，数据格式支持ND。每1bit代表1个布尔值，shape长度乘以8须不小于`grad`长度。
- **masked\_target**（`Tensor`）：必选参数，目标词ID映射到当前设备的局部索引，1维Tensor。数据类型支持`torch.int32`、`torch.int64`，数据格式支持ND。shape长度需与`grad`一致。
- **label\_smoothing**（`float`）：可选参数，标签平滑系数，当前仅支持0.0，默认值为0.0。
- **logits\_max**（`Tensor`）：可选参数，全局logits最大值，1维Tensor。数据类型支持`torch.float32`。`softmax`为None时必须提供。默认值为None。
- **sum\_exp\_logits**（`Tensor`）：可选参数，处理后的logits，1维Tensor。数据类型支持`torch.float32`。`softmax`为None时必须提供。默认值为None。
- **softmax**（`Tensor`）：可选参数，Softmax计算结果，2维Tensor。数据类型支持`torch.float32`。传入时走高性能模式，不传时走省显存模式（需提供`logits_max`和`sum_exp_logits`）。默认值为None。

## 返回值说明

- **input\_grad**（`Tensor`）：`input`的梯度，数据类型与`input`一致，shape为`[input.size(0), input.size(1)]`。
- **weight\_grad**（`Tensor`）：`weight`的梯度，数据类型与`input`一致，shape为`[weight.size(0), weight.size(1)]`。

## 约束说明

- `label_smoothing`当前仅支持0。
- `softmax`为None时，`logits_max`和`sum_exp_logits`必须同时提供。
- `softmax`非None时，`logits_max`和`sum_exp_logits`无效。

## 调用示例

- 高性能模式（传入Softmax）：

```python
import torch
import torch_npu

batch = 128
hidden = 64
vocab_size = 256

grad = torch.randn(batch, dtype=torch.float32).npu()
input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()
softmax = torch.randn(batch, vocab_size, dtype=torch.float32).npu()

input_grad, weight_grad = torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(
    grad, input_tensor, weight_tensor, target_mask, masked_target,
    label_smoothing=0.0, softmax=softmax
)
```

- 省显存模式（传入logits_max和sum_exp_logits）：

```python
import torch
import torch_npu

batch = 128
hidden = 64
vocab_size = 256

grad = torch.randn(batch, dtype=torch.float32).npu()
input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()
logits_max = torch.randn(batch, dtype=torch.float32).npu()
sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0

input_grad, weight_grad = torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(
    grad, input_tensor, weight_tensor, target_mask, masked_target,
    label_smoothing=0.0, logits_max=logits_max, sum_exp_logits=sum_exp_logits, softmax=None
)
```
