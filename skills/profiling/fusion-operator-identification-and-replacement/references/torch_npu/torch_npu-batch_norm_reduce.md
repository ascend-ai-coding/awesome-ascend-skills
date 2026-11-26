# torch_npu.batch_norm_reduce

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：该接口用于按BatchNorm通道维对输入Tensor进行归约，计算通道维以外维度上的元素和 `sum` 与平方和 `square_sum` 。

- 计算公式：对于4D Tensor输入 `(N, C, H, W)`，计算C轴以外的N、H、W轴求和与平方和。

    $$
    sum_i = \sum_{n=0}^{N-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} input_{(n,i,h,w)}
    $$

    $$
    square\_sum_i = \sum_{n=0}^{N-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} input_{(n,i,h,w)}^2
    $$

## 函数原型

```python
torch_npu.batch_norm_reduce(input, eps) -> (Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，表示待归约的输入Tensor。支持4D Tensor，shape为 `(N, C, H, W)` 。数据类型支持 `torch.float32`、`torch.float16`、`torch.bfloat16`。
- **eps** (`float`)：必选参数，保留参数。该参数不参与 `sum` 和 `square_sum` 的计算，建议按BatchNorm常用配置传入 `1e-5`。

## 返回值说明

- **sum** (`Tensor`)：输入Tensor按通道维归约后的元素和，为1D `torch.float32` Tensor，长度为通道维长度。输入为4D Tensor且shape为 `(N, C, H, W)` 时，输出shape为 `(C,)` 。
- **square_sum** (`Tensor`)：输入Tensor按通道维归约后的平方和，为1D `torch.float32` Tensor，shape同 `sum` 。

## 调用示例

```python
import torch
import torch_npu

input = torch.randn(2, 3, 12, 12, dtype=torch.float32, device="npu")

sum_out, square_sum_out = torch_npu.batch_norm_reduce(input, 1e-5)

print("sum_out.shape:", sum_out.shape)
print("sum_out.dtype:", sum_out.dtype)
print("square_sum_out.shape:", square_sum_out.shape)
print("square_sum_out.dtype:", square_sum_out.dtype)
```
