# torch_npu.npu_group_norm_swish

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：计算输入`input`的组归一化结果`y`、均值`mean`、标准差的倒数`rstd`，以及swish的输出。
- 计算公式：
  - GroupNorm: 公式中的$x$代表`input`， $E[x] = \bar{x}$ 代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^{n} (x_i - E[x])^2$ 代表$x$的方差，$\gamma$代表`weight`，$\beta$代表`bias`，则公式如下：
  $$
  \begin{cases}
  y & = \frac{x - E[x]}{\sqrt{{Var[x]} + eps}} * \gamma + \beta \\ 
  mean & = E[x] \\ 
  rstd & = \frac{1}{\sqrt{{Var[x]} + eps}}
  \end{cases}
  $$

  - swish：swish计算公式的$x$为GroupNorm公式得到的$y$。
  $$
  y = \frac{x}{1 + e^{-scale \cdot x}}
  $$
  
> **说明：**<br>
> 正向默认确定性实现。<br>
> <!-- npu="A3,910b" id4 -->
> 反向梯度在<term>Atlas A3训练系列产品</term>和<term>Atlas A2训练系列产品</term>上默认非确定性实现。
> <!-- end id4 -->
> <!-- npu="950" id5 -->
> 反向梯度在<term>Ascend 950DT系列产品</term>上默认确定性实现。
> <!-- end id5 -->
> 需要计算反向梯度场景时，若需要输出结果排除随机性，则需要[设置确定性计算开关](../determin_API_list.md)。

## 函数原型

```python
torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **input**(`Tensor`)：必选参数，表示需要进行组归一化的数据，支持2-8D张量，数据类型支持`torch.float16`，`torch.float32`，`torch.bfloat16`。

- **num_groups**(`int`)：必选参数，表示将`input`的第1维分为`num_groups`组，`input`的第1维必须能被`num_groups`整除。
  - `num_groups=1`：等价于LayerNorm（层归一化），对整个输入进行归一化，适用于序列建模、全连接层后等场景。
  - `num_groups=C`（C为通道数）：等价于InstanceNorm（实例归一化），每个通道独立归一化，适用于风格迁移、图像生成等场景。
  - `1 < num_groups < C`：标准GroupNorm，在通道维度上划分若干组进行归一化，`num_groups=32`是常用的默认值。
  > [!NOTE]
  > 
  > 当需要计算反向梯度时，`input.shape[1] / num_groups` 的结果不能超过4000。违反此约束可能导致训练时的错误。

- **weight**(`Tensor`)：必选参数，表示权重，支持1D张量，并且第0维大小与`input`的第1维相同；数据类型支持`torch.float16`，`torch.float32`，`torch.bfloat16`，并且需要与`input`一致。

- **bias**(`Tensor`)：必选参数，表示偏置，支持1D张量，并且第0维大小与`input`的第1维相同；数据类型支持`torch.float16`，`torch.float32`，`torch.bfloat16`，并且需要与`input`一致。

- **eps**(`float`)：可选参数，计算组归一化时加到分母上的值，以保证数值的稳定性。默认值为1e-5。

- **swish_scale**(`float`)：可选参数，swish计算的缩放因子。默认值为1.0。

## 返回值说明

- **y**(`Tensor`)：组归一化后经过Swish激活的最终输出，用于网络的前向传播。shape与`input`相同，数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。

- **mean**(`Tensor`)：每个分组的均值，用于反向传播时的梯度计算（需要与`y`一起保存）。shape为\(N, num_groups\)，其中N为`input`第0维大小。数据类型与`input`相同。

- **rstd**(`Tensor`)：每个分组标准差的倒数，用于反向传播时的梯度计算（需要与`y`一起保存）。shape为\(N, num_groups\)，其中N为`input`第0维大小。数据类型与`input`相同。

## 约束说明

- 当需要计算**反向梯度**时，`input.shape[1] / num_groups` 的结果不能超过4000。违反此约束可能导致训练时的错误。该约束仅在前向+反向传播场景下生效，纯推理场景不受此约束限制。
- `input`、`weight`、`bias`参数不支持含有-inf、inf或nan值。

## 调用示例

```python
import torch
import torch_npu
 
input = torch.randn(3, 4, 6, dtype=torch.float32).npu()
weight = torch.randn(input.size(1), dtype=torch.float32).npu()
bias = torch.randn(input.size(1), dtype=torch.float32).npu()
num_groups = input.size(1)
eps = 1e-5
swish_scale = 1.0
out, mean, rstd = torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=eps, swish_scale=swish_scale)
```
