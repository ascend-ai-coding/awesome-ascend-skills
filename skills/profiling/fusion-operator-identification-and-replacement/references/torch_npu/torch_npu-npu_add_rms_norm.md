# torch_npu.npu_add_rms_norm

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

- **API功能**：将Add计算与RMSNorm归一化融合，常用于大模型中将残差连接后的张量进行归一化处理。
- **计算公式**：

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  \operatorname{RMSNorm}(x_i)=\frac{x_i}{\operatorname{RMS}(\mathbf{x})} gamma_i, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## 函数原型

```python
torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=1e-06) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x1**(`Tensor`)：必选参数，表示用于Add计算的第一个输入，对应公式中的$x1$。数据格式支持$ND$，支持空Tensor，支持非连续Tensor。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。支持1-8维张量。
- **x2**(`Tensor`)：必选参数，表示用于Add计算的第二个输入，对应公式中的$x2$。数据格式支持$ND$，支持空Tensor，支持非连续Tensor。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。支持1-8维张量，shape和`x1`保持一致。
- **gamma**(`Tensor`)：必选参数，表示RMSNorm的缩放因子（权重）。对应公式中的$gamma$。数据格式支持$ND$，支持空Tensor，支持非连续Tensor。数据类型与`x1`的数据类型保持一致，shape要与`x1`需要norm的维度保持一致。
- **epsilon**(`float`)：可选参数，表示添加到分母中的值，以确保数值稳定，对应公式中的$epsilon$。默认值为`1e-6`。

## 返回值说明

- **yOut**(`Tensor`)：对应公式中的$RMSNorm(x)$，表示最后的输出。数据格式支持$ND$，支持空Tensor，支持非连续Tensor。数据类型和shape与输入`x1`的数据类型和shape保持一致。

- **rstdOut**(`Tensor`)：对应公式中$RMS(x)$的倒数，表示归一化后均方根的倒数。数据格式支持$ND$，支持空Tensor，支持非连续Tensor。数据类型仅支持`torch.float32`。维度数与`x1`保持一致，不需要norm的维度与`x1`对应维度保持一致，需要norm的维度为1。`rstdOut`shape与`x1`shape，`gamma`shape关系举例：
  - 若`x1`shape:(2,3,4,8)，`gamma`shape:(8)，`rstdOut`shape:(2,3,4,1)；
  - 若`x1`shape:(2,3,4,8)，`gamma`shape:(4,8)，`rstdOut`shape:(2,3,1,1)。

- **xOut**(`Tensor`)：对应公式中的$x$，表示`x1`与`x2`相加的计算结果。数据格式支持$ND$，支持空Tensor，支持非连续Tensor。数据类型和shape与输入`x1`的数据类型和shape保持一致。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式。
- 单算子模式下输入参数`x2`、`gamma`的数据类型必须和`x1`数据类型保持一致。
- 边界值场景说明：
  - 当输入是Inf时，输出为Inf。
  - 当输入是NaN时，输出为NaN。

<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：
  - 输入参数`x1`、`x2`、`gamma`、`yOut`、`xOut`的数据类型不支持`torch.bfloat16`。
  - 参数`rstdOut`在当前产品使用场景下无效。
<!-- end id5 -->

## 调用示例

- 单算子模式调用
    
    ```python
    import torch
    import torch_npu

    x1 = torch.randn((32,128), dtype=torch.float32).npu()
    x2 = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    y, rstd, x = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=1e-6)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import numpy as np

    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, gamma, eps):
            return torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=eps)
    # generate inputs
    x1 = torch.randn((32,128), dtype=torch.float32).npu()
    x2 = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    model = NetModel()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    y, rstd, x = model(x1, x2, gamma, 1e-6)
    ```
