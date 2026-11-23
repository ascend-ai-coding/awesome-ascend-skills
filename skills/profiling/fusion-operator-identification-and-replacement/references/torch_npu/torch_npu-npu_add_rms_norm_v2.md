# torch_npu.npu_add_rms_norm_v2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：AddRmsNorm算子是Add算子和RmsNorm算子的融合。其中RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，去掉了减去均值的部分。

- **计算公式**：

  $$
  x_i=x1_i+x2_i
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## 函数原型

```python
torch_npu.npu_add_rms_norm_v2(x1, x2, gamma, epsilon=1e-6) -> Tensor
```

## 参数说明

- **x1**（`Tensor`）：必选参数/输出，对应公式中的$x1$和$RmsNorm(x)$，表示用于Add计算的第一个输入和归一化后的最终输出结果。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，shape支持1-8维度。数据格式支持$ND$。不支持空Tensor。
- **x2**（`Tensor`）：必选参数/输出，对应公式中的$x2$和$x$，表示用于Add计算的第二个输入和Add计算的结果。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，shape和`x1`保持一致。数据格式支持$ND$。不支持空Tensor。
- **gamma**（`Tensor`）：必选参数，对应公式中的$gamma$，表示RmsNorm的缩放因子（权重）。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，shape要与`x1`需要Norm的维度保持一致。数据格式支持$ND$。不支持空Tensor。
- **epsilon**（`float`）：可选参数，对应公式中的$epsilon$，数据类型支持`torch.double`，默认值为1e-6。

## 返回值说明

**rstd**（`Tensor`）：对应公式中的$Rms(x)$，Norm计算的中间结果，用于反向计算使用。数据类型仅支持`torch.float32`，维度数与`x1`保持一致，不需要norm的维度与`x1`对应维度保持一致，需要norm的维度为1。`rstd`shape与`x1`shape，`gamma`shape关系举例：

  - 若`x1`shape:(2,3,4,8)，`gamma`shape:(8)，`rstd`shape:(2,3,4,1)；
  - 若`x1`shape:(2,3,4,8)，`gamma`shape:(4,8)，`rstd`shape:(2,3,1,1)。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式。
- 输出会覆盖原有的x1、x2。
- 单算子模式下输入参数`x2`、`gamma`的数据类型必须和`x1`数据类型保持一致。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    x1 = torch.randn((32,128), dtype=torch.float32).npu()
    x2 = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    rstd = torch_npu.npu_add_rms_norm_v2(x1, x2, gamma, epsilon=1e-5)
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
            return torch_npu.npu_add_rms_norm_v2(x1, x2, gamma, epsilon=eps)
    # generate inputs
    x1 = torch.randn((32,128), dtype=torch.float32).npu()
    x2 = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    model = NetModel()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    rstd = model(x1, x2, gamma, 1e-5)
    ```
