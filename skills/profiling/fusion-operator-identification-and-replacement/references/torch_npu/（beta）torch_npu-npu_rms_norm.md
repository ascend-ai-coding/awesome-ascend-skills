# （beta）torch_npu.npu_rms_norm

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
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

- API功能：RMSNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。
- 计算公式：

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i
  $$

  $$
  \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

## 函数原型

```python
torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor) 
```

## 参数说明

- **self**（`Tensor`）：必选参数，表示进行归一化计算的输入。对应计算公式中的$x$。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，shape支持2-8维度，数据格式支持$ND$。支持非连续Tensor，支持空Tensor。
- **gamma**（`Tensor`）：必选参数，表示进行归一化计算的缩放因子（权重），对应计算公式中的$g$。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，shape支持2-8维度，数据格式支持$ND$。shape需要满足gamma_shape = self_shape\[n:\], n < self_shape.dims()，通常为`self`的最后一维。支持非连续Tensor，支持空Tensor。
- **epsilon**（`float`）：可选参数，用于防止除0错误，对应计算公式中的`eps`。数据类型为`torch.double`，默认值为1e-6。

## 返回值说明

- **yOut**（`Tensor`）：表示进行归一化后的最终输出，对应计算公式的最终输出$RmsNorm(x)$。数据类型和shape与输入`self`一致。支持非连续Tensor，支持空Tensor。
- **rstdOut**（`Tensor`）：表示归一化后的标准差的倒数，rms_norm的中间结果，对应计算公式中的$Rms(x)$的倒数，用于反向计算。数据类型为`torch.float32`。shape与入参`self`的shape前几维一致，前几维指`self`的维度减去`gamma`的维度，表示不需要norm的维度。支持非连续Tensor，支持空Tensor。

## 约束说明

- 该接口支持单算子模式调用。

<!-- npu="950" id6 -->
- <term>Ascend 950PR&950DT系列产品</term>：该接口还支持TorchAir图模式调用。
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas推理系列产品</term>：`self`、`gamma`输入的尾轴长度必须大于等于32 Bytes。
<!-- end id7 -->
- 边界值场景说明：

  <!-- npu="310p" id8 -->
  - <term>Atlas推理系列产品</term>：输入不支持包含Inf和NaN。
  <!-- end id8 -->
  <!-- npu="950,A3,910b" id9 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Ascend 950PR&950DT系列产品</term>：当输入时Inf时，输出为Inf。当输入是NaN时，输出为NaN。
  <!-- end id9 -->

- 各产品支持数据类型及对应关系说明：

  <!-- npu="950,A3,910b" id10 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Ascend 950PR&950DT系列产品</term>：

    | `self`数据类型 | `gamma`数据类型 | `yOut`数据类型 | `rstdOut`数据类型 |
    | -------- | -------- | -------- | -------- |
    | `torch.float16` | `torch.float32` | `torch.float16` | `torch.float32` |
    | `torch.bfloat16` | `torch.float32` | `torch.bfloat16` | `torch.float32` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` |
    | `torch.float32` | `torch.float32`  | `torch.float32` | `torch.float32` |
  <!-- end id10 -->
  <!-- npu="910,310p" id11 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>：

    | `self`数据类型 | `gamma`数据类型 | `yOut`数据类型 | `rstdOut`数据类型 |
    | -------- | -------- | -------- | -------- |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` |
    | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` |
  <!-- end id11 -->

## 调用示例

- 单算子调用

    ```python
    import torch
    import torch_npu

    x = torch.randn((24, 1, 128), dtype=torch.bfloat16).npu()
    w = torch.randn((128), dtype=torch.bfloat16).npu()
    y, rstd = torch_npu.npu_rms_norm(x, w, epsilon=1e-5)
    ```

<!-- npu="950" id12 -->
- 图模式调用：该示例仅支持<term>Ascend 950PR&950DT系列产品</term>。

    ```python
    import torch
    import torch_npu
    import torchair
    import numpy as np


    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, gamma, eps):
            return torch_npu.npu_rms_norm(x, gamma, epsilon=eps)
    # generate inputs
    x = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    model = NetModel()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    y, rstd = model(x, gamma, 1e-5)
    ```
<!-- end id12 -->
